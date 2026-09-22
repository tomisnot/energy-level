# -*- coding: utf-8 -*-
"""机甲 MCP 写通道自检（Phase 2）——工具面 + 回执 + 可教学错误 + 体积闸 + 真连。

守的是 Phase 2 验收：
  A. 工具面 = 7 个薄封装（能力清单）。
  B. list_params 读参数表（含单位/范围/说明；默认藏 internal，D-B）。
  C. set_param 写回执带 resolved + choke point 读回的 readback（M2 声称vs实测）。
  D. 打错键名 → 可教学错误（含"你是不是想写 X"，D2）。
  E. 非法值 → invalid_value（报错不静默，D5）。
  F. get_activity diff-since-seq 只回增量（R1 头号原语）。
  G. undo 撤销（L2 可逆）。
  H. **所有回程走同一把体积闸**；灌爆 → 截断且报"截了多少+原文在哪"（I4）；
     致死性：小回程不被截。
  I. **真实 MCP 连接**（streamable-http）：线程起 server + 客户端 initialize/
     list_tools/call_tool——证明"harness 能连"（不永久 SKIP，D4）。

用临时 root（不污染 `<repo>/.mecha`）。server 起在 ephemeral 端口、daemon 线程。
"""
import asyncio
import json
import logging
import os
import socket
import sys
import tempfile
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

# 压低 uvicorn/mcp/httpx 的 INFO 噪声（自检输出要干净）
for _n in ('uvicorn', 'uvicorn.error', 'uvicorn.access', 'mcp',
           'mcp.server.streamable_http_manager', 'httpx', 'httpcore'):
    logging.getLogger(_n).setLevel(logging.ERROR)

from checks.checklib import Checker                        # noqa: E402
from mecha.session import Session                      # noqa: E402
from mecha.ops import Ops                             # noqa: E402
from mecha import server as SVR                       # noqa: E402
from mecha.bus import gate_value                       # noqa: E402
from mecha import runtime                              # noqa: E402
from src.adapter import EngineAdapter                      # noqa: E402

runtime.set_engine(EngineAdapter())   # 注入领域引擎（mecha 零领域 import）
from arc import Rubidium87                                 # noqa: E402
from src.core.static_atom import ArcCache                  # noqa: E402
from src.core import runspec as RS                         # noqa: E402
from src.core import waveforms as WF                       # noqa: E402
from src.core.atomic_levels import state_key               # noqa: E402

C = Checker('mecha_mcp')
_TMP = tempfile.mkdtemp(prefix='mecha_mcp_')
_session = Session(root=_TMP)
_ops = Ops(_session)
_srv, tools = SVR.create_server(_session, _ops, actor='ai-test', log_level='ERROR')


def call(name, **kw):
    """直接调已注册的工具函数（返回 JSON 字符串）→ dict。同一份代码路径。"""
    return json.loads(tools[name](**kw))


# ---------------- A. 工具面 ----------------
_names = sorted(k for k in tools if not k.startswith('_'))
C.group('A MCP 工具面 = 18 个薄封装（能力清单）', [
    ('create_server 返回 18 个工具（实际 %d）' % len(_names), len(_names) == 18),
    ('工具名与 TOOL_NAMES 一致', set(_names) == set(SVR.TOOL_NAMES)),
    ('含 async 作业面 + preview_behavior + 建场 set_drive/get_waves + list_levels + mecha_script',
     {'submit_job', 'job_status', 'cancel_job', 'list_jobs', 'preview_behavior',
      'set_drive', 'get_waves', 'list_levels', 'mecha_script'} <= set(_names)),
    ('工具可调、返回可解析 JSON（薄封装，逻辑在 session/ops）',
     isinstance(call('get_state'), dict)),
])

# ---------------- B. list_params ----------------
lp = call('list_params')
p0 = lp['params'][0] if lp.get('params') else {}
has_unit = any(p.get('unit') for p in lp['params'])
has_range = any(('lo' in p or 'hi' in p or 'enum' in p) for p in lp['params'])
C.group('B list_params 读参数表（单位/范围/说明；默认藏 internal，D-B）', [
    ('返回 count/returned/params', lp.get('count', 0) > 0 and 'params' in lp),
    ('每项带 name/family/type/desc',
     all({'name', 'family', 'type', 'desc'} <= set(p) for p in lp['params'])),
    ('有单位信息（unit）', has_unit),
    ('有值域/枚举信息（lo/hi/enum）', has_range),
    ('默认不露 internal（D-B）',
     all(p.get('tier') != 'internal' for p in lp['params'])),
    ('include_internal=true 才含 internal',
     call('list_params', include_internal=True)['count'] > lp['count']),
])

# ---------------- C. set_param 回执（M2） ----------------
r = call('set_param', key='field.B_gauss', value=21.7, reason='Zeeman 调谐')
C.group('C set_param 写回执带 resolved + readback（M2 声称vs实测）', [
    ('ok=True', r.get('ok') is True),
    ('resolved == 21.7', r.get('resolved') == 21.7),
    ('readback == resolved（choke point 产出，AI 伪造不了）',
     r.get('readback') == r.get('resolved') == 21.7),
    ('带 seq（写进了事件总线）', isinstance(r.get('seq'), int)),
    ('快照真的变了（get_state 读回）',
     call('get_state')['snapshot'].get('field.B_gauss') == 21.7),
])

# ---------------- D. 可教学错误（D2） ----------------
r = call('set_param', key='field.B_gaus', value=1.0, reason='手滑打错')
err = r.get('error', {})
C.group('D 打错键名 → 可教学错误（含"你是不是想写 X"）', [
    ('ok=False（不是 500、不静默）', r.get('ok') is False),
    ("error.kind == 'unknown_key'", err.get('kind') == 'unknown_key'),
    ("suggest 给出最可能的 field.B_gauss",
     'field.B_gauss' in (err.get('suggest') or [])),
    ('带 hint（下一步怎么办）', bool(err.get('hint'))),
    ('错误体可 JSON 序列化（能过 MCP 回程）',
     isinstance(json.dumps(err, ensure_ascii=False), str)),
])

# ---------------- E. 非法值（D5） ----------------
r = call('set_param', key='dynamic.t_max', value=-5, reason='坏值')
C.group('E 非法值 → invalid_value（报错不静默，D5）', [
    ('ok=False', r.get('ok') is False),
    ("error.kind == 'invalid_value'",
     r.get('error', {}).get('kind') == 'invalid_value'),
    ('非法值没落进快照（自愈回退）',
     'dynamic.t_max' not in call('get_state')['snapshot']),
])

# ---------------- F. diff-since-seq（R1） ----------------
seq0 = _session.bus.last_seq
call('set_param', key='field.E_vpcm', value=1.0, reason='DC-Stark')
act = call('get_activity', since_seq=seq0)
C.group('F get_activity diff-since-seq（只回增量，R1 头号原语）', [
    ('只回 seq0 之后的新事件（1 条）',
     act['count'] == 1 and all(e['seq'] > seq0 for e in act['events'])),
    ('带 last_seq（供下次 tail）', isinstance(act.get('last_seq'), int)),
    ('全量 since_seq=0 回得到更多', call('get_activity', since_seq=0)['count']
     > act['count']),
])

# ---------------- G. undo（L2） ----------------
r = call('undo', reason='撤销 DC-Stark')
C.group('G undo 撤销最近一次写入（L2 可逆）', [
    ('ok=True', r.get('ok') is True),
    ('E_vpcm 已从快照移除',
     'field.E_vpcm' not in call('get_state')['snapshot']),
    ('undo 也进了总线（kind=undo）',
     call('get_activity', since_seq=0)['events'][-1]['kind'] == 'undo'),
])

# ---------------- H. 体积闸（I4） + 致死性 ----------------
big = gate_value(_TMP, {'blob': 'x' * 200000}, max_bytes=65536)
small = gate_value(_TMP, {'a': 1, 'b': [1, 2, 3]}, max_bytes=65536)
C.group('H 体积闸（I4）：灌爆回程→截断且告知；小回程不截（致死性）', [
    ('超预算被截断（$truncated=True）', big.get('$truncated') is True),
    ('报了"截了多少"（kept < total）',
     big.get('total', 0) > big.get('kept', 0)),
    ('报了"原文在哪"（where + sha1）',
     'where' in big and len(big.get('sha1', '')) == 40),
    ('**致死性**：小回程原样返回（不被误截）', small == {'a': 1, 'b': [1, 2, 3]}),
])


# ---------------- I. 真实 MCP 连接（harness 能连） ----------------
def _connection_test():
    """线程起 streamable-http server + 客户端连；返回 {names, call_ok, count} 或 {error}。"""
    port = SVR.find_free_port(0)            # ephemeral
    def _run():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(srv_run(port))
        except Exception:                    # noqa: BLE001
            pass

    async def srv_run(p):
        await _srv.run_streamable_http_async(host='127.0.0.1', port=p,
                                             streamable_http_path='/mcp')

    th = threading.Thread(target=_run, daemon=True)
    th.start()
    # 等端口就绪
    up = False
    for _ in range(80):
        try:
            with socket.create_connection(('127.0.0.1', port), timeout=0.2):
                up = True
                break
        except OSError:
            time.sleep(0.1)
    if not up:
        return {'error': 'server 未在 8s 内就绪'}

    async def _client():
        from mcp.client.streamable_http import streamable_http_client
        from mcp import ClientSession
        url = 'http://127.0.0.1:%d/mcp' % port
        async with streamable_http_client(url) as streams:
            read, write = streams[0], streams[1]
            async with ClientSession(read, write) as sess:
                await sess.initialize()
                lt = await sess.list_tools()
                names = sorted(t.name for t in lt.tools)
                res = await sess.call_tool('list_params', {})
                txt = res.content[0].text if res.content else ''
                d = json.loads(txt) if txt else {}
                res2 = await sess.call_tool(
                    'set_param', {'key': 'dynamic.method', 'value': 'sambe',
                                  'reason': 'conn test'})
                d2 = json.loads(res2.content[0].text) if res2.content else {}
                return {'names': names, 'list_ok': not res.is_error,
                        'count': d.get('count'),
                        'set_ok': d2.get('ok') is True and not res2.is_error}

    try:
        return asyncio.run(_client())
    except Exception as e:                   # noqa: BLE001
        return {'error': '%s: %s' % (type(e).__name__, e)}


conn = _connection_test()
C.group('I 真实 MCP 连接（streamable-http；harness 能连，不永久 SKIP）', [
    ('连接无错（%s）' % conn.get('error', 'ok'), 'error' not in conn),
    ('initialize + list_tools 返回 16 个工具',
     conn.get('names') == sorted(SVR.TOOL_NAMES)),
    ('call_tool(list_params) 成功、count>0',
     conn.get('list_ok') is True and (conn.get('count') or 0) > 0),
    ('call_tool(set_param) 经传输写入成功', conn.get('set_ok') is True),
])

# ---- 建最小可运行配置（供 preview_behavior / async 作业工具测试）----
_cache = ArcCache(Rubidium87())
_G = [dict(n=60, l=0, j=0.5), dict(n=60, l=1, j=1.5)]
_st = RS.build_states(_cache, _G, 0.0, 0.0)
_freq = abs(_cache.E_mhz(60, 1, 1.5) - _cache.E_mhz(60, 0, 0.5))
_w = WF.Wave('opt', freq_mhz=_freq,
             envelope=WF.SquareEnv(0.0, 1.0, 1.0, 8.0), pol='π',
             intensity_mwcm2=1.0)
for _k, _v in {'groups.list': _G, 'dynamic.mode': 0, 'dynamic.method': 'mixed',
               'dynamic.t_max': 10.0, 'dynamic.npts': 11,
               'dynamic.decay_index': 1, 'dynamic.psi_single': state_key(_st[0]),
               'dynamic.waves_single': WF.serialize_waves([_w])}.items():
    _session.set(_k, _v, actor='setup', reason='判据配置')

# ---------------- J. preview_behavior + list_levels 一等工具 ----------------
pv = call('preview_behavior')
lv = call('list_levels')
_ca = call('mecha_script', script="result = {'nk': len(registry.all_settable_keys())}")
C.group('J preview_behavior 预检基组（D7）+ list_levels 语义寻址 + mecha_script 编排', [
    ('preview_behavior 返回基组规模 n_states>0', pv.get('n_states', 0) > 0),
    ('报被驱动跃迁 driven>=1（含 rabi）',
     len(pv.get('driven', [])) >= 1 and 'rabi_rad_s' in pv['driven'][0]),
    ('回答 basis_enough（基组够不够）', isinstance(pv.get('basis_enough'), bool)),
    ('list_levels 返回态（label + 身份键 key，供设初态）',
     lv.get('n_states', 0) > 0 and 'key' in lv['states'][0]
     and 'label' in lv['states'][0]),
    ('mecha_script 工具可多步编排（受限命名空间）',
     _ca.get('ok') is True and _ca.get('result', {}).get('nk', 0) >= 20),
])

# ---------------- K. async 作业面 + run_compute 超阈值自动转 async ----------------
r_sync = call('run_compute', reason='同步档', mode='sync')
_old_th = SVR.ASYNC_THRESHOLD_SEC
SVR.ASYNC_THRESHOLD_SEC = -1.0            # 压阈值 → auto 必转 async（单体 est_sec=0）
r_auto = call('run_compute', reason='自动档（阈值压低）', mode='auto')
SVR.ASYNC_THRESHOLD_SEC = _old_th
sj = call('submit_job', reason='经 MCP 工具发起')
_jid = sj.get('job_id')
_final = None
for _ in range(80):
    _j = call('job_status', job_id=_jid)
    if _j.get('status') in ('done', 'error', 'cancelled'):
        _final = _j
        break
    time.sleep(0.25)
_lj = call('list_jobs')
C.group('K async 作业面（长任务不阻塞写通道）+ run_compute 超阈值自动转 async', [
    ('run_compute mode=sync → 同步出结果（非 async）',
     r_sync.get('ok') is True and not r_sync.get('async', False)),
    ('run_compute mode=auto 且 est>阈值 → **自动转 async**（回 job_id，不阻塞）',
     r_auto.get('async') is True and bool(r_auto.get('job_id'))),
    ('submit_job → job_id；job_status 轮询到 done + 结果 ok',
     bool(_jid) and _final is not None and _final['status'] == 'done'
     and (_final.get('result') or {}).get('ok') is True),
    ('list_jobs 列出在册 job（断线可找回）',
     any(x['id'] == _jid for x in _lj.get('jobs', []))),
    ('job_status 进度是三槽 [frac,cancel,stage]',
     isinstance(_final.get('progress'), list) and len(_final['progress']) == 3),
])

sys.exit(0 if C.summary() else 1)
