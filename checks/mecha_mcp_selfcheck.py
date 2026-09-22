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
# P1-9d：`set_drive`/`get_waves` 已**搬回领域侧**（`src/adapter/mcp_tools.py`），
# 库的 `TOOL_NAMES` 只剩**通用**工具（18→16）；领域工具经 `engine.mcp_tools()` 自报。
_names = sorted(k for k in tools if not k.startswith('_'))
_domain = sorted(set(_names) - set(SVR.TOOL_NAMES))
C.group('A MCP 工具面 = 16 通用 + 领域自报（能力清单）', [
    ('库内通用工具 %d 个（实际注册总数 %d）'
     % (len(SVR.TOOL_NAMES), len(_names)), len(SVR.TOOL_NAMES) == 16),
    ('注册总数 = 通用 16 + 领域自报 %d' % len(_domain),
     len(_names) == len(SVR.TOOL_NAMES) + len(_domain)),
    ('**领域工具不在库里**（库 TOOL_NAMES 不含 set_drive/get_waves）',
     not ({'set_drive', 'get_waves'} & set(SVR.TOOL_NAMES))),
    ('**领域工具由本领域自报**（set_drive/get_waves 在）',
     {'set_drive', 'get_waves'} <= set(_names)),
    ('含 async 作业面 + preview_behavior + list_levels + mecha_script',
     {'submit_job', 'job_status', 'cancel_job', 'list_jobs',
      'preview_behavior', 'list_levels', 'mecha_script'} <= set(_names)),
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
    ('initialize + list_tools 与本地注册表一致（%d 个：通用 %d + 领域自报 %d）'
     % (len(_names), len(SVR.TOOL_NAMES), len(_domain)),
     conn.get('names') == sorted(_names)),
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

# ---------------- L. P0-3/P0-4：numpy 惰性 + 无建场层不崩（含致死性） ----------------
# P0-4：**鸭子类型**引擎（有引擎、无 make_fields 属性）—— 旧实现让 create_server 直接抛
# AttributeError（except 只捕 NotImplementedError/RuntimeError）⇒ 整个 server 起不来。
class _NoFieldsEngine:
    """有引擎、**没有** make_fields 的鸭子引擎（P0-4 致死性构造）。"""
    def engine_params(self): return {}
    def _rec(self, n, f, d):
        return {'name': n, 'type': 'float', 'unit': '', 'lo': None, 'hi': None,
                'desc': 'd', 'default': d, 'family': f, 'readonly': False}
    def config_schema(self): return {'x': self._rec('x', 'config', 0.0)}
    def solve_schema(self): return {}
    def solve_map(self): return {}
    def settable_keys(self): return ['x']
    def validate_wave_list(self, v): return v
    def make_cache(self): return object()
    def build_run(self, cache, snap, diag=False):
        return {'method': 'nofields', 'two_body': False}
    def check_run(self, run): return []
    def evolve_run(self, run, progress_arr=None): return {'ok': True}
    def result_summary(self, run, res, **kw): return {'ok': True}
    def summarize_run(self, run): return 'nofields'
    def estimate(self, cache, snap): return {'est_sec': 0.0}
    def preview(self, cache, snap): return {'ok': True}
    def list_levels(self, cache, snap): return []
    def run_annotations(self, snap): return {'n_waves': 0, 'overlap_warning': None}
    # P1-9d 后领域工具**由引擎自报**（库里没有）；本桩**自报**这两个工具但**不给**
    # make_fields ⇒ 正好验"声明了 needs_fields 而无建场层 → 可教学错误、不崩"。
    def mcp_tools(self):
        from src.adapter.mcp_tools import build_tools
        return build_tools()


_old_eng = runtime.engine()
try:
    runtime.set_engine(_NoFieldsEngine())
    _nf_tmp = tempfile.mkdtemp(prefix='mecha_nofields_')
    _nf_sess = Session(root=_nf_tmp)
    try:
        _nf_srv, _nf_tools = SVR.create_server(_nf_sess, Ops(_nf_sess),
                                               actor='ai', log_level='ERROR')
        _nf_crash = None
    except Exception as e:                                # noqa: BLE001
        _nf_crash = '%s: %s' % (type(e).__name__, e)
        _nf_tools = {}
    _sd = json.loads(_nf_tools['set_drive'](reason='r', src='a', tgt='b')) \
        if 'set_drive' in _nf_tools else {}
    _gw = json.loads(_nf_tools['get_waves']()) if 'get_waves' in _nf_tools else {}
    _sd_err = _sd.get('error') or {}
    _gw_err = _gw.get('error') or {}
    # P0-3：numpy 必须**不在** import 期被拉起（jobs 曾模块级 import numpy）
    import mecha.jobs as _mj
    _jobs_src = open(_mj.__file__, encoding='utf-8').read()
    _no_module_numpy = not any(
        l.strip().startswith('import numpy') for l in _jobs_src.splitlines())
finally:
    runtime.set_engine(_old_eng)

C.group('L P0-3/P0-4：numpy 惰性（不在 import 期）+ 无建场层不崩、回可教学错误', [
    ('**致死性(P0-4)**：鸭子引擎（无 make_fields）→ create_server **不抛**（%s）'
     % (_nf_crash or '未抛'),
     _nf_crash is None),
    ('工具面 = 通用 16 + 自报 2（不因缺建场层而少注册）',
     len(_nf_tools) == len(SVR.TOOL_NAMES) + 2),
    ('set_drive 回**可教学**错误（kind+hint+suggest 齐）',
     _sd.get('ok') is False and bool(_sd_err.get('hint'))
     and bool(_sd_err.get('suggest'))),
    ('get_waves 同样可教学（不再是裸 AttributeError）',
     _gw.get('ok') is False and _gw_err.get('kind') != 'AttributeError'
     and bool(_gw_err.get('hint'))),
    ('**致死性(P0-3)**：mecha/jobs.py 无模块级 `import numpy`',
     _no_module_numpy),
])

# ---------------- M. 引擎自报告警 ⇄ 实落事件 的**对拍**（P1-9c 要求 C） ----------------
# 这套设计把"告警内容"从库移到了引擎，于是**新的失效模式**是：
# "引擎自报的 alerts 与 ops 实际落的事件不一致"（少落/多落/字段丢）。
# 对拍放在**产出侧**（cockpit 是透传方，不该由它负责发现产出缺漏）。
from mecha.ops import Ops as _Ops2                     # noqa: E402
from mecha.session import Session as _Session2         # noqa: E402
from mecha import hooks as _hooks                      # noqa: E402


class _AlertEngine:
    """自报 2 条告警（不同 level/rule）+ 1 条框架级 degradation 的引擎。"""
    def engine_params(self): return {}
    def _rec(self, n):
        return {'name': n, 'type': 'float', 'unit': '', 'lo': None, 'hi': None,
                'desc': 'd', 'default': 0.0, 'family': 'config', 'readonly': False}
    def config_schema(self): return {'x': self._rec('x')}
    def solve_schema(self): return {}
    def solve_map(self): return {}
    def settable_keys(self): return ['x']
    def validate_wave_list(self, v): return v
    def make_cache(self): return object()
    def build_run(self, cache, snap, diag=False):
        return {'method': 'alerty', 'two_body': False, 'spec': dict(snap)}
    def check_run(self, run): return ['框架级：自检发现一处降级']
    def evolve_run(self, run, progress_arr=None): return {'ok': True}
    def result_summary(self, run, res, **kw): return {'ok': True}
    def summarize_run(self, run): return 'alerty'
    def estimate(self, cache, snap): return {'est_sec': 0.0}
    def preview(self, cache, snap): return {'ok': True}
    def list_levels(self, cache, snap): return []
    def run_meta(self, run): return {'method': run['method'], 'body': 'single'}
    def annotations(self, snap): return {'n_waves': 0, 'overlap_warning': None}
    def alerts(self, run, summary, annotations):
        return [{'rule': 'rule_a', 'level': 'warn', 'reason': '甲'},
                {'rule': 'rule_b', 'level': 'error', 'reason': '乙'}]


_old_e3 = runtime.engine()
try:
    runtime.set_engine(_AlertEngine())
    _a_tmp = tempfile.mkdtemp(prefix='mecha_alert_')
    _a_sess = _Session2(root=_a_tmp)
    _a_ops = _Ops2(_a_sess)
    _prep = _a_ops.prepare(actor='ai', reason='对拍', land_artifact=False)
    _ann = dict(_hooks.annotations(_AlertEngine(), _a_sess.effective()))
    _ann.setdefault('check_problems', list(_prep.get('problems') or []))
    _expect = _hooks.alerts(_AlertEngine(), _prep['run'], {'ok': True}, _ann)
    _a_ops.execute(_prep, actor='ai', reason='对拍', land_artifact=False)
    _evs = [e for e in _a_sess.activity(0) if e['kind'] == 'alert']
    _pairs = [((e.get('after') or {}).get('rule'),
               (e.get('provenance') or {}).get('level'), e.get('reason'))
              for e in _evs]
    _exp_pairs = [(a['rule'], a['level'], a['reason']) for a in _expect]
finally:
    runtime.set_engine(_old_e3)

C.group('M 引擎自报告警 ⇄ 实落 alert 事件**逐项对拍**（P1-9c 要求 C；含致死性）', [
    ('自报 %d 条 ⇒ 实落 %d 条（数量一致）' % (len(_expect), len(_evs)),
     len(_evs) == len(_expect)),
    ('rule/level/reason **逐项**传递（实落 == 自报）', _pairs == _exp_pairs),
    ('引擎自报的两条都在（rule_a/rule_b）',
     {p[0] for p in _pairs} >= {'rule_a', 'rule_b'}),
    ('level 分档正确（warn/error 各就位）',
     {p[0]: p[1] for p in _pairs}.get('rule_b') == 'error'),
    ('**框架级 degradation 与引擎自报合并**（check_run 的问题不被吞）',
     'degradation' in {p[0] for p in _pairs}),
    ('**致死性构造**：若少落一条，数量断言必红（%d != %d）'
     % (len(_evs), len(_expect) - 1), len(_evs) != len(_expect) - 1),
])

# ---------------- N. est_sec：**"键缺失"与"0 秒"必须可区分**（P0-5 的静默退化守卫） ----------------
# 旧写法 `estimate().get('est_sec') or 0.0` 把两者合并 ⇒ 引擎按别的键名给（或没给）时
# auto→async **永不触发**、长任务静默同步阻塞写通道。这条必须钉住，否则下次重构
# 又会退化回 `or 0.0`。
class _NoEstEngine(_AlertEngine):
    """estimate **不给** est_sec/sec 的引擎（模拟"键缺失"）。"""
    def estimate(self, cache, snap): return {'backend': 'sim'}


_old_e4 = runtime.engine()
try:
    runtime.set_engine(_NoEstEngine())
    _ne_tmp = tempfile.mkdtemp(prefix='mecha_noest_')
    _ne_sess = _Session2(root=_ne_tmp)
    _ne_srv, _ne_tools = SVR.create_server(_ne_sess, _Ops2(_ne_sess),
                                           actor='ai', log_level='ERROR')
    _ne_r = json.loads(_ne_tools['run_compute'](reason='缺 est_sec',
                                                mode='auto'))
    # 对照：est_sec=0.0（**有键**，只是值为 0）且阈值压低 ⇒ 应能转 async
    runtime.set_engine(_AlertEngine())
    _z_tmp = tempfile.mkdtemp(prefix='mecha_zero_')
    _z_sess = _Session2(root=_z_tmp)
    _z_srv, _z_tools = SVR.create_server(_z_sess, _Ops2(_z_sess),
                                         actor='ai', log_level='ERROR')
    _old_th2 = SVR.ASYNC_THRESHOLD_SEC
    SVR.ASYNC_THRESHOLD_SEC = -1.0
    try:
        _z_r = json.loads(_z_tools['run_compute'](reason='est_sec=0',
                                                  mode='auto'))
    finally:
        SVR.ASYNC_THRESHOLD_SEC = _old_th2
finally:
    runtime.set_engine(_old_e4)

C.group('N est_sec：**键缺失**与**0 秒**可区分（防 `or 0.0` 静默退化）', [
    ('键缺失 → **不静默**：回执带 `estimate_note` 说明原因',
     bool(_ne_r.get('estimate_note'))),
    ('键缺失 → 未误判成 async（按同步执行）', not _ne_r.get('async', False)),
    ('对照：est_sec=0.0（**有键**）且阈值压低 → **能**转 async',
     _z_r.get('async') is True and bool(_z_r.get('job_id'))),
    ('**致死性**：若退回 `or 0.0`，"键缺失"会被当成 0 而不再报 note（本断言必红）',
     bool(_ne_r.get('estimate_note'))),
])

sys.exit(0 if C.summary() else 1)
