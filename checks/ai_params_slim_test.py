# -*- coding: utf-8 -*-
"""自检：MCP 侧两项改造 —— ①波形包络瘦身/独立通道 ②两体结果标签。

用途: python -u checks/ai_params_slim_test.py

要钉住的两件事：
  ① `list_params` / `get_state` **不得整表回传包络采样**（实测一个 12001 点
     FileEnv 让单次响应到 1 MB）；完整曲线走 `get_envelope(path, index)`。
  ② 两体结果（pops_2d，22²=484 维）的标签必须来自 `states_2d` 的**成对标签**，
     不能照搬 22 个单原子标签 —— 否则 484 维布居被标成单原子态、索引 ≥22
     退化成裸数字（如 "247"），读数被误导。
"""
import asyncio
import json
import os
import sys
import threading
import time
import traceback

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ['QT_QPA_PLATFORM'] = 'offscreen'
os.environ['EL_MCP_PORT'] = '18767'
from PyQt5.QtWidgets import QApplication, QMessageBox

QMessageBox.information = staticmethod(lambda *a, **k: None)
QMessageBox.warning = staticmethod(lambda *a, **k: None)

import app_entry as main_mod

PASSES, FAILS = [], []
DONE = {}
NPTS = 501


def check(name, cond, detail=''):
    (PASSES if cond else FAILS).append(name)
    print(f'  {"PASS" if cond else "FAIL"}  {name}'
          + (f'   {detail}' if not cond and detail else ''), flush=True)


async def _drive(port):
    from mcp import ClientSession
    from mcp.client.streamable_http import streamable_http_client

    async with streamable_http_client(f'http://127.0.0.1:{port}/mcp') as (r, w):
        async with ClientSession(r, w) as s:
            await s.initialize()
            names = {t.name for t in (await s.list_tools()).tools}
            check('get_envelope 已注册', 'get_envelope' in names)

            async def call(name, **kw):
                res = await s.call_tool(name, kw)
                return json.loads(res.content[0].text)

            await call('load_archive', path=os.path.join('save_state', '微波驱动.json'))
            await call('set_param', path='dynamic.t_max', value=60.0)
            await call('set_param', path='dynamic.npts', value=61)

            # ---- 造一条 NPTS 点的 file 型包络 ----
            t = [i * 1200.0 / (NPTS - 1) for i in range(NPTS)]
            amp = [1.0 + 0.05 * (i % 7) / 6.0 for i in range(NPTS)]
            wv = [{'name': '瘦身自检', 'freq_mhz': 295724187.476,
                   'intensity': 8.3982e7, 'pol': 'π', 'offset_ns': 0.0,
                   'env': ['file', t, amp], 'drive': None}]
            await call('set_waves', waves=wv, atom='A')
            await call('set_waves', waves=[], atom='B')

            # ---- ① list_params 只给摘要 ----
            # 注意层级：list_params 返回 {参数名: {schema..., 'value': ...}}；
            # get_state 返回 {参数名: 值}。
            r1 = await call('list_params', prefix='dynamic.waves')
            ps = r1.get('params') or {}
            check('list_params 含 dynamic.waves_A',
                  'dynamic.waves_A' in ps, str(sorted(ps)[:8]))
            rec = ps.get('dynamic.waves_A') or {}
            ws = rec.get('value') or []
            check('waves_A 有 1 条波形', len(ws) == 1, str(len(ws)))
            env = (ws[0] or {}).get('env') if ws else None
            check('file 包络被换成摘要', isinstance(env, list) and len(env) == 2
                  and isinstance(env[1], dict), str(env)[:200])
            if isinstance(env, list) and len(env) == 2 and isinstance(env[1], dict):
                sm = env[1]
                check('摘要含点数 n', sm.get('n') == NPTS, str(sm.get('n')))
                check('摘要含 dt/跨度/幅值范围',
                      all(k in sm for k in ('dt_min_ns', 't_span_ns', 'amp_range')))
            size1 = len(json.dumps(r1, ensure_ascii=False))
            check(f'list_params 响应体不膨胀（{size1} B < 20000）', size1 < 20000,
                  str(size1))

            # ---- get_state 同样瘦身 ----
            r2 = await call('get_state', scope='dynamic')
            ws2 = (r2.get('params') or {}).get('dynamic.waves_A') or []
            e2 = (ws2[0] or {}).get('env') if ws2 else None
            check('get_state 的包络也是摘要', isinstance(e2, list) and len(e2) == 2
                  and isinstance(e2[1], dict), str(e2)[:200])
            size2 = len(json.dumps(r2, ensure_ascii=False))
            check(f'get_state 响应体不膨胀（{size2} B < 30000）', size2 < 30000,
                  str(size2))

            # ---- get_envelope 取全量 ----
            r3 = await call('get_envelope', path='dynamic.waves_A', index=0)
            check('get_envelope ok', r3.get('ok'), str(r3)[:200])
            check('get_envelope 返回等长 t/amp',
                  isinstance(r3.get('t_ns'), list) and isinstance(r3.get('amp'), list)
                  and len(r3['t_ns']) == len(r3['amp']) == NPTS,
                  f"{len(r3.get('t_ns') or [])} / {len(r3.get('amp') or [])}")
            # FileEnv 会**峰值归一化**（见 SKILL.md 语义陷阱），故取回的是
            # 归一化后的采样；这里按同规则比对，顺带钉住该语义。
            mx = max(amp)
            check('get_envelope 采样 = 峰值归一后的值',
                  abs(r3['amp'][0] - amp[0] / mx) < 1e-9
                  and abs(r3['amp'][-1] - amp[-1] / mx) < 1e-9,
                  f"{r3['amp'][0]} vs {amp[0] / mx}")
            r4 = await call('get_envelope', path='dynamic.waves_A', index=9)
            check('越界 index 被拒且提示总数', r4.get('ok') is False
                  and '共 1 条' in (r4.get('msg') or ''), str(r4)[:160])

            # ---- ② 两体结果标签 ----
            await call('set_param', path='dynamic.method', value='hard')
            await call('set_param', path='dynamic.mode', value=1)
            await call('set_param', path='dynamic.psi_A',
                       value=['mJ', 60, 1, 1.5, -0.5, -999975.847187])
            await call('set_param', path='dynamic.psi_B',
                       value=['mJ', 60, 0, 0.5, 0.5, -1017212.590406])
            await call('set_param', path='dynamic.t_max', value=20.0)
            await call('set_param', path='dynamic.npts', value=21)
            jb = await call('run_compute')
            check('两体计算已启动', jb.get('ok'), str(jb)[:160])
            jid = jb.get('job_id')
            for _ in range(160):
                st = await call('job_status', job_id=jid)
                if st.get('done'):
                    break
                await asyncio.sleep(0.5)
            check('两体计算完成', st.get('done'), str(st)[:160])
            jr = await call('job_result', job_id=jid)
            check('job_result ok', jr.get('ok'), str(jr)[:200])
            sm2 = jr.get('summary') or {}
            check('summary 标 two_body', sm2.get('two_body') is True,
                  str(sm2.get('two_body')))
            check('两体基维度 484', sm2.get('n_states') == 484,
                  str(sm2.get('n_states')))
            keys = list((sm2.get('final_pop') or {}).keys())
            check('final_pop 键是成对标签（含 " x "）',
                  bool(keys) and all(' x ' in k for k in keys), str(keys))
            check('无裸数字键（旧 bug 特征）',
                  all(not k.lstrip('#').isdigit() for k in keys), str(keys))
            sk = list((sm2.get('states') or [])[:3])
            check('states 前几项是成对标签',
                  bool(sk) and all(' x ' in x for x in sk), str(sk))


def _thread(port):
    try:
        asyncio.run(_drive(port))
    except Exception:
        FAILS.append('客户端异常:\n' + traceback.format_exc())
        print(traceback.format_exc(), flush=True)
    finally:
        DONE['x'] = True


def main():
    app = QApplication(sys.argv)
    win = main_mod.MainWindow()
    win.show()
    app.processEvents()
    port = win._init_ai()
    from checks.checklib import (require_ai_server as _need_ai)
    _need_ai(port, 'ai_params_slim_test.py')
    check('固定端口 18767', port == 18767, str(port))
    if not port:
        return 1
    t = threading.Thread(target=_thread, args=(port,), daemon=True)
    t.start()
    t0 = time.time()
    while not DONE.get('x') and time.time() - t0 < 600:
        app.processEvents()
        time.sleep(0.02)
    t.join(timeout=2)
    print(f'\n通过 {len(PASSES)} 项')
    if FAILS:
        print(f'失败 {len(FAILS)} 项:')
        for f in FAILS:
            print('  -', f)
        return 1
    print('全部通过 ✓')
    return 0


if __name__ == '__main__':
    sys.exit(main())
