# -*- coding: utf-8 -*-
"""run_store × MCP 自检：run_python_async / run_result / list_runs 三件套。

用途: python -u checks/ai_runstore_mcp_test.py

要钉住的核心性质（这是加这三个工具的唯一理由）：
  run_python 的脚本在 Qt 主线程 exec，所以长任务执行期间**任何经
  run_on_main 的工具都会被排队**。run_result 只读 runtime/ai_results/ 下的文件、
  不经主线程，因此必须能在任务飞行中立刻作答。本自检会在异步任务执行
  期间反复轮询，要求至少拿到一次 status='running' 且响应及时——若日后
  有人给 run_result 加上主线程依赖，这条断言会立刻失败。
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
os.environ['EL_MCP_PORT'] = '18766'      # 固定端口，与 ai_codeact_test(18765) 分开
from PyQt5.QtWidgets import QApplication, QMessageBox

QMessageBox.information = staticmethod(lambda *a, **k: None)
QMessageBox.warning = staticmethod(lambda *a, **k: None)

import app_entry as main_mod

PASSES, FAILS = [], []
DONE = {}

CODE = '''
out = []
for B in [15.0, 20.0, 25.0]:
    r0 = set_param('field.B_gauss', B, reason=f'async 自检 B={B}')
    r = run_compute_and_wait(timeout=240)
    out.append({'B': B, 'set_ok': r0['ok'], 'done': r.get('ok')})
    print(B, out[-1], flush=True)
result = out
'''


def check(name, cond, detail=''):
    (PASSES if cond else FAILS).append(name)
    print(f'  {"PASS" if cond else "FAIL"}  {name}' + (f'   {detail}' if not cond and detail else ''),
          flush=True)


async def _drive(port):
    from mcp import ClientSession
    from mcp.client.streamable_http import streamable_http_client

    async with streamable_http_client(f'http://127.0.0.1:{port}/mcp') as (r, w):
        async with ClientSession(r, w) as s:
            await s.initialize()
            names = {t.name for t in (await s.list_tools()).tools}
            for t in ('run_python', 'run_python_async', 'run_result', 'list_runs'):
                check(f'工具已注册：{t}', t in names)

            async def call(name, **kw):
                res = await s.call_tool(name, kw)
                return json.loads(res.content[0].text)

            out = await call('load_archive',
                             path=os.path.join('save_state', '微波驱动.json'))
            check('load_archive ok', out.get('ok'))
            await call('set_param', path='dynamic.t_max', value=100.0)
            await call('set_param', path='dynamic.npts', value=21)

            # ---- 同步 run_python 也应带 run_id ----
            r = await call('run_python', code='result = 1 + 1', timeout=60)
            check('同步 run_python 返回 run_id', bool(r.get('run_id')), str(r)[:160])
            sync_rid = r.get('run_id')
            rr = await call('run_result', run_id=sync_rid)
            check('同步 run 的记录可取回', rr.get('status') == 'ok'
                  and rr.get('result') == 2, str(rr)[:200])

            # ---- 异步：启动应立刻返回 ----
            t0 = time.time()
            a = await call('run_python_async', code=CODE, timeout=600)
            launch_s = time.time() - t0
            check('run_python_async ok', a.get('ok'), str(a)[:200])
            rid = a.get('run_id')
            check('异步返回 run_id', bool(rid))
            check('异步启动立即返回（<3s，不等脚本跑完）', launch_s < 3.0,
                  f'{launch_s:.2f}s')

            # ---- 飞行中轮询：这是本自检存在的意义 ----
            # 预算必须长于最慢用例：实测这个 3 轮扫描约 38 s，曾因只给
            # 120×0.25 s≈36 s 而"差 2 秒没等到"，误报 4 项失败。
            n_running, worst_lat, n_poll = 0, 0.0, 0
            deadline = time.time() + 240
            while time.time() < deadline:
                tp = time.time()
                st = await call('run_result', run_id=rid)
                worst_lat = max(worst_lat, time.time() - tp)
                n_poll += 1
                if st.get('status') == 'running':
                    n_running += 1
                if st.get('status') in ('ok', 'error'):
                    break
                await asyncio.sleep(0.5)

            check('任务飞行中能读到 status=running', n_running > 0,
                  f'n_running={n_running} / n_poll={n_poll}')
            check('飞行中查询响应及时（不经主线程）', worst_lat < 2.0,
                  f'worst={worst_lat:.2f}s')
            check('轮询最终等到终态', st.get('status') in ('ok', 'error'),
                  f'status={st.get("status")} n_poll={n_poll}')
            check('最终 status=ok', st.get('status') == 'ok', str(st)[:200])
            check('取回 result（两轮扫描）',
                  isinstance(st.get('result'), list) and len(st['result']) == 3,
                  str(st.get('result'))[:200])
            check('elapsed_s 已记', (st.get('elapsed_s') or 0) > 0,
                  str(st.get('elapsed_s')))
            check('stdout 已存', '15.0' in (st.get('stdout') or ''))

            # ---- 未知 run_id 的报错要可指导 ----
            bad = await call('run_result', run_id='run_不存在')
            check('未知 run_id → ok=False 且有指引',
                  bad.get('ok') is False and 'list_runs' in (bad.get('msg') or ''),
                  str(bad)[:200])

            # ---- list_runs 能找回 ----
            ls = await call('list_runs', limit=10)
            ids = [x.get('run_id') for x in ls.get('runs', [])]
            check('list_runs 含刚跑的异步 run', rid in ids, str(ids[:5]))
            check('list_runs 概览不夹带 result/stdout',
                  all('result' not in x and 'stdout' not in x
                      for x in ls.get('runs', [])))
            check('list_runs 给出目录', bool(ls.get('dir')), str(ls.get('dir')))


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
    _need_ai(port, 'ai_runstore_mcp_test.py')
    check('固定端口 18766', port == 18766, str(port))
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
