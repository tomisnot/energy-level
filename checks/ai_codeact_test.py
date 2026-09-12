# -*- coding: utf-8 -*-
"""CodeAct 通道端到端测试：run_python 一次调用完成"改参数→跑计算→扫描→
汇总"，断言 GUI 反映、journal 整段一步可 undo、result 回传正确。

用法: python -u checks/ai_codeact_test.py
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
os.environ['EL_MCP_PORT'] = '18765'      # 固定端口验证（层1）

from PyQt5.QtWidgets import QApplication, QMessageBox
QMessageBox.information = staticmethod(lambda *a, **k: None)
QMessageBox.warning = staticmethod(lambda *a, **k: None)

import app_entry as main_mod

PASSES, FAILS = [], []
DONE = {}


def check(name, cond):
    (PASSES if cond else FAILS).append(name)
    print(f'  {"PASS" if cond else "FAIL"}  {name}', flush=True)


CODE = '''
out = []
for B in [15.0, 25.0]:
    r0 = set_param('field.B_gauss', B, reason=f'CodeAct 扫描 B={B}')
    r = run_compute_and_wait(timeout=240)
    fp = (r.get('summary') or {}).get('final_pop') or {}
    top = sorted(fp.items(), key=lambda kv: -kv[1])[:1]
    out.append({'B': B, 'set_ok': r0['ok'], 'done': r.get('ok'),
                'top_final': top})
    print(B, out[-1])
result = out
'''


async def _drive(port, win):
    from mcp import ClientSession
    from mcp.client.streamable_http import streamable_http_client
    async with streamable_http_client(f'http://127.0.0.1:{port}/mcp') as (r, w):
        async with ClientSession(r, w) as s:
            await s.initialize()
            names = {t.name for t in (await s.list_tools()).tools}
            check('run_python 已注册', 'run_python' in names)

            async def call(name, **kw):
                res = await s.call_tool(name, kw)
                return json.loads(res.content[0].text)

            # 准备场景：微波驱动存档（hard 方法、小参数）
            out = await call('load_archive',
                             path=os.path.join('save_state', '微波驱动.json'))
            check('load_archive ok', out.get('ok'))
            await call('set_param', path='dynamic.t_max', value=100.0)
            await call('set_param', path='dynamic.npts', value=21)

            t0 = time.time()
            r = await call('run_python', code=CODE, timeout=600)
            print(f'   run_python 耗时 {time.time() - t0:.1f}s', flush=True)
            check('run_python ok', r.get('ok'))
            if not r.get('ok'):
                print('   error:', r.get('error'))
                print('   stdout:', r.get('stdout'))
                return
            res = r.get('result')
            check('result 两轮扫描', isinstance(res, list) and len(res) == 2)
            check('两轮均 set_ok+done',
                  all(x['set_ok'] and x['done'] for x in res))
            check('stdout 回传', '15.0' in r.get('stdout', ''))
            check('GUI B 场=25（末轮）',
                  abs(win.field.b_spin.value() - 25.0) < 1e-6)

            act = await call('get_activity', since_seq=0)
            rp = [o for o in act['ops'] if o['kind'] == 'run_python']
            check('journal 整段一条 run_python', len(rp) == 1)

            u = await call('undo', reason='撤掉 CodeAct 整段')
            check('undo 整段回滚', u.get('ok'))
            check('B 场回到脚本前(21.7)',
                  abs(win.field.b_spin.value() - 21.7) < 1e-6)

            # 失败路径：代码抛异常 → ok=False + error，快照仍记录
            r2 = await call('run_python', code='x = 1/0', timeout=30)
            check('异常捕获 ok=False', r2.get('ok') is False)
            check('异常含 traceback', 'ZeroDivisionError' in r2.get('error', ''))


def _thread(port, win):
    try:
        asyncio.run(_drive(port, win))
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
    _need_ai(port, 'ai_codeact_test.py')
    check('固定端口 18765', port == 18765)
    if not port:
        return 1
    t = threading.Thread(target=_thread, args=(port, win), daemon=True)
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
