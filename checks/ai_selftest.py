# -*- coding: utf-8 -*-
"""AI 协同端到端自检：离屏起主窗口 + MCP server，用真实 MCP 客户端驱动工具，
断言 GUI 反映 AI 操作（人机等效 P2）、journal 归因正确（P3）、undo 可逆。

用法: python -u checks/ai_selftest.py
退出码 0 = 全绿。
"""
import asyncio
import os
import sys
import threading
import traceback

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ['QT_QPA_PLATFORM'] = 'offscreen'

from PyQt5.QtWidgets import QApplication, QMessageBox
QMessageBox.information = staticmethod(lambda *a, **k: None)
QMessageBox.warning = staticmethod(lambda *a, **k: None)

import app_entry as main_mod

RESULTS = {}
PASSES = []
FAILS = []


def check(name, cond):
    (PASSES if cond else FAILS).append(name)
    print(f'  {"PASS" if cond else "FAIL"}  {name}')


async def _drive(port):
    from mcp import ClientSession
    from mcp.client.streamable_http import streamable_http_client
    url = f'http://127.0.0.1:{port}/mcp'
    async with streamable_http_client(url) as (read, write):
        async with ClientSession(read, write) as s:
            await s.initialize()
            tools = {t.name for t in (await s.list_tools()).tools}
            check('list_tools 齐全',
                  {'get_state', 'set_param', 'add_level_group', 'run_compute',
                   'get_activity', 'undo', 'list_params'} <= tools)

            async def call(name, **kw):
                r = await s.call_tool(name, kw)
                import json
                return json.loads(r.content[0].text)

            # --- 读 ---
            summ = await call('summary')
            check('summary 有 n_groups', 'n_groups' in summ)

            # --- 写：加能级组 → GUI 反映 ---
            win = QApplication.topLevelWidgets()
            win = next(w for w in win if isinstance(w, main_mod.MainWindow))
            before = len(win.groups_store.groups)
            r = await call('add_level_group', n=60, l=0, j=0.5,
                           reason='自检加 60S')
            check('add_level_group ok', r.get('ok'))
            check('GUI 能级组数 +1', len(win.groups_store.groups) == before + 1)

            # --- 写：改 B → 控件反映（人机同一路径） ---
            r = await call('set_param', path='field.B_gauss', value=33.3,
                           reason='自检改 B')
            check('set_param ok', r.get('ok'))
            check('B 控件=33.3', abs(win.field.b_spin.value() - 33.3) < 1e-6)

            # --- journal 归因：至少 2 条 ai op，reason 落库 ---
            act = await call('get_activity', since_seq=0)
            ai_ops = [o for o in act['ops'] if o['actor'] == 'ai']
            check('journal 记 ai op>=2', len(ai_ops) >= 2)
            check('journal 带 reason',
                  any(o['reason'] == '自检改 B' for o in ai_ops))

            # --- undo：撤销 set_param → B 回到旧值 ---
            old_b = 21.7
            r = await call('undo', reason='自检撤销 B')
            check('undo ok', r.get('ok'))
            check('B 回滚', abs(win.field.b_spin.value() - old_b) < 1e-6)

            # --- list_params 可读 ---
            lp = await call('list_params')
            check('list_params 有 B', 'field.B_gauss' in lp.get('params', {}))


def _client_thread(port):
    try:
        asyncio.run(_drive(port))
    except Exception:
        FAILS.append('客户端异常:\n' + traceback.format_exc())
    finally:
        RESULTS['__done__'] = True


def main():
    app = QApplication(sys.argv)
    win = main_mod.MainWindow()
    win.resize(1200, 800)
    win.show()
    app.processEvents()
    port = win._init_ai()
    from checks.checklib import (require_ai_server as _need_ai)
    _need_ai(port, 'ai_selftest.py')
    if not port:
        print('FAIL: AI server 未启动')
        return 1
    print(f'MCP 端点: http://127.0.0.1:{port}/mcp')
    t = threading.Thread(target=_client_thread, args=(port,), daemon=True)
    t.start()

    # 主线程泵事件循环（run_on_main 依赖），最多等 25s
    import time
    t0 = time.time()
    while not RESULTS.get('__done__') and time.time() - t0 < 25:
        app.processEvents()
        time.sleep(0.02)
    t.join(timeout=2)

    print(f'\n通过 {len(PASSES)} 项')
    if FAILS:
        print(f'失败 {len(FAILS)} 项:')
        for f in FAILS:
            print('  -', f)
        return 1
    if RESULTS.get('__done__'):
        print('全部通过 ✓')
        return 0
    print('超时未完成')
    return 1


if __name__ == '__main__':
    sys.exit(main())
