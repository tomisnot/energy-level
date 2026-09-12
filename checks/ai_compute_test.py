# -*- coding: utf-8 -*-
"""AI 计算通道端到端测试：load_archive → set_param → run_compute → job 轮询
→ job_result，断言 GUI 同步反映、结果摘要正确。

用法: python -u checks/ai_compute_test.py
（首次跑含 numba JIT 编译，可能较久）
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

from PyQt5.QtWidgets import QApplication, QMessageBox
QMessageBox.information = staticmethod(lambda *a, **k: None)
QMessageBox.warning = staticmethod(lambda *a, **k: None)

import app_entry as main_mod

PASSES, FAILS = [], []
DONE = {}


def check(name, cond):
    (PASSES if cond else FAILS).append(name)
    print(f'  {"PASS" if cond else "FAIL"}  {name}', flush=True)


async def _drive(port, win):
    from mcp import ClientSession
    from mcp.client.streamable_http import streamable_http_client
    async with streamable_http_client(f'http://127.0.0.1:{port}/mcp') as (r, w):
        async with ClientSession(r, w) as s:
            await s.initialize()

            async def call(name, **kw):
                res = await s.call_tool(name, kw)
                return json.loads(res.content[0].text)

            arch = os.path.join('save_state', '微波驱动.json')
            out = await call('load_archive', path=arch)
            check('load_archive ok', out.get('ok'))
            check('GUI 组数=3', len(win.groups_store.groups) == 3)

            # 缩短计算规模（AI 改参数 → 控件反映）
            out = await call('set_param', path='dynamic.t_max', value=200.0,
                             reason='自检缩短 t_max')
            check('set t_max ok', out.get('ok'))
            check('控件 t_max=200', abs(win.dynamic.tmax_sb.value() - 200.0) < 1e-6)
            await call('set_param', path='dynamic.npts', value=41,
                       reason='自检降点数')
            check('控件 npts=41', win.dynamic.npts_sb.value() == 41)

            t0 = time.time()
            out = await call('run_compute')
            check('run_compute 启动', out.get('ok'))
            if not out.get('ok'):
                print('   msg:', out.get('msg'))
                return
            jid = out['job_id']
            check('GUI 进入运行态', bool(win.dynamic._run_active))

            st = {}
            while time.time() - t0 < 420:
                st = await call('job_status', job_id=jid)
                if st.get('done'):
                    break
                await asyncio.sleep(1.0)
            check('计算完成', bool(st.get('done')))
            check('无错误', st.get('error') is None)
            print(f'   耗时 {time.time() - t0:.1f}s，'
                  f'状态 {st}', flush=True)
            check('GUI 退出运行态', not getattr(win.dynamic, '_run_active', True))

            res = await call('job_result', job_id=jid, save_npz=True)
            check('job_result ok', res.get('ok'))
            summ = res.get('summary', {})
            check('摘要含 top_states', bool(summ.get('top_states_avg_pop')))
            check('摘要含 npz 路径', bool(summ.get('npz')))
            if summ.get('npz'):
                check('npz 文件存在', os.path.exists(summ['npz']))
            print('   摘要:', json.dumps(summ, ensure_ascii=False)[:400], flush=True)

            # 人侧接管验证：AI 静默模式已还原，人再跑仍走正常路径
            check('进度框钩子已还原',
                  callable(getattr(win.dynamic, '_show_progress')))

            act = await call('get_activity', since_seq=0)
            kinds = [o['kind'] for o in act['ops'] if o['actor'] == 'ai']
            check('journal 含 run_compute', 'run_compute' in kinds)
            check('journal 含 load_archive', 'load_archive' in kinds)


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
    _need_ai(port, 'ai_compute_test.py')
    if not port:
        print('AI server 未启动')
        return 1
    t = threading.Thread(target=_thread, args=(port, win), daemon=True)
    t.start()
    t0 = time.time()
    while not DONE.get('x') and time.time() - t0 < 480:
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
