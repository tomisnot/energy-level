# -*- coding: utf-8 -*-
"""GUI 接线自检（t79）：mixed + 任意波形微波**不再改道 hard**。

用途: python -u checks/gui_arb_mixed_wiring_test.py

为什么这么测（检测器很关键）：
  改道 hard 时，GUI 会把 method 置为 'hard' 并跑 evolve_hard_pair；另一侧
  也跑 evolve_hard_pair 且输入完全相同 → **两次结果逐位相同（diff ≡ 0）**。
  而走新通道（mixed 自己的 cos(φ(t)) 路径）时两者只应一致到 ~1e-6。
  所以：
      diff == 0            → 仍然改道了（接线失败）
      0 < diff < 1e-4      → 走的是 mixed 新通道，且物理与 hard 一致 ✓

走 MCP 工具触发 run_compute，正是 GUI 的 _run 路径（含本节改动的分支）。
"""
import asyncio
import json
import os
import sys
import threading
import time
import traceback

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ['QT_QPA_PLATFORM'] = 'offscreen'
os.environ['EL_MCP_PORT'] = '18768'
from PyQt5.QtWidgets import QApplication, QMessageBox

QMessageBox.information = staticmethod(lambda *a, **k: None)
QMessageBox.warning = staticmethod(lambda *a, **k: None)

import app_entry as main_mod

PASSES, FAILS = [], []
DONE = {}
NU_MHZ = 17236.743          # 微波驱动.json 的 60S(+0.5) → 60P(-0.5) 载频
T_ARB, DT_ARB = 40.0, 0.01  # dt=0.01ns → Nyquist 50 GHz > 17.2 GHz


def check(name, cond, detail=''):
    (PASSES if cond else FAILS).append(name)
    print(f'  {"PASS" if cond else "FAIL"}  {name}'
          + (f'   {detail}' if not cond and detail else ''), flush=True)


def _arb_payload():
    tn = np.arange(0.0, T_ARB + 1e-9, DT_ARB)
    ph = 2.0 * np.pi * 1e-3 * NU_MHZ * tn
    amp = np.sin(np.pi * tn / T_ARB) ** 2 * np.cos(ph)
    return {'type': 'arb', 'name': '任意微波', 't_ns': tn.tolist(),
            'amp': amp.tolist(), 'pol': 'σ-', 'intensity': 5.5e-5,
            'offset_ns': 0.0}


async def _drive(port):
    from mcp import ClientSession
    from mcp.client.streamable_http import streamable_http_client

    async with streamable_http_client(f'http://127.0.0.1:{port}/mcp') as (r, w):
        async with ClientSession(r, w) as s:
            await s.initialize()

            async def call(name, **kw):
                res = await s.call_tool(name, kw)
                return json.loads(res.content[0].text)

            out = await call('load_archive',
                             path=os.path.join('save_state', '微波驱动.json'))
            check('load_archive ok', out.get('ok'))
            await call('set_param', path='dynamic.mode', value=1)
            await call('set_param', path='dynamic.t_max', value=T_ARB)
            await call('set_param', path='dynamic.npts', value=41)
            payload = _arb_payload()
            rw = await call('set_waves', waves=[payload], atom='A')
            check('任意波形已写入 A（type=arb）', rw.get('ok'), str(rw)[:160])
            await call('set_waves', waves=[], atom='B')

            async def run_once(method):
                await call('set_param', path='dynamic.method', value=method)
                jb = await call('run_compute')
                if not jb.get('ok'):
                    return None, jb
                jid = jb.get('job_id')
                st = None
                for _ in range(1200):
                    st = await call('job_status', job_id=jid)
                    if st.get('done'):
                        break
                    await asyncio.sleep(0.5)
                jr = await call('job_result', job_id=jid, save_npz=True)
                return jr, st

            jm, stm = await run_once('mixed')
            check('mixed 跑通（未因任意波形报错）',
                  bool(jm and jm.get('ok')), str(stm)[:200])
            jh, sth = await run_once('hard')
            check('hard 跑通', bool(jh and jh.get('ok')), str(sth)[:200])
            if not (jm and jm.get('ok') and jh and jh.get('ok')):
                return

            pm = (jm.get('summary') or {}).get('npz')
            ph = (jh.get('summary') or {}).get('npz')
            check('两侧 npz 均已落盘', bool(pm) and bool(ph), f'{pm} / {ph}')
            if not (pm and ph):
                return
            A = np.load(pm, allow_pickle=True)['pops_2d']
            B = np.load(ph, allow_pickle=True)['pops_2d']
            check('两侧形状一致', A.shape == B.shape, f'{A.shape} vs {B.shape}')
            d = float(np.abs(A - B).max())
            amp = float(np.ptp(A[:, 0])) + float(np.ptp(A).max())
            check('确有动力学（防空洞对照）', amp > 1e-3, f'幅度量级 {amp:.3e}')
            check('mixed 未改道 hard（否则 diff≡0）', d > 0.0, f'diff={d:.3e}')
            check('mixed ≡ hard（物理一致，<1e-4）', d < 1e-4, f'diff={d:.3e}')
            print(f'    mixed vs hard  max|Δpops| = {d:.3e}', flush=True)


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
    _need_ai(port, 'gui_arb_mixed_wiring_test.py')
    check('固定端口 18768', port == 18768, str(port))
    if not port:
        return 1
    t = threading.Thread(target=_thread, args=(port,), daemon=True)
    t.start()
    t0 = time.time()
    while not DONE.get('x') and time.time() - t0 < 900:
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
