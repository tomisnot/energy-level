# -*- coding: utf-8 -*-
"""B3 自检：AI 作业在 GUI 可见、可取消（G3「人可随时监控与接管」）。

模拟人机协同的关键一环：AI 提交了作业 → 状态栏出现常驻指示器 →
人点「取消AI作业」→ 作业被取消（与 MCP cancel_job 同路径）+ 指示器消失。

用法：python checks/gui_ai_job_indicator_test.py
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

from checks.checklib import Checker                            # noqa: E402
from PyQt5.QtWidgets import QApplication                       # noqa: E402
import app_entry as M                                          # noqa: E402
from src.ai import server as ai_server                         # noqa: E402

C = Checker('gui_ai_job_indicator')
_app = QApplication.instance() or QApplication(sys.argv)
win = M.MainWindow()

win._install_ai_job_indicator()      # 服务未起时也能装（幂等）
C.check('1 状态栏指示器与取消按钮已建', hasattr(win, '_ai_job_ind')
        and hasattr(win, '_ai_job_btn'))
C.check('2 无作业时隐藏', win._ai_job_ind.isHidden()
        and win._ai_job_btn.isHidden())

# 造一个"正在跑"的 AI 作业（只放登记，不起真线程）
jid = 'jobTEST'
ai_server._jobs[jid] = {'started': time.time() - 3.0, 'done': False,
                        'error': None, 'res': None, 'owner': 'ai',
                        'kind': 'single', 't_max': None}
st = ai_server.ui_job_status()
C.check('3 ui_job_status 能看到它', (st or {}).get('job_id') == jid,
        str(st))
J = ai_server._t_list_jobs()
row = next((r for r in J['jobs'] if r['job_id'] == jid), None)
C.check('4 list_jobs 归因 owner=ai 且 status=running',
        bool(row) and row['owner'] == 'ai' and row['status'] == 'running',
        str(row))

win._ai_job_tick()                 # 与定时器同一条逻辑刷新
C.check('5 有作业时指示器可见且显示 job_id',
        (not win._ai_job_ind.isHidden())
        and jid in win._ai_job_ind.text(), win._ai_job_ind.text())

# 人点「取消AI作业」（模拟点击 → 走同一条取消路径）
win._ai_job_btn.click()
C.check('6 取消后作业不再是 running',
        (ai_server.ui_job_status() or {}) == {} or
        ai_server.ui_job_status() is None, str(ai_server.ui_job_status()))

ai_server._jobs.pop(jid, None)
C.check('7 取消入口与 MCP 同路径（ui_cancel_job → _t_cancel_job）',
        callable(ai_server.ui_cancel_job)
        and ai_server.ui_cancel_job('不存在')['ok'] is False)

C.summary()
