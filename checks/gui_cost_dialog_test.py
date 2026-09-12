# -*- coding: utf-8 -*-
"""运行前成本对话框自检（t84）——钉住用户实测的"点不到对话框"bug。

bug 复现条件：`_run` 在开头起了一个 400ms 单发定时器，到点就弹出**进度框**，
而进度框是 `Qt.WindowModal` —— 模态阻塞的是整个父窗口（含挂在它下面的其它
对话框），于是后弹出的成本报告被压在下面、**根本点不到**。

因此本检查的**核心断言**是：`QMessageBox.exec_()` 被调用的那一刻，
进度框必须不可见、其定时器必须已停；选"继续"后进度框要恢复。

用法：python checks\\gui_cost_dialog_test.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

from checks.checklib import Checker                          # noqa: E402
from PyQt5.QtCore import QTimer, Qt                           # noqa: E402
from PyQt5.QtWidgets import QApplication, QDialog, QMessageBox  # noqa: E402

import app_entry as M                                         # noqa: E402
import src.core.evolve_mixed as EM                            # noqa: E402

C = Checker('gui_cost_dialog')
_app = QApplication.instance() or QApplication(sys.argv)
_orig_em = EM.evolve_mixed

_S = [{'label': 's%d' % i} for i in range(8)]      # 假基组（只为 len()）
BIG = {'dry_run': True, 'cost': {
    'ok': True, 'n_mw': 5, 'max_det_mhz': 9857.0, 'est_steps': 4.6e4,
    'budget': None, 'mw_cap_mhz': 1e5, 'n_drop_phase': 0,
    'alt': {4e4: {'cap_mhz': 5305.0, 'n_drop': 2, 'max_det_mhz': 3796.0,
                  'est_steps': 1.8e4, 'speedup': 2.6}},
    'worst': {'a': 0, 'b': 1, 'label_a': '60P m=-0.5', 'label_b': '58G m=-3.5',
              'abs_v_mhz': 3290.0, 'det_mhz': 7079.0}}}
CHEAP = {'dry_run': True, 'cost': {
    'ok': True, 'n_mw': 2, 'max_det_mhz': 500.0, 'est_steps': 1e3,
    'budget': None, 'mw_cap_mhz': 1e5, 'alt': {}, 'worst': None}}


def _mk_dyn(states=None):
    """真窗口的 DynamicPanel + 手工造出"进度框已弹出"的状态。"""
    win = M.MainWindow()
    dyn = win.dynamic
    dyn._dlg_timer = QTimer(dyn)
    dyn._dlg_timer.setSingleShot(True)
    dyn._dlg_timer.start(4000)                # 模拟 _run 开头的定时器
    dyn._dlg = QDialog(win)
    dyn._dlg.show()
    _app.processEvents()
    return win, dyn


# ── 1. 便宜算例：不弹窗、不打扰 ──────────────────────────────
EM.evolve_mixed = lambda *a, **k: CHEAP
_win, _dyn = _mk_dyn()
_called = []
QMessageBox.exec_ = lambda self: (_called.append(1), 0)[1]
_ok = _dyn._preflight_cost(_S, [], [0.0, 400.0], {})
C.check('1 便宜算例（Δ_max<3GHz）→ 不弹窗、直接跑',
        _ok is True and not _called, f'ok={_ok} called={len(_called)}')

# ── 2. 贵算例：弹窗那一刻进度框必须已撤下（核心 bug）──────────
EM.evolve_mixed = lambda *a, **k: BIG
_win, _dyn = _mk_dyn()
_seen = {}


def _fake_exec(self):
    _seen['dlg_visible'] = bool(_dyn._dlg is not None and _dyn._dlg.isVisible())
    _seen['timer_active'] = bool(_dyn._dlg_timer.isActive())
    _seen['title'] = self.windowTitle()
    _seen['buttons'] = [b.text() for b in self.buttons()]
    return 0


QMessageBox.exec_ = _fake_exec
_ok = _dyn._preflight_cost(_S, [], [0.0, 400.0], {})
C.check('2 对话框弹出时进度框已隐藏（否则点不到 —— 用户实测 bug）',
        _seen.get('dlg_visible') is False, str(_seen.get('dlg_visible')))
C.check('3 对话框弹出时进度框定时器已停（不会再冒出来盖住它）',
        _seen.get('timer_active') is False, str(_seen.get('timer_active')))
C.check('4 三个按钮齐备（完整物理 / 剔除远对 / 取消）',
        len(_seen.get('buttons', [])) == 3
        and any('完整物理' in b for b in _seen['buttons'])
        and any('剔除' in b for b in _seen['buttons'])
        and any('取消' in b for b in _seen['buttons']),
        str(_seen.get('buttons')))
C.check('5 选"完整物理" → 进度框恢复显示、仍可继续',
        _ok is True and _dyn._dlg is not None and _dyn._dlg.isVisible(),
        f'ok={_ok} visible={_dyn._dlg.isVisible() if _dyn._dlg else None}')

# ── 3. 取消：返回 False，且收尾后不留禁用按钮/僵尸定时器 ────────
_win, _dyn = _mk_dyn()
_win.dynamic.run_btn.setEnabled(False)        # 模拟 _run 已禁用按钮
QMessageBox.exec_ = lambda self: 0
QMessageBox.clickedButton = lambda self: [
    b for b in self.buttons() if '取消' in b.text()][0]
_ok = _dyn._preflight_cost(_S, [], [0.0, 400.0], {})
C.check('6 选"取消" → 返回 False（本次不起算）', _ok is False)
_dyn._finish_run_ui()                          # 调用方的收尾（与 _run 一致）
C.check('7 取消后收尾：按钮恢复、定时器停、进度框关闭',
        _dyn.run_btn.isEnabled()
        and not _dyn._dlg_timer.isActive()
        and _dyn._dlg is None,
        f'btn={_dyn.run_btn.isEnabled()} timer={_dyn._dlg_timer.isActive()}')

# ── 4. AI 发起的运行不弹窗（无人点会卡死主线程）───────────────
_win, _dyn = _mk_dyn()
_dyn._ai_initiated = True
_called2 = []
QMessageBox.exec_ = lambda self: (_called2.append(1), 0)[1]
_ok = _dyn._preflight_cost(_S, [], [0.0, 400.0], {})
C.check('8 AI 运行 → 不弹模态、走完整物理、只写日志',
        _ok is True and not _called2 and _dyn._dlg.isVisible(),
        f'ok={_ok} called={len(_called2)}')

EM.evolve_mixed = _orig_em
sys.exit(0 if C.summary() else 1)
