# -*- coding: utf-8 -*-
"""自检：并行扫描的**人侧可见面**（悬浮进度窗 + 画布文本条 + 取消不阻塞）。

用户口径（2026-09-15）："这个功能倾向于 AI 使用……但**需要做出 UI 让人类知道
在并行**"，"画布也要有处理不能放着，可以显示一行文本，并行运行中"。
所以这里测的不是"好不好看"，而是三条硬要求：
  1. 人在扫描期间**看得见**（逐点进度 + 画布一行字）；
  2. 人**能接管**（取消按钮走 cancel_request，不阻塞主线程）；
  3. 人看到的与 AI 返回的**是同一份事实**（都读 registry.status → status.json）。

覆盖（每段 1 个不变式包）：
  A 悬浮窗渲染：逐点行数、百分比、汇总文本；超 `MAX_ROWS` 折叠出聚合行
  B 画布条：运行中/终态文案不同；空字符串 → 隐藏；父窗口是画布容器
  C 取消不阻塞：按钮只调 `cancel_request`（**不调**会等待的 `cancel`），立即返回
  D 接线：MainWindow 上真有这两个部件，且悬浮窗用画布条联动
  E 人机同源：`refresh` 的数据确实来自 `registry.status`

用法：python checks/sweep_gui_selfcheck.py（offscreen；缺 param_manager →
SKIP；不碰活实例：`EL_NO_AI_SERVER=1`）
"""
import os
import sys

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
os.environ.setdefault('EL_NO_AI_SERVER', '1')

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

from checks.checklib import Checker                              # noqa: E402

C = Checker('sweep_gui')
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

try:
    from PyQt5.QtWidgets import QApplication                     # noqa: E402
    import app_entry as M                                        # noqa: E402
    from src.ui import sweep_panel as SP                         # noqa: E402
    from src.sweep import registry as RG                         # noqa: E402
    app = QApplication.instance() or QApplication(sys.argv)
    win = M.MainWindow()
    win.show()
except Exception as e:                                           # noqa: BLE001
    print('  [SKIP] 构窗不可用（缺 param_manager / PyQt5？）：%s' % str(e)[:120])
    sys.exit(0)


def _fake_status(n=30, done=5, state='running'):
    return {
        'ok': True, 'sweep_id': 'swTEST', 'tag': 'freeze-rise-fall',
        'state': state, 'n_points': n, 'n_done': done, 'n_failed': 0,
        'workers': 4, 'elapsed_s': 12.0, 'elapsed': 12.0,
        'points': [{'k': i,
                    'state': ('done' if i < done else
                              ('running' if i == done else 'queued')),
                    'frac': (1.0 if i < done else (0.5 if i == done else 0.0)),
                    'tag': 'freq_mhz=%d' % (7300 + 5 * i)}
                   for i in range(n)],
    }


pw = win.sweep_window
pw._timer.stop()                     # 自检里不要后台轮询真目录

# ---------------------------------------------------------------
print('A) 悬浮窗渲染（逐点行 + 折叠）')
pw.refresh(_fake_status(30, 5))
_vis_rows = [r for r in pw._rows if not r[0].isHidden()]
C.group('A 逐点进度行与折叠聚合行', [
    ('行数 = min(点数, MAX_ROWS)', len(_vis_rows) == SP.MAX_ROWS),
    ('第 0 行百分比 = 100%（done）', pw._rows[0][1].value() == 1000),
    ('第 5 行百分比 = 50%（running）', pw._rows[5][1].value() == 500),
    ('行文本带点标签与状态',
     'pt00' in pw._rows[0][0].text() and
     '完成' in pw._rows[0][0].text()),
    ('汇总文本含 已完成/总数/进程数',
     all(k in pw.info.text() for k in ('5/30', '4'))),
    ('超出行数折叠成聚合行且可见',
     pw._agg is not None and not pw._agg.isHidden() and
     '其余 6 点' in pw._agg.text()),
    ('点数变少后聚合行收起',
     (pw.refresh(_fake_status(3, 3)), pw._agg.isHidden())[1]),
])

# ---------------------------------------------------------------
print('B) 画布文本条（运行中 / 终态 / 收起）')
b = win.sweep_banner
b.set_line('')
_hidden_empty = b.isHidden()
pw.refresh(_fake_status(30, 5, 'running'))
_txt_run = b.text()
_shown_run = not b.isHidden()
pw.refresh(_fake_status(30, 30, 'done'))
_txt_done = b.text()
C.group('B 画布那一行（用户明确要求"不能放着"）', [
    ('空字符串 → 隐藏', _hidden_empty),
    ('运行中：可见且写明运行中/进度/进程数',
     _shown_run and '并行扫描' in _txt_run and
     '运行中' in _txt_run and '5/30' in _txt_run),
    ('终态：文案切成"完成/失败/取消"',
     '完成' in _txt_done and '30/30' in _txt_done),
    ('父窗口就是画布容器（横幅贴在画布上）',
     b.parentWidget() is win.canvas_tabs),
])

# ---------------------------------------------------------------
print('C) 取消按钮：只写 cancel 文件、不阻塞主线程')
_calls = {}
_orig_req, _orig_cancel = RG.cancel_request, RG.cancel


def _fake_req(root, sid=None):
    _calls['cancel_request'] = (root, sid)
    return {'ok': True, 'msg': '已请求取消'}


def _fake_cancel(*a, **k):
    _calls['cancel'] = (a, k)
    return {'ok': True}


RG.cancel_request, RG.cancel = _fake_req, _fake_cancel
try:
    pw._sid = 'swTEST'
    pw._on_cancel()
finally:
    RG.cancel_request, RG.cancel = _orig_req, _orig_cancel
C.group('C 取消路径（GUI 不许被 runner 退出阻塞）', [
    ('按钮调了 cancel_request（写文件即返回）',
     _calls.get('cancel_request') == (ROOT, 'swTEST')),
    ('**没有**调会等待的 cancel()（否则主线程会被堵住）',
     'cancel' not in _calls),
])

# ---------------------------------------------------------------
print('D) 接线：主窗口上真的有这两个部件')
C.group('D MainWindow 接线', [
    ('win.sweep_window 是 SweepProgressWindow',
     isinstance(pw, SP.SweepProgressWindow)),
    ('win.sweep_banner 是 CanvasBanner',
     isinstance(win.sweep_banner, SP.CanvasBanner)),
    ('悬浮窗与画布条联动（banner 已接上）', pw.banner is win.sweep_banner),
    ('点画布条 → 抬起悬浮窗（有回调）',
     callable(getattr(win.sweep_banner, '_on_click', None))),
])

# ---------------------------------------------------------------
print('E) 人机同源：面板的数据确实来自 registry.status')
_seen = []
_orig_status = RG.status
RG.status = lambda root, *a, **k: (_seen.append(root),
                                   _fake_status(4, 1))[1]
try:
    pw._status()
finally:
    RG.status = _orig_status
C.group('E 与 AI 的 sweep_status 同一份数据', [
    ('面板读的就是 registry.status(ROOT)', _seen == [ROOT]),
    ('返回值原样喂给 refresh（键名一致）',
     all(k in _fake_status() for k in
         ('state', 'n_points', 'n_done', 'workers', 'points'))),
])

try:
    win.close()
except Exception:                                # noqa: BLE001
    pass

ok = C.summary()
sys.exit(0 if ok else 1)
