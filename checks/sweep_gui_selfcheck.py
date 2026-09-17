# -*- coding: utf-8 -*-
"""自检：并行扫描的**人侧可见面**（底栏之上向上展开的一栏；取消不阻塞）。

## 口径变更（2026-09-16 用户裁决）——旧版是本文件的"反面教材"

用户实测两条："**点击最小化之后还是弹出**"、"画布上面那个并行小行多余了，
把他的职能跟底栏进度条那一堆合并"。根因是形态选错了：

- 旧实现是**独立浮窗**（`Qt.Tool | FramelessWindowHint | WindowStaysOnTopHint`）。
  点"最小化"→ `hide()`，但 1 Hz 轮询下一拍看到"有扫描在跑"又 `show()` 回来
  ⇒ **最小化在结构上不可能生效**（不是按钮没接上，是"谁决定可见性"被两个主人
  同时管着：用户与轮询）。
- 旧实现还有一块**画布顶部横幅**，职能与浮窗重复。

现形态：主窗口中央布局里的一个普通 `QWidget`（`app_entry` 用
`root.addWidget(...)` 加在 `top` 之后 ⇒ 位于底栏正上方）。
**几何由布局决定，"弹"这个概念不存在**；表头常显（承担原横幅职能），
逐点区可收放，且**收放状态归用户**，轮询不许覆盖。

## 覆盖（每段 1 个不变式包）

  A 逐点行渲染：行数、百分比、行文本、超 `MAX_ROWS` 折叠出聚合行
  B **形态**：不是独立窗口、在中央布局里、横幅类与旧浮窗类都已不存在
  C **收放语义**：收起后**轮询不会把它顶回来**（用户报的 bug 的直接回归）；
    展开可恢复；终态若干秒后整栏隐藏、把空间还给画布
  D 取消不阻塞：按钮只调 `cancel_request`（**不调**会等待的 `cancel`）
  E 人机同源：栏内数据确实来自 `registry.status`（= AI 的 `sweep_status`）
  F 表头承担原画布横幅的职能：运行中/终态文案、tag、进度、进程数

用法：python checks/sweep_gui_selfcheck.py（offscreen；缺 param_manager →
SKIP；不碰活实例：`EL_NO_AI_SERVER=1`）
"""
import os
import sys
import time

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


st = win.sweep_stack
st._timer.stop()                     # 自检里不要后台轮询真目录
_orig_status = st._status

# ---------------------------------------------------------------
print('A) 逐点行渲染（逐点行 + 折叠）')
st.refresh(_fake_status(30, 5))
_vis = [r for r in st._rows if not r[0].isHidden()]
C.group('A 逐点进度行与折叠聚合行', [
    ('行数 = min(点数, MAX_ROWS)', len(_vis) == SP.MAX_ROWS),
    ('第 0 行 100%（done）', st._rows[0][1].value() == 1000),
    ('第 5 行 50%（running）', st._rows[5][1].value() == 500),
    ('行文本带点号与标签',
     'pt00' in st._rows[0][0].text() and
     'freq_mhz' in st._rows[0][0].text()),
    ('百分比文本跟着走', st._rows[0][2].text() == '100%'),
    ('超出行数折叠成聚合行且可见',
     st._agg is not None and not st._agg.isHidden() and
     ('其余 %d 点' % (30 - SP.MAX_ROWS)) in st._agg.text()),
    ('点数变少后聚合行收起',
     (st.refresh(_fake_status(3, 3)), st._agg.isHidden())[1]),
])

# ---------------------------------------------------------------
print('B) 形态：不是弹窗、在布局里、旧部件已删')
_lay = win.centralWidget().layout()
C.group('B 形态（用户裁决："改成本弹窗…向上展开"）', [
    ('**不是独立窗口**（isWindow=False）', st.isWindow() is False),
    ('顶层窗口就是主窗口（window() is win）', st.window() is win),
    ('在主窗口中央布局里（不是浮层）',
     _lay is not None and _lay.indexOf(st) >= 0),
    ('布局里排在最底（在 canvas 之后 ⇒ 底栏正上方）',
     _lay.indexOf(st) > _lay.indexOf(win.canvas_tabs)),
    ('画布横幅类已删除', not hasattr(SP, 'CanvasBanner')),
    ('旧浮窗类已删除', not hasattr(SP, 'SweepProgressWindow')),
    ('主窗口上没有横幅/浮窗属性',
     not hasattr(win, 'sweep_banner') and not hasattr(win, 'sweep_window')),
    ('无扫描时不占高度（初始隐藏）', st.isHidden()),
])

# ---------------------------------------------------------------
print('C) 收放语义（用户报的"最小化后被弹回"的直接回归）')
st._status = lambda: _fake_status(30, 5, 'running')
st.tick()                                    # 有扫描 → 显形
_shown = not st.isHidden()
st._on_toggle()                              # 点"收起"
_c1 = (st._collapsed, st._scroll.isHidden())
st.tick()                                    # ★ 轮询一拍（旧版就是这里把它弹回来）
_c2 = (st._collapsed, st._scroll.isHidden(), not st.isHidden())
st._on_toggle()                              # 再点"展开"
_c3 = (st._collapsed, not st._scroll.isHidden())
st._terminal_seen = 'swTEST'                 # 造"终态已看过 + 过了 8 s"
st._idle_since = time.time() - 100
st._status = lambda: _fake_status(30, 30, 'done')
st.tick()
C.group('C 收起/展开 + 终态自动隐藏（轮询不许覆盖用户选择）', [
    ('有扫描 → 整栏显形', _shown),
    ('收起后逐点区隐藏、表头仍在',
     _c1[0] is True and _c1[1] is True),
    ('**轮询一拍后仍是收起的**（旧版 bug 已不可能复现）',
     _c2[0] is True and _c2[1] is True and _c2[2] is True),
    ('再点展开可恢复', _c3[0] is False and _c3[1] is True),
    ('终态 + 超时后整栏隐藏（空间还给画布）', st.isHidden()),
])

# ---------------------------------------------------------------
print('D) 取消按钮：只写 cancel 文件、不阻塞主线程')
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
    st._sid = 'swTEST'
    st._on_cancel()
finally:
    RG.cancel_request, RG.cancel = _orig_req, _orig_cancel
C.group('D 取消路径（GUI 不许被 runner 退出阻塞）', [
    ('按钮调了 cancel_request（写文件即返回）',
     _calls.get('cancel_request') == (ROOT, 'swTEST')),
    ('**没有**调会等待的 cancel()（否则主线程会被堵住）',
     'cancel' not in _calls),
])

# ---------------------------------------------------------------
print('E) 人机同源：栏内数据确实来自 registry.status')
_seen = []
_orig = RG.status
RG.status = lambda root, *a, **k: (_seen.append(root), _fake_status(4, 1))[1]
st._status = _orig_status                    # 恢复真实入口
try:
    got = st._status()
finally:
    RG.status = _orig
C.group('E 与 AI 的 sweep_status 同一份数据', [
    ('栏内读的就是 registry.status(ROOT)', _seen == [ROOT]),
    ('返回值直接喂给 refresh（键名一致）',
     all(k in got for k in
         ('state', 'n_points', 'n_done', 'workers', 'points'))),
])

# ---------------------------------------------------------------
print('F) 表头承担原画布横幅的职能')
st.refresh(_fake_status(30, 5, 'running'))
_t_run = st.info.text()
st.refresh(_fake_status(30, 30, 'done'))
_t_done = st.info.text()
C.group('F 表头 = 原来那一行字（run/终态）', [
    ('运行中：tag/进度/进程数/状态都在一行',
     all(k in _t_run for k in ('freeze-rise-fall', '5/30', '4', '运行'))),
    ('终态：切成"完成"且给全量进度',
     '完成' in _t_done and '30/30' in _t_done),
    ('表头始终可见（收起也不藏）', not st.title.isHidden()),
], 'running=%r done=%r' % (_t_run, _t_done))

st._status = _orig_status
try:
    win.close()
except Exception:                                # noqa: BLE001
    pass

ok = C.summary()
sys.exit(0 if ok else 1)