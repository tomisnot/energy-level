# -*- coding: utf-8 -*-
"""并行扫描的人侧可见面：**主窗口底栏之上向上展开的一栏**。

## 形态（2026-09-16 用户裁决，取代原来的右下角悬浮窗）

用户原话："点击最小化之后还是弹出，直接改成**不是弹窗**，而是底栏进度条那一块
的**向上展开**；然后画布上面那个并行小行多余了，把他的职能跟底栏进度条那一堆
合并。"

于是：
- **不再是独立窗口**（旧实现是 `Qt.Tool | FramelessWindowHint` 的浮窗，
  点"最小化"→ `hide()`，但 1 Hz 轮询下一次 tick 又 `show()` 回来 ⇒ 最小化无效）。
  现在它是主窗口中央布局里的一个普通 `QWidget`（`app_entry` 用
  `root.addWidget(...)` 加在 `top` 之后），**几何由布局决定，没有"弹"这个概念**。
- **画布顶部那行字（CanvasBanner）删掉**：它的职能（"现在有并行在跑"）并入本栏
  的**表头**——表头永远可见，逐点行才收放。
- **"最小化"变成"收起/展开"**：收起 = 只留表头（仍在底栏那一块，看得见进度），
  再点展开。这个状态由**用户**决定，轮询**不会**把它顶回来。

## 人机同源（P2，不变）

本栏只读 `src.sweep.registry.status()` → runner 写的 `status.json`，
与 AI 的 `sweep_status` 读的是**同一份事实**。
取消只调 `registry.cancel_request()`（写 cancel 文件即返回），
**绝不阻塞主线程**——等待与结论交给轮询（本项目 Qt 事故史的直接教训）。

## 高度预算

逐点行放在 `QScrollArea` 里（最高 `BODY_MAX_H`），点很多也不会把画布挤没；
每行是**单行**（标签 + 进度条 + 百分比），比"标签一行、条一行"省一半高度。
"""
import os

from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel,
                             QProgressBar, QPushButton, QScrollArea,
                             QSizePolicy)

MAX_ROWS = 16                      # 最多建多少行（其余折叠成聚合行）
BODY_MAX_H = 190                   # 逐点区最高像素（超出内部滚动）
IDLE_HIDE_S = 8.0                  # 结束后再留几秒让人看清"完成了"
POLL_MS = 1000

_STATE_TAG = {'queued': '排队', 'running': '运行', 'done': '完成',
              'failed': '失败', 'cancelled': '取消'}


def _short_tag(tag, n=30):
    tag = str(tag or '')
    return tag if len(tag) <= n else tag[:n - 1] + '…'


class SweepStack(QWidget):
    """底栏之上的一栏：表头（常显）+ 逐点进度（可收放）。"""

    def __init__(self, main, root=None, parent=None):
        super().__init__(parent)
        self.main = main
        self.root = root or os.path.dirname(os.path.dirname(
            os.path.dirname(os.path.abspath(__file__))))
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Maximum)
        self.setStyleSheet(
            'SweepStack{background:#f7f8fa;border-top:1px solid #d6d9de;}'
            'QLabel{color:#3a4150;}'
            'QPushButton{color:#3a4150;background:#eceff4;'
            'border:1px solid #d6d9de;border-radius:3px;padding:1px 8px;}'
            'QPushButton:hover{background:#e2e7ef;}'
            'QProgressBar{border:1px solid #d6d9de;border-radius:2px;'
            'background:#ffffff;text-align:center;color:#5b6270;'
            'font-size:8pt;}'
            'QProgressBar::chunk{background:#6c8cff;}')

        box = QVBoxLayout(self)
        box.setContentsMargins(8, 3, 8, 4)
        box.setSpacing(2)

        head = QHBoxLayout()
        head.setSpacing(6)
        self.title = QLabel('⚡ 并行扫描')
        self.title.setStyleSheet('font-weight:600;')
        self.info = QLabel('')
        self.info.setStyleSheet('color:#5b6270;')
        self.btn_toggle = QPushButton('收起')
        self.btn_toggle.setToolTip('收起逐点进度（表头保留；再点展开）')
        self.btn_toggle.clicked.connect(self._on_toggle)
        self.btn_cancel = QPushButton('取消扫描')
        self.btn_cancel.setToolTip('请求取消：worker 在安全点退出，已完成的点保留')
        self.btn_cancel.clicked.connect(self._on_cancel)
        head.addWidget(self.title)
        head.addWidget(self.info)
        head.addStretch(1)
        head.addWidget(self.btn_cancel)
        head.addWidget(self.btn_toggle)
        box.addLayout(head)

        # 逐点区（可滚动、可整体收起）
        self._host = QWidget()
        self._body = QVBoxLayout(self._host)
        self._body.setContentsMargins(0, 0, 0, 0)
        self._body.setSpacing(1)
        self._scroll = QScrollArea()
        self._scroll.setWidgetResizable(True)
        self._scroll.setFrameShape(QScrollArea.NoFrame)
        self._scroll.setMaximumHeight(BODY_MAX_H)
        self._scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self._scroll.setWidget(self._host)
        box.addWidget(self._scroll)

        self._rows = []            # [(QLabel, QProgressBar, QLabel)]
        self._agg = None           # 聚合行
        self._sid = None
        self._collapsed = False    # **用户**的收放选择（轮询不许覆盖它）
        self._idle_since = None
        self._terminal_seen = None

        self._timer = QTimer(self)
        self._timer.timeout.connect(self.tick)
        self._timer.start(POLL_MS)
        self.setVisible(False)

    # ---------------- 数据（与 AI 的 sweep_status 同一份） ----------------
    def _status(self):
        try:
            from ..sweep import registry as rg
            return rg.status(self.root)
        except Exception:                          # noqa: BLE001
            return {'ok': False}

    def tick(self):
        st = self._status()
        if not st.get('ok'):
            self._maybe_idle_hide()
            return
        state = st.get('state')
        if state in ('done', 'failed', 'cancelled'):
            if self._terminal_seen != st.get('sweep_id'):
                self._terminal_seen = st.get('sweep_id')
                self._idle_since = __import__('time').time()
            self.refresh(st)
            self._maybe_idle_hide()
            return
        self._idle_since = None
        self._terminal_seen = None
        self.refresh(st)
        self.setVisible(True)          # 只有"新扫描出现"才会显形；不看浮窗状态

    def _maybe_idle_hide(self):
        import time
        if self._idle_since is None:
            return
        if time.time() - self._idle_since > IDLE_HIDE_S:
            self._idle_since = None
            self.setVisible(False)     # 整栏收起，布局把空间还回画布

    def refresh(self, st):
        sid = st.get('sweep_id')
        self._sid = sid
        pts = st.get('points') or []
        n = len(pts)
        show = min(n, MAX_ROWS)
        while len(self._rows) < show:
            lab = QLabel('')
            lab.setFixedWidth(180)
            bar = QProgressBar()
            bar.setRange(0, 1000)
            bar.setFixedHeight(12)
            pct = QLabel('')
            pct.setFixedWidth(42)
            pct.setStyleSheet('color:#5b6270;')
            row = QHBoxLayout()
            row.setSpacing(6)
            row.addWidget(lab)
            row.addWidget(bar, 1)
            row.addWidget(pct)
            self._body.addLayout(row)
            self._rows.append((lab, bar, pct))
        for i, (lab, bar, pct) in enumerate(self._rows):
            if i >= show:
                lab.setVisible(False)
                bar.setVisible(False)
                pct.setVisible(False)
                continue
            p = pts[i]
            lab.setVisible(True)
            bar.setVisible(True)
            pct.setVisible(True)
            lab.setText('pt%02d %s' % (p.get('k', i), _short_tag(p.get('tag'))))
            frac = float(p.get('frac') or 0.0)
            bar.setValue(int(round(1000 * frac)))
            pct.setText('%.0f%%' % (100 * frac))
        if n > show:
            if self._agg is None:
                self._agg = QLabel('')
                self._agg.setStyleSheet('color:#5b6270;')
                self._body.addWidget(self._agg)
            rest = pts[show:]
            done = sum(1 for p in rest if p.get('state') == 'done')
            avg = (sum(float(p.get('frac') or 0.0) for p in rest) / len(rest)
                   if rest else 0.0)
            self._agg.setText('其余 %d 点：完成 %d（平均 %.0f%%）'
                              % (len(rest), done, 100 * avg))
            self._agg.setVisible(True)
        elif self._agg is not None:
            self._agg.setVisible(False)

        tag = st.get('tag') or sid or ''
        state_txt = _STATE_TAG.get(st.get('state'), st.get('state') or '')
        # 表头承担原画布那行字的职能（用户裁决 2026-09-16）
        self.info.setText(
            '%s · %s/%s 点 · %s 进程 · %.0fs · %s'
            % (tag, st.get('n_done'), st.get('n_points'), st.get('workers'),
               float(st.get('elapsed_s') or 0.0), state_txt))
        self._apply_collapsed()

    def _apply_collapsed(self):
        self._scroll.setVisible(not self._collapsed)
        self.btn_toggle.setText('展开' if self._collapsed else '收起')

    def _on_toggle(self):
        self._collapsed = not self._collapsed
        self._apply_collapsed()

    def _on_cancel(self):
        try:
            from ..sweep import registry as rg
            r = rg.cancel_request(self.root, self._sid)
            if self.main is not None:
                self.main.log(r.get('msg') or '已请求取消', 5000)
        except Exception as e:                     # noqa: BLE001
            if self.main is not None:
                self.main.log('取消失败：%s' % e, 5000)
