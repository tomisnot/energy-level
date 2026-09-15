# -*- coding: utf-8 -*-
"""并行扫描的人侧可见面：右下角悬浮进度窗 + 画布顶部文本条。

## 为什么要这两个东西

用户口径（2026-09-15）："这个功能倾向于 AI 使用，所以不太需要考虑人类操作
方便，但**需要做出 UI 让人类知道在并行**"。也就是 G3 的那一条——AI 跑的东西
人得看得见、能接管。

设计上刻意与 AI 侧**同源**（P2）：两者都只读 `src.sweep.registry.status()`，
而它读的是 runner 写的 `status.json`。所以"人看到的"和 `sweep_status` 返回的
永远是同一份事实，不存在两套口径。

## 两个部件的分工

- `SweepProgressWindow`：**详情**。纵向一叠逐点进度条（每行 = 一个扫描点），
  含进程数、已用时长、逐点标签（`freq_mhz=7380.0`）、以及「取消扫描」。
  超过 `MAX_ROWS` 点则把余下的折成一行"其余 N 点"聚合条（不无限长高）。
- `CanvasBanner`：**一眼可见**。画布顶部居中一行字（"⚡ 并行扫描运行中 · 12/25"），
  点它把悬浮窗拉到前面。用户原话："画布也要有处理不能放着，可以显示一行文本"。

两者都不写任何状态：取消按钮只调 `registry.cancel_request()`（写 cancel 文件
后立即返回），等待与结论交给轮询——主线程绝不被 runner 的退出阻塞
（本项目 Qt 事故史的直接教训）。
"""
import os

from PyQt5.QtCore import Qt, QTimer, QEvent
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel,
                             QProgressBar, QPushButton)

MAX_ROWS = 24                      # 超过就折叠成聚合行（窗不能无限长）
IDLE_HIDE_S = 8.0                  # 结束后再留几秒让人看清"完成了"
POLL_MS = 1000

_STATE_TAG = {'queued': '排队', 'running': '运行', 'done': '完成',
              'failed': '失败', 'cancelled': '取消'}


def _short_tag(tag, n=34):
    tag = str(tag or '')
    return tag if len(tag) <= n else tag[:n - 1] + '…'


class SweepProgressWindow(QWidget):
    """右下角悬浮窗：纵向一叠逐点进度条（人机同源数据）。"""

    def __init__(self, main, root=None, parent=None, banner=None):
        super().__init__(parent)
        self.main = main
        self.banner = banner           # CanvasBanner（可为 None）
        self.root = root or os.path.dirname(os.path.dirname(
            os.path.dirname(os.path.abspath(__file__))))
        self.setWindowFlags(Qt.Tool | Qt.FramelessWindowHint |
                            Qt.WindowStaysOnTopHint)
        self.setAttribute(Qt.WA_ShowWithoutActivating, True)
        self.setStyleSheet(
            'QWidget{background:#20242c;color:#e8eaf0;}'
            'QLabel{color:#e8eaf0;}'
            'QPushButton{color:#e8eaf0;background:#2f3542;border:1px solid '
            '#4a5266;border-radius:3px;padding:2px 8px;}'
            'QPushButton:hover{background:#3b4455;}'
            'QProgressBar{border:1px solid #4a5266;border-radius:2px;'
            'background:#171a20;text-align:center;color:#cfd4e0;'
            'font-size:8pt;}'
            'QProgressBar::chunk{background:#6c8cff;}')

        root_box = QVBoxLayout(self)
        root_box.setContentsMargins(8, 6, 8, 8)
        root_box.setSpacing(4)

        head = QHBoxLayout()
        self.title = QLabel('并行扫描')
        self.title.setStyleSheet('font-weight:600;')
        self.info = QLabel('')
        self.info.setStyleSheet('color:#9aa3b8;')
        self.btn_cancel = QPushButton('取消扫描')
        self.btn_cancel.setToolTip('请求取消（worker 在安全点退出，已完成的点保留）')
        self.btn_cancel.clicked.connect(self._on_cancel)
        self.btn_hide = QPushButton('最小化')
        self.btn_hide.setToolTip('只收起这个窗；扫描继续，画布上仍有进度提示')
        self.btn_hide.clicked.connect(self.hide)
        head.addWidget(self.title)
        head.addStretch(1)
        head.addWidget(self.info)
        head.addWidget(self.btn_cancel)
        head.addWidget(self.btn_hide)
        root_box.addLayout(head)

        self._rows = []            # [(QLabel, QProgressBar)]
        self._body = QVBoxLayout()
        self._body.setSpacing(2)
        root_box.addLayout(self._body)
        self._agg = None           # 折叠聚合行
        self._sid = None
        self._idle_since = None
        self._terminal_seen = None

        self._timer = QTimer(self)
        self._timer.timeout.connect(self.tick)
        self._timer.start(POLL_MS)
        self.hide()

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
        if not self.isVisible():
            self.show()
        self._reposition()

    def _maybe_idle_hide(self):
        import time
        if self._idle_since is None:
            return
        if time.time() - self._idle_since > IDLE_HIDE_S:
            self.hide()
            self._idle_since = None
            if self.banner is not None:
                self.banner.set_line('')           # 画布提示同步收起

    def refresh(self, st):
        sid = st.get('sweep_id')
        self._sid = sid
        pts = st.get('points') or []
        n = len(pts)
        show = min(n, MAX_ROWS)
        while len(self._rows) < show:
            i = len(self._rows)
            lab = QLabel('')
            lab.setStyleSheet('font-size:8pt;')
            bar = QProgressBar()
            bar.setRange(0, 1000)
            bar.setFixedHeight(12)
            self._body.addWidget(lab)
            self._body.addWidget(bar)
            self._rows.append((lab, bar))
        for i in range(len(self._rows)):
            lab, bar = self._rows[i]
            if i >= show:
                lab.hide()
                bar.hide()
                continue
            p = pts[i]
            lab.show()
            bar.show()
            lab.setText('pt%02d %s  %s' % (
                p.get('k', i), _short_tag(p.get('tag')),
                _STATE_TAG.get(p.get('state'), p.get('state') or '')))
            bar.setValue(int(round(1000 * float(p.get('frac') or 0.0))))
        if n > show:
            if self._agg is None:
                self._agg = QLabel('')
                self._agg.setStyleSheet('font-size:8pt;color:#9aa3b8;')
                self._body.addWidget(self._agg)
            rest = pts[show:]
            done = sum(1 for p in rest if p.get('state') == 'done')
            avg = (sum(float(p.get('frac') or 0.0) for p in rest) / len(rest)
                   if rest else 0.0)
            self._agg.setText('其余 %d 点：完成 %d（平均 %.0f%%）'
                              % (len(rest), done, 100 * avg))
            self._agg.show()
        elif self._agg is not None:
            self._agg.hide()
        self.title.setText('并行扫描 %s' % (st.get('tag') or sid or ''))
        self.info.setText('%s/%s · %s 进程 · %.0fs · %s'
                          % (st.get('n_done'), st.get('n_points'),
                             st.get('workers'), float(st.get('elapsed_s')
                                                      or 0.0),
                             st.get('state') or ''))
        self.adjustSize()
        self._reposition()
        # 画布那一行（用户口径：并行时画布不能"放着没人管"）
        if self.banner is not None:
            done, tot = st.get('n_done'), st.get('n_points')
            if st.get('state') in ('done', 'failed', 'cancelled'):
                self.banner.set_line('⚡ 并行扫描%s：%s/%s 点（%s）'
                                     % (_STATE_TAG.get(st.get('state'),
                                                       st.get('state')),
                                        done, tot, st.get('tag') or sid))
            else:
                self.banner.set_line(
                    '⚡ 并行扫描运行中：%s/%s 点 · %s 进程 · %.0fs'
                    '（点这里看逐点进度）'
                    % (done, tot, st.get('workers'),
                       float(st.get('elapsed_s') or 0.0)))

    def _reposition(self):
        """贴主窗口右下角（每 tick 一次，人拖动/缩放主窗也跟得上）。"""
        try:
            g = self.main.geometry() if self.main is not None else None
            if g is None:
                return
            self.move(g.right() - self.width() - 24,
                      g.bottom() - self.height() - 48)
        except Exception:                          # noqa: BLE001
            pass

    def _on_cancel(self):
        try:
            from ..sweep import registry as rg
            r = rg.cancel_request(self.root, self._sid)
            if self.main is not None:
                self.main.log(r.get('msg') or '已请求取消', 5000)
        except Exception as e:                     # noqa: BLE001
            if self.main is not None:
                self.main.log('取消失败：%s' % e, 5000)

    def raise_panel(self):
        self.show()
        self.raise_()
        self._reposition()


class CanvasBanner(QLabel):
    """画布顶部一行的"并行运行中"提示（点它 → 抬起悬浮窗）。

    用户口径："画布也要有处理不能放着，可以显示一行文本，并行运行中"。
    它是**只读**提示：不阻塞画布、不抢焦点，扫描结束后自行隐藏。
    """

    def __init__(self, parent, on_click=None):
        super().__init__(parent)
        self._on_click = on_click
        self.setStyleSheet(
            'background:rgba(108,140,255,0.92);color:#ffffff;'
            'border-radius:4px;padding:3px 10px;font-size:9pt;')
        self.setAlignment(Qt.AlignCenter)
        self.setAttribute(Qt.WA_TransparentForMouseEvents, False)
        self.setCursor(Qt.PointingHandCursor)
        self.setText('⚡ 并行扫描运行中')
        self.hide()
        parent.installEventFilter(self)

    def eventFilter(self, obj, ev):
        if ev.type() == QEvent.Resize:
            self._reposition()
        return False

    def _reposition(self):
        p = self.parentWidget()
        if p is None:
            return
        self.adjustSize()
        w = min(self.width(), max(120, p.width() - 24))
        self.move(int((p.width() - w) / 2), 8)

    def set_line(self, text):
        if not text:
            self.hide()
            return
        self.setText(text)
        self.adjustSize()
        self._reposition()
        self.show()

    def mousePressEvent(self, ev):
        if self._on_click is not None:
            try:
                self._on_click()
            except Exception:                      # noqa: BLE001
                pass
        super().mousePressEvent(ev)
