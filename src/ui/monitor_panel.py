# -*- coding: utf-8 -*-
"""AI 监控侧边栏（总纲 §1.2 事实 2：第一交付物；v2 默认挂载）。

人类在这里看"AI 在我的软件里干了什么"，两层信任并列（§5.2④）：

- **写权控制卡**（顶）：当前模式实时显示 + 接管/交权/禁写三按钮 + 网页控制台
  入口。mode 是协调态不落 History，靠 `bridge.watch_mode` 订阅刷新（HTTP 端点
  线程切的模式，经 pyqtSignal 编组回主线程——与史订阅同一 Qt 模式）。
- **概括层**（上）：`bridge.monitor.read()` 的人话读数——标明是派生的、
  可能错；与原始不一致时顶部红条显示 dispute_reason（原始赢，硬纪律 5）。
- **原始层**（下）：History 尾部事件表（seq/actor/键/值/理由）——append-only
  权威史，改不了。

线程：写事件可能来自 MCP 服务器线程（AI 写），回调里**不直接碰控件**，
经 pyqtSignal 编组回主线程刷新（Qt 标准模式）。

## Known Limitations and Deferred Work

- 原始层只显尾部 200 条（全量在 journal 里，面板不做分页/检索——等真有人
  要再加，不预先建）。
- 概括层键集由 mecha 侧 `el_summarizer` 钉死（消费配置的一部分）；本面板
  不自己发明概括。
- 网页控制台按钮只在装配起了人工控制端点（make_v2_bridge 默认路径）时出现；
  纯内存判据装配（for_gui）没有 `_human_port`，按钮隐藏——不假装能打开。
"""
from __future__ import annotations

from PyQt5.QtCore import Qt, pyqtSignal, QUrl
from PyQt5.QtGui import QDesktopServices
from PyQt5.QtWidgets import (QDockWidget, QHBoxLayout, QHeaderView, QLabel,
                             QPushButton, QTableWidget, QTableWidgetItem,
                             QVBoxLayout, QWidget)

from mecha.monitor import Claim

_RAW_TAIL = 200
_MAX_CELL = 60          # 值太长（psi 列表等）截断显示

_MODE_TEXT = {'human': '人写（HUMAN）', 'ai': 'AI 写（AI）',
              'locked': '全员禁写（LOCKED）'}
_MODE_COLOR = {'human': '#1a7f37', 'ai': '#b3261e', 'locked': '#8a6d1a'}


def _short(v) -> str:
    s = repr(v)
    return s if len(s) <= _MAX_CELL else s[:_MAX_CELL - 1] + '…'


class MonitorPanel(QWidget):
    """两层并列的监控读数面板 + 顶部写权控制卡。

    监控读数本身**只读、无任何域状态写回**（硬纪律：Monitor 附属化与写回
    已否决）；控制卡写的是**协调态 mode**（人侧独占的开关，与网页控制台
    /switch_mode 同一效果、同一条 authority 路径），不碰 Gate。
    """

    _changed = pyqtSignal()          # 史事件 → 编组回主线程
    _mode_changed = pyqtSignal(str)  # 写权模式变更 → 编组回主线程

    def __init__(self, bridge, parent=None):
        super().__init__(parent)
        self._bridge = bridge
        lay = QVBoxLayout(self)
        lay.setContentsMargins(6, 6, 6, 6)
        lay.setSpacing(4)

        # 存疑横幅（disputed 才显；原始赢的机械体现）
        self.banner = QLabel('')
        self.banner.setWordWrap(True)
        self.banner.setStyleSheet(
            'background:#fdecea; color:#b3261e; padding:4px 8px;'
            'border:1px solid #e57373; font-weight:600;')
        self.banner.hide()
        lay.addWidget(self.banner)

        self._build_control_card(lay)

        head = QHBoxLayout()
        cap = QLabel('概括（自动生成，可能错——与下方原始不一致时以原始为准）')
        cap.setStyleSheet('color:#8a6d1a; font-size:8pt;')
        head.addWidget(cap, 1)
        btn = QPushButton('刷新')
        btn.clicked.connect(self.refresh)
        head.addWidget(btn)
        lay.addLayout(head)

        # 概括层：两列表（条目 / 值）
        self.sum_table = QTableWidget(0, 2)
        self.sum_table.setHorizontalHeaderLabels(['概括', '值'])
        self.sum_table.horizontalHeader().setSectionResizeMode(
            0, QHeaderView.ResizeToContents)
        self.sum_table.horizontalHeader().setSectionResizeMode(
            1, QHeaderView.Stretch)
        self.sum_table.verticalHeader().setVisible(False)
        self.sum_table.setEditTriggers(QTableWidget.NoEditTriggers)
        lay.addWidget(self.sum_table, 2)

        lab = QLabel('原始事件（History 尾部，append-only 不可改）')
        lab.setStyleSheet('color:#5b6270; font-size:8pt;')
        lay.addWidget(lab)

        # 原始层：尾部事件表
        self.raw_table = QTableWidget(0, 5)
        self.raw_table.setHorizontalHeaderLabels(
            ['seq', 'actor', '键', '值', '理由'])
        self.raw_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeToContents)
        self.raw_table.horizontalHeader().setSectionResizeMode(
            3, QHeaderView.Stretch)
        self.raw_table.verticalHeader().setVisible(False)
        self.raw_table.setEditTriggers(QTableWidget.NoEditTriggers)
        lay.addWidget(self.raw_table, 3)

        # 订阅权威史：任何线程的写都编组回主线程刷新
        self._unsub = bridge.subscribe(lambda ev: self._changed.emit())
        self._changed.connect(self.refresh)
        self.refresh()
        # 订阅写权模式：端点线程切的模式也要即时反映（mode 不落史，订这里）
        self._unwatch = bridge.watch_mode(lambda m: self._mode_changed.emit(m))
        self._mode_changed.connect(self._refresh_mode)
        self._refresh_mode(getattr(bridge, 'mode', 'locked'))

    # ---- 写权控制卡（人侧开关；与网页控制台同一效果、同一条 authority 路径）----

    def _build_control_card(self, lay):
        card = QWidget()
        card.setStyleSheet('background:#f7f8fa; border:1px solid #d6d9de;')
        v = QVBoxLayout(card)
        v.setContentsMargins(8, 6, 8, 6)
        v.setSpacing(4)
        row = QHBoxLayout()
        row.addWidget(QLabel('写权模式：'))
        self.mode_label = QLabel('—')
        self.mode_label.setStyleSheet('font-weight:600;')
        row.addWidget(self.mode_label, 1)
        port = getattr(self._bridge, '_human_port', None)
        if port:
            web = QPushButton('网页控制台…')
            web.setToolTip('浏览器打开人工控制端点（token 在 .human-token）——'
                           '不在 GUI 前也能接管/交权')
            web.clicked.connect(lambda: QDesktopServices.openUrl(
                QUrl('http://127.0.0.1:%d/' % port)))
            row.addWidget(web)
        v.addLayout(row)
        btns = QHBoxLayout()
        self._mode_btns = {}
        for key, text, tip in (
                ('human', '人接管', '切回人类写权（AI 的写当场被拒）'),
                ('ai', '交给 AI', '把写权交给 AI（你的界面变只读，'
                                  'AI 的每笔写都记 actor=ai 进史）'),
                ('locked', '全员禁写', '维护态：人和 AI 都写不进')):
            b = QPushButton(text)
            b.setToolTip(tip)
            b.clicked.connect(lambda _c, k=key: self._switch(k))
            btns.addWidget(b)
            self._mode_btns[key] = b
        btns.addStretch(1)
        v.addLayout(btns)
        lay.addWidget(card)

    def _switch(self, mode):
        """人侧切模式（actor='gui' 在 human 侧 ⇒ authority 放行）。"""
        try:
            self._bridge.switch_mode(mode, actor='gui')
        except Exception as e:                              # noqa: BLE001
            self.mode_label.setText('切换被拒：%s' % e)

    def _refresh_mode(self, mode):
        self.mode_label.setText(_MODE_TEXT.get(mode, mode))
        self.mode_label.setStyleSheet(
            'font-weight:600; color:%s;' % _MODE_COLOR.get(mode, '#222'))
        win = self.window()
        binder = getattr(win, 'binder', None)
        if binder is not None:
            # 两模式门控接线（P2/D2）：AI 模式下面板控件物理只读
            binder.set_readonly(mode != 'human')
        if getattr(win, 'app_bridge', None) is self._bridge:
            try:
                win.log({'human': '写权在人（HUMAN）', 'ai': '写权已交给 AI——'
                         '界面只读，可随时点「人接管」夺回',
                         'locked': '全员禁写（LOCKED）'}[mode], 8000)
            except Exception:                               # noqa: BLE001
                pass

    def refresh(self):
        try:
            mv = self._bridge.monitor.read()
        except Exception as e:                              # noqa: BLE001
            self.banner.setText('监控读数失败：%s' % e)
            self.banner.show()
            return
        if mv.disputed:
            self.banner.setText('⚠ 概括存疑（原始赢）：%s' % mv.dispute_reason)
            self.banner.show()
        else:
            self.banner.hide()
        rows = list(mv.summary.items())
        self.sum_table.setRowCount(len(rows))
        for r, (k, v) in enumerate(rows):
            it0 = QTableWidgetItem(str(k))
            if isinstance(v, Claim):
                it1 = QTableWidgetItem(_short(v.value))
                it1.setToolTip('对账目标（原始键）：%s' % (v.target or k))
            else:
                it1 = QTableWidgetItem(_short(v))
            self.sum_table.setItem(r, 0, it0)
            self.sum_table.setItem(r, 1, it1)
        evs = self._bridge.history.events()[-_RAW_TAIL:]
        self.raw_table.setRowCount(len(evs))
        for r, e in enumerate(evs):
            for c, txt in enumerate((str(e.seq), e.actor, e.key,
                                     _short(e.value), e.reason)):
                it = QTableWidgetItem(txt)
                if c == 1 and e.actor in ('ai', 'mcp', 'codeact', 'job'):
                    it.setForeground(Qt.darkRed)            # AI 归因高亮
                self.raw_table.setItem(r, c, it)
        self.raw_table.scrollToBottom()


def attach_monitor_dock(win):
    """给 MainWindow 挂右侧监控 dock + 视图菜单开关（v2 分支专用）。"""
    dock = QDockWidget('AI 监控', win)
    dock.setObjectName('monitor_dock')
    dock.setAllowedAreas(Qt.RightDockWidgetArea)
    panel = MonitorPanel(win.app_bridge, dock)
    dock.setWidget(panel)
    win.addDockWidget(Qt.RightDockWidgetArea, dock)
    win._monitor_dock = dock
    # app_entry 只建了「文件」菜单；「视图」菜单本面板首建（v2 分支）。
    vm = win.menuBar().addMenu('视图')
    act = vm.addAction('AI 监控侧边栏')
    act.setCheckable(True)
    act.setChecked(True)
    act.toggled.connect(dock.setVisible)
    dock.visibilityChanged.connect(act.setChecked)
    return dock
