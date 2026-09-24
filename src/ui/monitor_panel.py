# -*- coding: utf-8 -*-
"""AI 监控侧边栏（总纲 §1.2 事实 2：第一交付物；v2 默认挂载）。

人类在这里看"AI 在我的软件里干了什么"，两层信任并列（§5.2④）：

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
"""
from __future__ import annotations

from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtWidgets import (QDockWidget, QHBoxLayout, QHeaderView, QLabel,
                             QPushButton, QTableWidget, QTableWidgetItem,
                             QVBoxLayout, QWidget)

from mecha.monitor import Claim

_RAW_TAIL = 200
_MAX_CELL = 60          # 值太长（psi 列表等）截断显示


def _short(v) -> str:
    s = repr(v)
    return s if len(s) <= _MAX_CELL else s[:_MAX_CELL - 1] + '…'


class MonitorPanel(QWidget):
    """两层并列的监控读数面板（只读，无任何写口）。"""

    _changed = pyqtSignal()          # 史事件 → 编组回主线程

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
