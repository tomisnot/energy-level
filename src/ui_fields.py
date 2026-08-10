# -*- coding: utf-8 -*-
"""场定义面板：静磁场 + 静电场 + 光场表格 + 关注跃迁 + 初态。

物理模型定义（单体和两体共用同一份"场"）：
- 静磁场 B (G) / 静电场 E (V/cm)：一行，产生 Zeeman / DC-Stark 能移
- 光场：QTableWidget（# | 频率 GHz | 光强 mW/cm² | 偏振），参考谱表样式，
  右键删除；双击编辑频率/光强，偏振下拉。
- 关注跃迁 combobox：选项 = 已创建光场 +（单体页）exchange。
  选某份光 → 该光为响应谱扫频对象（其余作能移）；选 exchange → 不画响应谱，
  画旧版两能谱对比。由 rebuild_focus(show_exchange) 按模式刷新。
- 初态 (n l j m)：两页响应谱共用。

失谐 / 扫描光强 / 作用范围：暂缓（以后加），scope 默认 'both'。
"""
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtWidgets import *


class FieldPanel(QWidget):
    """静磁场 B + 静电场 E + 光场表格 + 关注跃迁 + 初态。"""

    changed = pyqtSignal()
    POLS = ['σ+', 'π', 'σ-']

    def __init__(self, page, parent=None):
        """page：MainWindow（提供 cache/groups 等）。"""
        super().__init__(parent)
        self.page = page
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(4)
        # ---- 静磁场 + 静电场 + 添加光 一行 ----
        row = QHBoxLayout()
        row.addWidget(QLabel('磁场 B (G)'))
        self.b_spin = QDoubleSpinBox()
        self.b_spin.setRange(0.0, 1000000.0); self.b_spin.setDecimals(2)
        self.b_spin.setValue(0.0)
        self.b_spin.setToolTip('静磁场（量子化轴 z）')
        row.addWidget(self.b_spin, 1)
        row.addWidget(QLabel('电场 E (V/cm)'))
        self.e_spin = QDoubleSpinBox()
        self.e_spin.setRange(-1e6, 1e6); self.e_spin.setDecimals(3)
        self.e_spin.setValue(0.0)
        self.e_spin.setToolTip('静电场（DC-Stark 二阶能移）')
        row.addWidget(self.e_spin, 1)
        self.add_btn = QPushButton('＋添加光')
        self.add_btn.setToolTip(
            '添加一份光（双击表格单元格改频率/光强；右键删除；\n'
            '作用范围：两原子都打 / 只打 1 号 / 只打 2 号）')
        self.add_btn.clicked.connect(self.add_light)
        row.addWidget(self.add_btn)
        lay.addLayout(row)
        self.b_spin.valueChanged.connect(self._changed)
        self.e_spin.editingFinished.connect(self._changed)
        # ---- 光场表格 ----
        self.table = QTableWidget(0, 5)
        self.table.setHorizontalHeaderLabels(
            ['#', '频率 (GHz)', '光强 (mW/cm²)', '偏振', '作用范围'])
        self.table.verticalHeader().setVisible(False)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setEditTriggers(
            QAbstractItemView.DoubleClicked | QAbstractItemView.EditKeyPressed)
        hh = self.table.horizontalHeader()
        hh.setSectionResizeMode(QHeaderView.Stretch)
        hh.setSectionResizeMode(0, QHeaderView.Fixed)
        hh.setSectionResizeMode(3, QHeaderView.Fixed)
        hh.setSectionResizeMode(4, QHeaderView.Fixed)
        self.table.setColumnWidth(0, 32)
        self.table.setColumnWidth(3, 56)
        self.table.setColumnWidth(4, 100)
        self.table.verticalHeader().setDefaultSectionSize(22)
        self.table.setFixedHeight(150)
        self.table.itemChanged.connect(self._table_changed)
        self.table.itemSelectionChanged.connect(self._changed)
        self.table.setContextMenuPolicy(Qt.CustomContextMenu)
        self.table.customContextMenuRequested.connect(self._table_menu)
        lay.addWidget(self.table)
        # ---- 关注跃迁 combobox ----
        row = QHBoxLayout()
        row.addWidget(QLabel('关注跃迁'))
        self.focus_box = QComboBox()
        self.focus_box.setToolTip(
            '选择关注哪个跃迁：某份光 → 响应谱扫频对象（其余光作能移）；\n'
            '单体页的 exchange → 旧版两能谱对比（不画响应谱）')
        self.focus_box.currentIndexChanged.connect(self._changed)
        row.addWidget(self.focus_box, 1)
        lay.addLayout(row)
        # ---- 初态（两页响应谱共用）----
        row = QHBoxLayout()
        row.addWidget(QLabel('初态'))
        self.psi_edit = QLineEdit('60 0 0.5 0.5')
        self.psi_edit.setToolTip('初始态 (n l j m)，空格分隔')
        self.psi_edit.editingFinished.connect(self._changed)
        row.addWidget(self.psi_edit, 1)
        lay.addLayout(row)

    # ---------------- 值接口 ----------------
    def _changed(self, *_):
        self.changed.emit()

    def _table_changed(self, *_):
        self.rebuild_focus(self._focus_exchange)
        self.changed.emit()

    def b_gauss(self):
        return float(self.b_spin.value())

    def e_vpcm(self):
        return float(self.e_spin.value())

    # ---------------- 光场表格 ----------------
    def _pol_text(self, r):
        w = self.table.cellWidget(r, 3)
        return w.currentText() if w is not None else 'π'

    def add_light(self):
        r = self.table.rowCount()
        self.table.insertRow(r)
        it = QTableWidgetItem(str(r + 1))
        it.setFlags(Qt.ItemIsEnabled)          # 序号只读
        self.table.setItem(r, 0, it)
        self.table.setItem(r, 1, QTableWidgetItem('16.8267'))
        self.table.setItem(r, 2, QTableWidgetItem('1000'))
        pol = QComboBox()
        pol.addItems(self.POLS)
        pol.setCurrentText('π')
        pol.currentTextChanged.connect(self._table_changed)
        self.table.setCellWidget(r, 3, pol)
        scope = QComboBox()
        scope.addItems(['both', 'atom1', 'atom2'])
        scope.setCurrentText('both')
        scope.setToolTip('作用范围：两原子都打 / 只打 1 号 / 只打 2 号')
        scope.currentTextChanged.connect(self._table_changed)
        self.table.setCellWidget(r, 4, scope)
        self._table_changed()

    def _table_menu(self, pos):
        item = self.table.itemAt(pos)
        if item is None:
            return
        menu = QMenu(self)
        act = menu.addAction('删除此光')
        if menu.exec_(self.table.mapToGlobal(pos)) == act:
            self.table.removeRow(item.row())
            for r in range(self.table.rowCount()):
                self.table.item(r, 0).setText(str(r + 1))
            self._table_changed()

    # ---------------- 关注跃迁 ----------------
    def rebuild_focus(self, show_exchange):
        """刷新"关注跃迁"选项；保持当前选择。show_exchange：单体页加 exchange。"""
        self._focus_exchange = bool(show_exchange)
        cur = self.focus_box.currentIndex()
        self.focus_box.blockSignals(True)
        self.focus_box.clear()
        for r in range(self.table.rowCount()):
            it_f = self.table.item(r, 1)
            f = it_f.text() if it_f is not None else '?'
            self.focus_box.addItem(f'光{r + 1}: {f}GHz {self._pol_text(r)}')
        if show_exchange:
            self.focus_box.addItem('exchange')
        if self.focus_box.count() > 0:
            # 无选中（-1）默认选第一个；否则尽量保持原选择
            idx = 0 if cur < 0 else min(cur, self.focus_box.count() - 1)
            self.focus_box.setCurrentIndex(idx)
        self.focus_box.blockSignals(False)

    def focus_drive_idx(self):
        """关注某份光 → 光索引；关注 exchange（或空）→ -1。"""
        i = self.focus_box.currentIndex()
        if i < 0:
            return -1
        n = self.table.rowCount()
        if i < n:
            return i
        return -1                    # exchange

    def focus_is_exchange(self):
        """是否关注 exchange（选项索引 ≥ 光场数，仅单体页可出现）。"""
        return (self.focus_box.currentIndex() >= 0
                and self.focus_box.currentIndex() >= self.table.rowCount()
                and self._focus_exchange)

    # ---------------- 数据 ----------------
    def light_cfgs(self):
        """全部光的配置列表（frequency_hz/polarization/intensity/scope）。"""
        cfgs = []
        for r in range(self.table.rowCount()):
            it_f = self.table.item(r, 1)
            it_i = self.table.item(r, 2)
            if it_f is None or it_i is None:
                continue
            try:
                f_hz = float(it_f.text()) * 1e9
                inten = float(it_i.text())
            except ValueError:
                continue
            scope_w = self.table.cellWidget(r, 4)
            scope = scope_w.currentText() if scope_w is not None else 'both'
            cfgs.append(dict(frequency_hz=f_hz, polarization=self._pol_text(r),
                             intensity=inten, scope=scope, enabled=True))
        return cfgs

    def first_cfg(self):
        """第一份光的配置（用于单体 address 同步）；无光返回 None。"""
        cfgs = self.light_cfgs()
        return cfgs[0] if cfgs else None

    def psi_ket(self):
        """解析初态 'n l j m'；无效返回 None。"""
        parts = self.psi_edit.text().split()
        if len(parts) != 4:
            return None
        try:
            return (int(parts[0]), int(parts[1]),
                    float(parts[2]), float(parts[3]))
        except ValueError:
            return None
