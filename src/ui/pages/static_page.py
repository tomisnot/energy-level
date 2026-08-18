# -*- coding: utf-8 -*-
"""静态页：能级组管理 + 能级图/off-on 频谱 + 两体耦合雷达图。

自包含本页全部业务逻辑（从 能级.py 移入）：
- 能级组区：定义行（n/l/j）、列表（右键删除）、能级数值表、颜色分配
- 静态面板：能级图显示控件（模式/y 上下限/参考频率/谱x 范围）、
  off/on 谱表（构造/填充/点击联动雷达图）、选中峰频率标签
- 数据：groups（能级组列表）、field_B、address_cfg（第一份光同步）
- 画布：EnergyPlotPanel（能级图 + off/on 镜像频谱）

MainWindow 只装配：canvas → 左侧 tab，widget → 右侧面板，events → 订阅。
"""
import numpy as np
from PyQt5.QtCore import Qt, pyqtSignal, QObject
from PyQt5.QtGui import QColor, QIcon, QPixmap
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel,
                             QSpinBox, QDoubleSpinBox, QComboBox, QPushButton,
                             QListWidget, QListWidgetItem, QTableWidget,
                             QTableWidgetItem, QHeaderView, QLineEdit,
                             QMenu, QMessageBox, QAbstractItemView,
                             QSizePolicy)

from ...core import core
from .. import ui_radar
from ..ui_single import EnergyPlotPanel, COLOR_CYCLE


class StaticPage(QObject):
    """静态页：能级组管理 + 能级图/频谱 + 雷达图（信号载体，非 QWidget）。"""

    # 能级组增删事件（页面订阅刷新；也透传给主程序供动态页订阅）
    groups_changed = pyqtSignal()

    def __init__(self, main, parent=None):
        super().__init__(parent)
        self.main = main              # MainWindow（cache 等共用）
        self.cache = main.cache
        # 状态层：能级组 + 场 的单一数据源（阶段2）
        self.groups_store = None      # LevelGroupStore（主程序注入，含 field）
        self.field_state = None       # FieldState（主程序注入）
        self.groups = []              # 转发 groups_store.groups（见 refresh_store）
        self._sorted_groups = []
        self._y_view_user_set = False
        self._last_lines_off = None
        self._last_lines_on = None
        self._selected_spec_table = 'off'
        self._selected_spec_row = -1
        self._level_all = []
        self._level_by_key = {}

        # ---- 画布（左侧 tab）----
        self.canvas = EnergyPlotPanel()

        # ---- 右侧面板：能级组区 + 静态面板 拆成两个独立部件 ----
        # （主程序装配顺序：能级组区 → 场面板 → 静态面板）
        self.group_widget = QWidget()
        self.group_widget.setSizePolicy(
            QSizePolicy.Preferred, QSizePolicy.Fixed)   # 高度=内容，不让外部拉伸（否则底部留白）
        glay = QVBoxLayout(self.group_widget)
        glay.setContentsMargins(0, 0, 0, 0)
        glay.setSpacing(2)
        self._build_group_area(glay)      # 能级组区（定义/列表/数值表）

        self.panel_widget = QWidget()
        play = QVBoxLayout(self.panel_widget)
        play.setContentsMargins(0, 0, 0, 0)
        play.setSpacing(2)
        self._build_panel_area(play)      # 静态面板（能级图控件/谱表/雷达图）

        # widget = 两者合成（Page 注册/全屏适配兼容用）
        self.widget = QWidget()
        wlay = QVBoxLayout(self.widget)
        wlay.setContentsMargins(0, 0, 0, 0)
        wlay.setSpacing(5)
        wlay.addWidget(self.group_widget)
        wlay.addWidget(self.panel_widget)

        # 画布信号 → 谱表/雷达图联动
        self.canvas.peakPicked.connect(self._on_peak_picked)
        self.canvas.peakFreqsChanged.connect(self._update_peak_pair_label)
        self.groups_changed.connect(self.refresh)

    def bind_store(self, groups_store, field_state):
        """主程序注入共享状态（LevelGroupStore + FieldState）。"""
        self.groups_store = groups_store
        self.field_state = field_state
        self.groups = groups_store.groups
        groups_store.changed.connect(self._on_store_groups_changed)

    def _on_store_groups_changed(self):
        """store 能级组变化 → 同步列表 + 刷新。"""
        self.groups = self.groups_store.groups
        self._rebuild_y_combos()
        self.refresh_plot()
        self.groups_changed.emit()

    # ================= 能级组区 =================
    def _build_group_area(self, lay):
        # 定义能级：n/l/j/标签 挤一行
        row_add = QHBoxLayout()
        row_add.addWidget(QLabel('n'))
        self.n_spin = QSpinBox()
        self.n_spin.setRange(1, 200); self.n_spin.setValue(60)
        row_add.addWidget(self.n_spin)
        row_add.addWidget(QLabel('l'))
        self.l_spin = QSpinBox()
        self.l_spin.setRange(0, 199); self.l_spin.setValue(0)
        row_add.addWidget(self.l_spin)
        row_add.addWidget(QLabel('j'))
        self.j_spin = QDoubleSpinBox()
        self.j_spin.setRange(0.0, 10.0)
        self.j_spin.setSingleStep(0.5); self.j_spin.setDecimals(1)
        self.j_spin.setValue(0.5)
        row_add.addWidget(self.j_spin)
        self.n_spin.editingFinished.connect(self._on_nlj_changed)
        self.l_spin.editingFinished.connect(self._on_nlj_changed)
        self.j_spin.editingFinished.connect(self._on_nlj_changed)
        self.label_preview = QLabel(core.auto_label(60, 0, 0.5))
        row_add.addWidget(self.label_preview, 1)
        self.add_button = QPushButton('添加')
        self.add_button.clicked.connect(self.add_group)
        row_add.addWidget(self.add_button)
        self._group_add_row = row_add
        lay.addLayout(row_add)

        # 能级组列表（右键删除）——与能级管理表并排一行
        self.group_list = QListWidget()
        self.group_list.setContextMenuPolicy(Qt.CustomContextMenu)
        self.group_list.customContextMenuRequested.connect(self._show_context_menu)
        self.group_list.setFixedHeight(180)

        # 能级管理表（全部子能级直显）
        self.level_table = QTableWidget(0, 3)
        self.level_table.setHorizontalHeaderLabels(['能级组', 'mJ', 'E (MHz)'])
        self.level_table.horizontalHeader().setSectionResizeMode(
            0, QHeaderView.Stretch)
        self.level_table.horizontalHeader().setSectionResizeMode(
            1, QHeaderView.Fixed)
        self.level_table.setColumnWidth(1, 48)
        self.level_table.horizontalHeader().setSectionResizeMode(
            2, QHeaderView.Stretch)
        self.level_table.setFixedHeight(180)
        self.level_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.level_table.verticalHeader().setVisible(False)
        self.level_table.verticalHeader().setDefaultSectionSize(18)
        self.level_table.horizontalHeader().setFixedHeight(36)
        self.level_table.setToolTip(
            '全部 mJ 磁子能级（每行一个）；动态演化基组 = 本表全部能级，无需选择')
        self._group_row = QHBoxLayout()
        self._group_row.setSpacing(6)
        self._group_row.addWidget(self.group_list, 1)
        self._group_row.addWidget(self.level_table, 1)
        lay.addLayout(self._group_row)

    # ================= 静态面板区 =================
    def _build_panel_area(self, lay):
        panel = QWidget()
        pl = QVBoxLayout(panel)
        pl.setContentsMargins(0, 0, 0, 0)
        pl.setSpacing(4)

        # 能级图显示模式 + y 上下限 + 参考频率 + 谱x 上下限 挤一行
        row_y = QHBoxLayout()
        row_y.addWidget(QLabel('能级图'))
        self.level_mode_box = QComboBox()
        self.level_mode_box.addItems(['单一', 'off/on 双图'])
        self.level_mode_box.setToolTip(
            '能级图显示模式：\n'
            '  单一：一张能级图（Zeeman + AC 位移与箭头），off/on 跃迁连线叠加\n'
            '  off/on 双图：非寻址光（裸能级）与寻址光（移后）各一张并排')
        self.level_mode_box.currentIndexChanged.connect(self._on_level_mode_changed)
        row_y.addWidget(self.level_mode_box)
        row_y.addWidget(QLabel('y↓'))
        self.y_lower_combo = QComboBox()
        self.y_lower_combo.currentIndexChanged.connect(self._on_y_node_changed)
        row_y.addWidget(self.y_lower_combo, 1)
        row_y.addWidget(QLabel('↑'))
        self.y_upper_combo = QComboBox()
        self.y_upper_combo.currentIndexChanged.connect(self._on_y_node_changed)
        row_y.addWidget(self.y_upper_combo, 1)
        row_y.addWidget(QLabel('参考'))
        self.spec_ref_edit = QLineEdit('17.2368')
        self.spec_ref_edit.setPlaceholderText('空=不画')
        self.spec_ref_edit.setMaximumWidth(110)
        self.spec_ref_edit.returnPressed.connect(self._on_spectrum_ref_changed)
        row_y.addWidget(self.spec_ref_edit)
        row_y.addWidget(QLabel('谱x↓'))
        self.spec_freq_lo_edit = QLineEdit()
        self.spec_freq_lo_edit.setPlaceholderText('自动')
        self.spec_freq_lo_edit.returnPressed.connect(self._on_spectrum_range_changed)
        row_y.addWidget(self.spec_freq_lo_edit, 1)
        row_y.addWidget(QLabel('↑'))
        self.spec_freq_hi_edit = QLineEdit()
        self.spec_freq_hi_edit.setPlaceholderText('自动')
        self.spec_freq_hi_edit.returnPressed.connect(self._on_spectrum_range_changed)
        row_y.addWidget(self.spec_freq_hi_edit, 1)
        pl.addLayout(row_y)
        self._on_spectrum_ref_changed()

        # 双谱选中峰频率
        self.peak_pair_label = QLabel('')
        self.peak_pair_label.setStyleSheet(
            'font-family: Consolas; font-size: 9pt;')
        self.peak_pair_label.setWordWrap(False)
        self.peak_pair_label.setFixedHeight(16)
        pl.addWidget(self.peak_pair_label)

        # off/on 谱表
        self.spec_off_table = self._make_spec_table('off')
        pl.addWidget(self.spec_off_table)
        self.spec_on_table = self._make_spec_table('on')
        pl.addWidget(self.spec_on_table)

        # 两体耦合雷达图（面板最下方；parent = 本页 widget）
        self.radar = ui_radar.CouplingRadarPlot(self, parent=panel)
        pl.addWidget(self.radar)
        self.groups_changed.connect(self.radar.refresh)

        lay.addWidget(panel, 1)

    # ================= 页面钩子（主程序 Page 注册用） =================
    def show(self):
        """切到本页时刷新。"""
        self.refresh()

    def on_light(self):
        """场/光变化时刷新。"""
        self.refresh()

    def refresh(self):
        """静态页刷新：能级图/频谱/谱表 + 雷达图。"""
        self.radar.refresh()
        self._sync_address_from_lights()
        self.refresh_plot()

    # ================= 能级组管理 =================
    def _on_nlj_changed(self, *_):
        n = int(self.n_spin.value())
        l = int(self.l_spin.value())
        j = float(self.j_spin.value())
        ok, _ = core.validate_state(n, l, j)
        if ok:
            self.label_preview.setText(core.auto_label(n, l, j))

    def add_group(self):
        n = int(self.n_spin.value())
        l = int(self.l_spin.value())
        j = float(self.j_spin.value())
        ok, group = self.groups_store.add(n, l, j)
        if not ok:
            QMessageBox.warning(self.widget, '参数不合法', group)
            return
        pix = QPixmap(12, 12)
        pix.fill(QColor(group['color']))
        item = QListWidgetItem(
            QIcon(pix), f"{group['label']}  (n={n}, l={l}, j={j})")
        self.group_list.addItem(item)
        # store.changed → _on_store_groups_changed（重建 y 组合 + 刷新 + groups_changed）
        # 列表项已由本方法添加（store 不碰 UI）

    def _show_context_menu(self, pos):
        item = self.group_list.itemAt(pos)
        if item is None:
            return
        menu = QMenu(self.widget)
        delete_action = menu.addAction('删除此能级组')
        action = menu.exec_(self.group_list.mapToGlobal(pos))
        if action == delete_action:
            row = self.group_list.row(item)
            if row >= 0:
                self.group_list.takeItem(row)
                self.groups_store.remove_at(row)

    # ================= 能级图显示 =================
    def _on_spectrum_range_changed(self):
        lo = self._parse_range_edit(self.spec_freq_lo_edit)
        hi = self._parse_range_edit(self.spec_freq_hi_edit)
        self.canvas.set_spectrum_range(lo, hi)

    @staticmethod
    def _parse_range_edit(edit):
        text = edit.text().strip()
        if not text:
            return None
        try:
            return float(text)
        except ValueError:
            return None

    def _on_spectrum_ref_changed(self):
        text = self.spec_ref_edit.text().strip()
        try:
            f = float(text) if text else None
        except ValueError:
            f = None
        self.canvas.set_spectrum_ref(f)

    def _on_level_mode_changed(self, *_):
        mode = 'dual' if self.level_mode_box.currentIndex() == 1 else 'single'
        self.canvas.set_level_mode(mode)

    # ================= 谱表 =================
    def _make_spec_table(self, which):
        table = QTableWidget(0, 4)
        table.setHorizontalHeaderLabels(['Freq (GHz)', 'From', 'To', 'Pol'])
        table.setSelectionBehavior(QAbstractItemView.SelectRows)
        table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        table.verticalHeader().setVisible(False)
        table.setMinimumWidth(0)
        table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Fixed)
        table.setColumnWidth(0, 240)
        table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        table.horizontalHeader().setSectionResizeMode(2, QHeaderView.Stretch)
        table.horizontalHeader().setSectionResizeMode(3, QHeaderView.Fixed)
        table.setColumnWidth(3, 55)
        table.setMinimumHeight(140)
        table.setMaximumHeight(200)
        table.verticalHeader().setDefaultSectionSize(18)
        table.horizontalHeader().setFixedHeight(36)
        table.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Maximum)
        table.cellClicked.connect(
            lambda r, c: self._on_peak_table_clicked(which, r, c))
        return table

    @staticmethod
    def _populate_peak_table(table, lines):
        table.setRowCount(len(lines))
        for i, ln in enumerate(lines):
            nu, A, la, lb, qa, qb, col, pol = ln
            table.setItem(i, 0, QTableWidgetItem(f'{nu / 1e9:.4f}'))
            it1 = QTableWidgetItem(f'{la} {qa}')
            it1.setToolTip(f'{la} {qa}')
            table.setItem(i, 1, it1)
            it2 = QTableWidgetItem(f'{lb} {qb}')
            it2.setToolTip(f'{lb} {qb}')
            table.setItem(i, 2, it2)
            table.setItem(i, 3, QTableWidgetItem(pol))

    def _on_peak_picked(self, which, idx):
        """点击频谱峰：峰表滚动选中 + 联动雷达图跃迁对。"""
        table = self.spec_off_table if which == 'off' else self.spec_on_table
        table.clearSelection()
        if idx < 0 or idx >= table.rowCount():
            return
        self._selected_spec_table = which
        self._selected_spec_row = idx
        table.selectRow(idx)
        table.scrollToItem(table.item(idx, 0))
        it_src = table.item(idx, 1)
        it_tgt = table.item(idx, 2)
        if it_src is not None and it_tgt is not None:
            self.radar.set_transition(which, it_src.text(), it_tgt.text())

    def _update_peak_pair_label(self, off_f, on_f):
        if off_f is None and on_f is None:
            self.peak_pair_label.setText('')
            return
        parts = []
        if off_f is not None:
            parts.append(f'off {off_f:.4f} GHz')
        if on_f is not None:
            parts.append(f'on {on_f:.4f} GHz')
        if off_f is not None and on_f is not None:
            parts.append(f'Δ {(on_f - off_f) * 1000:+.1f} MHz')
        self.peak_pair_label.setText('   |   '.join(parts))

    def _on_peak_table_clicked(self, which, row, col):
        """谱表点击：高亮图中峰 + 联动雷达图跃迁对（off→原子1，on→原子2）。"""
        self._selected_spec_table = which
        self._selected_spec_row = row
        self.canvas.set_spectrum_highlight(which, row)
        table = self.spec_off_table if which == 'off' else self.spec_on_table
        if 0 <= row < table.rowCount():
            it_src = table.item(row, 1)
            it_tgt = table.item(row, 2)
            if it_src is not None and it_tgt is not None:
                self.radar.set_transition(which, it_src.text(), it_tgt.text())

    # ================= y 轴显示范围 =================
    def _rebuild_y_combos(self):
        self._sorted_groups = sorted(
            self.groups, key=lambda g: self.cache.E_mhz(g['n'], g['l'], g['j']))
        lo = self.y_lower_combo.currentIndex()
        hi = self.y_upper_combo.currentIndex()
        self.y_lower_combo.blockSignals(True)
        self.y_upper_combo.blockSignals(True)
        self.y_lower_combo.clear()
        self.y_upper_combo.clear()
        for g in self._sorted_groups:
            self.y_lower_combo.addItem(g['label'])
            self.y_upper_combo.addItem(g['label'])
        n = len(self._sorted_groups)
        if n > 0:
            if not self._y_view_user_set:
                lo, hi = 0, n - 1
            else:
                lo = max(0, min(lo, n - 1))
                hi = max(0, min(hi, n - 1))
                if hi < lo:
                    hi = lo
            self.y_lower_combo.setCurrentIndex(lo)
            self.y_upper_combo.setCurrentIndex(hi)
        self.y_lower_combo.blockSignals(False)
        self.y_upper_combo.blockSignals(False)
        self._sync_y_combos()

    def _on_y_node_changed(self, *_):
        self._y_view_user_set = True
        self._sync_y_combos()
        self.refresh_plot()

    def _sync_y_combos(self):
        lo = self.y_lower_combo.currentIndex()
        hi = self.y_upper_combo.currentIndex()
        if lo >= 0 and hi >= 0 and hi < lo:
            self.y_upper_combo.blockSignals(True)
            self.y_upper_combo.setCurrentIndex(lo)
            self.y_upper_combo.blockSignals(False)

    def _visible_groups(self):
        if not self._sorted_groups:
            return []
        lo = self.y_lower_combo.currentIndex()
        hi = self.y_upper_combo.currentIndex()
        if lo < 0 or hi < 0:
            return list(self._sorted_groups)
        return self._sorted_groups[max(0, lo):hi + 1]

    # ================= 绘制 =================
    def _sync_address_from_lights(self):
        """第一份光同步到 address_cfg（能级图/频谱计算）。"""
        if self.field_state is not None:
            self.field_state.sync_address_from_lights()

    def refresh_plot(self):
        """能级图 + off/on 频谱 + 谱表 + 能级数值表。"""
        try:
            visible = self._visible_groups()
            fs = self.field_state
            B = fs.B_gauss if fs is not None else 0.0
            ac = fs.address_cfg if fs is not None else {'enabled': False}
            rows = self.canvas.draw_levels(
                self.cache, B, self.groups, ac, True, visible)
            if self.canvas._lines_off is not self._last_lines_off:
                self._last_lines_off = self.canvas._lines_off
                self._populate_peak_table(self.spec_off_table,
                                          self.canvas._lines_off)
            if self.canvas._lines_on is not self._last_lines_on:
                self._last_lines_on = self.canvas._lines_on
                self._populate_peak_table(self.spec_on_table,
                                          self.canvas._lines_on)
            self._fill_level_table()
        except Exception as e:
            self.main.log(f'能级图刷新失败: {e}')
            QMessageBox.warning(self.widget, '刷新失败', str(e))

    def _fill_level_table(self):
        """共用能级表：每行一个 mJ 磁子能级。"""
        fs = self.field_state
        B = fs.B_gauss if fs is not None else 0.0
        ac = fs.address_cfg if fs is not None else {'enabled': False}
        self._level_all = core.build_level_rows(
            self.cache, B, self.groups, ac, self.groups)
        self._level_by_key = {
            (r['n'], r['l'], r['j'], r['mj']): r
            for r in self._level_all if r['kind'] == 'mJ'}
        ref = min((r['E0'] for r in self._level_all), default=0.0)
        lowest = {}
        for r in self._level_all:
            key = (r['n'], r['l'], r['j'])
            if key not in lowest or r['E0'] < lowest[key]['E0']:
                lowest[key] = r
        self.level_table.setRowCount(0)
        for g in self.groups:
            key = (g['n'], g['l'], g['j'])
            pix = QPixmap(12, 12)
            pix.fill(QColor(g['color']))
            lo = lowest.get(key)
            for m in np.arange(-g['j'], g['j'] + 1e-9, 1.0):
                r = self.level_table.rowCount()
                self.level_table.insertRow(r)
                self.level_table.setItem(
                    r, 0, QTableWidgetItem(QIcon(pix), g['label']))
                it_m = QTableWidgetItem(f'{m:g}')
                it_m.setTextAlignment(Qt.AlignCenter)
                self.level_table.setItem(r, 1, it_m)
                rm = self._level_by_key.get((key[0], key[1], key[2], float(m)))
                if rm is None:
                    rm = lo
                if rm is not None:
                    self.level_table.setItem(
                        r, 2, QTableWidgetItem(f"{rm['E0'] - ref:.3f}"))
                else:
                    self.level_table.setItem(r, 2, QTableWidgetItem(''))
