# -*- coding: utf-8 -*-
"""静态分析面板：能级图 + 谱表 + 雷达图。"""

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QComboBox, QLineEdit,
    QTableWidget, QTableWidgetItem, QHeaderView, QAbstractItemView,
)

from src.ui import ui_radar
from src.ui.ui_single import EnergyPlotPanel


class LevelAnalysisPanel(QWidget):
    """静态分析子模块：能级图、谱表、雷达图。"""

    def __init__(self, owner, parent=None):
        super().__init__(parent)
        self.owner = owner
        self.canvas = EnergyPlotPanel()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)
        self._build_panel_area(layout)

    def _build_panel_area(self, lay):
        panel = QWidget(); pl = QVBoxLayout(panel)
        pl.setContentsMargins(0, 0, 0, 0); pl.setSpacing(4)

        row_y = QHBoxLayout(); row_y.addWidget(QLabel('能级图'))
        self.level_mode_box = QComboBox(); self.level_mode_box.addItems(['单一', 'off/on 双图'])
        self.level_mode_box.setToolTip(
            '能级图显示模式：\n'
            '  单一：一张能级图（Zeeman + AC 位移与箭头），off/on 跃迁连线叠加\n'
            '  off/on 双图：非寻址光（裸能级）与寻址光（移后）各一张并排')
        self.level_mode_box.currentIndexChanged.connect(self.on_level_mode_changed)
        row_y.addWidget(self.level_mode_box)
        row_y.addWidget(QLabel('y↓')); self.y_lower_combo = QComboBox();
        self.y_lower_combo.currentIndexChanged.connect(self.on_y_node_changed); row_y.addWidget(self.y_lower_combo, 1)
        row_y.addWidget(QLabel('↑')); self.y_upper_combo = QComboBox();
        self.y_upper_combo.currentIndexChanged.connect(self.on_y_node_changed); row_y.addWidget(self.y_upper_combo, 1)
        row_y.addWidget(QLabel('参考')); self.spec_ref_edit = QLineEdit('17.2368');
        self.spec_ref_edit.setPlaceholderText('空=不画'); self.spec_ref_edit.setMaximumWidth(110)
        self.spec_ref_edit.returnPressed.connect(self.on_spectrum_ref_changed); row_y.addWidget(self.spec_ref_edit)
        row_y.addWidget(QLabel('谱x↓')); self.spec_freq_lo_edit = QLineEdit(); self.spec_freq_lo_edit.setPlaceholderText('自动')
        self.spec_freq_lo_edit.returnPressed.connect(self.on_spectrum_range_changed); row_y.addWidget(self.spec_freq_lo_edit, 1)
        row_y.addWidget(QLabel('↑')); self.spec_freq_hi_edit = QLineEdit(); self.spec_freq_hi_edit.setPlaceholderText('自动')
        self.spec_freq_hi_edit.returnPressed.connect(self.on_spectrum_range_changed); row_y.addWidget(self.spec_freq_hi_edit, 1)
        pl.addLayout(row_y)
        self.canvas.set_spectrum_ref(17.2368)

        self.peak_pair_label = QLabel(''); self.peak_pair_label.setStyleSheet('font-family: Consolas; font-size: 9pt;')
        self.peak_pair_label.setWordWrap(False); self.peak_pair_label.setFixedHeight(16); pl.addWidget(self.peak_pair_label)

        self.spec_off_table = self.make_spec_table('off'); pl.addWidget(self.spec_off_table)
        self.spec_on_table = self.make_spec_table('on'); pl.addWidget(self.spec_on_table)

        self.radar = ui_radar.CouplingRadarPlot(self.owner, parent=panel)
        pl.addWidget(self.radar)
        self.owner.groups_changed.connect(self.radar.refresh)
        lay.addWidget(panel, 1)

    def set_spectrum_ref(self, value):
        self.canvas.set_spectrum_ref(value)

    def set_level_mode(self, mode):
        self.canvas.set_level_mode(mode)

    @staticmethod
    def parse_range_edit(edit):
        text = edit.text().strip()
        if not text:
            return None
        try:
            return float(text)
        except ValueError:
            return None

    def on_spectrum_range_changed(self):
        lo = self.parse_range_edit(self.spec_freq_lo_edit)
        hi = self.parse_range_edit(self.spec_freq_hi_edit)
        self.canvas.set_spectrum_range(lo, hi)

    def on_spectrum_ref_changed(self):
        text = self.spec_ref_edit.text().strip()
        try:
            f = float(text) if text else None
        except ValueError:
            f = None
        self.canvas.set_spectrum_ref(f)

    def on_level_mode_changed(self, *_):
        mode = 'dual' if self.level_mode_box.currentIndex() == 1 else 'single'
        self.canvas.set_level_mode(mode)

    def on_y_node_changed(self, *_):
        self.owner._y_view_user_set = True
        self.sync_y_combos()
        self.owner.refresh_plot()

    def make_spec_table(self, which):
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
        table.setSizePolicy(table.sizePolicy().Expanding, table.sizePolicy().Maximum)
        table.cellClicked.connect(lambda r, c: self.on_peak_table_clicked(which, r, c))
        return table

    def on_peak_picked(self, which, idx):
        table = self.spec_off_table if which == 'off' else self.spec_on_table
        table.clearSelection()
        if idx < 0 or idx >= table.rowCount():
            return
        self.owner._selected_spec_table = which
        self.owner._selected_spec_row = idx
        table.selectRow(idx)
        table.scrollToItem(table.item(idx, 0))
        it_src = table.item(idx, 1)
        it_tgt = table.item(idx, 2)
        if it_src is not None and it_tgt is not None:
            self.radar.set_transition(which, it_src.text(), it_tgt.text())

    def on_peak_table_clicked(self, which, row, col):
        self.owner._selected_spec_table = which
        self.owner._selected_spec_row = row
        self.canvas.set_spectrum_highlight(which, row)
        table = self.spec_off_table if which == 'off' else self.spec_on_table
        if 0 <= row < table.rowCount():
            it_src = table.item(row, 1)
            it_tgt = table.item(row, 2)
            if it_src is not None and it_tgt is not None:
                self.radar.set_transition(which, it_src.text(), it_tgt.text())

    def update_peak_pair_label(self, off_f, on_f):
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

    def rebuild_y_combos(self):
        self.owner._sorted_groups = sorted(
            self.owner.groups, key=lambda g: self.owner.cache.E_mhz(g['n'], g['l'], g['j']))
        lo = self.y_lower_combo.currentIndex()
        hi = self.y_upper_combo.currentIndex()
        self.y_lower_combo.blockSignals(True)
        self.y_upper_combo.blockSignals(True)
        self.y_lower_combo.clear()
        self.y_upper_combo.clear()
        for g in self.owner._sorted_groups:
            self.y_lower_combo.addItem(g['label'])
            self.y_upper_combo.addItem(g['label'])
        n = len(self.owner._sorted_groups)
        if n > 0:
            if not self.owner._y_view_user_set:
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
        self.sync_y_combos()

    def sync_y_combos(self):
        lo = self.y_lower_combo.currentIndex()
        hi = self.y_upper_combo.currentIndex()
        if lo >= 0 and hi >= 0 and hi < lo:
            self.y_upper_combo.blockSignals(True)
            self.y_upper_combo.setCurrentIndex(lo)
            self.y_upper_combo.blockSignals(False)

    def visible_groups(self):
        if not self.owner._sorted_groups:
            return []
        lo = self.y_lower_combo.currentIndex()
        hi = self.y_upper_combo.currentIndex()
        if lo < 0 or hi < 0:
            return list(self.owner._sorted_groups)
        return self.owner._sorted_groups[max(0, lo):hi + 1]

    @staticmethod
    def populate_peak_table(table, lines):
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

    def refresh_view(self):
        visible = self.visible_groups()
        fs = self.owner.field_state
        B = fs.B_gauss if fs is not None else 0.0
        ac = fs.address_cfg if fs is not None else {'enabled': False}
        self.canvas.draw_levels(self.owner.cache, B, self.owner.groups, ac, True, visible)
        if self.canvas._lines_off is not self.owner._last_lines_off:
            self.owner._last_lines_off = self.canvas._lines_off
            self.populate_peak_table(self.spec_off_table, self.canvas._lines_off)
        if self.canvas._lines_on is not self.owner._last_lines_on:
            self.owner._last_lines_on = self.canvas._lines_on
            self.populate_peak_table(self.spec_on_table, self.canvas._lines_on)
        self.owner._fill_level_table()
