# -*- coding: utf-8 -*-
"""能级组管理 UI：定义、列表、数值表。"""

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QColor, QIcon, QPixmap
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QSpinBox, QDoubleSpinBox,
    QPushButton, QListWidget, QListWidgetItem, QTableWidget, QTableWidgetItem,
    QHeaderView, QAbstractItemView, QMessageBox,
)

from src import core


class EnergyGroupPanel(QWidget):
    """能级组管理子模块：定义、列表、数值表。"""

    def __init__(self, owner, parent=None):
        super().__init__(parent)
        self.owner = owner
        self.setSizePolicy(self.sizePolicy().Policy.Preferred, self.sizePolicy().Policy.Fixed)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(2)
        self._build_group_area(lay)

    def _build_group_area(self, lay):
        row_add = QHBoxLayout()
        row_add.addWidget(QLabel('n'))
        self.n_spin = QSpinBox(); self.n_spin.setRange(1, 200); self.n_spin.setValue(60)
        row_add.addWidget(self.n_spin)
        row_add.addWidget(QLabel('l'))
        self.l_spin = QSpinBox(); self.l_spin.setRange(0, 199); self.l_spin.setValue(0)
        row_add.addWidget(self.l_spin)
        row_add.addWidget(QLabel('j'))
        self.j_spin = QDoubleSpinBox(); self.j_spin.setRange(0.0, 10.0)
        self.j_spin.setSingleStep(0.5); self.j_spin.setDecimals(1); self.j_spin.setValue(0.5)
        row_add.addWidget(self.j_spin)
        self.n_spin.editingFinished.connect(self.on_nlj_changed)
        self.l_spin.editingFinished.connect(self.on_nlj_changed)
        self.j_spin.editingFinished.connect(self.on_nlj_changed)
        self.label_preview = QLabel(core.auto_label(60, 0, 0.5))
        row_add.addWidget(self.label_preview, 1)
        self.add_button = QPushButton('添加')
        self.add_button.clicked.connect(self.add_group)
        row_add.addWidget(self.add_button)
        lay.addLayout(row_add)

        self.group_list = QListWidget()
        self.group_list.setContextMenuPolicy(Qt.CustomContextMenu)
        self.group_list.customContextMenuRequested.connect(self.show_context_menu)
        self.group_list.setFixedHeight(180)

        self.level_table = QTableWidget(0, 3)
        self.level_table.setHorizontalHeaderLabels(['能级组', 'mJ', 'E (MHz)'])
        self.level_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.level_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Fixed)
        self.level_table.setColumnWidth(1, 48)
        self.level_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.Stretch)
        self.level_table.setFixedHeight(180)
        self.level_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.level_table.verticalHeader().setVisible(False)
        self.level_table.verticalHeader().setDefaultSectionSize(18)
        self.level_table.horizontalHeader().setFixedHeight(36)
        self.level_table.setToolTip('全部 mJ 磁子能级（每行一个）；动态演化基组 = 本表全部能级，无需选择')
        row = QHBoxLayout(); row.setSpacing(6)
        row.addWidget(self.group_list, 1)
        row.addWidget(self.level_table, 1)
        lay.addLayout(row)

    def on_nlj_changed(self, *_):
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
        ok, group = self.owner.groups_store.add(n, l, j)
        if not ok:
            QMessageBox.warning(self, '参数不合法', group)
            return
        pix = QPixmap() if False else None
        from PyQt5.QtGui import QPixmap
        pix = QPixmap(12, 12)
        pix.fill(QColor(group['color']))
        from PyQt5.QtGui import QIcon
        item = QListWidgetItem(QIcon(pix), f"{group['label']}  (n={n}, l={l}, j={j})")
        self.group_list.addItem(item)

    def show_context_menu(self, pos):
        item = self.group_list.itemAt(pos)
        if item is None:
            return
        from PyQt5.QtWidgets import QMenu
        menu = QMenu(self)
        delete_action = menu.addAction('删除此能级组')
        action = menu.exec_(self.group_list.mapToGlobal(pos))
        if action == delete_action:
            row = self.group_list.row(item)
            if row >= 0:
                self.group_list.takeItem(row)
                self.owner.groups_store.remove_at(row)

    def serialize_state(self):
        """本模块负责导出它自己的状态快照。"""
        return {'groups': [dict(g) for g in self.owner.groups_store.groups]}

    def restore_state(self, state):
        """按本模块状态快照恢复。"""
        groups = state.get('groups', []) if isinstance(state, dict) else []
        self.owner.groups_store.groups = [dict(g) for g in groups]
        self.owner.groups = self.owner.groups_store.groups
        self.group_list.clear()
        for g in self.owner.groups_store.groups:
            pix = QPixmap(12, 12)
            pix.fill(QColor(g['color']))
            self.group_list.addItem(QListWidgetItem(QIcon(pix),
                f"{g['label']}  (n={g['n']}, l={g['l']}, j={g['j']})"))
        self.owner.groups_store.changed.emit()
