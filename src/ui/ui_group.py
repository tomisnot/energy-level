# -*- coding: utf-8 -*-
"""能级组管理 UI：定义、列表、数值表。"""

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QColor, QIcon, QPixmap
from PyQt5.QtWidgets import (QComboBox,
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QSpinBox, QDoubleSpinBox,
    QPushButton, QListWidget, QListWidgetItem, QTableWidget, QTableWidgetItem,
    QHeaderView, QAbstractItemView, QMessageBox,
)

from src.core import core


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
        # B4（2026-09-12）：基组预设 + 维数提示
        #   裁剪基组的真需求不是"删一个组"，而是**按任务一键切换方案**并
        #   **事先看见维数代价**（上限是两体 dim=N²，不是组数）。
        #   走面板已有的原子 set_groups → 同一状态对象/一次 undo/同一归因。
        prow = QHBoxLayout()
        prow.setSpacing(6)
        prow.addWidget(QLabel('基组预设'))
        self.preset_combo = QComboBox()
        self.preset_combo.setToolTip('按任务把基组设为**恰好**这些能级组'
                                    '（原子操作：一次点击、一次撤销）')
        prow.addWidget(self.preset_combo, 1)
        self.preset_apply = QPushButton('应用')
        self.preset_apply.setToolTip('应用所选预设（声明式整表设置）')
        self.preset_apply.clicked.connect(self.apply_preset)
        prow.addWidget(self.preset_apply)
        self.preset_save = QPushButton('存为…')
        self.preset_save.setToolTip('把当前能级组存成预设（写 '
                                    'save_state/basis_presets.json）')
        self.preset_save.clicked.connect(self.save_preset)
        prow.addWidget(self.preset_save)
        self.dim_hint = QLabel('')
        self.dim_hint.setStyleSheet('color:#888; padding:0 8px;')
        prow.addWidget(self.dim_hint)
        lay.addLayout(prow)
        self._presets = {}
        self._load_presets()
        self._refresh_preset_combo()
        try:
            self.owner.groups_store.changed.connect(self._refresh_dim_hint)
        except Exception:                          # noqa: BLE001
            pass
        self._refresh_dim_hint()

    def on_nlj_changed(self, *_):
        n = int(self.n_spin.value())
        l = int(self.l_spin.value())
        j = float(self.j_spin.value())
        ok, _ = core.validate_state(n, l, j)
        if ok:
            self.label_preview.setText(core.auto_label(n, l, j))

    # ---------- B4：基组预设 + 维数提示 ----------
    def _presets_path(self):
        import os
        root = os.path.dirname(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__))))
        return os.path.join(root, 'save_state', 'basis_presets.json')

    def _load_presets(self):
        """读预设（内置三套在文件缺失时兜底）。"""
        import json
        import os
        self._presets = {
            '单体扫描集(12)': [[60, 0, 0.5], [60, 1, 1.5], [60, 1, 0.5],
                           [58, 4, 3.5], [58, 4, 4.5], [59, 2, 2.5],
                           [58, 3, 2.5], [58, 3, 3.5], [59, 2, 1.5],
                           [60, 2, 2.5], [61, 1, 1.5], [57, 4, 3.5]],
            '两体交换集(4)': [[60, 0, 0.5], [60, 1, 1.5], [6, 1, 1.5],
                           [58, 4, 3.5]],
            '两体+freeze(7)': [[60, 0, 0.5], [60, 1, 1.5], [6, 1, 1.5],
                            [59, 2, 2.5], [58, 3, 2.5], [58, 4, 3.5],
                            [58, 4, 4.5]],
        }
        try:
            p = self._presets_path()
            if os.path.exists(p):
                with open(p, encoding='utf-8') as f:
                    self._presets.update(json.load(f))
        except Exception:                          # noqa: BLE001
            pass

    def _refresh_preset_combo(self):
        cur = self.preset_combo.currentText()
        self.preset_combo.clear()
        for name in self._presets:
            self.preset_combo.addItem(name)
        if cur:
            i = self.preset_combo.findText(cur)
            if i >= 0:
                self.preset_combo.setCurrentIndex(i)

    def apply_preset(self):
        """应用所选预设：**声明式**把基组设为恰好这些组（一次 undo）。"""
        name = self.preset_combo.currentText()
        gs = self._presets.get(name)
        if not gs:
            return
        store = self.owner.groups_store
        want = [(int(n), int(l), float(j)) for (n, l, j) in gs]
        keep = {}
        for g in list(store.groups):
            keep[(int(g['n']), int(g['l']), float(g['j']))] = g
        out = []
        for (n, l, j) in want:                     # 保持预设顺序
            if (n, l, j) in keep:
                out.append(keep[(n, l, j)])        # 保留原配色
            else:
                ok, g = store.add(n, l, j)         # 新增（自动配色）
                if ok:
                    out.append(g)
        self.set_groups(out)                       # 面板已有：原子整表设置 + 广播
        self.log_apply(name, len(out)) if hasattr(self, 'log_apply') else None
        self._refresh_dim_hint()

    def save_preset(self):
        """把当前能级组存为预设（人与 AI 共用同一份 json）。"""
        from PyQt5.QtWidgets import QInputDialog
        import json
        name, ok = QInputDialog.getText(self, '存为预设', '预设名：')
        if not ok or not name.strip():
            return
        self._presets[name.strip()] = [
            [int(g['n']), int(g['l']), float(g['j'])]
            for g in self.owner.groups_store.groups]
        try:
            with open(self._presets_path(), 'w', encoding='utf-8') as f:
                json.dump(self._presets, f, ensure_ascii=False, indent=1)
        except Exception:                          # noqa: BLE001
            pass
        self._refresh_preset_combo()
        self.preset_combo.setCurrentText(name.strip())

    def _refresh_dim_hint(self):
        """实时维数提示：上限是**两体 dim = N²**，不是组数。"""
        try:
            gs = self.owner.groups_store.groups
        except Exception:                          # noqa: BLE001
            return
        n_est = 0
        for g in gs:
            try:
                n_est += int(round(2 * float(g['j']) + 1))
            except Exception:                      # noqa: BLE001
                pass
        two = n_est * n_est
        color = '#c62828' if two > 2000 else ('#ef6c00' if two > 900
                                             else '#888')
        self.dim_hint.setStyleSheet('color:%s; padding:0 8px;' % color)
        self.dim_hint.setText('%d 组 ≈%d 态 · 单体 dim=%d · 两体 dim=%d%s'
                              % (len(gs), n_est, n_est, two,
                                 '（两体很慢，建议裁剪）' if two > 2000 else ''))
        self.dim_hint.setToolTip('态数为估算（n=6P 等含超精细子能级时会更多）；'
                                 '两体维数 = 单体态数的平方')

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

    def set_groups(self, groups):
        """整表设置能级组（参数管理 write 通道；存档恢复/AI 载入同路）。"""
        self.owner.groups_store.groups = [dict(g) for g in (groups or [])]
        self.owner.groups = self.owner.groups_store.groups
        self.group_list.clear()
        for g in self.owner.groups_store.groups:
            pix = QPixmap(12, 12)
            pix.fill(QColor(g['color']))
            self.group_list.addItem(QListWidgetItem(QIcon(pix),
                f"{g['label']}  (n={g['n']}, l={g['l']}, j={g['j']})"))
        self.owner.groups_store.changed.emit()
