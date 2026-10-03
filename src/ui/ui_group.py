# -*- coding: utf-8 -*-
"""能级组管理 UI：定义、列表、数值表。"""

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QColor, QIcon, QPixmap
from PyQt5.QtWidgets import (
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
        # ⚠ 逐子能级裁剪**没有**工具条按钮（2026-09-14 用户裁决：与右键入口重复、
        #   且这一行控件太挤）。唯一入口 = **能级组列表的右键菜单**「子能级…」
        #   （见 show_context_menu：右键哪一行就改哪一行，每行都够得着）。
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
        # 2026-09-14 用户裁决：**删掉「基组预设 + 维数提示」整行**。
        #   理由（用户原话）："这个功能完全可以被别的更正规的功能替代……逐组增删、
        #   状态保存，不是可以完美覆盖基组预设吗？" ⇒ 入口收敛到
        #   ① 逐组增删（n/l/j + 添加 / 右键删除 / 右键「子能级…」裁剪）
        #   ② 正规的状态保存/恢复（存档 round-trip，含裁剪与配色）
        #   代价信息（原来的 dim 提示）由 **t84 运行前成本报告** 与 `estimate_cost()` 覆盖：
        #   前者在点「运行」时给出保留对数/最远失谐/预计步数并让你选"完整物理/剔远对",
        #   后者（AI 面）给 `basis_n/dim/est_seconds/est_peak_gb/mem_advice`。
        #   用户数据文件 `save_state/basis_presets.json` 当时保留在盘上（变惰性）；
        #   2026-09-14 项目瘦身时该文件已按用户批准一并删除。

    def on_nlj_changed(self, *_):
        n = int(self.n_spin.value())
        l = int(self.l_spin.value())
        j = float(self.j_spin.value())
        ok, _ = core.validate_state(n, l, j)
        if ok:
            self.label_preview.setText(core.auto_label(n, l, j))

    # ---------- （2026-09-14 已删）基组预设 + 维数提示 ----------
    # 用户裁决："这个功能完全可以被别的更正规的功能替代，把它完完全全删干净……"
    # 原来这里有 _presets_path / _load_presets / _refresh_preset_combo /
    # apply_preset / save_preset / _refresh_dim_hint 六个方法与一整行 UI。
    # 正规替代：① 逐组增删 + 右键「子能级…」裁剪；② 状态保存/恢复（存档 round-trip，
    # 含裁剪与配色）；③ 代价信息 → t84 运行前成本报告 + `estimate_cost()`。
    # 用户数据 `save_state/basis_presets.json` 已于 2026-09-14 随瘦身删除；
    # 内置三套预设的原始内容在 git 历史里可追（commit e0c11fb 之前）。

    def add_group(self):
        n = int(self.n_spin.value())
        l = int(self.l_spin.value())
        j = float(self.j_spin.value())
        ok, group = self.owner.groups_store.add(n, l, j)
        if not ok:
            QMessageBox.warning(self, '参数不合法', group)
            return
        self._refresh_rows()
        self.owner.groups_store.changed.emit()

    def _row_text(self, g):
        """行文本：带裁剪显示 `60P_{3/2}  (n=60, l=1, j=1.5)  (2/4)`。"""
        base = (f"{g['label']}  (n={g['n']}, l={g['l']}, j={g['j']})")
        try:
            n_all = len(core._group_sublevel_rows(
                self.owner.cache if hasattr(self.owner, 'cache') else None,
                int(g['n']), int(g['l']), float(g['j']), 0.0)) \
                if getattr(self.owner, 'cache', None) is not None else None
        except Exception:                          # noqa: BLE001
            n_all = None
        ms, Fs = core.trim_of(g)
        if ms is None and Fs is None:
            return base
        n_keep = len(list(ms if ms is not None else Fs))
        return base + ('  (%d/%s)' % (n_keep, n_all if n_all else '?'))

    def _refresh_rows(self):
        """按 store 重画列表（保留选中行）。"""
        cur = self.group_list.currentRow()
        self.group_list.clear()
        for g in self.owner.groups_store.groups:
            pix = QPixmap(12, 12)
            pix.fill(QColor(g['color']))
            self.group_list.addItem(QListWidgetItem(QIcon(pix), self._row_text(g)))
        if 0 <= cur < self.group_list.count():
            self.group_list.setCurrentRow(cur)

    def refresh_list(self):
        """对外入口：订阅 `groups_store.changed`（增删/裁剪/**恢复存档**都要重画）。

        旧设计只靠 `add_group`/裁剪/删除**显式**调 `_refresh_rows()` ⇒ 任何“不改
        控件而直接换 store”的路径（如恢复存档走 `store.replace()`）会让列表空着。
        收拢到信号上：单一真源，新增写路径不会漏刷新。
        """
        self._refresh_rows()

    def edit_selected_trim(self, row=None):
        """打开"子能级"勾选框（默认全选）；改完走 registry 同一条通道（P2）。

        ⚠ **"选中"指左侧那张「能级组管理」列表**（`self.group_list`，彩色条目那张）
        ——**不是**右侧 `level_table`（逐子能级列 mJ / E 的那张）。`row=None` 时取
        `self.group_list.currentRow()`；右键菜单传**显式 row**（右键哪行改哪行，
        每行都够得着）。**该语义刻意保持不变。**

        ⚠ 用户口径：**裁剪是为省算力/做收敛检验，不是把观测量调好看**；
        裁掉之后必须**观测量不变**，变了就老实报告（说明它承载物理或基组未收敛）。
        """
        if row is None:
            row = self.group_list.currentRow()
        gs = self.owner.groups_store.groups
        if not (0 <= row < len(gs)):
            QMessageBox.information(self, '提示', '先在列表里选中一个能级组。')
            return
        g = gs[row]
        cache = getattr(self.owner, 'cache', None)
        if cache is None:
            return
        B = 0.0
        try:
            B = float(self.owner.field.b_gauss())
        except Exception:                          # noqa: BLE001
            pass
        rows = core._group_sublevel_rows(cache, int(g['n']), int(g['l']),
                                         float(g['j']), B)
        if not rows:
            return
        from PyQt5.QtWidgets import (QDialog, QDialogButtonBox, QCheckBox,
                                     QScrollArea)
        dlg = QDialog(self)
        dlg.setWindowTitle('子能级裁剪 — %s' % g.get('label'))
        lay = QVBoxLayout(dlg)
        lay.addWidget(QLabel('勾选要**保留**的子能级（默认全选）；'
                             '全选 = 不裁剪（与改前逐位一致）'))
        area = QScrollArea(dlg); area.setWidgetResizable(True)
        host = QWidget(); hl = QVBoxLayout(host)
        ms, Fs = core.trim_of(g)
        chosen = set()
        if ms is not None:
            chosen = {(r['qstr']) for r in rows
                      if r.get('kind') == 'mJ' and float(r['mj']) in
                      {round(float(x), 6) for x in ms}}
        elif Fs is not None:
            want = {(round(float(a), 6), round(float(b), 6)) for a, b in Fs}
            chosen = {r['qstr'] for r in rows if r.get('kind') == 'F'
                      and (round(float(r['F']), 6), round(float(r['mF']), 6))
                      in want}
        boxes = []
        for r in rows:
            cb = QCheckBox('%s   （%.4f MHz）' % (r['qstr'], r['E_mhz']))
            cb.setChecked((not chosen) or (r['qstr'] in chosen))
            hl.addWidget(cb)
            boxes.append((cb, r))
        area.setWidget(host); lay.addWidget(area)
        bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel, dlg)
        bb.accepted.connect(dlg.accept); bb.rejected.connect(dlg.reject)
        lay.addWidget(bb)
        if dlg.exec_() != QDialog.Accepted:
            return
        keep = [r for cb, r in boxes if cb.isChecked()]
        if len(keep) == len(rows):                 # 全选 → 清掉裁剪字段
            g.pop('ms', None); g.pop('Fs', None)
        elif all(r.get('kind') == 'F' for r in keep):
            g['Fs'] = [[float(r['F']), float(r['mF'])] for r in keep]
            g.pop('ms', None)
        else:
            g['ms'] = [float(r['mj']) for r in keep if r.get('kind') == 'mJ']
            g.pop('Fs', None)
        self._refresh_rows()
        try:
            self.log_apply('子能级裁剪 %s' % g.get('label'),
                           len(keep)) if hasattr(self, 'log_apply') else None
        except Exception:                          # noqa: BLE001
            pass
        self.owner.groups_store.changed.emit()      # → registry 归因 + 重绘/成本重算

    def show_context_menu(self, pos):
        """能级组列表（`self.group_list`）右键菜单 —— **逐子能级裁剪的唯一入口**。

        2026-09-14 用户裁决：工具条上的「子能级…」按钮已删（与这里重复 + 那一行太挤），
        故**口径指引搬到本 action**（原按钮 tooltip 的两句话不能丢）：
        "裁剪选中能级组的子能级（默认全选；裁剪只为省算力/做收敛检验，
        判据是'观测量不变'）"。
        菜单作用在**右键命中的那一行**（`row = self.group_list.row(item)`，显式传参），
        所以每行都够得着；若未命中行则直接返回。
        """
        item = self.group_list.itemAt(pos)
        if item is None:
            return
        from PyQt5.QtWidgets import QMenu
        menu = QMenu(self)
        trim_action = menu.addAction('子能级…')
        trim_action.setToolTip('裁剪选中能级组的子能级（默认全选；裁剪只为省算力/'
                               '做收敛检验，判据是"观测量不变"）')
        trim_action.setStatusTip('裁剪能级组的子能级：默认全选；只为省算力/做收敛检验，'
                                 '判据是裁掉后观测量不变（变了要老实报告）')
        delete_action = menu.addAction('删除此能级组')
        action = menu.exec_(self.group_list.mapToGlobal(pos))
        row = self.group_list.row(item)
        if action == trim_action and row >= 0:
            self.edit_selected_trim(row)           # ← 显式 row：右键哪行改哪行
        elif action == delete_action and row >= 0:
            self.group_list.takeItem(row)
            self.owner.groups_store.remove_at(row)

    def set_groups(self, groups):
        """整表设置能级组（参数管理 write 通道；存档恢复/AI 载入同路）。

        **原样保留** `ms`/`Fs` 等附加字段（逐子能级裁剪）——这是**状态存档
        round-trip** 不丢字段的前提（原「基组预设」也走这条路，该功能已删）。
        """
        self.owner.groups_store.groups = [dict(g) for g in (groups or [])]
        self.owner.groups = self.owner.groups_store.groups
        self._refresh_rows()
        self.owner.groups_store.changed.emit()
