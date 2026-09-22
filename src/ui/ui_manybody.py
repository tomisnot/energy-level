# -*- coding: utf-8 -*-
"""多体页（P3-1 M0）：N 原子阵列的配置 + 点云 + 结果。

**页面结构**（裁决③ 2026-09-22："单独开一页，啥布局你自己发挥"；既有页一行不动）：
  中央（本页自己的 tab）：
    ① 原子表（x/y/z/Ω/δ + 逐原子 g/r 量子数）+ 添加/删除按钮
    ② 点云画布（x–y 平面，按 Ω 标注）
    ③ 结果画布（产物态布居 / 逐原子布居）
  右侧参数面板（`manybody_widget`，进 `page_stack` ⇒ 与动态页**互斥占位**，
  不与它抢同一块面板；这是 M3 纪律里唯一仍然生效的那条）：
    模板 / 相互作用项 / 求解器 / 二阶 downfolding / n_window / guard / ed_max_dim /
    初态 ψ0 + 成本提示 + 运行

**真值**：全部 `manybody.*` 键，经 choke point 写 session（人机同源 I5）——本页不持有
平行状态；点云/ψ0/terms 这些非标量走 `app_bridge.write`（`register_external` 的等价手写）。

**运行**：**借 `DynamicPanel._run`**（同一条运行流，I5），先 `set_nbody('many')` 把族
声明为多体，再用自己的回调渲染（`on_done`）。进度条/取消复用底栏常驻控件。

**M0 范围**（未做项显式列在 `NOT_YET`，不放假控件）：
  · 点云的**画布交互编辑**（点击加原子/拖拽）——M1；M0 用表编辑 + 只读点云；
  · 关联矩阵 C_ij 热图 / 纠缠熵 S 曲线——M2（且 S 依赖 P1-8 的 artifact 边界）；
  · `drive_segments` 分段驱动表、`psi0='sup'` 叠加——后续（见 `NOT_YET`）。

**已裁决不做**（裁决⑤ 2026-09-22，见 `DECIDED_NO`）：`manybody.seed`（core 已删）与
`manybody.energy_window_mhz`（core 已删）**都不上控件**，原因写在 `DECIDED_NO` 里并显示在
页面底部——**不留"待裁决"的悬空措辞**（否则后来人会以为还悬着）。
`manybody.template` 已由"预设选择器"**真实现**（core `TEMPLATE_TERMS`）⇒ 本页恢复
ising/xy 单选（**有物理效果**），potts3 下拉项**禁用**（未实现，D3 显式边界）；
`manybody.terms` 是**权威开关**，UI 上用三态控件完整表达 None/list/[] 并写明"显式覆盖预设"。
"""
import numpy as np
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel,
                             QPushButton, QComboBox, QLineEdit, QCheckBox,
                             QTableWidget, QTableWidgetItem, QHeaderView,
                             QAbstractItemView, QMessageBox)
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

from ..core import core
from . import _mecha_bind as MB
from ..core import protocol as _proto
# 模板→预设相互作用项的唯一真源在 core（本页**不复制**这张表，守 I2）
from ..core.manybody.model import TEMPLATE_TERMS

# M0 未做的能力（页面上显式说明，避免"以为有"）
NOT_YET = ('点云画布交互编辑（点击/拖拽加原子）', '关联矩阵 C_ij 热图与纠缠熵 S 曲线',
           '分段驱动 drive_segments 表', 'ψ0 叠加（sup）编辑器')

# **已裁决不做**（2026-09-22 工单 P2-1 / 裁决⑤）——写明原因，避免后来人以为还悬着。
# ⚠ `seed` 是"暂时不做"而非"永久不做"：它要真生效必须先有阵列生成器键
# `manybody.array={kind,n,spacing_um,jitter}`（core 注释里已写明）⇒ 届时**一并恢复**。
DECIDED_NO = (
    ('manybody.seed',
     '已删（core 37701ad）：registry 路径只调 add_atom（逐原子显式坐标），'
     '结构上无随机性可播种；将来加 manybody.array（阵列生成器）时一并恢复本控件'),
    ('manybody.energy_window_mhz',
     '已删（core）：按声明无法实现（能量窗需配 ref_energy_mhz，schema 无此键）；'
     '中间态截断已由 n_window + guard_mhz 覆盖'),
)

# 「相互作用项」三态（与 schema 一一对应，D2：优先级写在界面上）
_TERMS_MODES = ('跟随模板预设', '自定义勾选', '显式无相互作用')

_TERMS = ('I1', 'I2', 'I3')
_TERM_TIP = {'I1': '密度-密度 V·n_i n_j（vdW/blockade 主项）',
             'I2': '交换 J(σ⁺σ⁻)',
             'I3': '双激发 A(σ⁺σ⁺)'}
_COLS = ('x (µm)', 'y (µm)', 'z (µm)', 'Ω (MHz)', 'δ (MHz)',
         'g (n,l,j,m)', 'r (n,l,j,m)')


class ManyBodyPanel(QWidget):
    """多体页：左侧画布区（表/点云/结果）+ 右侧参数页（由主程序放进 page_stack）。"""

    def __init__(self, main, parent=None):
        super().__init__(parent)
        self.main = main
        self.cache = main.cache
        self._atoms = []            # 本地点云缓存（真值 = session 的 manybody.atoms）
        self._last_res = None
        self._build_canvases()
        self.manybody_widget = self._build_controls()
        self._build_module()
        self.refresh_from_session()

    # ================= 画布区 =================
    def _build_canvases(self):
        self.cloud_fig = Figure(figsize=(7, 3.0), dpi=100)
        self.cloud_ax = self.cloud_fig.add_subplot(111)
        self.cloud_canvas = FigureCanvas(self.cloud_fig)
        self.res_fig = Figure(figsize=(7, 3.4), dpi=100)
        self.res_ax = self.res_fig.add_subplot(111)
        self.res_canvas = FigureCanvas(self.res_fig)
        for fig in (self.cloud_fig, self.res_fig):
            fig.subplots_adjust(left=0.08, right=0.96, top=0.92, bottom=0.14)
        self._draw_cloud()
        self._draw_result(None)

    def _build_module(self):
        """本页根布局：原子表 → 点云 → 结果（上下三段，中央区自定版式）。"""
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(4)

        bar = QHBoxLayout()
        bar.setSpacing(6)
        self.atom_label = QLabel('原子（0）')
        self.atom_label.setStyleSheet('font-weight:600;')
        bar.addWidget(self.atom_label)
        self.btn_add = QPushButton('＋添加原子')
        self.btn_add.setToolTip('在能级组最低两组上摆一个原子（位置按已有原子自动错开 6 µm）')
        self.btn_add.clicked.connect(self._add_atom)
        bar.addWidget(self.btn_add)
        self.btn_del = QPushButton('删除选中')
        self.btn_del.clicked.connect(self._del_atom)
        bar.addWidget(self.btn_del)
        self.btn_fill = QPushButton('g/r 取能级组')
        self.btn_fill.setToolTip('把选中行的 g/r 量子数重置为能级组面板里最低的两个组')
        self.btn_fill.clicked.connect(self._fill_gr_from_groups)
        bar.addWidget(self.btn_fill)
        bar.addStretch(1)
        self.hint = QLabel('')
        self.hint.setStyleSheet('color:#6b7280;')
        bar.addWidget(self.hint)
        root.addLayout(bar)

        self.table = QTableWidget(0, len(_COLS))
        self.table.setHorizontalHeaderLabels(list(_COLS))
        self.table.verticalHeader().setVisible(False)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setEditTriggers(QAbstractItemView.AllEditTriggers)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table.setFixedHeight(132)
        self.table.itemChanged.connect(self._on_table_edited)
        root.addWidget(self.table)

        root.addWidget(self.cloud_canvas, 3)
        root.addWidget(self.res_canvas, 3)

    # ================= 参数页（进 page_stack） =================
    def _build_controls(self):
        panel = QWidget()
        p = QVBoxLayout(panel)
        p.setSpacing(6)
        reg = self.main.binder            # _mecha_bind.Binder（与动态页同一绑定层）

        row = QHBoxLayout()
        row.addWidget(QLabel('模板'))
        self.template_cb = MB.combobox(
            reg, 'manybody.template', ['ising', 'xy', 'potts3'],
            spec=MB.ParamSpec('str', options=['ising', 'xy', 'potts3'],
                              desc='局部空间/相互作用预设'))
        self.template_cb.setToolTip(
            '相互作用**预设**（core `TEMPLATE_TERMS`）：\n'
            '  ising → terms=I1（密度-密度 V·n_i n_j）\n'
            '  xy    → terms=I2（交换 J σ⁺σ⁻）\n'
            '两者都是每原子 2 能级（横向场由逐原子 Ω 提供）。\n'
            'potts3：3 能级/原子——**M1 未实现**，选中会在运行时明确报错。\n'
            '⚠ **优先级**：显式 manybody.terms 覆盖本预设（相互作用内容只有一个真源）。')
        # potts3 未实现 ⇒ 下拉项**禁用**（不是删掉：D3 显式边界，让人看得见"有这个档但没做"）
        _pi = self.template_cb.findText('potts3')
        if _pi >= 0:
            self.template_cb.model().item(_pi).setEnabled(False)
        self.template_cb.currentTextChanged.connect(
            lambda _t: self._refresh_terms_display())
        row.addWidget(self.template_cb)
        row.addWidget(QLabel('求解器'))
        self.solver_cb = MB.combobox(
            reg, 'manybody.solver', ['auto', 'ed', 'tebd'],
            spec=MB.ParamSpec('str', options=['auto', 'ed', 'tebd'],
                              desc='多体求解器'))
        self.solver_cb.setToolTip('auto：按 dim=2^N 与 ed_max_dim 自动判可行\n'
                                  'ed：精确对角化（M1）\n'
                                  'tebd：TeNPy/MPS——**M3 未实现**，选中会明确报错')
        _si = self.solver_cb.findText('tebd')
        if _si >= 0:
            self.solver_cb.model().item(_si).setEnabled(False)
        row.addWidget(self.solver_cb)
        row.addStretch(1)
        p.addLayout(row)

        # 相互作用项：**三态**控件，与 schema 一一对应（None / 显式 list / 显式 []）。
        # 为什么用 combo+勾选框而不是"就三个勾选框"：schema 里"跟随模板预设(None)"与
        # "显式不要任何相互作用项([])"是**两个不同的值**，三个勾选框表达不了这个区别，
        # 会造成"取消勾选到底等于哪个"的歧义（D2：多键/同旋钮必须写明优先级）。
        row2 = QHBoxLayout()
        self.terms_mode_cb = QComboBox()
        self.terms_mode_cb.addItems(list(_TERMS_MODES))
        self.terms_mode_cb.setToolTip(
            '跟随模板预设：manybody.terms=None（ising→I1 / xy→I2）\n'
            '自定义勾选：写显式列表，**覆盖**模板预设\n'
            '显式无相互作用：写 []（自由自旋，无 I1/I2/I3）')
        self.terms_mode_cb.currentIndexChanged.connect(self._on_terms_mode_changed)
        row2.addWidget(self.terms_mode_cb)
        self.term_cbs = {}
        for t in _TERMS:
            cb = QCheckBox(t)
            cb.setToolTip(_TERM_TIP[t])
            cb.stateChanged.connect(self._write_terms)
            row2.addWidget(cb)
            self.term_cbs[t] = cb
        self.terms_hint = QLabel('')
        self.terms_hint.setStyleSheet('color:#6b7280;')
        row2.addWidget(self.terms_hint)
        row2.addStretch(1)
        p.addLayout(row2)

        row3 = QHBoxLayout()
        self.second_order_cb = MB.checkbox(
            reg, 'manybody.second_order', True, text='二阶 downfolding',
            spec=MB.ParamSpec('bool', desc='Löwdin 二阶（vdW C6 从伙伴态涌现）'))
        self.second_order_cb.setToolTip('开：二阶 Löwdin downfolding（vdW C6 从伙伴态涌现）\n'
                                        '关：只一阶 V_dd')
        row3.addWidget(self.second_order_cb)
        row3.addWidget(QLabel('伙伴窗 n_window'))
        self.nwin_sb = MB.intbox(reg, 'manybody.n_window', 3, 0, 20,
                                 spec=MB.ParamSpec('int', desc='单原子基伙伴态窗口'))
        self.nwin_sb.setToolTip('单原子基的偶极伙伴态窗口 |Δn| ≤ 本值（补 vdW 中间态）')
        row3.addWidget(self.nwin_sb)
        row3.addStretch(1)
        p.addLayout(row3)

        row4 = QHBoxLayout()
        row4.addWidget(QLabel('guard (MHz)'))
        self.guard_sb = MB.spinbox(reg, 'manybody.guard_mhz', 1.0, 0.0, 1e3,
                                   decimals=3,
                                   spec=MB.ParamSpec('float', unit='MHz',
                                                     desc='近简并中间态剔除阈'))
        self.guard_sb.setToolTip('Löwdin downfolding 的近简并中间态剔除阈；'
                                 '被剔除的个数会在运行旁注里报出（不静默）')
        row4.addWidget(self.guard_sb)
        row4.addWidget(QLabel('ED 维度上限'))
        self.edmax_sb = MB.intbox(reg, 'manybody.ed_max_dim', 1 << 20, 2, 1 << 24,
                                  spec=MB.ParamSpec('int', desc='ED 维度上限 2^N'))
        self.edmax_sb.setToolTip('dim=2^N 超过本值 → solver=auto 明确报错（不静默降级）；\n'
                                 '调大需自负内存')
        row4.addWidget(self.edmax_sb)
        row4.addStretch(1)
        p.addLayout(row4)

        row5 = QHBoxLayout()
        row5.addWidget(QLabel('初态 ψ0'))
        self.psi0_cb = QComboBox()
        self.psi0_cb.addItems(['全基态 |g…g⟩', 'Lanczos 基态', '指定模式串'])
        self.psi0_cb.setToolTip('None=全基态 / "ground"=Lanczos 基态 / '
                                '"grg"=逐原子 g/r 模式串（原子 0 在左）\n'
                                '叠加（sup）编辑器待后续')
        self.psi0_cb.currentIndexChanged.connect(self._write_psi0)
        row5.addWidget(self.psi0_cb)
        self.psi0_edit = QLineEdit()
        self.psi0_edit.setPlaceholderText('如 grg（仅"指定模式串"时生效）')
        self.psi0_edit.setFixedWidth(160)
        self.psi0_edit.editingFinished.connect(self._write_psi0)
        self.psi0_edit.setVisible(False)
        row5.addWidget(self.psi0_edit)
        row5.addStretch(1)
        p.addLayout(row5)

        self.cost_label = QLabel('（点「预估成本」看规模；或切到本页自动预估）')
        self.cost_label.setWordWrap(True)
        self.cost_label.setStyleSheet('color:#374151;')
        p.addWidget(self.cost_label)

        row6 = QHBoxLayout()
        self.btn_cost = QPushButton('预估成本')
        self.btn_cost.setToolTip('纯查询：dim=2^N / ED 可行性 / 预计秒数与峰值内存'
                                 '（不标定、不演化）')
        self.btn_cost.clicked.connect(self.refresh_cost)
        row6.addWidget(self.btn_cost)
        row6.addStretch(1)
        self.run_btn = QPushButton('运行多体演化')
        self.run_btn.setToolTip('运行中再点 = 取消（与动态页共用底栏进度/取消）')
        self.run_btn.clicked.connect(self._on_run_clicked)
        row6.addWidget(self.run_btn)
        p.addLayout(row6)

        self.not_yet = QLabel('本页尚未做：' + '、'.join(NOT_YET) + '。')
        self.not_yet.setWordWrap(True)
        self.not_yet.setStyleSheet('color:#9ca3af; font-size:8pt;')
        p.addWidget(self.not_yet)
        # **已裁决不做**（不是"待裁决"）：写明键名与原因，避免后来人以为还悬着
        self.decided_no = QLabel(
            '已裁决不做：' + '；'.join('%s（%s）' % (k, why)
                                      for k, why in DECIDED_NO) + '。')
        self.decided_no.setWordWrap(True)
        self.decided_no.setStyleSheet('color:#9ca3af; font-size:8pt;')
        self.decided_no.setToolTip('\n'.join('%s\n  %s' % (k, why)
                                             for k, why in DECIDED_NO))
        p.addWidget(self.decided_no)
        p.addStretch(1)
        return panel

    # ================= 钩子（app_entry 的 modules 协议） =================
    def on_show(self):
        """切到本页：从 session 回填 + 预估成本（纯查询，无副作用）。"""
        self.refresh_from_session()
        self.refresh_cost()

    def on_light(self):
        """光场变化：多体的驱动来自 atoms 的 Ω/δ（不是场面板的光场表）⇒ 无需重算。"""
        return

    def refresh_groups(self):
        """能级组变化：只影响「g/r 取能级组」的默认值，不改已有原子。"""
        return

    # ================= 与 session 的往返 =================
    def refresh_from_session(self):
        br = getattr(self.main, 'app_bridge', None)
        if br is None:
            return
        try:
            atoms = br.read('manybody.atoms') or []
        except Exception:                               # noqa: BLE001
            atoms = []
        self._atoms = [dict(a) for a in atoms]
        # 相互作用项三态：None=跟随模板 / list=显式 / []=显式无
        terms = br.read('manybody.terms')
        self.terms_mode_cb.blockSignals(True)
        if terms is None:
            self.terms_mode_cb.setCurrentIndex(0)
        elif len(terms) == 0:
            self.terms_mode_cb.setCurrentIndex(2)
        else:
            self.terms_mode_cb.setCurrentIndex(1)
        self.terms_mode_cb.blockSignals(False)
        # 读路径（真 bug 修，2026-09-22 自测发现）：模式 1 的勾选框**必须**从会话现值
        # 同步——否则加载存档/undo 后勾选框是空的，而 session 里其实有显式列表
        # （显示与真值分叉；且下一次 `_write_terms` 会把空表当作用户意图写回去）。
        if isinstance(terms, (list, tuple)):
            for t, cb in self.term_cbs.items():
                cb.blockSignals(True)
                cb.setChecked(t in terms)
                cb.blockSignals(False)
        self._refresh_terms_display()
        psi = br.read('manybody.psi0')
        self.psi0_cb.blockSignals(True)
        if psi == 'ground':
            self.psi0_cb.setCurrentIndex(1)
        elif isinstance(psi, str) and psi:
            self.psi0_cb.setCurrentIndex(2)
            self.psi0_edit.setText(psi)
        else:
            self.psi0_cb.setCurrentIndex(0)
        self.psi0_cb.blockSignals(False)
        self.psi0_edit.setVisible(self.psi0_cb.currentIndex() == 2)
        self._rebuild_table()

    def _default_gr(self):
        """新原子的 g/r 默认值：能级组最低两组；无组时给 60S/60P 并在状态栏提示。"""
        gs = sorted(getattr(self.main, 'groups', []) or [],
                    key=lambda g: self.cache.E_mhz(g['n'], g['l'], g['j']))
        if len(gs) >= 2:
            g, r = gs[0], gs[1]
            return ([g['n'], g['l'], g['j'], float(g['j'])],
                    [r['n'], r['l'], r['j'], float(r['j'])])
        self.main.log('多体能级：能级组不足 2 个 → 原子默认用 60S1/2 / 60P3/2，'
                      '请在能级组面板添加实际组，或直接编辑表的 g/r 列', 8000)
        return [60, 0, 0.5, 0.5], [60, 1, 1.5, 1.5]

    def _write_atoms(self, reason):
        """点云 → session（choke point）。空表写 None=清除（registry 拒空 list）。"""
        br = getattr(self.main, 'app_bridge', None)
        if br is None:
            return
        try:
            br.write('manybody.atoms', list(self._atoms) or None,
                     reason=reason, actor='gui')
        except Exception as e:                          # noqa: BLE001
            QMessageBox.warning(self, '点云写入失败', str(e))

    def _add_atom(self):
        g, r = self._default_gr()
        n = len(self._atoms)
        self._atoms.append({'g': list(g), 'r': list(r),
                            'pos': [6.0 * n, 0.0, 0.0],
                            'Omega_mhz': 0.0, 'delta_mhz': 0.0})
        self._write_atoms('多体页：添加原子')
        self._rebuild_table()
        self.refresh_cost()

    def _del_atom(self):
        row = self.table.currentRow()
        if not (0 <= row < len(self._atoms)):
            self.main.log('多体页：先在原子表里选中一行再删除')
            return
        del self._atoms[row]
        self._write_atoms('多体页：删除原子')
        self._rebuild_table()
        self.refresh_cost()

    def _fill_gr_from_groups(self):
        row = self.table.currentRow()
        if not (0 <= row < len(self._atoms)):
            self.main.log('多体页：先选中一行再取能级组')
            return
        g, r = self._default_gr()
        self._atoms[row]['g'] = list(g)
        self._atoms[row]['r'] = list(r)
        self._write_atoms('多体页：g/r 取能级组')
        self._rebuild_table()

    def _rebuild_table(self):
        self.table.blockSignals(True)
        self.table.setRowCount(len(self._atoms))
        for i, a in enumerate(self._atoms):
            pos = list(a.get('pos') or [0, 0, 0])
            vals = [pos[0], pos[1], pos[2], a.get('Omega_mhz', 0.0),
                    a.get('delta_mhz', 0.0),
                    ','.join('%g' % x for x in (a.get('g') or [])),
                    ','.join('%g' % x for x in (a.get('r') or []))]
            for c, v in enumerate(vals):
                it = QTableWidgetItem(str(v))
                if c < 5:
                    it.setTextAlignment(Qt.AlignCenter)
                self.table.setItem(i, c, it)
        self.table.blockSignals(False)
        self.atom_label.setText('原子（%d）  dim=2^%d=%d'
                                % (len(self._atoms), len(self._atoms),
                                   1 << len(self._atoms)))
        self._draw_cloud()

    def _on_table_edited(self, item):
        """表编辑 → 解析 → 写 session（非法输入回退，不静默）。"""
        i, c = item.row(), item.column()
        if not (0 <= i < len(self._atoms)):
            return
        a = self._atoms[i]
        txt = (item.text() or '').strip()
        try:
            if c < 5:
                key = ('pos', c) if c < 3 else ('Omega_mhz' if c == 3 else 'delta_mhz')
                v = float(txt)
                if key[0] == 'pos':
                    pos = list(a.get('pos') or [0, 0, 0])
                    pos[key[1]] = v
                    a['pos'] = pos
                else:
                    a[key] = v
            else:
                q = [float(x) for x in txt.replace(' ', '').split(',')]
                if len(q) != 4:
                    raise ValueError('需 4 个数 (n,l,j,m)')
                a['g' if c == 5 else 'r'] = q
        except Exception as e:                          # noqa: BLE001
            QMessageBox.warning(self, '原子表输入非法',
                                '第 %d 行「%s」= %r 无法解析：%s\n（已回退）'
                                % (i + 1, _COLS[c], txt, e))
            self._rebuild_table()
            return
        self._write_atoms('多体页：编辑原子')
        self._draw_cloud()
        self.refresh_cost()

    def _preset_terms(self):
        """当前模板的预设项（core `TEMPLATE_TERMS`，不复制表）。"""
        return list(TEMPLATE_TERMS.get(self.template_cb.currentText(), ('I1',)))

    def _on_terms_mode_changed(self, idx):
        """切换三态：**进入"自定义勾选"时以当前生效值为起点**，不凭空重置。

        ⚠ 修一个真 bug（2026-09-22 自测发现）：模式 0（跟随预设）下勾选框显示的是
        **模板预设**（如 ising→I1）。若切到模式 1 时直接沿用这几个勾选框，会把会话里
        已有的**显式值**（如 ['I2']）静默改成预设（['I1']）——切个模式就把用户设置改了。
        故：进入模式 1 前先从会话现值（None → 模板预设）播种勾选框。
        """
        if int(idx) == 1:
            cur = None
            br = getattr(self.main, 'app_bridge', None)
            if br is not None:
                try:
                    cur = br.read('manybody.terms')
                except Exception:                       # noqa: BLE001
                    cur = None
            seed = (list(cur) if isinstance(cur, (list, tuple))
                    else self._preset_terms())
            for t, cb in self.term_cbs.items():
                cb.blockSignals(True)
                cb.setChecked(t in seed)
                cb.blockSignals(False)
        self._write_terms()

    def _refresh_terms_display(self):
        """按三态刷新勾选框的可用性/勾选与提示（**只显示，不写 session**）。"""
        mode = self.terms_mode_cb.currentIndex()
        if mode == 0:                       # 跟随模板预设
            eff = self._preset_terms()
            for t, cb in self.term_cbs.items():
                cb.blockSignals(True)
                cb.setChecked(t in eff)
                cb.setEnabled(False)
                cb.blockSignals(False)
            self.terms_hint.setText('跟随模板 %s → %s'
                                    % (self.template_cb.currentText(),
                                       '、'.join(eff) or '（无）'))
        elif mode == 1:                     # 自定义勾选（显式）
            for cb in self.term_cbs.values():
                cb.setEnabled(True)
            eff = [t for t in _TERMS if self.term_cbs[t].isChecked()]
            self.terms_hint.setText('显式：%s（覆盖模板预设）'
                                    % ('、'.join(eff) or '（空——请至少勾一项）'))
        else:                               # 显式无相互作用 []
            for t, cb in self.term_cbs.items():
                cb.blockSignals(True)
                cb.setChecked(False)
                cb.setEnabled(False)
                cb.blockSignals(False)
            self.terms_hint.setText('显式 []（自由自旋，无 I1/I2/I3）')

    def _write_terms(self):
        """三态 → `manybody.terms`：None（跟随模板）/ 显式 list / 显式 []。

        与 schema 一一对应；**显式覆盖模板**（core 侧若两者不一致会写 warning，
        本页把优先级直接写在 combo 与提示里，D2）。
        """
        mode = self.terms_mode_cb.currentIndex()
        if mode == 0:
            val = None
        elif mode == 2:
            val = []
        else:
            val = [t for t in _TERMS if self.term_cbs[t].isChecked()]
            if not val:
                # 自定义模式下全不勾 = 语义含糊（None 还是 []？）⇒ 不猜，回退"跟随模板"
                self.main.log('相互作用项：自定义模式下未勾任何项——已回退为'
                              '「跟随模板预设」（要显式清空请选「显式无相互作用」）', 8000)
                self.terms_mode_cb.blockSignals(True)
                self.terms_mode_cb.setCurrentIndex(0)
                self.terms_mode_cb.blockSignals(False)
                val = None
        br = getattr(self.main, 'app_bridge', None)
        if br is not None:
            try:
                br.write('manybody.terms', val, reason='多体页：相互作用项',
                         actor='gui')
            except Exception as e:                      # noqa: BLE001
                self.main.log('写 manybody.terms 失败：%s' % e)
        self._refresh_terms_display()
        self.refresh_cost()

    def _write_psi0(self):
        i = self.psi0_cb.currentIndex()
        self.psi0_edit.setVisible(i == 2)
        val = None if i == 0 else ('ground' if i == 1
                                   else (self.psi0_edit.text().strip() or None))
        br = getattr(self.main, 'app_bridge', None)
        if br is None:
            return
        try:
            br.write('manybody.psi0', val, reason='多体页：初态', actor='gui')
        except Exception as e:                          # noqa: BLE001
            self.main.log('写 manybody.psi0 失败：%s' % e)

    # ================= 成本预估（纯查询） =================
    def refresh_cost(self):
        """`protocol.estimate`（纯查询，不改状态）：dim/ED 可行性/预计秒数与内存。

        ⚠ 用 `src.core.protocol.estimate(cache, snap)`——**不是** `core.estimate`：
        `src.core.core` 是 `static_atom` 的别名模块，其上没有 estimate（实测踩到）。
        """
        br = getattr(self.main, 'app_bridge', None)
        if br is None:
            return
        snap = dict(br.effective() or {})
        snap['dynamic.mode'] = 2                     # 多体分支（不写回 session）
        try:
            est = _proto.estimate(self.cache, snap)
        except Exception as e:                       # noqa: BLE001
            self.cost_label.setText('成本预估失败：%s' % e)
            return
        n = est.get('n_atoms', 0)
        if not n:
            self.cost_label.setText('还没有原子——点「＋添加原子」摆 N 个原子。'
                                    '（多体基 = 2^N 个产物态）')
            return
        parts = ['N=%d 原子，dim=2^%d=%d' % (n, n, est.get('dim', 1 << n)),
                 'ED 可行：%s' % ('是' if est.get('ed_feasible') else '否')]
        if est.get('est_sec') is not None:
            parts.append('预计 ~%ss' % est['est_sec'])
        if est.get('est_peak_gb') is not None:
            parts.append('峰值 ~%s GB' % est['est_peak_gb'])
        txt = '　'.join(parts)
        warns = est.get('warnings') or []
        if warns:
            txt += '\n⚠ ' + '；'.join(str(x) for x in warns)
        self.cost_label.setText(txt)

    # ================= 绘制 =================
    def _draw_cloud(self):
        ax = self.cloud_ax
        ax.clear()
        ax.set_xlabel('x (µm)')
        ax.set_ylabel('y (µm)')
        ax.grid(alpha=0.3)
        if not self._atoms:
            ax.text(0.5, 0.5, '还没有原子（点「＋添加原子」）', ha='center',
                    va='center', transform=ax.transAxes, color='#9ca3af')
            self.cloud_canvas.draw_idle()
            return
        xs = [float((a.get('pos') or [0, 0, 0])[0]) for a in self._atoms]
        ys = [float((a.get('pos') or [0, 0, 0])[1]) for a in self._atoms]
        om = [abs(float(a.get('Omega_mhz') or 0.0)) for a in self._atoms]
        ax.scatter(xs, ys, s=90, c=om, cmap='plasma', edgecolors='#333', zorder=3)
        for i, (x, y) in enumerate(zip(xs, ys)):
            ax.annotate(str(i), (x, y), textcoords='offset points',
                        xytext=(0, 9), ha='center', fontsize=8)
        ax.set_title('原子点云（颜色 = |Ω| MHz；编号 = 表内行号/基序位）', fontsize=9)
        self.cloud_canvas.draw_idle()

    def _draw_result(self, res):
        ax = self.res_ax
        ax.clear()
        ax.set_xlabel('t (ns)')
        ax.set_ylabel('布居')
        ax.grid(alpha=0.3)
        if not res or 'pops' not in res or 't' not in res:
            ax.text(0.5, 0.5, '还没有结果（点「运行多体演化」）', ha='center',
                    va='center', transform=ax.transAxes, color='#9ca3af')
            self.res_canvas.draw_idle()
            return
        pops = np.asarray(res['pops'])
        t = np.asarray(res['t'])
        labels = [s.get('label', str(i)) for i, s in
                  enumerate(res.get('states') or [])]
        if pops.ndim == 1:
            pops = pops[:, None]
        if len(labels) != pops.shape[1]:            # 退化：逐原子边缘（dim 超上限时）
            labels = ['atom%d' % i for i in range(pops.shape[1])]
        avg = pops.mean(axis=0)
        order = np.argsort(avg)[::-1][:min(12, pops.shape[1])]
        for i in order:
            ax.plot(t, pops[:, i], lw=1.2, label=labels[i])
        loss = res.get('loss')
        if loss is not None and np.asarray(loss).max() > 1e-12:
            ax.plot(t, np.asarray(loss), lw=1.6, ls='--', color='#333',
                    label='loss（离场）')
        ax.margins(x=0)
        ax.set_xlim(float(t[0]), float(t[-1]))
        ax.legend(fontsize=7, ncol=1)
        ax.set_title('产物态布居（按平均布居取前 %d）' % len(order), fontsize=9)
        self.res_canvas.draw_idle()

    # ================= 运行（借 DynamicPanel 的同一条运行流） =================
    def _on_run_clicked(self):
        d = self.main.dynamic
        if getattr(d, '_run_active', False):
            d._cancel_run()
            self.run_btn.setText('正在取消…')
            return
        if not self._atoms:
            QMessageBox.information(self, '提示', '先摆原子（点「＋添加原子」）再运行。')
            return
        self.run_btn.setText('运行中（再点取消）')
        d.set_nbody('many')                 # 声明演化族（唯一入口，I5）
        d._run(on_done=self._on_result)

    def _on_result(self, result, error):
        """多体运行结束：收尾 UI（唯一入口）+ 从 artifact 取数组绘图。"""
        self.main.dynamic._finish_run_ui()
        self.run_btn.setText('运行多体演化')
        if error == 'cancelled':
            return
        if error is not None:
            QMessageBox.warning(self, '多体演化失败', str(error))
            return
        res = None
        aid = (result or {}).get('artifact_id')
        try:
            from mecha.ops import load_artifact
            loaded = (load_artifact(self.main.app_bridge.ops.root, aid)
                      if aid else None)
            if loaded is not None:
                res = loaded[1]
        except Exception as e:                          # noqa: BLE001
            self.main.log('多体结果载入失败：%s' % e)
        self._last_res = res
        self._draw_result(res)
        summ = (result or {}).get('summary') or {}
        self.main.log('多体运行完成：method=%s backend=%s %s'
                      % ((result or {}).get('method'), (result or {}).get('backend'),
                         {k: summ.get(k) for k in ('n_atoms', 'dim', 'ok')
                          if k in summ}), 12000)
