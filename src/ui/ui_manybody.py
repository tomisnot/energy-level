# -*- coding: utf-8 -*-
"""多体页（P3-1）：N 原子阵列的配置 + 点云 + 结果。

**一个模式按钮分两条工作流**（演化 与 平衡态 彻底分家）：页顶「时间演化 / 平衡态」互斥
按钮控制只显示对应组件（参数页的两个分区 + 中央区的两个结果画布都随之切换）：
  · 时间演化（`manybody.want=''`）：含时求解器 `manybody.solver`、初态 ψ0、ED 维度上限；
    结果画布 = 产物态布居 / ⟨n_i⟩(t) / C_ij。
  · 平衡态（`manybody.want∈gs/thermal/dos`）：请求量、后端 `manybody.eq_backend`、
    betas；**无 ψ0/时间轴概念**；结果画布 = E₀+gap+⟨n_i⟩ 或 E/C–β 或 ρ(E)。
  几何点云表、能级状态、标定旋钮（二阶/n_window/guard）、成本预估为两条腿（人/AI）共用。

**页面结构**（裁决③ 2026-09-22：“单独开一页，布局自己发挥”）：
  中央（本页自己的 tab）：点云画布 + 演化结果画布 / 平衡态结果区（按模式显隐）
  右侧参数页（`manybody_widget`，进 `page_stack` ⇒ 与动态页**互斥占位**）：
    能级状态 → **模式按钮** → 几何表 → 模式专属分区 → 标定/成本/运行

**真值**：全部 `manybody.*` 键，经 choke point 写 session（人机同源 I5）——本页不持有
平行状态；点云/ψ0/betas 这些非标量走 `app_bridge.write`。

**运行**：**借 `DynamicPanel._run`**（同一条运行流，I5），先 `set_nbody('many')`，再用自己
的回调渲染（`on_done`）。进度条/取消复用底栏常驻控件。

**未做项**显式列在 `NOT_YET`；**已裁决不做**列在 `DECIDED_NO`（不悬空措辞）。
相互作用项：模板/项勾选已从界面移除，由能级标定**恒全开**（I1+I2+I3）。g/r 直接取能级组裁剪后的
全部能级（需恰好 2 个，否则顶部一行红警告且不能加原子）。
"""
import numpy as np
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel,
                             QPushButton, QComboBox, QLineEdit, QCheckBox,
                             QTableWidget, QTableWidgetItem, QHeaderView,
                             QAbstractItemView, QMessageBox, QButtonGroup)
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

from ..core import core
from . import _mecha_bind as MB
from ..core import protocol as _proto
# 模板→预设相互作用项的唯一真源在 core（本页**不复制**这张表，守 I2）
from ..core.manybody.model import TEMPLATE_TERMS
# 观测量（M2）：只吃概率向量 ⇒ 可由 GUI 从联合 `pops` 重算，**不依赖 P1-8 的 artifact 边界**
from ..core.manybody.observables import (bit_matrix as _bit_matrix,
                                         corr_matrix as _corr_matrix)

# M0/M1/M2 未做的能力（页面上显式说明，避免"以为有"）
NOT_YET = ('纠缠熵 S 曲线（S 由 psi 算，需 P1-8 把 psi/S 落进 artifact）',
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
_COLS = ('x (µm)', 'y (µm)', 'z (µm)', 'Ω (MHz)', 'δ (MHz)')


class ManyBodyPanel(QWidget):
    """多体页：左侧画布区（表/点云/结果）+ 右侧参数页（由主程序放进 page_stack）。"""

    def __init__(self, main, parent=None):
        super().__init__(parent)
        self.main = main
        self.cache = main.cache
        self._atoms = []            # 本地点云缓存（真值 = session 的 manybody.atoms）
        self._last_res = None
        self._run_n_atoms = 0       # **运行时刻**的原子数（结果图解析基序要用它，别用当前表）
        self._build_canvases()
        self.manybody_widget = self._build_controls()
        self._build_module()
        # 能级组增删/裁剪 → 重算能级数（恰好 2 个才有 g/r），与动态页同一事件机制。
        try:
            main.groups_changed.connect(self.refresh_groups)
        except Exception:                               # noqa: BLE001
            pass
        self.refresh_from_session()

    # ================= 画布区 =================
    def _build_canvases(self):
        self.cloud_fig = Figure(figsize=(7, 3.0), dpi=100)
        self.cloud_ax = self.cloud_fig.add_subplot(111)
        self.cloud_canvas = FigureCanvas(self.cloud_fig)
        # 结果图三格（M2）：产物态布居 | 逐原子 ⟨n_i⟩(t) | 关联 C_ij（末态）
        self.res_fig = Figure(figsize=(10.5, 3.2), dpi=100)
        self.res_ax = self.res_fig.add_subplot(131)
        self.res_ax_atom = self.res_fig.add_subplot(132)
        self.res_ax_corr = self.res_fig.add_subplot(133)
        self.res_canvas = FigureCanvas(self.res_fig)
        self.cloud_fig.subplots_adjust(left=0.08, right=0.96, top=0.92, bottom=0.14)
        self.res_fig.subplots_adjust(left=0.06, right=0.97, top=0.88, bottom=0.17,
                                     wspace=0.34)
        # M1：点云画布交互（左键空白=加原子；左键点中原子并拖动=移动，**松开才落账**）
        self._drag_idx = None
        self.cloud_canvas.mpl_connect('button_press_event', self._on_cloud_press)
        self.cloud_canvas.mpl_connect('motion_notify_event', self._on_cloud_move)
        self.cloud_canvas.mpl_connect('button_release_event', self._on_cloud_release)
        self.cloud_canvas.setToolTip(
            '左键点空白：在该位置加一个原子\n'
            '左键点中原子并拖动：移动（松开时才写 session，避免事件风暴）\n'
            '删除/改 Ω/δ：用右侧参数页的表格与按钮（g/r 自动取自能级组）')
        self._draw_cloud()
        self._draw_result(None)
        # 平衡态结果区（仅“平衡态”模式可见）：左=主曲线、右=逐原子⟨n_i⟩/摘要格
        self.eq_summary = QLabel('（未求解）')
        self.eq_summary.setWordWrap(True)
        self.eq_summary.setStyleSheet('color:#374151;')
        self.eq_fig = Figure(figsize=(10.5, 3.0), dpi=100)
        self.eq_ax_main = self.eq_fig.add_subplot(121)
        self.eq_ax_txt = self.eq_fig.add_subplot(122)
        self.eq_canvas = FigureCanvas(self.eq_fig)
        self.eq_fig.subplots_adjust(left=0.08, right=0.96, top=0.88, bottom=0.16,
                                    wspace=0.28)
        self._draw_eq_result(None)

    def _build_module(self):
        """本页根（中央画布）：点云 + **按模式二选一**的结果区（演化三格 / 平衡态）。

        几何表/工具条在右侧参数页顶部（=主窗口场面板下方）。
        """
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(4)
        root.addWidget(self.cloud_canvas, 3)
        root.addWidget(self.res_canvas, 3)          # 时间演化结果
        root.addWidget(self.eq_summary, 0)          # 平衡态结果（+ 画布）
        root.addWidget(self.eq_canvas, 3)

    def _apply_mode(self):
        """按 `self._mode` 只显示对应组件（参数分区 + 中央结果区 + 运行按钮文案）。"""
        eq = self._mode == 'eq'
        self.sec_evolve.setVisible(not eq)
        self.sec_eq.setVisible(eq)
        self.res_canvas.setVisible(not eq)
        self.eq_canvas.setVisible(eq)
        self.eq_summary.setVisible(eq)
        self.btn_mode_evolve.setChecked(not eq)
        self.btn_mode_eq.setChecked(eq)
        self.run_btn.setText('求解平衡态' if eq else '运行多体演化')
        self._draw_cloud()

    # ================= 参数页（进 page_stack） =================
    def _build_controls(self):
        panel = QWidget()
        p = QVBoxLayout(panel)
        p.setSpacing(6)
        reg = self.main.binder            # _mecha_bind.Binder（与动态页同一绑定层）

        # —— 能级来源/校验：g/r 取自能级组裁剪后的能级，需恰好 2 个 ——
        self.level_status = QLabel('')
        self.level_status.setWordWrap(True)
        p.addWidget(self.level_status)

        # —— 模式按钮：演化 与 平衡态 彻底分家，只显示对应组件 ——
        self._mode = 'evolve'
        mbar = QHBoxLayout()
        mbar.setSpacing(6)
        mbar.addWidget(QLabel('模式'))
        self.btn_mode_evolve = QPushButton('时间演化')
        self.btn_mode_eq = QPushButton('平衡态')
        for _b in (self.btn_mode_evolve, self.btn_mode_eq):
            _b.setCheckable(True)
            _b.setStyleSheet(
                'QPushButton{padding:3px 10px;} QPushButton:checked'
                '{background:#1d4ed8; color:white; font-weight:600;}')
            mbar.addWidget(_b)
        self._mode_grp = QButtonGroup(self)
        self._mode_grp.setExclusive(True)
        self._mode_grp.addButton(self.btn_mode_evolve)
        self._mode_grp.addButton(self.btn_mode_eq)
        self.btn_mode_evolve.clicked.connect(lambda: self._set_mode('evolve'))
        self.btn_mode_eq.clicked.connect(lambda: self._set_mode('eq'))
        self.btn_mode_evolve.setToolTip(
            '含时演化：给初态 ψ0，沿时间轴演化（求解器 auto/ed/tebd）')
        self.btn_mode_eq.setToolTip(
            '平衡态：不求时间轴，直接解基态/有限温/态密度'
            '（后端 auto/ed/gpu-batch/lanczos/kpm/dmrg）')
        mbar.addStretch(1)
        p.addLayout(mbar)

        # 两个模式专属分区（先建容器，控件填在下面；放在几何表之后、标定之前）
        self.sec_evolve = QWidget()
        self._ev_lay = QVBoxLayout(self.sec_evolve)
        self._ev_lay.setContentsMargins(0, 0, 0, 0)
        self._ev_lay.setSpacing(6)
        self.sec_eq = QWidget()
        self._eq_lay = QVBoxLayout(self.sec_eq)
        self._eq_lay.setContentsMargins(0, 0, 0, 0)
        self._eq_lay.setSpacing(6)

        # —— 几何点云表（从中央画布顶部移到参数页顶部 ⇒ 主窗口里就在场面板正下方）——
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
        bar.addStretch(1)
        self.hint = QLabel('')
        self.hint.setStyleSheet('color:#6b7280;')
        bar.addWidget(self.hint)
        p.addLayout(bar)

        self.table = QTableWidget(0, len(_COLS))
        self.table.setHorizontalHeaderLabels(list(_COLS))
        self.table.verticalHeader().setVisible(False)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setEditTriggers(QAbstractItemView.AllEditTriggers)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table.setFixedHeight(132)
        self.table.itemChanged.connect(self._on_table_edited)
        p.addWidget(self.table)

        p.addWidget(self.sec_evolve)
        p.addWidget(self.sec_eq)

        # ================= 时间演化专属：含时求解器（ψ0 / ED 上限在下面一并归入）
        row = QHBoxLayout()
        row.addWidget(QLabel('含时求解器'))
        self.solver_cb = MB.combobox(
            reg, 'manybody.solver', ['auto', 'ed', 'tebd'],
            spec=MB.ParamSpec('str', options=['auto', 'ed', 'tebd'],
                              desc='多体含时求解器'))
        self.solver_cb.setToolTip('auto：按 dim=2^N 与 ed_max_dim 自动判可行\n'
                                  'ed：精确对角化（M1）\n'
                                  'tebd：TeNPy/MPS——**未实现**，选中会明确报错')
        _si = self.solver_cb.findText('tebd')
        if _si >= 0:
            self.solver_cb.model().item(_si).setEnabled(False)
        row.addWidget(self.solver_cb)
        row.addStretch(1)
        self._ev_lay.addLayout(row)

        # ================= 平衡态专属：请求量 + 后端 + betas =================
        req = QLabel('平衡态请求量')
        req.setStyleSheet('font-weight:600;')
        self._eq_lay.addWidget(req)
        rowe = QHBoxLayout()
        rowe.addWidget(QLabel('求什么'))
        self.want_cb = MB.combobox(
            reg, 'manybody.want', ['gs', 'thermal', 'dos'],
            spec=MB.ParamSpec('str', options=['gs', 'thermal', 'dos'],
                              desc='平衡态请求量'))
        self.want_cb.setToolTip('gs=基态（E₀/gap/⟨n_i⟩）\n'
                                'thermal=有限温 E/C/F–β（KPM 谱方法）\n'
                                'dos=态密度 ρ(E)（KPM）')
        rowe.addWidget(self.want_cb)
        rowe.addWidget(QLabel('后端'))
        self.backend_cb = MB.combobox(
            reg, 'manybody.eq_backend',
            ['auto', 'ed', 'gpu-batch', 'lanczos', 'kpm', 'dmrg'],
            spec=MB.ParamSpec('str',
                              options=['auto', 'ed', 'gpu-batch', 'lanczos',
                                       'kpm', 'dmrg'],
                              desc='平衡态求解后端'))
        self.backend_cb.setToolTip(
            'auto：按规模路由（小→ed、大稀疏→lanczos、有 GPU 且批量→gpu-batch、'
            'thermal/dos→kpm、大而几何规整→dmrg）\n'
            'ed/gpu-batch/lanczos/kpm：显式指定\n'
            'dmrg：TeNPy 柱形 DMRG——**任意点云**（多腿/无周期/逐键复 V·J·A）')
        rowe.addWidget(self.backend_cb)
        rowe.addStretch(1)
        self._eq_lay.addLayout(rowe)

        rowb = QHBoxLayout()
        rowb.addWidget(QLabel('χ (DMRG 键维)'))
        self.chi_sb = MB.intbox(reg, 'manybody.chi_max', 200, 2, 4000,
                                spec=MB.ParamSpec('int', desc='DMRG 最大键维 χ'))
        self.chi_sb.setToolTip('仅后端=dmrg 时生效：χ 越大越准、越慢（代价 ~χ³）。\n'
                               '先用 60~120 看趋势，要数字再上 200+；收敛看结果图里的 χ/truncerr')
        rowb.addWidget(self.chi_sb)
        rowb.addWidget(QLabel('betas (1/MHz)'))
        self.betas_edit = QLineEdit('0.1, 0.5, 1, 3, 10')
        self.betas_edit.setPlaceholderText('如 0.1, 0.5, 1, 3, 10（仅 thermal 用）')
        self.betas_edit.setToolTip('逆温度清单，逗号/空格分隔；β→∞ 趋基态。'
                                   'gs/dos 忽略本项')
        self.betas_edit.editingFinished.connect(self._write_betas)
        rowb.addWidget(self.betas_edit)
        rowb.addStretch(1)
        self._eq_lay.addLayout(rowb)

        _eq_note = QLabel('平衡态**不走时间轴**：没有初态 ψ0、也没有时长/步数——只解'
                          '当前几何+驱动所对应的哈密顿量。')
        _eq_note.setStyleSheet('color:#6b7280;')
        _eq_note.setWordWrap(True)
        self._eq_lay.addWidget(_eq_note)

        # 相互作用项：模板/项勾选已撤下——**恒全开（I1+I2+I3）**，第一性全相互作用单一口径。
        _terms_note = QLabel('相互作用：由能级标定自动全开（V·nn ＋ 交换 J ＋ 双激发 A）；'
                             '不再有“模板/项”开关。')
        _terms_note.setStyleSheet('color:#6b7280;')
        _terms_note.setWordWrap(True)
        p.addWidget(_terms_note)

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
        row4.addStretch(1)
        p.addLayout(row4)

        # 演化专属：ED 维度上限 + 初态 ψ0
        rowe2 = QHBoxLayout()
        rowe2.addWidget(QLabel('ED 维度上限'))
        self.edmax_sb = MB.intbox(reg, 'manybody.ed_max_dim', 1 << 20, 2, 1 << 24,
                                  spec=MB.ParamSpec('int', desc='ED 维度上限 2^N'))
        self.edmax_sb.setToolTip('dim=2^N 超过本值 → solver=auto 明确报错（不静默降级）；\n'
                                 '调大需自负内存')
        rowe2.addWidget(self.edmax_sb)
        rowe2.addStretch(1)
        self._ev_lay.addLayout(rowe2)

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
        self._ev_lay.addLayout(row5)

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
        self._apply_mode()                      # 初始只显示默认模式（时间演化）的组件
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
        """能级组变化：重算能级数→更新警告；恰好 2 个则把原子 g/r 同步为这 2 个。"""
        self._sync_atoms_gr()

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
        # 相互作用项**恒全开**（模板/项勾选已撤下）⇒ 不再回填/显示 terms。
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
        # 模式回填：`manybody.want` 非空 ⇒ 平衡态（只读控件，**不反写** session）。
        w = str(br.read('manybody.want') or '').lower()
        self._mode = 'eq' if w in ('gs', 'thermal', 'dos') else 'evolve'
        self.want_cb.blockSignals(True)
        if w in ('gs', 'thermal', 'dos'):
            self.want_cb.setCurrentText(w)
        self.want_cb.blockSignals(False)
        bts = br.read('manybody.betas')
        if bts:
            self.betas_edit.setText(', '.join('%g' % float(b) for b in bts))
        self._apply_mode()
        self._rebuild_table()
        self._refresh_level_status()

    def _base_states(self):
        """能级组「当前裁剪后的全部态」（原样，含 kind）→ (states, note)。

        与动态基组同一 `runspec.build_states` ⇒ 两页的能级基准不漂移。
        `note` 非空 = 解析失败的原因（不静默返空，否则界面只显示“0 个”而看不出所以）。
        """
        groups = [dict(g) for g in (getattr(self.main, 'groups', None) or [])]
        if not groups:
            return [], '能级组为空——先在能级组面板添加并裁剪。'
        br = getattr(self.main, 'app_bridge', None)
        B = E = 0.0
        if br is not None:
            try:
                B = float(br.read('field.B_gauss') or 0.0)
                E = float(br.read('field.E_vpcm') or 0.0)
            except Exception:                           # noqa: BLE001
                pass
        try:
            from src.core import runspec as _rs
            return list(_rs.build_states(self.cache, groups, B, E)), ''
        except Exception as e:                          # noqa: BLE001
            return [], '解析能级组失败：%s: %s' % (type(e).__name__, e)

    def _levels(self):
        """能级组裁剪后的全部能级 → [(n,l,j,m)…]；**只取 m_J 基**的态。"""
        sts, _note = self._base_states()
        return [(float(s['n']), float(s['l']), float(s['j']), float(s['m']))
                for s in sts if s.get('kind') == 'mJ']

    def _level_diag(self, n_mj):
        """mJ 能级数不是 2 时，给出具体原因（超精细 / 解析失败 / 就是多了）。"""
        sts, note = self._base_states()
        if note:
            return note
        n_f = sum(1 for s in sts if s.get('kind') != 'mJ')
        if n_f:
            return ('其中 %d 个是**超精细 (F,mF)** 展开的态（如 5S 基态）——多体的'
                    '2 能级基只定在 m_J 子能级上，请用 ms 裁剪或改用里德堡态。' % n_f)
        if not sts:
            return '能级组裁剪后一个态也不剩（ms/Fs 可能裁得太狠）。'
        return ''

    def _refresh_level_status(self):
        lv = self._levels()
        if len(lv) == 2:
            self.level_status.setText(
                '能级（来自能级组）：g=%s、r=%s — 恰好 2 个 ✓' % (lv[0], lv[1]))
            self.level_status.setStyleSheet('color:#047857;')
        else:
            why = self._level_diag(len(lv))
            self.level_status.setText(
                '⚠ 多体页的 g/r 取自「能级组」裁剪后的能级，需**恰好 2 个**；'
                '当前 m_J 基 %d 个。%s请在能级组面板处理到 2 个。'
                % (len(lv), (why + ' ') if why else ''))
            self.level_status.setStyleSheet('color:#b91c1c; font-weight:600;')

    def _sync_atoms_gr(self):
        """能级组变动：恰好 2 个时把已存原子 g/r 同步为这 2 个（g/r 由能级组决定）。"""
        lv = self._levels()
        if len(lv) != 2 or not self._atoms:
            self._refresh_level_status()
            return
        ch = False
        for a in self._atoms:
            if (list(a.get('g') or []) != list(lv[0])
                    or list(a.get('r') or []) != list(lv[1])):
                a['g'], a['r'] = list(lv[0]), list(lv[1])
                ch = True
        if ch:
            self._write_atoms('多体页：g/r 随能级组同步')
            self._rebuild_table()
        self._refresh_level_status()

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
        lv = self._levels()
        if len(lv) != 2:
            # g/r 直接取自能级组当前裁剪后的能级，需恰好 2 个。
            # 不弹模态框：顶部 `level_status` 已是常驻“一行警告”，另记入日志。
            self._refresh_level_status()
            self.main.log('多体页：需恰好 2 个能级才能加原子（请先在能级组面板裁剪），'
                          '当前 %d 个' % len(lv))
            return
        g, r = lv
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

    def _rebuild_table(self):
        self.table.blockSignals(True)
        self.table.setRowCount(len(self._atoms))
        for i, a in enumerate(self._atoms):
            pos = list(a.get('pos') or [0, 0, 0])
            vals = [pos[0], pos[1], pos[2], a.get('Omega_mhz', 0.0),
                    a.get('delta_mhz', 0.0)]
            for c, v in enumerate(vals):
                it = QTableWidgetItem(str(v))
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
            if c < 3:
                pos = list(a.get('pos') or [0, 0, 0])
                pos[c] = float(txt)
                a['pos'] = pos
            elif c == 3:
                a['Omega_mhz'] = float(txt)
            elif c == 4:
                a['delta_mhz'] = float(txt)
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

    # ================= 模式（时间演化 / 平衡态）与平衡态参数 =================
    def _set_mode(self, mode):
        """模式按钮：切换可见组件 **并**把 `manybody.want` 写成对应值（演化=''）。"""
        self._mode = 'eq' if str(mode) == 'eq' else 'evolve'
        self._apply_mode()
        self._write_want()

    def _write_want(self):
        """want 真值：演化=''（走含时路径）；平衡态=下拉选中（gs/thermal/dos）。

        经 choke point（app_bridge.write）——与控件同一写路（人机同路 I5）。
        """
        val = '' if self._mode == 'evolve' else (self.want_cb.currentText() or 'gs')
        br = getattr(self.main, 'app_bridge', None)
        if br is None:
            return
        try:
            br.write('manybody.want', val, reason='多体页：模式/请求量', actor='gui')
        except Exception as e:                          # noqa: BLE001
            self.main.log('写 manybody.want 失败：%s' % e)

    def _write_betas(self):
        """betas 文本 → float 清单（逗号/空格分隔）；空写 None（=core 默认 [1.0]）。"""
        vals = []
        for tok in self.betas_edit.text().replace(',', ' ').split():
            try:
                vals.append(float(tok))
            except ValueError:
                self.main.log('多体页：betas 忽略非法项 %r' % tok)
        br = getattr(self.main, 'app_bridge', None)
        if br is None:
            return
        try:
            br.write('manybody.betas', vals or None,
                     reason='多体页：逆温度清单', actor='gui')
        except Exception as e:                          # noqa: BLE001
            self.main.log('写 manybody.betas 失败：%s' % e)

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

    # ================= 点云画布交互（M1） =================
    def _pick_atom(self, event):
        """点中的原子下标（容差 = 坐标范围的 6%）；没点中 → -1。"""
        if event.xdata is None or event.ydata is None or not self._atoms:
            return -1
        xs = np.array([float((a.get('pos') or [0, 0, 0])[0]) for a in self._atoms])
        ys = np.array([float((a.get('pos') or [0, 0, 0])[1]) for a in self._atoms])
        x0, x1 = self.cloud_ax.get_xlim()
        y0, y1 = self.cloud_ax.get_ylim()
        tol = 0.06 * max(abs(x1 - x0), abs(y1 - y0), 1e-9)
        d = np.hypot(xs - float(event.xdata), ys - float(event.ydata))
        i = int(np.argmin(d))
        return i if d[i] <= tol else -1

    def _on_cloud_press(self, event):
        """左键空白 → 加原子（经 choke point）；左键点中原子 → 起拖。"""
        if getattr(event, 'button', None) != 1 or event.xdata is None:
            return
        i = self._pick_atom(event)
        if i >= 0:
            self._drag_idx = i
            if 0 <= i < self.table.rowCount():
                self.table.selectRow(i)
            return
        lv = self._levels()
        if len(lv) != 2:                        # g/r 由能级组决定，需恰好 2 个
            self._refresh_level_status()
            return
        g, r = lv
        self._atoms.append({'g': list(g), 'r': list(r),
                            'pos': [float(event.xdata), float(event.ydata), 0.0],
                            'Omega_mhz': 0.0, 'delta_mhz': 0.0})
        self._write_atoms('多体页：画布点击添加原子')
        self._rebuild_table()
        self.refresh_cost()

    def _on_cloud_move(self, event):
        """拖动中**只重画、不写 session**——每次 mouse-move 都写会灌爆总线（事件风暴）。"""
        if self._drag_idx is None or event.xdata is None or event.ydata is None:
            return
        if not (0 <= self._drag_idx < len(self._atoms)):
            return
        a = self._atoms[self._drag_idx]
        pos = list(a.get('pos') or [0, 0, 0])
        pos[0], pos[1] = float(event.xdata), float(event.ydata)
        a['pos'] = pos
        self._draw_cloud()

    def _on_cloud_release(self, event):
        """松开 → **落账一次**（choke point）+ 表/成本同步。"""
        if self._drag_idx is None:
            return
        self._drag_idx = None
        self._write_atoms('多体页：画布拖动移动原子')
        self._rebuild_table()
        self.refresh_cost()

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
        """结果三格（M2）：产物态布居 | 逐原子 ⟨n_i⟩(t) | 关联 C_ij（末态）。

        ⟨n_i⟩/C_ij **从联合 `pops` 重算**（`observables.atom_pops/corr_matrix` 只吃概率
        向量）——**不依赖 P1-8 的 artifact 边界**；只有当 `pops` 已退化为逐原子边缘
        （dim 超全布居上限，`engine_ed.py:336-341`）时 C_ij 不可得，此时**明说原因**。
        纠缠熵 S 需要 psi（不在 artifact 里）⇒ 若结果里带 S 就画，否则在标题写明来源。
        """
        ax = self.res_ax
        ax.clear()
        ax.set_xlabel('t (ns)')
        ax.set_ylabel('布居')
        ax.grid(alpha=0.3)
        for _a in (self.res_ax_atom, self.res_ax_corr):
            _a.clear()
        if not res or 'pops' not in res or 't' not in res:
            ax.text(0.5, 0.5, '还没有结果（点「运行多体演化」）', ha='center',
                    va='center', transform=ax.transAxes, color='#9ca3af')
            for _a in (self.res_ax_atom, self.res_ax_corr):
                _a.text(0.5, 0.5, '—', ha='center', va='center',
                        transform=_a.transAxes, color='#9ca3af')
            self.res_canvas.draw_idle()
            return
        pops = np.asarray(res['pops'], dtype=float)
        t = np.asarray(res['t'], dtype=float)
        labels = [s.get('label', str(i)) for i, s in
                  enumerate(res.get('states') or [])]
        if pops.ndim == 1:
            pops = pops[:, None]
        n_atoms = int(getattr(self, '_run_n_atoms', 0) or 0)
        dim_joint = (1 << n_atoms) if n_atoms else 0
        joint = bool(n_atoms and pops.shape[1] == dim_joint)
        if len(labels) != pops.shape[1]:            # 退化：逐原子边缘（dim 超上限时）
            labels = (['atom%d' % i for i in range(pops.shape[1])]
                      if not joint else ['#%d' % i for i in range(pops.shape[1])])
        # ---- 左：产物态布居（按平均布居取前 N） ----
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
        ax.legend(fontsize=6, ncol=1)
        ax.set_title('产物态布居（前 %d）' % len(order), fontsize=9)
        # ---- 中：逐原子 ⟨n_i⟩(t)（联合 → 重算；边缘 → 直接用） ----
        axa = self.res_ax_atom
        axa.set_xlabel('t (ns)')
        axa.set_ylabel(r'$\langle n_i\rangle$ 激发概率')
        axa.grid(alpha=0.3)
        try:
            if joint:
                bits = _bit_matrix(n_atoms, dim_joint)      # (n, dim)
                atom_t = pops @ bits.T                      # (n_t, n)
            else:
                atom_t = pops                               # 已是逐原子边缘
            for i in range(atom_t.shape[1]):
                axa.plot(t, atom_t[:, i], lw=1.3, label='atom%d' % i)
            axa.margins(x=0)
            axa.set_xlim(float(t[0]), float(t[-1]))
            axa.legend(fontsize=6, ncol=2)
            _s = res.get('S')
            if _s is not None and np.size(_s) == t.size:
                axb = axa.twinx()
                axb.plot(t, np.asarray(_s), lw=1.1, ls=':', color='#854F0B')
                axb.set_ylabel('S (nat)', fontsize=8)
            axa.set_title(r'逐原子 $\langle n_i\rangle$(t)%s'
                          % ('（由联合 pops 重算）' if joint else '（边缘）'),
                          fontsize=9)
        except Exception as e:                              # noqa: BLE001
            axa.text(0.5, 0.5, r'$\langle n_i\rangle$ 计算失败：%s' % e, ha='center', va='center',
                     transform=axa.transAxes, color='#A32D2D', fontsize=8)
        # ---- 右：关联 C_ij（末态，与引擎口径一致：engine_ed 用最后一步的 p） ----
        axc = self.res_ax_corr
        if joint and pops.shape[0]:
            try:
                C = _corr_matrix(_bit_matrix(n_atoms, dim_joint), pops[-1])
                im = axc.imshow(C, cmap='RdBu_r', vmin=-max(abs(C).max(), 1e-12),
                                vmax=max(abs(C).max(), 1e-12))
                axc.set_xticks(range(n_atoms))
                axc.set_yticks(range(n_atoms))
                axc.set_xlabel('j')
                axc.set_ylabel('i')
                self.res_fig.colorbar(im, ax=axc, fraction=0.046, pad=0.04)
                axc.set_title('关联 C_ij（末态，max|C|=%.3g）' % abs(C).max(),
                              fontsize=9)
            except Exception as e:                          # noqa: BLE001
                axc.text(0.5, 0.5, 'C_ij 计算失败：%s' % e, ha='center', va='center',
                         transform=axc.transAxes, color='#A32D2D', fontsize=8)
        else:
            axc.text(0.5, 0.5, 'C_ij 不可得：pops 已退化为逐原子边缘\n'
                               '（dim=2^%d 超全布居上限，联合分布未物化）'
                     % n_atoms if n_atoms else 'C_ij 不可得：未知原子数',
                     ha='center', va='center', transform=axc.transAxes,
                     color='#6b7280', fontsize=8)
            axc.set_xticks([])
            axc.set_yticks([])
            axc.set_title('关联 C_ij', fontsize=9)
        self.res_canvas.draw_idle()

    # ================= 运行（借 DynamicPanel 的同一条运行流） =================
    @staticmethod
    def _arr(res, key):
        """取结果里的数组字段。

        ⚠ 绝不能写 `np.asarray(res.get(key) or [], float)`：值是 numpy 数组时
        `arr or []` 会抛“truth value of an array ... is ambiguous”（实测整个结果图
        被自己的异常处理接住，只在图上打一行红字）。必须显式判 None。
        """
        v = res.get(key)
        if v is None:
            return np.zeros(0, dtype=float)
        try:
            return np.asarray(v, dtype=float).ravel()
        except (TypeError, ValueError):
            return np.zeros(0, dtype=float)

    def _draw_eq_result(self, res):
        """平衡态结果：主格画图 + 右格读数（与含时的三格结果图互斥显示）。

        gs → 逐原子 ⟨n_i⟩ 柱 + E₀/gap；thermal → E(β) 与 C(β)（双 y 轴）；dos → ρ(E)。
        """
        axm, axt = self.eq_ax_main, self.eq_ax_txt
        axm.clear()
        axt.clear()
        axt.set_axis_off()
        if not res:
            axm.text(0.5, 0.5, '还没有平衡态结果（选「求什么」后点「求解平衡态」）',
                     ha='center', va='center', transform=axm.transAxes,
                     color='#9ca3af')
            axt.text(0.5, 0.5, '—', ha='center', va='center',
                     transform=axt.transAxes, color='#9ca3af')
            self.eq_summary.setText('（未求解）')
            self.eq_canvas.draw_idle()
            return
        want = str(res.get('want') or res.get('method') or '')
        bits = []
        try:
            if want == 'gs':
                na = self._arr(res, 'n_atom')
                if na.size:
                    axm.bar(range(na.size), na, color='#2563EB')
                    axm.set_xticks(range(na.size))
                    axm.set_ylim(0, max(1.0, float(na.max()) * 1.15))
                    axm.set_xlabel('atom i')
                    axm.set_ylabel(r'$\langle n_i\rangle$ 激发概率')
                    axm.grid(alpha=0.3, axis='y')
                else:
                    axm.text(0.5, 0.5, r'该后端未给逐原子 $\langle n_i\rangle$',
                             ha='center',
                             va='center', transform=axm.transAxes, color='#6b7280')
                axm.set_title(r'基态 $\langle n_i\rangle$（后端 %s）'
                              % res.get('backend', '?'), fontsize=9)
                for k, lbl in (('E0', 'E0'), ('gap', 'gap'),
                               ('S', 'S'), ('chi', 'χ')):
                    if res.get(k) is not None:
                        bits.append('%s = %.6g %s' % (lbl, float(res[k]),
                                                      'MHz' if k in ('E0', 'gap') else ''))
            elif want == 'thermal':
                b = self._arr(res, 'betas')
                E = self._arr(res, 'E')
                C = self._arr(res, 'C')
                if b.size and E.size == b.size:
                    axm.plot(b, E, 'o-', color='#2563EB', label='E')
                    axm.set_xscale('log')
                    axm.set_xlabel('β (1/MHz)')
                    axm.set_ylabel('E (MHz)')
                    axm.grid(alpha=0.3)
                    handles, labels = axm.get_legend_handles_labels()
                    if C.size == b.size:
                        axb = axm.twinx()
                        axb.plot(b, C, 's--', color='#B45309', label='C')
                        axb.set_ylabel('C', fontsize=8)
                        h2, l2 = axb.get_legend_handles_labels()
                        handles, labels = handles + h2, labels + l2
                    axm.legend(handles, labels, fontsize=7, loc='best')
                    axm.set_title('有限温 E(β)/C(β)（后端 %s）'
                                  % res.get('backend', '?'), fontsize=9)
                    bits.append('β 点数 = %d' % b.size)
                    for k in ('E_first_last', 'C_first_last', 'F_first_last'):
                        v = res.get(k)
                        if v is not None and len(v) == 2:
                            bits.append('%s = [%.4g, %.4g]' % (k, v[0], v[1]))
                else:
                    axm.text(0.5, 0.5, '结果里没有 betas/E 数组', ha='center',
                             va='center', transform=axm.transAxes, color='#A32D2D')
            elif want == 'dos':
                e = self._arr(res, 'energies')
                rho = self._arr(res, 'rho')
                if e.size and rho.size == e.size:
                    axm.plot(e, rho, color='#7C3AED')
                    axm.set_xlabel('E (MHz)')
                    axm.set_ylabel('ρ(E)')
                    axm.grid(alpha=0.3)
                axm.set_title('态密度 ρ(E)（后端 %s）' % res.get('backend', '?'),
                              fontsize=9)
                bits.append('分箱 = %d' % e.size)
                for k in ('rho_max', 'E_at_rho_max'):
                    if res.get(k) is not None:
                        bits.append('%s = %.4g' % (k, float(res[k])))
            else:
                axm.text(0.5, 0.5, '不认识的 want=%r' % want, ha='center',
                         va='center', transform=axm.transAxes, color='#A32D2D')
        except Exception as e:                              # noqa: BLE001
            axm.text(0.5, 0.5, '绘图失败：%s' % e, ha='center', va='center',
                     transform=axm.transAxes, color='#A32D2D', fontsize=8)
        head = 'want=%s   后端=%s   N=%s   dim=%s' % (
            want, res.get('backend', '?'), res.get('n_atoms', '?'),
            res.get('dim', '?'))
        body = '\n'.join(bits) if bits else '（无额外读数）'
        # matplotlib 面板里把 E0 写成 mathtext（雅黑缺 `₀` 字形，直接用会变方框）；
        # 顶部 QLabel 走 Qt 字体，保留纯文本更可读。
        axt.text(0.02, 0.96, head + '\n' + body.replace('E0 =', r'$E_0$ ='),
                 va='top', ha='left', fontsize=8, transform=axt.transAxes)
        self.eq_summary.setText(head + '   ｜   ' + body.replace('\n', '  ·  '))
        self.eq_canvas.draw_idle()

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
        self._run_n_atoms = len(self._atoms)     # 记**运行时刻**的 N（结果图解析基序用）
        d.set_nbody('many')                 # 声明演化族（唯一入口，I5）
        d._run(on_done=self._on_result)

    def _on_result(self, result, error):
        """多体运行结束：收尾 UI（唯一入口）+ 按模式选结果渲染器。"""
        self.main.dynamic._finish_run_ui()
        self.run_btn.setText('求解平衡态' if self._mode == 'eq' else '运行多体演化')
        if error == 'cancelled':
            return
        if error is not None:
            QMessageBox.warning(self, '多体运行失败', str(error))
            return
        res = None
        aid = (result or {}).get('artifact_id')
        try:
            from src.v2_bind import load_artifact
            loaded = (load_artifact(self.main.app_bridge.ops.root, aid)
                      if aid else None)
            if loaded is not None:
                res = loaded[1]
        except Exception as e:                          # noqa: BLE001
            self.main.log('多体结果载入失败：%s' % e)
        self._last_res = res
        summ = (result or {}).get('summary') or {}
        if summ.get('equilibrium'):
            # 平衡态：数组在 artifact（betas/E/C/F/energies/rho），标量在 summary
            # （E0/gap/backend）——两边合并后才能画。
            merged = dict(summ)
            merged.update(res or {})
            self._draw_eq_result(merged)
            self.main.log('多体平衡态完成：want=%s backend=%s E0=%s'
                          % (summ.get('want'), summ.get('backend'),
                             summ.get('E0')), 12000)
            return
        self._draw_result(res)
        self.main.log('多体运行完成：method=%s backend=%s %s'
                      % ((result or {}).get('method'), (result or {}).get('backend'),
                         {k: summ.get(k) for k in ('n_atoms', 'dim', 'ok')
                          if k in summ}), 12000)
