# -*- coding: utf-8 -*-
"""动态分析面板（能级.py 新增 tab）：波形列表 → 实际电磁场 → 态演化。

画布（主程序窗口内，上下）：
  上：各波形实际电磁场包络 E_i(t)（waveforms 波形包络，无通道无 TTL）
  下：态布居演化 P(t)（evolve.evolve）

参数面板（借鉴时序/波形编辑器 + 按 evolve 需要）：
  波形管理（WavePanel：频率/光强/偏振/包络/偏移）· 初态 · 时间范围/点数 ·
    演化方法 · 自发辐射 · 运行
"""
import json
import time

import numpy as np
from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QEvent
from PyQt5.QtGui import QColor, QIcon, QPixmap
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
                             QLabel, QPushButton, QSpinBox, QDoubleSpinBox,
                             QComboBox, QListWidget, QListWidgetItem,
                             QMessageBox, QDialog, QProgressBar, QRadioButton,
                             QButtonGroup, QTabWidget, QFrame, QScrollArea,
                             QLineEdit, QCheckBox, QApplication)
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.colors import LogNorm
from matplotlib import colormaps
from matplotlib.ticker import MaxNLocator

from ..core import core, pair_core
from . import _mecha_bind as MB
from ..core.atomic_levels import make_levels
from ..core.evolve_util import decay, auto_decays  # t74 工具归位
from . import ui_waves
from ..core import waveforms
from . import ui_theme
from .ui_waves import state_key, find_state_index

# readout_cb 末尾"复杂配置…"哨兵项 data（t30：全能级读出率加权）
_COMPLEX_READOUT = -1
_COMPLEX_LABEL = '复杂配置…'


class DynamicPanel(QWidget):
    """动态分析：上光场波形、下态演化曲线。
    能级组共用主程序（右侧面板创建），这里仅按组选 m 子能级 + 特有演化参数。"""

    def __init__(self, main, parent=None):
        super().__init__(parent)
        self.main = main              # 主程序 MainWindow（读 groups 共用能级组）
        self.cache = main.cache
        self.main_groups = []         # 同步自主程序的 (n,l,j) 组
        self.wave_data = []           # 最近一次波形光强曲线（画图用）
        self.decay_cfgs = []          # 衰变通道 [{'from': label, 'to': label|None, 'gamma_mhz': f}]
        # 两体纠缠初态 spec（['pair_sup', […]] / None）——登记项
        # dynamic.psi_pair 的真值；两体运行时取代逐原子 psi_A/psi_B。
        # GUI 尚无编辑器（后议），目前由 AI 面 set_pair_superposition 设置。
        self._psi_pair_spec = None
        # 已结束但线程可能仍在收尾的 worker（强引用，防 QThread 在运行时被销毁
        # → Qt fail-fast；见 _finish_run_ui 的 2026-09-12 事故说明）。
        self._retired_workers = []
        # 逐原子**叠加态** spec（['sup', […]] / None）——登记项
        # dynamic.psi_single/A/B 的**持久真值**（#3/#5 裁决 A）。
        # combo 表达不了叠加态，故 spec 存这里；下拉显示哨兵项
        # "叠加态（自定义 n 项）"（显式回落显示，不再静默显示 0 号态）。
        self._psi_sups = {'single': None, 'A': None, 'B': None}
        # 能级组增删由主程序事件驱动刷新（统一机制，不散布同步调用）
        main.groups_changed.connect(self.refresh_groups)
        self._is_two_body = False
        # t82：检视器**常驻**（两个模式都存在，组件与后台参数一一绑定）；
        # 模式只决定它显不显示 + 运行采不采 diag。惰性建，但一旦建就不销毁。
        self._inspector = None
        self._god_mode = False
        self._last_god_kwargs = None   # t75：最近一次覆写（回填检视器旋钮）
        self._build_canvases()
        self.dynamic_widget = self._build_controls()
        self._build_module()
        self.refresh_groups()

    def _build_module(self):
        """模块层接口：动态页仅作为一个自包含模块，不再额外套一层 page 抽象。"""
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(2)
        root.addWidget(self.canvas_top, 5)
        root.addWidget(self.canvas_bot, 5)

    def show(self):
        self.refresh_groups()

    def on_light(self):
        self.refresh_groups()

    # ---------- 画布：上光场（A/B 双子图）、下演化 ----------
    def _build_canvases(self):
        # 顶部波形图：上下两个子图（单原子只用 A 格并拉满；双原子 A 上 B 下）
        self.fig_top = Figure(figsize=(7, 3.2), dpi=100)
        self.ax_topA = self.fig_top.add_subplot(211)
        self.ax_topB = self.fig_top.add_subplot(212)
        self.canvas_top = FigureCanvas(self.fig_top)
        self.fig_bot = Figure(figsize=(7, 3.6), dpi=100)
        self.ax_bot = self.fig_bot.add_subplot(111)
        self.canvas_bot = FigureCanvas(self.fig_bot)
        # 统一边距：left/right 与布居图一致 → 上下时间轴对齐；顶部不设标题省空间
        for fig in (self.fig_top, self.fig_bot):
            fig.subplots_adjust(left=0.07, right=0.95, top=0.97, bottom=0.08)
        for ax in (self.ax_topA, self.ax_topB, self.ax_bot):
            ax.set_xlabel('t (ns)'); ax.grid(alpha=0.3)
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
        self.ax_topB.set_visible(False)      # 单原子默认只显示 A 格

    # ---------- 布局 ----------
    def _build_controls(self):
        """动态分析特有参数面板（独立控件，由主程序放入右侧共用面板）。"""
        panel = QWidget()
        p = QVBoxLayout(panel)
        p.setSpacing(6)

        # ---- 单原子/双原子切换 ----
        # ---- V_dd 参数（仅双原子模式可见；t25：与"双原子"单选同一行紧凑）----
        self._vdd_box = QWidget()
        vdd_lay = QHBoxLayout(self._vdd_box)
        vdd_lay.setContentsMargins(0, 0, 0, 0)
        vdd_lay.setSpacing(2)
        vdd_lay.addWidget(QLabel('R:'))
        _reg = self.main.binder          # _mecha_bind.Binder（取代旧 param_reg）
        self.vdd_R_sb = MB.spinbox(_reg, 'dynamic.pair.R_um', 5.0, 0.1, 1000.0,
                                   decimals=1, width=128,
                                   spec=MB.ParamSpec('float', unit='μm',
                                                     desc='原子间距'))
        self.vdd_R_sb.setToolTip('原子间距 R（μm）')
        vdd_lay.addWidget(self.vdd_R_sb)
        vdd_lay.addWidget(QLabel('θ:'))
        self.vdd_theta_sb = MB.spinbox(_reg, 'dynamic.pair.theta_deg', 0.0,
                                       0.0, 180.0, decimals=1, width=128,
                                       spec=MB.ParamSpec('float', unit='°',
                                                         desc='V_dd 极角'))
        self.vdd_theta_sb.setToolTip('极角 θ（°）')
        vdd_lay.addWidget(self.vdd_theta_sb)
        vdd_lay.addWidget(QLabel('φ:'))
        self.vdd_phi_sb = MB.spinbox(_reg, 'dynamic.pair.phi_deg', 0.0,
                                     0.0, 360.0, decimals=1, width=128,
                                     spec=MB.ParamSpec('float', unit='°',
                                                       desc='V_dd 方位角'))
        self.vdd_phi_sb.setToolTip('方位角 φ（°）')
        vdd_lay.addWidget(self.vdd_phi_sb)
        self._vdd_box.setVisible(False)

        row_mode = QHBoxLayout()
        self._mode_group = QButtonGroup(self)
        self._radio_single = QRadioButton('单原子')
        self._radio_pair = QRadioButton('双原子')
        self._radio_single.setChecked(True)
        self._mode_group.addButton(self._radio_single, 0)
        self._mode_group.addButton(self._radio_pair, 1)
        self._mode_group.idClicked.connect(self._on_mode_changed)
        row_mode.addWidget(self._radio_single)
        row_mode.addWidget(self._radio_pair)
        row_mode.addWidget(self._vdd_box)      # t25：紧跟"双原子"后，同一行
        row_mode.addStretch(1)
        p.addLayout(row_mode)

        # ---- Sambe 高级参数行（t62：双原子 + sambe/auto 档显示——
        # per_freq_M/frozen_h_ns——全频段两体 Sambe 的精度/性能参数，t54 待接线）----
        self._sambe_adv_box = QWidget()
        sa_lay = QHBoxLayout(self._sambe_adv_box)
        sa_lay.setContentsMargins(0, 0, 0, 0)
        sa_lay.setSpacing(4)
        sa_lay.addWidget(QLabel('frozen_h(ns)'))
        self.frozen_h_sb = MB.spinbox(_reg, 'solve.FROZEN_H_NS', 0.0, 0.0,
                                      100.0, decimals=2, width=72,
                                      spec=MB.ParamSpec(
                                          'float', unit='ns',
                                          desc='Sambe 冻结段长（0=默认）'))
        self.frozen_h_sb.setToolTip(
            '全频段 Sambe 纯冻结段长（ns，包络变化尺度内）；0 = 默认路径')
        sa_lay.addWidget(self.frozen_h_sb)
        sa_lay.addWidget(QLabel('per_freq_M'))
        # per_freq_M：str 输入但 session 键 solve.PER_FREQ_M 是 int_or_list——留本地
        # QLineEdit，run 流程解析后写 session（B-1③-3），免 str→int_or_list 校验失败。
        self.perfreq_m_edit = QLineEdit()
        self.perfreq_m_edit.setFixedWidth(120)
        self.perfreq_m_edit.setPlaceholderText('[1,1] 或空=自动')
        self.perfreq_m_edit.setToolTip(
            '逐载频光子块阶数（列表 [M1,M2,…] 或 int 全体）；空 = 自动探针')
        sa_lay.addWidget(self.perfreq_m_edit)
        sa_lay.addStretch(1)
        self._sambe_adv_box.setVisible(False)
        # 修复（用户报告 QLayout::addChildLayout 警告）：sa_lay 已是
        # _sambe_adv_box 的布局，不能重复加进 p——应加的是控件本身。
        # 旧写法导致该行从未进入面板（双原子+sambe 下不可见）。
        p.addWidget(self._sambe_adv_box)

        # ---- 波形管理：单原子与双原子各一套独立面板（数据完全独立）----
        # 接力需求（单原子校准→双原子 A 沿用）由拖放实现：把波形组拖到
        # "单原子/双原子"选择区 = 单原子全部波形复制进原子 A 并切到双原子
        # （与 A→B 页签拖放同一 MIME 机制，见 eventFilter）。
        self.wave_panel = ui_waves.WavePanel()       # 单原子面板
        self.wave_panel.changed.connect(self._on_waves_changed)
        self.wave_panel_A = ui_waves.WavePanel()    # 双原子 A（独立实例）
        self.wave_panel_A.changed.connect(self._on_waves_changed)
        self.wave_panel_B = ui_waves.WavePanel()
        self.wave_panel_B.changed.connect(self._on_waves_changed)

        self._wave_single_host = QWidget()           # 单原子模式的槽位
        _wsh = QVBoxLayout(self._wave_single_host)
        _wsh.setContentsMargins(0, 0, 0, 0)
        _wsh.addWidget(self.wave_panel)

        self._wave_tab_A_host = QWidget()            # 双原子 tab-A 的槽位
        _wah = QVBoxLayout(self._wave_tab_A_host)
        _wah.setContentsMargins(0, 0, 0, 0)
        _wah.addWidget(self.wave_panel_A)

        self._wave_tab = QTabWidget()
        self._wave_tab.addTab(self._wave_tab_A_host, '原子 A')
        self._wave_tab.addTab(self.wave_panel_B, '原子 B')
        self._wave_tab.setVisible(False)              # 默认隐藏
        # 页签接受"复制波形"拖放（拖到原子B标签 → 复制一份到B，无需切页）
        self._tabbar = self._wave_tab.tabBar()
        self._tabbar.setAcceptDrops(True)
        self._tabbar.installEventFilter(self)

        p.addWidget(self._wave_single_host, 1)        # 单原子波形
        p.addWidget(self._wave_tab, 1)                # 双原子波形（默认隐藏）

        # ---- 能级：共用主程序"能级管理表"（日常数值 + m 选择），这里只选初态 ----
        # 单原子模式：单个 psi_combo
        self.psi_combo = QComboBox()
        # 双原子模式：两个 psi combo
        self._psi_box = QWidget()
        psi_lay = QHBoxLayout(self._psi_box)
        psi_lay.setContentsMargins(0, 0, 0, 0)
        psi_lay.setSpacing(4)
        psi_lay.addWidget(QLabel('A:'))
        self.psi_combo_A = QComboBox()
        psi_lay.addWidget(self.psi_combo_A, 1)
        psi_lay.addWidget(QLabel('B:'))
        self.psi_combo_B = QComboBox()
        psi_lay.addWidget(self.psi_combo_B, 1)
        self._psi_box.setVisible(False)

        row2 = QHBoxLayout()
        row2.addWidget(QLabel('初态'))
        row2.addWidget(self.psi_combo, 1)
        row2.addWidget(self._psi_box, 1)
        self._psi_box.setVisible(False)
        # 人在下拉里选了**具体能级** → 丢弃该原子的叠加态 spec（#3/#5 裁决 A：
        # 叠加态是持久真值，但人显式重选能级时必须让位，且要记账）。
        # 连接必须早于 pm_setup 的 touch_sig（register 在面板构造之后）——
        # 那时 read_fn 已看到 spec 被清掉，落账的才是"人选的键"。
        for _w, _cb in (('single', self.psi_combo),
                        ('A', self.psi_combo_A),
                        ('B', self.psi_combo_B)):
            _cb.currentIndexChanged.connect(
                lambda _i, w=_w: self._on_psi_pick(w))
        # t27：自发辐射从独立行移入初态行右侧（原"显示态数"位置）——
        # 单原子 psi_combo stretch=1 / 双原子 psi_box(A/B) 各 stretch=1，
        # 自发辐射档位自然宽度，整行协调不挤压。
        row2.addWidget(QLabel('自发辐射'))
        self.decay_mode_cb = QComboBox()
        self.decay_mode_cb.addItems(['自动（ARC 寿命）', '禁用', '自定义'])
        self.decay_mode_cb.setToolTip(
            '自动：按 ARC 权威能级寿命自动配置 Γ=1/τ'
            '（仅 τ≤100 µs 的低激发态，落点=丢失，运行后状态栏显示通道数）；'
            '自定义：手动配置衰变通道（从态/Γ/落点/纯退相）')
        self.decay_mode_cb.currentIndexChanged.connect(self._on_decay_mode)
        row2.addWidget(self.decay_mode_cb)
        p.addLayout(row2)

        # ---- 时间 / 方法 / 截断 一行 ----
        rowt = QHBoxLayout()
        rowt.addWidget(QLabel('t_max(ns)'))
        self.tmax_sb = MB.spinbox(_reg, 'dynamic.t_max', 1500.0, 10, 1e6,
                                  decimals=0,
                                  spec=MB.ParamSpec(
                                      'float', unit='ns', desc='演化总时长',
                                      aliases=['tmax', '时长']))
        rowt.addWidget(self.tmax_sb)
        rowt.addWidget(QLabel('点数'))
        self.npts_sb = MB.spinbox(_reg, 'dynamic.npts', 1501, 10, 50000,
                                  spec=MB.ParamSpec('int', desc='时间网格点数'))
        rowt.addWidget(self.npts_sb)
        rowt.addWidget(QLabel('方法'))
        # t25→t71：方法选项收敛——sambe：第一性原理 Sambe 光子块（实验室系无
        # RWA/无旋转系/无近远分流，引擎 sambe.evolve_sambe）；
        # hard：微波硬积分 DOP853 零近似 + 光场旋转系 + 耗散（numba 加速）；
        # mixed（默认，t71）：混合引擎——光场 Sambe 光子块 + 微波/V_dd 当场
        # （φ-帧，t62 起验证 ≡frozen 1e-5~1e-8 级；两体 5 倍降维 dim 1452）。
        # t71：删除 ip4（RWA RK4 快速粗算档）与 auto（route_auto 智能路由）——
        # 引擎层仍可用（evolve_rwa ip4 / route_auto 脚本调用），GUI 不再暴露；
        # 默认 mixed（t77 起亦支持纯微波：光子维退化 P=1）；
        # 仅任意波形（无单载频）不支持，UI 自动降级 hard。
        self.method_cb = MB.combobox(
            _reg, 'dynamic.method', ['sambe', 'hard', 'mixed'],
            spec=MB.ParamSpec('str', options=['sambe', 'hard', 'mixed'],
                              desc='演化引擎'))
        # 旧 pm.combobox current=2（默认 mixed）；MB.combobox 无此参 → 默认落 options[0]。
        # TODO(fine-tune)：app_entry 初始化 session 时 seed dynamic.method='mixed'。
        self.method_cb.currentTextChanged.connect(self._on_method_changed)
        rowt.addWidget(self.method_cb)
        # t82：上帝模式开关搬到菜单栏那一行（MainWindow 的「极简/上帝」分段钮）。
        # 这里只留一个模式徽标——检视器窗口此时已存在（无论哪个模式）。
        self.mode_badge = QLabel('')
        self.mode_badge.setToolTip('当前处于上帝模式：检视器已打开，'
                                   '运行会采集诊断数据')
        self.mode_badge.setStyleSheet(
            'color:%s; font-weight:600;' % ui_theme.COL_ACCENT)
        self.mode_badge.setVisible(False)
        rowt.addWidget(self.mode_badge)
        p.addLayout(rowt)

        # ---- 自发辐射（t27：档位选择移入初态行右侧；自定义折叠区保留原位）----
        # 自定义配置区（仅 自定义 档显示；自动/禁用 隐藏）
        self.decay_custom_box = QWidget()
        dcb = QVBoxLayout(self.decay_custom_box)
        dcb.setContentsMargins(0, 0, 0, 0)
        dcb.setSpacing(4)
        dcb.addWidget(QLabel('从态 → Γ(MHz) → 落点'))
        rowd1 = QHBoxLayout()
        self.decay_from_cb = QComboBox()
        self.decay_gamma_sb = QDoubleSpinBox()
        self.decay_gamma_sb.setRange(0.0, 1e6)
        self.decay_gamma_sb.setDecimals(3)
        self.decay_gamma_sb.setValue(4.35)
        self.decay_gamma_sb.setToolTip('衰变率 Γ（MHz）；落点选"丢失"时'
                                       '布居离开建模空间，画布叠加 loss 曲线')
        self.decay_to_cb = QComboBox()
        rowd1.addWidget(self.decay_from_cb, 1)
        rowd1.addWidget(self.decay_gamma_sb)
        rowd1.addWidget(self.decay_to_cb, 1)
        dcb.addLayout(rowd1)
        rowd2 = QHBoxLayout()
        self.decay_add_btn = QPushButton('添加通道')
        self.decay_add_btn.clicked.connect(self._add_decay)
        self.decay_del_btn = QPushButton('删除选中')
        self.decay_del_btn.clicked.connect(self._del_decay)
        rowd2.addWidget(self.decay_add_btn)
        rowd2.addWidget(self.decay_del_btn)
        dcb.addLayout(rowd2)
        self.decay_list = QListWidget()
        self.decay_list.setFixedHeight(56)
        self.decay_list.setSelectionMode(QListWidget.SingleSelection)
        dcb.addWidget(self.decay_list)
        p.addWidget(self.decay_custom_box)
        self.decay_custom_box.setVisible(False)     # 默认自动档

        # ---- 显示选项 + 荧光读出 / 显示态数（t25 + t27）----
        row_disp = QHBoxLayout()
        self.display_mode_cb = QComboBox()
        self.display_mode_cb.addItems(['态布居', '荧光读出'])
        self.display_mode_cb.setToolTip(
            '态布居：前 N 态布居曲线；荧光读出：指定读出能级上的概率'
            '（单原子 |⟨i|ψ⟩|²；双原子 A/B 各自约化概率）')
        self.display_mode_cb.currentIndexChanged.connect(self._on_display_mode)
        row_disp.addWidget(self.display_mode_cb)
        # t27：topn_sb（态布居专用，原初态行）移入显示选项行——与 readout_cb
        # 同位互斥（态布居→topn，荧光读出→readout）；readout_cb 去掉 stretch
        # 不再占满，运行按钮贴右侧。
        self.topn_sb = QSpinBox()
        self.topn_sb.setRange(1, 40)
        self.topn_sb.setValue(8)
        self.topn_sb.setToolTip('按最近一次结果的平均布居取前 N 个态绘制'
                                '（无需手动勾选，内部自动筛选）')
        # t30：参数更改触发画布重绘（有解暂存时直接转换，不重新运行）
        self.topn_sb.valueChanged.connect(self._redraw_last)
        row_disp.addWidget(self.topn_sb)
        self.readout_cb = QComboBox()
        self.readout_cb.setToolTip('荧光读出能级（基组全部子能级，F 区分）；'
                                   '末尾"复杂配置…"= 全能级读出率加权')
        self.readout_cb.setFixedWidth(300)
        # t30：currentIndexChanged（含程序化 refill）只重绘；activated（用户
        # 主动选择）才弹复杂配置窗——避免基组刷新/存档恢复重选哨兵项时模态
        # 弹窗阻塞（faulthandler 定位的挂起根因）
        self.readout_cb.currentIndexChanged.connect(self._redraw_last)
        self.readout_cb.activated.connect(self._on_readout_activated)
        row_disp.addWidget(self.readout_cb)
        self.readout_cb.setVisible(False)        # 初始态布居 → topn 显示
        row_disp.addStretch(1)
        self.run_btn = QPushButton('运行演化')
        # 2026-09-14 用户裁决：运行按钮**同时是取消按钮**（运行中文本变
        # "取消计算"），取代进度弹窗；AI 发起的运行走同一个 `_run()`，
        # 所以人机看到/能控制的完全是同一套组件（同一 `_run_active` 驱动）。
        # 常驻进度控件：窗口底栏由 app_entry 接管（挂到 QStatusBar 并覆盖这两个
        # 属性）；**独立构造**（自检/无主窗口）时用本地兜底，保证 _poll_progress
        # 永不遇到 None（见 _run 的崩溃说明）。
        if getattr(self, '_pbar', None) is None:
            self._pbar = QProgressBar(self)
            self._pbar.setRange(0, 1000)
        if getattr(self, '_stage_label', None) is None:
            self._stage_label = QLabel('空闲', self)
        self.run_btn.clicked.connect(self._on_run_clicked)
        row_disp.addWidget(self.run_btn)
        p.addLayout(row_disp)
        # 直接作为主程序右侧面板的普通区块（无独立滚动区，融入同一层级）
        return panel

    # ---------- 单原子/双原子切换 ----------
    def _on_mode_changed(self, mode_id):
        """0=单原子, 1=双原子。纯显隐切换（两套波形面板独立存在，不搬家）。"""
        self._is_two_body = (mode_id == 1)
        # V_dd 参数
        self._vdd_box.setVisible(self._is_two_body)
        # Sambe 高级行（双原子 + sambe/auto）
        self._update_sambe_adv_visibility()
        # 波形面板：单/双各一套，独立数据，只切容器显隐
        self._wave_single_host.setVisible(not self._is_two_body)
        self._wave_tab.setVisible(self._is_two_body)
        # 初态面板
        self.psi_combo.setVisible(not self._is_two_body)
        self._psi_box.setVisible(self._is_two_body)
        # 自发辐射：双原子模式现已支持（寿命/落点=丢失通道），档位放开
        self.decay_mode_cb.setEnabled(True)
        # 刷新初态下拉
        self._on_m_changed()

    def eventFilter(self, obj, ev):
        """波形拖放接力（与 ui_waves 拖拽源同一 MIME）：
        - 拖到 A/B 页签 → 复制该波形到对应原子（跨原子复制，原有）；
        - 拖到「双原子」单选钮 → 单原子面板全部波形深拷贝进原子 A
          并切到双原子（校准接力）。"""
        # ---- 「双原子」单选钮：整列表接力（仅接受从单原子面板拖来的）----
        if obj is getattr(self, '_radio_pair', None):
            t = ev.type()
            if t in (QEvent.DragEnter, QEvent.DragMove):
                if ev.mimeData().hasFormat(ui_waves._WAVE_MIME):
                    ev.setDropAction(Qt.CopyAction)
                    ev.accept()
                    return True
                return False
            if t == QEvent.Drop:
                try:
                    payload = json.loads(bytes(
                        ev.mimeData().data(ui_waves._WAVE_MIME)
                    ).decode('utf-8'))
                except Exception:
                    return False
                if payload.get('_src') != id(self.wave_panel):
                    return False
                import copy as _copy
                self.wave_panel_A.waves = _copy.deepcopy(
                    self.wave_panel.waves)
                self.wave_panel_A._rebuild()
                self.wave_panel_A.changed.emit()
                self._radio_pair.setChecked(True)
                self._on_mode_changed(1)
                self.main.log('[动态] 单原子波形已复制到原子 A，切到双原子')
                ev.acceptProposedAction()
                return True
        # ---- A/B 页签：单波形跨原子复制 ----
        if obj is getattr(self, '_tabbar', None):
            t = ev.type()
            if t in (QEvent.DragEnter, QEvent.DragMove):
                if ev.mimeData().hasFormat(ui_waves._WAVE_MIME):
                    ev.setDropAction(Qt.CopyAction)
                    ev.accept()
                    return True
                return False
            if t == QEvent.Drop:
                payload = bytes(ev.mimeData().data(ui_waves._WAVE_MIME))
                idx = self._tabbar.tabAt(ev.pos())
                if idx >= 0:
                    panel = (self.wave_panel_A if idx == 0
                             else self.wave_panel_B)
                    try:
                        if json.loads(payload.decode('utf-8')).get('_src') \
                                != id(panel):
                            panel.copy_wave_from(payload)
                            ev.acceptProposedAction()
                            return True
                    except Exception:
                        pass
                return False
        return super().eventFilter(obj, ev)

    # ---------- 能级：从共用能级组生成（切页/运行时刷新，数据共享不散布同步） ----------
    def refresh_groups(self):
        """能级组与 m 选择共用主程序"能级管理表"（单一数据源），这里只刷新初态/驱动。"""
        self.main_groups = list(self.main.groups)
        self._on_m_changed()

    def _on_m_changed(self, *_):
        """基组变化 → 刷新初态下拉（保留当前选择）+ 波形辅助 g/e 下拉。"""
        states = self._current_states()
        self._states = list(states)      # 供 _psi_key_for 把初态索引映射为身份键
        pending = getattr(self, '_pending_psi_keys', None)

        def _fill(combo, which, pending_key, fallback_text):
            # 重填期屏蔽信号：psi 参数接了 currentIndexChanged→note_change
            # （人侧流水），clear/addItem 的中间态会发假变更
            combo.blockSignals(True)
            combo.clear()
            for s in states:
                combo.addItem(s['label'])
            # 叠加态（持久 spec）优先：combo 表达不了叠加态，就加一个**显式**
            # 哨兵项（#3/#5 裁决 A 的硬要求②）——绝不静默显示成某个具体能级。
            sup = self._psi_sup(which)
            if sup is None and isinstance(pending_key, (list, tuple)) \
                    and len(pending_key) == 2 and pending_key[0] == 'sup':
                sup = pending_key
            if sup is not None:
                combo.addItem('叠加态（自定义 %d 项）' % len(sup[1]))
                combo.setCurrentIndex(combo.count() - 1)
                combo.blockSignals(False)
                return
            if pending_key:
                i = find_state_index(states, pending_key)
                if i >= 0:
                    combo.setCurrentIndex(i)
                combo.blockSignals(False)
                return
            if fallback_text:
                idx = combo.findText(fallback_text)
                if idx >= 0:
                    combo.setCurrentIndex(idx)
            elif combo.count():
                combo.setCurrentIndex(0)
            combo.blockSignals(False)

        # 单原子初态
        cur = self.psi_combo.currentText()
        _fill(self.psi_combo, 'single',
              pending.get('single') if pending else None, cur)
        # 双原子初态（A/B 各自独立选择）
        cur_t = [c.currentText() for c in (self.psi_combo_A, self.psi_combo_B)]
        for k, combo in enumerate((self.psi_combo_A, self.psi_combo_B)):
            _fill(combo, 'A' if k == 0 else 'B',
                  pending.get('A' if k == 0 else 'B') if pending else None,
                  cur_t[k])
        # 荧光读出态下拉（基组全部子能级；label 含 F 区分——低激发超精细
        # 态如 '6P3/2 F=3.0, m_F=+3.0'，里德堡如 '60S1/2 m_J=+0.5'；
        # t30：末尾"复杂配置…"哨兵项 + 权重随基组重置/恢复）
        cur_r = self.readout_cb.currentText()
        self.readout_cb.clear()
        for i, s in enumerate(states):
            self.readout_cb.addItem(s['label'], i)
        self.readout_cb.addItem('复杂配置…', _COMPLEX_READOUT)
        # t30：基组重建 → 读出率权重重置全 0（读出态=显示参数，不入册不存档）
        self._readout_weights = np.zeros(len(states), dtype=np.float64)
        # t30 微调：复杂配置只能用户主动选中——cur_r 为哨兵项时忽略，
        # 回退第一个真实能级（初始/重建都不得默认落在复杂配置上）
        want = cur_r if cur_r != _COMPLEX_LABEL else ''
        if want:
            idx = self.readout_cb.findText(want)
            self.readout_cb.setCurrentIndex(idx if idx >= 0 else 0)
        elif self.readout_cb.count():
            self.readout_cb.setCurrentIndex(0)
        self._pending_psi_keys = None
        self._refresh_decay_combos(states)
        # 基组变化 → 波形面板辅助 g/e 下拉数据源同步（三面板独立实例）
        for wp in (self.wave_panel, self.wave_panel_A, self.wave_panel_B):
            if wp is not None:
                wp.set_states(states, self.cache)

    # ---------- 自发辐射（Lindblad）通道配置 ----------
    def _on_decay_mode(self, *_):
        """三档：自动（ARC 寿命）/ 禁用 / 自定义——只显示需要的控件。"""
        mode = self.decay_mode_cb.currentIndex()      # 0 自动 / 1 禁用 / 2 自定义
        self.decay_custom_box.setVisible(mode == 2)
        #self.decay_auto_label.setVisible(mode == 0)

    def _decay_mode(self):
        """当前档位：'auto' / 'off' / 'custom'。"""
        return ('auto', 'off', 'custom')[self.decay_mode_cb.currentIndex()]

    def _refresh_decay_combos(self, states):
        """刷新从态/落点下拉（保留当前选择）。states 为空时清空。"""
        if not hasattr(self, 'decay_from_cb'):
            return                                    # 控件未建（信号早到）防御
        cur_from = self.decay_from_cb.currentText()
        cur_to = self.decay_to_cb.currentText()
        self.decay_from_cb.clear()
        self.decay_to_cb.clear()
        for s in states:
            self.decay_from_cb.addItem(s['label'])
            self.decay_to_cb.addItem(s['label'])
        self.decay_to_cb.addItem('（丢失）')
        if states:
            if cur_from:
                i = self.decay_from_cb.findText(cur_from)
                if i >= 0:
                    self.decay_from_cb.setCurrentIndex(i)
            if cur_to:
                i = self.decay_to_cb.findText(cur_to)
                if i >= 0:
                    self.decay_to_cb.setCurrentIndex(i)

    def _update_decay_list(self):
        self.decay_list.clear()
        for c in self.decay_cfgs:
            to = c['to'] if c['to'] else '丢失'
            self.decay_list.addItem(
                f"{c['from']} → {to}   Γ={c['gamma_mhz']:g} MHz")

    def _add_decay(self):
        """从当前下拉配置添加衰变通道（落点=末尾'（丢失）'项 → None）。"""
        n_states = self.decay_from_cb.count()
        if n_states == 0:
            return
        fi = self.decay_from_cb.currentIndex()
        ti = self.decay_to_cb.currentIndex()
        gamma = self.decay_gamma_sb.value()
        if gamma <= 0:
            QMessageBox.information(self, '提示', '衰变率 Γ 需 > 0。')
            return
        if ti >= n_states:
            to_label = None
        else:
            to_label = self.decay_to_cb.itemText(ti)
            if to_label == self.decay_from_cb.itemText(fi):
                QMessageBox.information(
                    self, '提示', '衰变态与落点相同 = 纯退相通道（不转移布居，'
                    '只衰减相干）。如需丢布居请选"（丢失）"。')
        self.decay_cfgs.append(
            {'from': self.decay_from_cb.itemText(fi), 'to': to_label,
             'gamma_mhz': float(gamma)})
        self._update_decay_list()
        self.decay_list.setCurrentRow(len(self.decay_cfgs) - 1)   # 选中新通道

    def _del_decay(self):
        row = self.decay_list.currentRow()
        if row >= 0 and row < len(self.decay_cfgs):
            del self.decay_cfgs[row]
            self._update_decay_list()

    def _has_decays(self):
        """是否已配置至少一个衰变通道。"""
        return bool(self.decay_cfgs)

    def _resolve_decay_cfgs(self, states):
        """decay_cfgs（按 label）→ evolve.decay 列表（解析为当前态索引）。
        找不到的 label（能级组已变）跳过并提示。"""
        labels = [s['label'] for s in states]
        out = []
        for c in self.decay_cfgs:
            fi = labels.index(c['from']) if c['from'] in labels else None
            ti = labels.index(c['to']) if (c['to'] and c['to'] in labels) \
                else None
            if fi is None:
                self.main.log(f'[动态] 衰变态 {c["from"]!r} 已不在能级列表中，'
                              '该通道被跳过')
                continue
            out.append(decay(fi, c['gamma_mhz'], ti))
        return out

    # ---------- 布居图：内部筛选（前 N 个平均布居最大的态） ----------
    def _top_pop_rows(self, states, res):
        """按最近结果的平均布居取前 N 个态（N=topn_sb；无结果时取前 N 个）。"""
        n = max(1, min(self.topn_sb.value(), len(states)))
        if res is not None:
            avg = np.asarray(res['pops']).mean(axis=0)
            order = np.argsort(avg)[::-1]
        else:
            order = np.arange(len(states))[::-1]
        return list(order[:n])

    def _current_states(self, pump=None):
        """动态基组 = 全部子能级（与共用能级表一致，无需选择）：
        低激发态展开超精细 F/mF（用场面板磁场 B），里德堡态保持 mJ。
        能量 = 组中心 + Zeeman(B) + DC-Stark(E)，与静态页/两体页同基准。
        label 含完整量子数，布居图/g/e 下拉共用。
        pump：可选 callable(frac)——每个能级组完成后上报（组内 DC-Stark
        的 dme/Numerov 首算可能很慢，给 GUI 预处理泵逐组反馈，t72）。

        **2026-09-15（P4 同源收敛）**：实现搬到 `src.core.runspec.build_states`
        （无 Qt）。并行扫描的 runner 调**同一个函数**，所以"扫描"在构造上就是
        "把人点运行的那份配置跑 N 次"——不会再有第二套基组装配各自漂移
        （SKILL 坑 18：离线脚本与登记册的 6P 展开规则曾不同）。
        """
        from ..core import runspec as _rs
        fld = getattr(self.main, 'field', None)
        B = (fld.b_gauss() if fld is not None
             else getattr(self.main, 'field_B', 0.0))
        E_vpcm = (fld.e_vpcm() if fld is not None else 0.0)
        return _rs.build_states(self.cache, list(self.main_groups), B, E_vpcm,
                                pump=pump)

    # ---------- 波形数据 ----------
    def _has_waves(self):
        """是否已配置至少一个波形（波形列表非空即算有效驱动源）。"""
        if self._is_two_body:
            return (len(self.wave_panel_A.waves) > 0
                    or len(self.wave_panel_B.waves) > 0)
        return len(self.wave_panel.waves) > 0

    def _active_wave_panel(self):
        """当前活跃的波形面板（双原子模式取选中 tab）。"""
        if self._is_two_body:
            idx = self._wave_tab.currentIndex()
            return self.wave_panel_A if idx == 0 else self.wave_panel_B
        return self.wave_panel

    def _on_waves_changed(self):
        """波形增删/参数变化 → 立即刷新顶部光强预览（不等待运行）。

        双原子模式：A/B 两面板**合并**并加 'A:'/'B:' 前缀（与 _run_two_body
        一致），供顶部双子图分组；单原子直接单面板。
        """
        t = np.linspace(0, self.tmax_sb.value(), self.npts_sb.value())
        if self._is_two_body:
            ea = waveforms.waveforms_to_env(self.wave_panel_A.waves, t)
            eb = waveforms.waveforms_to_env(self.wave_panel_B.waves, t)
            self.wave_data = (
                [(f'A:{nm}', th, env, f, I)
                 for nm, th, env, f, I in ea]
                + [(f'B:{nm}', th, env, f, I)
                   for nm, th, env, f, I in eb])
        else:
            wp = self._active_wave_panel()
            self.wave_data = waveforms.waveforms_to_env(wp.waves, t)
        self._draw_top(self.wave_data)

    # ---------- 状态管理：单一真值源 = mecha session ----------
    # 本面板不再持有 serialize_state/restore_state 平行清单；控件经 _mecha_bind
    # 直接读写 session。存档 = app_bridge.save（session 快照），恢复 = app_bridge.load。
    def _psi_key_for(self, combo):
        """组合框当前选中态的身份键（JSON 友好 list）；无可用时返回 None。"""
        i = combo.currentIndex()
        st = getattr(self, '_states', None)
        if st and 0 <= i < len(st):
            return list(state_key(st[i]))
        return None

    def _psi_sup(self, which):
        """该原子的**叠加态 spec**（['sup', […]]）——非法/未设 → None。

        #3/#5 裁决 A 的持久真值：叠加态存这里（不是 combo），AI/存档/undo
        全经登记项 `dynamic.psi_*`，其 read_fn/write_fn 落到本属性。
        """
        v = (getattr(self, '_psi_sups', None) or {}).get(which)
        if (isinstance(v, (list, tuple)) and len(v) == 2 and v[0] == 'sup'
                and isinstance(v[1], (list, tuple)) and v[1]):
            return list(v)
        return None

    def _psi0_for(self, which):
        """运行期初态（single/A/B）：**spec 优先**，无 spec 才用 combo 索引。

        硬要求（2026-09-12 用户裁决，P1 不许静默改初态）：
          · 设了 spec → 原样交给引擎（引擎侧 `_as_vector` 按 (n,l,j,m) 校验，
            缺项**报错**）；**绝不**回落到 0 号态；
          · 没设 spec 但下拉没选中基组内能级（索引越界/空基组）→ 报错，
            而不是把 -1 当"最后一个态"喂进引擎（`_onehot(N,-1)` 的静默错）。
        """
        combo = {'single': self.psi_combo, 'A': self.psi_combo_A,
                 'B': self.psi_combo_B}[which]
        spec = self._psi_sup(which)
        if spec is not None:
            return list(spec)
        i = combo.currentIndex()
        st = getattr(self, '_states', None) or []
        if not st or not (0 <= i < len(st)):
            raise ValueError(
                '初态未选定（%s）：下拉里没有可用的基组内能级'
                '（先添加能级组；叠加态用 set_superposition/%s）'
                % (which, 'set_pair_superposition' if which != 'single'
                   else 'set_superposition'))
        return i

    def _on_psi_pick(self, which):
        """人在下拉里选了**具体能级** → 丢弃该原子的叠加态 spec（显式让位）。"""
        combo = {'single': self.psi_combo, 'A': self.psi_combo_A,
                 'B': self.psi_combo_B}[which]
        st = getattr(self, '_states', None) or []
        if not (0 <= combo.currentIndex() < len(st)):
            return                      # 末尾哨兵项"叠加态（自定义…）"：不动 spec
        sups = dict(getattr(self, '_psi_sups', None) or {})
        if sups.get(which):
            sups[which] = None
            self._psi_sups = sups

    def _pair_psi0_spec(self):
        """两体纠缠初态 spec（dynamic.psi_pair）—— 非法/未设 → None。

        真值优先取注册表（存档/undo/AI 改都经它）；注册表不可用时回落到
        面板属性（自检/early 构造期）。只认 ['pair_sup', […]] 形态——
        别的值一律当"未设"，绝不把垃圾喂进引擎（P1：不许静默改物理）。
        """
        v = None
        br = getattr(self.main, 'app_bridge', None)
        if br is not None:
            try:
                v = br.get('dynamic.psi_pair')
            except Exception:                      # noqa: BLE001
                v = None
        if v is None:
            v = getattr(self, '_psi_pair_spec', None)
        if (isinstance(v, (list, tuple)) and len(v) == 2 and v[0] == 'pair_sup'
                and isinstance(v[1], (list, tuple)) and v[1]):
            return list(v)
        return None

    # ---------- 运行演化（后台线程 + 进度条 + 可取消） ----------
    def _run(self):
        if getattr(self, '_worker', None) is not None \
                and self._worker.isRunning():
            QMessageBox.information(self, '提示', '已有演化在运行，请先等待或取消。')
            return
        self._sweep_retired_workers()   # 上一轮线程若已退出，丢掉它的引用
        self.refresh_groups()          # 运行前取最新共用能级组（无需外部散布同步）
        if not self.main_groups:
            QMessageBox.information(self, '提示', '请先在主程序添加能级组。')
            return
        # t72 进度协议（engines 经 coupling.stage_progress 写入，GUI 映射）：
        #   prog[2]=阶段（0 准备 1 积分 2 收尾）、prog[0]=阶段内进度 0..1、
        #   prog[1]=取消标志。定时器在主线程预处理（基组/场/V_dd）之前启动，
        #   配合 _prep 的 processEvents 泵——预处理期间对话框同样显示阶段、
        #   进度与耗时，且可取消（此前 V_dd 等长预处理完全无反馈）。
        self.progress_arr = np.zeros(3)
        self._run_active = True
        self._prep_label = '准备'
        self._run_t0 = time.perf_counter()
        # 常驻进度控件（底栏）**不置 None**——轮询定时器（100ms）先于一切触发，
        # _poll_progress 直接写它们；缺失会变成 Qt 槽内 AttributeError →
        # PyQt qFatal → 原生崩溃（无 Python 回溯）。这里只清零 + 复位按钮。
        self._dlg = None
        self._reset_progress_ui()
        self.run_btn.setText('取消计算')      # 运行中：同一按钮即"取消计算"
        self.run_btn.setEnabled(True)
        self._dlg_timer = QTimer(self)
        self._dlg_timer.setSingleShot(True)
        self._dlg_timer.timeout.connect(self._show_progress)
        self._dlg_timer.start(400)
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._poll_progress)
        self._timer.start(100)
        try:
            states = self._current_states(
                pump=lambda f: self._prep('构建基组/子能级', 0.6 * f))
            self._prep('基组就绪', 0.65)
            # 无波形时须有实际耗散/自动通道，否则就是空演化 → 提示
            has_effect = self._has_waves()
            if not has_effect:
                mode = self._decay_mode()
                if mode == 'custom':
                    has_effect = bool(self.decay_cfgs)
                elif mode == 'auto':
                    # 门槛只做廉价判断（基组是否含低激发态），不触发 ARC 寿命计算：
                    # 此处的 auto_decays 曾致 Qt 原生堆损坏（0xC0000005，tab 切换后
                    # 确定性复现；与百度网盘无关，网盘关闭后依然崩）。
                    # auto_decays 仅在真正运行时计算一次。
                    has_effect = any(s['n'] <= 15 for s in states)
            if not has_effect:
                QMessageBox.information(
                    self, '提示',
                    '请先在下方波形区添加光波形（频率/光强/包络），'
                    '或开启/配置自发辐射（自动档需基组含 6P 等低激发态），'
                    '再运行演化。')
                return
            # 同步配置 → session（人机同路径 I5）；ops.prepare 在**主线程** build_run
            # （基组/场/V_dd/auto_decays 都碰 cache，ARC SQLite 线程亲和 → 必须主线程，
            # 见 B-1①）。单/两体同一入口（dynamic.mode 决定），不再手动分派。
            self._prep('同步配置 → 会话', 0.7)
            self._sync_run_config()
            self._run_states = states
            ops = self.main.app_bridge.ops
            if getattr(self.main.app_bridge, 'client', None) is not None:
                # Hub 模式（GUI 作 Hub 客户端）：run 在 Hub 侧（run_compute auto async）；
                # 本地不 prepare/execute（无本地 cache/RunSpec），也不 load 本地 artifact。
                from .evolve_worker import HubRunWorker
                self._hub_run = True
                self._prep('Hub 模式：演化在 Hub 侧', 1.0)
                self._worker = HubRunWorker(self.main.app_bridge, 'GUI 运行演化')
                self._worker.done.connect(self._on_evolve_done)
                self._worker.log_msg.connect(self.main.log)
                self._worker.start()
                return
            self._prep('构建 RunSpec（基组/场/V_dd/寿命）', 0.85)
            prepared = ops.prepare(actor='gui', reason='GUI 运行演化',
                                   diag=bool(self._god_mode))
            prepared = self._preflight_cost(prepared, ops)   # 成本报告（可能重建/取消）
            if prepared is None:
                self._prep('已取消（成本预检）', 0.0)
                return
            self._prep('开始演化', 1.0)
            from .evolve_worker import RunWorker
            self._worker = RunWorker(ops, prepared, self.progress_arr,
                                     actor='gui', reason='GUI 运行演化')
            self._worker.done.connect(self._on_evolve_done)
            self._worker.log_msg.connect(self.main.log)
            self._worker.start()
        except RuntimeError as e:
            # 主线程预处理被取消（_prep 泵检测 prog[1]≠0）；worker 内的
            # 取消走 done('cancelled') 信号，不经此处。
            if '取消' in str(e):
                return
            raise
        finally:
            # worker 已启动（_worker 非 None）时定时器/对话框由 done 回调收尾；
            # 其余返回路径（提示/取消/异常）在此统一清理。
            if getattr(self, '_worker', None) is None:
                self._finish_run_ui()

    def _sync_run_config(self):
        """运行前把当前 GUI 配置同步进 session（供 ops.prepare 读；人机同路径 I5）。

        MB 绑定的标准件（t_max/npts/method/pair.*/frozen_h）已实时写 session；这里补
        **非标准件**（能级组/波形/初态/衰变/per_freq_M）——旧时靠 register_panel_params
        的 external 登记，Re0 改为运行前一次性 write_many（单/两体同一入口）。
        """
        from ..core import waveforms as _wf
        two = bool(self._is_two_body)
        patch = {
            'groups.list': [dict(g) for g in self.main_groups],
            'dynamic.mode': 1 if two else 0,
            'dynamic.method': self.method_cb.currentText(),
            'dynamic.t_max': float(self.tmax_sb.value()),
            'dynamic.npts': int(self.npts_sb.value()),
            'dynamic.decay_index': int(self.decay_mode_cb.currentIndex()),
        }
        if patch['dynamic.decay_index'] == 2:            # 自定义衰变通道
            patch['dynamic.decays'] = [dict(c) for c in self.decay_cfgs]
        if two:
            patch['dynamic.waves_A'] = _wf.serialize_waves(list(self.wave_panel_A.waves))
            patch['dynamic.waves_B'] = _wf.serialize_waves(list(self.wave_panel_B.waves))
            patch['dynamic.psi_A'] = self._psi_sup('A') or self._psi_key_for(self.psi_combo_A)
            patch['dynamic.psi_B'] = self._psi_sup('B') or self._psi_key_for(self.psi_combo_B)
            patch['dynamic.pair.R_um'] = float(self.vdd_R_sb.value())
            patch['dynamic.pair.theta_deg'] = float(self.vdd_theta_sb.value())
            patch['dynamic.pair.phi_deg'] = float(self.vdd_phi_sb.value())
            pp = self._pair_psi0_spec()
            if pp is not None:
                patch['dynamic.psi_pair'] = pp
            if self.method_cb.currentText() == 'sambe':  # 两体 sambe 高级参数
                fh, pf = self._sambe_adv_params()
                if fh is not None:
                    patch['solve.FROZEN_H_NS'] = fh
                if pf is not None:
                    patch['solve.PER_FREQ_M'] = pf
        else:
            patch['dynamic.waves_single'] = _wf.serialize_waves(list(self.wave_panel.waves))
            patch['dynamic.psi_single'] = (self._psi_sup('single')
                                           or self._psi_key_for(self.psi_combo))
        patch = {k: v for k, v in patch.items() if v is not None}   # 不写垃圾进 session
        self.main.app_bridge.write_many(patch, reason='GUI 运行前同步配置', actor='gui')

    def _prep(self, label, frac):
        """主线程预处理进度泵（t72）：写阶段 0 进度 + 泵事件 + 取消检查。

        processEvents 让 400ms 对话框定时器/100ms 轮询定时器/重绘在长
        预处理（dme 查询、V_dd 构建）期间照常运行——否则主线程阻塞，
        对话框出不来、取消点不动。"""
        pa = self.progress_arr
        self._prep_label = label
        pa[2] = 0.0
        pa[0] = min(max(float(frac), 0.0), 1.0)
        QApplication.processEvents()
        if pa[1] != 0.0:
            raise RuntimeError('用户取消')

    def _finish_run_ui(self):
        """运行结束（完成/失败/取消/预处理中止）统一的 UI 收尾。

        ⚠ **2026-09-12 事故修（QThread 生命周期）**：这里**不得**在 QThread
        还在跑时丢掉它的最后一根引用。
        `done` 是在 worker 线程的 `run()` **内部** emit 的（queued 投递到主线程），
        主线程完全可能在 `run()` 尚未返回（正在退出 `_P.override` 上下文、释放
        大结果数组）时就处理完这个槽 → 原来的 `self._worker = None` 把唯一引用
        丢掉 → Qt 打印
            `QThread: Destroyed while thread is still running`
        并 **fail-fast**（实测 0xc0000409 / BEX64：两体 job20 dim=7056 跑完时
        进程静默消失、结果全丢）。
        现在：仍在跑 → 交给它的 `finished` 信号释放（那时线程确已退出）；已退出
        → 直接丢。收尾时刻的强引用保存在 `_retired_workers`。
        """
        if getattr(self, '_dlg_timer', None) is not None:
            self._dlg_timer.stop()
        if getattr(self, '_timer', None) is not None:
            self._timer.stop()
        if getattr(self, '_dlg', None) is not None:
            self._dlg.close()
        self._dlg = None
        # ⚠ 2026-09-14 修（用户实测：算完进度条停在 90% 多不动）：
        #   这里原来还有 `self._pbar = None` / `self._stage_label = None`——
        #   那是"进度框随运行销毁"时代的产物。现在它们是**常驻**控件
        #   （主窗口底栏），抹掉引用后没人再写、于是永远停在最后一次轮询值；
        #   而下游 `_reset_progress_ui` 判空也成了空操作。改为**显式收尾**。
        self._reset_progress_ui(idle=True)
        self._run_active = False
        w = getattr(self, '_worker', None)
        if w is not None:
            try:
                if w.isRunning():                 # 还在收尾 → 先留着引用
                    if w not in self._retired_workers:
                        self._retired_workers.append(w)
                    try:
                        w.finished.connect(self._sweep_retired_workers)
                    except Exception:             # noqa: BLE001
                        pass
            except RuntimeError:                  # C++ 对象已删（异常路径）
                pass
        self._worker = None
        self.run_btn.setEnabled(True)
        self.run_btn.setText('运行演化')          # 收尾：按钮复位为"运行"

    def _sweep_retired_workers(self):
        """丢掉**已经退出**的 worker 引用（防 `_retired_workers` 无界增长）。

        只丢 `isRunning()` 为 False 的——引用一丢 QThread 就会被销毁，
        所以"确实不跑了"是硬条件（见 `_finish_run_ui` 的事故说明）。
        """
        cur = list(getattr(self, '_retired_workers', None) or [])
        keep = []
        for w in cur:
            try:
                if w.isRunning():
                    keep.append(w)
            except RuntimeError:                  # C++ 对象已删
                pass
        self._retired_workers = keep

    def _field_B(self):
        """动态演化的磁场（从主窗口场面板取，供 Floquet AC 位移后端）。"""
        fld = getattr(self.main, 'field', None)
        return float(fld.b_gauss()) if fld is not None else 0.0

    def _sambe_adv_params(self):
        """Sambe 高级参数 → (frozen_h_ns, per_freq_M)；None = 默认/自动。"""
        fh = self.frozen_h_sb.value()
        frozen = fh if fh > 0 else None
        txt = self.perfreq_m_edit.text().strip()
        pf = None
        if txt:
            try:
                import ast
                pf = ast.literal_eval(txt)
                if isinstance(pf, list):
                    pf = [int(x) for x in pf]
                else:
                    pf = int(pf)
            except Exception:
                pf = None
                self.main.log(f'per_freq_M 解析失败：{txt!r}（用自动探针）')
        return frozen, pf

    def _on_method_changed(self, method):
        """方法切换（sambe/hard/mixed——t71 删 ip4/auto）；sambe 两体高级
        参数行显隐联动。"""
        self._update_sambe_adv_visibility()

    def _update_sambe_adv_visibility(self):
        """Sambe 高级行显隐：双原子 + sambe 时显示（t71：auto 档已删）。"""
        m = self.method_cb.currentText()
        adv = getattr(self, '_sambe_adv_box', None)
        if adv is not None:
            adv.setVisible(self._is_two_body and m == 'sambe')

    def _on_display_mode(self, *_):
        """显示选项切换：态布居 → 显示态数 topn；荧光读出 → 读出态下拉。
        （t27：topn/readout 同位互斥显隐；t30 微调：切回态布居关复杂配置窗）"""
        fluo = self.display_mode_cb.currentIndex() == 1
        self.readout_cb.setVisible(fluo)
        self.topn_sb.setVisible(not fluo)
        if not fluo:
            self._close_readout_dlg()
        self._redraw_last()

    def _on_readout_activated(self, idx):
        """t30：用户主动选中"复杂配置…"→ 弹权重配置窗（每次选中都弹/常驻）；
        选中真实能级 → 关复杂配置窗。仅 activated（用户交互）触发——
        currentIndexChanged 只重绘，程序化 refill 重选哨兵项不弹窗（防阻塞）。"""
        if idx >= 0 and self.readout_cb.itemData(idx) == _COMPLEX_READOUT:
            self._open_readout_config()
        else:
            self._close_readout_dlg()

    def _close_readout_dlg(self):
        """关复杂配置窗（显示模式切回态布居 / 选中真实能级 / 用户关闭）。"""
        dlg = getattr(self, '_readout_dlg', None)
        if dlg is not None:
            dlg.close()

    def _open_readout_config(self):
        """复杂配置：常驻非模态窗口列出全部能级 + 读出率（0-1，默认 0）。

        读出率 = 在该态前提下能被荧光计数的概率；**无确定/取消按钮**——
        每个 spinbox 改动**实时写回** self._readout_weights 并刷新画布；
        窗口随 readout_cb 选择而显隐（选"复杂配置…"弹出，选真实能级或
        切回态布居自动关闭），也可手动 X 关闭。已存在且可见 → raise_() 复用。
        """
        states = (getattr(self, '_run_states', None)
                  or getattr(self, '_states', None) or [])
        if not states:
            self.main.log('无基组，无法配置读出率')
            return
        dlg = getattr(self, '_readout_dlg', None)
        if dlg is not None and dlg.isVisible():
            dlg.raise_()
            dlg.activateWindow()
            return
        w = getattr(self, '_readout_weights', None)
        if w is None or len(w) != len(states):
            w = np.zeros(len(states))
        dlg = QDialog(self)
        dlg.setAttribute(Qt.WA_DeleteOnClose)
        dlg.setWindowTitle('复杂配置：能级读出率')
        lay = QVBoxLayout(dlg)
        lay.addWidget(QLabel(
            '读出率 = 在该态前提下能被荧光计数的概率（0-1，默认 0）；'
            '改动实时生效并刷新画布，无需确认。'))
        scroll = QScrollArea(dlg)
        scroll.setWidgetResizable(True)
        inner = QWidget()
        gl = QGridLayout(inner)
        gl.setColumnStretch(0, 1)
        for i, s in enumerate(states):
            gl.addWidget(QLabel(s['label']), i, 0)
            sb = QDoubleSpinBox()
            sb.setRange(0.0, 1.0)
            sb.setDecimals(3)
            sb.setValue(float(w[i]))
            sb.setFixedWidth(90)
            # t32：无确定/取消——值变化实时写回权重并刷新画布
            sb.valueChanged.connect(
                lambda v, i=i: self._set_readout_weight(i, v))
            gl.addWidget(sb, i, 1)
        scroll.setWidget(inner)
        lay.addWidget(scroll)
        dlg.setFixedWidth(540)
        self._readout_dlg = dlg
        dlg.destroyed.connect(
            lambda *a: setattr(self, '_readout_dlg', None))
        dlg.show()

    def _set_readout_weight(self, i, v):
        """复杂配置：单个能级读出率实时写回 + 刷新画布（t32）。"""
        w = getattr(self, '_readout_weights', None)
        if w is None or not (0 <= i < len(w)):
            return
        w[i] = float(v)
        self._redraw_last()

    def _redraw_last(self):
        """按当前显示选项重绘最近一次结果（无结果则不动）。

        修复（用户报告 KeyError: 'pops'）：按**结果自身的键**分派
        （pops_2d=两体 / pops=单原子）——运行后切换单/双原子模式或恢复
        存档时，_last_res 与当前 _is_two_body 可能不一致，旧逻辑按当前
        模式取键会 KeyError；结果与模式不符时按其自身类型重绘。"""
        res = getattr(self, '_last_res', None)
        if res is None:
            return
        states = getattr(self, '_run_states', None)
        if 'pops_2d' in res:
            self._draw_bot_pair(res)
        elif 'pops' in res and states is not None:
            self._draw_bot(states, res)

    def _sync_t_end(self, t):
        """把演化窗口告知波形面板 → 带宽显示随引擎查表上限收紧（t78）。"""
        if t is None or not len(t):
            return
        te = float(t[-1])
        for name in ('wave_panel', 'wave_panel_A', 'wave_panel_B'):
            p = getattr(self, name, None)
            if p is not None and hasattr(p, 'set_t_end'):
                try:
                    p.set_t_end(te)
                except Exception:                               # noqa: BLE001
                    pass

    def _check_arb_applicability(self, waves, states, t):
        """完整驱动（ArbitraryWave）的适用性体检 → 状态栏提示（t78）。

        任意波形走「文件即驱动」路径（无载频：H ⊃ amp(t)·A），三个已知陷阱：
          · 文件带宽不覆盖目标跃迁 → **静默零效果**（不是轻微失真）；
          · 基组含光学流形 → 该路径不参与光场旋转 → DOP853 步长被 THz 相位
            钉死（实测 2 ns 演化 >60 s 未完成）；
          · 超出引擎查表带宽的部分被折叠（f_max 已含该上限）。
        """
        from ..core.field_build import arb_applicability
        arbs = [w for w in (waves or [])
                if isinstance(w, waveforms.ArbitraryWave)]
        if not arbs or t is None or not len(t):
            return
        for w in arbs:
            try:
                warns = arb_applicability(self.cache, states, w, float(t[-1]))
            except Exception as e:                              # noqa: BLE001
                warns = [f'适用性检查失败（{type(e).__name__}: {e}）']
            for msg in warns:
                self.main.log(f'⚠ 任意波形「{w.name}」：{msg}')


    def _on_run_clicked(self):
        """运行/取消同一个按钮（2026-09-14 使用者裁决，取代进度弹窗）。

        运行中（`_run_active`）→ 取消；空闲 → 启动。与 AI 同源：AI 也调
        `self._run()`，故按钮文本与可取消性对两侧一致。
        """
        if getattr(self, '_run_active', False):
            self._cancel_run()
        else:
            self._run()

    def _reset_progress_ui(self, idle=False):
        """进度控件复位。常驻控件可能由主窗口底栏提供（见 app_entry），
        独立构造时是本地兜底，故一律判空后写。"""
        if getattr(self, '_pbar', None) is not None:
            self._pbar.setValue(0)
        if getattr(self, '_stage_label', None) is not None:
            self._stage_label.setText('空闲' if idle else '准备…')

    def _show_progress(self):
        """**已废除**（2026-09-14 使用者裁决）：进度改由窗口底栏常驻控件显示，
        不再弹对话框。保留此空方法只为兼容既有接线（`_dlg_timer`、
        `ai/server.py` 的 patch/restore）——调它不做任何事。
        """
        return

    def _cancel_run(self):
        """用户点取消：置取消标志，等待计算线程退出。"""
        self.progress_arr[1] = 1.0
        if self._dlg is not None:
            self._dlg.setWindowTitle('正在取消…')
        # 常驻底栏（取代弹窗）：取消也要有可见反馈
        if getattr(self, '_stage_label', None) is not None:
            self._stage_label.setText('正在取消…')

    def _poll_progress(self):
        """轮询进度数组（t72 阶段协议：prog[2]=0 准备/1 积分/2 收尾，
        prog[0]=阶段内进度）：
        - 准备（含 JIT 编译/装载与主线程 V_dd/基组）条 0→5%，标签带墙钟
          耗时——JIT 编译本身无内部进度，耗时数字即活性；
        - 积分条 5→95%（driver 逐步写 t/T；nogil 后轮询不再被编译内核
          饿死——修复前 GIL 被 njit 持有，整段积分进度条纹丝不动且取消
          按钮无响应）；
        - 收尾（跨扇区投影/loss/约化）条 95→100%。"""
        # ⚠ 2026-09-14 修（用户实测：算完停在 90% 多）：收尾后仍可能有**已排队**的
        #   定时器事件投递进来（`QTimer.stop()` 不撤销已 post 的 timeout），
        #   在此写入就会把终态覆盖回 ~95%。终态只由 `_finish_run_ui` 写。
        if not getattr(self, '_run_active', False):
            return
        pa = self.progress_arr
        stage = int(pa[2]) if len(pa) > 2 else 0
        p0 = min(max(float(pa[0]), 0.0), 1.0)
        el = time.perf_counter() - getattr(self, '_run_t0',
                                           time.perf_counter())
        if stage == 2:
            val = 950 + int(50 * p0)
            lab = f'收尾…（{el:.0f}s）'
        elif stage == 1:
            val = 50 + int(900 * p0)
            lab = f'积分 {min(int(100 * p0), 100)}%'
        else:
            lab = getattr(self, '_prep_label', '') or '准备/JIT 编译'
            val = int(50 * p0)
            lab = f'{lab}…（{el:.0f}s）'
        if getattr(self, '_pbar', None) is not None:
            self._pbar.setValue(val)
        if getattr(self, '_stage_label', None) is not None:
            self._stage_label.setText(lab)

    def _preflight_cost(self, prepared, ops):
        """mixed 单体起算前的**成本报告** → 用户选择。返回 None = 取消；否则返回
        （可能因成本档改动而重新 prepare 的）prepared。数据源改自 ops.prepare 的
        handle（RunSpec 内含 states/fields），不再手动建（B-1③）。

        背景（用户裁决）：成本预算曾作为引擎参数、默认 4e4 rad，构造期就按成本窗
        删对且静默 —— 那是"为了跑得快悄悄改模型"。现在默认不限制（solve.COST_BUDGET
        留空），真实耦合一律保留；代价与省法都摆到运行前这一次选择里。
        AI 发起的运行不弹窗（`_ai_initiated`）：只写日志、采用"完整物理"路径。
        """
        run = (prepared or {}).get('run') or {}
        if run.get('method') != 'mixed' or run.get('two_body'):
            return prepared              # 成本报告只针对单体 mixed（其余直接跑）
        try:
            import numpy as _np
            from ..core.evolve_mixed import evolve_mixed as _em
            states = run['states']
            fields = run['fields']
            t = _np.linspace(0, run['t_max'], run['npts'])
            keys = ('per_freq_M', 'r_crit', 'bw_opt', 'mw_cap', 'mw_max',
                    'mw_arb', 'phase_budget')
            kw = {k: v for k, v in (run.get('engine_kwargs') or {}).items() if k in keys}
            rep = (_em(states, fields, _np.zeros(len(states), complex), t,
                       dry_run=True, **kw) or {}).get('cost') or {}
        except Exception as e:                        # noqa: BLE001
            self.main.log(f'成本预检跳过（{e}）')
            return prepared
        dmax = float(rep.get('max_det_mhz') or 0.0)
        alts = rep.get('alt') or {}
        if not alts or dmax < 3000.0:
            return prepared                # 便宜算例：不打扰，直接跑
        alt = alts.get(4e4) or alts[sorted(alts)[-1]]
        worst = rep.get('worst') or {}
        head = (f'本次「完整物理」保留 {rep.get("n_mw", 0)} 对微波耦合，'
                f'最远失谐 {dmax:.4g} MHz')
        detail = (
            f'这些耦合 Θ 判据全部通过（真实存在，非数值噪声）。\n'
            f'预计积分步数 ~{rep.get("est_steps", 0):.3g}'
            f'（比只留近共振对慢约 {alt.get("speedup", 1):.1f}×）。\n'
            + (f'最远一对：{worst.get("label_a", "?")} ↔ '
               f'{worst.get("label_b", "?")}'
               f'（|v|={worst.get("abs_v_mhz", 0):.4g} MHz，'
               f'Δ={worst.get("det_mhz", 0):.4g} MHz）\n' if worst else '')
            + f'\n成本预算：'
              + ('不限制（保留全部真实耦合）'
                 if rep.get("budget") in (None, 0) else
                 f'{float(rep.get("budget")):g} rad')
              + f'（窗宽 {rep.get("mw_cap_mhz", 0):.4g} MHz）\n'
              f'若改上成本档 4e4 rad → 剔除 {alt.get("n_drop", 0)} 对，'
              f'最远失谐降到 {alt.get("max_det_mhz", 0):.4g} MHz。\n'
              f'注意：剔除的是**真实物理通道**（freeze 三光子梯子第三级'
              f'就是这么丢的）。')
        if getattr(self, '_ai_initiated', False):
            self.main.log(f'[成本预检] {head}；{detail.splitlines()[1]}'
                          f'AI 运行 → 采用完整物理路径（不弹窗）')
            return prepared
        # ⚠ 弹对话框之前**必须先把进度框撤下**：进度框在 _run 开头 400ms 后
        # 自动弹出，且是 `Qt.WindowModal` —— 模态阻塞的是整个父窗口（含挂在
        # 它下面的其它对话框），于是成本报告被压在下面、**根本点不到**（实测 bug）。
        timer = getattr(self, '_dlg_timer', None)
        if timer is not None:
            timer.stop()
        dlg = getattr(self, '_dlg', None)
        dlg_was = bool(dlg is not None and dlg.isVisible())
        if dlg_was:
            dlg.hide()
        from PyQt5.QtWidgets import QMessageBox
        box = QMessageBox(self.main)
        box.setWindowTitle('运行前成本报告')
        box.setIcon(QMessageBox.Information)
        box.setText(head)
        box.setInformativeText(detail)
        b_full = box.addButton('完整物理运行（推荐）', QMessageBox.AcceptRole)
        b_fast = box.addButton(f'剔除远失谐对（快 ~{alt.get("speedup", 1):.1f}×）',
                               QMessageBox.DestructiveRole)
        b_cancel = box.addButton('取消', QMessageBox.RejectRole)
        box.setDefaultButton(b_full)
        box.setWindowModality(Qt.ApplicationModal)   # 独立于父窗口，确保可点
        box.exec_()
        clicked = box.clickedButton()
        if clicked is b_cancel:
            # 取消＝本轮未起算：进度框不再恢复，由调用方 _finish_run_ui 收尾
            self.main.log('成本预检：用户取消本次运行')
            return None
        # 继续运行 → 恢复进度 UI。**不要**再启动 400ms 定时器：到点会再造
        # 一个进度框（旧框没人更新 → 屏幕上两个框、一个动一个不动，实测 bug）。
        if dlg_was and dlg is not None:
            dlg.show()
        else:
            self._show_progress()        # 幂等：已有框就直接返回
        if clicked is b_fast:
            # 写会话真值（solve.COST_BUDGET，运行级）→ 进存档、可 undo、AI 也看得见；
            # 配置变了 → 重新 prepare（build_run 按成本窗剔除远失谐对）。
            self.main.apply_solve_params({'solve.COST_BUDGET': 4e4})
            self.main.log(f'成本档已设 solve.COST_BUDGET=4e4 rad'
                          f'（剔除 {alt.get("n_drop", 0)} 对远失谐耦合，'
                          f'快 ~{alt.get("speedup", 1):.1f}×）')
            return ops.prepare(actor='gui', reason='GUI 运行（成本档后重建）',
                               diag=bool(self._god_mode))
        return prepared


    def set_god_mode(self, on):
        """菜单栏模式开关的唯一下游：显示/隐藏检视器 + 徽标。

        **不销毁检视器**——极简模式下它照样存在、诊断与引擎参数绑定照旧，
        只是窗口隐藏（登记跟真值源走，不跟显隐走）。
        """
        self._god_mode = bool(on)
        self.mode_badge.setText('· 上帝模式' if on else '')
        self.mode_badge.setVisible(bool(on))
        if on:
            self.show_inspector()
        else:
            self.hide_inspector()

    def show_inspector(self, diag=None):
        """确保检视器存在并显示（diag=None 时保留上次内容/占位说明）。"""
        self._open_inspector(diag)

    def hide_inspector(self):
        if self._inspector is not None:
            self._inspector.hide()

    def _on_inspector_rerun(self):
        """检视器「应用并重跑」→ 用旋钮当前值重跑演化。"""
        if not self._god_mode:
            self.main.set_mode(god=True)      # 唯一入口（会显示检视器）
        self.main.log('上帝模式：按检视器旋钮重跑…')
        self._run()

    def _open_inspector(self, diag=None):
        """确保检视器存在（非模态）并按需装载诊断。

        常驻语义（t82）：建一次就不再销毁；回填走会话现值（registry），
        保证组件与后台参数一一绑定。
        非 dict 入参（如 Qt 信号误传的 bool）一律按 None 处理。
        """
        from .solver_inspector import SolverInspector
        if not isinstance(diag, dict):
            diag = None
        if self._inspector is None:
            self._inspector = SolverInspector(self, win=self.main)
            self._inspector.rerun_requested.connect(self._on_inspector_rerun)
        # 回填走会话现值（引擎参数页从 registry 建：engine.* 只读 / solve.* 可写）
        self._inspector.refresh_param_display()
        if diag is not None:
            self._inspector.load_diag(diag)
        else:
            self._inspector.set_placeholder(
                '本次运行未产出诊断数据——diag 仅 mixed 引擎（单/两体）'
                '且方法未被自动降级时产出；hard / sambe 暂未接线。')
        self._inspector.show()
        self._inspector.raise_()
        self._inspector.activateWindow()
        return self._inspector

    def _on_evolve_done(self, result, error):
        """演化完成回调（RunWorker 回传 ops 结果 dict：summary + artifact_id）。

        大载荷（pops/t 数组）落 artifact（I3）→ 这里 load_artifact 取回绘图，与 AI
        跑出的产物同一份（人机同产物）。单/两体统一走这里（_is_two_body 分流绘图）。
        """
        if getattr(self, '_hub_run', False):
            # Hub 模式：产物在 Hub 侧，本地不 load artifact；只报 summary。
            self._hub_run = False
            if error:
                self.main.log('Hub 运行失败：%s' % error)
                self._finish_run_ui()
                return
            self.main.log('Hub 运行完成：%s' % ((result or {}).get('summary') or result))
            self._finish_run_ui()
            return
        self._finish_run_ui()
        if error == 'cancelled':
            return                                  # 静默（用户主动取消）
        if error is not None:
            QMessageBox.warning(
                self, '演化失败',
                f'{error}\n\n可尝试：降低光强、调整驱动频率/失谐，'
                '或减小 t_max/点数。')
            return
        # 从 artifact 取回演化数组（人机同产物；I3 大载荷不内联）
        res = None
        try:
            from ..mecha.ops import load_artifact
            aid = (result or {}).get('artifact_id')
            _ops = self.main.app_bridge.ops
            loaded = load_artifact(_ops.root, aid) if aid else None
            if loaded is not None:
                res = loaded[1]                     # (run, arrays, manifest) → arrays
        except Exception as e:                       # noqa: BLE001
            self.main.log('结果载入失败：%s' % e, 10000)
        self._last_res = res
        self._last_summary = (result or {}).get('summary')
        if self._god_mode:                           # 上帝模式 → 检视器（diag 接线见 B-2）
            self._open_inspector(None)
        self._draw_top(self.wave_data)
        states = getattr(self, '_run_states', None)
        if res is not None:
            if bool(self._is_two_body):
                self._draw_bot_pair(res)
            elif states is not None:
                self._draw_bot(states, res)


    def _draw_top(self, E_list):
        """顶部：各通道**光强曲线**（I = 峰值光强×包络）。

        - 单原子：只用 A 格（拉满整图）；双原子：A 波形上格、B 波形下格；
        - 每格内各通道独立 y 标度（twinx 外移），ylim 紧贴数据（无空白）；
        - 上/下格子与布居图共用 left/right 边距 → 时间轴完全对齐；
        - 无标题（省空间）。

        E_list: [(name, t, env, freq_mhz[, intensity_mwcm2]), ...]
        双原子模式下 name 带 'A:'/'B:' 前缀（分组绘制）。
        """
        # 重置：删除先前轮次遗留的全部 twin 轴（只保留两格主轴）
        for ax in list(self.fig_top.axes):
            if ax not in (self.ax_topA, self.ax_topB):
                self.fig_top.delaxes(ax)

        # 分组：双原子按前缀拆分；单原子全部进 A
        if self._is_two_body:
            grp = {'A': [e for e in E_list if str(e[0]).startswith('A:')],
                   'B': [e for e in E_list if str(e[0]).startswith('B:')]}
            if not grp['A'] and len(E_list) and any(
                    not str(e[0]).startswith(('A:', 'B:')) for e in E_list):
                grp['A'] = E_list          # 兼容无前缀旧数据 → 全算 A
            groups = [('A', grp['A']), ('B', grp['B'])]
        else:
            groups = [('A', list(E_list))]
        # 统一颜色映射（A/B 共用同一 log 色标）。注意：任意波形 freq=0，
        # 不参与 log 映射（log(0)=NaN 会让曲线绘制异常），统一用灰色。
        all_f = np.array([e[3] for e in E_list], float)
        ok_f = all_f[all_f > 0]
        fmin = max(ok_f.min(), 1e-3) if ok_f.size else 1e-3
        fmax = max(ok_f.max(), 1e-3) if ok_f.size else 1e-3
        if fmax <= fmin:
            fmax = fmin + 1.0
        norm = LogNorm(vmin=fmin, vmax=fmax)
        cmap = colormaps['plasma']
        ARB_COLOR = '#8a8a8a'          # 无单频（任意波形）固定灰

        def _color(f):
            return cmap(norm(f)) if f > 0 else ARB_COLOR

        # 布局（统一 left/right=0.07/0.95 与布居图对齐；无标题）
        axA = self.ax_topA
        axB = self.ax_topB
        if self._is_two_body:
            axB.set_visible(True)
            axA.set_position([0.07, 0.53, 0.88, 0.41])
            axB.set_position([0.07, 0.06, 0.88, 0.41])
        else:
            axB.set_visible(False)
            axA.set_position([0.07, 0.04, 0.88, 0.90])

        for ax, (tag, glist) in zip((axA, axB), groups):
            self._plot_wave_group(ax, tag, glist, _color)
        self.canvas_top.draw_idle()

    def _plot_wave_group(self, ax, tag, glist, color_of):
        """在给定 axes 内画一组波形曲线（单 y 轴，各通道归一化；无标题）。

        2026-08-31 修改：放弃 twinx 多 y 轴（多轴堆叠 + 零基线错位 + 与
        布居图对齐困难），改为**单 y 轴 + 逐通道归一化**——所有曲线画在
        同一坐标系（y 范围 [0,1]），形状/时序清晰、天然对齐，无轴堆叠。
        """
        ax.clear()
        ax.set_xlabel('t (ns)')
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.grid(alpha=0.3)
        ax.axhline(0.0, color='0.35', lw=0.8, alpha=0.6, zorder=0)  # 零基线
        if not glist:
            ax.set_ylabel('归一化光强')
            return
        has_I = len(glist[0]) >= 5
        ax.set_ylabel('归一化光强' if has_I else '归一化包络')
        for k, e in enumerate(glist):
            name, t, env, f = e[0], e[1], e[2], e[3]
            I = float(e[4]) if has_I else 1.0
            y = I * np.asarray(env, float)
            ymax = float(y.max()) if y.size else 0.0
            c = color_of(f)
            # 归一化到 [0,1]（峰值 = 1；全零时保持 0）
            yn = y / ymax if ymax > 1e-12 else y
            ax.plot(t, yn, lw=1.5, color=c, label=name)
            ax.fill_between(t, yn, 0.0, color=c, alpha=0.22)
        ax.set_ylim(0.0, 1.02)
        loc = MaxNLocator(nbins=4)
        ticks = np.asarray(loc.tick_values(0.0, 1.02))
        ticks = ticks[(ticks >= 0.0) & (ticks <= 1.02)]
        if not np.any(np.isclose(ticks, 0.0)):
            ticks = np.unique(np.append(ticks, 0.0))
        ax.set_yticks(ticks)
        ax.legend(fontsize=7, loc='upper right')
        # x 从 0 起 → 与下方布居图 x=0 对齐（waveforms_to_env 用 0 起始网格）
        x0 = float(glist[0][1][0])
        x1 = float(glist[0][1][-1])
        ax.set_xlim(0.0 if x0 < 0 else x0, max(x1, x0))

    def _draw_bot(self, states, res):
        """底部：态布居（只画前 N 个平均布居最大的态，N=topn_sb）+ loss 曲线。"""
        if self.display_mode_cb.currentIndex() == 1:
            self._draw_readout_single(states, res)
            return
        ax = self.ax_bot; ax.clear()
        ax.set_xlabel('t (ns)'); ax.set_ylabel('布居'); ax.grid(alpha=0.3)
        pops = res['pops']
        sel = set(self._top_pop_rows(states, res))
        for i, s in enumerate(states):
            if i not in sel:
                continue
            ax.plot(res['t'], pops[:, i], lw=1.2, label=s['label'])
        loss = np.asarray(res.get('loss', np.zeros(len(res['t']))))
        if loss.max() > 1e-12:
            ax.plot(res['t'], loss, lw=1.6, ls='--', color='#333333',
                    label='loss（离场）')
        ax.margins(x=0)                     # 取消 x 方向边距 → 贴齐 y 轴与右缘
        ax.set_xlim(res['t'][0], res['t'][-1])
        ax.legend(fontsize=8)
        self.canvas_bot.draw_idle()

    def _draw_bot_pair(self, res):
        """双原子底部：两体态布居（按平均布居取前 N 个 product 态）。"""
        if self.display_mode_cb.currentIndex() == 1:
            self._draw_readout_pair(res)
            return
        ax = self.ax_bot; ax.clear()
        pops_2d = res['pops_2d']
        states_2d = res.get('states_2d', [])
        N = pops_2d.shape[1]
        n_show = min(self.topn_sb.value(), N)
        avg = pops_2d.mean(axis=0)
        order = np.argsort(avg)[::-1]
        sel = set(order[:n_show])
        ax.set_xlabel('t (ns)'); ax.set_ylabel('布居'); ax.grid(alpha=0.3)
        for i in range(N):
            if i not in sel:
                continue
            label = states_2d[i]['label'] if i < len(states_2d) else str(i)
            ax.plot(res['t'], pops_2d[:, i], lw=1.2, label=label)
        loss = np.asarray(res.get('loss_A', np.zeros(len(res['t']))))
        loss += np.asarray(res.get('loss_B', np.zeros(len(res['t']))))
        if loss.max() > 1e-12:
            ax.plot(res['t'], loss, lw=1.6, ls='--', color='#333333',
                    label='loss')
        ax.margins(x=0)                     # 取消 x 方向边距 → 贴齐 y 轴与右缘
        ax.set_xlim(res['t'][0], res['t'][-1])
        ax.legend(fontsize=7, ncol=1)
        self.canvas_bot.draw_idle()

    def _readout_index(self, n_states):
        """荧光读出态索引（基组索引；越界/无选择回退 0；复杂配置哨兵透传）。"""
        i = self.readout_cb.currentData() if self.readout_cb.count() else None
        if i == _COMPLEX_READOUT:
            return _COMPLEX_READOUT
        if i is None or not (0 <= i < n_states):
            return 0
        return int(i)

    def _readout_weights_vec(self, n_states):
        """复杂配置权重向量（长度校验；无/不符 → 全 0）。"""
        w = getattr(self, '_readout_weights', None)
        if w is None or len(w) != n_states:
            return np.zeros(n_states)
        return np.asarray(w, float)

    def _draw_readout_single(self, states, res):
        """单原子荧光读出：|⟨读出态|ψ⟩|² 随时间；复杂配置 → Σ wᵢ·Pᵢ 加权期望。

        纯幺正演化下 pops 即 |⟨i|ψ⟩|²（精确读出概率）；耗散时 loss 不算
        读出（离场），曲线为 |⟨i|ψ⟩|² 的存活部分——无需 psi（record_amp），
        布居即投影模方。
        """
        ax = self.ax_bot; ax.clear()
        ax.set_xlabel('t (ns)'); ax.set_ylabel('读出概率'); ax.grid(alpha=0.3)
        i = self._readout_index(len(states))
        pops = res['pops']
        if i == _COMPLEX_READOUT:
            w = self._readout_weights_vec(len(states))
            y = pops @ w
            lab = '荧光计数期望 Σ wᵢ·Pᵢ' + ('（权重全 0）'
                                           if not np.any(w) else '')
            ax.plot(res['t'], y, lw=1.6, label=lab)
            ax.set_ylim(-0.02, max(1.02, 1.05 * float(y.max())))
        else:
            ax.plot(res['t'], pops[:, i], lw=1.6,
                    label=f'|⟨读出态|ψ⟩|²  {states[i]["label"]}')
            ax.set_ylim(-0.02, 1.02)
        ax.margins(x=0)
        ax.set_xlim(res['t'][0], res['t'][-1])
        ax.legend(fontsize=8)
        self.canvas_bot.draw_idle()

    def _draw_readout_pair(self, res):
        """双原子荧光读出：A/B 两原子在读出态上的约化概率；复杂配置 →
        Σ wᵢ·ρ_A/B 加权期望。

        约化：ρ_A(i) = Σ_j pops_2d[i·N+j]，ρ_B(i) = Σ_j pops_2d[j·N+i]
        （k=i·N+j 的 product 基排序；pops_2d 足够，无需 psi 投影）。
        """
        ax = self.ax_bot; ax.clear()
        ax.set_xlabel('t (ns)'); ax.set_ylabel('读出概率'); ax.grid(alpha=0.3)
        pops_2d = res['pops_2d']
        N = int(round(np.sqrt(pops_2d.shape[1])))
        i = self._readout_index(N)
        lab = self.readout_cb.currentText()
        if i == _COMPLEX_READOUT:
            w = self._readout_weights_vec(N)
            yA = (pops_2d.reshape(-1, N, N).transpose(0, 2, 1)
                  @ w).sum(axis=1)        # ρ_A(i) = Σ_j pops[i·N+j]
            yB = (pops_2d.reshape(-1, N, N) @ w).sum(axis=1)
            # ρ_B(i) = Σ_j pops[j·N+i]
            labA = 'A: 荧光计数期望 Σ wᵢ·ρ_A' + ('（权重全 0）'
                                                 if not np.any(w) else '')
            labB = 'B: Σ wᵢ·ρ_B'
            ax.plot(res['t'], yA, lw=1.4, label=labA)
            ax.plot(res['t'], yB, lw=1.4, ls='--', label=labB)
            ymax = max(1.02, 1.05 * float(max(yA.max(), yB.max())))
            ax.set_ylim(-0.02, ymax)
        else:
            ax.plot(res['t'], pops_2d[:, i * N:(i + 1) * N].sum(axis=1),
                    lw=1.4, label=f'A: ρ_A({lab})')
            ax.plot(res['t'], pops_2d[:, i::N].sum(axis=1),
                    lw=1.4, ls='--', label=f'B: ρ_B({lab})')
            ax.set_ylim(-0.02, 1.02)
        ax.margins(x=0)
        ax.set_xlim(res['t'][0], res['t'][-1])
        ax.legend(fontsize=8)
        self.canvas_bot.draw_idle()
