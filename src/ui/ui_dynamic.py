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

import numpy as np
from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QEvent
from PyQt5.QtGui import QColor, QIcon, QPixmap
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
                             QLabel, QPushButton, QSpinBox, QDoubleSpinBox,
                             QComboBox, QListWidget, QListWidgetItem,
                             QMessageBox, QDialog, QProgressBar, QRadioButton,
                             QButtonGroup, QTabWidget, QFrame, QScrollArea,
                             QLineEdit)
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.colors import LogNorm
from matplotlib import colormaps
from matplotlib.ticker import MaxNLocator

from ..core import core, evolve, pair_core
from . import ui_waves
from ..core import waveforms
from .evolve_worker import EvolveWorker
from .ui_waves import state_key, find_state_index

# ip4 的近/远分流边界（原共振 cut 控件，t25 移除 UI 后内部默认，单位 MHz）
_IP4_NEAR_CUTOFF_MHZ = 5000.0
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
        # 能级组增删由主程序事件驱动刷新（统一机制，不散布同步调用）
        main.groups_changed.connect(self.refresh_groups)
        self._is_two_body = False
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
        self.vdd_R_sb = QDoubleSpinBox()
        self.vdd_R_sb.setRange(0.1, 1000.0)
        self.vdd_R_sb.setValue(5.0)
        self.vdd_R_sb.setDecimals(1)
        self.vdd_R_sb.setFixedWidth(128)
        self.vdd_R_sb.setToolTip('原子间距 R（μm）')
        vdd_lay.addWidget(self.vdd_R_sb)
        vdd_lay.addWidget(QLabel('θ:'))
        self.vdd_theta_sb = QDoubleSpinBox()
        self.vdd_theta_sb.setRange(0.0, 180.0)
        self.vdd_theta_sb.setValue(0.0)
        self.vdd_theta_sb.setDecimals(1)
        self.vdd_theta_sb.setFixedWidth(128)
        self.vdd_theta_sb.setToolTip('极角 θ（°）')
        vdd_lay.addWidget(self.vdd_theta_sb)
        vdd_lay.addWidget(QLabel('φ:'))
        self.vdd_phi_sb = QDoubleSpinBox()
        self.vdd_phi_sb.setRange(0.0, 360.0)
        self.vdd_phi_sb.setValue(0.0)
        self.vdd_phi_sb.setDecimals(1)
        self.vdd_phi_sb.setFixedWidth(128)
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
        self.frozen_h_sb = QDoubleSpinBox()
        self.frozen_h_sb.setRange(0.0, 100.0)
        self.frozen_h_sb.setValue(0.0)
        self.frozen_h_sb.setDecimals(2)
        self.frozen_h_sb.setFixedWidth(72)
        self.frozen_h_sb.setToolTip(
            '全频段 Sambe 纯冻结段长（ns，包络变化尺度内）；0 = 默认路径')
        sa_lay.addWidget(self.frozen_h_sb)
        sa_lay.addWidget(QLabel('per_freq_M'))
        self.perfreq_m_edit = QLineEdit('')
        self.perfreq_m_edit.setPlaceholderText('[1,1] 或空=自动')
        self.perfreq_m_edit.setFixedWidth(120)
        self.perfreq_m_edit.setToolTip(
            '逐载频光子块阶数（列表 [M1,M2,…] 或 int 全体）；空 = 自动探针')
        sa_lay.addWidget(self.perfreq_m_edit)
        sa_lay.addStretch(1)
        self._sambe_adv_box.setVisible(False)
        p.addLayout(sa_lay)

        # ---- 波形管理：单原子与双原子「原子A」共用同一面板 ----
        # 先在单原子模式校准波形 → 切到双原子时原子 A 原样迁移，无需重配。
        # 实现：wave_panel_A 就是 wave_panel（同一实例），模式切换时把该实例
        # 在「单原子槽位」与「tab-A 槽位」之间整体迁移（Qt widget 可换 parent）。
        self.wave_panel = ui_waves.WavePanel()       # 共享面板（单原子 = 原子A）
        self.wave_panel.changed.connect(self._on_waves_changed)
        self.wave_panel_A = self.wave_panel          # 同一实例
        self.wave_panel_B = ui_waves.WavePanel()
        self.wave_panel_B.changed.connect(self._on_waves_changed)

        self._wave_single_host = QWidget()           # 单原子模式的槽位
        _wsh = QVBoxLayout(self._wave_single_host)
        _wsh.setContentsMargins(0, 0, 0, 0)
        _wsh.addWidget(self.wave_panel)

        self._wave_tab_A_host = QWidget()            # 双原子 tab-A 的槽位
        _wah = QVBoxLayout(self._wave_tab_A_host)
        _wah.setContentsMargins(0, 0, 0, 0)

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
        self.tmax_sb = QDoubleSpinBox()
        self.tmax_sb.setRange(10, 1e6); self.tmax_sb.setValue(1500.0)
        self.tmax_sb.setDecimals(0)
        rowt.addWidget(self.tmax_sb)
        rowt.addWidget(QLabel('点数'))
        self.npts_sb = QSpinBox()
        self.npts_sb.setRange(10, 50000); self.npts_sb.setValue(1501)
        rowt.addWidget(self.npts_sb)
        rowt.addWidget(QLabel('方法'))
        self.method_cb = QComboBox()
        # t25：选项重构——ip4：RWA 相互作用绘景 RK4（近/远分流用内部默认
        # 5 GHz，不暴露 UI）；sambe：第一性原理 Sambe 光子块（实验室系无
        # RWA/无旋转系/无近远分流，原 'exact' 更名，引擎 evolve_sambe）；
        # hard：微波硬积分 DOP853 零近似 + 光场旋转系 + 耗散（numba 加速）。
        # t62：加 mixed（混合引擎——光 Sambe 光子块 + 微波当场，强光近共振
        # + 微波场景；两体 mixed 未验收前两体页退回 sambe）与 auto（智能
        # 路由——route_auto 按场/能级选 hard/sambe/mixed）。
        self.method_cb.addItems(['ip4', 'sambe', 'hard', 'mixed', 'auto'])
        self.method_cb.setCurrentText('hard')      # t25：默认 hard
        self.method_cb.currentTextChanged.connect(self._on_method_changed)
        rowt.addWidget(self.method_cb)
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
        self.run_btn.clicked.connect(self._run)
        row_disp.addWidget(self.run_btn)
        p.addLayout(row_disp)
        # 直接作为主程序右侧面板的普通区块（无独立滚动区，融入同一层级）
        return panel

    # ---------- 单原子/双原子切换 ----------
    def _on_mode_changed(self, mode_id):
        """0=单原子, 1=双原子。切换面板可见性 + 共享波形面板迁移。"""
        self._is_two_body = (mode_id == 1)
        # V_dd 参数
        self._vdd_box.setVisible(self._is_two_body)
        # Sambe 高级行（双原子 + sambe/auto）
        self._update_sambe_adv_visibility()
        # 波形面板：共享面板（单原子 = 原子A）在槽位间迁移 + 容器显隐。
        # 注意：不再 setVisible 共享面板本身——它迁到哪个容器就由哪个容器
        # 展示（hidden 会让 tab-A"看起来没有波形"，且被布局压成最小高度）。
        self._place_shared_waves(self._is_two_body)
        self._wave_single_host.setVisible(not self._is_two_body)
        self._wave_tab.setVisible(self._is_two_body)
        # 初态面板
        self.psi_combo.setVisible(not self._is_two_body)
        self._psi_box.setVisible(self._is_two_body)
        # 自发辐射：双原子模式现已支持（寿命/落点=丢失通道），档位放开
        self.decay_mode_cb.setEnabled(True)
        # 刷新初态下拉
        self._on_m_changed()

    def _place_shared_waves(self, in_tab):
        """共享波形面板（wave_panel ≡ wave_panel_A）在单原子槽位 / tab-A 槽位间迁移。"""
        pan = self.wave_panel
        target = self._wave_tab_A_host if in_tab else self._wave_single_host
        if pan.parent() is target:
            return
        if pan.parent() is not None:
            pan.parent().layout().removeWidget(pan)
        target.layout().addWidget(pan)

    def eventFilter(self, obj, ev):
        """双原子页签栏的拖放：把波形拖到另一原子的标签（不切页）即复制到该原子。"""
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
        self._states = list(states)      # 供 serialize_state 把初态索引映射为身份键
        pending = getattr(self, '_pending_psi_keys', None)

        def _fill(combo, pending_key, fallback_text):
            combo.clear()
            for s in states:
                combo.addItem(s['label'])
            if pending_key:
                i = find_state_index(states, pending_key)
                if i >= 0:
                    combo.setCurrentIndex(i)
                    return
            if fallback_text:
                idx = combo.findText(fallback_text)
                if idx >= 0:
                    combo.setCurrentIndex(idx)
            elif combo.count():
                combo.setCurrentIndex(0)

        # 单原子初态
        cur = self.psi_combo.currentText()
        _fill(self.psi_combo,
              pending.get('single') if pending else None, cur)
        # 双原子初态（A/B 各自独立选择）
        cur_t = [c.currentText() for c in (self.psi_combo_A, self.psi_combo_B)]
        for k, combo in enumerate((self.psi_combo_A, self.psi_combo_B)):
            _fill(combo,
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
        # t30：基组重建 → 读出率权重重置全 0；恢复档有 pending 则用之
        self._readout_weights = np.zeros(len(states), dtype=np.float64)
        pw = getattr(self, '_pending_readout_weights', None)
        if pw is not None and len(pw) == len(states):
            self._readout_weights = np.asarray(pw, float)
        self._pending_readout_weights = None
        # t30 微调：复杂配置只能用户主动选中——cur_r/_pending_readout 为哨兵
        # 项时忽略，回退第一个真实能级（初始/恢复档都不得默认落在复杂配置上）
        want = cur_r
        if not want or want == _COMPLEX_LABEL:
            want = getattr(self, '_pending_readout', '')
        if not want or want == _COMPLEX_LABEL:
            want = ''
        if want:
            idx = self.readout_cb.findText(want)
            if idx >= 0:
                self.readout_cb.setCurrentIndex(idx)
            else:
                self.readout_cb.setCurrentIndex(0)
        elif self.readout_cb.count():
            self.readout_cb.setCurrentIndex(0)
        self._pending_readout = ''
        self._pending_psi_keys = None
        self._refresh_decay_combos(states)
        # 基组变化 → 波形面板辅助 g/e 下拉数据源同步（A ≡ 共享面板，去重）
        for wp in (self.wave_panel, self.wave_panel_B):
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
            out.append(evolve.decay(fi, c['gamma_mhz'], ti))
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

    def _current_states(self):
        """动态基组 = 全部子能级（与共用能级表一致，无需选择）：
        低激发态展开超精细 F/mF（用场面板磁场 B），里德堡态保持 mJ。
        能量 = 组中心 + Zeeman(B) + DC-Stark(E)，与静态页/两体页同基准。
        label 含完整量子数，布居图/g/e 下拉共用。"""
        fld = getattr(self.main, 'field', None)
        B = (fld.b_gauss() if fld is not None
             else getattr(self.main, 'field_B', 0.0))
        E_vpcm = (fld.e_vpcm() if fld is not None else 0.0)
        groups = list(self.main_groups)
        specs = []
        for g in groups:
            n, l, j = g['n'], g['l'], g['j']
            hf = core.get_hyperfine_levels(self.cache, n, l, j, B)
            if hf is None:
                for m in np.arange(-j, j + 1e-9, 1.0):
                    E = core._sublevel_energy(self.cache, n, l, j, m, B)
                    if abs(E_vpcm) > 1e-9:
                        E += core.dc_stark_shift_mhz(
                            self.cache, n, l, j, m, groups, E_vpcm, B)
                    specs.append(dict(n=n, l=l, j=j, kind='mJ', m=float(m),
                                      E_mhz=float(E)))
            else:
                for F, mF, e_hf, _w, amps in hf:
                    E0 = self.cache.E_mhz(n, l, j)
                    if abs(E_vpcm) > 1e-9:
                        E0 += core.dc_stark_shift_mhz(
                            self.cache, n, l, j, mF, groups, E_vpcm, B)
                    specs.append(dict(n=n, l=l, j=j, kind='F', F=F, mF=mF,
                                      E_hf=float(e_hf), E_mhz=float(E0),
                                      amps=amps))
        return evolve.make_levels(self.cache, specs)

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

    # ---------- 状态管理（序列化/恢复；文件 IO 与菜单入口在主程序） ----------
    def serialize_state(self):
        """只导出科学数据与参数，不导出显示参数。

        - 科学数据/参数：模式（单/双原子）、初态、t_max/点数、方法、
          共振截断、衰变档位与通道、V_dd 几何（R/θ/φ）、各原子波形
        - 不保存显示参数：显示态数 topn、波形预览 wave_data
          （预览曲线由波形重算，属于派生显示数据）
        """
        return {
            'dynamic': {
                'mode': 1 if self._is_two_body else 0,
                'psi': self.psi_combo.currentIndex(),
                'psi_A': self.psi_combo_A.currentIndex(),
                'psi_B': self.psi_combo_B.currentIndex(),
                # 身份键（跨 B 稳定，恢复按最近能量匹配；旧档无此字段时回退索引）
                'psi_key': self._psi_key_for(self.psi_combo),
                'psi_key_A': self._psi_key_for(self.psi_combo_A),
                'psi_key_B': self._psi_key_for(self.psi_combo_B),
                't_max': self.tmax_sb.value(),
                'npts': self.npts_sb.value(),
                'method': self.method_cb.currentText(),
                # t62：sambe 两体高级参数（frozen_h_ns / per_freq_M）
                'frozen_h_ns': self.frozen_h_sb.value(),
                'per_freq_M': self.perfreq_m_edit.text(),
                'decay_mode': self.decay_mode_cb.currentIndex(),
                # t25：显示选项（0 态布居 / 1 荧光读出）+ 读出态（身份文本）
                'display_mode': self.display_mode_cb.currentIndex(),
                'readout': self.readout_cb.currentText(),
                # t30：复杂配置读出率权重（索引随基组；无配置时全 0）
                'readout_weights': list(getattr(
                    self, '_readout_weights',
                    np.zeros(len(getattr(self, '_states', []))))),
            },
            'pair': {
                'R_um': self.vdd_R_sb.value(),
                'theta_deg': self.vdd_theta_sb.value(),
                'phi_deg': self.vdd_phi_sb.value(),
            },
            'waves': waveforms.serialize_waves(self.wave_panel.waves),
            # A ≡ 共享面板：只存一份（waves），避免任意波形采样数组双倍体积
            'waves_A': ([] if self.wave_panel_A is self.wave_panel
                        else waveforms.serialize_waves(
                            self.wave_panel_A.waves)),
            'waves_B': waveforms.serialize_waves(self.wave_panel_B.waves),
            'decays': [dict(c) for c in self.decay_cfgs],
        }

    def _psi_key_for(self, combo):
        """组合框当前选中态的身份键（JSON 友好 list）；无可用时返回 None。"""
        i = combo.currentIndex()
        st = getattr(self, '_states', None)
        if st and 0 <= i < len(st):
            return list(state_key(st[i]))
        return None

    def _serialize_dynamic(self):
        """兼容旧接口：由本模块导出自己的状态快照。"""
        return self.serialize_state()

    def restore_state(self, state):
        """按本模块状态快照恢复（旧档缺新字段时用默认值，向后兼容）。

        注意：旧档里的 wave_data（波形预览，派生显示数据）不再读取，
        预览曲线在恢复后由波形重算（_on_waves_changed）。
        """
        if not isinstance(state, dict):
            return
        dyn = state.get('dynamic', {})
        # 模式（科学/工作流参数）：先恢复，决定波形/初态面板可见性
        mode = int(dyn.get('mode', 0))
        self._is_two_body = (mode == 1)
        self._radio_single.setChecked(not self._is_two_body)
        self._radio_pair.setChecked(self._is_two_body)
        self._on_mode_changed(mode)
        # 科学参数
        self.tmax_sb.setValue(float(dyn.get('t_max', 1500.0)))
        self.npts_sb.setValue(int(dyn.get('npts', 1501)))
        # t25：旧档方法名映射（exact→sambe）；未知/缺省 → hard
        # t62：白名单加 mixed/auto——'auto' 原样恢复（新智能路由语义；
        # 极老档的 backend-auto 也落 auto——语义演进可接受）
        _m = {'exact': 'sambe'}.get(
            dyn.get('method', 'hard'), dyn.get('method', 'hard'))
        if _m not in ('ip4', 'sambe', 'hard', 'mixed', 'auto'):
            _m = 'hard'
        self.method_cb.setCurrentText(_m)
        # t62：sambe 两体高级参数恢复（缺省 = 默认 0/空）
        self.frozen_h_sb.setValue(float(dyn.get('frozen_h_ns', 0.0)))
        self.perfreq_m_edit.setText(str(dyn.get('per_freq_M', '')))
        # 显示选项 + 读出态（读出态文本在 _on_m_changed 填充时按 label 恢复）
        self.display_mode_cb.setCurrentIndex(
            int(dyn.get('display_mode', 0)))
        self._on_display_mode()
        self._pending_readout = str(dyn.get('readout', ''))
        # t30：复杂配置读出率权重（长度不符/缺失 → 全 0，_on_m_changed 应用）
        _rw = dyn.get('readout_weights')
        self._pending_readout_weights = (
            list(_rw) if isinstance(_rw, (list, tuple)) else None)
        # 自发辐射档位（双原子模式现支持寿命/丢失通道，同样恢复）
        self.decay_mode_cb.setCurrentIndex(int(dyn.get('decay_mode', 0)))
        pair = state.get('pair', {})
        self.vdd_R_sb.setValue(float(pair.get('R_um', 5.0)))
        self.vdd_theta_sb.setValue(float(pair.get('theta_deg', 0.0)))
        self.vdd_phi_sb.setValue(float(pair.get('phi_deg', 0.0)))
        # 波形（单原子 + 双原子 A/B；A ≡ 共享面板 → 只恢复一次）
        wv = state.get('waves')
        if wv:
            self.wave_panel.set_waves(waveforms.restore_waves(wv))
        wvA = state.get('waves_A')
        if wvA and self.wave_panel_A is not self.wave_panel:
            self.wave_panel_A.set_waves(waveforms.restore_waves(wvA))
        wvB = state.get('waves_B')
        if wvB:
            self.wave_panel_B.set_waves(waveforms.restore_waves(wvB))
        self.decay_cfgs = [dict(c) for c in state.get('decays', [])]
        self._update_decay_list()
        # 初态（基组下拉由 groups_changed 刷新填充；优先身份键，旧档回退索引）
        pkeys = {'single': dyn.get('psi_key'),
                 'A': dyn.get('psi_key_A'),
                 'B': dyn.get('psi_key_B')}
        self._pending_psi_keys = pkeys if any(
            k for k in pkeys.values() if k) else None
        if self._pending_psi_keys is None:
            self.psi_combo.setCurrentIndex(int(dyn.get('psi', 0)))
            self.psi_combo_A.setCurrentIndex(int(dyn.get('psi_A', 0)))
            self.psi_combo_B.setCurrentIndex(int(dyn.get('psi_B', 0)))
        self.main.groups_changed.emit()
        # 显示预览由波形重算（wave_data 不再存档）
        self._on_waves_changed()

    def _restore_dynamic(self, state):
        """兼容旧接口：按本模块快照恢复。"""
        self.restore_state(state)

    # ---------- 运行演化（后台线程 + 进度条 + 可取消） ----------
    def _run(self):
        if getattr(self, '_worker', None) is not None \
                and self._worker.isRunning():
            QMessageBox.information(self, '提示', '已有演化在运行，请先等待或取消。')
            return
        self.refresh_groups()          # 运行前取最新共用能级组（无需外部散布同步）
        if not self.main_groups:
            QMessageBox.information(self, '提示', '请先在主程序添加能级组。')
            return
        states = self._current_states()
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
        mode = self._decay_mode()
        if mode == 'custom':
            decays = self._resolve_decay_cfgs(states)
        elif mode == 'auto':
            decays = evolve.auto_decays(self.cache, states)
            if decays:
                self.main.log(f'[动态] 自动自发辐射：{len(decays)} 通道'
                              '（Γ=1/τ(ARC)，τ≤100µs，落点=丢失）')
        else:
            decays = []
        t = np.linspace(0, self.tmax_sb.value(), self.npts_sb.value())
        self._run_states = states
        # t28：prog 3 元素——[0]=全程进度(0-1)、[1]=取消标志、[2]=阶段
        # （0=准备/JIT、1=积分[driver 写 t/T]、2=收尾）；hard 的 JIT 装载
        # ~19s 期间进度条不再死 0%（阶段标签提示）。
        self.progress_arr = np.zeros(3)

        if self._is_two_body:
            self._run_two_body(states, t)
        else:
            self._run_single_body(states, t, decays)

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
        """t25：方法切换（ip4/sambe/hard）；near_cutoff 控件已移除，
        ip4 用内部默认 5 GHz。t62：sambe 两体高级参数行显隐联动。"""
        self._update_sambe_adv_visibility()

    def _update_sambe_adv_visibility(self):
        """Sambe 高级行显隐：双原子 + sambe（或 auto 可能路由 sambe）时显示。"""
        m = self.method_cb.currentText()
        adv = getattr(self, '_sambe_adv_box', None)
        if adv is not None:
            adv.setVisible(self._is_two_body and m in ('sambe', 'auto'))

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
        """按当前显示选项重绘最近一次结果（无结果则不动）。"""
        res = getattr(self, '_last_res', None)
        if res is None:
            return
        states = getattr(self, '_run_states', None)
        if self._is_two_body:
            self._draw_bot_pair(res)
        elif states is not None:
            self._draw_bot(states, res)

    def _run_single_body(self, states, t, decays):
        """单原子演化路径。"""
        waves = list(self.wave_panel.waves)
        method = self.method_cb.currentText()
        if method == 'auto':
            # t62 auto 智能路由（目标 4）：route_auto 按场/能级判 hard/sambe/
            # mixed——缓存精确判据（失谐+强度）；reason 显示状态栏
            from ..core.route import route_auto
            method, reason = route_auto(states, waves, two_body=False,
                                        cache=self.cache)
            self.main.log(f'auto 路由 → {method}：{reason}')
        if method == 'sambe':
            # 第一性原理场（Sambe 光子块）：全部偶极对耦合，无近/远分流与
            # 静态 AC 注入（原 'exact' 更名，引擎 evolve_sambe）
            fields = waveforms.build_fields_exact(self.cache, states, waves)
        elif method == 'mixed':
            # t62 混合引擎（光 Sambe 光子块 + 微波当场）——fields 同构
            # build_fields_exact（evolve_mixed 正式版内部做 far_sos AC 化）
            fields = waveforms.build_fields_exact(self.cache, states, waves)
            self.main.log('mixed：光场 Sambe 光子块 + 微波当场'
                          '（φ-帧，t62 验证 ≡frozen 1e-5~1e-8 级）')
        elif method == 'hard':
            # t20：两场引擎 meta（光旋转 + 微波硬积分）——普通波走
            # build_two_field；ArbitraryWave 走 arbitrary_mw_group 并入
            # mw_groups（cos(φ_f(t)) 任意相位路径，t18）。
            from ..core.evolve_twofield import (build_two_field,
                                                arbitrary_mw_group)
            regular = [w for w in waves
                       if not isinstance(w, waveforms.ArbitraryWave)]
            arb = [w for w in waves
                   if isinstance(w, waveforms.ArbitraryWave)]
            meta = build_two_field(self.cache, states, regular)
            for w in arb:
                meta['mw_groups'].append(
                    arbitrary_mw_group(self.cache, states, w))
            fields = meta
        else:
            # ip4：RWA 近/远分流用内部默认 5 GHz（t25 移除 UI 控件）
            fields = waveforms.build_fields(
                waves, states, self.cache,
                near_cutoff_mhz=_IP4_NEAR_CUTOFF_MHZ,
                B_gauss=self._field_B())
        self.wave_data = waveforms.waveforms_to_env(waves, t)
        psi0 = self.psi_combo.currentIndex()

        self._dlg = None; self._pbar = None
        self._dlg_timer = QTimer(self)
        self._dlg_timer.setSingleShot(True)
        self._dlg_timer.timeout.connect(self._show_progress)
        self._dlg_timer.start(400)
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._poll_progress)
        self._timer.start(100)

        self._worker = EvolveWorker(states, fields, psi0, t,
                                     method,
                                     'auto',
                                     self.progress_arr, decays=decays)
        self._worker.done.connect(self._on_evolve_done)
        self._worker.log_msg.connect(self.main.log)
        self._worker.start()

    def _run_two_body(self, states, t):
        """双原子演化路径（后台线程）。"""
        from src.core import pair_evolve
        waves_A = list(self.wave_panel_A.waves)
        waves_B = list(self.wave_panel_B.waves)
        method = self.method_cb.currentText()
        if method == 'auto':
            # t62 auto：两体路由（route_auto two_body=True——mixed 两体未验收
            # 前 route 判据保守回 sambe AC 化路径或 hard）
            from ..core.route import route_auto
            wa = list(self.wave_panel_A.waves) + list(self.wave_panel_B.waves)
            method, reason = route_auto(states, wa, two_body=True,
                                        cache=self.cache)
            self.main.log(f'auto 路由（两体）→ {method}：{reason}')
        if method == 'mixed':
            # 两体 mixed（用例 8 dim 1452）未验收前：退回 sambe（全频段
            # 两体 Sambe 可用）并提示
            self.main.log('mixed 两体引擎验收中——本次退回 sambe'
                          '（全频段两体 Sambe）')
            method = 'sambe'
        if method == 'hard':
            # t21：pair hard——各原子两场 meta（普通波 build_two_field +
            # ArbitraryWave→arbitrary_mw_group）；V_dd 在主线程预计算
            from ..core.evolve_twofield import (build_two_field,
                                                arbitrary_mw_group)

            def _meta(wvs):
                reg = [w for w in wvs
                       if not isinstance(w, waveforms.ArbitraryWave)]
                arb = [w for w in wvs
                       if isinstance(w, waveforms.ArbitraryWave)]
                m = build_two_field(self.cache, states, reg)
                for w in arb:
                    m['mw_groups'].append(
                        arbitrary_mw_group(self.cache, states, w))
                return m

            fields_A = _meta(waves_A)
            fields_B = _meta(waves_B)
        elif method == 'sambe':
            # 第一性两体（Sambe）：全部偶极对耦合，无近/远分流与静态 AC 注入
            fields_A = waveforms.build_fields_exact(self.cache, states,
                                                    waves_A)
            fields_B = waveforms.build_fields_exact(self.cache, states,
                                                    waves_B)
        else:
            # ip4：内部默认 5 GHz 近/远分流（t25）
            fields_A = waveforms.build_fields(
                waves_A, states, self.cache,
                near_cutoff_mhz=_IP4_NEAR_CUTOFF_MHZ,
                B_gauss=self._field_B())
            fields_B = waveforms.build_fields(
                waves_B, states, self.cache,
                near_cutoff_mhz=_IP4_NEAR_CUTOFF_MHZ,
                B_gauss=self._field_B())
        # 波形预览：合并 A+B
        env_A = waveforms.waveforms_to_env(waves_A, t)
        env_B = waveforms.waveforms_to_env(waves_B, t)
        self.wave_data = [(f'A:{nm}', th, env, f, I)
                          for nm, th, env, f, I in env_A] \
                        + [(f'B:{nm}', th, env, f, I)
                           for nm, th, env, f, I in env_B]
        psi0_A = self.psi_combo_A.currentIndex()
        psi0_B = self.psi_combo_B.currentIndex()
        R_um = self.vdd_R_sb.value()
        theta_deg = self.vdd_theta_sb.value()
        phi_deg = self.vdd_phi_sb.value()

        # ---- 自发辐射（寿命通道 v1：单原子 decay 列表，两原子共用 states）----
        # 在**主线程**构建：auto 档调 ARC getStateLifetime（SQLite 线程安全约束）。
        decays_A = decays_B = None
        mode = self.decay_mode_cb.currentIndex()
        if mode == 0:                       # 自动（ARC 寿命，落点=丢失）
            decays_A = evolve.auto_decays(self.cache, states)
            decays_B = evolve.auto_decays(self.cache, states)
            if decays_A or decays_B:
                self.main.log(f'[双原子] 自动寿命：A {len(decays_A)}'
                              f'/B {len(decays_B)} 条通道（落点=丢失）')
        elif mode == 2:                     # 自定义（双原子 v1：仅落点=丢失）
            lab2idx = {s['label']: i for i, s in enumerate(states)}
            outA, outB = [], []
            for c in self.decay_cfgs:
                fi = lab2idx.get(c['from'])
                if fi is None:
                    continue
                if c.get('to'):
                    raise NotImplementedError(
                        '双原子自定义衰变 v1 仅支持「丢失」落点'
                        '（spontaneous radiation 按寿命处理）；'
                        '显式落点请用单原子模式。')
                ch = evolve.decay(fi, float(c['gamma_mhz']), None)
                outA.append(ch)
                outB.append(ch)
            decays_A, decays_B = outA, outB
        # 注意：A/B 共用同一 states → 同一通道集（各自原子的"从态"相同，
        # 但两体基上原子 A/B 的 from 分别作用——由 pair_evolve 张量化）。

        # ---- V_dd 在主线程预计算（复用 self.cache）----
        # ARC 的 SQLite 连接不能跨线程（sqlite3: objects created in a
        # thread can only be used in that same thread）；pair_evolve 的
        # R_um+cache 路径会在工作线程里访问 cache → 崩溃。这里把 V_dd
        # 矩阵先在主线程算好，工作线程只做纯 numpy/scipy 演化。
        # 基组顺序 = 实际演化所用 states 顺序（与 pair_evolve 内部一致）。
        self.main.log(f'[双原子] 预计算 V_dd（R={R_um:g} μm, '
                      f'θ={theta_deg:g}°, φ={phi_deg:g}°）…')
        vdd_matrix = pair_core.vdd_sparse(
            self.cache, [dict(s) for s in states],
            R_um, theta_deg, phi_deg)
        n_dd = vdd_matrix.nnz
        self.main.log(f'[双原子] V_dd 非零元 {n_dd} 个，开始演化…')

        self._dlg = None; self._pbar = None
        self._dlg_timer = QTimer(self)
        self._dlg_timer.setSingleShot(True)
        self._dlg_timer.timeout.connect(self._show_progress)
        self._dlg_timer.start(400)
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._poll_progress)
        self._timer.start(100)

        from .evolve_worker import EvolveWorkerPair
        # t26：外部后端已退役归档（checks/_archive/）——
        # sambe 走 Python evolve_sambe_pair / evolve_exact_pair（同引擎）
        # t62：sambe 两体高级参数（frozen_h_ns/per_freq_M——面板，目标 5）
        frozen_h, per_freq_M = (self._sambe_adv_params()
                                if method == 'sambe' else (None, None))
        self._worker = EvolveWorkerPair(
            states, fields_A, fields_B, psi0_A, psi0_B,
            t, self.progress_arr,
            vdd_matrix=vdd_matrix,
            decays_A=decays_A, decays_B=decays_B,
            method=method,
            per_freq_M=per_freq_M, frozen_h_ns=frozen_h)
        self._worker._cache_ref = self.cache
        self._worker.done.connect(self._on_pair_evolve_done)
        self._worker.log_msg.connect(self.main.log)
        self._worker.start()

    def _show_progress(self):
        """延迟显示进度对话框（仅当计算仍在进行时）。"""
        if getattr(self, '_worker', None) is None or not self._worker.isRunning():
            return
        dlg = QDialog(self)
        dlg.setWindowTitle('演化')
        lay = QVBoxLayout(dlg)
        lay.setContentsMargins(14, 12, 14, 12)
        self._stage_label = QLabel('正在准备…')      # t28：阶段文本（_poll 更新）
        lay.addWidget(self._stage_label)
        self._pbar = QProgressBar(dlg)
        self._pbar.setRange(0, 1000)
        lay.addWidget(self._pbar)
        btn = QPushButton('取消', dlg)
        btn.clicked.connect(self._cancel_run)
        lay.addWidget(btn)
        dlg.setWindowModality(Qt.WindowModal)
        dlg.setFixedWidth(320)
        self._dlg = dlg
        dlg.show()

    def _cancel_run(self):
        """用户点取消：置取消标志，等待计算线程退出。"""
        self.progress_arr[1] = 1.0
        if self._dlg is not None:
            self._dlg.setWindowTitle('正在取消…')

    def _poll_progress(self):
        """轮询进度数组更新进度条 + 阶段标签（t28：阶段化覆盖 hard 的
        JIT 装载/积分/收尾全程——准备 5%、积分 5→95%、收尾 95%）。"""
        pa = self.progress_arr
        stage = int(pa[2]) if len(pa) > 2 else 0
        p0 = float(pa[0])
        if stage == 2:
            val, lab = 950, '收尾…'
        elif p0 >= 0.002:                       # 积分中（driver 写 t/T）
            val, lab = int(50 + 900 * p0), f'积分 {min(int(100 * p0), 100)}%'
        else:
            val, lab = 50, 'JIT 编译/准备…'
        if self._pbar is not None:
            self._pbar.setValue(val)
        if getattr(self, '_stage_label', None) is not None:
            self._stage_label.setText(lab)

    def _on_evolve_done(self, res, error):
        self._dlg_timer.stop()
        self._timer.stop()
        if self._dlg is not None:
            self._dlg.close()
            self._dlg = None
            self._pbar = None
        self._worker = None
        if error == 'cancelled':
            return                                  # 静默（用户主动取消）
        if error is not None:
            QMessageBox.warning(
                self, '演化失败',
                f'{error}\n\n可尝试：降低光强、调整驱动频率/失谐，'
                '或减小 t_max/点数。')
            return
        self._last_res = res
        self._draw_top(self.wave_data)
        states = getattr(self, '_run_states', None)
        if states is not None:
            self._draw_bot(states, res)

    def _on_pair_evolve_done(self, res, error):
        """双原子演化完成回调。"""
        self._dlg_timer.stop()
        self._timer.stop()
        if self._dlg is not None:
            self._dlg.close()
            self._dlg = None; self._pbar = None
        self._worker = None
        if error == 'cancelled':
            return
        if error is not None:
            QMessageBox.warning(self, '演化失败', f'{error}')
            return
        self._last_res = res
        self._draw_top(self.wave_data)
        self._draw_bot_pair(res)

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
