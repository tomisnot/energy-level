# -*- coding: utf-8 -*-
"""动态分析面板（能级.py 新增 tab）：波形列表 → 实际电磁场 → 态演化。

画布（主程序窗口内，上下）：
  上：各波形实际电磁场包络 E_i(t)（waveforms 波形包络，无通道无 TTL）
  下：态布居演化 P(t)（evolve.evolve）

参数面板（借鉴时序/波形编辑器 + 按 evolve 需要）：
  波形管理（WavePanel：频率/光强/偏振/包络/偏移）· 初态 · 时间范围/点数 ·
    演化方法 · 自发辐射 · 运行
"""
import numpy as np
from PyQt5.QtCore import Qt, QTimer, pyqtSignal
from PyQt5.QtGui import QColor, QIcon, QPixmap
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel,
                             QPushButton, QSpinBox, QDoubleSpinBox, QComboBox,
                             QListWidget, QListWidgetItem, QMessageBox,
                             QDialog, QProgressBar, QRadioButton,
                             QButtonGroup, QTabWidget, QFrame)
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.colors import LogNorm
from matplotlib import colormaps

from ..core import core, evolve, pair_core
from . import ui_waves
from ..core import waveforms
from .evolve_worker import EvolveWorker
from .ui_waves import state_key, find_state_index


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

    # ---------- 画布：上光场、下演化 ----------
    def _build_canvases(self):
        self.fig_top = Figure(figsize=(7, 3.2), dpi=100)
        self.ax_top = self.fig_top.add_subplot(111)
        self.fig_top.subplots_adjust(left=0.07, right=0.97, top=0.93,
                                     bottom=0.10)
        self.canvas_top = FigureCanvas(self.fig_top)
        self.fig_bot = Figure(figsize=(7, 3.6), dpi=100)
        self.ax_bot = self.fig_bot.add_subplot(111)
        self.fig_bot.subplots_adjust(left=0.07, right=0.97, top=0.93,
                                     bottom=0.10)
        self.canvas_bot = FigureCanvas(self.fig_bot)
        self.ax_top.set_title('光场光强 I(t)（各波形通道，颜色=频率）')
        self.ax_bot.set_title('态布居 P(t)')
        for ax in (self.ax_top, self.ax_bot):
            ax.set_xlabel('t (ns)'); ax.grid(alpha=0.3)
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
        self.ax_top.set_ylabel('光强 (mW/cm²)')
        self.ax_bot.set_ylabel('布居')

    # ---------- 布局 ----------
    def _build_controls(self):
        """动态分析特有参数面板（独立控件，由主程序放入右侧共用面板）。"""
        panel = QWidget()
        p = QVBoxLayout(panel)
        p.setSpacing(6)

        # ---- 单原子/双原子切换 ----
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
        row_mode.addStretch(1)
        p.addLayout(row_mode)

        # ---- V_dd 参数（仅双原子模式可见）----
        self._vdd_box = QWidget()
        vdd_lay = QHBoxLayout(self._vdd_box)
        vdd_lay.setContentsMargins(0, 0, 0, 0)
        vdd_lay.setSpacing(4)
        vdd_lay.addWidget(QLabel('R(μm)'))
        self.vdd_R_sb = QDoubleSpinBox()
        self.vdd_R_sb.setRange(0.1, 1000.0)
        self.vdd_R_sb.setValue(5.0)
        self.vdd_R_sb.setDecimals(1)
        vdd_lay.addWidget(self.vdd_R_sb)
        vdd_lay.addWidget(QLabel('θ(°)'))
        self.vdd_theta_sb = QDoubleSpinBox()
        self.vdd_theta_sb.setRange(0.0, 180.0)
        self.vdd_theta_sb.setValue(0.0)
        self.vdd_theta_sb.setDecimals(1)
        vdd_lay.addWidget(self.vdd_theta_sb)
        vdd_lay.addWidget(QLabel('φ(°)'))
        self.vdd_phi_sb = QDoubleSpinBox()
        self.vdd_phi_sb.setRange(0.0, 360.0)
        self.vdd_phi_sb.setValue(0.0)
        self.vdd_phi_sb.setDecimals(1)
        vdd_lay.addWidget(self.vdd_phi_sb)
        p.addWidget(self._vdd_box)
        self._vdd_box.setVisible(False)

        # ---- 波形管理：单原子 = 单 WavePanel；双原子 = QTabWidget(A/B) ----
        self.wave_panel = ui_waves.WavePanel()       # 单原子默认波形面板
        self.wave_panel.changed.connect(self._on_waves_changed)

        self._wave_tab = QTabWidget()
        self.wave_panel_A = ui_waves.WavePanel()
        self.wave_panel_A.changed.connect(self._on_waves_changed)
        self.wave_panel_B = ui_waves.WavePanel()
        self.wave_panel_B.changed.connect(self._on_waves_changed)
        self._wave_tab.addTab(self.wave_panel_A, '原子 A')
        self._wave_tab.addTab(self.wave_panel_B, '原子 B')
        self._wave_tab.setVisible(False)              # 默认隐藏

        p.addWidget(self.wave_panel, 1)               # 单原子波形
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
        row2.addWidget(QLabel('显示态数'))
        self.topn_sb = QSpinBox()
        self.topn_sb.setRange(1, 40)
        self.topn_sb.setValue(8)
        self.topn_sb.setToolTip('按最近一次结果的平均布居取前 N 个态绘制'
                                '（无需手动勾选，内部自动筛选）')
        row2.addWidget(self.topn_sb)
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
        rowt.addWidget(QLabel('共振cut(MHz)'))
        self.near_cutoff_sb = QDoubleSpinBox()
        self.near_cutoff_sb.setRange(1.0, 1e6)
        self.near_cutoff_sb.setDecimals(0)
        self.near_cutoff_sb.setValue(5000.0)
        self.near_cutoff_sb.setToolTip(
            'real_field 近/远分流边界：缺省 5 GHz 保守地板（相干包含整个'
            '超精细/精细流形）；协议场景（STIRAP/拉曼想只相干驱动目标对）'
            '可调小（如 50 MHz）——流形其余态作远失谐 AC 位移（二阶，Ω/Δ≪1 时准确）')
        rowt.addWidget(self.near_cutoff_sb)
        rowt.addWidget(QLabel('方法'))
        self.method_cb = QComboBox()
        # 动态分析只用相互作用绘景 RK4（auto 无耗散恒走 ip4；有耗散走
        # lindblad）。floquet/pwc 已封存至 src/_archive/evolve_floquet_pwc.py。
        self.method_cb.addItems(['auto', 'ip4'])
        rowt.addWidget(self.method_cb)
        p.addLayout(rowt)

        # ---- 自发辐射：自动（ARC 寿命）/ 禁用 / 自定义（一行选择 + 可折叠自定义区）----
        rowdm = QHBoxLayout()
        rowdm.addWidget(QLabel('自发辐射'))
        self.decay_mode_cb = QComboBox()
        self.decay_mode_cb.addItems(['自动（ARC 寿命）', '禁用', '自定义'])
        self.decay_mode_cb.setToolTip(
            '自动：按 ARC 权威能级寿命自动配置 Γ=1/τ'
            '（仅 τ≤100 µs 的低激发态，落点=丢失，运行后状态栏显示通道数）；'
            '自定义：手动配置衰变通道（从态/Γ/落点/纯退相）')
        self.decay_mode_cb.currentIndexChanged.connect(self._on_decay_mode)
        rowdm.addWidget(self.decay_mode_cb, 1)
        p.addLayout(rowdm)
        # self.decay_auto_label = QLabel(
        #     '自动：Γ=1/τ(ARC)，仅 τ≤100 µs 低激发态，落点=丢失；'
        #     '里德堡慢衰变/黑体辐射请用 自定义')
        # self.decay_auto_label.setWordWrap(True)
        # self.decay_auto_label.setFixedHeight(16)
        # p.addWidget(self.decay_auto_label)
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

        self.run_btn = QPushButton('运行演化')
        self.run_btn.clicked.connect(self._run)
        p.addWidget(self.run_btn)
        # 直接作为主程序右侧面板的普通区块（无独立滚动区，融入同一层级）
        return panel

    # ---------- 单原子/双原子切换 ----------
    def _on_mode_changed(self, mode_id):
        """0=单原子, 1=双原子。切换面板可见性。"""
        self._is_two_body = (mode_id == 1)
        # V_dd 参数
        self._vdd_box.setVisible(self._is_two_body)
        # 波形面板
        self.wave_panel.setVisible(not self._is_two_body)
        self._wave_tab.setVisible(self._is_two_body)
        # 初态面板
        self.psi_combo.setVisible(not self._is_two_body)
        self._psi_box.setVisible(self._is_two_body)
        # 自发辐射在双原子模式下禁用（pair_evolve 尚不支持 Lindblad）
        self.decay_mode_cb.setEnabled(not self._is_two_body)
        if self._is_two_body:
            self.decay_mode_cb.setCurrentIndex(1)  # 切到"禁用"
        # 刷新初态下拉
        self._on_m_changed()

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
        self._pending_psi_keys = None
        self._refresh_decay_combos(states)
        # 基组变化 → 波形面板辅助 g/e 下拉数据源同步
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
        """波形增删/参数变化 → 立即刷新顶部光强预览（不等待运行）。"""
        t = np.linspace(0, self.tmax_sb.value(), self.npts_sb.value())
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
                'decay_mode': self.decay_mode_cb.currentIndex(),
                'near_cutoff': self.near_cutoff_sb.value(),
            },
            'pair': {
                'R_um': self.vdd_R_sb.value(),
                'theta_deg': self.vdd_theta_sb.value(),
                'phi_deg': self.vdd_phi_sb.value(),
            },
            'waves': waveforms.serialize_waves(self.wave_panel.waves),
            'waves_A': waveforms.serialize_waves(self.wave_panel_A.waves),
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
        self.method_cb.setCurrentText(dyn.get('method', 'auto'))
        self.near_cutoff_sb.setValue(float(dyn.get('near_cutoff', 5000.0)))
        # 双原子模式下自发辐射被强制"禁用"（pair_evolve 暂不支持 Lindblad），
        # 只对单原子模式恢复档位
        if not self._is_two_body:
            self.decay_mode_cb.setCurrentIndex(int(dyn.get('decay_mode', 0)))
        pair = state.get('pair', {})
        self.vdd_R_sb.setValue(float(pair.get('R_um', 5.0)))
        self.vdd_theta_sb.setValue(float(pair.get('theta_deg', 0.0)))
        self.vdd_phi_sb.setValue(float(pair.get('phi_deg', 0.0)))
        # 波形（单原子 + 双原子 A/B）
        wv = state.get('waves')
        if wv:
            self.wave_panel.set_waves(waveforms.restore_waves(wv))
        wvA = state.get('waves_A')
        if wvA:
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
        self.progress_arr = np.zeros(2)

        if self._is_two_body:
            self._run_two_body(states, t)
        else:
            self._run_single_body(states, t, decays)

    def _run_single_body(self, states, t, decays):
        """单原子演化路径。"""
        waves = list(self.wave_panel.waves)
        fields = waveforms.build_fields(
            waves, states, self.cache,
            near_cutoff_mhz=self.near_cutoff_sb.value())
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
                                     self.method_cb.currentText(),
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
        fields_A = waveforms.build_fields(
            waves_A, states, self.cache,
            near_cutoff_mhz=self.near_cutoff_sb.value())
        fields_B = waveforms.build_fields(
            waves_B, states, self.cache,
            near_cutoff_mhz=self.near_cutoff_sb.value())
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
        self._worker = EvolveWorkerPair(
            states, fields_A, fields_B, psi0_A, psi0_B,
            t, self.progress_arr, vdd_matrix=vdd_matrix)
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
        lay.addWidget(QLabel('正在演化…（计算中可取消）'))
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
        """轮询进度数组更新进度条。"""
        if self._pbar is not None:
            self._pbar.setValue(int(self.progress_arr[0] * 1000))

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
        """顶部：各通道**光强曲线**（I = 峰值光强×包络）——颜色按频率 log 映射；
        每个通道独立 y 轴标度（twin axes 外移避让），防止光强差异过大压扁曲线。

        E_list: [(name, t, env, freq_mhz[, intensity_mwcm2]), ...]
        （旧存档缺 intensity → 退回画无量纲包络，单 y 轴。）
        """
        # 重置：移除旧 twin 轴，只留主轴
        for ax in list(self.fig_top.axes):
            if ax is not self.ax_top:
                self.fig_top.delaxes(ax)
        ax = self.ax_top
        ax.clear()
        ax.set_title('光场光强 I(t)（颜色=频率，对数尺度；每通道独立 y 轴）')
        ax.set_xlabel('t (ns)')
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.grid(alpha=0.3)
        if not E_list:
            ax.set_ylabel('光强 (mW/cm²)')
            self.canvas_top.draw_idle()
            return
        has_I = len(E_list[0]) >= 5
        n = len(E_list)
        freqs = np.array([e[3] for e in E_list], float)
        fmin, fmax = max(freqs.min(), 1e-3), max(freqs.max(), 1e-3)
        if fmax <= fmin:
            fmax = fmin + 1.0
        norm = LogNorm(vmin=fmin, vmax=fmax)
        cmap = colormaps['plasma']
        colors = cmap(norm(freqs))
        # 右侧 twin y 轴空间（每额外通道一条，外移 45px 避让）
        right = max(0.78, 0.97 - 0.09 * (n - 1))
        self.fig_top.subplots_adjust(left=0.07, right=right, top=0.93,
                                     bottom=0.10)
        unit = '光强 (mW/cm²)' if has_I else '包络'
        for k, (e, c) in enumerate(zip(E_list, colors)):
            name, t, env, f = e[0], e[1], e[2], e[3]
            I = float(e[4]) if has_I else 1.0
            y = I * np.asarray(env, float)
            if k == 0:
                target = ax
                target.set_ylabel(f'{name}  {unit}', color=c)
            else:
                target = ax.twinx()
                target.spines['right'].set_position(('outward',
                                                      45.0 * (k - 1)))
                target.spines['top'].set_visible(False)
                target.yaxis.label.set_color(c)
                target.tick_params(axis='y', colors=c, labelsize=8)
                target.set_ylabel(f'{name}  {unit}', color=c, fontsize=8)
            target.plot(t, y, lw=1.5, color=c,
                        label=(name if k == 0 else None))
            target.fill_between(t, y, 0.0, color=c, alpha=0.22)
            target.set_ylim(bottom=0.0)
        ax.legend(fontsize=8, loc='upper right')
        self.canvas_top.draw_idle()

    def _draw_bot(self, states, res):
        """底部：态布居（只画前 N 个平均布居最大的态，N=topn_sb）+ loss 曲线。"""
        ax = self.ax_bot; ax.clear()
        ax.set_title(f'态布居 P(t)（按平均布居取前 '
                     f'{min(self.topn_sb.value(), len(states))} 个态）')
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
        ax.legend(fontsize=8)
        self.canvas_bot.draw_idle()

    def _draw_bot_pair(self, res):
        """双原子底部：两体态布居（按平均布居取前 N 个 product 态）。"""
        ax = self.ax_bot; ax.clear()
        pops_2d = res['pops_2d']
        states_2d = res.get('states_2d', [])
        N = pops_2d.shape[1]
        n_show = min(self.topn_sb.value(), N)
        avg = pops_2d.mean(axis=0)
        order = np.argsort(avg)[::-1]
        sel = set(order[:n_show])
        ax.set_title(f'两体态布居 P(t)（前 {n_show} 个 product 态）')
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
        ax.legend(fontsize=7, ncol=1)
        self.canvas_bot.draw_idle()
