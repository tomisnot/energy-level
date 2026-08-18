# -*- coding: utf-8 -*-
"""动态分析面板（能级.py 新增 tab）：波形列表 → 实际电磁场 → 态演化。

画布（主程序窗口内，上下）：
  上：各波形实际电磁场包络 E_i(t)（waveforms 波形包络，无通道无 TTL）
  下：态布居演化 P(t)（evolve.evolve）

参数面板（借鉴时序/波形编辑器 + 按 evolve 需要）：
  波形管理（WavePanel：频率/光强/偏振/包络/偏移）· 初态 · 时间范围/点数 ·
  演化方法/后端 · 自发辐射 · 运行
"""
import numpy as np
from PyQt5.QtCore import Qt, QTimer, pyqtSignal
from PyQt5.QtGui import QColor, QIcon, QPixmap
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel,
                             QPushButton, QSpinBox, QDoubleSpinBox, QComboBox,
                             QListWidget, QListWidgetItem, QMessageBox,
                             QDialog, QProgressBar)
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.colors import LogNorm
from matplotlib import colormaps

from ..core import core, evolve
from . import ui_waves
from ..core import waveforms
from .evolve_worker import EvolveWorker


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
        self._build_canvases()
        self.dynamic_widget = self._build_controls()
        # DynamicPanel 只含画布（上光场、下演化）；特有参数区由主程序放入右侧面板（与静态页同级）
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(2)
        root.addWidget(self.canvas_top, 5)
        root.addWidget(self.canvas_bot, 5)
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

        # ---- 波形管理（无通道、无独立窗口：右侧面板直接配置；占剩余空间）----
        self.wave_panel = ui_waves.WavePanel()
        self.wave_panel.changed.connect(self._on_waves_changed)
        p.addWidget(self.wave_panel, 1)      # 波形组滚动区吃满动态页右侧高度

        # ---- 能级：共用主程序"能级管理表"（日常数值 + m 选择），这里只选初态 ----
        row2 = QHBoxLayout()
        row2.addWidget(QLabel('初态'))
        self.psi_combo = QComboBox()
        row2.addWidget(self.psi_combo, 1)
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
        rowt.addWidget(QLabel('t_max'))
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
        self.method_cb.addItems(['auto', 'floquet', 'ip4', 'pwc'])
        rowt.addWidget(self.method_cb)
        rowt.addWidget(QLabel('后端'))
        self.backend_cb = QComboBox()
        self.backend_cb.addItems(['auto', 'cpu', 'gpu'])
        rowt.addWidget(self.backend_cb)
        p.addLayout(rowt)
        rowcut = QHBoxLayout()
        rowcut.addWidget(QLabel('近共振截断(MHz)'))
        self.near_cutoff_sb = QDoubleSpinBox()
        self.near_cutoff_sb.setRange(1.0, 1e6)
        self.near_cutoff_sb.setDecimals(0)
        self.near_cutoff_sb.setValue(5000.0)
        self.near_cutoff_sb.setToolTip(
            'real_field 近/远分流边界：缺省 5 GHz 保守地板（相干包含整个'
            '超精细/精细流形）；协议场景（STIRAP/拉曼想只相干驱动目标对）'
            '可调小（如 50 MHz）——流形其余态作远失谐 AC 位移（二阶，Ω/Δ≪1 时准确）')
        rowcut.addWidget(self.near_cutoff_sb)
        rowcut.addStretch(1)
        p.addLayout(rowcut)

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
        self.decay_auto_label = QLabel(
            '自动：Γ=1/τ(ARC)，仅 τ≤100 µs 低激发态，落点=丢失；'
            '里德堡慢衰变/黑体辐射请用 自定义')
        self.decay_auto_label.setWordWrap(True)
        self.decay_auto_label.setFixedHeight(16)
        p.addWidget(self.decay_auto_label)
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

    # ---------- 能级：从共用能级组生成（切页/运行时刷新，数据共享不散布同步） ----------
    def refresh_groups(self):
        """能级组与 m 选择共用主程序"能级管理表"（单一数据源），这里只刷新初态/驱动。"""
        self.main_groups = list(self.main.static.groups)
        self._on_m_changed()

    def _on_m_changed(self, *_):
        """基组变化 → 刷新初态下拉（保留当前选择）+ 布居态列表 + 波形辅助 g/e 下拉。"""
        states = self._current_states()
        cur = self.psi_combo.currentText()          # 记录选择，避免运行后刷回第一个
        self.psi_combo.clear()
        for s in states:
            self.psi_combo.addItem(s['label'])
        if cur:
            idx = self.psi_combo.findText(cur)
            if idx >= 0:
                self.psi_combo.setCurrentIndex(idx)
        self._refresh_decay_combos(states)
        # 基组变化 → 波形面板辅助 g/e 下拉数据源同步（内部只在 states 变化时重建；
        # 构造期 wave_panel 可能未就绪，防御）
        wp = getattr(self, 'wave_panel', None)
        if wp is not None:
            wp.set_states(states, self.cache)

    # ---------- 自发辐射（Lindblad）通道配置 ----------
    def _on_decay_mode(self, *_):
        """三档：自动（ARC 寿命）/ 禁用 / 自定义——只显示需要的控件。"""
        mode = self.decay_mode_cb.currentIndex()      # 0 自动 / 1 禁用 / 2 自定义
        self.decay_custom_box.setVisible(mode == 2)
        self.decay_auto_label.setVisible(mode == 0)

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
        return len(self.wave_panel.waves) > 0

    def _on_waves_changed(self):
        """波形增删/参数变化 → 立即刷新顶部光强预览（不等待运行）。"""
        t = np.linspace(0, self.tmax_sb.value(), self.npts_sb.value())
        self.wave_data = waveforms.waveforms_to_env(self.wave_panel.waves, t)
        self._draw_top(self.wave_data)

    # ---------- 状态管理（序列化/恢复；文件 IO 与菜单入口在主程序） ----------
    def _serialize_dynamic(self):
        """动态页全部参数与数据 → dict（数据组装在 serialization 模块）。"""
        from . import serialization
        return serialization.serialize_dynamic(
            self.main.static.groups,
            self.main.field.b_gauss(),
            {
                'psi': self.psi_combo.currentIndex(),
                't_max': self.tmax_sb.value(),
                'npts': self.npts_sb.value(),
                'method': self.method_cb.currentText(),
                'backend': self.backend_cb.currentText(),
                'topn': self.topn_sb.value(),
                'decay_mode': self.decay_mode_cb.currentIndex(),
                'near_cutoff': self.near_cutoff_sb.value(),
            },
            self.wave_panel.waves,
            self.decay_cfgs,
            [(nm, np.asarray(th), np.asarray(env), f, I)
             for nm, th, env, f, I in self.wave_data])

    def _restore_dynamic(self, state):
        """按 _serialize_dynamic 输出恢复动态页。"""
        from . import serialization
        st = self.main.static
        store = self.main.groups_store
        # 1) 能级组（store 单一数据源 + 静态页共用区列表 + 画布 + 共用能级表）
        store.groups = [dict(g) for g in state.get('groups', [])]
        st.group_list.clear()
        for g in store.groups:
            pix = QPixmap(12, 12)
            pix.fill(QColor(g['color']))
            st.group_list.addItem(QListWidgetItem(
                QIcon(pix),
                f"{g['label']}  (n={g['n']}, l={g['l']}, j={g['j']})"))
        store.changed.emit()        # → st._on_store_groups_changed（重建+刷新+转发）
        # 2) 磁场 B（超精细基组依赖；设值会触发 on_light → refresh_groups）
        self.main.field.b_spin.setValue(float(state.get('B_gauss', 0.0)))
        self.refresh_groups()
        # 3) 动态参数
        dyn = state.get('dynamic', {})
        self.tmax_sb.setValue(float(dyn.get('t_max', 1500.0)))
        self.npts_sb.setValue(int(dyn.get('npts', 1501)))
        self.method_cb.setCurrentText(dyn.get('method', 'auto'))
        self.backend_cb.setCurrentText(dyn.get('backend', 'auto'))
        self.psi_combo.setCurrentIndex(int(dyn.get('psi', 0)))
        self.topn_sb.setValue(int(dyn.get('topn', 8)))
        # 自发辐射档位（0 自动 / 1 禁用 / 2 自定义；旧存档缺省自动）
        self.decay_mode_cb.setCurrentIndex(int(dyn.get('decay_mode', 0)))
        # 近共振截断（旧存档缺省 5 GHz 地板）
        self.near_cutoff_sb.setValue(float(dyn.get('near_cutoff', 5000.0)))
        # 4) 波形（新格式 'waves'；旧存档 'editor' 不再支持——旧编辑器已归档，
        #    旧存档只恢复能级组/B/动态参数/衰变，波形留空）
        wv = state.get('waves')
        if wv:
            self.wave_panel.set_waves(waveforms.restore_waves(wv))
        # 4b) 衰变通道（按 label 恢复；能级组/B 已先恢复 → 下拉已重建）
        self.decay_cfgs = [dict(c) for c in state.get('decays', [])]
        self._update_decay_list()
        # 5) 波形数据 + 顶部图（解析在 serialization 模块）
        self.wave_data = serialization.restore_wave_data(state.get('wave_data', []))
        self._draw_top(self.wave_data)
        self.main.groups_changed.emit()

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
        waves = list(self.wave_panel.waves)
        fields = waveforms.build_fields(
            waves, states, self.cache,
            near_cutoff_mhz=self.near_cutoff_sb.value())
        t = np.linspace(0, self.tmax_sb.value(), self.npts_sb.value())
        self.wave_data = waveforms.waveforms_to_env(waves, t)
        psi0 = self.psi_combo.currentIndex()
        self._run_states = states            # 供完成后绘图（与计算一致）

        # 进度反馈（延迟显示：计算 >400ms 才弹窗，短计算不打扰；
        # 完全可控，杜绝"空白弹窗闪现"）
        self.progress_arr = np.zeros(2)          # [进度 0..1, 取消标志]
        self._dlg = None
        self._pbar = None
        self._dlg_timer = QTimer(self)
        self._dlg_timer.setSingleShot(True)
        self._dlg_timer.timeout.connect(self._show_progress)
        self._dlg_timer.start(400)
        self._timer = QTimer(self)               # 轮询进度
        self._timer.timeout.connect(self._poll_progress)
        self._timer.start(100)

        self._worker = EvolveWorker(states, fields, psi0, t,
                                     self.method_cb.currentText(),
                                     self.backend_cb.currentText(),
                                     self.progress_arr, decays=decays)
        self._worker.done.connect(self._on_evolve_done)
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
                f'{error}\n\n可尝试：改用 pwc（快速预览）、降低光强、'
                '或调整驱动频率/失谐。')
            return
        self._last_res = res                        # 供布居图内部筛选用
        self._draw_top(self.wave_data)
        states = getattr(self, '_run_states', None)
        if states is not None:
            self._draw_bot(states, res)             # 自动按平均布居取前 N 个态画

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
