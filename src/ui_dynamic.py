# -*- coding: utf-8 -*-
"""动态分析面板（能级.py 新增 tab）：TTL 触发光脉冲 → 实际电磁场 → 态演化。

画布（主程序窗口内，上下）：
  上：各波形通道实际电磁场包络 E_i(t)（ttl_pulse 合成编辑器 + TTL 上升沿触发）
  下：态布居演化 P(t)（evolve.evolve）

参数面板（借鉴时序/波形编辑器 + 按 evolve 需要）：
  初态 · 时间范围/点数 · 演化方法/后端 · 运行
  脉冲驱动（g/e 态、Ω、Δ）在波形编辑器里配置，主程序读取生成演化场
"""
import numpy as np
from PyQt5.QtCore import Qt, QThread, QTimer, pyqtSignal
from PyQt5.QtGui import QColor, QIcon, QPixmap
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel,
                             QPushButton, QSpinBox, QDoubleSpinBox, QComboBox,
                             QListWidgetItem, QMessageBox, QDialog,
                             QProgressBar)
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.colors import LogNorm
from matplotlib import colormaps

from . import core, evolve, ttl_pulse


class _EvolveWorker(QThread):
    """演化计算线程：阻塞式 evolve 放后台，主线程轮询进度/可取消。"""

    done = pyqtSignal(object, object)       # (result_dict, error_str 或 None)
    log_msg = pyqtSignal(str)               # 算法提示/警告 → 主窗口状态栏

    def __init__(self, states, fields, psi0, t_grid, method, backend,
                 progress_arr, parent=None):
        super().__init__(parent)
        self.states = states
        self.fields = fields
        self.psi0 = psi0
        self.t_grid = t_grid
        self.method = method
        self.backend = backend
        self.progress_arr = progress_arr

    def run(self):
        try:
            # ac_shift 已内嵌各 field（随包络时变），无需另传
            res = evolve.evolve(self.states, self.fields, self.psi0,
                                self.t_grid, method=self.method,
                                backend=self.backend,
                                progress_arr=self.progress_arr,
                                log=self.log_msg.emit)
            self.done.emit(res, None)
        except RuntimeError as e:
            # 用户取消：progress_arr[1]≠0 → 不算错误，静默
            if self.progress_arr[1] != 0.0:
                self.done.emit(None, 'cancelled')
            else:
                self.done.emit(None, str(e))
        except Exception as e:
            self.done.emit(None, str(e))


class DynamicPanel(QWidget):
    """动态分析：上光场波形、下态演化曲线。
    能级组共用主程序（右侧面板创建），这里仅按组选 m 子能级 + 特有演化参数。"""

    def __init__(self, main, parent=None):
        super().__init__(parent)
        self.main = main              # 主程序 MainWindow（读 groups 共用能级组）
        self.cache = main.cache
        self.editor = None            # ttl_pulse.TimingPulseWindow
        self.main_groups = []         # 同步自主程序的 (n,l,j) 组
        self.wave_data = []           # 导入的波形数据 [(name, t, env, freq), ...]（主程序管理）
        # 能级组增删由主程序事件驱动刷新（统一机制，不散布同步调用）
        main.groups_changed.connect(self.refresh_groups)
        self._build_canvases()
        self.dynamic_widget = self._build_controls()
        # DynamicPanel 只含画布（上光场、下演化）；特有参数区由主程序放入右侧面板（与单体/两体同级）
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
        self.fig_top.subplots_adjust(left=0.09, right=0.97, top=0.90, bottom=0.15)
        self.canvas_top = FigureCanvas(self.fig_top)
        self.fig_bot = Figure(figsize=(7, 3.6), dpi=100)
        self.ax_bot = self.fig_bot.add_subplot(111)
        self.fig_bot.subplots_adjust(left=0.09, right=0.97, top=0.90, bottom=0.15)
        self.canvas_bot = FigureCanvas(self.fig_bot)
        self.ax_top.set_title('光场包络 E(t)（各波形通道，颜色=频率）')
        self.ax_bot.set_title('态布居 P(t)')
        for ax in (self.ax_top, self.ax_bot):
            ax.set_xlabel('t (ns)'); ax.grid(alpha=0.3)
        self.ax_top.set_ylabel('包络')
        self.ax_bot.set_ylabel('布居')

    # ---------- 布局 ----------
    def _build_controls(self):
        """动态分析特有参数面板（独立控件，由主程序放入右侧共用面板）。"""
        panel = QWidget()
        p = QVBoxLayout(panel)
        p.setSpacing(6)

        self.open_btn = QPushButton('打开 时序/波形 编辑器')
        self.open_btn.clicked.connect(self._open_editor)
        p.addWidget(self.open_btn)

        # ---- 能级：共用主程序"能级管理表"（日常数值 + m 选择），这里只选初态 ----
        row2 = QHBoxLayout()
        row2.addWidget(QLabel('初态（能级/m 见右侧共用表）：'))
        self.psi_combo = QComboBox()
        row2.addWidget(self.psi_combo, 1)
        p.addLayout(row2)

        # ---- 光场驱动配置（已移入波形编辑器：每脉冲 g/e 态、Ω、Δ）----
        p.addWidget(QLabel('脉冲驱动（g/e 态、Ω、Δ）在波形编辑器里配置；'
                           '主程序读取编辑器驱动生成演化场'))

        # ---- 布居图：只设"显示几个态"，具体哪些态内部按平均布居自动筛选 ----
        rowp = QHBoxLayout()
        rowp.addWidget(QLabel('布居图显示态数'))
        self.topn_sb = QSpinBox()
        self.topn_sb.setRange(1, 40)
        self.topn_sb.setValue(8)
        self.topn_sb.setToolTip('按最近一次结果的平均布居取前 N 个态绘制'
                                '（无需手动勾选，内部自动筛选）')
        rowp.addWidget(self.topn_sb)
        p.addLayout(rowp)

        # ---- 时间 / 方法 ----
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
        p.addLayout(rowt)
        rowm = QHBoxLayout()
        rowm.addWidget(QLabel('方法'))
        self.method_cb = QComboBox()
        self.method_cb.addItems(['auto', 'floquet', 'ip4', 'pwc'])
        rowm.addWidget(self.method_cb)
        rowm.addWidget(QLabel('后端'))
        self.backend_cb = QComboBox()
        self.backend_cb.addItems(['auto', 'cpu', 'gpu'])
        rowm.addWidget(self.backend_cb)
        p.addLayout(rowm)

        self.run_btn = QPushButton('运行演化')
        self.run_btn.clicked.connect(self._run)
        p.addWidget(self.run_btn)
        p.addStretch(1)
        # 直接作为主程序右侧面板的普通区块（无独立滚动区，融入同一层级）
        return panel

    # ---------- 能级：从共用能级组生成（切页/运行时刷新，数据共享不散布同步） ----------
    def refresh_groups(self):
        """能级组与 m 选择共用主程序"能级管理表"（单一数据源），这里只刷新初态/驱动。"""
        self.main_groups = list(self.main.groups)
        self._on_m_changed()

    def _on_m_changed(self, *_):
        """基组变化 → 刷新初态下拉（保留当前选择）+ 布居态列表 + 编辑器驱动下拉。"""
        states = self._current_states()
        cur = self.psi_combo.currentText()          # 记录选择，避免运行后刷回第一个
        self.psi_combo.clear()
        for s in states:
            self.psi_combo.addItem(s['label'])
        if cur:
            idx = self.psi_combo.findText(cur)
            if idx >= 0:
                self.psi_combo.setCurrentIndex(idx)
        if self.editor is not None:
            self.editor.set_states(states, self.cache)

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

    # ---------- 编辑器 ----------
    def _ensure_editor(self, show=False):
        """创建（若未创建）时序/波形编辑器；show=True 才显示窗口。"""
        if self.editor is None:
            self.editor = ttl_pulse.TimingPulseWindow(self.main)
            self.editor.import_requested.connect(self.import_wave)
        if show:
            self.editor.show(); self.editor.raise_()
        return self.editor

    def _open_editor(self):
        """打开时序/波形编辑器（独立窗口，父窗口 = 主窗口 → 生命周期联动；
        位置/大小由编辑器自身 QSettings 记忆）。"""
        self._ensure_editor(show=True)
        self.editor.set_states(self._current_states(), self.cache)

    def _has_pulses(self):
        """是否已配置至少一个**带驱动对（g/e）**的光脉冲。
        （编辑器自带的示例脉冲 drive=None，不算用户配置。）"""
        if self.editor is None:
            return False
        for wc in self.editor.wave.channels:
            for pu in wc.pulses:
                if pu.drive is not None:
                    return True
        return False

    def import_wave(self):
        """导入编辑器波形到主程序并立即画图（不等运行演化）。
        数据存 self.wave_data（主程序管理：导入/运行都会更新，画图用它）。"""
        if self.editor is None:
            return
        t = np.linspace(0, self.tmax_sb.value(), self.npts_sb.value())
        self.wave_data = ttl_pulse.field_waveforms(self.editor, t)
        self._draw_top(self.wave_data)

    # ---------- 状态管理（序列化/恢复；文件 IO 与菜单入口在主程序） ----------
    def _serialize_dynamic(self):
        """动态页全部参数与数据 → dict（含共用能级组、场 B、编辑器、波形数据）。"""
        return {
            'groups': [dict(g) for g in self.main.groups],
            'B_gauss': self.main.field.b_gauss(),
            'dynamic': {
                'psi': self.psi_combo.currentIndex(),
                't_max': self.tmax_sb.value(),
                'npts': self.npts_sb.value(),
                'method': self.method_cb.currentText(),
                'backend': self.backend_cb.currentText(),
                'topn': self.topn_sb.value(),
            },
            'editor': (ttl_pulse.serialize_editor(self.editor)
                       if self.editor is not None else None),
            'wave_data': [
                {'name': nm,
                 't': np.asarray(th).tolist(),
                 'env': np.asarray(env).tolist(),
                 'freq': float(f)}
                for nm, th, env, f in self.wave_data],
        }

    def _restore_dynamic(self, state):
        """按 _serialize_dynamic 输出恢复动态页。"""
        # 1) 能级组（共用区列表 + 画布 + 共用能级表）
        self.main.groups = []
        self.main.group_list.clear()
        for g in state.get('groups', []):
            self.main.groups.append(dict(g))
            pix = QPixmap(12, 12)
            pix.fill(QColor(g['color']))
            self.main.group_list.addItem(QListWidgetItem(
                QIcon(pix),
                f"{g['label']}  (n={g['n']}, l={g['l']}, j={g['j']})"))
        self.main._rebuild_y_combos()
        self.main.refresh_plot()
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
        # 4) 编辑器（TTL + 波形）——静默创建（不弹窗），恢复配置
        ed_state = state.get('editor')
        if ed_state is not None:
            self._ensure_editor(show=False)
            ttl_pulse.restore_editor(self.editor, ed_state)
            self.editor.set_states(self._current_states(), self.cache)
        # 5) 波形数据 + 顶部图（兼容旧存档：旧字段 E=载波，env 缺省时退回 E）
        wd = state.get('wave_data', [])
        self.wave_data = [
            (d['name'], np.array(d['t']),
             np.array(d.get('env', d.get('E', 0.0))),
             float(d.get('freq', 0.0)))
            for d in wd]
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
        # 未配置脉冲 → 明确提示并打开编辑器，不静默跑空演化
        if not self._has_pulses():
            QMessageBox.information(
                self, '提示',
                '请先在 时序/波形 编辑器中配置光脉冲（频率/光强/包络），'
                '再运行演化。\n已为你打开编辑器。')
            self._open_editor()
            return
        states = self._current_states()
        fields, _ = ttl_pulse.build_fields(self.editor, states, self.cache)
        t = np.linspace(0, self.tmax_sb.value(), self.npts_sb.value())
        self.wave_data = ttl_pulse.field_waveforms(self.editor, t)   # 取最新波形
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

        self._worker = _EvolveWorker(states, fields, psi0, t,
                                     self.method_cb.currentText(),
                                     self.backend_cb.currentText(),
                                     self.progress_arr)
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
        """顶部：各通道**包络**（不画载波）——颜色按频率 log 映射（半透明下面积）。

        E_list: [(name, t, env, freq_mhz), ...]（field_waveforms 输出）。
        """
        ax = self.ax_top; ax.clear()
        ax.set_title('光场包络 E(t)（颜色=频率，对数尺度）')
        ax.set_xlabel('t (ns)'); ax.set_ylabel('包络'); ax.grid(alpha=0.3)
        if E_list:
            freqs = np.array([f for _, _, _, f in E_list], float)
            fmin, fmax = max(freqs.min(), 1e-3), max(freqs.max(), 1e-3)
            if fmax <= fmin:
                fmax = fmin + 1.0
            norm = LogNorm(vmin=fmin, vmax=fmax)
            cmap = colormaps['plasma']
            for (name, t, env, f), c in zip(
                    E_list, cmap(norm(freqs))):
                ax.plot(t, env, lw=1.5, color=c, label=name)
                ax.fill_between(t, env, 0.0, color=c, alpha=0.25)
            ax.legend(fontsize=8)
        self.canvas_top.draw_idle()

    def _draw_bot(self, states, res):
        """底部：态布居（只画前 N 个平均布居最大的态，N=topn_sb）。"""
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
        ax.legend(fontsize=8)
        self.canvas_bot.draw_idle()
