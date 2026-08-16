# -*- coding: utf-8 -*-
"""TTL 时序 + 光脉冲 合成编辑器（独立窗口）与动态分析数据源。

布局（QSplitter 上下）：
  - 上半：TTL 时序编辑（复用 scripts/timing_editor 的画布/模型/表格）
  - 下半：波形通道编辑（复用 src/pulse 的轨道画布/可折叠参数面板）

触发绑定：波形通道 i ↔ TTL 通道 i，由该通道第一个（上升）沿触发，
脉冲绝对起点 = 上升沿 t_abs + 脉冲偏移。

主程序（能级.py 动态分析页）引用：
    from src import ttl_pulse, evolve
    win = ttl_pulse.TimingPulseWindow()       # 打开合成编辑器
    win.set_states(states)                    # 主程序基组 → 编辑器驱动下拉
    E_list = ttl_pulse.field_waveforms(win, t_grid)   # 各通道实际电磁场包络
    fields = ttl_pulse.build_fields(win, states)      # 脉冲驱动配置 → evolve.field
"""
import os
import sys
_SC = os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), 'scripts')
if _SC not in sys.path:
    sys.path.insert(0, _SC)

import numpy as np
from PyQt5.QtCore import Qt, QSettings, QPointF, QTimer, pyqtSignal
from PyQt5.QtWidgets import (QMainWindow, QWidget, QSplitter, QVBoxLayout,
                             QHBoxLayout, QToolBar, QAction, QDockWidget,
                             QListWidget, QListWidgetItem, QMenu,
                             QToolButton, QSpinBox, QPushButton, QLabel)

import timing_editor as te
from . import pulse
from . import ui_theme

# 全局主题（与主窗口一致）；独立使用本编辑器时也无需重复定义
_QSS = ui_theme.QSS


def bind_triggers(model, channels):
    """波形通道 i 由 TTL 通道 i 的第一个（上升）沿触发。"""
    for i, wc in enumerate(channels):
        trig = None
        if i < len(model.channels) and model.channels[i].edges:
            trig = model.channels[i].edges[0]   # 上升沿
        for pu in wc.pulses:
            pu.trigger = trig


class TTLPanel(QWidget):
    """上半：TTL 时序编辑。组合 TimingWindow 实例（零复制，联动完整）。
    左侧单开窄通道列：点击定位、右键删除（仿波形通道管理）。"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.host = te.TimingWindow()      # 逻辑宿主（全部联动方法都在此）
        self.host.setParent(self)          # 防 GC（不 show）
        self.model = self.host.model
        self.scene = self.host.scene
        self.view = self.host.takeCentralWidget()     # TimelineView
        dock = self.host.findChild(QDockWidget)       # '时序信息' dock
        self.tables = dock.widget()                   # 沿表 + 关系表容器
        self.host.removeDockWidget(dock)
        # ---- 左侧 TTL 通道列（窄，统一宽度）：字母标号 + 沿数 spin + 右键删除 + 底部添加 ----
        self.channel_col = QWidget()
        self.channel_col.setFixedWidth(ui_theme.CHANNEL_COL_W)
        cc = QVBoxLayout(self.channel_col)
        cc.setContentsMargins(4, 4, 4, 4)
        cc.setSpacing(4)
        cc.addWidget(QLabel('TTL'))
        self._rows_box = QVBoxLayout()          # 每通道一行：字母 + 沿数 spin
        self._rows_box.setSpacing(3)
        cc.addLayout(self._rows_box)
        self.add_btn = QPushButton('＋ 添加通道')
        self.add_btn.setToolTip('添加 TTL 通道（默认一高一低两个沿）')
        self.add_btn.clicked.connect(lambda: self.host._add_ttl())
        cc.addWidget(self.add_btn)
        cc.addStretch(1)
        self._rows = []                          # [(ch, row_widget)]
        self.host.channels_changed.connect(self._sync_rows)
        # TTL 区域：左通道列 + 时序画布 + 沿表/关系表面板
        root = QHBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(2)
        root.addWidget(self.channel_col, 0)
        root.addWidget(self.view, 1)
        root.addWidget(self.tables, 0)
        self.tables.setFixedWidth(340)
        self._sync_rows()

    def _sync_rows(self):
        """重建 TTL 通道行（字母 + 沿数 spin）；host.channels_changed 时调用。"""
        while self._rows_box.count():
            it = self._rows_box.takeAt(0)
            w = it.widget()
            if w is not None:
                w.deleteLater()
        self._rows = []
        for i, ch in enumerate(self.model.channels):
            row = QWidget()
            h = QHBoxLayout(row)
            h.setContentsMargins(0, 0, 0, 0)
            h.setSpacing(3)
            lab = QToolButton()
            lab.setText(chr(ord('A') + i))       # 极简：一个大写字母
            lab.setAutoRaise(True)
            lab.setStyleSheet(
                f'font-weight:bold; font-size:14pt; color:{ch.color};')
            lab.setToolTip(f'{ch.name}：点击定位到该轨道')
            lab.clicked.connect(lambda _, i=i: self._locate(i))
            spin = QSpinBox()
            spin.setRange(1, 20)
            spin.setValue(max(1, len(ch.edges) // 2))
            spin.setToolTip('高电位区间数（沿数 = 2×；每高电平一对 rise+fall）')
            # 右键落在行上（删除菜单），不弹 spin 默认菜单
            spin.setContextMenuPolicy(Qt.PreventContextMenu)
            spin.valueChanged.connect(
                lambda v, ch=ch: QTimer.singleShot(
                    0, lambda: self.host.set_high_count(ch, v)))
            h.addWidget(lab)
            h.addWidget(spin, 1)
            row.setContextMenuPolicy(Qt.CustomContextMenu)
            row.customContextMenuRequested.connect(
                lambda pos, ch=ch, row=row: self._row_menu(pos, ch, row))
            self._rows_box.addWidget(row)
            self._rows.append((ch, row))

    def _row_menu(self, pos, ch, row):
        m = QMenu(self)
        a = m.addAction(f'删除通道 {ch.name}')
        a.triggered.connect(lambda: self.scene._del_channel(ch))
        m.exec_(row.mapToGlobal(pos))

    def _locate(self, i):
        """点击字母 → 画布垂直定位到该轨道（水平保持当前位置）。"""
        if i < 0 or i >= len(self.model.channels):
            return
        vp = self.view.viewport().rect()
        cx = self.view.mapToScene(vp.center()).x()      # 当前视图中心场景 x
        y = self.scene._lane_y(i) + te.ROW_H / 2
        self.view.centerOn(QPointF(cx, y))


class PulsePanel(QWidget):
    """下半：波形通道编辑。组合 PulseEditor 实例（零复制）。"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.host = pulse.PulseEditor()    # 逻辑宿主
        self.host.setParent(self)
        self.channels = self.host.channels
        self.canvas = self.host.takeCentralWidget()   # PulseCanvas（轨道画布）
        dock = self.host.findChild(QDockWidget)       # '脉冲参数' dock
        self.panel = dock.widget()
        self.host.removeDockWidget(dock)
        # 波形区域：左画布右参数面板；面板固定宽（容纳 g/e 长能级描述下拉）
        root = QHBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(2)
        root.addWidget(self.canvas, 1)
        root.addWidget(self.panel, 0)
        self.panel.setFixedWidth(440)


class TimingPulseWindow(QMainWindow):
    """合成编辑器：上 TTL 时序、下波形通道（上下分别布局，独立窗口）。

    数据源角色：主程序（动态分析页）只读上拉；
    import_requested：工具栏"导入波形到主程序"按钮 → 通知主程序立即取数画图。
    窗口几何（位置/大小/分栏比例）用 QSettings 记忆，关闭后重开还原。
    """
    import_requested = pyqtSignal()

    _SETTINGS_KEY = 'editor/geometry'

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle('TTL 时序 + 光脉冲 编辑器')
        self.resize(1180, 760)
        self._live_sync = True               # 实时同步：编辑即刷新主面板波形
        self.ttl = TTLPanel(self)
        self.wave = PulsePanel(self)
        split = QSplitter(Qt.Vertical)
        split.addWidget(self.ttl)
        split.addWidget(self.wave)
        split.setSizes([420, 320])
        self.setCentralWidget(split)
        self.setStyleSheet(_QSS)     # 中性灰阶质感（与主窗口同一主题）
        self._build_toolbar()
        bind_triggers(self.ttl.model, self.wave.channels)
        # 波形参数变化 → 实时同步主面板（脉冲结构/参数变更）
        self.wave.host.changed.connect(self._on_wave_changed)
        # 恢复上次关闭时的几何（位置/大小/分栏比例）
        s = QSettings('EnergyLevel', 'EnergyLevel')
        g = s.value(self._SETTINGS_KEY)
        if g is not None:
            self.restoreGeometry(g)

    def _on_wave_changed(self):
        """波形通道/脉冲参数变化：实时同步开时通知主程序刷新波形预览。"""
        if self._live_sync:
            self.import_requested.emit()

    def closeEvent(self, ev):
        """关闭时记忆几何，供下次打开还原。"""
        s = QSettings('EnergyLevel', 'EnergyLevel')
        s.setValue(self._SETTINGS_KEY, self.saveGeometry())
        super().closeEvent(ev)

    def set_states(self, states, cache=None):
        """主程序把演化基组 + ArcCache 传入（g/e 下拉数据源 / 辅助 Rabi）。"""
        self.wave.host.set_states(states, cache)

    def _build_toolbar(self):
        """顶部工具栏只保留「导出 / 导入」；通道与脉冲的增删已融入画布
        （通道旁 ✕ 删除、底部 ＋ 添加）与右侧面板（＋高斯 / ＋方波）。"""

        def _mk(title):
            tb = QToolBar(title); tb.setMovable(False)
            self.addToolBar(tb)

            def _act(label, fn):
                a = QAction(label, self); a.triggered.connect(fn)
                tb.addAction(a)
            return _act

        # ---- 导出组（总） ----
        actX = _mk('导出')
        actX('导出时序', self.ttl.host._export)
        actX('导出脉冲', self.wave.host._export)
        # ---- 主程序组（导入波形 → 主程序立即画图） ----
        actM = _mk('主程序')
        actM('导入波形到主程序', self.import_requested.emit)
        # 实时同步开关：编辑即刷新主面板波形预览（默认开）
        self._live_act = QAction('实时同步', self)
        self._live_act.setCheckable(True)
        self._live_act.setChecked(self._live_sync)
        self._live_act.toggled.connect(
            lambda on: setattr(self, '_live_sync', on))
        tb = QToolBar('主程序'); tb.setMovable(False)
        tb.addAction(self._live_act)
        self.addToolBar(tb)


# ================= 动态分析数据源 =================
def field_waveforms(win, t_grid):
    """各波形通道**每个脉冲**的光强曲线数据（主页面画光强，不画载波）。

    返回 [(label, t, env, freq_mhz, intensity_mwcm2), ...]——每脉冲一条：
    env 为 0~1 包络，光强曲线 = intensity × env（物理光强）；多脉冲通道
    各自频率/光强独立（颜色=频率映射）。曾整通道只取首脉冲频率 → 颜色错误
    （已修）。采样直接用输出网格 t_grid（包络慢变，无需高采样）。
    """
    t = np.asarray(t_grid, float)
    bind_triggers(win.ttl.model, win.wave.channels)
    out = []
    for wc in win.wave.channels:
        single = len(wc.pulses) == 1
        for k, pu in enumerate(wc.pulses):
            label = wc.name if single else f'{wc.name}#{k + 1}'
            out.append((label, t, np.asarray(pu.waveform(t), float),
                        float(pu.freq_mhz), float(pu.intensity_mwcm2)))
    return out


def build_fields(win, states, cache, near_cutoff_mhz=None):
    """由编辑器脉冲的「真实光场参数」生成演化输入。

    每脉冲 (freq_mhz, intensity_mwcm2, pol, waveform) → evolve.real_field：
    自动按自适应失谐判据展开近共振驱动对（Ω 真实、Δ 相对裸差）+
    远失谐 AC 位移。返回 (fields 列表, None)：
    - 近共振对进各 field 的驱动矩阵（RWA 演化自动产生 dressed 频移）；
    - 远失谐 AC 位移内嵌 field['ac_shift']，随该场包络**时变**（不返回到
      常数向量——常数 ac 全时段生效会破坏其他时段的驱动）。
    near_cutoff_mhz：透传 real_field 的近/远分流边界（协议场景可调小）。
    调用方直接 evolve.evolve(states, fields, ...) 即可，无需 ac_shift_mhz。
    """
    from . import evolve
    bind_triggers(win.ttl.model, win.wave.channels)
    fields = []
    for wc in win.wave.channels:
        for pu in wc.pulses:
            fld, a = evolve.real_field(
                cache, states, pu.freq_mhz, pu.intensity_mwcm2, pu.pol,
                waveform=pu.waveform, near_cutoff_mhz=near_cutoff_mhz)
            if fld is not None:
                fields.append(fld)          # 每脉冲一个多对驱动场（含时变 ac）
            elif np.any(np.asarray(a, float) != 0.0):
                # 纯远失谐（无近共振对）：ac 位移仍要随包络生效 → 零驱动场
                acv = np.asarray(a, float)
                fields.append({
                    'kind': 'multi', 'M': np.diag(acv),
                    'A': np.zeros((len(states), len(states))),
                    'ac_shift': acv,
                    'g_idx': [], 'e_idx': [], 'omega0_mhz': 1.0,
                    'omega_L_mhz': pu.freq_mhz, 'waveform': pu.waveform})
    return fields, None


# ================= 状态管理（编辑器序列化 / 恢复） =================
def serialize_editor(win):
    """合成编辑器（TTL 时序 + 波形通道）→ JSON 可序列化 dict。"""
    ttl = []
    for ch in win.ttl.model.channels:
        edges = []
        for k, e in enumerate(ch.edges):
            d = {'kind': e.kind, 'anchored': e.anchored, 'root_abs': e.root_abs}
            if e.relation is not None:
                p = e.relation.parent
                d.update(parent_ch=(p.owner.index if p.owner is not None else 0),
                         parent_idx=(p.owner.edges.index(p)
                                     if p.owner is not None else k),
                         offset_ns=e.relation.offset_ns)
            edges.append(d)
        ttl.append({'name': ch.name, 'color': ch.color, 'edges': edges})
    wf = []
    for wc in win.ttl.model.wf_channels:
        d = {'name': wc.name, 'color': wc.color, 'wf_type': wc.wf_type,
             'width_ns': wc.width_ns, 'sigma_ns': wc.sigma_ns,
             'offset_ns': wc.offset_ns}
        if wc.trigger is not None:
            d['trigger_ch'] = (wc.trigger.owner.index
                               if wc.trigger.owner is not None else 0)
            d['trigger_idx'] = (wc.trigger.owner.edges.index(wc.trigger)
                                if wc.trigger.owner is not None else 0)
        wf.append(d)
    chans = []
    for ch in win.wave.channels:
        pulses = []
        for pu in ch.pulses:
            env = pu.envelope
            if isinstance(env, pulse.GaussianEnv):
                e = ('gauss', env.center_ns, env.sigma_ns, env.peak)
            else:
                e = ('square', env.start_ns, env.edge_ns, env.height,
                     env.fall_start_ns)
            pulses.append({'freq_mhz': pu.freq_mhz,
                           'intensity': pu.intensity_mwcm2,
                           'pol': pu.pol, 'offset_ns': pu.offset_ns,
                           'env': e, 'drive': pu.drive})
        chans.append({'name': ch.name, 'pulses': pulses})
    return {'ttl': {'channels': ttl, 'wf_channels': wf}, 'channels': chans}


def restore_editor(win, data):
    """按 serialize_editor 输出重建编辑器（TTL model + 波形通道）。"""
    m = win.ttl.model
    m.channels = []
    m.wf_channels = []
    ch_edges = []
    for chs in data['ttl']['channels']:
        ch_edges.append([te.Edge(es['kind'],
                                 anchored=es.get('anchored', False),
                                 root_abs=es.get('root_abs', 0.0))
                         for es in chs['edges']])
    # 关系绑定（父沿可能跨通道，先建全部沿再绑）
    for ci, chs in enumerate(data['ttl']['channels']):
        for k, es in enumerate(chs['edges']):
            pc, pi = es.get('parent_ch'), es.get('parent_idx')
            if pc is not None and 0 <= pc < len(ch_edges) \
                    and 0 <= pi < len(ch_edges[pc]):
                te._with_rel(ch_edges[pc][pi], ch_edges[ci][k],
                             es['offset_ns'])
    for ci, chs in enumerate(data['ttl']['channels']):
        m.channels.append(te.TTLChannel(chs['name'], chs['color'],
                                        ch_edges[ci]))
    for wcs in data['ttl']['wf_channels']:
        trig = None
        if 'trigger_ch' in wcs:
            tc, ti = wcs['trigger_ch'], wcs['trigger_idx']
            if 0 <= tc < len(ch_edges) and 0 <= ti < len(ch_edges[tc]):
                trig = ch_edges[tc][ti]
        wc = te.WfChannel(wcs['name'], wcs['color'], wf_type=wcs['wf_type'],
                          width_ns=wcs['width_ns'], sigma_ns=wcs['sigma_ns'],
                          trigger=trig, offset_ns=wcs['offset_ns'])
        m.wf_channels.append(wc)
    m.solve()
    win.ttl.scene.rebuild()
    # 波形通道 + 脉冲（就地清空：PulsePanel.channels 与 host.channels 同一列表引用）
    ed = win.wave.host
    new_ch = []
    for chs in data['channels']:
        wc = pulse.WaveChannel(chs['name'])
        for ps in chs['pulses']:
            env = ps['env']
            if env[0] == 'gauss':
                env_obj = pulse.GaussianEnv(env[1], env[2], env[3])
            else:
                env_obj = pulse.SmoothSquareEnv(env[1], env[2], env[3], env[4])
            dr = ps.get('drive')
            if dr is not None:
                dr = dict(g=tuple(dr['g']), e=tuple(dr['e']))
            wc.pulses.append(pulse.Pulse(
                ps['freq_mhz'], env_obj, pol=ps['pol'],
                offset_ns=ps['offset_ns'],
                intensity_mwcm2=ps.get('intensity', 100.0), drive=dr))
        new_ch.append(wc)
    ed.channels.clear()
    ed.channels.extend(new_ch)
    ed._cur = 0 if ed.channels else -1
    ed._refresh()
    bind_triggers(win.ttl.model, win.wave.channels)
