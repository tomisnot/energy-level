# -*- coding: utf-8 -*-
"""TTL 触发光脉冲模块（独立于 TTL 编辑面板）。

简洁原则：一个脉冲 = 频率 + 振幅调制形状 + 偏振。
振幅调制形状总共只有两种：

  - GaussianEnv     高斯包络：中心 center_ns、展宽 sigma_ns、峰值 peak
  - SmoothSquareEnv 缓变方波：开始上升 start_ns、沿时长 edge_ns、
                      目标高度 height、开始下降 fall_start_ns

Pulse 可绑定一个 TTL 沿（任意带 ``.t_abs`` 的对象，如 timing_editor.Edge）
作为播放触发；``waveform()`` 返回 evolve.field 兼容的包络 callable（t→[0,1]），
可直接送入 evolve.evolve 做含时演化。

用法::

    from src import pulse, evolve
    p = pulse.Pulse(6.8e6, pulse.GaussianEnv(400.0, 60.0),
                    pol='π', trigger=some_edge, offset_ns=0.0)
    fields = [p.to_field(1, 2, omega0_mhz=20.0)]
"""
import numpy as np
from . import evolve
from . import core as _core
from . import ui_theme

# 偏振约定（与 evolve.group_field 的 _POL_Q 一致）
POLS = ('π', 'σ+', 'σ-')


def _smart_unit(value, units):
    """智能选单位：显示值 ∈ [1, 1000) 的最小单位（大数用大单位，小数用小单位）。

    units = [(名, 系数 MHz/单位 或 mW/单位, decimals), ...] 按系数降序。
    value ≤ 0 时返回最小单位（最大系数，如 THz / W）。
    """
    for i, (_, coef, _) in enumerate(units):
        v = value / coef
        if 1.0 <= v < 1000.0:
            return i
    return 0 if value > 0 else len(units) - 1


def _smooth(x):
    """x∈[0,1] → 0..1 余弦缓变（raised cosine，两端导数连续=0）。"""
    x = np.clip(x, 0.0, 1.0)
    return 0.5 * (1.0 - np.cos(np.pi * x))


class GaussianEnv:
    """高斯振幅包络：peak·exp(−(τ−center)²/2σ²)，τ 以脉冲起点为 0。"""
    def __init__(self, center_ns, sigma_ns, peak=1.0):
        self.center_ns = float(center_ns)
        self.sigma_ns = float(sigma_ns)
        self.peak = float(peak)
        assert self.sigma_ns > 0, '展宽须为正'
        assert 0 < self.peak <= 1.0, '峰值须在 (0,1]'

    def __call__(self, tau):
        tau = np.asarray(tau, float)
        return self.peak * np.exp(-(tau - self.center_ns) ** 2
                                  / (2.0 * self.sigma_ns ** 2))

    def __repr__(self):
        return (f'GaussianEnv(center={self.center_ns}ns, '
                f'sigma={self.sigma_ns}ns, peak={self.peak})')


class SmoothSquareEnv:
    """缓变方波包络：start→(start+edge) 余弦上升、平台到 fall_start、
    (fall_start)→(fall_start+edge) 余弦下降，目标高度 height。
    fall_start 缺省 = start + 2·edge（两沿相接的最小脉冲）。"""
    def __init__(self, start_ns, edge_ns, height=1.0, fall_start_ns=None):
        self.start_ns = float(start_ns)
        self.edge_ns = float(edge_ns)
        self.height = float(height)
        self.fall_start_ns = (float(fall_start_ns)
                              if fall_start_ns is not None
                              else self.start_ns + 2 * self.edge_ns)
        assert self.edge_ns >= 0, '沿时长非负'
        assert self.fall_start_ns >= self.start_ns, '下降位置须在上升之后'
        assert 0 < self.height <= 1.0, '目标高度须在 (0,1]'

    def __call__(self, tau):
        t = np.asarray(tau, float)
        if self.edge_ns <= 0:                    # 退化：理想方波
            return np.where((t >= self.start_ns) & (t <= self.fall_start_ns),
                            self.height, 0.0)
        rise = _smooth((t - self.start_ns) / self.edge_ns)
        fall = 1.0 - _smooth((t - self.fall_start_ns) / self.edge_ns)
        env = self.height * rise * fall
        return np.where(t < self.start_ns, 0.0,
                        np.where(t > self.fall_start_ns + self.edge_ns,
                                 0.0, env))

    def __repr__(self):
        return (f'SmoothSquareEnv(start={self.start_ns}ns, '
                f'edge={self.edge_ns}ns, height={self.height}, '
                f'fall_start={self.fall_start_ns}ns)')


class Pulse:
    """一个由 TTL 触发的光脉冲：频率 + 包络形状 + 偏振。

    trigger：任意带 ``.t_abs`` 的对象（如 timing_editor 的沿），
             或 None（绝对起点 = offset_ns）。
    """
    def __init__(self, freq_mhz, envelope, pol='π',
                 trigger=None, offset_ns=0.0, drive=None, intensity_mwcm2=100.0):
        assert pol in POLS, f'偏振须为 {POLS}'
        self.freq_mhz = float(freq_mhz)
        self.envelope = envelope
        self.pol = pol
        self.trigger = trigger
        self.offset_ns = float(offset_ns)
        self.intensity_mwcm2 = float(intensity_mwcm2)   # 真实参数：光强
        # 辅助驱动对（仅用于辅助定频率/光强；不参与演化结算）
        self.drive = drive   # dict(g=spec, e=spec)

    # ---------- 定位 ----------
    def start(self):
        """脉冲绝对起点（ns）：TTL 沿时刻 + 偏移。"""
        if self.trigger is None:
            return self.offset_ns
        return float(self.trigger.t_abs) + self.offset_ns

    def end(self):
        """脉冲结束绝对时间（包络≈0 的粗略上界）。"""
        e = self.envelope
        if isinstance(e, GaussianEnv):
            return self.start() + e.center_ns + 5 * e.sigma_ns
        return self.start() + e.fall_start_ns + e.edge_ns

    # ---------- 波形（evolve.field 兼容） ----------
    def waveform(self, t):
        """包络 callable：t→[0,1]，已按 TTL 触发做绝对定位。"""
        t = np.asarray(t, float)
        return self.envelope(t - self.start())

    # ---------- evolve 集成 ----------
    def to_field(self, g, e, omega0_mhz, detune_mhz=0.0):
        """生成 evolve.field（单对驱动），峰值 Rabi = omega0_mhz·包络峰值。"""
        return evolve.field(g, e, omega0_mhz=omega0_mhz,
                            waveform=self.waveform,
                            detune_mhz=detune_mhz)

    def __repr__(self):
        tr = getattr(self.trigger, 'desc', None)
        if tr is None:
            tr = f'@{self.start():.1f}ns' if self.trigger is None \
                else str(self.trigger)
        return (f'Pulse(freq={self.freq_mhz:.6g}MHz, pol={self.pol}, '
                f'trigger={tr}{self.offset_ns:+.0f}ns, {self.envelope})')


class WaveChannel:
    """波形通道（两级管理第一级）：与 TTL 通道一一对应，内部管理多个脉冲。"""
    def __init__(self, name, pulses=None):
        self.name = name
        self.pulses = list(pulses or [])

    def __repr__(self):
        return f'WaveChannel({self.name!r}, {len(self.pulses)} pulses)'


# ================= 独立 UI：TTL 触发光脉冲编辑器（自己维护脉冲列表） =================
try:
    import sys as _sys
    from PyQt5.QtWidgets import (
        QApplication, QMainWindow, QWidget, QDialog, QVBoxLayout, QHBoxLayout,
        QFormLayout, QDoubleSpinBox, QComboBox, QPushButton, QScrollArea,
        QToolButton, QToolBar, QAction, QTextEdit, QDockWidget, QLabel,
        QSplitter, QListWidget, QListWidgetItem, QAbstractItemView,
        QSizePolicy, QMenu)
    from PyQt5.QtCore import Qt, QRect, pyqtSignal
    from PyQt5.QtGui import QPainter, QColor, QFont, QPen, QPainterPath
    _UI = True
except ImportError:
    _UI = False

_PAL = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6',
        '#1abc9c', '#e67e22', '#16a085', '#c0392b', '#2980b9']


if _UI:

    class PulseCanvas(QWidget):
        """多通道轨道画布：每通道一行轨道，同通道脉冲曲线叠加；点击轨道选中通道。
        通道增删/选中在左侧通道列完成，画布保持干净。"""
        def __init__(self, editor):
            super().__init__()
            self.editor = editor
            self.setMinimumSize(480, 200)

        def _t_max(self):
            t = 1000.0
            for ch in self.editor.channels:
                for pu in ch.pulses:
                    t = max(t, pu.end())
            return t * 1.05

        def _channel_at(self, y):
            chs = self.editor.channels
            if not chs:
                return None
            mT, mB = 8, 22
            th = (self.height() - mT - mB) / len(chs)
            i = int((y - mT) / th)
            return i if 0 <= i < len(chs) else None

        def mousePressEvent(self, ev):
            chs = self.editor.channels
            i = self._channel_at(ev.y())
            if i is not None and i != self.editor._cur:
                self.editor._select_channel(i)
                ev.accept()
                return
            super().mousePressEvent(ev)

        def paintEvent(self, ev):
            p = QPainter(self)
            p.fillRect(self.rect(), QColor('white'))
            chs = self.editor.channels
            if not chs:
                p.setPen(QColor('#999'))
                p.drawText(self.rect(), Qt.AlignCenter,
                           '尚无波形通道：左侧「＋ 通道」添加')
                return
            W, H = self.width(), self.height()
            mL, mR, mT, mB = 12, 16, 8, 22
            n = len(chs)
            th = (H - mT - mB) / n
            pw = W - mL - mR
            t_max = self._t_max()
            for i, ch in enumerate(chs):
                y0 = mT + i * th
                sel = i == self.editor._cur
                p.fillRect(QRect(mL, int(y0), pw, int(th) + 1),
                           QColor('#e3f0fc' if sel else '#fafafa'))
                base = y0 + th * 0.80
                amp = th * 0.55
                for k, pu in enumerate(ch.pulses):
                    p.setPen(QPen(QColor(_PAL[k % len(_PAL)]), 2))
                    path = QPainterPath()
                    ts = np.linspace(0, t_max, 300)
                    ys = pu.waveform(ts)
                    for m, (t, y) in enumerate(zip(ts, ys)):
                        x = mL + t / t_max * pw
                        yy = base - float(y) * amp
                        if m == 0:
                            path.moveTo(x, yy)
                        else:
                            path.lineTo(x, yy)
                    p.drawPath(path)
                p.setPen(QPen(QColor('#ccc'), 1))
                p.drawLine(mL, int(y0 + th), W - mR, int(y0 + th))
            # 全局时间轴
            p.setPen(QColor('#555'))
            p.drawLine(mL, H - mB, W - mR, H - mB)
            p.setFont(QFont('Sans Serif', 9))
            for k in range(5):
                t = t_max * k / 4
                x = mL + t / t_max * pw
                p.drawLine(int(x), H - mB, int(x), H - mB + 5)
                p.drawText(QRect(int(x) - 40, H - mB + 6, 80, 14),
                           Qt.AlignCenter, f'{t:.0f}ns')

    class PulseEditor(QMainWindow):
        """波形编辑器（两级管理）：左侧通道列（增删/选中）+ 画布多通道轨道；
        右侧 QToolBox 每脉冲一个可折叠参数页（参数裸露、即时生效）。
        changed：任何参数/结构变化时发出（供主窗口实时同步）。"""
        changed = pyqtSignal()

        def __init__(self):
            super().__init__()
            self.setWindowTitle('TTL 触发光脉冲编辑器')
            self.resize(1180, 700)
            self.channels = []          # 第一级：波形通道
            self.states = []            # 主程序演化基组（驱动 g/e 下拉数据源）
            self.cache = None           # 主程序 ArcCache（辅助 Rabi 计算）
            self._cur = -1              # 当前通道索引
            self.canvas = PulseCanvas(self)
            # 中央：左侧通道列（列表 + 右键增删 + 底部添加按钮）+ 轨道画布
            self.channel_list = QListWidget()
            self.channel_list.setMinimumWidth(0)   # 填满列宽（不按内容撑）
            self.channel_list.currentRowChanged.connect(self._list_select)
            self.channel_list.setContextMenuPolicy(
                Qt.CustomContextMenu)
            self.channel_list.customContextMenuRequested.connect(
                self._list_menu)
            self.channel_list.setToolTip(
                '波形通道：点击选中；右键添加/删除')
            col = QWidget()
            col.setFixedWidth(ui_theme.CHANNEL_COL_W)   # 与 TTL 左列统一
            cl = QVBoxLayout(col)
            cl.setContentsMargins(4, 4, 4, 4)
            cl.setSpacing(4)
            cl.addWidget(QLabel('通道'))
            cl.addWidget(self.channel_list, 1)
            add_b = QPushButton('＋ 添加通道')
            add_b.setToolTip('添加波形通道（删除在列表右键）')
            add_b.clicked.connect(self._add_channel)
            cl.addWidget(add_b)
            split = QSplitter(Qt.Horizontal)
            split.addWidget(col)
            split.addWidget(self.canvas)
            split.setStretchFactor(0, 0)
            split.setStretchFactor(1, 1)
            split.setSizes([100, 800])
            self.setCentralWidget(split)
            self._build_panel()
            self._build_toolbar()
            # 默认一个通道 + 两个示例脉冲
            self._add_channel()
            ch = self._current()
            ch.pulses.append(Pulse(6.8e6, GaussianEnv(400.0, 60.0), pol='π'))
            ch.pulses.append(Pulse(6.9e6, SmoothSquareEnv(
                200.0, 40.0, fall_start_ns=800.0), pol='σ+'))
            self._refresh()

        # ---------- 右侧面板：滚动区 + 每脉冲一个可折叠参数组（箭头开合） ----------
        def _build_panel(self):
            dock = QDockWidget('脉冲参数（当前通道）', self)
            self._panel_scroll = QScrollArea()
            self._panel_scroll.setWidgetResizable(True)
            self._panel_scroll.setMinimumWidth(320)
            host = QWidget()
            self._panel_box = QVBoxLayout(host)
            self._panel_box.setContentsMargins(6, 6, 6, 6)
            self._panel_box.setSpacing(6)
            self._panel_scroll.setWidget(host)
            dock.setWidget(self._panel_scroll)
            self.addDockWidget(Qt.RightDockWidgetArea, dock)

        def _current(self):
            return (self.channels[self._cur]
                    if 0 <= self._cur < len(self.channels) else None)

        def _select_channel(self, i):
            self._cur = i
            # 同步左侧通道列表（防递归：blockSignals）
            self.channel_list.blockSignals(True)
            self.channel_list.setCurrentRow(i)
            self.channel_list.blockSignals(False)
            self._refresh()

        def _list_select(self, i):
            if i >= 0 and i != self._cur:
                self._select_channel(i)

        def _list_menu(self, pos):
            """通道列右键：添加通道 / 删除该通道。"""
            m = QMenu(self)
            m.addAction('＋ 添加通道').triggered.connect(self._add_channel)
            it = self.channel_list.itemAt(pos)
            if it is not None:
                m.addAction('－ 删除该通道').triggered.connect(
                    self._del_channel)
            m.exec_(self.channel_list.mapToGlobal(pos))

        def _sync_channel_list(self):
            """重建左侧通道列表（保持 _cur 选中）。"""
            self.channel_list.blockSignals(True)
            self.channel_list.clear()
            for i, ch in enumerate(self.channels):
                it = QListWidgetItem(f'{ch.name} · {len(ch.pulses)} 脉冲')
                if i == self._cur:
                    it.setSelected(True)
                self.channel_list.addItem(it)
            self.channel_list.blockSignals(False)

        def set_states(self, states, cache=None):
            """主程序把演化基组能级 + ArcCache 传入（g/e 下拉数据源 / 辅助 Rabi）。"""
            self.states = list(states)
            self.cache = cache
            self._refresh()

        def _add_channel(self, name=None):
            name = name or f'波形{len(self.channels)}'
            self.channels.append(WaveChannel(name))
            self._cur = len(self.channels) - 1
            self._refresh()
            self.changed.emit()

        def _del_channel(self):
            if not self.channels:
                return
            self.channels.pop(self._cur
                              if 0 <= self._cur < len(self.channels) else -1)
            self._cur = min(self._cur, len(self.channels) - 1)
            self._refresh()
            self.changed.emit()

        def _del_channel_at(self, i):
            """删除第 i 通道（列表右键用）。"""
            if 0 <= i < len(self.channels):
                self.channels.pop(i)
                self._cur = min(self._cur, len(self.channels) - 1)
                self._refresh()
                self.changed.emit()

        def _build_toolbar(self):
            tb = QToolBar('脉冲'); tb.setMovable(False)
            self.addToolBar(tb)
            for label, fn in (
                    ('＋通道', self._add_channel),
                    ('－通道', self._del_channel),
                    ('＋高斯', lambda: self._add_pulse(GaussianEnv(400, 60))),
                    ('＋方波', lambda: self._add_pulse(SmoothSquareEnv(
                        200, 40, fall_start_ns=800))),
                    ('－脉冲', self._delete),
                    ('导出代码', self._export)):
                a = QAction(label, self)
                a.triggered.connect(fn); tb.addAction(a)

        # ---------- 脉冲（第二级，作用于当前通道） ----------
        def _add_pulse(self, env, freq_mhz=6.8e6, pol='π'):
            ch = self._current()
            if ch is None:
                return
            ch.pulses.append(Pulse(freq_mhz, env, pol=pol))
            self._refresh()
            self.changed.emit()

        def _delete(self):
            ch = self._current()
            if ch is not None and ch.pulses:
                ch.pulses.pop()
                self._refresh()
                self.changed.emit()

        def _pulse_title(self, pu, idx):
            e = pu.envelope
            kind = '高斯' if isinstance(e, GaussianEnv) else '方波'
            return f'p{idx} {kind} · {pu.freq_mhz:.3g}MHz · {pu.pol}'

        def _pulse_content(self, pu, idx, head):
            """单个脉冲的裸露参数区（每参数一行 spinbox，编辑即时生效）。"""
            w = QWidget()
            f = QFormLayout(w)
            f.setContentsMargins(10, 8, 10, 8)
            f.setSpacing(5)
            # 字段列横向拉伸：参数行横跨整个面板宽度
            f.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)

            def _spin(lo, hi, dec, val, step=1.0, suffix=''):
                sb = QDoubleSpinBox()
                sb.setRange(lo, hi); sb.setDecimals(dec)
                sb.setSingleStep(step); sb.setValue(val)
                if suffix:
                    sb.setSuffix(suffix)
                # 最小宽 + 可拉伸：字段列随面板宽度填充（空间充分利用）
                sb.setMinimumWidth(110)
                sb.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
                return sb

            # 频率 + 单位档（THz/GHz/MHz/kHz/Hz；内部统一存 MHz）。
            # 精度：decimals 保证该单位下 1 kHz 绝对精度（log10(coef)+3），
            # 单位切换纯显示层（内部 double 全程不丢位）；
            # 智能默认单位选显示值 ∈ [1,1000)，微波→GHz、光频→THz。
            _FREQ_UNITS = [('THz', 1e6, 9), ('GHz', 1e3, 6),
                           ('MHz', 1.0, 3), ('kHz', 1e-3, 0),
                           ('Hz', 1e-6, 0)]   # (名, MHz/单位, decimals)
            _fi = _smart_unit(pu.freq_mhz, _FREQ_UNITS)
            freq = _spin(0, 1e9, _FREQ_UNITS[_fi][2],
                         pu.freq_mhz / _FREQ_UNITS[_fi][1], 1e-3)
            freq.setSingleStep(1.0 / _FREQ_UNITS[_fi][1])  # 当前单位下 1 MHz 步进
            freq.setToolTip('脉冲载频；可切换 THz/GHz/MHz/kHz/Hz 单位\n'
                            '内部 MHz 全精度（double），切换单位不丢位')
            freq_unit = QComboBox()
            freq_unit.setFixedWidth(70)
            for name, _, _ in _FREQ_UNITS:
                freq_unit.addItem(name)
            freq_unit.setCurrentIndex(_fi)
            row_f = QHBoxLayout()
            row_f.addWidget(freq, 1)
            row_f.addWidget(freq_unit)
            f.addRow('频率', row_f)

            # 光强 + 单位档（W/mW/μW/nW per cm²；内部统一存 mW/cm²）。
            # 精度：每单位 6 位小数（显示值 ∈[1,1000) 时绝对精度 ≤1e-6 当前
            # 单位，最小 nW 档 = 1e-12 mW/cm²，远低于任何实验不确定度）。
            _INTEN_UNITS = [('W/cm²', 1e3, 6), ('mW/cm²', 1.0, 6),
                            ('μW/cm²', 1e-3, 6), ('nW/cm²', 1e-6, 6)]
            _ii = _smart_unit(pu.intensity_mwcm2, _INTEN_UNITS)
            inten = _spin(0, 1e9, _INTEN_UNITS[_ii][2],
                          pu.intensity_mwcm2 / _INTEN_UNITS[_ii][1], 10.0)
            inten.setToolTip('光强：与频率共同决定真实驱动强度（Rabi），结算用；\n'
                             '可切换 W/mW/μW/nW 单位（内部 mW/cm² 全精度）')
            inten_unit = QComboBox()
            inten_unit.setFixedWidth(70)
            for name, _, _ in _INTEN_UNITS:
                inten_unit.addItem(name)
            inten_unit.setCurrentIndex(_ii)
            row_i = QHBoxLayout()
            row_i.addWidget(inten, 1)
            row_i.addWidget(inten_unit)
            f.addRow('光强', row_i)
            pol = QComboBox()
            pol.addItems(list(POLS)); pol.setCurrentText(pu.pol)
            f.addRow('偏振', pol)
            e = pu.envelope
            if isinstance(e, GaussianEnv):
                spins = [('中心', _spin(0, 1e7, 1, e.center_ns, 10, ' ns')),
                         ('展宽 σ(半径)', _spin(1e-3, 1e6, 1, e.sigma_ns, 1,
                                               ' ns')),
                         ('峰值', _spin(0, 1, 3, e.peak, 0.1))]
                spins[1][1].setToolTip(
                    '高斯标准差 σ（半径量纲，从中心到拐点）；'
                    '半高全宽 FWHM ≈ 2.355σ')
            else:
                spins = [('开始上升', _spin(0, 1e7, 1, e.start_ns, 10, ' ns')),
                         ('沿时长', _spin(0, 1e6, 1, e.edge_ns, 1, ' ns')),
                         ('目标高度', _spin(0, 1, 3, e.height, 0.1)),
                         ('开始下降', _spin(0, 1e7, 1, e.fall_start_ns,
                                          10, ' ns'))]
            for lab, sb in spins:
                f.addRow(lab, sb)
            off = _spin(-1e9, 1e9, 1, pu.offset_ns, 10, ' ns')
            f.addRow('触发偏移', off)

            # ---- 辅助（可折叠：g/e 对帮定真实参数；结算用 频率+光强+偏振）----
            drive_head = QToolButton()
            drive_head.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
            drive_head.setCheckable(True)
            drive_head.setChecked(True)
            drive_head.setText('辅助')
            drive_head.setArrowType(Qt.DownArrow)
            drive_head.setStyleSheet(
                'QToolButton{background:#f0f3f7;border:1px solid #d5dae2;'
                'border-radius:4px;padding:3px 8px;text-align:left;'
                'font-weight:bold;color:#4a5568;}')
            f.addRow(drive_head)
            body = QWidget()
            bf = QFormLayout(body)
            bf.setContentsMargins(10, 4, 4, 4)
            bf.setSpacing(4)
            bf.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)

            def _spec_of(s):
                if s.get('kind') == 'F':
                    return ('F', s['n'], s['l'], s['j'], s['F'], s['mF'])
                return ('mJ', s['n'], s['l'], s['j'], s['m'])

            def _drow(s):
                return dict(n=s['n'], l=s['l'], j=s['j'], kind=s['kind'],
                            mj=s.get('mj'), mF=s.get('mF'), amps=s.get('amps'))

            g_combo = QComboBox()
            g_combo.setMaxVisibleItems(14)     # 长标签下拉展开时限制高度
            g_combo.setMinimumWidth(260)       # 尽量宽：完整显示能级描述
            g_combo.addItem('— 未配置 —', None)
            for s in self.states:
                g_combo.addItem(s['label'], _spec_of(s))
            bf.addRow('g 态', g_combo)
            e_combo = QComboBox()
            e_combo.setMaxVisibleItems(14)
            e_combo.setMinimumWidth(260)
            e_combo.addItem('— 未配置 —', None)
            for s in self.states:
                e_combo.addItem(s['label'], _spec_of(s))
            bf.addRow('e 态', e_combo)
            hint = QLabel('未配置')
            hint.setStyleSheet(
                'font-family: Consolas; font-size: 9pt; color: #0b5394;')
            bf.addRow('真实反馈', hint)
            # 目标 Ω + 「定光强」同行（省一行、利用标签列右侧空间）
            tgt_om = _spin(0, 1e5, 4, 1.0, 1.0, ' MHz')
            setI_b = QPushButton('定光强')
            setI_b.setToolTip('反推光强使该对 Rabi = 目标 Ω，写回上方"光强"')
            row_om = QHBoxLayout()
            row_om.addWidget(tgt_om, 1)
            row_om.addWidget(setI_b)
            bf.addRow('目标 Ω', row_om)
            # 目标 Δ + 「定频率」同行
            tgt_det = _spin(-1e6, 1e6, 3, 0.0, 1.0, ' MHz')
            tgt_det.setToolTip('失谐相对裸能级差（不含能移），无歧义')
            setF_b = QPushButton('定频率')
            setF_b.setToolTip('频率 = 裸差 + Δ，写回上方"频率"')
            row_det = QHBoxLayout()
            row_det.addWidget(tgt_det, 1)
            row_det.addWidget(setF_b)
            bf.addRow('目标 Δ', row_det)
            f.addRow(body)

            def _toggle_drive(on):
                body.setVisible(on)
                drive_head.setArrowType(Qt.DownArrow if on else Qt.RightArrow)
            drive_head.toggled.connect(_toggle_drive)

            def _find_state(spec):
                for i, s in enumerate(self.states):
                    if _spec_of(s) == spec:
                        return i + 1
                return 0

            if pu.drive is not None:
                g_combo.setCurrentIndex(_find_state(pu.drive['g']))
                e_combo.setCurrentIndex(_find_state(pu.drive['e']))

            def _freq_unit_coef():
                return _FREQ_UNITS[freq_unit.currentIndex()][1]

            def _inten_unit_coef():
                return _INTEN_UNITS[inten_unit.currentIndex()][1]

            def _apply_freq_unit():
                """按当前频率单位同步 spin 的 decimals/步进（不改变内部值）。"""
                dec = _FREQ_UNITS[freq_unit.currentIndex()][2]
                freq.setDecimals(dec)
                freq.setSingleStep(1.0 / _freq_unit_coef())

            def _apply_inten_unit():
                dec = _INTEN_UNITS[inten_unit.currentIndex()][2]
                inten.setDecimals(dec)
                inten.setSingleStep(10.0 / _inten_unit_coef())

            def _pair():
                """返回 (低态, 高态) 或 None。"""
                g_spec, e_spec = g_combo.currentData(), e_combo.currentData()
                if not g_spec or not e_spec:
                    return None
                gs = next((s for s in self.states if _spec_of(s) == g_spec), None)
                es = next((s for s in self.states if _spec_of(s) == e_spec), None)
                if gs is None or es is None:
                    return None
                return (gs, es) if gs['E_mhz'] <= es['E_mhz'] else (es, gs)

            def _pol_q():
                return {'π': 0, 'σ+': 1, 'σ-': -1}.get(pol.currentText(), 0)

            def _real_feedback(*_):
                """真实反馈：当前光强下这对的 Rabi + 当前频率失谐（相对裸差）。"""
                pr = _pair()
                if pr is None:
                    hint.setText('未配置')
                    return
                gs, es = pr
                nu_mhz = es['E_mhz'] - gs['E_mhz']          # 裸差
                f_mhz = freq.value() * _freq_unit_coef()
                delta = f_mhz - nu_mhz
                cache = getattr(self, 'cache', None)
                om = 0.0
                if cache is not None:
                    d = _core.dipole_amplitude(
                        cache, _drow(gs), _drow(es), _pol_q())
                    om = _core.rabi_mhz(
                        abs(d),
                        _core.field_e0_sq(inten.value() * _inten_unit_coef()))
                hint.setText(f'Rabi = {om:.4g} MHz（当前光强）\n'
                             f'Δ = {delta:+.4g} MHz（当前频率−裸差）')

            def _set_intensity(*_):
                pr = _pair()
                cache = getattr(self, 'cache', None)
                if pr is None or cache is None:
                    return
                d = _core.dipole_amplitude(
                    cache, _drow(pr[0]), _drow(pr[1]), _pol_q())
                if abs(d) < 1e-12:
                    hint.setText('该对在所选偏振下无耦合（Δm 不匹配）')
                    return
                mw = _core.intensity_mwcm2_from_rabi(abs(d), tgt_om.value())
                # 内部值直接写全精度（不经 spin——QDoubleSpinBox.setValue 会
                # round 到 decimals，THz/GHz 档曾因此丢 MHz 级精度）；
                # spin 仅刷新显示（智能切单位 + decimals 同步）
                pu.intensity_mwcm2 = mw
                ui = _smart_unit(mw, _INTEN_UNITS)
                inten_unit.setCurrentIndex(ui)      # 先切单位（显示旧值换算）
                inten.blockSignals(True)
                inten.setValue(mw / _INTEN_UNITS[ui][1])
                _apply_inten_unit()
                inten.blockSignals(False)
                hint.setText(f'I = {mw / _INTEN_UNITS[ui][1]:.6g} '
                             f'{_INTEN_UNITS[ui][0]}（目标 Ω 对应）')
                _real_feedback()
                head.setText(self._pulse_title(pu, idx))
                self.changed.emit()
            setI_b.clicked.connect(_set_intensity)

            def _set_freq(*_):
                pr = _pair()
                if pr is None:
                    return
                nu_mhz = pr[1]['E_mhz'] - pr[0]['E_mhz']
                f_mhz = nu_mhz + tgt_det.value()
                # 内部值直接写全精度（不经 spin 的 decimals round）
                pu.freq_mhz = f_mhz
                # 智能切单位：写回后的频率落在哪个档（如微波 → GHz）
                ui = _smart_unit(f_mhz, _FREQ_UNITS)
                freq_unit.setCurrentIndex(ui)
                freq.blockSignals(True)
                freq.setValue(f_mhz / _FREQ_UNITS[ui][1])
                _apply_freq_unit()
                freq.blockSignals(False)
                _real_feedback()
                head.setText(self._pulse_title(pu, idx))
                self.changed.emit()
            setF_b.clicked.connect(_set_freq)

            def _upd(*_):
                pu.freq_mhz = freq.value() * _freq_unit_coef()
                pu.intensity_mwcm2 = inten.value() * _inten_unit_coef()
                pu.pol = pol.currentText()
                if isinstance(e, GaussianEnv):
                    e.center_ns = spins[0][1].value()
                    e.sigma_ns = spins[1][1].value()
                    e.peak = spins[2][1].value()
                else:
                    e.start_ns = spins[0][1].value()
                    e.edge_ns = spins[1][1].value()
                    e.height = spins[2][1].value()
                    e.fall_start_ns = spins[3][1].value()
                pu.offset_ns = off.value()
                g_spec, e_spec = g_combo.currentData(), e_combo.currentData()
                if g_spec and e_spec:
                    pu.drive = dict(g=g_spec, e=e_spec)   # 辅助对（不参与结算）
                else:
                    pu.drive = None
                _real_feedback()
                self.canvas.update()
                head.setText(self._pulse_title(pu, idx))
                self.changed.emit()            # 参数变化 → 主窗口实时同步

            def _on_unit(*_):
                # 切换单位保持脉冲频率不变（仅换算显示值；decimals 随单位）
                freq.blockSignals(True)
                freq.setValue(pu.freq_mhz / _freq_unit_coef())
                _apply_freq_unit()
                freq.blockSignals(False)
                _real_feedback()

            def _on_inten_unit(*_):
                # 切换单位保持光强不变（仅换算显示值；decimals 随单位）
                inten.blockSignals(True)
                inten.setValue(pu.intensity_mwcm2 / _inten_unit_coef())
                _apply_inten_unit()
                inten.blockSignals(False)
                _real_feedback()
            freq_unit.currentIndexChanged.connect(_on_unit)
            inten_unit.currentIndexChanged.connect(_on_inten_unit)
            freq.valueChanged.connect(_upd)
            inten.valueChanged.connect(_upd)
            pol.currentTextChanged.connect(_upd)
            for _, sb in spins:
                sb.valueChanged.connect(_upd)
            off.valueChanged.connect(_upd)
            g_combo.currentIndexChanged.connect(_upd)
            e_combo.currentIndexChanged.connect(_upd)
            setI_b.clicked.connect(lambda: self.changed.emit())
            setF_b.clicked.connect(lambda: self.changed.emit())
            return w

        def _pulse_group(self, pu, idx, expanded):
            """单个脉冲的可折叠组：标题行（箭头+摘要+✕删除）点击开合。"""
            box = QWidget()
            v = QVBoxLayout(box)
            v.setContentsMargins(0, 0, 0, 0)
            v.setSpacing(0)
            head = QToolButton()
            head.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
            head.setCheckable(True)
            head.setChecked(expanded)
            head.setText(self._pulse_title(pu, idx))
            head.setArrowType(Qt.DownArrow if expanded else Qt.RightArrow)
            head.setStyleSheet(
                'QToolButton{background:#e8eef7;border:1px solid #c5d3e5;'
                'border-radius:4px;padding:4px 8px;text-align:left;'
                'font-weight:bold;}')
            # 宽度跟随面板（Ignored：不按文本 sizeHint 撑宽 → 小面板不溢出）
            head.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Fixed)
            head.setMinimumWidth(0)
            dl = QPushButton('✕')
            dl.setFixedSize(24, 24)
            dl.setToolTip('删除此脉冲')
            hr = QWidget()
            h = QHBoxLayout(hr)
            h.setContentsMargins(0, 0, 0, 0); h.setSpacing(3)
            h.addWidget(head, 1); h.addWidget(dl)
            v.addWidget(hr)
            content = self._pulse_content(pu, idx, head)
            content.setVisible(expanded)
            v.addWidget(content)

            def _toggle(on):
                content.setVisible(on)
                head.setArrowType(Qt.DownArrow if on else Qt.RightArrow)
            head.toggled.connect(_toggle)
            dl.clicked.connect(lambda: self._delete_pulse(idx))
            return box

        def _delete_pulse(self, idx):
            ch = self._current()
            if ch is not None and 0 <= idx < len(ch.pulses):
                ch.pulses.pop(idx)
                self._refresh()
                self.changed.emit()

        def _refresh(self):
            self._sync_channel_list()          # 左侧通道列表与结构同步
            while self._panel_box.count():
                it = self._panel_box.takeAt(0)
                w = it.widget()
                if w is not None:
                    # 先关闭所有 QComboBox 下拉并隐藏旧组，再删除：
                    # 避免切换通道时残留的 popup 视图被销毁瞬间闪现空白窗口
                    for cb in w.findChildren(QComboBox):
                        cb.hidePopup()
                    w.hide()
                    w.deleteLater()
            ch = self._current()
            pulses = ch.pulses if ch is not None else []
            for i, pu in enumerate(pulses):
                self._panel_box.addWidget(
                    self._pulse_group(pu, i, expanded=(i == 0)))
            # 加脉冲按钮（与折叠脉冲同一层次）
            addrow = QWidget()
            ar = QHBoxLayout(addrow)
            ar.setContentsMargins(0, 0, 0, 0); ar.setSpacing(6)
            for text, env in (('＋高斯', GaussianEnv(400, 60)),
                              ('＋方波', SmoothSquareEnv(
                                  200, 40, fall_start_ns=800))):
                b = QPushButton(text)
                b.clicked.connect(lambda _, e=env: self._add_pulse(e))
                ar.addWidget(b)
            self._panel_box.addWidget(addrow)
            self._panel_box.addStretch(1)
            self.canvas.update()

        def _export_text(self):
            lines = ['# 波形通道（与 TTL 一一对应）→ 每通道多个脉冲',
                     'from src import pulse', '']
            lines.append('channels = [')
            for ch in self.channels:
                lines.append(f'    pulse.WaveChannel({ch.name!r}, [')
                for i, pu in enumerate(ch.pulses):
                    e = pu.envelope
                    if isinstance(e, GaussianEnv):
                        env_s = (f'pulse.GaussianEnv({e.center_ns:g}, '
                                 f'{e.sigma_ns:g}, {e.peak:g})')
                    else:
                        env_s = (f'pulse.SmoothSquareEnv({e.start_ns:g}, '
                                 f'{e.edge_ns:g}, {e.height:g}, '
                                 f'fall_start_ns={e.fall_start_ns:g})')
                    lines.append(f'        pulse.Pulse({pu.freq_mhz:g}, '
                                 f'{env_s}, pol={pu.pol!r}, '
                                 f'offset_ns={pu.offset_ns:g}),  # p{i}')
                lines.append('    ]),')
            lines.append(']')
            return '\n'.join(lines)

        def _export(self):
            dlg = QDialog(self); dlg.setWindowTitle('导出脉冲代码')
            lay = QVBoxLayout(dlg)
            te = QTextEdit(); te.setPlainText(self._export_text())
            te.setFont(QFont('Consolas', 9))
            lay.addWidget(te)
            b = QPushButton('关闭'); b.clicked.connect(dlg.accept)
            lay.addWidget(b)
            dlg.resize(680, 460); dlg.exec_()

    def run_ui():
        """启动独立波形编辑器：python -m src.pulse"""
        app = QApplication(_sys.argv)
        w = PulseEditor(); w.show()
        _sys.exit(app.exec_())

else:
    def run_ui():
        raise RuntimeError('需要 PyQt5 才能启动 UI')


if __name__ == '__main__':
    run_ui()
