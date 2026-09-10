# -*- coding: utf-8 -*-
"""波形管理面板（动态页右侧面板内嵌，无独立窗口、无通道、无列表）。

每个波形 = 一个可折叠参数组（QToolButton 箭头标题 + ✕ 删除），样式与参数
编辑直接沿用旧版 TTL 波形编辑器的 PulsePanel（src/_archive/pulse.py）：

  [▼ 波形1 高斯 · 6.83e6MHz · π        ✕]
  | 频率 [spin][单位档]                 |   ← 每个参数行编辑即时生效
  | 光强 [spin][单位档]                 |
  | 偏振 [combo]                       |
  | 中心/σ/峰值 或 起点/沿/高度/下降     |
  | 起点偏移 [spin]                     |
  | [▸辅助]  g/e 态 + 目标Ω「定光强」   |   ← 可折叠辅助区（不参与结算）
  |          + 目标Δ「定频率」          |

不做：面板内波形预览图（直接看主面板动态页顶部光强画布）、
      通道/列表管理（每个波形组件自带 ✕ 删除）。
数据：self.waves（waveforms.Wave 列表），changed 信号 → 动态页刷新。
"""
import json

import numpy as np
from PyQt5.QtCore import Qt, pyqtSignal, QEvent, QMimeData
from PyQt5.QtGui import QDrag
from PyQt5.QtWidgets import (QApplication, QWidget, QVBoxLayout, QHBoxLayout,
                             QGridLayout, QFormLayout, QLabel, QDoubleSpinBox,
                             QComboBox, QPushButton, QToolButton, QScrollArea,
                             QMenu, QAction)

from ..core import core
from ..core import waveforms

# ---- 单位档（沿用旧编辑器；内部统一 MHz / mW/cm²，切换纯显示层） ----
_FREQ_UNITS = [('THz', 1e6, 9), ('GHz', 1e3, 6),
               ('MHz', 1.0, 3), ('kHz', 1e-3, 0), ('Hz', 1e-6, 0)]
_INTEN_UNITS = [('W/cm²', 1e3, 6), ('mW/cm²', 1.0, 6),
                ('μW/cm²', 1e-3, 6), ('nW/cm²', 1e-6, 6)]


def _smart_unit(value, units):
    """选显示值 ∈ [1,1000) 的单位档下标。"""
    if value <= 0:
        return len(units) - 1
    for i, (_, coef, _) in enumerate(units):
        if coef <= value < coef * 1000:
            return i
    return 0 if value > 0 else len(units) - 1


# 拖拽复制波形（跨原子）的 MIME 类型：JSON 载荷 = serialize_waves([w])[0] + '_src'
_WAVE_MIME = 'application/x-energy-level-wave'


# ---------------- 子能级唯一身份键（跨模块共用） ----------------
def state_key(s):
    """子能级身份键：(kind, n, l, j, m, energy_rounded)。

    - kind='F' 时 m = m_F；kind='mJ' 时 m = m_J
    - 末位能量（MHz，6 位小数）使键跨 B 稳定、与 F 标号无关、天然无重复；
      旧存档（无能量位）经 find_state_index 前缀回退兼容。
    """
    en = round(float(s['E_mhz']), 6)
    if s.get('kind') == 'F':
        return ('F', int(s['n']), int(s['l']), float(s['j']),
                float(s['mF']), en)
    return ('mJ', int(s['n']), int(s['l']), float(s['j']),
            float(s['m']), en)


def find_state_index(states, spec):
    """states 中匹配身份键 spec 的第一个索引（0 基）；找不到返回 -1。

    匹配顺序：① 完整键（含能量）精确；② 前 5 元 (kind,n,l,j,m) 精确
    （兼容旧存档无能量位 / 能量微漂）；③ (kind,n,l,j,m) 组内能量最近
    （B 漂移后 F 标号/能量变化）。
    """
    if not spec:
        return -1
    spec = tuple(spec)
    keys = [state_key(s) for s in states]
    # ① 完整键精确（含能量）
    for i, k in enumerate(keys):
        if k == spec:
            return i
    # ② 前 5 元精确（旧存档 / 同 (n,l,j,m) 能量微漂）
    if len(spec) >= 5:
        base = spec[:5]
        for i, k in enumerate(keys):
            if k[:5] == base:
                return i
    # ③ (kind,n,l,j,m) 组内能量最近
    base = spec[:5]
    best, bd = -1, None
    ref_e = spec[5] if len(spec) > 5 else None
    for i, k in enumerate(keys):
        if k[:5] != base:
            continue
        d = abs(k[5] - ref_e) if ref_e is not None else 0.0
        if bd is None or d < bd:
            best, bd = i, d
    return best


class WavePanel(QWidget):
    """波形管理：垂直堆叠可折叠波形组（＋高斯/＋方波 添加，组内 ✕ 删除）。"""

    changed = pyqtSignal()      # 波形增删/参数变化 → 动态页刷新

    def __init__(self, states=None, cache=None, parent=None):
        super().__init__(parent)
        self.waves = []          # waveforms.Wave 列表
        self.states = list(states or [])    # 演化基组（辅助 g/e 下拉）
        self.cache = cache                  # ArcCache（定光强/真实反馈）
        self._states_sig = None             # 最近一次基组签名（set_states 用）

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(4)

        row = QHBoxLayout()
        row.addWidget(QLabel('波形'))
        row.addStretch(1)
        b_g = QPushButton('＋高斯')
        b_g.setToolTip('添加高斯包络波形')
        b_g.clicked.connect(lambda: self._add('gauss'))
        row.addWidget(b_g)
        b_s = QPushButton('＋方波')
        b_s.setToolTip('添加方波包络波形')
        b_s.clicked.connect(lambda: self._add('square'))
        row.addWidget(b_s)
        b_a = QToolButton()
        b_a.setText('＋任意 ▾')
        b_a.setPopupMode(QToolButton.InstantPopup)
        b_a.setToolTip('从文件导入任意时间形状（txt/csv/dat/npy）。两种语义：\n'
                       '· 光强包络：文件=光强形状，载频由波形自带 → 三引擎全支持\n'
                       '· 完整驱动：文件=驱动信号本身（含振荡），无载频 → 仅 hard')
        act_env = QAction('作为光强包络（配载频）', b_a)
        act_env.setToolTip('文件给出 0..1 的光强形状，载频另填（默认取基组内\n'
                           '最小偶极跃迁）。sambe / mixed / hard 全支持，推荐。')
        act_env.triggered.connect(lambda: self._add('envfile'))
        act_drv = QAction('作为完整驱动（无载频，仅 hard）', b_a)
        act_drv.setToolTip('文件本身就是时域驱动信号（必须含目标跃迁的振荡，\n'
                           '否则静默零效果）。仅 hard 支持；采样率须过 Nyquist。')
        act_drv.triggered.connect(lambda: self._add('drive'))
        _menu = QMenu(b_a)
        _menu.addAction(act_env)
        _menu.addAction(act_drv)
        b_a.setMenu(_menu)
        row.addWidget(b_a)
        root.addLayout(row)

        # 波形组滚动区（可折叠组随波形数增长）
        self._scroll = QScrollArea()
        self._scroll.setWidgetResizable(True)
        self._scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        # 垂直滚动条恒显：宽度恒定，折叠/展开时控件宽度绝不跳变
        self._scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
        self._host = QWidget()
        self._box = QVBoxLayout(self._host)
        self._box.setContentsMargins(0, 0, 0, 0)
        self._box.setSpacing(6)
        self._scroll.setWidget(self._host)
        root.addWidget(self._scroll, 1)

    # ---------------- 波形管理 ----------------
    def set_states(self, states, cache=None):
        """基组变化 → 更新辅助 g/e 下拉数据源（仅在基组内容变化时重建）。"""
        states = list(states or [])
        sig = [(s['label'], s.get('kind'), s.get('n'), s.get('l'), s.get('j'))
               for s in states]
        if sig != self._states_sig:
            self.states = states
            self._states_sig = sig
            if cache is not None:
                self.cache = cache
            self._rebuild()

    def set_waves(self, waves):
        """恢复波形列表（存档恢复）并重建折叠组。"""
        self.waves = list(waves)
        self._rebuild()
        self.changed.emit()

    def _add(self, kind):
        if kind == 'gauss':
            env = waveforms.GaussianEnv(200.0, 40.0)
        elif kind == 'square':
            env = waveforms.SquareEnv(50.0, 20.0)
        elif kind == 'envfile':
            self._add_env_file()
            return
        else:                                   # 'drive'：完整时域驱动
            self._add_drive_file()
            return
        self.waves.append(waveforms.Wave(
            f'波形{len(self.waves) + 1}', 6.8e6, env,
            offset_ns=0.0, intensity_mwcm2=100.0))
        self._rebuild()
        self.changed.emit()

    def _add_env_file(self):
        """＋任意 → 作为**光强包络**：文件给形状，载频另填（保留载频语义）。

        与「完整驱动」的关键区别：本路径产出普通 `Wave`（载频 + 包络 callable），
        因此 sambe / mixed / hard 三引擎都支持，也不会踩"无载频"那串坑。
        """
        from PyQt5.QtWidgets import QInputDialog, QMessageBox
        data = self._load_samples('载入光强包络文件')
        if data is None:
            return
        t, a = data
        try:
            env = waveforms.FileEnv(t, a)
        except Exception as e:                                  # noqa: BLE001
            QMessageBox.warning(self, '载入失败', str(e))
            return
        f0 = self._suggest_freq('π')
        f, ok = QInputDialog.getDouble(
            self, '载频', '该包络作用的载频 (MHz)：\n'
            '（默认取基组内**耦合最强**的跨流形偶极跃迁；'
            '请按你的驱动意图核对）',
            f0, 1e-3, 1e12, 6)
        if not ok:
            return
        self.waves.append(waveforms.Wave(
            f'包络{len(self.waves) + 1}', float(f), env, pol='π',
            offset_ns=0.0, intensity_mwcm2=100.0))
        self._rebuild()
        self.changed.emit()

    def _add_drive_file(self):
        """＋任意 → 作为**完整驱动**：文件即驱动信号（无载频），仅 hard 支持。"""
        data = self._load_samples('载入完整驱动文件')
        if data is None:
            return
        t, a = data
        self.waves.append(waveforms.ArbitraryWave(
            f'驱动{len(self.waves) + 1}', t, a, intensity_mwcm2=100.0))
        self._rebuild()
        self.changed.emit()

    def _suggest_freq(self, pol='π'):
        """按当前基组建议载频（MHz）= 该偏振下**偶极最强**的**跨流形**跃迁。

        为什么要排除同流形 (n,l,j) 相同的对：那是 Zeeman/超精细劈裂（几十 MHz
        量级），取"最小跃迁"会选中它——看着像算出来的、其实几乎不是用户想驱动
        的对象，属静默坑。跨流形取最强耦合更接近"你打算驱动的跃迁"。
        退路：无跨流形对→用全部对；再无→6.8e6（与高斯/方波默认一致）。
        """
        try:
            q = {'π': 0, 'σ+': 1, 'σ-': -1}.get(pol, 0)
            st, ck = getattr(self, 'states', None), getattr(self, 'cache', None)
            if not st or ck is None:
                return 6.8e6
            from ..core.field_build import _sublevel_row
            cross, any_ = [], []
            for i in range(len(st)):
                for j in range(i + 1, len(st)):
                    d = core.dipole_amplitude(ck, _sublevel_row(st[i]),
                                              _sublevel_row(st[j]), q)
                    if abs(d) <= 1e-12:
                        continue
                    nu = abs(float(st[j]['E_mhz']) - float(st[i]['E_mhz']))
                    any_.append((abs(d), nu))
                    same = ((st[i]['n'], st[i]['l'], st[i]['j'])
                            == (st[j]['n'], st[j]['l'], st[j]['j']))
                    if not same:
                        cross.append((abs(d), nu))
            pool = cross or any_
            if not pool:
                return 6.8e6
            return float(max(pool, key=lambda x: x[0])[1])
        except Exception:                                       # noqa: BLE001
            return 6.8e6

    def _load_samples(self, title='载入波形文件'):
        """文件 → (t_ns, amp)。两列(时间ns,幅值)；单列(幅值)+QInputDialog 等间距。
        返回 None = 用户取消/失败。载入形状与语义无关，包络/完整驱动共用。"""
        from PyQt5.QtWidgets import (QFileDialog, QInputDialog, QMessageBox)
        path, _ = QFileDialog.getOpenFileName(
            self, title, '',
            '波形文件 (*.txt *.csv *.dat *.npy);;所有文件 (*)')
        if not path:
            return None
        try:
            if path.lower().endswith('.npy'):
                arr = np.load(path)
                if arr.ndim == 2:
                    t, a = arr[:, 0].astype(float), arr[:, 1].astype(float)
                elif arr.ndim == 1:
                    dt, ok = QInputDialog.getDouble(
                        self, '等间距载入', '采样间隔 dt (ns):',
                        0.02, 1e-6, 1e4, 6)
                    if not ok:
                        return None
                    t = np.arange(arr.size) * dt
                    a = arr.astype(float)
                else:
                    raise ValueError('npy 须为 1D(幅值) 或 2D(时间, 幅值)')
            else:
                arr = np.atleast_2d(np.loadtxt(path, comments='#'))
                if arr.shape[1] == 1:
                    dt, ok = QInputDialog.getDouble(
                        self, '等间距载入', '采样间隔 dt (ns):',
                        0.02, 1e-6, 1e4, 6)
                    if not ok:
                        return None
                    t = np.arange(arr.shape[0]) * dt
                    a = arr[:, 0]
                elif arr.shape[1] >= 2:
                    t, a = arr[:, 0], arr[:, 1]
                else:
                    raise ValueError('文件须至少 1 列')
            t = np.asarray(t, float)
            a = np.asarray(a, float)
            if t.size < 2 or a.size != t.size:
                raise ValueError('样品点过少或时间/幅值长度不一致')
            if np.any(np.diff(t) <= 0):
                raise ValueError('时间样品必须严格递增')
            return t, a
        except Exception as e:
            QMessageBox.warning(self, '载入失败', str(e))
            return None

    def _delete(self, idx):
        if 0 <= idx < len(self.waves):
            del self.waves[idx]
            self._rebuild()
            self.changed.emit()

    def _rebuild(self):
        """重建全部波形折叠组（保留展开状态）。"""
        expanded = []
        for i in range(self._box.count()):
            it = self._box.itemAt(i)
            if it is not None:
                w = it.widget()
                if w is not None and hasattr(w, '_expanded'):
                    expanded.append(w._expanded)
        while self._box.count():
            it = self._box.takeAt(0)
            if it.widget() is not None:
                it.widget().deleteLater()
        for i, w in enumerate(self.waves):
            exp = expanded[i] if i < len(expanded) else True
            self._box.addWidget(self._wave_group(w, i, exp))
        self._box.addStretch(1)

    # ---------------- 单个波形折叠组 ----------------
    def _wave_title(self, w, idx):
        if isinstance(w, waveforms.ArbitraryWave):
            return (f'波形{idx + 1} 任意 · {len(w.t_ns)}点 · '
                    f'dt={w.dt_min_ns():.3g}ns · {w.pol}')
        e = w.envelope
        if isinstance(e, waveforms.FileEnv):
            kind = '包络文件'
        elif isinstance(e, waveforms.GaussianEnv):
            kind = '高斯'
        else:
            kind = '方波'
        return f'波形{idx + 1} {kind} · {w.freq_mhz:.3g}MHz · {w.pol}'

    # ---------------- 拖拽复制（拖到另一原子的页签 = 复制到该原子） ----------------
    def eventFilter(self, obj, ev):
        """波形折叠组上的拖动手势：按住标题左键拖动 → QDrag。

        只负责"带出"载荷；放下由 ui_dynamic 的页签 drop 处理（跨原子复制）。
        返回 False：press/release 仍传给控件，展开/折叠点击不受影响。
        """
        wave = getattr(obj, '_drag_wave', None)
        if wave is None:
            return super().eventFilter(obj, ev)
        t = ev.type()
        if t == QEvent.MouseButtonPress and ev.button() == Qt.LeftButton:
            self._drag_press = (obj, ev.pos())
        elif (t == QEvent.MouseMove and ev.buttons() & Qt.LeftButton
              and getattr(self, '_drag_press', None)
              and self._drag_press[0] is obj):
            if (ev.pos() - self._drag_press[1]).manhattanLength() \
                    > QApplication.startDragDistance():
                d = waveforms.serialize_waves([wave])[0]
                d['_src'] = id(self)          # 目标据此忽略"拖回自己"
                mime = QMimeData()
                mime.setData(_WAVE_MIME, json.dumps(
                    d, ensure_ascii=False).encode('utf-8'))
                drag = QDrag(obj)
                drag.setMimeData(mime)        # 必须挂载：无 mime 不显示拖拽
                drag.exec_(Qt.CopyAction)
                self._drag_press = None
        elif t == QEvent.MouseButtonRelease:
            self._drag_press = None
        return False

    def copy_wave_from(self, payload):
        """从拖拽载荷复制一个波形到本面板尾部（深拷贝，名字加"(副本)"）。

        payload: _WAVE_MIME 的 bytes（JSON）。失败返回 False。
        """
        try:
            d = json.loads(payload.decode('utf-8'))
            d.pop('_src', None)
            w = waveforms.restore_waves([d])[0]
        except Exception:
            return False
        w.name = f'{w.name}(副本)'
        self.waves.append(w)
        self._rebuild()
        self.changed.emit()
        return True

    def _wave_group(self, w, idx, expanded=True):
        """单个波形的可折叠组：标题行（箭头+摘要+✕删除）点击开合。"""
        box = QWidget()
        box._expanded = expanded
        # 拖动手势绑定：整个组（标题为主要把手）都可作为拖拽源
        box._drag_wave = w
        box.installEventFilter(self)
        v = QVBoxLayout(box)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(0)

        head = QToolButton()
        head._drag_wave = w
        head.installEventFilter(self)
        head.setToolTip('点击折叠/展开参数；按住左键拖动到另一原子的'
                        '页签（如「原子 B」）= 复制该波形到那个原子')
        head.setCursor(Qt.OpenHandCursor)     # 悬停提示"可拖"
        head.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        head.setCheckable(True)
        head.setChecked(expanded)
        head.setText(self._wave_title(w, idx))
        head.setArrowType(Qt.DownArrow if expanded else Qt.RightArrow)
        head.setStyleSheet(
            'QToolButton{background:#e8eef7;border:1px solid #c5d3e5;'
            'border-radius:4px;padding:4px 8px;text-align:left;'
            'font-weight:bold;}')
        head.setSizePolicy(__import__('PyQt5.QtWidgets',
                                      fromlist=['QSizePolicy']).QSizePolicy
                           .Ignored, __import__('PyQt5.QtWidgets',
                                                fromlist=['QSizePolicy'])
                           .QSizePolicy.Fixed)
        head.setMinimumWidth(0)
        dl = QPushButton('✕')
        dl.setFixedSize(24, 24)
        dl.setToolTip('删除此波形')
        hr = QWidget()
        h = QHBoxLayout(hr)
        h.setContentsMargins(0, 0, 0, 0); h.setSpacing(3)
        h.addWidget(head, 1); h.addWidget(dl)
        v.addWidget(hr)

        content = self._wave_content(w, idx, head)
        content.setVisible(expanded)
        v.addWidget(content)

        def _toggle(on):
            box._expanded = on
            content.setVisible(on)
            head.setArrowType(Qt.DownArrow if on else Qt.RightArrow)
        head.toggled.connect(_toggle)
        dl.clicked.connect(lambda: self._delete(idx))
        return box

    # ---------------- 波形参数内容 ----------------
    def set_t_end(self, t_end_ns):
        """告知演化窗口 → 带宽显示随引擎查表上限收紧（t78）。

        `f_max` 有两个上限：文件 Nyquist 与引擎查表 Nyquist（≈0.5·N_TAB/T，
        T=620 ns 时仅 ~106 GHz）。只报前者会过度承诺，故窗口已知时取较小值。
        """
        if getattr(self, '_t_end_ns', None) == t_end_ns:
            return
        self._t_end_ns = float(t_end_ns) if t_end_ns else None
        self._rebuild()

    def _envfile_info(self, e):
        """文件包络摘要（点数/采样/时长/带宽）。"""
        f = e.f_max_mhz(getattr(self, '_t_end_ns', None))
        return (f'{len(e.t_ns)} 点 · dt_min={e.dt_min_ns():.3g} ns · '
                f'时长={e.span_ns():.4g} ns · 带宽≥{f:.4g} MHz')

    def _arb_info(self, w):
        """完整驱动摘要（样品统计 + 求解提示）。"""
        f = w.f_max_mhz(getattr(self, '_t_end_ns', None))
        return (f'{len(w.t_ns)} 点 · dt_min={w.dt_min_ns():.3g} ns · '
                f'带宽≥{f:.4g} MHz · 幅值峰值归一 · 文件即驱动（须含目标'
                '跃迁的振荡，否则零效果；仅 hard 支持）')

    def _arb_content(self, w, idx, head):
        """任意波形参数区：光强 | 偏振 | 偏移 + 样品信息/重新载入。
        无频率（无单载频）→ 无辅助定参区。"""
        from PyQt5.QtWidgets import QMessageBox

        def _label(txt):
            lb = QLabel(txt)
            lb.setStyleSheet('color:#4a5568; font-size:9pt;')
            return lb

        def _spin(lo, hi, dec, val, step=1.0, suffix='', min_w=100):
            sb = QDoubleSpinBox()
            sb.setRange(lo, hi); sb.setDecimals(dec)
            sb.setSingleStep(step); sb.setValue(val)
            if suffix:
                sb.setSuffix(suffix)
            sb.setMinimumWidth(min_w)
            return sb

        widget = QWidget()
        outer = QVBoxLayout(widget)
        outer.setContentsMargins(10, 8, 10, 8)
        outer.setSpacing(4)
        row = QGridLayout()
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(6)

        _ii = _smart_unit(w.intensity_mwcm2, _INTEN_UNITS)
        inten = _spin(0, 1e9, _INTEN_UNITS[_ii][2],
                      w.intensity_mwcm2 / _INTEN_UNITS[_ii][1], 10.0,
                      min_w=160)
        inten.setToolTip('峰值光强（决定场绝对幅度：E(t)=E0·amp(t)）')
        inten_unit = QComboBox()
        inten_unit.setFixedWidth(64)
        for name, _, _ in _INTEN_UNITS:
            inten_unit.addItem(name)
        inten_unit.setCurrentIndex(_ii)
        ri = QHBoxLayout()
        ri.setSpacing(2)
        ri.addWidget(inten, 1)
        ri.addWidget(inten_unit)
        row.addWidget(_label('光强'), 0, 0)
        row.addLayout(ri, 0, 1)

        pol = QComboBox()
        pol.addItems(list(waveforms.POLS))
        pol.setCurrentText(w.pol)
        pol.setMinimumWidth(110)
        row.addWidget(_label('偏振'), 0, 2)
        row.addWidget(pol, 0, 3)

        off = _spin(-1e9, 1e9, 1, w.offset_ns, 10, ' ns', min_w=130)
        off.setToolTip('波形绝对起点相对演化 t=0 的偏移')
        row.addWidget(_label('偏移'), 0, 4)
        row.addWidget(off, 0, 5)
        outer.addLayout(row)

        info = QLabel(self._arb_info(w))
        info.setStyleSheet('font-family:Consolas; font-size:9pt; color:#0b5394;')
        outer.addWidget(info)
        reload_b = QPushButton('重新载入波形文件…')
        reload_b.setToolTip('替换样品数据（保留光强/偏振/偏移）')
        outer.addWidget(reload_b)

        def _upd(*_):
            w.intensity_mwcm2 = (inten.value()
                                 * _INTEN_UNITS[inten_unit.currentIndex()][1])
            w.pol = pol.currentText()
            w.offset_ns = off.value()
            head.setText(self._wave_title(w, idx))
            self.changed.emit()

        def _on_inten_unit(*_):
            inten.blockSignals(True)
            inten.setValue(w.intensity_mwcm2
                           / _INTEN_UNITS[inten_unit.currentIndex()][1])
            inten.blockSignals(False)

        def _reload(*_):
            data = self._load_samples('重新载入完整驱动文件')
            if data is None:
                return
            t, a = data
            try:
                w.t_ns = t
                w.amp = a / np.max(np.abs(a))
            except Exception as e:
                QMessageBox.warning(widget, '载入失败', str(e))
                return
            info.setText(self._arb_info(w))
            head.setText(self._wave_title(w, idx))
            self.changed.emit()

        inten_unit.currentIndexChanged.connect(_on_inten_unit)
        inten.valueChanged.connect(_upd)
        pol.currentTextChanged.connect(_upd)
        off.valueChanged.connect(_upd)
        reload_b.clicked.connect(_reload)
        return widget

    def _wave_content(self, w, idx, head):
        """单个波形的紧凑参数区（网格多列，利用加宽的主面板；编辑即时生效）。

        布局（QVBox：参数网格 + 辅助折叠区分离，折叠辅助不影响参数宽度）：
          网格：行0 频率|光强|偏振；行1 包络参数|起点偏移
          下方：[▸辅助]（g/e 态 + 目标Ω「定光强」+ 目标Δ「定频率」）
        任意波形（ArbitraryWave）走 _arb_content：无频率/无辅助。
        """
        if isinstance(w, waveforms.ArbitraryWave):
            return self._arb_content(w, idx, head)
        widget = QWidget()
        outer = QVBoxLayout(widget)
        outer.setContentsMargins(10, 8, 10, 8)
        outer.setSpacing(4)
        grid_host = QWidget()
        g = QGridLayout(grid_host)
        g.setContentsMargins(0, 0, 0, 0)
        g.setSpacing(6)
        # 列宽固定分配：频率/光强数字组件列（1,3）双倍宽于偏振列（5），
        # 承载数字的组件更宽；辅助折叠不影响参数宽度
        g.setColumnStretch(1, 2)
        g.setColumnStretch(3, 2)
        g.setColumnStretch(5, 1)

        def _spin(lo, hi, dec, val, step=1.0, suffix='', min_w=100):
            sb = QDoubleSpinBox()
            sb.setRange(lo, hi); sb.setDecimals(dec)
            sb.setSingleStep(step); sb.setValue(val)
            if suffix:
                sb.setSuffix(suffix)
            sb.setMinimumWidth(min_w)
            sb.setSizePolicy(
                __import__('PyQt5.QtWidgets', fromlist=['QSizePolicy'])
                .QSizePolicy.Expanding,
                __import__('PyQt5.QtWidgets', fromlist=['QSizePolicy'])
                .QSizePolicy.Fixed)
            return sb

        def _label(txt):
            lb = QLabel(txt)
            lb.setStyleSheet('color:#4a5568; font-size:9pt;')
            return lb

        # ============ 行0：频率 | 光强 | 偏振 ============
        _fi = _smart_unit(w.freq_mhz, _FREQ_UNITS)
        freq = _spin(0, 1e9, _FREQ_UNITS[_fi][2],
                     w.freq_mhz / _FREQ_UNITS[_fi][1], 1e-3, min_w=200)
        freq.setSingleStep(1.0 / _FREQ_UNITS[_fi][1])
        freq.setToolTip('波形载频；可切换 THz/GHz/MHz/kHz/Hz 单位\n'
                        '内部 MHz 全精度（double），切换单位不丢位')
        freq_unit = QComboBox()
        freq_unit.setFixedWidth(64)
        for name, _, _ in _FREQ_UNITS:
            freq_unit.addItem(name)
        freq_unit.setCurrentIndex(_fi)
        row_f = QHBoxLayout()
        row_f.setSpacing(4)
        row_f.addWidget(freq, 1)
        row_f.addWidget(freq_unit)
        g.addWidget(_label('频率'), 0, 0)
        g.addLayout(row_f, 0, 1)

        _ii = _smart_unit(w.intensity_mwcm2, _INTEN_UNITS)
        inten = _spin(0, 1e9, _INTEN_UNITS[_ii][2],
                      w.intensity_mwcm2 / _INTEN_UNITS[_ii][1], 10.0,
                      min_w=200)
        inten.setToolTip('光强：与频率共同决定真实驱动强度（Rabi），结算用；\n'
                         '可切换 W/mW/μW/nW 单位（内部 mW/cm² 全精度）')
        inten_unit = QComboBox()
        inten_unit.setFixedWidth(64)
        for name, _, _ in _INTEN_UNITS:
            inten_unit.addItem(name)
        inten_unit.setCurrentIndex(_ii)
        row_i = QHBoxLayout()
        row_i.setSpacing(2)
        row_i.addWidget(inten, 1)
        row_i.addWidget(inten_unit)
        g.addWidget(_label('光强'), 0, 2)
        g.addLayout(row_i, 0, 3)

        pol = QComboBox()
        pol.addItems(list(waveforms.POLS))
        pol.setCurrentText(w.pol)
        pol.setMinimumWidth(120)
        g.addWidget(_label('偏振'), 0, 4)
        g.addWidget(pol, 0, 5)

        # ============ 行1：包络参数 | 起点偏移（独立 HBox，不占网格列，防溢出） ============
        e = w.envelope
        if isinstance(e, waveforms.GaussianEnv):
            spins = [('中心', _spin(0, 1e7, 1, e.center_ns, 10, ' ns', 90)),
                     ('σ', _spin(1e-3, 1e6, 1, e.sigma_ns, 1, ' ns', 90)),
                     ('峰值', _spin(0, 1, 3, e.peak, 0.1, min_w=90))]
        elif isinstance(e, waveforms.FileEnv):
            spins = []                       # 包络形状来自文件 → 无参数旋钮
        else:
            spins = [('起点', _spin(0, 1e7, 1, e.start_ns, 10, ' ns', 68)),
                     ('沿', _spin(0, 1e6, 1, e.edge_ns, 1, ' ns', 68)),
                     ('高度', _spin(0, 1, 3, e.height, 0.1, min_w=68)),
                     ('下降', _spin(0, 1e7, 1, e.fall_start_ns, 10, ' ns', 68))]
        env_row = QHBoxLayout()
        env_row.setSpacing(4)
        _QSP = __import__('PyQt5.QtWidgets', fromlist=['QSizePolicy']).QSizePolicy
        for lab, sb in spins:
            cell = QWidget()
            cell.setSizePolicy(_QSP.Ignored, _QSP.Preferred)
            cl = QHBoxLayout(cell)
            cl.setContentsMargins(0, 0, 0, 0)
            cl.setSpacing(3)
            cl.addWidget(_label(lab))
            cl.addWidget(sb, 1)
            env_row.addWidget(cell, 1)
        if isinstance(e, waveforms.FileEnv):
            # 包络来自文件：显示样品信息 + 重新载入（形状由文件定，无旋钮）
            _fi = QLabel(self._envfile_info(e))
            _fi.setStyleSheet('font-family:Consolas; font-size:9pt;'
                              'color:#0b5394;')
            _rb = QPushButton('重新载入包络文件…')
            _rb.setToolTip('替换包络样品（保留载频/光强/偏振/偏移）')
            env_row.addWidget(_fi, 2)
            env_row.addWidget(_rb, 1)

            def _reload_env(_e=e, _lb=_fi):
                from PyQt5.QtWidgets import QMessageBox
                data = self._load_samples('重新载入光强包络文件')
                if data is None:
                    return
                try:
                    e2 = waveforms.FileEnv(*data)
                except Exception as ex:                          # noqa: BLE001
                    QMessageBox.warning(widget, '载入失败', str(ex))
                    return
                _e.t_ns, _e.amp = e2.t_ns, e2.amp
                _lb.setText(self._envfile_info(_e))
                self.changed.emit()

            _rb.clicked.connect(_reload_env)
        off = _spin(-1e9, 1e9, 1, w.offset_ns, 10, ' ns', min_w=100)
        off.setToolTip('波形绝对起点相对演化 t=0 的偏移')
        off_cell = QWidget()
        off_cell.setSizePolicy(_QSP.Ignored, _QSP.Preferred)
        oc = QHBoxLayout(off_cell)
        oc.setContentsMargins(0, 0, 0, 0)
        oc.setSpacing(3)
        oc.addWidget(_label('偏移'))
        oc.addWidget(off, 1)
        env_row.addWidget(off_cell, 1)
        # 行1 独立于网格（不占网格列）：宽度 = 面板全宽，HBox 均分防溢出

        # ============ 行2：辅助（可折叠） ============
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
        # 辅助区独立于参数网格（折叠只改高度，不影响网格列宽）
        grid_host.setSizePolicy(
            __import__('PyQt5.QtWidgets', fromlist=['QSizePolicy'])
            .QSizePolicy.Ignored,
            __import__('PyQt5.QtWidgets', fromlist=['QSizePolicy'])
            .QSizePolicy.Preferred)
        outer.addWidget(grid_host)
        outer.addLayout(env_row)      # 包络参数行占面板全宽，均分防溢出
        outer.addWidget(drive_head)

        body = QWidget()
        body.setSizePolicy(
            __import__('PyQt5.QtWidgets', fromlist=['QSizePolicy'])
            .QSizePolicy.Ignored,
            __import__('PyQt5.QtWidgets', fromlist=['QSizePolicy'])
            .QSizePolicy.Preferred)
        bg = QGridLayout(body)
        bg.setContentsMargins(12, 2, 4, 2)
        bg.setSpacing(6)
        bg.setColumnStretch(1, 1)
        bg.setColumnStretch(3, 1)

        def _spec_of(s):
            return state_key(s)

        def _drow(s):
            return dict(n=s['n'], l=s['l'], j=s['j'], kind=s['kind'],
                        mj=s.get('mj'), mF=s.get('mF'), amps=s.get('amps'))

        g_combo = QComboBox()
        g_combo.setMaxVisibleItems(14)
        g_combo.addItem('— 未配置 —', None)
        for s in self.states:
            g_combo.addItem(s['label'], _spec_of(s))
        bg.addWidget(_label('g 态'), 0, 0)
        bg.addWidget(g_combo, 0, 1)
        e_combo = QComboBox()
        e_combo.setMaxVisibleItems(14)
        e_combo.addItem('— 未配置 —', None)
        for s in self.states:
            e_combo.addItem(s['label'], _spec_of(s))
        bg.addWidget(_label('e 态'), 0, 2)
        bg.addWidget(e_combo, 0, 3)

        hint = QLabel('未配置')
        hint.setStyleSheet(
            'font-family: Consolas; font-size: 9pt; color: #0b5394;')
        bg.addWidget(hint, 1, 1, 1, 3)

        tgt_om = _spin(0, 1e5, 4, 1.0, 1.0, ' MHz')
        setI_b = QPushButton('定光强')
        setI_b.setToolTip('反推光强使该对 Rabi = 目标 Ω，写回上方"光强"')
        row_om = QHBoxLayout()
        row_om.setSpacing(2)
        row_om.addWidget(tgt_om, 1)
        row_om.addWidget(setI_b)
        bg.addWidget(_label('目标Ω'), 2, 0)
        bg.addLayout(row_om, 2, 1)

        tgt_det = _spin(-1e6, 1e6, 3, 0.0, 1.0, ' MHz')
        tgt_det.setToolTip('失谐相对裸能级差（不含能移），无歧义')
        setF_b = QPushButton('定频率')
        setF_b.setToolTip('频率 = 裸差 + Δ，写回上方"频率"')
        row_det = QHBoxLayout()
        row_det.setSpacing(2)
        row_det.addWidget(tgt_det, 1)
        row_det.addWidget(setF_b)
        bg.addWidget(_label('目标Δ'), 2, 2)
        bg.addLayout(row_det, 2, 3)

        outer.addWidget(body)

        def _toggle_drive(on):
            body.setVisible(on)
            drive_head.setArrowType(Qt.DownArrow if on else Qt.RightArrow)
        drive_head.toggled.connect(_toggle_drive)

        def _find_state(spec):
            i = find_state_index(self.states, spec)
            return i + 1 if i >= 0 else 0

        if w.drive is not None:
            g_combo.setCurrentIndex(_find_state(w.drive['g']))
            e_combo.setCurrentIndex(_find_state(w.drive['e']))

        def _freq_unit_coef():
            return _FREQ_UNITS[freq_unit.currentIndex()][1]

        def _inten_unit_coef():
            return _INTEN_UNITS[inten_unit.currentIndex()][1]

        def _apply_freq_unit():
            dec = _FREQ_UNITS[freq_unit.currentIndex()][2]
            freq.setDecimals(dec)
            freq.setSingleStep(1.0 / _freq_unit_coef())

        def _apply_inten_unit():
            dec = _INTEN_UNITS[inten_unit.currentIndex()][2]
            inten.setDecimals(dec)
            inten.setSingleStep(10.0 / _inten_unit_coef())

        def _pair():
            g_spec, e_spec = g_combo.currentData(), e_combo.currentData()
            if not g_spec or not e_spec:
                return None
            gi = find_state_index(self.states, g_spec)
            ei = find_state_index(self.states, e_spec)
            gs = self.states[gi] if gi >= 0 else None
            es = self.states[ei] if ei >= 0 else None
            if gs is None or es is None:
                return None
            return (gs, es) if gs['E_mhz'] <= es['E_mhz'] else (es, gs)

        def _pol_q():
            return {'π': 0, 'σ+': 1, 'σ-': -1}.get(pol.currentText(), 0)

        def _real_feedback(*_):
            pr = _pair()
            if pr is None:
                hint.setText('未配置')
                return
            gs, es = pr
            nu_mhz = es['E_mhz'] - gs['E_mhz']
            f_mhz = freq.value() * _freq_unit_coef()
            delta = f_mhz - nu_mhz
            cache = getattr(self, 'cache', None)
            om = 0.0
            if cache is not None:
                d = core.dipole_amplitude(cache, _drow(gs), _drow(es), _pol_q())
                om = core.rabi_mhz(
                    abs(d), core.field_e0_sq(inten.value() * _inten_unit_coef()))
            hint.setText(f'Rabi = {om:.4g} MHz（当前光强）\n'
                         f'Δ = {delta:+.4g} MHz（当前频率−裸差）')

        def _set_intensity(*_):
            pr = _pair()
            cache = getattr(self, 'cache', None)
            if pr is None or cache is None:
                return
            d = core.dipole_amplitude(cache, _drow(pr[0]), _drow(pr[1]), _pol_q())
            if abs(d) < 1e-12:
                hint.setText('该对在所选偏振下无耦合（Δm 不匹配）')
                return
            mw = core.intensity_mwcm2_from_rabi(abs(d), tgt_om.value())
            w.intensity_mwcm2 = mw
            ui = _smart_unit(mw, _INTEN_UNITS)
            inten_unit.setCurrentIndex(ui)
            inten.blockSignals(True)
            inten.setValue(mw / _INTEN_UNITS[ui][1])
            _apply_inten_unit()
            inten.blockSignals(False)
            hint.setText(f'I = {mw / _INTEN_UNITS[ui][1]:.6g} '
                         f'{_INTEN_UNITS[ui][0]}（目标 Ω 对应）')
            _real_feedback()
            head.setText(self._wave_title(w, idx))
            self.changed.emit()
        setI_b.clicked.connect(_set_intensity)

        def _set_freq(*_):
            pr = _pair()
            if pr is None:
                return
            nu_mhz = pr[1]['E_mhz'] - pr[0]['E_mhz']
            f_mhz = nu_mhz + tgt_det.value()
            w.freq_mhz = f_mhz
            ui = _smart_unit(f_mhz, _FREQ_UNITS)
            freq_unit.setCurrentIndex(ui)
            freq.blockSignals(True)
            freq.setValue(f_mhz / _FREQ_UNITS[ui][1])
            _apply_freq_unit()
            freq.blockSignals(False)
            _real_feedback()
            head.setText(self._wave_title(w, idx))
            self.changed.emit()
        setF_b.clicked.connect(_set_freq)

        def _upd(*_):
            w.freq_mhz = freq.value() * _freq_unit_coef()
            w.intensity_mwcm2 = inten.value() * _inten_unit_coef()
            w.pol = pol.currentText()
            if isinstance(e, waveforms.GaussianEnv):
                e.center_ns = spins[0][1].value()
                e.sigma_ns = spins[1][1].value()
                e.peak = spins[2][1].value()
            elif not isinstance(e, waveforms.FileEnv):
                e.start_ns = spins[0][1].value()
                e.edge_ns = spins[1][1].value()
                e.height = spins[2][1].value()
                e.fall_start_ns = spins[3][1].value()
            w.offset_ns = off.value()
            g_spec, e_spec = g_combo.currentData(), e_combo.currentData()
            if g_spec and e_spec:
                w.drive = dict(g=g_spec, e=e_spec)
            else:
                w.drive = None
            _real_feedback()
            head.setText(self._wave_title(w, idx))
            self.changed.emit()

        def _on_unit(*_):
            freq.blockSignals(True)
            freq.setValue(w.freq_mhz / _freq_unit_coef())
            _apply_freq_unit()
            freq.blockSignals(False)
            _real_feedback()

        def _on_inten_unit(*_):
            inten.blockSignals(True)
            inten.setValue(w.intensity_mwcm2 / _inten_unit_coef())
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
        return widget
