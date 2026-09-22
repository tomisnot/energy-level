# -*- coding: utf-8 -*-
"""纯波形数据层：一个波形 = 频率 + 光强 + 偏振 + 包络(高斯|方波) + 起始偏移。

取代旧「TTL 时序 + 波形通道」两层编辑器模型：
  旧：TimingPulseWindow（TTL 触发层）→ WaveChannel（通道层）→ Pulse
  新：直接管理一串 Wave（无通道、无 TTL 触发、无独立窗口）。

每个 Wave 的绝对起点 = offset_ns（ns，相对演化 t=0）。
包络参数：
  高斯：center_ns（相对起点）、sigma_ns、peak(0,1]
  方波：start_ns（相对起点）、edge_ns、height(0,1]、fall_start_ns（缺省 start+2·edge）

数据源接口（供动态页复用）：
  waveforms_to_env(waves, t_grid) -> [(label, t, env, freq_mhz, intensity), ...]
  serialize_waves(waves) / restore_waves(data) -> Wave 列表

场构造入口（现行三引擎全在此）：
  · `field_build.build_fields_exact` —— sambe / mixed 的 kind='exact' 场；
  · `evolve_twofield.build_two_field` —— hard / mixed 的两场 meta；
  · `evolve_twofield.arbitrary_mw_group` —— ArbitraryWave → 微波组
    （Hilbert 分解 env/φ，走 hard 的 cos(φ_f(t)) 任意相位路径）。

旧 ip4/Lindblad 管线的 `build_fields`（本模块）与 `field_build.arbitrary_field`
已删除——三引擎都不用它们（见 docs/t74b兼容层拆除.md）。
"""
import numpy as np


from .field_build import build_fields_exact  # noqa: F401

from .hard import _TAB_N

# 与旧 pulse.py 一致的包络形状（数据类抽离 UI，纯计算）
POLS = ['σ+', 'π', 'σ-', '椭圆偏']


def _smooth(x):
    """x∈[0,1] → 0..1 余弦缓变（raised cosine，两端导数连续=0）。"""
    x = np.clip(x, 0.0, 1.0)
    return 0.5 * (1.0 - np.cos(np.pi * x))


def _f_max_mhz(dt_min_ns, t_end_ns=None):
    """可表达带宽上限（MHz）= min(样品 Nyquist, 引擎查表 Nyquist, 波长约束)。

    样品 Nyquist = 0.5/dt_min（GHz→MHz 用 0.5e3/dt_ns）。
    引擎侧：驱动包络/相位被预采样成 `_TAB_N` 点查表，其 Nyquist
    = 0.5·(_TAB_N−1)/T。**只报前者会过度承诺**（T=620 ns 时表上限仅 ~106 GHz，
    恰好压在微波/光场分界上），故 t_end_ns 给定时取两者较小值。
    """
    fn = 0.0 if dt_min_ns <= 0 else 0.5e3 / float(dt_min_ns)
    if t_end_ns and float(t_end_ns) > 0 and fn > 0.0:
        fn = min(fn, 0.5e3 * (_TAB_N - 1) / float(t_end_ns))
    return fn


class GaussianEnv:
    """高斯振幅包络：peak·exp(−(τ−center)²/2σ²)，τ 以波形起点为 0。"""
    kind = 'gauss'

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

    def span(self, lo, hi):
        """包络在 [lo,hi]（**包络相对时间**）上的 (inf, sup)——**精确**。

        GaussianEnv 单峰 ⇒ 极值只可能在区间两端或峰位取得（O(1)，与区间宽度
        无关）。这是"这段包络是否恒常"的**唯一权威答案**，取代固定点数采样
        （事故说明见 `field_build.env_span`）。
        """
        a, b = float(self.center_ns), float(self.sigma_ns)
        lo, hi = float(lo), float(hi)
        cands = [lo, hi]
        if lo <= a <= hi:
            cands.append(a)
        vals = [self.peak * float(np.exp(-((c - a) ** 2) / (2.0 * b * b)))
                for c in cands]
        return min(vals), max(vals)

    def dt_feature_ns(self):
        """最窄变化特征时长（ns）：高斯 ⇒ σ。"""
        return float(self.sigma_ns)

    def __repr__(self):
        return (f'GaussianEnv(center={self.center_ns}ns, '
                f'sigma={self.sigma_ns}ns, peak={self.peak})')


class SquareEnv:
    """缓变方波包络：start→(start+edge) 余弦上升、平台到 fall_start、
    (fall_start)→(fall_start+edge) 余弦下降，目标高度 height。
    fall_start 缺省 = start + 2·edge（两沿相接的最小脉冲）。"""
    kind = 'square'

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

    def span(self, lo, hi):
        """包络在 [lo,hi] 上的 (inf, sup)。

        分段单调（升沿/平台/降沿/零区各自单调）⇒ 极值在折点或区间端点取得。
        唯一例外：`fall_start < start+edge` 时升沿与降沿**重叠**，乘积在重叠区
        可能有一个内部极值 ⇒ 额外取该重叠区中点与相邻候选点中点。
        """
        lo, hi = float(lo), float(hi)
        knots = [self.start_ns, self.start_ns + self.edge_ns,
                 self.fall_start_ns, self.fall_start_ns + self.edge_ns]
        pts = [lo, hi] + [k for k in knots if lo <= k <= hi]
        ov_lo, ov_hi = self.fall_start_ns, self.start_ns + self.edge_ns
        if ov_hi > ov_lo:
            mid = 0.5 * (ov_lo + ov_hi)
            if lo <= mid <= hi:
                pts.append(mid)
        pts.sort()
        pts = pts + [0.5 * (a + b) for a, b in zip(pts[:-1], pts[1:])]
        vals = [float(np.asarray(self(np.array([c])), float).flat[0])
                for c in pts]
        return min(vals), max(vals)

    def dt_feature_ns(self):
        """最窄变化特征时长（ns）：方波 ⇒ 沿时长（0 = 理想跳变，无特征尺度）。"""
        return float(self.edge_ns)


class FileEnv:
    """**文件导入的光强包络**（保留载频）：env(τ) = interp(τ, t_ns, |amp|)。

    与 `ArbitraryWave` 的分工（两者都从文件来，但语义完全不同）：
      · `FileEnv`  —— 只提供 0..1 的**光强包络**；载频由所附 `Wave` 决定。
        因为载频明确，**sambe / mixed / hard 三引擎全支持**（sambe 正是靠
        "包络自适应分段"处理任意形状包络：不连续、任意分段、快速调制都行）。
        文件里的振荡会被**忽略**（取 |amp| 峰值归一）——它表示的是光强形状。
      · `ArbitraryWave` —— 文件是**完整时域驱动信号**，无载频概念，只有 hard 认。

    ⚠ 本类**故意不定义 `start_ns`**：hard/twofield 用 `hasattr(env,'start_ns')`
    识别方波的不连续沿，若本类带上该属性会被误当成方波边界。
    """
    kind = 'file'

    def __init__(self, t_ns, amp):
        t = np.asarray(t_ns, float)
        a = np.asarray(amp, float)
        if t.ndim != 1 or a.ndim != 1 or t.size != a.size or t.size < 2:
            raise ValueError('包络文件样品须为等长一维数组（至少 2 点）')
        if np.any(np.diff(t) <= 0):
            raise ValueError('包络文件时间样品必须严格递增')
        mx = float(np.max(np.abs(a)))
        if mx <= 0:
            raise ValueError('包络文件幅值全零')
        self.t_ns = t
        self.amp = np.abs(a) / mx          # 光强包络：非负、峰值归一

    def __call__(self, tau):
        return np.interp(np.asarray(tau, float), self.t_ns, self.amp,
                         left=0.0, right=0.0)

    def span(self, lo, hi):
        """包络在 [lo,hi] 上的 (inf, sup)——精确。

        `np.interp(..., left=0, right=0)` 是**分段线性 + 区外钳 0** ⇒ 极值在
        样品点或区间端点取得。
        """
        lo, hi = float(lo), float(hi)
        m = (self.t_ns >= lo) & (self.t_ns <= hi)
        vals = [float(x) for x in self.amp[m]]
        vals.append(float(np.asarray(self(np.array([lo])), float).flat[0]))
        vals.append(float(np.asarray(self(np.array([hi])), float).flat[0]))
        return min(vals), max(vals)

    def dt_feature_ns(self):
        """最窄变化特征时长（ns）：文件包络 ⇒ 最小样品间隔。"""
        return self.dt_min_ns()

    def span_ns(self):
        """包络的绝对时长（ns）。"""
        return float(self.t_ns[-1] - self.t_ns[0])

    def dt_min_ns(self):
        d = np.diff(self.t_ns)
        d = d[d > 0]
        return float(d.min()) if d.size else 0.0

    def f_max_mhz(self, t_end_ns=None):
        """包络可表达带宽（MHz）：min(文件 Nyquist, 引擎查表 Nyquist)。"""
        return _f_max_mhz(self.dt_min_ns(), t_end_ns)

    def __repr__(self):
        return (f'FileEnv(N={len(self.t_ns)}点, dt_min={self.dt_min_ns():.3g}ns,'
                f' span={self.span_ns():.3g}ns)')

    def __repr__(self):
        return (f'SquareEnv(start={self.start_ns}ns, '
                f'edge={self.edge_ns}ns, height={self.height}, '
                f'fall_start={self.fall_start_ns}ns)')


class Wave:
    """一个独立波形：频率 + 光强 + 偏振 + 包络 + 起始偏移（无通道无触发）。"""

    def __init__(self, name, freq_mhz, envelope, pol='π',
                 offset_ns=0.0, intensity_mwcm2=100.0, drive=None,
                 elliptical=None):
        assert pol in POLS, f'偏振须为 {POLS}'
        self.name = name or '波形'
        self.freq_mhz = float(freq_mhz)
        self.envelope = envelope
        self.pol = pol
        # 椭偏参数（α/θ/ε，度）：仅 pol=='椭圆偏' 时有效；含义见
        # static_atom.elliptical_coeffs。None = 未配置（构建场时会报错）。
        # 选了「椭圆偏」但没给参数 → 补齐为 (0,0,0)（= x 线偏，有明确物理
        # 含义）。**不能**在这里抛异常：GUI 里常在 Qt 槽内建场，异常会变成
        # PyQt qFatal → 进程原生崩溃且无回溯（本项目踩过多次）。
        if pol == '椭圆偏' and not elliptical:
            elliptical = dict(alpha_deg=0.0, theta_deg=0.0, eps_deg=0.0)
        self.elliptical = dict(elliptical) if elliptical else None
        self.offset_ns = float(offset_ns)
        self.intensity_mwcm2 = float(intensity_mwcm2)
        # 辅助驱动对（仅用于辅助定频率/光强；不参与演化结算）
        self.drive = drive   # dict(g=spec, e=spec)

    # ---------- 定位 ----------
    def start(self):
        """波形绝对起点（ns）。"""
        return self.offset_ns

    def end(self):
        """波形结束绝对时间（包络≈0 的粗略上界）。"""
        e = self.envelope
        if isinstance(e, GaussianEnv):
            return self.start() + e.center_ns + 5 * e.sigma_ns
        if isinstance(e, FileEnv):
            return self.start() + e.t_ns[-1]
        return self.start() + e.fall_start_ns + e.edge_ns

    def waveform(self, t):
        """包络 callable：t→[0,1]，已按 offset 做绝对定位。"""
        t = np.asarray(t, float)
        return self.envelope(t - self.start())

    # ---------- 包络变差（"这段是否恒常"的权威答案） ----------
    def span(self, t0, t1):
        """包络在**绝对时间** [t0,t1] 上的 (inf, sup)；包络不支持解析 ⇒ None。

        消费方见 `field_build.env_span`（它同时处理裸 callable 的兜底）。
        """
        fn = getattr(self.envelope, 'span', None)
        if not callable(fn):
            return None
        return fn(float(t0) - self.offset_ns, float(t1) - self.offset_ns)

    def dt_feature_ns(self):
        """最窄变化特征时长（ns）；包络不支持 ⇒ None。"""
        fn = getattr(self.envelope, 'dt_feature_ns', None)
        return None if not callable(fn) else float(fn())

    def __repr__(self):
        return (f'Wave({self.name!r}, freq={self.freq_mhz:.6g}MHz, '
                f'pol={self.pol}, @{self.start():.1f}ns, {self.envelope})')


class ArbitraryWave:
    """任意波形：用户给出的**完整时域信号**（无单一载频概念）。

    - t_ns / amp ：样品数组（ns / 无量纲幅值，载入时峰值归一 |amp|≤1，
      可负——负幅值即电场反向；任意频谱、啁啾、多频叠加都由此表达）；
    - intensity_mwcm2：峰值光强（决定场绝对幅度：E(t) = E0·amp(t)）；
    - pol / offset_ns：偏振 / 波形绝对起点偏移；无单频 → 无辅助定参：
      drive 恒为 None（频率/失谐概念不适用）。

    **结算路径（t18 起，t74/t78 定型）**：`evolve_twofield.arbitrary_mw_group()`
    对它做解析信号分解（Hilbert）：amp(t) = env(t)·cos(φ(t))，env/φ 打包成
    查表，并入两场 meta 的 `mw_groups`，由 **hard 的 cos(φ_f(t)) 任意相位
    路径**求解。

    **引擎支持面**（现行仅 sambe / hard / mixed 三款）：
      · 微波场支持任意波形 —— hard 直接支持；**sambe / mixed 选中时会自动
        改道 hard**（src/ui/ui_dynamic.py 有日志提示）。
      · 光学场**不支持**任意波形：无载频 → 不参与光场旋转 → RHS 带 THz
        相位 → 积分器被钉死。光学只能用 `FileEnv`（光强包络 + 载频），
        **三款引擎全支持**。

    ⚠ 旧 `method='arbitrary'`（实验室系相互作用绘景 RK4）与 `build_fields`
    管线已于 t74 归档（src/_archive/evolve_arbitrary.py），不要再引用。
    """
    kind = 'arb'

    def __init__(self, name, t_ns, amp, pol='π', intensity_mwcm2=100.0,
                 offset_ns=0.0, elliptical=None):
        assert pol in POLS, f'偏振须为 {POLS}'
        t = np.asarray(t_ns, float)
        a = np.asarray(amp, float)
        if t.ndim != 1 or a.ndim != 1 or t.size != a.size or t.size < 2:
            raise ValueError('任意波形样品须为等长一维数组（至少 2 点）')
        if np.any(np.diff(t) <= 0):
            raise ValueError('任意波形时间样品必须严格递增')
        mx = float(np.max(np.abs(a)))
        if mx <= 0:
            raise ValueError('任意波形幅值全零')
        self.name = name or '任意波形'
        self.t_ns = t
        self.amp = a / mx                 # 峰值归一
        self.pol = pol
        self.intensity_mwcm2 = float(intensity_mwcm2)
        self.offset_ns = float(offset_ns)
        # 椭偏参数（与 Wave 对称）：serialize_waves 会写 'elliptical'、restore_waves 会回传，
        # 缺了这个形参 arb 波**无法往返**（潜在 bug，建场层 arb 支持暴露后补上）。
        self.elliptical = elliptical
        self.drive = None                 # 无单频 → 辅助定参不适用

    # ---- 兼容层（预览/标题/存档共用）----
    @property
    def freq_mhz(self):
        return 0.0

    @property
    def envelope(self):
        """返回包络采样函数（t→|amp(t)|），供 waveforms_to_env 预览。"""
        off = self.offset_ns
        t_s, a_s = self.t_ns, np.abs(self.amp)

        def _env(tau):
            tau = np.asarray(tau, float) - off
            return np.interp(tau, t_s, a_s, left=0.0, right=0.0)
        return _env

    def start(self):
        return self.offset_ns

    def end(self):
        return self.offset_ns + self.t_ns[-1]

    def waveform(self, t):
        """幅值采样（含符号，第一性原理用）：t→a(t)∈[−1,1]。"""
        t = np.asarray(t, float)
        return np.interp(t - self.offset_ns, self.t_ns, self.amp,
                         left=0.0, right=0.0)

    def dt_min_ns(self):
        d = np.diff(self.t_ns)
        d = d[d > 0]
        return float(d.min()) if d.size else 0.0

    def f_max_mhz(self, t_end_ns=None):
        """可表达带宽上限（MHz）：min(文件 Nyquist, 引擎查表 Nyquist)。

        t_end_ns 给定时把引擎相位查表上限一并算进来（否则只报文件 Nyquist，
        会过度承诺——见 `_f_max_mhz`）。
        """
        return _f_max_mhz(self.dt_min_ns(), t_end_ns)

    def __repr__(self):
        return (f'ArbitraryWave({self.name!r}, N={len(self.t_ns)}点, '
                f'dt_min={self.dt_min_ns():.3g}ns, f_max={self.f_max_mhz():.3g}'
                f'MHz, pol={self.pol}, @{self.offset_ns:.1f}ns)')


# ================= 数据源（动态页演化输入） =================
def waveforms_to_env(waves, t_grid):
    """各波形的光强曲线数据（画光强，不画载波）。

    返回 [(label, t, env, freq_mhz, intensity_mwcm2), ...]——每波形一条。
    env 为 0~1 包络，光强曲线 = intensity × env。采样直接用输出网格 t_grid。
    """
    t = np.asarray(t_grid, float)
    out = []
    for w in waves:
        out.append((w.name, t, np.asarray(w.waveform(t), float),
                    float(w.freq_mhz), float(w.intensity_mwcm2)))
    return out


# ================= 状态管理（序列化 / 恢复） =================
def serialize_waves(waves):
    """Wave / ArbitraryWave 列表 → JSON 可序列化 dict。"""
    out = []
    for w in waves:
        if isinstance(w, ArbitraryWave):
            out.append({'type': 'arb', 'name': w.name,
                        't_ns': list(w.t_ns), 'amp': list(w.amp),
                        'pol': w.pol, 'intensity': w.intensity_mwcm2,
            'elliptical': getattr(w, 'elliptical', None),
                        'offset_ns': w.offset_ns})
            continue
        env = w.envelope
        if isinstance(env, GaussianEnv):
            e = ('gauss', env.center_ns, env.sigma_ns, env.peak)
        elif isinstance(env, FileEnv):
            e = ('file', list(env.t_ns), list(env.amp))
        else:
            e = ('square', env.start_ns, env.edge_ns, env.height,
                 env.fall_start_ns)
        d = {'name': w.name, 'freq_mhz': w.freq_mhz,
             'intensity': w.intensity_mwcm2, 'pol': w.pol,
        'elliptical': getattr(w, 'elliptical', None),
             'offset_ns': w.offset_ns, 'env': e, 'drive': w.drive}
        out.append(d)
    return out


def restore_waves(data):
    """serialize_waves 输出 → Wave/ArbitraryWave 列表。"""
    waves = []
    for d in (data or []):
        if d.get('type') == 'arb':
            waves.append(ArbitraryWave(
                d.get('name', '任意波形'), d['t_ns'], d['amp'],
                pol=d.get('pol', 'π'),
        elliptical=d.get('elliptical'),
                intensity_mwcm2=d.get('intensity', 100.0),
                offset_ns=d.get('offset_ns', 0.0)))
            continue
        env = d['env']
        if env[0] == 'gauss':
            env_obj = GaussianEnv(env[1], env[2], env[3])
        elif env[0] == 'file':
            env_obj = FileEnv(env[1], env[2])
        else:
            env_obj = SquareEnv(env[1], env[2], env[3], env[4])
        dr = d.get('drive')
        if dr is not None:
            dr = dict(g=tuple(dr['g']), e=tuple(dr['e']))
        waves.append(Wave(
            d.get('name', '波形'), d['freq_mhz'], env_obj, pol=d['pol'],
            elliptical=d.get('elliptical'),
            offset_ns=d.get('offset_ns', 0.0),
            intensity_mwcm2=d.get('intensity', 100.0), drive=dr))
    return waves
