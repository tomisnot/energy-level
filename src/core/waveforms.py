# -*- coding: utf-8 -*-
"""纯波形数据层：一个波形 = 频率 + 光强 + 偏振 + 包络(高斯|方波) + 起始偏移。

取代旧「TTL 时序 + 波形通道」两层编辑器模型：
  旧：TimingPulseWindow（TTL 触发层）→ WaveChannel（通道层）→ Pulse
  新：直接管理一串 Wave（无通道、无 TTL 触发、无独立窗口）。

每个 Wave 的绝对起点 = offset_ns（ns，相对演化 t=0）。
包络参数：
  高斯：center_ns（相对起点）、sigma_ns、peak(0,1]
  方波：start_ns（相对起点）、edge_ns、height(0,1]、fall_start_ns（缺省 start+2·edge）

数据源接口（对齐旧 ttl_pulse.field_waves / build_fields，供动态页复用）：
  waveforms_to_env(waves, t_grid) -> [(label, t, env, freq_mhz, intensity), ...]
  build_fields(waves, states, cache, near_cutoff_mhz=None) -> fields 列表
  serialize_waves(waves) / restore_waves(data) -> Wave 列表
"""
import numpy as np

from . import evolve

# 与旧 pulse.py 一致的包络形状（数据类抽离 UI，纯计算）
POLS = ['σ+', 'π', 'σ-', '椭圆偏']


def _smooth(x):
    """x∈[0,1] → 0..1 余弦缓变（raised cosine，两端导数连续=0）。"""
    x = np.clip(x, 0.0, 1.0)
    return 0.5 * (1.0 - np.cos(np.pi * x))


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

    def __repr__(self):
        return (f'SquareEnv(start={self.start_ns}ns, '
                f'edge={self.edge_ns}ns, height={self.height}, '
                f'fall_start={self.fall_start_ns}ns)')


class Wave:
    """一个独立波形：频率 + 光强 + 偏振 + 包络 + 起始偏移（无通道无触发）。"""

    def __init__(self, name, freq_mhz, envelope, pol='π',
                 offset_ns=0.0, intensity_mwcm2=100.0, drive=None):
        assert pol in POLS, f'偏振须为 {POLS}'
        self.name = name or '波形'
        self.freq_mhz = float(freq_mhz)
        self.envelope = envelope
        self.pol = pol
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
        return self.start() + e.fall_start_ns + e.edge_ns

    def waveform(self, t):
        """包络 callable：t→[0,1]，已按 offset 做绝对定位。"""
        t = np.asarray(t, float)
        return self.envelope(t - self.start())

    def __repr__(self):
        return (f'Wave({self.name!r}, freq={self.freq_mhz:.6g}MHz, '
                f'pol={self.pol}, @{self.start():.1f}ns, {self.envelope})')


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


def build_fields(waves, states, cache, near_cutoff_mhz=None):
    """由波形列表的「真实光场参数」生成演化输入。

    每波形 (freq_mhz, intensity_mwcm2, pol, waveform) → evolve.real_field：
    自动按自适应失谐判据展开近共振驱动对（Ω 真实、Δ 相对裸差）+
    远失谐 AC 位移。返回 fields 列表（每波形一个多对驱动场，含时变 ac）。
    """
    fields = []
    for w in waves:
        fld, a = evolve.real_field(
            cache, states, w.freq_mhz, w.intensity_mwcm2, w.pol,
            waveform=w.waveform, near_cutoff_mhz=near_cutoff_mhz)
        if fld is not None:
            fields.append(fld)
        elif np.any(np.asarray(a, float) != 0.0):
            acv = np.asarray(a, float)
            fields.append({
                'kind': 'multi', 'M': np.diag(acv),
                'A': np.zeros((len(states), len(states))),
                'ac_shift': acv,
                'g_idx': [], 'e_idx': [], 'omega0_mhz': 1.0,
                'omega_L_mhz': w.freq_mhz, 'waveform': w.waveform})
    return fields


# ================= 状态管理（序列化 / 恢复） =================
def serialize_waves(waves):
    """Wave 列表 → JSON 可序列化 dict。"""
    out = []
    for w in waves:
        env = w.envelope
        if isinstance(env, GaussianEnv):
            e = ('gauss', env.center_ns, env.sigma_ns, env.peak)
        else:
            e = ('square', env.start_ns, env.edge_ns, env.height,
                 env.fall_start_ns)
        d = {'name': w.name, 'freq_mhz': w.freq_mhz,
             'intensity': w.intensity_mwcm2, 'pol': w.pol,
             'offset_ns': w.offset_ns, 'env': e, 'drive': w.drive}
        out.append(d)
    return out


def restore_waves(data):
    """serialize_waves 输出 → Wave 列表。"""
    waves = []
    for d in (data or []):
        env = d['env']
        if env[0] == 'gauss':
            env_obj = GaussianEnv(env[1], env[2], env[3])
        else:
            env_obj = SquareEnv(env[1], env[2], env[3], env[4])
        dr = d.get('drive')
        if dr is not None:
            dr = dict(g=tuple(dr['g']), e=tuple(dr['e']))
        waves.append(Wave(
            d.get('name', '波形'), d['freq_mhz'], env_obj, pol=d['pol'],
            offset_ns=d.get('offset_ns', 0.0),
            intensity_mwcm2=d.get('intensity', 100.0), drive=dr))
    return waves
