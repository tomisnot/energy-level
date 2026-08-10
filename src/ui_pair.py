# -*- coding: utf-8 -*-
"""响应谱画布 + 统一响应谱计算（两页共用，src 包）。

- ResponsePlot：响应谱画布（x = 相对扫频光频率的失谐 MHz，y = 归一化强度）
- response_curves：统一响应谱计算（n_body=1 单体 / 2 两体，同一引擎）。
  响应谱对"关注跃迁"（drive_idx 对应光）扫频，其余光作能移（ac_stark），
  静电场 dc_stark_vpcm 作 DC-Stark 能移。
"""
import numpy as np
from PyQt5.QtWidgets import *
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

from . import driver

COLOR_CYCLE = [
    '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
    '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
    '#aec7e8', '#ffbb78', '#98df8a', '#ff9896', '#c5b0d5',
]


class ResponsePlot(QWidget):
    """响应谱画布：x = 相对扫频光频率的失谐 (MHz)，y = 归一化强度。"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.fig = Figure(figsize=(7, 5), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.fig.subplots_adjust(left=0.1, right=0.97, top=0.92, bottom=0.12)
        self.canvas = FigureCanvas(self.fig)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(self.canvas)
        self.peaks = []   # [(光描述, 失谐MHz, 频率Hz, 强度), ...]

    def render(self, curves, title):
        """curves: [(x_mhz, y_norm, color, label), ...]；同时填充 self.peaks。"""
        ax = self.ax
        ax.clear()
        self.peaks = []
        for x, y, c, lab in curves:
            ax.plot(x, y, color=c, lw=1.0, label=lab)
        ax.axhline(0, color='#888888', lw=0.5)
        ax.axvline(0, color='#888888', lw=0.5, ls='--')
        ax.set_xlabel('相对光频率失谐 (MHz)')
        ax.set_ylabel('线性响应强度 (归一化)')
        if curves:
            ax.legend(fontsize=8, loc='upper right')
        if title:
            ax.set_title(title, fontsize=10)
        self.canvas.draw_idle()


def response_curves(cache, groups, lights, psi_ket, n_body, R_um=None,
                    B_gauss=0.0, theta_deg=0.0, phi_deg=0.0, m_total=None,
                    energy_window_mhz=None, ref_energy_mhz=None,
                    window_mhz=200.0, drive_idx=0, dc_stark_vpcm=0.0):
    """统一响应谱计算：返回 (curves, peaks, system)。

    响应谱对"关注跃迁"（drive_idx 对应光）扫频；其余光全部作能移
    （static ac_stark）；静电场 dc_stark_vpcm 作 DC-Stark 能移。
    否则共振光作能移会把自己 dressed 推走（0 峰）。

    curves: [(x_mhz, y_norm, color, label), ...] 供画布
    peaks : [(label, 失谐MHz, freq_hz, strength), ...] 供峰表
    system: build_system 结果（供后续 CW 演化复用）
    """
    if not lights:
        return [], [], None
    if drive_idx >= len(lights) or drive_idx < 0:
        drive_idx = 0
    drive = lights[drive_idx]
    shifts = [lg for k, lg in enumerate(lights) if k != drive_idx]
    static = {}
    if shifts:
        static['ac_stark'] = [{**lg, 'enabled': True} for lg in shifts]
    if dc_stark_vpcm:
        static['dc_stark'] = dc_stark_vpcm
    if n_body == 1:
        system = driver.build_system(cache, groups, 1, B_gauss=B_gauss,
                                     static_cfg=static)
    else:
        system = driver.build_system(
            cache, groups, 2, R_um=R_um, B_gauss=B_gauss,
            static_cfg=dict(static, vdw=True, exchange=True),
            theta_deg=theta_deg, phi_deg=phi_deg, m_total=m_total,
            energy_window_mhz=energy_window_mhz, ref_energy_mhz=ref_energy_mhz)
    dc = dict(kind='mw', group_a=dict(n=groups[0]['n'], l=groups[0]['l'],
                                      j=groups[0]['j']),
              group_b=dict(n=groups[-1]['n'], l=groups[-1]['l'],
                           j=groups[-1]['j']),
              pol=drive['polarization'], om_mhz=1.0,
              scope=drive.get('scope', 'both'))
    try:
        lin = driver.response_spectrum_linear(system, dc, psi_ket)
    except Exception:
        return [], [], system
    f = lin['freq_hz']
    if f.size == 0:
        return [], [], system
    center = drive['frequency_hz']
    w = window_mhz * 1e6
    mask = np.abs(f - center) <= w
    if not np.any(mask):
        return [], [], system
    x = (f[mask] - center) / 1e6
    s = lin['strength'][mask]
    smax = s.max()
    y = s / smax if smax > 0 else s
    col = COLOR_CYCLE[0]
    lab = (f'{drive["polarization"]} {drive.get("scope", "both")}'
           f' I={drive["intensity"]:g}')
    curves = [(x, y, col, lab)]
    peaks = [(lab, float(xi), float(fi), float(si))
             for xi, fi, si in zip(x, f[mask], s)]
    return curves, peaks, system
