# -*- coding: utf-8 -*-
"""60S1/2 m=±0.5 的 AC-Stark 位移 vs 寻址光强（direct 对角化后端）。

寻址光：ν = 295724.048 GHz，偏振 σ⁻，B = 21.7 G，强度 0 → 5.6e8 mW/cm²。
基组：6P3/2 + 60S（6P3/2 是与 60S 近共振的耦合伙伴，位移由此产生）。

计算：core.direct_ac_stark_sweep —— 一次构建 Floquet 耦合结构（耦合 ∝ √I 重标定），
对全部光强点批量 direct 对角化；基组窗口取软件默认（win=3 全基组）。
每次对角化约 1.3s，故采样 33 点（含 I=0）：低区对数步长捕捉近共振快变，
高区线性步长覆盖 1e7→5.6e8 的饱和段。
"""
import os
import sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
import time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from arc import Rubidium87
import core
_DIR = os.path.dirname(os.path.abspath(__file__))  # 输出到脚本所在目录

FREQ_HZ = 295724.048e9      # 寻址光频率 (Hz)
POL = 'σ-'
B_G = 21.7                  # 磁场 (G)
I_MAX = 5.6e8               # 最大光强 (mW/cm²)

atom = Rubidium87()
cache = core.ArcCache(atom)
groups = [
    {'n': 6, 'l': 1, 'j': 1.5, 'label': '6P_{3/2}', 'color': '#1f77b4'},
    {'n': 60, 'l': 0, 'j': 0.5, 'label': '60S_{1/2}', 'color': '#ff7f0e'},
]
cfg = {'enabled': True, 'intensity': 1.0,
       'frequency_hz': FREQ_HZ, 'polarization': POL}

# ---- 混合采样：低区对数（近共振快速变化）+ 高区线性（饱和段）----
I_log = np.geomspace(1e3, 1e7, 18)
I_lin = np.linspace(1e7, I_MAX, 15)[1:]        # 去掉与 I_log 末点重复的 1e7
intensities = np.concatenate(([0.0], I_log, I_lin))

t0 = time.time()
keys, shifts = core.direct_ac_stark_sweep(
    cache, groups, cfg, B_G, intensities,
    sos_n_window=core.DIRECT_SOS_N_WINDOW)     # 软件默认全基组
print(f'基组 {len(keys)} 个子能级 × {len(intensities)} 个光强点，'
      f'耗时 {time.time() - t0:.1f}s', flush=True)


def extract(label, qstr):
    for k in keys:
        if core.auto_label(k[0], k[1], k[2]) == label and k[3] == qstr:
            return shifts[k]
    raise ValueError(f'找不到 {label} {qstr}')


sm = extract('60S_{1/2}', 'm_J=-0.5')    # MHz
sp = extract('60S_{1/2}', 'm_J=+0.5')    # MHz

# ---- 绘图：左线性（0→I_MAX，用户要求的范围）、右对数 ----
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 5))
for ax in (ax1, ax2):
    ax.plot(intensities, sm, 'o-', ms=3, lw=1.5, label='60S m_J=-0.5', color='#d62728')
    ax.plot(intensities, sp, 's-', ms=3, lw=1.5, label='60S m_J=+0.5', color='#1f77b4')
    ax.set_xlabel('Intensity (mW/cm²)')
    ax.set_ylabel('AC Stark shift (MHz)')
    ax.grid(True, alpha=0.3)
    ax.legend()
ax1.set_xlim(0, I_MAX)
ax1.set_title('linear x  (0 → 5.6e8)')
ax2.set_xscale('log')
ax2.set_title('log x')
fig.suptitle(f'60S AC shift vs intensity  |  ν={FREQ_HZ/1e9:.3f} GHz, '
             f'σ-, B={B_G} G, direct 对角化')
fig.tight_layout()
out_png = os.path.join(_DIR, 'ac_shift_60S.png')
fig.savefig(out_png, dpi=150)
print(f'已保存 {out_png}', flush=True)

# ---- 代表性数值 ----
print('\n代表性光强下的 60S m=±0.5 位移：', flush=True)
for i in (0, 5, 10, 14, 18, 22, -1):
    I = intensities[i]
    print(f'I = {I:12.4g} mW/cm²   Δν(m=-0.5) = {sm[i]:+10.4f} MHz'
          f'   Δν(m=+0.5) = {sp[i]:+10.4f} MHz', flush=True)
print('完成', flush=True)
