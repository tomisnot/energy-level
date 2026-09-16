# -*- coding: utf-8 -*-
"""时间相关分析（第一步）：把 V 与 κ 从引擎里直接算出来，用于建立沿的模型。

模型（两能级交换 + 光移解谐）：
  交换通道 |P_A(-0.5) S_B(+0.5)> ↔ |S_A(-0.5) P_B(+1.5)>，耦合 V（V_dd）；
  光提供微分光移 δ(I) = κ·(I − I_res)；广义 Rabi 频率 Ω=√(V²+δ²)，
  转移幅度 = V²/(V²+δ²)  ⇒ 峰值对"交换窗口内的时间平均失谐"敏感。

本脚本只算：① 通道耦合 V（以及竞争通道）② 各态的微分光移 → κ。

用法：python scripts/tmp/_model_V_kappa.py
"""
import io
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8',
                              errors='replace')

from arc import Rubidium87                                        # noqa: E402
from src.core import static_atom as core                          # noqa: E402
from src.core import pair_core as pc                              # noqa: E402
from src.core.atomic_levels import make_levels                    # noqa: E402
from src.core.evolve_mixed import acize_physical                  # noqa: E402
from src.core.waveforms import Wave, FileEnv                      # noqa: E402

B_GAUSS, R_UM, THETA = 21.7, 14.1, 31.1
I3, I4 = 2.0e8, 8.3982e7
Q_FS = 295724187.476
GROUPS = [{'n': 60, 'l': 0, 'j': 0.5, 'label': '60S_{1/2}', 'color': '#1f77b4'},
          {'n': 60, 'l': 1, 'j': 1.5, 'label': '60P_{3/2}', 'color': '#ff7f0e'},
          {'n': 6, 'l': 1, 'j': 1.5, 'label': '6P_{3/2}', 'color': '#2ca02c'}]

cache = core.ArcCache(Rubidium87())
rows = core._base_level_rows(cache, B_GAUSS, GROUPS)
specs = []
for r in rows:
    lab = '%s %s' % (r['label'], r['qstr'])
    if r['kind'] == 'F':
        specs.append(dict(n=r['n'], l=r['l'], j=r['j'], kind='F', F=r['F'],
                          mF=r['mF'], amps=r['amps'], E_mhz=r['E'], E_hf=0.0,
                          label=lab))
    else:
        specs.append(dict(n=r['n'], l=r['l'], j=r['j'], kind='mJ', m=r['mj'],
                          E_mhz=r['E'], label=lab))
states = make_levels(cache, specs)
N = len(states)
print('  基组 %d 态；R=%g μm, θ=%g°' % (N, R_UM, THETA))

# ---- ① V_dd 通道耦合（两体基，索引 = i*N + j）----
V = pc.vdd_sparse(cache, [dict(s) for s in states], R_UM, THETA, 0.0)
V = V.tocsr()
idx = {(s['label']): i for i, s in enumerate(states)}
ch = {
    '换前 |P_A(-0.5)S_B(+0.5)>': idx['60P_{3/2} m_J=-0.5'] * N
                                 + idx['60S_{1/2} m_J=+0.5'],
    '换后 |S_A(-0.5)P_B(+1.5)>': idx['60S_{1/2} m_J=-0.5'] * N
                                 + idx['60P_{3/2} m_J=+1.5'],
}
ks = list(ch)
i1, i2 = ch[ks[0]], ch[ks[1]]
print('\n  === ① 交换通道直接耦合 ===')
print('    V(换前→换后) = %.4f MHz' % float(np.abs(V[i1, i2])))
print('    （周期 ≈ 1/(2V) = %.0f ns ⇒ 半交换 ≈ %.0f ns）'
      % (1e3 / (2 * abs(V[i1, i2])), 1e3 / (4 * abs(V[i1, i2]))))

# 换前态与其他态的竞争耦合（前 8 大）
col = np.asarray(V[i1].todense()).ravel() if hasattr(V[i1], 'todense') else None
row = V.getrow(i1).toarray().ravel()
order = np.argsort(-np.abs(row))
print('\n  === 换前态耦合最强的 8 个末态（含自身）===')
for i in order[:8]:
    if abs(row[i]) < 1e-9:
        continue
    tag = '  ← 换后' if i == i2 else ''
    print('    %8.4f MHz  %s%s' % (row[i], states[i // N]['label'][:26] + ' × '
                                   + states[i % N]['label'][:26], tag))

# ---- ② 微分光移 → κ ----
print('\n  === ② 频移光对各态的 AC 位移（= κ·I 的基础）===')
w = Wave('fs', Q_FS, FileEnv([0.0, 1.0], [1.0, 1.0]), pol='π',
         intensity_mwcm2=I3)
flds = [dict(kind='exact', omega_L_mhz=Q_FS, A=None, waveform=lambda t: 1.0,
             label='fs', intensity_mwcm2=I3, pol='π')]
try:
    A_new, shifts = acize_physical([s['E_mhz'] for s in states], flds)
    sh = np.asarray(shifts[0], float)
    print('    逐态位移（前 6 个 + 交换相关态），单位 MHz @ I=%.3g：' % I3)
    for i, s in enumerate(states[:6]):
        print('      %-30s %+9.4f' % (s['label'], sh[i]))
    d = (sh[idx['60P_{3/2} m_J=-0.5']] + sh[idx['60S_{1/2} m_J=+0.5']]
         - sh[idx['60S_{1/2} m_J=-0.5']] - sh[idx['60P_{3/2} m_J=+1.5']])
    print('\n    换前/换后两通道的**微分**光移 δ(I3) = %+.4f MHz' % d)
    print('    ⇒ κ = δ/I3 = %+.4e MHz per (mW/cm²)' % (d / I3))
    print('    ⇒ 现行 I4=%.3g 时 δ(I4) = %+.4f MHz'
          % (I4, d * I4 / I3))
    print('    ⇒ 若 I4 是静态引擎选的共振点，则 δ(I4) 应 ≈ 0（实测 %+.4f）'
          % (d * I4 / I3))
except Exception as e:                                  # noqa: BLE001
    print('    acize_physical 调用失败：%s: %s' % (type(e).__name__, e))