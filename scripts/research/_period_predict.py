# -*- coding: utf-8 -*-
"""从旧几何(job3, R=12/θ=30)测交换振荡周期，给新几何(R=14.1/θ=31.1)的可证伪预言。

V_dd(θ,R) ∝ (1−3cos²θ)/R³：
  θ=30° → 1−3cos²= −1.25 ； θ=31.1° → −1.1993
  R：12³=1728 ； 14.1³=2803.2
  比值 = (−1.1993/2803.2)/(−1.25/1728) = 0.5915
交换频率 ∝ V_dd ⇒ 周期应放大约 1/0.5915 = 1.69 倍。
"""
import glob
import io
import os
import sys

import numpy as np

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8',
                              errors='replace')
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BEF = '(60P_{3/2} m_J=-0.5) x (60S_{1/2} m_J=+0.5)'
AFT = '(60S_{1/2} m_J=-0.5) x (60P_{3/2} m_J=+1.5)'

ratio = ((-1.1993 / 2803.2) / (-1.25 / 1728.0))
print('  V_dd(新)/V_dd(旧) = %.4f  ⇒ 周期比应为 %.3f'
      % (ratio, 1 / ratio))

g = glob.glob(os.path.join(ROOT, 'runtime', 'ai_results', 'ai_job3_*.npz'))
d = np.load(sorted(g)[-1], allow_pickle=True)
lab = [str(x['label']) for x in d['states_2d']]
t = np.asarray(d['t'], float)
P2 = np.real(np.asarray(d['pops_2d'], float))
a = P2[:, lab.index(AFT)]
b = P2[:, lab.index(BEF)]
# 找 a 的局部极大（t>=300，避开光开着的窗口）
idx = [i for i in range(1, len(a) - 1)
       if a[i] >= a[i - 1] and a[i] > a[i + 1] and t[i] >= 300]
pk = [(float(t[i]), float(a[i])) for i in idx]
print('\n  旧几何 换后态 的局部极大（t≥300 ns）：')
for tt, vv in pk:
    print('    t=%6.0f ns   P=%.4f' % (tt, vv))
if len(pk) >= 2:
    per = np.diff([p[0] for p in pk])
    print('\n  相邻峰间隔: %s ns  → 平均周期 ≈ %.0f ns'
          % (np.round(per, 1).tolist(), float(per.mean())))
    print('  ⇒ 新几何周期预言 ≈ %.0f ns（放大约 1.69 倍）'
          % (float(per.mean()) / ratio))
# 换前态峰
idx2 = [i for i in range(1, len(b) - 1)
        if b[i] >= b[i - 1] and b[i] > b[i + 1] and t[i] >= 300]
pk2 = [(float(t[i]), float(b[i])) for i in idx2]
if len(pk2) >= 2:
    per2 = np.diff([p[0] for p in pk2])
    print('  （换前态峰间隔 %s → 平均 %.0f ns）'
          % (np.round(per2, 1).tolist(), float(per2.mean())))
result = {'V_ratio': float(ratio), 'period_old_ns': float(np.diff([p[0] for p in pk]).mean()) if len(pk) >= 2 else None,
          'period_new_pred_ns': float(np.diff([p[0] for p in pk]).mean() / ratio) if len(pk) >= 2 else None}