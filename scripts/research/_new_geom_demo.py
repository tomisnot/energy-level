# -*- coding: utf-8 -*-
"""新几何（R=14.1um, theta=31.1deg）下用**布居随时间演化**论证两条机制。

论证目标：
  (1) 交换峰值由"能参与交换的布居"决定 —— 看 B:60P(+0.5) 的漏出（第二支 π 期间）
  (2) 上升沿 → 改 200–250 ns 处光强 → 改漏出 → 改峰值
  (3) 下降沿 → 改交换进行时的光移 → 改峰值与周期（已由旧几何扫描单独证明）

图：(a) 光强包络  (b) B:60P(+0.5) 漏出 vs 时间  (c) 交换两通道 vs 时间
    (d) 漏出峰值 vs 交换峰值（散点，新几何三点）
"""
import glob
import io
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8',
                              errors='replace')
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RID = 'run_20260915_203117_000009'
OUT = os.path.join(ROOT, 'ui_design', 'screenshots', 'new_geom_mechanism.png')
BEF = '(60P_{3/2} m_J=-0.5) x (60S_{1/2} m_J=+0.5)'
AFT = '(60S_{1/2} m_J=-0.5) x (60P_{3/2} m_J=+1.5)'
COL = {0.0: '#111111', 50.0: '#1f77b4', 100.0: '#d62728'}

j = json.load(io.open(os.path.join(ROOT, 'runtime', 'ai_results', RID + '.json'),
                      encoding='utf-8'))
rows = (j.get('result') or {}).get('rows') or []
print('  R = %s um, theta = %s deg' % ((j.get('result') or {}).get('R_um'),
                                       (j.get('result') or {}).get('theta_deg')))
S = {}
for r in rows:
    g = glob.glob(os.path.join(ROOT, 'runtime', 'ai_results',
                               'ai_%s_*.npz' % r['job']))
    if not g:
        continue
    d = np.load(sorted(g)[-1], allow_pickle=True)
    lab = [str(x['label']) for x in d['states_2d']]
    t = np.asarray(d['t'], float)
    A = np.real(np.asarray(d['pops_A'], float))
    B = np.real(np.asarray(d['pops_B'], float))
    P2 = np.real(np.asarray(d['pops_2d'], float))
    a, b = P2[:, lab.index(AFT)], P2[:, lab.index(BEF)]
    out = 0.5 * (A[:, [2, 4]].sum(axis=1) + B[:, [2, 4]].sum(axis=1))
    S[r['tr']] = dict(t=t, a=a, b=b, out=out, Bp=B[:, 4], job=r['job'],
                      pk=float(a.max()), t_pk=float(t[int(np.argmax(a))]),
                      out_max=float(out.max()),
                      out_250=float(B[int(np.argmin(np.abs(t - 250))), 4]))

fig, axes = plt.subplots(1, 4, figsize=(19.5, 4.5))
# (a) 光强包络
tt = np.linspace(0, 800, 8001)
I3, I4, tr0 = 2.0e8, 8.3982e7, 50.0
p0 = np.where((tt >= 120) & (tt < 280), I3, np.where(tt >= 280, I4, 0.0))
r50 = np.clip((tt - 120) / tr0, 0, 1)
p50 = np.where(tt < 120, 0.0, np.where(tt < 120 + tr0, I3 * r50 ** 2,
               np.where(tt < 270, I3, np.where(tt < 290,
                        I3 + (I4 - I3) * (tt - 270) / 20, I4))))
r100 = np.clip((tt - 120) / 100.0, 0, 1)
p100 = np.where(tt < 120, 0.0, np.where(tt < 220, I3 * r100 ** 2,
                np.where(tt < 270, I3, np.where(tt < 290,
                         I3 + (I4 - I3) * (tt - 270) / 20, I4))))
axes[0].plot(tt, p0, color=COL[0.0], lw=2, label='τ=0')
axes[0].plot(tt, p50, color=COL[50.0], lw=2, ls='--', label='τ=50 ns')
axes[0].plot(tt, p100, color=COL[100.0], lw=2, ls=':', label='τ=100 ns')
axes[0].axvspan(200, 250, color='#ff9800', alpha=0.18)
axes[0].annotate('第二支 π\n(漏出窗口)', xy=(225, I3 * 0.5), fontsize=9,
                 ha='center', color='#e65100')
axes[0].axvspan(270, 290, color='#2ca02c', alpha=0.15)
axes[0].set_xlabel('t (ns)')
axes[0].set_ylabel('光强 (a.u.)')
axes[0].set_title('(a) 三档的光强包络（唯一差别）', fontsize=11)
axes[0].legend(fontsize=9)
axes[0].grid(alpha=0.25)

# (b) 漏出
for tr in sorted(S):
    axes[1].plot(S[tr]['t'], S[tr]['Bp'], color=COL[tr], lw=1.9,
                 label='τ=%g：B:60P(+0.5)' % tr)
    axes[1].plot(S[tr]['t'], S[tr]['out'], color=COL[tr], lw=1.1, alpha=0.5,
                 ls='--', label='τ=%g：两原子其它 m 合计' % tr)
axes[1].axvspan(200, 250, color='#ff9800', alpha=0.15)
axes[1].set_xlabel('t (ns)')
axes[1].set_ylabel('布居')
axes[1].set_title('(b) 跑出工作子能级：峰在 250 ns（π 期间）', fontsize=11)
axes[1].legend(fontsize=8)
axes[1].grid(alpha=0.25)

# (c) 交换两通道
for tr in sorted(S):
    axes[2].plot(S[tr]['t'], S[tr]['b'], color=COL[tr], lw=1.9,
                 label='τ=%g：换前态' % tr)
    axes[2].plot(S[tr]['t'], S[tr]['a'], color=COL[tr], lw=1.9, ls='--',
                 label='τ=%g：换后态' % tr)
axes[2].set_xlabel('t (ns)')
axes[2].set_ylabel('布居')
axes[2].set_title('(c) 交换两通道：峰值高度与时刻', fontsize=11)
axes[2].legend(fontsize=8, ncol=2)
axes[2].grid(alpha=0.25)

# (d) 漏出 vs 峰值
for tr in sorted(S):
    axes[3].plot(S[tr]['out_max'], S[tr]['pk'], 'o', ms=11, color=COL[tr])
    axes[3].annotate('τ=%g' % tr, (S[tr]['out_max'], S[tr]['pk']),
                     textcoords='offset points', xytext=(8, 6), fontsize=10)
axes[3].set_xlabel('max 跑出工作子能级（布居）')
axes[3].set_ylabel('交换峰值 max P(换后态)')
axes[3].set_title('(d) 漏得越多 → 峰值越低？', fontsize=11)
axes[3].grid(alpha=0.25)

for ax in axes[1:3]:
    for x in (70, 200):
        ax.axvline(x, color='#999', ls='--', lw=1)
    ax.axvline(120, color='#c62828', ls=':', lw=1)
fig.suptitle('新几何 R=14.1 μm, θ=31.1°：漏出（第二支 π 期间）与交换峰值的关系',
             fontsize=12)
fig.tight_layout(rect=(0, 0, 1, 0.92))
fig.savefig(OUT, dpi=132)

print('\n  %-7s %-6s %10s %8s %12s %12s %12s'
      % ('τ_r', 'job', 'maxP(后)', '@t', 'max漏(其它m)', 'B:60P+0.5@250',
         '换后@800'))
for tr in sorted(S):
    print('  %-7g %-6s %10.4f %8.0f %12.4f %12.4f %12.4f'
          % (tr, S[tr]['job'], S[tr]['pk'], S[tr]['t_pk'], S[tr]['out_max'],
             S[tr]['out_250'], float(S[tr]['a'][-1])))
print('\n图 → %s' % OUT)