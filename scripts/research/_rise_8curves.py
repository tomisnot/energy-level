# -*- coding: utf-8 -*-
"""8 条曲线：4 种上升沿取值 × {P(换前态), P(换后态)}，随时间演化。

τ_r ∈ {0, 50, 75, 100} ns —— 0 = 理想阶跃（参照），其余覆盖实验的 50–100 ns。
颜色区分 τ_r，线型区分通道（实线=换前态，虚线=换后态）。
左：全窗 [0,800]；右：放大 [240,480]（差别显现处）。
"""
import glob
import io
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
OUT = os.path.join(ROOT, 'ui_design', 'screenshots', 'rise_8curves.png')

JOBS = {0.0: 'job3', 50.0: 'job5', 75.0: 'job6', 100.0: 'job7'}
BEF = '(60P_{3/2} m_J=-0.5) x (60S_{1/2} m_J=+0.5)'   # 换前态
AFT = '(60S_{1/2} m_J=-0.5) x (60P_{3/2} m_J=+1.5)'   # 换后态
COLORS = {0.0: '#111111', 50.0: '#1f77b4', 75.0: '#2ca02c', 100.0: '#d62728'}

D = {}
for tr, job in JOBS.items():
    g = glob.glob(os.path.join(ROOT, 'runtime', 'ai_results',
                               'ai_%s_*.npz' % job))
    if not g:
        print('  缺 %s 的 npz' % job)
        continue
    d = np.load(sorted(g)[-1], allow_pickle=True)
    lab = [str(x['label']) for x in d['states_2d']]
    P2 = np.real(np.asarray(d['pops_2d'], float))
    D[tr] = dict(t=np.asarray(d['t'], float),
                 bef=P2[:, lab.index(BEF)], aft=P2[:, lab.index(AFT)],
                 f=os.path.basename(sorted(g)[-1]))

fig, axes = plt.subplots(1, 2, figsize=(14.5, 5.4),
                         gridspec_kw={'width_ratios': [1.45, 1]})
for tr in sorted(D):
    t, b, a, c = D[tr]['t'], D[tr]['bef'], D[tr]['aft'], COLORS[tr]
    lab = 'τ=%.0f' % tr + ('（阶跃）' if tr == 0 else ' ns')
    axes[0].plot(t, b, color=c, ls='-', lw=1.9, label='%s：P(换前态)' % lab)
    axes[0].plot(t, a, color=c, ls='--', lw=1.9, label='%s：P(换后态)' % lab)
    axes[1].plot(t, b, color=c, ls='-', lw=2.0)
    axes[1].plot(t, a, color=c, ls='--', lw=2.0)

for ax in axes:
    for x in (70, 200):
        ax.axvline(x, color='#999', ls='--', lw=1.1)
    ax.axvline(120, color='#c62828', ls=':', lw=1.2)
    ax.axvspan(270, 290, color='#2ca02c', alpha=0.13)
    ax.set_xlabel('t (ns)')
    ax.set_ylabel('布居')
    ax.grid(alpha=0.25)
axes[0].set_xlim(0, 800)
axes[0].set_ylim(-0.02, 1.02)
axes[0].set_title('全窗 [0, 800] ns（8 条曲线 = 4 种上升沿 × 2 个交换通道）',
                  fontsize=11)
axes[1].set_xlim(240, 480)
axes[1].set_title('放大 [240, 480] ns：差别正好从绿色区（衔接下降沿）之后显现',
                  fontsize=11)
axes[1].annotate('换后态在此起飞\n（交换被"打开"）', xy=(300, 0.05), fontsize=9,
                 color='#555')
axes[0].legend(fontsize=8.5, ncol=2, loc='upper right')
fig.suptitle('交换两通道布居随时间：换前态 = P[(60P, m=-0.5)x(60S, m=+0.5)]，'
             '换后态 = P[(60S, m=-0.5)x(60P, m=+1.5)]；'
             '灰虚线 = 微波 pi 脉冲 @70/200 ns，红点线 = 光起 120 ns，'
             '绿区 = 衔接下降沿 270-290 ns', fontsize=10.5)
fig.tight_layout(rect=(0, 0, 1, 0.93))
fig.savefig(OUT, dpi=135)

print('  曲线来源：' + '; '.join('τ=%g→%s' % (k, D[k]['f']) for k in sorted(D)))
at = lambda t, a, tq: float(a[int(np.argmin(np.abs(t - tq)))])
print('\n  %-8s %10s %10s %10s %10s' % ('τ_r(ns)', 'P前(280)', 'P后(300)',
                                        'P前(末)', 'P后(末)'))
for tr in sorted(D):
    t, b, a = D[tr]['t'], D[tr]['bef'], D[tr]['aft']
    print('  %-8g %10.4f %10.4f %10.4f %10.4f'
          % (tr, at(t, b, 280), at(t, a, 300), float(b[-1]), float(a[-1])))
print('\n图 → %s' % OUT)