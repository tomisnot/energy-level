# -*- coding: utf-8 -*-
"""四联图：上升沿 τ_r 如何最终改变交换（用已有五档 4 ns 网格数据）。

(a) 输入差异：光强包络（阶跃 vs 25 ns 上升；衔接处 270→290 下降沿**五档相同**）
(b) 换前/换后两通道布居 vs 时间（τ=0 与 τ=25 对照）
(c) D(t)=P(换后)−P(换前) 全窗
(d) D(t) 在 200–400 ns 的放大——**差别出现的时刻**
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
OUT = os.path.join(ROOT, 'ui_design', 'screenshots', 'rise_onset.png')
JOBS = {0.0: 'job3', 25.0: 'job4', 50.0: 'job5', 75.0: 'job6', 100.0: 'job7'}
BEF = '(60P_{3/2} m_J=-0.5) x (60S_{1/2} m_J=+0.5)'
AFT = '(60S_{1/2} m_J=-0.5) x (60P_{3/2} m_J=+1.5)'

D = {}
for tr, job in JOBS.items():
    g = glob.glob(os.path.join(ROOT, 'runtime', 'ai_results',
                               'ai_%s_*.npz' % job))
    if not g:
        continue
    d = np.load(sorted(g)[-1], allow_pickle=True)
    lab = [str(x['label']) for x in d['states_2d']]
    P2 = np.real(np.asarray(d['pops_2d'], float))
    D[tr] = dict(t=np.asarray(d['t'], float),
                 b=P2[:, lab.index(BEF)], a=P2[:, lab.index(AFT)],
                 A_P=np.real(np.asarray(d['pops_A'], float))[:, 2:6].sum(axis=1))
t = D[0.0]['t']

fig, axes = plt.subplots(1, 4, figsize=(19, 4.4))
# (a) 输入
tr, I3, I4 = 25.0, 2.0e8, 8.3982e7
tt = np.linspace(0, 800, 8001)
prof_s = np.where((tt >= 120) & (tt < 280), I3, np.where(tt >= 280, I4, 0.0))
r = np.clip((tt - 120) / tr, 0, 1)
prof_25 = np.where(tt < 120, 0.0,
                   np.where(tt < 120 + tr, I3 * r ** 2,
                            np.where(tt < 270, I3,
                                     np.where(tt < 290,
                                              I3 + (I4 - I3) * (tt - 270) / 20,
                                              I4))))
axes[0].plot(tt, prof_s, lw=2.2, label='τ=0（阶跃）')
axes[0].plot(tt, prof_25, lw=2.2, ls='--', label='τ=25 ns（光强线性上升）')
axes[0].axvspan(120, 145, color='#c62828', alpha=0.12)
axes[0].annotate('唯一输入差异\n（上升沿 120→145）', xy=(132, I3 * 0.55),
                 fontsize=9, ha='center', color='#c62828')
axes[0].axvspan(270, 290, color='#2ca02c', alpha=0.15)
axes[0].annotate('衔接下降沿\n（五档**相同**）', xy=(280, I3 * 0.25), fontsize=9,
                 ha='center', color='#2ca02c')
axes[0].set_xlabel('t (ns)')
axes[0].set_ylabel('光强 (a.u.)')
axes[0].set_title('(a) 输入：唯一差别在上升沿', fontsize=11)
axes[0].legend(fontsize=9)
axes[0].grid(alpha=0.25)

# (b) 两通道布居（τ=0 / τ=25）
for tr_, c, ls in ((0.0, '#1f77b4', '-'), (25.0, '#d62728', '--')):
    if tr_ in D:
        axes[1].plot(t, D[tr_]['b'], color=c, ls=ls, lw=1.8,
                     label='τ=%g：换前态' % tr_)
        axes[1].plot(t, D[tr_]['a'], color=c, ls=ls, lw=1.2, alpha=0.75,
                     label='τ=%g：换后态' % tr_)
axes[1].set_xlabel('t (ns)')
axes[1].set_ylabel('布居')
axes[1].set_title('(b) 两个交换通道布居 vs 时间', fontsize=11)
axes[1].legend(fontsize=8)
axes[1].grid(alpha=0.25)

# (c)(d) D(t)
for ax, xlim, ttl in ((axes[2], None, '(c) D(t)=P(换后)−P(换前) 全窗'),
                      (axes[3], (200, 420), '(d) 放大 200–420 ns：差别从哪出')):
    for tr_, ls in ((0.0, '-'), (25.0, '--'), (50.0, '-.'), (75.0, ':'),
                    (100.0, (0, (3, 1, 1, 1)))):
        if tr_ in D:
            ax.plot(t, D[tr_]['a'] - D[tr_]['b'], ls=ls, lw=1.8,
                    label='τ=%g' % tr_)
    ax.axhline(0, color='#888', lw=1)
    ax.axvline(290, color='#2ca02c', lw=1.4, alpha=0.7)
    ax.set_xlabel('t (ns)')
    ax.set_title(ttl, fontsize=11)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    if xlim:
        ax.set_xlim(*xlim)
        ax.annotate('衔接下降沿 结束处\n差别自此显现', xy=(295, 0.15), fontsize=9,
                    color='#2ca02c')
fig.suptitle('上升沿如何改变交换：布居在 τ≈290 ns 之前几乎相同，'
             '差别**恰在衔接下降沿之后**显现（相位印记 → 被跳变读出）', fontsize=12)
fig.tight_layout(rect=(0, 0, 1, 0.90))
fig.savefig(OUT, dpi=130)
print('图 → %s' % OUT)