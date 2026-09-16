# -*- coding: utf-8 -*-
"""相位印记复现：阶跃 vs 25 ns 上升（细网格 1 ns）——布居/相干随时间的差别从哪来。

三档（run_20260915_175644_000007）：
  step    : 上升沿 = 理想阶跃
  sqrt25  : 25 ns 上升，**光强线性**（包络 ∝ √t）
  lin25   : 25 ns 上升，**幅度线性**（包络 ∝ t）—— 波形形状不同的对照

看点：若 sqrt25/lin25 的交换通道差曲线 D(t) 与 step 是**同形状、整体平移**，
      即为"相位印记"（上升沿做功只改相位、不改当时的布居）。
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
OUT = os.path.join(ROOT, 'ui_design', 'screenshots', 'phase_imprint.png')

TAGS = ['step', 'sqrt25', 'lin25']
BEF = '(60P_{3/2} m_J=-0.5) x (60S_{1/2} m_J=+0.5)'
AFT = '(60S_{1/2} m_J=-0.5) x (60P_{3/2} m_J=+1.5)'

data = {}
# 三档的 job 号从**那次 run 的结果 JSON** 里精确读取（不靠"取最新"猜）
import json
RID = 'run_20260915_175644_000007'
rj = os.path.join(ROOT, 'runtime', 'ai_results', RID + '.json')
TAG2JOB = {}
if os.path.exists(rj):
    j = json.load(io.open(rj, encoding='utf-8'))
    for r in ((j.get('result') or {}).get('rows') or []):
        TAG2JOB[r['tag']] = r.get('job')
print('  tag→job:', TAG2JOB)
for tag in TAGS:
    job = TAG2JOB.get(tag)
    if not job:
        continue
    g = glob.glob(os.path.join(ROOT, 'runtime', 'ai_results',
                               'ai_%s_*.npz' % job))
    if not g:
        continue
    f = sorted(g)[-1]
    dd = np.load(f, allow_pickle=True)
    lab = [str(x['label']) for x in dd['states_2d']]
    t = np.asarray(dd['t'], float)
    P2 = np.real(np.asarray(dd['pops_2d'], float))
    pA = np.real(np.asarray(dd['pops_A'], float))
    data['t'] = t
    data[tag] = dict(f=os.path.basename(f), t=t,
                     before=P2[:, lab.index(BEF)], after=P2[:, lab.index(AFT)],
                     A_P=pA[:, 2:6].sum(axis=1))
print('  已取: ' + ', '.join('%s=%s' % (k, data[k]['f']) for k in TAGS
                             if k in data))

t = data['t']
fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.6))
# (a) 输入差异：光强包络
tr, I3, I4 = 25.0, 2.0e8, 8.3982e7
amp4 = np.sqrt(I4 / I3)
tt = np.linspace(0, 800, 4001)
prof_step = np.where((tt >= 120) & (tt < 280), I3, np.where(tt >= 280, I4, 0.0))
r = np.clip((tt - 120) / tr, 0, 1)
prof25 = I3 * r ** 2 * ((tt < 270) | (tt > 290) | (tt >= 270) * 0)  # 上升(光强线性)
prof25 = np.where(tt < 120, 0.0, np.where(tt < 120 + tr, I3 * r ** 2,
                  np.where(tt < 270, I3, np.where(tt < 290,
                           I3 + (I4 - I3) * (tt - 270) / 20, I4))))
axes[0].plot(tt, prof_step, lw=2, label='step（理想阶跃）')
axes[0].plot(tt, prof25, lw=2, ls='--', label='sqrt25（光强线性上升）')
axes[0].set_xlabel('t (ns)')
axes[0].set_ylabel('光强 (a.u.)')
axes[0].set_title('(a) 三档的**唯一输入差异**：上升沿', fontsize=11)
axes[0].legend(fontsize=9)
axes[0].grid(alpha=0.25)

# (b) 两个交换通道的布居
for tag, st in (('step', '-'), ('sqrt25', '--'), ('lin25', ':')):
    if tag in data:
        axes[1].plot(data[tag]['t'], data[tag]['before'],
                     color='#1f77b4', ls=st, lw=1.7,
                     label='%s: 换前态' % tag)
        axes[1].plot(data[tag]['t'], data[tag]['after'],
                     color='#d62728', ls=st, lw=1.7,
                     label='%s: 换后态' % tag)
axes[1].set_xlabel('t (ns)')
axes[1].set_ylabel('布居')
axes[1].set_title('(b) 交换两通道布居 vs 时间', fontsize=11)
axes[1].legend(fontsize=8)
axes[1].grid(alpha=0.25)

# (c) D(t) = 换后 − 换前，看是否"同形状、整体平移"
for tag, st in (('step', '-'), ('sqrt25', '--'), ('lin25', ':')):
    if tag in data:
        axes[2].plot(data[tag]['t'], data[tag]['after'] - data[tag]['before'],
                     ls=st, lw=1.8, label=tag)
axes[2].axhline(0, color='#888', lw=1)
axes[2].set_xlabel('t (ns)')
axes[2].set_ylabel(r'$D(t)=P_{后}-P_{前}$')
axes[2].set_title('(c) 交换振荡：形状是否相同、整体平移？', fontsize=11)
axes[2].grid(alpha=0.25)
axes[2].legend(fontsize=9)
for ax in axes:
    for x, c in ((70, '#888'), (200, '#888'), (120, '#c62828')):
        ax.axvline(x, color=c, ls='--' if x != 120 else ':', lw=1)
fig.suptitle('相位印记复现：上升沿不改制备布居，改的是**相位** ⇒ 后面交换的振荡相位不同',
             fontsize=12)
fig.tight_layout(rect=(0, 0, 1, 0.92))
fig.savefig(OUT, dpi=130)

# 定量：互相关找时移（窗口 250–800 ns）
w = (t >= 250) & (t <= 800)
if 'step' in data and 'sqrt25' in data:
    a = (data['step']['after'] - data['step']['before'])[w]
    b = (data['sqrt25']['after'] - data['sqrt25']['before'])[w]
    a = a - a.mean(); b = b - b.mean()
    c = np.correlate(b, a, mode='full')
    lag = (np.arange(c.size) - (len(a) - 1)) * (t[1] - t[0])
    shift = float(lag[int(np.argmax(c))])
    print('\n  D(t) 互相关时移（sqrt25 相对 step）= %+.1f ns（正=滞后）' % shift)
    print('  三档的通道布居在 t=120/150/250 ns 处的值：')
    at = lambda arr, tq: float(arr[int(np.argmin(np.abs(t - tq)))])
    for tag in TAGS:
        if tag in data:
            print('    %-7s 前态 %.4f/%.4f/%.4f   后态 %.4f/%.4f/%.4f'
                  % (tag, at(data[tag]['before'], 120), at(data[tag]['before'], 150),
                     at(data[tag]['before'], 250), at(data[tag]['after'], 120),
                     at(data[tag]['after'], 150), at(data[tag]['after'], 250)))
print('\n图 → %s' % OUT)
