# -*- coding: utf-8 -*-
"""图10：非理想因素的敏感性（三组判决 + 偏振细扫）。"""
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
plt.rcParams['font.size'] = 11
plt.rcParams['axes.linewidth'] = 1.0
plt.rcParams['savefig.bbox'] = 'tight'
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TMP = os.path.join(ROOT, 'scripts', 'tmp')
FIG = os.path.join(ROOT, 'ui_design', 'screenshots', 'edge_study')
RED, BLUE, GREEN, ORANGE, GREY = ('#d62728', '#1f77b4', '#2ca02c', '#ff7f0e',
                                  '#7f7f7f')


def L(n):
    return json.load(io.open(os.path.join(TMP, n), encoding='utf-8'))


def main():
    R = L('_nonideal_v2_res.json')
    by = {r['tag']: r for r in R['rows']}
    base = by['P0/pi基准']
    PM = L('_nonideal_pol_model.json')

    fig, axes = plt.subplots(1, 3, figsize=(16.4, 5.0))

    # ---- (a) 同一积分、不同形状 ----
    ax = axes[0]
    tags = ['S1/lin_chi14.5', 'S2/exp_chi14.5', 'S3/step_chi14.5']
    lbl = ['线性\n(基准)', '指数缓降', '阶梯']
    Q = [5.6, 4.2, 8.4]
    fr = [by[t]['frac'] for t in tags]
    aa = [by[t]['A0'] for t in tags]
    xx = np.arange(3)
    ax.bar(xx - 0.2, fr, 0.38, color=GREEN, alpha=0.9, label=r'两通道分配 $a/s$')
    ax.bar(xx + 0.2, aa, 0.38, color=BLUE, alpha=0.9, label=r'交换振幅 $A_0$')
    for i, (f_, a_, q_) in enumerate(zip(fr, aa, Q)):
        ax.text(i - 0.2, f_ + 0.012, '%.3f' % f_, ha='center', fontsize=9)
        ax.text(i + 0.2, a_ + 0.012, '%.3f' % a_, ha='center', fontsize=9)
    ax.set_xticks(xx)
    ax.set_xticklabels(['%s\n$\\int\\delta^2$=%.1f' % (l, q)
                        for l, q in zip(lbl, Q)])
    ax.set_ylabel('值')
    ax.set_title('(a) 同一"光强积分 $\\int I\\,dt$"，不同时间结构\n'
                 '（$\\int\\delta\\,dt$ 三者相同）', fontsize=11.5)
    ax.legend(fontsize=9); ax.grid(alpha=0.2, axis='y')
    ax.annotate('$a/s$ 变 16.9%\n$A_0$ 变 22.7%', xy=(0.5, 0.42),
                xycoords='axes fraction', fontsize=9.5, color=RED,
                ha='center')

    # ---- (b) 张量光移：偏振细扫 ----
    ax = axes[1]
    rows = sorted(PM['rows'], key=lambda r: r['alpha'])
    al = np.array([r['alpha'] for r in rows])
    fr2 = np.array([r['frac'] for r in rows])
    f0 = np.array([r['f0'] * 1000 for r in rows])
    ca = np.array([r['A0'] + r['C'] for r in rows])
    ax.plot(al, fr2, 'o-', color=GREEN, lw=2, ms=6, label=r'$a/s$（分配）')
    ax.plot(al, ca / ca.max() * fr2.max(), 's--', color=BLUE, lw=1.8, ms=5,
            label=r'$C+A_0$（归一，交换质量）')
    ax.set_xlabel(r'椭圆偏角 $\alpha$  (°)')
    ax.set_ylabel(r'$a/s$')
    ax.set_title('(b) 偏振的"张量光移"改分配：\n'
                 '1° 即 +6.5%，但 $C+A_0$ 掉 7.6%', fontsize=11.5)
    ax.legend(fontsize=9, loc='center right'); ax.grid(alpha=0.22)
    ax2 = ax.twinx()
    ax2.plot(al, f0, '^:', color=ORANGE, lw=1.6, ms=5)
    ax2.set_ylabel(r'主频 $f_0$  (MHz)', color=ORANGE)
    ax2.tick_params(axis='y', colors=ORANGE)
    ax2.annotate('主频被拖低 1.9%\n（张量项的指纹）', xy=(8, f0[6]),
                 xytext=(0.08, 0.16), textcoords='axes fraction',
                 fontsize=9, color=ORANGE,
                 arrowprops=dict(arrowstyle='->', color=ORANGE, lw=1.1))

    # ---- (c) 敏感性排序 ----
    ax = axes[2]
    names = ['光脉冲\n时间结构', '张量光移\n(偏振)', '微波边沿\n形状']
    dfr = [16.91, 19.22, 1.95]
    daa = [22.71, 10.31, 22.50]
    xx = np.arange(3)
    ax.bar(xx - 0.2, dfr, 0.38, color=GREEN, alpha=0.9, label=r'$|\Delta(a/s)|$ (%)')
    ax.bar(xx + 0.2, daa, 0.38, color=BLUE, alpha=0.9, label=r'$|\Delta A_0|$ (%)')
    for i, (d1, d2) in enumerate(zip(dfr, daa)):
        ax.text(i - 0.2, d1 + 0.5, '%.1f' % d1, ha='center', fontsize=9)
        ax.text(i + 0.2, d2 + 0.5, '%.1f' % d2, ha='center', fontsize=9)
    ax.set_xticks(xx); ax.set_xticklabels(names)
    ax.set_ylabel('相对基准的最大变化 (%)')
    ax.set_ylim(0, 28)
    ax.set_title('(c) 敏感性排序：都改"分配"，\n'
                 '微波形状主要改"制备"', fontsize=11.5)
    ax.legend(fontsize=9); ax.grid(alpha=0.2, axis='y')

    fig.suptitle('图10  三类"校准消不掉"的非理想因素：谁的效力大、改的是什么',
                 fontsize=14, y=1.03)
    for ext in ('png', 'pdf'):
        fig.savefig(os.path.join(FIG, 'fig10_nonideal.%s' % ext), dpi=300)
    print('  图 → fig10_nonideal.png/.pdf')


if __name__ == '__main__':
    main()