# -*- coding: utf-8 -*-
"""研究绘图的最终版（含跨族、平滑度、时序、分解）。

产出 6 张图（PNG + PDF，300 dpi）：
  fig1_edge_shapes     三种边沿轨迹（受控变量）
  fig2_peak_removal    峰值 = 包络采样 ⇒ 必须分离 A0 / C
  fig3_redistribution  机制核心：峰值时刻的通道分解 + 无泄漏判据
  fig4_scaling         规律：损失 vs 净相位积累 + 双变量判别
  fig5_verdict         判决总表（排除法）
  fig6_mechanism       机制总图：分配项 vs 总量项 + 各族 a/s(Δφ)
"""
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
sys.path.insert(0, os.path.join(ROOT, 'scripts', 'research'))
TMP = os.path.join(ROOT, 'scripts', 'tmp')
FIGDIR = os.path.join(ROOT, 'ui_design', 'screenshots', 'edge_study')
os.makedirs(FIGDIR, exist_ok=True)

IRES = 0.42
BLUE, RED, GREEN, ORANGE, GREY = ('#1f77b4', '#d62728', '#2ca02c',
                                  '#ff7f0e', '#7f7f7f')


def load(n):
    return json.load(io.open(os.path.join(TMP, n), encoding='utf-8'))


def save(fig, stem):
    for ext in ('png', 'pdf'):
        fig.savefig(os.path.join(FIGDIR, '%s.%s' % (stem, ext)), dpi=300)
    print('  → %s' % stem)
    plt.close(fig)


def fig1():
    from _edge_profiles import tri, down
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.0), sharey=True)
    cases = [('理想阶跃  ($\\tau_f$ = 0)', down(0.0), BLUE),
             ('单向下降  ($\\tau_f$ = 50 ns)', down(50.0), RED),
             ('对称往返  ($\\tau$ = 25 ns)', tri(1.00, 25.0, 25), GREEN)]
    for ax, (title, br, col) in zip(axes, cases):
        t = np.array([b[0] for b in br], float)
        v = np.array([b[1] for b in br], float)
        ax.plot(t, v, '-', color=col, lw=2.4)
        ax.axhline(IRES, color=GREY, ls='--', lw=1.2)
        ax.text(1500, IRES + 0.03, r'$I_{\rm res}=0.42I_3$', color=GREY, fontsize=9)
        ax.axvspan(255, 305, color='0.92', zorder=0)
        ax.set_xlim(100, 420); ax.set_ylim(0, 1.12)
        ax.set_xlabel('时间 (ns)'); ax.set_title(title, fontsize=11.5)
        ax.grid(alpha=0.22)
        ax.annotate('', xy=(305, 1.05), xytext=(255, 1.05),
                    arrowprops=dict(arrowstyle='<->', color='0.35', lw=1.1))
        ax.text(280, 1.075, '作用窗口', ha='center', fontsize=9, color='0.3')
    axes[0].set_ylabel(r'光强  $I/I_3$')
    fig.suptitle('图1  三种下降沿轨迹：全部回到同一末态光强 $I_{\\rm res}$（受控变量）',
                 fontsize=13, y=1.03)
    save(fig, 'fig1_edge_shapes')


def fig2():
    L = load('_edge_lockin.json')['rows']
    fam = [('F/tf=0', 0), ('F/tf=10', 10), ('F/tf=20', 20), ('F/tf=35', 35),
           ('F/tf=50', 50), ('F/tf=70', 70), ('F/tf=100', 100)]
    d = {r['tag']: r for r in L}
    x = np.array([t for _, t in fam], float)
    pk = np.array([d[t]['pk'] for t, _ in fam])
    a0 = np.array([d[t]['A0'] for t, _ in fam])
    cc = np.array([d[t]['C'] for t, _ in fam])

    fig, axes = plt.subplots(1, 2, figsize=(12.8, 4.7))
    ax = axes[0]
    ax.plot(x, pk, 'o-', color=RED, lw=2, ms=7,
            label=r'前人的主指标：峰值 $\max_t P$(换后态)')
    ax.plot(x, cc + a0, 's--', color=BLUE, lw=2, ms=6,
            label=r'可达最大值 $C+A_0$（剔除相位采样）')
    ax.fill_between(x, cc, cc, color=BLUE, alpha=0)
    ax.plot(x, a0, '^:', color=GREEN, lw=2, ms=6, label=r'交换振幅 $A_0$')
    ax.set_xlabel(r'下降沿宽度 $\tau_f$  (ns)'); ax.set_ylabel('布居 / 幅度')
    ax.set_title(r'(a) 三条量：峰值与 $C+A_0$ 几乎重合，$A_0$ 低得多', fontsize=11.5)
    ax.legend(fontsize=8.8, loc='lower left'); ax.grid(alpha=0.22)

    ax = axes[1]
    s = np.array([r['C'] + r['A0'] for r in L]); p = np.array([r['pk'] for r in L])
    ax.plot([0.6, 0.82], [0.6, 0.82], color=GREY, ls='--', lw=1.3,
            label=r'理想：峰值 $=C+A_0$')
    ax.scatter(s, p, s=48, color=BLUE, zorder=3, edgecolor='white')
    ax.set_xlabel(r'$C + A_0$'); ax.set_ylabel('观测峰值')
    ax.set_title('(b) 12/12 条落在对角线上（最大偏差 %.1e）'
                 % np.max(np.abs(s - p)), fontsize=11.5)
    ax.legend(fontsize=9); ax.grid(alpha=0.22)
    fig.suptitle('图2  指标纠偏：为什么"峰值"不能直接当交换能力', fontsize=13, y=1.02)
    save(fig, 'fig2_peak_removal')


def fig3():
    L = load('_edge_leak_test.json')['rows']
    order = ['C/step', 'B/hold10@0.71', 'B/hold25@0.71', 'B/hold50@0.71',
             'C/down25', 'C/down50', 'A/tri25_d1.00', 'A/tri50_d1.00',
             'A/tri100_d1.00']
    R = [next(r for r in L if r['tag'] == k) for k in order]
    lbl = ['阶跃', 'B/10', 'B/25', 'B/50', 'C/25', 'C/50', 'A/25', 'A/50', 'A/100']
    a = np.array([r['pk'] for r in R]); b = np.array([r['bef_at_pk'] for r in R])
    s = a + b; ref = R[0]; idx = np.arange(len(R))

    fig, axes = plt.subplots(1, 3, figsize=(15.6, 4.8))
    ax = axes[0]
    ax.bar(idx, b, 0.62, color=RED, alpha=0.78, label='换前态 $b$（源通道）')
    ax.bar(idx, a, 0.62, bottom=b, color=GREEN, label='换后态 $a$（目标通道）')
    ax.plot(idx, s, 'k_', ms=19, mew=2.5, label='$s=a+b$（子空间总和）')
    ax.set_xticks(idx); ax.set_xticklabels(lbl, rotation=45)
    ax.set_ylabel('峰值时刻布居'); ax.legend(fontsize=8.5, loc='lower left')
    ax.set_title('(a) 总和几乎不变，分配在变', fontsize=11.5)
    ax.grid(alpha=0.2, axis='y')

    ax = axes[1]
    frac = a / s
    ax.bar(idx, frac, 0.62, color=[GREEN if f >= frac[0] else RED for f in frac],
           alpha=0.85)
    ax.axhline(frac[0], color=BLUE, ls='--', lw=1.6, label='理想阶跃 %.3f' % frac[0])
    ax.set_xticks(idx); ax.set_xticklabels(lbl, rotation=45)
    ax.set_ylabel(r'目标通道占比  $a/s$'); ax.set_ylim(0.6, 1.0)
    ax.set_title('(b) 分配从 %.3f 变到 %.3f' % (max(frac), min(frac)), fontsize=11.5)
    ax.legend(fontsize=9); ax.grid(alpha=0.2, axis='y')

    ax = axes[2]
    da = np.abs(a - ref['pk']); ds = np.abs(s - (ref['pk'] + ref['bef_at_pk']))
    ax.plot(idx, da, 'o-', color=RED, lw=2, ms=7,
            label=r'表观损失 $|\Delta a|$（最大 %.3f）' % da.max())
    ax.plot(idx, ds, 's-', color=BLUE, lw=2, ms=6,
            label=r'子空间总量变化 $|\Delta s|$（最大 %.3f）' % ds.max())
    ax.set_xticks(idx); ax.set_xticklabels(lbl, rotation=45)
    ax.set_ylabel('相对理想阶跃的变化')
    ax.set_title('(c) 无泄漏判据：峰值时堆外布居 $\\approx 0$', fontsize=11.5)
    ax.legend(fontsize=9); ax.grid(alpha=0.22)
    fig.suptitle('图3  机制核心：下降沿改的是"两通道分配"，不是布居总数',
                 fontsize=13, y=1.03)
    save(fig, 'fig3_redistribution')


def fig4():
    rows = load('_edge_loss_vs_phase.json')['rows']
    X1 = np.abs(np.array([r['dphi'] for r in rows]))
    y = np.array([r['loss_rel'] for r in rows])
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 4.8))
    ax = axes[0]
    cm = {'C': BLUE, 'B': ORANGE, 'A': GREEN}
    for r, xx, yy in zip(rows, X1, y):
        ax.scatter(xx, yy * 100, s=95, color=cm[r['tag'][0]], zorder=3,
                   edgecolor='white', linewidth=1.2)
        off = (-6, -15) if xx > X1.max() * 0.9 else (5, 5)
        ha = 'right' if xx > X1.max() * 0.9 else 'left'
        ax.annotate(r['tag'], (xx, yy * 100), xytext=off,
                    textcoords='offset points', fontsize=8.2, color='0.3', ha=ha)
    k, bb = np.polyfit(X1, y, 1)
    xs = np.linspace(0, X1.max() * 1.06, 50)
    ax.plot(xs, (k * xs + bb) * 100, '--', color=RED, lw=1.8,
            label=r'线性律 %.3f$|\Delta\phi|$%+.3f（corr=%.2f）'
                  % (k, bb, np.corrcoef(X1, y)[0, 1]))
    ax.axhline(0, color=GREY, lw=1.1); ax.axhspan(-8, 0, color=GREEN, alpha=0.08)
    ax.text(1.5, -5.0, '增益区', color=GREEN, fontsize=9.5)
    ax.set_xlabel(r'沿造成的净相位积累 $|\Delta\phi|$  (rad)')
    ax.set_ylabel('相对理想阶跃的振幅损失 (%)')
    ax.set_title('(a) 本轮判决族：损失 ∝ 净相位积累', fontsize=11.5)
    ax.legend(fontsize=9, loc='upper left'); ax.grid(alpha=0.22)

    ax = axes[1]
    names = [r'$|\Delta\phi|$' + '\n(净相位积累)', '单向行程\n(方向性)']
    vals = [1.0364, 0.4018]
    bars = ax.bar(names, vals, 0.5, color=[RED, BLUE], alpha=0.88)
    ax.axhline(0.3, color=GREY, ls=':', lw=1.2)
    ax.text(1.45, 0.33, '显著阈值', color=GREY, fontsize=8.6, ha='right')
    for b_, v in zip(bars, vals):
        ax.text(b_.get_x() + b_.get_width() / 2, v + 0.03, '%.2f' % v,
                ha='center', fontsize=11.5, fontweight='bold')
    ax.set_ylabel('标准化回归系数'); ax.set_ylim(0, 1.25)
    ax.set_title('(b) 双变量判别：相位项主导（2.6×）', fontsize=11.5)
    ax.grid(alpha=0.2, axis='y')
    fig.suptitle('图4  规律：交换能力的损失由净相位积累控制', fontsize=13, y=1.02)
    save(fig, 'fig4_scaling')


def fig5():
    cands = [
        (r'平台段光强积分 $\int I\,dt$', 0.0, '判决否定（同 S 末态差 7.7 倍）'),
        (r'超额积分 $\int(I-I_{res})^+dt$', 0.0, '前人否定（E1/hold100 理想）'),
        (r'折线平滑度（折点数）', -0.153, '判决否定（1→1000 只差 0.8%）'),
        (r'停留时间 $\tau_f$', -0.994, '单调但退化（非机制）'),
        (r'单向行程（方向性）', -0.341, '次要项（标准化 0.40）'),
        (r'净相位积累 $|\Delta\phi|$', 0.460, '主控项之一'),
        (r'两通道分配 $a/s$', -0.629, '主控项（直接观测量）'),
    ]
    fig, ax = plt.subplots(figsize=(11.6, 5.0))
    y = np.arange(len(cands))[::-1]
    cols = [GREY, GREY, GREY, ORANGE, BLUE, RED, '#9467bd']
    ax.barh(y, [c[1] for c in cands], 0.58, color=cols, alpha=0.88)
    ax.axvline(0, color='k', lw=0.9)
    ax.set_yticks(y); ax.set_yticklabels([c[0] for c in cands])
    ax.set_xlim(-1.3, 1.3)
    ax.set_xlabel('与"交换振幅损失"的相关系数（定性否定者为 0）')
    for yy, c in zip(y, cands):
        off = 0.05 if c[1] >= 0 else -0.05
        ax.text(c[1] + off, yy, c[2], va='center',
                ha='left' if c[1] >= 0 else 'right', fontsize=9.2, color='0.25')
    ax.grid(alpha=0.2, axis='x')
    ax.set_title('图5  判决总表：候选控制量的排除法', fontsize=13, pad=12)
    save(fig, 'fig5_verdict')


def fig6():
    D = load('_edge_decomp.json')
    rows = D['rows']
    fig = plt.figure(figsize=(14.5, 5.0))
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 1.15])

    # (a) 分解项 vs 实际 Δa
    ax = fig.add_subplot(gs[0])
    loss = np.array([r['loss'] for r in rows])
    A = np.array([r['A'] for r in rows]); B = np.array([r['B'] for r in rows])
    o = np.argsort(loss)
    ax.bar(np.arange(len(rows)), A[o], 0.7, color=GREEN,
           label='分配项  $s_{ref}\\cdot\\Delta(a/s)$')
    ax.bar(np.arange(len(rows)), B[o], 0.7, bottom=np.clip(A[o], None, 0),
           color=ORANGE, label='总量项  $(a/s)_{ref}\\cdot\\Delta s$')
    ax.plot(np.arange(len(rows)), loss[o], 'k.', ms=6,
            label=r'实际 $\Delta a$（恒等式校验 1e-16）')
    ax.axhline(0, color='k', lw=0.8)
    ax.set_xticks([]); ax.set_xlabel('各实验点（按 Δa 排序）')
    ax.set_ylabel(r'对 $\Delta a$ 的贡献')
    ax.set_title('(a) 振幅损失分解：分配项占 68%、总量项占 32%', fontsize=11.5)
    ax.legend(fontsize=9, loc='upper left'); ax.grid(alpha=0.2, axis='y')

    # (b) a/s vs Δφ（各族）
    ax = fig.add_subplot(gs[1])
    fams = {'A': ('本轮 A 族（对称往返）', GREEN, 'o'),
            'B': ('本轮 B 族（停留式）', ORANGE, 's'),
            'C': ('本轮 C 族（单向）', BLUE, '^'),
            'T': ('时序对照', '#9467bd', 'D'),
            'F': ('前人 F 族（单向）', RED, 'v')}
    ref = D['ref']          # C/step 被排除在 rows 之外，用 ref 里的参照值
    for key, (lab, col, mk) in fams.items():
        g = [r for r in rows if r['tag'].startswith(key + '/')]
        if not g:
            continue
        x = np.array([r['phi'] for r in g]); y = np.array([r['frac'] for r in g])
        o = np.argsort(x)
        ax.plot(x[o], y[o], mk + '-', color=col, lw=1.6, ms=7, alpha=0.9, label=lab)
    ax.axhline(ref['frac'], color=GREY, ls='--', lw=1.3,
               label='理想阶跃 $a/s$=%.3f' % ref['frac'])
    ax.set_xlabel(r'沿造成的相位差 $\Delta\phi$  (rad，折叠到 $(-\pi,\pi]$)')
    ax.set_ylabel(r'目标通道占比  $a/s$')
    ax.set_title('(b) 相位调制的分配：各族各成一条曲线', fontsize=11.5)
    ax.legend(fontsize=8.6, loc='lower left'); ax.grid(alpha=0.22)
    fig.suptitle('图6  机制总图：下降沿通过"相位 → 两通道分配"改变交换能力',
                 fontsize=13, y=1.02)
    save(fig, 'fig6_mechanism')


def fig7():
    """两组"排除性"对照：平滑度 与 边沿时刻。"""
    ALL = load('_edge_all.json')['rows']
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 4.6))

    # (a) 平滑度：同一轨迹，折点数 1→1000（用 _edge_all.json 里实际存在的键）
    ns = sorted(int(k.split('_n')[-1]) for k in ALL if k.startswith('S/ramp_n'))
    ax = axes[0]
    A0 = [ALL['S/ramp_n%d' % n]['A0'] for n in ns]
    pk = [ALL['S/ramp_n%d' % n]['pk'] for n in ns]
    ax.semilogx(ns, A0, 'o-', color=BLUE, lw=2, ms=7, label=r'交换振幅 $A_0$')
    ax.semilogx(ns, pk, 's--', color=RED, lw=1.6, ms=5, label='观测峰值')
    ax.axhline(A0[-1], color=GREY, ls=':', lw=1.2)
    ax.set_xlabel('折线采样点数 $n_{seg}$（同一物理轨迹）')
    ax.set_ylabel('幅度 / 峰值')
    ax.set_title('(a) 平滑度不是原因：$A_0$ 在 $n_{seg}$ 1→1000 只差 %.1f%%'
                 % (100 * abs(A0[0] - A0[-1]) / A0[-1]), fontsize=11.5)
    ax.legend(fontsize=9); ax.grid(alpha=0.22, which='both')

    # (b) 边沿时刻：step 与 ramp 分别
    ax = axes[1]
    ts = [255, 305, 400, 600]
    st = [ALL['T/step@%d' % t]['A0'] for t in ts]
    rp = [ALL['T/ramp50@%d' % t]['A0'] for t in (255, 305)]
    ax.plot(ts, st, 'o-', color=RED, lw=2, ms=8, label='阶跃在 $t_j$')
    ax.plot([255, 305], rp, 's--', color=BLUE, lw=2, ms=8,
            label='50 ns 下坡在 $t_j$')
    ax.axhline(ALL['C/step']['A0'], color=GREY, ls=':', lw=1.2,
               label='$t_j$=255 ns 阶跃（参照）')
    ax.set_xlabel('边沿时刻 $t_j$  (ns)'); ax.set_ylabel(r'交换振幅 $A_0$')
    ax.set_title('(b) 时序有影响，且不影响"光强积分"结论', fontsize=11.5)
    ax.legend(fontsize=9); ax.grid(alpha=0.22)
    fig.suptitle('图7  两组排除性对照：平滑度（无影响）与边沿时刻（有影响）',
                 fontsize=13, y=1.02)
    save(fig, 'fig7_controls')


if __name__ == '__main__':
    fig1(); fig2(); fig3(); fig4(); fig5(); fig6(); fig7()
    print('\n  图 → %s' % FIGDIR)