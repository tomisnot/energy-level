# -*- coding: utf-8 -*-
"""下降沿机制判决：把三族数据压成"峰值 vs 超额积分"与"δ_eff vs 超额积分"。

判决：
  · F 与 E1 若落在**同一条曲线**上 ⇒ H1（时间平均失谐）成立 ✓
  · E3(阶梯50) 若 ≈ F(τ_f=50)（同时长同积分，仅形状不同）⇒ H1 成立（形状无关）
  · Ω 由相邻峰间隔给出（Ω=2π/T_period）；δ_eff=√(max(Ω²−V²,0))，V=1.2056 MHz 参照
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
J = os.path.join(ROOT, 'scripts', 'tmp', '_fall_mech.json')
FIG = os.path.join(ROOT, 'ui_design', 'screenshots', 'fall_mechanism.png')
V_MHZ = 1.2056

d = json.load(io.open(J, encoding='utf-8'))
res = d['res']
rows = []
for r in res:
    p = os.path.join(ROOT, 'scripts', 'tmp', r['npz'])
    if not os.path.exists(p):
        continue
    z = np.load(p)
    t, a = np.asarray(z['t'], float), np.asarray(z['aft'], float)
    # 相邻极大（t>300，跳过噪声：只取局部极大且值 > 0.5*max）
    mx = a.max()
    pk = [i for i in range(2, len(a) - 2)
          if a[i] >= a[i - 1] and a[i] > a[i + 1] and t[i] > 300 and a[i] > 0.5 * mx]
    # 合并过近的峰（<40 ns）
    keep = []
    for i in pk:
        if not keep or t[i] - t[keep[-1]] > 40:
            keep.append(i)
    Tper = float(np.mean(np.diff([t[i] for i in keep[:4]]))) if len(keep) >= 2 else None
    Om = (1e3 / Tper) if Tper else None            # MHz（周期 ns → 1/ns = 1000 MHz）
    d_eff = float(np.sqrt(max(Om ** 2 - V_MHZ ** 2, 0.0))) if Om else None
    rows.append(dict(tag=r['tag'], pk=r['pk'], pair_sum=r['pair_sum'],
                     excess=r['excess_ns'], t_pk=r['t_pk'], T=Tper,
                     Om=Om, d_eff=d_eff, npeaks=len(keep)))

print('  %-16s %8s %10s %10s %10s %10s' % ('tag', '峰值', '超额(ns)', '周期(ns)', 'Ω(MHz)', 'δ_eff'))
for r in sorted(rows, key=lambda x: x['excess']):
    print('  %-16s %8.4f %10.1f %10s %10s %10s'
          % (r['tag'], r['pk'], r['excess'],
             '%.0f' % r['T'] if r['T'] else '-',
             '%.3f' % r['Om'] if r['Om'] else '-',
             '%.3f' % r['d_eff'] if r['d_eff'] else '-'))

fam = {'F': [r for r in rows if r['tag'].startswith('F/')],
       'E1': [r for r in rows if r['tag'].startswith('E1/')],
       'E3': [r for r in rows if r['tag'].startswith('E3/')]}
fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.6))
for k, rs in fam.items():
    if not rs:
        continue
    rs = sorted(rs, key=lambda x: x['excess'])
    axes[0].plot([r['excess'] for r in rs], [r['pk'] for r in rs], 'o-',
                 lw=1.9, label='%s 族' % k)
    if all(r['d_eff'] is not None for r in rs):
        axes[1].plot([r['excess'] for r in rs], [r['d_eff'] for r in rs], 'o-',
                     lw=1.9, label='%s 族' % k)
axes[0].set_xlabel('交换窗口内"超额光强积分" ∫(I−I_res)dt  (I₃·ns)')
axes[0].set_ylabel('交换峰值')
axes[0].set_title('(a) 判决：三族是否落在同一条曲线上（H1）', fontsize=11)
axes[0].legend(fontsize=9)
axes[0].grid(alpha=0.25)
axes[1].set_xlabel('超额积分 (I₃·ns)')
axes[1].set_ylabel('δ_eff = √(Ω²−V²)  (MHz)')
axes[1].set_title('(b) 有效失谐（由振荡周期反推）', fontsize=11)
axes[1].legend(fontsize=9)
axes[1].grid(alpha=0.25)
fig.suptitle('下降沿机制判决：F(细扫沿宽) / E1(错误强度停留) / E3(同积分不同形状)', fontsize=12)
fig.tight_layout(rect=(0, 0, 1, 0.92))
fig.savefig(FIG, dpi=130)

e3 = fam['E3'][0] if fam['E3'] else None
f50 = next((r for r in fam['F'] if 'tf=50' in r['tag']), None)
if e3 and f50:
    print('\n  === E3(阶梯50) vs F(tf=50)：同时长、同积分、仅形状不同 ===')
    print('    E3  峰值 %.4f  超额 %.1f' % (e3['pk'], e3['excess']))
    print('    F50 峰值 %.4f  超额 %.1f' % (f50['pk'], f50['excess']))
    print('    差 = %+.4f  ⇒ %s' % (e3['pk'] - f50['pk'],
          '同曲线（H1 成立：形状无关）' if abs(e3['pk'] - f50['pk']) < 0.01
          else '不同曲线（形状/速率另有贡献，H1 不完备）'))
print('\n图 → %s' % FIG)