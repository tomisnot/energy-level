# -*- coding: utf-8 -*-
"""把散乱的各族统一起来：损失对 |Δφ| 的"斜率"由什么决定？

观察（_edge_cross_family 与 _edge_all）：各族各自近似成一条直线
    loss ≈ k_fam · |Δφ|，但 k_fam 不同：
        F 0.236 / A 0.186 / C 0.060 / B -0.045
⇒ 若 k_fam 能被某个**轨迹量**排序解释，机制就统一了。

候选"行程量"（都只由轨迹决定，与相位无关）：
  P1 光强行程  ∫|dI/dt|dt / I3        总走过多少强度
  P2 共振行程  ∫|dI/dt|·w(I)dt，w 在 I_res 附近加权（穿越共振的行程）
  P3 单向行程  |∫dI/dt dt|            净方向性
  P4 未回到 I_res 前的最大偏离          excursion 深度
"""
import io
import json
import os
import sys

import numpy as np

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8',
                              errors='replace')
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, 'scripts', 'research'))
from _edge_profiles import profiles                # noqa: E402

TMP = os.path.join(ROOT, 'scripts', 'tmp')
IRES = 0.42
T0, T1 = 255.0, 2000.0


def path_metrics(br):
    """轨迹量（只依赖折线，与相位无关）。"""
    t = np.array([b[0] for b in br], float)
    v = np.array([b[1] for b in br], float)
    m = (t >= T0) & (t <= T1)
    tt, vv = t[m], v[m]
    if tt.size < 2:
        return dict(P1=0, P2=0, P3=0, P4=0)
    dv = np.diff(vv)
    P1 = float(np.abs(dv).sum())
    # 在 I_res 附近加权（高斯核宽 0.1）
    mid = 0.5 * (vv[1:] + vv[:-1])
    w = np.exp(-((mid - IRES) / 0.10) ** 2)
    P2 = float((np.abs(dv) * w).sum())
    P3 = float(abs(dv.sum()))
    P4 = float(np.max(np.abs(vv - IRES)))
    return dict(P1=P1, P2=P2, P3=P3, P4=P4)


def main():
    T = json.load(io.open(os.path.join(TMP, '_edge_all.json'),
                          encoding='utf-8'))['rows']
    ref = T['C/step']
    # 各族：折线来源
    br_of = {}
    for kind in ('A', 'B', 'C', 'ctrl', 'smooth'):
        for tag, br, n in profiles(kind):
            br_of[tag] = br

    fams = {'F': [], 'A': [], 'B': [], 'C': [], 'T': [], 'S': []}
    for tag, r in T.items():
        key = tag.split('/')[0]
        if key not in fams:
            continue
        dphi = (r['phi0'] - ref['phi0'] + np.pi) % (2 * np.pi) - np.pi
        fams[key].append(dict(tag=tag, dphi=float(dphi),
                              loss=r['A0'] - ref['A0'],
                              br=br_of.get(tag)))

    print('  === 各族斜率 k（loss 对 |Δφ| 的线性系数）与轨迹量 ===')
    print('  %-6s %8s %9s %8s %8s %8s %8s' %
          ('族', 'n点', 'k(/rad)', 'P1', 'P2', 'P3', 'P4'))
    rows = []
    for k, g in fams.items():
        g = [r for r in g if r['br'] is not None]
        if len(g) < 3:
            continue
        x = np.abs(np.array([r['dphi'] for r in g]))
        y = np.array([r['loss'] for r in g])
        # 强制过原点更稳（理想阶跃 = 0 损失、Δφ=0）
        kk = float((x * y).sum() / (x * x).sum())
        pm = path_metrics(g[0]['br'])
        rows.append(dict(fam=k, n=len(g), k=kk, **pm))
        print('  %-6s %8d %9.3f %8.2f %8.2f %8.2f %8.2f'
              % (k, len(g), kk, pm['P1'], pm['P2'], pm['P3'], pm['P4']))

    print('\n  === 斜率与轨迹量的相关（看哪个能排序各族）===')
    K = np.array([r['k'] for r in rows])
    for p in ('P1', 'P2', 'P3', 'P4'):
        X = np.array([r[p] for r in rows])
        if X.std() > 0:
            print('     corr(k, %s) = %+.3f' % (p, np.corrcoef(X, K)[0, 1]))
    print('\n  ⇒ 若某量与 k 强相关，它就是"损伤效率"的轨迹控制量；')
    print('     若都不相关 ⇒ 轨迹量与相位不可分离，需更细的通道级模型。')

    out = os.path.join(TMP, '_edge_path_metrics.json')
    json.dump(rows, io.open(out, 'w', encoding='utf-8'), indent=1)
    print('\n  数据 → %s' % out)


if __name__ == '__main__':
    main()