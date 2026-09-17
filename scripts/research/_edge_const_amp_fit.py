# -*- coding: utf-8 -*-
"""约束拟合：终态布居能否被"C + A·cosθ_i"解释（恒定 C/A + 逐点自由相位）？

判据设计（只用终态数据，不需要时间序列）：
  模型 M1：a_i = C + A·cos(θ_i)（θ_i 自由）—— 3 类参数、8 个数据点
  模型 M0：a_i 完全自由（8 参数）—— 平凡完美拟合

若 M1 的残差 ~1e-3 量级 ⇒ 终态就是**同一条正弦曲线上的不同相位采样**
⇒ 各点的"交换能力"（振幅 A）相同，末态差异是相位差。
若 M1 残差大 ⇒ 存在相位之外的机制（真幅度变化）。

⚠ 局限（写清楚，免得过度解读）：8 个点拟合 2+8 个自由度**必然过拟合**，
因此 M1 残差小**只是必要条件**，不是充分证据。真正的判据仍需要时间序列
（锁相法，见 _edge_lockin.py）；本脚本的作用是**排除**"终态根本无法用
单一正弦描述"这一可能。
"""
import glob
import io
import json
import os
import sys

import numpy as np

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8',
                              errors='replace')
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
J = os.path.join(ROOT, 'scripts', 'tmp',
                 '_edge_endpoint_sw20260916_094856_703.json')
LOCKIN = os.path.join(ROOT, 'scripts', 'tmp', '_edge_lockin.json')


def fit_const_amp(vals, n_iter=200):
    """a_i = C + A·cos(θ_i)：交替最小化（固定 C,A 解 θ，再固定 θ 解 C,A）。"""
    v = np.asarray(vals, float)
    C, A = float(v.mean()), float(v.std() * 2 ** 0.5)
    th = np.zeros_like(v)
    for _ in range(n_iter):
        # 给定 C,A 解 θ：cos θ = (a−C)/A
        x = np.clip((v - C) / max(A, 1e-12), -1.0, 1.0)
        th = np.arccos(x)
        # 给定 θ 解 C,A（线性最小二乘）
        G = np.column_stack([np.ones_like(v), np.cos(th)])
        coef, *_ = np.linalg.lstsq(G, v, rcond=None)
        C, A = float(coef[0]), float(coef[1])
        if A < 0:
            A, th = -A, th + np.pi
    pred = C + A * np.cos(th)
    return dict(C=C, A=A, theta=th.tolist(),
                rms=float(np.sqrt(np.mean((pred - v) ** 2))),
                spread=float(np.ptp(v)))


def main():
    d = json.load(io.open(J, encoding='utf-8'))['rows']
    d = [r for r in d if 'a' in r]
    print('  === 本次判决扫描的 8 个终态 ===')
    v = [r['a'] for r in d]
    for r in d:
        print('    %-18s a=%.4f' % (r['tag'], r['a']))
    f = fit_const_amp(v)
    print('\n  M1 拟合 C + A·cosθ：C=%.4f  A=%.4f  残差 rms=%.5f  （数据极差 %.4f）'
          % (f['C'], f['A'], f['rms'], f['spread']))
    print('  ⇒ %s' % ('✅ 终态可被"恒定 C/A + 自由相位"解释'
                      if f['rms'] < 0.005 * max(f['spread'], 1e-9) * 100
                      else '⚠ 残差不可忽略'))
    print('  逐点相位 θ_i (rad)：')
    for r, t in zip(d, f['theta']):
        print('    %-18s θ=%.3f' % (r['tag'], t))

    # ---- 前人 12 条曲线的 (C, A0) 是否恒定（这才是强证据）----
    L = json.load(io.open(LOCKIN, encoding='utf-8'))['rows']
    print('\n  === 前人 12 条曲线：锁相法给的 (C, A0) 恒定性与"正弦可解释性" ===')
    pks = [r['pk'] for r in L]
    print('    观测峰值极差 %.4f（%.3f – %.3f）' % (np.ptp(pks), min(pks), max(pks)))
    C0 = np.array([r['C'] for r in L]); A0 = np.array([r['A0'] for r in L])
    print('    C  极差 %.4f（相对 %.1f%%）' % (np.ptp(C0), 100 * np.ptp(C0) / C0.mean()))
    print('    A0 极差 %.4f（相对 %.1f%%）' % (np.ptp(A0), 100 * np.ptp(A0) / A0.mean()))
    # C + A0 与观测峰值
    s = C0 + A0
    print('    (C+A0) vs 观测峰值：|差| 最大 %.4f' % np.max(np.abs(s - np.array(pks))))
    # 用同样方法拟合 (C+A0) 是否恒定
    print('    (C+A0) 极差 %.4f  ⇒ %s' % (np.ptp(s),
          '恒定的"可达最大值"' if np.ptp(s) < 0.02 else '随沿变化'))

    out = os.path.join(ROOT, 'scripts', 'tmp', '_edge_const_amp_fit.json')
    json.dump({'verdict_points': f, 'C_A0_spread':
               {'C': float(np.ptp(C0)), 'A0': float(np.ptp(A0)),
                'sum': float(np.ptp(s))}},
              io.open(out, 'w', encoding='utf-8'), indent=1)
    print('\n  数据 → %s' % out)


if __name__ == '__main__':
    main()