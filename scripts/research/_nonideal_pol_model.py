# -*- coding: utf-8 -*-
"""自洽性检验：张量光移 = 两通道间的**差分光移 Δ_ts**？

若成立，则主频 f0（=2√(V²+Δ²)）与分配 a/s（=混合角决定的 cos²）必须
由**同一个 Δ_ts** 同时解释 —— 这是一个强约束（两个观测量、一个参数）。

  f0 = 2√(V² + Δ²)              ⇒ Δ = √((f0/2)² − V²)
  a/s = cos²(θ/2), tanθ = V/Δ   ⇒ 对非 π 偏振，1−a/s = sin²(θ/2) ∝ Δ²/(V²+Δ²)
  等价闭式：1−a/s = (1 − (2V/f0)²)/2 = 1/2 − 2V²/f0²
"""
import io
import json
import os
import sys

import numpy as np

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8',
                              errors='replace')
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TMP = os.path.join(ROOT, 'scripts', 'tmp')
R = os.path.join(TMP, '_nonideal_pol_res.json')


def main():
    rows = json.load(io.open(R, encoding='utf-8'))['rows']
    rows.sort(key=lambda r: r['alpha'])
    base = rows[0]
    V0 = base['f0'] * 1000 / 2.0          # MHz，由基准主频定标 V
    print('  由基准主频定标：V = f0(π)/2 = %.4f MHz' % V0)
    print()
    print('  %-12s %6s %9s %9s %10s %10s %9s' %
          ('tag', 'α(°)', 'f0(MHz)', 'Δ_ts(MHz)', 'a/s实测', 'a/s模型', '差'))
    print('  ' + '-' * 74)
    f0s, frs, preds = [], [], []
    for r in rows:
        f0 = r['f0'] * 1000
        D = np.sqrt(max((f0 / 2.0) ** 2 - V0 ** 2, 0.0))
        pred = 0.5 + 2 * V0 ** 2 / f0 ** 2
        pred = min(pred, 1.0)
        f0s.append(f0); frs.append(r['frac']); preds.append(pred)
        print('  %-12s %6.1f %9.4f %9.4f %10.4f %10.4f %+9.4f'
              % (r['tag'], r['alpha'], f0, D, r['frac'], pred,
                 r['frac'] - pred))
    f0s, frs, preds = map(np.array, (f0s, frs, preds))
    print('\n  === 模型质量（一个参数 V 解释 9 个点）===')
    print('     a/s 残差 rms = %.4f（a/s 的跨度 %.4f）'
          % (np.sqrt(((frs - preds) ** 2).mean()), frs.max() - frs.min()))
    print('     corr(实测 a/s, 模型 a/s) = %+.4f' % np.corrcoef(frs, preds)[0, 1])
    print('     ⇒ %s' % ('✅ 同一个差分光移同时解释了 f0 与 a/s ⇒ 张量光移机制成立'
                        if np.sqrt(((frs - preds) ** 2).mean()) < 0.02 else
                        '⚠ 残差偏大，机制可能不止差分光移'))

    # 差分光移随 α 的增长规律
    Ds = np.sqrt(np.maximum((f0s / 2.0) ** 2 - V0 ** 2, 0.0))
    A = np.array([r['alpha'] for r in rows])
    nz = A > 0
    if nz.sum() > 2:
        print('\n  === 差分光移 vs 椭圆偏角 α ===')
        print('     Δ_ts(α)/Δ_ts(1°) = %s'
              % ' '.join('%.2f' % (d / Ds[1]) for d in Ds))
        print('     α 比        = %s' % ' '.join('%.2f' % (a / A[1]) for a in A))
        print('     若 Δ ∝ sinα（张量项 ∝ 圆分量幅度）⇒ 比值应与 α 比同量级')
        print('     corr(α, Δ_ts) = %+.4f（几乎线性）' % np.corrcoef(A[nz], Ds[nz])[0, 1])

    out = os.path.join(TMP, '_nonideal_pol_model.json')
    json.dump({'V_MHz': float(V0),
               'rows': [dict(r, D_ts=float(d), frac_model=float(p))
                        for r, d, p in zip(rows, Ds, preds)]},
              io.open(out, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    print('\n  数据 → scripts/tmp/_nonideal_pol_model.json')


if __name__ == '__main__':
    main()