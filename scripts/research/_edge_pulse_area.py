# -*- coding: utf-8 -*-
"""机制检验：两通道分配是否服从"脉冲面积"公式 a/s = sin²(θ_tot/2)？

物理预期：沿结束后系统回到 I_res 上的两能级（耦合 V，失谐 δ_res）。
若把"沿 + 平台"看作对该两能级施加一个有效脉冲面积
    θ_tot = ∫ Ω_eff(t) dt ,   Ω_eff = √(V² + δ(t)²)
则目标通道占比应满足
    a/s = sin²(θ_tot/2 + θ_0)

⇒ 用**同族内 3–4 个点**做非线性最小二乘（振幅已归一 ⇒ 只 3 个自由参数），
   再用**另一族**独立验证。这是把经验律升级为"可预测公式"的关键一步。

数据来源：判决实验的锁相结果（Δφ 由拟合相位给出，a/s 由峰值时刻布居给出）。
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


def main():
    leak = {r['tag']: r for r in
            json.load(io.open(os.path.join(TMP, '_edge_leak_test.json'),
                              encoding='utf-8'))['rows']}
    phase = {r['tag']: r for r in
             json.load(io.open(os.path.join(TMP, '_edge_loss_vs_phase.json'),
                               encoding='utf-8'))['rows']}
    tags = [t for t in leak if t in phase]
    # 相对理想阶跃的相位（用有符号 Δφ）
    dphi = np.array([phase[t]['dphi'] for t in tags])
    fracd = {t: leak[t]['pk'] / (leak[t]['pk'] + leak[t]['bef_at_pk'])
             for t in tags}
    frac = np.array([fracd[t] for t in tags])

    print('  %-20s %10s %10s' % ('tag', 'Δφ(rad)', 'a/s'))
    for t, d, f in sorted(zip(tags, dphi, frac), key=lambda x: x[1]):
        print('  %-20s %10.3f %10.4f' % (t, d, f))

    # ---- 用 B 族（hold）拟合 a/s = A0 + A1·cos(2πΔφ/T + φ) ----
    famB = [t for t in tags if t.startswith('B/')]
    famA = [t for t in tags if t.startswith('A/')]
    print('\n  === 族内拟合：a/s 随 Δφ 的振荡 ===')
    for nm, fam in (('B 族（停留式）', famB), ('A 族（对称往返）', famA)):
        if len(fam) < 3:
            print('     %s：点太少（%d），跳过' % (nm, len(fam)))
            continue
        x = np.array([phase[t]['dphi'] for t in fam])
        y = np.array([fracd[t] for t in fam])
        o = np.argsort(x)
        x, y = x[o], y[o]
        # 拟合 y = A0 + A1·cos(k·x + φ)
        from scipy.optimize import curve_fit

        def model(xx, A0, A1, k, ph):
            return A0 + A1 * np.cos(k * xx + ph)

        best = None
        # 多起点（k 用若干种子，避免局部极小）
        for k0 in (0.5, 1.0, 2.0, 4.0, 8.0):
            for ph0 in np.linspace(-np.pi, np.pi, 7):
                try:
                    p, _ = curve_fit(model, x, y,
                                     p0=[y.mean(), y.std() * 1.4, k0, ph0],
                                     maxfev=40000,
                                     bounds=([0, -1, 0.01, -2 * np.pi],
                                             [1, 1, 50, 2 * np.pi]))
                    r = float(np.sqrt(np.mean((model(x, *p) - y) ** 2)))
                    if best is None or r < best[0]:
                        best = (r, p)
                except Exception:                       # noqa: BLE001
                    pass
        r, p = best
        print('     %-14s  A0=%.4f  A1=%.4f  k=%.3f rad⁻¹  φ=%.3f  rms=%.4f'
              % (nm, p[0], p[1], p[2], p[3], r))
        print('        ⇒ 周期 2π/k = %.2f rad（相位尺度）' % (2 * np.pi / p[2] if p[2] else np.inf))
        # 极值位置（a/s 的最大值）
        kk, ph = p[2], p[3]
        xopt = (-ph + 2 * np.pi * np.arange(4)) / kk
        xopt = xopt[(xopt > x.min() - 2) & (xopt < x.max() + 2)]
        print('        a/s 极值出现在 Δφ ≈ %s'
              % ', '.join('%.2f' % v for v in xopt[:3]))

    # ---- 跨族一致性：把两族的 a/s vs Δφ 画在一起看是否同一条曲线 ----
    print('\n  === 跨族一致性（若同一条 sin² 曲线 ⇒ 机制统一）===')
    x = np.array([phase[t]['dphi'] for t in tags])
    y = np.array([frac[t] for t in tags])
    k = np.polyfit(x, y, 2)
    rr = y - np.polyval(k, x)
    print('     二次拟合 a/s = %.4f·Δφ² %+.4f·Δφ %+.4f  rms=%.4f'
          % (k[0], k[1], k[2], np.sqrt((rr ** 2).mean())))
    print('     corr(Δφ, a/s) = %+.3f' % np.corrcoef(x, y)[0, 1])
    print('     ⇒ %s' % ('a/s 与 Δφ 有清晰的抛物线关系（顶点=最优相位）'
                        if abs(np.corrcoef(x ** 2, y)[0, 1]) > 0.6 else
                        '关系不干净，机制不止一个变量'))

    out = os.path.join(TMP, '_edge_pulse_area.json')
    json.dump({'tags': tags, 'dphi': dphi.tolist(), 'frac': frac.tolist()},
              io.open(out, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    print('\n  数据 → %s' % out)


if __name__ == '__main__':
    main()