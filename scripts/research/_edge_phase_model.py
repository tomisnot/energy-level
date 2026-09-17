# -*- coding: utf-8 -*-
"""下降沿机制：定量标定 κ 与 V_dd，检验"净相移"单参数模型。

接力 docs/研究-上升沿下降沿对交换峰值的影响.md 的 §8.2/§8.5：
  · 前人用"相邻峰间隔"抽 Ω；改用**整曲线阻尼正弦拟合**（更可靠）
  · 由 Ω(I) = sqrt(V² + δ(I)²)、δ = κ·(I − I_res) 同时定标 V 与 κ
  · 用定标结果检验新假设：峰值由**净相移 φ_net = ∫δ dt**（而非超额积分）决定

数据源：scripts/tmp/_fall_mech_*.npz（前人 E3 判决实验，窗口 2000 ns）
只读，不改软件状态。
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
TMP = os.path.join(ROOT, 'scripts', 'tmp')
J = os.path.join(TMP, '_fall_mech.json')

# 前人标定的参照值（docs §4）
V_REF = 1.2056            # MHz，目标通道 V_dd
IRES = 0.42               # I_res / I3
I3 = 2.0e8                # mW/cm²


def fit_damped(t, y, t0, t1):
    """在 [t0,t1] 内最小二乘拟合 y = C + A e^{-(t-t0)/tau} cos(2πf(t-t0) + φ)。

    频率用 FFT 给初值（比"找峰间隔"稳），再做 NLS（scipy 不可用时退化为网格）。
    """
    m = (t >= t0) & (t <= t1)
    tt, yy = t[m] - t0, y[m]
    if tt.size < 16:
        return None
    # FFT 初值
    yc = yy - yy.mean()
    w = np.hanning(yc.size)
    sp = np.abs(np.fft.rfft(yc * w))
    fr = np.fft.rfftfreq(yc.size, d=float(np.median(np.diff(tt))))
    k = int(np.argmax(sp[1:]) + 1)
    f0 = float(fr[k])
    try:
        from scipy.optimize import curve_fit

        def model(x, C, A, f, phi, tau):
            return C + A * np.exp(-x / tau) * np.cos(2 * np.pi * f * x + phi)

        p0 = [float(yy.mean()), float(yc.std() * 2 ** 0.5), f0, 0.0, 1e4]
        bnd = ([0, 0, f0 * 0.5, -np.pi, 100], [1, 1, f0 * 2.0, 3 * np.pi, 1e6])
        popt, _ = curve_fit(model, tt, yy, p0=p0, bounds=bnd, maxfev=40000)
        res = float(np.sqrt(np.mean((model(tt, *popt) - yy) ** 2)))
        return dict(C=popt[0], A=abs(popt[1]), f=popt[2], phi=popt[3],
                    tau=popt[4], rms=res, method='nls')
    except ImportError:
        return dict(C=float(yy.mean()), A=float(yc.std() * 2 ** 0.5), f=f0,
                    phi=None, tau=None,
                    rms=float(np.sqrt(np.mean((yc) ** 2))), method='fft')
    except Exception as e:                                     # noqa: BLE001
        return dict(C=float(yy.mean()), A=float(yc.std() * 2 ** 0.5), f=f0,
                    phi=None, tau=None, rms=None, method='fft(%s)' % type(e).__name__)


def main():
    d = json.load(io.open(J, encoding='utf-8'))
    rows = []
    for r in d['res']:
        p = os.path.join(TMP, r['npz'])
        if not os.path.exists(p):
            continue
        z = np.load(p)
        t = np.asarray(z['t'], float)
        a = np.asarray(z['aft'], float)
        # 稳态窗口：衔接结束(≈330)后到末尾；避开 F 族在 500 附近的瞬态
        fx = fit_damped(t, a, 600.0, 2000.0)
        rows.append(dict(tag=r['tag'], pk=r['pk'], t_pk=r['t_pk'],
                         excess=r['excess_ns'], pair_sum=r['pair_sum'], fit=fx))

    print('  %-16s %8s %8s %9s %8s %7s %9s' %
          ('tag', '峰值', '峰位', 'f(ns^-1)', 'Tper(ns)', 'Ω(MHz)', 'rms'))
    print('  ' + '-' * 76)
    for r in sorted(rows, key=lambda x: x['excess']):
        f = r['fit']
        T = (1.0 / f['f']) if f and f['f'] else None
        Om = (f['f'] * 1000.0) if f and f['f'] else None      # 1/ns → MHz
        print('  %-16s %8.4f %8.0f %9.5f %8s %7s %9s'
              % (r['tag'], r['pk'], r['t_pk'],
                 f['f'] if f else float('nan'),
                 '%.1f' % T if T else '-', '%.3f' % Om if Om else '-',
                 '%.2e' % f['rms'] if f and f['rms'] is not None else '-'))

    # ---- 用 F 族（同一 I_res 末态）标定 V 与 κ ----
    # 各点到稳态后都在 I_res 上 ⇒ Ω 应当相同；真正随 I 变的是 E1/阶梯段
    print('\n  === 稳态段(600–2000 ns, 均在 I_res 上)的频率一致性 ===')
    fs = [r['fit']['f'] for r in rows if r['fit'] and r['fit']['f']]
    if fs:
        print('     f 中位 %.5f ns^-1  → Ω = %.4f MHz   (散布 %.2e)'
              % (float(np.median(fs)), float(np.median(fs)) * 1000,
                 float(np.std(fs))))
        print('     由 Ω=√(V²+δ²) 反推：若稳态 δ=0 ⇒ V = %.4f MHz（前人静态直算 %.4f）'
              % (float(np.median(fs)) * 1000, V_REF))

    out = os.path.join(TMP, '_edge_phase_model.json')
    json.dump({'rows': rows}, io.open(out, 'w', encoding='utf-8'),
              ensure_ascii=False, indent=1)
    print('\n  数据 → %s' % out)


if __name__ == '__main__':
    main()
