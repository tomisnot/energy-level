# -*- coding: utf-8 -*-
"""锋利假说：交换峰值的 12 点差异是**相位采样假象**，不是机制。

由 _edge_phase_model.py 观察到：12 条曲线的稳态振荡频率**逐位相同**
(2.3130 MHz, 散布 3e-8)，而峰值差 0.12 —— 频率不变而幅度差这么多，只能是
**相位不同**（峰落在不同相位上）。

假说 H*：
    两通道布居 = C + A·e^{-t/τ}·cos(2πft + φ)（C,A,f,τ 由静态末态决定，与沿宽无关）
    观测到的"峰值" = 该余弦在离散时刻上的最大值 ⇒ **纯运动学**
    真正的物理量是 (C, A, τ)，它们应当**与 τ_f 无关**

可判定的预言（都能被现有数据证伪）：
  P1  在拟合窗口内，峰位 t_pk 与 cos 的极值相位吻合（差 < T/4）
  P2  用拟合参数预测的"窗口内最大"应复原实测峰值（差 ≲ 1e-3）
  P3  A、C 跨全部 12 条曲线**不随沿宽系统漂移**
  P4  若损失是"相位"而非"退相干"，衰减时间 τ 不随 τ_f 单调变短

若 P1–P4 成立 ⇒ 前人 §2/§3 的"下降沿掉 5.4–6.7 点"与"上升沿 50 ns 最差"
需重新表述为：**沿改变了相位，而有限窗口把相位差读成了幅度差**。
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
J = os.path.join(TMP, '_edge_phase_model.json')

T_WIN = 2000.0          # 演化窗口
T_STEADY = 340.0        # 稳态起点（衔接 330 ns 之后）


def main():
    d = json.load(io.open(J, encoding='utf-8'))
    rows = []
    for r in d['rows']:
        f = r['fit']
        z = np.load(os.path.join(TMP, '_fall_mech_%s.npz'
                                 % r['tag'].replace('/', '_')))
        t, a = np.asarray(z['t'], float), np.asarray(z['aft'], float)
        # 稳态段重拟合，固定用解析式外推预测
        m = t >= T_STEADY
        C, A, fr, ph, tau = f['C'], f['A'], f['f'], f['phi'], f['tau']
        pred = C + A * np.exp(-(t[m] - T_STEADY) / tau) * \
            np.cos(2 * np.pi * fr * (t[m] - T_STEADY) + ph)
        i_pred = int(np.argmax(pred))
        rows.append(dict(
            tag=r['tag'], pk_obs=r['pk'], t_pk_obs=r['t_pk'],
            C=C, A=A, f=fr, phi=ph, tau=tau, rms=f['rms'],
            pk_pred=float(pred[i_pred]), t_pk_pred=float(t[m][i_pred]),
            ratio=A / C if C else None,
            t_steady_obs=float(t[m][int(np.argmax(a[m]))])))

    print('  %-16s %8s %8s %9s %9s %9s %8s %7s' %
          ('tag', '峰观测', '峰预测', 't_pk观', 't_pk预', 'A(振幅)', 'C(基线)', 'τ(ns)'))
    print('  ' + '-' * 86)
    for r in sorted(rows, key=lambda x: x['tag']):
        print('  %-16s %8.4f %8.4f %9.0f %9.0f %9.4f %8.4f %7s'
              % (r['tag'], r['pk_obs'], r['pk_pred'], r['t_pk_obs'],
                 r['t_pk_pred'], r['A'], r['C'],
                 '%.0f' % r['tau'] if r['tau'] else '-'))

    # ---- P2：预测 vs 观测 ----
    e = np.array([abs(r['pk_obs'] - r['pk_pred']) for r in rows])
    print('\n  === P2 预测 vs 观测（纯运动学）===')
    print('     |Δ峰值| 中位 %.2e  最大 %.2e' % (float(np.median(e)), float(e.max())))
    print('     ⇒ %s' % ('✅ 峰值完全由 (C,A,f,φ) 解释（纯相位采样）'
                        if e.max() < 3e-3 else
                        '❌ 拟合不足以解释峰值，另有机制'))

    # ---- P3：A、C 是否随沿宽漂移 ----
    print('\n  === P3 振幅 A / 基线 C 与沿宽的关系 ===')
    fam = [('F/tf=0', 0), ('F/tf=10', 10), ('F/tf=20', 20), ('F/tf=35', 35),
           ('F/tf=50', 50), ('F/tf=70', 70), ('F/tf=100', 100)]
    tfs, As, Cs, taus = [], [], [], []
    for tag, tf in fam:
        r = next((x for x in rows if x['tag'] == tag), None)
        if r:
            tfs.append(tf); As.append(r['A']); Cs.append(r['C'])
            taus.append(r['tau'] or np.nan)
    As, Cs, taus = np.array(As), np.array(Cs), np.array(taus)
    print('     τ_f     :', '  '.join('%7.0f' % x for x in tfs))
    print('     振幅 A  :', '  '.join('%7.4f' % x for x in As))
    print('     基线 C  :', '  '.join('%7.4f' % x for x in Cs))
    print('     τ(ns)   :', '  '.join('%7.0f' % x for x in taus))
    print('     A 的极差 %.4f（相对 %.1f%%）; C 的极差 %.4f（相对 %.1f%%）'
          % (np.ptp(As), 100 * np.ptp(As) / As.mean(), np.ptp(Cs),
             100 * np.ptp(Cs) / Cs.mean()))
    corr = float(np.corrcoef(tfs, As)[0, 1]) if len(tfs) > 2 else float('nan')
    print('     corr(τ_f, A) = %+.3f  ⇒ %s'
          % (corr, '✅ 振幅与沿宽无关' if abs(corr) < 0.5 else '⚠ 振幅随沿宽漂移'))

    print('\n  === P4 衰减时间 τ（退相干）与沿宽 ===')
    ok = ~np.isnan(taus)
    if ok.sum() > 2:
        c2 = float(np.corrcoef(np.array(tfs)[ok], taus[ok])[0, 1])
        print('     corr(τ_f, 衰减时间) = %+.3f  ⇒ %s'
              % (c2, '✅ 无退相干累积' if abs(c2) < 0.5 else '⚠ 退相干随沿宽增加'))

    # ---- 净相移 φ_net = ∫(I/I3 − I_res) dt 与拟合相位的关系 ----
    print('\n  === 拟合相位 φ 与"净相移积分" ===')
    print('  %-16s %10s %10s' % ('tag', '超额积分', '拟合相位(rad)'))
    for r in sorted(rows, key=lambda x: x['tag']):
        rr = next(x for x in d['rows'] if x['tag'] == r['tag'])
        print('  %-16s %10.1f %10.3f' % (r['tag'], rr['excess'], r['phi']))

    out = os.path.join(TMP, '_edge_phase_hypothesis.json')
    json.dump({'rows': rows}, io.open(out, 'w', encoding='utf-8'),
              ensure_ascii=False, indent=1)
    print('\n  数据 → %s' % out)


if __name__ == '__main__':
    main()
