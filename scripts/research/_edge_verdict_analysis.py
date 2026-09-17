# -*- coding: utf-8 -*-
"""下降沿判决实验的分析：把每个扫描点的"交换能力"用锁相法干净提取。

输入：runtime/sweeps/<sid>/pt_XXXX.npz（record='final' 也带终态布居与态标签）
方法：与 _edge_lockin.py 同（Hann 窗 FFT 定主频 → 三基最小二乘投影）
      ⇒ 得 C(基线)、A0(幅度)、f0(频率)、φ0(相位)

判决要点：
  K1  末态同光强 I_res ⇒ 所有点的 (C, A0, f0) 应一致；若 A0 实际不同，
      差异就是**沿造成的真损失**（与末态哈密顿量无关）
  K2  φ0 是否只由 ∫δ dt 决定：用轨迹算 J = ∫₀²⁰⁰ (I/I3 − 0.42) dt，
      检验 φ0 与 J 的关系是否单调/线性
  K3  交叉校验：C/step 与 C/down50 应与前人 F/tf=0(0.7813) 与 F/tf=50(0.7395)
      一致 ⇒ 证明本次 resubmit 通道与离线通道等价
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
IRES = 0.42
I3 = 2.0e8
T_FS_ON = 120.0


def lockin(t, y, t_lo=340.0, t_hi=2000.0):
    m = (t >= t_lo) & (t <= t_hi)
    tt, yy = t[m] - t_lo, y[m]
    n = tt.size
    if n < 32:
        return None
    w = np.hanning(n)
    sp = np.abs(np.fft.rfft((yy - yy.mean()) * w))
    fr = np.fft.rfftfreq(n, d=float(np.median(np.diff(tt))))
    k = int(np.argmax(sp[1:]) + 1)
    if 0 < k < sp.size - 1:
        a, b, c = sp[k - 1], sp[k], sp[k + 1]
        den = (a - 2 * b + c)
        dk = 0.5 * (a - c) / den if den else 0.0
        f0 = float(fr[k] + dk * (fr[1] - fr[0]))
    else:
        f0 = float(fr[k])
    G = np.column_stack([np.ones(n), np.cos(2 * np.pi * f0 * tt),
                         np.sin(2 * np.pi * f0 * tt)])
    coef, *_ = np.linalg.lstsq(G, yy, rcond=None)
    C, cc, ss = coef
    return dict(C=float(C), A0=float(np.hypot(cc, ss)),
                phi0=float(np.arctan2(-ss, cc)), f0=f0,
                rms=float(np.sqrt(np.mean((G @ coef - yy) ** 2))))


def profile_from_spec(sid):
    """每点的波形折线 + J = ∫(I/I3 − I_res)dt。

    注意：扫描 spec 的 points **不能带 tag 元数据**（apply_delta 会把每个键
    当参数路径校验），所以 tag 从本地档 `_edge_profiles.json` 按**点序**对齐；
    折线本身从 spec 的 `dynamic.waves_A` 读（确保算的是实际提交的轨迹）。
    """
    p = os.path.join(ROOT, 'runtime', 'sweeps', sid, 'spec.json')
    spec = json.load(io.open(p, encoding='utf-8'))
    pf = os.path.join(ROOT, 'scripts', 'tmp', '_edge_profiles.json')
    tags = json.load(io.open(pf, encoding='utf-8'))['tags'] if os.path.exists(pf) else []
    out = {}
    for i, pt in enumerate(spec.get('points') or []):
        wa = pt.get('dynamic.waves_A')
        if not wa:
            continue
        tag = tags[i] if i < len(tags) else 'pt%d' % i
        env = wa[2]['env']
        t = np.asarray(env[1], float)
        amp = np.asarray(env[2], float)
        I = amp ** 2                      # 幅度 → I/I3
        m = t >= T_FS_ON
        tt, II = t[m], I[m]
        J = float(np.trapezoid(II - IRES, tt)) if tt.size > 1 else 0.0
        integ = float(np.trapezoid(II, tt)) if tt.size > 1 else 0.0
        out[tag] = dict(J=J, I_integral=integ)
    return out


def main():
    sid = sys.argv[1] if len(sys.argv) > 1 else None
    if sid is None:
        sdir = max(glob.glob(os.path.join(ROOT, 'runtime', 'sweeps', 'sw*')),
                   key=os.path.getmtime)
        sid = os.path.basename(sdir)
    sdir = os.path.join(ROOT, 'runtime', 'sweeps', sid)
    print('  扫描 %s' % sid)

    K_AFT = '(60S_{1/2} m_J=-0.5) x (60P_{3/2} m_J=+1.5)'
    K_BEF = '(60P_{3/2} m_J=-0.5) x (60S_{1/2} m_J=+0.5)'
    prof = profile_from_spec(sid)

    rows = []
    for f in sorted(glob.glob(os.path.join(sdir, 'pt_*.npz'))):
        z = np.load(f, allow_pickle=False)
        t = np.asarray(z['series_t'], float)
        pops = np.asarray(z['series_pop'], float)
        labels = [str(x) for x in z['states']]
        ia = labels.index(K_AFT) if K_AFT in labels else None
        ib = labels.index(K_BEF) if K_BEF in labels else None
        if ia is None:
            print('   %s 缺少换后态标签（前 4 个：%s）'
                  % (os.path.basename(f), labels[:4]))
            continue
        a = pops[:, ia]
        b = pops[:, ib] if ib is not None else np.zeros_like(a)
        # tag 从 spec 的 points 顺序推（pt_XXXX 与 points 同序）
        idx = int(os.path.basename(f)[3:7])
        spec = json.load(io.open(os.path.join(sdir, 'spec.json'),
                                 encoding='utf-8'))
        pts = spec.get('points') or []
        tag = pts[idx].get('tag') if idx < len(pts) else 'pt%d' % idx
        L = lockin(t, a)
        rows.append(dict(k=idx, tag=tag, pk=float(a.max()),
                         t_pk=float(t[int(np.argmax(a))]),
                         end_a=float(a[-1]), end_b=float(b[-1]),
                         J=prof.get(tag, {}).get('J'),
                         **({} if L is None else L)))

    rows.sort(key=lambda r: r['k'])
    print('\n  %-16s %7s %7s %7s %7s %8s %8s %8s' %
          ('tag', '观测峰', 'A0', 'C', 'C+A0', 'f0(MHz)', 'φ0', 'J(ns)'))
    print('  ' + '-' * 78)
    for r in rows:
        print('  %-16s %7.4f %7.4f %7.4f %7.4f %8.4f %8.3f %8.1f'
              % (r['tag'], r['pk'], r.get('A0', float('nan')),
                 r.get('C', float('nan')), r.get('C', 0) + r.get('A0', 0),
                 r.get('f0', float('nan')) * 1000, r.get('phi0', float('nan')),
                 r['J'] if r['J'] is not None else float('nan')))

    # ---- K1：同末态下 A0 的一致性 ----
    print('\n  === K1 同末态 I_res 下的交换幅度 A0（差异 = 沿造成的真损失）===')
    A0s = [r['A0'] for r in rows if 'A0' in r]
    if A0s:
        print('     A0 范围 %.4f – %.4f（相对展宽 %.1f%%）'
              % (min(A0s), max(A0s), 100 * (max(A0s) - min(A0s)) / np.mean(A0s)))
    print('     f0 范围 %s MHz'
          % (np.ptp([r['f0'] for r in rows if 'f0' in r]) * 1000))

    # ---- K2：φ0 与 J ----
    print('\n  === K2 φ0 与净相移积分 J = ∫(I/I3 − I_res)dt ===')
    ok = [r for r in rows if r['J'] is not None and 'phi0' in r]
    if len(ok) > 2:
        J = np.array([r['J'] for r in ok])
        P = np.unwrap(np.array([r['phi0'] for r in ok]))
        print('     corr(J, φ0) = %+.3f' % np.corrcoef(J, P)[0, 1])
        # 线性拟合 φ0 = k·J + b
        k, b = np.polyfit(J, P, 1)
        resid = P - (k * J + b)
        print('     φ0 ≈ %.5f·J + %.3f   残差 rms %.3f rad'
              % (k, b, float(np.sqrt(np.mean(resid ** 2)))))
        # 与 "Ω 积分" 模型比较：φ = ∫√(V²+δ²)dt
        print('     对比：若相位来自 ∫δdt，斜率应 ~ κ；若来自 ∫√(V²+δ²)dt，斜率更大')

    # ---- K3：交叉校验 ----
    print('\n  === K3 交叉校验（本通道 vs 前人离线通道）===')
    ref = {'C/step': 0.7813, 'C/down50': 0.7395}
    for r in rows:
        if r['tag'] in ref:
            print('     %-14s 本次 %.4f   前人 %.4f   差 %+.4f'
                  % (r['tag'], r['pk'], ref[r['tag']], r['pk'] - ref[r['tag']]))

    out = os.path.join(ROOT, 'scripts', 'tmp', '_edge_verdict_%s.json' % sid)
    json.dump({'sid': sid, 'rows': rows}, io.open(out, 'w', encoding='utf-8'),
              ensure_ascii=False, indent=1)
    print('\n  数据 → %s' % out)


if __name__ == '__main__':
    main()