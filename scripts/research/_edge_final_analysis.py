# -*- coding: utf-8 -*-
"""最终分析：以**波形指纹**识别轨迹（不依赖 tag/点序），跨多个 sweep 汇总。

为什么用指纹：扫描 spec 不能带 tag（会被当参数路径校验），而不同 sweep 的点序
又各不相同 ⇒ 唯一可靠的轨迹身份是**它自己的波形**。本脚本从每点 npz 读
`series_t/series_pop`，从 spec.json 读该点的 `dynamic.waves_A[2].env` 折线，
与 `_edge_profiles` 里各 profile 的折线做指纹匹配。

输出：
  · 每个 profile 的 (C, A0, f0, φ0) —— 锁相提取（见 _edge_lockin.py 的方法与稳健性）
  · 判决表：A 族（对称往返）vs C 族（单向）的 A0 对比 ⇒ 回答 P1/P2
  · 交叉校验：C/step、C/down50 vs 前人离线通道
  · 收敛检查：同轨迹不同 npts 的 A0
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
sys.path.insert(0, os.path.join(ROOT, 'scripts', 'research'))
from _edge_profiles import profiles, one, IRES          # noqa: E402

K_AFT = '(60S_{1/2} m_J=-0.5) x (60P_{3/2} m_J=+1.5)'
K_BEF = '(60P_{3/2} m_J=-0.5) x (60S_{1/2} m_J=+0.5)'
LEGACY = {                                  # 前人离线通道（交叉校验锚）
    'C/step': 0.7813, 'C/down50': 0.7395,
    'F/tf=0': 0.7813, 'F/tf=50': 0.7395,
}


def lockin(t, y, t_lo=340.0, t_hi=2000.0):
    m = (t >= t_lo) & (t <= t_hi)
    tt, yy = t[m] - t_lo, y[m]
    n = tt.size
    if n < 64:
        return None
    w = np.hanning(n)
    sp = np.abs(np.fft.rfft((yy - yy.mean()) * w))
    fr = np.fft.rfftfreq(n, d=float(np.median(np.diff(tt))))
    k = int(np.argmax(sp[1:]) + 1)
    if 0 < k < sp.size - 1:
        a, b, c = sp[k - 1], sp[k], sp[k + 1]
        den = a - 2 * b + c
        f0 = float(fr[k] + (0.5 * (a - c) / den if den else 0.0) * (fr[1] - fr[0]))
    else:
        f0 = float(fr[k])
    G = np.column_stack([np.ones(n), np.cos(2 * np.pi * f0 * tt),
                         np.sin(2 * np.pi * f0 * tt)])
    coef, *_ = np.linalg.lstsq(G, yy, rcond=None)
    return dict(C=float(coef[0]), A0=float(np.hypot(coef[1], coef[2])),
                phi0=float(np.arctan2(-coef[2], coef[1])), f0=f0,
                rms=float(np.sqrt(np.mean((G @ coef - yy) ** 2))))


def traces():
    """→ {tag: dict(npz, npts, J, I_int)} 指纹匹配。

    参考折线**由 one() 生成**（与提交时同一条代码路径）⇒ 逐位一致，
    不受 _fs 的"去重改写"影响。
    """
    ref = {}
    for kind in ('A', 'B', 'C', 'fine'):
        for tag, br, n in profiles(kind):
            env = one(tag, br, n)['dynamic.waves_A'][2]['env']
            t = np.array(env[1], float)
            v = np.array(env[2], float) ** 2          # 幅度 → I/I3
            m = t >= 255.0
            tc = np.clip(t[m], 255.0, 2000.0)
            ref[tag] = dict(t=t, v=v, npts=n,
                            J=float(np.trapezoid(v[m] - IRES, tc)),
                            I_int=float(np.trapezoid(v[m], tc)))
    out = {}
    for sid_dir in sorted(glob.glob(os.path.join(ROOT, 'runtime', 'sweeps', 'sw*')),
                          key=os.path.getmtime):
        sp = os.path.join(sid_dir, 'spec.json')
        if not os.path.exists(sp):
            continue
        spec = json.load(io.open(sp, encoding='utf-8'))
        pts = spec.get('points') or []
        # points 的键是**完整参数路径**（'dynamic.waves_A'），不是末段名
        if not any(len((p or {}).get('dynamic.waves_A') or []) == 3 for p in pts):
            continue          # 非本实验的 sweep
        for i, p in enumerate(pts):
            wa = p.get('dynamic.waves_A')
            if not wa or len(wa) < 3:
                continue
            env = wa[2].get('env') or []
            if env[:1] != ['file']:
                continue
            tt = np.array(env[1], float)
            vv = np.array(env[2], float) ** 2          # 幅度 → I/I3
            hit = None
            for tag, r in ref.items():
                if r['t'].size == tt.size and np.allclose(r['t'], tt, atol=1e-6):
                    hit = tag
                    break
            if hit is None:
                continue
            f = os.path.join(sid_dir, 'pt_%04d.npz' % i)
            if not os.path.exists(f):
                continue
            key = (hit, ref[hit]['npts'])
            if key in out:
                continue          # 同一轨迹可能被多次提交（失败批次也在目录里）
            z = np.load(f, allow_pickle=False)
            if 'series_t' not in z.files:
                continue
            labels = [str(x) for x in z['labels']]
            if K_AFT not in labels:
                continue
            t = np.asarray(z['series_t'], float)
            pops = np.asarray(z['series_pop'], float)
            a = pops[:, labels.index(K_AFT)]
            L = lockin(t, a)
            if L is None:
                continue
            out[key] = dict(tag=hit, sid=os.path.basename(sid_dir), npts=ref[hit]['npts'],
                            J=ref[hit]['J'], I_int=ref[hit]['I_int'],
                            pk=float(a.max()), pk_t=float(t[int(np.argmax(a))]),
                            **L)
    return out


def main():
    T = traces()
    if not T:
        print('  ⚠ 没有可用轨迹（可能还没跑完，或 record 不是 pops）')
        return
    # 同一 tag 可能有多个 npts（收敛检查点）⇒ 优先取 npts 与 profile 默认一致的
    T = {(t, r['npts']): r for (t, n), r in T.items()}
    order = ['C/step', 'C/down25', 'C/down50', 'B/hold10@0.71', 'B/hold25@0.71',
             'B/hold50@0.71', 'A/tri25_d1.00', 'A/tri50_d1.00', 'A/tri100_d1.00',
             'X/tri50_d1.00_n501', 'X/tri50_d1.00_n2001']
    rows = []
    for k in order:
        cand = [r for (t, n), r in T.items() if t == k]
        if cand:
            rows.append(min(cand, key=lambda r: abs(r['npts'] - 1001)))
    print('  %-22s %5s %8s %8s %8s %8s %9s %8s'
          % ('tag', 'npts', '观测峰', 'A0', 'C', 'C+A0', 'f0(MHz)', 'φ0'))
    print('  ' + '-' * 84)
    for r in rows:
        print('  %-22s %5d %8.4f %8.4f %8.4f %8.4f %9.4f %8.3f'
              % (r['tag'], r['npts'], r['pk'], r['A0'], r['C'], r['C'] + r['A0'],
                 r['f0'] * 1000, r['phi0']))

    # ---- P4：频率一致性 ----
    f0s = [r['f0'] for r in rows]
    print('\n  === P4 主频一致性 ===')
    print('     f0 中位 %.5f ns^-1 (%.4f MHz)，相对散布 %.2e'
          % (float(np.median(f0s)), float(np.median(f0s)) * 1000,
             float(np.std(f0s) / np.mean(f0s))))

    # ---- P1/P2：A 族 vs C 族 ----
    def get(tag):
        return next((r for r in rows if r['tag'] == tag), None)
    cs, c50 = get('C/step'), get('C/down50')
    tri = [get('A/tri25_d1.00'), get('A/tri50_d1.00'), get('A/tri100_d1.00')]
    print('\n  === P1/P2 判决：对称往返(A) vs 单向下降(C) 的交换幅度 A0 ===')
    if cs:
        print('     C/step      A0=%.4f  (理想参照)' % cs['A0'])
    if c50:
        print('     C/down50    A0=%.4f  (单向 50 ns)' % c50['A0'])
    got = [r for r in tri if r]
    for r in got:
        print('     %-12s A0=%.4f' % (r['tag'], r['A0']))
    if c50 and got:
        best = max(got, key=lambda r: r['A0'])
        lossC = (cs['A0'] - c50['A0']) / cs['A0'] * 100 if cs else float('nan')
        lossA = (cs['A0'] - best['A0']) / cs['A0'] * 100 if cs else float('nan')
        print('\n     单向损失 %.1f%%  vs  往返最小损失 %.1f%%  ⇒ %s'
              % (lossC, lossA,
                 '✅ P1 成立：往返显著恢复幅度' if lossA < lossC * 0.5 else
                 '⚠ P1 不成立：往返同样掉幅度'))
        if len(got) >= 2:
            sp = max(r['A0'] for r in got) - min(r['A0'] for r in got)
            print('     A 族跨 τ 的 A0 散布 %.4f ⇒ %s'
                  % (sp, '✅ P2 成立（与时长无关）' if sp < 0.01 else '⚠ P2 不成立'))
    for r in got + ([c50] if c50 else []) + ([cs] if cs else []):
        if r['tag'] in LEGACY:
            print('     交叉校验 %-12s 本次 %.4f  前人 %.4f  差 %+.4f'
                  % (r['tag'], r['pk'], LEGACY[r['tag']], r['pk'] - LEGACY[r['tag']]))

    # ---- 收敛检查 ----
    fine = get('X/tri50_d1.00_n2001')
    coarse = get('A/tri50_d1.00')
    if fine and coarse:
        print('\n  === P5 时间网格收敛 ===')
        print('     n2001 A0=%.4f  n1001 A0=%.4f  差 %+.4f ⇒ %s'
              % (fine['A0'], coarse['A0'], fine['A0'] - coarse['A0'],
                 '✅ 数值收敛' if abs(fine['A0'] - coarse['A0']) < 0.005 else '⚠ 未收敛'))

    out = os.path.join(ROOT, 'scripts', 'tmp', '_edge_final.json')
    json.dump({'rows': rows}, io.open(out, 'w', encoding='utf-8'),
              ensure_ascii=False, indent=1)
    print('\n  数据 → %s' % out)


if __name__ == '__main__':
    main()