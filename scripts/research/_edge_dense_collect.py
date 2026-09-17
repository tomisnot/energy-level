# -*- coding: utf-8 -*-
"""密扫数据的收集器（独立模块，供分析与绘图复用）。

按**波形指纹**匹配：遍历所有 sweep 的 spec.json，取每点的
`dynamic.waves_A[2].env` 折线，与"正常下降沿 τ_f"的指纹比对 ⇒
拿到该 τ_f 的 npz（含 series_t / series_pop）。
"""
import glob
import io
import json
import os

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TMP = os.path.join(ROOT, 'scripts', 'tmp')
K_AFT = '(60S_{1/2} m_J=-0.5) x (60P_{3/2} m_J=+1.5)'
K_BEF = '(60P_{3/2} m_J=-0.5) x (60S_{1/2} m_J=+0.5)'
IRES, T0, TMAX = 0.42, 280.0, 2000.0


def normal(tau, n_pts=None):
    """正常下降沿：平台 I3 → 280 ns 处线性下坡 τ 到 I_res（前人 F 族同构）。"""
    n = n_pts or max(8, int(round(tau)))
    br = [(0.0, 0.0), (119.999, 0.0), (120.0, 1.0), (T0, 1.0)]
    for k in range(1, n + 1):
        br.append((T0 + tau * k / n, 1.0 - (1.0 - IRES) * k / n))
    br.append((TMAX, IRES))
    out, last = [], None
    for t, v in br:
        if last is not None and t <= last:
            continue
        out.append((t, v))
        last = t
    return out


def env_of(tau):
    br = normal(tau)
    t = np.array([b[0] for b in br], float)
    return t, np.sqrt(np.array([b[1] for b in br], float))


def collect(taus=None):
    """→ {tau: dict(t, a, b, sid)}"""
    if taus is None:
        taus = json.load(io.open(os.path.join(TMP, '_edge_dense.json'),
                                 encoding='utf-8'))['taus']
    ref = {float(t): env_of(float(t)) for t in taus}
    out = {}
    for sd in sorted(glob.glob(os.path.join(ROOT, 'runtime', 'sweeps', 'sw*')),
                     key=os.path.getmtime):
        sp = os.path.join(sd, 'spec.json')
        if not os.path.exists(sp):
            continue
        spec = json.load(io.open(sp, encoding='utf-8'))
        for i, p in enumerate(spec.get('points') or []):
            wa = p.get('dynamic.waves_A')
            if not wa or len(wa) != 3 or wa[2].get('env', [None])[0] != 'file':
                continue
            tt = np.array(wa[2]['env'][1], float)
            aa = np.array(wa[2]['env'][2], float)
            for tau, (rt, ra) in ref.items():
                if rt.size != tt.size or not np.allclose(rt, tt, atol=1e-6) \
                        or not np.allclose(ra, aa, atol=1e-9):
                    continue
                if tau in out:
                    continue
                f = os.path.join(sd, 'pt_%04d.npz' % i)
                if not os.path.exists(f):
                    continue
                z = np.load(f, allow_pickle=False)
                if 'series_t' not in z.files:
                    continue
                labels = [str(x) for x in z['labels']]
                if K_AFT not in labels:
                    continue
                t = np.asarray(z['series_t'], float)
                P = np.asarray(z['series_pop'], float)
                out[tau] = dict(t=t, a=P[:, labels.index(K_AFT)],
                                b=P[:, labels.index(K_BEF)],
                                sid=os.path.basename(sd))
    return out