# -*- coding: utf-8 -*-
"""下降沿机制判决实验（离线并行；父进程做全部 ARC 活，子进程只演化）。

三族（全部：上升沿=阶跃@120ns，平台 I3，衔接处降到 I_res=0.42·I3）：
  F  细扫 τ_f = 0/10/20/35/50/70/100 ns        —— δ̄ 的函数形状
  E1 陡降到 0.6·I3 → 停 Δt → 陡降到 I_res        —— 把"沿时长"与"失谐时间"解耦
  E3 50ns **阶梯**（0.71 停 50ns）——与 F(50) 同**时长**且同**积分**，只有形状不同

判决逻辑：
  H1（时间平均失谐）⇒ E1 的峰值应落在与 F 相同的"峰值 vs δ̄"曲线上；
                      E3 应与 F(50) 几乎相同（同积分）而非按"越陡越好"排列。
用法：python scripts/tmp/_fall_mechanism.py [--jobs N] [--only F|E1|E3]
"""
import argparse
import io
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8',
                              errors='replace')

from arc import Rubidium87                                        # noqa: E402
from src.core import static_atom as core                          # noqa: E402
from src.core import pair_core as _pc                             # noqa: E402
from src.core.waveforms import Wave, GaussianEnv, FileEnv         # noqa: E402
from src.core.field_build import build_fields_exact               # noqa: E402
from src.core.atomic_levels import make_levels                    # noqa: E402
from src.core.evolve_mixed import evolve_mixed_pair               # noqa: E402

B_GAUSS, R_UM, THETA = 21.7, 14.1, 31.1
I3, IRES = 2.0e8, 0.42
I_MW = 5.5e-05
Q_FS, Q_MW = 295724187.476, 17236.743
TON, TMAX, NPTS = 120.0, 2000.0, 1001
GROUPS = [{'n': 60, 'l': 0, 'j': 0.5, 'label': '60S_{1/2}', 'color': '#1f77b4'},
          {'n': 60, 'l': 1, 'j': 1.5, 'label': '60P_{3/2}', 'color': '#ff7f0e'},
          {'n': 6, 'l': 1, 'j': 1.5, 'label': '6P_{3/2}', 'color': '#2ca02c'}]
K_BEF = '(60P_{3/2} m_J=-0.5) x (60S_{1/2} m_J=+0.5)'
K_AFT = '(60S_{1/2} m_J=-0.5) x (60P_{3/2} m_J=+1.5)'
OUT = os.path.join(ROOT, 'scripts', 'tmp', '_fall_mech')
_B = {}


def basis():
    if 'states' in _B:
        return _B['cache'], _B['states']
    cache = core.ArcCache(Rubidium87())
    rows = core._base_level_rows(cache, B_GAUSS, GROUPS)
    specs = []
    for r in rows:
        lab = '%s %s' % (r['label'], r['qstr'])
        if r['kind'] == 'F':
            specs.append(dict(n=r['n'], l=r['l'], j=r['j'], kind='F', F=r['F'],
                              mF=r['mF'], amps=r['amps'], E_mhz=r['E'],
                              E_hf=0.0, label=lab))
        else:
            specs.append(dict(n=r['n'], l=r['l'], j=r['j'], kind='mJ',
                              m=r['mj'], E_mhz=r['E'], label=lab))
    _B['cache'] = cache
    _B['states'] = make_levels(cache, specs)
    return _B['cache'], _B['states']


def fs_wave(breaks):
    """按 (t_ns, I/I3) 折线给包络（幅度=sqrt(I/I3)）。"""
    t = np.asarray([b[0] for b in breaks], float)
    v = np.asarray([b[1] for b in breaks], float)
    env = np.sqrt(np.clip(v, 0.0, None))
    keep = np.concatenate([[True], np.diff(t) > 1e-9])
    return Wave('频移光', Q_FS, FileEnv(t[keep].tolist(), env[keep].tolist()),
                pol='π', intensity_mwcm2=I3)


def prof_F(tf):
    if tf <= 0:
        return [(0, 0), (TON - 0.1, 0), (TON, 1.0), (280 - 1e-3, 1.0),
                (280.0, IRES), (TMAX, IRES)]
    return [(0, 0), (TON - 0.1, 0), (TON, 1.0), (280 - tf / 2.0, 1.0),
            (280 + tf / 2.0, IRES), (TMAX, IRES)]


def prof_hold(dt, lvl=0.6):
    return [(0, 0), (TON - 0.1, 0), (TON, 1.0), (280 - 1e-3, 1.0),
            (280.0, lvl), (280.0 + dt, lvl), (280.0 + dt + 1e-3, IRES),
            (TMAX, IRES)]


def prof_stair():
    # 与 F(50) 同积分：线性 50ns 的超额 = 0.5*50*(1-0.42)=14.5
    # 阶梯：0.71-0.42=0.29 ⇒ 停 50ns ⇒ 超额 14.5 ✓
    return [(0, 0), (TON - 0.1, 0), (TON, 1.0), (280 - 1e-3, 1.0),
            (280.0, 0.71), (330.0, 0.71), (330.001, IRES), (TMAX, IRES)]


def tasks(only=None):
    T = []
    if only in (None, 'F'):
        for tf in (0.0, 10.0, 20.0, 35.0, 50.0, 70.0, 100.0):
            T.append(('F/tf=%g' % tf, prof_F(tf)))
    if only in (None, 'E1'):
        for dt in (0.0, 25.0, 50.0, 100.0):
            T.append(('E1/hold%g@0.6' % dt, prof_hold(dt)))
    if only in (None, 'E3'):
        T.append(('E3/stair50', prof_stair()))
    return T


def prep(item):
    tag, breaks = item
    cache, states = basis()
    mw70 = Wave('p70', Q_MW, GaussianEnv(70.0, 20.0, 1.0), pol='σ-',
                intensity_mwcm2=I_MW)
    mw200 = Wave('p200', Q_MW, GaussianEnv(200.0, 20.0, 1.0), pol='σ-',
                 intensity_mwcm2=I_MW)
    fs = fs_wave(breaks)
    fA = build_fields_exact(cache, states, [mw70, mw200, fs])
    fB = build_fields_exact(cache, states, [mw70, mw200])
    V = _pc.vdd_sparse(cache, [dict(s) for s in states], R_UM, THETA, 0.0)
    i0 = [i for i, s in enumerate(states)
          if s.get('label', '').startswith('60S') and s.get('mj') is not None
          and abs(float(s['mj']) - 0.5) < 1e-6][0]
    # 折线在 [280, 280+200] 内的"失谐超额" ∫(I/I3 − IRES)+ dt（粗估控制量）
    tt = np.arange(280.0, 480.0, 0.5)
    ii = np.interp(tt, [b[0] for b in breaks], [b[1] for b in breaks])
    excess = float(np.trapezoid(np.clip(ii - IRES, 0, None), tt))
    return dict(tag=tag, breaks=breaks, states=states, fA=fA, fB=fB, V=V,
                i0=int(i0), excess_ns=excess)


def one(p):
    t = np.linspace(0.0, TMAX, NPTS)
    t0 = time.time()
    out = evolve_mixed_pair(p['states'], p['fA'], p['fB'], p['i0'], p['i0'], t,
                            vdd_matrix=p['V'], tol=1e-8)
    P2 = np.real(np.asarray(out['pops_2d'], float))
    PA = np.real(np.asarray(out['pops_A'], float))
    PB = np.real(np.asarray(out['pops_B'], float))
    lab = [str(x['label']) for x in out['states_2d']]
    a, b = P2[:, lab.index(K_AFT)], P2[:, lab.index(K_BEF)]
    i = int(np.argmax(a))
    path = '%s_%s.npz' % (OUT, p['tag'].replace('/', '_'))
    np.savez_compressed(path, t=t, aft=a, bef=b,
                        A_P=PA[:, 2:6].sum(axis=1), out_m=0.5 * (
                            PA[:, [2, 4]].sum(axis=1) + PB[:, [2, 4]].sum(axis=1)))
    # 前两个极值（用于拟合频率）
    idx = [j for j in range(2, len(a) - 2)
           if a[j] >= a[j - 1] and a[j] > a[j + 1] and t[j] > 300]
    return dict(tag=p['tag'], t_s=round(time.time() - t0, 1),
                pk=float(a[i]), t_pk=float(t[i]),
                ct=float((a - b).max()), pair_sum=float(a[i] + b[i]),
                bef_at_pk=float(b[i]), excess_ns=p['excess_ns'],
                peaks=[float(t[j]) for j in idx[:3]],
                npz=os.path.basename(path))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--jobs', type=int, default=8)
    ap.add_argument('--only', default=None)
    a = ap.parse_args()
    T = tasks(a.only)
    print('  任务 %d 个，%d 进程；窗口 %.0f ns / %d 点' % (len(T), a.jobs, TMAX, NPTS),
          flush=True)
    print('  父进程准备（ARC 查询全在此）…', flush=True)
    pl = []
    for i, it in enumerate(T):
        pl.append(prep(it))
        if (i + 1) % 4 == 0 or i == len(T) - 1:
            print('    准备 %d/%d' % (i + 1, len(T)), flush=True)
    print('  开始并行演化…', flush=True)
    t0 = time.time()
    res = []
    with ProcessPoolExecutor(max_workers=a.jobs) as ex:
        for r in ex.map(one, pl):
            res.append(r)
            print('  %-16s 峰值 %.4f @%4.0f  两通道和 %.4f  超额积分 %6.1f ns  %.0fs'
                  % (r['tag'], r['pk'], r['t_pk'], r['pair_sum'],
                     r['excess_ns'], r['t_s']), flush=True)
    print('\n  ===== 按峰值降序 =====')
    for r in sorted(res, key=lambda x: -x['pk']):
        print('  %-16s 峰值 %.4f  峰值时刻 %4.0f  两通道和 %.4f  前态@峰 %.4f  超额 %6.1f'
              % (r['tag'], r['pk'], r['t_pk'], r['pair_sum'],
                 r['bef_at_pk'], r['excess_ns']))
    print('  总用时 %.0fs' % (time.time() - t0))
    json.dump({'R_um': R_UM, 'theta_deg': THETA, 'TMAX': TMAX,
               'res': res}, io.open(OUT + '.json', 'w', encoding='utf-8'),
              ensure_ascii=False, indent=1)
    print('  数据 → %s.json / %s_*.npz' % (OUT, OUT))


if __name__ == '__main__':
    main()