# -*- coding: utf-8 -*-
"""新几何下 (上升沿 τ_r × 下降沿 τ_f) 二维扫描——**离线并行版**（不走软件串行队列）。

复用软件引擎：build_level_rows_direct 造基组（与 GUI 同源同序）、
build_fields_exact 造场、evolve_mixed_pair 演化。

用法：
  python scripts/tmp/_scan2d_par.py --probe      # 只跑 1 点，与软件结果对拍
  python scripts/tmp/_scan2d_par.py              # 20 点并行（10 进程）
  python scripts/tmp/_scan2d_par.py --jobs 12
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
from src.core.waveforms import Wave, GaussianEnv, FileEnv         # noqa: E402
from src.core.field_build import build_fields_exact               # noqa: E402
from src.core.atomic_levels import make_levels                    # noqa: E402
from src.core.evolve_mixed import evolve_mixed_pair               # noqa: E402

B_GAUSS = 21.7
R_UM, THETA = 14.1, 31.1
I3, I4 = 2.0e8, 8.3982e7
RAT = I4 / I3
Q_FS = 295724187.476        # 频移光频率 (MHz)
Q_MW = 17236.743            # 微波 π 频率 (MHz)
I_MW = 5.5e-05
T0, TON, TMAX, NPTS = 280.0, 120.0, 800.0, 801
GROUPS = [{'n': 60, 'l': 0, 'j': 0.5, 'label': '60S_{1/2}', 'color': '#1f77b4'},
          {'n': 60, 'l': 1, 'j': 1.5, 'label': '60P_{3/2}', 'color': '#ff7f0e'},
          {'n': 6, 'l': 1, 'j': 1.5, 'label': '6P_{3/2}', 'color': '#2ca02c'}]
K_BEF = '(60P_{3/2} m_J=-0.5) x (60S_{1/2} m_J=+0.5)'
K_AFT = '(60S_{1/2} m_J=-0.5) x (60P_{3/2} m_J=+1.5)'
OUTJSON = os.path.join(ROOT, 'scripts', 'tmp', '_scan2d_par.json')

_B = {}


def basis():
    if 'states' not in _B:
        cache = core.ArcCache(Rubidium87())
        # 与 GUI 完全同源：_base_level_rows 给"组的保留子能级行"（里德堡=mJ、
        # 低激发=(F,mF) 带 amps，E=E0 无 AC），再按 make_levels 的 spec 键名转换。
        rows = core._base_level_rows(cache, B_GAUSS, GROUPS)
        specs = []
        for r in rows:
            lab = '%s %s' % (r['label'], r['qstr'])   # 与 GUI 同格式
            if r['kind'] == 'F':
                specs.append(dict(n=r['n'], l=r['l'], j=r['j'], kind='F',
                                  F=r['F'], mF=r['mF'], amps=r['amps'],
                                  E_mhz=r['E'], E_hf=0.0, label=lab))
            else:
                specs.append(dict(n=r['n'], l=r['l'], j=r['j'], kind='mJ',
                                  m=r['mj'], E_mhz=r['E'], label=lab))
        _B['cache'] = cache
        _B['states'] = make_levels(cache, specs)
    return _B['cache'], _B['states']


def fs_wave(tr, tf):
    """强度 I(t)=I3·R(t)·F(t)；包络=sqrt(I/I3)（幅度因子）。"""
    ts = np.unique(np.concatenate([
        [0.0, TON - 0.1, TON],
        np.linspace(TON, TON + max(tr, 1e-3), 40) if tr > 0 else [TON + 1e-3],
        [T0 - max(tf, 1e-3) / 2.0] if tf > 0 else [T0 - 1e-3],
        np.linspace(T0 - tf / 2.0, T0 + tf / 2.0, 40) if tf > 0 else [T0],
        [T0 + max(tf, 1e-3) / 2.0, TMAX]]))
    R = np.clip((ts - TON) / tr, 0.0, 1.0) if tr > 0 else (ts >= TON) * 1.0
    F = (1.0 + (RAT - 1.0) * np.clip((ts - (T0 - tf / 2.0)) / tf, 0.0, 1.0)
         if tf > 0 else np.where(ts >= T0, RAT, 1.0))
    env = np.sqrt(np.clip(R * F, 0.0, None))
    keep = np.concatenate([[True], np.diff(ts) > 1e-9])
    return Wave('频移光', Q_FS, FileEnv(ts[keep].tolist(), env[keep].tolist()),
                pol='π', intensity_mwcm2=I3)


def one(task):
    tr, tf = task
    cache, states = basis()
    t = np.linspace(0.0, TMAX, NPTS)
    mw70 = Wave('π70', Q_MW, GaussianEnv(70.0, 20.0, 1.0), pol='σ-',
                intensity_mwcm2=I_MW)
    mw200 = Wave('π200', Q_MW, GaussianEnv(200.0, 20.0, 1.0), pol='σ-',
                 intensity_mwcm2=I_MW)
    fs = fs_wave(tr, tf)
    fA = build_fields_exact(cache, states, [mw70, mw200, fs])
    fB = build_fields_exact(cache, states, [mw70, mw200])
    i0 = [i for i, s in enumerate(states)
          if s.get('label', '').startswith('60S') and s.get('mj') is not None
          and abs(float(s['mj']) - 0.5) < 1e-6]
    i0 = int(i0[0])
    t0 = time.time()
    out = evolve_mixed_pair(states, fA, fB, i0, i0, t,
                            R_um=R_UM, theta_deg=THETA, cache=cache, tol=1e-8)
    P2 = np.real(np.asarray(out['pops_2d'], float))
    PA = np.real(np.asarray(out['pops_A'], float))
    PB = np.real(np.asarray(out['pops_B'], float))
    lab = [str(x['label']) for x in out['states_2d']]
    a, b = P2[:, lab.index(K_AFT)], P2[:, lab.index(K_BEF)]
    outm = 0.5 * (PA[:, [2, 4]].sum(axis=1) + PB[:, [2, 4]].sum(axis=1))
    i = int(np.argmax(a))
    return dict(tr=tr, tf=tf, i0=i0, t_s=round(time.time() - t0, 1),
                pk=float(a[i]), t_pk=float(t[i]), ct=float((a - b).max()),
                out_max=float(outm.max()), final_aft=float(a[-1]),
                final_bef=float(b[-1]),
                npz=None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--probe', action='store_true')
    ap.add_argument('--jobs', type=int, default=10)
    a = ap.parse_args()

    cache, states = basis()
    print('  基组 %d 个态：' % len(states))
    for i, s in enumerate(states[:6]):
        print('    [%2d] %s' % (i, s.get('label')))
    print('    ... 共 %d 个（6P 应为 16 个超精细态）' % len(states))
    if len(states) != 22:
        print('   基组数与软件(22)不符，先停下核对')

    if a.probe:
        tr, tf = 50.0, 20.0
        print('\n  探针：τ_r=%g, τ_f=%g（软件实测峰值 0.7395 @500 ns）' % (tr, tf))
        r = one((tr, tf))
        print('  离线：峰值 %.4f @%.0f ns  对比度 %.4f  最大漏 %.4f  用时 %.0fs'
              % (r['pk'], r['t_pk'], r['ct'], r['out_max'], r['t_s']))
        json.dump(r, io.open(OUTJSON, 'w', encoding='utf-8'), ensure_ascii=False,
                  indent=1)
        return

    tasks = [(tr, tf) for tr in (0.0, 25.0, 50.0, 75.0, 100.0)
             for tf in (0.0, 20.0, 50.0, 100.0)]
    print('\n  %d 点并行（%d 进程）…' % (len(tasks), a.jobs))
    t0 = time.time()
    res = []
    with ProcessPoolExecutor(max_workers=a.jobs) as ex:
        for r in ex.map(one, tasks):
            res.append(r)
            print('    r=%5.1f f=%5.1f  峰值 %.4f @%.0f  对比度 %.4f  漏 %.4f  %.0fs'
                  % (r['tr'], r['tf'], r['pk'], r['t_pk'], r['ct'],
                     r['out_max'], r['t_s']), flush=True)
    TRS = sorted({r['tr'] for r in res})
    TFS = sorted({r['tf'] for r in res})
    print('\n  === 交换峰值 ===')
    print('  %-8s' % 'τ_r\\τ_f' + ''.join('%10s' % ('%g' % f) for f in TFS))
    for tr in TRS:
        row = ''.join('%10.4f' % next(r['pk'] for r in res
                                      if r['tr'] == tr and r['tf'] == tf)
                      for tf in TFS)
        print('  %-8g' % tr + row)
    print('\n  === 对比度 max(P后−P前) ===')
    print('  %-8s' % 'τ_r\\τ_f' + ''.join('%10s' % ('%g' % f) for f in TFS))
    for tr in TRS:
        row = ''.join('%10.4f' % next(r['ct'] for r in res
                                      if r['tr'] == tr and r['tf'] == tf)
                      for tf in TFS)
        print('  %-8g' % tr + row)
    print('\n  === 最大漏出（其它 m） ===')
    print('  %-8s' % 'τ_r\\τ_f' + ''.join('%10s' % ('%g' % f) for f in TFS))
    for tr in TRS:
        row = ''.join('%10.4f' % next(r['out_max'] for r in res
                                      if r['tr'] == tr and r['tf'] == tf)
                      for tf in TFS)
        print('  %-8g' % tr + row)
    best = max(res, key=lambda r: r['pk'])
    worst = min(res, key=lambda r: r['pk'])
    print('\n  最优 r=%g f=%g → %.4f ; 最差 r=%g f=%g → %.4f ; 极差 %.4f'
          % (best['tr'], best['tf'], best['pk'], worst['tr'], worst['tf'],
             worst['pk'], best['pk'] - worst['pk']))
    print('  总用时 %.0fs' % (time.time() - t0))
    json.dump({'R_um': R_UM, 'theta_deg': THETA, 'res': res},
              io.open(OUTJSON, 'w', encoding='utf-8'), ensure_ascii=False,
              indent=1)
    print('  数据 → %s' % OUTJSON)


if __name__ == '__main__':
    main()