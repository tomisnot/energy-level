# -*- coding: utf-8 -*-
"""以"让交换尽可能充分"为目标的三组并行扫描（离线，与软件同源引擎）。

A. 两个沿：τ_r ∈ {0,50,100} × τ_f ∈ {0,20,50,100}
B. 交换期间光强 I4/I3 ∈ {0, 0.1, 0.2, 0.42(现行), 0.6, 0.8, 1.0}  （机制#1：微分光移解谐）
C. 衔接时刻 t_j ∈ {230, 255, 280(现行), 330, 380} ns        （高位光场持续时间）
用法：python scripts/tmp/_sweep_exchange.py [--jobs N] [--only A|B|C]
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
I3 = 2.0e8
I_MW = 5.5e-05
Q_FS, Q_MW = 295724187.476, 17236.743
TON, TMAX, NPTS = 120.0, 800.0, 801
GROUPS = [{'n': 60, 'l': 0, 'j': 0.5, 'label': '60S_{1/2}', 'color': '#1f77b4'},
          {'n': 60, 'l': 1, 'j': 1.5, 'label': '60P_{3/2}', 'color': '#ff7f0e'},
          {'n': 6, 'l': 1, 'j': 1.5, 'label': '6P_{3/2}', 'color': '#2ca02c'}]
K_BEF = '(60P_{3/2} m_J=-0.5) x (60S_{1/2} m_J=+0.5)'
K_AFT = '(60S_{1/2} m_J=-0.5) x (60P_{3/2} m_J=+1.5)'
OUTJSON = os.path.join(ROOT, 'scripts', 'tmp', '_sweep_exchange.json')
_B = {}


def basis():
    """建基组。⚠ ARC 的 sqlite 缓存在缺条目时会**重算并重建表**；多进程同时首次
    访问会互相踩（sqlite3.OperationalError: no such table: hfsDataAB）⇒ 带重试。"""
    if 'states' in _B:
        return _B['cache'], _B['states']
    last = None
    for attempt in range(6):
        try:
            cache = core.ArcCache(Rubidium87())
            rows = core._base_level_rows(cache, B_GAUSS, GROUPS)
            specs = []
            for r in rows:
                lab = '%s %s' % (r['label'], r['qstr'])
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
        except Exception as e:                       # noqa: BLE001
            last = e
            _B.pop('states', None)
            time.sleep(1.5 + 2.5 * attempt)
    raise last


def fs_wave(tr, tf, i4_ratio, tj):
    """上升沿 tr（120 起）、平台 I3 到 tj、下降沿 tf（中心 tj）降到 i4_ratio*I3。"""
    ratio = float(i4_ratio)
    t_lo, t_hi = tj - max(tf, 1e-3) / 2.0, tj + max(tf, 1e-3) / 2.0
    ts = np.unique(np.concatenate([
        [0.0, TON - 0.1, TON],
        np.linspace(TON, TON + max(tr, 1e-3), 40) if tr > 0 else [TON + 1e-3],
        [t_lo] if tf > 0 else [tj - 1e-3],
        np.linspace(t_lo, t_hi, 40) if tf > 0 else [tj],
        [t_hi, TMAX]]))
    R = np.clip((ts - TON) / tr, 0.0, 1.0) if tr > 0 else (ts >= TON) * 1.0
    F = (1.0 + (ratio - 1.0) * np.clip((ts - t_lo) / tf, 0.0, 1.0)
         if tf > 0 else np.where(ts >= tj, ratio, 1.0))
    env = np.sqrt(np.clip(R * F, 0.0, None))
    keep = np.concatenate([[True], np.diff(ts) > 1e-9])
    return Wave('频移光', Q_FS, FileEnv(ts[keep].tolist(), env[keep].tolist()),
                pol='π', intensity_mwcm2=I3)


def prep(task):
    """**父进程**：建基组 + 建场 + 算 V_dd（全部 ARC 查询在此完成）。
    返回可 pickle 的载荷（states/场 dict/sparse V），子进程不再碰 ARC。
    ⚠ 原因：ARC 的 sqlite 缓存在多进程并发下会互相踩
    （OperationalError: no such table: literatureDME/hfsDataAB）。"""
    tag, tr, tf, i4, tj = task
    cache, states = basis()
    mw70 = Wave('π70', Q_MW, GaussianEnv(70.0, 20.0, 1.0), pol='σ-',
                intensity_mwcm2=I_MW)
    mw200 = Wave('π200', Q_MW, GaussianEnv(200.0, 20.0, 1.0), pol='σ-',
                 intensity_mwcm2=I_MW)
    fs = fs_wave(tr, tf, i4, tj)
    fA = build_fields_exact(cache, states, [mw70, mw200, fs])
    fB = build_fields_exact(cache, states, [mw70, mw200])
    from src.core import pair_core as _pc
    V = _pc.vdd_sparse(cache, [dict(s) for s in states], R_UM, THETA, 0.0)
    i0 = [i for i, s in enumerate(states)
          if s.get('label', '').startswith('60S') and s.get('mj') is not None
          and abs(float(s['mj']) - 0.5) < 1e-6][0]
    return dict(tag=tag, tr=tr, tf=tf, i4=i4, tj=tj, states=states,
                fA=fA, fB=fB, V=V, i0=int(i0))


def one(p):
    """**子进程**：只做演化（纯 numba，不碰 ARC）。"""
    tag, tr, tf, i4, tj = p['tag'], p['tr'], p['tf'], p['i4'], p['tj']
    t = np.linspace(0.0, TMAX, NPTS)
    t0 = time.time()
    out = evolve_mixed_pair(p['states'], p['fA'], p['fB'], p['i0'], p['i0'], t,
                            vdd_matrix=p['V'], tol=1e-8)
    P2 = np.real(np.asarray(out['pops_2d'], float))
    PA = np.real(np.asarray(out['pops_A'], float))
    PB = np.real(np.asarray(out['pops_B'], float))
    lab = [str(x['label']) for x in out['states_2d']]
    a, b = P2[:, lab.index(K_AFT)], P2[:, lab.index(K_BEF)]
    outm = 0.5 * (PA[:, [2, 4]].sum(axis=1) + PB[:, [2, 4]].sum(axis=1))
    i = int(np.argmax(a))
    tot = float(a[i] + b[i])
    return dict(tag=tag, tr=tr, tf=tf, i4=i4, tj=tj, t_s=round(time.time() - t0, 1),
                pk=float(a[i]), t_pk=float(t[i]), ct=float((a - b).max()),
                out_max=float(outm.max()), pair_sum=tot,
                bef_at_pk=float(b[i]))


def build_tasks(only=None):
    T = []
    if only in (None, 'A'):
        for tr in (0.0, 50.0, 100.0):
            for tf in (0.0, 20.0, 50.0, 100.0):
                T.append(('A/r%g/f%g' % (tr, tf), tr, tf, 0.42, 280.0))
    if only in (None, 'B'):
        for i4 in (0.0, 0.1, 0.2, 0.42, 0.6, 0.8, 1.0):
            T.append(('B/I4=%.2f' % i4, 0.0, 0.0, i4, 280.0))
    if only in (None, 'C'):
        for tj in (230.0, 255.0, 280.0, 330.0, 380.0):
            T.append(('C/tj=%g' % tj, 0.0, 20.0, 0.42, tj))
    return T


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--jobs', type=int, default=10)
    ap.add_argument('--only', default=None)
    a = ap.parse_args()
    tasks = build_tasks(a.only)
    print('  任务 %d 个，%d 进程并行' % (len(tasks), a.jobs))
    print('  父进程准备（建场 + V_dd，ARC 查询全在此完成）…', flush=True)
    payloads = []
    for i, tk in enumerate(tasks):
        payloads.append(prep(tk))
        if (i + 1) % 4 == 0 or i == len(tasks) - 1:
            print('    已准备 %d/%d' % (i + 1, len(tasks)), flush=True)
    print('  准备完成，开始并行演化（子进程不碰 ARC）', flush=True)
    t0 = time.time()
    res = []
    with ProcessPoolExecutor(max_workers=a.jobs) as ex:
        for r in ex.map(one, payloads):
            res.append(r)
            print('  %-12s 峰值 %.4f @%3.0f ns  对比度 %.4f  两通道和 %.4f  '
                  '漏 %.4f  %.0fs'
                  % (r['tag'], r['pk'], r['t_pk'], r['ct'], r['pair_sum'],
                     r['out_max'], r['t_s']), flush=True)
    print('\n  ===== 汇总（按峰值降序）=====')
    for r in sorted(res, key=lambda x: -x['pk']):
        print('  %-12s 峰值 %.4f  两通道和 %.4f  对比度 %.4f'
              % (r['tag'], r['pk'], r['pair_sum'], r['ct']))
    b = max(res, key=lambda x: x['pk'])
    print('\n  最优: %s → 峰值 %.4f（两通道和 %.4f）'
          % (b['tag'], b['pk'], b['pair_sum']))
    print('  总用时 %.0fs' % (time.time() - t0))
    json.dump({'R_um': R_UM, 'theta_deg': THETA, 'res': res},
              io.open(OUTJSON, 'w', encoding='utf-8'), ensure_ascii=False)
    print('  数据 → %s' % OUTJSON)


if __name__ == '__main__':
    main()