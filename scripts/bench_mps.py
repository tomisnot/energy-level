# -*- coding: utf-8 -*-
"""MPS 性能/观测开销基准（不进入默认判据）。

示例：
  python scripts/bench_mps.py --calib checks/fixtures/kagome_n12.json --chi 64 \
      --t-max 2 --npts 3 --dt 2
  python scripts/bench_mps.py --calib checks/fixtures/kagome_n24.json --gpu \
      --chi 128 --t-max 10 --npts 2 --dt 10
  python scripts/bench_mps.py --calib checks/fixtures/kagome_n24.json --chi 64 \
      --pauli 96 --pauli-order 3 --t-max 10 --npts 3 --dt 10

输出一行 JSON（stdout），含墙钟、内部步数、engine build/reuse、measure 占比；
`chi_actual` 是全过程实际最大键维（`chi` 是 cap，不能当实际值）。
大系统只在显式调用时运行；默认参数是小规模。
"""
import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core.manybody.runspec_mb import drive_from_calib


def make_pauli(n, count, order):
    rng = np.random.default_rng(20261106)
    specs = []
    for s in range(int(count)):
        sites = sorted(rng.choice(n, size=min(order, n), replace=False).tolist())
        specs.append({'label': 's%d' % s,
                      'terms': [[int(i), 'xyz'[int(rng.integers(0, 3))]]
                                for i in sites]})
    return specs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--calib', required=True)
    ap.add_argument('--chi', type=int, default=64)
    ap.add_argument('--npts', type=int, default=3)
    ap.add_argument('--t-max', type=float, default=2.0)
    ap.add_argument('--dt', type=float, default=2.0)
    ap.add_argument('--psi0', default=None, help='默认一个 r + 其余 g')
    ap.add_argument('--omega', type=float, default=0.5,
                    help='均匀 Rabi 驱动 MHz（夹具本身通常无 Ω）')
    ap.add_argument('--pauli', type=int, default=0, help='Pauli 串数')
    ap.add_argument('--pauli-order', type=int, default=3)
    ap.add_argument('--gpu', action='store_true')
    ap.add_argument('--algorithm', default=None,
                    help='默认 CPU=tdvp、GPU=swap-tebd-gpu')
    ap.add_argument('--dtype', choices=('float32', 'float64'), default='float64')
    a = ap.parse_args()

    calib = json.load(open(a.calib, encoding='utf-8'))
    n = len(calib['atoms'])
    t = np.linspace(0.0, a.t_max, a.npts)
    Om, de, ga = drive_from_calib(calib)
    Om = np.full(n, float(a.omega), dtype=complex)
    psi0 = a.psi0 or ('r' + 'g' * (n - 1))
    specs = make_pauli(n, a.pauli, a.pauli_order) if a.pauli else None
    t0 = time.perf_counter()
    if a.gpu:
        from src.core.manybody.engine_mps_gpu import evolve_mps_gpu
        alg = a.algorithm or 'swap-tebd'
        res = evolve_mps_gpu(calib, psi0=psi0, t_grid=t, Om=Om, delta=de,
                             gamma=ga, algorithm=alg, chi_max=a.chi,
                             truncerr=1e-10, max_step_ns=a.dt,
                             want_observables=bool(specs), want_xy_corr=False,
                             want_corr=False, want_entropy=False,
                             dtype=a.dtype, pauli_strings=specs)
        try:
            import torch
            peak = (torch.cuda.max_memory_allocated() / 2**20
                    if torch.cuda.is_available() else None)
        except Exception:
            peak = None
    else:
        from src.core.manybody import backends
        alg = backends.normalize_time_backend(a.algorithm or 'tdvp')
        if alg == 'tdvp-dense':
            from src.core.manybody.tdvp_dense import evolve_tdvp_dense
            res = evolve_tdvp_dense(
                calib, psi0=psi0, t_grid=t, Om=Om, delta=de, gamma=ga,
                chi_max=a.chi, truncerr=1e-10, max_step_ns=a.dt,
                want_observables=bool(specs), want_entropy=False,
                pauli_strings=specs)
        elif alg == 'tdvp-mpo':
            from src.core.manybody.mpo_tdvp import evolve_mpo_tdvp
            if specs:
                raise ValueError('--algorithm tdvp-mpo 当前不支持 Pauli；'
                                 '改用 tdvp-dense/tdvp 或去掉 --pauli')
            res = evolve_mpo_tdvp(
                calib, psi0=psi0, t_grid=t, Om=Om, delta=de, gamma=ga,
                chi_max=a.chi, truncerr=1e-10, max_step_ns=a.dt)
        else:
            from src.core.manybody.engine_mps import evolve_mps
            if backends.TIME_BACKENDS[alg].module != 'engine_mps':
                raise ValueError('bench_mps 的 CPU 路径只接受 MPS 后端；'
                                 'algorithm=%r 请用对应入口' % alg)
            res = evolve_mps(calib, psi0=psi0, t_grid=t, Om=Om, delta=de, gamma=ga,
                             algorithm=alg, chi_max=a.chi, truncerr=1e-10,
                             max_step_ns=a.dt, want_observables=bool(specs),
                             want_xy_corr=False, want_corr=False,
                             want_entropy=False, pauli_strings=specs)
        peak = None
    wall = time.perf_counter() - t0
    err = res.get('error_report') or {}
    timing = err.get('timings_s') or {}
    chi_hist = res.get('chi')
    out = {
        'calib': os.path.basename(a.calib), 'n': n, 'chi': a.chi,
        'npts': a.npts, 't_max_ns': a.t_max, 'dt_ns': a.dt,
        'gpu': bool(a.gpu), 'dtype': a.dtype,
        'pauli': a.pauli, 'pauli_order': a.pauli_order,
        'wall_s': round(wall, 6),
        'n_internal_steps': err.get('n_internal_steps'),
        'engine_builds': timing.get('engine_builds'),
        'engine_reuses': timing.get('engine_reuses'),
        'measure_s': timing.get('measure_s'),
        'measure_frac': (round(float(timing['measure_s']) / wall, 6)
                         if timing.get('measure_s') is not None and wall > 0 else None),
        'peak_cuda_mb': peak,
        'max_chi': err.get('max_chi'),
        'chi_actual': err.get('max_chi'),
        'chi_final': (int(max(chi_hist)) if chi_hist else err.get('max_chi')),
    }
    print(json.dumps(out, ensure_ascii=False, indent=2))
    return 0


if __name__ == '__main__':
    sys.exit(main())
