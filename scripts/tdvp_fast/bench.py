# -*- coding: utf-8 -*-
"""Independent benchmark for the dense/Torch TDVP prototypes.

Examples
--------
python scripts/tdvp_fast/bench.py --case n12 --chi 64 --device cpu --reference tenpy --runs 3
python scripts/tdvp_fast/bench.py --case n24 --chi 64 --device gpu --dtype complex64 --runs 3
"""
import argparse
import json
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

from src.core.manybody.runspec_mb import drive_from_calib  # noqa: E402

CASES = {
    'n12': 'checks/fixtures/kagome_n12.json',
    'n24': 'checks/fixtures/kagome_n24.json',
}


def _load(case):
    path = os.path.join(ROOT, CASES[case])
    calib = json.load(open(path, encoding='utf-8'))
    n = len(calib['atoms'])
    Om, de, ga = drive_from_calib(calib)
    Om = np.full(n, 0.5, dtype=complex)
    return calib, n, Om, de, ga


def run_custom(calib, n, Om, de, ga, args):
    t = np.linspace(0.0, args.t_max, args.npts)
    kw = dict(psi0='r' + 'g' * (n - 1), t_grid=t, Om=Om, delta=de, gamma=ga,
              chi_max=args.chi, truncerr=0.0 if args.truncerr0 else 1e-10,
              max_step_ns=args.dt)
    if args.device == 'cpu':
        from scripts.tdvp_fast.dense_tdvp import evolve_dense
        return evolve_dense(calib, method=args.method, **kw)
    from scripts.tdvp_fast.torch_tdvp import evolve_torch
    return evolve_torch(calib, device='cuda', dtype=args.dtype,
                        taylor_tol=args.taylor_tol,
                        taylor_order=args.taylor_order,
                        svd_device=args.svd_device, **kw)


def run_tenpy(calib, n, Om, de, ga, args):
    from src.core.manybody.engine_mps import evolve_mps
    t = np.linspace(0.0, args.t_max, args.npts)
    return evolve_mps(calib, psi0='r' + 'g' * (n - 1), t_grid=t, Om=Om,
                      delta=de, gamma=ga, algorithm='tdvp', chi_max=args.chi,
                      truncerr=0.0 if args.truncerr0 else 1e-10,
                      max_step_ns=args.dt, krylov_dim=20,
                      want_observables=False, want_corr=False,
                      want_entropy=False, want_xy_corr=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--case', choices=sorted(CASES), default='n12')
    ap.add_argument('--chi', type=int, default=64)
    ap.add_argument('--t-max', type=float, default=20.0)
    ap.add_argument('--npts', type=int, default=3)
    ap.add_argument('--dt', type=float, default=2.0)
    ap.add_argument('--runs', type=int, default=3)
    ap.add_argument('--device', choices=('cpu', 'gpu'), default='cpu')
    ap.add_argument('--dtype', choices=('complex64', 'complex128'), default='complex64')
    ap.add_argument('--method', choices=('auto', 'taylor', 'lanczos'), default='auto')
    ap.add_argument('--taylor-tol', type=float, default=1e-8)
    ap.add_argument('--taylor-order', type=int, default=16)
    ap.add_argument('--svd-device', choices=('cuda', 'cpu'), default='cuda')
    ap.add_argument('--truncerr0', action='store_true')
    ap.add_argument('--reference', choices=('none', 'tenpy', 'ed'), default='none')
    args = ap.parse_args()

    calib, n, Om, de, ga = _load(args.case)
    times = []
    last = None
    for _ in range(max(1, args.runs)):
        t0 = time.perf_counter()
        last = run_custom(calib, n, Om, de, ga, args)
        times.append(time.perf_counter() - t0)
    out = {
        'case': args.case, 'n': n, 'chi_max': args.chi,
        't_max': args.t_max, 'npts': args.npts, 'dt': args.dt,
        'device': args.device, 'dtype': args.dtype if args.device == 'gpu' else 'complex128',
        'method': args.method,
        'svd_device': args.svd_device if args.device == 'gpu' else None,
        'wall_s_median': round(float(np.median(times)), 6),
        'wall_s_runs': [round(x, 6) for x in times],
        'max_chi': last['max_chi'],
        'pops_last': [round(float(x), 12) for x in last['pops_atom'][-1][:8]],
    }
    if args.reference != 'none':
        t0 = time.perf_counter()
        if args.reference == 'tenpy':
            ref = run_tenpy(calib, n, Om, de, ga, args)
        else:
            from src.core.manybody.engine_ed import evolve_manybody
            t = np.linspace(0.0, args.t_max, args.npts)
            ref = evolve_manybody(calib, psi0='r' + 'g' * (n - 1), t_grid=t,
                                  Om=Om, delta=de, gamma=ga, want_corr=False)
        out['reference'] = args.reference
        out['reference_wall_s'] = round(time.perf_counter() - t0, 6)
        out['max_abs_dpops'] = float(np.max(np.abs(last['pops_atom'] - ref['pops_atom'])))
        if args.reference == 'tenpy':
            out['reference_max_chi'] = ref['error_report']['max_chi']
    print(json.dumps(out, ensure_ascii=False, indent=2))
    return 0


if __name__ == '__main__':
    sys.exit(main())
