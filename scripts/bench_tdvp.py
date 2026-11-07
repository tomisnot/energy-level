# -*- coding: utf-8 -*-
"""Small TDVP parameter sweep benchmark.

This is an experimental benchmark and does not change solver defaults.  It
prints one JSON object per configuration so the output can be collected or
compared with a later run.
"""
import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core.manybody.runspec_mb import drive_from_calib


def _ints(text):
    return [int(x.strip()) for x in text.split(',') if x.strip()]


def _floats(text):
    return [float(x.strip()) for x in text.split(',') if x.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--calib', required=True)
    ap.add_argument('--chi', type=int, default=32)
    ap.add_argument('--t-max', type=float, default=8.0)
    ap.add_argument('--npts', type=int, default=5)
    ap.add_argument('--omega', type=float, default=0.5)
    ap.add_argument('--dt', default='1,2,4')
    ap.add_argument('--algorithm', default='tdvp,tdvp-adaptive')
    ap.add_argument('--krylov-dim', default='6,10,20')
    ap.add_argument('--krylov-tol', default='1e-5,1e-6,1e-10')
    ap.add_argument('--adaptive-switch-chi', default='8,16')
    args = ap.parse_args()

    calib = json.load(open(args.calib, encoding='utf-8'))
    n = len(calib['atoms'])
    t = np.linspace(0.0, args.t_max, args.npts)
    _om, de, ga = drive_from_calib(calib)
    om = np.full(n, args.omega, dtype=complex)

    from src.core.manybody.engine_mps import evolve_mps

    for algorithm in [x.strip() for x in args.algorithm.split(',') if x.strip()]:
        for dt in _floats(args.dt):
            for kd in _ints(args.krylov_dim):
                for kt in _floats(args.krylov_tol):
                    switches = (_ints(args.adaptive_switch_chi)
                                if algorithm == 'tdvp-adaptive' else [None])
                    for switch in switches:
                        t0 = time.perf_counter()
                        result = evolve_mps(
                            calib, psi0='r' + 'g' * (n - 1), t_grid=t,
                            Om=om, delta=de, gamma=ga, algorithm=algorithm,
                            chi_max=args.chi, truncerr=1e-10,
                            max_step_ns=dt, krylov_dim=kd,
                            krylov_tol=kt,
                            adaptive_switch_chi=(switch or 16),
                            want_observables=False, want_corr=False,
                            want_entropy=False, want_xy_corr=False)
                        wall = time.perf_counter() - t0
                        err = result.get('error_report') or {}
                        timing = err.get('timings_s') or {}
                        print(json.dumps({
                            'algorithm': algorithm,
                            'dt_ns': dt,
                            'krylov_dim': kd,
                            'krylov_tol': kt,
                            'adaptive_switch_chi': switch,
                            'wall_s': round(wall, 6),
                            'engine_run_s': timing.get('engine_run_s'),
                            'measure_s': timing.get('measure_s'),
                            'n_internal_steps': err.get('n_internal_steps'),
                            'engine_builds': timing.get('engine_builds'),
                            'engine_reuses': timing.get('engine_reuses'),
                            'max_chi': err.get('max_chi'),
                            'adaptive_switched_at': err.get('adaptive_switched_at'),
                            'max_norm_dev': err.get('max_norm_dev'),
                        }, ensure_ascii=False))


if __name__ == '__main__':
    main()
