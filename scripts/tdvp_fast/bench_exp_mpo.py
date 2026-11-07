# -*- coding: utf-8 -*-
"""Reproducible 1D power-law benchmark: exact MPO vs exponential-sum MPO."""
import argparse
import json
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

from scripts.tdvp_fast.exp_mpo import fit_exponentials_grid, build_exp_mpo
from scripts.tdvp_fast.dense_tdvp import build_mpo, evolve_dense
from src.core.manybody.dmrg import terms_from_calib


def make_calib(n, power=3.0):
    cal = {'atoms': [{'id': i, 'pos': [float(i), 0.0, 0.0]} for i in range(n)],
           'pairs': []}
    for i in range(n):
        for j in range(i + 1, n):
            cal['pairs'].append({'i': i, 'j': j, 'J': 1.0 / (j - i) ** power})
    return cal


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=96)
    ap.add_argument('--k', type=int, default=8)
    ap.add_argument('--power', type=float, default=3.0)
    ap.add_argument('--chi', type=int, default=64)
    ap.add_argument('--t-max', type=float, default=20.0)
    ap.add_argument('--npts', type=int, default=3)
    ap.add_argument('--dt', type=float, default=2.0)
    ap.add_argument('--runs', type=int, default=1)
    args = ap.parse_args()

    n = args.n
    cal = make_calib(n, args.power)
    Om = np.full(n, 0.5, dtype=complex)
    de = np.zeros(n)
    ga = np.zeros(n)
    order, onsite, couplings, info = terms_from_calib(cal, Om, de, ga, order=None)
    W_exact = build_mpo(onsite, couplings, n)
    t = np.linspace(0.0, args.t_max, args.npts)

    def run(W):
        times = []
        last = None
        for _ in range(max(1, args.runs)):
            t0 = time.perf_counter()
            last = evolve_dense(cal, psi0='r' + 'g' * (n - 1), t_grid=t, Om=Om,
                                delta=de, gamma=ga, chi_max=args.chi,
                                truncerr=1e-10, max_step_ns=args.dt,
                                method='auto', W=W)
            times.append(time.perf_counter() - t0)
        return float(np.median(times)), last

    exact_wall, exact = run(W_exact)
    q, c, fit = fit_exponentials_grid(lambda m: 1.0 / m ** args.power,
                                      n - 1, args.k, grid=256)
    W_exp = build_exp_mpo(n, q, c, onsite=onsite)
    exp_wall, exp = run(W_exp)
    out = {
        'n': n, 'k': args.k, 'power': args.power, 'chi_max': args.chi,
        't_max': args.t_max, 'npts': args.npts, 'dt': args.dt,
        'exact_D': int(max(w.shape[0] for w in W_exact)),
        'exp_D': int(max(w.shape[0] for w in W_exp)),
        'fit': fit,
        'exact_wall_s': round(exact_wall, 6),
        'exp_wall_s': round(exp_wall, 6),
        'speedup': round(exact_wall / exp_wall, 3),
        'max_abs_dpops': float(np.max(np.abs(exact['pops_atom'] - exp['pops_atom']))),
        'exact_chi': exact['max_chi'], 'exp_chi': exp['max_chi'],
        'q_range': [float(q.min()), float(q.max())],
    }
    print(json.dumps(out, ensure_ascii=False, indent=2))
    return 0


if __name__ == '__main__':
    sys.exit(main())
