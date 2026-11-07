# -*- coding: utf-8 -*-
"""Constant-bond-dimension MPO for 1D power-law couplings.

J(m) is approximated by sum_k c_k q_k**m (q_k in (0,1)).  One MPO channel per
exponential/operator pair: start with q*opA, continue with q*I, end with c*opB.
"""
import numpy as np
from scipy.optimize import least_squares

from src.core.manybody.tdvp_dense import I2, NMAT, PMAT, MMAT, OPMAP


def fit_exponentials(J, m_max, n_terms, n_restarts=8, seed=0):
    """Fit J(m) ~ sum_k c_k q_k**m for m=1..m_max."""
    m = np.arange(1, m_max + 1, dtype=float)
    y = np.asarray(J(m), dtype=float)
    scale = np.maximum(np.abs(y), 1e-8)
    rng = np.random.default_rng(seed)
    best = None
    for r in range(n_restarts):
        # initial q spread over decades; c from a linear least-squares solve
        q0 = np.exp(-np.linspace(0.05, 3.0, n_terms) * (1.0 + 0.15 * r))
        A = q0[:, None] ** m[None, :]
        c0 = np.linalg.lstsq(A.T, y, rcond=None)[0]
        p0 = np.concatenate([np.log(q0 / (1.0 - q0)), c0])
        def resid(p):
            q = 1.0 / (1.0 + np.exp(-p[:n_terms]))
            c = p[n_terms:]
            pred = np.sum(c[:, None] * q[:, None] ** m[None, :], axis=0)
            return (pred - y) / scale
        try:
            res = least_squares(resid, p0, max_nfev=2000)
        except Exception:
            continue
        if best is None or res.cost < best.cost:
            best = res
    q = 1.0 / (1.0 + np.exp(-best.x[:n_terms]))
    c = best.x[n_terms:]
    pred = np.sum(c[:, None] * q[:, None] ** m[None, :], axis=0)
    return q, c, dict(max_abs=float(np.max(np.abs(pred - y))),
                      max_rel=float(np.max(np.abs(pred - y) / scale)),
                      rms_rel=float(np.sqrt(np.mean(((pred - y) / scale) ** 2))))


def build_exp_mpo(n, q, c, onsite=None, op_pairs=(('P', 'M'), ('M', 'P'))):
    """Build a constant-D MPO for sum_{a<b} J(b-a) opA_a opB_b (plus onsite)."""
    K = len(q)
    G = K * len(op_pairs)
    S0, SD = 0, G + 1
    D = G + 2
    W = [np.zeros((D, D, 2, 2), dtype=complex) for _ in range(n)]
    # identity / done; left boundary has only S0, right only SD
    W[0][S0, S0] += I2
    for i in range(1, n - 1):
        W[i][S0, S0] += I2
        W[i][SD, SD] += I2
    W[n - 1][SD, 0] += I2
    # onsite
    if onsite is not None:
        for i, (delta, om2) in enumerate(onsite):
            op = complex(delta) * NMAT + complex(om2) * PMAT + np.conj(complex(om2)) * MMAT
            if i < n - 1:
                W[i][S0, SD] += op
            else:
                W[n - 1][S0, 0] += op
    g = 1
    for k in range(K):
        for opA, opB in op_pairs:
            # start at every site: S0 -> g with q*opA
            for i in range(n - 1):
                W[i][S0, g] += q[k] * OPMAP[opA]
            # continue with q*I at every internal/end site
            for i in range(1, n - 1):
                W[i][g, g] += q[k] * I2
            # end at every site: g -> SD with c*opB
            for i in range(1, n - 1):
                W[i][g, SD] += c[k] * OPMAP[opB]
            W[n - 1][g, 0] += c[k] * OPMAP[opB]
            g += 1
    return W


def fit_exponentials_grid(J, m_max, n_terms, grid=256, q_min=1e-3, q_max=0.999,
                          refine=True):
    """OMP-on-a-q-grid fit; monotone in n_terms and free of local minima.

    The basis is q_k**m on a log-spaced grid; OMP selects the K most useful
    decays and solves nonnegative least squares on the selected set.  An
    optional nonlinear polish refines q and c.
    """
    from scipy.optimize import nnls, least_squares

    m = np.arange(1, m_max + 1, dtype=float)
    y = np.asarray(J(m), dtype=float)
    scale = np.maximum(np.abs(y), 1e-8)
    q_grid = np.geomspace(q_min, q_max, int(grid))
    A = q_grid[:, None] ** m[None, :]                 # (grid, M)
    norms = np.linalg.norm(A, axis=1)
    selected = []
    residual = y.copy()
    for _ in range(int(n_terms)):
        corr = (A @ residual) / np.maximum(norms, 1e-300)
        corr[selected] = -np.inf
        idx = int(np.argmax(corr))
        selected.append(idx)
        As = A[selected].T
        try:
            c, _ = nnls(As, y, maxiter=10 * len(y))
        except RuntimeError:
            c = np.maximum(np.linalg.lstsq(As, y, rcond=None)[0], 0.0)
        residual = y - As @ c
    q = q_grid[selected].copy()
    c = c.copy()
    if refine:
        # keep q in (q_min, q_max) via logit; c free (starting from NNLS)
        lo, hi = np.log(q_min / (1 - q_min)), np.log(q_max / (1 - q_max))
        p0 = np.concatenate([np.clip(np.log(q / (1 - q)), lo + 1e-6, hi - 1e-6), c])
        def resid(p):
            qq = 1.0 / (1.0 + np.exp(-np.clip(p[:len(q)], -60, 60)))
            cc = p[len(q):]
            pred = np.sum(cc[:, None] * qq[:, None] ** m[None, :], axis=0)
            return (pred - y) / scale
        try:
            res = least_squares(resid, p0, max_nfev=4000)
            q = 1.0 / (1.0 + np.exp(-np.clip(res.x[:len(q)], -60, 60)))
            c = res.x[len(q):]
        except Exception:
            pass
    pred = np.sum(c[:, None] * q[:, None] ** m[None, :], axis=0)
    return q, c, dict(max_abs=float(np.max(np.abs(pred - y))),
                      max_rel=float(np.max(np.abs(pred - y) / scale)),
                      rms_rel=float(np.sqrt(np.mean(((pred - y) / scale) ** 2))))
