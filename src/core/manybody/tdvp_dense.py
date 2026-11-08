# -*- coding: utf-8 -*-
"""Dense CPU two-site TDVP backend (finite open chain; no charge/return_state).

Product backend for ``manybody.solver='tdvp-dense'``.  It uses plain numpy
tensors (no TeNPy ``np_conserved`` objects) and the same Hamiltonian convention
as ``src/core/manybody/assemble.py``.  The sweep semantics match TeNPy's
``TwoSiteTDVPEngine``; ``scripts/tdvp_fast/`` keeps the benchmark driver and the
experimental CUDA port.  Cross-backend checks live in
``checks/manybody_tdvp_dense_selfcheck.py``.  The product wrapper supports
piecewise-constant drive segments and arbitrary Pauli strings; the pure
``evolve_dense`` helper below is the static research path.

Conventions
-----------
* local basis ``(|g>, |r>) = (0, 1)``; site 0 is MSB for full-space vectors,
  while MPS sites are stored in the explicit ``order`` (identity here).
* MPO tensor at site i: ``W[i][wL, wR, s_out, s_in]``.
* MPS tensor at site i: ``B[i][l, s, r]`` with canonical form ``(fL, fR)``:
  stored ``B = S_L**fL * A * S_R**fR``, matching TeNPy's form algebra.
"""
import os
from contextlib import nullcontext

import numpy as np

I2 = np.eye(2, dtype=complex)
NMAT = np.diag([0.0, 1.0]).astype(complex)
PMAT = np.array([[0.0, 0.0], [1.0, 0.0]], dtype=complex)   # |r><g|
MMAT = np.array([[0.0, 1.0], [0.0, 0.0]], dtype=complex)   # |g><r|
SXMAT = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex)
OPMAP = {'I': I2, 'Nhat': NMAT, 'N': NMAT, 'P': PMAT, 'M': MMAT, 'SX': SXMAT}


def build_mpo(onsite, couplings, n):
    """Build a position-dependent dense MPO from ``terms_from_calib`` output.

    One channel per (start site, opA, opB).  Channels are sorted by start site
    so that at bond i the active set is a prefix of the channel list; the local
    MPO bond dimension is therefore ``2 + #channels started before i`` rather
    than the full ``2 + #channels`` at every site.  This is exact (no
    approximation) and roughly halves the average contraction cost for a chain
    whose interactions start at different sites.
    """
    groups = {}
    for coeff, a, opA, b, opB in couplings:
        a, b = int(a), int(b)
        if a >= b:
            raise ValueError('couplings must satisfy a < b')
        groups.setdefault((a, opA, opB), []).append((b, complex(coeff)))

    keys = sorted(groups, key=lambda k: (k[0], k[1], k[2]))
    channel_of = {key: g for g, key in enumerate(keys, start=1)}
    G = len(keys)
    start_site = {g: key[0] for key, g in channel_of.items()}

    # local bond lists: boundary states are singleton; internal bonds carry
    # S0, the channels already started, and SD (done) as the last index.
    bonds = []
    for i in range(n + 1):
        if i == 0:
            bonds.append([0])
        elif i == n:
            bonds.append([-1])          # -1 is the right-boundary done state
        else:
            active = [g for g in range(1, G + 1) if start_site[g] < i]
            bonds.append([0] + active + [-1])
    pos = [{v: j for j, v in enumerate(lst)} for lst in bonds]

    W = [np.zeros((len(bonds[i]), len(bonds[i + 1]), 2, 2), dtype=complex)
         for i in range(n)]

    # identity / done channels (S0 propagates; SD propagates after an operator)
    for i in range(n - 1):
        W[i][pos[i][0], pos[i + 1][0]] += I2
    for i in range(1, n - 1):
        W[i][pos[i][-1], pos[i + 1][-1]] += I2
    W[n - 1][pos[n - 1][-1], pos[n][-1]] += I2

    # onsite terms
    for i, (delta, om2) in enumerate(onsite):
        op = complex(delta) * NMAT + complex(om2) * PMAT + np.conj(complex(om2)) * MMAT
        if i < n - 1:
            W[i][pos[i][0], pos[i + 1][-1]] += op
        else:
            W[n - 1][pos[n - 1][0], pos[n][-1]] += op

    # start / continue / end of every two-body channel
    for key in keys:
        g = channel_of[key]
        a = key[0]
        opA = key[1]
        W[a][pos[a][0], pos[a + 1][g]] += OPMAP[opA]
    for g in range(1, G + 1):
        a = start_site[g]
        for k in range(a + 1, n - 1):
            W[k][pos[k][g], pos[k + 1][g]] += I2
    for key in keys:
        g = channel_of[key]
        opB = key[2]
        for b, coeff in groups[key]:
            if b < n - 1:
                W[b][pos[b][g], pos[b + 1][-1]] += coeff * OPMAP[opB]
            else:
                W[n - 1][pos[n - 1][g], pos[n][-1]] += coeff * OPMAP[opB]
    return W


def mpo_dense(W):
    """Contract an MPO to a dense (2^n, 2^n) matrix (tiny-N validation only)."""
    n = len(W)
    d = 2
    # einsum subscripts must be letters (n <= 20 for this tiny-N helper)
    letters = 'abcdefghijklmnopqrstuvwxyz'
    if n > 10:
        raise ValueError('mpo_dense only supports n <= 10')
    subs = []
    for i in range(n):
        subs.append(letters[i] + letters[i + 1] + letters[10 + i] + letters[20 + i])
    out = ''.join(letters[10 + i] for i in range(n)) + ''.join(letters[20 + i] for i in range(n))
    H = np.einsum(','.join(subs) + '->' + out, *W)
    return H.reshape(d ** n, d ** n)


class DenseMPS:
    """Finite open MPS with TeNPy-compatible (fL, fR) canonical forms.

    ``S`` has length n+1 (S[0] = S[n] = [1]).  ``B[i]`` has shape
    ``(chi_i, 2, chi_{i+1})`` and its stored form is ``(fL, fR)``, meaning
    ``B = S_i**fL * A_i * S_{i+1}**fR``.
    """

    def __init__(self, B, S, form):
        self.B = list(B)
        self.S = list(S)
        self.form = list(form)
        self.norm = 1.0

    @classmethod
    def product(cls, n, spec=None, dtype=complex):
        """``spec``: None/string/sequence of local 2-vectors."""
        if spec is None:
            vecs = [np.array([1.0, 0.0], dtype=dtype) for _ in range(n)]
        elif isinstance(spec, str):
            s = spec.replace('|', '').replace('>', '').replace(' ', '')
            if len(s) != n or any(c not in 'gr' for c in s):
                raise ValueError('product state string must have length n over g/r')
            vecs = [np.array([1.0, 0.0], dtype=dtype) if c == 'g'
                    else np.array([0.0, 1.0], dtype=dtype) for c in s]
        else:
            vecs = [np.asarray(v, dtype=dtype).reshape(2) for v in spec]
            if len(vecs) != n:
                raise ValueError('product state needs n local vectors')
        B = [v.reshape(1, 2, 1) for v in vecs]
        S = [np.array([1.0], dtype=float)] + [np.array([1.0], dtype=float) for _ in range(n)]
        return cls(B, S, [(0.0, 1.0)] * n)

    _FORMS = {'A': (1.0, 0.0), 'B': (0.0, 1.0), 'Th': (1.0, 1.0),
              'G': (0.0, 0.0), 'C': (0.5, 0.5)}

    def get_B(self, i, form=None):
        fL, fR = self.form[i]
        if isinstance(form, str):
            form = self._FORMS[form]
        nfL, nfR = (fL, fR) if form is None else form
        if nfL is None:
            nfL = fL
        if nfR is None:
            nfR = fR
        B = self.B[i]
        if nfL != fL:
            B = B * (self.S[i] ** (nfL - fL))[:, None, None]
        if nfR != fR:
            B = B * (self.S[i + 1] ** (nfR - fR))[None, None, :]
        return B

    def set_B(self, i, B, form):
        self.B[i] = B
        self.form[i] = form

    def get_theta(self, i, n=2, formL=1.0, formR=1.0):
        if n == 1:
            return self.get_B(i, (formL, formR))
        theta = self.get_B(i, (formL, None))
        old_fR = self.form[i][1]
        for k in range(1, n):
            new_fR = None if k + 1 < n else formR
            Bk = self.get_B(i + k, (1.0 - old_fR, new_fR))
            old_fR = self.form[i + k][1]
            theta = np.tensordot(theta, Bk, axes=([2], [0]))
        return theta

    def to_dense(self):
        """Materialize the full state vector (tiny-N validation only)."""
        n = len(self.B)
        v = np.array([1.0 + 0.0j])
        for i in range(n):
            B = self.get_B(i, (1.0, 1.0))
            # contract left bond; B shape (chiL, 2, chiR)
            v = np.tensordot(v, B, axes=([-1], [0]))
            # v axes: (p0..p_{i-1}, s_i, chiR) -> move chiR last; it already is
            v = v.reshape(-1, B.shape[2])
        # final right bond is 1
        return v.reshape(-1)


def update_LP(LP, B, W):
    """LP[i] (chiL, D_i, chiL) + B (chiL, d, chiR) + W -> LP[i+1] (chiR, D_{i+1}, chiR).

    Explicit pairwise contractions are much faster than a four-operand einsum:
    the latter lets numpy build an unnecessarily large intermediate.
    """
    # T1[a,k,s,b] = sum_l LP[l,a,k] B[l,s,b]^*
    T1 = np.tensordot(LP, B.conj(), axes=([0], [0]))          # (a,k,s,b)
    # T2[x,k,t,b] = sum_{a,s} W[a,x,s,t] T1[a,k,s,b]
    T2 = np.tensordot(W, T1, axes=([0, 2], [0, 2]))            # (x,t,k,b)
    T2 = T2.transpose(0, 2, 1, 3)                              # (x,k,t,b)
    # out[b,x,d] = sum_{k,t} T2[x,k,t,b] B[k,t,d]
    out = np.tensordot(T2, B, axes=([1, 2], [0, 1]))           # (x,b,d)
    return out.transpose(1, 0, 2)


def update_RP(RP, B, W):
    """RP[i] (chiL, D_i, chiL) + B (chiL, d, chiR) + W -> RP[i-1] (chiL, D_{i-1}, chiL)."""
    # T1[x,d,l,s] = sum_b RP[b,x,d] B[l,s,b]^*
    T1 = np.tensordot(RP, B.conj(), axes=([0], [2]))           # (x,d,l,s)
    T1 = T1.transpose(0, 1, 2, 3)
    # T2[a,d,l,t] = sum_{x,s} W[a,x,s,t] T1[x,d,l,s]
    T2 = np.tensordot(W, T1, axes=([1, 2], [0, 3]))            # (a,t,d,l)
    T2 = T2.transpose(0, 2, 3, 1)                              # (a,d,l,t)
    # out[l,a,k] = sum_{d,t} T2[a,d,l,t] B[k,t,d]
    out = np.tensordot(T2, B, axes=([1, 3], [2, 1]))           # (a,l,k)
    return out.transpose(1, 0, 2)


def init_env(mps, W):
    n = len(mps.B)
    LP = [None] * (n + 1)
    RP = [None] * (n + 1)
    LP[0] = np.ones((1, W[0].shape[0], 1), dtype=complex)
    RP[n] = np.ones((1, W[n - 1].shape[1], 1), dtype=complex)
    for i in range(n):
        LP[i + 1] = update_LP(LP[i], mps.get_B(i, 'A'), W[i])
    for i in range(n - 1, -1, -1):
        RP[i] = update_RP(RP[i + 1], mps.get_B(i, 'B'), W[i])
    return LP, RP


def lanczos_expm(matvec, theta, alpha, krylov_dim=20, tol=1e-10):
    """exp(alpha * H) theta for Hermitian H, full reorthogonalization."""
    norm0 = float(np.linalg.norm(theta))
    if norm0 == 0.0:
        return theta.copy()
    v = theta.reshape(-1) / norm0
    V = [v]
    alphas = []
    betas = []
    for _ in range(int(krylov_dim)):
        w = matvec(V[-1].reshape(theta.shape)).reshape(-1)
        a = float(np.real(np.vdot(V[-1], w)))
        w = w - a * V[-1]
        if betas:
            w = w - betas[-1] * V[-2]
        for u in V:
            w = w - np.vdot(u, w) * u
        b = float(np.linalg.norm(w))
        alphas.append(a)
        if b < tol:
            break
        betas.append(b)
        V.append(w / b)
    m = len(alphas)
    if m == 0:
        return theta.copy()
    T = np.diag(alphas)
    if m > 1:
        T += np.diag(betas[:m - 1], 1) + np.diag(betas[:m - 1], -1)
    evals, evecs = np.linalg.eigh(T)
    coeff = evecs @ (np.exp(alpha * evals) * evecs[0, :])
    out = np.zeros_like(V[0])
    for j in range(m):
        out += coeff[j] * V[j]
    return (out * norm0).reshape(theta.shape)


def two_site_step(mps, i, W, LP, RP, alpha, chi_max=200, truncerr=1e-10,
                  krylov_dim=20, tol=1e-10, method='auto'):
    theta = mps.get_theta(i, 2)
    ops = _two_site_ops_blas(LP[i], W[i], W[i + 1], RP[i + 2])
    theta = local_expm(lambda th: _matvec_two_blas(ops, th), theta, alpha,
                       krylov_dim, tol, method)
    chiL, d, d2, chiR = theta.shape
    mat = theta.reshape(chiL * d, d2 * chiR)
    U, S, Vh = np.linalg.svd(mat, full_matrices=False)
    if S.size == 0:
        raise RuntimeError('empty SVD')
    keep = int(np.sum(S > max(truncerr * S[0], 1e-15)))
    keep = max(1, min(int(chi_max), keep, S.size))
    U = U[:, :keep]
    S = S[:keep]
    Vh = Vh[:keep, :]
    sn = float(np.linalg.norm(S))
    if sn > 0:
        mps.norm *= sn
        S = S / sn
    mps.set_B(i, U.reshape(chiL, d, keep), (1.0, 0.0))
    mps.S[i + 1] = S
    mps.set_B(i + 1, Vh.reshape(keep, d2, chiR), (0.0, 1.0))
    return keep, S


def one_site_step(mps, j, W, LP, RP, alpha, krylov_dim=20, tol=1e-10,
                  method='auto'):
    theta = mps.get_theta(j, 1)
    ops = _one_site_ops_blas(LP[j], W[j], RP[j + 1])
    theta = local_expm(lambda th: _matvec_one_blas(ops, th), theta,
                       alpha, krylov_dim, tol, method)
    mps.set_B(j, theta, (1.0, 1.0))


def sweep(mps, W, LP, RP, dt_ns, chi_max=200, truncerr=1e-10,
          krylov_dim=20, tol=1e-10, method='auto', one_site=True):
    n = len(mps.B)
    dt = float(dt_ns) * (2.0 * np.pi * 1e-3)
    for i in range(0, n - 2):
        two_site_step(mps, i, W, LP, RP, -0.5j * dt, chi_max, truncerr,
                      krylov_dim, tol, method)
        LP[i + 1] = update_LP(LP[i], mps.B[i], W[i])
        if one_site:
            one_site_step(mps, i + 1, W, LP, RP, +0.5j * dt,
                          krylov_dim, tol, method)
    for i in range(n - 2, -1, -1):
        alpha = -1.0j * dt if i == n - 2 else -0.5j * dt
        two_site_step(mps, i, W, LP, RP, alpha, chi_max, truncerr,
                      krylov_dim, tol, method)
        if i > 0:
            RP[i + 1] = update_RP(RP[i + 2], mps.B[i + 1], W[i + 1])
            if one_site:
                one_site_step(mps, i, W, LP, RP, +0.5j * dt,
                              krylov_dim, tol, method)


def expectation_pops(mps, op=NMAT):
    n = len(mps.B)
    out = np.zeros(n, dtype=float)
    for i in range(n):
        th = mps.get_theta(i, 1)
        out[i] = float(np.real(np.einsum('lsr,st,ltr->', th.conj(), op, th)))
    return out


def entropy_mid(mps):
    s = np.asarray(mps.S[len(mps.B) // 2], dtype=float)
    p = s * s
    p = p[p > 0]
    return float(-np.sum(p * np.log(p)))


def evolve_dense(calib, psi0=None, t_grid=None, Om=None, delta=None,
                 gamma=None, terms=None, chi_max=200, truncerr=1e-10,
                 max_step_ns=None, krylov_dim=20, tol=1e-10, progress=None,
                 method='auto', order=None, W=None, one_site=True):
    """Standalone static finite two-site TDVP evolution (research path)."""
    from src.core.manybody.dmrg import terms_from_calib
    from src.core.manybody.couplings import active_terms

    n = len(calib.get('atoms') or [])
    if n < 2:
        raise ValueError('dense TDVP needs n >= 2')
    t = np.asarray(t_grid, float)
    if t.ndim != 1 or t.size < 2 or np.any(np.diff(t) < 0):
        raise ValueError('t_grid must be a nondecreasing 1-D array with >=2 points')
    Om = np.zeros(n, complex) if Om is None else np.asarray(Om, complex)
    delta = np.zeros(n) if delta is None else np.asarray(delta, float)
    gamma = np.zeros(n) if gamma is None else np.asarray(gamma, float)
    if np.any(np.abs(gamma) > 0.0):
        raise ValueError('dense TDVP prototype only supports gamma=0 (Hermitian)')
    selected = dict(calib, terms=list(active_terms(calib) if terms is None else terms))
    order, onsite, couplings, info = terms_from_calib(selected, Om, delta, gamma,
                                                      order=order)
    order = list(map(int, order))
    if W is None:
        W = build_mpo(onsite, couplings, n)
    mps = DenseMPS.product(n, psi0)
    if order != list(range(n)):
        mps = DenseMPS([mps.B[a] for a in order],
                       [np.array([1.0])] + [np.array([1.0]) for _ in range(n)],
                       [(0.0, 1.0)] * n)
    LP, RP = init_env(mps, W)
    t_total = float(t[-1] - t[0])
    max_step_ns = (None if max_step_ns is None or max_step_ns <= 0
                   else float(max_step_ns))
    pops = np.zeros((len(t), n), dtype=float)
    ent = np.zeros(len(t), dtype=float)
    def _pops():
        pm = expectation_pops(mps)
        pa = np.empty(n, dtype=float)
        pa[order] = pm
        return pa

    pops[0] = _pops()
    ent[0] = entropy_mid(mps)
    for k in range(1, len(t)):
        span = float(t[k] - t[k - 1])
        nsub = 1 if max_step_ns is None else max(1, int(np.ceil(span / max_step_ns)))
        for a, b in zip(np.linspace(t[k - 1], t[k], nsub + 1)[:-1],
                        np.linspace(t[k - 1], t[k], nsub + 1)[1:]):
            sweep(mps, W, LP, RP, b - a, chi_max, truncerr, krylov_dim, tol,
                  method, one_site)
            if progress is not None:
                progress(float((k - 1 + (b - t[k - 1]) / span) / (len(t) - 1)))
        pops[k] = _pops()
        ent[k] = entropy_mid(mps)
    if progress is not None:
        progress(1.0)
    return dict(t=t, pops_atom=pops, S=ent, n=n, chi=[len(s) for s in mps.S],
                max_chi=max(len(s) for s in mps.S))


def taylor_expm(matvec, theta, alpha, tol=1e-10, max_order=24):
    """exp(alpha*H) theta by a truncated Taylor series (small |alpha|*||H||).

    Each iteration is one matvec, so this is typically much cheaper than a
    20-step Lanczos when ``max_step_ns`` keeps the local phase small.  The loop
    stops when the last term is below ``tol`` relative to the input norm; if it
    does not converge within ``max_order`` the caller should fall back to
    Lanczos.
    """
    norm0 = float(np.linalg.norm(theta))
    if norm0 == 0.0:
        return theta.copy(), 0
    out = theta.copy()
    term = theta.copy()
    for k in range(1, int(max_order) + 1):
        term = matvec(term) * (alpha / k)
        out = out + term
        if float(np.linalg.norm(term)) <= tol * norm0:
            return out, k
    return None, int(max_order)


def local_expm(matvec, theta, alpha, krylov_dim=20, tol=1e-10,
               method='auto', taylor_max_order=24):
    """Apply exp(alpha*H) with either Taylor or Lanczos.

    ``auto`` tries Taylor first (cheap for small local phases) and falls back
    to Lanczos if the Taylor series does not reach ``tol``.
    """
    if method in ('taylor', 'auto'):
        out, order = taylor_expm(matvec, theta, alpha, tol, taylor_max_order)
        if out is not None:
            return out
        if method == 'taylor':
            raise RuntimeError('Taylor series did not converge; increase max_order')
    return lanczos_expm(matvec, theta, alpha, krylov_dim, tol)


def _two_site_ops_blas(L, W_i, W_next, R):
    """Precompute BLAS-ready matrices for the two-site effective H matvec."""
    chiL = L.shape[0]
    d = W_i.shape[2]
    Dm = W_i.shape[1]
    chiR = R.shape[0]
    # LW[x,s,t,b,l] -> rows (x,s,b), cols (l,s)
    LW = np.tensordot(L, W_i, axes=([1], [0]))       # (x,l,b,s,t)
    LW = LW.transpose(0, 3, 4, 2, 1)                  # (x,s,t,b,l)
    LW2 = LW.transpose(0, 1, 3, 4, 2).reshape(chiL * d * Dm, chiL * d)
    # RW[b,y,t,s,r] -> rows (b,t,r), cols (y,t2)  [after transpose below]
    RW = np.tensordot(W_next, R, axes=([1], [1]))     # (b,t,s,y,r)
    RW = RW.transpose(0, 3, 1, 2, 4)                  # (b,y,t,s,r)
    # rows (b,s,r) [s is the input physical t of site i+1], cols (y,t2)
    RW2 = RW.transpose(0, 3, 4, 1, 2).reshape(Dm * d * chiR, chiR * d)
    return LW2, RW2, (chiL, d, chiR, Dm)


def _matvec_two_blas(ops, theta):
    LW2, RW2, (chiL, d, chiR, Dm) = ops
    th1 = theta.reshape(chiL * d, d * chiR)
    tmp1 = LW2 @ th1                                  # (chiL*d*D, d*chiR)
    tmp = tmp1.reshape(chiL, d, Dm, d, chiR)
    tmp2 = tmp.reshape(chiL * d, Dm * d * chiR)
    out = tmp2 @ RW2                                  # (chiL*d, chiR*d)
    return out.reshape(chiL, d, chiR, d).transpose(0, 1, 3, 2)


def _one_site_ops_blas(L, W, R):
    chiL = L.shape[0]
    d = W.shape[2]
    Dm = W.shape[1]
    chiR = R.shape[0]
    LW = np.tensordot(L, W, axes=([1], [0]))          # (x,l,b,s,t)
    LW = LW.transpose(0, 3, 4, 2, 1)                  # (x,s,t,b,l)
    LW2 = LW.transpose(0, 1, 3, 4, 2).reshape(chiL * d * Dm, chiL * d)
    # R[y,b,r] -> rows (b,r), cols y
    R2 = R.transpose(1, 2, 0).reshape(Dm * chiR, chiR)
    return LW2, R2, (chiL, d, chiR, Dm)


def _matvec_one_blas(ops, theta):
    LW2, R2, (chiL, d, chiR, Dm) = ops
    th1 = theta.reshape(chiL * d, chiR)
    tmp1 = LW2 @ th1                                  # (chiL*d*D, chiR)
    tmp = tmp1.reshape(chiL * d, Dm * chiR)
    out = tmp @ R2                                    # (chiL*d, chiR)
    return out.reshape(chiL, d, chiR)


# ---------------------------------------------------------------------------
# Product entry point
# ---------------------------------------------------------------------------
from .runtime import (check_cancel, report_progress, reject_options,
                      segment_drive, segment_intervals)
from .results import time_result


def _psi0_to_vectors(spec, n):
    """Product-state spec (atom order) -> list of local 2-vectors."""
    if spec is None:
        return [np.array([1.0, 0.0], complex) for _ in range(n)]
    if isinstance(spec, str):
        if spec == 'ground':
            raise ValueError('tdvp-dense 不支持 psi0=ground；请用 ed 或先给乘积初态')
        s = spec.replace('|', '').replace('>', '').replace(' ', '')
        if len(s) != n or any(c not in 'gr' for c in s):
            raise ValueError('tdvp-dense 的 psi0 模式串须为长度 N 的 g/r')
        return [np.array([1.0, 0.0], complex) if c == 'g'
                else np.array([0.0, 1.0], complex) for c in s]
    if isinstance(spec, (int, np.integer)) and not isinstance(spec, bool):
        if not 0 <= int(spec) < (1 << n):
            raise ValueError('tdvp-dense 的 psi0 产物态索引越界')
        return [np.array([1.0, 0.0], complex) if ((int(spec) >> (n - 1 - i)) & 1) == 0
                else np.array([0.0, 1.0], complex) for i in range(n)]
    if isinstance(spec, dict):
        kind = spec.get('kind', 'product')
        if kind != 'product':
            raise ValueError('tdvp-dense 只支持乘积初态，收到 kind=%r' % kind)
        raw = spec.get('amplitudes', spec.get('local_states'))
        if raw is None:
            raise ValueError('tdvp-dense product 初态需要 amplitudes/local_states')
        spec = raw
    if isinstance(spec, (list, tuple)) and len(spec) == n:
        out = []
        for v in spec:
            a = np.asarray(v, complex).reshape(-1)
            if a.size != 2:
                raise ValueError('tdvp-dense 局域向量须为长度 2')
            out.append(a)
        return out
    raise ValueError('tdvp-dense 无法解析 psi0=%r' % (spec,))


def _observables_local(mps, n):
    mats = dict(sx=SXMAT,
                sy=np.array([[0.0, -1.0j], [1.0j, 0.0]], complex),
                sz=np.diag([1.0, -1.0]).astype(complex),
                sp=PMAT, sm=MMAT)
    out = {k: np.zeros(n, complex) for k in mats}
    phase = np.full(n, np.nan)
    for i in range(n):
        th = mps.get_theta(i, 1)
        for k, op in mats.items():
            out[k][i] = np.einsum('lsr,st,ltr->', th.conj(), op, th)
        amp = abs(out['sm'][i])
        if amp > 1e-12:
            phase[i] = float(np.angle(out['sm'][i]))
    out['phase'] = phase
    return out


def _blas_thread_limit():
    """Opt-in BLAS thread cap for this run (EL_TDVP_BLAS_THREADS).

    BLAS threads are process-global; the backend does not choose them by
    default.  Small systems are faster with 1 thread, larger ones with ~8.
    """
    raw = os.environ.get('EL_TDVP_BLAS_THREADS')
    if not raw:
        return nullcontext()
    try:
        from threadpoolctl import threadpool_limits
        n = int(raw)
        if n <= 0:
            return nullcontext()
        return threadpool_limits(limits=n)
    except Exception:
        return nullcontext()


_PAULI_DENSE = {
    'x': np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex),
    'y': np.array([[0.0, -1.0j], [1.0j, 0.0]], dtype=complex),
    'z': np.array([[1.0, 0.0], [0.0, -1.0]], dtype=complex),
}


def _pauli_expectation_dense(mps, specs):
    """<psi|P|psi> for Pauli strings, specs in MPS order.

    Transfer-matrix contraction on the finite MPS (no 2^N materialization):
    E' [rB,rK] = sum E[lB,lK] conj(B[lB,s,rB]) O[s,t] B[lK,t,rK].
    Uses form 'B' on every site so the bond singular values are included once.
    """
    n = len(mps.B)
    if not specs:
        return []
    Bs = [mps.get_B(i, 'B') for i in range(n)]
    out = []
    for spec in specs:
        ops = {int(s): _PAULI_DENSE[ax] for s, ax in spec['terms']}
        E = np.ones((1, 1), dtype=complex)
        for i in range(n):
            B = Bs[i]
            T = np.tensordot(E, B.conj(), axes=([0], [0]))   # (lK,s,rB)
            O = ops.get(i)
            if O is None:
                E = np.tensordot(T, B, axes=([0, 1], [0, 1]))
            else:
                T2 = np.tensordot(T, O, axes=([1], [0])).transpose(0, 2, 1)
                E = np.tensordot(T2, B, axes=([0, 1], [0, 1]))
        out.append(complex(E[0, 0]))
    return out


@time_result(approximation='two-site TDVP (dense CPU)')
def evolve_tdvp_dense(calib, psi0=None, t_grid=None, Om=None, delta=None,
                      gamma=None, terms=None, segments=None, progress_arr=None,
                      log=None, chi_max=200, truncerr=1e-10, max_step_ns=None,
                      krylov_dim=20, krylov_tol=1e-10, mps_order=None,
                      want_observables=False, want_corr=False, want_entropy=True,
                      link_pairs=None, loops=None, plan=None, return_state=False,
                      charge_sector=None, pauli_strings=None, drive_interp='left',
                      **kwargs):
    """Dense CPU two-site TDVP (finite open chain; no charge/return/corr).

    Supports piecewise-constant drive segments and arbitrary Pauli strings.
    Segments rebuild only the onsite MPO at each boundary and re-initialize the
    environments; the two-body couplings and the MPS state are kept.
    """
    reject_options(kwargs)
    if plan is not None and plan.get('projected'):
        raise ValueError('tdvp-dense 不支持 projected HamiltonianPlan；请用 ed')
    if return_state:
        raise ValueError('tdvp-dense 不支持 return_state；请用 tdvp/ed')
    if charge_sector is not None:
        raise ValueError('tdvp-dense 不支持 charge_sector；请用 tdvp')
    if link_pairs or loops:
        raise ValueError('tdvp-dense 暂不支持 link_pairs/loops；请用 tdvp')
    if want_corr:
        raise ValueError('tdvp-dense 暂不支持 corr；请用 tdvp/ed')
    if gamma is not None and np.any(np.asarray(gamma) != 0):
        raise ValueError('tdvp-dense 只支持厄米演化 gamma=0')

    n = len(calib.get('atoms') or [])
    if n < 2:
        raise ValueError('tdvp-dense 需要至少 2 个原子')
    t = np.asarray(t_grid, float)
    if t.ndim != 1 or t.size < 2 or not np.isfinite(t).all() or np.any(np.diff(t) < 0):
        raise ValueError('tdvp-dense 时间网格须为至少两个单调不减的有限点')
    Om = np.zeros(n, complex) if Om is None else np.asarray(Om, complex)
    delta = np.zeros(n) if delta is None else np.asarray(delta, float)
    gamma = np.zeros(n) if gamma is None else np.asarray(gamma, float)
    if Om.shape != (n,) or delta.shape != (n,) or gamma.shape != (n,):
        raise ValueError('tdvp-dense 的 Om/delta/gamma 长度必须等于原子数')
    from .couplings import active_terms
    from .dmrg import terms_from_calib
    from .observables import normalize_pauli_strings, pauli_specs_to_mps_order

    selected = dict(calib, terms=list(active_terms(calib) if terms is None else terms))
    order, onsite, couplings, info = terms_from_calib(selected, Om, delta, gamma,
                                                      order=mps_order)
    order = list(map(int, order))
    W = build_mpo(onsite, couplings, n)

    vecs_atom = _psi0_to_vectors(psi0, n)
    mps = DenseMPS([vecs_atom[a].reshape(1, 2, 1) for a in order],
                   [np.array([1.0])] + [np.array([1.0]) for _ in range(n)],
                   [(0.0, 1.0)] * n)
    mps.norm = 1.0
    LP, RP = init_env(mps, W)

    max_step_ns = (None if max_step_ns is None or float(max_step_ns) <= 0
                   else float(max_step_ns))
    n_t = len(t)
    pops = np.zeros((n_t, n), float)
    ent = np.zeros(n_t, float)
    obs_series = None
    if want_observables:
        obs_series = {k: np.zeros((n_t, n), complex)
                      for k in ('sx', 'sy', 'sz', 'sp', 'sm')}
        obs_series['phase'] = np.full((n_t, n), np.nan)

    pauli_specs_atom = normalize_pauli_strings(pauli_strings, n_atoms=n)
    pauli_specs_mps = None
    pauli_labels = None
    if pauli_specs_atom:
        pauli_specs_mps = pauli_specs_to_mps_order(pauli_specs_atom, order)
        pauli_labels = [s['label'] for s in pauli_specs_atom]
    pauli_series = (np.zeros((n_t, len(pauli_specs_mps)), complex)
                    if pauli_specs_mps else None)

    def _pops_atom():
        pm = expectation_pops(mps)
        pa = np.empty(n, float)
        pa[order] = pm
        return pa

    def _record(k):
        pops[k] = _pops_atom()
        ent[k] = entropy_mid(mps) if want_entropy else 0.0
        if obs_series is not None:
            o = _observables_local(mps, n)
            for key in ('sx', 'sy', 'sz', 'sp', 'sm', 'phase'):
                arr = np.empty(n, dtype=o[key].dtype)
                arr[order] = o[key]
                obs_series[key][k] = arr
        if pauli_series is not None:
            pauli_series[k] = _pauli_expectation_dense(mps, pauli_specs_mps)
        if progress_arr is not None:
            progress_arr[0] = (k + 1) / n_t

    check_cancel(progress_arr)
    _record(0)
    n_internal = 0
    n_segment_builds = 0
    current_seg_key = None
    with _blas_thread_limit():
        for k in range(1, n_t):
            if segments:
                steps = []
                for left, right in segment_intervals(t[k - 1], t[k], segments):
                    nsub = (1 if max_step_ns is None else
                            max(1, int(np.ceil((right - left) / max_step_ns))))
                    grid = np.linspace(left, right, nsub + 1)
                    steps.extend(zip(grid[:-1], grid[1:]))
            else:
                span = float(t[k] - t[k - 1])
                nsub = 1 if max_step_ns is None else max(1, int(np.ceil(span / max_step_ns)))
                steps = zip(np.linspace(t[k - 1], t[k], nsub + 1)[:-1],
                            np.linspace(t[k - 1], t[k], nsub + 1)[1:])
            for a, b in steps:
                check_cancel(progress_arr)
                if segments:
                    mid = 0.5 * (float(a) + float(b))
                    key = max([float(s['t0']) for s in segments
                               if float(s['t0']) <= mid],
                              default=min(float(s['t0']) for s in segments))
                    if key != current_seg_key:
                        Om_k, de_k = segment_drive(mid, segments, n, interp=drive_interp)
                        _, on_k, cp_k, _ = terms_from_calib(
                            selected, Om_k, de_k, gamma, order=order)
                        W = build_mpo(on_k, cp_k, n)
                        LP, RP = init_env(mps, W)
                        current_seg_key = key
                        n_segment_builds += 1
                sweep(mps, W, LP, RP, float(b - a), chi_max, truncerr,
                      krylov_dim, krylov_tol, method='auto')
                n_internal += 1
                check_cancel(progress_arr)
                if log is not None:
                    log('[tdvp-dense] t=%.6g chi=%d' % (b, max(len(s) for s in mps.S)))
            _record(k)
    if progress_arr is not None:
        progress_arr[0] = 1.0

    states = [{'label': 'atom%d@%s' % (a['id'], tuple(a.get('pos', ()))),
               'atom_id': a['id']} for a in calib['atoms']]
    err = dict(backend='tdvp-dense', algorithm='two-site-dense', n_atoms=n,
               n_t=n_t, chi_max=int(chi_max), truncerr=float(truncerr),
               max_chi=int(max(len(s) for s in mps.S)),
               max_step_ns=max_step_ns, n_internal_steps=n_internal,
               mps_order=order, norm_end=float(mps.norm),
               local_expm='taylor/lanczos auto',
               blas_threads=(int(os.environ['EL_TDVP_BLAS_THREADS'])
                             if os.environ.get('EL_TDVP_BLAS_THREADS') else None))
    if segments:
        err['n_segments'] = len(segments)
        err['segment_model_builds'] = n_segment_builds
        err['drive_interp'] = drive_interp
    if pauli_specs_mps:
        err['pauli_count'] = len(pauli_specs_mps)
    obs = None
    if obs_series is not None:
        obs = dict(obs_series)
    if pauli_series is not None:
        if obs is None:
            obs = {}
        obs['pauli_strings'] = pauli_series
        obs['pauli_string_labels'] = pauli_labels
    loss = 1.0 - float(mps.norm) if mps.norm is not None else None
    return dict(t=t, pops=pops, pops_atom=pops, states=states,
                method='manybody', backend='tdvp-dense',
                S=ent if want_entropy else None, corr=None, loss=loss,
                error_report=err, observables=obs, diag=None)
