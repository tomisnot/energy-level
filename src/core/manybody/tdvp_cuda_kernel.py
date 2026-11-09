# -*- coding: utf-8 -*-
"""Shared CUDA two-site TDVP kernel, including the CPU SVD branch.

Promoted from scripts/tdvp_fast/torch_tdvp.py after N24 benchmarks. The
prototype compatibility entry and the product backend use this same kernel.
"""
import math
import numpy as np
import torch

from .tdvp_dense import build_mpo, expectation_pops as _np_pops

_FORMS = {'A': (1.0, 0.0), 'B': (0.0, 1.0), 'Th': (1.0, 1.0),
          'G': (0.0, 0.0), 'C': (0.5, 0.5)}


class TorchMPS:
    def __init__(self, B, S, form):
        self.B = list(B)
        self.S = list(S)
        self.form = list(form)

    @classmethod
    def product(cls, n, spec=None, device='cuda', dtype=torch.complex64):
        if spec is None:
            vecs = [torch.tensor([1.0, 0.0], dtype=dtype, device=device)
                    for _ in range(n)]
        elif isinstance(spec, str):
            s = spec.replace('|', '').replace('>', '').replace(' ', '')
            if len(s) != n or any(c not in 'gr' for c in s):
                raise ValueError('product state string must have length n over g/r')
            vecs = [torch.tensor([1.0, 0.0] if c == 'g' else [0.0, 1.0],
                                 dtype=dtype, device=device) for c in s]
        else:
            vecs = [torch.as_tensor(v, dtype=dtype, device=device).reshape(2)
                    for v in spec]
        B = [v.reshape(1, 2, 1) for v in vecs]
        one = torch.ones(1, dtype=torch.float32, device=device)
        S = [one.clone() for _ in range(n + 1)]
        return cls(B, S, [(0.0, 1.0)] * n)

    def get_B(self, i, form=None):
        fL, fR = self.form[i]
        if isinstance(form, str):
            form = _FORMS[form]
        nfL, nfR = (fL, fR) if form is None else form
        if nfL is None:
            nfL = fL
        if nfR is None:
            nfR = fR
        B = self.B[i]
        if nfL != fL:
            B = B * (self.S[i] ** (nfL - fL)).reshape(-1, 1, 1)
        if nfR != fR:
            B = B * (self.S[i + 1] ** (nfR - fR)).reshape(1, 1, -1)
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
            theta = torch.tensordot(theta, Bk, dims=([2], [0]))
        return theta


def update_LP(LP, B, W):
    T1 = torch.tensordot(LP, B.conj(), dims=([0], [0]))       # (a,k,s,b)
    T2 = torch.tensordot(W, T1, dims=([0, 2], [0, 2]))         # (x,t,k,b)
    T2 = T2.permute(0, 2, 1, 3)                                 # (x,k,t,b)
    out = torch.tensordot(T2, B, dims=([1, 2], [0, 1]))        # (x,b,d)
    return out.permute(1, 0, 2)


def update_RP(RP, B, W):
    T1 = torch.tensordot(RP, B.conj(), dims=([0], [2]))        # (x,d,l,s)
    T2 = torch.tensordot(W, T1, dims=([1, 2], [0, 3]))         # (a,t,d,l)
    T2 = T2.permute(0, 2, 3, 1)                                 # (a,d,l,t)
    out = torch.tensordot(T2, B, dims=([1, 3], [2, 1]))        # (a,l,k)
    return out.permute(1, 0, 2)


def init_env(mps, W):
    n = len(mps.B)
    device = mps.B[0].device
    dtype = mps.B[0].dtype
    LP = [None] * (n + 1)
    RP = [None] * (n + 1)
    LP[0] = torch.ones((1, W[0].shape[0], 1), dtype=dtype, device=device)
    RP[n] = torch.ones((1, W[n - 1].shape[1], 1), dtype=dtype, device=device)
    for i in range(n):
        LP[i + 1] = update_LP(LP[i], mps.get_B(i, 'A'), W[i])
    for i in range(n - 1, -1, -1):
        RP[i] = update_RP(RP[i + 1], mps.get_B(i, 'B'), W[i])
    return LP, RP


def _two_site_ops_blas(L, W_i, W_next, R):
    chiL = L.shape[0]
    d = W_i.shape[2]
    Dm = W_i.shape[1]
    chiR = R.shape[0]
    LW = torch.tensordot(L, W_i, dims=([1], [0]))       # (x,l,b,s,t)
    LW = LW.permute(0, 3, 4, 2, 1)                      # (x,s,t,b,l)
    LW2 = LW.permute(0, 1, 3, 4, 2).reshape(chiL * d * Dm, chiL * d)
    RW = torch.tensordot(W_next, R, dims=([1], [1]))    # (b,t,s,y,r)
    RW = RW.permute(0, 3, 1, 2, 4)                      # (b,y,t,s,r)
    RW2 = RW.permute(0, 3, 4, 1, 2).reshape(Dm * d * chiR, chiR * d)
    return LW2, RW2, (chiL, d, chiR, Dm)


def _matvec_two_blas(ops, theta):
    LW2, RW2, (chiL, d, chiR, Dm) = ops
    tmp1 = LW2 @ theta.reshape(chiL * d, d * chiR)
    tmp = tmp1.reshape(chiL, d, Dm, d, chiR)
    out = tmp.reshape(chiL * d, Dm * d * chiR) @ RW2
    return out.reshape(chiL, d, chiR, d).permute(0, 1, 3, 2)


def _one_site_ops_blas(L, W, R):
    chiL = L.shape[0]
    d = W.shape[2]
    Dm = W.shape[1]
    chiR = R.shape[0]
    LW = torch.tensordot(L, W, dims=([1], [0]))         # (x,l,b,s,t)
    LW = LW.permute(0, 3, 4, 2, 1)                      # (x,s,t,b,l)
    LW2 = LW.permute(0, 1, 3, 4, 2).reshape(chiL * d * Dm, chiL * d)
    R2 = R.permute(1, 2, 0).reshape(Dm * chiR, chiR)
    return LW2, R2, (chiL, d, chiR, Dm)


def _matvec_one_blas(ops, theta):
    LW2, R2, (chiL, d, chiR, Dm) = ops
    tmp1 = LW2 @ theta.reshape(chiL * d, chiR)
    tmp = tmp1.reshape(chiL * d, Dm * chiR)
    out = tmp @ R2
    return out.reshape(chiL, d, chiR)


def _taylor_order(alpha, hnorm, tol, max_order):
    x = abs(alpha) * hnorm * 1.25
    k = 1
    fact = 1.0
    while k < max_order:
        fact *= (k + 1)
        if x ** (k + 1) / fact < tol:
            return k
        k += 1
    return max_order


def taylor_expm(matvec, theta, alpha, tol=1e-6, max_order=12):
    norm0 = torch.linalg.vector_norm(theta).item()
    if norm0 == 0.0:
        return theta
    w = matvec(theta)
    hnorm = torch.linalg.vector_norm(w).item() / norm0
    order = _taylor_order(alpha, hnorm, tol, max_order)
    out = theta.clone()
    term = theta.clone()
    for k in range(1, order + 1):
        term = matvec(term) * (alpha / k)
        out = out + term
    return out


def _svd_guarded(mat, chi_max, truncerr, res_tol=1e-9, cond_max=1e8):
    """Fast gesvda SVD with a residual/condition guard and default fallback."""
    try:
        U, S, Vh = torch.linalg.svd(mat, full_matrices=False, driver='gesvda')
        rec = (U * S) @ Vh
        den = torch.linalg.matrix_norm(mat)
        res = float(torch.linalg.matrix_norm(mat - rec) / den) if float(den) > 0 else 1.0
        S_np = S.detach().cpu().numpy()
        keep0 = int(np.sum(S_np > max(truncerr * S_np[0], 1e-12)))
        keep0 = max(1, min(int(chi_max), keep0, len(S_np)))
        cond = float(S_np[0] / S_np[keep0 - 1]) if S_np[keep0 - 1] > 0 else float('inf')
    except Exception:
        res, cond = float('inf'), float('inf')
    if np.isfinite(res) and np.isfinite(cond) and res <= res_tol and cond <= cond_max:
        return U, S, Vh, 'gesvda', res, cond
    U, S, Vh = torch.linalg.svd(mat, full_matrices=False)
    return U, S, Vh, 'fallback', res, cond


def two_site_step(mps, i, W, LP, RP, alpha, chi_max=200, truncerr=1e-10,
                  taylor_tol=1e-6, taylor_order=12, svd_device='cuda',
                  svd_driver=None, svd_method='svd', guard_stats=None):
    theta = mps.get_theta(i, 2)
    ops = _two_site_ops_blas(LP[i], W[i], W[i + 1], RP[i + 2])
    theta = taylor_expm(lambda th: _matvec_two_blas(ops, th), theta, alpha,
                        taylor_tol, taylor_order)
    chiL, d, d2, chiR = theta.shape
    mat = theta.reshape(chiL * d, d2 * chiR)
    if svd_method == 'guarded':
        U, S, Vh, used, res, cond = _svd_guarded(mat, chi_max, truncerr)
        S_np = S.detach().cpu().numpy()
        keep = int(np.sum(S_np > max(truncerr * S_np[0], 1e-12)))
        keep = max(1, min(int(chi_max), keep, len(S_np)))
        U = U[:, :keep]
        S = S[:keep]
        Vh = Vh[:keep, :]
        if guard_stats is not None:
            guard_stats[used] = guard_stats.get(used, 0) + 1
            guard_stats['max_res'] = max(guard_stats.get('max_res', 0.0), res)
            guard_stats['max_cond'] = max(guard_stats.get('max_cond', 0.0), cond)
    elif svd_device == 'cpu':
        # numpy/LAPACK is faster than cuSOLVER for complex128 at chi<=128.
        U_np, S_np, Vh_np = np.linalg.svd(mat.detach().cpu().numpy(),
                                          full_matrices=False)
        keep = int(np.sum(S_np > max(truncerr * S_np[0], 1e-12)))
        keep = max(1, min(int(chi_max), keep, len(S_np)))
        U = torch.as_tensor(U_np[:, :keep], dtype=theta.dtype, device=theta.device)
        S = torch.as_tensor(S_np[:keep], dtype=torch.float32 if theta.dtype == torch.complex64 else torch.float64,
                            device=theta.device)
        Vh = torch.as_tensor(Vh_np[:keep, :], dtype=theta.dtype, device=theta.device)
    else:
        if svd_driver:
            U, S, Vh = torch.linalg.svd(mat, full_matrices=False,
                                        driver=svd_driver)
        else:
            U, S, Vh = torch.linalg.svd(mat, full_matrices=False)
        S_np = S.detach().cpu().numpy()
        keep = int(np.sum(S_np > max(truncerr * S_np[0], 1e-12)))
        keep = max(1, min(int(chi_max), keep, len(S_np)))
        U = U[:, :keep]
        S = S[:keep]
        Vh = Vh[:keep, :]
    sn = torch.linalg.vector_norm(S)
    if float(sn) > 0.0:
        S = S / sn
    mps.set_B(i, U.reshape(chiL, d, keep), (1.0, 0.0))
    mps.S[i + 1] = S
    mps.set_B(i + 1, Vh.reshape(keep, d2, chiR), (0.0, 1.0))
    return keep


def one_site_step(mps, j, W, LP, RP, alpha, taylor_tol=1e-6, taylor_order=12):
    theta = mps.get_theta(j, 1)
    ops = _one_site_ops_blas(LP[j], W[j], RP[j + 1])
    theta = taylor_expm(lambda th: _matvec_one_blas(ops, th), theta, alpha,
                        taylor_tol, taylor_order)
    mps.set_B(j, theta, (1.0, 1.0))


def sweep(mps, W, LP, RP, dt_ns, chi_max=200, truncerr=1e-10,
          taylor_tol=1e-6, taylor_order=12, svd_device='cuda',
          svd_driver=None, svd_method='svd', guard_stats=None):
    n = len(mps.B)
    dt = float(dt_ns) * (2.0 * math.pi * 1e-3)
    for i in range(0, n - 2):
        two_site_step(mps, i, W, LP, RP, -0.5j * dt, chi_max, truncerr,
                      taylor_tol, taylor_order, svd_device, svd_driver,
                      svd_method, guard_stats)
        LP[i + 1] = update_LP(LP[i], mps.B[i], W[i])
        one_site_step(mps, i + 1, W, LP, RP, +0.5j * dt,
                      taylor_tol, taylor_order)
    for i in range(n - 2, -1, -1):
        alpha = -1.0j * dt if i == n - 2 else -0.5j * dt
        two_site_step(mps, i, W, LP, RP, alpha, chi_max, truncerr,
                      taylor_tol, taylor_order, svd_device, svd_driver,
                      svd_method, guard_stats)
        if i > 0:
            RP[i + 1] = update_RP(RP[i + 2], mps.B[i + 1], W[i + 1])
            one_site_step(mps, i, W, LP, RP, +0.5j * dt,
                          taylor_tol, taylor_order)


def expectation_pops(mps, op=None, device='cpu'):
    n = len(mps.B)
    out = np.zeros(n, dtype=float)
    if op is None:
        op = torch.tensor([[0.0, 0.0], [0.0, 1.0]], dtype=mps.B[0].dtype,
                          device=mps.B[0].device)
    for i in range(n):
        th = mps.get_theta(i, 1)
        out[i] = float(torch.real(torch.einsum('lsr,st,ltr->', th.conj(), op, th)).item())
    return out


def evolve_torch(calib, psi0=None, t_grid=None, Om=None, delta=None,
                 gamma=None, terms=None, chi_max=200, truncerr=1e-10,
                 max_step_ns=None, taylor_tol=1e-6, taylor_order=12,
                 device='cuda', dtype='complex64', progress=None, order=None,
                 svd_device='cuda', svd_driver=None, svd_method='svd',
                 guard_stats=None):
    from src.core.manybody.dmrg import terms_from_calib
    from src.core.manybody.couplings import active_terms

    if device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA is not available')
    tdtype = torch.complex128 if dtype == 'complex128' else torch.complex64
    n = len(calib.get('atoms') or [])
    t = np.asarray(t_grid, float)
    if t.ndim != 1 or t.size < 2 or np.any(np.diff(t) < 0):
        raise ValueError('t_grid must be a nondecreasing 1-D array with >=2 points')
    Om = np.zeros(n, complex) if Om is None else np.asarray(Om, complex)
    delta = np.zeros(n) if delta is None else np.asarray(delta, float)
    gamma = np.zeros(n) if gamma is None else np.asarray(gamma, float)
    if np.any(np.abs(gamma) > 0.0):
        raise ValueError('torch TDVP prototype only supports gamma=0')
    selected = dict(calib, terms=list(active_terms(calib) if terms is None else terms))
    order, onsite, couplings, info = terms_from_calib(selected, Om, delta, gamma,
                                                      order=order)
    order = list(map(int, order))
    W_np = build_mpo(onsite, couplings, n)
    W = [torch.as_tensor(w, dtype=tdtype, device=device) for w in W_np]
    mps = TorchMPS.product(n, psi0, device=device, dtype=tdtype)
    if order != list(range(n)):
        one = torch.ones(1, dtype=torch.float32, device=device)
        mps = TorchMPS([mps.B[a] for a in order], [one.clone() for _ in range(n + 1)],
                       [(0.0, 1.0)] * n)
    LP, RP = init_env(mps, W)
    pops = np.zeros((len(t), n), dtype=float)
    ent = np.zeros(len(t), dtype=float)
    def _pops():
        pm = expectation_pops(mps)
        pa = np.empty(n, dtype=float)
        pa[order] = pm
        return pa

    pops[0] = _pops()
    ent[0] = _entropy(mps)
    max_chi = 0
    for k in range(1, len(t)):
        span = float(t[k] - t[k - 1])
        nsub = 1 if max_step_ns is None else max(1, int(np.ceil(span / max_step_ns)))
        for a, b in zip(np.linspace(t[k - 1], t[k], nsub + 1)[:-1],
                        np.linspace(t[k - 1], t[k], nsub + 1)[1:]):
            sweep(mps, W, LP, RP, b - a, chi_max, truncerr,
                  taylor_tol, taylor_order, svd_device, svd_driver,
                  svd_method, guard_stats)
            max_chi = max(max_chi, max(len(s) for s in mps.S))
            if progress is not None:
                progress(float((k - 1 + (b - t[k - 1]) / span) / (len(t) - 1)))
        pops[k] = _pops()
        ent[k] = _entropy(mps)
    if progress is not None:
        progress(1.0)
    return dict(t=t, pops_atom=pops, S=ent, n=n, max_chi=max_chi,
                device=device, dtype=dtype, guard_stats=guard_stats)


def _entropy(mps):
    s = mps.S[len(mps.B) // 2].detach().cpu().numpy().astype(float)
    p = s * s
    p = p[p > 0]
    return float(-np.sum(p * np.log(p)))
