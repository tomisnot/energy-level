# -*- coding: utf-8 -*-
"""Stoquastic stochastic-series-expansion (SSE) sampler for spin-1/2 arrays.

This is deliberately an honest first-stage QMC backend: arbitrary graphs and
long-range bonds are accepted, while complex/negative-sign matrix elements are
rejected unless a local +/- gauge removes the sign.  The sampler uses a fixed
operator string with diagonal insertion/removal and reversible off-diagonal
pair updates.  It is intended for thermal observables and low-temperature
ground-state estimates, not real-time evolution.
"""
import itertools
import numpy as np

__all__ = ['check_stoquastic', 'sse_run']


def _real(x, name, tol=1e-12):
    z = complex(x)
    if abs(z.imag) > tol:
        raise ValueError('SSE 需要实 Hamiltonian；%s=%r 含复相位' % (name, x))
    return float(z.real)


def _build_terms(calib, Om, delta, terms, gauge=True, allowed=None):
    atoms = calib.get('atoms') or []
    n = len(atoms)
    use = set(calib.get('terms') or ('I1', 'I2', 'I3')) if terms is None else set(terms)
    # Keep the complex dtype until _real checks every transverse matrix
    # element.  Casting here would hide a phase or emit only a NumPy warning.
    Om = np.zeros(n, complex) if Om is None else np.asarray(Om, complex)
    delta = np.zeros(n) if delta is None else np.asarray(delta, float)
    if Om.shape != (n,) or delta.shape != (n,):
        raise ValueError('SSE 的 Om/delta 长度必须等于原子数')

    # Each off-diagonal matrix element must become <= 0 after a local sign
    # gauge.  Constraints are g_i*g_j = parity and g_i = parity for flips.
    edges = []
    off_raw = []
    for i, x in enumerate(Om):
        c = _real(x / 2.0, 'Omega[%d]/2' % i)
        if abs(c) > 1e-14:
            off_raw.append(dict(kind='flip', i=i, j=None, coeff=c))
            edges.append((i, i, -1 if c > 0 else 1))
    for p in calib.get('pairs') or []:
        i, j = int(p['i']), int(p['j'])
        if 'I2' in use and abs(p.get('J', 0)) > 1e-14:
            c = _real(p['J'], 'J[%d,%d]' % (i, j))
            off_raw.append(dict(kind='exchange', i=i, j=j, coeff=c))
            edges.append((i, j, -1 if c > 0 else 1))
        if 'I3' in use and abs(p.get('A', 0)) > 1e-14:
            c = _real(p['A'], 'A[%d,%d]' % (i, j))
            off_raw.append(dict(kind='pair', i=i, j=j, coeff=c))
            edges.append((i, j, -1 if c > 0 else 1))

    g = np.ones(n, int)
    if gauge:
        seen = np.zeros(n, bool)
        adj = [[] for _ in range(n)]
        for i, j, parity in edges:
            if i == j:
                adj[i].append((i, parity))
            else:
                adj[i].append((j, parity)); adj[j].append((i, parity))
        for root in range(n):
            if seen[root]:
                continue
            seen[root] = True; g[root] = 1
            stack = [root]
            while stack:
                i = stack.pop()
                for j, parity in adj[i]:
                    want = g[i] * parity
                    if not seen[j]:
                        seen[j] = True; g[j] = want; stack.append(j)
                    elif g[j] != want:
                        raise ValueError('SSE sign problem：局部 +/- gauge 不一致，'
                                         '该 Hamiltonian 不是 stoquastic')

    diag = []
    shift = 0.0
    for i, d in enumerate(delta):
        d = _real(d, 'delta[%d]' % i)
        if abs(d) > 1e-14:
            # Keep every diagonal matrix element strictly positive.  A zero
            # weight would freeze local basis-state updates in a fixed-string
            # sampler even though the physical Boltzmann weight is finite.
            c = abs(d) + 1.0
            diag.append(dict(kind='onsite', i=i, j=None, coeff=d, shift=c))
            shift += c
    for p in calib.get('pairs') or []:
        i, j = int(p['i']), int(p['j'])
        if 'I1' in use and abs(p.get('V', 0)) > 1e-14:
            v = _real(p['V'], 'V[%d,%d]' % (i, j))
            c = abs(v) + 1.0
            diag.append(dict(kind='density', i=i, j=j, coeff=v, shift=c))
            shift += c
    # A positive identity operator keeps insertion/removal well-defined even
    # when the physical Hamiltonian has no diagonal terms.
    diag.append(dict(kind='identity', i=None, j=None, coeff=0.0, shift=1.0))
    shift += 1.0

    off = []
    for x in off_raw:
        fac = g[x['i']] if x['j'] is None else g[x['i']] * g[x['j']]
        h = x['coeff'] * fac
        if h > 1e-12:
            raise ValueError('SSE sign problem：%s[%s] 经 gauge 后仍为正矩阵元 %.6g'
                             % (x['kind'], x['i'], h))
        off.append(dict(kind=x['kind'], i=x['i'], j=x['j'], weight=-h,
                        allowed=allowed))
    return diag, off, float(shift), g


def check_stoquastic(calib, Om=None, delta=None, terms=None, gauge=True):
    """检查并返回 SSE 所需的 +/- gauge；失败时抛出带原因的 ValueError。"""
    diag, off, shift, g = _build_terms(calib, Om, delta, terms, gauge=gauge)
    return dict(ok=True, n_diag=len(diag), n_off=len(off), energy_shift=shift,
                gauge=g.tolist(), average_sign=1.0)


def _diag_weight(op, state):
    if op['kind'] == 'identity':
        return op['shift']
    if op['kind'] == 'onsite':
        h = op['coeff'] * ((state >> op['i']) & 1)
    else:
        h = op['coeff'] * (((state >> op['i']) & 1) * ((state >> op['j']) & 1))
    return op['shift'] - h


def _apply_off(op, state):
    i, j = op['i'], op['j']
    if op['kind'] == 'flip':
        ns = state ^ (1 << i)
        return None if op.get('allowed') is not None and ns not in op['allowed'] else ns
    bi, bj = (state >> i) & 1, (state >> j) & 1
    if op['kind'] == 'exchange':
        ns = None if bi == bj else state ^ (1 << i) ^ (1 << j)
    else:
        ns = None if bi != bj else state ^ (1 << i) ^ (1 << j)
    return None if ns is None or (op.get('allowed') is not None and ns not in op['allowed']) else ns


def _propagate(string, state, diag, off):
    before = []
    weight = 1.0
    s = int(state)
    for code in string:
        before.append(s)
        if code == -1:
            continue
        if code >= 0:
            w = _diag_weight(diag[code], s)
        else:
            op = off[-code - 2]
            w = op['weight']
            ns = _apply_off(op, s)
            if ns is None:
                return None, 0.0
            s = ns
        if w <= 0.0:
            return None, 0.0
        weight *= w
    if s != state:
        return None, 0.0
    return before, weight


def _sample_beta(calib, beta, Om, delta, terms, *, sweeps, burn_in, measure_every,
                 M, seed, gauge, want_corr, allowed=None):
    if beta <= 0:
        raise ValueError('SSE beta 必须 > 0')
    diag, off, shift, g = _build_terms(calib, Om, delta, terms, gauge=gauge, allowed=allowed)
    if not off:
        off = []
    n = len(calib.get('atoms') or [])
    M = int(M or max(32, 4 * n + 16))
    if M < 4:
        raise ValueError('SSE operator string 长度 M 至少为 4')
    rng = np.random.default_rng(seed)
    string = [-1] * M
    state = (int(rng.choice(np.asarray(sorted(allowed), dtype=int)))
             if allowed is not None else (int(rng.integers(0, 1 << n)) if n else 0))
    before, weight = _propagate(string, state, diag, off)
    if before is None:
        raise RuntimeError('SSE 初始配置无效')
    accepted = proposed = 0
    diag_count = len(diag)
    off_count = len(off)

    def nonidentity():
        return sum(x != -1 for x in string)

    def valid_weight(candidate=None, st=state):
        if candidate is None:
            candidate = string
        b, w = _propagate(candidate, st, diag, off)
        return b, w

    total_steps = int(burn_in) + int(sweeps) * int(measure_every)
    e_samples, order_samples, n_samples, c_samples = [], [], [], []
    for step in range(total_steps):
        # Standard diagonal insertion/removal sweep.
        # A full random permutation improves mixing; the insertion/removal
        # ratio below is for a fixed position, so the common 1/M proposal
        # factor cancels in both directions.
        for p in rng.permutation(M):
            ncur = nonidentity()
            if string[p] == -1:
                b = int(rng.integers(diag_count))
                w = _diag_weight(diag[b], before[p])
                if w > 0 and ncur < M:
                    ratio = beta * w * diag_count / max(1, M - ncur)
                    proposed += 1
                    if rng.random() < min(1.0, ratio):
                        cand = list(string); cand[p] = b
                        b2, w2 = valid_weight(cand)
                        if b2 is not None:
                            string, before, weight = cand, b2, w2; accepted += 1
            elif string[p] >= 0:
                b = string[p]; w = _diag_weight(diag[b], before[p])
                if w > 0 and ncur > 0:
                    ratio = (M - ncur + 1) / max(beta * w * diag_count, 1e-300)
                    proposed += 1
                    if rng.random() < min(1.0, ratio):
                        cand = list(string); cand[p] = -1
                        b2, w2 = valid_weight(cand)
                        if b2 is not None:
                            string, before, weight = cand, b2, w2; accepted += 1

        # Reversible pair insertion/removal supplies the off-diagonal sector.
        if off_count:
            for _ in range(max(1, M // 4)):
                ids = np.flatnonzero(np.asarray(string) == -1)
                if len(ids) >= 2 and rng.random() < 0.5:
                    p, q = sorted(rng.choice(ids, 2, replace=False))
                    oi = int(rng.integers(off_count)); code = -oi - 2
                    cand = list(string); cand[p] = cand[q] = code
                    b2, w2 = valid_weight(cand)
                    if b2 is not None:
                        ncur = nonidentity()
                        ratio = (beta * beta * off[oi]['weight'] ** 2 * off_count
                                 / max(1, (M - ncur) * (M - ncur - 1)))
                        proposed += 1
                        if rng.random() < min(1.0, ratio):
                            string, before, weight = cand, b2, w2; accepted += 1
                else:
                    pos = [p for p, x in enumerate(string) if x < -1]
                    if len(pos) < 2:
                        continue
                    p, q = sorted(rng.choice(pos, 2, replace=False))
                    if string[p] != string[q]:
                        continue
                    oi = -string[p] - 2
                    cand = list(string); cand[p] = cand[q] = -1
                    b2, w2 = valid_weight(cand)
                    if b2 is not None:
                        ncur = nonidentity()
                        denom = beta * beta * off[oi]['weight'] ** 2 * off_count
                        ratio = max(1, (M - ncur + 1) * (M - ncur + 2)) / max(denom, 1e-300)
                        proposed += 1
                        if rng.random() < min(1.0, ratio):
                            string, before, weight = cand, b2, w2; accepted += 1

        # Trace over the initial basis state: a local bit flip is a symmetric
        # proposal and preserves the periodic operator-string constraint.
        if n:
            for i in rng.permutation(n):
                i = int(i); ns = state ^ (1 << i)
                if allowed is not None and ns not in allowed:
                    continue
                b2, w2 = valid_weight(string, ns)
                if b2 is not None and weight > 0 and rng.random() < min(1.0, w2 / weight):
                    state, before, weight = ns, b2, w2

        if step >= burn_in and (step - burn_in) % measure_every == 0:
            p = int(rng.integers(M)); s = before[p]
            ncur = nonidentity()
            e_samples.append(shift - ncur / beta)
            order_samples.append(ncur)
            n_samples.append([(s >> i) & 1 for i in range(n)])
            if want_corr:
                v = np.asarray(n_samples[-1], float)
                c_samples.append(np.outer(v, v))

    e = np.asarray(e_samples, float)
    na = np.asarray(n_samples, float)
    if len(e) > 3 and float(np.var(e)) > 0:
        x = e - e.mean()
        ac = np.correlate(x, x, mode='full')[len(e)-1:] / np.arange(len(e), 0, -1)
        ac = ac / ac[0]
        tau = 0.5
        for a in ac[1:min(len(ac), 200)]:
            if a <= 0:
                break
            tau += float(a)
    else:
        tau = 0.5
    out = dict(E=float(e.mean()), E_err=float(e.std(ddof=1) / np.sqrt(max(1, len(e)))) if len(e) > 1 else np.nan,
               n_atom=na.mean(axis=0) if len(na) else np.zeros(n),
               n_atom_err=(na.std(axis=0, ddof=1) / np.sqrt(len(na)) if len(na) > 1 else np.full(n, np.nan)),
               C=float(np.var(order_samples, ddof=1) / beta**2) if len(order_samples) > 1 else np.nan,
               corr=(np.mean(c_samples, axis=0) - np.outer(na.mean(axis=0), na.mean(axis=0))
                     if want_corr and c_samples else None),
               samples=int(len(e)), average_sign=1.0,
               autocorrelation_time=float(tau),
               effective_samples=float(len(e) / max(1.0, 2.0 * tau)),
               acceptance=(accepted / proposed if proposed else 0.0),
               gauge=g.tolist(), energy_shift=shift)
    out['F'] = np.nan  # SSE energy estimators do not determine ln Z directly.
    return out


def sse_run(calib, betas=(1.0,), Om=None, delta=None, terms=None,
            sweeps=2000, burn_in=500, measure_every=5, M=None, seed=0,
            gauge=True, want_corr=True, plan=None):
    """Run SSE for one or more inverse temperatures.

    Returns a unified equilibrium-style dictionary.  ``betas`` use 1/MHz,
    matching the rest of the many-body API.  A complex or non-stoquastic input
    raises ``ValueError`` with an explicit sign-problem diagnosis.
    """
    if plan is not None and plan.get('projected'):
        calib = plan['calib']; Om = np.asarray(plan['Om'], complex)
        delta = np.asarray(plan['delta'], float); terms = plan.get('terms')
        allowed = set(map(int, plan['states']))
    else:
        allowed = None
    betas = np.asarray(list(betas), float)
    if betas.ndim != 1 or len(betas) == 0 or np.any(betas <= 0):
        raise ValueError('SSE betas 必须是正数列表')
    check = check_stoquastic(calib, Om, delta, terms, gauge=gauge)
    rows = []
    for i, beta in enumerate(betas):
        rows.append(_sample_beta(calib, float(beta), Om, delta, terms,
                                 sweeps=sweeps, burn_in=burn_in,
                                 measure_every=measure_every, M=M,
                                 seed=int(seed) + i, gauge=gauge,
                                 want_corr=want_corr, allowed=allowed))
    return dict(want='thermal', backend='sse-qmc', betas=betas,
                E=np.asarray([r['E'] for r in rows]),
                E_err=np.asarray([r['E_err'] for r in rows]),
                C=np.asarray([r['C'] for r in rows]),
                F=np.full(len(rows), np.nan),
                n_atom=np.asarray([r['n_atom'] for r in rows]),
                corr=rows[-1]['corr'], average_sign=np.ones(len(rows)),
                samples=np.asarray([r['samples'] for r in rows]),
                autocorrelation_time=np.asarray([r['autocorrelation_time'] for r in rows]),
                effective_samples=np.asarray([r['effective_samples'] for r in rows]),
                acceptance=np.asarray([r['acceptance'] for r in rows]),
                gauge=check['gauge'], sign_problem_status='absent',
                constraint=(plan.get('constraint') if plan is not None else None),
                diagnostics=rows)
