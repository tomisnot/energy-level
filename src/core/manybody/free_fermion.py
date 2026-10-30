# -*- coding: utf-8 -*-
"""Applicability guard for an optional free-fermion backend.

This deliberately refuses generic Rydberg point clouds: Jordan-Wigner strings
make long-range XY and arbitrary complex couplings non-quadratic.
"""
import numpy as np

def check_applicability(calib, terms=None, Om=None):
    terms = set(calib.get('terms', ()) if terms is None else terms)
    if not terms.issubset({'I1', 'I2', 'I3'}):
        return False, '未知相互作用项'
    if Om is not None and np.any(np.asarray(Om, complex) != 0):
        return False, '局部翻转驱动 Omega 非零，不是数守恒二次费米子模型'
    atoms = calib.get('atoms') or []; n = len(atoms)
    pairs = calib.get('pairs') or []
    if 'I1' in terms and any(abs(complex(p.get('V', 0))) > 1e-12 for p in pairs):
        return False, '非零 I1/V 是密度-密度相互作用，不是自由费米子'
    if 'I3' in terms and any(abs(complex(p.get('A', 0))) > 1e-12 for p in pairs):
        return False, '非零 I3/A 会产生配对项，当前后端只支持数守恒 XY'
    pairs = [p for p in pairs if 'I2' in terms and abs(complex(p.get('J',0))) > 1e-12]
    edges = {(min(int(p['i']), int(p['j'])), max(int(p['i']), int(p['j']))) for p in pairs}
    if any(b-a != 1 for a,b in edges):
        return False, '长程 XY 需要 Jordan-Wigner string，不能自动视为 quadratic'
    if any(abs(np.imag(complex(p.get('J', 0)))) > 1e-12 for p in pairs):
        return False, '复 J 的规范与边界条件未声明，拒绝静默映射'
    return True, '1D nearest-neighbour real XY is quadratic'


def solve_ground_state(calib, delta=None, charge_sector=None, terms=None, Om=None):
    """Solve the applicable number-conserving quadratic Hamiltonian.

    The returned energy is the sum of occupied one-particle eigenvalues.  A
    requested ``charge_sector`` fixes particle number; otherwise all negative
    orbitals are occupied.  This backend intentionally does not construct the
    many-body Hilbert matrix.
    """
    ok, reason = check_applicability(calib, terms, Om)
    if not ok:
        raise ValueError('free-fermion 后端不适用：' + reason)
    atoms = calib.get('atoms') or []; n = len(atoms)
    d = np.zeros(n, float) if delta is None else np.asarray(delta, float)
    if d.shape != (n,) or not np.isfinite(d).all(): raise ValueError('delta 必须是有限、长度为 N 的数组')
    use = set(calib.get('terms', ()) if terms is None else terms)
    h = np.diag(d.astype(complex))
    for p in calib.get('pairs') or []:
        i,j=int(p['i']),int(p['j']); J=float(np.real(p.get('J', 0.0))) if 'I2' in use else 0.
        h[i,j] += J; h[j,i] += J
    eps, vec = np.linalg.eigh(h)
    if charge_sector is None:
        occ = np.flatnonzero(eps < 0.0)
    else:
        if isinstance(charge_sector, (bool,np.bool_)) or int(charge_sector) != charge_sector:
            raise ValueError('charge_sector 必须是整数')
        k=int(charge_sector)
        if k < 0 or k > n: raise ValueError('charge_sector 越界')
        occ = np.argsort(eps)[:k]
    e0 = float(np.sum(eps[occ]))
    unocc = np.delete(eps, occ)
    gap = (float(np.min(np.abs(eps))) if charge_sector is None and len(eps) else
           float(np.min(unocc)-np.max(eps[occ])) if len(occ) and len(unocc) else 0.)
    return dict(E0=e0, gap=max(0.0, gap), backend='free-fermion',
                one_particle_energies=eps, occupied=np.asarray(occ, int),
                n_atom=np.sum(np.abs(vec[:,occ])**2, axis=1),
                charge_sector=(int(len(occ)) if charge_sector is None else int(charge_sector)))

__all__ = ['check_applicability', 'solve_ground_state']
