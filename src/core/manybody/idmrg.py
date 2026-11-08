# -*- coding: utf-8 -*-
"""Translation-invariant infinite MPS / iDMRG for two-level quantum models.

The input is deliberately **not** a Rydberg calib and not an array-shape enum:
it is a unit cell plus generic local terms.  A Rydberg cell can be converted by
``from_rydberg_cell``; the engine itself only knows the two-level site and the
operator names ``I/X/Y/Z/N/P/M``.

Spec (JSON-friendly)::

    {"cell": [{"site": 0, "pos": [0, 0]}, ...],
     "terms": [
       {"kind": "onsite", "site": 0, "op": "N", "coeff": 0.1},
       {"kind": "pair", "i": 0, "j": 1, "dx": 0,
        "op_i": "P", "op_j": "M", "coeff": 0.4},
       {"kind": "pair", "i": 0, "j": 0, "dx": 1,
        "op_i": "N", "op_j": "N", "coeff": 0.1}
     ]}

``dx`` is the unit-cell offset of ``j`` relative to ``i``; the infinite
direction is periodic by construction.  A circumference is encoded by the
intra-cell terms (present = periodic cylinder, absent = open strip).
"""
import numpy as np

__all__ = ['validate_infinite_spec', 'infinite_model', 'idmrg_ground_state',
           'from_rydberg_cell']

_OP_NAMES = ('I', 'X', 'Y', 'Z', 'N', 'P', 'M', 'SX', 'SY', 'SZ')
_MAX_DX = 64


def _complex(value):
    try:
        c = complex(value)
    except Exception as exc:  # noqa: BLE001
        raise ValueError('iDMRG 系数 %r 不是复数（复杂耦合用 "a+bj" 字符串）'
                         % (value,)) from exc
    if not np.isfinite(c.real) or not np.isfinite(c.imag):
        raise ValueError('iDMRG 系数必须有限：%r' % (value,))
    return c


def validate_infinite_spec(spec):
    """结构校验并返回规范化副本；不改变物理约定。"""
    if not isinstance(spec, dict):
        raise ValueError('manybody.idmrg 须是映射')
    cell = spec.get('cell')
    terms = spec.get('terms')
    if not isinstance(cell, (list, tuple)) or not cell:
        raise ValueError('manybody.idmrg.cell 须是非空列表')
    out_cell = []
    for k, site in enumerate(cell):
        if not isinstance(site, dict):
            raise ValueError('manybody.idmrg.cell[%d] 须是映射' % k)
        out_cell.append(dict(site))
    n = len(out_cell)
    if n < 2:
        raise ValueError('iDMRG 的 TeNPy 两站点 DMRG 需要单胞至少 2 个 site；'
                         '一维链请给 2-site 单胞')
    if not isinstance(terms, (list, tuple)) or not terms:
        raise ValueError('manybody.idmrg.terms 须是非空列表')
    out_terms = []
    for k, term in enumerate(terms):
        if not isinstance(term, dict):
            raise ValueError('manybody.idmrg.terms[%d] 须是映射' % k)
        kind = str(term.get('kind') or '')
        coeff = _complex(term.get('coeff'))
        if kind == 'onsite':
            site = int(term.get('site', -1))
            if not 0 <= site < n:
                raise ValueError('terms[%d].site 越界（cell=%d）' % (k, n))
            op = str(term.get('op') or '')
            if op not in _OP_NAMES:
                raise ValueError('terms[%d].op 不支持：%r（可选 %s）'
                                 % (k, op, '/'.join(_OP_NAMES)))
            out_terms.append({'kind': 'onsite', 'site': site,
                              'op': op, 'coeff': coeff})
            continue
        if kind != 'pair':
            raise ValueError('terms[%d].kind 须为 onsite/pair，得到 %r'
                             % (k, kind))
        i, j = int(term.get('i', -1)), int(term.get('j', -1))
        dx = term.get('dx', 0)
        if isinstance(dx, bool) or int(dx) != dx or int(dx) < 0:
            raise ValueError('terms[%d].dx 须为非负整数' % k)
        dx = int(dx)
        if not (0 <= i < n and 0 <= j < n):
            raise ValueError('terms[%d].i/j 越界（cell=%d）' % (k, n))
        if dx > _MAX_DX:
            raise ValueError('terms[%d].dx=%d 超过上限 %d' % (k, dx, _MAX_DX))
        if dx == 0 and i == j:
            raise ValueError('terms[%d]: 同 site/dx=0 的 pair 请写 onsite' % k)
        if dx == 0 and i > j:
            i, j = j, i
        op_i = str(term.get('op_i') or '')
        op_j = str(term.get('op_j') or '')
        if op_i not in _OP_NAMES or op_j not in _OP_NAMES:
            raise ValueError('terms[%d].op_i/op_j 不支持：%r/%r'
                             % (k, op_i, op_j))
        out_terms.append({'kind': 'pair', 'i': i, 'j': j, 'dx': dx,
                          'op_i': op_i, 'op_j': op_j, 'coeff': coeff})
    return {'cell': out_cell, 'terms': out_terms}


def _site(conserve=None):
    from tenpy.networks.site import SpinHalfSite
    site = SpinHalfSite(conserve=conserve)
    site.add_op('Nhat', np.array([[0, 0], [0, 1]], dtype=complex))
    site.add_op('P', np.array([[0, 0], [1, 0]], dtype=complex))
    site.add_op('M', np.array([[0, 1], [0, 0]], dtype=complex))
    site.add_op('SZ', np.array([[1, 0], [0, -1]], dtype=complex))
    if conserve is None:
        site.add_op('SX', np.array([[0, 1], [1, 0]], dtype=complex))
        site.add_op('SY', np.array([[0, -1j], [1j, 0]], dtype=complex))
    return site


def _op_name(op):
    return {'N': 'Nhat', 'I': 'Id', 'X': 'SX', 'Y': 'SY', 'Z': 'SZ'}.get(op, op)


def infinite_model(spec, *, charge_sector=None, mps_dtype='complex'):
    """Build a TeNPy infinite ``Lattice`` MPO from a cell + generic terms."""
    _require_tenpy()
    spec = validate_infinite_spec(spec)
    n = len(spec['cell'])
    conserve = 'Sz' if charge_sector is not None else None
    if conserve is not None:
        # Conserved-sector admissibility is a property of the **whole term**,
        # not of each operator in isolation.  P_i M_j (and M_i P_j) move one
        # excitation from j to i and therefore conserve total N even though a
        # single P or M does not.
        delta_n = {'I': 0, 'N': 0, 'Z': 0, 'SZ': 0,
                   'P': 1, 'M': -1}
        bad = []
        for t in spec['terms']:
            if t['kind'] == 'onsite':
                if delta_n.get(t['op']) != 0:
                    bad.append(t)
            else:
                di = delta_n.get(t['op_i'])
                dj = delta_n.get(t['op_j'])
                if di is None or dj is None or di + dj != 0:
                    bad.append(t)
        if bad:
            raise ValueError(
                'charge_sector 下这些项的净 ΔN≠0：%s；'
                '允许 N/Z/I 及 P_i M_j（总激发数守恒），'
                'X/Y/SX/SY 或 P_i P_j/M_i M_j 请去掉扇区'
                % [t.get('op') or (t.get('op_i'), t.get('op_j'))
                   for t in bad[:6]])
    site = _site(conserve)
    from tenpy.models.model import CouplingMPOModel
    from tenpy.models.lattice import Lattice
    class _InfiniteCluster(CouplingMPOModel):
        def init_sites(self, params):
            return [site] * n

        def init_lattice(self, params):
            return Lattice([1], [site] * n, bc='periodic',
                           bc_MPS='infinite')

        def init_terms(self, params):
            for t in spec['terms']:
                if t['kind'] == 'onsite':
                    self.add_onsite(t['coeff'], t['site'], _op_name(t['op']))
                else:
                    self.add_coupling(t['coeff'], t['i'], _op_name(t['op_i']),
                                      t['j'], _op_name(t['op_j']), t['dx'])

    model = _InfiniteCluster(dict())
    return model


def _require_tenpy():
    try:
        import tenpy  # noqa: F401
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError('iDMRG 需要 TeNPy（pip install physics-tenpy）') from exc


def from_rydberg_cell(cell, pairs):
    """Convenience adapter: Rydberg-like cell + V/J/A/Omega/delta → generic terms.

    ``pairs`` entries use ``i,j,dx,V,J,A``; onsite uses atom ``Omega_mhz`` and
    ``delta_mhz``.  This is only a builder; the iDMRG engine stays generic.
    """
    terms = []
    for k, atom in enumerate(cell):
        if atom.get('Omega_mhz'):
            om = complex(atom.get('Omega_mhz', 0.0)) * np.exp(
                1j * float(atom.get('Omega_phase_rad', 0.0)))
            terms.append({'kind': 'onsite', 'site': k, 'op': 'P',
                          'coeff': om / 2.0})
            terms.append({'kind': 'onsite', 'site': k, 'op': 'M',
                          'coeff': np.conj(om) / 2.0})
        de = float(atom.get('delta_mhz', 0.0))
        if de:
            terms.append({'kind': 'onsite', 'site': k, 'op': 'N',
                          'coeff': de})
    for p in pairs:
        i, j, dx = int(p['i']), int(p['j']), int(p.get('dx', 0))
        if p.get('V'):
            terms.append({'kind': 'pair', 'i': i, 'j': j, 'dx': dx,
                          'op_i': 'N', 'op_j': 'N', 'coeff': complex(p['V'])})
        if p.get('J'):
            jc = complex(p['J'])
            terms.append({'kind': 'pair', 'i': i, 'j': j, 'dx': dx,
                          'op_i': 'P', 'op_j': 'M', 'coeff': jc})
            terms.append({'kind': 'pair', 'i': i, 'j': j, 'dx': dx,
                          'op_i': 'M', 'op_j': 'P', 'coeff': np.conj(jc)})
        if p.get('A'):
            ac = complex(p['A'])
            terms.append({'kind': 'pair', 'i': i, 'j': j, 'dx': dx,
                          'op_i': 'P', 'op_j': 'P', 'coeff': ac})
            terms.append({'kind': 'pair', 'i': i, 'j': j, 'dx': dx,
                          'op_i': 'M', 'op_j': 'M', 'coeff': np.conj(ac)})
    return {'cell': [dict(a) for a in cell], 'terms': terms}


def idmrg_ground_state(spec, *, charge_sector=None, chi_max=128,
                       truncerr=1e-10, max_sweeps=20, mixer=True,
                       mps_dtype='complex', progress_arr=None):
    """Run TeNPy infinite two-site DMRG; return bulk observables per unit cell."""
    _require_tenpy()
    spec = validate_infinite_spec(spec)
    n = len(spec['cell'])
    if charge_sector is not None:
        if isinstance(charge_sector, bool) or int(charge_sector) != charge_sector:
            raise ValueError('charge_sector 必须是整数')
        charge_sector = int(charge_sector)
        if not 0 <= charge_sector <= n:
            raise ValueError('charge_sector 越界（cell=%d）' % n)
    model = infinite_model(spec, charge_sector=charge_sector,
                           mps_dtype=mps_dtype)
    from tenpy.networks.mps import MPS
    from tenpy.algorithms.dmrg import TwoSiteDMRGEngine
    from .runtime import check_cancel, report_progress
    check_cancel(progress_arr)
    p_state = ([1] * charge_sector + [0] * (n - charge_sector)
               if charge_sector is not None else [0] * n)
    psi = MPS.from_product_state(model.lat.mps_sites(), p_state,
                                 bc='infinite', dtype=complex)

    class CancellableDMRG(TwoSiteDMRGEngine):
        def run_iteration(self):
            check_cancel(progress_arr)
            out = super().run_iteration()
            check_cancel(progress_arr)
            report_progress(progress_arr,
                            min(0.99, self.sweeps / max(2, max_sweeps + 2)))
            return out

    opts = dict(max_sweeps=int(max_sweeps), min_sweeps=2, mixer=bool(mixer),
                trunc_params=dict(chi_max=int(chi_max),
                                  trunc_cut=float(truncerr)))
    eng = CancellableDMRG(psi, model, opts)
    E, psi = eng.run()
    psi = psi.copy()
    psi.canonical_form(renormalize=False)
    n_atom = np.asarray(psi.expectation_value('Nhat'), dtype=float).reshape(-1)
    S = np.asarray(psi.entanglement_entropy(), dtype=float).reshape(-1)
    if len(S) == 0:
        S = np.zeros(n)
    try:
        xi = float(psi.correlation_length())
    except Exception:  # noqa: BLE001 - correlation length is optional
        xi = None
    try:
        terr = float(np.max(np.asarray(eng.trunc_err_list, dtype=float)))
    except Exception:  # noqa: BLE001
        terr = float('nan')
    E = float(E)
    if charge_sector is not None and abs(float(n_atom.sum()) - charge_sector) > 1e-8:
        raise ValueError('iDMRG 最终占据 %.6f 与 charge_sector=%d 不一致'
                         % (float(n_atom.sum()), charge_sector))
    return {
        'want': 'gs', 'backend': 'tenpy-idmrg', 'infinite': True,
        'n_atoms': n, 'n_sites': None, 'unit_cell_size': n,
        'E_per_site': E, 'E_per_cell': E * n,
        'n_atom': n_atom, 'S_per_bond': float(np.mean(S)),
        'entanglement_entropy': S, 'corr_length': xi,
        'chi': int(max(psi.chi)) if len(psi.chi) else 1,
        'trunc_err': terr, 'sweeps': int(eng.sweeps),
        'solver_converged': bool(eng.is_converged()),
        'charge_sector': charge_sector,
        'error_report': {'solver_converged': bool(eng.is_converged()),
                         'sweeps': int(eng.sweeps), 'trunc_err': terr},
    }
