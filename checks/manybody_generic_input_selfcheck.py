# -*- coding: utf-8 -*-
"""Small cross-backend check for generic complex states/drives/observables."""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np

from src.core.manybody.assemble import build_H_sparse
from src.core.manybody.engine_ed import evolve_manybody
from src.core.manybody.hamiltonian_plan import build_plan, compile_sparse
from src.core.manybody.qmc import check_stoquastic


def ok(name, cond):
    print("  [%s] %s" % ("PASS" if cond else "FAIL", name))
    return bool(cond)


cal = {
    'atoms': [{'id': 0, 'pos': (0, 0, 0)}, {'id': 1, 'pos': (1, 0, 0)}],
    'pairs': [{'i': 0, 'j': 1, 'V': .2, 'J': .4 + .1j, 'A': .1 - .2j}],
    'terms': ['I1', 'I2', 'I3'],
}
om = np.array([1.2 * np.exp(.3j), .4 * np.exp(-.7j)])
delta = np.array([.2, -.1])
spec = {'kind': 'product', 'amplitudes': [[1, 1j], [1, -1]]}
t = np.array([0., .1])

H = build_H_sparse(cal, om, delta, cal['terms']).toarray()
plan = build_plan(cal, om, delta, terms=cal['terms'])
Hp = compile_sparse(plan).toarray()
r = evolve_manybody(cal, spec, t, om, delta, want_observables=True,
                    loops=[[0, 1]])
ob = r['observables']
checks = [
    ok('complex Hamiltonian Hermitian', np.max(abs(H - H.conj().T)) < 1e-12),
    ok('HamiltonianPlan equals full assembler', np.max(abs(H - Hp)) < 1e-12),
    ok('generic product state normalized', abs(r['norms'][0] - 1.0) < 1e-12
       if 'norms' in r else abs(r['loss'][0]) < 1e-12),
    ok('X/Y/Z and phase outputs', all(k in ob for k in
                                      ('sx', 'sy', 'sz', 'sp', 'sm', 'phase'))),
    ok('XY correlation and loop output', ob['xy_corr'].shape == (2, 2)
       and len(ob['loop_phase']) == 1 and len(ob['loop_winding']) == 1),
]
try:
    check_stoquastic(cal, om, delta)
except ValueError as exc:
    checks.append(ok('QMC rejects complex transverse phase', '复相位' in str(exc)))
else:
    checks.append(ok('QMC rejects complex transverse phase', False))

print('[manybody_generic_input] PASS=%d FAIL=%d' %
      (sum(checks), len(checks) - sum(checks)))
raise SystemExit(0 if all(checks) else 1)
