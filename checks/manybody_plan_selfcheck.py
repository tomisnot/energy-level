# -*- coding: utf-8 -*-
"""HamiltonianPlan / blockade-PXP 装配层验证。"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from checks.checklib import Checker
from src.core.manybody.hamiltonian_plan import build_plan, compile_sparse
from src.core.manybody.engine_ed import evolve_manybody

C = Checker('manybody_plan')
c = {'atoms': [dict(id=i, pos=(float(i), 0., 0.)) for i in range(4)],
     'terms': ['I1'],
     'pairs': [dict(i=0, j=1, V=10., J=0., A=0.),
               dict(i=1, j=2, V=10., J=0., A=0.),
               dict(i=2, j=3, V=10., J=0., A=0.)]}
p = build_plan(c, Om=np.full(4, -0.5), delta=np.zeros(4), terms=['I1'],
               constraint={'kind': 'pxp', 'threshold_mhz': 5.})
C.group('PXP plan', [('projected', p['projected']), ('dim', p['dim'] == 8),
                      ('full_dim', p['full_dim'] == 16),
                      ('edges', len(p['blockade_edges']) == 3),
                      ('csr', compile_sparse(p).shape == (8, 8))])
r = evolve_manybody(c, psi0='gggg', t_grid=np.linspace(0., 2., 3),
                    Om=np.full(4, -0.5), delta=np.zeros(4), terms=['I1'],
                    plan=p, want_corr=False)
C.check('reduced ED backend', r['backend'] == 'ed-plan-pxp')
C.check('reduced ED population shape', r['pops'].shape == (3, 4))
full = build_plan(c, constraint={'kind': 'none'})
C.check('full plan compact', full['states'] is None and full['dim'] == 16)
sys.exit(0 if C.summary() else 1)
