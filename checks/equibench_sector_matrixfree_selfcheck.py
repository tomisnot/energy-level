# -*- coding: utf-8 -*-
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from checks.checklib import Checker
from src.core.manybody.assemble import build_H_sparse
from src.core.manybody.matrix_free import hamiltonian_linear_operator
from src.core.manybody.equilibrium import ground_state
from src.core.manybody.cost_mb import effective_dim_from_sector, estimate_mb_peak_gb

C = Checker('sector_matrixfree')
c={'atoms':[{'id':i,'pos':(float(i),0,0)} for i in range(4)],
   'terms':['I1','I2'],
   'pairs':[{'i':i,'j':i+1,'V':.2,'J':-.3,'A':0} for i in range(3)]}
H=build_H_sparse(c,Om=np.zeros(4),delta=np.zeros(4))
L=hamiltonian_linear_operator(c,Om=np.zeros(4),delta=np.zeros(4))
x=np.random.default_rng(1).normal(size=16)+1j*np.random.default_rng(2).normal(size=16)
C.close('matrix-free matvec', np.max(np.abs(H@x-L@x)), 0., atol=1e-12, rtol=0.)
labels=np.array([int(i).bit_count() for i in range(16)])
r=ground_state(H, dense_max=0, k=2, sector=2, sector_labels=labels)
idx=np.where(labels==2)[0]; ref=np.linalg.eigvalsh(H.toarray()[np.ix_(idx,idx)])[0]
C.close('sector E0', r['E0'], ref, atol=1e-10, rtol=0.)
C.check('sector index', r['dim_sub']==6 and len(r['sector_idx'])==6)
C.check('effective dim', effective_dim_from_sector(4,2)==6)
C.check('cost reduction', estimate_mb_peak_gb(20,effective_dim=184756) < estimate_mb_peak_gb(20))
sys.exit(0 if C.summary() else 1)
