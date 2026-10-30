# -*- coding: utf-8 -*-
"""GPU matrix-free/KPM smoke benchmark; clean SKIP on hosts without CUDA."""
import numpy as np
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from src.core.manybody import matrix_free, kpm, equilibrium
try:
    import torch
except Exception:
    torch = None

if torch is None or not torch.cuda.is_available():
    print('SKIP gpu backends: CUDA unavailable')
    raise SystemExit(0)

cal = {'atoms':[{'id':i} for i in range(4)], 'terms':['I1','I2'],
       'pairs':[{'i':i,'j':i+1,'V':0.2,'J':0.7} for i in range(3)]}
r = matrix_free.torch_lanczos_ground_state(cal, Om=np.ones(4), m=24)
assert np.isfinite(r['E0']) and r['dim']==16
op = matrix_free.hamiltonian_linear_operator(cal, np.ones(4))
H = np.column_stack([op @ np.eye(16)[:, i] for i in range(16)])
cpu = equilibrium.ground_state(op, dense_max=0, k=2)
assert abs(r['E0'] - cpu['E0']) < 1e-6
mu,c,s = kpm.kpm_moments_gpu(H, num=16, nvec=8, seed=1)
assert mu.shape==(17,) and np.isfinite(mu).all()
# A nonzero spectral centre catches a missing c*I subtraction in GPU KPM.
diagonal = np.diag(np.linspace(-2.,3.,32))
reference,cc,rr = kpm.kpm_moments(diagonal,num=24,nvec=1,exact=True)
for dtype,tolerance in [('float64',1e-10),('float32',2e-4)]:
    actual,c,r = kpm.kpm_moments_gpu(diagonal,num=24,nvec=1,seed=1,dtype=dtype)
    assert abs(c-cc)<1e-10 and abs(r-rr)<1e-10
    assert np.max(abs(actual-reference))<tolerance, (dtype,np.max(abs(actual-reference)))
print('[gpu_backends] PASS=5 FAIL=0 SKIP=0')
