# -*- coding: utf-8 -*-
"""GPU MPS TEBD correctness/dispatch self-check.

The test is intentionally a generic nearest-neighbour two-level model, not an
XY-specific fixture.  It compares the CUDA backend with exact ED and CPU TEBD
on local observables, links, norm, and the public Runspec dispatch table.
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from checks.checklib import Checker
from src.core import runspec
from src.core import runspec as runspec_module
from src.core.manybody import runspec_mb as mb_runspec
from src.core.manybody.engine_ed import evolve_manybody
from src.core.manybody.engine_mps import evolve_mps

C = Checker('manybody_mps_gpu')
try:
    import torch
except Exception:
    torch = None
if torch is None or not torch.cuda.is_available():
    print('SKIP manybody_mps_gpu: CUDA unavailable')
    raise SystemExit(0)
from src.core.manybody.engine_mps_gpu import evolve_mps_gpu

n = 6
cal = {
    'atoms': [{'id': i, 'pos': (float(i), 0., 0.)} for i in range(n)],
    'pairs': [{'i': i, 'j': i + 1, 'V': .17 + .03 * i,
               'J': -.61 + .04j * (i + 1), 'A': .05j}
              for i in range(n - 1)],
    'terms': ['I1', 'I2', 'I3'],
}
psi = {'kind': 'product', 'amplitudes': [
    [1.0 / np.sqrt(2), np.exp(1j * .17 * i) / np.sqrt(2)] for i in range(n)]}
t = np.array([0., .05, .1, .2])
kw = dict(psi0=psi, t_grid=t, Om=np.zeros(n, complex),
          delta=np.linspace(-.2, .3, n), want_observables=True,
          link_pairs=[(0, 1), (1, 2), (0, 3), (2, 0)], loops=[[0, 1, 2]],
          want_xy_corr=False,
          max_step_ns=.05)
ed_kw = dict(kw)
ed_kw.pop('max_step_ns', None)
ed = evolve_manybody(cal, **ed_kw)
cpu = evolve_mps(cal, algorithm='tebd', chi_max=64, truncerr=1e-12, **kw)
gpu = evolve_mps_gpu(cal, chi_max=64, truncerr=1e-12, dtype='float64', **kw)
C.check('dispatch table exposes tebd-gpu', runspec.engine_fn('tebd-gpu', 'many') is evolve_mps_gpu)

# Exercise the real configuration-to-execution path.  The RunSpec only takes an
# inline calib, so the fixture hands the calibrated object in directly.
params = {
    'manybody.calib': {'atoms': [dict(id=a['id'], pos=a['pos'],
                                     Omega_mhz=0., delta_mhz=0., gamma_mhz=0.)
                                for a in cal['atoms']],
                       'pairs': list(cal.get('pairs') or [])},
    'manybody.solver': 'tebd-gpu',
    'manybody.ed_max_dim': 4096, 'manybody.dtype': 'float32',
    'manybody.device': 'cuda', 'manybody.psi0': psi,
    'manybody.link_pairs': [[0, 1]], 'dynamic.t_max': .02,
    'dynamic.npts': 2,
}
run = mb_runspec.build_run_manybody(None, params)
routed = runspec_module.evolve_run(run)
C.check('Runspec routes tebd-gpu end-to-end', routed['backend'] == 'tebd-torch-cuda')
C.check('pops vs CPU TEBD', np.max(np.abs(gpu['pops'] - cpu['pops'])) < 2e-7)
C.check('sx vs ED', np.max(np.abs(gpu['observables']['sx'] - ed['observables']['sx'])) < 2e-7)
C.check('link correlations vs ED', np.max(np.abs(gpu['observables']['link_corr'] - ed['observables']['link_corr'])) < 2e-7)
C.check('loop diagnostics shape', gpu['observables']['loop_phase_series'].shape == (len(t), 1))
C.check('norm conservation', np.max(np.abs(gpu['norms'] - 1.0)) < 2e-8)
C.check('internal steps are separated', gpu['error_report']['n_internal_steps'] == 4)
try:
    bad = dict(cal)
    bad['pairs'] = list(cal['pairs']) + [{'i': 0, 'j': 2, 'V': .1, 'J': 0., 'A': 0.}]
    evolve_mps_gpu(bad, **kw)
except ValueError as exc:
    C.check('long-range coupling is explicit capability error', '最近邻' in str(exc))
else:
    C.check('long-range coupling is explicit capability error', False)
sys.exit(0 if C.summary() else 1)
