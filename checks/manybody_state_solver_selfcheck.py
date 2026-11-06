"""Generic fixed-number solvers, certified GPU gaps and finite-state handoff.

Fixtures are small irregular spin graphs, independent of any lattice/project.
"""
import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import tempfile
import warnings
from unittest.mock import patch
import numpy as np
from scipy.linalg import expm
from src.core.manybody import assemble, equilibrium, eq_route, eq_engine, dmrg
from src.core.manybody.matrix_free import (number_sector_basis, torch_lanczos_ground_state,
                                          hamiltonian_linear_operator)
from src.core.manybody.engine_mps import evolve_mps
from src.core.manybody.state import save_mps_checkpoint, load_mps_checkpoint
from src.core.manybody.hamiltonian_plan import build_plan
from src.core.manybody import runspec_mb
from src.core import runspec
from src.core.evolve_util import CONV


def check(name, condition):
    if not condition:
        raise AssertionError(name)
    print('[PASS]', name, flush=True)


def rejects(name, fn):
    try:
        fn()
    except (ValueError, RuntimeError):
        check(name, True)
    else:
        check(name, False)


n = 4
cal = dict(atoms=[dict(id=i, pos=(float(i), 0., 0.)) for i in range(n)],
           terms=['I2'], pairs=[dict(i=i,j=j,J=.2/(j-i)+.03j*(j-i),V=0.,A=0.)
                               for i in range(n) for j in range(i+1,n)])
delta = np.array([-2.,-1.5,1.,1.5])
H = assemble.build_H_sparse(cal, delta=delta)
general = dict(cal, terms=['I1','I2','I3'],pairs=[dict(p,V=.17,A=.04+.07j) for p in cal['pairs']])
drive=np.array([.1j,.2,.3j,-.4])
reference=assemble.build_H_sparse(general,Om=drive,delta=delta)
for basis in (np.arange(16),np.array([14,0,7,3,1])):
    operator=hamiltonian_linear_operator(general,Om=drive,delta=delta,basis_indices=basis)
    vector=np.arange(len(basis))+1j*np.arange(len(basis))[::-1]
    check('vectorized generic matvec basis size %d'%len(basis),
          np.max(np.abs(operator@vector-reference[basis][:,basis]@vector))<1e-12)
idx = number_sector_basis(n, 1)
hs = H[idx][:,idx].toarray()
ev = np.linalg.eigvalsh(hs)
spec = dict(calib=cal, n_atoms=n, want='gs', method='ed', charge_sector=1,
            delta=delta, hamiltonian_plan=build_plan(cal,delta=delta), return_state=True)
got = eq_engine.run_equilibrium(spec)
check('product ED respects fixed occupation', abs(got['E0']-ev[0])<1e-11
      and abs(got['n_atom'].sum()-1)<1e-11 and len(got['vec'])==4)
for number in (0,1,2,4):
    basis = number_sector_basis(n, number)
    target = np.linalg.eigvalsh(H[basis][:,basis].toarray())[0]
    for backend in ('auto','lanczos','ed'):
        with patch.object(eq_route, '_system_H', side_effect=AssertionError('full matrix forbidden')):
            out = eq_route.solve_equilibrium(dict(calib=cal, delta=delta,n_sites=n,
                        charge_sector=number), backend=backend)
        check('direct sector %d backend %s' % (number,backend), abs(out['E0']-target)<1e-10)
with patch.object(eq_route, '_system_H', side_effect=AssertionError('full matrix forbidden')):
    eq_route.solve_equilibrium(dict(calib=cal,n_sites=n,delta=delta),backend='lanczos')
check('full-space Lanczos retains LinearOperator', True)
custom = np.diag(np.arange(16.)-8.)
with patch.object(eq_route, 'hamiltonian_linear_operator', side_effect=AssertionError('explicit H ignored')):
    out = eq_route.solve_equilibrium(dict(calib=cal,H=custom,n_sites=n),backend='lanczos')
check('explicit Hamiltonian takes precedence', abs(out['E0']+8.)<1e-10)
rejects('nonconserving source cannot be projected',lambda: eq_route.solve_equilibrium(
    dict(calib=cal,n_sites=n,Om=np.ones(n),charge_sector=1),backend='ed'))
rejects('fractional charge rejected',lambda: number_sector_basis(n,1.5))
rejects('unimplemented thermal charge rejected',lambda: eq_route.solve_equilibrium(
    dict(calib=cal,n_sites=n,charge_sector=1),want='thermal',backend='kpm'))
thermal = equilibrium.thermal_from_h(H,beta=1.,n_atoms=n,basis_indices=number_sector_basis(n,2))
check('canonical thermal reduced basis', abs(thermal['n_atom'].sum()-2.)<1e-12)

try:
    import torch
except ImportError:
    torch = None
if torch is not None and torch.cuda.is_available():
    full_ev = np.linalg.eigvalsh(H.toarray())
    for seed in (0,1):
        gpu = torch_lanczos_ground_state(cal,delta=delta,m=16,seed=seed,return_state=True)
        check('GPU Ritz gap and residual seed %d' % seed, gpu['converged']
            and abs(gpu['gap']-(full_ev[1]-full_ev[0]))<1e-9
            and max(gpu['residuals'])<1e-9)
    zero = dict(atoms=cal['atoms'],pairs=[],terms=['I2'])
    gpu = torch_lanczos_ground_state(zero,delta=[-1.,0.,0.,1.],m=16)
    check('GPU degenerate ground-space gap',gpu['converged'] and abs(gpu['gap'])<1e-9)
    unconverged = torch_lanczos_ground_state(cal,delta=delta,m=2,max_restarts=0)
    check('uncertified GPU gap is NaN',not unconverged['converged'] and np.isnan(unconverged['gap']))
else:
    print('SKIP GPU certification: CUDA unavailable',flush=True)

with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter('always')
    gs = dmrg.dmrg_cluster(cal,delta=delta,charge_sector=1,
        chi_max=16,max_sweeps=6,truncerr=1e-12,return_state=True)
check('DMRG ED energy agreement',abs(gs['E0']-ev[0])<1e-10)
check('DMRG truncation option consumed',not any('unused option' in str(w.message)
      and 'truncerr' in str(w.message) for w in caught))
state=gs['state']
check('DMRG middle cut is explicit',gs['entropy_cut']==2 and
      abs(gs['S_mid']-state['psi'].entanglement_entropy(bonds=[2])[0])<1e-12)
with tempfile.TemporaryDirectory() as folder:
    save_mps_checkpoint(state,Path(folder)/'state.h5')
    restored=load_mps_checkpoint(Path(folder)/'state.h5')
    check('HDF5 state and charge round trip',restored['charge_sector']==1
          and restored['order']==state['order']
          and abs(abs(restored['psi'].overlap(state['psi']))-1.)<1e-11)

# A local unitary makes the eigenstate a nonstationary, correlated initial state.
state['psi'].apply_local_op(state['order'].index(0),'SZ',unitary=True)
before=state['psi'].copy()
kw=dict(delta=delta,chi_max=16,truncerr=1e-12,max_step_ns=.05/CONV,
        return_state=True,want_corr=True,want_observables=True)
whole=evolve_mps(cal,psi0=state,t_grid=np.array([0.,.2,.4])/CONV,**kw)
first=evolve_mps(cal,psi0=state,t_grid=np.array([0.,.2])/CONV,**kw)
second=evolve_mps(cal,psi0=first['state'],t_grid=np.array([0.,.2])/CONV,**kw)
check('state input remains unchanged',abs(abs(before.overlap(state['psi']))-1.)<1e-12)
check('DMRG to TDVP preserves charge',whole['error_report']['charge_sector']==1
      and np.max(np.abs(whole['pops_atom'].sum(axis=1)-1))<1e-10)
check('split evolution matches continuous evolution',
      np.max(np.abs(whole['pops_atom'][-1]-second['pops_atom'][-1]))<1e-9)
check('charged local observables are supported',np.max(np.abs(whole['observables']['sx']))<1e-10)
params={'manybody.calib':{'atoms':[dict(id=a['id'],pos=a['pos']) for a in cal['atoms']],
                          'pairs':list(cal.get('pairs') or [])},
        'manybody.psi0':state,'manybody.solver':'auto','manybody.return_state':True,
        'dynamic.t_max':.05/CONV,'dynamic.npts':2,'manybody.max_step_ns':.05/CONV}
prepared=runspec_mb.build_run_manybody(None,params)
routed=runspec.evolve_run(prepared)
check('RunSpec routes native state and returns successor',prepared['method']=='tdvp'
      and routed['state']['charge_sector']==1)
from types import SimpleNamespace
from mecha_v2.bridge import ElOps
import pickle
with tempfile.TemporaryDirectory() as folder:
    artifact=ElOps._land(SimpleNamespace(root=Path(folder)),dict(labels=['a']),whole)
    with (Path(folder)/(artifact+'.pkl')).open('rb') as handle:
        _,payload=pickle.load(handle)
    check('artifact preserves research arrays and state',
          {'observables','error_report','state','pops_atom'}<=payload.keys())
rejects('mismatched charge cannot silently restart',lambda: evolve_mps(cal,psi0=state,
    t_grid=[0.,1.],charge_sector=2))
rejects('number-changing drive rejected for charged state',lambda: evolve_mps(cal,psi0=state,
    t_grid=[0.,1.],Om=np.ones(n)))

# Jump boundaries must be integrated exactly even between output samples.
two=dict(atoms=cal['atoms'][:2],terms=['I2'],pairs=[dict(i=0,j=1,J=.7,V=0.,A=0.)])
segments=[dict(t0=0.,delta=[0.,0.]),dict(t0=.13/CONV,delta=[.8,-.8])]
r=evolve_mps(two,psi0='gr',t_grid=[0.,.4/CONV],segments=segments,
    charge_sector=1,chi_max=4,max_step_ns=1./CONV,want_corr=False)
h0=assemble.build_H_sparse(two).toarray()
h1=assemble.build_H_sparse(two,delta=[.8,-.8]).toarray()
v=expm(-1j*.27*h1)@expm(-1j*.13*h0)@np.eye(4)[:,1]
check('piecewise time boundaries are respected',abs(r['pops_atom'][-1,0]-abs(v[2])**2)<1e-9)

with patch('tenpy.networks.mps.MPS.correlation_function',side_effect=RuntimeError('contraction failed')):
    rejects('failed correlation is not physical zero',lambda: evolve_mps(
        cal,psi0=state,t_grid=[0.,1.],want_corr=True))
print('PASS generic state/solver regressions',flush=True)
