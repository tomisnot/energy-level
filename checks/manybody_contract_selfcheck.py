"""Small, asymmetric public-entry regressions for the many-body contract."""
import os
import sys
from unittest.mock import patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from scipy.linalg import expm
from scipy.sparse import csr_matrix
from checks.checklib import Checker
from src.core import runspec, protocol
from src.core.evolve_util import CONV
from src.core.manybody import runspec_mb as mb, eq_route, kpm
from src.core.manybody.backends import (TIME_BACKENDS, ConfigConflict,
                                       require_equilibrium_features,
                                       time_backend_names,
                                       validate_time_capabilities)
from src.core.manybody.basis import independent_set_count, independent_states, basis_bits
from src.core.manybody.assemble import build_H_sparse
from src.core.manybody.hamiltonian_plan import build_plan, compile_sparse
from src.core.manybody.engine_ed import evolve_manybody
from src.core.manybody.classical_spin import evolve_classical_spin
from src.core.manybody.engine_peps import evolve_peps
from src.core.manybody.engine_trajectories import evolve_trajectories
from src.core.manybody.free_fermion import solve_ground_state

C = Checker('manybody_contract')


def rejects(name, fn):
    try:
        fn()
    except (ValueError, TypeError, RuntimeError):
        C.check(name, True)
    else:
        C.check(name, False)


def calibration(n):
    return dict(atoms=[dict(id=i, pos=(float(n-i), 0., 0.)) for i in range(n)],
                pairs=[], terms=['I1', 'I2', 'I3'])


def params(cal):
    return {'dynamic.mode': 2, 'dynamic.t_max': .2, 'dynamic.npts': 2,
            'manybody.max_step_ns': .1,
            'manybody.calib': {'atoms': [dict(id=i, pos=list(a['pos']),
                                             Omega_mhz=1. + i, delta_mhz=.1 * i)
                                         for i, a in enumerate(cal['atoms'])],
                               'pairs': list(cal.get('pairs') or [])}}


# B5：能力冲突必须是 typed ConfigConflict，带冲突键，供 MCP 映射。
try:
    validate_time_capabilities('tdvp-dense', charge_sector=1)
    conflict_kind = None
except ConfigConflict as exc:
    conflict_kind = (type(exc).__name__, exc.keys)
except Exception as exc:  # noqa: BLE001
    conflict_kind = ('WRONG', type(exc).__name__, str(exc))
try:
    require_equilibrium_features('ed', {'mps_order'})
    eq_conflict = None
except ConfigConflict as exc:
    eq_conflict = exc.keys
except Exception as exc:  # noqa: BLE001
    eq_conflict = ('WRONG', type(exc).__name__)
# P2：距离截断是真模型选项，丢弃必须留痕。
cut_cal = {'atoms': [{'id': i, 'pos': [p, 0.0, 0.0]} for i, p in enumerate([0., 1., 2., 10.])],
           'pairs': [{'i': 0, 'j': 1, 'J': 0.4},
                     {'i': 1, 'j': 2, 'J': 0.4},
                     {'i': 2, 'j': 3, 'J': 0.4}]}
cut_warn = []
cut_calib, _m, _t = mb.resolve_model(
    {'manybody.calib': cut_cal, 'manybody.distance_cutoff_um': 2.0},
    warnings=cut_warn)
C.group('距离截断（P2）', [
    ('只保留近程对', len(cut_calib['pairs']) == 2),
    ('丢弃项留痕', any(d.get('reason') == 'r_cut_product'
                       for d in (cut_calib.get('dropped') or []))),
    ('warning 可见', any('distance_cutoff_um' in str(x) for x in cut_warn)),
])

C.group('配置冲突是 typed（B5）', [
    ('时间能力冲突类型/键',
     conflict_kind == ('ConfigConflict', ('manybody.charge_sector',))),
    ('平衡态冲突键', eq_conflict == ('manybody.mps_order',)),
])

cal = calibration(2)
schema = protocol.config_schema()
C.check('retired interaction controls are not writable',
        not {'manybody.terms','manybody.template'} & set(protocol.settable_keys()))
C.check('legacy model keys are gone from the schema',
        not {'manybody.atoms', 'manybody.calib_file', 'manybody.n_window',
             'manybody.r_cut_um', 'manybody.second_order', 'manybody.guard_mhz',
             'manybody.periods_um', 'manybody.B_gauss'} & set(protocol.settable_keys()))
C.check('schema and backend entries agree',
        schema['manybody.solver']['enum'] == time_backend_names()
        and all(callable(runspec.engine_fn(name,'many')) for name in TIME_BACKENDS))
C.check('every mapped backend option has a schema default', all(
    'manybody.'+suffix in schema for backend in TIME_BACKENDS.values()
    for suffix, _kw in backend.options))

# calib 只认行内形态：file/arc/terms 都必须当场拒
for bad, why in (({'file': 'x.json'}, 'file'), ({'arc': {'B_gauss': 1.}}, 'arc'),
                 (dict(cal, terms=['I2']), 'terms')):
    rejects('calib rejects %s form' % why, lambda b=bad: mb.validate_calib(b))

# 通道由数值推导（没有 terms 输入）
_derived = mb.validate_calib({'atoms': [{'id': 0, 'pos': [0, 0, 0]},
                                        {'id': 1, 'pos': [1, 0, 0]}],
                              'pairs': [{'i': 0, 'j': 1, 'J': 1.5},
                                        {'i': 0, 'j': 1, 'V': 0.0, 'A': 0.0}]})
C.check('active channels are derived from pair numbers',
        _derived['terms'] == ['I2'], _derived['terms'])
_withV = mb.validate_calib({'atoms': [{'id': 0, 'pos': [0, 0, 0]},
                                      {'id': 1, 'pos': [1, 0, 0]}],
                            'pairs': [{'i': 0, 'j': 1, 'J': 1.0, 'V': 2.0}]})
C.check('V≠0 turns on I1', 'I1' in _withV['terms'], _withV['terms'])

# 指纹：同样数字同样指纹；改一个数就变
_fp1 = mb.calib_summary(_derived)['fingerprint']
_changed = mb.validate_calib({'atoms': [{'id': 0, 'pos': [0, 0, 0]},
                                        {'id': 1, 'pos': [1, 0, 0]}],
                              'pairs': [{'i': 0, 'j': 1, 'J': 1.5000001}]})
C.check('calib fingerprint is content-addressed',
        _fp1 == mb.calib_summary(_derived)['fingerprint']
        and _fp1 != mb.calib_summary(_changed)['fingerprint'], _fp1)

for solver, kw in [('swap-tebd-gpu','max_step_ns'),('peps-real-time','chi_max'),
                   ('tdvp-trajectories','truncerr'),('tebd-gpu','memory_budget_mb')]:
    request=dict(params(cal), **{'manybody.solver':solver,'manybody.max_step_ns':.123,
                 'manybody.chi_max':7,'manybody.truncerr':1e-7,'manybody.memory_budget_mb':512.})
    spec=runspec.build_run(None,request)
    expected={'max_step_ns':.123,'chi_max':7,'truncerr':1e-7,'memory_budget_mb':512.}[kw]
    C.check('%s forwards %s'%(solver,kw),spec['engine_kwargs'][kw]==expected)

for solver in ('classical-spin','peps-real-time','tdvp-trajectories'):
    rejects('RunSpec rejects unsupported segments '+solver,lambda: runspec.build_run(
        None,dict(params(cal),**{'manybody.solver':solver,
                                'manybody.drive_segments':[dict(t0=0.,Om=[1.,2.])]})))
for fn in (evolve_classical_spin,evolve_peps,evolve_trajectories):
    rejects('direct entry rejects segments '+fn.__name__, lambda: fn(cal,t_grid=[0.,.1],segments=[dict(t0=0.)]))
    rejects('direct entry cancels '+fn.__name__, lambda: fn(cal,t_grid=[0.,.1],progress_arr=[0.,1]))

# Shared sparse compiler must preserve the complex matrix and arbitrary basis order.
cal3=calibration(3)
cal3['pairs']=[dict(i=0,j=1,V=.6,J=.3+.2j,A=.1-.07j),
               dict(i=1,j=2,V=.2,J=-.1+.3j,A=.04j)]
om=[.3j,.1-.2j,.4]; de=[-.3,.2,.8]
full=build_H_sparse(cal3,Om=om,delta=de)
indices=np.array([4,0,3,1,2])
reduced=build_H_sparse(cal3,Om=om,delta=de,basis_indices=indices)
C.check('one compiler preserves complex projected matrix',
        np.max(abs(reduced.toarray()-full[indices][:,indices].toarray()))<1e-12)
plan=build_plan(cal3,Om=om,delta=de,constraint={'kind':'pxp','edges':[[0,1],[1,2]]})
basis=np.asarray(plan['states'])
C.check('plan compiler equals independent full-space projection',np.max(abs(
    compile_sparse(plan).toarray()-full[basis][:,basis].toarray()))<1e-12)
before=set(plan)
compile_sparse(plan)
C.check('compilation does not mutate a shared plan',set(plan)==before)
r=evolve_manybody(cal3,psi0='rgg',t_grid=[0.,.2],plan=plan,record_psi=True)
C.check('projected site identities and column labels',np.max(abs(r['pops_atom'][0]-[1,0,0]))<1e-12
        and r['pops'].shape[1]==len(r['states']) and r['pops_kind']=='site_occupation')
C.check('projected physical entropy and state basis are explicit',abs(r['S'][0])<1e-12
        and len(r['psi_amp'])==len(basis) and r['basis']['bit_order']=='site0-msb')
g=evolve_manybody(cal3,psi0='ground',t_grid=[0.,.2],plan=plan)
C.check('projected ground initial state runs',np.isfinite(g['pops_atom']).all())
with patch.object(csr_matrix,'toarray',side_effect=AssertionError('dense allocation forbidden')):
    sparse=evolve_manybody(cal3,psi0='rgg',t_grid=[0.,.2],plan=plan,dense_max=0)
C.check('projected sparse propagation agrees with dense',np.max(abs(sparse['pops_atom']-r['pops_atom']))<1e-11)
gs=eq_route.solve_equilibrium(dict(calib=cal3,plan=plan,n_sites=3),backend='ed',return_state=True)
expected=basis_bits(3,basis) @ abs(gs['vec'])**2
C.check('projected ground observables use the same site order',np.max(abs(gs['n_atom']-expected))<1e-12)
number_plan=build_plan(calibration(3),delta=[-1.,0.,1.],
                      constraint={'kind':'pxp','edges':[[0,1],[1,2]]})
number_gs=eq_route.solve_equilibrium(dict(calib=calibration(3),plan=number_plan,
    n_sites=3,charge_sector=1),backend='ed',return_state=True)
C.check('charge and blockade compose with full-space basis identities',
        abs(number_gs['E0']+1)<1e-12 and np.max(abs(number_gs['n_atom']-[1,0,0]))<1e-12
        and set(number_gs['sector_idx'])=={1,2,4})
rejects('invalid initial projection is visible',lambda:evolve_manybody(cal3,psi0='rrg',t_grid=[0.,1.],plan=plan))
C.check('independent set counting avoids full enumeration',independent_set_count(40,[])==1<<40
        and independent_set_count(30,[[i,i+1] for i in range(29)])==2178309)
rejects('projected basis enumeration is bounded',lambda: independent_states(40,[],max_states=20))

# A drive discontinuity between output times must be integrated exactly.
cal1=calibration(1)
segments=[dict(t0=0.,Om=[1.]),dict(t0=13.,Om=[0.],delta=[2.])]
seg=evolve_manybody(cal1,psi0='g',t_grid=[0.,50.],segments=segments)
v=expm(-1j*CONV*37*build_H_sparse(cal1,delta=[2.]).toarray()) @ (
  expm(-1j*CONV*13*build_H_sparse(cal1,Om=[1.]).toarray()) @ np.array([1.,0.]))
C.check('ED splits exact segment boundaries',np.max(abs(seg['pops'][-1]-abs(v)**2))<1e-12)
rejects('free fermions refuse local flips',lambda:eq_route.solve_equilibrium(
    dict(calib=cal,n_sites=2,Om=[1.,1.]),backend='free-fermion'))
free=solve_ground_state(cal,delta=[-2.,3.])
ref=eq_route.solve_equilibrium(dict(calib=cal,n_sites=2,delta=[-2.,3.]),backend='ed')
C.check('zero inactive interactions allow free fermions and correct unconstrained gap',
        abs(free['E0']-ref['E0'])<1e-12 and abs(free['gap']-ref['gap'])<1e-12
        and np.max(abs(free['n_atom']-ref['n_atom']))<1e-12)
rejects('unsupported equilibrium task/backend refused',lambda:eq_route.solve_equilibrium(
    dict(calib=cal,n_sites=2),want='thermal',backend='dmrg'))
rejects('explicit exact thermal computation respects memory budget',lambda:eq_route.solve_equilibrium(
    dict(calib=cal,n_sites=2),want='thermal',backend='ed',memory_budget_mb=.0001))
thermal_spec=runspec.build_run(None,dict(params(cal),**{
    'manybody.want':'thermal','manybody.eq_backend':'ed','manybody.betas':[.3,1.],
    'manybody.kpm_num':32,'manybody.kpm_nvec':1,'manybody.qmc_burn_in':0}))
thermal=runspec.evolve_run(thermal_spec)
C.check('thermal product entry carries beta axes and preserves zero options',
        np.array_equal(thermal['betas'],[.3,1.]) and thermal_spec['qmc']['burn_in']==0
        and thermal['data_schema']['quantities']['E']['axes']==['beta'])

# KPM DOS must not obtain a full spectrum just to choose the output grid.
Hs=csr_matrix(np.diag(np.linspace(-2,3,32)))
with patch.object(csr_matrix,'todense',side_effect=AssertionError('dense DOS forbidden')):
    dos=eq_route.solve_equilibrium(dict(H=Hs,n_sites=5),want='dos',backend='kpm',num=32,nvec=2)
C.check('KPM DOS stays sparse and has real energy axis',np.isfinite(dos['rho']).all()
        and dos['energies'][0]<-1.9 and dos['energies'][-1]>2.9)

# Exercise every declared product entry on a real two-site numerical problem.
for name in TIME_BACKENDS:
    if 'gpu' in name:
        import torch
        if not torch.cuda.is_available():
            C.skip('public entry '+name,'CUDA unavailable')
            continue
    spec=runspec.build_run(None,dict(params(cal),**{'manybody.solver':name,
        'manybody.n_trajectories':1,'manybody.chi_max':4}))
    progress=[0.,0]
    out=runspec.evolve_run(spec,progress_arr=progress)
    C.check('public entry '+name,out['pops'].shape[1]==len(out['states'])
            and np.isfinite(out['pops']).all() and out['data_schema']['version']==1
            and abs(progress[0]-1.)<1e-12
            and all(axis in out['data_schema']['axes']
                    for quantity in out['data_schema']['quantities'].values()
                    for axis in quantity['axes']))

# The persistence boundary must retain data semantics and optional results.
import pickle
import tempfile
from pathlib import Path
from types import SimpleNamespace
from mecha_v2.bridge import ElOps
with tempfile.TemporaryDirectory() as folder:
    aid=ElOps._land(SimpleNamespace(root=Path(folder)),dict(labels=['site0','site1','site2']),r)
    with (Path(folder)/(aid+'.pkl')).open('rb') as stream:
        _labels,stored=pickle.load(stream)
    C.check('artifact roundtrip retains projected basis, descriptors and all engine fields',
            stored.keys()==r.keys() and np.array_equal(stored['basis_states'],r['basis_states'])
            and stored['data_schema']==r['data_schema'])

sys.exit(0 if C.summary() else 1)
