"""Independent dense/ED references for finite-MPS analysis and public forwarding."""
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
    os.environ[key]='1'
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import mecha_site
import numpy as np
from scipy.linalg import expm
from checks.checklib import Checker
from src.core.manybody import dmrg,eq_route,eq_engine,runspec_mb
from src.core.manybody.assemble import build_H_sparse
from src.core.manybody.mps_dtype import select_mpo_dtype,select_mps_dtype
from src.core.manybody.mps_analysis import (mpo_sum_variance,permutation_expectation,
    analyze_mps,partition_model_terms,variance_workspace_bytes)
from src.core.manybody.state import make_mps_state,save_mps_checkpoint,load_mps_checkpoint
from src.core.manybody.engine_mps import evolve_mps
from src.core.evolve_util import CONV
from src.core import runspec,protocol
from tenpy.networks.mps import MPS
from tenpy.networks.site import BosonSite
from tenpy.linalg import np_conserved as npc

C=Checker('manybody_mps_analysis')
rng=np.random.default_rng(31031)


def rejects(fn):
    try:fn()
    except (ValueError,TypeError,RuntimeError):return True
    return False


def dense(psi):
    return psi.get_theta(0,n=psi.L).transpose(['vL']+[f'p{i}' for i in range(psi.L)]+['vR']).to_ndarray().reshape(-1)*psi.norm


def atom_vector(state):
    psi=state['psi'];a=dense(psi).reshape([2]*psi.L)
    for i,s in enumerate(psi.sites):
        a=np.take(a,np.argsort(np.real(np.diag(s.get_op('Nhat').to_ndarray()))),axis=i)
    return a.transpose(np.argsort(state['order'])).reshape(-1)


def permuted(vec,p,d=2):
    n=len(p);result=np.empty_like(vec)
    for source,digits in enumerate(np.ndindex(*([d]*n))):
        target=[0]*n
        for i,j in enumerate(p):target[j]=digits[i]
        result[np.ravel_multi_index(tuple(target),[d]*n)]=vec[source]
    return result


def graph(n,phases):
    return dict(atoms=[dict(id=i,pos=(float(n-i),.07*(i%2),0.)) for i in range(n)],terms=['I1','I2'],
        pairs=[dict(i=i,j=j,V=.13*(i+1)/(j+1),J=(.4+.05*i)*np.exp(1j*(.31*(j-i) if phases else 0.)),A=0.)
               for i in range(n) for j in range(i+1,n) if j-i in (1,3)])


def phase_kick(psi,site,diagonal):
    leg=psi.sites[site].leg
    op=npc.Array.from_ndarray(np.diag(diagonal),[leg,leg.conj()],labels=['p','p*'])
    psi.apply_local_op(site,op,unitary=True)


n=6;delta=np.array([.17,-.3,.11,.43,-.26,.07]);order=[3,0,5,1,4,2]
real_cal=graph(n,False);complex_cal=graph(n,True)
gs={}
for name,cal in [('real',real_cal),('complex',complex_cal)]:
    h=build_H_sparse(cal,delta=delta).toarray();codes=np.array([i for i in range(1<<n) if i.bit_count()==3])
    exact=np.linalg.eigvalsh(h[np.ix_(codes,codes)])[0]
    r=eq_route.solve_equilibrium(dict(calib=cal,n_sites=n,delta=delta,charge_sector=3,mps_order=order),
        backend='dmrg',chi_max=16,max_sweeps=8,return_state=True,
        mps_diagnostics=dict(variance=True,permutation=list(range(n))))
    gs[name]=r;v=atom_vector(r['state'])
    C.group(name+' charged ground / explicit order / dtype / Result',[
        ('ED ground energy',abs(r['E0']-exact)<1e-10),
        ('saved vector energy',abs(np.vdot(v,h@v)-r['E0'])<1e-10),
        ('complete norm',abs(np.vdot(v,v)-1)<1e-12),
        ('dtype',r['state']['psi'].dtype.kind==('f' if name=='real' else 'c')),
        ('identity',abs(r['permutation_overlap']-1)<1e-12),
        ('variance',r['energy_variance']<1e-10),
        ('semantics',r['data_schema']['quantities']['energy_variance']['unit']=='MHz^2'),
        ('actual options',r['used_options']['mps_dtype']=='auto' and 'solver_converged' in r['error_report'])])
    state=r['state'];psi=state['psi'].copy()
    phase_kick(psi,2,np.exp(1j*np.array([.2,-.7])))
    kicked=make_mps_state(psi,order,3);v=atom_vector(kicked);e=np.vdot(v,h@v)
    target=np.linalg.norm(h@v-e*v)**2
    _,on,cp,_=dmrg.terms_from_calib(cal,delta=delta,order=order)
    parts=partition_model_terms(on,cp,conserve='Sz',groups=3)
    measure=analyze_mps(kicked,mpos=parts)
    C.group(name+' nonstationary partition variance vs independent CSR',[
        ('nonzero residual',target>1e-3),('variance',abs(measure['energy_variance']-target)<1e-10),
        ('energy',abs(measure['energy_expectation']-e)<1e-11),
        ('input unchanged',abs(abs(psi.overlap(kicked['psi']))-1)<1e-12)])

# Compiled real MPOs with unused complex local operators still select float.
_,on,cp,_=dmrg.terms_from_calib(real_cal,delta=delta,order=order)
model=dmrg.cluster_model(on,cp,conserve='Sz',mps_dtype='complex')
real_model=select_mpo_dtype(model.H_MPO)
alias=eq_route.solve_equilibrium(dict(calib=real_cal,n_sites=n,delta=delta),backend='dmrg',
    sector=3,chi_max=16,max_sweeps=6,return_state=True)
C.check('DMRG sector and charge_sector share the same contract',alias['charge_sector']==3 and
        abs(alias['E0']-gs['real']['E0'])<1e-10 and alias['resolved_backend']=='dmrg')
selected=eq_route.solve_equilibrium(dict(calib=real_cal,n_sites=n,delta=delta,terms=['I2'],charge_sector=3),
    backend='dmrg',chi_max=16,max_sweeps=6)
indices=np.array([i for i in range(64) if i.bit_count()==3])
selected_h=build_H_sparse(real_cal,delta=delta,terms=['I2']).toarray()
C.group('DMRG honors explicit research terms without mutating calibration',[
    ('selected Hamiltonian',abs(selected['E0']-np.linalg.eigvalsh(selected_h[np.ix_(indices,indices)])[0])<1e-10),
    ('calibration unchanged',real_cal['terms']==['I1','I2'])])
C.group('strict dtype and gauge-safe copies',[
    ('compiled real detected',real_model.dtype.kind=='f'),
    ('forced real complex H refused',rejects(lambda: dmrg.dmrg_cluster(complex_cal,delta=delta,mps_dtype='real'))),
    ('complex state refused',rejects(lambda: select_mps_dtype(kicked['psi'],'real'))),
    ('original MPO unchanged',model.H_MPO.dtype.kind=='c'),
    ('invalid policy refused',rejects(lambda:select_mpo_dtype(model.H_MPO,'float32')))])

# Exact arbitrary cyclic permutations: complex expectations, single-site
# blocks, disjoint blocks and atom/MPS order conversion are all independent.
sites=model.lat.mps_sites();psi=gs['complex']['state']['psi'].copy()
phase_kick(psi,0,[1.,1j]);state=make_mps_state(psi,order,3);v=atom_vector(state)
cases=[(list(range(n)),[1]*n),([2,1,4,0,3,5],None),([3,0,5,1,4,2],[4,2])]
# Last permutation may cross the suggested split in MPS order: use an
# explicitly guaranteed [1,2,1,2] block mapping in that order instead.
local=[0,2,1,3,5,4];p=list(range(n))
for i,j in enumerate(local):p[order[i]]=order[j]
cases[-1]=(p,[1,2,1,2])
sub=[]
for p,blocks in cases:
    exact=np.vdot(v,permuted(v,p));r=analyze_mps(state,permutation=p,blocks=blocks)
    sub.append((str(p),abs(r['permutation_overlap']-exact)<1e-11))
C.group('atom-index permutations including mixed one-site blocks',sub)

# Three-level, complex, non-involutive permutation: no assumption <U> is real.
sites=[BosonSite(Nmax=2,conserve=None)]*3
v=rng.normal(size=27)+1j*rng.normal(size=27);v/=np.linalg.norm(v)
theta=npc.Array.from_ndarray(v.reshape(3,3,3),[s.leg for s in sites],labels=['p0','p1','p2'])
qudit=MPS.from_full(sites,theta,bc='finite');p=[1,2,0]
r=permutation_expectation(qudit,p)
C.group('non-Hermitian permutation on qudits',[
    ('complex witness',abs(np.vdot(v,permuted(v,p,3)).imag)>1e-3),
    ('orientation and complex value',abs(r['permutation_overlap']-np.vdot(v,permuted(v,p,3)))<1e-12)])

# Non-Hermitian *parts* must produce the residual of their sum, not <H^2>.
model=dmrg.cluster_model([(0.,0.)]*3,[(.7+.4j,0,'P',2,'M')],mps_dtype='complex')
from tenpy.algorithms.exact_diag import ExactDiag
ed=ExactDiag(model);ed.build_full_H_from_mpo();h=ed.full_H.to_ndarray()
sites=model.lat.mps_sites();v=rng.normal(size=8)+1j*rng.normal(size=8);v/=np.linalg.norm(v)
psi=MPS.from_full(sites,npc.Array.from_ndarray(v.reshape(2,2,2),[s.leg for s in sites],labels=['p0','p1','p2']),bc='finite')
r=mpo_sum_variance(psi,[model.H_MPO]);e=np.vdot(v,h@v)
C.check('non-Hermitian sum uses H-dagger H',abs(r['energy_variance']-np.linalg.norm(h@v-e*v)**2)<1e-12)
other=dmrg.cluster_model([(0.,0.)]*3,[(-.2+.3j,0,'M',1,'P')],mps_dtype='complex')
ed=ExactDiag(other);ed.build_full_H_from_mpo();summed=h+ed.full_H.to_ndarray()
r=mpo_sum_variance(psi,[model.H_MPO,other.H_MPO]);e=np.vdot(v,summed@v)
C.check('non-Hermitian cross terms retain complex conjugation',abs(r['energy_variance']-np.linalg.norm(summed@v-e*v)**2)<1e-12)

# A cyclic charge uses modulo arithmetic in the workspace estimate too.
cal=graph(4,True);cal['terms']=['I1','I2','I3']
for bond in cal['pairs']:bond['A']=.17+.09j
_,on,cp,_=dmrg.terms_from_calib(cal,delta=delta[:4],order=list(range(4)))
parity_model=dmrg.cluster_model(on,cp,conserve='parity',mps_dtype='auto')
pg=dmrg.dmrg_run(parity_model,chi_max=8,max_sweeps=6,want_state=True)['psi']
phase_kick(pg,1,[1.,1j]);pv=atom_vector(make_mps_state(pg,list(range(4))))
ph=build_H_sparse(cal,delta=delta[:4]).toarray();pe=np.vdot(pv,ph@pv)
parts=partition_model_terms(on,cp,conserve='parity',groups=2)
r=mpo_sum_variance(pg,parts)
C.check('cyclic charge variance vs independent CSR',abs(r['energy_variance']-np.linalg.norm(ph@pv-pe*pv)**2)<1e-10)

with patch('src.core.manybody.mps_analysis._cross',side_effect=AssertionError('allocated past budget')):
    budget_rejected=rejects(lambda:mpo_sum_variance(psi,[model.H_MPO],memory_budget_mb=1e-6))
with patch.object(MPS,'get_theta',side_effect=AssertionError('large theta allocated')):
    permutation_budget=rejects(lambda:permutation_expectation(psi,[2,1,0],memory_budget_mb=1e-6))
long_product=MPS.from_product_state([psi.sites[0]]*65,[0]*65,bc='finite')
C.group('resource, cancellation and malformed-input rejection',[
    ('variance budget before contraction',budget_rejected),('permutation budget before expansion',permutation_budget),
    ('cancelled',rejects(lambda:permutation_expectation(psi,[0,1,2],progress_arr=np.array([0.,1.])))),
    ('invalid permutation',rejects(lambda:permutation_expectation(psi,[0,0,2]))),
    ('unaddressable block refuses before allocation',rejects(lambda:permutation_expectation(long_product,list(reversed(range(65)))))),
    ('wrong blocks',rejects(lambda:permutation_expectation(psi,[2,1,0],blocks=[1,2]))),
    ('unknown diagnostics',rejects(lambda:eq_route.solve_equilibrium(dict(calib=real_cal,n_sites=n),backend='dmrg',mps_diagnostics={'typo':True}))),
    ('ED cannot ignore MPS options',rejects(lambda:eq_route.solve_equilibrium(dict(H=h,n_sites=3),backend='ed',mps_diagnostics={'variance':True}))),
    ('real-time cannot be forced real',rejects(lambda:select_mps_dtype(psi,'real',complex_required=True)))])

# Native snapshots survive serialization and warm starts; real-to-complex
# promotion is verified against independent full-matrix time propagation.
state=gs['real']['state'];old=state['psi'].copy()
with tempfile.TemporaryDirectory() as folder:
    path=Path(folder)/'ground.h5';save_mps_checkpoint(state,path);restored=load_mps_checkpoint(path)
    warm=eq_route.solve_equilibrium(dict(calib=real_cal,n_sites=n,delta=delta,psi0=restored),backend='dmrg',chi_max=16,max_sweeps=6,return_state=True)
    excited=restored['psi'].copy();excited.apply_local_op(2,'SZ',unitary=True)
    excited=make_mps_state(excited,order,3)
    r=evolve_mps(real_cal,psi0=excited,delta=delta,t_grid=[0.,.05/CONV],charge_sector=3,
        max_step_ns=.05/CONV,chi_max=16,return_state=True,mps_order=order)
    single=evolve_mps(real_cal,psi0=excited,delta=delta,t_grid=[0.,.05/CONV],charge_sector=3,
        max_step_ns=.05/CONV,chi_max=16,return_state=True,mps_order=order,algorithm='tdvp-single')
    exact=expm(-1j*.05*build_H_sparse(real_cal,delta=delta).toarray())@atom_vector(excited)
    C.group('HDF5 / warm-start / complex time propagation',[
        ('warm energy',abs(warm['E0']-gs['real']['E0'])<1e-10),
        ('order retained',warm['order']==order),('charge inherited',warm['charge_sector']==3),
        ('complex time container',r['state']['psi'].dtype.kind=='c'),
        ('ED propagated state',abs(abs(np.vdot(exact,atom_vector(r['state'])))-1)<1e-10),
        ('one-site ED propagation',abs(abs(np.vdot(exact,atom_vector(single['state'])))-1)<1e-10),
        ('input remains unchanged',abs(abs(old.overlap(state['psi']))-1)<1e-12),
        ('mismatched order refused',rejects(lambda:dmrg.dmrg_cluster(real_cal,delta=delta,psi0=restored,order=list(range(n))))),
        ('bad integer order refused',rejects(lambda:dmrg.terms_from_calib(real_cal,order=[0,1,2,3,4,4]))),
        ('charge mismatch refused',rejects(lambda:dmrg.dmrg_cluster(real_cal,psi0=restored,charge_sector=2)))])
    C.check('charged product state is not silently projected',rejects(lambda:dmrg.dmrg_cluster(
        real_cal,charge_sector=3,psi0=dict(kind='product',amplitudes=[[1.,1.]]*6))))

params={'dynamic.mode':2,'manybody.want':'gs','manybody.eq_backend':'auto',
    'manybody.calib':{'atoms':[dict(id=a['id'],pos=a['pos'],delta_mhz=float(delta[i]))
                               for i,a in enumerate(real_cal['atoms'])],
                      'pairs':list(real_cal.get('pairs') or [])},
    'manybody.charge_sector':3,'manybody.chi_max':16,'manybody.max_sweeps':6,
    'manybody.return_state':True,'manybody.mps_dtype':'auto','manybody.mps_order':order,
    'manybody.mps_diagnostics':dict(variance=True,permutation=list(range(n)))}
spec=eq_engine.build_equilibrium(None,params);r=runspec.evolve_run(spec)
schema=protocol.config_schema()
class Progress:
    def __init__(self):self.values=[0.,0.];self.history=[]
    def __getitem__(self,i):return self.values[i]
    def __setitem__(self,i,v):
        self.values[i]=v
        if i==0:self.history.append(v)
progress=Progress();eq_engine.run_equilibrium(spec,progress_arr=progress)
C.group('product schema -> RunSpec -> actual DMRG -> Result',[
    ('schema policy',schema['manybody.mps_dtype']['enum']==['auto','real','complex']),
    ('auto respects requested MPS features',r['backend']=='dmrg'),
    ('forwarded order',r['order']==order),('variance',r['energy_variance']<1e-10),
    ('identity',abs(r['permutation_overlap']-1)<1e-12),
    ('real execution',r['error_report']['tensor_dtype']=='float64'),
    ('descriptors',{'energy_variance','permutation_overlap'}<=r['data_schema']['quantities'].keys()),
    ('RunSpec keeps requested diagnostics',spec['mps_diagnostics']['variance'])])
C.check('nested DMRG/variance/permutation progress remains monotonic',
        np.all(np.diff(progress.history)>=-1e-14) and abs(progress.history[-1]-1)<1e-14)

sys.exit(0 if C.summary() else 1)
