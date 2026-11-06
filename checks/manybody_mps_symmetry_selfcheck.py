"""Explicit permutation and sector preparation vs independent full vectors."""
import os,sys
from pathlib import Path
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[key]='1'
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from checks.checklib import Checker
from src.core.manybody import dmrg,eq_route
from src.core.manybody.state import make_mps_state
from src.core.manybody.mps_symmetry import apply_site_permutation,project_permutation_sector
from src.core.manybody.assemble import build_H_sparse
from tenpy.linalg import np_conserved as npc
from tenpy.networks.mps import MPS

C=Checker('manybody_mps_symmetry');rng=np.random.default_rng(3103101)
n=6;order=[2,5,0,3,1,4]
model=dmrg.rydberg_cluster_model([(0.,0.)]*n,[])
sites=model.lat.mps_sites();v=rng.normal(size=64)+1j*rng.normal(size=64);v/=np.linalg.norm(v)
psi=MPS.from_full(sites,npc.Array.from_ndarray(v.reshape([2]*n),[s.leg for s in sites],
                    labels=[f'p{i}' for i in range(n)]),bc='finite')
state=make_mps_state(psi,order)


def atom_vec(state):
    p=state['psi'];a=p.get_theta(0,n=p.L).to_ndarray().reshape([2]*p.L)*p.norm
    for i,s in enumerate(p.sites):
        a=np.take(a,np.argsort(np.real(np.diag(s.get_op('Nhat').to_ndarray()))),axis=i)
    return a.transpose(np.argsort(state['order'])).reshape(-1)


def permute(vec,permutation):
    n=len(permutation)
    out=np.zeros_like(vec)
    for i in range(len(vec)):
        j=0
        for a,b in enumerate(permutation):j|=((i>>(n-1-a))&1)<<(n-1-b)
        out[j]=vec[i]
    return out


def rejects(fn):
    try:fn()
    except (ValueError,TypeError,RuntimeError):return True
    return False


original=atom_vec(state);p=[5,4,3,2,1,0];expected=permute(original,p)
applied=apply_site_permutation(state,p)
cyclic=[1,2,0,4,5,3];cyclic_result=apply_site_permutation(state,cyclic)
C.group('arbitrary permutations preserve physical direction / metadata / phase',[
    ('involution vector',np.linalg.norm(atom_vec(applied['state'])-expected)<1e-11),
    ('cyclic vector',np.linalg.norm(atom_vec(cyclic_result['state'])-permute(original,cyclic))<1e-11),
    ('metadata',applied['state']['order']==order),
    ('input unchanged',np.linalg.norm(atom_vec(state)-original)<1e-12)])
prepared={}
for parity in [-1,1]:
    r=project_permutation_sector(state,p,parity);prepared[parity]=r
    exact=.5*(original+parity*expected);weight=float(np.vdot(exact,exact).real);exact/=np.sqrt(weight)
    C.group('explicit parity %d projection' % parity,[
        ('normalized vector',np.linalg.norm(atom_vec(r['state'])-exact)<1e-10),
        ('weight retained',abs(r['projection_weight']-weight)<1e-12),
        ('actual parity',abs(r['permutation_overlap']-parity)<1e-11),
        ('no leakage',r['error_report']['parity_leakage']<1e-11),
        ('weight semantics',r['data_schema']['quantities']['projection_weight']['quantity']=='symmetry_projection_weight'),
        ('no DMRG constraint claim',not r['error_report']['sector_constrained_dmrg'])])

r=project_permutation_sector(state,p,1,chi_max=1)
compressed=atom_vec(r['state']);actual=np.vdot(compressed,permute(compressed,p))
C.group('compression is explicit and reports measured sector leakage',[
    ('requested chi',max(r['state']['psi'].chi)<=1),
    ('nonzero error',r['error_report']['compression_error']>0),
    ('independent post-compression parity',abs(r['permutation_overlap']-actual)<1e-11),
    ('correct leakage',abs(r['error_report']['parity_leakage']-(1-actual.real)/2)<1e-11),
    ('original projection weight',abs(r['projection_weight']-prepared[1]['projection_weight'])<1e-12)])
C.group('unsupported/zero/resource-limited projections refuse explicitly',[
    ('zero sector weight',rejects(lambda:project_permutation_sector(prepared[1]['state'],p,-1))),
    ('non-involution',rejects(lambda:project_permutation_sector(state,cyclic,1))),
    ('invalid parity',rejects(lambda:project_permutation_sector(state,p,True))),
    ('budget',rejects(lambda:project_permutation_sector(state,p,1,memory_budget_mb=1e-6))),
    ('invalid chi',rejects(lambda:project_permutation_sector(state,p,1,chi_max=0)))])

large=18;large_model=dmrg.rydberg_cluster_model([(0.,0.)]*large,[])
product=MPS.from_product_state(large_model.lat.mps_sites(),[1]+[0]*(large-1),bc='finite')
large_result=project_permutation_sector(make_mps_state(product,list(range(large))),list(reversed(range(large))),1,
                                       memory_budget_mb=2.)
occup=large_result['state']['psi'].expectation_value('Nhat')
C.group('long reflection projection stays in the MPS representation',[
    ('probability',abs(large_result['projection_weight']-.5)<1e-12),
    ('actual parity',abs(large_result['permutation_overlap']-1)<1e-11),
    ('analytic end occupations',abs(occup[0]-.5)<1e-11 and abs(occup[-1]-.5)<1e-11 and np.max(abs(occup[1:-1]))<1e-11)])

cm=dmrg.rydberg_cluster_model([(0.,0.)]*4,[(.7+.2j,i,'P',i+1,'M') for i in range(3)]+
    [(.7-.2j,i,'M',i+1,'P') for i in range(3)],conserve='Sz')
cg=dmrg.dmrg_run(cm,chi_max=8,max_sweeps=6,want_state=True,product_state=[1,0,0,0])['psi']
cs=make_mps_state(cg,[2,0,3,1],1);cp=[3,2,1,0];cv=atom_vec(cs)
cr=project_permutation_sector(cs,cp,1);exact=.5*(cv+permute(cv,cp));cw=np.linalg.norm(exact)**2;exact/=np.sqrt(cw)
C.group('charged projection preserves physical basis / total charge / weight',[
    ('full vector',np.linalg.norm(atom_vec(cr['state'])-exact)<1e-10),
    ('sector metadata',cr['state']['charge_sector']==1),
    ('total occupation',abs(np.sum(cr['state']['psi'].expectation_value('Nhat'))-1)<1e-11),
    ('probability',abs(cr['projection_weight']-cw)<1e-11)])

cal=dict(atoms=[dict(id=i,pos=(float(i),0,0)) for i in range(n)],terms=['I2'],
         pairs=[dict(i=i,j=i+1,J=.7,V=0.,A=0.) for i in range(n-1)])
out=eq_route.solve_equilibrium(dict(calib=cal,n_sites=n,psi0=prepared[1]['state']),
    backend='auto',chi_max=16,max_sweeps=6,return_state=True)
C.group('projected native Result is a compatible warm start',[
    ('auto DMRG',out['backend']=='dmrg'),('order',out['order']==order),
    ('actual ED energy',abs(out['E0']-np.linalg.eigvalsh(build_H_sparse(cal).toarray())[0])<1e-10)])
sys.exit(0 if C.summary() else 1)
