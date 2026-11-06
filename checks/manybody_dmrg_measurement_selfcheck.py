"""DMRG observables must use a canonical copy, even at a sweep limit."""
from pathlib import Path
import os,sys
from unittest.mock import patch
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[key]='1'
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from checks.checklib import Checker
from src.core.manybody import dmrg
from src.core.manybody.assemble import build_H_sparse
from tenpy.algorithms.dmrg import TwoSiteDMRGEngine

C=Checker('manybody_dmrg_measurement')
n=6;delta=[.17,-.3,.11,.43,-.26,.07]
cal=dict(atoms=[dict(id=i,pos=(i,0.,0.)) for i in range(n)],terms=['I2'],
    pairs=[dict(i=i,j=i+1,J=.7,V=0.,A=0.) for i in range(n-1)])
h=build_H_sparse(cal,delta=delta).toarray();codes=[i for i in range(1<<n) if i.bit_count()==3]
reference=np.linalg.eigvalsh(h[np.ix_(codes,codes)])[0]
original=TwoSiteDMRGEngine.run;raw_norm=[]


def gauge_output(engine):
    energy,psi=original(engine)
    psi=psi.copy();psi.convert_form('B')
    i=n//2-1;a=psi.get_B(i,form=None).copy();b=psi.get_B(i+1,form=None).copy()
    factors=np.linspace(.7,1.3,a.shape[2])
    a.iscale_axis(factors,'vR');b.iscale_axis(1/factors,'vL')
    psi.set_B(i,a,form=None);psi.set_B(i+1,b,form=None)
    raw_norm.append(float(np.real(psi.overlap(psi,ignore_form=True))))
    return energy,psi


try:
    with patch.object(TwoSiteDMRGEngine,'run',gauge_output):
        out=dmrg.dmrg_cluster(cal,delta=delta,charge_sector=3,chi_max=16,max_sweeps=6,
            return_state=True,diagnostics=dict(variance=True,permutation=list(range(n))))
    C.group('finite normalized noncanonical optimizer output',[
        ('input physical norm',abs(raw_norm[0]-1)<1e-11),
        ('ED energy',abs(out['E0']-reference)<1e-10),
        ('exact charge in measured occupations',abs(sum(out['n_atom'])-3)<1e-11),
        ('variance',out['energy_variance']<1e-10),
        ('identity',abs(out['permutation_overlap']-1)<1e-11),
        ('native state norm',abs(out['state']['psi'].overlap(out['state']['psi'])-1)<1e-11)])
except (ValueError,RuntimeError) as exc:
    C.check('finite normalized noncanonical optimizer output',False,repr(exc))
sys.exit(0 if C.summary() else 1)
