# -*- coding: utf-8 -*-
"""ED/MPS cross-check for generic nearest-neighbour link correlator time series."""
import os,sys
sys.path.insert(0,os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from checks.checklib import Checker
from src.core.manybody.engine_ed import evolve_manybody
from src.core.manybody.engine_mps import evolve_mps
C=Checker('manybody_link_observable')
cal={'atoms':[{'id':0,'pos':(0,0,0)},{'id':1,'pos':(1,0,0)}], 'pairs':[{'i':0,'j':1,'V':0.,'J':-1.,'A':0.}], 'terms':['I2']}
psi={'kind':'product','amplitudes':[[1/np.sqrt(2),1/np.sqrt(2)],[1/np.sqrt(2),1j/np.sqrt(2)]]}; t=np.array([0.,1.,2.]); kw=dict(psi0=psi,t_grid=t,Om=np.zeros(2,complex),delta=np.zeros(2),terms=['I2'],want_observables=True,link_pairs=[(0,1)],want_xy_corr=False)
e=evolve_manybody(cal,**kw); m=evolve_mps(cal,algorithm='tdvp',chi_max=8,truncerr=1e-10,**kw)
C.close('ED/MPS link correlator',float(np.max(np.abs(e['observables']['link_corr']-m['observables']['link_corr']))),0,atol=2e-8)
C.check('link pair labels',m['observables']['link_pairs']==[(0,1)])
C.check('link time series shape',m['observables']['link_corr'].shape==(3,1))
sys.exit(0 if C.summary() else 1)
