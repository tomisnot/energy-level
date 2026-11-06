# -*- coding: utf-8 -*-
import os,sys
sys.path.insert(0,os.path.dirname(os.path.dirname(__file__)))
try:
 import tenpy
except Exception:
 print('SKIP DMRG U(1): TeNPy unavailable'); raise SystemExit(0)
import numpy as np
from src.core.manybody.dmrg import dmrg_cluster
cal={'atoms':[{'id':i,'pos':(float(i),0,0)} for i in range(4)],'terms':['I1','I2'],
 'pairs':[{'i':i,'j':i+1,'V':.1,'J':.4,'A':0.0} for i in range(3)]}
r=dmrg_cluster(cal,Om=np.zeros(4),delta=np.array([-.2,-.1,.1,.2]),chi_max=32,max_sweeps=4,charge_sector=2)
assert r['backend']=='tenpy-dmrg-cluster' and np.isfinite(r['E0'])
print('[dmrg_u1] PASS=2 FAIL=0')
