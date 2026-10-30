# -*- coding: utf-8 -*-
import os,sys
sys.path.insert(0,os.path.dirname(os.path.dirname(__file__)))
import numpy as np
from src.core.manybody.symmetry import reflection_capability, reflection_projector, project_hamiltonian
from src.core.manybody.matrix_free import hamiltonian_linear_operator
cal={'atoms':[{'id':i} for i in range(4)],'terms':['I2'],
     'pairs':[{'i':0,'j':1,'J':.5},{'i':1,'j':2,'J':.7},{'i':2,'j':3,'J':.5}]}
ok,_=reflection_capability(cal,Om=[1,2,2,1],delta=[.1,.2,.2,.1]); assert ok
Pp=reflection_projector(4,1); Pm=reflection_projector(4,-1)
assert np.allclose(Pp.T@Pp,np.eye(Pp.shape[1])) and np.allclose(Pm.T@Pm,np.eye(Pm.shape[1]))
op=hamiltonian_linear_operator(cal,Om=[0,0,0,0],delta=[.1,.2,.2,.1],terms=['I2'])
H=np.column_stack([op@np.eye(16)[:,i] for i in range(16)])
assert np.allclose(np.sort(np.linalg.eigvalsh(H)),np.sort(np.r_[np.linalg.eigvalsh(project_hamiltonian(H,4,1)),np.linalg.eigvalsh(project_hamiltonian(H,4,-1))]))
print('[symmetry] PASS=3 FAIL=0')
