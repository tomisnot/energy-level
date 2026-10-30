# -*- coding: utf-8 -*-
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import numpy as np
from src.core.manybody.free_fermion import check_applicability, solve_ground_state
from src.core.manybody.matrix_free import hamiltonian_linear_operator

cal = {'atoms':[{'id':0},{'id':1},{'id':2}], 'terms':['I2'],
       'pairs':[{'i':0,'j':1,'J':0.7},{'i':1,'j':2,'J':0.4}]}
ok, _ = check_applicability(cal); assert ok
r = solve_ground_state(cal, delta=[-0.2, 0.1, 0.3], charge_sector=1)
op = hamiltonian_linear_operator(cal, delta=[-0.2,0.1,0.3], terms=['I2'])
H = np.column_stack([op @ np.eye(8)[:,i] for i in range(8)])
idx = np.array([1,2,4])  # one-excitation basis in MSB convention
e = np.linalg.eigvalsh(H[np.ix_(idx, idx)])[0]
assert abs(r['E0'] - e) < 1e-10
bad = dict(cal, terms=['I1'], pairs=[{'i':0,'j':1,'J':0.7,'V':1.0}])
assert not check_applicability(bad)[0]
print('[free_fermion] PASS=3 FAIL=0')
