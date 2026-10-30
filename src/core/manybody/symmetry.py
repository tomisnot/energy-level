# -*- coding: utf-8 -*-
"""Explicit, opt-in symmetry helpers for one-dimensional chains.

Only reflection (site i -> N-1-i) is implemented.  We require the caller to
declare a chain ordering and verify mirrored parameters; arbitrary point
clouds are rejected rather than guessed to be translation/reflection
symmetric.
"""
import numpy as np

def reflection_capability(calib, Om=None, delta=None, tol=1e-10):
    atoms = calib.get('atoms') or []; n=len(atoms)
    if n < 2: return False, '链长不足'
    d=np.zeros(n) if delta is None else np.asarray(delta,float)
    o=np.zeros(n) if Om is None else np.asarray(Om,float)
    if len(d)!=n or len(o)!=n or not (np.allclose(d,d[::-1],atol=tol) and np.allclose(o,o[::-1],atol=tol)):
        return False, 'Omega/delta 不具反射对称'
    bonds={}
    for p in calib.get('pairs') or []:
        i,j=sorted((int(p['i']),int(p['j'])))
        if j-i != 1: return False, '仅支持按给定链序的最近邻反射'
        key=(i,j); bonds[key]=(complex(p.get('J',0)),complex(p.get('V',0)),complex(p.get('A',0)))
    for (i,j),v in bonds.items():
        if (n-1-j,n-1-i) not in bonds or not all(abs(x-y)<tol for x,y in zip(v,bonds[(n-1-j,n-1-i)])):
            return False, '镜像键参数不一致'
    return True, 'reflection'

def reflection_projector(n, parity):
    """Return an orthonormal projector from reflection parity to bit basis."""
    n=int(n); parity=int(parity)
    if parity not in (-1,1): raise ValueError('parity 必须为 +1 或 -1')
    cols=[]; seen=set()
    for s in range(1<<n):
        if s in seen: continue
        t=0
        for i in range(n):
            if (s>>i)&1: t |= 1 << (n-1-i)
        seen.add(s); seen.add(t)
        if s==t:
            if parity==1: cols.append({s:1.0})
        elif parity==1: cols.append({s:2**-0.5,t:2**-0.5})
        else: cols.append({s:2**-0.5,t:-2**-0.5})
    P=np.zeros((1<<n,len(cols)))
    for j,c in enumerate(cols):
        for i,v in c.items(): P[i,j]=v
    return P

def project_hamiltonian(H, n, parity):
    P=reflection_projector(n, parity)
    return P.T.conj() @ np.asarray(H) @ P

__all__=['reflection_capability','reflection_projector','project_hamiltonian']
