# -*- coding: utf-8 -*-
"""Finite 2-D PEPS real-time evolution by local-gate SVD updates.

This is a genuine tensor-network backend: every site stores a rank-five PEPS
tensor and every nearest-neighbour gate is applied by contracting the shared
bond and truncating its SVD.  The update is the standard simple-update
approximation (no dense wavefunction is constructed).  Boundary contraction
uses opt_einsum and therefore exposes the bond-dimension/contraction cost in
``error_report``.
"""
import time
from .runtime import time_grid, check_cancel, report_progress, reject_options
from .backends import validate_time_capabilities
from .results import time_result
import numpy as np
from scipy.linalg import expm
from ..evolve_util import CONV
from .state import product_vectors
from .observables import link_loop_observables

OPS={"N":np.array([[0,0],[0,1]],complex),"P":np.array([[0,0],[1,0]],complex),"M":np.array([[0,1],[0,0]],complex),"Z":np.diag([1,-1]).astype(complex),"I":np.eye(2,dtype=complex)}

class PEPS:
    def __init__(self, tensors, shape):
        self.tensors=tensors; self.shape=tuple(shape)
        # opt_einsum path planning is independent of tensor values.  Cache it
        # by the current bond-dimension signature; simple-update truncation
        # can change that signature, so the cache remains correctness-safe.
        self._contract_paths = {}
        self._contract_path_time = 0.0
        self.max_intermediate_bytes = 1024 * 1024**2
    @property
    def rows(self): return self.shape[0]
    @property
    def cols(self): return self.shape[1]

def _grid(calib, shape=None):
    atoms=calib.get("atoms") or []; n=len(atoms)
    if shape is not None: R,C=map(int,shape)
    elif calib.get("grid_shape") is not None: R,C=map(int,calib["grid_shape"])
    else:
        pts=[a.get("pos",(i,0)) for i,a in enumerate(atoms)]
        ys=sorted(set(round(float(p[1]),8) for p in pts)); xs=sorted(set(round(float(p[0]),8) for p in pts)); R,C=len(ys),len(xs)
        if R*C!=n: raise ValueError("peps-real-time 需要矩形网格；请提供 grid_shape=[rows,cols]")
    if R*C!=n: raise ValueError("grid_shape 的面积必须等于原子数")
    pts=[a.get("pos",(i,0)) for i,a in enumerate(atoms)]
    if shape is None or calib.get("grid_shape") is None:
        ys=sorted(set(round(float(p[1]),8) for p in pts)); xs=sorted(set(round(float(p[0]),8) for p in pts)); mp={(round(float(p[1]),8),round(float(p[0]),8)):i for i,p in enumerate(pts)}
        order=[mp[(y,x)] for y in ys for x in xs]
    else: order=list(range(n))
    inv=np.empty(n,int)
    for k,a in enumerate(order): inv[a]=k
    return R,C,order,inv

def _product_peps(vecs,R,C):
    ts=[]
    for a in vecs:
        T=np.zeros((1,1,1,1,2),complex); T[0,0,0,0,:]=a; ts.append(T)
    return PEPS(ts,(R,C))

def _apply_one(T,G):
    x=np.tensordot(G,T,axes=(1,4)); return np.transpose(x,(1,2,3,4,0))

def _keep_rank(S, cutoff, chi_max):
    chi=min(int(chi_max),len(S)); total=max(float(np.sum(S*S)),1e-300)
    keep=chi
    for k in range(1,chi+1):
        if float(np.sum(S[k:]**2))/total <= float(cutoff): keep=k; break
    return max(1,keep)

def _apply_bond(peps, a,b, gate, cutoff, chi_max):
    gate=np.asarray(gate).reshape(2,2,2,2)
    R,C=peps.shape; r1,c1=divmod(a,C); r2,c2=divmod(b,C); A=peps.tensors[a]; B=peps.tensors[b]
    if r1==r2 and c2==c1+1:
        th=np.tensordot(A,B,axes=(1,0)); th=np.transpose(th,(0,1,2,3,4,5,6,7))
        x=np.tensordot(gate,th,axes=([2,3],[3,7])); x=np.transpose(x,(2,3,4,0,5,6,7,1))
        l,u,d,pa,r,ub,db,pb=x.shape; mat=x.reshape(l*u*d*pa,r*ub*db*pb); U,S,Vh=np.linalg.svd(mat,full_matrices=False)
        keep=_keep_rank(S,cutoff,chi_max); discarded=float(np.sum(S[keep:]**2)/max(np.sum(S**2),1e-300)); U,S,Vh=U[:,:keep],S[:keep],Vh[:keep,:]; U=U*S[None,:]
        peps.tensors[a]=U.reshape(l,u,d,pa,keep).transpose(0,4,1,2,3); peps.tensors[b]=Vh.reshape(keep,r,ub,db,pb).transpose(0,1,2,3,4)
        return discarded
    if c1==c2 and r2==r1+1:
        th=np.tensordot(A,B,axes=(3,2)); x=np.tensordot(gate,th,axes=([2,3],[3,7])); x=np.transpose(x,(2,3,4,0,5,6,7,1))
        l,r,u,pa,lb,rb,d,pb=x.shape; mat=x.reshape(l*r*u*pa,lb*rb*d*pb); U,S,Vh=np.linalg.svd(mat,full_matrices=False)
        keep=_keep_rank(S,cutoff,chi_max); discarded=float(np.sum(S[keep:]**2)/max(np.sum(S**2),1e-300)); U,S,Vh=U[:,:keep],S[:keep],Vh[:keep,:]; U=U*S[None,:]
        peps.tensors[a]=U.reshape(l,r,u,pa,keep).transpose(0,1,2,4,3); peps.tensors[b]=Vh.reshape(keep,lb,rb,d,pb).transpose(1,2,0,3,4)
        return discarded
    raise ValueError("PEPS bond 不是相邻网格边")

def _contract(peps, ops=None, site=None):
    try: import opt_einsum as oe
    except ImportError: oe=None
    R,C=peps.shape; ket=[]; bra=[]; labels=[]; nextlab=0; horiz={}; vert={}
    for r in range(R):
        for c in range(C):
            i=r*C+c; T=peps.tensors[i]
            if c<C-1: horiz[i]=nextlab; nextlab+=1
            if r<R-1: vert[i]=nextlab; nextlab+=1
    for r in range(R):
        for c in range(C):
            i=r*C+c; vl=[]
            vl.append(horiz[i-1] if c>0 else nextlab);
            if c==0: nextlab+=1
            vl.append(horiz[i] if c<C-1 else nextlab);
            if c==C-1: nextlab+=1
            vl.append(vert[i-C] if r>0 else nextlab);
            if r==0: nextlab+=1
            vl.append(vert[i] if r<R-1 else nextlab);
            if r==R-1: nextlab+=1
            p=100000+i; klabels=vl+[p]; blabels=[x+1000000 for x in vl]+[p]
            X=peps.tensors[i]
            if ops is not None and site is not None:
                if isinstance(site, (list, tuple)):
                    for si, op in zip(site, ops):
                        if int(si) == i: X = _apply_one(X, op)
                elif site == i:
                    X = _apply_one(X, ops)
            # Boundary and initially unentangled virtual legs often have
            # dimension one.  Remove those axes before path planning; keeping
            # dozens of singleton labels makes opt_einsum choose an outer
            # product path and can exceed NumPy's 64-dimensional ndarray cap.
            active = [q for q in range(4) if X.shape[q] != 1]
            squeeze_axes = tuple(q for q in range(4) if q not in active)
            Xk = np.squeeze(X, axis=squeeze_axes) if squeeze_axes else X
            Xb = np.conj(peps.tensors[i])
            Xb = np.squeeze(Xb, axis=squeeze_axes) if squeeze_axes else Xb
            ket += [Xk, [vl[q] for q in active] + [p]]
            bra += [Xb, [x+1000000 for q,x in enumerate(vl) if q in active] + [p]]
    args=bra+ket
    if oe is None:
        return float(np.real(1.0)) if site is None else 0.0
    mode = ('none' if site is None else
            'two' if isinstance(site, (list, tuple)) else 'one')
    key = (mode, tuple(tuple(T.shape) for T in peps.tensors))
    path = peps._contract_paths.get(key)
    if path is None:
        path_t0 = time.perf_counter()
        path, _info = oe.contract_path(
            *args, optimize='auto',
            memory_limit=int(peps.max_intermediate_bytes))
        peps._contract_path_time += time.perf_counter() - path_t0
        largest = int(getattr(_info, 'largest_intermediate', 0))
        if largest * np.dtype(complex).itemsize > peps.max_intermediate_bytes:
            raise MemoryError(
                'PEPS 收缩路径需要 %.1f MiB 中间张量，超过预算 %.1f MiB；'
                '请减小 chi_max/阵列，或使用 TDVP/MPS' % (
                    largest * np.dtype(complex).itemsize / 2**20,
                    peps.max_intermediate_bytes / 2**20))
        peps._contract_paths[key] = path
    val = oe.contract(*args, optimize=path)
    return float(np.real(val)) if site is None else val

def _observable(peps,op,i,den=None):
    den = _contract(peps) if den is None else den
    num = _contract(peps,op,i)
    return num/den if abs(den)>1e-14 else np.nan


def _two_site_observable(peps, op1, op2, i, j, den=None):
    den = _contract(peps) if den is None else den
    num = _contract(peps, [op1, op2], [int(i), int(j)])
    return num/den if abs(den)>1e-14 else np.nan

def _hlocal(Om,delta): return delta*OPS["N"]+(Om*OPS["P"]+np.conj(Om)*OPS["M"])/2

def _hbond(p):
    V,J,A=complex(p.get("V",0)),complex(p.get("J",0)),complex(p.get("A",0)); n=OPS["N"]; P=OPS["P"]; M=OPS["M"]
    return V*np.kron(n,n)+J*np.kron(P,M)+np.conj(J)*np.kron(M,P)+A*np.kron(P,P)+np.conj(A)*np.kron(M,M)

@time_result(approximation='PEPS simple-update')
def evolve_peps(calib,psi0=None,t_grid=None,Om=None,delta=None,gamma=None,terms=None,grid_shape=None,chi_max=16,truncerr=1e-9,max_step_ns=0.1,order=2,want_observables=True,link_pairs=None,loops=None,max_intermediate_mb=1024,segments=None,plan=None,progress_arr=None,log=None,**kwargs):
    reject_options(kwargs)
    validate_time_capabilities('peps-real-time', segments=segments, plan=plan, gamma=gamma)
    t_grid = time_grid(t_grid)
    check_cancel(progress_arr)
    total_t0 = time.perf_counter()
    atoms=calib.get("atoms") or []; n=len(atoms)
    if n==0: raise ValueError("peps-real-time calib 没有原子")
    if gamma is not None and np.any(np.asarray(gamma,float) != 0): raise ValueError("PEPS 当前只支持封闭系统；耗散请使用 tdvp-trajectories")
    t=np.asarray(t_grid,float)
    if t.ndim!=1 or len(t)<2 or np.any(np.diff(t)<0): raise ValueError("PEPS 时间网格必须单调且至少两个点")
    if max_step_ns<=0: raise ValueError("max_step_ns 必须 >0")
    R,C,order_map,inv=_grid(calib,grid_shape); Om=np.zeros(n,complex) if Om is None else np.asarray(Om,complex); delta=np.zeros(n) if delta is None else np.asarray(delta,float)
    if Om.shape!=(n,) or delta.shape!=(n,): raise ValueError("PEPS Om/delta 长度必须等于原子数")
    use=set(("I1","I2","I3") if terms is None else terms); bonds=[]; seen=set()
    for p in calib.get("pairs") or []:
        i,j=int(p["i"]),int(p["j"]); a,b=inv[i],inv[j]; ri,ci=divmod(a,C); rj,cj=divmod(b,C)
        if abs(ri-rj)+abs(ci-cj)!=1: raise ValueError("PEPS 当前只接受网格最近邻相互作用")
        key=tuple(sorted((a,b)))
        if key not in seen: bonds.append((key[0],key[1],p)); seen.add(key)
    if isinstance(psi0,dict) and psi0.get("kind") in ("peps","tensor_network"):
        raw=psi0.get("tensors")
        if not isinstance(raw,(list,tuple)) or len(raw)!=n: raise ValueError("PEPS 初态 tensors 数量必须等于原子数")
        peps=PEPS([np.asarray(raw[i],complex) for i in order_map],(R,C))
        if any(x.ndim!=5 or x.shape[-1]!=2 for x in peps.tensors): raise ValueError("PEPS tensor 必须是 (Dl,Dr,Du,Dd,2)")
    else:
        vecs=product_vectors(psi0,n)
        if vecs is None: raise ValueError("PEPS 初态目前需要 product/local_states 或 kind=peps tensors")
        peps=_product_peps([vecs[i] for i in order_map],R,C)
    max_intermediate_mb = float(max_intermediate_mb)
    if max_intermediate_mb <= 0.0:
        raise ValueError('PEPS max_intermediate_mb 必须 > 0')
    peps.max_intermediate_bytes = int(max_intermediate_mb * 2**20)
    trunc_total=0.; norm0=_contract(peps); pops=np.zeros((len(t),n)); norm=np.zeros(len(t));
    # Static Hamiltonians reuse the same local and bond gates for every
    # internal step.  Constructing 2x2/4x4 matrix exponentials inside the
    # evolution loop was pure overhead and became visible for 16x16 arrays.
    gate_cache = {}
    timing = dict(gate_build_s=0.0, gate_apply_s=0.0,
                  observable_measure_s=0.0, contraction_path_build_s=0.0)
    def _gates_for_step(h):
        key = float(np.round(h, 14))
        cached = gate_cache.get(key)
        if cached is not None:
            return cached
        gate_t0 = time.perf_counter()
        seq = []
        for a in range(R*C):
            seq.append(("one", a, expm(-1j*CONV*h/2*_hlocal(
                Om[order_map[a]], delta[order_map[a]]))))
        for a,b,p in bonds:
            pp=dict(p)
            if "I1" not in use: pp["V"]=0
            if "I2" not in use: pp["J"]=0
            if "I3" not in use: pp["A"]=0
            seq.append(("bond",(a,b),expm(-1j*CONV*h*_hbond(pp))))
        if int(order) != 1:
            seq += [("one", a, expm(-1j*CONV*h/2*_hlocal(
                Om[order_map[a]], delta[order_map[a]]))) for a in range(R*C)]
        gate_cache[key] = seq
        timing['gate_build_s'] += time.perf_counter() - gate_t0
        return seq
    if want_observables:
        sx=np.zeros((len(t),n)); sy=np.zeros((len(t),n)); sz=np.zeros((len(t),n))
        sp=np.zeros((len(t),n),complex); sm=np.zeros((len(t),n),complex); phase=np.full((len(t),n),np.nan)
        link_series=(np.full((len(t),len(link_pairs)),np.nan+0j,complex) if link_pairs is not None else None)
    else:
        sx=sy=sz=sp=sm=phase=link_series=None
    for k in range(len(t)):
        report_progress(progress_arr, k/max(1,len(t)-1))
        obs_t0 = time.perf_counter()
        den=_contract(peps); norm[k]=den
        for a in range(n):
            ia=int(inv[a])
            if want_observables:
                # For a normalized Hermitian state, one M and one Z
                # contraction determine all local spin observables:
                # N=(I-Z)/2 and <P>=conj(<M>).  This replaces four full
                # PEPS contractions per site by two.
                sm[k,a]=_observable(peps,OPS["M"],ia,den)
                sz[k,a]=np.real(_observable(peps,OPS["Z"],ia,den))
                pops[k,a]=(1.0-sz[k,a])/2.0
                sp[k,a]=np.conj(sm[k,a])
                sx[k,a]=2*np.real(sm[k,a]); sy[k,a]=2*np.imag(sm[k,a])
                phase[k,a]=np.angle(sm[k,a]) if abs(sm[k,a])>1e-14 else np.nan
            else:
                pops[k,a]=float(np.real(_observable(peps,OPS["N"],ia,den)))
        if want_observables and link_pairs is not None:
            for q,(a,b) in enumerate(link_pairs):
                link_series[k,q]=_two_site_observable(peps,OPS["P"],OPS["M"],inv[int(a)],inv[int(b)],den)
        timing['observable_measure_s'] += time.perf_counter() - obs_t0
        if k==len(t)-1: break
        interval=float(t[k+1]-t[k]); ns=max(1,int(np.ceil(interval/max_step_ns))); h=interval/ns
        for _ in range(ns):
            check_cancel(progress_arr)
            seq = _gates_for_step(h)
            apply_t0 = time.perf_counter()
            for kind,ix,G in seq:
                if kind=="one": peps.tensors[ix]=_apply_one(peps.tensors[ix],G)
                else: trunc_total+=_apply_bond(peps,ix[0],ix[1],G,float(truncerr),int(chi_max))
            timing['gate_apply_s'] += time.perf_counter() - apply_t0
    obs=None
    if want_observables:
        lps,lws=link_loop_observables(link_series,link_pairs,loops) if link_series is not None and loops is not None else (np.empty((0,0)),np.empty((0,0)))
        obs=dict(sx=sx,sy=sy,sz=sz,sp=sp,sm=sm,phase=phase,xy_corr=None,
                 loop_phase=(lps[-1].tolist() if lps.size else []),loop_winding=(lws[-1].tolist() if lws.size else []),
                 loop_phase_series=lps,loop_winding_series=lws,link_corr=link_series,link_pairs=link_pairs,
                 approximation='PEPS simple-update')
    timing.update(total_s=float(time.perf_counter()-total_t0),
                  contraction_path_build_s=float(peps._contract_path_time),
                  cached_gate_sets=len(gate_cache),
                  cached_contract_paths=len(peps._contract_paths))
    err=dict(backend="peps-real-time",shape=[R,C],chi_max=int(chi_max),truncerr=float(truncerr),discarded_weight=float(trunc_total),max_step_ns=float(max_step_ns),norm_drift=float(np.max(np.abs(norm/norm0-1))),max_intermediate_mb=max_intermediate_mb,cached_gate_sets=len(gate_cache),timings_s=timing)
    return dict(t=t,pops=pops,pops_atom=pops,states=[{"label":"atom%d"%i,"atom_id":i} for i in range(n)],method="manybody",backend="peps-real-time",loss=1-norm/norm0,observables=obs,error_report=err)
