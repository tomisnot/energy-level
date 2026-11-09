"""Finite two-site CUDA TDVP with an explicit CPU/GPU SVD choice.

The benchmarked hybrid uses dtype=float32 (complex64 tensors), svd_device=cpu.
Native checkpoints share the existing tdvp-dense representation.
"""
import numpy as np

from . import tdvp_dense as dense
from .backends import validate_time_capabilities
from .couplings import active_terms
from .dmrg import terms_from_calib
from .observables import normalize_pauli_strings, pauli_specs_to_mps_order
from .results import time_result
from .runtime import check_cancel, reject_options, segment_drive, time_steps
from .state import product_vectors


@time_result(approximation='two-site CUDA TDVP, Taylor local propagation')
def evolve_tdvp_cuda(calib, psi0=None, t_grid=None, Om=None, delta=None,
                     gamma=None, terms=None, segments=None, progress_arr=None,
                     log=None, chi_max=200, truncerr=1e-10, max_step_ns=0.1,
                     device='cuda', dtype='float64', svd_device='cpu',
                     svd_driver=None, taylor_tol=None, taylor_order=24,
                     mps_order=None, return_state=False, want_observables=True,
                     want_entropy=True, want_corr=False, link_pairs=None,
                     loops=None, pauli_strings=None, plan=None,
                     charge_sector=None, drive_interp='left', **kwargs):
    reject_options(kwargs)
    validate_time_capabilities('tdvp-cuda', segments=segments, plan=plan,
                              gamma=gamma, psi0=psi0, return_state=return_state,
                              charge_sector=charge_sector, mps_order=mps_order,
                              pauli_strings=pauli_strings)
    if want_corr or link_pairs or loops:
        raise ValueError('tdvp-cuda supports explicit Pauli strings, not corr/link_pairs/loops')
    import torch
    from . import tdvp_cuda_kernel as kernel
    target=torch.device(device)
    if target.type!='cuda' or not torch.cuda.is_available():
        raise RuntimeError('tdvp-cuda requires an available CUDA device')
    dtypes={'float32':torch.complex64, 'complex64':torch.complex64,
            'float64':torch.complex128, 'complex128':torch.complex128}
    if dtype not in dtypes:
        raise ValueError('tdvp-cuda dtype must be float32/float64/complex64/complex128')
    if svd_device not in ('cpu','cuda'):
        raise ValueError('svd_device must be cpu or cuda')
    if svd_driver not in (None,'gesvd','gesvdj') or (svd_device=='cpu' and svd_driver is not None):
        raise ValueError('svd_driver applies only to CUDA SVD and must be gesvd/gesvdj')
    if isinstance(chi_max,bool) or int(chi_max)!=chi_max or chi_max<1:
        raise ValueError('chi_max must be a positive integer')
    if not np.isfinite(truncerr) or not 0<=truncerr<1:
        raise ValueError('truncerr must be a finite relative singular-value cutoff in [0,1)')
    if isinstance(taylor_order,bool) or int(taylor_order)!=taylor_order or taylor_order<1:
        raise ValueError('taylor_order must be a positive integer')
    tdtype=dtypes[dtype]
    tolerance=(2e-7 if tdtype==torch.complex64 else 1e-10) if taylor_tol is None else float(taylor_tol)
    if not np.isfinite(tolerance) or tolerance<=0:
        raise ValueError('taylor_tol must be finite and positive')
    if max_step_ns is not None and (not np.isfinite(max_step_ns) or max_step_ns<=0):
        raise ValueError('max_step_ns must be finite and positive')
    n=len(calib.get('atoms') or [])
    t=np.asarray(t_grid,float)
    if n<2 or t.ndim!=1 or len(t)<2 or not np.isfinite(t).all() or np.any(np.diff(t)<0):
        raise ValueError('tdvp-cuda requires N>=2 and a finite nondecreasing time grid')
    Om=np.zeros(n,complex) if Om is None else np.asarray(Om,complex)
    delta=np.zeros(n) if delta is None else np.asarray(delta,float)
    if drive_interp not in ('left','midpoint'):
        raise ValueError('drive_interp must be left or midpoint')
    if gamma is not None and (np.asarray(gamma).shape!=(n,) or not np.isfinite(gamma).all()):
        raise ValueError('gamma must be a finite array of length N')
    if Om.shape!=(n,) or delta.shape!=(n,) or not np.isfinite(Om).all() or not np.isfinite(delta).all():
        raise ValueError('Om/delta must be finite arrays of length N')
    selected=dict(calib,terms=list(active_terms(calib) if terms is None else terms))
    order,onsite,couplings,_=terms_from_calib(selected,Om,delta,np.zeros(n),order=mps_order)
    order=list(map(int,order))
    if isinstance(psi0,dict) and psi0.get('kind')=='mps':
        initial=dense._load_dense_state(psi0,n,order)
        if abs(initial.norm-1.)>1e-8:
            raise ValueError('tdvp-cuda requires a normalized native checkpoint')
        if any(not np.isfinite(b).all() for b in initial.B):
            raise ValueError('Native MPS tensors must be finite')
        for i,s in enumerate(initial.S):
            bond=initial.B[i].shape[0] if i<n else 1
            if s.ndim!=1 or len(s)!=bond or not np.isfinite(s).all() or np.any(s<=0):
                raise ValueError('Native MPS singular values must be positive finite bond vectors')
        if any(len(f)!=2 or not np.isfinite(f).all() for f in initial.form):
            raise ValueError('Native MPS forms must be finite pairs')
    else:
        vectors=product_vectors(psi0,n)
        if vectors is None:
            raise ValueError('Prepare a native state before using tdvp-cuda')
        initial=dense.DenseMPS([vectors[a].reshape(1,2,1) for a in order],
                               [np.ones(1) for _ in range(n+1)],[(0.,1.)]*n)
    real_dtype=torch.float32 if tdtype==torch.complex64 else torch.float64
    mps=kernel.TorchMPS([torch.tensor(b,dtype=tdtype,device=target) for b in initial.B],
                        [torch.tensor(s,dtype=real_dtype,device=target) for s in initial.S],initial.form)
    matrices=dense.build_mpo(onsite,couplings,n)
    mpo=[torch.as_tensor(w,dtype=tdtype,device=target) for w in matrices]
    lp,rp=kernel.init_env(mps,mpo)
    specs=normalize_pauli_strings(pauli_strings,n_atoms=n) or []
    mapped=pauli_specs_to_mps_order(specs,order) if specs else []
    pops=np.zeros((len(t),n))
    entropy=np.zeros(len(t))
    norm_errors=[]
    observations={key:np.zeros((len(t),n),complex) for key in ('sx','sy','sz','sp','sm')} if want_observables else None
    if observations is not None:
        observations['phase']=np.full((len(t),n),np.nan)
    pauli=np.zeros((len(t),len(specs)),complex) if specs else None
    snapshot=None

    def record(k):
        nonlocal snapshot
        snapshot=dense.DenseMPS([b.detach().cpu().numpy().copy() for b in mps.B],
                                [s.detach().cpu().numpy().copy() for s in mps.S],list(mps.form))
        pops[k,order]=dense.expectation_pops(snapshot)
        if not np.isfinite(pops[k]).all():
            raise RuntimeError('Nonfinite CUDA TDVP state, reduce the step or increase precision')
        entropy[k]=dense.entropy_mid(snapshot) if want_entropy else 0.
        norm_errors.append(float(np.max(np.abs(dense.expectation_pops(snapshot,op=dense.I2)-1.))))
        if observations is not None:
            values=dense._observables_local(snapshot,n)
            for key in observations:
                observations[key][k,order]=values[key]
        if pauli is not None:
            pauli[k]=dense._pauli_expectation_dense(snapshot,mapped)
        if progress_arr is not None:
            progress_arr[0]=(k+1)/len(t)

    check_cancel(progress_arr)
    steps=0
    segment_builds=0
    current_drive=(tuple(Om),tuple(delta))
    with dense._blas_thread_limit():
        record(0)
        for k in range(1,len(t)):
            for a,b in time_steps(t[k-1],t[k],segments,max_step_ns):
                check_cancel(progress_arr)
                if segments:
                    om,de=segment_drive((a+b)/2,segments,n,interp=drive_interp)
                    key=(tuple(om),tuple(de))
                    if key!=current_drive:
                        # Exchange channels are fixed. Refresh only onsite MPO entries.
                        operators=np.array([de[atom]*dense.NMAT+om[atom]*.5*dense.PMAT+
                                            np.conj(om[atom])*.5*dense.MMAT for atom in order])
                        onsite_gpu=torch.as_tensor(operators,dtype=tdtype,device=target)
                        for i in range(n):
                            mpo[i][0,-1]=onsite_gpu[i]
                        lp,rp=kernel.init_env(mps,mpo)
                        current_drive=key
                        segment_builds+=1
                kernel.sweep(mps,mpo,lp,rp,float(b-a),chi_max=int(chi_max),truncerr=truncerr,
                             taylor_tol=tolerance,taylor_order=int(taylor_order),
                             svd_device=svd_device,svd_driver=svd_driver)
                steps+=1
                check_cancel(progress_arr)
            record(k)
            if log is not None:
                log('[tdvp-cuda] t=%.6g chi=%d'%(t[k],max(len(s) for s in mps.S)))
    torch.cuda.synchronize(target)
    if pauli is not None:
        if observations is None:
            observations={}
        observations.update(pauli_strings=pauli,pauli_string_labels=[s['label'] for s in specs])
    errors=dict(backend='tdvp-cuda',algorithm='two-site-tdvp',n_atoms=n,
                n_internal_steps=steps,max_chi=max(len(s) for s in mps.S),chi_max=int(chi_max),
                dtype=str(tdtype).split('.')[-1],device=str(target),svd_device=svd_device,
                svd_driver=svd_driver,taylor_tol=tolerance,taylor_order=int(taylor_order),
                local_expm='Taylor with heuristic order estimate',max_step_ns=max_step_ns,
                tf32_matmul_allowed=bool(torch.backends.cuda.matmul.allow_tf32),
                truncerr=truncerr,truncation_semantics='relative singular-value amplitude cutoff',
                max_local_norm_deviation=max(norm_errors),mps_order=order,
                segment_model_builds=segment_builds,drive_interp=drive_interp,
                n_segments=len(segments or []),pauli_count=len(specs))
    states=[dict(label='atom%d'%a['id'],atom_id=a['id']) for a in calib['atoms']]
    return dict(t=t,pops=pops,pops_atom=pops,states=states,method='manybody',backend='tdvp-cuda',
                S=entropy if want_entropy else None,corr=None,loss=None,diag=None,
                observables=observations,error_report=errors,
                **({'state':dense.make_dense_state(snapshot,order)} if return_state else {}))
