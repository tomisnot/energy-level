"""Explicit tensor-factor permutations and involutive-sector preparation.

Projection prepares an initial state; it does not constrain subsequent DMRG.
No geometry, model symmetry, or physical interpretation is inferred here.
"""
import numpy as np
from .mps_analysis import _state, _permutation, _copy_workspace, permutation_expectation
from .resources import require_workspace, budget_bytes
from .runtime import check_cancel,report_progress,progress_range
from .state import make_mps_state
from .results import describe_result

__all__=['apply_site_permutation','project_permutation_sector']


def apply_site_permutation(state, permutation, *, memory_budget_mb=512., progress_arr=None):
    """Independent native MPS for U|psi>, using untruncated adjacent swaps.

    permutation maps source atom indices to target indices. Local factors
    must share a spin/boson basis. Fermion exchange signs are not implemented.
    """
    check_cancel(progress_arr);budget_bytes(memory_budget_mb);psi=_state(state)
    p=list(permutation);order=list(state['order'])
    _permutation(psi,p,None,order)  # validates indices, physical bases, statistics
    positions=np.argsort(order);mapping=[int(positions[p[a]]) for a in order]
    desired=list(np.argsort(mapping));labels=list(range(psi.L))
    require_workspace(_copy_workspace(psi),memory_budget_mb,'Site-permutation MPS copy')
    transformed=psi.copy();transformed.canonical_form(renormalize=False,cutoff=0.)
    maximum=_copy_workspace(transformed);error=0.;max_chi=max(transformed.chi,default=1);swaps=0
    for destination,wanted in enumerate(desired):
        check_cancel(progress_arr)
        j=labels.index(wanted);distance=j-destination
        while j>destination:
            check_cancel(progress_arr);i=j-1
            left=transformed.get_B(i).shape[0];right=transformed.get_B(i+1).shape[2]
            d=transformed.sites[i].dim*transformed.sites[i+1].dim
            estimate=_copy_workspace(transformed)+32*16*left*right*d
            require_workspace(estimate,memory_budget_mb,'Untruncated site-permutation SVD')
            err=transformed.swap_sites(i,swap_op=None,trunc_par=dict(chi_max=None,svd_min=None,trunc_cut=0.))
            error+=float(err.eps);maximum=max(maximum,estimate);max_chi=max(max_chi,max(transformed.chi,default=1))
            labels[i],labels[j]=labels[j],labels[i];j-=1;swaps+=1
            report_progress(progress_arr,(destination+(distance-(j-destination))/distance)/psi.L)
        report_progress(progress_arr,(destination+1)/psi.L)
    if error>1e-12:raise RuntimeError('Untruncated permutation lost weight beyond roundoff')
    result=dict(backend='tenpy-mps-permutation',resolved_backend='tenpy-mps-permutation',n_sites=psi.L,
        state=make_mps_state(transformed,order,state.get('charge_sector')),
        used_options=dict(memory_budget_mb=memory_budget_mb),
        error_report=dict(swaps=swaps,truncation_error=error,max_intermediate_chi=max_chi,
                          workspace_estimate_mb=maximum/2**20,hamiltonian_symmetry_verified=False))
    report_progress(progress_arr,1.)
    return describe_result(result)


def project_permutation_sector(state, permutation, parity, *, blocks=None,
        chi_max=None, truncerr=0., min_weight=1e-12, memory_budget_mb=512., progress_arr=None):
    """Prepare normalized (I+parity*U)|psi>, returning the projection weight.

    U must be an involutive site permutation. Optional SVD compression is
    explicit; its error and the actual post-compression <U> are reported.
    This does not verify [H,U]=0 or select the lower-energy sector.
    """
    check_cancel(progress_arr);psi=_state(state);p=list(permutation)
    if isinstance(parity,(bool,np.bool_)) or not isinstance(parity,(int,np.integer)) or parity not in (-1,1):
        raise ValueError('parity must be +1 or -1')
    _permutation(psi,p,blocks,list(state['order']))
    if any(p[p[i]]!=i for i in range(psi.L)):
        raise ValueError('Sector preparation currently requires an involution U^2=I')
    if not np.isfinite(min_weight) or not 0<=min_weight<1:
        raise ValueError('min_weight must lie in [0,1)')
    if chi_max is not None and (isinstance(chi_max,(bool,np.bool_)) or not isinstance(chi_max,(int,np.integer)) or chi_max<1):
        raise ValueError('chi_max must be a positive integer or None')
    if not np.isfinite(truncerr) or not 0<=truncerr<1:
        raise ValueError('truncerr must lie in [0,1)')
    transformed=apply_site_permutation(state,p,memory_budget_mb=memory_budget_mb,
                                       progress_arr=progress_range(progress_arr,0.,.6))
    other=transformed['state']['psi'];overlap=complex(psi.overlap(other))
    if abs(overlap.imag)>1e-9 or abs(overlap.real)>1+1e-9:
        raise RuntimeError('Invalid expectation of involutive permutation')
    weight=float(np.clip((1+parity*overlap.real)/2,0.,1.))
    if weight<=min_weight:
        raise ValueError('Requested sector has zero or insufficient projection weight: %.16g' % weight)
    if psi.L<2:
        projected=psi.copy()
    else:
        size=0
        for i in range(psi.L):
            a=psi.get_B(i);b=other.get_B(i)
            size+=(a.shape[0]+b.shape[0])*a.shape[1]*(a.shape[2]+b.shape[2])
        require_workspace(24*16*size,memory_budget_mb,'Exact MPS linear combination')
        check_cancel(progress_arr)
        projected=psi.add(other,.5,.5*parity,cutoff=0.)
        projected.norm/=np.sqrt(weight)
    compression_error=0.;retained=1.
    if chi_max is not None or truncerr>0:
        check_cancel(progress_arr)
        estimate=_copy_workspace(projected)+32*16*max(b.size for b in [projected.get_B(i) for i in range(projected.L)])
        require_workspace(estimate,memory_budget_mb,'Projected-state compression')
        err=projected.compress_svd(dict(chi_max=chi_max,svd_min=None,trunc_cut=truncerr))
        compression_error=float(err.eps);retained=float(np.real(projected.overlap(projected)))
        if not np.isfinite(retained) or retained<=0:raise RuntimeError('Compression produced a zero/nonfinite state')
        projected.norm/=np.sqrt(retained)
    prepared=make_mps_state(projected,state['order'],state.get('charge_sector'))
    if blocks is None:
        # A long reflection can have a small MPS but an exponential closed
        # block. Without explicit blocks, audit via another exact SWAP MPS.
        check=apply_site_permutation(prepared,p,memory_budget_mb=memory_budget_mb,
                                    progress_arr=progress_range(progress_arr,.6,1.))
        after=complex(prepared['psi'].overlap(check['state']['psi']))
    else:
        after=permutation_expectation(prepared['psi'],p,blocks=blocks,order=state['order'],
            memory_budget_mb=memory_budget_mb,progress_arr=progress_range(progress_arr,.6,1.))['permutation_overlap']
    return describe_result(dict(backend='tenpy-mps-projection',resolved_backend='tenpy-mps-projection',
        n_sites=psi.L,state=prepared,projection_weight=weight,permutation_overlap=after,
        used_options=dict(parity=parity,chi_max=chi_max,truncerr=truncerr,
                          min_weight=min_weight,memory_budget_mb=memory_budget_mb),
        error_report=dict(compression_error=compression_error,compression_retained_norm=retained,
            parity_leakage=float(max(0.,(1-parity*after.real)/2)),
            hamiltonian_symmetry_verified=False,sector_constrained_dmrg=False,
            permutation=transformed['error_report'])))
