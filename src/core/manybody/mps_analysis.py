"""Resource-aware finite-MPS diagnostics for explicitly supplied operators.

Site permutations are tensor-product (spin/boson) permutations, not fermionic
orbital exchanges. A diagnostic never certifies a symmetry of a Hamiltonian.
"""
from collections import Counter
import math
import time
import numpy as np
from .resources import require_workspace, budget_bytes
from .runtime import check_cancel, report_progress, progress_range
from .results import describe_result

__all__ = ['mpo_sum_variance', 'permutation_expectation', 'analyze_mps',
           'partition_model_terms', 'variance_workspace_bytes', 'validate_diagnostics']


def validate_diagnostics(options, n):
    if options is None:
        return {}
    if not isinstance(options,dict) or set(options)-{'variance','permutation','blocks','corr'}:
        raise ValueError('MPS diagnostics accepts only variance, permutation, blocks, corr')
    if 'corr' in options and not isinstance(options['corr'],(bool,np.bool_)):
        raise ValueError('diagnostics.corr must be boolean')
    if 'variance' in options and not isinstance(options['variance'],(bool,np.bool_)):
        raise ValueError('diagnostics.variance must be boolean')
    if options.get('blocks') is not None and options.get('permutation') is None:
        raise ValueError('diagnostics.blocks requires a permutation')
    if options.get('permutation') is not None:
        p=options['permutation']
        if (not isinstance(p,(list,tuple,np.ndarray)) or len(p)!=n or
                any(isinstance(i,(bool,np.bool_)) or not isinstance(i,(int,np.integer)) for i in p) or
                sorted(p)!=list(range(n))):
            raise ValueError('diagnostics.permutation must contain every atom index once')
    return dict(options)


def _finite(psi):
    if psi.bc != 'finite' or psi.grouped != 1:
        raise ValueError('Requires an ungrouped finite MPS')


def _state(state):
    from .state import validate_mps_state
    validate_mps_state(state, len(state.get('order', [])))
    return state['psi']


def _copy_workspace(psi):
    return 4 * sum(int(psi.get_B(i, form=None).size) * psi.get_B(i, form=None).dtype.itemsize
                   for i in range(psi.L))


def _hist(leg):
    return Counter(tuple(q) for q in leg.qconj * leg.to_qflat())


def _convolve(a, b, chinfo):
    result = Counter()
    for qa, na in a.items():
        for qb, nb in b.items():
            q = tuple(chinfo.make_valid(np.asarray(qa, dtype=int)+np.asarray(qb, dtype=int)))
            result[q] += na*nb
    return result


def variance_workspace_bytes(psi, a, b):
    """Conservative charge-block bound for <b^dagger a>; includes copy/SVD space.

    Supports trivial, U(1), cyclic and combined abelian charge information.
    The estimate is not an operating-system allocator peak guarantee.
    """
    _finite(psi)
    if a.L != psi.L or b.L != psi.L or a.bc != 'finite' or b.bc != 'finite':
        raise ValueError('MPO/MPS lengths or boundary conditions differ')
    for mpo in (a,b):
        if any(s.leg!=t.leg for s,t in zip(psi.sites,mpo.sites)):
            raise ValueError('MPO/MPS physical bases or charges differ')
    if a.explicit_plus_hc or b.explicit_plus_hc:
        raise ValueError('Materialize explicit_plus_hc MPOs before variance contraction')
    maximum = 1
    for i in range(psi.L):
        v = psi.get_B(i).get_leg('vR')
        # The second operator is daggered: conjugate virtual-leg orientation.
        legs = [v, v.conj(), a.get_W(i).get_leg('wR'), b.get_W(i).get_leg('wR').conj()]
        counts = Counter({tuple([0]*psi.chinfo.qnumber): 1})
        for leg in legs:
            counts = _convolve(counts, _hist(leg), psi.chinfo)
        maximum = max(maximum, max(counts.values(), default=1))
    physical = max(s.dim for s in psi.sites)
    itemsize = np.dtype(np.result_type(psi.dtype, a.dtype, b.dtype, np.complex128)).itemsize
    return int(_copy_workspace(psi) + 16*itemsize*physical**2*maximum)


def _cross(psi, a, bdag):
    """Return <psi|b^dagger a|psi> with no compressed applied state."""
    from tenpy.linalg import np_conserved as npc
    theta = psi.get_theta(0, n=1)
    wa = a.get_W(0).take_slice(a.get_IdL(0), 'wL')
    wb = bdag.get_W(0).take_slice(bdag.get_IdL(0), 'wL')
    env = npc.tensordot(theta, wa.replace_label('wR','wa'), axes=['p0','p*'])
    env = npc.tensordot(env, wb.replace_label('wR','wb'), axes=['p','p*'])
    env = npc.tensordot(theta.conj(),env,axes=[['vL*','p0*'],['vL','p']])
    for i in range(1,psi.L):
        t=psi.get_B(i,form='B')
        env=npc.tensordot(env,t,axes=['vR','vL'])
        env=npc.tensordot(env,a.get_W(i).replace_label('wR','wa'),axes=[['wa','p'],['wL','p*']])
        env=npc.tensordot(env,bdag.get_W(i).replace_label('wR','wb'),axes=[['wb','p'],['wL','p*']])
        env=npc.tensordot(env,t.conj(),axes=[['vR*','p'],['vL*','p*']])
    env=env.take_slice([a.get_IdR(psi.L-1),bdag.get_IdR(psi.L-1)],['wa','wb'])
    return complex(npc.trace(env,'vR','vR*'))


def mpo_sum_variance(psi, mpos, *, memory_budget_mb=512., progress_arr=None):
    """For H=sum(mpos), compute ||(H-<H>)psi||² exactly by all cross terms.

    Parts need not be Hermitian. For Hermitian H this is its energy variance;
    no Hermiticity or ground-state claim is inferred. Input psi is normalized.
    """
    _finite(psi); budget_bytes(memory_budget_mb)
    mpos=list(mpos)
    if not mpos:
        raise ValueError('At least one explicitly supplied MPO is required')
    check_cancel(progress_arr)
    estimate=max(variance_workspace_bytes(psi,a,b) for i,a in enumerate(mpos) for b in mpos[i:])
    require_workspace(estimate,memory_budget_mb,'Exact MPO-sum variance')
    norm=complex(psi.overlap(psi))
    if not np.isfinite(norm) or abs(norm-1.)>1e-8:
        raise ValueError('Variance requires a normalized MPS; source weight is never discarded')
    state=psi.copy();state.canonical_form(renormalize=False,cutoff=0.)
    energies=np.array([p.expectation_value(state) for p in mpos],complex)
    daggers=[p.dagger() for p in mpos]
    total=0.; scale=0.; done=0; count=len(mpos)*(len(mpos)+1)//2
    for i,a in enumerate(mpos):
        for j in range(i,len(mpos)):
            check_cancel(progress_arr)
            moment=_cross(state,a,daggers[j]);product=np.conj(energies[j])*energies[i]
            factor=1 if i==j else 2
            total+=factor*float(np.real(moment-product));scale+=factor*(abs(moment)+abs(product))
            done+=1;report_progress(progress_arr,done/count)
    roundoff=128*np.finfo(float).eps*max(1.,scale)
    if not np.isfinite(total) or total < -roundoff:
        raise RuntimeError('Invalid residual variance beyond estimated roundoff')
    return dict(energy=complex(energies.sum()),energy_variance=max(0.,total),
                groups=len(mpos),contractions=count,workspace_estimate_mb=estimate/2**20,
                roundoff_estimate=roundoff,method='exact cross-MPO contractions',
                definition='||(H-<H>)psi||^2; H=sum(explicit MPO parts)',
                hermiticity_verified=False)


def _permutation(psi, permutation, blocks, order):
    n=psi.L
    for seq in (permutation,order):
        if len(seq)!=n or any(isinstance(i,(bool,np.bool_)) or not isinstance(i,(int,np.integer)) for i in seq) or sorted(seq)!=list(range(n)):
            raise ValueError('permutation/order must be complete integer site permutations')
    inverse=np.argsort(order);mapping=[int(inverse[permutation[order[i]]]) for i in range(n)]
    if blocks is None:
        blocks=[];first=0
        while first<n:
            end=mapping[first]
            i=first
            while i<=end:
                end=max(end,mapping[i]);i+=1
            blocks.append(end-first+1);first=end+1
    if (any(isinstance(k,(bool,np.bool_)) or not isinstance(k,(int,np.integer)) or k<=0 for k in blocks)
            or sum(blocks)!=n):
        raise ValueError('Positive integer blocks must partition the MPS chain')
    maps=[];first=0
    for size in blocks:
        p=[mapping[first+i]-first for i in range(size)]
        if sorted(p)!=list(range(size)):
            raise ValueError('Permutation crosses the declared block boundary')
        for i,j in enumerate(p):
            left=psi.sites[first+i];right=psi.sites[first+j]
            if left.dim!=right.dim or left.leg != right.leg:
                raise ValueError('Permuted sites must share physical basis and charges')
            if np.any(np.asarray(getattr(left,'JW_exponent',0))):
                raise ValueError('Tensor-product permutation does not implement fermion exchange signs')
        maps.append(p);first+=size
    return list(blocks),maps


def permutation_expectation(psi, permutation, *, blocks=None, order=None, memory_budget_mb=512., progress_arr=None):
    """Exact <U>, with source atom i moved to permutation[i]. May be complex.

    order lists atom indices at MPS positions. No state expansion, SVD
    truncation, geometry inference, or assumption that U commutes with H.
    """
    from tenpy.linalg import np_conserved as npc
    _finite(psi);check_cancel(progress_arr);budget_bytes(memory_budget_mb)
    order=list(range(psi.L)) if order is None else list(order)
    blocks,maps=_permutation(psi,list(permutation),blocks,order)
    estimate=_copy_workspace(psi);first=0
    for size in blocks:
        d=math.prod(s.dim for s in psi.sites[first:first+size])
        if d>int(np.iinfo(np.intp).max):
            raise ValueError('Permutation block exceeds addressable array dimensions; supply smaller closed blocks')
        chiL=psi.get_B(first).shape[0];chiR=psi.get_B(first+size-1).shape[2]
        estimate=max(estimate,_copy_workspace(psi)+8*16*chiL*chiR*d)
        first+=size
    require_workspace(estimate,memory_budget_mb,'Block-permutation expectation')
    norm=complex(psi.overlap(psi))
    if not np.isfinite(norm) or abs(norm-1.)>1e-8:
        raise ValueError('Permutation expectation requires a normalized MPS')
    state=psi.copy();state.canonical_form(renormalize=False,cutoff=0.)
    leg=state.get_B(0).get_leg('vL')
    env=npc.Array.from_ndarray(np.ones((1,1),complex),[leg,leg.conj()],labels=['bra','ket'])
    first=0
    for b,(size,p) in enumerate(zip(blocks,maps)):
        check_cancel(progress_arr)
        # TeNPy get_theta(n=1) unconditionally returns form (1,1), ignoring
        # formL/formR. A transfer chain instead needs the right-canonical B.
        theta=(state.get_B(first,form='B').replace_label('p','p0') if size==1 else
               state.get_theta(first,n=size,formL=0.,formR=1.))
        transformed=theta.copy();transformed.ireplace_labels([f'p{i}' for i in range(size)],[f't{i}' for i in range(size)])
        inverse=np.argsort(p)
        transformed=transformed.transpose(['vL']+[f't{i}' for i in inverse]+['vR'])
        transformed.ireplace_labels([f't{i}' for i in inverse],[f'p{i}' for i in range(size)])
        env=npc.tensordot(env,transformed,axes=['ket','vL'])
        env=npc.tensordot(theta.conj(),env,axes=[['vL*']+[f'p{i}*' for i in range(size)],['bra']+[f'p{i}' for i in range(size)]])
        env.ireplace_labels(['vR*','vR'],['bra','ket']);first+=size
        report_progress(progress_arr,(b+1)/len(blocks))
    value=complex(env.to_ndarray()[0,0])*abs(state.norm)**2  # final 1x1 scalar only
    if not np.isfinite(value) or abs(value)>1+1e-8:
        raise RuntimeError('Permutation expectation requires a normalized finite state')
    return dict(permutation_overlap=value,blocks=blocks,workspace_estimate_mb=estimate/2**20,
                method='exact block-permutation transfer',hamiltonian_symmetry_verified=False)


def partition_model_terms(onsite, couplings, *, conserve=None, groups=1, mps_dtype='auto'):
    """Split explicit generic one/two-site terms without spatial assumptions."""
    from .dmrg import rydberg_cluster_model
    if isinstance(groups,(bool,np.bool_)) or not isinstance(groups,(int,np.integer)) or groups<1:
        raise ValueError('groups must be a positive integer')
    onsite=list(onsite);couplings=list(couplings)
    n=len(onsite);chunk=max(1,math.ceil(n/int(groups)));parts=[]
    for first in range(0,n,chunk):
        last=min(n,first+chunk)
        on=[pair if first<=i<last else (0.,0.) for i,pair in enumerate(onsite)]
        cp=[term for term in couplings if first<=term[1]<last]
        if cp or any(abs(d)+abs(o)>0 for d,o in on):
            parts.append(rydberg_cluster_model(on,cp,conserve=conserve,mps_dtype=mps_dtype).H_MPO)
    if not parts:
        parts=[rydberg_cluster_model(onsite,couplings,conserve=conserve,mps_dtype=mps_dtype).H_MPO]
    return parts


def analyze_mps(state, *, mpos=None, permutation=None, blocks=None, memory_budget_mb=512., progress_arr=None):
    """Public native-state diagnostic Result following data_schema version 1."""
    check_cancel(progress_arr);psi=_state(state);start=time.perf_counter()
    result=dict(backend='tenpy-mps-analysis',resolved_backend='tenpy-mps-analysis',
                n_sites=psi.L,order=list(state['order']),charge_sector=state.get('charge_sector'),
                used_options=dict(memory_budget_mb=memory_budget_mb,blocks=blocks))
    if mpos is None and permutation is None:
        raise ValueError('Request explicit MPO parts and/or a site permutation')
    if mpos is not None:
        v=mpo_sum_variance(psi,mpos,memory_budget_mb=memory_budget_mb,
            progress_arr=progress_range(progress_arr,0.,.5 if permutation is not None else 1.))
        result['energy_expectation']=v.pop('energy');result['energy_variance']=v['energy_variance'];result['variance_report']=v
    if permutation is not None:
        p=permutation_expectation(psi,permutation,blocks=blocks,order=state['order'],
                                  memory_budget_mb=memory_budget_mb,
                                  progress_arr=progress_range(progress_arr,.5 if mpos is not None else 0.,1.))
        result['permutation_overlap']=p.pop('permutation_overlap');result['permutation_report']=p
    result['error_report']=dict(elapsed_s=time.perf_counter()-start,exact_contractions=True,
                              interpretation='Diagnostics do not certify a ground state or a Hamiltonian symmetry')
    return describe_result(result)
