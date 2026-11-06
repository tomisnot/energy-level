"""Lossless finite tensor-network dtype selection; no model-specific heuristics."""
import numpy as np
from .backends import MPS_DTYPES

POLICIES = MPS_DTYPES


def validate_policy(policy):
    if policy not in POLICIES:
        raise ValueError('mps_dtype must be auto, real or complex')
    return policy


def _real(array):
    if not np.isfinite(array.norm()):
        raise ValueError('Tensor contains nonfinite entries')
    return array.unary_blockwise(np.imag).norm(ord=np.inf) == 0.


def select_mpo_dtype(mpo, policy='auto'):
    """Copy a finite MPO, choosing float64 only for exactly real tensors.

    The test uses charge blocks, never a dense tensor/Hilbert-space expansion.
    Local operator dictionaries can contain unused complex operators.
    """
    from tenpy.networks.mpo import MPO
    validate_policy(policy)
    if mpo.bc != 'finite' or mpo.grouped != 1:
        raise ValueError('dtype selection requires an ungrouped finite MPO')
    real = all([_real(mpo.get_W(i)) for i in range(mpo.L)])
    if policy == 'real' and not real:
        raise ValueError('real MPO requested but compiled tensors have imaginary entries')
    dtype = np.float64 if real and policy != 'complex' else np.complex128
    tensors = [(mpo.get_W(i).unary_blockwise(np.real) if dtype == np.float64 else mpo.get_W(i)).astype(dtype)
               for i in range(mpo.L)]
    return MPO(mpo.sites, tensors, bc=mpo.bc, IdL=mpo.IdL, IdR=mpo.IdR,
               max_range=mpo.max_range, explicit_plus_hc=mpo.explicit_plus_hc,
               mps_unit_cell_width=mpo.unit_cell_width)


def select_mps_dtype(psi, policy='auto', *, complex_required=False):
    """Independent copy retaining phase, gauge, Schmidt values and norm.

    ``complex_required`` is used for real-time evolution and complex MPOs.
    ``real`` refuses to discard *any* nonzero imaginary amplitude.
    """
    from tenpy.networks.mps import MPS
    validate_policy(policy)
    if psi.bc != 'finite' or psi.grouped != 1:
        raise ValueError('dtype selection requires an ungrouped finite MPS')
    real = all([_real(psi.get_B(i, form=None)) for i in range(psi.L)])
    if policy == 'real' and (not real or complex_required):
        raise ValueError('real MPS requested for complex tensors or a complex operation')
    dtype = np.float64 if real and policy != 'complex' and not complex_required else np.complex128
    tensors = [(psi.get_B(i, form=None).unary_blockwise(np.real) if dtype == np.float64
                else psi.get_B(i, form=None)).astype(dtype) for i in range(psi.L)]
    sv = [psi.get_SL(i) for i in range(psi.L)] + [psi.get_SR(psi.L-1)]
    # Mixers can have matrix-valued Schmidt objects. Preserve them, but never
    # silently cast a complex gauge factor to real.
    if dtype == np.float64 and any(hasattr(s, 'norm') and not _real(s) for s in sv):
        if policy == 'real':
            raise ValueError('Complex Schmidt gauge cannot be converted to real')
        return select_mps_dtype(psi, 'complex')
    if dtype == np.float64:
        sv = [s.unary_blockwise(np.real).astype(dtype) if hasattr(s,'unary_blockwise') else s for s in sv]
    return MPS(psi.sites, tensors, sv, bc=psi.bc, form=psi.form,
               norm=psi.norm, unit_cell_width=psi.unit_cell_width)
