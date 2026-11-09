# -*- coding: utf-8 -*-
"""Compatibility entry for the CUDA TDVP kernel promoted into core."""
from src.core.manybody.tdvp_cuda_kernel import *  # noqa: F401,F403
from src.core.manybody.tdvp_cuda_kernel import (  # noqa: F401
    _FORMS, _two_site_ops_blas, _matvec_two_blas,
    _one_site_ops_blas, _matvec_one_blas, _taylor_order,
    _svd_guarded, _entropy)
