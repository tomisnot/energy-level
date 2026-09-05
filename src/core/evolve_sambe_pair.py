# -*- coding: utf-8 -*-
"""兼容 shim（t74）：Sambe 两体引擎已并入 src/core/sambe.py；两场两体
变体已归位 evolve_twofield.py——本文件仅再导出，一个过渡期后删除。"""
from .sambe import (  # noqa: F401
    evolve_sambe_pair, evolve_exact_pair, _PairSambeSetup, _pair_propagate,
    _pair_m_probe, _dense_decomp, _prop_step, _embed_pair, _physical_pops,
    _defect_pass, _subspace_segment, _volterra_rhs, _volterra_rule,
    _DENSE_MAX_PAIR, _NODE_LADDER, _M_MAX, _CONV, _gl_nodes,
)


def __getattr__(name):
    # 两场两体家族（t74 归位 evolve_twofield）——惰性转发
    tf_names = ('build_two_field_pair', 'vdd_rot_filter', '_pair_mw_Ms',
                'evolve_two_field_pair', '_embed2d', '_TF_DENSE_MAX',
                '_TF_MW_NRES_BW_MHZ', '_VDD_RWA_TOL_MHZ')
    if name in tf_names:
        from . import evolve_twofield as _tf
        return getattr(_tf, name)
    raise AttributeError(name)
