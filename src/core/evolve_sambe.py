# -*- coding: utf-8 -*-
"""兼容 shim（t74）：Sambe 单体引擎已并入 src/core/sambe.py——本文件仅
再导出，一个过渡期后删除（届时请改从 src.core.sambe 导入）。"""
from .sambe import (  # noqa: F401
    evolve_sambe, evolve_exact, SambeSetup, _SambeSetup,
    _m_analytic, _group_res_peak, _probe_pops, _peak_const_groups,
    _const_env, _seg_decomp, _apply_prop, _pops_of, _embed,
    _frozen_apply, _defect_sweep, _defect_at, _propagate_adaptive,
    _reconstruct, _env_all_constant, _run_sambe, setup_P,
    _M_START, _M_MAX, _M_ANALYTIC_BW_MHZ, _DENSE_MAX,
    _ENV_PROBE_N, _DT_MIN_FRAC, _CONV, _gl_nodes, _GL_CACHE,
    frozen_seg_propagate, sector_sum_pops, sector_sum_amp, env_at,
    photon_geometry, embed, reconstruct_psi, physical_pops, _group_by_freq)
from .field_build import (  # noqa: F401  （t74 场构造归位 field_build）
    build_fields_exact, _validate_fields, _sublevel_row)


def __getattr__(name):
    # evolve_reference 归位独立模块（t74）——惰性转发保持旧属性访问
    if name == 'evolve_reference':
        from .evolve_reference import evolve_reference
        return evolve_reference
    raise AttributeError(name)
