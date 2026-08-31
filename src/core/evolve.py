# -*- coding: utf-8 -*-
"""单原子动态模拟引擎（evolve.py，t19 拆分后兼容 shim）。

原 evolve.py 已按方案 A（docs/src重命名方案.md）拆分为：
  - atomic_levels.py    能级构造（make_levels / gaussian / square / step）
  - field_build.py      场构造（field / group_field / real_field / arbitrary_field）
  - evolve_rwa.py       单原子 RWA 演化（ip4 / Lindblad / evolve 调度）
  - evolve_arbitrary.py 任意波形演化（实验室系 RK4）

本文件仅 re-export 全部公共与内部名——外部调用（UI/checks/scripts/封存模块）
零改动（from src.core import evolve / from src.core.evolve import X 照常）。
"""
from .atomic_levels import gaussian, square, step, make_levels
from .field_build import (field, group_field, real_field, arbitrary_field,
                          _env_samples, _sublevel_row, _floquet_ac_static,
                          _CONV, _POL_Q, _ENV_SAMPLES, _lifetime_cache)
from .evolve_rwa import (hamiltonian, _field_geometry, _assign_reference,
                         _ip4_reference, _ip4_vtilde_impl, _ip4_vtilde_sp,
                         _ip4_vtilde, _ip4_rk4_core_numpy, _ip4_rk4_core,
                         _evolve_ip4, decay, auto_decays, _resolve_state_idx,
                         _e_rot, _decay_ops, _lindblad_eligible,
                         _evolve_lindblad_eig, _lindblad_rhs_impl,
                         _lindblad_rk4_step_impl, _lindblad_rk4_core_impl,
                         _lindblad_rhs, _lindblad_rk4_step, _lindblad_rk4_core,
                         _evolve_lindblad, _evolve_full, _evolve_full_hybrid,
                         _onehot, _as_vector, evolve, rwa_error_estimate,
                         group_pops, level_pops, IP4_RWA_THRESH, _CONV,
                         _POL_Q, _ENV_SAMPLES, _lifetime_cache)
from .evolve_arbitrary import (_arb_interp, _arb_apply, _arb_rk4_core,
                               _evolve_arbitrary, _CONV)
