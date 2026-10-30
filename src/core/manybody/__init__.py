# -*- coding: utf-8 -*-
"""Two-level many-body models, physical calibration and numerical backends.

Product calls enter through core.runspec. See backends.py for capability
names, basis.py for site/basis identity, and results.py for data semantics.
Low-level model/solver APIs remain available for explicit research models.
"""

# 平衡态产品入口（ED/GPU批量/Lanczos/KPM/DMRG 经 eq_route 路由，runspec method=gs/thermal/dos 可达）。
from .eq_engine import solve_equilibrium, build_equilibrium, run_equilibrium  # noqa: F401
from .hamiltonian_plan import build_plan, compile_sparse, plan_summary  # noqa: F401
from .free_fermion import check_applicability as free_fermion_check, solve_ground_state as free_fermion_ground_state  # noqa: F401
from .topology import (fukui_chern, loop_flux, twist_pairs, twist_calibration, wilson_loop,
                       spectral_flow, entanglement_spectrum, topology_result,
                       topology_report)  # noqa: F401

__all__ = ['solve_equilibrium', 'build_equilibrium', 'run_equilibrium',
           'build_plan', 'compile_sparse', 'plan_summary',
           'free_fermion_check', 'free_fermion_ground_state',
           'fukui_chern', 'loop_flux', 'twist_pairs', 'twist_calibration', 'wilson_loop',
           'spectral_flow', 'entanglement_spectrum', 'topology_result',
           'topology_report', 'evolve_classical_spin', 'evolve_trajectories', 'evolve_peps']

from .classical_spin import evolve_classical_spin
from .engine_trajectories import evolve_trajectories
from .engine_peps import evolve_peps
