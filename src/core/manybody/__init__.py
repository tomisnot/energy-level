# -*- coding: utf-8 -*-
"""core 的多体引擎家族（N 原子阵列，ED 先行）：模型卡 + 装配层 + ED 演化。

设计见 内部记录（未随仓发布，存档在仓外）（该稿基于旧框架，pm_setup/route/src.ai 已删，
接入以 docs 里「多体引擎并入 core」方案为准）。路线（2026-09-17 裁决）：
- 几何完全自由（任意点云，不定死阵列）；
- 相互作用 = **4 维对空间 V_dd 全精确**：一阶直接收缩（含球基张量 T_qq'
  的 (θ,φ) 复相位结构），二阶 vdW 用 Löwdin downfolding 从伙伴态涌现
  （C3/C6 不硬编码，与 static_pair 同一哲学）；
- 后端 TeNPy（ED 先行，MPS 在 M3）；耗散 v1 缓做。

本包是 core 的一部分（第 7 个引擎家族），只读复用同包 static_pair/static_atom，
经 runspec 唯一分派点（dynamic.mode==2）接入语义层，遵守统一 Result 契约。
"""

# 平衡态产品入口（ED/GPU批量/Lanczos/KPM/DMRG 经 eq_route 路由，runspec method=gs/thermal/dos 可达）。
from .eq_engine import solve_equilibrium, build_equilibrium, run_equilibrium  # noqa: F401

__all__ = ['solve_equilibrium', 'build_equilibrium', 'run_equilibrium']
