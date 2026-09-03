# 目标 6 设计：AC 化进 fields/两体引擎正式路径

> 2026-09-06 船长设计（t62 收官后实现）。t61 的 AC 化是 tempsimu 原型
> （_t61_acize.py 占位 + 22 态验证 norm 1.75e-5）；单原子版已正式化
> （evolve_mixed.ac_ize_physical_E，ae04d11 验证用例 1/3/V5）。本文档设计
> 全频段两体 Sambe（evolve_sambe_pair）的自动 AC 化正式路径。

## 1. 现状

- 单原子：evolve_mixed 入口自动 AC 化（ac_ize_physical_E：r>r_crit 远强对
  剔除 + far_sos 位移注入物理 E）✓ 已验证
- 两体（evolve_sambe_pair）：无 AC 化参数——22 态两体跑 AC 化需外部原型
  （t61）；不 AC 化则远强对（60S↔60P @295.7THz r=0.046）需 M=4 → dim 爆炸
- t61 结果：AC 化后 620ns norm 1.75e-5（vs 1.08e-2），dim 4356 vs 13068

## 2. 设计：ac_ize 进 evolve_sambe_pair

**入口**：`evolve_sambe_pair(..., ac_ize=True)`（默认 False 零行为变化）

**逻辑**（复用 evolve_mixed.ac_ize_physical_E——单原子 E 数组版）：
1. fields_A/B 的光场组（kind='exact'，omega>1e5）每场 A → r 判据分类
   （r=|A|/ω > r_crit=0.0135 → 远强对）
2. 远强对从 A 剔除 + far_sos 位移累加 E_shift（单原子向量——两体
   E_pair=E_i+E_j 自动含）
3. E_shift 注入：传播用 E+E_shift（K0_pair 构建前）——同单原子
   evolve_mixed 的 E2 语义
4. 返回 error_report 记 n_ac_pairs/E_shift 范围（诊断）

**关键**：A/B 场驱动同一 states（共用单原子 E 数组）→ 单 E_shift 向量对
A/B 场位移都适用（物理：AC 位移是单原子性质，两体态 E_i+E_j 位移=δ_i+δ_j）。

## 3. 与 per_freq_M/frozen_h_ns 的配合

- AC 化在 M 判据**之前**（剔除后近对 M 自动判据小——远强对不再推高 M）
- frozen_h_ns 不变（段长由包络定——AC 化只改对角）
- GUI：sambe 两体档默认 ac_ize=True（若 per_freq_M 未显式给大 M）——
  或由 route_auto two_body 的 sambe 路由自动带 ac_ize

## 4. 验证链（t61 复用）

- 3 态 C1≈A（t60 场景）→ 22 态短窗 → 620ns 全程 norm（t61 1.75e-5 复现）
- 回归 11/11+11/11（ac_ize 默认 False 零行为变化）

## 5. 实施顺序

1. evolve_mixed.ac_ize_physical_E 已就绪（单原子版）——extract 公共函数
2. evolve_sambe_pair 加 ac_ize 参数 + E_shift 注入点（K0_pair 前）
3. 验证 + GUI sambe 两体默认开
4. （可选）build_fields_exact 层 ac_ize 选项（GUI fields 构建处）

## 6. 依赖

- t62 收官（8 用例 + 620ns 数据）后实施——避免主线期间大改
- EP6 保护（9bde559）已就位（本改动不碰稀疏路径）
