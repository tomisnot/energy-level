# 有限 MPS 分析与资源契约

本契约覆盖有限 MPS 静态实数路径、精确算符和方差、显式 site 置换期望，以及基态接续
所需的排序与初态入口。几何、场纹理、离散群的物理解释和科学验收预算属于调用方。

## 产品配置与统一结果

产品仍走 `schema_mb -> eq_engine RunSpec -> eq_route -> dmrg`。新增配置：

| 键 | 默认 | 语义 |
|---|---|---|
| `manybody.mps_dtype` | `auto` | 通用 calib DMRG 的 `auto/real/complex` 静态张量策略 |
| `manybody.mps_order` | `None` | DMRG 或 CPU MPS 演化的显式 MPS 顺序，按位置列出原子下标 |
| `manybody.mps_diagnostics` | `None` | 通用 calib DMRG 的 `{variance: bool, permutation: [...], blocks: [...]}` |

`manybody.psi0` 的原生 TeNPy 状态或局部乘积态可作为 DMRG 初态；`manybody.return_state` 保持既有返回约定。明确请求 MPS 功能时，平衡态 `auto` 选择兼容的 DMRG。显式 ED 等不支持路径会拒绝这些选项；显式矩阵 H、projected Plan、旧 ladder 快捷接口不能伪装成支持新功能。

`permutation[i]=j` 表示把原子下标 i 的物理态移到 j。它不是稳定 `atom.id` 的编号表，也不是直接作用于 MPS 位置的表。`mps_order[k]=i` 则表示第 k 个 MPS 位置对应原子下标 i，转换仅有一个定义点。交换重排时整体交换 `(site, operator)`，保留复交换耦合的方向。

结果保留既有 E0、n_atom、S、chi、order、state。新增 `energy_variance`、`permutation_overlap` 和详细 `diagnostics`，并在 `data_schema.version=1` 描述其量、轴和单位。`error_report` 给实际 tensor/MPO dtype、sweep 数、收敛标志、最大及最终截断误差；`used_options` 记录所用策略。收敛标志或小方差不构成基态唯一性与尺寸收敛证明。

## Python 研究调用

```python
from src.core.manybody.eq_route import solve_equilibrium

# calib 由模型层提供，delta 是逐原子的 onsite 系数，单位 MHz。
out = solve_equilibrium(
    dict(calib=calib, delta=delta, n_sites=N, charge_sector=k,
         mps_order=order),
    backend='dmrg', chi_max=256, max_sweeps=40,
    mps_dtype='auto', memory_budget_mb=512,
    mps_diagnostics={'variance': True, 'permutation': permutation,
                     'blocks': blocks}, return_state=True)

# 原生接续：不重建乘积态，不改变荷区，不静默重排。
next_out = solve_equilibrium(
    dict(calib=calib, delta=new_delta, n_sites=N, psi0=out['state']),
    backend='dmrg', chi_max=512, return_state=True)
```

直接 `dmrg_rydberg_cluster` 使用参数 `psi0/order/mps_dtype/diagnostics/memory_budget_mb`；统一路由中的 order 则位于 `system['mps_order']`。直接数值模型可调用 `terms_from_calib(..., order=...)`，所有时间分段沿用同一显式顺序。接续 CPU 时间演化时，若使用了自定义顺序，应明确传 `evolve_mps(..., mps_order=order)`，或者产品配置 `manybody.mps_order`。

## 无相位丢失的 dtype 策略

`mps_dtype.select_mpo_dtype` 在编译后的守恒块张量中检查有限性和虚部，不构造完整 H 或把稀疏块张量稠密化。

- `auto`：所有张量虚部严格为零才用 float64，否则保留 complex128。
- `real`：遇到任何非零虚部明确拒绝；不使用容差偷偷删除相位。
- `complex`：使用 complex128。

MPS 使用公共构造器复制，保留局部基、荷、规范形式、奇异值、范数。实数 MPO 可以与带相位的复数初态共同使用。该策略不要求局部算符字典完全为实：未参与当前 Hamiltonian 的复算符不妨碍实数路径。它不尝试改变虚键规范以“寻找”实表示。

CPU 时间演化在 `exp(-iHt)` 前显式升回复数，包括单站点 TDVP；静态实数策略不能强制时间容器保持实数。已有 GPU `manybody.dtype=float32/float64` 仍是独立精度协议。

低层使用已编译模型时，可先通过 `rydberg_cluster_model(..., mps_dtype=...)` 或 `select_mpo_dtype` 选择 MPO 表示，再交给 `dmrg_run`；后者选择与模型兼容的 MPS dtype，不原地修改调用方的模型。统一 calib 入口会完成这两步。研究 API 的显式 `system['terms']` 同样传入 DMRG，且不修改共享 calib；产品配置仍固定全相互作用。

## 精确 MPO 和方差

```python
from src.core.manybody.mps_analysis import analyze_mps, mpo_sum_variance

# H 的定义就是这些显式 MPO 之和，不把任意分组冒充另一个完整 H。
diagnostics = analyze_mps(state, mpos=parts, memory_budget_mb=512)
```

`mpo_sum_variance(psi, parts)` 计算 `||(H-<H>)psi||²`，`H=sum(parts)`；对厄米 H 等于能量方差。计算所有 `<B†A>` 交叉项，不通过压缩后的 H|psi> 或随机采样近似。各个分组可以非厄米，结果不自动认证整体 H 的厄米性。与普通 `<H²>-<H>²` 的不同在非厄米输入上有明确含义。

高层 DMRG 按通用 onsite/二站点项的左端点分组，从少量组开始，依据荷块工作区估计增加组数。不会按某种晶格、边界、场模式或物理通道分支。低层 `partition_model_terms` 提供同一构建能力。正负矩消减的舍入估计、分组数、交叉项数和工作区估计均返回；超出舍入估计的负值或非有限值会抛错。

默认新增诊断工作区预算为 512 MiB；高层使用显式 `memory_budget_mb` 时跟随该值。预算控制新增精确收缩及其规范副本的保守工作区，不是整个 DMRG 流程、操作系统或分配器峰值的硬上限。低层 API 的显式 `None` 遵循通用资源接口，表示不设该预算。工作区不够时明确拒绝，不填零、不把未计算值当成精确方差。分组越多，交叉项成本按组数平方增长。

有限 MPS 的归一化是输入契约；没有自动丢弃算符作用态的谱权重。`explicit_plus_hc` 的简写 MPO 应先物化，当前明确拒绝，以免重复或漏掉伴随项。

## 显式块置换

```python
from src.core.manybody.mps_analysis import analyze_mps
r = analyze_mps(state, permutation=permutation, blocks=blocks,
                memory_budget_mb=512)
```

`blocks` 是沿 MPS 链的连续块长度，各块长度之和必须为 N，每块在置换下闭合。省略时由给定置换计算最小连续闭合块；若它覆盖很长链，物理腿乘积会指数增长，资源守卫会在生成该块张量前拒绝。

收缩得到整个态的 `<psi|U|psi>`，不截断 MPS、不生成 `2**N` 向量。一般循环置换不自伴，结果可以是真实复数。局部物理基和荷必须匹配。本接口是自旋/玻色张量积因子置换，拒绝需要 Jordan–Wigner 交换符号的费米子 site；不会把几何名字猜成某个对称操作。

原生 Result 接口沿用 g/r 二能级状态协议；低层 `permutation_expectation` 可用于共享物理基的有限 qudit MPS。得到期望值不证明 `[H,U]=0`，也不强制 DMRG 留在某个离散扇区。对近简并分支仍需独立初态、匹配 χ 与各探测参数处的能量比较。

## 二阶置换的显式扇区准备

```python
from src.core.manybody.mps_symmetry import project_permutation_sector

prepared = project_permutation_sector(
    state, permutation, parity=+1, blocks=blocks,
    chi_max=512, memory_budget_mb=1024)
weight = prepared['projection_weight']
next_initial_state = prepared['state']
```

`apply_site_permutation` 通过相邻 SWAP 与不丢弃非零奇异值的 SVD 应用明确的张量积因子置换。逐次检查工作区，保留原子顺序元数据和荷；不改变输入。

`project_permutation_sector` 目前只接受 `U²=I` 与 ±1 扇区，显式准备 `(I±U)|psi>/2`。返回归一态，同时单独保留投影概率 `projection_weight`，不丢弃物理权重。零或低于显式 `min_weight` 的分量会拒绝。一般有限群/非二阶置换的不可约表示投影尚不在本接口中。

不提供 chi 上限时保留投影态；显式提供 `chi_max/truncerr` 才压缩。结果记录 SVD 压缩误差、保留范数及**实际复测**的置换期望与扇区泄漏。给定 blocks 时可用分块 transfer 高效复测；未给 blocks 时使用额外的精确 SWAP 态重叠，避免长反射被迫生成指数物理块。预算不足会拒绝，输入始终保持独立。

投影不检查 Hamiltonian 是否与 U 对易，不决定哪个扇区能量较低，也不把后续普通 DMRG 变成对称性约束优化。调用方应将返回原生态作为 warm start，并继续测量其扇区性质。

## 验证入口

`checks/manybody_mps_analysis_selfcheck.py` 与 `checks/manybody_mps_symmetry_selfcheck.py` 纳入 `run_checks.py --core`。它们以不对称、含复相位模型和非定态为参照，覆盖独立 CSR/ED、U(1)/Z₂、复值循环置换、qudit、HDF5、两站点/单站点 TDVP、产品 RunSpec、预算/取消/不支持输入拒绝及分段进度；投影另以稠密向量及长链解析态检验权重、方向与压缩后的泄漏。既有 DMRG、状态接续与统一协议检查继续保留。
