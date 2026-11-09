# 使用反馈与需求（面向 CSL 项目的实测证据）

> 2026-11-07 · 使用者：CSL 项目侧 · 对象：EL 多体引擎维护者
> 全部数字都有复现命令（附录 A）。凡是我没实测的，标「未验证」，不写成结论。

---

## 0. 一页摘要

| 编号 | 项 | 类型 | 影响 | 建议优先级 |
|---|---|---|---|---|
| **B1** | iDMRG 扇区判据把守恒的 `P_iM_j` 也拒了（`idmrg.py:136`） | **bug** | 被迫跑无 U(1) 的稠密 iDMRG，慢 10–100×，结果不收敛 | **P0（一行改动）** |
| **B2** | `tdvp-dense` 的 dt 收敛是**一阶**（应为二阶） | **bug/精度** | 达到 1e−3 需要 dt≈0.2 ns，步数 ×25 | **P0** |
| **B3** | `tdvp-mpo` 与 `tdvp` 同输入末态差 **1.5e−2** | **疑似 bug** | 同一物理两条路给出不同答案 | **P0（需定性）** |
| **B4** | `bench_mps.py --algorithm tdvp-dense` 直接报错 | **工具** | 官方基准工具测不了新后端 | P1 |
| **B5** | 遗留键导致 `internal_error` 而非配置冲突 | UX | 排错成本 | P2 |
| **B6** | 摘要层丢 `corr`、`n_atom` 被拍平 | UX | 结果不可直接读 | P2 |
| **P1** | 性能契约按 `χ_max` 而非**实际 χ** 标定 | 度量 | “快 43×”在真实工况不成立（实测 8.1 s/步） | **P0** |
| **P2** | 长程全连使每步 ∝ χ³·N³，24 点已不可用 | 算法 | 24 点一次撤场 5–12 h，42 点数天 | P1 |
| **R1** | iDMRG 不暴露任意 Pauli / χ（手性） | 能力 | **无法用它算本项目最重要的量** | **P0** |
| **R2** | `drive_segments` 是左端点阶梯，一阶 | 精度 | 强场 ramp 要几千段才收敛 | **P0** |
| **R3** | 无系综/噪声循环 | 能力 | 位置/速度/光移弥散只能外部手搓 | P1 |
| **R4** | `want=spectrum` + 制备末态 | **未验证** | 见 §4.4，需确认能否用上一次的末态 | P2 |

**最该先做的两件**：B1（一行条件，解锁 iDMRG 的 U(1)）和 B2（二阶收敛，等效提速 4–25×）。

---

## 1. 我用它做了什么（上下文）

在 12 点呼吸 kagome（h=0.3）上做**准绝热撤场**：乘积初态 → 指数降 δ → 读 χ、P_z、E。
需要的两条通路：

1. `tdvp-dense` + `drive_segments` + `pauli_strings`（含时）；
2. `eq_backend=idmrg` 沿 δ 扫 ξ（静态，判 CSL）。

---

## 2. 阻断性缺陷

### B1（P0）iDMRG 扇区判据过严

**现象**

```text
ValueError: charge_sector 下 P/M/X/Y 单算符破缺守恒；请改用 N/Z 或去掉扇区
```

**但我的项全是 `P_i M_j`**——把激发从 j 搬到 i，**总 Sz 严格守恒**。

**定位** `src/core/manybody/idmrg.py:130-136`

```python
bad = [t for t in spec['terms']
       if (t['kind'] == 'onsite' and t['op'] in break_ops)
       or (t['kind'] == 'pair'
           and (t['op_i'] in break_ops or t['op_j'] in break_ops))]
```

只要 pair 里出现 P/M 就拒，不看**组合**。

**影响**：去掉 `charge_sector` 后 iDMRG 在无对称性的全空间跑。实测 6 点胞、χ=64、**单点 921.8 s**，且结果不收敛（ξ=55.9、S/bond=1.856、χ 顶满 cap，而物理值应 ξ≈1.6a）。这是本项目目前**唯一有希望覆盖 CSL 宽度**的路线，被这条挡住。

**建议**：按**组合**判守恒，而不是按单算符。做法很简单——维护一张 (op_i, op_j) 的粒子数变化表：

| 组合 | Δn | 处理 |
|---|---|---|
| `N`,`Z`,`I` 任意配对 | 0 | 允许 |
| `P_i M_j` / `M_i P_j` | 0 | **允许（现在被误拒）** |
| `P_i P_j` / `M_i M_j` | ±2 | 拒绝 |
| 单站点 `P`/`M`/`X`/`Y` | ≠0 | 拒绝 |

即：`Δn(op_i) + Δn(op_j) == 0` 才放行，其中 Δn(P)=+1、Δn(M)=−1，N/Z/I 为 0。

**验证建议**：修好后 δ=0 的单点耗时应显著低于 921.8 s，且 ξ 应落到 O(1–5)a。

### B2（P0）`tdvp-dense` 的 dt 收敛是一阶

**现象**（12 点、χ=128、T=500 ns、强场 15.4 MHz、与 ED 对拍）

| dt (ns) | 步数 | max\|Δ布居\| |
|---|---|---|
| 10 | 50 | 6.19e−2 |
| 5 | 100 | 3.09e−2 |
| 2 | 250 | 1.20e−2 |

dt 减半误差减半 ⇒ **一阶**。标准两站点 TDVP（Strang 分裂 + 单站点修正）应是二阶。

**影响**：要把 χ(t) 做到 1e−3，需 dt≈0.2 ns，步数 ×25。这是“24 点不可行”的直接原因之一。

**建议**：先确认 `sweep()` 的左右扫是否**各做一次**（只做单向右扫会给出一阶）。若已是双向，则检查 1-site 修正项的 ±dt/2 符号与作用顺序。

### B3（P0）`tdvp-mpo` 与 `tdvp` 结果不一致

**现象**（12 点、χ=64、10 步、同一初态/同一静态驱动、无 Pauli）

| 后端 | 每步 | 与 `tdvp` 末态布居差 |
|---|---|---|
| `tdvp` | 0.82 s | — |
| `tdvp-mpo` | 0.60 s | **1.5e−2** |
| `tdvp-dense` | 0.019 s | 9.1e−7 |

`tdvp-dense` 能对到 9e−7，说明问题不在“TDVP 本身”，而在 `tdvp-mpo` 路径。

**建议**：用它去对 `tdvp` 与 `tdvp-dense`（后者明显更可信），最小复现见附录 A3。

### B4（P1）官方基准工具不认新后端

```text
python scripts/bench_mps.py --algorithm tdvp-dense
→ ValueError: MPS algorithm 只支持 tebd/tdvp/tdvp-single/tdvp-adaptive，得到 'tdvp-dense'
```

`--algorithm` 被原样塞进 `evolve_mps`，而新后端在 `backends.py` 里是独立条目。

**建议**：`bench_mps.py` 走 `backends.TIME_BACKENDS` 查表分派，而不是硬编码四个名字。

### B5（P2）遗留键报 `internal_error`

上一步设了 `manybody.mps_order`，接着用 `want=thermal`：

```text
internal_error: ValueError: ed does not support equilibrium options: mps_order
```

被包装成 `internal_error`，用户在 MCP 层看不到“哪两个键冲突”。

**建议**：把“某后端不接受的选项”统一走成 `ok:false / kind:config_conflict / 列出冲突键`，而不是内层异常。

### B6（P2）摘要层丢信息

- `want=thermal` 的 **`corr` 不进摘要**（artifact 里有，形状 (3,12,12) 正确）；
- `n_atom` 被拍平成 36 个数，丢失 (3,12) 的形状。

**建议**：摘要里 `corr` 至少给 `first/last`（像 E/C/F 那样）；`n_atom` 保留二维形状或按 β 分条。

---

## 3. 性能

### P1（P0）性能契约应按**实际 χ** 标定

我按官方口径测到的 43×提速（12 点 χ_max=64）：

```
tdvp 8.19 s / tdvp-mpo 6.02 s / tdvp-dense 0.19 s （10 步）
```

但那次的 **`max_chi=2`**——单激发、无纠缠，χ³ 项根本没被触发。

真实工况（4 激发 + 强场，χ 顶到 cap）：

| 场景 | χ_max | 每步 |
|---|---|---|
| 12 点 | 64（实际 2） | 0.019 s |
| **24 点** | 8 | 0.24 s |
| **24 点** | 16 | 0.66 s |
| **24 点** | 32 | 1.94 s |
| **24 点** | **64** | **8.13 s** |
| 24 点（`tdvp` TeNPy） | 64 | 19.1 s |
| 24 点（`tdvp-single`） | — | 1.05 s，但 **max_chi=1**（乘积态出发永远长不出纠缠，对本用途无效） |

**建议**：

1. 基准用例换成“有真实纠缠的态”（多激发 + 非零场）；
2. 性能契约里同时记录 **`chi_actual`**，只报 `chi_max` 会误导；
3. `scripts/bench_mps.py` 输出里加 `chi_actual`。

### P2（P1）长程全连的结构性代价

偶极模型是全连：24 点 276 对、42 点 861 对。MPO 键维 **W ≈ 2N**（24 点实测 48），两站点每步代价 ∝ χ³·W²·N = **χ³·N³**。

- 12→24 点本身 8×，叠加 χ 从 2 到 64 的 ~60×，合起来 ~480×（实测吻合）；
- 42 点 χ=256 外推 **~800 s/步**，一次 22 µs 撤场 = 数天。

**建议**（按性价比）：

1. 给 `manybody` 加一个**距离截断**选项（`r_cut`），把 1/r³ 尾巴砍掉。24 点上 r≤3a 可把 W 从 48 降到 ~10，W² 降 20 倍。需要一个配套判据（截断对 ξ、χ_bulk 的影响由调用方自查）。
2. 或者提供 **`χ_actual` 自适应**：先跑 10 步看真实 χ 需求，再决定是否值得跑全程。

---

## 4. 接口需求

### R1（P0）iDMRG 要能测任意 Pauli / 手性

现在 iDMRG 只回 `E_per_site / E_per_cell / S_per_bond / n_atom / corr_length / chi / trunc_err`。

但本项目最重要的量是 **χ = ⟨S_i·(S_j×S_k)⟩**（6 条三体 Pauli）。**没有它，iDMRG 无法回答“δ=0 是不是 CSL”**——而这正是我上一轮说“静态可行性便宜、先用 iDMRG”的前提。前提本身不成立。

**建议**：给 iDMRG 加 `pauli_strings`（或至少 `corr`），在无限 MPS 上用 `expectation_value_term` 逐条算。项在 cell 内即可（例：`[(0,'x'),(1,'y'),(2,'z')]`，不需要跨 cell 的 dx）。

### R2（P0）Ramp 的表示：现在是左端点阶梯，一阶

`drive_segments` 在每个区间取**左端点**的常数值。同一段指数 ramp（τ=1.6/J₁、12 点 ED）：

| nseg | (E−E₀)/N |
|---|---|
| 40 | 0.0566 |
| 80 | 0.0476 |
| 160 | 0.0399 |
| 320 | 0.0420 |

**nseg 从 40 加到 320 还漂 35%**，且 χ 完全不收敛。对照：我自己写的中点格式 600 步就收敛到 2%。

**建议**（任选其一，效果相同）：

1. `drive_segments` 加 `interp:"midpoint"`（取区间中点值）——最小改动，直接把阶梯误差从 O(dt) 降到 O(dt²)；
2. 或加解析规格 `{kind:"exp", delta0, tau, t_total}`，由引擎内部按 δ 的变化率自适应细分；
3. ED 路径尤其需要（ED 的“步长”就是段长，没有内部细分）。

### R3（P1）系综/噪声循环

位置（0.1/0.1/0.6 µm）、速度（0.03 µm/µs）、逐原子光移弥散（1%，一天漂到 3%）都要抽样。

**注意**：逐原子 δ 现在已经能表达（`drive_segments.delta` 长度 = N），所以**缺的只是循环与结果契约**：

```python
run_ensemble(calib, sampler, observables, n_samples, seed)
# → 每个观测量 (mean, stderr, n_valid)，必须区分“无条件”与“后选”
```

### R4（P2）`want=spectrum` 以制备末态为源

schema 里 `manybody.source` 已有 `{kind:"psi0"}`，desc 说“省略时默认用 psi0 作 |0⟩”。**这条我没实测**，需要确认：能不能把“上一次撤场的末态”作为源（R3 的边缘响应要用）。

### R5（P2）`tdvp-dense` 的能力位写清楚

它现在拒绝 `charge_sector`、`return_state`、`gamma`。其中 `return_state` 被拒意味着**拿不到末态矢量**，没法把“制备末态”喂给后续的 `want=spectrum`（R4）或再做一次驱动。建议至少给 `return_state`。

---

## 5. 优先级与预期收益

| 顺序 | 项 | 改动量 | 预期收益 |
|---|---|---|---|
| 1 | **B1** iDMRG 守恒判据 | 一行条件 | iDMRG 从“跑不动”变成“能扫”，慢 10–100× 的问题消失 |
| 2 | **B2** dt 二阶收敛 | 几行 | 达到同样精度步数 ÷4~25 |
| 3 | **R2** 中点/解析 ramp | 小 | 每个 ramp 的精度提升一个量级，段数需求大幅下降 |
| 4 | **R1** iDMRG 的 Pauli/χ | 中 | 让 iDMRG 能回答本项目的核心问题 |
| 5 | **P1** 按实际 χ 标定 | 小（工具） | 避免“快 43×”这类误导性结论 |
| 6 | **B3/B4** MPO 不一致 + 基准工具 | 小 | 可信度 |
| 7 | **R3** 系综 | 中 | 实验误差预算 |

---

## 附录 A：复现命令

环境：`D:\code-nosync\energy-level`，EL 通过 mecha 启动（改 `.py` 后必须重启）。

**A1 三后端对拍（12 点）**

```powershell
python scripts/bench_tdvp_variants_local.py --t-max 20 --dt 2 --chi 64
# → tdvp 8.19s / tdvp-mpo 6.02s(max|Δpop|=1.5e-2) / tdvp-dense 0.19s(9.1e-7)
```

**A2 dt 收敛（12 点、强场）**

```powershell
python scripts/verify_tdvp_dense_ramp.py
# τ=500ns 段、dt=10/5/2 → max|Δpops| = 6.19e-2 / 3.09e-2 / 1.20e-2
```

**A3 24 点每步成本**

```powershell
python src/ramp24.py --tau 0.5 --tmax 40 --step 4 --chi 64 --nseg 2
# → wall=81.3s / 10 步 = 8.13 s/步；total excitation=12.000000（守恒 ✓）
```

**A4 iDMRG（B1 的复现）**

```powershell
python src/idmrg_scan.py --chi 64 --deltas 0,0.1 --sweeps 8
# → 扇区被拒 → 回落无扇区：921.8s / 203.3s；ξ=55.9 / 97.5；n_cell≈0.5
```

**A5 分段阶梯（R2 的复现）**

```powershell
python src/ramp_window_scan.py --nseg 20
# τ=1.6/J1 时改 --nseg 40/80/160/320 看 (E-E0)/N 漂移
```

**A6 MCP 端到端（新接口可用性）**

```text
build_calib(kagome_h030_n12) → mode=2, solver=tdvp-dense, segments+pauli_strings, run_sim
→ 24 条三体串全返回；summary.observables.pauli_strings.items 结构正常
```

---

## 附录 B：做得好的地方（别改坏了）

- `tdvp-dense` 接受 `segments` + `pauli_strings` 后**一次跑通**，24 条三体串结果与 ED 量级一致；
- **Pauli 求值开销可忽略**：同一次撤场，不带 Pauli 139.95 s vs 带 24 条 138.44 s（1%）；
- 新的摘要契约（`mode: top/final/first_last/list` + 逐项 `items`）比我预期的好用，`pick` 投影省了很多传输；
- `want=thermal` + `eq_backend=ed` 真的给出了 `corr (nβ,n,n)`，这是我之前提的硬缺口，补上了；
- 拒绝面很干净：四个“不支持”的错误消息都直接告诉我该用哪个后端，没有静默降级。
