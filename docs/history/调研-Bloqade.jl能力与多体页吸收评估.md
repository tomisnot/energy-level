# 调研：Bloqade.jl 能算什么 · 与我们多体页的重叠/差距 · 吸收可行性

> **缘起**：用户拉取了 `external/Bloqade.jl`（QuEra 的中性原子/里德堡阵列计算引擎），认为其"蒙特卡洛算法"与我们的
> **多体页**功能有重叠，且我们对多体页性能不满意、很少用，想知道能否"把 Bloqade 吸收进来"。
> **本轮范围 = 只调查**：先把 Bloqade 能算什么钉死，再对照我们多体页现状给出重叠/差距与吸收路线的可行性判断。
> **本文不改动任何代码**；路线/优先级供后续实现会话执行。
>
> **证据基线**：逐条带 `文件:行`。我们多体页现状取自代码锚定的《物理引擎-专家手册.md》§6/§11（唯一真相源）。
> **一句话结论见 §0。**

---

## 0. 结论先行（TL;DR）

1. **Bloqade.jl 是 Julia 单仓多包引擎**（13+ 子包 + `Yao` 生态），Python 侧 `external/Bloqade.jl/python/` 只是个
   **三函数薄封装**，靠 `juliacall` 跨语言调运行时 Julia——**"吸收代码"在语言层面不成立**，要么跨语言桥接、要么只搬"算法思想"。
2. **它和我们多体页是"同一个物理对象的两套抽象"**：都是"中性原子点云 + 逐原子二/三能级 + 里德堡相互作用 + 含时驱动"的
   多体量子演化。但 Bloqade 的模型更"教科书"（Ising：Ω/Δ/φ + $C_6/r^6\,\hat n_i\hat n_j$，**RWA 已内建**），
   我们的模型更"显微"（ARC 偶极矩阵元 → 精确两体 4 维对空间 + Löwdin downfolding 的 vdW/C6、交换 J、双激发 A、$m_J$/Zeeman）。
   **两者不互相取代。**
3. **Bloqade 真正的、我们缺失的价值有三块**：① **受限里德堡子空间（blockade/RHS/PXP）**——这是它相对我们 $2^N$ 全空间 ED 的
   **性能命门**；② **QMC（SSE/worldline）求基态/热态** 与 **噪声蒙特卡洛波函数（Aquila 误差模型）**——我们多体页**完全没有**这两类
   统计/随机方法；③ **连续包络驱动 / GPU 传播 / 自适应高阶积分**。
4. **推荐**：不做跨语言桥（会把 Julia 运行时塞进我们已打包分发的 exe）；**把 ① 受限子空间 作为最高性价比的对症移植**
   （直接解掉"n≤12 才快、n≤20 封顶"的悬崖），把 ②③ 列为"新能力候选"逐个评估。桥接仅可作**离线基准/对照**外挂，不进主产品路径。
   详见 §7。

---

## 1. Bloqade.jl 架构与依赖现实（决定"怎么吸收"）

### 1.1 它是 Julia monorepo
- 顶层 `src/Bloqade.jl:6-12` 纯 `@reexport` 聚合：`Yao`、`BloqadeMIS`、`BloqadeODE`、`BloqadeExpr`、`BloqadeKrylov`、
  `YaoSubspaceArrayReg`、`BloqadeWaveforms`；`Bloqade.jl:16-42` 再导出 `BloqadeLattices` 的格子生成器。
- `lib/` 下 **13 个子包**（各带 `Project.toml`）：`BloqadeCUDA`（GPU）、`BloqadeDormandPrince`（DP8 积分器）、
  `BloqadeExpr`（哈密顿表达式）、`BloqadeGates`（预置门脉冲序列）、`BloqadeKrylov`（expmv）、`BloqadeLattices`（晶格）、
  `BloqadeMIS`（最大独立集）、`BloqadeNoisy`（噪声）、`BloqadeODE`（含时薛定谔）、`BloqadeQMC`（量子蒙特卡洛）、
  `BloqadeSchema`（Aquila 硬件 schema）、`BloqadeWaveforms`（波形）、`YaoSubspaceArrayReg`（子空间寄存器）。
- 依赖 `Project.toml:5-35`：`Yao`、`OrdinaryDiffEq`/`DifferentialEquations`、`KrylovKit`、`Graphs`、`BitBasis`、`CUDA`、`FFTW`、`Zygote` 等
  ——**一整套 Julia 科学生态**，非纯 Python 可 import 的东西。

### 1.2 官方对"Python 版"的定位（关键情报）
- `README.md:16`："Bloqade.jl 已达稳定、近期不大改；推荐改用 **bloqade-python**（QuEra 新版 Python SDK）访问 Aquila"。
- `README.md:55`：计划把 **Bloqade.jl 做成 bloqade-python 背后的高性能仿真后端**。
  → 也就是说：**"Python 前端 + Julia 高性能后端"是官方路线**，但它要的是**他们自己的 bloqade-python**，不是让我们把它缝进现有 Python 项目。

### 1.3 `python/` 封装 = 三函数垫片（几乎没用）
- `python/bloqade/__init__.py:1-3`：`import juliacall` + `using Bloqade`（**运行时必须有 Julia + 已安装 Bloqade**）。
- `python/bloqade/julia.py:22-35`：仅暴露 `rydberg_h`、`unit_disk_graph`、`mis_postprocessing` 三个函数，其余全靠用户在 Julia REPL 里做。
- **含义**：这条 Python 通道**不是**给我们"用 Python API 调引擎"的完整 SDK，只是 MIS 场景的最小胶水。想调全能力就得写 Julia，
  且要在打包产物里附带 Julia 运行时 + 编译延迟——**对我们"双击 exe 给无 Python 用户"的分发模型（见 README 旧版启动/PyInstaller）是灾难**。

---

## 2. Bloqade 能算什么 —— 能力矩阵（调查主体）

### 2.1 物理模型
| 模型 | 哈密顿量 | 证据 | 备注 |
|---|---|---|---|
| **二能级 Rydberg Ising** | $\sum_j\frac{\Omega_j}{2}(|g_j\rangle\langle r_j|+\text{h.c.})-\sum_j\Delta_j\hat n_j+\sum_{j<k}\frac{C_6}{r_{jk}^6}\hat n_j\hat n_k$ | `hamiltonians.md:12-15` | **RWA 已内建**（驱动项就是旋转波形式，无反旋项）；默认 $C_6=2\pi\times862690$ MHz·µm⁶（⁸⁷Rb 70S）|
| 全局 / **位依赖**驱动 | Ω、Δ、φ 可为标量、逐格矢量、或 waveform(t) | `hamiltonians.md:78-96` | 支持逐原子 Ω/Δ（与我们同构）|
| **三能级**（hyperfine+Rydberg）| 两束脉冲：|0⟩↔|1⟩（hf）、|1⟩↔|r⟩ | `3-level.md:15-17,39` | 通用量子计算载体 |
| **量子门 / 脉冲序列** | 单比特门、5-pulse CZ、**Levine-Pichler** 门 | `3-level.md:52-195`；`BloqadeGates` | 门基(gate-based)路线，**与我们无关**（我们不做 QC 门编译）|
| **PXP / 约束局域哈密顿** | 子空间投影掉长程尾，只留近邻约束 | `subspace.md:115-133` | 量子多体物理/scar 研究用 |
| 晶格规范理论(LGT)、QAOA | 见 examples/4.LGT、6.MWIS | `examples/`（目录）| 专题研究，非通用 |

> **与我们模型的关键差**：Bloqade 相互作用 = 单一 $C_6/r^6\,\hat n_i\hat n_j$（Ising 密度-密度）。我们 = **ARC 偶极矩阵元**
> 严格算的两体 4 维对空间，含**交换 J（σ⁺σ⁻）、双激发 A（σ⁺σ⁺）、二阶 Löwdin downfolding**、$m_J$/Zeeman/B 场/光-微波 Floquet
> （《物理引擎-专家手册》§4/§6）。**Bloqade 表达不了我们的精细结构多能级与光微波混合**；反之我们的多体页也没做 Bloqade 的门/LGT。

### 2.2 数值后端（"凭什么快"）
| 后端 | 干什么 | 证据 | 我们有无 |
|---|---|---|---|
| **ODE（DifferentialEquations.jl）** | 含时薛定谔实时演化；默认 DP8，推荐 Vern6/7/8、`AutoVern9(Rodas5)` 自动刚性切换；**自适应步长** | `emulation.md:19-20,102,111-130` | 有（自研 DOP853，《专家手册》§3.3）|
| **Krylov（expmv）** | 时段内定常 H 的 $e^{-i\Delta t H}$；对 piecewise-constant/QAOA 更高效 | `emulation.md:133-165`；`BloqadeKrylov` | 部分（多体用 `expm_multiply` 逐区间，《专家手册》§6；单体参考锚有）|
| **受限子空间 SubspaceArrayReg / BitSubspace** | **把 $2^N$ 全空间裁成"合法构型基"**：位运算生成允许 bitstring、压缩索引 | `subspace.md:1-52`；`YaoSubspaceArrayReg/src/type.jl`；`BloqadeMIS/src/bsubspace.jl` | **❌ 无**（我们恒在 $2^N$ 全空间，《专家手册》§6：`dim=2^N` 是唯一主约束）|
| **GPU（CUDA）** | 态矢演化上卡 | `BloqadeCUDA`；`docs/cuda.md`；`multithreading.md` | 仅 sambe 两体**研究原型**（torch，《专家手册》§3.5），多体无 |

> **子空间是命门**：3×3 方格（9 原子）$R_s=5.2$µm 时，合法态 **63 « 全空间 512**（`subspace.md:42`）。阻塞越强、维度坍缩越狠，
> 可达 N 从 ~12 直接推到几十。**这正是我们对多体页性能不满意的根因**（见 §6）。

### 2.3 统计 / 优化 / 开放系统（三类我们完全没有的方法）
| 模块 | 本质 | 能产出 | 证据 |
|---|---|---|---|
| **BloqadeQMC** | **SSE / worldline 量子蒙特卡洛**（cluster 更新 + 抖动队列）：TFIM / LTFIM / Rydberg 的虚时采样 | **基态 + 有限温混态**（大 N 免 $2^N$），磁化/关联/纠缠熵/nematic；带 autocorrelation、jackknife/bootstrap 误差分析 | `BloqadeQMC.jl:31-47`；`ising/groundstate.jl`、`mixedstate.jl`、`cluster.jl` |
| **BloqadeMIS** | 最大独立集：绝热量子算法 + **可微**（`gibbs_loss` 配 Zygote 梯度）变分调脉冲；`mis_postprocessing` 把采样比特串规约到独立集 | MIS 问题的量子优化求解（实验演示达 289 qubit） | `mis.md:7-25`；`loss.jl`；`BloqadeMIS.jl:15-29` |
| **BloqadeNoisy** | **蒙特卡洛波函数**（量子跳 + 相干抖动 + 读出误差）：Aquila 误差模型 | 含噪演化轨迹集 → 均值 + 误差；`measure_noisy`/`expectation_value_noisy` | `noise_models.jl:4-19`（dephasing/relaxation/site·detuning·Rabi jitter/p01·p10）；`BloqadeNoisy.jl:22-33`；`problem.jl`（492 行）|

> 用户说的"蒙特卡洛"最可能指 **Noisy 的 MC 波函数** 或 **QMC**。二者都是**随机/统计**方法，与我们的**确定性幺正/非厄米**演化
> 是**互补**关系，不是同类替代——我们多体页目前**只有确定性 ED**（《专家手册》§6 三档：稠密 eigh / 稀疏 expm / 守恒扇区分块）。

### 2.4 可观测量与硬件接口
- 观测：`rydberg_density`、`rydberg_corr`、`bitstring_hist`、`get_average_rydberg_densities`（`src/Bloqade.jl:43`、`observables.jl`）——
  与我们多体观测（逐原子激发/关联 $C_{ij}$/纠缠熵，《专家手册》§6 line 279）**基本一一对应**。
- 硬件：`BloqadeSchema` 提供 **Aquila 设备能力表**（256 qubit、75×76 µm、Ω≤15.8 rad/µs、|Δ|≤125 rad/µs、slew-rate、4µs 时长等，
  `capabilities.md:18-58`）+ 任务提交。**这块是"上真机"，与我们纯仿真定位无关**（避免误吸收）。
- 波形/模拟编程：`BloqadeWaveforms`（piecewise_linear/constant + 解析函数 + 时长/斜率约束，`waveform.md`）——我们多体页**缺连续包络**，这块是**可参照的接口设计**。

---

## 3. 我们的多体页现在算什么（对照基线）

摘自《物理引擎-专家手册.md》§6/§11（代码锚定）：
- **模型**：任意点云 + 逐原子二能级 (g,r) + 逐原子 Ω/δ/Γ；相互作用 = `static_pair` **全精确 4 维对空间**（复用 ARC 偶极，不硬编码 C6）
  + 一阶（标准序 $\langle p|V|q\rangle$，保留 (θ,φ) 复相位）+ 二阶 **Löwdin downfolding**（vdW/C6 从伙伴态涌现，`|Δn|≤n_window=3` 截断）；
  项开关 I1 密度-密度 / I2 交换 J / I3 双激发 A（§11.1，`assemble.py`）。
- **后端**：$2^N$ **全空间 ED**——dim≤`_MB_DENSE_MAX=4096`（n≤12）且厄米走稠密 `eigh` 全时刻向量化；否则稀疏 `expm_multiply` **逐区间步进**；
  **仅当无驱动+无双激发 A**（粒子数守恒）才扇区分块对角化（§6 line 273）。维度闸 `_MB_MAX_DIM=1<<20`（n≤20）；全布居物化闸 65536（n≤16）。
- **驱动**：逐原子常数 Ω/δ 或**分段常数** `drive_segments`；**无高斯/方波包络、无 chirp、无任意波形**（§11.2-3）。
- **耗散**：逐原子非厄米 γ（布居按 $e^{-\Gamma t}$）；L3/场致损耗未接多体（§8.9/§11）。
- **缺**：`tebd`/MPS **只 raise**（§11.2-1）；无受限子空间；无 GPU；无 QMC；无 MC 噪声；`entropy_at_cut` 越界静默 0（§11.3①）。
- **官方自评"能不能给人用"的 P1 缺口**（§11.4）：① 连续包络驱动 ② 熵越界改 None ③ |m|≤j 校验 ④ 成本实测标定。**连它自己都第一条就点"性能/能力不足"。**

---

## 4. 重叠 vs 差距（一页对照）

| 维度 | Bloqade | 我们多体页 | 判定 |
|---|---|---|---|
| 语言/栈 | Julia + Yao/DiffEq/CUDA | Python + numpy/scipy/numba(+torch 原型) | **不可直接搬代码** |
| 物理抽象 | 教科书 Ising（C6/r^6+RWA），另有 3-level/门 | ARC 显微（mJ/Zeeman/J/A/downfolding），RWA-free 单两体引擎 | 各有所长，**不互相取代** |
| 多体实时演化 | ODE/Krylov **＋受限子空间** | 全空间 ED（无子空间） | **Bloqade 胜在可扩展性** |
| 大 N（n>15） | 子空间/QMC 可达 | 悬崖（n≤12 快 / n≤20 封顶） | **我们最弱、Bloqade 最强处** |
| GPU | 有 | 无（原型除外） | Bloqade 领先 |
| 连续包络脉冲 | 有波形层 | 无（只常数/分段） | Bloqade 领先 |
| 基态/热态 | **QMC** + （ED 需借 KrylovKit） | 无 QMC；基态靠 `static_pair.pair_eigen`/downfolding（小系统） | **互补**（我们给的是显微有效模型，非采样）|
| 噪声开放系统 | **MC 波函数**（Aquila 模型） | 确定性非厄米 γ；无 MC 噪声 | **互补**（我们已有 L2/L4 场致损耗，《专家手册》§5，可对接）|
| 门基 QC / 上真机 | 强 | 无关 | **不吸收**（我们不做门编译/Aquila 提交）|
| 光/微波 Floquet、AC-Stark、两体 Sambe | 无 | 强 | **我们独有，别被带偏** |

---

## 5. 单位/约定差异（移植必须先建的换算层）
- Bloqade：Ω/Δ 用 **rad/µs**，时间 **µs**，$C_6=2\pi\times862690$（MHz·µm⁶ 名义）/ 设备表 5.42×10⁶ rad/µs·µm⁶（`capabilities.md:43`）。
- 我们：能量/耦合/Rabi 用 **MHz（周频）**，时间 **ns**，$C=2\pi\times10^{-3}$ rad/(MHz·ns)，$V_{dd}$ 由 ARC 偶极现算（《专家手册》§4/§7.5 单位约定）。
- → 距离尺度（µm）与"⁸⁷Rb 里德堡"一致，数量级可对齐，**可作交叉验证锚**；但 rad/µs↔MHz/ns 的换算 + "Bloqade 的 C6 常数 vs 我们 ARC 现算"
  的差异，是任何"对照基准"落地前要先核对清楚的第一件事。

---

## 6. "我们多体页为什么慢"——根因 vs Bloqade 快在哪
1. **根因（我们）**：恒在 $2^N$ **全空间**做 ED；有全局驱动 Ω 时**粒子数不守恒 → 扇区分块失效 → 完整 $2^N$**（《专家手册》§6 line 273/277）。
   稠密快路径 n≤12、硬上限 n≤20，稀疏逐区间 `expm_multiply` 无自适应误差控制。
2. **Bloqade 的三个加速杠杆**：① **受限子空间**（阻塞把维度指数级裁掉，`subspace.md:42` 的 512→63）；② **自适应高阶 ODE/Krylov**；
   ③ **GPU**。其中 ① 是**结构性**收益（改变可及 N），且**纯 Python 可实现**（位运算枚举合法 bitstring + 压缩索引，无外部依赖）。
3. → **对症结论**：想缓解"性能不满意"，**移植"受限子空间表征"这一思想**比换积分器/上 GPU 收益更大、风险更低，且不引 Julia。

---

## 7. 吸收路线评估（供决策，未开工）

### 路线 A — Python↔Julia 桥（直接调 Bloqade）
- **代价**：硬依赖运行时 Julia + 装好 Bloqade（`__init__.py:1-3`）；打包 exe 附带 Julia 编译链——与我们"给无 Python 用户双击"分发模型冲突；
  跨语言搬运大态矢；**模型不对齐**（Bloqade 只有 C6+RWA，喂不进我们的 ARC/mJ/AC-Stark/光微波 Floquet）。
- **判定**：❌ 不进主产品路径。✅ **可作离线"科研对照/基准"外挂**（可选依赖、脚本级）：在我们多体页退化到 RHS/PXP 极限时，
  用它验证数值一致（当"第三方权威锚"，类似我们对 `evolve_reference` 的用法）。

### 路线 B — 移植算法**思想**到 Python/numba 栈（不搬代码，搬 idea）★推荐
- **B1 受限里德堡子空间（RHS）表征** ★最高性价比、最对症：枚举阻塞半径 $R_s$ 内的合法 bitstring 作基 + 位运算压缩索引
  （参照 `YaoSubspaceArrayReg/type.jl`、`bsubspace.jl` 的 `BitSubspace`），把哈密顿在子空间内组装/传播。
  **收益**：解掉 n≤12 悬崖，阻塞强时可及 N 推到 20–30+；**验收**：同一 PXP/RHS 场景，子空间维数 ≪ $2^N$、与全空间 ED 结果在
  投影误差内一致、可达 N 显著上移。**定位**：新增 `src/core/manybody/` 下子空间基 + 传播；复用 `assemble.calibrate_pair` 的矩阵元。
- **B2 连续包络驱动**：补高斯/方波/chirp（我们多体只常数/分段，《专家手册》§11.4 P1-1；Bloqade `waveform.md` 是好的接口范式）。
- **B3 Krylov/自适应 expmv**：给多体稀疏传播加自适应（现 `expm_multiply` 逐区间无误差控制，《专家手册》§6 line 274）。
- **B4 GPU**：把已有 sambe 两体 torch 原型思路推广到多体传播（可选，收益随 N 才显现）。
- **B5 蒙特卡洛波函数（MC 噪声）**：Aquila 型跳+抖动+读出误差（`noise_models.jl`）——是**新能力**，与我们 §5 场致损失/L2/L4/非厄米 γ
  天然互补、纯 Python 可移植；但属"加功能"非"提性能"，优先级排在 B1/B2 后。
- **B6 QMC（SSE/worldline 基态/热态）**：**全新范式**，专治"大 N 基态/热态"，但**sign problem + 只给统计量 + 实时幺正做不了**，
  与我们"含时演化"主定位偏离；**建议单列专题评估**，不进近期路线。

### 路线 C — 只做能力对标（划边界，防误吸收）
- 明确 **不吸收**：门基 QC（CZ/Levine-Pichler）、LGT、Aquila 真机提交、gate-based QAOA（我们不做 QC 编译）。
- 明确 **我们独有、勿被替换**：光/微波 Floquet、AC-Stark、两体 Sambe、ARC 显微相互作用、$m_J$/Zeeman。

### 建议优先级（下一步交给实现会话时可细化为规划文档）
1. **B1 受限子空间**（对症"性能不满意"，纯 Python，风险可控）→ 先。
2. **B2 连续包络驱动**（对症"没怎么用=能力不足"，且多体做真实脉冲的前置）→ 次。
3. **路线 A 作为可选基准外挂**（对照/回归用，非产品路径）→ 平行小成本验证。
4. B5 MC 噪声、B3 自适应 expmv、B4 GPU、B6 QMC → 各自单独立项评估。

---

## 8. 待用户拍板（影响下一轮规划方向）
- **"吸收"的真实诉求**是 **(a) 提性能/扩可及 N**（→ 直奔 B1），**(b) 补能力**（连续驱动/噪声/基态采样 → B2/B5/B6），
  还是 **(c) 想要一个权威对照来校验我们多体页**（→ 路线 A 基准外挂）？三者代价与产物差别很大。
- 是否接受"只搬思想、用 Python/numba 重写"这一前提（拒绝引入 Julia 运行时）？
- 多体页要服务的**典型规模**（几十？上百原子？）与**是否只要基态/也可要实时动力学**——决定 B1 vs B6 的取舍。

### 8.1 决策快照（用户 2026-10 回复，供下一轮参照）
- **诉求**：本轮**只到调研**，暂不立项实现/规划。
- **技术前提**：用户明确 **可接受 Julia 桥**（不排斥在产物中引入 Julia 运行时）。
  → 这**放宽了路线 A 的定位**：§7 里"跨语言桥"从"仅离线对照外挂"升级为**可纳入正式候选**——好处是能直调 Bloqade 全能力、
  保真度高；代价仍是**分发附带 Julia 运行时 + 编译延迟**，与我们"双击 exe 给无 Python 用户"的模型需另行权衡（可做成
  "首次运行拉取/可选组件"缓解）。纯 Python 移植（路线 B）依旧成立，只是不再是**唯一**选择。
- **待下一轮再定的仍是大方向**：性能(B1) / 能力(B2·B5·B6) / 桥接保真(A)，三选一或组合。

---

## 附：证据清单（文件:行）
- 架构/依赖：`src/Bloqade.jl:6-42`、`Project.toml:5-35`、`python/bloqade/{__init__.py:1-3,julia.py:22-35,types.py}`、`README.md:16,55`。
- 模型：`docs/src/hamiltonians.md:12-15,78-96,205-219`、`3-level.md:15-17,52-195`、`subspace.md:115-133`。
- 后端/子空间：`docs/src/subspace.md:1-52`、`emulation.md:19-20,102,111-165`、`lib/YaoSubspaceArrayReg/src/type.jl`、`lib/BloqadeMIS/src/bsubspace.jl`、`lib/BloqadeMIS/src/subspace.jl`。
- 统计/噪声：`lib/BloqadeQMC/src/BloqadeQMC.jl:31-47`、`lib/BloqadeMIS/{loss.jl,BloqadeMIS.jl}`、`docs/src/mis.md:7-25`、`lib/BloqadeNoisy/{BloqadeNoisy.jl:22-33,noise_models.jl:4-19,problem.jl}`。
- 观测/硬件/波形：`src/observables.jl`、`docs/src/capabilities.md:18-58`、`docs/src/waveform.md`、`docs/src/schema.md`。
- 我们多体页现状：《物理引擎-专家手册.md》§1（分派）、§4（V_dd/downfolding）、§5（耗散/场致损失）、§6（多体 ED/维度闸）、§8（缺陷）、§11（完善度评估与 P1 缺口）。

*本文为调研快照；结论若与后续代码演进不符，以《物理引擎-专家手册.md》（代码锚定）为准。*
