# 方案：求解动词收敛——扇区切片与参数 ramp 纳入均衡态引擎

> 状态：**待审查**（2026-10-02，Rydberg FCI 论文复现任务的架构收尾）
> 审查人确认后才动 `src/core/`；当前只有 `equilibrium.ground_state` 已加 `v0/evals`（见 §4.1，向后兼容、独立可回退）。

## 1. 背景与问题

arXiv:2202.00699（Rydberg FCI）复现任务中，为三能级 V 型里德堡阵列新建了模型装配模块
`src/core/manybody/rydberg_fci.py`。该任务的四类计算（单粒子能带、扇区谱、E(N) 标定、
绝热 ramp）暴露了求解层的碎片化：

| 求解动词 | 现有实现 | 论文脚本里的重复实现 |
|---|---|---|
| 全空间最低本征对 | `equilibrium.ground_state`（eigh/Lanczos，任意 2^N 厄米 H） | `rydberg_fci.low_eigenpairs`（同质，重复） |
| 固定激励数扇区内的基态 | **无** | 脚本本地 `sector_lowest`（切片 + eigsh） |
| 参数 ramp 逐点瞬时基态（warm-start） | **无** | 脚本本地 `_ramp`（LinearOperator + 循环） |
| E(N) → 化学势边界标定 | **无** | 脚本本地 `calibrate_mu`（diff + 平台中点公式） |

> 前 3 行是**收进引擎**的通用动词；最后一行按 §3 裁决 5 **留在脚本**——它含
> 论文端点策略（offset 负裕量、平台中点取法），换篇论文就变。

即：模型装配层 ×2（`assemble` 二能级口径、`rydberg_fci` 三能级口径——后者有正当物理理由），
求解动词层也 ×2.5 份。不收敛的话，下一个模型（Bose-Hubbard / 1+1d U(1)）会带来第三份拷贝。

## 2. 目标架构（北极星）

**求解动词一个目录（`equilibrium` + `eq_route`），模型装配做可插拔后端。**

> "一个目录"只对**通用**求解动词成立。什么是通用，判据见 §3 裁决 5：换一篇论文
> 会不会变。扫描协议（`sweep_profile`）、端点标定（`calibrate_mu`）会变 ⇒ 归论文侧。

```
UI / runspec ──► eq_engine ──► eq_route.solve_equilibrium   ← 求解编排（路由 + 扇区 + ramp）
                                     │
                     ┌───────────────┼────────────────┐
                     ▼               ▼                ▼
               equilibrium      dmrg(TeNPy)        kpm            ← 算法库
                     ▲
        ┌────────────┴────────────┐
   assemble.build_H_sparse   rydberg_fci.build_H_*      ← 模型后端（可插拔）
   （二能级 calib 口径）      （三能级 V 型口径）
```

## 3. 设计裁决（及理由）

1. **扇区切片不是新算法，是执行模式**——内核仍是 eigh/eigsh，只是输入空间换成固定 N 子块。
   所以不新增 backend，而是作为 `want='gs'` 的约束参数进路由层。
   塞进算法层是错的：DMRG 有自己的等价物（U(1) charge）、KPM 不适用扇区——只有路由层
   知道如何按 backend 正确映射。
2. **ramp 是编排不是算法**——对一串 H(τ) 依次调 `ground_state` + 传递 warm-start 初值。
   warm-start 对结果零影响（只影响速度），不需要用户决策。
3. **扇区自动判定、ramp 显式意图**（复刻体系既有哲学 `use_sectors='auto'` / `backend='auto'`）：
   - 扇区的可机检判据 = 激发数守恒（`engine_ed._is_excitation_conserving` 同款思路）；
     守恒校验不通过时**报错**（不许静默退化为全空间——那是把"算错"伪装成"没约束"）。
   - ramp/标定是用户意图（"我要扫参数"），UI 显式呈现；系统不猜。
4. **µ 标定是 ramp 的前置步骤**（先扫 E(N) 定 δ 端点），不是独立求解动词。
5. **【2026-10-02 晚裁决，覆盖本方案 §4.2 第 4/5 条】扫描协议不是求解动词**——
   "独立求解动词"还不够，**与具体论文无关**才是通用引擎的入场券。据此把
   `sweep_profile`（三段式波形）与 `calibrate_mu`（δ 端点取法）**摘出 `eq_route`**：
   - `sweep_profile` 的三段划分（Ω 升-平台-降、δ 线性扫）是论文 Fig.4(a) 的做法，
     世界上没有唯一的 ramp 波形，通用引擎不该持有其一；
   - `calibrate_mu` 的 `delta0 = µ_c(0→1) − offset`（offset=0.3 是论文给的负裕量）
     与 `delta1 = 平台中央` 同样是论文的端点策略。
   - 判据：**换一篇论文这套东西会不会变**？会 → 论文侧；不会 → 引擎侧。
     按此，`want='gs' + sector=int`、`want='ramp'` 留在引擎（前者是子空间约束、
     后者是"一串 H 依次求基态"，都不含论文知识），波形与端点取法归论文侧。
   - 去向：`sweep_profile` → `rydberg_fci.py`（零依赖纯函数，可随该模块被
     spec 独立加载）；`calibrate_mu` → `scripts/sim_fci_rydberg_2202p00699.py`
     本地（内部循环调通用动词 `solve_equilibrium(want='gs', sector=N)`）。

## 4. 改动清单（逐文件）

### 4.1 `equilibrium.py`（算法层，唯一改动）——**已实施**
- `ground_state(H, dense_max=2048, seed=0, k=2, v0=None)`：
  - 新增 `v0`：透传给 `eigsh`（Lanczos 分支）；`None` 时保持现行为（固定种子自动造）。
    dense 分支忽略（eigh 无初值概念）。
  - 新增 `LinearOperator` 支持：`sp.issparse` 不中且是 LinearOperator 时直接进 eigsh 分支
    （大维 ramp 不落 CSR）。
  - 返回 dict **新增键** `evals`（升序前 k 个本征值数组）——`E0/gap` 语义不变，向后兼容。

### 4.2 `eq_route.py`（路由编排层，主要改动）
1. **`solve_equilibrium` 的 system 解耦模型后端**：
   - `system['H']`：稀疏/稠密/LinearOperator 直传（优先于 `calib` 路径）；
   - `system['H_builder']`：`callable(Om, delta) -> 可eigsh对象`（ramp 用；标量或逐原子
     Om/delta 均由模型侧决定）；
   - `system['sector_labels']`：1 维逐基矢扇区号（int 数组；扇区模式必需）。
   - `n_sites` 推断：无 `calib` 时从 `sector_labels`/`H` 维度取，或要求显式给。
2. **`want='gs'` 新增 `sector=None|int` 参数**：
   - `sector=None`：现行为（全空间）。
   - `sector=int`：`idx = where(sector_labels==sector)` → 子空间矩阵 → 按**子空间维数**
     （而非全维）走 ed/lanczos → 返回同构 Result（E0/gap/evals/vec 均为子空间口径），
     另带 `sector=..., dim_sub=...`。**守恒校验**：抽样检查非零元行列扇区号一致
     （`H_ij≠0 ⇒ labels[i]==labels[j]`，抽 max(1000, 1%) 个非零元）——不过即 raise。
   - dmrg 后端遇 `sector=int` → raise（TeNPy charge 等价能力后续接，不静默）。
3. **新增 `want='ramp'`**：`ramp=dict(Om_t, de_t, n_tot=None, L=None, v0_warm=True)`，
   其中 `Om_t/de_t` 是逐 τ 的驱动序列（与 `sweep_profile` 输出同构）、`n_tot` 为 1 维
   逐基矢总激发数对角算符（H_builder 口径不含 δ 项时用于补 −δ·n_tot）、`L` 为密度归一
   的格点数。循环：`H_i = system['H_builder'](Om_t[i], de_t[i])` → `ground_state(H_i,
   k=2, v0=上一点 vec)`。返回 `dict(taus, gaps(E1−E0), ns(⟨N_tot⟩/L), Om_t, de_t,
   backend)`。**大维自动走 LinearOperator**：模型侧可给
   `rydberg_fci.ramp_operator(H0, B, n_tot, Om, delta)` 产物（见 4.3）。
4. ~~新增模块级 `sweep_profile`~~ **已撤销**（见 §3 裁决 5）：三段式波形归
   `rydberg_fci`（论文协议，零依赖纯函数）。
5. ~~新增模块级 `calibrate_mu`~~ **已撤销**（见 §3 裁决 5）：δ 端点标定归论文脚本，
   由脚本循环调本门面的通用动词 `want='gs', sector=N` 自行合成。

   > 撤销理由摘要：两者是**论文实验协议**而非求解动词。§4.2 实际落地的只有上面
   > 第 1–3 条（模型后端解耦、`sector` 参数、`want='ramp'`），这三项才是与模型/
   > 协议无关的通用能力。引擎侧因此保持：`__all__ = ['route_equilibrium',
   > 'solve_equilibrium']`。

### 4.3 `rydberg_fci.py`（模型层，收缩为纯装配）
- **删除**：`low_eigenpairs`、`sector_lowest`、`adiabatic_ramp`（求解动词归 §4.1/§4.2；
  这三个是 2026-10-02 上午误加的临时沉淀，无调用方，直接删）。
- **`sweep_profile` 留在/回归本模块**（按 §3 裁决 5）：零依赖纯函数，与 spec 独立
  加载兼容；`__all__` 同时清掉从未定义的 `low_eigenpairs`（它曾让
  `from rydberg_fci import *` 直接 AttributeError）。
- **新增**：`ramp_operator(H0, B, n_tot, Om, delta) -> LinearOperator`——模型侧的
  H(τ)=H0+Ω·B−δ·N_tot 组装（ramp 编排的模型接口，纯构造无求解）。
- **保留**：几何（honeycomb_sites/honeycomb_cluster/pair_shells）、基矢与装配
  （build_states_full/build_H_full/build_H_eff）、单粒子工具（bloch_H/band_structure/
  kpath_KGM/chern_fukui/band_metrics）、site_density、PAPER/EFFECTIVE 参数集。

### 4.4 `checks/`（回归判据）
1. `checks/fci_rydberg_selfcheck.py`：G 组两处 `F.low_eigenpairs` 冒烟 →
   `equilibrium.ground_state`。`sweep_profile` 判据**不动**（按 §3 裁决 5 它本就
   归 `rydberg_fci`，无需迁移）。其余 25 判据不动（物理数值判据与求解层无关）。
2. **新增 `checks/eq_sector_ramp_check.py`**（判据全部对照手工构造/解析值）：
   - `ground_state` v0 确定性（同 H 同 v0 两次 → E0 位级一致）；`evals` 键与手工稠密谱一致；
   - `system['H']` 直传 == calib 路径（小二能级模型双口径对照）；
   - 扇区切片：L=4 三能级全模型（dim=81）sector=1/2 的 E0/gap/evals == 手工切片稠密谱；
   - 守恒校验：往扇区 H 里掺跨扇区项 → `sector=int` 必须 raise；
   - `want='ramp'`：L=4 小 ramp 3 τ 点，逐点 E0 == 直接构造 H(τ) 后 `ground_state` 的
     E0（atol 1e-10）；`ns` == ⟨N_tot⟩/L 手工值；v0_warm 开/关结果一致；
   - µ 标定链路（E 组）：L=4 全模型 n_target=1, n_until=2 → **用通用动词
     `solve_equilibrium(want='gs', sector=N)` 逐扇区取 E(N)**，E0 对照手工切片
     稠密谱、`mu_c==diff(E)`、端点公式复核。不测专用入口（`calibrate_mu` 已归
     论文脚本），钉的是"通用动词组合能复现这条标定链"。

### 4.5 论文脚本 `scripts/sim_fci_rydberg_2202p00699.py`（切换调用方）
- 删本地 `sector_lowest` / `_ramp` / `_L_of_dim`；
- fig1d：装配后走**本地 `calibrate_mu`**（按 §3 裁决 5，端点取法是论文协议，
  归脚本；其内部逐扇区调 `eq_route.solve_equilibrium(want='gs', sector=N)`）；
- fig2b：走 `solve_equilibrium(want='gs', sector=N)`；
- fig4：走 `want='ramp'`（有效模型与全模型都是 `H_builder` 闭包 + `ramp_operator`）；
- 导入路径：`venv` 补装 `numba` 后改常规 `from src.core.manybody import eq_route,
  equilibrium`（摆脱对 `rydberg_fci` 的 spec_from_file_location 旁路——根因是
  `src/core/__init__.py` 连带 `evolve_twofield → coupling → numba`；装 numba 是
  打通整个包的正常导入，比继续绕路更对症）。

### 4.6 `cost_mb.py`（一行扩展，可后置）
`estimate_mb_cost` 增加 ramp 口径：总成本 ≈ n_tau × 单点成本（复用现有 ed_feasible）。

## 5. 明确不改的

- `engine_ed.py`（时间演化）、`dmrg.py`、`kpm.py` 算法本体；
- `runspec_mb.py` / `schema_mb.py` / `eq_engine.py`（UI 接入是下一步薄层，本方案只到
  `eq_route` 这层可测内核）；
- 四张图的数据与绘图逻辑（只换求解调用，物理与画图代码不动）。

## 6. 兼容性与回归判据

- `ground_state` 新增键/参数均向后兼容（旧调用点零改动）；
- `solve_equilibrium` 旧签名不变，新能力全部走新增可选参数/新 want 值；
- `rydberg_fci` 删除的 3 个函数（`low_eigenpairs`/`sector_lowest`/`adiabatic_ramp`）
  均无调用方（已 grep 确认）；`sweep_profile` 有调用方且**不迁移**（按 §3 裁决 5
  留在本模块），调用方无需改动；`calibrate_mu` 的调用方改为脚本本地函数；
- **数值回归判据**：切换后论文四图全部缓存命中重算，与既有缓存对照
  （fig1c/fig1d/fig2b atol 1e-8；fig4 有效模型 ramp 逐点 atol 1e-6）。
  已知允许差异源：Lanczos 的 v0 种子约定不同（脚本旧版实数 v0 vs equilibrium 复数 v0），
  本征值差异量级 ~1e-10，判据已覆盖。

## 7. 测试计划（先短后长，用户裁决）

1. **快**：`checks/eq_sector_ramp_check.py` + `fci_rydberg_selfcheck.py`（< 3 min）；
2. **快**：论文脚本 `--skip-full-ramp` 重算 fig1c/fig1d/fig2b/fig4-有效模型
   （约 5 min，其中 20 簇谱读缓存）——对缓存数值；
3. **慢**：全模型 ramp（69 万维 × 49 τ 点，~55 min）后台重跑——对缓存数值后出终版 fig4。

## 8. 风险与回退

- 风险 1：`ground_state` 的 maxiter 约定（`dim*20`）与旧脚本（20000 固定）不同，
  大维扇区可能收敛行为略异 → 判据 atol 已覆盖；若个别点不收敛，`ramp` 编排层做
  一次 tol=1e-6 重试（沿用旧脚本策略）。
- 风险 2：numba 安装失败（py3.13 wheel 可用性）→ 回退方案：继续按路径加载，
  为 `equilibrium/observables/kpm/assemble` 手工注册伪包骨架（约 30 行加载器），
  仅论文脚本使用；venv 隔离环境无污染。
- 风险 3：`solve_equilibrium` 变复杂后回归破坏 UI 链 → 本方案不触碰
  `eq_engine`/`runspec` 分派点（docstring 声明的 runspec_contract 不动），
  全部新能力走新增可选参数。

## 9. 给审查人的问题清单

1. `sector` 放 `solve_equilibrium` 参数（方案）vs 独立 `solve_sector` 入口——前者
   让"扇区"成为 gs 的修饰，后者动词更扁平。倾向前者（扇区与 backend 正交）。
2. ~~`sweep_profile` 挪 `eq_route` 且去默认值~~ → **裁决：不同意**（2026-10-02 晚）。
   去默认值这一步是对的，但"挪 `eq_route`"是错的：把论文波形收进通用求解门面，
   等于让引擎替论文做协议选择。已回落 `rydberg_fci`，见 §3 裁决 5。
3. ramp 的 δ 项约定：`H_builder(Om, delta)` 返回**含 δ 项**的 H（方案 A，要求模型侧
   自己处理）vs 不含 δ、由编排层统一补 −δ·n_tot（方案 B，rydberg_fci 全模型当前
   是三矩阵分解口径）。方案 B 对 LinearOperator 路径省一次闭包包装，但把"δ 是对角
   的"这个假设抬进了编排层。**当前倾向 A**（δ 的处理是模型知识），`n_tot` 仅用于
   密度观测。
4. 守恒校验的抽样比例（max(1000, 1%) 非零元）是否足够？
