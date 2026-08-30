# AI 队友任务书：两场引擎（2026-08-30 重写版）

> 船长写给代码调试队友（aly:qwen3.7-flash），**当前有效版本**。
> 历史探索记录已压缩到 §A（教训清单），不再赘述滚动过程。
> 每轮开工先读本文档 §READ FIRST + §0 状态，再动手。

---

## ⭐ READ FIRST（每轮开工前必读）

**当前激活任务**：
- **t6 = 修两体引擎 `evolve_exact_pair.py` 的同频合并 bug**（船长已定位，见 §1）
- 完成后接 t7（两体交换验证）。多光场拍频（§2.3）是其后的独立待办。

**完成任务判定标准**（照这个做，不要自由发挥）：
1. bug 修复 + 现有 EP1（直积）EP2（交换）回归通过；
2. 修复前后数字对比汇报。

**mailbox 规则**：船长历史消息均已并入本文档，无需逐条回复旧消息。
**回报纪律**：改了什么（文件:行）、验证数字、与上一版本对比。不确定就明说。

---

## 0. 当前状态（2026-08-30 快照）

### 已完成并结案
- ✅ 单原子 `evolve_exact.py` 同频合并 bug 已修（t4）：`Kc_dense` 逐波配对，回归全绿
- ✅ 单原子 `evolve_two_field.py` 同款 bug 已修（t5 附带）：620ns 光学窗口三路对照
  Sambe vs exact 1.13e-4 ✅ / Hard vs exact 1.72e-4 ✅ / Hard vs Sambe 6.3e-5 ✅
- ✅ hard 微波组装**无 bug**（DOP853 族与 reference 一致）——此前"hard 有 bug"是
  evolve_exact 自身 bug 造成的假象

### 引擎选型结论（§11 精简，2026-08-30 定论）
单原子尺度 Sambe 与 hard **均已与权威一致（620ns 三路 ≤1.7e-4）**——选型在
单原子层面已闭环，**不再有 t7（圈3 交换）独立选型任务**。
两体是剩余验证点，但以"复用旧引擎 + 修 bug"为路径，非"对比选型"：
- t6（进行中）：修 `evolve_exact_pair.py` 同频合并 bug（旧代码残留隐患）
- 修后顺带验证两体直积（V_dd=0 ≡ 单原子）作为回归，不再单列为选型判据

### 任务地图（2026-08-30 重排）
| 任务 | 状态 | 内容 |
|---|---|---|
| t1-t5 | ✅ 结案 | 认知/排查/硬算验证/单原子同频修复/重验 |
| **t6** | 🔧 进行中 | **修两体同频合并 bug**（§1，非"新框架"） |
| ~~t7~~ | ❌ 已取消 | 圈3 交换选型判据——过时，选型已由 620ns 三路完成 |
| §2.3 | ⏳ 待办 | 多光场拍频（W3/W4 参考对齐，独立） |
| — | ⏳ 待办 | hard 性能曲线（rtol 扫描 + 1500ns 全场景 ≤900s） |

### 资产盘点
`docs/核心资产盘点表.md`——两体工具（pair_evolve 的 tensor_states/_as_product_psi/
_reduce_A/_reduce_B/_tensor_field_atom/_pair_loss_vectors）与两体 Sambe 骨架
（evolve_exact_pair._PairSambeSetup）**全部成熟，直接复用，不重写**。

---

## 1. 🎯 t6 任务：修两体引擎同频合并 bug（最高优先）

### 1.1 船长定位的 bug（代码行级，已通读确认）

`evolve_exact_pair.py` 与单原子 `evolve_exact.py` 修复前**同款错误结构**：

```
_pair_groups（:57-94）：同频场 K2 直接相加 —— :92  hit['K2'] = hit['K2'] + K2
                           envs append               :93  hit['envs'].append(wf)
_PairSambeSetup.c_mids（:179-182）：Σ env —— :181  sum(env_wf(mid))
_PairSambeSetup.K_seg（:191-199）：K_static + Σ c·Kc_ext
Kc_ext（:158-172）：由 0.5·K2 建成（:170）
⟹ 段耦合 = (Σenv)·0.5·(ΣK2) —— 与单原子原版 (Σenv)·(ΣA) 完全同类错误
```

**后果**：两体场景含 A/B 侧同频场（如两边都有 17.2367 GHz 微波）时，耦合被
Σenv 和 ΣK2 交叉污染 → 2×/多倍错误。EP 系列自检（2 态/原子、无同频多场）未暴露。

### 1.2 修复方案（与 t4 同款，复制已验证模式）

1. `_pair_groups`：同频组改为 `dict(omega_mhz, K2_w=[K2_w 每波形], envs=[wf])`
   —— **不再提前相加 K2**；
2. `_PairSambeSetup`：`Kc_ext` → `Kc_ext_w[gi] = [每波形 Kc_w]`；
3. `c_mids`/`c_at_nodes`/`K_seg`：按键 (env_w, Kc_w) **逐波配对**
   `Σ_w c_w·Kc_w`；
4. `_defect_pass`（:251）：同样逐波 dc_w = env_w(nodes) − env_w(mid)；
5. 稀疏路径 `_subspace_segment` 的 R（`_volterra_rhs` :286）同步逐波配对。

**禁改**：物理模型（V_dd 构造、单位、初态约定）。只修数值配对。

### 1.3 回归（必须跑并回报）

- `checks/evolve_exact_pair_selfcheck.py` 的 EP1a/b（直积 ≤5e-6/5e-7）、
  EP2a/b（交换首峰 10.30ns ≤2%、幅度 ≥0.99）、EP3/EP4/EP5 全部；
- 单场组与改前**逐位一致**（Σ 退化为单元素是回归核心）；
- 若出现任何 EP 失败：停下汇报，不要自行改动其它逻辑。
- 产出：`tempsimu/_tf_pair_direct.py`（V_dd=0 直积 vs 单原子直积 ≤1e-4）——
  这是 t7 交换验证的前提。

---

## 2. 已修/已证结论（勿回退、勿重做）

### 2.1 单原子同频合并（§10 结案）
`evolve_exact.py` 三处（Kc_dense 逐波列表 / _seg_decomp 逐波配对 /
_defect_sweep 加 tb+逐波 dc）+ `_propagate_adaptive` 两调点。
回归：_tf_smoke[A]=0.5021、_tf_combo [1..4] 全 0.0013、hard vs exact 1.7e-4 ✅

### 2.2 hard 微波组装 ≠ bug（结案）
`evolve_hard` 与 `evolve_reference`（同为 DOP853 实验室系零近似）diff 1.93e-9——
正确性由权威同族背书。**不要**再花时间"修 hard 微波"。

### 2.3 多光场拍频（§8，**待办**，独立于 t6）
W3/W4 都耦合 6P↔60S，但旋转参考只对齐第一个光场 → W4 失谐 +680.8（应 +420）。
**修法（T10 已验证修正，2026-08-30）**：光场项必须用**复指数** exp 而非实 cos：
`V += c_f(t)·exp(i·_CONV·(ω_f−ω_ref_f)·t)·A_f`（非对角矩阵元，下三角配厄米共轭）。
**T10 结论（debugger 验证 + 船长独立复核）**：
- V1 精确旋转系 vs 实验室系 = 5.16e-09 ✅（旋转系框架数学严格等价）
- V2 RWA+exp(iδt) = 9.92e-03（MHz 类比 RWA 误差 ~1%；真实 THz 光学 (Ω/ω)²~1e-14 可忽略）✅
- V3 cos(δt) = 95.3% ❌——**实 cos 含 e^{±iδt} 双慢边带 → 虚假反旋驱动**，船长 §8 原公式已修正
- 脚本：`tempsimu/_tf_beat_theory.py`
→ **拍频方案数学成立，实现时用 exp 相位；sambe K 带时变相位因子 exp(i·δ_f·t)**。

### 2.4 单原子 Sambe 退役结论 → **已撤除**（620ns 三路一致后）
此前"退役"基于 0.038 范数偏差，绝大部分源于同频合并 bug；修复后
2.5e-4（稳定不发散），Sambe 恢复候选。**不要再提退役**。

---

## 3. 数字基线（复现用，不要瞎调）

- 场景：`tempsimu/_scenario_models.py::load_scene()` → (cache, 22states, wa, wb, pair)
- 光学失谐：W3=+620 MHz、W4=+420 MHz（自检 PASS）
- 微波：17236.743 MHz ×2 高斯（60S↔60P，Ω≈5.1 MHz，π 脉冲面积 511 MHz·ns）
- 已知锚：交换首峰 10.30 ns = π/(2·|V_ex|)；V_dd max|V|≈3.82 MHz（R=10μm θ=45° φ=0）
- 两体维度目标：N²=484 × P_mw=5（仅微波进光子块）→ 2420 维稠密 eigh 可行

---

## 4. 工程纪律

- 工作区 `D:\学\BaiduSyncdisk\编程\code\python\Energy Level`
- **BaiduSyncdisk 同步可能横跳**：大改用整文件 Write
- 改动前读文件；改动后立刻跑对应自检确认不回归
- 长任务 `run_in_background`；超时先查段数/范数日志
- 禁止：改物理模型/单位/常数、跑全量 selftest_all（太慢）、
  改动 `docs/` 中船长写的方向文档（如需修正先汇报）
- **代码组织原则（船长 2026-08-30 修订）**：
  - 追求"单一实现"是**倾向**而非教条：逐波配对这类"一处修复多处受影响"的
    改动，优先统一收束；但若新旧路径有真实性能鸿沟（如稠密 vs 稀疏）或
    接口演进成本过高，**保留明确标注的兼容分支是合理的**——关键在于
    **分叉点必须有注释说明"为什么分叉、各分支适用场景"，禁止无注解的分支**。
  - 每次改动后自查：是否留下了无用的 deprecated 字段/死分支？若有，本轮顺手删。

---

## 12. 🚀 hard 路线：Julia 经验迁移 + 内核加速（2026-08-30 船长定，harddev 负责）

**分工**：harddev（chenlab-official/deepseek-v4-flash）负责 hard 路线与 Julia 资产迁移；
debugger（aly/qwen3.7-flash）继续 t6（两体同频合并）。两线并行，t6 无关。

**hard 路线目标（实用速度）**：
- 单原子 22 态 620ns：≤5 s（现 194 s，rtol=1e-8）
- 两体（t6 修好后的 evolve_exact_pair 或 hard 两体）：≤2 min
- 零近似不变（DOP853 硬积分是 hard 的身份，**禁止解析化载波**——那是变成 Sambe）

**Julia 资产地图（harddev 自己研究，参考为主，可照搬成熟代码思路）**：
- 环境：`D:\julia-1.10.10\bin`、depot `D:\julia\depot`（JULIA_DEPOT_PATH）、TUNA 镜像、16 线程
- 源码：`julia/src/*.jl`（sambe/densepath/sparsepath/wosc/mprobe/block/evolve/io）
- 文档：`docs/Julia引擎交接文档.md`（含实测数据、三种构造约定、性能教训）
- 关键可迁移思想：① JIT 内核化（=> numba njit）；② 包络/载波**预计算查表**（`_pack_pair_terms` 模式）；③ 稀疏结构在 IP 下保留（对角 u 不破坏非零位）；④ gen-3 逆 Volterra-GMRES 的"收敛与巨大对角无关"（两体 hard 参考）
- ⚠️ 陷阱：Wosc/Filon 解析振荡 → 那是把 hard 变成 Sambe，方向错误；hard 的独立价值 = 不解析化、可交叉验证 Sambe

**hard 三件套实施顺序（harddev）**：
1. numba njit 化 `evolve_hard` 的 RHS（u·V·u† @ y + cos/env 求值）→ 620ns ≤5s 验收
2. 查表化 cos/env（tgrid 均匀采样 + 线性插值）→ 再省 2-5×
3. 两体 hard：k=i·N+j 张量 + V_dd + 稀疏矩阵乘 + numba（依赖 t6 的两体框架方向）

**同步认知**：开工前读本文档全文（尤其 READ FIRST/§0/§6/§A）+ `docs/核心资产盘点表.md`。

## 13. 🏗️ 完整重构决策与判据（2026-08-30 船长定，用户授权自主推进）

**背景**：用户确认"两体能用的算法单体也能用"——调查确认当前单体/两体
**分裂严重**（30-40% 重复）：_SambeSetup/_PairSambeSetup、_dop853_step/
_dop853_step_pair、_embed/_embed_pair、_defect_sweep/_defect_pass 均为
同一逻辑两份拷贝。正确架构 = **核心层（与 N 无关）+ 调度层（单体/两体张量化）**。

**决策判据（t8 结果门槛）**：
- ✅ **t8 良好**（两体 hard 对照通过 + 单原子 620ns ≤5s 达标）→ **全面推进完整重构**：
  1. 先 git 基线（427df61 已存）
  2. 核心层：sambe.py（光子块装配/缺陷修正，任意 N）、dop853.py（DOP853 内核，
     numba，任意 N）、rk4.py、fields.py
  3. 调度层：tensor.py（tensor_states/场镜像/V_dd 注入）+ evolve.py（单体入口）
     + evolve_pair.py（两体入口）
  4. 每完成一层存一次 git；每层跑既有回归（EP1-EP5/smoke/620ns 三路）
  5. 最终：单文件核心 + 两个薄调度入口，消除重复
- ❌ **t8 不佳**（两体对照失败或加速不达标）→ 与 debugger/harddev 分析根因：
  若 hard 路线原理性失败 → 回退 hard 为主、以 Sambe 为核心（它已 2s 且两体 2420 维
  可行）；若仅实现细节问题 → 修复后仍按重构推进；架构方案也可调整（如仅抽工具层）。

**重构原则**：先保存 git；每层小步提交；回归不破；dead code 顺手删；
分叉点必须有注解（§4 纪律）。**用户长期不干涉，里程碑式汇报**。

## A. 历史教训清单（每次遇到分歧时对照）

1. **范数守恒 ≠ 物理正确**：hard 4.6e-8、exact 2.9e-4 都守恒，却曾给错答案
   （同频合并让 π 脉冲变 2π，布居绕回）。判断物理必须三路交叉（exact/hard/
   reference），且含至少一个"人工未改"路径。
2. **两个错误实现"对拍一致"是假一致**：evolve_exact 与 evolve_two_field 曾各带
   同款同频合并 bug，错得一样 → "sambe≡exact"骗过所有人。交叉验证时若两实现
   共享同一错误来源（复制代码），一致性无意义。
3. **同频合并 bug 模式**：`(Σenv)·0.5·(ΣA)` ≠ `0.5·Σ(env_w·A_w)`——凡把同频场
   的 A 先相加、又用 Σenv 单标量的地方都要怀疑。检查清单：Kc 构造 / c_mid /
   defect_sweep / K_seg。
4. **权威引擎也可能有 bug**：evolve_exact 曾被当权威，结果自身带同款 bug。
   权威 = "独立实现 + 已对拍 + 无共源错误"。
5. **监督看数据不看报告**：修复后必须重跑全部依赖结论的测试（timeline 意识）。

---

*文档版本：2026-08-30 重写（替代此前 338 行的滚动历史）。历史细节如需追溯见
git log / 各 tempsimu 脚本。*