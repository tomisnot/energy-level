# 认知固化：引擎家族现状与全频段 Sambe 方向（换方向前快照）

> 2026-09-03 船长整理。本文档固化**当前全部引擎认知**，供换方向（弃混合引擎、
> 研究"全频段固定频率任意包络、一律 Sambe"）后快速上手。包含：项目全貌、
> 引擎家族现状、关键物理/数学认知、问题史与教训、混合引擎放弃原因、
> 全频段 Sambe 的既有基础与可行性预判、代码地图与快速上手。
> 配套文档：docs/AI队友任务书-两场引擎.md（历史任务书，部分已过时）、
> docs/t34核查结论.md、docs/t38混合引擎原型报告.md、docs/t39STIRAP仲裁.md。

---

## 0. 项目全貌（30 秒版）

- **物理**：Rb-87 里德堡"交换全过程"仿真——两原子、22 态/原子（60S×2 + 60P×4 +
  6P3/2 超精细×16）、微波（17 GHz 驱动 60S↔60P）+ 双光场（295.7 THz 驱动
  6P↔60S，失谐 +620/+420 MHz）、典型 t_max=620-1500 ns。
- **目标**：第一性原理准确 + 秒~分钟级。
- **单位约定**：频率 MHz、时间 ns；`_CONV = 2π×1e-3` rad/(MHz·ns)；Rabi 周频
  Ω/2π（π 脉冲 = 1/(2Ω) µs）；正失谐=蓝失谐。
- **工作区**：`D:\学\BaiduSyncdisk\编程\code\python\Energy Level`
- **权威参考**：`evolve_reference`（实验室系 DOP853）≡ hard 1.93e-9；
  `evolve_sambe frozen` ≡ STIRAP 类场景仲裁正确（t39）。

---

## 1. 引擎家族现状（核心资产地图）

### 1.1 单原子引擎（src/core/）

| 引擎 | 文件 | 机制 | 现状 |
|---|---|---|---|
| **ip4** | evolve_rwa.py | RWA 近/远分流 + 链式参考 + RK4 | **有问题**：多频参考污染（问题.json 现象①）、旁观态混叠（Julia §11 P1）|
| **sambe**（原 exact）| evolve_sambe.py | **全频段 Sambe 光子块**（实验室系无 RWA）+ frozen 纯冻结分段 | ✅ **STIRAP/拉曼类正确**（t39 仲裁），frozen 模式 24.6s/3400ns |
| **hard** | hard.py + dop853.py | 光场旋转系 RWA + 微波 cos 硬积分 DOP853 | ✅ 620ns 2.1s；**光场链式参考缺陷**（STIRAP 用不了，rot_pick THz 残差）|
| **mixed**（新）| evolve_mixed.py | 光场 Sambe 光子块 + 微波 DOP853（Strang/φ-帧）| ❌ **放弃**（t40-t44 五连败，见 §4）|

### 1.2 两体引擎

| 引擎 | 文件 | 机制 | 现状 |
|---|---|---|---|
| **evolve_sambe_pair** | evolve_sambe_pair.py | 全频段两体 Sambe | 基础可用；光场也进光子块 → 维数爆炸（60500 维不可用）|
| **evolve_two_field_pair** | evolve_sambe_pair.py | **两体两场**：光场旋转系 + 微波-only 光子块 | ✅ **另一 AI 补全（t30 批次1-3），自检 11/11**，dim=2420 |
| pair_evolve | pair_evolve.py | secular RWA 两体 | 旧，多波形稀疏待重构 |

### 1.3 支撑层

- **static_atom.py**（core）：能级/偶极/AC-Stark/偏振（t23 实数化 13.2×）
- **static_pair.py**（pair_core）：vdd_sparse/交换耦合
- **sambe.py**：photon_geometry（任意 N 光子块，混合进制，Ho-Chu 多模 Floquet）
- **dop853.py**：DOP853 核心层（numba first-class RHS 参数，t11）
- **tensor.py**：两体张量化（_reduce_A/B）

---

## 2. 关键物理/数学认知（这些是踩坑换来的，勿重踩）

### 2.1 Sambe 光子块（原理）

把含时振荡驱动 cos(ωt) 改写成扩展空间的**静态矩阵**：引入光子数自由度 n，
光场耦合 |i,n⟩↔|j,n±1⟩（同时改变原子态与光子数），光子块对角 = E_i − n·ω
吸收载波 → 扩展空间哈密顿定常 → eigh 精确对角化。

- 光学载频 M=1（3 块）即可（多光子概率 (Ω/ω)²~1e-14）；微波 M=2（5 块）多光子显式
- **只支持固定频率**：变频/任意相位（chirp）破坏周期性，光子块无法构造
  （build_fields_exact 对 ArbitraryWave 显式抛 NotImplementedError）
- **支持任意包络**：包络是慢调制，不影响光子块结构（frozen 模式已证）

### 2.2 frozen 模式（t36/t37，Sambe 时变包络的解药）

**问题**：原 defect 路径（段内冻结 + Duhamel 缺陷修正 Gauss 积分）在 THz 光子块
下死局——积分核 e^{iC·ε·t} 含 THz 本征振荡，Gauss 节点无法解析 37.8 万周期 →
段长被钉死 0.125ns → STIRAP 3400ns 需 5-13 分钟。

**解法（t36）**：砍掉缺陷修正，纯冻结分段——每段取中点 K(c_mid) 一次 eigh +
精确指数。段长由**包络缓变性**决定（σ=400ns → h=1ns 布居偏差 8e-8），与 THz
解耦。STIRAP 3400ns：h=1ns 88s → t37 eigh 单线程加速后 **h=2ns 24.6s**、
收敛 3.5e-7。**frozen_h_ns 参数（None=原 defect 路径行为不变）**。

### 2.3 φ-帧（相互作用绘景，t41 关键实验）

φ = e^{iC·D_full·t}·X（D_full = 光子块全对角 E−n·ω）——把 THz 对角吸进全局
相位，剩下耦合项全可控：
- 光场共振对 ΔD≈失谐（MHz 慢）✓
- 微波同扇区 ΔD=能级差（cos 干涉）✓

**单 DOP853 直接积分 φ-帧方程即可**（船长验证 3 态 300ns max|Δpops|=4.22e-8）。
Strang 分裂 + eigh 冻结是多余的（t41 平台 1.5e-2 的根源，dt 无关）。

### 2.4 远失谐对 AC 位移（t43/t44 教训，混合引擎失败核心）

**光子块里远失谐对（如 60S↔60P @ 295.7 THz，Rabi 13.6 THz、ΔD=±295.7 THz）
两个边带方向的 AC 位移符号相反、几乎抵消** → 净位移只有 ±18 MHz（2 态诊断），
而非单方向 Ω²/4Δ（±156 GHz）。**ref 转移正常的原因**：净位移仅使微波失谐
−36 MHz 仍共振。

**22 态却净 ±198 MHz**（多 mJ 远对叠加不抵消）——注入后微波失共振（60S末=0.52
不转移）✗。**结论：混合引擎的微波 cos 处理与 ref 的微波光子块（M=2）存在机制
差异，简单 AC 注入补不回**——这是混合引擎放弃的直接原因。

### 2.5 其他已验证结论

- **拍频必须 exp(iδt) 非 cos**（t10）：cos(δt) 有 e^{±iδt} 双慢边带 → 虚假反旋
  驱动，95% 误差
- **光场 RWA 振幅 = 0.5·rabi**（t17）：cos(ωt)=½(e^{iωt}+e^{−iωt}) 旋转系只留
  共振项 → 0.5 因子；此前 AC-Stark 偏 4×
- **近共振过滤必要性**：微波近共振对 |ΔD|−ω<5GHz（MW_NRES_BW_MHZ）；光场反旋
  对 |ΔD|<20GHz（BW_OPT）——THz 相位项不可冻结/不可积分
- **同频合并 bug 三处修复**：Kc 构造逐波配对，禁止 (Σenv)·(ΣA)
- **问题.json 现象①**（ip4 参考污染）：60S 被光场平移 THz 而 60P 不共享 →
  微波对假差；hard 旋转低流形绕开，ip4 仍存在（t34）

---

## 3. 问题史与教训（§A 清单扩展）

0. **numba cache 掩盖回归**：@njit(cache=True) 缓存键=字节码哈希，改名不改函数体
   可复用旧缓存 → 物理变更被旧机器码掩盖。**对策：物理/内核变更后清
   __pycache__/*.nbc 再自检**。
1. **范数守恒 ≠ 物理正确**：hard 4.6e-8、exact 2.9e-4 都守恒却给错答案。
2. **两个错误实现"对拍一致"= 假一致**：必须有人工未改的独立裁判
   （evolve_reference）。
3. **权威引擎也可能有 bug**：evolve_exact 同频合并 bug 曾使 hard 被误判。
4. **缺陷修正 THz 死局**（t33 Q1）：全频段 Sambe 时变包络原始路径不可行。
5. **混合引擎五连败**（t40-t44）：E 相位双计 → φ-帧 → 单 DOP853 → AC 注入 →
   净 AC 注入——微波 cos vs 光子块的机制差异是硬伤（§2.4）。
6. **Stark 位移符号要双向算**：单方向 Ω²/4Δ 是伪值，光子块里双向边带抵消。

---

## 4. 混合引擎为什么放弃（决策记录）

| 任务 | 尝试 | 失败点 |
|---|---|---|
| t40 | K_noE + U_mw lab 转换（Strang）| 300ns 光场失共振（60S=0 vs 0.946）|
| t41 | φ-帧 + Strang + eigh | 22 态平台 1.5e-2（dt 无关，分裂/eigh 多余）|
| t42 | 单 DOP853 直接积 φ-帧 | 22 态中段 2.36e-2（60S 不再激发）|
| t43 | 单方向 Ω²/4Δ AC 注入 | 伪值 446 GHz → 微波失共振 |
| t44 | eigh 法净 AC 注入（±18→±198）| 22 态注入后仍破坏转移（60S末=0.52）|

**根因**：mixed 的"微波 cos 硬积分"与 ref 的"微波光子块 M=2"在强 AC 场景下
dressed 机制不等效；光场远失谐对的净位移（22 态 ±198 MHz）无法用常数对角
补回而不破坏微波共振。**混用两套微波处理机制是根本矛盾**。

**教训**：要保微波物理与 ref 一致，微波必须和光场用**同一机制**（都光子块，
或都硬积分）——混用必出现机制差异。

---

## 5. 新方向：全频段固定频率任意包络、一律 Sambe（可行性评估）

### 5.1 定义

不分光场/微波（弃 split_by_freq 两场分流），所有场一律进 Sambe 光子块
（各自载频一个光子维度，Ho-Chu 多模 Floquet），固定频率 + 任意包络。

### 5.2 既有基础（好消息：核心已验证）

- **单原子全频段 Sambe + frozen** = 正是 evolve_sambe(frozen_h_ns>0)——
  **STIRAP 3400ns 24.6s、收敛 3.5e-7、t39 仲裁确认为正确引擎**（已可用！）
- **620ns 基线（光+微波全频段）**：t39 的 ref 就是 evolve_sambe frozen
  （22 态），已跑通——**微波也进光子块（M=2）场景 frozen 已实证可用**
- **两体两场 Sambe**（evolve_two_field_pair，另一 AI 补全）：光场旋转系 +
  微波-only 光子块，dim=2420，自检 11/11 ✅——可作为两体方向的起点

### 5.3 关键待研究问题

1. **两体全频段维数**：光场也进光子块 → 60500 维不可用。两条路：
   a. 保持"光场旋转系 + 微波光子块"（两体两场，已实现）——严格说不是"全频段
      Sambe"，但两体场景维数可行
   b. 全频段两体：需维数缩减（对称性分块 / 稀疏 / 只保留近共振光子块）
2. **微波 M=2 的 frozen 性能**：620ns 已证可行，1500ns 需实测（目标秒~分钟级）
3. **变频/任意相位**：Sambe 光子块不支持（周期性破坏）——变频仍需 hard 或
   ArbitraryWave 路径（这是明确边界，不是全频段 Sambe 的能力）
4. **多光场拍频**：全频段 Sambe 用 _group_by_freq 按载频分组自然处理
   （各自光子维度），不需要两场引擎的 exp(iδt) 拍频——**这是全频段相对两场的
   优势**

### 5.4 初步可行性结论

**单原子全频段 Sambe（frozen）= 已可用、已仲裁正确**——新方向的核心已站住。
两体需二选一：两体两场（光旋转+微波光子块，现成）或全频段两体（需维数攻关）。
变频是明确边界（用 hard）。**建议：先固化单原子全频段为推荐路径，两体用
两体两场，变频用 hard——三引擎按场景分工，不再混合**。

---

## 6. 代码地图与快速上手

### 6.1 关键入口

```python
# 单原子全频段 Sambe（推荐，STIRAP 类）
from src.core.evolve_sambe import evolve_sambe
fields = waveforms.build_fields_exact(cache, states, waves)  # 全偶极对
r = evolve_sambe(states, fields, psi0, t, tol=1e-6,
                 frozen_h_ns=2.0, log=...)                   # frozen 模式

# 两体两场 Sambe（另一 AI 补全）
from src.core.evolve_sambe_pair import evolve_two_field_pair
meta_A = build_two_field(cache, states, waves_A)
meta_B = build_two_field(cache, states, waves_B)
r = evolve_two_field_pair(states, meta_A, meta_B, psi0_A, psi0_B, t,
                          R_um=..., theta_deg=..., cache=cache)
```

### 6.2 自检清单（全绿才算健康）

```bash
python -u checks/evolve_exact_selfcheck.py        # sambe 11 项
python -u checks/two_field_pair_selfcheck.py      # 两体两场 11 项（另一 AI）
python -u checks/hard_numba_selfcheck.py          # hard 9 项
python -u checks/ui_selftest.py                   # GUI 57 项
python -u checks/selftest_all.py                  # 全量（慢，谨慎）
```

### 6.3 关键 tempsimu 脚本

| 脚本 | 用途 |
|---|---|
| _t36_frozen_3400.py / _t36_compare.py | STIRAP frozen 全程 + 对比 |
| _t39_conv.py / _t39_ref_check.py | frozen 收敛序列 / 参考核查 |
| _t41_single_dop853.py | φ-帧单 DOP853（船长 4.22e-8 实验）|
| _t43_diag_ac.py | 远失谐对 AC 位移双向抵消诊断 |
| _t42_diag.py | mixed vs ref 逐态差异定位 |
| _t37_profile.py | frozen 段耗时剖析（eigh 94%）|
| _t35_frozen_proto.py | 纯冻结原型（船长 40ns 窗段长-精度表）|

### 6.4 存档场景（save_state/）

- 交换全过程.json：双原子 22 态，光+微波混合，hard 方法（R=12/θ=30°）
- STIRAP.json / 拉曼.json：已转 v2 格式（v1 存 _legacy/）
- 问题.json：ip4 参考污染现象①复现场景（v1 格式，读取需兼容路径）

---

## 7. 待办（按新方向重新排序）

1. **【推荐】固化单原子全频段 Sambe 为默认**：GUI method 接线 frozen 模式
   （目前 GUI sambe 仍走 defect 路径，时变包络会死局）
2. **两体方向选型**：两体两场（现成，11/11）vs 全频段两体（维数攻关）
3. **1500ns 全场景性能实测**（frozen 微波 M=2）
4. 变频脉冲：明确走 hard（cos(φ_f(t)) 任意相位路径，t18）
5. 遗留：README LaTeX 化（待用户确认）、julia/ 目录是否删（当前保留参考）

---

## 8. 会话中反复确认的用户偏好（合作模式）

- 先调查构思再动手；改 src 后立即 ast.parse + import 冒烟
- 队员任务必须走 agent_teams 正式立项；关键认知写进任务书（消息有延迟）
- 数值/物理结论要量化（数字/表格），不靠报告
- 新功能单一实现是倾向非教条；性能与正确性并重，不为复用牺牲性能
- 物理/内核变更后清 numba 缓存再自检（§A 第 0 条）
