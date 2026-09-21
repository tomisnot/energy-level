# Re0 重构 · 物理引擎沙盒

> **这里只有引擎**（`src/core/`）。GUI / AI / session / kernel / model 一律不在——
> 为了能在不被界面和框架牵动的前提下，把引擎的**调用接口**、**能级基准**、
> **参数归属**重做一遍。
>
> **原则：不套层。** 改动全部落在 `src/core/` 内部；不在引擎外面加适配皮。

---

## 现状

| | 源仓 | 沙盒 | 变化 |
|---|---:|---:|---:|
| `src/core/` | 25 文件 / 11722 行 | **22 文件 / 11369 行** | 删 4 文件（−839 行）**+ 新增 `field_loss.py`** |

删掉的 4 个文件：`params.py`(357) · `route.py`(100) · `evolve_engine.py`(54) · `project.py`(33)。
其余 −256 行来自删死代码（`real_field` 120 + `_floquet_ac_static` 43 + `group_field` 16）、
删 `evolve_hard` 滞后拷贝 137、合并 5 处重名、删 25 处死 import、文档精简。

**基线**：`##RE0_SUMMARY scripts=27 fail=0` —— **27 判据 / 198 项 / 0 FAIL**（21 物理 + 6 机甲框架）。

⚠ **这个目录同时有别的会话在改。** 2026-09-19 本会话按一份**已批准的计划**实施了
“第 14 刀”两条主线：耗散契约修正（C1/C2/C5，新增 `dissip_contract_selfcheck`）与
场致加热损失接入（新增 `field_loss.py` / `build_field_loss` / `field_loss_{,engine_}selfcheck`）。
**同日随后**又按《`docs/AI机甲框架-开发计划.md`》实施了 **Phase 0–5**：新增 `src/mecha`
（语义内核）+ `src/cockpit`（独立只读 Web 驾驶舱），让 **AI 经语义层接入引擎**（基线 21→26）——
逐阶段记录见 **`docs/AI机甲框架-实施记录.md`**。
期间发现基线行被另一会话改至 19——**上述 26 是本会话实跑的最新树**。
**改动前先确认无别的会话在同一目录干活**（见文末"沙盒环境注意"）。

判据是**改造期的守门人**：任何动作先跑基线，绿了才继续。

---

## 目录

```
src/core/                 引擎本体（**23 个文件**；含 `param.py` 就地参数声明）
    static_atom.py        Rb-87 原子物理（ARC 封装 / 偶极 / AC+DC Stark / 能级行 / 谱线）
        └ ★ base_sublevels()  **能级基准唯一处**（静态页与动态基组共用，两页 label 同源）
    static_pair.py        两体相互作用（V_dd / 本征 / 交换耦合 / vdW）
    atomic_levels.py      能级构造（make_levels / state_key）
    field_build.py        场构造（field / build_fields_exact）
    waveforms.py          波形数据层（Wave / 任意波形 / 三种包络）
    runspec.py            ★ **配置层 + 唯一分派点**：登记册快照 → RunSpec → 跑 → 结果摘要
    sambe.py              Sambe 引擎（单体 + 两体 + 核心层）
    evolve_mixed.py       混合引擎（光 Sambe 光子块 + 微波硬积分）
    evolve_twofield.py    两场引擎
    hard.py               hard 路线（numba JIT）——**`evolve_hard` 的正本**
    evolve_rwa.py         RWA（ip4 / Lindblad）——**数值权威参照**，GUI 不暴露
    evolve_reference.py   参考积分器（权威锚）
    evolve_sambe_pair_gpu.py  GPU 传播（**torch 的唯一入口**，函数体内惰性导入）
    dop853.py             DOP853 核心（**被 coupling/hard 共用的数值内核**）
    coupling.py           RHS 打包公共层（+ `PHASE_TOL` / `R_CRIT` / `PHASE_BUDGET`；**`rhs_phi_pairs` 时变损耗块**）
    field_loss.py         场致加热与损失模型（L2 散射含饱和·修正 F3 / L3 TrapModel / L4 光电离；纯公式）
    tensor*.py            两体张量化
    evolve_util.py        衰变 / 初态 / 耗散工具（叶子模块，只 import numpy）
    diag.py / cost.py     诊断摘要 / 成本估算
    manybody/             多体引擎家族（N 原子阵列 ED；模型卡/装配/演化/观测量）
src/_archive/             evolve_floquet_pwc.py（Floquet 权威锚，被两个判据引用）
src/mecha/                ★ **机甲框架语义内核**（引擎之外；`core ↛ mecha` 单向）
    registry.py           参数语义单一查询口（**收集** core 的 Param，非复制）
    bus.py                append-only 事件总线（唯一真值；指针+sha1；体积闸）
    session.py            会话态 + 写 choke point（set→校验→发事件；no side doors）
    ops.py                run/estimate/preview/rerun/propose → runspec 唯一分派
    server.py             MCP 写通道（streamable-HTTP，localhost，人运行 launcher）
    subscribe.py          只读订阅端点（SSE，可远程鉴权）
    codeact.py / jobs.py / autonomy.py   受限编排 / async 作业 / 分级自治+后端路由
    fields.py             建场层：波形/含时场语义构造（set_drive/π脉冲/Ramsey/STIRAP/ARP）
    errors.py             可教学错误（kind/hint/suggest）
src/cockpit/              ★ **独立只读 Web 驾驶舱**（黑匣子；读 journal 文件，不 import mecha）
src/__init__.py           极简：**刻意不带 ARC 补丁**（见下"两个刻意的取舍"）
.mecha/                   运行时数据（journal/payloads/artifacts/jobs；gitignore）
checks/                   **27 个判据** + 1 个接线判据（默认跳过）+ checklib + pair_evolve
run_checks.py             物理基线跑
docs/                     18 份（见下）
old/                      归档，**不在活动路径上**（见下）
```

### `docs/` —— 按用途分

| 文档 | 什么时候读 |
|---|---|
| **`引擎重构总览与调用说明书.md`** | ★ **先读这个**：11 刀总览 + **调用说明书** + **UI 衔接工单**（§三） |
| **`AI机甲框架-实施记录.md`** | ★ **AI 接入怎么做的**：Phase 0–5 逐阶段交付/判据/不套层结论/踩到的坑 |
| `AI机甲框架-开发原则.md` | 纲领：第一性原理/不变量/分级自治/拓扑（只定原则，不绑项目） |
| `AI机甲框架-开发计划.md` | 把纲领落到 Re0 的五阶段计划 + 决策点 D-A~D-E + harvest 落点 |
| `机甲协议-draft.md` | 从仿真实例①蒸馏的语义层合规协议 DRAFT（N=1 不冻结） |
| `引擎-调用病症与改进方向.md` | 病历：9 类病症 + 9 个改进方向（方向表已逐条更新状态） |
| `遗留坑清单.md` | 主动复查的遗留坑（真未完成 / 已清 / 沙盒⇔产品仓缺口 / 待裁决）|
| `场致加热与损失-接入沙盒.md` | 第 14 刀主线二：`dissipation_lab`→core 的分层接入（H1–H3）/ opt-in / 不套层与后续 H3b·H4 |
| `语义层-我们有什么缺什么.md` | 「语义层」的四种意思 + Re0 的 6 块语义碎片 + **缺的是「单一定义点」**（改写了迁移项 C1） |
| `引擎层该有状态管理吗.md` | **只论沙盒**：引擎接口已是纯函数；扫出 1 处真状态（`_AC_UI_CFG`，实测同输入差 7.9 倍）+ 8 处记忆化 + 第 11 刀的漏网 |
| `runspec审查.md` | `runspec.py` 逐段审查：硬编码是不是债 / 3 个静默洞 / 死代码 / 重复 |
| `params归属-分析与管理方案.md` | 32 个参数的读者统计 + 三个方案 + 代价 |
| `引擎接口规范化-第1-3刀.md` | 返回契约统一 / hard 去重 / 删第二套入口 |
| `静态页与动态页-能级共用与频移引擎的甄别.md` | 两页差在哪、AC 频移引擎的现状 |
| `能级基准统一与清理.md` | 能级收成一处 / 静态页补 DC-Stark / 死代码清理 |
| `接口契约与重名清理.md` | 参数白名单改签名派生 / 重名 5→0 / diag 消息 / 两条自我更正 |
| `判据公开面.md` | 【F】私有耦合 20+11 → 3（白名单化）/ 6 份单位常量单源化 |
| `还需改什么-与理由.md` | 更早的待办清单（部分已被后续各刀追上，状态见文内更正段） |

### `old/` —— 归档（不删，可回溯）

```
old/lab/                  另一个人做过的**适配层**（2188 行，已否决——套层）
old/AI-调用架构.md         那份适配层的设计文档
old/checks/               判据的一份副本（沙盒重排遗留）
old/{lab_adapter,route_auto,mw_bench_firstprinciples,params_registry}_*.py
                          4 个过时判据（目标已删 / 不是判据的基准 / 纯注册表判据）
old/_*.py                 ~33 个一次性探针与迁移脚本（结论已进 docs/）
```

---

## 怎么跑

```powershell
$env:PYTHONPATH=''; $env:QT_QPA_PLATFORM='offscreen'
$env:EL_NO_AI_SERVER='1'; $env:PYTHONIOENCODING='utf-8'
cd 'D:\desktop\Re0重构'

& D:\python310\python.exe run_checks.py              # 物理基线（约 6 分钟）
& D:\python310\python.exe run_checks.py --list       # 列出判据
& D:\python310\python.exe run_checks.py --only sambe # 只跑名字含该子串的
& D:\python310\python.exe checks\evolve_exact_selfcheck.py   # 也可以单跑
```

判据一律以**独立进程**逐个跑（脚本间无共享状态）——与原 `selftest_all` 同一纪律。
退出码 = FAIL 的脚本数（0 = 全绿）。

---

## 基线（已实测全绿）

```
##RE0_SUMMARY scripts=27 fail=0 seconds=358
```

| 判据 | 项 | 覆盖的物理 |
|---|---:|---|
| `core_selfcheck` | 6 | ARC / 公式矩阵 / 能谱 / AC-Stark / F↔mJ 矩阵元 |
| `evolve_exact_selfcheck` | 11 | Sambe vs **封存 Floquet** 权威 |
| `evolve_exact_pair_selfcheck` | 11 (1 SKIP) | 两体 Sambe：交换峰 / vdW / 寿命 |
| `pair_evolve_selfcheck` | 12 | 两体 vs **pairinteraction**（第三方权威） |
| `two_field_pair_selfcheck` | 11 | 两场两体：旋转筛选 / 实验室系对拍 |
| `mixed_nolight_selfcheck` | 10 | mixed 退化边界（无光 / 无耦合） |
| `mixed_arb_mw_selfcheck` | 3 | 任意微波 |
| `mw_pair_keep_selfcheck` | 4 | 微波候选筛选（相位 / 带宽判据） |
| `hard_vdd_pack_selfcheck` | 9 | hard 打包 + 显式零矩阵 |
| `evolve_dissipative_selfcheck` | 6 | 自发辐射 / 耗散 / 交换耦合 |
| `manybody_m0_selfcheck` | 20 | 4 维对空间 V_dd **全精确标定** |
| `manybody_m1_ed_selfcheck` | 7 | 多体 ED 引擎：N=1 拉比 / N=2 blockade 解析 3 能级 / 对称扇区 / 张量积 / 非厄米衰减 |
| `manybody_integration_selfcheck` | 10 | 多体接入语义层：schema/校验/build_run 路由/Result 契约/查询/产物/线程拆分 |
| `hard_dedup_selfcheck` | 5 | hard 家族去重 + `solve.MAX_STEP_HARD` 可达性 + 压步长不改变物理 |
| **`levels_twopage_selfcheck`** | 7 | **两页同基准**（态集合 / 能量 / **label** 逐位一致）+ DC-Stark 两页生效 + 超精细加权 |
| **`engine_contract_selfcheck`** | 6 | 分派表唯一 / 参数白名单＝签名 / torch 惰性 / diag 归属 / 无跨文件重名 / **判据私有依赖有界** |
| **`engine_params_selfcheck`** | 4 | 参数就地定义的归属 / **派生量一致** / `solve.*` 无死映射 / 关键默认值不漂 |
| **`runspec_contract_selfcheck`** | 5 | **缺键 / 非法值不静默**（必填键、`decay_index`、未知 method） |
| `ai_h_snapshot_guard_selfcheck` | 9 | H 快照尺寸闸 |
| `checks_sentinel` | 2 | 判据存活哨兵 |
| **`dissip_contract_selfcheck`** | 3 | 耗散契约：**C1 E_shift 透传逐位等价** / C2 参考路径 error_report / C5 报错文案指真实档 |
| **`field_loss_selfcheck`** | 7 | 场致加热损失**模型**锚：A7/A8/A1（含饱和·修正 F3 取反）/ L4 光电离 / A9 n_crit / 双计·近共振 / build_field_loss 装配 |
| **`field_loss_engine_selfcheck`** | 2 | **引擎接线**：split_field_loss 分项数学 + mixed 消费 field_loss（散射抽布居 + 分项守恒 + 开/关差异）|
| **`mecha_kernel_selfcheck`** | 8 | 机甲语义内核：registry 收集/值逐位、set 回执 readback、I6 schema 交叉、I3/I4/M4 总线、**I1 无侧门 AST 守卫（致死性）** |
| **`mecha_mcp_selfcheck`** | 11 | MCP 写通道：14 工具、resolved+readback、可教学错误、diff-since-seq、体积闸、async 作业面、**真实 streamable-http 连接** |
| **`cockpit_selfcheck`** | 5 | 驾驶舱：①杀项目后读死前全史（T5）②diff-since-seq ③告警致死性 ④只读 AST 守卫 ⑤远程鉴权+POST→405 |
| **`mecha_codeact_selfcheck`** | 5 | CodeAct 编排+边界致死性、preview 不演化、可复现产物+rerun+内容寻址、async 断线找回 |
| **`mecha_backend_selfcheck`** | 6 | 硬件档预留：backend 路由表、sim 逐位等价现状+AI 侧零改动、L3 三闸、Reserved、dry-run、分级自治 |
| **`fields_selfcheck`** | 10 | 建场层：freq/Rabi 换算对拍、π 时长、校验致死、指针化、patch 隔离、preview 联动、**不套层逐位对拍 serialize_waves**、arb 改道、AI 面接线 |

**合计 198 项**（153 物理 + 45 机甲框架）。

### 一个被**显式跳过**的接线判据（不是删掉）

`solve_max_step_hard_selfcheck` —— **D 包**用 `pm_setup._SOLVE_SPECS` / `god_split`
（旧 GUI 登记册）；**A–C 包是物理**，其中"压步长不改变物理"已迁进 `hard_dedup_selfcheck`。
它在**源仓仍然是绿的**，只是沙盒里没有 `src/ui` 与 `pm_setup`。

---

## 已做的 14 刀

| 刀 | 治什么 | 一句话 |
|---|---|---|
| **1** | 【D】返回契约分叉 | `pops_2d`→**`pops`**、`states_2d`→**`states`**，单体两体同一个键。`result_labels` 里那个"漏判就标错 N² 维布居"的分支**消失了** |
| **2** | 【A】【B】`evolve_hard` 双实现 | 实测裁决（`max|Δ|=0.0`，逐行 diff 仅 11 行）⇒ **删滞后拷贝 137 行**；**接通 `solve.MAX_STEP_HARD`** |
| **3** | 【A】三份分派 | **删 `evolve_engine.py`**；`runspec.evolve_run` 成为**唯一分派点** |
| **4** | 【C】两页两套能级实现 | **收成 `static_atom.base_sublevels` 一处**；三处消费方各自投影。`E_vpcm=0` 时逐位不变 |
| **5** | 【C】静态页无 DC-Stark | **补上**；并修掉超精细态把 `mF` 当 `mJ` 传的**静默失效**（旧行为恒返回 0） |
| **6** | 【G】死代码 + 过时判据 | 删 `project.py`/`route.py`/`real_field`/`_floquet_ac_static`/`group_field`；归档 4 个过时判据 |
| **7** | 参数白名单手抄 | **改为按引擎签名派生**（`inspect.signature`）；新增 `_ENGINE_TABLE` 作为**唯一分派表** |
| **8** | 【B】跨文件重名 5 处 | **全部合并 → 0 处**（含公开名 `env_at`）。`_embed`/`_reconstruct` 的成因是**两个 setup 对象接口不一致**，正解是统一协议 |
| **9** | `diag` 消息怪调用方 | `summarize(None)` 改为**点名引擎**（真相：只有 mixed 家族产出诊断） |
| **10** | 【F】判据绑私有 API | **私有耦合 20+11 → 3**（白名单化 + 判据强制）；11 个能力名去下划线；**6 份重复单位常量单源化**；删 25 处死 import |
| **11** | **`params.py` 全局注册表** | **整层消灭**：32 个参数**就地定义**在各引擎模块；覆写栈消失 |
| **12** | 两页 label 不同字 | **统一到 `auto_label`**：`base_sublevels` 算一次，`build_states` 显式传给 `make_levels` ⇒ **同源同字**；人工名搬进 **`group_label`** |
| **13** | `runspec` 的 3 个静默洞 | 必填键缺失 / `t_max=0` / `npts<2` / 非法 `decay_index` → **一律 `ValueError`**（原来会静默跑 0 ns 空演化） |
| **14a** | 耗散契约（调查文档 C1/C2/C5）| sambe+decays **透传 `E_shift`**（旧静默丢）· `evolve_reference` **补 `error_report`** · hard 报错文案改指 sambe。判据 `dissip_contract_selfcheck` |
| **14b** | 场致加热与损失接入（`dissipation_lab`→core）| **不套层**分三层：H1 模型进 `field_loss.py`·H2 `build_field_loss` 装配·H3 `rhs_phi_pairs` 时变损耗块 + mixed 消费（**opt-in 默认关**，loss_base=None 逐位不变）。hard/两体与 arrhenius 列 H3b/H4。见 `docs/场致加热与损失-接入沙盒.md` |

**两条撤回**（都是我先前讲错的）：
- 【H】"`torch` 拖住 core" —— 实测 `import src.core` 0.95 s、不拉 torch
- 【G】"`dop853.py` 零公开 API" —— 它就是被 `coupling`/`hard` 共用的数值内核，core 之外无人需要

---

## 下一步

### AI 接入 —— **已落地**（机甲框架 Phase 0–5，2026-09-19）

AI 现经 `src/mecha` 语义内核接入引擎（不再需旧 `src/ai`）：

```powershell
& D:\python310\python.exe -m src.mecha.server     # 起 MCP 写通道（localhost，人运行；harness 靠 .mcp-port attach）
& D:\python310\python.exe -m src.cockpit.server   # 起独立只读驾驶舱（浏览器打开；黑匣子，项目死了也在）
```

详见 `docs/AI机甲框架-实施记录.md` 与 `docs/机甲协议-draft.md`。
**调用 AI 的使用指南见 `src/mecha/SKILL.md`**（16 个 MCP 工具 + 标准流程 + 契约，让 AI 不用试错）。

### 旧 GUI（`src/ui`）接线 —— 仍是**产品仓的下一阶段**（沙盒无 src/ui）

工单在 **`docs/引擎重构总览与调用说明书.md` §三**（A 必改 9 / B 应改 7 / C 需决策 3）。
其中 **C1**（参数语义单一查询口）已由 `mecha/registry.py` 解决；**C2**（`_` 前缀参数要不要上 UI）
已由 `Param(tier='internal')` 默认隐藏解决——GUI 可直接消费 registry（与 AI 同一查询口）。

---

## 两个刻意的取舍

1. **不带 ARC 数据补丁**。正式仓的 `src/__init__.py` 在导入期把 ARC 的 `dataFolder`
   指向项目内 `.arcdata/`，代价是任何 `import src.*` 都要拖进 sympy + matplotlib + scipy
   （实测 ~5.9 s）。沙盒**不装这个补丁**，ARC 走默认的 `~/.arc-data`（本机已存在，73 文件）
   ⇒ `import src.core` **~1.9 s**。
   *若某台机器没有 `~/.arc-data`：跑一次 `python -c "import arc; arc.setup_data_folder()"`。*

2. **只搬"只依赖引擎"的判据**。源仓涉及 `core` 的自检共 **34 个**（2026-09-18 实测；
   早先测是 33——**源仓在被外部编辑**），其中"不碰
   `src.ui`/`src.ai`/`src.kernel`/`src.sweep`/`PyQt5`/`pm_setup`"的**真判据 17 个**
   （外加 `pair_evolve.py` 作支撑）。

   沙盒的 **18 个判据** = 那 17 个 **− 4**（目标已删或需 GUI：`mw_bench_firstprinciples` /
   `params_registry` / `route_auto` / `solve_max_step_hard`）**+ 5**（新写的：
   `engine_contract` / `engine_params` / `runspec_contract` / `hard_dedup` / `levels_twopage`）。

   另外 17 个验的是**接线**（GUI/MCP/扫描），不是物理，留在原仓。
   分类用 AST 全仓解析得出，不是靠文件名猜。
   （`manybody_m0_selfcheck` 依赖 `src.manybody`——那 3 个文件**一并搬进沙盒**了，
   所以它算"只依赖引擎"。）

---

## 沙盒环境注意（实测）

**这个目录同时有别的程序在改。** 已知三次：

1. 有人把 `checks/` 整个移到 `old/checks/`，导致全部判据因路径解析失败而红
   （判据用 `dirname(dirname(__file__))` 定位仓根，搬到 `old/` 后根就变成了 `old/`）。
   已核：**判据自己不做任何文件操作**。处置：把判据**复制**回 `checks/`（`old/` 保留），
   并让 `run_checks.py` 同时认 `checks/` 与 `old/checks/`。
2. 另一个人做了 `src/lab/` 适配层（2188 行）——**已否决并归档到 `old/lab/`**。
3. 本 README 一度被改成"描述 `src/lab` 存在、`src/core` 25 个文件原样"——**与事实不符**，已重写。

**在这里改动前，先确认没有别的会话正在同一个目录里干活。**

---

## 改造纪律

1. **先证明等价，再换路**——逐位对拍（`max|Δ| = 0`），不接受"接口通了"。
2. **覆盖只迁移，不删除**——判据换锚到新结构，子条件只增不减。
3. **每步一个可回滚提交**，先跑基线绿再动手。
4. **不套层**——改引擎本身，不在外面加适配皮。
5. **注释写事实，不写编年史。**
6. **跑基线期间不动引擎**——曾因此拿到一次"假红"（半途修改的树）。
7. **跨文件重名 0 处、死 import 0 处、判据的私有依赖 4 项（白名单化）**——
   这三条由判据强制（`engine_contract_selfcheck` 的 E/F 包），不要绕过它们。
