# 上帝模式的 AI 接入 — 设计（t82 构思）

> 前置阅读：`docs/AI协同-项目目标与原则.md`（P1~P6 顶层依据）、
> `docs/AI协同-实现方案草案.md`（v3 换轨：登记册即存档、AI 可及面 = 在册面）、
> `docs/参数管理改造-议题清单.md`（议题 0 三轴分离、议题 3 消灭平行清单）。
> 本文把"上帝模式"这个**界面模式**映射到那套人机等效原则上，给出可落地的
> 分期方案。**本文件只定架构与接口；已落地部分逐项标注。**

---

## 1. 先把三样东西拆开（否则必然违反 P5/P6）

"上帝模式"一词混了三类东西，AI 接入必须先分类：

| 类别 | 具体内容 | 按原则该怎么处理 |
|---|---|---|
| **语义状态** | 引擎参数覆写（`PHASE_TOL`/`PHASE_BUDGET`/`R_CRIT`/`MW_MAX_MHZ`…） | P5：唯一真值源，必须入册；AI 可读写（经 MCP `set_param`） |
| **只读观测** | 运行诊断：AC 化分类、M 判定、扇区结构、耦合对清单、H(t) 快照、步长历史、扇区轨迹（`out['diag']`） | P1：AI 要"理解为什么"，但这是**读**通道；单独工具、默认摘要 |
| **视图状态** | 模式开关、检视器窗口显隐/位置/滚动、面板折叠、配色 | P6：AI **默认不碰**（人负责）；最多允许"查询当前模式" |

**关键判断**：上帝模式的 AI 价值 90% 在第二类（诊断可见 → AI 能解释
"为什么这次跑得慢/为什么这条路径死了"），第一类只是把已有的引擎参数接到
现有 `set_param` 上，第三类基本不该给 AI。

## 1b. 统一命名空间（t83：让"旧的入册参数"与"新的控制参数"同构）

上帝模式的定位是**凌驾各页的模式**，检视器只是它的"监视器"；将来会有更多
**真正的控制参数**（L2/L3：逐载频 M、AC 化逐对手动 keep/AC、哈密顿注入项、
手动管线分步…）。为免每加一个就新开一条通道，统一成：

| 约定 | 内容 |
|---|---|
| **键** | 会话 store 的键 = 注册表全路径 `<group>.<NAME>`（`engine.PHASE_TOL` / `solve.MAX_STEP`） |
| **分组** | `engine.*` 数值精度/成本（params.py 的 SPECS 32 项）；`solve.*` 求解器级 kwarg；规划 `acize.*` / `inject.*` / `pipe.*` |
| **登记** | 一条通道 `pm_setup.register_god_group(reg, win, group, editors=…)`——登记 + 可选绑控件，**幂等**（已登记不重复 register，免得洗掉灯的 widget 绑定） |
| **真值** | 一个 store：`MainWindow.engine_overrides`（全路径键；兼容旧裸键） |
| **映射** | 一个映射点 `pm_setup.god_split(store)` → `(param_overrides, engine_kwargs)`；新分组只改这里 + `_SOLVE_SPECS` |
| **下游** | 数值容差 → `params.override` 快照；求解器 kwarg → `engine_kwargs` |
| **灯** | 框架原生：**注册必须绑控件**（`_paint()` 在 `widget is None` 时直接返回——只注册不绑=永远没灯）；人输入=黄灯、提交后清（项目既有约定）；AI/程序写=绿灯+来源紫 |
| **非法输入** | 编辑期**自愈**：立刻回退到上次有效值 + 计数行红字说明 + tooltip 给原因；不弹模态、不写 store、不牵连其它项 |
| **一路径一控件** | 同一参数出现在两页时，第二页做**镜像**（求解页 r_crit/tol 镜像 `engine.R_CRIT`/`engine.DEFAULT_RTOL`），不抢框架的 widget 绑定 |

已按该约定收敛的**平行通道**：`r_crit` / `tol` / `max_step` / `per_freq_M`
原先各走 `_last_god_kwargs → engine_kwargs`，现分别是
`engine.R_CRIT` / `engine.DEFAULT_RTOL` / `solve.MAX_STEP` / `solve.PER_FREQ_M`
（与引擎参数页同一批注册项）；旧存档的平铺 kwarg 由唯一映射表转全路径。

## 2. 已落地（t82/t83）

### 2.1 引擎参数入册 = AI 自动可及

`src/pm_setup._reg_engine_params` 把 `src/core/params.py` 的 `SPECS`
逐条登记为 `engine.<NAME>`（32 项）：

- **真值源**：`MainWindow.engine_overrides`（会话覆写）叠加
  `params.DEFAULTS`；`read_fn` 返回"覆写优先"的当前生效值；
- **AI 路径**：`set_param('engine.PHASE_TOL', 1e-3, reason=...)` →
  框架 `set()` → 校验 → `write_fn` → `win.set_engine_overrides` →
  检视器页即时回填（人看得见 AI 在改什么，P2/P3）；
- **人的路径**：检视器「引擎参数」页 → `engine_overrides_changed` →
  同一个 store → `reg.note_change('engine.X', source='user')` 记账；
- **与模式无关**：登记跟真值源走，不跟显隐走（议题 0 三轴分离）——
  极简模式下这些参数照样在册、照样生效；
- **不进 range 硬校验**：`SPECS.lo/hi` 只是提示，框架 `range` 会拒绝越界
  值，探索性取值不该被拦（已并进 `desc`）。

已验：`checks/gui_mode_switch_test.py` 第 6~11 项 + 第 10 项
"engine.* 全部在册 == 可调参数数"。

### 2.2 模式开关与"AI 不改视图"

模式切换是 `MainWindow.set_mode(god)` 唯一入口（菜单栏行分段按钮 →
`dynamic.set_god_mode`）。**不做成 MCP 工具**：它是视图状态（P6）。
若将来确有需要，应做成"提示"而非"设置"（见 §3.3）。

## 3. 待落地（按价值排序）

### 3.1 诊断读取通道（✅ 已落地 t83）

已实现（`src/ai/server.py`）：

| 工具 | 返回 |
|---|---|
| `get_diag(scope='')` | 摘要：meta（N/P/dim/t_end/tol/mw_cap/M…）、acize（逐场 n_before/n_kept/位移范围）、photon（Ms/P/dim/strides/nvecs）、pairs（对数 / 微波与光场-V_dd 计数 / max\|v\| / ΔD 统计）、solver（nfev/n_acc/n_rej/norm_dev/dt 分位数） |
| `get_diag(scope=pairs\|steps\|sectors\|traj\|H, limit=N, save_npz=)` | 分项明细：pairs 按 \|v\| 降序带态标签；steps 给 dt 分位数 + 降采样；sectors 给 D_full 统计；traj 给逐扇区幅值；H 给形状或落盘 npz 路径 |
| `run_compute(diag=True)` | AI 发起的运行**默认采诊断**（人发起仍跟界面模式，不让人无谓付成本），由 `dyn._ai_wants_diag` 承载 |

自检 `checks/ai_diag_test.py`（16 项）：用**真实跑出来的 diag** 逐项核对摘要
数值、明细截断/排序/标签、错误路径；物理锚 = freeze 三光子梯子末级
58F→58G 的 \|v\|>1 GHz、Δ≈7.08 GHz 在 pairs 里可见（诊断能回答"路径是不是
被删了"）。

以下为原始设计要点（保留）：

现状：`out['diag']` 只在 `diag=True` 且 mixed 引擎时产出，且**只喂 GUI**；
AI 拿不到（`job_result` 只给布居摘要）。

建议新增两个只读工具：

| 工具 | 返回 |
|---|---|
| `get_diag(scope='')` | 最近一次运行的诊断**摘要**：AC 化（r 判据分类计数、位移范围）、M 判定（逐载频 M + r）、扇区结构（P/dim/strides）、耦合对统计（保留/成本剔除/相位剔除）、步长历史分位数、归一化偏差 |
| `get_diag(scope='pairs'\|'sectors'\|'H'\|'steps'\|'traj', limit=N)` | 分项**明细**（默认截断 + 落盘 `.npz` 给路径，与 `job_result(save_npz=True)` 同纪律） |

要点：
- **摘要默认、明细要显式要**（草案 §4 风险 3：大数组灌爆 context）；
- 与 `job_result` 一致：大数组给文件路径，不直接回灌；
- **不要求先开上帝模式**：`diag` 的采集成本由"是否需要诊断"决定，而不是
  由界面模式决定。建议策略：`run_compute(diag=None)` 时 **AI 发起的运行默认
  采 diag**（AI 的用途就是要解释），人发起的按界面模式（P6 精神：别让人
  无谓地付成本）。这一条需要新增 `run_compute(diag=...)` 参数。

### 3.2 引擎参数的"作用域"语义（避免一个坑）

现在 `engine.*` 是**会话级**覆写（跑完不还原，存档会带上——因为
"登记即存档"）。要明确两件事并写进 SKILL.md：

1. **运行级 vs 会话级**：AI 若只想"这一次试试"，应走
   `run_python` 里的 `with params.override({...})`（线程本地快照，退出即
   还原），而不是 `set_param('engine.X')`（会改会话真值、进存档）；
2. **存档会带走引擎参数**：这是特性（算例可复现），但 AI 覆盖前应
   `save_archive` 或依赖 `undo`（P3 已具备）。

建议在 `engine.*` 的 `desc` 里加一句"改动进会话与存档；一次性的请用
CodeAct 的 params.override"，并且 `set_param` 的 `reason` 强制非空
（现状可选）。

### 3.3 视图状态的读权限（低价值，做的话只做"读"）

`get_state(scope='view')` 返回 `{god_mode, inspector_visible, active_tab}`。
用途很窄：AI 想判断"人是不是正在看检视器的耦合对页"来决定解释的详略。
**不提供写**。若将来真需要 AI 主动拉起检视器（例如它想让人看某张图），
正确形态是"**建议**"而非"设置"：`suggest_view(tab='pairs')` → GUI 弹一条
可点确认的提示条（人点头才切）——这样才不违反 P6 与 G3（人可随时接管）。

### 3.4 SKILL.md 补章

给 AI 写清"上帝模式能干什么、我该怎么用"：
- 何时看诊断（结果反直觉 / 布居零变化 / 耗时异常）；
- `engine.*` 清单与典型调整场景（例：freeze 三光子路径消失 →
  先查 `get_diag(scope='pairs')` 看第三级是否被剔，再考虑调
  `engine.PHASE_BUDGET`）；
- 一次性 vs 会话级的写法（§3.2）；
- 坑：`engine.*` 不是"精度越高越好"——`PHASE_BUDGET` 调大直接换运行时。

## 4. 分期与验收

| 期 | 内容 | 验收 |
|---|---|---|
| **A（已做）** | 引擎参数入册 `engine.*` + 检视器页三向绑定 | `gui_mode_switch_test` 14 项；`ai_params_slim_test` 不回归 |
| **B** | `get_diag(scope)` 摘要 + 明细（截断/落盘）+ `run_compute(diag=)` | 新 `checks/ai_diag_test.py`：跑一次 freeze 场景，断言摘要含三级梯子保留数；明细返回路径 |
| **C** | SKILL.md 补章 + 一条"AI 读诊断→调引擎参数→复跑"的端到端脚本 | 新会话仅凭 skill 完成：布居零变化 → 诊断定位 → 调整 → 复跑出结果 |
| **D（可选）** | `get_state(scope='view')` 只读；`suggest_view` 建议通道 | 手工 + 截图断言 |

## 5. 与原则的逐条对照（自检表）

| 原则 | 本设计如何满足 |
|---|---|
| P1 语义优先 | 诊断走结构化 `get_diag`，不是截图；引擎参数走 `set_param` 而非点按钮 |
| P2 同一状态 | 人（检视器页）与 AI（`set_param`）写同一个 `engine_overrides`；读同一个 `reg.read` |
| P3 可逆可归因 | 走框架 `set()`/`note_change` → 灯 + op + journal + `undo`；AI 的 `reason` 可见 |
| P4 协议中立 | 只加 MCP 工具与在册参数，无私有通道 |
| P5 单一真值源 | 引擎参数真值 = `engine_overrides`（+ 默认值派生）；GUI/MCP 都是渲染器 |
| P6 状态边界 | 模式开关/窗口显隐归视图状态，AI 不写；只读查询也只列最小集 |
