# AI 交互 · 架构设计（本项目）

> 读者：要在这套软件上继续加"AI 可用的面"的人。
> 定位：**只讲"AI 怎么进来、看到什么、能动什么、凭什么可信"**；工程心得与反模式见
> `docs/心得-人机共用的GUI与MCP项目.md`，AI 侧使用体感（18 条实测不适）见
> `docs/MCP使用体感-不适清单.md`，逐项核查与整改见
> `docs/核查报告-文档与代码不一致与整改项.md`。
> 本文的 `file:LINE` 锚点截至 2026-09-16；行号漂移时按符号名搜。

---

## 1. 总览

**一等公民**（AI 与 GUI 共用同一份，不是两套）：

| 一等公民 | 载体 | 锚点 |
|---|---|---|
| 状态册子（**唯一真值源**） | `ParamRegistry`（定义即注册） | `src/pm_setup.py`；面板→注册 `app_entry.py:315` |
| 引擎调用构造 | `runspec.build_run`（登记册快照 → RunSpec） | `src/core/runspec.py:413` |
| 作业 | 作业登记表（`owner=ai\|human`） | `src/ai/server.py:443/237/681/795` |
| 诊断 | diag 契约 + 摘要卡 | `src/core/diag.py:26/94/135` |
| 扫描 | `src/sweep/spec.py`（轴展开）+ `registry.py`（生命周期）+ `runner.py`（执行） | `spec.py:41/171`、`registry.py:144/260/282`、`runner.py:338` |

**AI 从哪进来**（两条通道，同一套语义）：

```
任何 MCP harness ──HTTP──▶ 进程内 MCP server（36 个工具，src/ai/server.py）
                              │  工具 handler 跑在 MCP 线程
                              ▼
                    dispatch.run_on_main()（QueuedConnection → Qt 主线程）
                              │                    src/ai/dispatch.py:58
                              ▼
        SimState 语义门面（src/ai/sim_state.py）──▶ store / 引擎 / 登记表
                              ▲
CodeAct（run_python，同一批函数注入命名空间）src/ai/codeact.py:73
                              ▲
并行扫描：独立 runner 进程 ──spawn──▶ N 个 worker     src/sweep/runner.py:1（docstring）
```

三条通道的**分工**与代价（选择准则）：

| 通道 | 语义 | 代价 / 边界 |
|---|---|---|
| 细粒度工具（36 个） | 一步一调用，人在 GUI 实时看见 | 每步一个回合；写入会占主线程（`server.py:459` 会报"占用作业"） |
| `run_python` / `run_python_async`（CodeAct） | 一次调用多步编排（扫描/后处理） | 脚本**在主线程 exec** ⇒ 长任务必须用 async 版（`server.py:1209`），否则别的工具被排队 |
| `submit_sweep`（并行扫描） | 声明 N 个参数点，多进程算完 | 走**独立 runner 进程**（见 §2.4）；结果按点落盘，回程只给摘要 |

---

## 2. 接口面设计

### 2.1 可改面 = 在册面（`list_params` 是唯一权威）
- 参数只声明一次：schema = UI 绑定 = 存档 = MCP 可及面 = 撤销单元；引擎读取走
  `override()`，人的写入走 `write_fn`（`src/pm_setup.py`）。
- **显示类字段不入册**（图例色/展开态/读出权重）⇒ 不入存档、AI 改不到 ⇒
  "在册面 = AI 可及面"这条不变量自动成立。
- 不用它会怎样：AI 能改的集合与人在界面能改的集合悄悄分叉；存档与 UI 各存一份。

### 2.2 语义状态面（**不让 AI 猜索引/能量**）
| 工具 | 语义 | 锚点 |
|---|---|---|
| `list_levels()` | 只列**当前基组内合法**的身份键 `(kind,n,l,j,m/F,m_F)` | `sim_state.py:267` |
| `set_initial_state(atom,n,l,j,m)` | 按身份设初态；**不在基组就报错**，绝不静默回落 0 号态 | `sim_state.py:281` |
| `set_groups(groups)` | **声明式整表**：原子 + 幂等；回 `added/removed/kept` + **`resolved`（每组实际子能级数与身份键）** | `sim_state.py:307/449` |
| `set_superposition` / `set_pair_superposition` | 逐原子 / 两体纠缠初态（存"持久真值"，不是 combo 选中项） | `sim_state.py:586/518` |

- **为什么**：AI 拿不到合法清单时必然去猜能量或索引——猜错会静默落到别的态（真实踩过）。
- **`resolved` 是回执**：写了 ≠ 引擎侧真按你说的展开；见 §5 的 E1。

### 2.3 声明式扫描（`src/sweep/`）
- **唯一构建路径**：`runspec.build_run(cache, params) → RunSpec`（`runspec.py:413`）；
  runner 就是"**同一份配置跑 N 次**"，所以**扫描与人点运行同源**（`runner.py:1` 接缝 1）。
- **表达式**：轴键 = **登记册全路径**（`spec.py:41 parse_path`），支持
  `dynamic.t_max` / `field.lights[0].intensity` / `dynamic.waves_A[0].env` / `groups.list`；
  轴值支持 `[a,b,c]` / `linspace` / `arange` / `logspace` / `value`（消歧义）→
  `build_points`（`spec.py:171`）做**笛卡尔积**（轴名排序，最后一个轴最快变化）。
  **AI 只需声明"哪个参数取哪些值"**，不该自己写网格展开与结果归并。
- **生命周期**：`submit`（立即返回，`registry.py:144`）→ `status`（逐点，`:260`）→
  `result`（`:282`）→ `cancel`（`:417`，先优雅后强杀）→ `list_sweeps`（`:362`）。
- **独立 runner 进程的意义**：① 不占 Qt 主线程；② **内存随进程归还**（多体/长演化）；
  ③ runner 崩了软件照常活，且能 headless 单跑复现（`runner.py:1`）。
  已知约束（同 docstring）：需要非 frozen 解释器；子进程强制单线程
  （`runner.py:115`）；**ARC 的 SQLite 多进程并发建表会 `database is locked`**
  ⇒ 父进程先建缓存/预热再放 worker。

### 2.4 大数组的"专用通道"
包络采样不进工具响应：摘要只给形状/量程/步长，完整曲线走 `get_envelope(path,index)`
（`server.py:85-105/165`）。作业全量数组走 `job_result(save_npz=True)` / `result_states`
落 npz 给路径（`server.py:795/840`；保留最近 `_NPZ_KEEP` 份，`server.py:775`）。

---

## 3. 契约与不变量（AI 依赖的硬契约）

1. **diag 键名契约**：`meta / acize / photon / pairs / solver / trajectory /
   H_snapshots`（写进 `src/ui/solver_inspector.py` 顶部 docstring）；
   摘要卡面向 AI（`diag.summarize` `diag.py:94`，告警 `warnings` `:135`）。
2. **作业登记**：`owner=ai|human`（AI 发起的在 `server.py:467` 标 `ai`；GUI 里的在
   `:258` 以 `(gui)/human` 出现），`done` 一到即落盘 json + npz（`:518-528`）；
   `thread_finished` **只记状态、不参与线程生命周期**。
3. **进度协议**：`progress_arr = [阶段内进度, 取消标志, 阶段]`——引擎内核、GUI、扫描
   共用同一条；取消 = 置 `[1]=1.0`（`runner.py:1` 接缝 2）。前置条件：内核 `nogil`。
4. **代码新鲜度**：`summary()['code']['stale']`（`src/ai/code_freshness.py:88/133`）+
   CodeAct 返回 `stale_code_warning`（`codeact.py:271`）——进程内代码改了必须重启，
   否则 AI 会拿"旧行为"当"新证据"。
5. **产物路径唯一真值源**：`src/ai/paths.py:13-16`（runtime/、_journal/、ai_results/、
   scratch/）；不要在别处拼路径。
6. **反向契约（同样重要）**：**契约里没有的字段，读出来永远是 `None`** ⇒ 告警哑掉。
   实例：`diag` 曾读 `n_microwave`（契约里根本没有 ⇒ 三个字段全 None、"微波候选对为 0"
   的告警**从不触发**；修法 = 按契约键 `is_mw` 现算，`diag.py:26`）；
   `n_drop_cap/n_drop_phase` 也曾**只有读、没有写**（哑告警；现在写入侧是
   `n_drop_bw/max_drop_bw_abs_v_mhz`，`evolve_mixed.py:1102-1106`，告警 `diag.py:175`）。
   **判据**：任何"读 meta/契约键"的代码，都要能指出**写它的那一行**。

---

## 4. AI 可观测性与可归因

- **操作流水 journal**：`Op{seq,ts,actor,kind,detail,reason,snapshot}`（`journal.py:107`），
  每条带来源与动机；`get_activity(since_seq)` 读（`server.py:1159`）；
  **detail 有 4 KB 体积闸**（超限只留形状/长度/前 8 项 + `sha1`，`journal.py:30/81`）。
- **撤销**：`undo()` 回滚"最近一个 AI op 的变更前快照"（`server.py:1163`；
  `sim_state.restore_snapshot` `:816`）——粒度粗但绝对正确。
- **扫描算一步**：`submit_sweep` 在流水里是一条 op；逐点状态在 `status.json`
  （GUI 侧有逐点进度窗，见 `docs/并行扫描-设计与验收.md`）。
- **AI 看人眼所见**：`screenshot()`（`server.py:1175`）——**接口只给路径**，需要
  harness 自带读图能力（体感 #18）。
- **长任务断线可找回**：`run_python_async`（`server.py:1209`）→ `run_result(run_id)`
  （`:1240`）/ `list_runs`（`:1265`）；记录落 `runtime/ai_results/`（`src/ai/run_store.py`）。

---

## 5. 失败模式与防护（现象 → 根因 → 现在怎么办 → 锚点）

| # | 现象 | 根因 | 防护 / 锚点 |
|---|---|---|---|
| 1 | 诊断告警从不触发（看起来"没问题"） | 读**契约外键** ⇒ 永远 `None` | 按契约键现算（`diag.py:26`）；"读键必须能指出写它的行" |
| 2 | 自检"全绿"但某守卫**从没跑过** | 位置/导入依赖 ⇒ 该守卫**永久 SKIP** | 例：镜像一致性守卫改为 **AST 解析源码取常量**（`checks/sweep_spec_selfcheck.py:235-262`）⇒ 换机器/拷贝也活着 |
| 3 | 并行/多进程下 `no such table: literatureDME / hfsDataAB` | ARC 的 SQLite **首访 CREATE TABLE**，多进程同时来就锁死（实测不预热只成 1/14） | **父进程先建基组+场+V_dd 并预热**，子进程只跑 numba（`runner.py:1` 已知约束） |
| 4 | AI 发起的运行被记成 `owner=human`；作业完成后 `elapsed_s` 还在涨 | 归属字段没在**发起处**写；耗时按"当前时间"现算 | AI 路径显式标 `owner='ai'`（`server.py:467`）；终态冻结耗时（登记表 + `thread_finished`） |
| 5 | 一条扫描结果把 AI 上下文灌爆 | 结果默认**全量回传**（含逐点数组） | `sweep_result(with_points=True)` 可关；逐点摘要 + npz 路径（`registry.py:282`）；同理 `params` 截断 40 条（`codeact.py:47`） |
| 6 | 新加的工具报 `unknown tool`（**软件没在跑时也报同样错**） | MCP 客户端的**工具表在会话启动时加载**，运行期新增不可见 | 工具面变更后**重开会话**；别把这条错当"服务端没起"（`@mcp.tool` 共 36 个，`src/ai/server.py`） |
| 7 | 大响应被上层截断，而 AI 不知道 | 回程某些字段没走同一把体积闸（实测：`list_params.value` 292 B，而 `last_committed` 241041 B） | 已知未清（建议与锚点：`docs/AI协同-分工与请示.md:339-346`、`SKILL.md:620-633`）——**同一份数据的安全口径必须一致** |
| 8 | 脚本里 `get_envelope(...)` → `NameError` | CodeAct 命名空间与工具面**不同集**，且白名单里没写"不在其中的重要名字" | 已知未清（`codeact.py:73`；体感 #1）。**边界要么能用、要么写明** |

**三条最贵的**（都在真实事故里付过代价）：
- **写入侧通了 ≠ 引擎侧通了**：稀疏分支 `Kc_dense=None` ⇒ 存档/复选全对、光场耦合
  化为乌有（加一个远组：配对数 1206→150、观测量 0.8956→0.0060）。**验收必须打到引擎入口**。
- **假绿比红更危险**：永远 `SKIP` 的自检、"SKIP 也算通过"的聚合口径。
- **性能阈值绝不能改模型**：成本预算曾**静默删耦合对**（三光子梯子第三级被删 ⇒
  布居恒 0、零报错）⇒ 现在成本剔除要**大声报出来**（`diag.py:175`）。

---

## 6. 人机协同约定（可写性与可见性）

- **AI 写入即人可见**：控件闪紫框 + 状态栏流水（含 reason）；AI 发起的计算走
  **与"人点运行"同一条路径**（`runspec.evolve_run`，`runspec.py:580`）。
- **AI 路径不许弹模态**：模态框会卡死主线程 ⇒ AI 上下文把 `QMessageBox` 降级为
  状态栏日志（`server._QuietModals`，`server.py:379`；用在 `:495/:503`）。
- **人是主人**：软件实例生命周期（启动/重启/端口）由人拥有；**自检绝不抢活实例端口**
  （`EL_NO_AI_SERVER` + `.mcp-port` 归属判定，`server.py:1381/1403/1416`）；
  offscreen + `EL_NO_AI_SERVER=1` 是自检的固定边界。
- **可撤销**：AI 的每一步都可 `undo`；破坏性恢复前自动快照兜底。

---

## 7. 给后来者的设计准则

1. **先设计"状态册子"，再让 GUI 与 AI 都当消费者**；不要从 UI 控件倒推状态。
2. **每个写入面都必须配读回面 + 一个能"判死"的自检**（含故意弄坏的致死性证明）；
   没有读回的写入 = 没接线的开关。
3. **契约外字段就是会撒谎的字段**：读它之前，先指出**写它的那一行**。
4. **假绿比红更危险**：SKIP 要显式计数；"没跑过"和"跑过了"必须能从摘要里区分。
5. **独立进程是并行设施的正确隔离边界**（Qt 进程里塞进程池 = 把两族风险叠在一起）。
6. **性能阈值绝不能改模型**：任何"为了快"的丢弃都必须**大声报告**或交给人选。
7. **大对象永远给"通道 + 路径"**：回程只给摘要与指针，完整数组按需另取
   （且**所有**回程口走同一把体积闸）。

---

## 附 A：AI 侧体感清单的架构归类（引用不重复）
`docs/MCP使用体感-不适清单.md`（18 条 + 复现清单）——“让我白烧回合”的是
#1（CodeAct 缺 `get_envelope`，对应 §5-8）、#2/#3（回程安全口径不一致，§5-7）、
#4（错误消息不给别名线索）、#10（作业记录不含配置 ⇒ 无法复现）、#11（提交前看不到
"这条脉冲会驱动什么"）。它们分别落在：**命名空间与工具面同集**、**回程统一体积闸**、
**错误消息可教学**、**作业自带上可重跑配置**、**提交前行为预览** 五个可加的面上。

## 附 B：加一个新 AI 面时的清单
1. 它在**登记册**里吗（`list_params` 能看到、能写、能存档）？
2. 有**语义入口**吗（别让 AI 传索引/能量）？回执里有**可自检的量**（如 `resolved`）吗？
3. **读回面**是什么？它读的键**谁写**（§3-6）？
4. 大数组走**通道 + 路径**了吗？回程会不会灌爆上下文？
5. 失败时**报错还是静默降级**？（默认必须报错）
6. 有**自检 + 致死性证明**吗？会不会**永久 SKIP**？
7. 人与 AI 是**同一条代码路径**吗？
