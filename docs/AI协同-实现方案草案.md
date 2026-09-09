# AI 协同 — 实现方案草案（给自己看）

> **实施状态（2026-09-09）**：P0-P5 已全部落地，验证全绿
> （ai_selftest 11/11 · ai_compute_test 17/17 · ui_selftest 57/57 无回归）。
> 与草案的主要偏差：① P1 未做"值入核"大重构，改为 **SimState 门面**
> （读写收口到既有人用路径：控件 setValue / groups_store / set_waves），
> 人机等效照样成立，风险归零；真"值入核"留作后续可选演进。
> ② 确认策略按用户拍板 = 先斩后奏（写直放行 + journal 快照 + undo）。
> ③ 进度框在 AI 发起的计算中静默（防模态卡死主线程），画布/状态栏照常。
> 代码：src/ai/{dispatch,journal,sim_state,server}.py + app_entry._init_ai
> + checks/ai_selftest.py + checks/ai_compute_test.py + skills/energy-level/SKILL.md。
>
> **追加（同日，高效方案三层落地）**：
> ① 固定端口：默认 8765（`EL_MCP_PORT` 可改，被占回退随机+日志提示），
>    harness 配置可写死，SKILL.md 有样板；
> ② 细粒度 MCP 工具保留（交互/逐步监督场景）；
> ③ CodeAct：src/ai/codeact.py + `run_python` 工具——受限命名空间（只注入
>    语义入口函数，与 MCP 工具同名同返回形状）、主线程 exec、
>    run_compute_and_wait 泵 GUI 事件、整段一个 undo 单元 + 一条 journal。
>    验证：checks/ai_codeact_test.py 13/13（两轮 B 场扫描一次调用完成）。

> 顶层依据见《AI协同-项目目标与原则.md》。本文是 Energy Level 项目
> 落地的具体技术方案，含现状盘点、架构、分期与风险。

## 0. 现状盘点（已核实）

| 资产 | 位置 | 与本方案的关系 |
|---|---|---|
| 共享状态雏形 | `src/ui/app_state.py`：`LevelGroupStore`（groups 列表 + `changed` 信号）、`FieldState`（+`changed`） | 状态核的起点；**缺陷**：`FieldState` 持有 `field_panel` 控件引用（读 B/E/光要经控件），`LevelGroupStore` 干净 |
| 序列化 | 各 UI 组件自治 `serialize_state`/`restore_state`（`ui_group.py:109`、`fields.py:170`、`ui_dynamic.py:671`），`app_entry.py:232` 装配 | 快照语义、真值源在控件里——**要反转**：状态入核，序列化降级为核的导出 |
| 长计算 | `src/ui/evolve_worker.py`：QThread + `progress_arr` 轮询 + `done/log_msg` 信号 | 可直接复用为 job 执行器；AI 侧包一层 job 登记表即可 |
| 校验 | `core.validate_state(n,l,j)` 等 | AI 写入路径必须过同一校验（人机等效的"法"） |
| 存档 | `save_state/*.json` | schema 即语义状态 schema 的蓝本 |
| 验证环 | `ui_design/`（截图 + 几何断言）、`checks/ui_selftest.py` | Phase 3 后扩展"AI 操作 → GUI 断言"用例 |

## 1. 目标架构

```
                 ┌──────────────────────────────┐
   人 ──GUI──►   │        SimState（状态核）      │  ◄──MCP tools── AI harness
                 │  语义参数唯一真值源 + 校验      │
                 │  每次变更 → OpJournal(带actor) │
                 └──────┬────────────┬───────────┘
              changed信号│            │changed信号
                 ┌──────▼──────┐ ┌───▼─────────────┐
                 │ PyQt5 控件层 │ │ MCP server(进程内 │
                 │ （渲染+反向写 │ │  HTTP端点）渲染+  │
                 │   回，带actor)│ │  写入，actor=ai） │
                 └─────────────┘ └─────────────────┘
                        └──── EvolveWorker / JobRegistry ────┘
```

要点：
- **GUI 与 MCP 是对称的两个"外设"**：都只通过 store 的语义方法读写，
  都带 actor 参数（`user` / `ai:<session>`），都订阅 `changed` 广播。
- **Qt 线程亲和**：MCP 请求在处理线程收到后，经信号
  （`Qt.QueuedConnection`）marshal 到主线程执行 store 变更，回包用
  `threading.Event` 等待结果。store 本身只允许主线程写（断言保护）。

## 2. 关键设计

### 2.1 SimState（状态核）

- 新建 `src/state/sim_state.py`：纯数据 + 语义方法，**不 import PyQt5**
  （信号经一个极薄的 Qt 适配子类，或注入回调）。
- 内容 = 现 `save_state/*.json` 的全部语义字段：能级组、B/E 场、地址光、
  动态页参数、波形序列、初态、耗散通道、引擎覆写（上帝模式）。
- 迁移策略：`FieldState` 的控件引用摘除（B/E/光值入核，控件改订阅渲染）；
  `LevelGroupStore.groups` 入核；`ui_dynamic` 的 dyn_params/waves 入核。
- 序列化：`sim_state.to_dict()/from_dict()` 成为唯一实现，各 UI 组件的
  `serialize_state/restore_state` 退役为"控件↔核"的绑定代码。
- 边界（P6）：核内只有语义状态；缩放/配色/窗口态留在控件层，不进核。

### 2.2 OpJournal + 撤销

- `src/state/journal.py`：`Op = {seq, ts, actor, kind, path, before, after, reason}`。
- store 每个语义方法内部记 before/after → 追加 journal → `changed.emit()`。
- 撤销 = 反向应用（栈语义；Phase 2 先做整 op 撤销，不做字段级合并）。
- journal 同时是：GUI"AI 动态"面板数据源、MCP `get_activity` 工具数据源、
  调试日志。落盘 `save_state/_journal/<日期>.jsonl`（可配开关）。

### 2.3 MCP server（进程内，HTTP）

- `src/ai/mcp_server.py`：用官方 `mcp` Python SDK 的 **streamable HTTP**
  transport，随 app 启动在 `127.0.0.1:<port>`（默认 8765，端口写进
  `.mcp-port` 文件供 harness 配置/脚本发现）。
  - 选 HTTP 而非 stdio 的理由：stdio 要求 harness 拉起子进程，与"软件先启动"
    矛盾；HTTP 让 app 自己当 server，harness 配置 URL 即连，天然支持
    "软件启动、harness 随后启动"。
- 工具集（v1）：
  | 工具 | 说明 |
  |---|---|
  | `get_state(scope?)` | 导出语义状态（可按模块裁剪，防灌 context） |
  | `set_param(path, value, reason?)` | 白名单路径写（如 `field.B_z`、`dynamic.t_end_us`），过核内校验 |
  | `add_level_group(n,l,j)` / `remove_level_group(idx)` | 复用 `validate_state` |
  | `set_waveform(...)` / `add_waveform(...)` | 脉冲序列编辑 |
  | `run_compute(config?)` | 用当前状态起 EvolveWorker，立即返回 `job_id` |
  | `job_status(job_id)` / `job_result(job_id)` | 进度百分比/日志；结果：布居曲线**落盘 npz/json + 摘要统计**（保真度、耗时、路由理由），大数组只给路径 |
  | `cancel_job(job_id)` | 复用 progress_arr 取消位 |
  | `get_activity(since_seq)` | 读 journal（人侧也可读） |
  | `undo(n?)` | 撤销最近 n 个 op（默认只撤 AI 自己的） |
  | `save_archive(path)` / `load_archive(path)` | 走核序列化 |
  | `screenshot(target?)` | 可选；复用 ui_design 截图机制，供 AI"看一眼" |
- 写类工具统一 `actor="ai"`；`reason` 进 journal，人侧可见。

### 2.4 GUI 侧协同表现（G3 的 2、3、4）

- **状态栏"AI 动态"条**：订阅 journal，滚动显示"AI: set_param field.B_z
  12.5→13.0（原因…）"。
- **控件级高亮**：`changed` 广播携带 actor；被 AI 改的控件短暂描边/徽标
  （如 1.5s 紫色边框），人改的不高亮（避免自闪）。
- **运行中锁**：AI 或人任一方发起计算时，参数面板进入只读提示条
  "计算中（发起人：AI）"，完成解锁——复用现有 worker 流程。
- **接管**：人随时可点撤销按钮撤 AI 最近操作；Phase 4 加"AI 请求确认"
  模式（危险 op 如覆盖存档/超长计算弹确认框，MCP 侧阻塞等待）。

### 2.5 Skill 文档（G4）

- `skills/energy-level/SKILL.md`：工具清单、典型工作流（"复现 STIRAP：
  add_level_group×3 → set_waveform×2 → run_compute → job_result"）、
  单位约定（MHz/μs/Gauss）、坑（引擎路由、截断判据）。
- 放项目内，harness 侧配置指向；AI 使用中可 PR 式自改（迭代环）。

## 3. 分期

| 期 | 内容 | 验收 |
|---|---|---|
| **P0 探针**（不重构） | 进程内 MCP server + 只读工具 `get_state`（临时从现有 serialize_state 拼）/`screenshot` | harness 里 AI 能答"当前能级组是什么" |
| **P1 状态核** | `src/state/sim_state.py` 落地；FieldState 摘控件引用；ui_dynamic 参数入核；序列化收口 | 现有 `checks/ui_selftest.py` 全绿；GUI 行为无变化（纯重构） |
| **P2 写通道+日志** | journal/actor/undo；`set_param`/`add_level_group` 等写工具；Qt 线程 marshal | 脚本模拟 AI 写入 → GUI 值变化 + journal 正确 + undo 还原；新增 selftest 用例 |
| **P3 协同 UI** | 状态栏 AI 动态、控件高亮、计算锁 | 手工 + ui_shot 断言：AI 改 B_z 后控件边框出现、状态栏有记录 |
| **P4 计算通道** | `run_compute`/`job_*`/`cancel_job`；确认模式 | AI 端到端：改参数→跑演化→拿摘要→人眼看 GUI 同步 |
| **P5 skill 化** | SKILL.md + 典型工作流回归脚本 | 新会话 AI 仅凭 skill 文档完成一次完整"设定→计算→解读" |

P1 是最大块（触及 fields/ui_dynamic 内部），但也是其余一切的前提；
P0 可先行验证 harness 连接体验，成本约一天级。

## 4. 风险与对策

1. **FieldState 控件耦合比想象深**（fields.py 读控件值）→ P1 先做
   fields 面板的"值入核、控件双向绑定"试点，模式验证后再推 ui_dynamic。
2. **Qt 跨线程**：MCP 线程直接碰 store 会崩 → 所有写路径强制
   queued-signal marshal + 主线程断言；P2 加并发 selftest。
3. **大结果灌爆 context**：布居曲线可达 MB 级 → 结果工具默认返回
   摘要+文件路径，`job_result(full=true)` 才给数组。
4. **AI 误操作覆盖人工作**：存档类 op 一律走 journal 可 undo +
   写前自动快照 `save_state/_autosnap/`（保留最近 N 份）。
5. **harness 更换**：全部走标准 MCP + SKILL.md，不写私有能力；
   P0 起即用官方 SDK。
6. **PyInstaller 打包**：HTTP 端口/`.mcp-port` 文件在冻结环境路径处理，
   P4 验证（非阻塞项，开发期不打包）。

## 5. 开放问题（待定夺）

- 确认策略默认值：v1 是"AI 写操作全部直放行（可 undo）"还是
  "部分 op 需人点头"？建议 v1 直放行 + undo，P4 再加确认。
- 多窗口/多项目实例：一个 app 一个端口，harness 多 server 配置可接
  多实例——暂不做协调层。
- 视图状态要不要给 AI 只读（如"当前缩放到的能量区间"）？建议 P4 后按需。
