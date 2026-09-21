# AI 机甲框架 · 开发计划（Re0 重构）

> **定位**：把《AI机甲框架-开发原则.md》落到 `D:\desktop\Re0重构` 这张白纸上。
> **依据**：纲领 = `docs/AI机甲框架-开发原则.md`；Re0 现状 = `README.md` +
> `docs/引擎重构总览与调用说明书.md` + `docs/语义层-我们有什么缺什么.md` +
> `docs/引擎层该有状态管理吗.md` + `docs/params归属-分析与管理方案.md`。
> **范围**：只针对 Re0 沙盒（仿真档）；硬件档在 Phase 5 只预留接口，不实现。

---

## 0. 前提与四条硬约束

Re0 已经把引擎打磨成**纯函数 + 唯一分派点**（`runspec.build_run → RunSpec →
runspec.evolve_run`），21 判据 / 153 项 / 0 FAIL，且 `src/ai`·`src/ui`·`session`·
`kernel` 全部剥离。本计划在此之上**只加"引擎之外"的语义内核与投影**，遵守：

1. **core 不反向 import 新层**（否则成环）——`docs/params归属` §2 的硬约束。
2. **状态管理住在引擎之外**——`docs/引擎层该有状态管理吗.md` 结论：引擎=纯函数+记忆化。
3. **不套层**（纲领 §13 判据）：新层必须是**单一定义点**，不造并行第二真源；
   `old/lab/`（2188 行）因造并行 DISPATCH/spec/Result 被否决，是反面教材。
4. **判据基线常绿**：任何动作先跑 `run_checks.py`，绿了才继续；新层配**带致死性证明**的判据。

---

## 1. 现状盘点：已具备 vs 缺口

| 框架要素（纲领） | Re0 现状 | 缺口 |
|---|---|---|
| 计算 choke point | ✅ `runspec.evolve_run` 唯一分派（`_ENGINE_TABLE`） | — |
| 结果契约统一 | ✅ `res['states']`/`res['pops']`（六引擎同键，第1刀） | — |
| 语义寻址（不猜索引） | ✅ `state_key`/`find_state_index`/`resolve_psi` **绝不静默回落** | — |
| 不静默降级（D5/I6） | ✅ `build_run.warnings`、`_required` 报错、`_filter_kw` 报 dropped | — |
| 自检 + 致死性（D4） | ✅ 21 判据；`engine_contract`/`runspec_contract`（缺键不静默） | 需加**新层**判据 |
| 大数组不内联（I3） | ✅ `summarize_run`/`result_summary` 不搬数组 | 缺 **npz 落盘 + 指针 + sha1** 纪律 |
| 成本预估 | ✅ `cost.py` | 缺**行为预览**（这条脉冲驱动哪些跃迁 / 基组缺谁） |
| 诊断"为什么"通道 | ✅ `diag.py`（仅 mixed 家族产出） | — |
| **参数元数据单一定义点（C1）** | ❌ `params.py` 已删（第11刀），label/unit/range/desc **无家** | **最大缺口** |
| **写 choke point（`set()`）** | ❌ `params` 只是裸 dict 在函数间传递 | 缺 |
| **事件总线（append-only journal）** | ❌ 引擎无状态，无"谁改了什么"记录 | 缺 |
| **会话态（session）** | ❌ 无（RunSpec 冻结 + 会话覆写该在引擎之外） | 缺 |
| **MCP server（写通道）** | ❌ 已剥离 | 缺 |
| **订阅端点（读通道）** | ❌ | 缺 |
| **驾驶舱（独立只读投影）** | ❌ | 缺 |
| **async runner + 可复现指针（D8）** | ❌ `sweep`/`runner` 已剥离 | 缺 |
| **CodeAct（调语义层，D1）** | ❌ | 缺 |

**一句话**：Re0 已经把"**引擎侧的语义**"（结果契约、身份键、不静默）做扎实了；
缺的全是"**引擎之外的状态与投影**"——registry / 事件总线 / 会话 / 两条通道 / 驾驶舱。

---

## 2. 架构落点（在 Re0 里长什么样）

```
src/core/        纯函数引擎（不动数值；只在 Phase 1 采纳 Param 声明，见 D-A）
   ▲ import（单向）
src/mecha/       语义内核 = 引擎之外的状态与纪律机器
   registry.py   参数元数据【单一定义点】(解 C1) + Param 声明收集
   bus.py        append-only 事件总线（通用事件 schema + 大载荷指针+sha1 + 体积闸 + 轮转）
   session.py    会话态：当前 params 快照 + 覆写；写 choke point set()→校验→发事件
   ops.py        语义操作（run/estimate/preview/undo）经 choke point → runspec
   server.py     MCP server（写通道，localhost）+ 只读订阅端点（读通道，SSE/WebSocket，可远程鉴权）
   codeact.py    CodeAct 命名空间（只暴露 registry/session/ops，不暴露裸 core）
src/cockpit/     【Web】独立只读驾驶舱（浏览器前端 + 订阅 diff-since-seq；HUD 灯/告警/飞行记录仪；可跨项目复用）
```

- **依赖方向**：`cockpit → mecha.bus(只读)`；`mecha → core`；**`core ↛ mecha`**。
- **两条通道一个真值**（纲领 T1）：写走 `server → session.set → bus`；读走 `bus → cockpit`。
- **命名待拍板**（D-C）：`mecha` 只是提案。

---

## 3. 分期计划（每期：目标 / 做什么 / 验收 / 不套层检查）

> **实施状态（2026-09-19）：Phase 0–5 全部交付完成**，每阶段一个可回滚提交、
> 判据常绿（基线 21→26）。逐阶段做了什么 / 判据 / 不套层结论 / 踩到的坑
> 见 **`docs/AI机甲框架-实施记录.md`**；协议蒸馏见 **`docs/机甲协议-draft.md`**。
> 以下是原计划（保留作依据）。

### Phase 0 · 边界裁决与 harvest（半天，纯文档）
- **做什么**：① 拍板 §4 决策点；② 把 `old/AI-调用架构.md` 里**被否决结构之外的
  好点子**列成 harvest 清单（渐进披露 `capabilities`、句柄+增量 patch、内容寻址缓存、
  dry-run estimate、sweep+metrics、构造性错误 hint/suggest、**backend 路由列**）——
  这些将在 Phase 2/4 以"单一定义点"的方式重建，而非照搬其套层结构。
- **验收**：决策点有书面结论；harvest 清单进本文件附录。
- **不套层检查**：确认 harvest 的每一项都落到 `mecha`（引擎之外），不在 core 里加皮。

### Phase 1 · 语义内核（registry + 写 choke point + 事件总线）——**地基**
- **做什么**：
  - `registry.py`：参数元数据**单一定义点**（解 C1）。**按方案 B 实现（D-A 已定）**：
    core 里用零依赖 `Param(默认值, desc, unit=, lo=, hi=, tier=)` 就地声明，
    `mecha` 写**收集器**扫各引擎模块的 `Param` 实例拼成全表；表是**扫出来的**，
    不是登记进去的 ⇒ 无第二真源。**`_` 前缀内部数值保护参数标 `tier='internal'`，
    默认不对 AI/GUI 露面（D-B 已定）**。同时收 `runspec.SOLVE_KWARG` 作为 `solve.*` 唯一映射。
    落地顺序：先在 `sambe`（独占 9 参数）试点，逐位对拍引擎读到的值不变，再铺开（`params归属` §4）。
  - `session.py`：会话态（当前 params 快照 + 覆写栈）；**写 choke point** `set(key, value,
    actor, reason)` → 校验（单位/范围/必填，非法**自愈回退**不静默）→ 发事件。
  - `bus.py`：append-only 事件日志 `{seq,ts,actor,kind,target,before,after,reason,
    provenance,reversibility}`；大载荷走指针+sha1；体积闸；轮转。
- **验收（新判据 `mecha_kernel_selfcheck`，带致死性）**：
  1. `list_params` 能返回每个参数的 name/type/unit/range/desc/default + 对应引擎形参；
  2. 每次 `set()` 产生**恰好一条**带 actor/reason/before/after 的事件；
  3. **无侧门**：AST 扫描证明"改会话态的路径只有 `session.set`"（故意加一条裸写入→判据变红）；
  4. 非法值被拒且回退（致死性：喂越界值必须报错，不能静默通过）；
  5. `run_checks.py` 21 判据仍 0 FAIL（若采方案 B，逐位对拍引擎读到的常量值不变）。
- **不套层检查**：registry 是**收集**不是**复制**；core 不 import mecha；无并行分派表。

### Phase 2 · MCP 写通道 + 通用工具 + 读回回执
- **做什么**：`server.py` 起 MCP server（localhost，端口文件发现）。通用工具：
  `list_params`（分页/前缀，轻量优先）、`set_param`（经 session.set，回执带 `resolved`）、
  `get_state`、`get_activity(since_seq)`、`undo`、`estimate_cost`、`run_compute`
  （复用 `runspec` 唯一分派，**与人点运行同一路径** I5）。错误消息**可教学**（D2：
  打错键名先给最可能的 + 别名线索）。
- **验收（新判据 `mecha_mcp_selfcheck`）**：harness 能连；能读参数表（含单位/范围）；
  能写并拿到 `resolved` 回执；能 `undo`；错误消息含"你是不是想写 X"；所有回程走**同一把
  体积闸**、截断必带"截了多少+原文在哪"（I4）。
- **不套层检查**：工具只是 `session/ops` 的薄封装，不含业务逻辑副本。

### Phase 3 · 驾驶舱（Web 独立只读投影）+ 订阅端点
- **做什么**：`mecha.bus` 开**只读订阅端点**（`diff-since-seq` 增量推送，SSE/WebSocket）；
  `src/cockpit/` = **Web 前端**（D-E 已定；浏览器打开，可作远程只读看板）+ 一个静态服务：
  HUD 归因灯（AI 改动标记 + reason）、告警灯（异常规则）、飞行记录仪（查询/回放/undo 视图）。
  **独立于项目存活**（黑匣子，纲领 T5）。
- **验收（新判据 `cockpit_selfcheck`）**：
  1. 驾驶舱独立进程，**杀掉项目后仍显示死前最后状态 + 完整事件史**；
  2. `diff-since-seq` 增量：连续读只回新事件，不重灌全量；
  3. 告警灯有**致死性证明**（构造一次"静默降级"→灯必须亮；永不亮的告警视为 FAIL）；
  4. 驾驶舱**只读**：无任何写回项目的通道（AST 守卫）；
  5. **远程只读**可用（鉴权后另一台设备浏览器能看），且**不暴露写端点**（T6）。
- **不套层检查**：驾驶舱无自己的真值，只投影总线（I2）；本地只留视图偏好（P6）。

### Phase 4 · CodeAct + async job + 可复现指针 + 行为预览
- **做什么**：
  - `codeact.py`：一次调用多步编排，命名空间**只暴露 `registry/session/ops`，不暴露裸
    core**（D1/D3：显式声明"不在其中的重要名字"）。
  - async job：长任务可发起、可轮询、**断线可找回**；进度用 `[阶段内进度,取消标志,阶段]`。
  - 可复现指针（D8）：每个产物自带"能重跑它自己的完整配置"（RunSpec 已可 pickle，落盘 + 编号回指）。
  - 行为预览（D7）：提交长算前的**纯查询**入口——"这条脉冲会驱动哪些跃迁、各多强、
    基组缺谁"（复用 `cost.py` + `static_atom.collect_transitions`，不改物理）。
- **验收**：CodeAct 里 `get_*` 边界显式；async 断线重连能取回结果；产物→配置链可查；
  预览入口能在**不跑演化**的前提下回答"基组够不够"。
- **不套层检查**：CodeAct 调语义层，禁止绕过 `session.set` 直改状态。

### Phase 5 · 硬件档预留（只留接口，不实现）
- **做什么**：把 `ops` 的后端路由做成**带 `backend` 列的表**（harvest 自 `old/lab` 的
  DISPATCH 北极星）：`backend='sim'`（现状）/ 未来 `'hardware'`，**AI 侧零改动**。
  预留：分级自治 L3 的"提议→人确认→互锁"闸；数字孪生 dry-run（硬件序列先跑 `backend='sim'`）。
- **验收**：路由表存在且 `backend='sim'` 逐位等价现状；L3 闸在仿真档默认放行但**结构就位**。
- **不实现**：真实设备驱动、硬实时内核、物理互锁固件——不在 Re0 范围。

---

## 4. 关键决策点（D-A/D-B/D-C/D-E 已拍板 2026-09-19；D-D 待硬件实例落地）

| # | 决策 | 选项 | 结论 | 权衡 |
|---|---|---|---|---|
| **D-A** | C1 参数元数据的单一定义点放哪 | ①方案B：core 里零依赖 `Param` 描述符就地声明 + `mecha` 收集器；②元数据全放 `mecha`，core 保持裸常量 | **✅ 已定：方案 B** | ①值+元数据**同处声明**=真单一点，但要动 core 常量（逐位对拍守）；②不动 core，但"值在 core、元数据在 mecha"=两处，会漂（`params归属` 方案A 的病） |
| **D-B** | C2：`_` 前缀的 ~20 个内部数值保护参数（`_SPARSE_*`/`_VOLT_*`/`_DENSE_MAX`…）要不要在 AI/GUI 露面 | 露面 / 不露面（`tier` 分层，默认隐藏） | **✅ 已定：不露面**（tier 分层） | 它们本质是"引擎内部数值保护"，暴露只增噪声与误调风险；保留 `tier='internal'` 供深挖 |
| **D-C** | 新包命名 | `mecha` / `helm` / `bridge` / 其他 | **✅ 已定：`mecha`**（语义内核，`src/mecha/`）+ `cockpit`（驾驶舱，`src/cockpit/`） | 纯命名，不影响架构；与文档标题「AI 机甲框架」一致 |
| **D-D** | 协议何时冻结（纲领 §11 时序警告） | Phase 1 后 / Phase 3 后 / 硬件实例落地后 | **Phase 3 后标 draft，硬件落地再冻 v1** | 别在 N=1 冻结；Phase 3 后有"仿真"一个完整实例可蒸馏 |
| **D-E** | 驾驶舱技术栈 | 独立 Qt App / Web(浏览器) / TUI | **✅ 已定：Web**（为远程跨设备计算/控制留空间） | 需独立进程 + 只读订阅；Web 便于远程只读看板（T6）。边界见 §4.1 |

### 4.1 D-E（Web 驾驶舱）的边界后果

选 Web 是为"以后远程跨设备计算/控制留空间"，但必须守住纲领 T6 的安全边界：

- **只读订阅端点**（驾驶舱用）：可**远程暴露**，但必须**鉴权**；它只能读事件总线，
  物理上无写回通道（Phase 3 验收 4/5）。
- **写/控制端点**（MCP）：**默认只走 localhost**，永不因"驾驶舱是 Web"就跟着暴露。
- **远程"控制/计算"是未来项**：真要远程下发控制，必须 ① 经鉴权隧道、② 过分级自治
  （L3 不可逆动作仍"提议→人确认→互锁"，纲领 §5）、③ 写端点仍不裸露公网。
  **Web 驾驶舱今天只解决"远程看"，不解决"远程动"**——后者留给硬件档（Phase 5+）。

---

## 5. 非目标（明确不做）

- **不重造 agent 循环**——用 harness 当宿主（纲领 §9）。
- **不套第二真源**——任何新层过 §13 判据。
- **不把状态塞进 core**——引擎保持纯函数 + 记忆化。
- **不在 N=1 冻结协议**（D-D）。
- **不动物理引擎数值**——改造期判据常绿是硬门槛；Phase 1 按方案 B 把 core 常量包进 `Param(...)`
  声明（**只改声明形式、不改数值**），逐位对拍守"引擎读到的值不变"。
- **不做真实硬件驱动 / 硬实时内核**——Phase 5 只留接口。

---

## 6. 风险与缓解

| 风险 | 缓解 |
|---|---|
| **并发会话**（README 明确警告"这个目录同时有别的会话在改"） | 除 Phase 1 把 core 常量包进 `Param`（只改声明形式）外，**其余只加 `src/mecha`·`src/cockpit` 与 docs**；动手前确认无别的会话在同目录 |
| **import 成环** | `mecha → core` 单向；registry 用**收集器**（消费方扫 core），core 不认识 mecha |
| **方案 B 动 core 常量引入回归** | 逐位对拍判据守（引擎读到的值不变）；先在 `sambe`（独占 9 参数）试点再铺开（`params归属` §4） |
| **驾驶舱退化成第二真源** | 判据强制"驾驶舱只读、无写回通道"（AST 守卫）；本地只留视图偏好 |
| **假绿** | 每个新判据带致死性证明；SKIP 显式计数（D4） |

---

## 附：harvest 清单（Phase 0 已裁决落点，来自 `old/AI-调用架构.md`）

> 以下是被否决的 `old/lab`（2188 行，造并行 DISPATCH/spec/Result 第二真源）里**值得重建**
> 的点子。**重建为单一定义点，非照搬套层结构**——每项标注落到哪个 `mecha` 模块、哪一阶段：

| harvest 点子 | 落点（单一定义点） | 阶段 | 怎么不套层 |
|---|---|---|---|
| 渐进披露 `capabilities(level=)`：整个调用面 ≈565 token，按需拉单条签名 | `mecha/registry.py` 的 `list_params(prefix,tier,page)` + `describe(key)`；MCP 工具面即能力清单 | P1/P2 | 表是**扫 core 的 `Param` 出来的**，不是另抄一份目录 |
| 句柄 + 增量 patch：首轮发整份规格，之后只发 `{"id":..,"patch":{..}}` | `mecha/session.py` 的会话快照 + `set(key,value)` 逐键写 | P1 | 快照是**唯一会话态**，patch 只是 `set` 的批量形式 |
| 内容寻址缓存（`rid`）：同规格二次提交零算力命中 | `mecha/ops.py` 的产物落 `.mecha/artifacts/<sha1>/`（RunSpec 指纹寻址） | P4 | 缓存键 = RunSpec 内容 sha1，不引入平行规格对象 |
| dry-run `estimate`：提交前看代价 | `mecha/ops.py` 的 `estimate()` 复用 `core/cost.py`；P4 扩成行为预览 | P2/P4 | 只**调用** core 既有估算，不重写物理 |
| `sweep + metrics`：一次表达 N 次运行，回程只给标量表 | `mecha/codeact.py` + async job（P4）：编排 N 次 `ops.run`，回程走标量摘要 | P4 | 每次运行仍走 `runspec` 唯一分派（I5 人机同路径） |
| 构造性错误 `kind/hint/suggest`：第一次失败就能改对 | `mecha/errors.py`（并入 D2 可教学错误消息） | P1 | 错误由 choke point 产出，不是另建校验副本 |
| **backend 路由列**：`DISPATCH` 加一列 `backend='hardware'`，AI 侧零改动 | `mecha/ops.py` 的后端路由表（`backend='sim'` 现状 / `'hardware'` 未来） | P5 | 路由表是**声明**，`'sim'` 逐位等价现状 `runspec` 分派 |

**一句话**：`old/lab` 的病是「把引擎的解析/分派/结果契约又抄了一遍」（第二真源）；
本次重建的每一项都**落在引擎之外**（状态/事件/投影/编排），且**收集或复用** core 的既有
定义点，绝不新造平行规格。
