# AI 协同二期升级方案 —— 语义状态 · 作业模型 · 结果模型 · 模型卡

> 顶层依据：《AI协同-项目目标与原则.md》（G1–G4 / P1–P6）。
> 触发：2026-09-11/12 的 freeze 调参会话——AI 通过 MCP + 本地脚本做了
> 数千次演化（单原子扫描、偏振扫描、抖动分析、两体验证），暴露出**一批
> 违反 P1/P2/P3/P5 的缺口**。本文把这些缺口归到原则层面，给出二期方案。
> 与《引擎调用与多原子扩展-待办.md》的关系：那里是**逐条 bug/摩擦清单**
> （A1–A11、S1–S7），本文是**按原则重构的升级路径**，清单是其施工单。

## 一、把这次的翻车逐条对回原则（先看清违反了哪一条）

| 现场事实（都有证据） | 违反 | 本质 |
|---|---|---|
| AI 为设初态**猜能量键** `[…,-1000430.0]`，结果初态落到 `59D_{3/2} m=+1.5` 而运行仍报 ok | **P2 / P5 / P1** | AI 被要求操作"渲染产物（序列化键）"，而不是"语义对象（能级 n,l,j,m）"；且没有真值源校验 |
| 按标签前缀判重删组 → **删掉 60P_3/2、留下 60P_1/2**；`'58G_{'` 让 58G_9/2 漏加 | **P1 / P5** | 索引进出的"组操作"是 GUI 渲染细节；语义操作应是"让基组包含这组能级"（声明式、幂等、返回增量） |
| AI 发起的重活**用户看不到、取消不了**，AI 也拿不到 job_id → 双方卡在"已有演化在运行" | **G3 / P2 / P3** | 作业（job）不是一等语义对象：无 owner、无可见性、无统一取消路径 |
| 重活打穿 MCP 传输层（`-32001`/`fetch failed`）留下**孤儿计算** | **P3 / G3 / P1** | 缺少**事前成本语义**与"重活即异步作业"的默认路径 |
| 两体 `save_npz` 不回传路径、`final_pop` 只给 top-N（取不到指定交换态） | **P1 / P5** | "给我 X 态的布居"是语义请求；"按大小排前五"是渲染 |
| `get_state('dynamic')` 与 `summary()` 字段不一致（mode/pair 为 None） | **P5 / P6** | 两个渲染器各自拼装状态，没有唯一规范化快照 |
| 微波波在基组内**零候选对**（缺中间态）→ 完全静默，白跑一整轮 | **P1 / G4** | 引擎缺"模型有效性自述"：不告诉调用方"这个模型根本表达不了你要的过程" |

结论：这次的问题**不是"工具少几个"，而是缺四个一等对象**——语义状态、作业、结果、模型卡。
逐个补丁（A1–A11）能止痛，但按 P5 应当补对象。

## 二、二期方案：四个一等对象 + 一个可报告的模型选择

### A. 语义状态面（Semantic state surface）——治 A2 / A7 / A9

原则落点：**P1 语义优先 + P5 单一真值源 + P6 状态边界**。
规定：**AI 面只接受语义操作，索引/键/标签一律是渲染产物**。新增（MCP 与 CodeAct 同名同义）：

```python
list_levels(scope='basis'|'available')      # 每个能级的身份 (n,l,j,m)+label+是否在基组
set_groups([(n,l,j), …])                    # 声明式、原子、幂等；回 {added, removed, kept}
set_initial_state(atom, n, l, j, m)         # 内部解析键；不在基组则**报错**（不再静默）
get_config()                                # 规范化运行配置（唯一真值源快照）
```

- `dynamic.psi_*` 的"身份键"降级为**内部实现**：GUI 存它、AI 不碰它。
- 兼容：`add/remove_level_group` 保留但标 deprecated；`summary()/get_state` 继续回
  渲染视图（含已加的 `groups_nlj` 结构化字段）。
- 验收：把这次的两处误操作写成自检——`set_groups` 传入 {60S,60P_3/2,58G_7/2,58G_9/2,59D_5/2,58F_5/2}
  时**必须**得到这 6 组（不论初始是 12 组还是 3 组）；`set_initial_state(60,1,1.5,-0.5)` 后
  该态初布居 = 1，且**若该组不在基组 → 返回 ok=False**。

### B. 作业模型（Job model）——治 A3 / A4，直接服务 G3 验收

原则落点：**G3 人可随时监控与接管 + P2 双方共同语言 + P3 可归因**。
作业进语义状态，人机共用一条取消路径：

```python
list_jobs()                     # [{job_id, owner:'human'|'ai', kind, status, elapsed_s,
                                #   progress, cost_est, model_card?, run_id}]
cancel_job(job_id)              # 唯一取消入口：GUI 按钮与 MCP 都调它
estimate_cost(config)           # 事前成本（复用引擎 _cost_report；两体也要有）
run_compute(..., mode='auto')   # 超阈值 → 自动异步 + 立即回 job_id（不再打穿传输层）
```

- GUI：作业列表（含 AI 作业）+ 取消按钮**覆盖 AI 作业**；忙时消息带占用者（已实现）。
- 验收 = **G3 协同验收脚本**（见 §四）：AI 跑扫描 → 人在 GUI 取消其中一个 → 双方状态一致、无孤儿。

### C. 结果模型（Result model）——治 A5 / A6

原则落点：**P1（语义取数）+ P5（结果的唯一真值）**。

```python
result.final_pop(states=[…])            # 指定态（名字/身份键都收）
result.timeseries(states=[…], t0, t1)   # 指定态的时间序列
result.by_manifold()                    # 按流形/原子归并
result.model_card()                     # 见 D
result.path('npz')                      # 大数组落盘路径（实现细节，固定键名）
```

- `summary`/`final_pop(top-N)` 明确标注为**投影**，不再是 AI 取数的唯一途径。

### D. 模型卡（Model card）——把"静默物理"变成"自述"

原则落点：**P1（AI 要能解释为什么）+ G4（坑进文档也进运行时）**。
复用已有的 `get_diag` 与成本报告，**每次运行默认产出一张卡**：

```
model: mixed | dim=1296 | pairs 5100 (kept 5100, drop_cap 0, drop_phase 0)
       AC-ized 0 | P=1 M=[] | maxΩ/ω=0.71 | tol/max_step=… | T=250ns
basis: 7 组 / 48 态 | 目标跃迁 60P→58G 需要中间态 59D,58F：**在册**
warnings: ["微波波 freq=7380MHz 在基组内 0 个候选对——疑缺中间态（若不在册会在此列出）"]
```

- **必须有"目标跃迁是否可表达"的判定**：给定波形与基组，若到达任何目标态所需的
  中间态不在册 → 直接告警（本会话的冻结失败即此类，靠 `evolve_mixed` 零配对告警已能兜住一部分）。
- 卡进三处：GUI（运行详情/工具提示）、journal（可追溯）、MCP 结果（`result.model_card()`）。

### E. 分时截断与拼接（S7）——作为**显式的模型选择**

原则落点：**P5**——截断是模型的一部分，若逐段变化就必须**被报告**，不许隐身。
故 S7 的前置条件是把 D 做完（每段一张模型卡 + 段边界泄漏/范数检查）。
先做零改动验证（`psi_amp` 串联，1 段 vs 3/5 段），再上 `segments=[…]` API。

## 三、优先级与工作量（按"违反原则的严重度 × 成本"）

| 序 | 项 | 治什么 | 成本 | 备注 |
|---|---|---|---|---|
| 1 | **A 语义状态面** | A2/A7/A9（**唯一会静默给错物理**的一类） | 中 | `set_groups`/`set_initial_state`/`list_levels`/`get_config` |
| 2 | **D 模型卡** | 静默物理、白跑、G4 | 低（多数是把已有 diag 汇总） | 收益/成本比最高 |
| 3 | **B 作业模型** | A3/A4、G3 验收 | 中 | GUI 作业列表 + 统一取消 + 成本预检 |
| 4 | **C 结果模型** | A5/A6 | 低—中 | 纯接口层，可先做 `final_pop(states=…)` |
| 5 | **E S7 分时截断** | 提速 | 高 | D 之后再做 |

已完成（2026-09-12）：A1（只读工具进 CodeAct）、A3 部分（`running_job`/忙消息/`cancel_job`）、
A5（npz 顶层键）、A9（`get_state` 的 `cfg`）、S4 部分（零候选对告警）、A11（SKILL 坑 8–12）；
**二期 A 主体**（`set_groups` / `list_levels` / `set_initial_state` / `get_config`，自检 `checks/ai_semantic_state_selfcheck.py` 含**事故回归**两条）、
**二期 D**（结果里带 `model_card`：配对/AC 化/P·M/maxΩ·ω/范数偏差 + **可表达性判定**与 `warnings[]`）。
仍待做：B 作业模型（GUI 作业列表 + 取消覆盖 AI + 成本预检）、C 的 `final_pop(states=…)`、E（S7 分时截断，前置是 D 的逐段卡）。

## 四、验收：把 G3 写成可跑的脚本

新增 `checks/ai_collab_acceptance.py`（人机协同验收，需 GUI 起一次）：

1. 软件启动 + MCP 连上；
2. AI（脚本扮演）执行：`set_groups` 建基组 → `set_initial_state` → `estimate_cost` → `run_compute`；
3. **人在 GUI 上取消 AI 的作业** → 断言：`list_jobs` 状态变 cancelled、无孤儿、journal 有 owner 归因；
4. AI 重跑并取结果：`result.final_pop(states=[…])` 与 `timeseries` 与 GUI 曲线一致（P2）；
5. 全程 GUI 无模态阻塞、人可正常操作其它参数（G3-3）；
6. 汇总为 0/1 判定，纳入 `checks/selftest_all.py`。

这条脚本同时是**二期的验收基线**：A/B/C/D 四项各自通过它的一条断言。

## 五、G4 机制化：陷阱登记册与运行时校验同源

本会话 4 次踩坑都是"文档里补一句"（SKILL 第 8–12 条）。但**文档不会在运行时拦你**。
建议：`docs/pits.yml`（机读）+ 两个消费者——
① `skills/energy-level/SKILL.md` 生成"坑"一节（G4：说明书即 prompt）；
② 引擎/服务端的运行时校验器（如"零候选对"、"psi 不在基组"、"Ω/ω 越界"）直接引用同一批 id，
   命中即进模型卡 `warnings`。
这样"人改核心 → AI 世界模型同步受益"（G4）就有了**机制**，而不是靠 AI 记得去改文档。

## 六、不做（守 P4/非目标）

- 不引入 harness 私有能力；新接口一律 MCP。
- 不做多 AI 并发（journal 串行化已留余地）。
- 不做"AI 复刻 GUI 操作路径"（P1 明确排除）；视觉通道只用于"人看到了什么"的核验。
