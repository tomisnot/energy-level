# AI 机甲框架 · 回归主仓库 · 迁移计划

> **方向（已定）**：**Re0重构 = 新主干**；旧仓（`Energy Level`）= **零件库 + 历史载体**。
> 不是"把 Re0 插进旧版"，而是"**从旧版挑有用的 → 充分清洗 → 补进 Re0 → Re0 整体回来
> 覆盖旧版**"，且**保留旧仓提交历史**。
> **本文档落在旧仓 `docs/`**（迁移在旧仓工作树内以提交叠加完成，故计划随工作地）。
> **框架依据**：《AI机甲框架-开发原则》《开发计划》《建场层设计》《实施记录》（均在
> `Re0重构/docs`）。

---

## 0. 已定决策（本轮拍板，计划的硬约束）

| # | 决策 | 落点 |
|---|---|---|
| **M1** | **GUI 全量搬**，旧模块从模块角度基本无废止；内部逻辑可精简、已精简的参数欢迎 | §4 |
| **M2** | **god-mode 检视器保留**（对人类有用）；内部参数走 `tier` 深挖开关（D-B：默认隐藏、可展开） | §4.4 |
| **M3** | **页面布局绝不改**（精心调过）——只精简逻辑，不动控件位置/尺寸/排版 | §4.5（铁律） |
| **M4** | **周边全丢**：`microwave_pulse` / `scripts/research` / `ui_design` / `src/julia` 不进主干 | §3 |
| **M5** | **save_state 不写旧版适配代码**：理解旧档语义 → **一次性转成新档格式**（物理量未变，只是格式） | §6.2 |
| **M6** | **保留旧仓提交历史** | §7 |
| **M7** | **GUI + 模式 + sweep 一轮全上**（不分阶段削减范围） | §4/§5 |

---

## 1. 目标终态（新主干长什么样）

```
src/core/       Re0 的清洗引擎（纯函数 + param.py + 唯一分派 runspec）  ← 旧 core 丢弃
src/mecha/      语义内核（registry/bus/session/ops/server/subscribe/codeact/jobs/autonomy/errors）
src/cockpit/    Web 只读驾驶舱（AI 模式的监督面 + 黑匣子）
src/ui/         清洗后的 GUI（全量移植，布局不变；人工模式的控制台）
src/sweep/      清洗后的并行扫描（改调 mecha.ops / Re0 runspec）
src/manybody/   取旧仓与 Re0 的超集
checks/         物理判据（Re0 26）+ 重锚到 mecha 的接线判据（旧仓 17）
docs/           两仓文档合并
save_state/     全部转到新格式
```

- **依赖方向（单向，守卫强制）**：`core`（纯物理 + 调用协议 `core/protocol.py`）↛ 任何上层；
  `adapter`（唯一同时 import core+mecha 的胶水，实现 mecha 的 `EngineProtocol`，含建场层 fields）；
  `mecha`（纯框架、零领域 import，`mecha_purity` 提炼守卫看守）；`ui / cockpit / sweep / dsh插件 → mecha`
  （经注入的 adapter 拿领域）。
- **人类投影，一条真值（【2026-09-20 修正】GUI 角色收窄；【2026-09-21 再修正】AI 模式 GUI 关闭）**：**GUI = 人工操控面**
  （人工模式可写；**AI 模式 GUI 直接关闭/不启动，不挂只读界面**——要接管就切回 GUI 模式连 Hub 夺权，
  而非挂一个啥也不能动的只读窗）。**AI 模式的配置态
  监督面移到 web cockpit 的配置态结构树**（不再看 GUI 参数）。cockpit = 独立进程黑匣子/飞行记录仪
  + 远程只读看板 + **配置态结构树**（schema join 现值，`/api/config-tree`）。三者都是 `mecha` 语义
  状态的投影，共写同一 choke point / 同一事件总线（I2/I5）。
- **进程拓扑（【2026-09-20 修正】：Hub 独立，取代先前"GUI 进程=项目进程、进程内托管 MCP"）**：
  **Hub 是独立进程**（`python -m src.adapter.serve`，headless 可跑），托管 mecha session+bus+MCP
  server+注入的 engine；人启动 Hub = 权威（T4）。**GUI / cockpit / dsh 插件都是 Hub 的客户端**：
  GUI 经 AppBridge（传输可插拔：本地进程内 / 未来远程），cockpit 独立进程读 journal 文件（黑匣子 T5
  + 远程只读 T6），dsh 插件经 MCP 连 Hub。→ GUI 可关而 AI 不断连（修先前致命伤），且天然支持跨设备（需求③）。

---

## 2. 两个模式（架构支点）：互斥的是"写权"，不是"状态"

> **单一控制权（single control authority）**：任一时刻只有一方能写语义状态。
> **但状态/历史/总线只有一个**——模式只切换"谁持写权"，绝不搞两套状态（否则请回双真源）。

| 模式 | 写权 | MCP 写通道 | GUI | 人的位置 |
|---|---|---|---|---|
| **AI 监工** | AI | **live**（经 dsh 插件的自愈 MCP 桥） | **关闭/不启动**（不挂只读界面）；监督看 web cockpit；要接管=切回 GUI 模式 | 主要看 **web cockpit 配置态结构树**（+事件流+告警）；要夺权就切回 GUI 模式 |
| **自己干** | 人 | **门控**（AI 写→可教学报错"人工模式，AI 无写权"） | 正常操作，经 choke point 写 | 在 GUI 里开车 |

- **交接协议**（原子 + 记账 + 静默 + 重定位）：切换发一条 `authority_switch` 事件（actor/reason）；
  **AI→人** 前先把 AI 静默（在飞 async job 收尾/取消/孤儿对账）；**人→AI** 前让 AI 用
  **定位包 / diff-since-seq** 重新站定（R1 orientation 原语）。
- **【2026-09-20 修正】AI 模式监督面 = web cockpit 配置态树，不再是 GUI**：用户明确"AI 模式下不看
  GUI 参数，看 web 上自动生成的结构树"。故 cockpit **新增**配置态结构树（schema join 现值，
  `/api/config-tree`，已实现+判据绿）作为 AI 模式主监督面；GUI 退回**纯人工操控**；
  **【2026-09-21 再修正】AI 模式下 GUI 关闭**（不挂只读/暗色界面），要接管=切回 GUI 模式连 Hub 夺权。
  GUI 仍去掉旧框架耦合（`pm_setup`/`engine_overrides`/`set_god_mode`）。
- **cockpit 职责**：① 黑匣子/飞行记录仪（独立进程读 journal，T5）、② 远程只读看板（Web/鉴权，T6）、
  ③ 独立告警、④ **配置态结构树**（AI 模式主监督面）。数据源仍是 journal（Hub 落 `registry_schema`
  事件 + set 事件；cockpit 只读、不 import mecha，黑匣子纯净）。
- **AI 操作历史**：GUI 内嵌一个**流水面板**（bus 投影：seq/actor/kind/target/before→after/reason）
  + 就地高亮"刚改的"。**注意**：in-GUI 历史随进程死，**文件级黑匣子仍是 cockpit 的职责，不可替代**（T5）。
- **泛化**：这个写权 token 就是硬件档（§5/§6）的 L3 控制权交接，前向兼容。
- **【2026-09-20 交付】AI 侧写通道 = dsh 集成插件**（`Re0重构/dsh/`，TS/Cordis，out-of-tree，`--patch`
  加载）：接管 MCP 连接生命周期（自愈重连，治 dsh stock HTTP 桥"重启即永久 404"死区），工具原生注册
  进 dsh，cockpit 配置树经 `ctx.slots` 头部动作 iframe 呈现。已活体验证（含杀 Hub 重启自愈）。保留
  独立 MCP server 跨 harness。详见 `Re0重构/dsh/README.md`。

---

## 3. Phase A · 打捞清单（triage manifest，先做，产出书面清单）

| 旧仓子系统 | 处置 | 依据 |
|---|---|---|
| `src/ui/*`（14 文件，~10k 行） | **移植 + 清洗**（全量，布局不变） | M1/M2/M3 |
| `src/sweep/*` | **移植 + 清洗**（改调 mecha.ops/runspec） | M7 |
| `src/manybody/*` | **取超集**（Re0 已有 3 文件，diff 后合并） | 不丢能力 |
| 17 个接线判据（GUI/MCP/sweep） | **重锚到 mecha**（覆盖只迁移不删除） | §6.1 |
| `src/core/*`（旧） | **丢弃**，用 Re0 core | Re0 core 是其清洗超集 |
| `src/ai/*`（旧 MCP/sim_state/codeact/dispatch/run_store） | **丢弃**，mecha 取代 | 纲领 §9 |
| `pm_setup.py` / god-mode 管线 / `engine_overrides` / `params.py` | **丢弃**，mecha.registry/session 取代 | C1/C2 已解 |
| `dissipation_lab/*` | **裁决**（Re0 已把 field_loss 折进 core；除非有 Re0 没有的模型，否则丢） | 第14刀 |
| `microwave_pulse` / `scripts/research` / `ui_design` / `src/julia` | **丢弃**（不进主干） | **M4** |
| `scripts/` 其余（如 `setup_arcdata`） | Phase A 逐个裁决（ARC 数据初始化类多半留） | 不误伤 |
| `save_state/*.json` | **一次性转新格式**（不写运行时旧档适配器） | **M5** |
| `.arcdata` / `requirements.txt` / `build.bat` / `*.spec` | 保留并按新主干校对 | 运行环境 |

**Phase A 验收**：一份书面打捞清单（每项 keep-clean / discard / migrate-data），**覆盖前逐项核对
"能力没丢"**（防最大风险 = 功能静默丢失）。

---

## 4. Phase B · 清洗 + 移植 GUI（最大块）

### 4.1 剥离清单（旧框架耦合，全部拆掉）
`pm_setup`（`import pm_setup as pm`）、`god_split` / `god_read` / `god_write` / `_SOLVE_SPECS`、
`engine_overrides` / `set_engine_overrides`、`set_god_mode` 的 AI 侧、`from ..core.params` /
`_P.override` / `_P.DEFAULTS`、旧 `src.ai` / `sim_state`。

### 4.2 改接 mecha（GUI 成为语义层的纯消费者）
- 参数元数据/校验 → `mecha.registry`（`list_params`/`describe`/`validate`；C1/C2 已解）。
- 读写状态 → `mecha.session`（**经同一 choke point**，人机同路径 I5；每次写发事件、可 undo）。
- 运行/预估/预览 → `mecha.ops`（`run`/`estimate`/`preview`；`run` 复用 runspec 唯一分派）。
- 波形编辑（`ui_waves`）→ 对接**建场层**（《建场层设计》：底层存物理量 + 辅助捷径；旧 GUI 的
  「定光强/定频率/实时反馈/`_suggest_freq`」正是辅助层的现成范本，解绑 Qt 后与 AI 共用同一批
  core 换算函数）。

### 4.3 加两模式（§2）
控制权 token + GUI 的"人工模式可写 / AI 模式**只读+实时刷新**（订阅 bus 重渲染）+横幅+归因高亮" +
**AI 流水面板**（bus 投影，可停靠 dock 或检视器新 tab，不回流布局）+ 交接协议 + `authority_switch` 事件。

### 4.4 保留 god-mode 检视器（M2）
`solver_inspector` 等检视页**保留**，改为 `mecha.registry` 的人类读写面；`tier='internal'` 的
内部参数**默认折叠、给一个"显示内部参数"开关**供人深挖（AI 侧仍默认隐藏，D-B）。

### 4.5 布局铁律（M3，最高优先）
- **控件位置/尺寸/间距/排版/字体 逐像素不变**；只允许精简**内部逻辑**与已被精简的参数。
- **保留**清洗后的实时渲染（AI 模式只读实时刷新 + 归因高亮），但去掉旧框架耦合；新增的横幅/高亮/
  流水面板**不得回流既有排版**（横幅走状态栏/overlay，流水走 dock 或检视器新 tab），**绝不重排**。
- **验证手段**：迁移期**临时**借用旧 `ui_design` 的 layout-audit / 截图 diff 技术逐页比对
  （`ui_design` 本身按 M4 不进主干，仅作一次性验证脚手架）；每页移植后出"迁移前 vs 迁移后"截图，
  逐像素对齐才算过。

### 4.6 施工进度与逐文件清洗清单（持续更新）

**已建成并验证（staging `Re0重构/`，全量基线 29 判据 / 0 FAIL / 294s）**：
- `src/mecha/authority.py`：两模式写权 token（gate/switch/静默闸）+ `session.py` 接线
  （set/set_many/undo 过 gate；可选、默认不门控→向后兼容）+ `authority_selfcheck`（7/20）。
- `src/mecha/app_bridge.py`：GUI↔mecha 对接缝（捆 Bus/Authority/Session/Ops/JobRunner/registry；
  取代 pm_setup+archive+src.ai+engine_overrides/god_mode）+ `app_bridge_selfcheck`（7/25）。
- `src/mecha/fields.py` + `fields_selfcheck`（11）：**建场层已由并发会话实现**（随快照进 staging）。
- `src/__init__.py`：合并惰性 UI 导出（`from src import fields/ui_dynamic/...`）+ 引擎别名；ARC 补丁留到 Phase E。
- GUI 源码已 copy 进 staging（`src/ui/*` 14 文件 + `app_entry.py`/`main.py`/`能级.py`），**尚未清洗**。

**逐文件清洗清单（外科式最小 diff，只动框架耦合行，布局代码一行不碰 M3）**：

| 文件 | 耦合点 → 改成 |
|---|---|
| `evolve_worker.py` | `from ..core import params as _P`（**已删**）/`_P.override`/`_P.get('DEFAULT_RTOL/ATOL')` → 整体瘦身为 `app_bridge.run(progress_arr=…)` 的 QThread 包装（engine_kwargs 走 `solve.*`→runspec） |
| `fields.py` | `pm_setup`/`param_reg` → `app_bridge`/`registry`；`core.set_ac_stark_config`（**已删**）→ `build_level_rows` 显式传 `n_photon`/`sos_n_window`（A10） |
| `ui_dynamic.py` | `pm_setup`/`param_reg`/`engine_overrides`/`set_engine_overrides`/`god_split`/`_P.override`/`archive.snapshot/restore`/`pops_2d`→`pops` → `app_bridge`(write/effective/run/save/load)；build_run 必填键（B7）；**加两模式只读门控 + bus 订阅实时刷新 + 归因高亮 + AI 流水面板** |
| `solver_inspector.py` | `pm_setup.register_god_group`/`god_write`/`engine_overrides` → `registry`（含 internal tier 深挖开关 M2）；`set_engine_overrides` → `app_bridge.write('solve.*'/'engine.*')` |
| `app_entry.py` | `new_registry`/`register_panel_params`/`archive`/`_init_ai(src.ai.server/journal)`/`set_engine_overrides`/`god_write`/`set_mode` → 组装 `AppBridge` + 进程内 MCP（后台线程）+ **模式切换=人的动作**（AI 不能自授权）；`build_level_rows` 补 `E_vpcm`（A1）、人工名走 `group_label`（A8） |
| `app_state.py` | `from ..core import core`（走 Re0 兼容别名，多半免改）；`LevelGroupStore/FieldState` 若与 session 重叠则收口到 app_bridge |
| 其余（`ui_theme`/`ui_group`/`ui_radar`/`ui_spectrum`/`ui_single`/`hamiltonian_viewer`/`sweep_panel`） | 耦合轻，逐个核对（多为纯 Qt widget + 直调引擎，走 Re0 兼容别名） |

**Phase B 验证闸**：全 ui 移植后 → offscreen 实例化 `MainWindow` 不报错 → 截图逐像素比对
（借 `D:\desktop\UI design` 的 layout-audit 技术，M3）→ 再跑全量基线常绿。

---

## 5. Phase C · 移植 sweep（+ manybody 取超集）

- `src/sweep`：轴键改用 **registry 全路径**；执行改调 `mecha.ops`/Re0 `runspec`（唯一分派）；
  **独立 runner 进程保留**（隔离 + 内存随进程归还 + headless 可复现）；生命周期
  submit/status/result/cancel/list 对接 mecha。
- 注意 Re0 已知约束：ARC 的 SQLite 多进程并发建表会 `database is locked` ⇒ 父进程先建基组/场/
  V_dd 预热再放 worker（旧 runner 的老规矩，别丢）。
- `src/manybody`：Re0 已有 3 文件；与旧仓 diff 取**超集**，判据 `manybody_m0_selfcheck` 保绿。

---

## 6. Phase D · 判据重锚 + save_state 转换 + 清洗守卫 + 文档

### 6.1 接线判据重锚（覆盖只迁移不删除）
旧仓 17 个验"接线"（GUI/MCP/sweep）的判据，重锚到 mecha/cockpit；每个带**致死性证明**；
SKIP 显式计数（D4）。物理判据以 Re0 的 26 为准。

### 6.2 save_state 一次性转格式（M5，**不写运行时旧档适配器**）
- 做法：**先逐字段理解旧档语义**（旧 `params` 快照：`groups.list`/`field.*`/`dynamic.*`/`solve.*`/
  `engine_overrides`/波形 `serialize_waves` 同构/初态身份键或旧标签/叠加态…），写一个**一次性转换脚本**
  把每个 `.json` 重表达成新档格式（新 registry 键 + `pops`/`states` 新契约 + 身份名）。
- 用户判断"物理量没变、只是格式"——转换脚本**逐档转换 + 抽样对拍**（转换前后关键物理量一致）后，
  **脚本用完即弃**（不进主干、不留旧格式读取分支）。
- 转换失败的档**大声列出**（不静默丢），交人工裁决。

### 6.3 清洗致死守卫（"充分清洗"的可验证定义）
一条判据：在主干里 `grep` 旧框架符号 **必须 0 命中**——
`pm_setup | core\.params | engine_overrides | god_split | god_read | god_write | set_god_mode | sim_state | _P\.override | _SOLVE_SPECS`。
故意留一处旧耦合 → 判据变红（致死性）。另加**依赖方向守卫**（AST：`core` 不 import `ui/mecha`）。

### 6.4 文档合并
两仓 `docs/` 合并；README 从"纯引擎沙盒"改写为"完整产品"；迁移记录归档。

---

## 7. Phase E · 覆盖切换（M6：保留旧仓提交历史）

**策略**：迁移工作**在旧仓（`Energy Level`）工作树内、以一连串可回滚提交叠加在旧历史 `H_old` 之上**
完成——旧仓历史**天然保留**。每步先跑基线绿再提交：

1. 用 Re0 core 覆盖旧 `src/core`（删 params.py/route.py/evolve_engine.py/project.py 等旧文件）。
2. 加入 `src/mecha` + `src/cockpit`（Re0 现状）。
3. 删旧 `src/ai` + `pm_setup` + god-mode 管线。
4. 移植清洗后的 `src/ui`（Phase B）+ `src/sweep`（Phase C）+ manybody 超集。
5. save_state 一次性转换（§6.2）。
6. 重锚接线判据 + 清洗守卫（§6）。
7. **全基线绿**（物理 26 + 重锚接线）后，旧 `src/ai` 等正式退役删除 → 切换完成。

- **可选**：若还想保留 Re0 的 Phase 0–5 细粒度历史，用 `git subtree`/`git merge
  --allow-unrelated-histories` 把 Re0 的提交接进来；否则以"内容移植 + 旧仓提交"为准（满足 M6 即可）。
- **切换判据**：主干同时具备 引擎 + mecha + cockpit + GUI(全量,布局不变) + sweep + 判据全绿 +
  save_state 全转 + 清洗守卫 0 命中。

---

## 8. 风险与缓解

| 风险 | 缓解 |
|---|---|
| **功能静默丢失**（最大风险） | Phase A 打捞清单 + Phase E 覆盖前逐项能力核对；判据"覆盖只迁移不删除" |
| **布局漂移**（M3 红线） | 逐页"迁移前 vs 后"截图逐像素 diff；删 AI-render 不回流布局 |
| **清洗不彻底**（旧符号残留成暗桩） | §6.3 grep 致死守卫 + 依赖方向 AST 守卫 |
| **大爆炸式合并难回滚** | 每 Phase 可回滚提交 + 基线常绿才提交（README 改造纪律） |
| **save_state 语义误读** | 转换前逐字段理解 + 抽样对拍关键物理量 + 失败档大声列出 |
| **并发会话/目录写锁**（本轮已踩） | 施工前确认无别的会话在同目录；跨目录写用重试 + 确认成功前不删暂存 |
| **sweep 多进程 ARC 锁表** | 父进程预热建缓存再放 worker（旧 runner 老规矩） |

---

## 9. 验收清单（cutover checklist）

1. `grep` 旧框架符号 = **0 命中**（致死判据证明会变红）。
2. 依赖方向守卫：`core ↛ ui/mecha`（AST）。
3. 全基线绿：物理 26 判据 + 重锚后的接线判据，0 FAIL，SKIP 显式计数。
4. **布局逐像素不变**：每页截图 diff 通过。
5. 两模式切换 + 交接协议有判据（含致死性：AI 模式下 AI 能写、人工模式下 AI 写被拒）。
6. `save_state` 全部转到新格式且能读回；无旧格式运行时分支。
7. 能力核对：GUI 各页 / god-mode 检视器 / sweep / manybody / 建场辅助 全在新主干且可用。
8. 旧仓提交历史保留（`git log` 可追溯到 `H_old`）。

---

## 附：与既有框架文档的关系

- 本计划是《开发计划》Phase 1–5（已在 Re0 落地）**之后的"产品化回归"**阶段。
- 建场层（《建场层设计》）在 Phase B 与 `ui_waves` 对接：GUI 的「定光强/定频率/实时反馈」与
  AI 的辅助捷径**共用同一批 core 换算函数**（I5 人机同源）——移植 GUI 与建场层落地应协同排期。
- 所有"不套层 / 单一真源 / choke point / 事件总线 / 分级自治"纪律沿用《开发原则》，本计划不另立规矩。
