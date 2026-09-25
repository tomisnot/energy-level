# Energy Level · Re0 —— Rb-87 里德堡原子仿真软件

> **一句话**：把"静态能级计算 + 动态原子演化 + 多原子阵列"做成一台**人机共用**的仿真设备——
> 人用 GUI 开、AI 经 MCP 语义层开、两者共用**同一份**语义状态与同一条事件总线。
>
> **当前形态（2026-09-25 核对）**：新主干（Re0）已覆盖旧版；引擎 + GUI + sweep + 多体 + 机甲语义内核
> 全在。**AI 机甲框架已提炼为独立库 `mecha`**（本机实测 **0.2.0**，editable 装于
> `D:\code-nosync\mecha`），本仓不再持有 `src/mecha` / `src/cockpit`；EL 自己的 v2 领域层
> 住在 `mecha_v2/`。索引见 `docs/README.md`（**读文档前先看它**）。

---

## 1. 三个能力层次

| 层 | 在哪 | 状态 |
|---|---|---|
| **静态能级** | `src/core/static_atom.py` / `atomic_levels.py` / `field_build.py` | ✅ 能级组、AC/DC Stark、偶极矩阵元、谱线；两页（静态/动态）共用 `base_sublevels()` 单一基准 |
| **动态演化** | `src/core/{sambe,hard,evolve_mixed,evolve_twofield,evolve_rwa}.py` | ✅ 三主力引擎（sambe / hard / mixed）+ RWA 数值权威参照；唯一分派点 `runspec.evolve_run` |
| **多原子阵列** | `src/core/manybody/`（8 文件） | ✅ ED 引擎（N≈3–10 全链）：model / assemble / engine_ed / runspec_mb / observables |
| **并行扫描** | `src/sweep/` | ✅ 声明式扫描，独立 runner 进程 |

## 2. 三种控制方式

| 方式 | 通道 | 状态 |
|---|---|---|
| **GUI** | PyQt5，`app_entry.py` → `src/ui/*` | ⚠ 布局逐像素未动（M3 无偏离）；god-mode 检视器保留（引擎常量只读）。**已知失能**：「窗口/光子块」两控件静默失能（迁移后无人消费）；写权门控 `set_readonly` 有定义、MainWindow 从未调用。逐条见 `docs/待办-经复核的问题清单.md` §D1/§D2（该清单成文于 v1 期，部分条目随 v1 退役已失效，见其顶注） |
| **AI** | MCP 写通道（权威进程内起服务器：`app_entry.py --authority`；或独立进程 `python -m mecha_v2.hub`），**14 个工具** | ✅ 语义寻址 / 可教学错误 / 体积闸 / 指针化 / async 作业 / CodeAct 编排 / 建场层 |
| **远程跨设备** | 只读监控端点（权威进程内，**动态端口** + 项目根 `.monitor-port` 发现，仅 127.0.0.1 + CORS）；远程写未作正式能力验收 | ⚠ 只读监控可用（`/status` `/activity` `/history` `/config`）；写通道默认只 localhost（T6）。旧文档里的 cockpit `--host/--token` 随 v1 退役 |

## 3. 目录（当前真实结构）

```
src/core/          引擎本体（25 文件）
    static_atom.py     ★ base_sublevels() = 能级基准唯一处
    static_pair.py     两体相互作用（V_dd / 交换耦合 / vdW）
    atomic_levels.py / field_build.py / waveforms.py
    runspec.py         ★ 唯一分派点：登记册快照 → RunSpec → 跑 → 结果摘要
    sambe.py / hard.py / evolve_mixed.py / evolve_twofield.py / evolve_rwa.py
    evolve_reference.py / evolve_sambe_pair_gpu.py / dop853.py / coupling.py
    field_loss.py      场致加热与损失（L2 散射 / L3 TrapModel / L4 光电离）
    param.py           ★ 就地参数声明（Param 描述符；无覆写栈）
    protocol.py        ★ 领域面协议（供 mecha_v2 适配成 mecha 的引擎协议）
    manybody/          多体 ED 引擎家族
mecha_v2/          ★ EL 自己的 v2 领域层（唯一同时 import core 与 mecha 的胶水）
    engine.py          ElEngine：实现 mecha 的 EngineProtocol（schema/run/summary/…）
    tools.py           build_tool_registry：AI 工具面（14 个 MCP 工具）
    bridge.py          V2Bridge：GUI ↔ 同一道 Gate 的操作者适配器（+ ElOps）
    hub.py             ToolRegistry → MCP 服务器的投影 + 独立进程入口
src/v2_bind.py     装配点 make_v2_bridge + 只读监控端点（动态端口；写 .monitor-port）
src/ui/            GUI（16 个 .py，含 _mecha_bind.py：widget ↔ mecha 绑定层）
src/sweep/         并行扫描（4 文件）
src/_archive/      封存的 Floquet 权威锚（仍被判据引用）
checks/            45 个 .py（43 判据 + checklib/pair_evolve 两个支撑模块）
dsh/               dsh（Cordis）集成插件：自愈 MCP 桥 + 监控面板（TS）
docs/              文档（顶层 38 份 + _history/ 归档；索引见 docs/README.md）
```

**依赖方向（单向，判据强制）**：`core`（纯物理，不认识框架）← 胶水层 → `mecha`。
胶水层 = EL 自己的 `mecha_v2/`（engine/tools/bridge/hub）+ 装配点 `src/v2_bind.py`
（旧的 `src/adapter/` 已随 v1 退役，**不在本仓**）；`ui / sweep / dsh → mecha`。
`core ↛ mecha` 由 `mecha_purity_selfcheck` 看守。

## 4. 外部依赖：`mecha`（AI 机甲纪律机器，**硬依赖**）

框架本体**不在本仓**，是独立库。本机实测（2026-09-25）：`pip show mecha` → version **0.2.0**、
Editable project location `D:\code-nosync\mecha`；`import mecha.gate` →
`D:\code-nosync\mecha\mecha\gate.py`。

```
D:\code-nosync\mecha                              (v0.2.0, editable 安装)
    mecha/    gate(唯一写入口) / history(append-only) / authority(写权模式) /
              surface(能力面) / tools / assembly(具名装配点) / data_layout /
              monitor / write_lease / commands / roles / scopes / visibility /
              artifacts / toolhost / format_version
              providers/  approval / codeact / jobs / log_local / skills / spill
```

- **依赖声明**：`requirements.txt` 用**绝对路径** `-e D:\code-nosync\mecha` 声明——EL 与
  mecha 已**不在同一父目录**。0.2.0 的 `[project.optional-dependencies]` 是**空的**、
  也没有 `cockpit` 包 ⇒ **不能**写 `mecha[...]`。
- **它是硬依赖，不是"可选 AI 依赖"**：`app_entry.py:53-54` 顶层
  `from src.v2_bind import make_v2_bridge` → `mecha_v2.bridge` 顶层 `import mecha.*`
  （**无 try/except**）⇒ 缺 mecha 时软件直接 ImportError 起不来，**没有降级路径**。
  AI 模式另需 `mcp>=2,<3` + `uvicorn`，两者已在 `requirements.txt` 真声明。
- **`mecha-sdk`（0.1.0，`D:\学\...\mecha-sdk`）是已退役的旧库，EL 跑不了它**：它没有
  `mecha.authority.Mode`，而 `src/v2_bind.py:214` 需要 `Mode`。凡文档里写 "mecha-sdk" 的
  地方请按"历史件"读（各文档顶部有横幅）。
- 三件套：**spec（合规契约）+ Python SDK（合规实现）+ dsh 插件（驾驶舱 UX）**。

## 5. 怎么跑

```powershell
# 基线：判据全档（约 10 分钟）；分档 --core（约 20s）/ --fast（约 150s）
#   --list 看判据清单，--only <子串> 单跑
$env:PYTHONPATH=''; $env:QT_QPA_PLATFORM='offscreen'; $env:PYTHONIOENCODING='utf-8'
& D:\python310\python.exe run_checks.py

# 正常使用：看门人 launcher 弹窗选模式（AI / GUI），进程交接与写权落位都由它编排
& D:\python310\python.exe launcher.py

# 等价单跑——GUI 模式：纯本地软件，不起任何端点，写权在人
& D:\python310\python.exe app_entry.py --gui

# 等价单跑——AI 模式：headless 权威，进程内起 MCP 服务器 + 只读监控端点（动态端口）
& D:\python310\python.exe app_entry.py --authority
#   两个发现文件都写在项目根：.mcp-port（MCP 端点）/ .monitor-port（监控端点）；
#   dsh 插件的面板自己经同源路由 /re0-mecha/monitor-url 取地址，人不用抄端口。

# 独立进程 Hub（无 GUI 的 headless 部署；写权威落 --data-dir，默认 .mecha-v2-hub）
& D:\python310\python.exe -m mecha_v2.hub --root .
```

> MCP 端口是**随机空闲口**，由权威写进项目根 `.mcp-port`；dsh 插件每次建会话现读它
> （端口漂移自愈）。旧文档里的 `python -m src.adapter.serve` 与
> `python -m cockpit.server` 都是**死命令**——`src/adapter/` 与 `cockpit` 随 v1 退役，
> 本仓已不存在。

AI 侧调用指南（两份**需手工同步**）：`skills/energy-level/SKILL.md`（**权威源，入库**；
其「工具清单」表被 `checks/docs_consistency_selfcheck.py` 与真注册表双向互钉）与
`.dsh/skills/re0-mecha/SKILL.md`（dsh 工作区副本，gitignored）。两份都对
`mecha_v2/tools.py` 注册的 **14 个** v2 工具。

## 6. 基线（实测）

```
##RE0_SUMMARY scripts=46 fail=0 seconds=609
```

覆盖：核心物理（ARC / AC-Stark / 偶极）、三引擎对拍封存 Floquet 权威锚、两体 vs `pairinteraction`
第三方权威、耗散与非幺正、场致加热损失模型与接线、多体 ED（M0 全精确标定 / M1 解析解 / 语义层接入）、
两页能级同基准、引擎契约与参数归属、机甲语义内核（含**无侧门 AST 致死刑**）、MCP 通道、
只读监控端点与只读守卫、CodeAct 边界、GUI 接线与运行流、sweep、建场层。

## 7. 已知未解决（诚实交代）

> 逐条登记（含证据与最小改法）见 **`docs/待办-经复核的问题清单.md`**。以下为要点。

1. **框架提炼在"实例②"上被证伪**：独立库 `mecha-sdk` 自称领域无关，但 `Get Paper` 试接后写了
   `Get Paper/docs/MECHA-INDICTMENT.md`——六条指控（物理方言进强制 ABC、`ops.py` 无条件
   `import numpy`、`server.py` 身份硬编码 `re0-mecha`、抢 repo 根 `.mcp-port`/8765、`undo` 仅 LIFO、
   pickle RunSpec）**在当时 mecha-sdk HEAD 仍成立**；Get Paper 已自建方案未用本库。
   ⇒ spec 的"两实例落地再冻 v1"这条验收**当时不成立**。
   > ⚠ **2026-09-25 注**：本条针对的是**已退役的 `mecha-sdk` 0.1.0**。EL 现运行的框架是
   > `mecha` 0.2.0（`D:\code-nosync\mecha`，模块集完全不同，见 §4）。这六条指控在 0.2.0 上
   > **是否仍成立尚未复核**——本仓不做结论，只标注前提已变。
2. **提炼守卫只守 import**：`checks/mecha_purity_selfcheck.py` 是 AST import 扫描，且 `numpy` 在
   白名单里 ⇒ 上面六条语义级偏差全在盲区（46/0 全绿与六条指控并存，不矛盾）。
3. **`dsh/` 尚未独立成仓**：`docs/dsh插件-发布骨架.md` §2 的五项发布动作一项未勾。
4. **`field.E_vpcm` 量级待裁决**：表单允许 ±1e6 V/cm，而 n=60 的 Inglis-Teller 极限约 25 V/cm。

## 8. 纪律（沿用，未变）

1. **先证明等价，再换路**——逐位对拍（`max|Δ| = 0`），不接受"接口通了"。
2. **覆盖只迁移，不删除**——判据换锚到新结构，子条件只增不减；文档过时**归档**不删。
3. **每步一个可回滚提交**，先跑基线绿再动手。
4. **不套层**——改引擎本身，不在外面加适配皮；新层必须是单一定义点。
5. **注释写事实，不写编年史。**
6. **假绿比红更危险**——SKIP 显式计数，每个判据要有致死性证明。
