# Energy Level · Re0 —— Rb-87 里德堡原子仿真软件

> **一句话**：把"静态能级计算 + 动态原子演化 + 多原子阵列"做成一台**人机共用**的仿真设备——
> 人用 GUI 开、AI 经 MCP 语义层开、两者共用**同一份**语义状态与同一条事件总线。
>
> **当前形态（2026-09-22）**：新主干（Re0）已覆盖旧版；引擎 + GUI + sweep + 多体 + 机甲语义内核
> 全在。**AI 机甲框架已提炼为独立库 `mecha-sdk`**，本仓不再持有 `src/mecha` / `src/cockpit`。
> 索引见 `docs/README.md`（**读文档前先看它**）。

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
| **GUI** | PyQt5，`app_entry.py` → `src/ui/*` | ⚠ 布局逐像素未动（M3 无偏离）；god-mode 检视器保留（引擎常量只读）。**但有已知失能**：in-process 模式可用（`RE0_GUI_INPROC=1`），**Hub 客户端模式是坏的**（`hub_bridge.py` 读错键 + 订阅载荷形状不符）；**「窗口/光子块」两控件静默失能**（迁移后无人消费）；两模式门控 `set_readonly` 从未被 MainWindow 调用。逐条见 `docs/待办-经复核的问题清单.md` §A1/§C1–C3 |
| **AI** | MCP 写通道（`python -m src.adapter.serve`），**18 个工具** | ✅ 语义寻址 / 可教学错误 / 体积闸 / 指针化 / async 作业 / CodeAct 编排 / 建场层 |
| **远程跨设备** | 只读：cockpit `--host 0.0.0.0 --token`；写：`--allow-remote-write`（仅仿真 L2） | ⚠ **只读监控已可用**；远程写有接缝但未作为正式能力验收（T6：写通道默认只 localhost） |

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
    protocol.py        ★ 领域面协议（供 adapter 适配成 mecha 的引擎协议）
    manybody/          多体 ED 引擎家族
src/adapter/       ★ 唯一同时 import core 与 mecha 的胶水层
    __init__.py        EngineAdapter：实现 mecha 的 EngineProtocol
    fields.py          建场层（领域动词 set_drive/ramsey/stirap/…）
    serve.py           Hub launcher（人运行 `python -m src.adapter.serve`）
src/ui/            GUI（15 文件）+ _mecha_bind.py（widget ↔ mecha 绑定层）
src/sweep/         并行扫描（4 文件）
src/_archive/      封存的 Floquet 权威锚（仍被判据引用）
checks/            46 个判据 + checklib
dsh/               dsh（Cordis）集成插件：自愈 MCP 桥 + 驾驶舱面板（TS）
docs/              文档（顶层 24 份现行 + _history/ 归档；索引见 docs/README.md）
```

**依赖方向（单向，判据强制）**：`core`（纯物理，不认识框架）← `adapter` → `mecha`；
`ui / sweep / dsh → mecha`。`core ↛ mecha` 由 `mecha_purity_selfcheck` 看守。

## 4. 外部依赖：`mecha-sdk`（AI 机甲纪律机器）

框架本体**不在本仓**，是独立库：

```
D:\学\BaiduSyncdisk\编程\code\python\mecha-sdk     (v0.1.0, editable 安装)
    mecha/    registry(typed) / session(唯一 choke point) / bus(append-only) /
              authority / autonomy / ops / jobs / codeact / server(MCP) / errors /
              archives / interlock / hub_client + hub_bridge / protocol(EngineProtocol 缝)
    cockpit/  黑匣子只读 Web 投影（独立进程读磁盘 journal，项目死后仍在）
    docs/spec-draft.md   合规规范 draft v0.1
    examples/            ToyEngine + SchemaOnlyAdapter（两类项目的最小接入）
```

三件套：**spec（合规契约）+ Python SDK（合规实现）+ dsh 插件（驾驶舱 UX）**。

> ⚠ **本仓 `requirements.txt` 尚未声明 `mecha-sdk` 依赖**（当前靠 editable 安装）。换机器或打包
> 前必须补，否则 `import mecha` 直接失败。

## 5. 怎么跑

```powershell
# 基线：46 判据 / 0 FAIL（约 10 分钟）；--list 看判据，--only <子串> 单跑
$env:PYTHONPATH=''; $env:QT_QPA_PLATFORM='offscreen'; $env:PYTHONIOENCODING='utf-8'
& D:\python310\python.exe run_checks.py

# 起 Hub（MCP 写通道，localhost；人运行 = 权威，harness 只 attach 读 .mcp-port）
& D:\python310\python.exe -m src.adapter.serve

# 起驾驶舱（独立只读进程，读 .mecha/journal；项目死了也还在 —— 黑匣子）
& D:\python310\python.exe -m cockpit.server
```

AI 侧调用指南：`.dsh/skills/re0-mecha/SKILL.md`（工作区副本）与 `mecha-sdk/mecha/SKILL.md`
（领域无关版）。**注意**：副本里"权威源 = `src/mecha/SKILL.md`"已失效——那个位置不存在了。

## 6. 基线（实测）

```
##RE0_SUMMARY scripts=46 fail=0 seconds=609
```

覆盖：核心物理（ARC / AC-Stark / 偶极）、三引擎对拍封存 Floquet 权威锚、两体 vs `pairinteraction`
第三方权威、耗散与非幺正、场致加热损失模型与接线、多体 ED（M0 全精确标定 / M1 解析解 / 语义层接入）、
两页能级同基准、引擎契约与参数归属、机甲语义内核（含**无侧门 AST 致死刑**）、MCP 通道、
cockpit 黑匣子与只读守卫、CodeAct 边界、GUI 接线与运行流、sweep、建场层。

## 7. 已知未解决（诚实交代）

> 逐条登记（含证据与最小改法）见 **`docs/待办-经复核的问题清单.md`**。以下为要点。

1. **框架提炼在"实例②"上被证伪**：独立库 `mecha-sdk` 自称领域无关，但 `Get Paper` 试接后写了
   `Get Paper/docs/MECHA-INDICTMENT.md`——六条指控（物理方言进强制 ABC、`ops.py` 无条件
   `import numpy`、`server.py` 身份硬编码 `re0-mecha`、抢 repo 根 `.mcp-port`/8765、`undo` 仅 LIFO、
   pickle RunSpec）**在当前 mecha-sdk HEAD 仍成立**；Get Paper 已自建方案未用本库。
   ⇒ spec 的"两实例落地再冻 v1"这条验收**尚不成立**。
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
