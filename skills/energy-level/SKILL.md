---
name: energy-level
description: 通过 MCP 操控 Rb-87 里德堡原子仿真软件 Energy Level（mecha v2）——管理能级组、场与微波/光波形，运行含时演化，并解读布居转移结果。含单位约定、v2 工具面、典型工作流与已知物理坑。
whenToUse: 当需要操控 Energy Level 软件（改参数、加能级组、配波形、跑演化计算、取布居结果）或解读其计算结果时。软件由人经统一启动器 `python -m mecha --all` 起（写权在 AI 侧），MCP 端点见项目根 .mcp-port；GUI 模式下 AI 物理不可见（无端点）。
---

# SKILL: Energy Level — Rb-87 里德堡原子仿真（AI 操作指南 · mecha v2）

> 本文件是 AI harness 的 skill 说明书：连接方式、工具清单、工作流、单位约定、
> 物理坑。AI 使用中若发现文档与实态不符，应回头修正本文件（自我迭代）。
> **本版对齐 mecha v2**：工具面 = `mecha_v2/tools.py` 注册的 MCP 工具，共 **14 个**
> （清单见下方「## 工具清单」表；该表由 `checks/ai_surface_selfcheck.py` 的 C 组与真
> 注册表**双向互钉**，多一个/漏一个即判据红）。宿主 MCP 面**就是**这张注册表，
> **没有**额外的 `list_tools` 工具可调。v1 的 GUI 内嵌 AI 面
> （run_python/submit_sweep/get_diag/screenshot/set_groups…）**在 v2 不存在**，
> 见末节"v2 暂缺的能力"。

## 连接

**软件怎么起**（人经统一启动器 `python -m mecha --all`；切写权**随时可切**）：

- **默认（headless）**：软件进程（无 GUI 窗口）+ dsh。软件进程
  起 MCP 服务器，端口写项目根 `.mcp-port`（裸端口，兼容 dsh `resolveHubUrl`），
  你 attach 到 `http://127.0.0.1:<port>/mcp`。**写权自动在 AI 侧**（人授权了
  这个模式）。
- **人接管时**：写权切到人侧 ⇒ ⚠ **你的写会被当场拒**（**进程不重启、连接不断**），
  但**读仍照常**。等人在面板上点「交还给 AI」即可。
  ⚠ 另有「纯本地」形态（`app_entry.py --gui`，独立进程、不起任何端点，AI 物理看不见它）
  —— 那不是 mecha 起的。

**端点换端口/换进程**（实测报告 §0）：权威重启后端口会变（自动选空闲口）。
**re0-mecha dsh 插件已自愈**：每次重连现读 `.mcp-port`，漂移后下一次调用自动
连上新端口，无需重启 dsh。但**插件之外的客户端**（自写 MCP 客户端、写死 URL 的
IDE 配置）不会自愈——那些须新开会话/改配置。

**监控面**：权威进程另起**只读**监控端点（`/status`、`/activity`、`/history`、
`/config`），dsh 右侧栏「机甲监控」面板消费它——飞行记录仪（每笔写
seq/actor/before→after/reason，逐行可折叠）+ 配置态（族→前缀分层树）。
**端口是动态分配的**（OS 选，写在项目根 `.monitor-port`）；面板自己经 dsh 同源只读
路由 `/re0-mecha/monitor-url` 取地址，**没有人需要抄端口**，也没有硬编码端口可漂移。
**控制权不经你这条端点**：人的夺权/交权走**控制端点**（`.control-port`，只给框架/面板
⇒ 你**拿不到它的地址**）。你要知道自己此刻能不能写，就调只读的 `read_write_authority()`。
AI 面改不了写权模式、伪造不了 human 归因（Authority 的通道判定，与端点无关）。

**其他 IDE / 客户端接入**：本仓**当前没有** `.trae/mcp.json`（该配置曾存在，已不在
仓内——2026-09-25 核实）。要用别的 MCP 客户端，请让它指向
`http://127.0.0.1:<.mcp-port 里的端口>/mcp`，并在该客户端里**新开会话**生效
（工具在会话启动时加载）。注意这类写死 URL 的客户端**不会**像 dsh 插件那样自愈。

## 工具清单

| 工具 | 作用 | 关键参数 |
|---|---|---|
| `query_params` | 可写配置键的元数据：type/unit/lo/hi/enum/desc/default | — |
| `query_levels` | 基组每个子能级：**label + 身份键 key + 能量 MHz** | — |
| `read_config` | 当前所有已设配置键（快照）——确认改动是否生效 | — |
| `read_waves` | 当前驱动波清单（逐波摘要，不内联大数组） | `atom("single"/"A"/"B")` |
| `set_config` | 写一个配置键（过门校验；类型/枚举/值域/wave_list 当场判）。回执自带回读：`before`（写前旧值）+`resolved`（落地的值）+`readback`（读回的值）——写一次就自证，不必再查 | `key, value, reason`（中文摘要进留痕） |
| `reset_config` | 把当前所有已设键清空回默认（含驱动波）——**上手先收拾台面用它**。回执列 `cleared` 清单 | `reason` |
| `set_drive` | **在跃迁上建驱动**：给 Rabi/光强 + 脉冲形状 → 自动算频率/光强/时长。可选参数（pol/detuning_mhz/mode/atom/name）**不传即用默认**（π/0/append/single/drive） | `src, tgt, rabi_mhz｜intensity_mwcm2, pulse("pi"/"pi2")｜duration_ns, pol, detuning_mhz, atom("single"/"A"/"B"), mode("append"/"replace"), name` |
| `estimate_run` | dry-run 成本（不跑演化）；单体回 `est_sec` 粗估 | — |
| `preview_run` | 行为预览：驱动哪些跃迁 / **基组够不够**（`basis_enough`） | — |
| `run_sim` | 跑演化，回终态布居摘要（不含大数组） | `cancel_after_sec`（可选超时取消） |
| `submit_job` | 长任务提交后台，立即回 `job_id`（提交时刻固定配置） | — |
| `read_job` | 查任务：给 `job_id` 查单个（done 含 result），省略则列全部 | `job_id` |
| `cancel_job` | 请求取消后台任务（下一步边界生效） | `job_id` |
| `run_actions` | **多步编排**：一串 `{op,args}` 按序执行，任一步失败即停在该步 | `actions` |

### 两种驱动形态的选择
| 场景 | 用什么 | 原因 |
|---|---|---|
| 单步探索、需人逐步监督 | 细粒度工具（`set_config`/`run_sim`/…） | 每步在对话+GUI 双通道可见 |
| 参数扫描、批量对比、多步组合 | `run_actions`（CodeAct 结构化动作） | 一次调用=多步，省回合 |

`run_actions` 约定：
- 每步是 `{"op": 名称, "args": 对象}`。op 可为内置 `estimate`/`summary`/`run`/`schema`
  （顶层工具名 `estimate_run`/`run_sim` 也可直接当 op 用），
  或只读查询名 `read_config`/`query_params`/`preview_run`/`query_levels`/`read_waves`。
- **不 eval 任意代码**（库只编排自家工具，任意代码是宿主 code-runtime 的权力）。
- run/estimate 的配置三种给法：`args.spec` 整表替换；**`args.spec_patch` 只改列出的键、
  其余沿用当前配置（扫描首选——别每点重抄全量）**；都不给=当前配置。顶层 `spec_patch`
  同义（步没给才生效）。
- **`pick`**：键路径列表（如 `["final_top_pop.60P_{3/2} m_J=-0.5","method"]`，点分层、
  整键优先匹配），投影每步 output 只回关注量——多点扫描控体积用它；写错的键如实列进
  `_missing`（不静默 null）。
- 回执 `{ok, steps:[{index,op,ok,detail,output}], stopped_at}`——**每步自带
  output**（不用再按位置对齐）与 detail（成功=短摘要，失败=原因）；`stopped_at=-1` = 全成。
- 每步留可追溯记录（逐子调用审计，不污染账）。

**失败长什么样**（机器可判别）：工具失败时回执 `isError=true`，正文是
`{"ok": false, "error": {"message", "info": {"kind", "hint", "suggest"}}}` 的
JSON——**读 `info.kind` 决定怎么改**（如 `unknown_key`/`bad_enum`/`bad_value`），
hint 给下一步指引。成功回执才是普通 JSON 正文。

## 配置键（`set_config` 的 key）

可写面 = `query_params` 返回的清单（唯一权威）。除标量外这些结构化键也可整表写：

- `groups.list`（必填）：`[{"n":60,"l":0,"j":0.5}, …]` 能级组；逐子能级裁剪见坑 6
- `field.B_gauss`(Gauss) · `field.E_vpcm`(V/cm)
- `dynamic.mode`(0单原子/1双原子/2多体) · `dynamic.method`(`mixed`｜`sambe`｜`hard`) ·
  `dynamic.t_max`(ns,>0) · `dynamic.npts`(≥2) · `dynamic.decay_index`(0自动/1禁/2自定义)
- `dynamic.psi_single` / `psi_A` / `psi_B`：初态**身份键**（query_levels 的 `key`，非索引，跨 B 稳定）
- `dynamic.psi_pair`：两体纠缠初态 spec（见坑 4），两体模式下**取代** psi_A/psi_B
- `dynamic.decays`：自定义衰变通道 list，元素 `{'from':态键,'to':态键|None,'gamma_mhz':Γ}`（见坑 5）
- `dynamic.waves_single` / `waves_A` / `waves_B`：波形——**用 `set_drive` 建、`read_waves` 读，别手搓**
- `solve.*`：运行级旋钮（`MAX_STEP`/`MAX_STEP_HARD`/`PER_FREQ_M`/`FROZEN_H_NS`/`COST_BUDGET`）
- `engine.*`：**只读引擎常量**，`set_config` 会被当场拒（`readonly_key`）——改模型不是运行级的事

## 单位约定（务必遵守）

- 磁场 B：**Gauss**；静电场 E：**V/cm**
- 微波频率：**MHz**（波形 freq_mhz）；光频：**Hz**；光强：**mW/cm²**
- 时间轴 t_max / 波形包络参数：**ns**；衰变率 Γ：**MHz**
- 能级组：n 主量子数（1-200）、l 轨道角动量（0-199）、j 半整数/整数

## 典型工作流

**复现一次微波 π 脉冲**：
1. `set_config('groups.list',[{"n":60,"l":0,"j":0.5},{"n":60,"l":1,"j":1.5}])`
2. `query_levels` → 拿 60S / 60P 各子能级的**身份键 key**
3. `set_config('dynamic.psi_single', <60S 的 key>)` 设初态（用 key 不用 label）
4. `set_drive(src=<60S key>, tgt=<60P key>, rabi_mhz=5.0, pulse='pi', pol='σ-')`
   —— 频率/光强/时长自动算
5. `set_config('dynamic.method','hard')`、`set_config('dynamic.t_max',500)`
6. `preview_run` 看基组够不够 → `estimate_run` 看耗时 → `run_sim` 跑
7. 从回执 `final_top_pop` / `top_states_avg_pop` 解读布居转移

**两体纠缠初态（"交换后的结果"当初态）**：
`set_config('dynamic.psi_pair', ['pair_sup', [[nA,lA,jA,mA, nB,lB,jB,mB, re, im], …]])`
→ `set_config('dynamic.mode', 1)` → `submit_job` 异步跑。注意对称组合（两 amp 都 1）
是**暗态**，布居不摆（坑 4）。

**参数扫描**：`run_actions` 串多步，每步 `{"op":"run","args":{"spec_patch":{只写变化量}}}`，
顶层带 `pick` 只回关注量（如 `["final_top_pop.<目标态label>"]`）——别每点重抄全量 spec、
也别把 50 点全量布居灌进上下文。或循环调 `submit_job` + `read_job` 轮询。
别往 context 灌大数组（run_sim 只回摘要，全量在 artifact）。

## 波形格式（`set_drive` 建、`read_waves` 读；手搓 `set_config` 时的元素形状）

```json
{"name":"波形1","freq_mhz":6800.0,"intensity":0.05,"pol":"π",
 "offset_ns":0.0,"env":["gauss",200.0,40.0,1.0]}
```
- env 高斯：`["gauss", 中心ns, σ_ns, 峰值]`
- env 方波：**必须 5 个元素** `["square", 起点ns, 沿ns, 高度, 下降起点ns]`
  （第 5 个是平台结束/开始下降的时刻；只给 4 个会被反序列化器判"波形数据不合法"——
  实测踩过。平台到演化末尾就写 `t_max`）
- env 文件（光强包络）：`["file", t_ns 数组, amp 数组]`——保留载频，三引擎全支持
- pol：微波/光场用 π/σ+/σ-；椭圆偏另带 `'elliptical': {'alpha_deg','theta_deg','eps_deg'}`
- **完整驱动（`type:'arb'`）**：无载频的完整时域信号，只有 `hard` 直接支持，
  `sambe`/`mixed` 遇它自动改道 hard（回执 `arb_note` 提示）

### ⚠ 光强包络的两个语义陷阱（实测标定，务必按配方换算）
`FileEnv` 名为"光强包络"，但实现上：
1. **峰值归一化**——文件幅值被除以自身峰值，绝对标度丢弃。整体标度只能靠
   `intensity` 字段，峰值永远钉死在 `intensity` 上。
2. **它是振幅因子，不是光强因子**——耦合 ∝ √I·env，即 **光强(t) = intensity × env(t)²**。

**→ 想要 I(t) = I₀·(1+δ(t))（δ ∈ [−a,+b]）的正确配方**：
```
intensity = I₀ · (1 + b)                 # 峰值
env 采样   = √[ (1 + δ(t)) / (1 + b) ]    # 峰值恰为 1
```
**反面教材**：把绝对光强曲线直接塞进文件 → 被静默归一化 → 整体光位移悄悄错掉。

## 引擎支持面（现行 sambe / hard / mixed 三款）

| 场类型 | sambe | hard | mixed |
|---|---|---|---|
| 微波·任意波形（完整驱动） | 改道 hard | ✅ 直接支持 | 改道 hard |
| 微波·任意包络（FileEnv） | ✅ | ✅ | ✅ |
| 光学·任意包络（FileEnv） | ✅ | ✅ | ✅ |
| 光学·任意波形 | ❌ | ❌ | ❌ |

- 光学场不支持任意波形：无载频 → RHS 带 THz 相位 → 积分器被钉死。光学只能用 env 文件。
- **耗散通道支持面**：`hard`/`mixed`（单、两体）只支持**丢失**（`to=None`）；显式落点/纯退相
  仅**单原子 + `sambe`**（转 Lindblad 主方程）支持。要跑"自发辐射落回基态"这类内空间转移，
  必须 `dynamic.mode=0` + `dynamic.method='sambe'`。

## 坑（已知 · 物理/引擎级，v2 仍成立）

1. **mixed 纯微波亦可**：没光场时光子维退化 P=1，等价 hard 同物理。会强制改道的只有
   **任意波形（完整驱动）**；光强包络（FileEnv）是普通 Wave，不受影响。
2. **run_sim 同步预处理可能久**：两体 V_dd 预计算可达数十秒。长任务改用 `submit_job`
   异步 + `read_job` 轮询，别同步干等。
3. **双原子模式**：`dynamic.pair.*` 仅双原子模式生效；切模式
   `set_config('dynamic.mode', 0|1)`。
4. **两体纠缠初态暗态**：`psi_pair` 对称组合 `(|ab⟩+|ba⟩)/√2` 只吃全局相位、布居不动。
   想看交换摆动，要么给相对相位（如 `[0,1]`=i），要么用逐原子直积初态。分量按
   (n,l,j,m) 解析（不是索引），缺项报错列出，绝不静默丢。
5. **初态键别凭能量猜**：`psi` 用身份键（query_levels 的 `key`）。猜错能量会**静默选到
   别的能级**（实测落到 59D_3/2 却报 ok）。
6. **🚨 基组裁剪要给"观测量收敛证据"，否则结论不可引用（灾难级）**：同一脉冲，7 组
   → C=0.006、13 组 → C=0.9982（差在缺 59D_3/2/60D_5/2 时某双光子通道不被抵消）。
   做法：≥3 个基组（如 7/10/13 组）跑同一脉冲，观测量变化 ≤1% 才敢下结论。**主判据
   永远是观测量；配对数只是诊断线索**。逐子能级裁剪：`groups.list` 元素可带
   `ms:[mJ,…]`（与 `Fs:[[F,m_F],…]` 互斥）只留组内特定子能级；不写=全展开（逐位一致）。
   ⚠ 不能按"6P 只留两三个"裁——mixed 的光移只来自基组内态对的 ARC 矩阵元，
   **没有 ARC 伙伴态注入** ⇒ 基组里没有 6P，6P 的贡献就是缺失的（漏算）。
7. **多光子通道要求中间态在基组里**：做 60P→58G 三光子踢出，基组里必须有 59D 与 58F
   作中间态；只放 60P/58G 会"频率对、强度对，却一点都踢不动"。
8. **结果反直觉时的诊断套路**：① 布居**完全不动** → 看目标跃迁在不在保留集里（多为
   基组缺中间态，或成本窗砍了远失谐梯级）；② 跑得**异常慢** → 光强过大 Ω≫失谐 →
   步长被钉死，先降 intensity 别硬等。
9. **零波形 = 跑不了**：没波形也没实际耗散通道时，演化**拒绝启动**（既有设计，空演化
   没意义，不是缺陷）。静止类验收（只验初态/基底）必须加一个"不落在时间窗内"的脉冲
   （中心远在 t_max 外的窄高斯）。
10. **`solve.MAX_STEP` 只是"初始步长上限"**：之后步长由误差控制自由增长。要**真**压
    步长用 `solve.MAX_STEP_HARD`（默认关；只对 mixed/hard 的 DOP853 生效，sambe 走自带传播子）。
11. **软件实例的生命周期归人管**：不自己 `python app_entry.py`、不杀 python 进程；需要新代码
    生效就报"需要重启"，由人重启。
12. **教训：写入侧通了 ≠ 引擎侧通了——验收必须打到引擎**：任何"接口已实现"的结论，
    必须用**真引擎跑一次**（哪怕合成小基组）才算数。
13. **π 脉冲标定别手搓方波**（实测第二轮 §2）：手搓 20 ns 沿的方波在同一时长给
    0.993839，而 `set_drive` 自动算的 π 脉冲（1 ns 沿）给 1.000000——**脉冲沿吃掉
    有效面积**。做标定必须用 `set_drive`（或显式控制沿），否则把"沿的锅"读成物理。
14. **t_max 不跟随脉冲时长——实验设计陷阱**（实测第二轮 4-3）：`run_sim` 回的是
    **演化窗口末端**的布居。t_max 晚于脉冲结束 ⇒ 读到的是"脉冲后又自由演化了一段"；
    周期恰好整除时看不出来，换个 Ω 就读错。`set_drive` 回执带 `pulse_end_ns` 与
    `t_max_note` 对账提示（不拦止——自由演化可能是故意的）；扫描时长时把 t_max
    与脉冲时长一起 patch。

## v2 暂缺的能力（v1 有、v2 未接线——别按 v1 名字调）

以下在 v1 的 GUI 内嵌 AI 服务里存在，**v2 的 MCP 面不提供**，需要时走"下一步"：

- **`run_python` / 任意代码执行**：v2 用 `run_actions`（结构化动作，不 eval）。
  要真跑 Python 是宿主 code-runtime 的权力，不在库面。
- **`submit_sweep` 并行扫描**（多进程网格）：v2 未接线。当前用 `run_actions` 串多步
  或循环 `submit_job`。多进程扫描器属 Provider 接线推迟项。
- **`get_diag` 运行诊断**（pairs/steps/sectors/traj/H）：v2 未接线。诊断信息暂从
  `run_sim` 回执的 `warnings` + `preview_run` 读。
- **`screenshot` / `undo` / 存档面**：
  - 操作流水（v1 的 `get_activity`）——**不在 MCP 工具面**，在只读监控端点的
    `/history`（端口见项目根 `.monitor-port`；dsh 右侧栏「机甲监控」面板消费它，
    飞行记录仪页逐笔可折叠）；域当前态用 `read_config`；
  - `undo`——v2 史 append-only，撤销=反向 set 记新事件，等接 GUI 撤销按钮（ADR 已定设计）；
  - 存档面——journal 即存档，要读历史直接开 `.mecha-v2/journal.jsonl`；
  - `screenshot`——GUI 侧能力，非 MCP 工具。
- **`set_groups`/`set_superposition`/`set_pair_superposition`/`add_level_group` 等具名写工具**：
  v2 统一走 `set_config`（写 `groups.list`/`dynamic.psi_*`）。裁剪/叠加态的写法见上"配置键"与坑 6。

## 副本与同步责任（三份 SKILL）

同一套 AI 指南在本仓有**三份**，职责不同。**漂移了以本文件为准**：

| 文件 | 角色 | 谁读 |
|---|---|---|
| `skills/energy-level/SKILL.md` | **权威**（入库） | 人 / 工具 |
| `.dsh/skills/energy-level/SKILL.md` | **镜像**（须与权威**逐字节相同**） | dsh 加载 |
| `.dsh/skills/re0-mecha/SKILL.md` | **工作副本**（另一份文档：独立坑列表 + v1→v2 工具名对照） | **AI 实际读的那份** |

改法与纪律：

- 先改**权威**，再同步镜像与工作副本。工具面的**唯一来源是注册表**
  （`mecha_v2/tools.py`）；本文件的「工具清单」表与它互钉，**不许在别处再手抄一份**。
- ⚠ **`.dsh/` 是 gitignored**：`git diff` / `git status` / 门禁**都看不见那两份**——
  历史上正因此漏改过一处固定端口，而它恰是 AI 真正读的那份。查漂移要**按内容扫**：
  `python run_checks.py --only ai_surface`（禁固定端口、禁退役工具名的活指令、
  三份工具面一致、镜像逐字节一致）。
- 新增第四份副本时**必须**登记进 `checks/ai_surface_selfcheck.py` 的 `SURFACES`：
  未登记的文件会被该守卫判为违规（"漏登记"本身违规）。
