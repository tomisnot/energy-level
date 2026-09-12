---
name: energy-level
description: 通过 MCP 操控 Rb-87 里德堡原子仿真软件 Energy Level——管理能级组、场与微波/光波形，运行含时演化，并解读布居转移结果。含单位约定、v3 登记册 AI 可及面、典型工作流与已知坑。
whenToUse: 当需要操控 Energy Level 软件（改参数、加能级组、配波形、跑演化计算、取布居结果）或解读其计算结果时。软件需处于运行状态，MCP 端点见项目根 .mcp-port（默认 8765）。
---

# SKILL: Energy Level — Rb-87 里德堡原子仿真（AI 操作指南）

> 本文件是 AI harness 的 skill 说明书：连接方式、工具清单、工作流、
> 单位约定、坑。AI 使用中若发现文档与实态不符，应回头修正本文件
> （自我迭代，G4）。

## 连接

软件启动后自动开放 streamable HTTP 端点：默认端口 **8765**（环境变量
`EL_MCP_PORT` 可改；被占自动回退随机端口），实际端口始终写入项目根
`.mcp-port` 文件。

**Trae IDE 接入（本项目已配好）**：项目根 `.trae/mcp.json` 已写入：

```json
{"mcpServers": {"energy-level": {"url": "http://127.0.0.1:8765/mcp"}}}
```

需在 Trae 设置 → MCP 打开"启用项目级 MCP"，然后**新开会话**生效
（工具在会话启动时加载）。

其他 harness：Claude Code 用 `claude mcp add --transport http
energy-level http://127.0.0.1:8765/mcp`；通用 JSON 形态同上。
若端口非默认（GUI 状态栏"● AI 在线 :端口"可见），改配置或读 `.mcp-port`。

## 两种驱动形态的选择（重要）

| 场景 | 用什么 | 原因 |
|---|---|---|
| 单步探索、需要人逐步监督 | 细粒度工具（set_param/run_compute/…） | 每步在对话+GUI 双通道可见 |
| 参数扫描、批量对比、多步组合、结果后处理 | **run_python**（CodeAct） | 一次调用=任意多操作，省回合；大数组进程内消化 |

`run_python` 约定：
- **命名空间白名单（仅这些；没有 os / sys / subprocess / open）**：
  `summary() get_state(scope) set_param(path,v,reason='')
  add_level_group(n,l,j) remove_level_group(i) set_waves(list,atom)
  save_archive(path) load_archive(path)
  run_compute_and_wait(timeout=1800,poll=1.0,save_npz=False)
  activity(since=0) np(=numpy) json time ROOT`
- 要自己写文件请**拼字符串**：`np.savez(ROOT + '/ai_results/x.npz', ...)`。
  用 `os.path.join` 会直接 `NameError`——这是刻意的设计边界（状态变更必须
  全部经 ParamRegistry），不是缺陷。
- **把结论赋给 `result` 变量**（自动回传，JSON 化）；过程信息用 `print`。
- 数组不要整体回传（>50 元素会被截成形状摘要）——print 统计量或存 npz。
- undo 语义：整段脚本算**一步**，撤销回到脚本执行前。
- **结果自动落盘**（2026-09-10 加）：每次执行都写 `ai_results/<run_id>.json`
  （含 result / stdout / **完整代码** / 状态 / 耗时），保留最近 200 份。
  返回值含 `run_id`。所以**脚本末尾报错或客户端断线都不会丢结果**。
- 超长任务用 `run_python_async(code)` 立即拿 `run_id` 后台跑，再用
  `run_result(run_id)` 轮询；该查询**只读文件、不经 Qt 主线程**，因此长任务
  执行期间也能用（`run_python` 的脚本占用主线程，别的工具会被排队）。
  断线重连后用 `list_runs()` 找回之前那次跑的 run_id。
- 示例（B 场扫描一次调用）：

```python
out = []
for B in [20, 25, 30]:
    set_param('field.B_gauss', B, reason=f'扫描 B={B}')
    r = run_compute_and_wait()
    out.append({'B': B, 'final': r.get('summary', {}).get('final_pop')})
    print(out[-1])
result = out
```

## 工具清单

| 工具 | 作用 | 备注 |
|---|---|---|
| `summary` | 轻量摘要（组数/场/方法/波形数） | **每次会话先调这个建立认知** |
| `get_state(scope)` | 全量/分模块状态导出 | scope ∈ 空/groups/field/dynamic |
| `list_params(prefix)` | 参数 schema：type/unit/desc/range/options/value | set_param 的字典；**波形包络只给摘要** |
| `get_envelope(path, index)` | 取某条波形的**原始包络采样数组** | 大数组专用通道（list_params 只给摘要，防响应膨胀）|
| `set_param(path, value, reason)` | 改一个在册参数（登记面=可改面） | 过控件校验；reason 写动机 |
| `add_level_group(n,l,j,reason)` | 加能级组 | 自动校验+配色 |
| `remove_level_group(index,reason)` | 删能级组 | 索引见 get_state |
| `set_waves(waves, atom, reason)` | 整表替换波形 | atom=single/A/B（三套独立），格式见下 |
| `run_compute(diag=true)` | 启动演化计算 | 立即返回 job_id，GUI 同步显示。**diag=true（默认）**额外采集运行诊断供 `get_diag` 用（仅 mixed 引擎产出） |
| `get_diag(scope,job_id,limit,save_npz)` | **只读：运行诊断** | 理解"为什么"的主通道。scope 空=摘要；`pairs`/`steps`/`sectors`/`traj`/`H` 给明细（截断/落盘）。job_id 空=最近一次或 GUI 当前 |
| `job_status(job_id)` | 查进度 | 轮询到 done=true |
| `job_result(job_id, save_npz)` | 取结果摘要 | 大数组落 npz 给路径 |
| `cancel_job(job_id)` | 取消计算 | |
| `get_activity(since_seq)` | 操作流水（人+AI） | 了解"现在什么状况" |
| `undo(reason)` | 撤销最近一个 AI 操作 | 先斩后奏的兜底 |
| `save_archive(path)` / `load_archive(path)` | JSON 存档读写 | 与菜单同源，可 undo |
| `screenshot()` | 主窗口截图 → PNG 路径 | 了解人眼所见 |
| `run_python(code)` | **CodeAct**：一次调用跑多步编排 | 见上"两种驱动形态" |
| `run_python_async(code)` | **CodeAct 异步版**：立即返回 run_id | 长扫描/大系综；断线不丢结果 |
| `run_result(run_id)` | 取运行状态与结果 | 只读文件，**长任务期间也能查** |
| `list_runs(limit)` | 列出最近运行记录 | 断线重连后找回 run_id |

## v3 登记册的 AI 可及面（2026-09 参数管理改造）

可改面 = 登记在册面，`list_params()` 是唯一权威清单。除标量外这些
结构化参数也可 `set_param` 整表读写：

- `field.lights`：光场行 list（`frequency_hz`/`intensity`/`polarization`/
  `scope`/`enabled`，椭圆偏另带 `elliptical`）；
- `dynamic.waves_single` / `waves_A` / `waves_B`：波形（用 `set_waves` 更顺手）；
- `dynamic.psi_single` / `psi_A` / `psi_B`：初态**身份键**（非索引，跨 B 稳定）；
  读得到键即可写回；
- `dynamic.mode`：0 单原子 / 1 双原子（= 人点单选钮同路）；
- `dynamic.decays`：自定义衰变通道。

### 引擎/求解器控制量（2026-09 t83/t84：上帝模式的控制面已对 AI 开放）

同一个注册表里还有两类**控制参数**（键 = 全路径）：

- **`engine.*`** —— `src/core/params.py` 的 31 个**引擎**参数（默认值/分层/单位/
  说明见 `list_params('engine.')`）。最常用的几个：
  | 键 | 作用 | 何时动它 |
  |---|---|---|
  | `engine.PHASE_TOL` | 弱耦合丢弃门 Θ（rad，默认 1e-4） | 少留无贡献的弱对（提速）；调大可能丢多光子通道 |
  | `engine.R_CRIT` | AC 化边界 r=Ω/ω（默认 0.0135） | 强驱动下远强对被剔出光子块 |
  | `engine.MW_MAX_MHZ` | 微波/光场分界 + 候选窗绝对上界（1e5） | 一般不动 |
  | `engine.BW_OPT` / `OPT_NRES_BW_MHZ` / `AC_NEAR_CUTOFF_MHZ` | 各引擎的对过滤/分流窗 | 模型边界，改=换模型 |
- **`solve.*`** —— **运行级**参数（只管这一次怎么算，不改模型）：
  | 键 | 作用 | 默认 |
  |---|---|---|
  | `solve.COST_BUDGET` | 微波候选窗的**成本上限**（rad）；窗宽 = 值/(C·T) | **空 = 不限制** |
  | `solve.MAX_STEP` | DOP853 最大步长（ns） | 空 = 自适应 |
  | `solve.PER_FREQ_M` | 逐载频光子块 M | 空 = 自动探针 |
  | `solve.FROZEN_H_NS` | 两体纯冻结段长（ns） | 0 = 全自适应 |

**成本预算已从"引擎参数"改为"运行级参数"（t84 用户裁决）**：它曾经默认 4e4 rad、
在构造期就静默按成本窗删耦合对 —— 那是"为了跑得快悄悄改模型"。现在：
① 默认**不限制**，真实存在的耦合一律保留；② 人在点"运行"时会看到**成本报告**
对话框（保留对数 / 最远失谐 / 预计步数，可选"完整物理"或"剔除远对换速度"）；
③ AI 发起的运行**不弹窗**（`_ai_initiated` 旗标），走完整物理路径并只写日志。
**你要提速就显式 `set_param('solve.COST_BUDGET', 4e4)`**，并在结论里说明你
剔掉了哪些远失谐通道 —— 那等于改物理，必须交代。

改这些 = 改**会话真值**（进存档、可 undo、GUI 检视器即时回填+亮灯）。
只想"这一次试试"就用 CodeAct 的 `params.override`（见 `run_python` 说明），
别污染会话/存档。

**不入册 = AI 碰不到**（也不入存档）：显示态数 topn、荧光读出态/权重、
显示模式、关注跃迁；检视器的**窗口显隐/模式开关**属视图状态（AI 不改）。

## 语义状态面（二期 A，2026-09-12 —— 首选，别再手拼索引/键）

| 工具 | 用法 | 为什么 |
|---|---|---|
| `set_groups([(n,l,j),…])` | **声明式**让基组恰好包含这些组；原子、幂等，回 `{added,removed,kept,psi_warnings}` | 按索引增删 + 按标签字符串判重会**删错组/漏加组**（实测两次：`'60P'` 先匹配 60P_3/2、`'58G_{'` 漏掉 58G_9/2） |
| `list_levels()` | 基组内每个子能级 `(n,l,j,m)` + **合法身份键** | `psi` 的 `options` 是 null，**猜能量会静默选到别的能级**（实测落到 59D_3/2 且报 ok） |
| `set_initial_state(atom,n,l,j,m)` | 按语义身份设初态；不在基组 → **报错** | 同上，根治静默错态 |
| `get_config()` | 规范化运行配置（groups/mode/pair/psi/t_max/npts/方法/波数） | `get_state()` 与 `summary()` 字段曾不一致，脚本无法自检配置 |

运行时还会给**模型卡**（见下条），其中 `warnings` 会点名"微波候选对为 0（疑缺中间态）"
与"maxΩ/ω>1（出 RWA 适用区）"这类静默物理问题。

## 单位约定（务必遵守）

- 磁场 B：**Gauss**；静电场 E：**V/cm**
- 光频：GUI 显示 GHz，但参数 `field.lights` 的 `frequency_hz` 字段用 **Hz**
  （v3 登记册格式；6800 GHz → `6800e9`）；光强：**mW/cm²**
- 微波频率：**MHz**（波形 freq_mhz）
- 时间轴 t_max / 波形包络参数：**ns**；衰变率 Γ：**MHz**
- 能级组：n 主量子数（1-200）、l 轨道角动量（0-199）、j 半整数/整数

## 典型工作流

**复现一次微波 π 脉冲**：
1. `summary` 看现状
2. `add_level_group(60,0,0.5)` + `add_level_group(60,1,1.5)`（或
   `load_archive('save_state/微波驱动.json')` 一步到位）
3. `set_waves([{name:'π脉冲', freq_mhz:17236.743, intensity:5.5e-5,
   pol:'σ-', offset_ns:0, env:['gauss',70,20,1.0]}])`
4. `set_param('dynamic.method','hard')`、`set_param('dynamic.t_max',500)`
5. `run_compute` → `job_status` 轮询 → `job_result(save_npz=true)`
6. 从 `top_states_avg_pop` / `final_pop` 解读布居转移

**改参数扫描**：`set_param` 逐个改 → 每次 `run_compute`；演化数组落在
`tempsimu/ai_job*.npz`（`run_compute_and_wait(save_npz=True)` 会给路径），
用本地脚本读 npz 做精细分析，别往 context 灌数组。**脚本级的 result /
stdout / 代码**则自动落在 `ai_results/`（见上"结果自动落盘"）。

## 波形格式（set_waves 的 waves 列表元素）

```json
{"name":"波形1","freq_mhz":6800.0,"intensity":0.05,"pol":"π",
 "offset_ns":0.0,"env":["gauss",200.0,40.0,1.0]}
```
- env 高斯：`["gauss", 中心ns, σ_ns, 峰值]`
- env 方波：**必须 5 个元素** `["square", 起点ns, 沿ns, 高度, 下降起点ns]`
  （第 5 个是平台结束/开始下降的时刻；`serialize_waves` 就是这么输出的。
  ⚠ 只给 4 个会被反序列化器判"波形数据不合法: list index out of range"——
  实测踩过。平台到演化末尾就写 `t_max`）
- **env 文件（光强包络，AI 可直接下发）**：`["file", t_ns 数组, amp 数组]`
  —— 保留载频，**sambe / mixed / hard 三引擎全支持**，是"任意包络 / 随机
  抖动 / 任意分段调制"的正确通道。实测与同参数方波 `env` **逐位等价**。
- pol：微波用 π/σ+/σ-；光场同
- **完整驱动（`type:'arb'`）不是上面这个**：它的文件是**完整时域信号、无载频**。
  结算路径 = `arbitrary_mw_group()`（Hilbert 分解 amp(t)=env·cosφ）→ 并入微波组
  → **hard 的 `cos(φ_f(t))` 任意相位路径**。

  **引擎支持面（现行仅 sambe / hard / mixed 三款）**：

  | 场类型 | sambe | hard | mixed |
  |---|---|---|---|
  | 微波·任意波形（完整驱动） | 改道 hard | ✅ 直接支持 | 改道 hard |
  | 微波·任意包络 | ✅ | ✅ | ✅ |
  | 光学·任意包络（`FileEnv`） | ✅ | ✅ | ✅ |
  | 光学·任意波形 | ❌ | ❌ | ❌ |

  - **微波场支持任意波形**：hard 直接支持；**sambe / mixed 选中时会自动改道 hard**（GUI 日志有提示）。
  - **光学场不支持任意波形**：无载频 → 不参与光场旋转 → RHS 带 THz 相位 → 积分器被钉死
    （源码实测警告："2 ns 演化 >60 s 未完成"）。光学只能用上面的 env 文件（光强包络 + 载频）。
  - 任意波形暂不通过 `set_waves` 提供，需走存档。

### ⚠ 光强包络的两个语义陷阱（实测标定，务必按配方换算）

`FileEnv` 名为"光强包络"，但实现上：

1. **峰值归一化** —— 文件里的幅值会被除以自身峰值，绝对标度被丢弃。
   实测：`[1,1,1,1]` 与 `[0.5,0.5,0.5,0.5]` 在**同 `intensity`** 下结果
   `max|Δpops| = 0.0`（逐位相同）。所以 **文件只表达形状，整体标度只能靠
   `intensity` 字段**，且峰值永远被钉死在 `intensity` 上。
2. **它是振幅因子，不是光强因子** —— 代码链为 `field_e0_sq` 线性（E0²∝I）、
   `rabi_mhz` 取 √，故耦合 ∝ √I·env，即 **光强(t) = intensity × env(t)²**。

**→ 想要 I(t) = I₀·(1+δ(t))（δ ∈ [−a, +b]）的正确配方**：

```
intensity = I₀ · (1 + b)                 # 峰值
env 采样   = √[ (1 + δ(t)) / (1 + b) ]    # 峰值恰为 1
```

若要同时把**平均**光强锁在 I₀（隔离"均值漂移"这个混淆因素）：
先做峰值归一化，再令 `intensity = I₀ / mean(env²)`。

**反面教材**：把绝对光强曲线直接塞进文件 → 被静默归一化 → 整体光位移悄悄
错掉。而按物理文档，光位移差几 MHz 就等于交换从共振变解谐。

## 坑（已知）

1. **mixed 纯微波亦可**（t77 起）：没有光场时光子维退化 P=1，等价于微波/V_dd
   当场精确积分（与 hard 同物理）。会强制改道的是**任意波形（完整驱动）**——
   sambe / mixed 遇它自动改用 hard（无载频，光子块引擎不支持），GUI 日志有提示；
   **光强包络（`FileEnv`）是普通 Wave，不受影响**。
2. **run_compute 阻塞预处理**：两体 V_dd 预计算可达数十秒，工具返回前
   主线程在泵进度——超时设长一点（默认 300s）。
3. **undo 是整快照回滚**：撤销 set_param 会连带恢复该步之后的所有
   同 op 影响；逐步操作逐步撤，别攒。
4. **双原子模式**：`dynamic.pair.*` 仅双原子模式生效；切模式用
   `set_param('dynamic.mode', 0|1)`（0 单原子 / 1 双原子，= 人点单选钮同路）。
5. **上帝模式怎么用（t83 更新，旧说明"不开放给 AI"已作废）**：
   - **能用的**：引擎/求解器控制参数（`engine.*` / `solve.*`，见上）与
     **运行诊断**（`get_diag`）。诊断是回答"为什么"的主通道；
   - **不碰的**：检视器窗口显隐、极简/上帝模式开关（视图状态，P6）。
6. **结果反直觉时的诊断套路**（比猜参数快得多）：
   ① `get_diag('')` 看摘要（AC 化剔了几对、M=几、P/dim、步长分位数、归一偏差）；
   ② 布居**完全不动** → `get_diag('pairs')` 看目标跃迁在不在保留集里
   （不在 → 多为 `engine.PHASE_BUDGET` 砍了远失谐梯级，或基组缺中间态）；
   ③ 跑得**异常慢** → 看 `solver.dt_ns` 与 `nfev`（光强过大 → Ω≫失谐 → 步长被钉死；
   先降 intensity，别硬等）。
7. 计算失败时 `job_result` 的 msg 含原因；常见=光强过大不收敛，
   降 intensity 或换 hard。

8. **多光子通道要求中间态在基组里**（2026-09-12 实测）：引擎的多光子耦合是由基组内**真实的单光子
   矩阵元**串起来的。做 60P→58G 的三光子踢出时，基组里必须有 59D（≈10.2 GHz）与 58F（≈11.2 GHz）
   作中间态；只放 60P/58G 会出现"频率对、强度对，却一点都踢不动"（实测 58G 布居为 0）。
   裁基组前先确认目标跃迁的中间态在不在。
9. **AI 发起的重活没有弹窗、用户也取消不了**（2026-09-12 实测）：AI 发起的 `run_compute` 不弹成本框；
   GUI 的"取消"不覆盖 AI 作业，AI 侧也拿不到 job_id → 一旦提交重活双方都被卡住（后续
   `run_compute_and_wait` 全被拒："已有演化在运行（人或 AI 发起）"）。**重活一律用 `run_python_async`**，
   两体+光学场的窗口开短（t_max ≤ 200 ns、npts ≈ 101），提交前先估量级。
10. **两体模式 `save_npz` 不回传 npz 路径**（2026-09-12）：单体正常、两体拿不到 →
    取不到逐时刻曲线，改用 `job_result` 或 `get_diag(scope='traj')`。
11. **基组编辑会静默重置初态**（2026-09-12）：改完组后 `psi_A` 曾从 60P_3/2 变回 60S（无提示）；
    改完组务必用 `list_params('dynamic.psi_A')` 复核 `value` 与 `last_committed` 是否一致。
12. **初态键的能量若猜错会静默选到别的能级**：`psi` 的 `options` 目前是 null，不要凭能量猜；
    实测猜错的键导致初态落到 `59D_{3/2} m=+1.5` 而运行仍报 ok。
## 人机协同约定

- 你的每次写操作 GUI 实时反映（控件闪紫框 + 状态栏流水），人可随时
  点"撤销AI"回滚你的最近一步——**操作带 reason 是好习惯**。
- 人可能在你两次调用之间改了参数：动手前 `get_activity(since_seq)`
  或 `summary` 复核现状，别基于过期认知写入。
- 长计算期间人可能取消（GUI 取消按钮）：`job_status.error='cancelled'`
  即此情形，不要重试同一配置。
