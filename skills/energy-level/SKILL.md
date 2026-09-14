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

### ⚠ 先看 `summary()['code']`（T3/A12，2026-09-12 加）
引擎/服务端是**进程内**代码：改了 `src/` 不重启，你看到的仍是旧行为（本会话曾因此白烧整轮）。
```python
summary()['code']
# {'stale': True, 'n_stale': 7, 'n_files': 118, 'age_s': 812.0,
#  'loaded_at': …, 'src_core_mtime': …, 'src_ai_mtime': …,
#  'newest': [{'path': 'src/ai/sim_state.py', 'mtime': …, 'stale': True}, …],
#  'hint': '⚠ 代码已陈旧：… **需要重启软件** 才生效；当前结果仍来自旧代码。'}
```
- `stale=True` → **别下结论**：先让人重启再重跑；
- `run_python` 的返回里也带 `stale_code_warning`（陈旧时一句话，新鲜时 `None`）；
- 基准 = "本进程加载代码的时刻"（`server.py` 导入期盖戳，first-wins）→ 只要有人改过 `src/` 就能立刻看出。

## 工具清单

| 工具 | 作用 | 备注 |
|---|---|---|
| `summary` | 轻量摘要（组数/场/方法/波形数）+ **`code` 代码新鲜度** | **每次会话先调这个建立认知**；`code.stale=True` ⇒ 进程跑的是**旧代码**，让人重启 |
| `get_state(scope)` | 全量/分模块状态导出 | scope ∈ 空/groups/field/dynamic |
| `list_params(prefix)` | 参数 schema：type/unit/desc/range/options/value | set_param 的字典；**波形包络只给摘要** |
| `get_envelope(path, index)` | 取某条波形的**原始包络采样数组** | 大数组专用通道（list_params 只给摘要，防响应膨胀）|
| `set_param(path, value, reason)` | 改一个在册参数（登记面=可改面） | 过控件校验；reason 写动机 |
| `add_level_group(n,l,j,reason)` | 加能级组 | 自动校验+配色 |
| `remove_level_group(index,reason)` | 删能级组 | 索引见 get_state |
| `set_waves(waves, atom, reason)` | 整表替换波形 | atom=single/A/B（三套独立），格式见下 |
| `set_superposition(atom, components)` | 设**逐原子叠加态**初态 | 分量须在基组内，自动归一；存为**持久真值**（下拉显示"叠加态（自定义 n 项）"）。见坑 14 |
| `set_pair_superposition(components, reason)` | 设**两体纠缠初态**（交换后的结果） | 分量须在当前两体基内；存 `dynamic.psi_pair`，**取代** psi_A/psi_B；传 `[]` 清除。见坑 13 |
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
- `dynamic.psi_pair`（2026-09-12 加）：**两体纠缠初态 spec**，形如
  `['pair_sup', [[nA,lA,jA,mA, nB,lB,jB,mB, re, im], …]]`，默认 `None`。
  用 `set_pair_superposition` 写（别手拼）；两体模式下**取代** psi_A/psi_B。
  新登记项 ⇒ **只在重启后的进程里有**（旧进程会报"没登记"）；
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
  | `solve.MAX_STEP` | DOP853 **初始**步长上限（ns）——⚠ 只管起步 | 空 = 自适应 |
  | `solve.MAX_STEP_HARD` | DOP853 步长**硬上限**（ns，每步复核） | **空/0 = 不限制**（默认；置值会显著增加步数与耗时） |
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
| `set_groups([(n,l,j),…])` | **声明式**让基组恰好包含这些组；原子、幂等，回 `{added,removed,kept,retrimmed,resolved,psi_warnings,field_warnings}` | 按索引增删 + 按标签字符串判重会**删错组/漏加组**（实测两次：`'60P'` 先匹配 60P_3/2、`'58G_{'` 漏掉 58G_9/2）。**逐子能级裁剪**见坑 24 |
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

**两体纠缠初态（"交换后的结果"当 freeze 的初态）**：
```python
set_groups([(60,0,0.5),(60,1,1.5)])            # 两体交换集（按需加中间态）
r = set_pair_superposition([                    # (|P_A,S_B⟩ + i|S_A,P_B⟩)/√2
    {'pair': [[60,1,1.5,-0.5],[60,0,0.5,+0.5]], 'amp': [1,0]},
    {'pair': [[60,0,0.5,+0.5],[60,1,1.5,-0.5]], 'amp': [0,1]}])
set_param('dynamic.mode', 1)                    # 双原子（否则只有 mode_warning）
# r['info']['populations'] 复核占据谱；缺项会在这里报错
submit_compute()                                # 重活走异步，别同步等
```
注意：对称组合（两个 amp 都是 1）是**暗态**，布居不摆（坑 13）。

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
- **椭圆偏（2026-09-12 补文档）**：`pol='椭圆偏'` 时同一元素再加一个键
  `'elliptical': {'alpha_deg': …, 'theta_deg': …, 'eps_deg': …}`
  （α=θ=ε=0 ⇒ σ± 等权 = ⊥B 线偏）。实测可用（波形读回与结果都对）；
  这是"复现 σ± 等权 / 椭圆偏"的唯一通道。
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
10. **两体作业的 npz 现在**自动**落盘（2026-09-12 起，原坑已修）**：AI 作业一完成，
    `_jobs` 里就存了 `ai_results/job_<jid>.json`（摘要）**与全量 npz**
    （`tempsimu/ai_job<jid>_*.npz`）；`job_result(jid)` **即使不给 `save_npz`**
    也会回传该 npz 路径（`summary.npz` / 顶层 `npz`，带 `npz_auto=True`）。
    仍建议：**跑完立刻** `job_result(jid, save_npz=True)` 再存一份并记下路径
    （npz 只保留最近 100 份）。
11. **基组编辑会静默重置初态**（2026-09-12）：改完组后 `psi_A` 曾从 60P_3/2 变回 60S（无提示）；
    改完组务必用 `list_params('dynamic.psi_A')` 复核 `value` 与 `last_committed` 是否一致。
12. **初态键的能量若猜错会静默选到别的能级**：`psi` 的 `options` 目前是 null，不要凭能量猜；
    实测猜错的键导致初态落到 `59D_{3/2} m=+1.5` 而运行仍报 ok。
13. **两体纠缠初态怎么写**（2026-09-12 加，交换类实验的前置）：
    `set_pair_superposition(components)`，分量 =
    `{'pair': [[nA,lA,jA,mA], [nB,lB,jB,mB]], 'amp': [re,im]}`
    （或 `{'mag':…, 'phase_deg':…}`；省略 amp 默认 1）。
    - 分量按 **(n,l,j,m) 解析**（不是索引）→ 跨 B/基组重排稳定；缺项**报错并列出**
      缺失项，绝不静默丢；
    - 自动归一，**相对相位保留**（`[1,0]` 与 `[0,1]` = 相位差 i）；
    - 存进 `dynamic.psi_pair`，两体模式下**取代** psi_A/psi_B；反过来设
      `set_initial_state`/`set_superposition` 会清掉它（返回 `pair_spec_cleared: True`）；
    - `dynamic.mode=0` 时返回带 `mode_warning`（**不**自动替你切模式）；
    - ⚠ **对称组合 `(|ab⟩+|ba⟩)/√2` 是交换哈密顿量的暗态**：只吃全局相位、布居不动。
      想看交换摆动，要么给相对相位（如 `[0,1]`=i），要么用逐原子直积初态。
14. **逐原子叠加态初态现在真的可用了**（2026-09-12 修，原为两处叠在一起的缺陷）：
    ① **引擎侧格式错配**：`build_sup_spec` 存 `['sup', [[n,l,j,m,re,im],…]]`，而
    `_as_vector` 的 sup 分支只认旧形态 `[[spec, amp],…]` → 规范形态的 `item[0]`（= n，如 60）
    被当成**索引** → `IndexError`（即"叠加态初态"从未在引擎里跑通）；
    ② **GUI 侧通道丢失**：spec 只进 `_pending_psi_keys`，`find_state_index` 匹配不到 →
    运行期读 `psi_combo.currentIndex()` → **静默回落 0 号态**。
    现在：spec 是**持久真值**（`dynamic.psi_*` 读回就是 spec），下拉显示
    "叠加态（自定义 n 项）"，`_psi0_for` **spec 优先且失败即报错**（不再静默回落）。
    三种写法都吃：`[n,l,j,m,re,im]`（规范）/ `[spec, amp]` / `[n,l,j,m]`（幅度默认 1）。
15. **`solve.MAX_STEP` 只是"初始步长上限"**（2026-09-12 实测）：`dop853.py` 只在**起步**取
    `min(…, max_step)`，之后步长由误差控制自由增长——实测请求 0.05 ns 的 10 ns 演化里
    出现 **2.2 ns** 的接受步（7 步走完，nfev=98）。要**真**压步长用新键
    **`solve.MAX_STEP_HARD`**（默认关；置值后每步复核，代价是步数/耗时上升；
    只对 mixed/hard 的 DOP853 生效——sambe 走自带传播子）。
16. **软件实例的生命周期归人管；跑 GUI 自检前先问**（2026-09-12 事故后立规）：
    GUI 自检会构 `MainWindow` + 真实 ARC 基组，与**正在跑实验的实例**争 ARC SQLite；
    且它的 `win.close()` 会触发 `stop_server()`（曾把活实例的 `.mcp-port` 删掉，
    已修：分工文档 §六 #6）。规则：① 不自己 `python main.py`、不杀 python 进程；
    ② GUI 自检用 `QT_QPA_PLATFORM=offscreen`，人在跑实验时**先问**；
    ③ 需要新代码生效就报"需要重启"，由人重启。
17. **⚠ 基组裁剪必须给"观测量收敛证据"，否则结论不可引用**（2026-09-12 使用者实测，
    **灾难级**）：同一个 freeze 脉冲（7380 MHz / I=5 / σ=16.99 / 中心 50.96 / 椭圆偏 α=θ=ε=0），
    只看 `C = P_end(60S|初态60S) − P_end(60P|初态60P)`：
    **7 组 → C=0.006**（60S 保留 0.7144，28.6% 落到 **59D_5/2 m=+2.5**：缺 59D_3/2/60D_5/2
    时那条 Δm=+2 双光子通道不被抵消）；**13 组 → C=0.9982**（60S 保留 0.9994）。
    `field.n_photon` 2 vs 4 **逐位相同** → 这不是多光子窗口问题，是**基组本身**。
    → 做法：≥3 个基组（如 7/10/13 组）跑同一脉冲，`C` 与目标态保留的**变化 ≤1%** 才敢下结论；
    → ⭐ **主判据永远是观测量；配对数只是诊断线索**（使用者 2026-09-12 写死）：
      E1 那套"配对数 150→1206"在**完备基**下**不复现**（实测：13 组 178 对 → 加**极远**
      70S_1/2 仍 178 对且 `C_dual` 逐位不变；加近旁无关能级只 +33%/态、观测量动 0.9%）
      ⇒ "配对数爆炸"不具物理意义，**裁剪基组真正要担心的是观测量未收敛**；
    → 未知跃迁先问"中间态/抵消通道在不在基组里"（坑 8 同源）；
    → 两个观测量可能**在不同基组大小上收敛**（本轮：60S 保留 10 组就收敛、60P 踢出要到
      13 组）——别用一个基组的大小代表"全都收敛了"；
    → ⚠ **做基组判据要挑"不敏感的工作点"**：I=5/7380 MHz 上 kick 那一路是**窄三光子
      共振**（相邻 5 MHz 摆动 ±0.15；**已定性 = 物理**，见坑 20），而 60S 保留单调平滑。
      **别把窄共振当"配对选择 bug"**；基组测试放到不敏感处（如 I=4，那里 60S 保留 ≡ 1.0000）做。
18. **离线批量通道的基组与登记册不同源**（2026-09-12）：手工脚本用
    `make_levels(..., kind='mJ')` 把 6P_3/2 展开成 4 个 mJ 态，而软件登记册展开成
    **16 个 (F,m_F)** 态——两者基组定义不同。本次两条通道 4 位小数吻合（碰巧：6P 在
    296 THz 外）。**交叉验证前先确认基组同源**，否则"对得上"是运气、"对不上"是白查。
19. **⚠ 教训：写入侧通了 ≠ 引擎侧通了——验收必须打到引擎**（2026-09-12）：
    "叠加态初态"曾按"写入侧已实现"记进文档（spec 能写进 param、能读回），
    但**引擎侧 `_as_vector` 只认另一种形态** → 一跑就 `IndexError`，即**从未跑通过**。
    同一轮的 `psi_pair` 也踩过"GUI 通道丢失、运行期静默回落 0 号态"。
    → 任何"接口已实现"的结论，必须用**真引擎跑一次**（哪怕合成小基组）才算数；
      自检要打在**引擎入口**，不是打在存储/序列化层。
20. **freeze kick 在 7380 MHz/I=5 是"窄三光子共振"——这是物理，不是抖动 bug**
    （2026-09-12 使用者 `tol` 复核定性；详见待办 §10）：
    - 同一 ω 下变 `tol`(1e-7/1e-9) 与 `npts`(1501/3001)，`C` 极差仅 **3×10⁻⁷~6×10⁻⁷**，
      比 5 MHz 步长引起的 ~0.15 **小六个数量级** ⇒ 结构物理；±1% 平坦区只 **1/45** 格点；
    - 机制：60P→58G 的**裸共振在 7219–7347 MHz**，同一 J 内相邻子能级间隔/3 ≈ **9–11 MHz**
      （对上实测峰间隔 10 MHz）；实测峰比裸共振**高 ~+40 MHz**（I=5 的 AC Stark）
      ⇒ **不能按裸共振布点，必须扫**；共振宽 ≈ 1/(3τ) ≈ **4 MHz**（τ=FWHM 40 ns）；
    - **交付条件**：必须给**频率精度要求（~1–2 MHz 级）**；"参数抖动平坦"在当前方案不成立；
    - **平坦化路线**：❌ **纯圆偏振不行**（使用者全窗口扫描：**σ+ ≤ 0.42、σ− ≤ 0.37**，
      多数 0.0–0.3 甚至为负：只剩单条 Δm=±3 路径，每圆分量场强 1/√2 ⇒ 三光子振幅
      ∝(1/√2)³，同强度补不回来；提 I 则 60S 损伤）
      ｜ ⚠ 更短脉冲（要更高 I，有损伤风险）｜ ⭐ **啁啾/绝热脉冲**（扫频 ⇒ 天然宽带；
      通道：微波**任意波形（完整时域驱动）** 由 `hard` 支持，`sambe/mixed` 自动改道
      `hard`；⚠ 任意波形不经 `set_waves`，需走存档）。
    - **平坦性的正确表述（使用者 2026-09-12 实测）**：椭圆偏在 **7355–7400 MHz 有宽的高
      平台**（`C_ent` 0.74–0.99，叠 ~5 MHz 精细结构）→ 交付口径是
      **"保证 0.79、标称 0.99"**（±10 MHz 窗口的保证值 = **0.786 @ 中心 7375**），
      而**不是**"完全不平坦"。路线选择等用户裁决。
21. **⚠ 长两体作业曾"跑完时整进程消失、成果全丢"**（2026-09-12 事故；销毁者仍未定性）：
    - 现象：13 组 / dim=7056 的 AI 两体作业在尾段进程**静默消失**（Windows：
      faulting module **Qt5Core.dll**、exception **0xc0000409**、WER BEX64）；
      stderr 最后一句 `QThread: Destroyed while thread is still running`；
      `.mcp-port` 未变 ⇒ 不是正常关窗；结果随进程丢失。
    - **机制已证实**：那句话 = `qFatal` in `QThread::~QThread`（`running && !finished`）——
      隔离复现（销毁运行中的 QThread）给出**同一句消息 + 退出码 0xC0000409**。
    - **触发者未定位；假说 B（内存）已被实测排除**：同类型两体作业（dim=4624）全程
      **私有内存逐位不动 1358 MB**、系统可用 ≥5.2 GB ⇒ 不是"dim 吃爆内存"。
      已排除的还有：AI 没持强引用（AI 与 GUI 走**同一条** `dyn._run()`，worker 都存在
      面板 `self._worker`）、无 `sip.delete`/`deleteLater`、无 Qt 父对象、
      `src/` 内无 `quit()/sys.exit` 可达路径。**崩溃时确无 Python traceback**
      （使用者抓到原 stderr：8 条 `Terminating session` + 唯一一条 Qt 原生消息）⇒
      **没有 Python 层触发者**，销毁在 C++ 侧。
      **WER 全 dump 已配好**（`tempsimu/crashdumps`、DumpType=2、DumpCount=3）→
      下次崩溃自动落 dump，用 cdb 看 `~QThread` 调用栈即可钉死。
      ✅ **已修并验**：`_finish_run_ui` 不再丢"仍在跑的 worker"的最后一根引用
      （`_retired_workers` + `finished` 释放，同时覆盖取消路径）；自检
      `checks/gui_worker_lifetime_test.py` **10/10 通过**（含"收尾时线程仍在跑"的时序）。
      ⚠ **现场保全**：每个实例用**带 pid 的日志名**，崩溃后先复制现场再重启
      （实测事故：实例 stderr 被下一次启动的同路径日志**截断覆盖**）。
22. **⚠ 零波形 = 跑不了**（2026-09-12 使用者实测）：没有波形、也没有实际耗散/自动寿命通道时，
    `run_compute` 会**拒绝启动**（`ok=False`、`msg="未启动（校验未过或无波形，见 GUI 日志）"`、
    作业不登记）。**这是既有设计（空演化没意义），不是缺陷**。
    → **静止类验收（只验初态/基底是否正确）必须加一个"不落在时间窗内"的脉冲**，
      例如中心远在 `t_max` 之外的窄高斯：这样演化等于静止，但校验能过。
      **引擎级验收范例（#3/#5 叠加态，2026-09-12 使用者现场跑）**：
      `set_superposition('single', [{'state':[60,0,0.5,0.5],'amp':[1,0]},
      {'state':[60,1,1.5,-0.5],'amp':[0,1]}])` + 一个窗外的微波脉冲 →
      **终点布居 `60S(+0.5)=0.50178`、`60P(−0.5)=0.49822`、其余全 0**
      （旧 bug 会给 `60S(−0.5)=1.0000`、其余 0）⇒ 叠加态**确实进了引擎**，
      读回 `get_config()['config']['psi']['single']` 也是规范 spec。
      —— 这就是坑 19"验收必须打到引擎"的正面样板。
23. **🚨 性能阈值曾静默删掉整批光场耦合（E1 真凶，2026-09-12 定位并修复）**：
    `SambeSetup` 在 `dim > engine._DENSE_MAX`(=2200) 时把 `Kc_dense` 置 None（只是
    "稠密/稀疏"的**性能**选择），而 `evolve_mixed._build_pair_geometry` 原来只在稠密分支里
    收光场近对 ⇒ **超阈值时光场耦合整批消失、光场变成没接线**。实测（频移光只在 A 上）：
    - 3 组 + 光（dim=1452）→ `n_pairs=1206`、`B:60P=0.8956`（光**有效**）；
    - 3 组 + 58G_7/2 + 光（dim=2700）→ 修前 `n_pairs=150`、`B:60P=0.0060`
      （**与不加光逐位相同 = 光完全失效**）；修后 `n_pairs=1590`、`B:60P=0.8956`，
      且"只加一个无关组"的观测量变化 = **0.0000**（物理预期）。
    → **怎么第一时间发现**：跑完看 `get_diag(scope='pairs')`——**有光场波形却看不到光场对**
      就是它；另一个指纹是"观测量与不加光**逐位相同**"。
    → **记住这条纪律**：**性能阈值（稠密/稀疏、成本窗、M）绝不能改模型**。
    → 受影响面：**两体 + 光学场且 dim > 2200** 的运行（微波/纯 V_dd 不受影响）。
      修复见 `evolve_mixed._build_pair_geometry`（稠密缺失时回落到稀疏 `Kc_ext`，
      稠密区逐位不变）；回归锚 `checks/e1_pair_forensics.py`（7 项）。
24. **逐子能级裁剪（2026-09-12 加）：只留组内特定子能级**——`dim = N²·P` 直接受益
    （含光学场 P=3）。三种写法等价（`set_groups` 都吃）：
    ```python
    set_groups([(60,0,0.5), (60,1,1.5, [-0.5, 1.5]), (6,1,1.5)])          # (n,l,j,ms)
    set_groups([{'n':60,'l':1,'j':1.5,'ms':[-0.5,1.5]}, {'n':60,'l':0,'j':0.5}])
    set_groups([{'n':6,'l':1,'j':1.5,'Fs':[[3.0,0.0],[3.0,1.0]]}])        # 超精细 (F,m_F)
    ```
    · **组的身份仍是 (n,l,j)**（配色/图例/预设/身份键/两体配对几何都不变）；
      `ms` 与 `Fs` **互斥**，`ms=[mJ,…]`、`Fs=[[F,m_F],…]`；
    · **不写 = 全展开**（默认路径与改前**逐位一致**——这是硬要求）；
    · 返回里 **`resolved`** 给每组**实际**子能级数与身份键 ⇒ **裁剪是否真生效可自检**
      （落实坑 19）；`retrimmed` 列出被改裁剪的组；
    · GUI：入口 = **能级组列表（左栏彩色条目那张）右键菜单的「子能级…」**
      —— 2026-09-14 用户裁决**删掉了**工具条上的同名按钮（重复 + 那一行太挤）；
      菜单作用于**右键命中的那一行**（每行都够得着），勾选框**默认全选**，
      行内显示 `60P₃/₂ (2/4)`，dim 提示实时体现收益。
      ⚠ "选中"指**左栏能级组列表**，**不是**右栏 `level_table`（逐子能级列 mJ/E 那张）；
    · ⚠ **口径（用户原话）**：裁剪是为**省算力/做收敛检验**，**不是**把观测量调好看——
      **判据是"裁掉之后观测量不变"**；变了说明那个子能级承载物理（或基组没收敛），
      **要老实报告**，不能拿"裁剪后更漂亮"当成果。
    · ⚠ **不能按"6P 只留两三个"来裁**（反证在案）：mixed 引擎的光移**只**来自
      "基组内态对的 ARC 矩阵元 + 精确传播"，**没有任何 ARC 伙伴态注入**
      （`field_build.build_fields_exact` docstring：全部偶极允许对都进耦合矩阵、
      无近/远分流、不注入静态能移；`evolve_mixed` 不调 `real_field`）
      ⇒ **基组里没有 6P，6P 对频移光的贡献就是缺失的（漏算），ARC 不会补上**。
      归档里 `docs/_archive/论文级方案调研-交换全过程场景.md:279` 的实测是
      **F=1+F=2 合计占光移 60.7%**（与 F=3 同量级，"没有天然弱耦合子集"）
      ⇒ `docs/_archive/Julia引擎交接文档.md:705` 那条"6P 只留两三个"的旧建议
      **已被推翻，勿照做**。
    · 护栏：被裁掉的子能级若正被 `dynamic.psi_*` 引用（按**变更前**取值判！）
      → `psi_warnings` 点名；若它是已配置场需要的耦合伙伴
      （|Δl|=1、|Δm|≤1、跃迁落在波形 `freq ± engine.BW_OPT` 内）→ `field_warnings` 点名。
    · 自检 `checks/sublevel_trim_selfcheck.py`（**25 项**：向后兼容/过滤/存档往返/
      三写法幂等+resolved/护栏/合法裁剪⇒观测量不变）。
    - **你现在就该做的**：① 大两体作业先读 `estimate_cost()` 的
      `est_peak_gb / avail_ram_gb / mem_advice`（锚点 dim=4624→1.36 GB，dim² 外推，
      **偏保守上界**；软建议峰值 ≤ 可用内存 40%）；
      ② 作业一完成**摘要 + 全量 npz 已自动落盘**（`ai_results/job_<jid>.json`、
      `tempsimu/ai_job<jid>_*.npz`），进程随后死掉也拿得到；
      ③ 仍建议跑完立刻 `job_result(jid, save_npz=True)` 再存一份并记路径。

## 人机协同约定

- 你的每次写操作 GUI 实时反映（控件闪紫框 + 状态栏流水），人可随时
  点"撤销AI"回滚你的最近一步——**操作带 reason 是好习惯**。
- 人可能在你两次调用之间改了参数：动手前 `get_activity(since_seq)`
  或 `summary` 复核现状，别基于过期认知写入。
- 长计算期间人可能取消（GUI 取消按钮）：`job_status.error='cancelled'`
  即此情形，不要重试同一配置。
