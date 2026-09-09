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
- 可用名字：`summary() get_state() set_param(path,v,reason='')
  add_level_group(n,l,j) remove_level_group(i) set_waves(list,atom)
  run_compute_and_wait(timeout=1800,poll=1.0,save_npz=False)
  activity(since=0) np json time ROOT`
- **把结论赋给 `result` 变量**（自动回传，JSON 化）；过程信息用 `print`。
- 数组不要整体回传（>50 元素会被截成形状摘要）——print 统计量或存 npz。
- undo 语义：整段脚本算**一步**，撤销回到脚本执行前。
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
| `list_params` | 可改参数 path 与当前值 | set_param 的字典 |
| `set_param(path, value, reason)` | 改白名单参数 | 过控件校验；reason 写动机 |
| `add_level_group(n,l,j,reason)` | 加能级组 | 自动校验+配色 |
| `remove_level_group(index,reason)` | 删能级组 | 索引见 get_state |
| `set_waves(waves, atom, reason)` | 整表替换波形 | 格式见下 |
| `run_compute()` | 启动演化计算 | 立即返回 job_id，GUI 同步显示 |
| `job_status(job_id)` | 查进度 | 轮询到 done=true |
| `job_result(job_id, save_npz)` | 取结果摘要 | 大数组落 npz 给路径 |
| `cancel_job(job_id)` | 取消计算 | |
| `get_activity(since_seq)` | 操作流水（人+AI） | 了解"现在什么状况" |
| `undo(reason)` | 撤销最近一个 AI 操作 | 先斩后奏的兜底 |
| `save_archive(path)` / `load_archive(path)` | JSON 存档读写 | 与菜单同源，可 undo |
| `screenshot()` | 主窗口截图 → PNG 路径 | 了解人眼所见 |

## 单位约定（务必遵守）

- 磁场 B：**Gauss**；静电场 E：**V/cm**
- 光频率：**GHz**（存档字段 frequency_ghz）；光强：**mW/cm²**
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

**改参数扫描**：`set_param` 逐个改 → 每次 `run_compute`；结果文件在
`tempsimu/ai_job*.npz`，用本地脚本读 npz 做精细分析，别往 context 灌数组。

## 波形格式（set_waves 的 waves 列表元素）

```json
{"name":"波形1","freq_mhz":6800.0,"intensity":0.05,"pol":"π",
 "offset_ns":0.0,"env":["gauss",200.0,40.0,1.0]}
```
- env 高斯：`["gauss", 中心ns, σ_ns, 峰值]`
- env 方波：`["square", 起点ns, 沿ns, 高度]`（可选第 5 元下降起点）
- pol：微波用 π/σ+/σ-；光场同
- 任意波形（arb）暂不通过 set_waves 提供（需采样数组，走存档）

## 坑（已知）

1. **mixed 引擎需光场**：纯微波波形会被自动降级 hard（GUI 日志有提示，
   `get_activity`/`screenshot` 可见）。
2. **run_compute 阻塞预处理**：两体 V_dd 预计算可达数十秒，工具返回前
   主线程在泵进度——超时设长一点（默认 300s）。
3. **undo 是整快照回滚**：撤销 set_param 会连带恢复该步之后的所有
   同 op 影响；逐步操作逐步撤，别攒。
4. **双原子模式**：`dynamic.pair.*` 仅双原子模式生效；模式切换目前
   不开放给 AI（视图/工作流耦合深，需人在 GUI 上切）。
5. **上帝模式（diag 旋钮）**不开放给 AI：检视器属人侧调试通道。
6. 计算失败时 `job_result` 的 msg 含原因；常见=光强过大不收敛，
   降 intensity 或换 hard。

## 人机协同约定

- 你的每次写操作 GUI 实时反映（控件闪紫框 + 状态栏流水），人可随时
  点"撤销AI"回滚你的最近一步——**操作带 reason 是好习惯**。
- 人可能在你两次调用之间改了参数：动手前 `get_activity(since_seq)`
  或 `summary` 复核现状，别基于过期认知写入。
- 长计算期间人可能取消（GUI 取消按钮）：`job_status.error='cancelled'`
  即此情形，不要重试同一配置。
