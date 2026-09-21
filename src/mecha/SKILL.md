---
name: re0-mecha
description: 通过 MCP 语义层驱动 Re0 里德堡原子物理引擎时使用——发现参数/能级、配置实验、建驱动场、预检、跑演化（长任务自动异步）、监控、撤销。调用任何 re0-mecha 工具前先读本文件，照流程走即可，无需试错。
---

# Re0 机甲框架 · AI 调用指南

你通过 **MCP 写通道**操作一个语义内核（registry + 会话 choke point + 事件总线），它替你调
物理引擎（`runspec` 唯一分派）。铁律：**每次写都经 choke point、记一条事件、可 undo**；
所有键/值过契约校验；**错误消息会教你怎么改**（按 `suggest` 改，别盲试）。

> **权威源**：本文件 = `src/mecha/SKILL.md`（随代码走，git 里这份为真）。若 harness 读的是
> 工作区副本，两者应一致；发现漂移以本文件为准（副本需从项目同步）。

## 连接
人先起服务（你不 spawn）：`python -m src.adapter.serve` → 写 `.mcp-port`。
你 attach 到 `http://127.0.0.1:<port>/mcp`（streamable-HTTP，仅 localhost）。
❗ 启动开关是 `RE0_MECHA_NO_SERVER`（**不是**旧项目的 `EL_NO_AI_SERVER`——后者现在只管判据跳过）。

## ⚠ 波形纪律（最危险的坑：静默给错答案）
`set_drive`/`add_wave` **按 name 追加或替换**：同一 name → 替换；不同 name → 追加。
扫描时若逐点用不同 name，波形会 **累积**、多个驱动时间重叠 ⇒ 你测的是“多驱叠加”而非单脉冲，
**结果像物理、却不报错**（真实踩过：扫 Ω 时 23 条波叠加，数据周期性翻转差点被当成发现）。
- **单驱动扫描**：复用同一 `name`（或不传 name，默认名=跃迁）→ 自动替换，永远 1 条。
- **每轮扫描前**：用 `mecha_script` 跑 `fields.clear_waves(atom)` 清干净。
- **`get_waves` 与 `preview_behavior` 都报 `n_waves`（=`n_drives`）+ `overlap_warning`**（set_drive/add_wave 回执也带）——**跑前先看一眼**。
- 合法多波不误报：Ramsey（同跃迁但不重叠）/ STIRAP（重叠但不同跃迁）→ `overlap_warning=None`。

## 标准流程（一次实验，照做即可）
1. `set_param` 配好骨架：`groups.list` → `field.B_gauss` → `dynamic.{mode,method,t_max,npts,decay_index}`
2. `list_levels` → 拿到基组每个态的 **label + 身份键 key**
3. `set_param("dynamic.psi_single", <key>)` 设初态 —— **必须用 list_levels 给的 key，不能用 label**
4. `set_drive(reason, src, tgt, rabi_mhz=…, pulse="pi")` 建驱动场（src/tgt 用 label 或 key 都行）；`get_waves` 核对
5. `preview_behavior` 预检：这条驱动了哪些跃迁、**基组够不够**（不跑演化，秒回）
6. `estimate_cost` 看代价 → `run_compute(reason=…)` 跑（两体/长任务**自动转 async** 回 `job_id`）
7. `get_activity(since_seq)` tail 事件；出错或想回退用 `undo`

## 工具（16 个）
| 工具 | 用途 | 关键参数 |
|---|---|---|
| `list_params` | 可调参数表（默认藏引擎内部参数） | `prefix, family, include_internal, page` |
| `list_levels` | 基组能级态：**label + 身份键 + 能量** | — |
| `get_state` | 当前配置快照 + 定位包 | — |
| `get_activity` | diff-since-seq 事件（tail 总线，只回增量） | `since_seq` |
| `set_param` | 写一个配置键（回执带 `resolved`+`readback`） | `key, value, reason` |
| `undo` | 撤销最近一次写入 | `reason` |
| `set_drive` | **在跃迁上建驱动**：给 Rabi/强度 + 脉冲类型 → 自动算 freq/intensity/时长 | `reason, src, tgt, rabi_mhz｜intensity_mwcm2, pulse("pi"/"pi2")｜dur_ns, pol, detuning_mhz, atom, edge_ns, name` |
| `get_waves` | 波形摘要（不内联大数组） | `atom` |
| `estimate_cost` | dry-run 成本（不跑演化） | — |
| `preview_behavior` | 行为预览：驱动哪些跃迁/各多强/**基组够不够** | — |
| `run_compute` | 跑演化；`mode=auto` 超阈值(30s)自动转 async | `reason, mode("auto"/"sync"/"async"), diag` |
| `submit_job`/`job_status`/`cancel_job`/`list_jobs` | 长任务异步面（**断线可按 job_id 找回**） | `reason` / `job_id` |
| `codeact` → **`mecha_script`** | **多步编排**：一段脚本（循环/批量/条件），命名空间只有语义层门面 + `math` | `script`（把产物赋给 `result`） |

## 配置键（`set_param` 的 key）
- `groups.list`（必填）：`[{"n":60,"l":0,"j":0.5}, …]` 能级组
- `field.B_gauss`(Gauss) · `field.E_vpcm`(V/cm)
- `dynamic.mode`(0单体/1两体) · `dynamic.method`(`mixed`｜`sambe`｜`hard`) ·
  `dynamic.t_max`(ns,>0) · `dynamic.npts`(≥2) · `dynamic.decay_index`(0自动/1禁/2自定义)
- `dynamic.psi_single`：**身份键**（list_levels 的 `key`）；两体用 `psi_A`/`psi_B`/`psi_pair`
- `dynamic.waves_single`：波形——**用 `set_drive`/`get_waves` 建读，别手搓**；两体 `waves_A`/`waves_B`
- `solve.MAX_STEP`/`MAX_STEP_HARD`/`PER_FREQ_M`/`FROZEN_H_NS`/`COST_BUDGET`：运行级旋钮

## 契约（照做就不试错）
- **单位**：B=Gauss，E=V/cm，t=ns，微波 freq=**MHz**，光频=**Hz**，Γ=MHz。
- **每次写要 `reason`**（归因）；回执 `readback` 是引擎实测读回，与你声称的 `resolved` 并排核对。
- **错误可教学**：`{ok:false, error:{kind,hint,suggest}}`。打错键名→`suggest` 给最可能的；
  非法值→**当场拒、快照不变**（不静默）。读 `hint`/`suggest` 改，别重复试。
- **大数组走指针**：结果/包络不内联，只给 `{path,sha1,shape}`；被截断会告诉你 `kept/total/where`。
- **`psi_single` 用身份键，不是 label**；`set_drive` 的 `src/tgt` 可用 label 或 key。
- **π 脉冲时长自动算**（`t_π=500/Ω_MHz` ns）；给 `rabi_mhz` + `pulse="pi"` 即可，别手算时长。
- **arb（任意波形）只 `hard` 支持**：`sambe` 遇 arb 会自动改道 hard（回执有 `arb_note` 提示）。
- **两体/长演化别同步等**：`run_compute` 自动转 async 回 `job_id`，用 `job_status` 轮询。
- **journal 只记成功的状态变更**：失败的写入（打错键/非法值）**当场抛错、不进事件总线**——
  驾驶舱流水因此保持干净；你的“试错痕迹”在 harness transcript（另一层），不污染领域记录。

## 多步编排（`mecha_script`）
需要循环/批量/条件（如扫 N 个失谐、STIRAP/Ramsey 序列）时用 `mecha_script(script=…)`：
命名空间**只暴露** `registry / session / ops / fields / jobs` 门面 **+ `math`**（sin/cos/pi…）——
够不到裸引擎，改状态只能经 `session.set`（自动记事件、可 undo）。脚本把产物赋给 `result`，`print` 被捕获回传。
- 建场：`fields.set_drive/pi_pulse/pi2_pulse/square_mw/ramsey/stirap/arp`、`fields.add_wave(spec)`、`fields.clear_waves(atom)`。
- `add_wave`/`set_sequence` 的 **spec 键**：`kind`(square/gauss/file/arb) + `name/freq_mhz/pol/intensity_mwcm2/offset_ns`
  ；按 kind：square→`start_ns/edge_ns/height/fall_start_ns`；gauss→`center_ns/sigma_ns/peak`；file/arb→`t_ns/amp`。
  **打错键当场报错并给建议**（不静默用默认）。
- 扫描模板：`fields.clear_waves('single')` → 逐点 `fields.set_drive(..., name='sweep')`（同名替换）→ `ops.run(...)`。

> 边界（够不到的重要名字）：`runspec/build_run/evolve_run/static_atom/sambe/numpy/waveforms/Wave`
> 等裸引擎与裸波形类**都不在**命名空间——建场走 `fields`，跑演化走 `ops.run`/`run_compute`。
