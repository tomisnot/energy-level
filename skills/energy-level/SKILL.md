---
name: energy-level
description: 通过 MCP 操控 Rb-87 里德堡原子仿真软件 Energy Level（mecha v2）——配置能级/场/波形，运行单体、两体、多体含时与平衡态计算，读取布居、关联、纠缠熵和 Pauli 字符串。含工具签名、返回字段、完整调用示例、单位与拒绝面。
whenToUse: 需要操作 Energy Level（改参数、加能级组、建驱动、跑演化或平衡态、读结果）时。软件由人经统一启动器 `python -m mecha --all` 起，MCP 地址在项目根 `.mcp-port`；`app_entry.py --gui` 是纯本地独立进程，AI 看不到。
---

# Energy Level · AI 调用手册

这份文档是 MCP 工具面的**唯一操作说明**。工具 schema 与 `query_params` 的运行时返回是最终权威；
本文给的是“怎么组合、参数长什么样、返回什么、什么时候会被拒”。不要凭 v1 记忆或猜键名调用。

## 0. 先建立心智模型

```
配置快照（扁平键）  +  manybody.calib（多体模型，唯一货币）
        │
        ├─ 单体/两体：groups.list + field.* + dynamic.*（波形/初态/衰变）
        └─ 多体：manybody.*（含 time / gs / thermal / dos / spectrum）
        │
        ▼  estimate_run / preview_run（可选，不跑）
        run_sim（同步，回摘要）／ submit_job（异步，回 job_id）
        │
        ▼
摘要（top 态、布居、E0/gap…） + artifact（完整数组，pickle）+ 运行卡片
```

三条铁律：

1. **写只经 `set_config` / `set_drive` / `reset_config` / `build_calib`**；读用 `read_config` / `query_params` / `query_levels` / `read_waves` / `read_write_authority`。
2. **不要自己实现库里已有的能力**：多体装配、ED/MPS/DMRG/KPM/QMC、Pauli 串、源态谱、运行卡片都在库里；MCP 没暴露的走 GUI / Python API / artifact，见 §14。
3. **失败先读 `error.info.kind` / `hint` / `suggest`**，不要重试硬撞；写被拒先 `read_write_authority`。

## 1. 连接与写权

- MCP 端点：`http://127.0.0.1:<.mcp-port>/mcp`（streamable HTTP，仅 localhost）。
- 端口每次重启会变；dsh 插件会现读 `.mcp-port` 自愈，自写客户端需重连/改配置。
- `read_write_authority()` 只读：返回当前谁能写。AI 侧写恒归因 `ai`，不能改模式、不能伪造 human。
- 写被拒（`wrong_side` / `locked`）时**读仍可用**；等人切回 AI，不要杀进程/重启软件。
- 改 `.py` 源码后必须由人重启 EL（Python 不热加载）；不要自己 `app_entry.py`。

## 2. 工具清单（15 个领域工具 + 1 个只读元工具）

| 工具 | 作用 | 参数 | 成功回执关键字段 |
|---|---|---|---|
| `query_params` | 可写键元数据 | — | `keys{key:{type,unit,lo,hi,enum,desc,default}}` |
| `query_levels` | 当前基组态 | — | `n_states`, `states[{label,key,E_mhz,…}]` |
| `read_config` | 当前快照 | — | `config`（可能很大，含 calib） |
| `read_waves` | 驱动波摘要 | `atom="single"/"A"/"B"` | `waves[{name,freq_mhz,intensity_mwcm2,pol,offset_ns,envelope}]` |
| `set_config` | 写一个键 | `key,value,reason` | `applied,before,resolved,readback` |
| `reset_config` | 全部清回默认（含波形/calib） | `reason` | `cleared,n` |
| `build_calib` | 文件/ARC → `manybody.calib` | `build` 或 `file` 或 `atoms`+`arc`；`reason` | `calib_summary,build_receipt,build` |
| `set_drive` | 在跃迁上建波 | `src,tgt`；`rabi_mhz` 或 `intensity_mwcm2`；`pulse` 或 `duration_ns`；`pol,detuning_mhz,mode,atom,name,reason` | `key,wave,n_waves,pulse_end_ns,t_max_note?` |
| `estimate_run` | 干跑成本 | — | `est_sec` 及规模字段 |
| `preview_run` | 干跑预览 | — | `basis_enough`, 驱动/模型摘要；多体含 `n_segments,segment_t0,max_Omega_mhz,max_delta_mhz,pauli_count` |
| `run_sim` | 同步跑一次 | `cancel_after_sec?` | `ok,method,n_t,n_states,top_states_avg_pop,final_top_pop,summary,pauli_strings?,pauli_strings_total?,monitor_id,artifact_id?,warnings?` |
| `submit_job` | 后台任务 | — | `job_id,state` |
| `read_job` | 查任务 | `job_id?`（省略=列表） | `state`；done 带 `result`，failed 带 `error` |
| `cancel_job` | 请求取消 | `job_id` | `state`（协作式，终态以 `read_job` 为准） |
| `run_actions` | 多步编排 | `actions,spec?,spec_patch?,pick?` | `steps[{index,op,ok,detail,output}],stopped_at` |
| `read_write_authority` | 只读自省 | — | 当前 mode/谁能写 |

没有 `list_tools`；工具表就是宿主注册表。不要再找 `set_groups`、`run_compute`、`get_state` 这类 v1 名字。

## 3. 单体最小可跑流程

```jsonc
// 1) 能级组
set_config("groups.list", [{"n":60,"l":0,"j":0.5},{"n":60,"l":1,"j":1.5}])
// 2) 取身份键
query_levels()
// → states: [{"label":"60S 1/2 m_J=+0.5","key":["mJ",60,0,0.5,0.5,0.0],"E_mhz":…}, …]
// 3) 初态：原样回传 key
set_config("dynamic.psi_single", ["mJ",60,0,0.5,0.5,0.0])
// 4) 驱动（自动算频率/光强/时长）
set_drive(src=<60S key>, tgt=<60P key>, rabi_mhz=5.0, pulse="pi", pol="σ-")
// 5) 时间/方法
set_config("dynamic.mode", 0)
set_config("dynamic.method", "mixed")      // 可选 mixed / sambe / hard
set_config("dynamic.t_max", 500)
set_config("dynamic.npts", 501)
// 6) 预检与运行
preview_run(); estimate_run(); run_sim()
```

读结果：`final_top_pop` 是**演化窗口末端**的布居；`top_states_avg_pop` 是时间平均。
要“脉冲结束瞬间”的布居，把 `t_max` 调到 `set_drive` 回执的 `pulse_end_ns` 附近。

补充：

- 逐子能级裁剪：`groups.list` 元素可带 `ms:[mJ,…]` 或 `Fs:[[F,m_F],…]`（互斥）；
  不写=全展开。裁完重新 `query_levels`。
- 耗散：`dynamic.decay_index` 0/1/2（自动/禁/自定义）；自定义写
  `dynamic.decays=[{"from":<key>,"to":<key>|null,"gamma_mhz":Γ}]`；两体只支持 `to=null`。

## 4. 两体

```jsonc
set_config("dynamic.mode", 1)
set_config("dynamic.psi_A", <A key>)
set_config("dynamic.psi_B", <B key>)
set_config("dynamic.pair.R_um", 6.0)
set_config("dynamic.pair.theta_deg", 90.0)
set_config("dynamic.pair.phi_deg", 0.0)
set_drive(src=…, tgt=…, rabi_mhz=1.0, pulse="pi2", atom="A")
set_drive(src=…, tgt=…, rabi_mhz=1.0, pulse="pi2", atom="B")
```

- 两体只读 `waves_A`/`waves_B`；`waves_single` 在 mode=1 被忽略。
- 两体 + `psi_pair` 时仍要给 `psi_A`/`psi_B`（占位也可）；`psi_pair` 的对称组合可能是暗态。
  形状：`["pair_sup", [[nA,lA,jA,mA,nB,lB,jB,mB,re,im], …]]`。
- 两体寿命/耗散只支持“落点=丢失”（`to=None`）。

## 5. 多体：`calib` 是唯一模型输入

```jsonc
"manybody.calib": {
  "atoms": [{"id":0,"pos":[0,0,0],"Omega_mhz":0,"Omega_phase_rad":0,
             "delta_mhz":0,"gamma_mhz":0}],
  "pairs": [{"i":0,"j":1,"V":0.0,"J":"0.7+0.2j","A":0.0}]
}
```

- 位置 µm、能量/耦合 MHz、时间 ns；`V` 必须实数；`J/A` 可为复，JSON 里用字符串 `"0.7+0.2j"`（或实数）。
- **没有 `terms` 输入**：V≠0→I1、J≠0→I2、A≠0→I3；不用的通道写精确 `0.0`。
- 文件/ARC 不是 RunSpec 写法：用 `build_calib` 转成 calib。

```jsonc
// 文件（读 <EL>/.mecha-v2/calib/ 下的 JSON）
build_calib(build={"source":"file","file":"n36.json"})
// ARC（原子 g/r 四元组 + pos）
build_calib(build={"source":"arc",
  "atoms":[{"g":[60,0,0.5,0.5],"r":[60,1,1.5,1.5],"pos":[0,0,0]}],
  "arc":{"B_gauss":21.7,"n_window":3,"r_cut_um":7.0,"second_order":true,"guard_mhz":1.0}})
```

直接 `set_config("manybody.calib", {...})` 也能写，但大 calib 的 `before` 会全量回显；
换模型优先 `build_calib`。`manybody.build` 是构造器入参，运行时不读。

## 6. 多体时间演化

```jsonc
set_config("dynamic.mode", 2)
set_config("dynamic.t_max", 1000)
set_config("dynamic.npts", 501)
set_config("manybody.solver", "ed")        // auto/ed/tdvp/tdvp-dense/tebd/tdvp-mpo/tdvp-single/tdvp-adaptive/...
set_config("manybody.psi0", "rggrgg")      // 模式串 / 产物态索引 / ['sup',…] / {kind:"product",…}
set_config("manybody.want", "")            // 空 = 含时
set_config("manybody.max_step_ns", 0.1)    // MPS/PEPS/经典/轨迹的内部步长
run_sim()
```

常用可写键（完整清单以 `query_params` 为准）：

- `manybody.solver`：含时后端；`auto` 只在 ED/TDVP 间选。`tdvp-dense` 是纯 CPU 稠密两站点 TDVP（有限开链/分段驱动/厄米/乘积初态；支持 segments、Pauli 与原生 MPS `return_state`；拒绝 charge/corr），需显式选择；环境变量 `EL_TDVP_BLAS_THREADS` 可限制其 BLAS 线程。
- `manybody.psi0`：初态；`"ground"`、`{kind:"product"}`、`{kind:"mps"}`、`{kind:"ground_of",…}`、`{kind:"permutation_sector",…}`。
- `manybody.charge_sector`：固定激发数（ED/Lanczos/DMRG；非守恒输入会拒）。
- `manybody.constraint`：`{kind:"none"/"blockade"/"pxp",mode:"full"/"project",threshold_mhz,edges}`；投影基当前主要给 ED。
- `manybody.distance_cutoff_um`：可选距离截断（µm，按显式 pos 欧氏距离，不做周期最小像）；只保留 R≤cutoff 的对，丢弃项进 `calib.dropped`/warnings，属近似，需自查 ξ/χ。
- `manybody.drive_segments`：`[{t0,Om:[N],delta:[N],phase:[N]}]`；长度必须=N。
- `manybody.drive_interp`：`left`（默认，左端点阶梯，逐位兼容）/ `midpoint`（段条目视为线性 ramp 控制点，取段中点值；仅 ED/CPU MPS/tdvp-dense，ramp 二阶）。
- `manybody.loops` / `manybody.link_pairs`：环路/有向边；闭环边自动并入 `link_pairs`。
- GPU `manybody.dtype="float32"` = complex64，短程实测约 2×；它是近似开关，长时窗需 float64 对拍。
  24 点长程 `swap-tebd` 仍是 O(pair×distance) SVD，不建议当扫描工具。
- `manybody.pauli_strings`：见 §9。
- `manybody.chi_max/truncerr/max_sweeps/mps_order/mps_dtype/mps_diagnostics/return_state/memory_budget_mb/dtype/device`：MPS/资源选项。

## 7. 多体平衡态与谱

```jsonc
set_config("dynamic.mode", 2)
set_config("manybody.want", "gs")           // gs / thermal / dos / spectrum
set_config("manybody.eq_backend", "ed")     // ed/lanczos/gpu-lanczos/dmrg/idmrg/kpm/qmc/free-fermion
set_config("manybody.charge_sector", 4)     // 可选
run_sim()
```

- `want="thermal"` 配 `manybody.betas`（1/MHz）；`want="dos"` 由 KPM 生成 `energies/rho`
  （`manybody.kpm_num`/`kpm_nvec` 控精度）。
- `eq_backend="ed"` + `want="thermal"` 且 N≤12 时给 `n_atom (n_beta,N)` 与
  `corr (n_beta,N,N)`（连通占据 `⟨n_i n_j⟩−⟨n_i⟩⟨n_j⟩`）；N>12 仍给 E/C/F，
  但带 `corr_unavailable_reason`，不填零。局域线性响应用两次 δ± 热态有限差分，不在引擎里。
- `want="spectrum"`：`manybody.source` 可给 `{kind:"vector"|"psi0"|"diagonal"|"sparse"|"full_sparse"|"site_weights"}`；
  省略 source 时默认用 `manybody.psi0` 作 `|0⟩`（模式串/向量会物化，原生 MPS 会拒）。
  源态谱只支持 ED/Lanczos 类后端，DMRG/QMC/KPM 会拒。
- 平衡态结果：`E0`、`gap`、`n_atom`、`corr`、`evals`；热态另加 `E/C/F+betas`；
  DOS `energies/rho`；谱 `spectral/energies/source_norm`。
- DMRG 扫描：`return_state=True` 取 `state.kind="mps"`，下一点作为
  `manybody.psi0={kind:"mps",...}` 传入；order/charge 由 state 携带，不要手改。
- iDMRG：`eq_backend="idmrg"` + `manybody.idmrg={cell,terms}`（不能同时给 `manybody.calib`）。
  通用 `cell+terms` 不是阵列枚举；`cell>=2`、一个平移方向、局部维数 2、`dx<=64`。
  结果用 `E_per_site/E_per_cell/S_per_bond/corr_length/chi/trunc_err`，没有有限体系 `E0`/`gap`。
  `manybody.pauli_strings` 也支持（cell 内任意阶；X/Y 自动展开成 P/M，固定 Sz 只留净 ΔN=0 项），
  可算手性 χ 等三体串；Rydberg 单胞用 `from_rydberg_cell` 转 terms，但 iDMRG 核心不绑定 Rydberg。
- `estimate_run` 对多体 `charge_sector` 不缩维（按 2^N 估，偏高）；`preview_run` 对多体是模型摘要，不是跃迁预览。

## 8. 系综驱动（Python API，不是 MCP 工具）

```python
from src.core.manybody import run_ensemble
r = run_ensemble(params, sampler, n_samples=100, seed=1234,
                 observables=('pops_atom', 'corr'),
                 conditional=lambda res: res['pops_atom'][-1, 0] > 0.5,
                 return_samples=False)
# r['unconditional'] / r['conditional'] -> {mean, stderr, n}
# r['n_valid'], r['rejected'], r['error_report']
```

- `sampler(rng) -> calib` 或 `(calib, overrides)`；可改初态做初始缺陷，原子数/ID/pair 会被校验。
- 内部复用 `protocol.build_run + evolve_run`，ED/MPS 走同一分派；不新写求解器。
- `stderr = std(ddof=1)/sqrt(n)`；n=1 为 NaN；无条件/后选两套，后选不得冒充无条件。
- 速度不是 calib 字段：shot 间初始位置偏移折进 `sampler` 的 `pos`；
  时变位置/Doppler 属新物理模型，不支持。
- 因为 sampler 是 callable，MCP 不暴露；MCP 批量扫描用 `run_actions` 的 `spec_patch`；
  CLI 薄包装 `scripts/run_ensemble.py` 可做位置抖动/初态翻转。

## 9. Pauli 字符串（通用高阶观测）

```jsonc
set_config("manybody.pauli_strings",
  [[[0,"x"],[1,"y"],[2,"z"]], {"label":"z0","terms":[[0,"z"]]}])
```

- 引擎只算 `⟨Π σ^a_i⟩`；手性、结构因子、连通减法由调用方组合，引擎不认“chirality”这种物理名。
- 支持：含时 `ed`、CPU MPS 四档、GPU MPS 两档、轨迹（系综平均）、经典自旋（因子化）；
  平衡态 `ed/lanczos/gpu-lanczos` 的 gs、`dmrg` 的 gs。PEPS/QMC/KPM/free-fermion/gpu-batch 拒绝并给替代。
- 结果：含时 `observables.pauli_strings (n_t,n_strings)` + `observables.pauli_string_labels`；
  平衡态顶层 `pauli_strings (n_strings,)` + `pauli_string_labels`。
- MCP `run_sim`/`read_job` 摘要按 `data_schema.summary` 策略给前 12 条终值
  （`pauli_strings`、`pauli_strings_total`、`pauli_strings_time_ns`）；完整数组仍在 artifact。
- MPS 用局域算符之积（σˣ=P+M、σʸ=iP−iM、σᶻ=SZ）；实哈密顿量本征态的奇数 σʸ 串期望恒 0（时间反演），
  要非零手性用复模型或非本征态。近简并基态上单态期望不唯一，先看 `gap`。

## 10. `run_actions`：扫描与多步

```jsonc
run_actions({
  "actions": [
    {"op": "estimate"},
    {"op": "run", "args": {"spec_patch": {"dynamic.t_max": 500}}},
    {"op": "run", "args": {"spec_patch": {"dynamic.t_max": 600}}}
  ],
  "pick": ["final_top_pop", "method", "ok"]
})
```

- op：`estimate`/`run`/`summary`/`schema`，或只读查询 `read_config`/`query_params`/`preview_run`/`query_levels`/`read_waves`。
- `estimate_run`/`run_sim` 可当 op 别名。
- `spec` 整表替换；`spec_patch` 只改列出的键；都不给=当前快照。
- **`run_actions` 不能 `set_config`**；批量改配置要多次 `set_config`，只有跑/估/读能进编排。
- `pick` 是键路径列表（整键优先，再点分层）；缺失键进 `_missing`，不静默 null。
- 每步 `output` 自带；`stopped_at=-1` 全成。

## 11. 长任务、摘要与 artifact

```jsonc
submit_job()                    // 固定提交时快照
read_job(job_id)                // 轮询；done 带 result
cancel_job(job_id)              // 协作式；以 read_job 终态为准
```

- `run_sim` 只回摘要，不搬大数组；摘要由 `data_schema.summary` 策略派生（标量/前几条终值），完整 `pops/t/observables` 在 artifact。
- `want=thermal` 的摘要另外给 `corr_first_last`（首/末 β 的 n×n 矩阵）和 `n_atom` 的逐 β 行，不把二维关联拍平。
- `run_sim` 回执 `artifact_id`/`monitor_id`；artifact 在
  `http://127.0.0.1:<.run-monitor-port>/api/runs/<id>/artifact`（pickle 的 `(meta,res)`）。
- 运行卡片是只读投影，不能创建/重跑/删除运行；页面端口是 `.run-monitor-port`，不是 `.monitor-port`。
- 取消后 `read_job` 的终态与部分结果以回执为准；不要假设 `cancel` 立即停。

## 12. 单位与身份键

- B=Gauss，E=V/cm，t=ns，微波频率=MHz，光频=Hz，Γ=MHz，光强=mW/cm²。
- 能级身份键来自 `query_levels` 的 `key`（含 kind/n/l/j/m/E），**不要用 label 或猜索引**。
- 两体初态分量按 `(n,l,j,m)` 解析；多体 calib 的 `g/r` 是四元组。
- 多体 H 的耦合是**圈频 MHz**（不是 rad）；论文的 rad/µs 要先 /2π。
- 基序：原子 0 在 MSB，`|g⟩=0, |r⟩=1`，`S^z=1/2−n`；本仓 `sp/sm` 与教科书 σ± 编号相反。

## 13. 错误与拒绝面（不要试错）

失败回执是 `isError=true`，正文：

```json
{"ok":false,"error":{"message":"…","info":{"kind":"…","hint":"…","suggest":"…"}}}
```

常见 `kind`：`unknown_key`（先 `query_params`）、`bad_enum`/`bad_value`/`bad_type`（看 schema）、
`readonly_key`（`engine.*` 不可写）、`missing_arguments`、`bad_drive`（身份键/偏振/强度时长）、
`build_failed`（配置组不成 RunSpec）、`unknown_job`、`gate_denied`/`wrong_side`（写权）。
`hint`/`suggest` 是给下一步的，不要原样重试。

明确的拒绝（属设计，不是缺陷）：

- 零波形且无耗散通道 → 拒绝空演化。
- 多体 `V` 含虚部、`terms` 与数值不一致、`charge_sector` 非守恒 → 拒绝。
- PXP/blockade 投影给非 ED 后端 → 拒绝。
- PEPS 非矩形网格/长程键；DMRG projected Plan；GPU 固定荷；不匹配的原生 MPS → 拒绝。
- MPS 路径 `gamma≠0`；两体显式落点/纯退相 → 拒绝。

## 14. MCP 没暴露什么（别在 MCP 里造轮子）

| 想要 | 现状 | 正确做法 |
|---|---|---|
| 静态谱/AC-Stark 扫描 | 无专用 MCP 工具；`query_levels` 只给能级 | 用 GUI 静态页，或进程内 Python `src.core.protocol.static_*` |
| 并行扫描提交 | 无 `submit_sweep` | `run_actions` 串行多步，或循环 `submit_job`；独立 runner 属脚本/GUI |
| 运行诊断 `get_diag` | MCP 无 | 看 `run_sim` 的 `warnings`、`preview_run`；完整数据在 artifact |
| 任意 Python / `run_python` | 无 | 宿主 code-runtime，不是库工具 |
| 截图/undo/存档面 | 无 | GUI；journal 即存档；撤销=反向写新事件 |
| 具名 `set_groups`/`set_superposition` | 无 | `set_config("groups.list"/"dynamic.psi_*")` |
| 多体专用 `set_atoms` | 无 | `build_calib` 写 `manybody.calib` |

## 15. 当前支持面（选后端先看这里）

含时：`ed`（精确小维）、`mixed`（默认，光 Sambe+微波当场）、`sambe`（无 RWA）、`hard`（微波硬积分）、
CPU MPS（`tebd`/`tdvp`/`tdvp-single`/`tdvp-adaptive`/`tdvp-mpo`/`tdvp-dense`）、GPU MPS（`tebd-gpu`/`swap-tebd-gpu`）、
`tdvp-trajectories`、`classical-spin`、`peps-real-time`。

平衡态：`ed`、`lanczos`、`gpu-lanczos`、`gpu-batch`、`kpm`/`gpu-kpm`、`dmrg`、`qmc`、`free-fermion`。

- `auto`（含时）只选 ED/TDVP；不会静默降级到经典。
- `kpm/qmc/free-fermion` 没有多体态，不能给 Pauli；
- 轨迹/经典/PEPS 是近似语义，必须看 `observables.approximation`。
- 具体适用条件以 `preview_run`、`query_params`、`docs/引擎能力边界.md` 为准。

## 16. 深入文档

- 契约与全部多体键/结果字段：`docs/多体接口与维护约定.md`
- 引擎适用边界与后端选择：`docs/引擎能力边界.md`
- 配置/运行/排障：`docs/用户手册.md`
- 代码地图与模块职责：`docs/代码总览.md`
- 有限 MPS 资源契约：`docs/有限MPS分析与资源契约.md`
- 性能/观测开销微基准：`python scripts/bench_mps.py --help`；输出同时给 `chi_actual`（全过程实际最大 χ）与 `chi`（cap），只报 cap 会误导。

## 17. 维护规则

- 工具签名/返回字段以 `mecha_v2/tools.py` 与运行时 schema 为准；本文与代码冲突时先改本文。
- 本文是权威；`.dsh/skills/energy-level/SKILL.md` 是镜像（同一文件），不要手改镜像。
- 新增/删除工具、配置键、后端时，同步更新本文的 §2/§5/§13/§15；`checks/skill_mcp_contract_selfcheck.py`
  机械校验工具名单与镜像，`run_checks.py` 默认 smoke 会跑它。
- 跑测试默认 `python run_checks.py`（smoke），共享层改动用 `--cascade`/`--changed`。
