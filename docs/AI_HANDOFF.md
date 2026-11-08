# 接手必读（给新接手的 AI / 人）

> 这份文档的目标只有一个：**让没参与过前文的接手者在 20 分钟内知道该做什么、不该做什么、以及哪些地方最容易理解错**。
> 它与 `用户手册.md`（软件怎么用）、`多体接口与维护约定.md`（契约细节）、`开发者指南.md`（改代码）互补；
> 冲突时以**代码 + `el_query_params` 的实际返回**为准，本文档会跟着更新。

---

## 0. 一页速览

| 你需要的 | 在哪 |
|---|---|
| 软件是什么 | EL = 里德堡原子（Rb-87）多体仿真软件；通过 mecha 暴露 15 个领域工具 + 1 个只读元工具；页面是只读投影 |
| 唯一入口 | 由 mecha launcher 启动的 `pythonw app_entry.py --mcp`；**绝不要自己再起一个 server** |
| 端口 | 读仓库根下 `.mcp-port` / `.monitor-port` / `.run-monitor-port` / `.control-port`，**每次重启都会变** |
| 配置 | `el_query_params`（权威清单）→ `el_set_config`；持久化在 `.mecha-v2/journal.jsonl`，重启后保留 |
| 卡片 | `.mecha-v2/run-cards/<run_id>.{meta,viz}.json`；最多 30 张、磁盘上限 1 GiB、最旧先删 |
| 模型 | `manybody.calib`（atoms + pairs 的 V/J/A）是**唯一输入**；文件/ARC 由 `build_calib` 先转成它 |
| 长任务 | `el_submit_job` → DSH 后台 job → 终态通知（不要用 `run_sim` 跑分钟级作业） |
| 判据 | `python run_checks.py`（主干 smoke，<1 min）；`--cascade` 失败再展开 core |
| 多体开发/取舍 | [多体开发纲领](多体开发纲领.md)（长期）；施工排期见 [施工规划](施工规划-多体观测量与共享内核.md) |

---

## 1. 系统与进程

### 1.1 单端点铁律

- EL 只有**一个** HTTP/MCP 端点，由 mecha launcher 拉起。页面的 `run-monitor` 端口只是这个进程里的一个只读视图。
- 不要在别处再起 `app_entry.py`：Windows 的 `SO_REUSEADDR` 会让第二个进程"绑定成功"而实际抢同一个端口，症状是请求随机 404/连接被重置，极难排查。
- 页面**只读**：它从 `.mecha-v2/run-cards/` 读，不写任何运行记录。任何"页面能不能改结果"的想法都是误解。

### 1.2 启动 / 重启

用 mecha 的软件管理工具（`open_software` / `close_software`，id = `EL`）。

**已知假象**：这两个调用经常返回 `Error: Connection closed`。这是 launcher 重启子进程时聚合连接断开导致的，**不代表失败**。正确确认方式是：

```powershell
# 1) 状态（state 必须是 ready）
(Get-Content "D:\code-nosync\mecha\_mecha_runtime\state.json" -Raw | ConvertFrom-Json).children |
  Where-Object { $_.id -eq 'EL' } | Select-Object state,port,ready

# 2) 端口文件（重启后端口会变，永远读文件）
Get-Content .mcp-port,.run-monitor-port

# 3) 页面活着
curl.exe -s -o NUL -w "%{http_code}" "http://127.0.0.1:<run-monitor-port>/"   # 期望 200
```

轮询时给 5–10 秒；`starting` 是正常中间态。若 `state=failed` 或 `last_error` 非空，去读 `D:\code-nosync\mecha\_mecha_runtime\logs\EL.log`。

### 1.3 改了 EL 源码之后

**必须重启 EL**（Python 进程不会热加载）。`web/run_monitor/*.js` 是个例外：静态文件，刷新页面即可（但 `mecha_v2/run_monitor.py` 这种后端改动仍要重启）。

---

## 2. 配置与状态

- 工具面（16 个）：`set_config` / `read_config` / `reset_config` / `query_params` / `estimate_run` / `preview_run` / `run_sim` / `query_levels` / `submit_job` / `read_job` / `cancel_job` / `set_drive` / `read_waves` / `run_actions` / `read_write_authority` / **`build_calib`**（文件/ARC → calib）。
- **`el_query_params` 是唯一权威**：类型、枚举、单位、值域都在里面。写配置前先查，别猜。
- `el_set_config` 的回执会带 `resolved` 与 `readback`；**大值（如 calib）会把 `before` 全量回显**，注意输出体积。
- `el_reset_config` 清空所有已设置键（含波形），用于"开始一轮新实验前收拾台面"。
- 配置**持久化**：重启后快照恢复（`.mecha-v2/journal.jsonl`）。所以"重启后配置还在"是正常的，不要以为是缓存。

### 2.1 写权

`el_read_write_authority` 显示当前谁能写。AI 面与人工面通过 mecha gate 协调；写被拒时先读它，不要重试硬撞。

---

## 3. 多体模型：`calib` 是唯一输入

```json
"manybody.calib": {
  "atoms": [{"id": 0, "pos": [0.0, 0.0, 0.0],
             "delta_mhz": -0.1, "Omega_mhz": 0.0, "gamma_mhz": 0.0}],
  "pairs": [{"i": 0, "j": 1, "V": 0.0, "J": 1.05, "A": 0.0}]
}
```

- 单位：位置 µm，能量 MHz，时间 ns。
- 哈密顿量：`H = Σ[δᵢnᵢ + (Ωᵢ/2)σˣᵢ − (i/2)Γᵢnᵢ] + Σ[V nᵢnⱼ + J σ⁺ᵢσ⁻ⱼ + A σ⁺ᵢσ⁺ⱼ + h.c.]`
- **符号陷阱**：投影映射是 `|g⟩↔|↑⟩`，因此 `Sᶻ = 1/2 − n`。若用 `n − 1/2`，`E₀` 不变但 `⟨O_m⟩` 整体反号。
- **没有 `terms` 输入**：活跃通道由 `pairs` 的数值决定（V≠0→I1，J≠0→I2，A≠0→I3）。不用某通道就把系数写成精确的 `0.0`。
- `file`/`arc` **不是** calib 的写法：见下面的构造器。

### 3.1 装配：`manybody.build` 就是构造器的入参

快照里唯一"不是 calib"的装配物，是 **`manybody.build`**——它和 `el_build_calib` 收的是
**同一份数据**，不是给人看的备注（否则必然退化成摆设）：

```jsonc
{"source": "file", "file": "n36_mass_g0.1.json"}
{"source": "arc", "atoms": [{g,r,pos,…}],
 "arc": {B_gauss, n_window, r_cut_um, second_order, guard_mhz, periods_um}}
```

- `el_build_calib` 不传参时读它重跑一次装配；传参会覆盖并**写回同一份**；
- 产物写 `manybody.calib`；**回执由 calib 现算**（`build_receipt`：指纹/对数/Σ|J|/尾界/丢对/剔除），
  不另存副本，所以不会漂；
- 两个**短**的阅读字段：`manybody.label`（卡片标题）、`manybody.notes`（折叠区备注）——只用于读卡。

### 3.2 不在 RunSpec 里的键

多体只认 `manybody.calib`；字段表/ARC 参数不是 RunSpec 输入，文件名与 ARC 参数先由
`el_build_calib` 转成数。多体运行**不读** `field.*`——B 场只属于单/两体，或 ARC 构造器。
- 周期边界不是引擎开关：把像求和算进 `pairs` 的 J 值即可（项目侧 `calib_build.from_positions(periods_um=…, images=…)` 就是干这个的）。

---

## 4. 请求类型（`manybody.want`）

| 值 | 含义 | 关键配套键 |
|---|---|---|
| `""`（默认） | 含时演化 | `dynamic.t_max/npts`、`manybody.solver` |
| `gs` | 基态 | `manybody.eq_backend` |
| `thermal` | 有限温 | `manybody.betas` |
| `dos` | 态密度 | `manybody.kpm_*` |
| `spectrum` | 源态谱 | `manybody.source` + `manybody.spectrum` |

**两个后端名别搞混**：

- `manybody.eq_backend`（`ed/lanczos/dmrg/kpm/qmc/free-fermion/…`）管**平衡态**；
- `manybody.solver`（`ed/tdvp/tebd/…`）只对**含时**有效。把 `dmrg` 写进 `solver` 会被枚举直接拒绝——这是设计，不是 bug。

---

## 5. 初态（`manybody.psi0`）

| 形态 | 说明 |
|---|---|
| `null` | 全基态 `\|g…g⟩` |
| `123` | 产物态索引（site0 = MSB） |
| `"rgrg"` | 模式串，长度必须 = N |
| `"ground"` | 先求基态再做初态 |
| `["sup", [[模式, amp], …]]` | 叠加 |
| `{kind:"product", amplitudes:[[a_g,a_r],…]}` | 一般局部乘积态 |
| `{kind:"mps", backend:"tenpy", …}` | 原生态（Python API 返回的对象；跨会话用 HDF5 checkpoint） |
| `{kind:"ground_of", calib:{…}}` | 初态是**另一个模型**的基态（淬火/制备态） |
| `{kind:"permutation_sector", base:<任意 psi0>, permutation:[…], sign:±1}` | 对合站点置换的 ± 本征扇区 |

要点：

- `ground_of` 与 `permutation_sector` 由后端决定落地形态：ED → 全空间向量（维度上限 `2^20`）；MPS → 原生 TeNPy 态，**order/charge 与本次运行一致**。
- 制备结果会写进 RunSpec 的 `quench` 字段并打 warning（`prepared_by`、`projection_weight`、`sector_overlap`、`base_overlap`）。
- `permutation_sector` 的 `permutation` 必须是**对合**（`p[p[i]]==i`），`sign` 只能 ±1；权重为 0 的扇区会明确报错。

---

## 6. MPS / DMRG 的四条硬规则（血泪）

1. **必须显式给 `manybody.mps_order`，用 rung 顺序。**
   默认 `cylinder_order` 会把同一个物理 rung 拆到链的不同位置（实测 `n_legs=8`、腿长不等），χ=512 也压不住截断：能量偏 0.019、截断 4e-4，直接触发 `TenpyInconsistencyError: max truncation error exceeded`。
   N=lm×ln×3（kagome，site id = `(m*ln + n)*3 + sub`）的 rung 顺序：

   ```python
   order = [(m*ln + n)*3 + sub for n in range(ln) for m in range(lm) for sub in range(3)]
   ```

2. **初态的 order 必须与运行一致。** 否则 `Warm-start order mismatch; no implicit reordering`。`permutation_sector`/`ground_of` 的 MPS 准备已经会跟随 `mps_order`。

3. **诊断很吃内存。** 精确方差/置换诊断默认 512 MiB 预算常常不够（"即使逐项分组也装不下"），给 `manybody.memory_budget_mb: 4096`。

4. **投影只约束初态，不约束 DMRG；而且 DMRG 不保证留在该扇区。**
   实测（N36 2×6，χ=256）：从 ⟨P⟩=−1 的初态出发，末态 ⟨P⟩=**+0.9999984**——扇区没被守住。
   同一模型两次运行的末态 ⟨P⟩ 都是 +1，但能量差 7.7×10⁻⁶，其中**能量高的一次 `solver_converged=false`（撞 sweep 上限）**。
   所以：
   - 比较能量前先看 `diagnostics.solver_converged`，未收敛的差值是**数值**不是物理；
   - 要声称"某扇区的基态"，必须（a）末态 ⟨P⟩ 匹配，且（b）`solver_converged=true`；
   - 真正的扇区分辨求解（求解过程中约束对称性）目前**引擎不支持**，是已知能力缺口。

---

## 7. 结果与卡片

- 结果对象是 `data_schema.quantities/axes` 描述的数组集合：`n_atom`（逐原子占据）、`corr`（连通占据关联）、`evals`、`spectral`（源态谱）、`E/C/F`（热力学）等；标量进卡片 metrics。
- 摘要（`run_sim` / `read_job` 返回）只给标量 + 少量数组；**完整数据在 artifact**：`/api/runs/<monitor_id>/artifact`。
- 卡片 = `.mecha-v2/run-cards/<id>.meta.json`（元数据，失败卡也带 `error` 文本）+ `.viz.json`（可视化投影）。页面 `/api/runs` 只列最近 30 张。
- 卡片输入区有一个 **模型来源** 行：`显式 calib（不碰 ARC）` / `calib 文件引用` / `ARC 标定（真实原子）`。三者**最终都汇成同一种 calib** 进 RunSpec——ARC 只是 calib 的生产者，不是并列的执行路径。
- 非 ARC 运行不显示「磁场 B」与「具体能级」两栏（那是 ARC 路的概念）；初态显示人话（如 `置换扇区 sign=-1（投影权重 0.5）→ MPS`），相互作用显示 `I2（交换 J）` 这类译名。
- 卡片输入区几何取**本次运行的 calib**；节点数值在 24–48 站点且半径 ≥7px 时显示；色标默认按数据实际范围，可切回语义范围（占据 0–1 / 自旋 ±1）。
- 轴单位来自 `data_schema.axes.<axis>.unit`（随 dataset 下发为 `axis_unit`），前端常量表只作兜底。

---

## 8. 长任务

- 分钟级以上的作业一律 `el_submit_job`（立刻返回 `job_id`），DSH 会把它桥成后台 job 并在终态通知里带结果。
- EL 内部作业**串行**；`el_cancel_job` 在下一个步边界生效，已结束的取消无副作用。
- 不要用 `run_sim` 跑长作业（会占住调用通道），也不要轮询得太密。

---

## 9. 判据与自检

```powershell
python run_checks.py --core          # 核心集（含下面的多体判据）
python run_checks.py --fast          # 更宽的场景集
python checks/manybody_source_spectrum_selfcheck.py   # 单个跑
```

| 判据 | 管什么 |
|---|---|
| `manybody_contract_selfcheck.py` | RunSpec/结果契约、calib 解析、投影基、错误路径（48 项） |
| `manybody_source_spectrum_selfcheck.py` | 源态谱对**精确对角化**：节点/权重/峰位/和规则/扇区泄漏 |
| `manybody_quench_selfcheck.py` | `ground_of` 淬火：ED 与 MPS 两条路对精确时间演化 + 态矢入口 |
| `manybody_prepare_selfcheck.py` | `permutation_sector`（ED/MPS、order 跟随、拒绝路径）+ 构造器（文件→calib 等价性与越界拒绝） |
| `run_monitor_selfcheck.py` / `run_visualization_selfcheck.py` | 卡片存储、进度、惰性结果、只读 HTTP、可视化投影 |

**新增引擎能力必须自带判据**，并登记进 `run_checks.py` 的 `CORE`。没有判据的能力视为不存在。

---

## 10. 症状 → 原因 → 处理

| 症状 | 原因 | 处理 |
|---|---|---|
| `tool execution arguments must be losslessly JSON-serializable` | 参数里含 `-0.0`/NaN/Inf（DSH 参数层拒绝） | 归一化 `-0.0→0.0`（`validate_calib` 已自动做）；重试一次也可能成功（历史上是抖动） |
| `TenpyInconsistencyError: max truncation error exceeded` | MPS 顺序差或 χ 不足 | 给 rung `mps_order`；再不够加 χ |
| `Exact variance cannot fit budget even with individual term groups` | 诊断工作区超 `memory_budget_mb` | 提到 4096 MiB 或关掉诊断 |
| `Warm-start order mismatch; no implicit reordering` | 初态 `order` ≠ 运行 `mps_order` | 让初态跟随 `mps_order`（已修）；或不要显式给 order |
| `DMRG initial/final occupation does not match charge_sector` | 初态不在扇区，或末态非规范 | 检查初态 r 数；必要时给更多 sweep / 更大 χ |
| `manybody.solver='dmrg' 不在枚举 [...]` | 后端名放错键 | 平衡态用 `manybody.eq_backend` |
| `ed does not support equilibrium options: mps_order` | 从 MPS 后端切回 ED，但 MPS 专用键仍留在配置里 | 把 `mps_order`/`mps_diagnostics` 置 `null`（引擎**拒绝静默忽略**未知/不适用选项，这是故意的） |
| 卡片输入几何画成方阵 | 几何来源不是本次 calib | 取本次 calib；重跑旧卡一次 |
| 卡片节点没有数值 | 半径阈值（<11px 时省略） | 已放宽到 7px；或放大画布 |
| 色标一片同色 | 占据类数据被固定映射到 0–1 | 控制条切「色标：自动范围」 |
| 两次运行能量差很小（1e-6 量级） | 很可能是收敛差异，不是物理 | 先看卡片 `diagnostics.solver_converged`；未收敛的差值是数值 |
| 想比较"对称扇区"的能量 | 投影只改初态，DMRG 守不住扇区 | 同时核对末态 `permutation_overlap` 与 `solver_converged`；真正的扇区约束求解尚未实现 |

---

## 11. 典型工作流（可直接照抄）

### A. 小体系 ED 基态（秒级）

```json
{"dynamic.mode": 2, "manybody.want": "gs", "manybody.eq_backend": "ed",
 "manybody.calib": {"atoms": [...], "pairs": [...]}}
```

### B. 大体系 DMRG（分钟级）

```json
{"dynamic.mode": 2, "manybody.want": "gs", "manybody.eq_backend": "dmrg",
 "manybody.calib": <build_calib(file="n36_mass_g0.1.json") 的产物>,
 "manybody.charge_sector": 18, "manybody.chi_max": 256, "manybody.max_sweeps": 30,
 "manybody.mps_order": [0,1,2,18,...], "manybody.memory_budget_mb": 4096,
 "manybody.psi0": {"kind": "permutation_sector", "base": "rrrr...gggg",
                   "permutation": [18,19,...,17], "sign": -1}}
```

### C. 源态谱

```json
{"manybody.want": "spectrum", "manybody.eq_backend": "ed",
 "manybody.source": {"kind": "diagonal", "values": [0,1,1,2,...]},
 "manybody.spectrum": {"eta": 0.05, "n_krylov": 200, "n_omega": 1201}}
```

### D. 淬火（初态是另一个模型的基态）

```json
{"manybody.psi0": {"kind": "ground_of", "calib": {"atoms":[...], "pairs":[...]}},
 "dynamic.t_max": 100, "dynamic.npts": 501}
```

### E. 查卡片 / 取完整结果

```
GET http://127.0.0.1:<run-monitor-port>/api/runs
GET http://127.0.0.1:<run-monitor-port>/api/runs/<id>/result
GET http://127.0.0.1:<run-monitor-port>/api/runs/<id>/artifact
```

---

## 12. 文件地图

| 路径 | 管什么 |
|---|---|
| `src/core/runspec.py` | 总装配入口：`build_run` / `evolve_run` / 摘要；按 `want` 分派平衡态与含时 |
| `src/core/manybody/tdvp_dense.py` | 稠密 CPU 两站点 TDVP（`tdvp-dense`）；支持 `drive_segments`、`pauli_strings`、原生 MPS `return_state`；拒绝 charge/corr |
| `src/core/manybody/runspec_mb.py` | 多体装配：calib 解析与校验、两条装配路、`psi0` 制备（`ground_of` / `permutation_sector`） |
| `src/core/manybody/eq_engine.py` | 平衡态 RunSpec（`want=gs/thermal/dos/spectrum`），含源态谱与制备态初态 |
| `src/core/manybody/eq_route.py` | 后端路由（ed/lanczos/kpm/dmrg/qmc/free-fermion），源态谱与关联的求解 |
| `src/core/manybody/spectrum.py` | Lanczos + 连分式展宽 + 未展宽求积 |
| `src/core/manybody/dmrg.py` | TeNPy DMRG 封装、MPS 顺序、诊断 |
| `src/core/manybody/mps_symmetry.py` | 站点置换与 ± 扇区投影 |
| `src/core/manybody/schema_mb.py` | `manybody.*` 配置键的权威描述与校验 |
| `src/core/manybody/results.py` | 结果数据契约（`data_schema`） |
| `mecha_v2/run_monitor.py` | 卡片存储（落盘、上限、只读快照） |
| `mecha_v2/visualization.py` | 结果 → 卡片投影（dataset 轴/单位/标签） |
| `web/run_monitor/` | 页面（`visualization.js` 管画布、色标、节点标注） |
| `checks/` | 全部判据；`run_checks.py` 是入口 |

---

## 13. 权威文档索引

| 想知道 | 读 |
|---|---|
| 软件怎么用、单位、波形 | `docs/用户手册.md` |
| 多体接口契约、两条装配路、psi0 全语法 | `docs/多体接口与维护约定.md` |
| 引擎算法清单与边界 | `docs/多体接口与维护约定.md`、`docs/引擎能力边界.md` |
| MPS 分析与资源契约 | `docs/有限MPS分析与资源契约.md` |
| 改代码的约定 | `docs/开发者指南.md` |
| 本轮数值进展与后续计划（项目侧） | `D:\desktop\数值尝试\01_QED3_质量形变与禁闭动力学\PLAN_20261031.md` |

---

## 14. 今天（2026-11-01）验证过的基准值

> 这些数字是回归参照；改了引擎就重跑核对，差了就要能解释。

| 场景 | 配置 | 结果 |
|---|---|---|
| N24 2×4 质量 g=0.1 | calib + 图像求和 ±16384、χ=512、M=0、rung 顺序 | `E0=-14.409565197`（旧流程 −14.409565196489）、质量均值 `-0.00452292`、`Var/N=1.09e-9` |
| 同上 χ=64 | 同上 | `E0=-14.409362`、质量均值 `-0.0045193`（34 s） |
| N36 2×6 质量 g=0.1，收敛 ✅ | χ=256、rung 顺序、初态 ⟨P⟩=−1、`solver_converged=true` | `E0=-21.637008422`、质量均值 `-0.0055456`、S=1.430、末态 ⟨P⟩=+0.9999984 |
| 同上，**未收敛** ❌ | 初态 ⟨P⟩=+1、撞 30 sweep 上限 | `E0=-21.637000695`、质量均值 `-0.0051493`、S=0.446、末态 ⟨P⟩=+0.9999995 |
| 旧流程 N36 χ=256 | — | `-21.637001693` / `-21.637008662` |

**怎么读这张表**：两个 N36 数字**不是**对称扇区的劈裂——末态 ⟨P⟩ 都是 +1，能量高的那次是
`solver_converged=false` 的未收敛解。可信值是 `-21.637008422`。旧流程的两个数很可能是同一
现象（其中一个是未收敛/局部极小），不要按"两支"解释，除非能拿出各自的末态 ⟨P⟩ + 收敛标志。
