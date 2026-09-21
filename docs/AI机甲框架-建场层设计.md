# AI 机甲框架 · 建场层设计（波形/含时场的语义构造）

> **定位**：把"建含时电磁场"这块**唯一还退化成裸结构**的语义面，补成一个既符合
> 《AI机甲框架-开发原则》、又让 AI **得心应手**的适配层。
> **背景**：wave 是仿真实验里**调节最频繁**的对象；Re0 现状只有 `dynamic.waves_*`
> 一个 `type='list'` 的可写键（橡皮图章校验），AI 得**手搓** `serialize_waves` 的元组
> 编码、**猜** `freq_mhz`（光学 ~3e8 MHz）、**猜** 强度↔Rabi 关系。这违反已落地的
> I6/D5/D2，也是旧项目"MCP 不适清单"#1/#2/#3 的老痛点。
> **落点**：新增 `src/mecha/fields.py`（适配层，物理专属，§11「不可提炼的适配」）；
> `src/core` **一行不改**，只**调用** `waveforms` / `static_atom`（`rabi_mhz` /
> `intensity_mwcm2_from_rabi` / `dipole_amplitude`）/ `field_build` / `atomic_levels` 既有函数。
> **分层裁决（已定）**：底层只存物理量（`freq_mhz`/`intensity_mwcm2`）保最大自由度；
> 共振频率/失谐/Rabi/π时长 都是**辅助层的按需捷径**（复用旧 GUI `ui_waves` 辅助区同一批 core 函数）。

---

## 0. 设计目标（一句话）

> AI 说"**在 60S→60P 上打一个 Ω/2π=50 MHz 的 π 脉冲**"就能建好；改一个参数只发一个
> patch；建完即 preview 验证；大数组走指针；全程经 choke point（一条事件、可 undo）。

四条硬要求，逐条对齐纲领：
1. **语义入口，不猜结构**（§2.2）：AI 给物理意图（跃迁/Rabi/脉冲类型），不给元组编码。
2. **契约锁死**（P0.4 / I6 / D5）：非法波形在 `set()` **当场**报错，不拖到 `build_run`。
3. **大载荷走指针**（I3）：file/arb 包络永不内联进快照/事件/回执。
4. **经 choke point、可归因可逆**（I1/I2/M1/M2）：每次构造 = 一条 `set` 事件，带 readback。

---

## 1. 三层 API（渐进披露；越常用的越高层）

| 层 | 面向 | 入口 | 何时用 |
|---|---|---|---|
| **L2 意图级** | 最常用 | `set_drive` / 脉冲协议预设 | "在某跃迁上打个 π/2π/STIRAP/Ramsey/啁啾" |
| **L1 逐波级** | 常用 | `add_wave` / `edit_wave` / `remove_wave` / `get_waves` / `set_sequence` | 精确控制单波的频率/强度/偏振/包络 |
| **L0 原始级** | 少用 | `set_waves`（整表替换，收 serialize 同构 list） | 存档恢复、"我知道我在干什么"的批量导入 |

**三层都经同一个 choke point 写 `dynamic.waves_<atom>`**，区别只在"AI 要表达多少细节"。
L0 保留是为了兼容 `serialize_waves` 存档；**默认引导 AI 用 L2/L1**。

---

## 2. 分层：底层存物理量（最大自由度）+ 辅助层给捷径（得心应手）

**裁决（已定，对应上轮判断点 #1/#2）**：底层**只存物理量**——`freq_mhz`、`intensity_mwcm2`
（就是引擎结算吃的那两个值）；"共振频率/失谐/Rabi/π 时长"都是**辅助层的按需换算**，
算完**写回物理量**，辅助本身**不参与结算、不进快照真值**。这与旧 GUI 的波形辅助区
**完全同构**（见 §2.6 复用表）：既保留"直接填物理量"的最大自由度，又给"按跃迁/Rabi 表达"的捷径。

> 对齐纲领：底层物理量 = 表达自由度拉满（AI 可直接设任意 freq/intensity）；辅助捷径 +
> `wave_list` 校验 = 契约锁死。二者写的是**同一份物理量快照**（不引入第二真源）。

### 2.1 辅助①「定频率」：freq = 裸能级差 + Δ（复用旧 GUI `_set_freq`）
```
freq_mhz = |E_mhz[tgt] − E_mhz[src]| + detuning_mhz
```
- 依据：`field_build.py:229` 的共振判据 `det = |ΔE_mhz| − omega_L_mhz`。
- `src`/`tgt` 用**身份名/label**（如 `'60S_{1/2} m_J=+0.5'` 或组名 `'60S'`），**不用索引**
  （§2.2）；解析走 `atomic_levels.find_state_index`，**不在基组就报错**（绝不静默）。
- **失谐相对裸能级差**（`E_mhz` 含超精细+Zeeman+DC-Stark，但**不含动态 AC-Stark 能移**）
  ——旧 GUI tooltip 明确如此、"无歧义"，本层照搬。
- **存的是绝对 `freq_mhz`**：改磁场/静电场后 `E_mhz` 变、但**已存载频不会自动更新**（这就是
  "底层存物理量"的代价）；辅助层可用 `drive` 记录的跃迁**一键重算**（re-drive），但真值始终
  是那个绝对 `freq_mhz`。
- 默认载频建议复用旧 `_suggest_freq(pol)`：该偏振下**偶极最强的跨流形**跃迁——**刻意排除
  同流形**（Zeeman/超精细劈裂几十 MHz，选它=静默坑）。

### 2.2 辅助②「定光强」：目标 Rabi → 强度（core 已有逆换算，直接反解）
```
d = dipole_amplitude(cache, g, e, q)                  # 该偏振 q 的偶极矩阵元
intensity_mwcm2 = intensity_mwcm2_from_rabi(|d|, Ω_target)   # core 已有逆换算，不绕平方律
```
- **core 里双向换算都已存在**：`rabi_mhz(d,E0²)`（强度→Rabi）与 `intensity_mwcm2_from_rabi(d,Ω)`
  （Rabi→强度）——辅助层**直接调**，不重写物理。
- **选择定则**（旧 GUI 同款）：`|d| < 1e-12` → "该对在所选偏振下无耦合（Δm 不匹配）"，
  **当场报错**，不静默建一条零效果波。AI 给 `rabi_mhz`（捷径）或直接给 `intensity_mwcm2`（自由度）。

### 2.3 脉冲时长由 Rabi 算（π/2π 脉冲一键）
```
布居 ∝ sin²(π·Ω_MHz·t_µs)  ⇒  t_π(ns) = 500/Ω_MHz ，  t_{π/2}(ns) = 250/Ω_MHz
```
- 依据：`build_fields_exact` docstring 的周频约定（"π 脉冲 = 1/(2Ω) µs"）。
- 方波平台长 = `t_π − 2·edge_ns`（扣除两个缓变沿）。

### 2.4 存意图不存真值：复用既有 `drive` 字段（不引入新概念 = 坐实不套层）
- `Wave.drive = {'g': g_spec, 'e': e_spec}`（`serialize_waves`/`restore_waves` 已保留）。
- 辅助层把 g/e 跃迁写进 `drive` ⇒ 存档自带"这条波辅助对着哪个跃迁"，但 **`freq_mhz`/
  `intensity_mwcm2` 才是结算真值**（drive 不参与结算，正如旧 GUI 辅助区"不参与结算"）。
- 改磁场/静电场后，`drive` 在则可提示"裸差已变、载频建议重算"（辅助重跑，**不自动改真值**）。

### 2.5 辅助④「实时反馈」= 场的 claim-vs-measured（M2）
- 镜像旧 GUI `_real_feedback`：给一条波的**当前物理量**，反算并报出
  `Rabi = rabi_mhz(|d|, field_e0_sq(I))`、`Δ = freq − 裸差`。
- 用途：AI 直接填了 freq/intensity（走自由度、没用捷径）后，`describe_drive(wave)` 仍能
  告诉它"这实际是 Ω=? Δ=?"——**透明可核对**，是场版本的声称 vs 实测。

### 2.6 旧 GUI 辅助 → AI 辅助 复用映射（同一批 core 函数，I5 人机同源）

| 旧 GUI（给人，`ui_waves.py`） | 调的 core 函数 | AI 辅助（本层，无 Qt 纯函数） |
|---|---|---|
| g/e 下拉 + `_pair()` | `find_state_index` | `set_drive(g,e,...)` 的身份解析 |
| 「定频率」`_set_freq`（裸差+Δ） | `E_mhz` 差 | `freq_for(g,e,detuning)` |
| 「定光强」`_set_intensity` | `dipole_amplitude`+`intensity_mwcm2_from_rabi` | `intensity_for_rabi(g,e,pol,Ω)` |
| 实时反馈 `_real_feedback` | `rabi_mhz`+`field_e0_sq` | `describe_drive(wave)`（读回 Ω/Δ） |
| `_suggest_freq(pol)` | `dipole_amplitude`（跨流形最强） | `suggest_freq(pol)`（默认载频建议） |
| `fields.level_shift_at_intensity` | AC-Stark 扫描 | `intensity_for_ac_shift(level,target)`（可选进阶） |

**结论**：旧 GUI 辅助区就是一套"物理量 ⇄ 跃迁语义"的双向换算器，只是绑在 Qt 控件上。
本层把同一批 core 函数包成**无 Qt 的纯函数辅助**，人和 AI 共用同一套换算（I5 同源）。

---

## 3. 契约与校验：把 `type='list'` 换成 `wave_list`

registry 新增类型 **`wave_list`**，`validate` 逐波检查（**在 `set()` 当场**，I6/D5）：

| 字段 | 规则 | 非法时 |
|---|---|---|
| `env[0]` / `kind` | ∈ `{gauss, square, file, arb}` | InvalidValue + suggest 合法集 |
| `pol` | ∈ `POLS = ['σ+','π','σ-','椭圆偏']` | InvalidValue + suggest |
| `elliptical` | `pol=='椭圆偏'` 时**必须**带 `{alpha_deg,theta_deg,eps_deg}` | InvalidValue（缺参不给静默按 π 算） |
| `freq_mhz`/`intensity` | 数值、> 0（intensity）；`arb` 无载频→`freq_mhz` 忽略 | InvalidValue |
| gauss | 需 `(center_ns, sigma_ns>0, peak∈(0,1])` | InvalidValue |
| square | 需 `(start_ns, edge_ns≥0, height∈(0,1])`，`fall_start` 可缺省 | InvalidValue |
| file/arb | `t_ns`/`amp` 等长、严格递增、≥2 点，**或**给 `env_path` 指针 | InvalidValue |

错误体走 `MechaError.to_dict()`（kind/hint/suggest，D2）——**第一次失败就能改对**。

**引擎支持面提示**（不静默，D5）：`arb`（无载频）**只 hard 支持**；`build_fields_exact`
（sambe/mixed 的 exact 场）遇 arb 会 raise。构造 `arb` 时回执**主动提示**"将改道 hard"
（对齐 `runspec.effective_method` 的既有改道规则）。

---

## 4. 大数组走指针（I3；根治旧 #2/#3 撑爆回程）

- **写**：`add_wave(kind='file'|'arb', env_path='...')` 或 `env=[t,amp]`——若数组超阈值，
  构造器把它落 `.mecha/payloads/<sha1>.npz`，**快照里只存指针** `{path,sha1,shape,dtype}`
  （复用 `bus.payload_pointer`）。同 sha1 二次写入零成本命中（内容寻址）。
- **读**：`get_waves(atom)` 每波只回**摘要**：
  ```
  {name, kind, freq_mhz, pol, intensity_mwcm2, offset_ns,
   env: gauss→{center,sigma,peak} | square→{start,edge,height,fall_start}
        | file/arb→{N, dt_min_ns, span_ns, f_max_mhz, $pointer}}
  ```
  **绝不内联 12001 点**；要曲线走 `get_envelope(atom, name)` 单独取（并过体积闸）。
- **带宽预检**（D7）：file/arb 回执带 `f_max_mhz`（`waveforms._f_max_mhz`：样品 Nyquist vs
  引擎查表 Nyquist 取小）+ 一句"当前 t_max/npts 能否表达"，别让 AI 提交不可表达的波形。

---

## 5. 频繁调节的人机工程（"得心应手"的具体落点）

1. **句柄 + 增量 patch**（harvest）：`edit_wave(atom, name|index, patch={'intensity_mwcm2': 5.0})`
   —— 改一个参数只发一个 patch，**不重发整表**；仍是一条 `set` 事件、可 undo。
2. **命名寻址**：波按 `name` 寻址（`edit_wave(atom,'pump',{...})`），不用记索引。
3. **一次表达多波**：`set_sequence(atom, [spec1, spec2, ...])` 声明式建整套
   （如 STIRAP 的 pump+Stokes、Ramsey 的两个 π/2），**一条事件**、原子写入（全成或全不成）。
4. **命名脉冲协议**（L2 预设，最高频）：
   - `pi_pulse(atom, transition, pol, rabi_mhz, ...)` / `pi2_pulse(...)`（时长自动 §2.3）
   - `ramsey(atom, transition, sep_ns, ...)`（两个 π/2 + 间隔）
   - `stirap(atom, pump, stokes, ...)`（counter-intuitive 时序）
   - `arp(atom, transition, chirp_mhz, ...)`（啁啾扫频，走 arb 或 file）
   - `square_mw(atom, freq_mhz, dur_ns, ...)`（纯微波方波，最常见）
5. **建完即验**：每个构造器回执附**轻量 preview 摘要**（driven transitions + det + rabi +
   `basis_enough`），或提示"调 `preview()` 看这条驱动了什么"（D7 + M2）。
6. **可教学默认**：`pol='π'`、`offset_ns=0`、`intensity`/`rabi` 缺一个则 hint 提示二选一；
   缺包络参数时 hint 列出该 kind 需要哪些。
7. **渐进披露**：`describe('dynamic.waves_single')` 才给完整 envelope schema；列表视图只给摘要。

---

## 6. 事件 / 可逆 / 无侧门（框架一致）

- 每个构造器最终 = 一次 `session.set('dynamic.waves_<atom>', new_list, actor, reason)`
  → 一条 `kind='set'` 事件；`before/after` 过体积闸（大数组指针化）；`reversibility='undo'`。
- **无侧门**（I1）：构造器**只能经 `session.set`** 改 waves；`mecha_kernel_selfcheck` 的
  AST 守卫已覆盖 `_snapshot`/`_write`，fields 不得开新写口。
- **CodeAct 门面**（D1）：新增 `fields` facade，只转发上述公开方法；`NOT_EXPOSED` 补
  `waveforms`/`Wave`/`ArbitraryWave`/`GaussianEnv`… 裸类（AI 不手搓对象，走构造器）。

---

## 7. MCP / CodeAct 暴露策略（工具面不膨胀）

他们自豪"工具面仅 7 个"，故**克制**：
- **MCP 只加 2 个高频工具**：`set_drive(...)`（L2 主力）+ `get_waves(atom)`（读回）。
- **其余走 CodeAct 的 `fields` 门面**：`add_wave/edit_wave/remove_wave/set_sequence/
  pi_pulse/ramsey/stirap/arp/square_mw/set_waves/get_envelope`——组合自由拉满、契约锁死。
- 理由：多波编排/协议预设本就适合"一段脚本"，塞进 MCP 工具面会让工具数爆炸（渐进披露原则）。

---

## 8. 判据 `fields_selfcheck`（带致死性，D4）

| 包 | 断言 | 致死性 |
|---|---|---|
| **换算对拍** | `set_drive(60S→60P, detuning=0)` 的 `freq_mhz` **逐位** == `|ΔE_mhz|` | 故意用错能级→必须不等 |
| **Rabi 反推** | 给 `rabi_mhz=Ω`，`collect_transitions` 复算该波 == Ω（容差内） | 平方律写错→超差 |
| **π 时长** | `t_π == 500/Ω_MHz`（ns） | — |
| **校验致死** | `pol='x'` / gauss 缺 `sigma` / `freq` 非数值 → **set 当场** InvalidValue | 证明不是 build_run 才炸 |
| **指针化** | 设 12001 点 file 包络：事件/回执**无内联数组**、只含 sha1 指针；回程字节 < 阈值 | 灌爆→截断报 kept/total/where（I4） |
| **patch 隔离** | `edit_wave` 改一字段，其余波/字段**逐位不变** | — |
| **preview 联动** | `set_drive` 后 `preview` 报该跃迁 driven 且 `basis_enough` 正确 | 缺中间态→`missing_from_basis` 非空 |
| **不套层** | 构造器产出的存档经 `restore_waves` **逐位 ==** 手搓 `serialize_waves` 同构结果 | 证明是同一数据模型，非并行表示（§13） |
| **arb 改道** | 建 `arb` 波→回执提示"改道 hard"；在 sambe/mixed 下与 `effective_method` 一致 | — |

---

## 9. 不套层声明（§13 自查）

- **单一真值**：波形的唯一真值仍是快照里 `serialize_waves` 同构的 dict；构造器只"正确拼它 +
  校验它 + 记一条事件"，**不引入并行表示**（判据"不套层"包逐位对拍 `restore_waves`）。
- **core 不改**：只调用 `waveforms`（Wave/包络/serialize）、`static_atom`（collect_transitions/
  E_mhz）、`field_build`（rabi 约定）、`atomic_levels`（身份解析）既有函数。
- **依赖方向**：`fields → core` 单向；`core ↛ mecha` 不破。
- **属适配层**（§11）：物理专属、不可提炼进本体——落在 Re0，天经地义。

---

## 10. 施工顺序（建议）

1. registry 加 `wave_list` 类型 + 校验（§3）——先把橡皮图章换成真契约，**独立可测**。
2. `fields.py`：`get_waves`/`set_waves`(L0) + `add_wave`/`edit_wave`/`remove_wave`(L1)，
   全经 choke point；含指针化（§4）。
3. 语义换算（§2）：`set_drive` + `pi_pulse`/`square_mw`（L2 高频两件）。
4. 协议预设（§5.4）：`ramsey`/`stirap`/`arp` + `set_sequence`。
5. CodeAct `fields` 门面 + MCP 加 `set_drive`/`get_waves`（§7）。
6. `fields_selfcheck`（§8）逐包补，每步跑基线常绿。

> 每步一个可回滚提交；先跑 `run_checks.py` 绿了再动手（README 改造纪律）。
