# 原子能级计算器（Rb-87 / 通用碱金属）

一个面向里德堡原子实验的**原子能级 + 寻址光 AC-Stark + 跃迁频谱**计算与可视化工具。界面可直接交互：实时调整磁场 / 寻址光参数，直观看到能级位移、超精细结构、谱线变化，并通过点击谱峰定位跃迁对。

> 本项目以 Rb-87 为默认原子，但物理核心与 ARC 库解耦，**可切换到任意 ARC 支持的碱金属原子**（Cs、Li、Na、K…）。

---

## 一、功能总览

| 能力 | 说明 |
|---|---|
| 能级结构 | fine 结构 + 超精细 + Zeeman（B 场任意），数值对角化 |
| AC-Stark | 近共振 dressed 精确解 + 远失谐非 RWA SOS，失谐精确到子能级（含超精细/Zeeman） |
| 跃迁频谱 | 全部偶极允许跃迁，爱因斯坦 A 系数强度，偏振分辨 |
| 镜像频谱对比 | 一块画布：off 谱向上、on 谱向下，共用一条频率轴——共振的线上下对齐，非共振错开，肉眼发现交换相互作用 |
| 点击定位 | 点击谱峰 → 高亮 + 峰表定位 + 能级图跃迁连线 |
| 寻址光参数 | 直接输入频率 / 或"选中谱线 + 失谐"确定频率（目标跃迁对模式） |
| 回归自检 | `core_selfcheck.py` 44 项：手写算法与 ARC 官方函数逐项对拍 |

---

## 二、安装与运行

**环境**：Python 3.10+（实测 3.10）。**依赖**：

```
numpy  scipy  matplotlib  PyQt5  arc
```

（`arc` 为 [ARC — Alkaline Rydberg Calculator](https://arc-alkali-rydberg-calculator.readthedocs.io/) 库，提供碱金属量子亏损、波函数、偶极矩阵元等。）

**运行**：

```bash
python 能级.py          # 启动 GUI
python core_selfcheck.py  # 运行回归自检（44 项，全过退出码 0）
```

> Windows 环境如 `python` 解析异常，用绝对路径：`& "D:\python310\python.exe" 能级.py`

**文件结构**：

```
core.py            # 物理核心（无 UI 依赖，可单独审查）
能级.py            # UI 入口（PyQt5 + matplotlib，直接渲染 core 输出）
core_selfcheck.py  # 回归自检：手写算法 vs ARC 官方函数
core_ac_directcheck.py  # AC-Stark 直接哈密顿量对角化交叉验证（可选 torch-GPU）
# 其余 .py / .svg / .pptx 为早期独立实验脚本与资料，未纳入当前架构
```

---

## 三、使用说明书

### 3.1 界面布局

- **左侧画布**（自上而下）：
  - 能级图：横轴为子能级（m_J / m_F），纵轴为能量；受寻址光影响的子能级画"移前灰线 + 箭头 + 移后粗线"并标注位移。
    **显示模式可切换**（右侧"能级图"下拉）：`单一`（一张图，off/on 跃迁连线叠加）/
    `off/on 双图`（非寻址光裸能级 + 寻址光移后各一张并排，分别标注对应频谱选中的跃迁连线）
  - **镜像频谱**：off 谱（无寻址光）向上、on 谱（有寻址光）向下，共用一条频率轴；根部在 y=0 对齐。
    选中峰不再在图上标频率，改由右侧面板一行显示（off 峰频率 | on 峰频率 | 频率差）
- **右侧控制面板**：**磁场（单独一行）** / 寻址光 / **频移计算后端** / **光强扫描** / 添加能级组 / 显示范围 / 参考线 / 谱横轴范围
  - **磁场 B (G)**：单独一行（回车生效）
  - **频移后端**（address 光/偏振 同行下拉框）：`矩阵对角化`（默认，Floquet 直接对角化，
    自洽含近共振 dressed + 远失谐反旋项 + 多态耦合，精度最高）/ `解析/微扰`
    （逐伙伴 RWA dressed + 远失谐 SOS，速度最快）。切换即重算全部能级/谱线。
  - **偏振**：σ⁺ / π / σ⁻ 标准偏振，或 **任意椭偏**（选中后显示 α°/θ°/ε° 三个参数）——
    支持任意完全偏振光的 AC 位移：α=传播方向与 z 轴夹角（决定 π 分量占比）、
    θ=偏振椭圆主轴方位角、ε=椭偏角（0=线偏，±45°=圆偏）
  - **光强扫描**（光强行下方）：目标频移 (MHz) + 作用侧（目标侧 To / 源侧 From）+
    "扫描光强"按钮——固定 B/偏振/频率，对数扫描光强并迭代，使目标子能级频移达到设定值
- **底部两个峰表**：off 峰表（无寻址光谱线）、on 峰表（有寻址光谱线），列 = 频率 / From / To / 偏振

### 3.2 基本流程

1. **添加能级组**：右侧输入 `n / l / j`（如 60/0/0.5 添加 60S），点"添加"；重复可添加多个组（6P、60P…）
2. **设置磁场**：`B (G)` 输入框（回车生效），如 21.7 G
3. **打开寻址光**：勾选"address 光"，设光强（mW/cm²，可拖滑块）、偏振（σ⁺/π/σ⁻，或"任意椭偏"用 α/θ/ε 描述）
4. **确定频率**（两种方式）：
   - 直接输入 `频率 (GHz)`
   - 或：在下方峰表选中一条谱线 → 设"失谐 Δ (MHz)" → 点"**获取自能谱**"（进入目标跃迁对模式）
5. **观察**：能级图看 AC 位移；镜像频谱看哪些线被共振影响（上下对齐 = 共振）
6. **点击谱峰**：在镜像频谱点击一条线 → 该峰高亮、对应峰表滚动定位、能级图上画出跃迁连线；点击空白清除
7. **双峰对比**：分别点击 off/on 谱各一个峰 → 右侧一行显示 off/on 峰频率与频率差
8. **光强扫描**：峰表选中谱线 → 设目标频移与作用侧 → 点"扫描光强"，得到目标频移所需光强

### 3.3 寻址光两种"确定频率"模式（重要）

| 模式 | 进入方式 | 显示 | 失谐组件 |
|---|---|---|---|
| **直接频率** | 手动改"频率 (GHz)" | **最近跃迁对**（core 按失谐最小选子能级对） | 红边框（Δ 不生效）|
| **目标跃迁对** | 峰表选中 + "获取自能谱" | **目标跃迁对**（所选谱线，失谐以组件为准） | 正常（回车即按 Δ 重算频率）|

**记忆规则**：失谐框**红边框** = 当前是直接频率模式，Δ 无效；正常样式 = 目标跃迁对模式，Δ 生效。改频率框会回到直接频率模式。

### 3.4 其他控件

- **能级图模式**（相对显示 同行）：`单一` / `off/on 双图`——切换能级图画布显示模式
- **目标频移 + 扫描光强**：先在峰表选中谱线 → 设目标频移(MHz)与作用侧(From/To) → 点"扫描光强"，
  软件保持 B/偏振/频率不变，对数扫描光强（1e-6~1e9 mW/cm²）并迭代，把满足目标频移的光强写入光强框
  （扫描统一用矩阵对角化后端，多态自洽、结果更准；强驱动非单调时提示最接近点）
- **双峰频率面板**：镜像频谱选中 off/on 峰后，右侧一行显示 off 峰频率 | on 峰频率 | Δ(±MHz)
- **相对显示**：能级图能量相对最低子能级
- **y 下限/上限**：按能级组节点裁剪能级图视野（含能移前/后）
- **参考频率**：镜像频谱竖虚线（默认 17.2368 GHz，清空不画）
- **谱x 下限/上限**：频谱横轴范围（空 = 自动，两图自动一致）
- **能级数值表**：各子能级能量与 AC 位移（相对最低子能级）
- **能级组列表**：右键删除组；图例只显示视野内组

### 3.5 常见使用提示

- 想观察"某对能级在共振下的大位移"：**偏振方向决定哪侧共振**。例：σ⁻ 下 6P3/2 F=3 mF=3 ↔ 60S mJ=0.5 是 6P3/2 侧大位移；要让 60S mJ=+0.5 大位移请用 σ⁺（60S 0.5 →σ⁺→ 6P3/2 1.5）。
- B=0 时超精细 F 内各 mF 简并，无法从频率区分同 F 不同 mF 的跃迁；B≠0（默认 21.7 G）才可区分。
- 频率框为 3 位小数 GHz（约 1 MHz 精度），亚 MHz 失谐会被舍入。

---

## 四、开发者文档

### 4.1 架构与设计原则

```
          core.py（纯物理，无 Qt/matplotlib）
            │  返回行 dict / 谱线元组 / 跃迁对元组
            ▼
         能级.py（UI：PyQt5 + matplotlib）
            │  直接渲染 core 输出字段，不做二次换算
            ▼
       画布 / 峰表 / 交互
```

**核心原则**：UI 只消费 core 返回的数据结构，**不重复任何物理公式**——"结果即所见，所见即结果"，从源头避免算一遍、再换算一遍的 bug。数据流全部经 `ArcCache` 缓存（能级能量、Landé g、超精细常数、偶极矩阵元），磁场/频率/视野变化不触发 ARC 重算，滑块实时刷新。

### 4.2 `core.py` API 参考

**常量**

| 常量 | 值 | 含义 |
|---|---|---|
| `EV_TO_HZ` | e/h | ARC 能量（eV）→ Hz |
| `MU_B_MHZ_PER_G` | μB/h/1e4/1e6 | MHz/G |
| `RYDBERG_N_MIN` | 20 | n≥20 视为里德堡（忽略核自旋）|
| `AC_FAR_DETUNE_CUTOFF_HZ` | 5e9 | 近共振/远失谐分流阈值 |
| `SOS_BASIS_N_MIN/MAX` | 5 / 100 | SOS 基组 n 范围 |
| `POLARIZATION_Q` | σ⁺:1, π:0, σ⁻:-1 | 偏振 → q |
| `elliptical_coeffs(α,θ,ε)` | — | 任意椭偏 → 球基复系数 (c_{+1},c_0,c_{−1})，Σ\|c\|²=1 |
| `address_pol_coeffs(cfg)` | — | cfg 有 'elliptical' 用椭偏系数，否则单 q 系数（所有物理量据此叠加）|
| `SHIFT_BACKEND` | 'direct' | 频移后端：'direct'（矩阵对角化，默认）/ 'legacy'（解析/微扰）|
| `DIRECT_SOS_N_WINDOW` | 3 | 直接后端自动纳入每个已添加组偶极伙伴的 \|Δn\| 窗口 |

**`ArcCache(atom)`**：`atom` 为 ARC 原子实例（如 `Rubidium87()`）。
- `E_mhz(n,l,j)` / `gJ(l,j)` / `hfs(n,l,j)` / `dme(...)`：缓存态能量（MHz）、Landé g、超精细 A/B（Hz）、偶极矩阵元（原子单位）
- `I` / `gI`：核自旋与核 g 因子，**取自原子实例**（`atom.I` / `atom.gI`），换原子无需改代码

**能级**
- `get_hyperfine_levels(cache, n, l, j, B)`：|mJ,mI⟩ 基下按 mF 分块对角化 `H = A·I·J + B·Q + μB·B(gJ·Jz + gI·Iz)`，返回 `(F, mF, E_mhz, mJ_weights, mJ_amps)`；里德堡返回 None
- `build_level_rows(cache, field_B, groups, address_cfg, visible, backend=None)`：**UI 直接绘制的行数据**（x/E0/E/ac/affected/label/qstr/n/l/j/kind/mj 或 mF/amps）；`backend` 默认取 `SHIFT_BACKEND`，可选 'direct' / 'legacy'，两后端行结构一致

**AC-Stark**
- `ac_stark_shift_mhz(cache, group, m, groups, address_cfg, B_gauss=0, E_sub=None)`：子能级总位移（legacy 后端）
- `direct_ac_stark_shifts(cache, groups, address_cfg, B_gauss=0, n_photon=1, sos_n_window=None)`：**direct 后端**——一次 Floquet 矩阵对角化得全部子能级位移（numpy，无需 GPU）。基组 = 已添加组 + 各组偶极伙伴（|Δn|≤窗口），耦合矩阵只依赖（基组,B,强度,偏振）不依赖频率，已做模块级缓存，频率变化只重对角化。**远失谐里德堡↔里德堡耦合按频率屏蔽**（|ω₀−ω|>ω₀ 的对净位移≈0 但会在光子块边界产生虚假 ~Ω²/ω 位移，屏蔽后不影响亚 MHz 精度；近共振对如 60S↔60P3/2 保留）
- `pair_contribution_mhz(...)`：单个伙伴贡献，按失谐分流近共振 dressed / 远失谐 SOS。**双通道**：fwd（源为低态吸收，m2=m+q）+ rev（源为高态发射，m2=m−q，dme 参数交换），仅保留旋转波对，跳过反旋转波对（偏振不匹配，否则 q=0 时双通道重复计数）
- 失谐一律用**子能级能量差**（fine + 超精细 + Zeeman），与 `collect_transitions` 显示完全一致

**跃迁与频谱**
- `collect_transitions(cache, groups, address_cfg, field_B)` → `(label_src, qstr_src, label_tgt, qstr_tgt, det_hz, omega_hz)`：每组对**失谐最小**的耦合子能级对（超精细含 F/mF，里德堡 mJ）+ 该对失谐与 Rabi
- `spectrum_lines(cache, rows, ac_on)` → 8 元组 `(nu_hz, A, label_a, label_b, qstr_a, qstr_b, color, pol)`：全部偶极允许跃迁，爱因斯坦 A 强度
- `transition_strength_q(cache, a, b, q)`：单偏振强度，**含 mJ 匹配 / mF 守恒选择定则**

### 4.3 物理模型要点（与易错点）

1. **AC-Stark 分流**：
   - 近共振（|Δ| < max(5 GHz, Rabi)）：dressed 两能级精确解，源态位移 ½(√(Δ²+Ω²)−Δ)，共振给 Autler-Townes
   - 远失谐：非 RWA 二阶项 `−(E0²/4ħ)·Σ|d|²·2ω_ki/(ω_ki²−ω²)`（rotating+counter-rotating，即 ARC `DynamicPolarizability`）。**RWA 在远失谐区失效**，只留 rotating 会把里德堡-里德堡对位移高估 ~10⁴ 倍
2. **子能级失谐（最精细）**：当前态为超精细态时 `E_sub = fine + e_hf`；伙伴按 mJ=m+q 分量分解到各 (F,mF)，按 |振幅|²/(2I+1) 加权（核自旋平均）。**缺 (2I+1) 归一化会高估 4 倍**
3. **跃迁禁忌**：`transition_strength_q` 必须命中对方 mJ（mJ-mJ 需 m2==b.mj 等），且 F-F 需 mF 守恒——否则会把"到其他 mJ 的耦合"误归到本对
4. **偏振方向**：同一对能级，σ⁻ 驱动低能态吸收、σ⁺ 驱动高能态吸收，两者共振侧不同（见 3.5）
5. **ARC 约定**：`getTransitionRate` 用 **asymmetric** 约化矩阵元，我们的 Σ_A 与它恒差因子 2（selfcheck T6 已固化）

### 4.4 单位约定

| 量 | 单位 |
|---|---|
| 能级 / AC 位移 | MHz |
| 跃迁 / 寻址光频率 | Hz（UI 显示 GHz）|
| 偶极矩阵元 | 原子单位（a.u.，ARC 返回）|
| 光强 | mW/cm²（→W/m² 因子 10）|
| 磁场 | G |

### 4.5 换原子指南

`能级.py` 中 `MainWindow` 实例化原子的位置（`self.atom = Rubidium87()`）换成目标原子实例即可，例如：
`Caesium()`（Cs-133）、`Rubidium85()`、`Lithium6/7()`、`Sodium()`、`Potassium39/40/41()`。
- 核自旋 I / gI 自动从 `atom.I` / `atom.gI` 读取
- **注意**：Li/Na/K 的 `gI = 0.0` 是 ARC 数据限制（未提供真实核 g 因子），Zeeman 核项会缺失，需自行补充
- `RYDBERG_N_MIN`、`SOS_BASIS_N_MIN/MAX` 为通用物理阈值，通常无需改

### 4.6 测试与验证（`core_selfcheck.py`）

44 项断言，手写算法与 ARC 官方函数逐项对拍（T7/T9 显式用 legacy 后端；
T11 校验 direct 与 legacy 双后端在低/中强度下亚 MHz 一致）：

| 测试 | 内容 |
|---|---|
| T1 | 核自旋 I/gI（Rb87、Cs133 原子钟频率）|
| T2 | J=1/2 超精细能量 vs `breitRabi` + `getHFSEnergyShift` |
| T3 | J=3/2（含 B 四极项）vs `getHFSEnergyShift`（breitRabi 对 J>1/2 近似，差 2.5 MHz）|
| T4 | 5S×6P3/2 全部 384 项超精细矩阵元 vs `getDipoleMatrixElementHFS` |
| T5 | Σ\|d\|² vs `getReducedMatrixElementJ`²（比值=1）|
| T6 | 爱因斯坦 A 逐条 vs 标准公式；Σ_A/getTransitionRate=2（ARC 约定）|
| T7 | 基本功能回归 + collect_transitions 6 元组/失谐最小 |
| T8 | 远失谐 SOS vs `DynamicPolarizability`（j=1/2 比值恒 −0.5）|
| T9 | 子能级共振 → AC=Ω/2，且与 collect_transitions Rabi 一致 |
| T10 | F↔mJ 矩阵元 vs `getDipoleMatrixElementHFStoFS`（含禁忌=0）|| T11 | 双后端：direct 矩阵对角化 vs legacy 解析/微扰（行数 + 亚 MHz 位移一致） |
修改 `core.py` 后请运行 `python core_selfcheck.py` 确认无回归。
**AC-Stark 直接对角化交叉验证（`core_ac_directcheck.py`）**：

core 的 AC-Stark 是"逐伙伴分流"（近共振 RWA dressed + 远失谐 SOS + 核自旋平均），
`core_ac_directcheck.py` 构造含光子数的 Floquet 哈密顿量并直接数值对角化，
一次自洽包含 dressed + SOS（含反旋项）+ 多态耦合，作为独立"地面真值"对拍：

```bash
python core_ac_directcheck.py          # 全量（大基组自动用 torch-GPU 并行）
python core_ac_directcheck.py --quick  # 只跑快速场景
python core_ac_directcheck.py --no-torch
```

已验证/可发现的问题：
- S1（Rydberg–Rydberg 全基组，近/远失谐 + 邻态共振）：core 与直接法高度一致
- S3（6P3/2↔60S，σ⁻）：core 曾漏掉"上态"共振耦合（`pair_contribution_mhz` 只用
  m2=m+q 的吸收方向，60S m_J=+0.5 作为高态时漏掉 σ⁻ 与 6P F=3mF=3 的共振，
  位移低估十余倍）——**已修复**（新增 rev 发射方向通道，见 4.2）
- direct 曾受 3 光子块 Floquet 边界截断误差影响：远失谐里德堡↔里德堡耦合
  （60S↔60P ~100GHz）在光子块边界只剩旋转波方向 → 虚假 ~Ω²/2ω 位移破坏
  6P↔60S 共振（6P 报告 ~0）——**已修复**（按频率屏蔽远失谐里德堡对）
- **强驱动（Ω 与超精细劈裂可比，如 I~10⁷ mW/cm²、6P↔60S 位移 ~20MHz）**：
  direct 捕捉到 {6P F3mF3↔60S+0.5} 与 {6P F3mF2↔60S−0.5} 两个近简并 AT 对的
  多态杂化（更可靠）；legacy 逐对独立模型忽略多态相干 → 符号/大小均不可靠。
  亚 MHz 精度请以 direct 为默认；强驱动时两法都不能保证亚 MHz
- S1c：强驱动下 RWA 反旋项（Bloch-Siegert）随 Ω/ω 增大，超出容差需用直接法
- S5：F–F 相干 vs 核自旋平均模型差异（本场景测得很小，视通道而定）
### 4.7 pairinteraction 独立校验（对比脚本保留在仓库）

已用专业软件 [pairinteraction](https://github.com/pairinteraction/pairinteraction)（v2.5.0，源在 `../pairinteraction`，`pip install pairinteraction` 已装）对本项目做独立对拍：

- **覆盖度**：pairinteraction 覆盖 能级（fine/mJ 级）、偶极矩阵元、DC-Stark、pair potential、lifetime；但 **没有 AC-Stark/Floquet**（仅 DC 电场），且 **Rb SQDT 模型忽略超精细**（仅 MQDT 物种如 Sr87/Yb 含超精细），也没有爱因斯坦 A 直接 API。
- **数值对比**（`compare_pairinteraction.py` / `compare_pairinteraction2.py`）：
  | 量 | 本项目(ARC) | pairinteraction | 差异 |
  |---|---|---|---|
  | fine 跃迁频率（60S↔60P 等） | — | — | 完全一致（0 GHz） |
  | 6P3/2↔60S 频率 | 295723.72 GHz | 295728.00 GHz | 0.0014%（量子缺陷来源略不同） |
  | 偶极矩阵元（Ryd↔Ryd） | — | — | 0.002% |
  | 偶极矩阵元（低↔Ryd / 低↔低） | — | — | 0.75% / 0.9% |
  | 60S Zeeman @21.7G | 60.814145 MHz | 60.814141 MHz | 4 Hz |
  | 远失谐 SOS(ω→0) | −33.83 kHz | DC-Stark/2 = −33.89 kHz | 0.2% |
- **SOS↔DC 等价原理**：本项目远失谐 SOS 项 `−(E0²/4ħ)Σ|d|²·2ω_ki/(ω_ki²−ω²)` 在 ω→0 时退化为 `½·DC Stark`（同 E0）。上表最后一行即用 pairinteraction 的 DC-Stark 验证本项目 SOS 公式 + 偶极矩阵元 + 能量分母。
- **结论**：pairinteraction **不能替代**本项目的 AC-Stark（含超精细）与跃迁频谱任务，但可作为 fine 级能级、偶极矩阵元、远失谐 SOS 的独立校验基准，结果与专业软件一致到 0.2% 以内。

### 4.8 维护注意（历史踩坑）

- 远失谐 SOS 曾缺 ħ 因子（低估 ~10³⁴ 倍）；dressed 曾用 fine 频率（未含超精细能移）；`transition_strength_q` 曾缺 mJ 匹配导致非物理线——均已修复并固化为 T6/T8/T9/T10
- 避免用 `c` 作循环变量（遮蔽光速）导致 ComplexWarning
- `pair_shift_mhz` 已废弃，统一用 `_dressed_shift_mhz` + 子能级失谐
- 测试 offscreen 需 `os.environ['QT_QPA_PLATFORM']='offscreen'`；stderr 有大量 QFontDatabase 警告，过滤即可

---

*架构演进：单一 `能级.py` → 物理/UI 两文件（core.py + 能级.py）→ 回归自检（core_selfcheck.py）→ 交互增强（点击定位 / 镜像频谱 / 目标跃迁对模式）→ 双频移后端（direct 矩阵对角化 / legacy 解析微扰，UI 可切换）。*
