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
| 回归自检 | `core_selfcheck.py` 38 项：手写算法与 ARC 官方函数逐项对拍 |

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
python core_selfcheck.py  # 运行回归自检（38 项，全过退出码 0）
```

> Windows 环境如 `python` 解析异常，用绝对路径：`& "D:\python310\python.exe" 能级.py`

**文件结构**：

```
core.py            # 物理核心（无 UI 依赖，可单独审查）
能级.py            # UI 入口（PyQt5 + matplotlib，直接渲染 core 输出）
core_selfcheck.py  # 回归自检：手写算法 vs ARC 官方函数
# 其余 .py / .svg / .pptx 为早期独立实验脚本与资料，未纳入当前架构
```

---

## 三、使用说明书

### 3.1 界面布局

- **左侧画布**（自上而下）：
  - 能级图：横轴为子能级（m_J / m_F），纵轴为能量；受寻址光影响的子能级画"移前灰线 + 箭头 + 移后粗线"并标注位移
  - **镜像频谱**：off 谱（无寻址光）向上、on 谱（有寻址光）向下，共用一条频率轴；根部在 y=0 对齐
- **右侧控制面板**：磁场 / 寻址光 / 添加能级组 / 显示范围 / 参考线 / 谱横轴范围
- **底部两个峰表**：off 峰表（无寻址光谱线）、on 峰表（有寻址光谱线），列 = 频率 / From / To / 偏振

### 3.2 基本流程

1. **添加能级组**：右侧输入 `n / l / j`（如 60/0/0.5 添加 60S），点"添加"；重复可添加多个组（6P、60P…）
2. **设置磁场**：`B (G)` 输入框（回车生效），如 21.7 G
3. **打开寻址光**：勾选"address 光"，设光强（mW/cm²，可拖滑块）、偏振（σ⁺/π/σ⁻）
4. **确定频率**（两种方式）：
   - 直接输入 `频率 (GHz)`
   - 或：在下方峰表选中一条谱线 → 设"失谐 Δ (MHz)" → 点"**获取自能谱**"（进入目标跃迁对模式）
5. **观察**：能级图看 AC 位移；镜像频谱看哪些线被共振影响（上下对齐 = 共振）
6. **点击谱峰**：在镜像频谱点击一条线 → 该峰高亮、对应峰表滚动定位、能级图上画出跃迁连线；点击空白清除

### 3.3 寻址光两种"确定频率"模式（重要）

| 模式 | 进入方式 | 显示 | 失谐组件 |
|---|---|---|---|
| **直接频率** | 手动改"频率 (GHz)" | **最近跃迁对**（core 按失谐最小选子能级对） | 红边框（Δ 不生效）|
| **目标跃迁对** | 峰表选中 + "获取自能谱" | **目标跃迁对**（所选谱线，失谐以组件为准） | 正常（回车即按 Δ 重算频率）|

**记忆规则**：失谐框**红边框** = 当前是直接频率模式，Δ 无效；正常样式 = 目标跃迁对模式，Δ 生效。改频率框会回到直接频率模式。

### 3.4 其他控件

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

**`ArcCache(atom)`**：`atom` 为 ARC 原子实例（如 `Rubidium87()`）。
- `E_mhz(n,l,j)` / `gJ(l,j)` / `hfs(n,l,j)` / `dme(...)`：缓存态能量（MHz）、Landé g、超精细 A/B（Hz）、偶极矩阵元（原子单位）
- `I` / `gI`：核自旋与核 g 因子，**取自原子实例**（`atom.I` / `atom.gI`），换原子无需改代码

**能级**
- `get_hyperfine_levels(cache, n, l, j, B)`：|mJ,mI⟩ 基下按 mF 分块对角化 `H = A·I·J + B·Q + μB·B(gJ·Jz + gI·Iz)`，返回 `(F, mF, E_mhz, mJ_weights, mJ_amps)`；里德堡返回 None
- `build_level_rows(cache, field_B, groups, address_cfg, visible)`：**UI 直接绘制的行数据**（x/E0/E/ac/affected/label/qstr/n/l/j/kind/mj 或 mF/amps）

**AC-Stark**
- `ac_stark_shift_mhz(cache, group, m, groups, address_cfg, B_gauss=0, E_sub=None)`：子能级总位移
- `pair_contribution_mhz(...)`：单个伙伴贡献，按失谐分流近共振 dressed / 远失谐 SOS
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

38 项断言，手写算法与 ARC 官方函数逐项对拍：

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
| T10 | F↔mJ 矩阵元 vs `getDipoleMatrixElementHFStoFS`（含禁忌=0）|

修改 `core.py` 后请运行 `python core_selfcheck.py` 确认无回归。

### 4.7 维护注意（历史踩坑）

- 远失谐 SOS 曾缺 ħ 因子（低估 ~10³⁴ 倍）；dressed 曾用 fine 频率（未含超精细能移）；`transition_strength_q` 曾缺 mJ 匹配导致非物理线——均已修复并固化为 T6/T8/T9/T10
- 避免用 `c` 作循环变量（遮蔽光速）导致 ComplexWarning
- `pair_shift_mhz` 已废弃，统一用 `_dressed_shift_mhz` + 子能级失谐
- 测试 offscreen 需 `os.environ['QT_QPA_PLATFORM']='offscreen'`；stderr 有大量 QFontDatabase 警告，过滤即可

---

*架构演进：单一 `能级.py` → 物理/UI 两文件（core.py + 能级.py）→ 回归自检（core_selfcheck.py）→ 交互增强（点击定位 / 镜像频谱 / 目标跃迁对模式）。*
