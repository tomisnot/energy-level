# 原子能级计算器（Rb-87 / 通用碱金属）

面向里德堡原子实验的**原子能级 + 寻址光 AC-Stark + 两体耦合雷达图 + 单原子/双原子含时演化**计算与可视化工具。
界面可直接交互：实时调整磁场 / 电场 / 光场，直观看到能级位移、超精细结构、谱线变化；动态页直接配置光波形列表，模拟任意多场同时作用下的态演化（单原子 / 双原子）。

> 本项目以 Rb-87 为默认原子，但物理核心与 ARC 库解耦，**可切换到任意 ARC 支持的碱金属原子**（Cs、Li、Na、K…）。

本文档以**当前项目的分层架构为行文结构**：自顶向下（入口 → 界面层 → 核心科学计算层 → 检测与脚本层），
讲到底层 `src/core` 时**精细记录核心代码的科学计算算法**（公式、约定、常量、容差、关键修复），
供物理实现核对与二次开发。

---

## 目录

- [一、项目概览](#一项目概览)
- [二、层次架构总览](#二层次架构总览)
- [三、入口层：能级.py](#三入口层能级py)
- [四、界面层 src/ui](#四界面层-srcui)
- [五、核心科学计算层 src/core（算法详录）](#五核心科学计算层-srccore算法详录)
- [六、检测与脚本层 checks/ scripts/](#六检测与脚本层-checks-scripts)
- [七、安装与运行](#七安装与运行)
- [八、使用手册](#八使用手册)
- [九、精度、单位与约定速查](#九精度单位与约定速查)
- [十、已知限制与路线图](#十已知限制与路线图)

---

# 一、项目概览

## 1.1 这是什么

一个单窗口桌面应用（PyQt5 + matplotlib），把「里德堡原子物理计算」拆成两条可独立使用的路径：

1. **静态路径**：给定能级组（n, l, j）、磁场 B、寻址光（频率/光强/偏振），实时计算
   各子能级的 Zeeman / 超精细 / AC-Stark / DC-Stark 位移，绘制能级图与 off/on 镜像频谱，
   并在静态页底部绘制**两体耦合强度雷达图**（R × θ）。
2. **动态路径**：给定能级组与任意多组**时变光波形**（频率/光强/偏振/包络/起始偏移），
   求解含时薛定谔方程 / Lindblad 主方程，绘制各态布居随时间演化 P(t)。
   支持**单原子**与**双原子（偶极-偶极相互作用 V_dd）**两种模式。

## 1.2 功能总览

| 能力 | 说明 |
|---|---|
| 能级结构 | fine 结构 + 超精细 + Zeeman（B 场任意），数值对角化；静态/动态两页共用同一能级组 |
| AC-Stark | Floquet 矩阵对角化（`direct` 后端，第一性原理，唯一后端；legacy 逐对算法已于 2025-08 删除） |
| DC-Stark | 二阶微扰 SOS（含超精细/Zeeman 子能级），E² 标度已验证 |
| 两体耦合雷达图 | 固定 φ 扫描 R×θ，极坐标躺平热图显示 \|V_ex\|；跃迁对由 off/on 谱点击选择 |
| 光场表格 | 多光管理：频率/光强/偏振/作用范围；右键导出频率/扫描光强；椭圆偏振 |
| 失谐约定 | **正 = 蓝失谐**（ν_光 > ν_跃迁），全项目统一 |
| 单原子动态模拟 | 独立引擎（`evolve.py`）：**ip4**（相互作用绘景 RK4，默认）/ lindblad；floquet、pwc 已封存至 `src/_archive/evolve_floquet_pwc.py`（2025-08，动态分析只用 ip4），见 §5.3 |
| 双原子动态模拟 | `pair_evolve.py`：H_A(t)⊗I + I⊗H_B(t) + V_dd（secular RWA），见 §5.4 |
| 波形管理面板 | 动态页直接配置可折叠波形组：高斯/方波包络、定光强/定频率辅助 |
| 自发辐射 | 动态引擎内建 **Lindblad 主方程**（decay 通道：指数衰变/显式落点/纯退相） |
| 存档 | 「文件→保存/恢复状态」；**只保存科学数据与参数，不保存显示参数**（见 §4.8） |
| 回归自检 | `checks/selftest_all.py` 一键全跑（约 90 s）：core 22 + 耗散 18 + UI 35 + 两体 11 |

## 1.3 两条计算路径的边界

- **静态能移 ≠ 动态演化**：静态页的 AC/DC 位移是「稳态」物理（dressed / 微扰）；
  动态页**不算斯塔克能移**——所有光都是动态驱动项（RWA 旋转系修正除外，那是坐标变换）。
  两者共用能级组与偶极矩阵元，但物理模型独立。
- **两体静态 ≠ 两体动态**：`pair_core.py`（静态）对角化含 V_dd 的 pair 哈密顿得 vdW/交换能移；
  `pair_evolve.py`（动态）在旋转系下做 secular RWA 演化，**不含二阶 vdW 能移**（见 §5.4.4 局限）。

---

# 二、层次架构总览

## 2.1 分层依赖图

```
┌─────────────────────────────────────────────────────────────┐
│ 第 0 层  入口  能级.py                                        │
│          只装配与编排：模块表、tab/切页循环、菜单（存档入口）  │
└─────────────────────────────────────────────────────────────┘
        │ 创建/调用
┌─────────────────────────────────────────────────────────────┐
│ 第 1 层  界面层  src/ui/                                     │
│  app_state（单一数据源）· fields（B/E/光场）· ui_group        │
│  ui_single/ui_spectrum/ui_pair/ui_radar（静态页）             │
│  ui_waves（波形面板）· ui_dynamic（动态页，单/双原子）         │
│  evolve_worker（演化线程）· serialization（存档）· ui_theme   │
└─────────────────────────────────────────────────────────────┘
        │ 依赖（只调用，不做二次换算）
┌─────────────────────────────────────────────────────────────┐
│ 第 2 层  核心科学计算  src/core/（无 UI 依赖）                 │
│  core.py      单原子物理：能级/偶极/AC/DC-Stark/频谱/偏振      │
│  pair_core.py 两体静态：V_dd/pair 哈密顿/vdW/交换/雷达图       │
│  evolve.py    单原子含时演化：ip4 / lindblad / auto（floquet·pwc 已封存）│
│  pair_evolve.py 两体含时演化：张量基/ secular RWA / RK4       │
│  waveforms.py 波形数据层：Wave/包络/build_fields             │
│  driver.py    统一驱动引擎：静态→驱动→RWA→演化/响应谱          │
└─────────────────────────────────────────────────────────────┘
        │ 交叉验证（权威参考）
┌─────────────────────────────────────────────────────────────┐
│ 第 3 层  检测与脚本  checks/ scripts/                         │
│  selftest_all（4 个核心回归）· _archive（按需归档自检）        │
│  scripts/（pairinteraction 对比、setup_arcdata、造态脚本）    │
└─────────────────────────────────────────────────────────────┘
```

## 2.2 各层职责一句话

| 层 | 职责 | 关键约束 |
|---|---|---|
| 入口（能级.py） | 装配页面、切页循环、文件菜单 | 不实现业务逻辑，不做逐模块特例 |
| 界面层 src/ui | 渲染 core 输出、收集参数、后台线程编排 | **ARC/SQLite 只在主线程访问**（§4.7）；序列化各模块自治（§4.8） |
| 核心层 src/core | 全部物理模型与数值计算 | 无 UI 依赖；结果即所见、所见即结果 |
| 检测层 checks | 回归钉死物理正确性 | 权威参考 = solve_ivp / pairinteraction / 精确 expm |

## 2.3 数据流

```
能级组（LevelGroupStore） ──► fields（B/E/光场） ──► core.build_level_rows ──► 静态画布/谱表/雷达图
                                                      │
能级组 + 波形（WavePanel）──► waveforms.build_fields ──► evolve.evolve / pair_evolve ──► 动态画布
                                                      │（worker 线程，纯数值）
「文件→保存状态」──► 各模块 serialize_state ──► JSON（只含科学数据）
```

---

# 三、入口层：能级.py

`能级.py`（约 280 行）只做三件事：

1. **装配**：创建 `EnergyGroupPanel`、`LevelAnalysisPanel`（静态）、`DynamicPanel`（动态）、
   `FieldPanel`（场面板），组装 `QTabWidget`（静态/动态两个 tab）+ 右侧 `QScrollArea` 控制区。
2. **切页循环**：`_on_canvas_tab_changed(idx)` 切换 `QStackedWidget` 与场面板显隐
   （静态页显示完整场面板含光场表格；动态页只显示 B/E 行）。
3. **文件菜单**：`文件 → 保存状态 / 恢复状态`，调用各模块自己的 `serialize_state()/restore_state()`。

```python
# 能级.py _save_state（存档装配：各模块自治，主程序只拼装）
snapshot = {
    'meta': {'version': 2},          # v2：只存科学数据/参数（见 §4.8）
    'groups': self.group_panel.serialize_state(),
    'field': self.field.serialize_state(),
    'dynamic': self.dynamic.serialize_state(),
}
```

其余业务逻辑全部下沉到模块内部（静态页计算在 `ui_spectrum`、动态页在 `ui_dynamic`），
主程序不做逐模块特例——这是「装配与编排」与「业务实现」分离的架构原则。

---

# 四、界面层 src/ui

界面层依赖 `src/core`（单向），负责：渲染、参数收集、后台线程编排、存档。

## 4.1 共享状态层 `app_state.py`

把页面与控件间的裸属性访问收口到统一 store，页面持 store 引用（构造注入），
store 变更经 `pyqtSignal` 广播（页面订阅刷新），不散布同步调用。

- **`LevelGroupStore`**：能级组单一数据源（增删/颜色/排序/变更事件）。
  `add()` 校验合法性（`core.validate_state`）并自动分配标签与颜色；`sorted_by_energy()` 供静态页排序。
- **`FieldState`**：磁场/电场/地址光 cfg 的单一读写入口；`sync_address_from_lights()`
  把第一份光同步为静态页的「寻址光」（AC-Stark 计算输入）。

## 4.2 场定义面板 `fields.py`

- 静磁场 B (G)、静电场 E (V/cm)：两页共用、恒显。
- 光场表格：序号 / 频率 (GHz) / 光强 (mW/cm²) / 偏振 / 作用范围；单击编辑、右键菜单
  （导出频率、扫描光强、椭圆偏振参数、删除）。
- 关注跃迁 / 初态（响应谱参数，暂时隐藏 UI，序列化保留为科学参数）。
- `serialize_state()`：只导出业务状态（B/E/lights/focus/psi），**不含纯显示状态**。

## 4.3 能级组管理 `ui_group.py`

定义 (n,l,j) → 校验 → 加入 `LevelGroupStore`；组列表（带色块）与数值表
（各 m 子能级能量相对最低态）。增删经 store 广播，两页自动刷新。

## 4.4 静态分析画布

- **`ui_single.py` / `ui_spectrum.py`**：能级图（off/on 镜像）+ 谱表 + 谱峰点击联动
  （点击 off 谱 → 原子 1 跃迁、on 谱 → 原子 2 跃迁，喂给雷达图）。
- **`ui_pair.py`**：响应谱画布（计算层保留，UI 暂缓）。
- **`ui_radar.py`**：两体耦合强度雷达图——固定 φ，扫描 R（对数刻度）× θ（0~180°），
  颜色 = \|V_ex\|（耦合强度），鼠标悬停实时读数 R/θ/φ/J。

## 4.5 波形管理面板 `ui_waves.py`

可折叠波形组（无通道、无 TTL、无独立窗口）：＋高斯 / ＋方波，每个折叠组标题栏 ✕ 删除；
参数多列并排（频率/光强带智能单位、偏振、包络、起点偏移）；「辅助」区选 g/e 态 +
目标 Ω「定光强」+ 目标 Δ「定频率」，内部 MHz 全精度写回。数据模型在 `waveforms.py`（§5.6）。

## 4.6 动态分析页 `ui_dynamic.py`

- 左侧上下双画布：上 = 各波形光强 I(t)（每波形独立 y 轴标度），下 = 态布居演化 P(t)。
- 右侧面板：单/双原子模式切换（radio）、V_dd 参数（R/θ/φ，仅双原子可见）、
  波形面板（单原子一个 / 双原子 A、B 两个 tab）、初态（单原子一个 / 双原子 A、B 各一个）、
  t_max/点数/共振截断/方法、自发辐射三档。
- 运行编排：`_run()` → 主线程预计算（能级、波形场、**双原子 V_dd**，见 §4.7）→
  `EvolveWorker(EvolveWorkerPair)` 后台线程 → 进度条轮询/可取消 → 结果画图。

## 4.7 演化工作线程 `evolve_worker.py`（线程安全约定）

- `EvolveWorker`：单原子——主线程已把 `states`（含能量）与 `fields`（含耦合矩阵）
  算好，工作线程只做纯数值演化。
- `EvolveWorkerPair`：双原子——**V_dd 在主线程预计算**（复用主线程 ArcCache），
  以 `vdd_matrix` 传入；工作线程不触碰 cache。

> **线程安全原则（重要）**：ARC 的 `AlkaliAtom` 底层是 SQLite，连接不能跨线程使用
> （`sqlite3: objects created in a thread can only be used in that same thread`）。
> 因此**所有 ARC 计算（能级、偶极矩阵元、V_dd）必须发生在主线程**；
> 工作线程只做 numpy/scipy 数值计算（`EvolveWorkerPair` 只收预计算的 `vdd_matrix`）。

## 4.8 存档序列化（各模块自治）

序列化按模块自治：`ui_group` / `fields` / `ui_dynamic` 各自实现
`serialize_state()/restore_state()`，主程序只装配调用（能级.py `_save_state`）。

**存档策略：只保存科学数据与参数，不保存显示参数。**

- 科学数据/参数：能级组 (n,l,j)、B/E 场、光场表格、波形（频率/光强/偏振/包络/偏移）、
  双原子 mode / R·θ·φ / A·B 波形 / A·B 初态、衰变通道、t_max/点数/方法/共振截断。
- 不保存：显示态数 `topn`、波形预览曲线 `wave_data`（派生显示数据，恢复后由波形重算）、
  后端选择（实现细节）、窗口布局。
- `meta.version = 2`（v1 旧档含 topn/wave_data/backend；restore 用 `get()` 默认值兼容）。

`src/ui/serialization.py` 仅保留旧档兼容助手（`restore_wave_data` 供外部转换旧档），
新存档不再经过集中式序列化。

## 4.9 主题 `ui_theme.py`

全局 QSS 主题（深色/统一配色），界面样式与物理计算完全解耦。

---

# 五、核心科学计算层 src/core（算法详录）

> 本节是本文档的核心：**精细记录每个核心模块的科学计算算法**——
> 物理模型、公式、单位、常量、容差、实现函数、关键修复。
> 所有模块无 UI 依赖，可单独导入审查。

## 5.0 层内结构与单位约定

### 5.0.1 文件职责

| 文件 | 职责 | 依赖 |
|---|---|---|
| `core.py` | 单原子静态物理 | ARC 库（原子数据） |
| `pair_core.py` | 两体静态 | core |
| `evolve.py` | 单原子含时演化 | core（仅偶极矩阵元） |
| `pair_evolve.py` | 两体含时演化 | evolve + pair_core |
| `waveforms.py` | 波形数据层 | evolve |
| `driver.py` | 统一驱动引擎（响应谱） | core + pair_core |

### 5.0.2 单位约定（全层统一）

| 量 | 单位 | 备注 |
|---|---|---|
| 能级 / 耦合 / 失谐 / Rabi | **MHz** | 内部 Python float64 全精度 |
| 频率（光参数） | Hz | 界面 GHz/THz 显示层换算 |
| 偶极矩阵元 | 原子单位 a.u.（e·a₀） | ARC 返回值 |
| 光强 | mW/cm² | 换算 W/m² 因子 10 |
| 时间（演化） | ns | 角频率 rad/ns |
| 几何 | 间距 μm，角度 ° | |

关键换算常数（`core.py` / `evolve.py`）：

```
EV_TO_HZ          = e/h              # eV → Hz（ARC getEnergy 返回 eV 绑定能）
MU_B_MHZ_PER_G    = μ_B/(h·1e10)     # MHz/G
_CONV             = 2π × 1e-3        # MHz·ns → rad（角频率换算）
MW_PER_CM2_TO_W_PER_M2 = 10.0        # 光强换算
A0 = Bohr 半径（m）; E_CHARGE = 基本电荷（C）
```

## 5.1 `core.py` 单原子物理

### 5.1.1 物理模型总览

1. **能级**：里德堡态（n ≥ 20）只画 fine-structure 的 m_J 子能级（忽略核自旋）；
   低激发态在 \|m_J, m_I⟩ 基下按 m_F 分块数值对角化超精细 + Zeeman 哈密顿量。
2. **AC-Stark**：对每个偶极耦合伙伴按失谐分流——近共振 → dressed 两能级精确解；
   远失谐 → 非 RWA 二阶 SOS（rotating + counter-rotating 都保留）。
3. **跃迁频谱**：子能级间全部偶极允许跃迁，强度用爱因斯坦 A 系数。
4. **频移后端**：唯一后端 = `direct`（Floquet 直接矩阵对角化，第一性原理；
   legacy「逐伙伴解析/微扰」已于 2025-08 删除）。

### 5.1.2 能级构造算法

**里德堡态（n ≥ `RYDBERG_N_MIN` = 20）**：

```
E(n,l,j,m) = E0(n,l,j) + g_J(l,j) · μ_B/ħ · B · m
```

- 每个 m_J 一条子能级；核自旋忽略（超精细对里德堡态可忽略）。
- 实现：`_base_level_rows` → `_sublevel_energy`；g_J 由 ARC `getLandegjExact` 缓存（`ArcCache.gJ`）。

**低激发态超精细 + Zeeman（数值对角化）**：`get_hyperfine_levels(cache, n, l, j, B)`

在 \|m_J, m_I⟩ 基（`mJ_list × mI_list`，I 取自 `atom.I`）下，哈密顿量：

```
H = A·I·J + B·Q + μ_B·B·(g_J·J_z + g_I·I_z)
```

- **I·J 矩阵**：对角 `m_J·m_I` + 升降算符项
  `I₊J₋: (m_J,m_I)→(m_J−1,m_I+1)`、`I₋J₊: (m_J,m_I)→(m_J+1,m_I−1)`，
  系数 `½√(I(I+1)−m_I(m_I±1))·√(j(j+1)−m_J(m_J∓1))`。
- **电四极项 Q**（仅 J>1/2 且 I>1/2，B 常数非零）：
  `Q = 3(I·J)² + 1.5(I·J) − I(I+1)·j(j+1)·1`，除以 `2I(2I−1)j(2j−1)`。
- **按 m_F 分块**（`m_F = m_J + m_I` 守恒，对角化块大小 ≤ 2I+1）：
  - 零场块对角化用于标注 F（零场能量 `E_F = (A/2)[F(F+1)−I(I+1)−j(j+1)]`，
    按排序匹配 F 量子数）；
  - 加 B 后再对角化 `H_hf + H_z`，得本征能量 E、本征向量。
- 返回 `(F, m_F, E_mhz, mJ_weights, mJ_amps)`：
  - `mJ_weights[m_J]`：各电子 m_J 分量占比（对 m_I 求和，界面/谱线显示用）；
  - `mJ_amps[m_J]`：复振幅（m_I = m_F − m_J 由分块唯一确定，**供跃迁偶极矩阵元计算**）。
- 里德堡态或无超精细数据 → 返回 None（忽略核自旋）。

### 5.1.3 偶极矩阵元算法

**ARC 原始矩阵元**：`ArcCache.dme(n1,l1,j1,m1, n2,l2,j2,m2, q)` 直接查 ARC
`getDipoleMatrixElement` 并按键缓存（`_dme` 字典）。

**统一入口 `dipole_amplitude(cache, a, b, q)` = ⟨b|d_q|a⟩（a.u.，复）**，四种情况：

| a 类型 | b 类型 | 规则 |
|---|---|---|
| mJ | mJ | 需 `b.mj == a.mj + q`，直接查 dme |
| F | mJ | 遍历 a 的 m_J 分量：`Σ amp_a(mj)* · dme(mj → b.mj)`，需 `mj+q == b.mj` |
| mJ | F | 遍历 b 的 m_J 分量：`Σ amp_b(mj)* · dme(b.mj → a.mj)`，需 `mj+q == a.mj` |
| F | F | 需 m_F 守恒（`b.mF == a.mF + q`，核自旋不参与），双分量求和 |

- 偶极算符只作用于电子（Δm_J = q）；缺 mJ 匹配会把「与其他 mJ 态的耦合」错误归到本对
  （实测 bug：6P3/2 F=3 mF=−1 ↔ 60S mJ=+0.5 在 σ⁻ 下被误报非零，实际是到 60S mJ=−0.5）。
- F–F 缺 m_F 守恒检查会给出违反选择定则的非零强度（ARC HFS 版对不满足者返回 0）。
- `transition_strength_q = |dipole_amplitude|²`（谱线/驱动共用同一矩阵元逻辑）。

**Rabi 换算**（`field_e0_sq` / `rabi_mhz` / `intensity_mwcm2_from_rabi`）：

```
E0² = 2·I/(ε₀·c)              # I 为 W/m²
Ω = E0·|d|/ħ                   # d 转 SI：|d|·e·a₀
Ω_MHz = Ω/(2π)/1e6
```

### 5.1.4 AC-Stark 物理公式

AC-Stark 唯一后端 = direct（Floquet 矩阵对角化，§5.1.5）。下述物理公式为其内化内容
（legacy「逐伙伴解析/微扰」逐态实现已于 2025-08 删除）：

- **近共振 dressed 两能级精确解**（所有失谐区成立：远失谐回微扰 Ω²/4Δ，共振给 Autler-Townes ±Ω/2）：

```
ΔE源 = ½(sign(Δ)·√(Δ²+Ω²) − Δ)      （Δ = 2π·失谐，Ω = Rabi，均 rad/s）
ΔE目标 = 与之严格反对称
```

  direct 完整对角化在共振处源态位移 ≈ **−Ω/2**（含超精细多态/反旋项自洽）。
- **远失谐非 RWA 二阶 SOS**（rotating + counter-rotating 都保留；只留 rotating 会把
  里德堡-里德堡对位移高估 ~10⁴ 倍）：

```
ΔE = −(E0²/4ħ) · |d|² · 2ω_ki/(ω_ki² − ω²)     （ω_ki = 子能级能量差角频率，带符号）
```

  由 `far_sos_shift_mhz` 实现，同时被动态引擎 `evolve.real_field` 复用于时变 AC 位移；
  ω→0 极限退化为 DC-Stark/2（T8 回归）。
- **伙伴子能级**（`_partner_sublevels`，核自旋平均权重 |amp|²/(2I+1)）供
  `dc_stark_shift_mhz` 复用（DC-Stark 独立物理，唯一实现）。
- **椭圆偏振**（`elliptical_coeffs(α,θ,ε)` → 球基复系数 (c₊₁,c₀,c₋₁)，Σ|c|²=1）：
  传播 k=(sinα,0,cosα)，垂直基 e1/e2，Jones 矢量到球基变换；耦合按各 q 分量
  `|c_q|²` 加权（同一 (m,m′) 对只能经唯一 q 耦合，无干涉）。

### 5.1.5 AC-Stark：direct 后端实现（Floquet 矩阵对角化）

`direct_ac_stark_shifts(cache, groups, address_cfg, B_gauss, n_photon=1)`：

**基组**：全部已添加组子能级 + 各组偶极伙伴（|Δn| ≤ `DIRECT_SOS_N_WINDOW`=3）。

**Floquet 哈密顿量**（光子块 M = 2·n_photon+1 = 3）：

```
H_F = Σ_n (H0 + n·ħω)⊗|n⟩⟨n|  +  (V/2)⊗(|n+1⟩⟨n| + |n⟩⟨n+1|)
```

- 对角块 `H0 + (p−1)·freq_mhz·1`；相邻块耦合 `C = E0·C_unit`，
  其中 `C_unit_ij = −d_ij·(e·a0)/(2h·1e6)`（MHz，单位 E0），`E0² = 2I/(ε0c)`。
- 一次 `np.linalg.eigh` 对角化 → 自洽包含近共振 dressed + 远失谐反旋项 + 多态耦合 + 超精细相干性。

**远失谐里德堡↔里德堡屏蔽**：对 ω0=|E_i−E_j| 满足 `|ω0−ω|>ω0` 的对，净位移
≈ Ω²ω0/(2ω²) 可忽略，但有限光子块截断下该对在光子块边界态上只剩旋转波方向通道，
产生虚假 ~Ω²/ω 位移并破坏共振——故直接屏蔽该耦合。近共振对保留真实 dressed/SOS 贡献。

**位移归属**（`_floquet_shifts`）：本征态对光子块 0 的投影重叠确定归属（取最大者）；
共振 50/50 劈裂（重叠 <30%·max）时取离裸能量最近的分支。

**光强扫描**（`direct_ac_stark_sweep`）：耦合 C ∝ E0 ∝ √I，基组与对角块不随 I 变——
只构建一次（基组, E, C_unit），对每个 I 重标定耦合并重对角化，快一个量级；
可用精简基组（window=1）定位再用全基组（window=2，误差 <0.3%）局部迭代收敛。

**行结构**（`build_level_rows` / `_base_level_rows` / `_fill_ac`）：
每行 `{x, E0, E, ac, affected, label, qstr, color, n, l, j, kind, mj/mF, amps}`；
`affected = |ac| > 0.05 MHz`（`AC_EFFECT_THRESHOLD_MHZ`）。唯一后端 direct，
UI 直接调用 `build_level_rows`（不再有后端切换开关）。

### 5.1.6 DC-Stark

`dc_stark_shift_mhz(cache, n, l, j, m, groups, E_vpcm, B_gauss)`：二阶 SOS（第一性原理）：

```
ΔE_i = Σ_k |⟨k|d|i⟩|² E²/(E_i − E_k)     （静电场无旋转波：全部偶极中间态贡献）
```

- 遍历已添加组 + `dipole_partners` 未添加的完整基组；吸收（m+q）/发射（m−q）两方向，
  dme 参数相应交换；核自旋平均权重。
- 量纲：ΔE(J) = Σ d_cm²E²/(E_i−E_k)；换算 MHz 除以 h²·1e12（h=2πħ）。
- E² 标度已验证（checks）。

### 5.1.7 频谱与偏振

**谱线**（`spectrum_lines`）：子能级对 (a,b) 全部偶极允许跃迁：

```
A = ω³/(3πε₀ħc³) · |⟨a|d_q|b⟩|² · (e·a₀)²     （爱因斯坦 A 系数，s⁻¹，作谱线强度）
```

- 矩阵元按 m_J 分辨（`dipole_amplitude` 四分支）；偏振系数存在时按 `|c_q|²` 加权，
  pol 标注非零 q 对应的 σ+/π/σ−；`ac_on=True` 时用 AC 移后能量计算频率。
- 同 (n,l,j) 组内不产生谱线（跳过）。

**跃迁对查找**（`collect_transitions` / `_closest_sublevel_pair`）：组对间取
**失谐最小**的耦合子能级对（而非矩阵元最强）——与当前寻址光频率最匹配的跃迁对，
含超精细/Zeeman 的子能级失谐与 Rabi（rad/s）。任意偏振取全局失谐最小的 q 分量。

## 5.2 `pair_core.py` 两体静态

### 5.2.1 偶极-偶极相互作用 V_dd

```
H_pair(R) = H1 ⊗ 1 + 1 ⊗ H2 + V_dd(R)
V_dd = (1/R³) Σ_qq' d1_q · T_qq' · d2_q'，   T = e_qᵀ(1 − 3nnᵀ)e_q'
```

- n = (sinθcosφ, sinθsinφ, cosθ) 为两原子连线方向（θ=0 → n∥z）。
- θ=0 退化：`V_dd = (1/R³)[−d1₋₁d2₊₁ − d1₊₁d2₋₁ − 2d1₀d2₀]`（`VDD_QCOEFF`）。
- 单位换算：`_DD_CONV = Hartree_Hz/1e6/(1e-6/a₀)³`（MHz·μm³/(a.u.)²）。

**实现**（`_vdd_tensor` + `_dipole_sparse` + `vdd_sparse`）：

- `_dipole_sparse(cache, single, q)`：单原子偶极矩阵 D_q（N×N，CSR），
  `D_q[i,i2] = ⟨i2|d_q|i⟩`（a.u.），选择定则 Δl=±1、|Δj|≤1 预过滤；
  以 `_single_fp` 为键 LRU 缓存（容量 64）。
- `vdd_sparse`：`V = (1/R³) Σ_ab T_ab · kron(D_a, D_b)`（pair 基 k=i·N+j），LRU 缓存。

### 5.2.2 基组构造与截断

- **单原子基**（`single_atom_basis`）：已添加组 + 每组偶极伙伴 |Δn|≤n_window
  （自动补 vdW 中间态，如 60S 的 59-61 S/P/D）。
- **pair 索引**（`pair_indices`）：笛卡尔积 (i,j)，k=i·N+j；可选截断：
  总磁量子数 m_total（m₁+m₂ 分块，V_dd 保 m₁+m₂）；能量窗口
  `|E_a+E_b−E_ref| ≤ energy_window_mhz`（可选）。
- 重截断逻辑的 pair 系统构建/对角化按 (groups, B, n_window, R, θ, φ, 截断) LRU 缓存
  （容量 32/64），返回一律拷贝（`build_system` 会原地改 H 对角，防污染缓存）。

### 5.2.3 pair 哈密顿与本征

- `pair_hamiltonian`：`H = diag(E_pair) + V_dd`（子空间稠密复矩阵，CSR 索引后 toarray）。
- `pair_eigen`：`np.linalg.eigh` → evals（升序 MHz）+ evecs（列=本征态）。
- `pair_component(res, a_row, b_row)`：各本征态在 |a⊗b⟩ 的权重 |⟨ψ|ab⟩|²。

### 5.2.4 实用提取

- **交换耦合**（`exchange_coupling`）：一阶项 ⟨a|V_dd|b⟩（MHz）——直接查矩阵元不需对角化；
  默认 60S↔60P 交换对 |SP⟩↔|PS⟩。
- **vdW 位移**（`vdw_shift`）：对角化后从 |ref⊗ref⟩ 支（overlap 最大本征态）提取
  `shift = E_支 − 2E_ref`（二阶项 ∝1/R⁶，**不在 raw 对角**）。
- **耦合强度雷达图**（`coupling_strength_map`）：固定 φ 扫 R×θ，
  `|V_c| = |⟨tgt1⊗tgt2|V_dd|src1⊗src2⟩|`——两原子各自跃迁对的偶极向量
  `d_a(q) = ⟨tgt1|d_q|src1⟩`，`V_c = (1/R³)|Σ T_ab·d_a·d_b|`；
  交换对只在里德堡态之间（n ≥ `RYDBERG_N_MIN`）有物理意义，否则全零。

## 5.3 `evolve.py` 单原子含时演化

### 5.3.1 物理模型与场算符

```
H(t) = H0 − Σ_i ref_i·P_i + Σ_k [ (Ω_k(t)/2)·A_k ] + BS     （RWA，旋转系）
H(t) = H0 + Σ_k Ω_k(t)·cos(ω_L,k·t)·A_k                      （非 RWA 精确，实验室系）
```

- H0：对角 = 各能级能量 E_mhz（用户输入，可含 Zeeman 等任意静态）。
- **统一场算符**（Pulser 范式）：每个物理光场 = 一个算符 `M_k = A_k + diag(ac_k)`——
  近共振驱动对的真实 Rabi 耦合矩阵 A_k（含选择定则与相对强度）与远失谐态的
  二阶 AC 位移 diag(ac_k) 同属一个场，随包络 env_k(t) 一起时变
  （避免"常数 AC 位移全时段生效"的物理错误）。
- 三种场工厂：
  - `field(g, e, Ω, waveform, Δ)`：单对驱动，A 为 0/1 单耦合；
  - `group_field(cache, states, ga, gb, pol, Ω)`：组驱动，A 各 (m_g,m_e) 耦合
    = |dme|/max|dme|（相对强度来自 core 偶极矩阵元）；
  - `real_field(cache, states, freq, intensity, pol)`：真实光场→按自适应失谐分流
    （近共振合并成多对驱动场、远失谐 SOS 位移进 diag(ac)）。

### 5.3.2 旋转参考系算法

**级联 RWA 参考**（`_assign_reference` / `_ip4_reference`）：

以 (n,l,j) 组为单位，从能量最低组 ref=0 出发，沿每个 RWA 场的耦合对传播
`ref[高能组] = ref[低能组] + ω_L,k`——链式/树状驱动图使每个驱动对在旋转系里共振，
正确处理**共享能级**的多驱动（如光频 6P↔60S 与微波 60S↔60P 同开时 60S 是中间态，
级联旋转 ref[60P] = ref[60S] + ω_mw）。闭环不一致的驱动留下残留拍频 δ
（RWA 丢弃并警告，误差 ~Ω/|δ|）。纯位移场（无 g_idx）不参与传播。

**非 RWA 场共享参考**：全部耦合组共享同一参考（e 跟随 g），使微波耦合为纯 cos 快项
（比全非 RWA 快 ~10⁴ 倍）。multi 场耦合多组时必须**逐对传播**（否则 E_rot 伪劈裂 → 相位爆炸）。

**旁观态归一**（`_e_rot`）：旁观态（无耦合）E_rot 归零；耦合态减共同参考——
避免光频量级参考使 u(t) 相位爆炸、步长估计被污染。

### 5.3.3 算法 1：相互作用绘景 RK4（`_evolve_ip4`，第一性原理，唯一纯幺正算法）

> 动态分析只用 ip4（单体 RK4 足够快）；精确 Floquet 与 RWA 分段常数（pwc）
> 已封存至 `src/_archive/evolve_floquet_pwc.py`（§5.3.6）。

适用：任意时变包络；光频（Ω/ωL < `IP4_RWA_THRESH`=1e-4）→ RWA，微波等强驱动 → 非 RWA。

**相互作用绘景**：H0 = diag(E_rot) 相位精确除去，积分
`Ṽ(t) = Σ s_k(t)·u·M_k·u†`（u = e^{iE_rot t}，统一场算符对角位移相位自动抵消）。

**全 RWA + 常数包络 → 一次对角化精确解**：`H = diag(E_rot) + Σ(Ω/2·c)·A + c·diag(ac)`，
`eigh` → `ψ(t) = V·diag(e^{−iεt})·V†·ψ0`（RWA 内精确，秒出）。

**步长三重约束 + 下限/上限**（`_evolve_ip4` 与 `_evolve_lindblad` 同款）：

```
w_max   = 最快真实振荡频率：
          RWA 场     → |E_rot_i − E_rot_j|（旋转系失谐）
          非 RWA 场  → |E_rot_i − E_rot_j| + ω_L（载波）
cmax    = max Ω·‖A‖（耦合幅度，RK4 精度/稳定约束用）
dt_carrier = 2π/(_CONV·w_max)/frac            # frac=ip4_frac（默认 20 → ~3e-5 误差）
dt_acc     = 0.06/(_CONV·max(cmax,1e-6))      # dt·‖Ṽ‖ ≲ 0.06（经验 → 布居误差 1e-4~1e-5）
dt_stab    = 2.0/(_CONV·max(cmax,1e-6))       # RK4 稳定性上界
dt_res     = t_end/500                        # 至少 500 步
dt         = min(dt_carrier, dt_acc, dt_stab, dt_res, 0.05)
```

**numba 稀疏内核**（`_ip4_rk4_core`）：稀疏耦合 CSR（非零元并集），
S 预计算到 1/4 步长子网格（RK4 阶段时刻 4m, 4m+2, 4m+4 精确采样），
输出时刻从步首做精确部分步进（`rk4_step_arb`），nogil 进度轮询/取消；
无 numba 时 numpy 回退（`_ip4_rk4_core_numpy`）。`real_omega0_mhz` = |A|.max()
用于 RWA 分类（multi 场），避免用归一化 omega0=1 误判。

实测：N=20 高斯脉冲 400 ns → 29 s（T/20）/ 54 s（T/40），误差 3e-5 / 1e-6。

### 5.3.4 算法 2：Lindblad 主方程（自发辐射等马尔可夫耗散）

```
dρ/dt = −i[H(t),ρ] + Σ_k (L_k ρ L_k† − ½{L_k†L_k, ρ})
L_k = √Γ_k |to_k⟩⟨from_k|      （decay(g, Γ, e) 工厂）
```

- **单位约定**：Γ 单位 MHz → 速率 Γ×1e-3 /ns；**跳变算符不乘 2π**（速率 = Γ×1e-3）。
- **旋转系下跳变算符时不变**（LρL† 与 L†L 中相位自动抵消）→ 耗散项与旋转参考选择无关，精确。
- 落点 e=None → 虚拟 loss 态（下标 N）：布居只进不出，`loss = ρ[N,N]`，Σpops+loss=1。
- g==e → 纯退相通道（|g⟩⟨g| 型测量算符，相干衰减速率 Γ/2，布居不变）。
- **`lindblad_eig`**：旋转系下 H 恒定（全 RWA + 包络恒常 + 无残留拍频）→
  Liouvillian 本征分解精确解（机器精度，成本与时长无关）：
  `ρ_vec(t) = Σ_k e^{λ_k t} v_k (w_k†ρ0)`，
  `Lv = −i·_CONV·(I⊗H − H†⊗I) + Σ_k (L_k*⊗L_k − ½(I⊗L_k†L_k + (L_k†L_k)*⊗I))`。
- **`lindblad`**：矩阵 RK4（ρ 为 N×N），任意时变包络，含非 RWA 载波。
  **numba 内核**（`_lindblad_rk4_core`）：H 稀疏 + ρ 厄米 → 对易子一次点乘
  （`d = −i(X−X†)`，X=Hρ）、S 预计算到 dt/2 子网格（RK4 阶段时刻全为 dt/2 整数倍 →
  全步精确采样；仅输出部分步线性插值）、rank-1 跳变 O(N²) 快速路径、nogil 可取消。
  实测非 RWA 微波场景从 numpy 纯循环 ~40 分钟降到 ~1 分钟（42×，机器精度一致）。
- **`auto_decays`**：按 ARC 权威寿命自动配置（Γ = 1/τ，`getStateLifetime`；
  只含 τ ≤ 100 µs 低激发态，落点=丢失；排除里德堡慢衰变/黑体）。

### 5.3.5 auto 调度与参考积分器

`evolve()` 一站式入口（2025-08 起 auto 无耗散恒走 ip4）：

```
auto（无耗散）：恒 'ip4'（相互作用绘景 RK4，第一性原理）
auto（有耗散）：旋转系 H 恒定 → 'lindblad_eig'；否则 → 'lindblad'
显式 'ip4' 配耗散 → 自动切换 Lindblad 路径并 log 提示
```

参考积分器（内部保留，仅 checks 交叉验证）：`_evolve_full`（实验室系 solve_ivp 非 RWA，
最无偏基准）、`_evolve_full_hybrid`（光频 RWA + 其余非 RWA，快 ~10⁴ 倍）。

`rwa_error_estimate`：每驱动 RWA 误差估计 `est_err ≈ Ω/(4ωL)`

### 5.3.6 封存算法（`src/_archive/evolve_floquet_pwc.py`）

精确 Floquet 传播子 与 RWA 分段常数（pwc）已从 evolve.py 封存（2025-08，动态分析
只用 ip4）。封存模块独立可用，与旧 `evolve.evolve(method='floquet'/'pwc')` 等价：

```
from src._archive.evolve_floquet_pwc import evolve_floquet, evolve_pwc
r = evolve_floquet(states, fields, psi0, t_grid)     # 常数包络 + 单频 → TDSE 精确解
r = evolve_pwc(states, fields, psi0, t_grid, bs=True)  # RWA 分段常数（最快预览 + BS 修正）
```

- **Floquet**：`H(t)=diag(E)+cos(ωt)ΣA_k`，单周期中点法积分（n_sub=2000）→ U(T) 本征分解
  （准能量 ε=−arg λ/T）→ `ψ(t)=U(τ)·W·diag(e^{−iεnT})·W†·ψ0`——TDSE 精确解，
  成本不随时长增长（周期驱动研究用）。
- **pwc**：每段 H 取中点 → eigh 传播；numpy 逐段 / torch GPU 批量；精度 ~Ω/ωL
  （微波强驱动 RWA 误差可达 ~3%）；`bs` = Bloch–Siegert 一阶修正（Ω²/4ω_L 对角）。

注意：`evolve.evolve(method='floquet'/'pwc')` 现在会抛 ValueError 并指向本模块
（不静默）。相关基准（`checks/evolve_matrix.py`、`mw_bench_firstprinciples.py`、
`evolve_rwa_bench.py`、`acceptance_state.py`）均已改引封存模块。
（实测系数 ~1/4；多能级 + 长时可比单对估计大 ~20 倍，精算勿依赖 RWA）。

## 5.4 `pair_evolve.py` 两体含时演化

### 5.4.1 张量基组与场

```
H(t) = [H_A(t) ⊗ I_B] + [I_A ⊗ H_B(t)] + V_dd
```

- 两体基组 = 单体 states 的笛卡尔积 |i⟩⊗|j⟩（N² 维），k = i·N + j（`tensor_states`）。
- 单体场 → 两体场（`_tensor_field_atom`）：`M_A⊗I_B` 或 `I_A⊗M_B`（`np.kron`）；
  ac 位移对角加 `ac_2d[i·N+j] = ac[i]`（A）或 `ac[j]`（B）；
  g_idx/e_idx 相应扩展（A：`gi·N+j`；B：`i·N+gi`）；透传 `real_omega0_mhz`（RWA 分类）。
- 两体初态 `|ψ_A⟩⊗|ψ_B⟩`（`_as_product_psi`）；两体旋转参考
  `ref_2d[i·N+j] = ref_A[i] + ref_B[j]`（`pair_rotate_frame` 复用单原子 `_ip4_reference`）。

### 5.4.2 V_dd 旋转系 secular RWA（核心判据）

`_secular_filter(V_dense, E_rot_2d, ratio=SECULAR_RATIO=10.0)`：

```
V_{αβ} 在相互作用绘景中的振荡频率 = |E_rot_β − E_rot_α|（E_rot 含 V_dd 对角位移）
|Δ| < ratio·|V| → 保留（缓变，RWA）；否则高振荡项平均为零 → 丢弃
```

- 对角位移 V_diag = Re(V[α,α]) 并入 E_rot；V_sec 为保留的非对角稀疏矩阵。
- 输入可为稠密 ndarray 或 scipy 稀疏（稀疏时只遍历 COO 非零元，大基组不爆内存）。
- `vdd_secular(cache, single, R, θ, φ, ref_A, ref_B, N, E_rot_2d)`：cache 路径
  （`pair_core.vdd_sparse` → `_secular_filter`），E_rot_2d 缺省时从 E_mhz − ref 重建。

> 判据必须用**真实旋转系能量差**（含 V_dd 对角位移）——若误用旋转参考之差，
> 无光场时恒 0，会把远失谐 vdW 元误当 secular 保留（T5 回归钉死）。

### 5.4.3 两体演化主循环（`_pair_ip4`）

- **RWA 分类**（`_is_rwa`，步长估计与项组装共用同一判据）：
  `ω_L > 0 且 Ω/ω_L < IP4_RWA_THRESH` → RWA；`ω_L = 0` 常数场（V_dd secular）按非 RWA
  处理（cos(0)=1，无载波振荡）。
- **对角化精确路径**：全 RWA + 常数包络 + **无振荡非 RWA 场** → 一次 eigh
  （非 RWA 微波即使包络恒常也含 cos(ω_L t) 载波，哈密顿量时变，不能走该路径）。
- **步长**（三类场分别计入 w_max）：
  RWA 场 → \|E_rot_i − E_rot_j\|；非 RWA 场 → +ω_L；**ω_L=0 常数场 → \|E_rot_i − E_rot_j\|**
  （保留项的振荡频率必须被步长解析）。
- **旁观态归一**：耦合态集合 = 驱动场 g/e 索引 ∪ V_sec 非零元索引；
  耦合态减公共参考、旁观态 E_rot 归零（同单原子 `_e_rot`，防相位爆炸）。
- 相互作用绘景 RK4：`interaction = c·Σ_k s_k(t)·(u·M_k·u†·ψ)`，u = e^{iE_rot t}；
  发散防护（|ψ|>100 或非有限 → RuntimeError）；progress_arr 进度/取消。

### 5.4.4 pair_evolve 主入口

`pair_evolve(states, fields_A, fields_B, psi0_A, psi0_B, vdd_matrix=..., R_um/θ/φ/cache=...)`：

- V_dd 来源：`vdd_matrix`（外部直接给）或 `R_um + cache`（自动计算）；
  **cache 必传**（缺则 ValueError——防 V_dd 静默为零，两体退化为两独立单体）。
- **两体 Lindblad 暂不支持**：decays_A/B 非空显式 NotImplementedError。
- **实现约定**（回归钉死于 `checks/pair_evolve_selfcheck.py`）：
  secular 判据用旋转系能量差（§5.4.2）；步长三类场分别计入 w_max；
  对角化路径仅当无振荡非 RWA 场；GUI 双原子在主线程预计算 V_dd（§4.7 线程安全）；
  旁观态 E_rot 归一。
- 返回 `t, pops_2d, pops_A, pops_B, states_2d`（pops_A/B 为约化布居，trace out 另一原子）。

**模型局限**：secular RWA 丢弃远失谐元素 → 二阶 vdW 能移不在演化哈密顿中
（与 pair_core 静态设计一致）；近简并交换/Förster 动力学精确（T3 与精确 expm 一致至 1e-5）。

## 5.5 `driver.py` 统一驱动引擎

### 5.5.1 分层设计（driver-centric）

```
静态层 pair_static    ：非 driver 因素 → H_static（完整 pair 基，含 vdW/交换）
驱动层 driver_operator：driver → (D, ω, Ω)
化简层 rwa_hamiltonian：旋转系 RWA → H_rwa（高能侧减 ω，加 Ω/2·D）
演化层 evolve          ：定常哈密顿一次对角化 → 布居时间演化
```

- `driver_operator`：kind='mw'（微波 group_a↔group_b，Rabi 由光强算
  `Ω = E0·|d|max/ħ` 或用户直给 om_mhz；scope=both/atom1/atom2 → D = kron 张量）；
  kind='ex'（交换驱动 |a⊗b⟩↔|b⊗a⟩，Ω = |V_ex(R)|）。
- `rwa_evolve`：`H = diag + Σ(Ω/2)(|i⟩⟨j|+h.c.)`，**Ω 为完整 Rabi 频率（内部取半）**——
  ⚠ 不要把半耦合 Ω/2 传进来（Rabi 减半，常见坑）。
- `full_evolve`：不 RWA 的 solve_ivp 完整含时积分（最无偏基准，交叉验证 RWA 实现）。

### 5.5.2 RWA 化简（光子数 BFS）

`rwa_hamiltonian`：每态按光子数减 n_k·ω。`_photon_numbers` 沿驱动算符 D 从初始态
BFS 推断光子数（微波驱动 SS=0、SP/PS=1、PP=2 → 正确支持双光子级联 SS→SP→PP）；
D 连通分量外未触及的态记 0。`H_rwa = H_static − diag(n_k·ω) + (Ω/2)·D`。

### 5.5.3 dressed 投影（vdW 位移进入封锁）

`_dressed_hamiltonian`：静态 H 全对角化 → dressed 能级（含 vdW）→ 驱动算符投影到
本征基 → RWA 演化 → 投影回 raw 基布居。vdW（C₆ 二阶位移）通过静态对角化进入
dressed 能级，**封锁物理完整**。`pair_drive_dressed` 为两体默认路径。

### 5.5.4 线性响应谱（黄金规则）

`response_spectrum_linear`：**一次静态对角化**提取全部共振：

```
S(ω) = Σ_k |⟨k|D|i⟩|² δ(ω − (E_k − E_i))
```

静态 H（含 vdW/交换/AC-Stark）对角化一次 → 驱动算符 D（**真实偶极矩阵元**，
含 scope 单原子寻址）投影到本征基 → 初态 |i⟩ 的全部共振频率与相对强度一次算出
（O(N²)），比逐频率含时演化快 ~N_freq 倍。`response_spectrum`（逐频率扫驱动）为
对照实现。`build_system(n_body=1/2)` 提供统一系统接口（`drive` / `response_spectrum` 共用）；
`static_cfg` 的能移光（`ac_stark`）走 `core.direct_ac_stark_shifts`（Floquet 对角化，2025-08 起，
legacy 逐态 `ac_stark_shift_mhz` 已删），DC 静电场（`dc_stark`）走 `dc_stark_shift_mhz`；
两体能移 = 两单原子位移之和（`shift_i + shift_j`）。

## 5.6 `waveforms.py` 波形数据层

- **包络工厂**：`GaussianEnv(center, σ, peak)`——`peak·exp(−(τ−center)²/2σ²)`；
  `SquareEnv(start, edge, height, fall_start)`——余弦缓变方波（raised cosine，
  两端导数连续为 0，`fall_start` 缺省 = start+2·edge 最小脉冲）。
- **`Wave`**：一个独立波形 = 频率 + 光强 + 偏振 + 包络 + 起始偏移（offset_ns，
  相对演化 t=0 绝对定位）；`drive`（g/e 辅助对）仅用于定参，不参与演化结算。
- **`waveforms_to_env(waves, t_grid)`**：各波形光强曲线（画光强不画载波），
  直接用输出网格采样。
- **`build_fields(waves, states, cache, near_cutoff_mhz)`**：每波形 → `evolve.real_field`
  （近共振驱动对 + 远失谐 AC 位移，见 §5.3.1）；near_cutoff 默认 5 GHz 保守地板
  （协议场景可调小，如 50 MHz，流形其余态作远失谐 AC）。
- **序列化**：`serialize_waves` / `restore_waves`（波形 = 科学数据，参与存档）。

---

# 六、检测与脚本层 checks/ scripts/

## 6.1 回归自检矩阵（`checks/selftest_all.py` 一键全跑，约 90 s）

| 自检 | 项数 | 覆盖 | 权威参考 |
|---|---|---|---|
| `core_selfcheck.py` | 22 | 能级/偶极/AC 位移（direct 唯一后端）/频谱/偏振/far_sos ω→0==DC | 解析公式、ARC 官方函数、direct 自洽 |
| `evolve_dissipative_selfcheck.py` | 18 | 解析衰变 e^{−Γt}、CW/时变、退相、ac 约定、numba≡numpy | 独立 Liouvillian 本征参考、solve_ivp |
| `ui_selftest.py` | 35 | 主窗口/波形面板/动态页/存档往返（含双原子参数、线程安全） | UI 冒烟 + 状态往返断言 |
| `pair_evolve_selfcheck.py` | 11 | 两体 V_dd 矩阵元/能谱/vdW 位移/交换动力学/secular 判据 | **pairinteraction 2.x + 精确 expm** |

其余自检（pair/unified、STIRAP/拉曼、pulse、Floquet 交叉、单原子 evolve）归档于
`checks/_archive/`，需要时单独运行，不进日常回归。

## 6.2 pairinteraction 权威交叉验证（两体检测设计）

`checks/pair_evolve_selfcheck.py` 按"**权威开源软件 + 最简单直接**"标准设计：

- 所有物理输入（能级、V_dd、两体哈密顿）由 **pairinteraction 2.x** 独立给出
  （`BasisAtom`/`SystemAtom`/`BasisPair`/`SystemPair`，B=0 基组精确对齐）；
- 动力学参考 = 同一哈密顿量的**精确矩阵指数演化**（`scipy.linalg.expm`），
  不引入任何与实现共享的约定/内部函数；
- T1 权威 V_dd（\|V\| 逐元 rel < 2e-3，相位鲁棒——ARC 与 PI 有合法规范差，
  约半数元素差一个 −1，不影响物理，能谱/动力学已证）；
- T2 权威能级（去公共参考谱差 < 0.5 MHz + vdW 位移）；
- T3 权威动力学（交换振荡 vs expm < 1e-5 + 首峰时刻 = π/(2\|V_ex\|)）；
- T4 单体一致性（V_dd=0 ≡ 单原子引擎，验证张量化）；
- T5 secular 判据回归（远失谐丢弃/近简并保留）。
- pairinteraction 未安装时整组 SKIP（不误报）。

## 6.3 脚本工具（`scripts/`）

- `compare_pairinteraction.py` / `compare_pairinteraction2.py`：本项目（ARC）vs
  pairinteraction 的能级/偶极矩阵元/Zeeman/DC-Stark 交叉验证。
- `setup_arcdata.py`：预生成 `.arcdata/` ARC 缓存。
- `make_stir_rap_states.py`：STIRAP/拉曼存档生成。
- `demo_pair_drive.py` / `demo_unified.py`：两体驱动/统一引擎演示。
- `ac_shift_curve.py`：AC 位移曲线扫描。

`yuchen/` 为两体模拟参考实验目录（pairinteraction + qutip 有效哈密顿/含时演化脚本）。

---

# 七、安装与运行

**环境**：Python 3.10+（实测 3.10）。**依赖**：

```
numpy  scipy  matplotlib  PyQt5  arc   (可选: numba 加速, torch GPU,
                                        pairinteraction 两体权威检测)
```

（`arc` 为 [ARC — Alkaline Rydberg Calculator](https://arc-alkali-rydberg-calculator.readthedocs.io/)，
提供碱金属量子亏损、波函数、偶极矩阵元。首次运行生成项目内 `.arcdata/` 缓存，
可 `python scripts\setup_arcdata.py` 预生成。`pairinteraction`（pip install pairinteraction，
2.x）只用于 `checks/pair_evolve_selfcheck.py`，未安装时该检测自动 SKIP。）

**运行**（在项目根目录内）：

```bash
python 能级.py            # 启动 GUI
python checks\selftest_all.py   # 一键跑全部回归自检（约 90 s）
python checks\evolve_matrix.py  # 动态引擎测评矩阵
```

> Windows 环境如 `python` 解析异常，用绝对路径：`& "D:\python310\python.exe" 能级.py`

## 7.1 打包为 Windows exe（PyInstaller，onedir 首发版）

依赖锁定在 `requirements.txt`（**numba 必需**，ip4/lindblad 演化内核；
torch/pairinteraction/qutip 一律不装——软件本体不用，PyInstaller 会沿 import 图
收集，装了会把巨量无关代码打进 exe。注意 PyPI 上正确的 ARC 发行名是
`arc_alkali_rydberg_calculator`，`arc` 是另一个无关小库）。

```bash
# 一次性：建构建环境（用 D:\python310，Python 3.10）
D:\python310\python.exe -m venv build_venv
# 之后每次发版只需：
build.bat                      # 一键安装依赖(清华镜像) + PyInstaller 打包
```

- 产物：`dist\EnergyLevel\EnergyLevel.exe`（连同整个文件夹一起分发）。
- 入口：`main.py`（ASCII 名转发 `能级.py`，避开中文脚本名构建问题）。
- ARC 数据：`src/__init__.py` 打包模式（`sys.frozen`）会把随包 `.arcdata` 种子
  首次启动复制到 `%LOCALAPPDATA%\EnergyLevel\.arcdata`（可写、持久），
  避免 onefile/只读目录写缓存崩溃；开发模式行为不变。

---

# 八、使用手册

## 8.1 静态页

1. **添加能级组**：输入 `n / l / j`（如 60/0/0.5 添加 60S），点"添加"；重复可添加多组。
2. **设置磁场**：`B (G)` 输入框（如 21.7 G，两页共用）。
3. **添加光场**：点"添加光"，表格里填频率/光强/偏振/作用范围。
4. **右键导出频率**：选跃迁对 + 失谐 Δ → 频率写回表格（内部 MHz 全精度）。
5. **右键扫描光强**：选单一能级 + 目标位移 → 反推光强写回。
6. **观察**：能级图（off/on 镜像）、谱表、谱峰点击联动、雷达图（静态页最下方）。

## 8.2 动态页（单原子）

1. 配置波形（＋高斯/＋方波，折叠组 ✕ 删除）；辅助定参（g/e 态 + 目标Ω定光强 /
   目标Δ定频率）。
2. 选初态、t_max/点数、方法（auto / ip4；auto 无耗散恒走 ip4，有耗散走 lindblad）、自发辐射三档。
3. 运行演化 → 后台线程 + 进度条可取消 → 上图画波形光强、下图画布居演化（含 loss 虚线）。

## 8.3 动态页（双原子）

1. 切"双原子"模式 → 波形面板变为 A/B 两个 tab（各自驱动对应原子）。
2. 设 V_dd 几何：R(μm)、θ(°)、φ(°)。
3. 选 A/B 初态；运行 → 主线程预计算 V_dd（日志显示非零元数）→ 工作线程演化 →
   画 A/B 约化布居。
   （自发辐射在双原子模式强制禁用——pair_evolve 暂不支持 Lindblad。）
4. **交换振荡示例**：A 初态 60S、B 初态 60P、R~5 μm，
   \|60S,60P⟩↔\|60P,60S⟩ 交换拍频 ≈ V_ex（pairinteraction 校验，见 §6.2 T3）。
   纯 V_dd 实验（无驱动波形）因 UI 门槛需先加一个波形才能运行（见 §10.1 限制 3）。

## 8.4 常见提示

- 想观察"某对能级在共振下的大位移"：偏振方向决定哪侧共振（σ⁻ 下 6P3/2 F=3 mF=3
  ↔ 60S mJ=0.5 是 6P3/2 侧大位移；60S mJ=+0.5 用 σ⁺）。
- B=0 时超精细 F 内各 mF 简并，无法从频率区分同 F 不同 mF；B≠0（默认 21.7 G）才可区分。
- **失谐 Δ 正 = 蓝失谐**；导出频率 = 跃迁频率 + Δ。
- 光场做"频移阻塞"须远失谐（Δ≫Ω）：近失谐光场会真实转移布居（Ω²/(Δ²+Ω²) 量级）。

---

# 九、精度、单位与约定速查

| 项 | 约定 |
|---|---|
| 内部频率 | 统一 MHz（float64 全精度）；界面 GHz/THz 智能单位纯显示换算 |
| 失谐符号 | 正 = 蓝失谐（ν_光 > ν_跃迁），全项目统一 |
| RWA 分界 | Ω/ωL < 1e-4 → RWA；≥ → 非 RWA（ip4 保留 cos 反旋转项，第一性原理） |
| 衰变约定 | Γ MHz → 速率 Γ×1e-3/ns；跳变算符 √Γ\|to⟩⟨from\| **不乘 2π** |
| 步长 | dt = min(载波/frac, 0.06/(_CONV·cmax), 2/(_CONV·cmax), t_end/500, 0.05) |
| secular | \|E_rot 差\| < 10·\|V\| 保留（V_dd 旋转系 RWA） |
| AC-Stark 后端 | direct（Floquet 矩阵对角化，唯一；legacy 逐对分流已删 2025-08） |
| 动态近/远分流 | real_field：\|Δ\| < max(near_cutoff, Ω) → 近共振驱动对；否则远失谐 SOS 位移 |
| 存档 | 只存科学数据/参数；meta.version=2（旧档 get() 默认兼容） |
| 权威验证 | 两体 V_dd/能谱/动力学 vs pairinteraction + expm（rel ≤ 2e-3 / 1e-5） |

---

# 十、已知限制与路线图

## 10.1 已知限制

1. **两体 Lindblad 未实现**：`pair_evolve` 的 decays_A/B 显式拒绝；GUI 双原子模式
   自发辐射强制禁用。
2. **两体 secular RWA 不含二阶 vdW 能移**：远失谐元素被丢弃；近简并交换/Förster
   动力学精确。需要 vdW 位移进演化的场景用 `driver.pair_drive_dressed`（静态对角化）。
3. **双原子模式无波形时不能运行**：UI 门槛要求至少一个波形（纯 V_dd 实验需加占位波形）。
4. **Qt 原生堆损坏风险**：GUI 主线程事件流里调用 ARC `getStateLifetime`
   （auto_decays）有堆损坏风险——已通过"门槛只做廉价判断、寿命计算只在运行时算一次"
   规避；ARC 计算与 Qt 控件操作不穿插（见 §4.7 线程安全约定）。
5. **响应谱 UI 暂缓**：`ui_pair.py` 计算层保留，未接入静态页。

## 10.2 路线图（TODO）

1. 整理 UI 管理，权责分明，更好控制尺寸（static_page 可能删掉更好）。
2. 开发任意脉冲模拟，实现外部脉冲文件载入。
3. 两体（pair_evolve 已接 pairinteraction 权威检测）：两体 Lindblad 衰变、
   二阶 vdW 能移入演化、GUI 双原子自发辐射档位启用。
4. 动态模拟两体效应（GUI 已通：主线程预计算 V_dd → 工作线程演化）。

---

*README 按当前项目层次架构重写（2025-08）；核心算法记录见 §5，与 `src/core` 源码逐条对应。*
