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
| AC-Stark | Floquet 矩阵对角化（`direct` 后端，第一性原理，唯一后端；legacy 逐对算法已于 2025-08 删除）；光子块阶数与偶极伙伴窗口**面板可调**（默认 n=4 / window=3），静态页与动态页共用（2025-09） |
| DC-Stark | 二阶微扰 SOS（含超精细/Zeeman 子能级），E² 标度已验证 |
| 两体耦合雷达图 | 固定 φ 扫描 R×θ，极坐标躺平热图显示 \|V_ex\|；跃迁对由 off/on 谱点击选择 |
| 光场表格 | 多光管理：频率/光强/偏振/作用范围；右键导出频率/扫描光强；椭圆偏振 |
| 失谐约定 | **正 = 蓝失谐**（ν_光 > ν_跃迁），全项目统一 |
| 单原子动态模拟 | 独立引擎：`evolve.py`（**ip4** 相互作用绘景 RK4，默认 / lindblad）+ `evolve_exact.py`（**exact** 第一性原理：实验室系无 RWA/无旋转系/无近远分流，Sambe 光子块 + 包络自适应，误差由 tol 控制，见 §5.3.7）；floquet、pwc 已封存至 `src/_archive/evolve_floquet_pwc.py`（2025-08），见 §5.3 |
| 双原子动态模拟 | `pair_evolve.py`：H_A(t)⊗I + I⊗H_B(t) + V_dd（ip4：secular RWA）+ `evolve_exact_pair.py`（**exact**：完整 V_dd 无 secular，见 §5.3.8），见 §5.4 |
| 波形管理面板 | 动态页直接配置可折叠波形组：高斯/方波包络、定光强/定频率辅助 |
| 自发辐射 | 动态引擎内建 **Lindblad 主方程**（decay 通道：指数衰变/显式落点/纯退相）；**双原子模式亦支持**（寿命/落点=丢失，NonHermitian 确定性演化，2025-09） |
| 存档 | 「文件→保存/恢复状态」；**只保存科学数据与参数，不保存显示参数**（见 §4.8） |
| 回归自检 | `checks/selftest_all.py` 一键全跑（约 2 分钟）：core 22 + 耗散 18 + 单体 exact 11 + 两体 exact 10 + UI 35 + 两体 12（pairinteraction 未装时权威段 SKIP） |

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
│  evolve_exact.py 第一性原理含时引擎：method='exact'，无 RWA/旋转系/分流│
│  evolve_exact_pair.py 第一性两体引擎：完整 V_dd + Krylov 缺陷修正  │
│  pair_evolve.py 两体含时演化：张量基/ secular RWA / RK4       │
│  waveforms.py 波形数据层：Wave/包络/build_fields             │
│  driver.py    统一驱动引擎：静态→驱动→RWA→演化/响应谱          │
└─────────────────────────────────────────────────────────────┘
        │ 交叉验证（权威参考）
┌─────────────────────────────────────────────────────────────┐
│ 第 3 层  检测与脚本  checks/ scripts/                         │
│  selftest_all（5 个核心回归）· _archive（按需归档自检）        │
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
- **AC-Stark 精度控件**（B/E 行右侧）：**窗口**（偶极伙伴 Δn，1–5，默认 3）与
  **光子块**（Floquet 阶数，1–12，默认 4）。两者经 `core.set_ac_stark_config` 写入
  全局配置，静态页（能级图/位移/扫描）与动态页（`real_field` 位移）**共用**。
  提示：窗口=1 快速定位（~60 态），窗口=3 全伙伴（~400 态，冷启动数十秒，结果缓存后同参秒回）。
- 光场表格：序号 / 频率 (GHz) / 光强 (mW/cm²) / 偏振 / 作用范围；单击编辑、右键菜单
  （导出频率、扫描光强、椭圆偏振参数、删除）。
- 关注跃迁 / 初态（响应谱参数，暂时隐藏 UI，序列化保留为科学参数）。
- `serialize_state()`：只导出业务状态（B/E/window/n_photon/lights/focus/psi），**不含纯显示状态**。

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
>
> **2026-08-31 修订**：按 T19「功能驱动重命名」更新文件/函数名——
> 旧名（core.py / pair_core.py / evolve.py / evolve_exact.py 等）已拆分/改名，
> 兼容别名层在 `src/__init__.py` 与 `src/core/__init__.py` 保留，外部调用零改动。

## 5.0 层内结构与单位约定

### 5.0.1 文件职责（T19 功能命名后）

| 文件 | 职责 | 依赖 |
|---|---|---|
| `static_atom.py` | 单原子静态物理（能级/偶极/AC/DC-Stark/频谱） | ARC 库（原子数据） |
| `static_pair.py` | 两体静态（V_dd/vdW/交换/雷达图） | static_atom |
| `atomic_levels.py` | 能级构造（make_levels/gaussian/square/step） | static_atom（偶极） |
| `field_build.py` | 场构造（field/group_field/real_field/arbitrary_field） | static_atom |
| `evolve_rwa.py` | 单原子 RWA 演化（ip4 相互作用绘景 / Lindblad / auto） | static_atom + field_build |
| `evolve_arbitrary.py` | 任意波形演化（实验室系 RK4，第一性原理） | static_atom |
| `evolve_sambe.py` | 单原子 Sambe 光子块第一性引擎（method='exact'）+ reference | static_atom + sambe |
| `evolve_sambe_pair.py` | 两体 Sambe 第一性引擎（method='exact' 双原子） | evolve_sambe + static_pair + tensor |
| `evolve_twofield.py` | 两场引擎（光场旋转系 + 微波）：`evolve_hard`（硬算）/ `evolve_two_field`（微波 Sambe） | evolve_sambe + hard |
| `hard.py` | hard 引擎内核（单原子/两体，numba 加速，cos(φ(t)) 任意相位） | dop853 + evolve_sambe |
| `pair_evolve.py` | 两体 RWA 演化（secular） | evolve_rwa + static_pair + tensor |
| `waveforms.py` | 波形数据层（Wave/GaussianEnv/SquareEnv/ArbitraryWave） | — |
| `dop853.py` | DOP853 核心层（rhs 参数化驱动，任意 N） | hard（系数） |
| `sambe.py` | Sambe 光子块几何核心（photon_geometry，任意 N） | — |
| `tensor.py` | 两体张量化核心（tensor_states/场镜像/V_dd 注入/约化/丢失） | evolve_rwa（decay） |
| `driver.py` | 统一驱动引擎（RWA/响应谱/dressed） | static_atom + static_pair |

> 旧名兼容：`src/core/__init__.py` 增量别名层（core→static_atom、
> pair_core→static_pair、evolve_exact→evolve_sambe、evolve_exact_pair→
> evolve_sambe_pair、evolve_two_field→evolve_twofield、hard_numba/hard_pair→hard）；
> `src/__init__.py` re-export 顶层旧名——**UI/GUI/checks/scripts 引用零改动**。

### 5.0.2 单位约定（全层统一）

| 量 | 单位 | 备注 |
|---|---|---|
| 能级 / 耦合 / 失谐 / Rabi | **MHz** | 内部 Python float64 全精度 |
| 频率（光参数） | Hz | 界面 GHz/THz 显示层换算 |
| 偶极矩阵元 | 原子单位 a.u.（e·a₀） | ARC 返回值 |
| 光强 | mW/cm² | 换算 W/m² 因子 10 |
| 时间（演化） | ns | 角频率 rad/ns |
| 几何 | 间距 μm，角度 ° | |

关键换算常数（`static_atom.py` / `evolve_rwa.py` / `evolve_sambe.py`）：

```
MU_B_MHZ_PER_G    = μ_B/(h·1e10)     # MHz/G
_CONV             = 2π × 1e-3        # MHz·ns → rad（角频率换算）
MW_PER_CM2_TO_W_PER_M2 = 10.0        # 光强换算
A0 = Bohr 半径（m）; E_CHARGE = 基本电荷（C）
```

## 5.1 `static_atom.py` 单原子静态物理

### 5.1.1 物理模型总览

1. **能级**：里德堡态（n ≥ 20）只画 fine-structure 的 m_J 子能级（忽略核自旋）；
   低激发态在 |m_J, m_I⟩ 基下按 m_F 分块数值对角化超精细 + Zeeman 哈密顿量。
2. **AC-Stark**：唯一后端 = `direct`（Floquet 直接矩阵对角化，第一性原理；
   legacy「逐伙伴解析/微扰」已于 2025-08 删除）。
3. **跃迁频谱**：子能级间全部偶极允许跃迁，强度用爱因斯坦 A 系数。
4. 适用条件：静态（稳态）物理——能级图、寻址光 AC 位移、DC 位移、谱线。
   与动态演化解耦（§1.3）；动态页不算斯塔克能移。

### 5.1.2 能级构造算法

**里德堡态（n ≥ `RYDBERG_N_MIN` = 20）**：

```
E(n,l,j,m) = E0(n,l,j) + g_J(l,j) · μ_B/ħ · B · m
```

- 每个 m_J 一条子能级；核自旋忽略（超精细对里德堡态可忽略）。
- 实现：`atomic_levels.make_levels`（见 §5.3）→ `static_atom._base_level_rows` /
  `_sublevel_energy`；g_J 由 ARC `getLandegjExact` 缓存（`ArcCache.gJ`）。

**低激发态超精细 + Zeeman（数值对角化）**：`get_hyperfine_levels(cache, n, l, j, B)`

在 |m_J, m_I⟩ 基（`mJ_list × mI_list`，I 取自 `atom.I`）下，哈密顿量：

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

- 偶极算符只作用于电子（Δm_J = q）；缺 mJ 匹配会把「与其他 mJ 态的耦合」错误归到本对。
- `transition_strength_q = |dipole_amplitude|²`（谱线/驱动共用同一矩阵元逻辑）。

**Rabi 换算**（`field_e0_sq` / `rabi_mhz` / `intensity_mwcm2_from_rabi`）：

```
E0² = 2·I/(ε₀·c)              # I 为 W/m²
Ω = E0·|d|/ħ                   # d 转 SI：|d|·e·a₀
Ω_MHz = Ω/(2π)/1e6             # Ω/2π 周频；共振 π 脉冲 = 1/(2Ω) µs
```

### 5.1.4 AC-Stark 物理公式

AC-Stark 唯一后端 = direct（Floquet 矩阵对角化，§5.1.5）。下述物理公式为其内化内容：

- **近共振 dressed 两能级精确解**（所有失谐区成立：远失谐回微扰 Ω²/4Δ，
  共振给 Autler-Townes ±Ω/2）：

```
ΔE源 = ½(sign(Δ)·√(Δ²+Ω²) − Δ)      （Δ = 2π·失谐，Ω = Rabi，均 rad/s）
ΔE目标 = 与之严格反对称
```

- **远失谐非 RWA 二阶 SOS**（rotating + counter-rotating 都保留；只留 rotating 会把
  里德堡-里德堡对位移高估 ~10⁴ 倍）：

```
ΔE = −(E0²/4ħ) · |d|² · 2ω_ki/(ω_ki² − ω²)     （ω_ki = 子能级能量差角频率，带符号）
```

  由 `far_sos_shift_mhz` 实现；目前仅作 `B_gauss=None` 旧调用时的回退
  （动态位移后端已统一为 direct，2025-09）。
- **椭圆偏振**（`elliptical_coeffs(α,θ,ε)` → 球基复系数 (c₊₁,c₀,c₋₁)，Σ|c|²=1）：
  传播 k=(sinα,0,cosα)，垂直基 e1/e2，Jones 矢量到球基变换；耦合按各 q 分量
  `|c_q|²` 加权（同一 (m,m′) 对只能经唯一 q 耦合，无干涉）。

### 5.1.5 AC-Stark：direct 后端实现（Floquet 矩阵对角化）

`direct_ac_stark_shifts(cache, groups, address_cfg, B_gauss,
                        n_photon=None, sos_n_window=None)`：

**基组**：全部已添加组子能级 + 各组偶极伙伴（|Δn| ≤ `DIRECT_SOS_N_WINDOW`=3）。

**Floquet 哈密顿量**（光子块 M = 2·n_photon+1；默认 n_photon=4 → 9 块）：

```
H_F = Σ_n (H0 + n·ħω)⊗|n⟩⟨n|  +  (V/2)⊗(|n+1⟩⟨n| + |n⟩⟨n+1|)
```

- 对角块 `H0 + (p−n_photon)·freq_mhz·I`；相邻块耦合 `C = E0·C_unit`，
  其中 `C_unit_ij = −d_ij·(e·a0)/(2h·1e6)`（MHz，单位 E0），`E0² = 2I/(ε0c)`。
- 一次 `np.linalg.eigh` 对角化 → 自洽包含近共振 dressed + 远失谐反旋项 + 多态耦合 + 超精细相干性。
- 适用条件：任意失谐（近/远统一处理）；强光/多态/超精细相干场景 n_photon 需 ≥4。

**两级基组收敛**（window 面板）：AC 位移 = 全部偶极伙伴中间态求和，远伙伴按
~1/ΔE³ 衰减 → window=1 粗算、2/3 精算，两级差即截断误差上界（window=2→3 差 <0.3%）。

**结果缓存**：`_DIRECT_SHIFT_CACHE`（键 = 基组/B/偏振/阶数/频率/强度）——同参直接
返回；改频/改强只重标定耦合并重对角化。**面板全局配置**（`set_ac_stark_config`）。

**位移归属**（`_floquet_shifts`）：本征态对光子块 0 的投影重叠确定归属；共振 50/50
劈裂（重叠 <30%·max）时取离裸能量最近的分支。

**光强扫描**（`direct_ac_stark_sweep`）：耦合 C ∝ E0 ∝ √I，基组/对角块不随 I 变——
只构建一次，对每个 I 重标定耦合并重对角化，快一个量级。

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

**跃迁对查找**（`collect_transitions`）：组对间取**失谐最小**的耦合子能级对
（与当前寻址光频率最匹配），含超精细/Zeeman 子能级失谐与 Rabi（rad/s）。

## 5.2 `static_pair.py` 两体静态

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

### 5.2.3 pair 哈密顿与本征

- `pair_hamiltonian`：`H = diag(E_pair) + V_dd`（子空间稠密复矩阵）。
- `pair_eigen`：`np.linalg.eigh` → evals（升序 MHz）+ evecs（列=本征态）。
- `pair_component(res, a_row, b_row)`：各本征态在 |a⊗b⟩ 的权重 |⟨ψ|ab⟩|²。

### 5.2.4 实用提取

- **交换耦合**（`exchange_coupling`）：一阶项 ⟨a|V_dd|b⟩（MHz）——直接查矩阵元不需对角化；
  默认 60S↔60P 交换对 |SP⟩↔|PS⟩。
- **vdW 位移**（`vdw_shift`）：对角化后从 |ref⊗ref⟩ 支（overlap 最大本征态）提取
  `shift = E_支 − 2E_ref`（二阶项 ∝1/R⁶）。
- **耦合强度雷达图**（`coupling_strength_map`）：固定 φ 扫 R×θ，
  `|V_c| = |⟨tgt1⊗tgt2|V_dd|src1⊗src2⟩|`。

## 5.3 `atomic_levels.py` + `field_build.py` 能级与场构造

### 5.3.1 能级构造（`atomic_levels.py`）

- `make_levels(cache, specs)`：规格列表 → 能级行（含 n/l/j/m/mj/mF/kind/E_mhz/amps），
  为全部引擎（RWA/Sambe/hard/两体）的基组来源。
- `gaussian` / `square` / `step`：包络函数（历史遗留，演进到 `waveforms.py` 的
  GaussianEnv/SquareEnv 数据层；此处保留供兼容）。

### 5.3.2 场构造（`field_build.py`）

三种场工厂（供 `evolve_rwa` 的演化引擎消费）：

- `field(g, e, Ω, waveform, Δ)`：单对驱动，A 为 0/1 单耦合；
- `group_field(cache, states, ga, gb, pol, Ω)`：组驱动，A 各 (m_g,m_e) 耦合
  = |dme|/max|dme|（相对强度来自 static_atom 偶极矩阵元）；
- `real_field(cache, states, freq, intensity, pol, B_gauss=None, ...)`：真实光场→
  近共振合并成多对驱动场；AC 位移后端 = direct（Floquet，见 §5.1.5）；
  `B_gauss=None`（旧调用）回退 `far_sos_shift_mhz` 逐对求和。近共振 Rabi 驱动
  矩阵 A 仍按 near_cutoff 筛选（RK4 步长的 RWA 硬约束）。
- `arbitrary_field`：任意波形场（实验室系，无载频概念）——消费方为 `evolve_arbitrary`。

## 5.4 `evolve_rwa.py` 单原子 RWA 演化

### 5.4.1 物理模型与场算符

```
H(t) = H0 − Σ_i ref_i·P_i + Σ_k [ (Ω_k(t)/2)·A_k ] + BS     （RWA，旋转系）
H(t) = H0 + Σ_k Ω_k(t)·cos(ω_L,k·t)·A_k                      （非 RWA 精确，实验室系）
```

- H0：对角 = 各能级能量 E_mhz（用户输入，可含 Zeeman 等任意静态）。
- **统一场算符**：每个物理光场 = 一个算符 `M_k = A_k + diag(ac_k)`——
  近共振驱动对的 Rabi 耦合矩阵 A_k 与远失谐态的二阶 AC 位移 diag(ac_k) 同属
  一个场，随包络 env_k(t) 一起时变（避免"常数 AC 位移全时段生效"）。

### 5.4.2 旋转参考系算法

**级联 RWA 参考**（`_assign_reference` / `_rwa_reference`）：

以 (n,l,j) 组为单位，从能量最低组 ref=0 出发，沿每个 RWA 场的耦合对传播
`ref[高能组] = ref[低能组] + ω_L,k`——链式/树状驱动图使每个驱动对在旋转系里共振，
正确处理**共享能级**的多驱动（如光频 6P↔60S 与微波 60S↔60P 同开时 60S 是中间态）。
闭环不一致的驱动留下残留拍频 δ（RWA 丢弃并警告，误差 ~Ω/|δ|）。

**非 RWA 场共享参考**：全部耦合组共享同一参考（e 跟随 g），使微波耦合为纯 cos 快项
（比全非 RWA 快 ~10⁴ 倍）。

**旁观态归一**（`_e_rot`）：旁观态（无耦合）E_rot 归零；耦合态减共同参考——
避免光频量级参考使 u(t) 相位爆炸。

### 5.4.3 算法 1：相互作用绘景 RK4（`_evolve_rwa`，第一性原理，唯一纯幺正算法）

> 动态分析只用 ip4（单体 RK4 足够快）；精确 Floquet 与 pwc 已封存至
> `src/_archive/evolve_floquet_pwc.py`。

适用：任意时变包络；光频（Ω/ωL < `IP4_RWA_THRESH`=1e-4）→ RWA，微波等强驱动 → 非 RWA。

**相互作用绘景**：H0 = diag(E_rot) 相位精确除去，积分
`Ṽ(t) = Σ s_k(t)·u·M_k·u†`（u = e^{iE_rot t}）。

**全 RWA + 常数包络 → 一次对角化精确解**：`H = diag(E_rot) + Σ(Ω/2·c)·A + c·diag(ac)`，
`eigh` → `ψ(t) = V·diag(e^{−iεt})·V†·ψ0`（秒出）。

**步长三重约束 + 下限/上限**：

```
w_max   = 最快真实振荡频率（RWA 场 → |ΔE_rot|；非 RWA → |ΔE_rot| + ω_L）
cmax    = max Ω·‖A‖
dt_carrier = 2π/(_CONV·w_max)/frac            # frac=20 → ~3e-5 误差
dt_acc     = 0.06/(_CONV·max(cmax,1e-6))
dt_stab    = 2.0/(_CONV·max(cmax,1e-6))
dt         = min(dt_carrier, dt_acc, dt_stab, t_end/500, 0.05)
```

**numba 稀疏内核**（`_rwa_rk4_core`）：稀疏耦合 CSR + S 预计算到 1/4 步子网格；
无 numba 时 numpy 回退（`_rwa_rk4_core_numpy`）。实测 N=20 高斯脉冲 400 ns → 29 s，
误差 3e-5。

### 5.4.4 算法 2：Lindblad 主方程（自发辐射等马尔可夫耗散）

```
dρ/dt = −i[H(t),ρ] + Σ_k (L_k ρ L_k† − ½{L_k†L_k, ρ})
L_k = √Γ_k |to_k⟩⟨from_k|      （decay(g, Γ, e) 工厂）
```

- **单位约定**：Γ 单位 MHz → 速率 Γ×1e-3 /ns；跳变算符不乘 2π。
- **旋转系下跳变算符时不变**（相位抵消）→ 耗散项与旋转参考选择无关，精确。
- 落点 e=None → 虚拟 loss 态；g==e → 纯退相通道。
- **`lindblad_eig`**：旋转系下 H 恒定 → Liouvillian 本征分解精确解（机器精度）。
- **`lindblad`**：矩阵 RK4（任意时变包络，含非 RWA）。**numba 内核**
  （`_lindblad_rk4_core`）：H 稀疏 + ρ 厄米 → 对易子一次点乘 + rank-1 跳变快速路径。
  实测非 RWA 微波从 numpy ~40 min 降到 ~1 min（42×）。
- **`auto_decays`**：按 ARC 权威寿命自动配置（Γ = 1/τ，落点=丢失，排除里德堡慢衰变）。

### 5.4.5 auto 调度与参考积分器

```
auto（无耗散）：恒 'ip4'（相互作用绘景 RK4）
auto（有耗散）：旋转系 H 恒定 → 'lindblad_eig'；否则 → 'lindblad'
```

参考积分器（内部保留，仅 checks 交叉验证）：`_evolve_reference_lab`（实验室系
solve_ivp 非 RWA，最无偏基准）、`_evolve_reference_hybrid`（光频 RWA + 其余非 RWA，
快 ~10⁴ 倍）。

`rwa_error_estimate`：每驱动 RWA 误差估计 `est_err ≈ Ω/(4ωL)`。

## 5.5 `evolve_arbitrary.py` 任意波形演化

**适用**：ArbitraryWave（无单频、任意时域信号——任意频谱/啁啾/多频叠加）。

```
H(t) = H0 + Σ_k amp_k(t)·V̂_k        （实验室系，无 RWA/无旋转系/无近远分流）
```

- V̂ 含**复相位**（dipole_amplitude 复振幅）——任意信号中不一致的相对相位正确干涉。
- 步长受信号 Nyquist 约束（`f_max_mhz = 0.5/dt_min`）——慢（含 THz 高频时锁死），
  属第一性原理必要的代价。
- 实现：`_evolve_arbitrary`（实验室系相互作用绘景 RK4，`_arb_rk4_core`）。
- **注**：hard 路径（§5.8）提供"旋转系载波剥离 + cos(φ(t)) 残差"的更快任意波形
  方案（`evolve_twofield.arbitrary_mw_group`，2026-08-31）。

## 5.6 `evolve_sambe.py` 单原子 Sambe 第一性引擎

### 5.6.1 物理模型（实验室系 TDSE）

```
H(t) = H0 + Σ_k Ω_k·env_k(t)·cos(ω_k t)·A_k        （实验室系，无模型近似）
```

无 RWA、无旋转系、无近/远失谐分流、无静态能移注入：AC 位移、Bloch–Siegert、远失谐
效应全部从精确传播中自然涌现。Ω_k = static_atom.rabi_mhz；A_k 为全偶极允许对耦合矩阵。
**场构造**：`build_fields_exact(cache, states, waves)`（无 near_cutoff、无 Floquet 注入）。

### 5.6.2 数值算法（`evolve_sambe`；含耗散时转 `evolve_reference`）

1. **Sambe (t,t') 光子块扩展空间**：每个相异载频一个光子维（阶数 M 由常数包络探针
   递增收敛）；块对角承载载波相位 → 成本与段内载波周期数无关。
2. **包络自适应分段**：段内包络冻结于中点值，段传播 = 冻结哈密顿精确指数；
   链状态全程扩展空间连续传播（边界不重嵌入，无微运动泄漏），仅输出时刻投影物理态。
3. **Duhamel 缺陷修正**：`δ(t) = −iC∫ U_f(t,s)·[K(s)−K(c_mid)]·ψ(s) ds`，
   本征基可分离 + Gauss 求积（节点 32/64/128，节点加密差即误差估计）+ 两级 Picard。
4. **误差控制**：单旋钮 `tol`（默认 1e-6）；`error_report` 附段数/残差/范数/光子块数。
5. **参考积分器** `evolve_reference`：H0 相互作用绘景 + DOP853（幺正）/BDF（Lindblad）。
6. **常数包络快速路径**：单次扩展空间对角化，成本与时长无关。

### 5.6.3 光子块阶数（逐载频 M）

`_m_analytic`：`(r_g/4)^{2(M_g−1)} ≤ tol`（r_g = 近共振对最大 Ω/ω）。光学 r~1e-7 → M=1
（3 块）；微波 r~5e-4 → M=2（5 块）。离共振巨矩阵元（如光学频 60S↔60P Ω~19 GHz）
不参与判定（`_group_res_peak`，近共振带宽 `_M_ANALYTIC_BW_MHZ`=20 GHz）——其物理贡献
是二阶微扰，光子块阶数只由共振对决定。逐载频 M=(1,1,2) → 45 扇区（统一 M 会 125 扇区）。

**实测**：两能级高斯 400 ns tol=1e-7 → 与 DOP853 参考一致至 3.5e-8；N=20 微波多能级
400 ns / tol=1e-6 → ≈22 s、与参考一致至 2.1e-6。

## 5.7 `evolve_sambe_pair.py` 两体 Sambe 第一性引擎

### 5.7.1 模型（无 secular、无 RWA、无旋转系、无近/远分流）

```
H_2d(t) = [H_A0 + Σ_f Ω_f env_f(t)cos(ω_f t)·d̂_A] ⊗ I
        + I ⊗ [H_B0 + Σ_f Ω_f env_f(t)cos(ω_f t)·d̂_B] + V_dd
```

V_dd 完整保留（不做 `pair_evolve` 的 `_secular_filter`）：近简并交换、远失谐 vdW 通道
全部自然涌现。A/B 同频场共享一个光子维。

### 5.7.2 数值算法

1. **两体 Sambe 扩展空间**：索引 idx = k + N²·p；K_static（E_pair + 光子偏移 +
   V_dd 块对角 + 寿命 NonHermitian）与逐频组光子邻接耦合一次预计算；
   扩展维 ≤ `_DENSE_MAX`（700）自动转稠密。
2. **冻结中点段传播**：稀疏 expm_multiply（Krylov）；稠密自动厄米判定本征分解
   （绝对容差判厄米，能量标度 1e6 MHz）。
3. **Krylov Duhamel 缺陷修正**：Gauss 节点阶梯（16/32/64），包络近平坦时跳过。
4. **寿命耗散**：复用 `tensor._pair_loss_vectors` → NonHermitian 对角 −iΓ/2。
5. **返回契约与 `pair_evolve` 同键**（GUI 双原子直接复用）+ error_report。

**实测**：交换场景首峰 10.30 ns ≡ π/(2C·|V_ex|)，P_max=1.0000；vdW 远失谐通道 ≡
两体实验室系稠密 DOP853（2e-10）；V_dd=0 ≡ 单原子 exact 直积（1.4e-7）。
**大基组现状**：扩展维 >700 走稀疏 Krylov——实测 N=20/原子（N²·P=3600）>25 min
未出结果，实用不可行；待"两场两体变体"（光场旋转 + 微波-only 光子块，dim=2420 稠密）。

## 5.8 `evolve_twofield.py` 两场引擎（光场旋转系 + 微波）

### 5.8.1 核心思想（用户提议 2026-08-29）

把电磁场按频段拆开：**光场用旋转系解析吸收载波，微波硬积分（或 Sambe 光子块）**——
消除 THz 光学载波使 DOP853 步长爆炸的问题，只解析 17 GHz 微波载波。

### 5.8.2 场分类（`split_by_freq`）

`MW_MAX_MHZ = 1e5`（100 GHz）：载频 ≤100 GHz 视为微波，>视为光场。
（2026-08-31 规划：改为自适应阈值，基于能级间隔分布，见 t20。）

### 5.8.3 算法 A：`evolve_hard`（微波硬算，零近似）

```
H(t) = diag(E_rot) + Σ_opt c_f(t)·exp(i·δ_f·t)·A_opt + Σ_mw c_f(t)·cos(φ_f(t))·A_mw
```

- **光场**：旋转系（6P 流形 +ω_opt，耦合停在失谐）；光学 RWA（反旋 Bloch–Siegert
  相位 4.4e-5 rad ≪ tol）；**多光场拍频** exp(i·δ_f·t)（t10 验证：exp 正确、cos 错误 95%）。
- **微波**：cos(φ_f(t)) 完整载波**硬积分**（φ 任意：定频=chirp 特例，t18 统一路径）——
  多光子、Bloch–Siegert、任意谐波、任意时域信号天然全含，零近似。
- **V_dd**（两体）：实验室系静态 → IP 相位扭曲 e^{iC·(E_lab[k]−E_lab[l])t}，按 dE 分组。
- 误差：DOP853 自适应（rtol/atol 显式）；RHS 厄米 → 范数机器精度。
- 性能：单原子 620ns 4s（45×numba）、两体 1500ns 84s（7×）；步长由最高瞬时频率决定。

**核心层**：`hard.py`（`evolve_hard_fast`/`evolve_hard_pair`，numba）+ `dop853.py`
（rhs 参数化 DOP853 驱动，任意 N）。

### 5.8.4 算法 B：`evolve_two_field`（微波 Sambe 光子块）

- 微波进扩展空间（M=2，5 块，多光子显式）；光场以 `(A_opt⊗I_P)` 块对角进入。
- 620ns 三路（exact/hard/sambe 两场）一致 ≤6.3e-5；单原子 620ns 2.0s（最快）。
- 两体扩展（dim=2420 稠密）为 t15 路径 A（qwen 额度恢复后）。

### 5.8.5 任意波形微波（`arbitrary_mw_group`，2026-08-31）

ArbitraryWave → hard 兼容组：Hilbert 解析分解 `amp(t) = env(t)·cos(φ(t))`（精确 ~1e-11）
+ 全偶极实振幅 A + env/φ 插值——UI 的 ArbitraryWave 无缝喂 hard。

## 5.9 `hard.py` + `dop853.py` hard 内核

### 5.9.1 DOP853 核心层（`dop853.py`）

- `_dop853_driver(rhs, ...)`：scipy solve_ivp(DOP853) 主循环的 numba 移植
  （SAFETY=0.9、MIN/MAX_FACTOR、order=7 select_initial_step、RMS×complex 误差范数、
  contd8 稠密输出）——**rhs 参数化**（numba first-class 函数参数），任意 N 通用。
- `_dop853_step` / `_dop853_dense_interp`：单步与稠密插值。
- 系数（_A/_C/_B/_E3/_E5/_D）复用 `hard.py`（scipy 1.15.2 移植，2.9e-10 基线验证）。
- `_env_tab` / `_env_at`：包络均匀查表 + 线性插值（误差 ~3e-10 ≪ rtol）。
- **核心-调度分层**：单体/两体 RHS 各一个 njit，共用一套驱动（消除 ~450 行重复）。

### 5.9.2 hard 引擎（`hard.py`）

- `evolve_hard_fast`（单原子）/ `evolve_hard_pair`（两体）：numba RHS + dop853 驱动。
- `build_hard_rhs`：fields_meta → njit RHS 闭包（光场 exp 拍频 + 微波 cos(φ(t)) 查表）。
- 性能：620ns 单原子 3.8-4.0s（45.6×）、两体 1500ns 83.6s、V_dd 过滤 `_VDD_MIN_FRAC=1e-4`
  （5184→144 物理元素）。

## 5.10 `pair_evolve.py` 两体 RWA 演化

### 5.10.1 张量基组与场

```
H(t) = [H_A(t) ⊗ I_B] + [I_A ⊗ H_B(t)] + V_dd
```

两体基组 = 单体笛卡尔积（`tensor.tensor_states`）；场镜像 `_tensor_field_atom`
（M_A⊗I_B / I_A⊗M_B）；初态 `tensor._as_product_psi`；旋转参考 `pair_rotate_frame`。

### 5.10.2 V_dd 旋转系 secular RWA

`_secular_filter(V_dense, E_rot_2d, ratio=SECULAR_RATIO=10.0)`：

```
V_{αβ} 振荡频率 = |E_rot_β − E_rot_α|（E_rot 含 V_dd 对角位移）
|Δ| < ratio·|V| → 保留；否则高振荡项平均为零 → 丢弃
```

> 判据必须用**真实旋转系能量差**（含 V_dd 对角位移）——误用旋转参考之差会把
> 远失谐 vdW 元误当 secular 保留（T5 回归钉死）。

### 5.10.3 演化主循环（`_pair_rwa`）

- RWA 分类（`_is_rwa`）；对角化精确路径（全 RWA + 常数包络 + 无振荡非 RWA 场）；
- 步长三类场分别计入 w_max；旁观态归一（同 `_e_rot`）；
- numba 并行 RK4（`_pair_rwa_jit`，~45×）；两体寿命耗散（NonHermitian）；
- 远失谐散射 depump（report-only）。

**模型局限**：secular RWA 丢弃远失谐元素 → 二阶 vdW 能移不在演化哈密顿中。

## 5.11 `driver.py` 统一驱动引擎

### 5.11.1 分层设计

```
静态层 pair_static    ：非 driver 因素 → H_static（含 vdW/交换）
驱动层 driver_operator：driver → (D, ω, Ω)
化简层 rwa_hamiltonian：旋转系 RWA → H_rwa
演化层 evolve          ：定常哈密顿一次对角化 → 布居演化
```

### 5.11.2 RWA 化简 + dressed 投影 + 线性响应

- `rwa_hamiltonian`：每态按光子数减 n_k·ω（`_photon_numbers` BFS，支持双光子级联）。
- `_dressed_hamiltonian`：静态 H 全对角化 → dressed 能级（含 vdW）→ 驱动投影本征基。
- `response_spectrum_linear`：一次静态对角化 + 黄金规则 `S(ω) = Σ_k |⟨k|D|i⟩|² δ(ω−ΔE_k)`。

## 5.12 `waveforms.py` 波形数据层

- **包络工厂**：`GaussianEnv(center, σ, peak)`、`SquareEnv(start, edge, height, fall_start)`
  （raised cosine 缓变方波，两端导数连续为 0）。
- **`Wave`**：频率 + 光强 + 偏振 + 包络 + 起始偏移；`drive`（g/e 辅助对）仅定参。
- **`ArbitraryWave`**：任意时域信号（样品 t_ns/amp，任意频谱/啁啾/多频），
  消费方 = `evolve_arbitrary`（实验室系）或 `evolve_twofield.arbitrary_mw_group`（hard）。
- **`build_fields(waves, states, cache, near_cutoff_mhz)`**：每波形 → `field_build.real_field`。
- **序列化**：`serialize_waves` / `restore_waves`（波形 = 科学数据，参与存档）。

## 5.13 `sambe.py` + `tensor.py` 核心层（重构 2b/3 产物）

- **`sambe.photon_geometry(E, groups, Ms)`**：Sambe 光子块几何（混合进制 stride /
  量子数 / 块对角 / 重构相位 / 中央光子块）——任意 N 通用，`_SambeSetup` 与
  `_PairSambeSetup` 共用（消除几何构造重复）。
- **`tensor.py`**：两体张量化工具——`tensor_states`（基组）、`tensor_fields`
  （场镜像）、`_as_product_psi`（初态）、`_reduce_A/_reduce_B`（约化布居）、
  `_pair_loss_vectors`（寿命丢失）。全部引擎（pair_evolve/evolve_sambe_pair/hard_pair）共用。

---

> 兼容性：全部旧模块名/函数名经 `src/__init__.py` 与 `src/core/__init__.py` 别名层
> 保留——`from src.core import core` / `from src.core.evolve_exact import ...` 等
> 仍可用（T19 后内部代码已改用新名，兼容层只服务外部）。

# 六、检测与脚本层 checks/ scripts/

## 6.1 回归自检矩阵（`checks/selftest_all.py` 一键全跑，约 90 s）

| 自检 | 项数 | 覆盖 | 权威参考 |
|---|---|---|---|
| `core_selfcheck.py` | 22 | 能级/偶极/AC 位移（direct 唯一后端，n_photon 默认 4）/频谱/偏振/far_sos ω→0==DC | 解析公式、ARC 官方函数、direct 自洽 |
| `evolve_dissipative_selfcheck.py` | 18 | 解析衰变 e^{−Γt}、CW/时变、退相、ac 约定、numba≡numpy | 独立 Liouvillian 本征参考、solve_ivp |
| `ui_selftest.py` | 35 | 主窗口/波形面板/动态页/存档往返（含双原子参数、线程安全） | UI 冒烟 + 状态往返断言 |
| `pair_evolve_selfcheck.py` | 12 | 两体 V_dd 矩阵元/能谱/vdW 位移/交换动力学/secular 判据 | **pairinteraction 2.x + 精确 expm**（未装时权威段 SKIP，本地 PASS 6） |

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
   （**自发辐射档位可用**：auto=ARC 寿命（Γ=1/τ，双原子同样按寿命流出，loss_A/B
   曲线画出）；自定义档仅支持"丢失"落点，显式落点提示用单原子模式。）
4. **交换振荡示例**：A 初态 60S、B 初态 60P、R~5 μm，
   \|60S,60P⟩↔\|60P,60S⟩ 交换拍频 ≈ V_ex（pairinteraction 校验，见 §6.2 T3）。
   纯 V_dd 实验（无驱动波形）因 UI 门槛需先加一个波形才能运行（见 §10.1 限制 3）。
   **多 mJ 近简并交换通道**：B 场下 60S/60P 流形存在多条近简并交换通道（同 Δm 组合
   天然共振；异 Δm 通道靠频移光归零），目标通道峰可能被 7 MHz 级竞争通道压低——
   这是全基组物理（参考脚本 4 态模型忽略这些通道），隔离目标对时交换仍精确
   （引擎验证：V=1.45 MHz、ΔE≈0 → 交换峰 1.000）。

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
| exact 引擎 | method='exact'：无 RWA/旋转系/分流；单体 §5.3.7 + 两体完整 V_dd 无 secular §5.3.8；tol 单旋钮（默认 1e-6），误差随 error_report 可见；Rabi 周频约定（π 脉冲 = 1/(2Ω) µs）；单体含耗散转参考路径，两体支持寿命 NonHermitian |
| 衰变约定 | Γ MHz → 速率 Γ×1e-3/ns；跳变算符 √Γ\|to⟩⟨from\| **不乘 2π** |
| 步长 | dt = min(载波/frac, 0.06/(_CONV·cmax), 2/(_CONV·cmax), t_end/500, 0.05) |
| secular | \|E_rot 差\| < 10·\|V\| 保留（V_dd 旋转系 RWA） |
| AC-Stark 后端 | direct（Floquet 矩阵对角化，唯一；legacy 逐对分流已删 2025-08）；默认 n_photon=4、window=3（面板可调，静态+动态共用） |
| 动态位移后端 | real_field：B_gauss 提供时走 Floquet（不分近/远）；far_sos 仅 B=None 回退（2025-09） |
| 动态近/远分流 | real_field：\|Δ\| < max(near_cutoff, Ω) → 近共振驱动对（A 矩阵）；位移侧由 Floquet 统一计算，分流只用于驱动筛选 |
| 存档 | 只存科学数据/参数；meta.version=2（旧档 get() 默认兼容） |
| 权威验证 | 两体 V_dd/能谱/动力学 vs pairinteraction + expm（rel ≤ 2e-3 / 1e-5） |

---

# 十、已知限制与路线图

## 10.1 已知限制

1. **两体 Lindblad 仅"落点=丢失"通道**：寿命/自发性辐射通道（NonHermitian + loss）
   已实现；**显式落点 / 纯退相**（需密度矩阵回写与相干衰减）仍显式拒绝，
   提示改用单原子引擎。
2. **两体 secular RWA 不含二阶 vdW 能移**：远失谐元素被丢弃；近简并交换/Förster
   动力学精确。需要 vdW 位移进演化的场景：method='exact' 两体路径（完整 V_dd，
   §5.3.8）或 `driver.pair_drive_dressed`（静态对角化）。
3. **双原子模式无波形时不能运行**：UI 门槛要求至少一个波形（纯 V_dd 实验需加占位波形）。
4. **Qt 原生堆损坏风险**：GUI 主线程事件流里调用 ARC `getStateLifetime`
   （auto_decays）有堆损坏风险——已通过"门槛只做廉价判断、寿命计算只在运行时算一次"
   规避；ARC 计算与 Qt 控件操作不穿插（见 §4.7 线程安全约定）。
5. **响应谱 UI 暂缓**：`ui_pair.py` 计算层保留，未接入静态页。
6. **多频 RWA 参考传播（待复核边界，2025-09 观察）**：同一能级同时被"光频 RWA 场"
   （如 295.7 THz 近共振 60S↔6P）与微波场驱动时，旋转参考按驱动对平移 ω_L，
   可能使**未参与该光参考的其它态**（如 60P）与平移态（60S）之间出现 ~ω_L 量级的
   旋转系假差。实测（存档 `save_state/问题.json`）：A 光近共振 60S↔6P、B 侧无光时，
   目标交换对能量差 ≈ −295.7 THz → 无交换；B 侧加同频**微弱光**后两侧平移恰好
   抵消、差恢复（现象消失）——**属巧合性对称，非物理**。性质待复核（可能需要
   "多参考相互作用绘景"），列此备忘，未改动既有算法。
   **规避**：此类多频混叠场景可直接用 method='exact'（无旋转系，无此问题）。
7. **exact 引擎范围**：单体（幺正 + 耗散转参考路径）与两体（幺正 + 寿命
   NonHermitian，§5.3.8）已实现；ArbitraryWave、段内耗散超算符（两体显式落点/
   纯退相）、光频载波强驱动大基组的性能优化（振荡求积/numba 内核）待后续；
   强驱动（Ω/ω ≳ 1e-2）场景光子块需求增大，耗时随段数/维度增长，运行前留意日志。
8. **⚠ ip4/Lindblad 在含远失谐旁观者组的基组上静默失效（P1，2026-08 发现）**：
   基组含能量远移的旁观者组（如实测案例：60S/60P₃/₂ 之外挂 6P₃/₂ 超精细组，
   对角能量 −296.7 THz）时，微波驱动**零激发**（π 脉冲处布居不变）、出现非物理
   布居尖峰与 mJ 泄漏，而**守恒诊断（Σpops+loss）保持 3.8e-9 完美**——引擎自检
   不可见。与光学场无关（去光学场对照故障依旧）；疑似旋转参考/步长与
   295.7-THz 级对角跨度的混叠（§10.1-6 多频参考问题的姊妹表现）。机理待诊断
   （复现：`tempsimu/_real_ip4_ref.py`；对照与数据：`tempsimu/_real_census.py`）。
   **规避**：此类场景暂用 method='exact'（Julia 内核已可跑，见
   `docs/Julia引擎交接文档.md`）；GUI 用户在基组含低激发态组且波形含微波时
   需知晓此风险。

## 10.2 路线图（TODO）

1. 整理 UI 管理，权责分明，更好控制尺寸（static_page 可能删掉更好）。
2. 开发任意脉冲模拟，实现外部脉冲文件载入。
3. 两体 Lindblad：显式落点 / 纯退相通道（密度矩阵回写）——当前仅"落点=丢失"。
4. 复核多频 RWA 参考传播（§10.1 条目 6）：必要时实现"光频场独立旋转系
   （多参考相互作用绘景）"，使"光频+微波同挂一个态"场景稳健。
5. 动态模拟两体效应（GUI 已通：主线程预计算 V_dd → 工作线程演化）。
6. exact 引擎扩展：段内耗散超算符（单体替代转参考路径、两体支持显式落点/纯退相）、
   大基组性能优化（缺陷扫描 numba 化/振荡解析求积）；回归稳定后考虑将
   auto 默认切到 exact。（两体 V_dd 接入已完成，§5.3.8，2026-08）

---

*README 按当前项目层次架构重写（2025-08）；核心算法记录见 §5，与 `src/core` 源码逐条对应。
2025-09 更新：AC-Stark 光子块/窗口面板化与全局配置、动态位移 Floquet 化、两体寿命耗散、
判据-演化一致的常数光位移并入、numba 两体内核、回归基线（90–110 s 全绿）。
2026-08 更新：新增第一性原理含时引擎 `evolve_exact.py`（method='exact'，§5.3.7：
Sambe 光子块 + 包络自适应 + Duhamel 缺陷修正，无 RWA/旋转系/近远分流，
误差由 tol 控制并随 error_report 可见；含耗散转参考路径），回归基线新增 11 项。
2026-08 更新（两体）：新增 `evolve_exact_pair.py`（method='exact' 双原子，§5.3.8：
完整 V_dd 无 secular，两体 Sambe 稀疏/Krylov 缺陷修正，寿命 NonHermitian），
GUI 双原子模式方法下拉可选，回归基线新增 10 项（EP1–EP5）。*
