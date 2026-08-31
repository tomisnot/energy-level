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

> 本节精细记录核心科学计算层的**物理模型、数值算法、公式与实现映射**。
> 所有模块无 UI 依赖，可单独导入审查。
>
> **2026-08-31 修订**：按 T19「功能驱动重命名」更新文件/函数名；公式改用 LaTeX
> （GUI 已启用 KaTeX 渲染）。旧模块名经 `src/__init__.py` 兼容别名保留，外部零改动。

---

## 5.0 层内总览

### 5.0.1 分层与文件职责

```
┌ 演化层（含时）────────────────────────────────────────────┐
│ evolve_rwa        RWA 演化（相互作用绘景 RK4 / Lindblad）    │
│ evolve_arbitrary  任意波形演化（实验室系 RK4，第一性原理）     │
│ evolve_sambe      单原子 Sambe 光子块第一性引擎               │
│ evolve_sambe_pair 两体 Sambe 第一性引擎                       │
│ evolve_twofield   两场引擎：evolve_hard（硬算）/ 微波 Sambe   │
│ hard               hard 引擎内核（单/两体 numba + cos(φ(t))） │
│ pair_evolve       两体 RWA 演化（secular）                    │
│ driver            统一驱动引擎（RWA/dressed/线性响应谱）      │
└───────────────────────────────────────────────────────────┘
┌ 静态层（稳态）────────────────────────────────────────────┐
│ static_atom       单原子静态（能级/偶极/AC/DC-Stark/频谱）    │
│ static_pair       两体静态（V_dd/vdW/交换/雷达图）            │
└───────────────────────────────────────────────────────────┘
┌ 构造层 ──────────────────────────────────────────────────┐
│ atomic_levels     能级构造（make_levels）                    │
│ field_build       场构造（field/group/real/arbitrary_field）│
│ waveforms         波形数据层（Wave/Envelope/ArbitraryWave）  │
└───────────────────────────────────────────────────────────┘
┌ 核心层（与维度 N 无关，调度层共用）───────────────────────┐
│ sambe   Sambe 光子块几何（photon_geometry）                 │
│ dop853  DOP853 积分驱动（rhs 参数化，任意 N）               │
│ tensor  两体张量化（基组/场镜像/约化/寿命丢失）             │
└───────────────────────────────────────────────────────────┘
```

旧名兼容：`src/core/__init__.py` 增量别名层（core→static_atom、pair_core→
static_pair、evolve_exact→evolve_sambe、evolve_exact_pair→evolve_sambe_pair、
evolve_two_field→evolve_twofield、hard_numba/hard_pair→hard）；`src/__init__.py`
re-export 顶层旧名——**UI/GUI/checks/scripts 引用零改动**。

### 5.0.2 单位约定

| 量 | 单位 | 备注 |
|---|---|---|
| 能级 / 耦合 / 失谐 / Rabi | **MHz** | float64 全精度 |
| 偶极矩阵元 | a.u.（e·a₀） | ARC 返回值 |
| 光强 | mW/cm² | →W/m² 因子 10 |
| 时间（演化） | ns | 角频率 rad/ns |
| 几何 | μm / ° | |

关键换算常数：

$$C \equiv \texttt{\_CONV} = 2\pi\times10^{-3} \ \text{(rad/ns per MHz)},\qquad \mu_B^{\text{MHz/G}} = \mu_B/(h\cdot10^{10})$$

失谐约定：**正 = 蓝失谐**（ν_光 > ν_跃迁），全项目统一。

---

## 5.1 单原子静态物理（`static_atom.py`）

### 5.1.1 里德堡能级

适用：n ≥ `RYDBERG_N_MIN`（=20），只含 fine-structure m_J 子能级（核自旋忽略）。

$$E(n,l,j,m) = E_0(n,l,j) + g_J(l,j)\cdot \mu_B B\, m / \hbar$$

实现：`atomic_levels.make_levels` → `_base_level_rows` / `_sublevel_energy`；
g_J 由 ARC `getLandegjExact` 缓存（`ArcCache.gJ`）。

### 5.1.2 低激发态超精细 + Zeeman

适用：低激发态（如 6P），在 $|m_J, m_I\rangle$ 基下按 $m_F$ 分块数值对角化：

$$H = A\,\mathbf{I}\!\cdot\!\mathbf{J} + B\,Q + \mu_B B\,(g_J J_z + g_I I_z)$$

- **I·J 矩阵**：对角 $m_J m_I$ + 升降算符项，系数
  $\tfrac12\sqrt{I(I+1)-m_I(m_I\pm1)}\sqrt{j(j+1)-m_J(m_J\mp1)}$；
- **电四极 Q**（仅 J>1/2 且 I>1/2）：$Q = \big[3(\mathbf{I}\!\cdot\!\mathbf{J})^2 + \tfrac32\mathbf{I}\!\cdot\!\mathbf{J} - I(I{+}1)j(j{+}1)\big]/\big[2I(2I{-}1)j(2j{-}1)\big]$；
- **m_F 分块**（块 ≤ 2I+1）：零场对角化标注 F，加 B 再对角化得本征能量/向量；
- 返回 `(F, m_F, E_mhz, mJ_weights, mJ_amps)`——`mJ_amps`（复振幅）供跃迁矩阵元。

实现：`get_hyperfine_levels`。F 标号采用**绝热约定**（Breit–Rabi 惯例，B 连续延拓）。

### 5.1.3 偶极矩阵元与 Rabi

统一入口 $d_q \equiv \langle b|d_q|a\rangle$（`dipole_amplitude`，a.u. 复）：

| a | b | 规则 |
|---|---|---|
| mJ | mJ | 需 $m_b = m_a + q$，直接查 `ArcCache.dme` |
| F | mJ | $\sum_{m_J} \text{amp}_a(m_J)^*\, d_{\text{dme}}(m_J \to m_b)$ |
| mJ | F | 对称求和 |
| F | F | 需 $m_F$ 守恒（核自旋不参与），双分量求和 |

偶极算符只作用于电子（$\Delta m_J = q$）；缺 mJ 匹配会错归耦合（有实测 bug 记录）。

**Rabi 换算**（`field_e0_sq` / `rabi_mhz` / `intensity_mwcm2_from_rabi`）：

$$E_0^2 = \frac{2I}{\varepsilon_0 c},\qquad \Omega = \frac{E_0\,|d|}{\hbar},\qquad \Omega_{\text{MHz}} = \frac{\Omega}{2\pi}\times10^{-6}$$

共振 $\pi$ 脉冲 $= \tfrac{1}{2\Omega}\,\mu\text{s}$（Ω/2π 周频约定，全项目统一）。

### 5.1.4 AC-Stark（direct 后端，Floquet 对角化）

唯一后端 = `direct_ac_stark_shifts`（第一性原理；legacy 逐伙伴算法 2025-08 删除）。
基组 = 已添加组子能级 + 偶极伙伴（$\lvert\Delta n\rvert\le 3$）。Floquet 哈密顿：

$$H_F = \sum_n (H_0 + n\hbar\omega)\otimes|n\rangle\langle n| \;+\; \frac{V}{2}\otimes\big(|n{+}1\rangle\langle n| + |n\rangle\langle n{-}1|\big)$$

- 光子块 $M = 2\,n_{\text{photon}}+1$（默认 4 → 9 块，近共振 n≥2 收敛、n≥4 全频稳定）；
- 相邻块耦合 $C = E_0\,C_{\text{unit}}$，$C_{\text{unit},ij} = -d_{ij}\,e\,a_0/(2h\times10^6)$（MHz/单位 E₀）；
- 一次 `np.linalg.eigh` → 自洽含近共振 dressed + 远失谐反旋 + 多态 + 超精细相干；
- **两级基组收敛**（window 面板）：位移 = 全偶极伙伴求和，远伙伴按 $1/\Delta E^3$ 衰减，
  window=1 粗算、2/3 精算，两级差即截断上界（window=2→3 差 <0.3%）；
- **位移归属**（`_floquet_shifts`）：本征态对光子块 0 投影重叠最大者；共振 50/50 劈裂
  （重叠 <30%·max）取离裸能最近分支；
- **光强扫描**（`direct_ac_stark_sweep`）：$C\propto\sqrt{I}$，基组/对角块不随 I 变——一次构建、
  逐 I 重标定重对角化，快一个量级。

**物理极限**（内化内容）：近共振 dressed 两能级精确解
$\Delta E_{\text{src}} = \tfrac12\big(\text{sgn}(\Delta)\sqrt{\Delta^2+\Omega^2}-\Delta\big)$
（远失谐回 $\Omega^2/4\Delta$，共振给 Autler–Townes $\pm\Omega/2$）；远失谐非 RWA 二阶 SOS

$$\Delta E = -\frac{E_0^2}{4\hbar}\,|d|^2\,\frac{2\omega_{ki}}{\omega_{ki}^2 - \omega^2}$$

（rotating+counter-rotating 都保留；只留 rotating 会高估 ~10⁴ 倍）。`far_sos_shift_mhz`
仅作 `B_gauss=None` 旧调用回退。椭圆偏振 `elliptical_coeffs` → 球基复系数
$(c_{+1}, c_0, c_{-1})$，$\sum|c_q|^2=1$，耦合按 $|c_q|^2$ 加权。

### 5.1.5 DC-Stark 与频谱

**DC-Stark**（`dc_stark_shift_mhz`，二阶 SOS）：

$$\Delta E_i = \sum_k \frac{|\langle k|d|i\rangle|^2 E^2}{E_i - E_k}$$

遍历全偶极中间态（吸收 $m+q$ / 发射 $m-q$ 两方向）+ 核自旋平均；E² 标度已验证。

**谱线**（`spectrum_lines`，爱因斯坦 A 系数）：

$$A = \frac{\omega^3}{3\pi\varepsilon_0\hbar c^3}\,|\langle a|d_q|b\rangle|^2\,(e a_0)^2$$

矩阵元按 m_J 分辨；偏振按 $|c_q|^2$ 加权；`ac_on=True` 用 AC 移后能量。
**跃迁对查找**（`collect_transitions`）：组对间取**失谐最小**（非矩阵元最强）的耦合子能级对。

---

## 5.2 两体静态物理（`static_pair.py`）

### 5.2.1 偶极-偶极 V_dd

$$H_{\text{pair}}(R) = H_1\otimes 1 + 1\otimes H_2 + V_{dd}(R),\qquad
V_{dd} = \frac{1}{R^3}\sum_{qq'} d_{1q}\,T_{qq'}\,d_{2q'}$$

其中 $T = e_q^{\mathsf T}(1-3\hat n\hat n^{\mathsf T})e_{q'}$，$\hat n=(\sin\theta\cos\phi,\sin\theta\sin\phi,\cos\theta)$
为连线方向（θ=0 → n∥z）。θ=0 退化：
$V_{dd}=\tfrac{1}{R^3}\big[-d_{1,-1}d_{2,+1}-d_{1,+1}d_{2,-1}-2d_{1,0}d_{2,0}\big]$。

实现：`_dipole_sparse`（单原子 $D_q$，CSR，Δl=±1/|Δj|≤1 预过滤，LRU 缓存）
+ `_vdd_tensor` + `vdd_sparse`（$V=\tfrac{1}{R^3}\sum_{ab}T_{ab}\,D_a\otimes D_b$，pair 基 k=i·N+j）。

### 5.2.2 基组 / 本征 / 实用提取

- 单原子基 `single_atom_basis`（已添加组 + 偶极伙伴 |Δn|≤n_window）；pair 索引
  `pair_indices`（k=i·N+j，可选 m_total / energy_window 截断）；
- `pair_hamiltonian`：$H=\text{diag}(E_{\text{pair}})+V_{dd}$；`pair_eigen`：eigh；
  `pair_component`：$|\langle\psi|ab\rangle|^2$；
- **交换耦合** `exchange_coupling`：一阶项 $\langle a|V_{dd}|b\rangle$（MHz，直接查矩阵元，
  默认 60S↔60P |SP⟩↔|PS⟩）；
- **vdW 位移** `vdw_shift`：对角化后从 $|\text{ref}\otimes\text{ref}\rangle$ 支提取
  $\text{shift}=E_{\text{支}}-2E_{\text{ref}}$（二阶 ∝1/R⁶）；
- **雷达图** `coupling_strength_map`：$|V_c|=\big|\langle\text{tgt}_1\otimes\text{tgt}_2|V_{dd}|\text{src}_1\otimes\text{src}_2\rangle\big|$。

---

## 5.3 波形与场构造（`waveforms.py` + `field_build.py`）

### 5.3.1 波形数据层（`waveforms.py`）

| 类/函数 | 功能 |
|---|---|
| `GaussianEnv(center,σ,peak)` | $peak\cdot e^{-(\tau-\text{center})^2/2\sigma^2}$ |
| `SquareEnv(start,edge,height,fall)` | raised-cosine 缓变方波（两端导数连续为 0） |
| `Wave` | 频率+光强+偏振+包络+offset；`drive` 辅助对仅定参 |
| `ArbitraryWave` | 任意时域信号（样品 t_ns/amp，任意频谱/啁啾/多频） |
| `build_fields` | 每波形 → `field_build.real_field`（近共振 + AC 位移） |
| `serialize_waves`/`restore_waves` | 波形存档（科学数据） |

### 5.3.2 场构造（`field_build.py`）

统一场算符（Pulser 范式）：每个物理光场 = `M_k = A_k + diag(ac_k)`——近共振 Rabi 耦合
矩阵 $A_k$ 与远失谐 AC 位移 diag(ac_k) 同属一场，随包络 $c_k(t)$ 时变
（避免"常数 AC 位移全时段生效"的物理错误）。

- `field(g,e,Ω,c,Δ)`：单对驱动，A 为 0/1 单耦合；
- `group_field`：组驱动，A = |dme|/max|dme|（相对强度）；
- `real_field`：真实光场 → 近共振多对驱动 + AC 位移（direct 后端）；near_cutoff 筛 A
  （RK4 步长的 RWA 硬约束）；`B_gauss=None` 回退 `far_sos_shift_mhz`；
- `arbitrary_field`：任意波形场（实验室系，无载频），消费方 = `evolve_arbitrary`。

---

## 5.4 RWA 演化（`evolve_rwa.py`）

### 5.4.1 物理模型

$$H(t) = H_0 - \sum_i \text{ref}_i P_i + \sum_k \tfrac{\Omega_k(t)}{2} A_k \quad(\text{RWA, 旋转系})$$
$$H(t) = H_0 + \sum_k \Omega_k(t)\cos(\omega_{L,k} t)\,A_k \quad(\text{非 RWA, 实验室系})$$

适用：任意时变包络；光频（$\Omega/\omega_L < \texttt{IP4\_RWA\_THRESH}{=}10^{-4}$）→ RWA，
微波等强驱动 → 非 RWA。

### 5.4.2 旋转参考系（级联 RWA）

以 (n,l,j) 组为单位，从最低能组 ref=0 出发，沿每个 RWA 场耦合对传播
`ref[高能组] = ref[低能组] + ω_L,k`——正确处理**共享能级**的多驱动（如光频 6P↔60S
与微波 60S↔60P 同开时 60S 是中间态，级联旋转 ref[60P] = ref[60S] + ω_mw）。
闭环不一致 → 残留拍频 δ（RWA 丢弃并警告，误差 ~Ω/|δ|）。非 RWA 场共享参考
（e 跟随 g）使微波为纯 cos 快项（快 ~10⁴ 倍）。旁观态 E_rot 归零（`_e_rot`，
防 u(t) 相位爆炸）。

实现：`_assign_reference` / `_rwa_reference` / `_e_rot`。

### 5.4.3 算法 1：相互作用绘景 RK4（`_evolve_rwa`，纯幺正，默认）

相互作用绘景 $H_0=\text{diag}(E_{\text{rot}})$ 精确除去，积分 $\tilde V(t)=\sum_k s_k(t)\,u\,M_k\,u^\dagger$。

**全 RWA + 常数包络 → 一次对角化精确解**：$H=\text{diag}(E_{\text{rot}})+\sum(\Omega/2\cdot c)A+c\,\text{diag}(ac)$，
$\psi(t)=V\,e^{-i\varepsilon t}V^\dagger\psi_0$（秒出）。

**步长三重约束**：

$$w_{\max} = \begin{cases} |E_{\text{rot},i}-E_{\text{rot},j}| & \text{RWA}\\ |E_{\text{rot},i}-E_{\text{rot},j}|+\omega_L & \text{非 RWA}\end{cases},\quad
dt = \min\!\Big(\frac{2\pi}{C\,w_{\max}/\text{frac}},\ \frac{0.06}{C\,c_{\max}},\ \frac{2.0}{C\,c_{\max}},\ \frac{t_{\text{end}}}{500},\ 0.05\Big)$$

frac=20 → ~3e-5 误差。numba 稀疏内核 `_rwa_rk4_core`（S 预计算到 1/4 步子网格）；
无 numba 时 numpy 回退。实测 N=20 高斯 400 ns → 29 s，误差 3e-5。

### 5.4.4 算法 2：Lindblad 主方程（自发辐射等）

$$\frac{d\rho}{dt} = -i[H(t),\rho] + \sum_k\Big(L_k\rho L_k^\dagger - \tfrac12\{L_k^\dagger L_k,\rho\}\Big),\qquad
L_k = \sqrt{\Gamma_k}\,|to_k\rangle\langle from_k|$$

- Γ 单位 MHz → 速率 Γ×1e-3 /ns；跳变算符**不乘 2π**；
- 旋转系下跳变算符时不变（相位抵消）→ 耗散与旋转参考选择无关，精确；
- 落点 e=None → 虚拟 loss 态（Σpops+loss=1）；g==e → 纯退相；
- **`lindblad_eig`**（旋转系 H 恒定）：Liouvillian 本征分解
  $\rho(t)=\sum_k e^{\lambda_k t}v_k\,(w_k^\dagger\rho_0)$，机器精度；
- **`lindblad`**（任意时变包络）：矩阵 RK4，numba 内核 `_lindblad_rk4_core`
  （H 稀疏 + ρ 厄米 → 对易一次点乘 + rank-1 跳变）；实测非 RWA 微波 40 min→1 min（42×）；
- `auto_decays`：ARC 权威寿命自动配置（Γ=1/τ，落点=丢失）。

### 5.4.5 auto 调度与参考积分器

```
auto（无耗散）：恒 ip4；auto（有耗散）：旋转系 H 恒定 → lindblad_eig，否则 lindblad
```

参考积分器（仅 checks 交叉验证）：`_evolve_reference_lab`（实验室系 solve_ivp 非 RWA，
最无偏）、`_evolve_reference_hybrid`（光 RWA + 其余非 RWA，快 ~10⁴ 倍）。
`rwa_error_estimate`：$e_{\text{est}}\approx\Omega/(4\omega_L)$。

---

## 5.5 任意波形演化（`evolve_arbitrary.py`）

适用：ArbitraryWave（无单频、任意时域信号）。

$$H(t) = H_0 + \sum_k \text{amp}_k(t)\,\hat V_k \quad(\text{实验室系，无 RWA/旋转系/近远分流})$$

$\hat V$ 含复相位（dipole_amplitude 复振幅）——不一致相对相位正确干涉。步长受信号
Nyquist 约束（`f_max=0.5/dt_min`），慢但第一性原理。实现：`_evolve_arbitrary`
（相互作用绘景 RK4，`_arb_rk4_core`）。
**注**：hard 路径（§5.8）提供"旋转系载波剥离 + cos(φ(t)) 残差"的更快任意波形方案。

---

## 5.6 Sambe 第一性引擎（`evolve_sambe.py` + `evolve_sambe_pair.py`）

### 5.6.1 物理模型（实验室系 TDSE，无模型近似）

$$H(t) = H_0 + \sum_k \Omega_k\,c_k(t)\cos(\omega_k t)\,A_k$$

无 RWA/无旋转系/无近远分流/无静态能移注入：AC 位移、Bloch–Siegert、远失谐效应全部
自然涌现。场构造 `build_fields_exact`（无 near_cutoff、无 Floquet 注入）。

### 5.6.2 数值算法（`evolve_sambe`；耗散转 `evolve_reference`）

1. **Sambe (t,t') 光子块扩展空间**：每相异载频一个光子维，块对角承载载波相位
   → 成本与段内载波周期数无关；
2. **包络自适应分段**：段内冻结中点值，冻结哈密顿精确指数传播；链状态全程扩展空间
   连续传播（边界不重嵌入，无微运动泄漏），仅输出时刻投影物理态；
3. **Duhamel 缺陷修正**：
   $$\delta(t) = -iC\int_{t_a}^{t} U_f(t,s)\big[K(s)-K(c_{\text{mid}})\big]\psi(s)\,ds$$
   本征基可分离 + Gauss 求积（节点 32/64/128，加密差即误差估计）+ 两级 Picard；
4. **误差控制**：单旋钮 `tol`（默认 1e-6），`error_report` 附段数/残差/范数/光子块数；
5. **参考积分器** `evolve_reference`：H0 相互作用绘景 + DOP853（幺正）/BDF（Lindblad）；
6. **常数包络快速路径**：单次扩展空间对角化。

### 5.6.3 光子块阶数（逐载频 M）

$$(r_g/4)^{2(M_g-1)} \le \texttt{tol},\qquad r_g = \frac{\Omega_{\text{共振对}}}{\omega_g}$$

光学 r~1e-7 → M=1（3 块）；微波 r~5e-4 → M=2（5 块）。离共振巨矩阵元（如光学频
60S↔60P Ω~19 GHz）不参与判定（`_group_res_peak`，带宽 `_M_ANALYTIC_BW_MHZ`=20 GHz）。
逐载频 M=(1,1,2) → 45 扇区（统一 M 会 125）。实测：tol=1e-7 → 与 DOP853 一致 3.5e-8；
N=20 微波 400 ns / tol=1e-6 → ≈22 s、一致 2.1e-6。

### 5.6.4 两体扩展（`evolve_sambe_pair.py`，公开函数 `evolve_sambe_pair`）

$$H_{2d}(t) = \big[H_{A0} + \textstyle\sum_f \Omega_f c_f(t)\cos(\omega_f t)\,\hat d_A\big]\otimes I \;+\; I\otimes[\cdots] + V_{dd}$$

- 两体 Sambe 扩展空间（idx = k + N²·p）；K_static（E_pair + 光子偏移 + V_dd 块对角 +
  寿命 NonHermitian）与逐频光子邻接预计算；扩展维 ≤ `_DENSE_MAX`(700) 转稠密；
- 冻结中点段传播：稀疏 expm_multiply（Krylov）/ 稠密自动厄米 eigh（绝对容差判厄米）；
- Krylov Duhamel 缺陷修正（Gauss 节点 16/32/64）；寿命复用 `tensor._pair_loss_vectors`；
- 实测：交换首峰 10.30 ns ≡ π/(2C|V_ex|)，P=1.0000；vdW 通道 ≡ 稠密 DOP853（2e-10）；
  V_dd=0 ≡ 单原子 exact 直积（1.4e-7）。
- **大基组现状**：>700 维稀疏 Krylov 慢（N=20/原子 N²·P=3600 >25 min 不可行）——
  待"两场两体变体"（光旋转 + 微波-only 光子块，dim=2420 稠密）。

---

## 5.7 两场引擎与 hard（`evolve_twofield.py` + `hard.py` + `dop853.py`）

### 5.7.1 核心思想（2026-08-29 用户提议）

把电磁场按频段拆开：**光场旋转系解析吸收载波，微波硬积分（或 Sambe 光子块）**——
消除 THz 光学载波导致的步长爆炸，只解析 17 GHz 微波载波。

### 5.7.2 场分类（`split_by_freq`）

$f \le \texttt{MW\_MAX\_MHZ}{=}10^5$ MHz → 微波，否则光场。（t20 规划：自适应阈值。）

### 5.7.3 算法 A：`evolve_hard`（微波硬算，零近似）

$$H(t) = \text{diag}(E_{\text{rot}}) + \sum_{\text{opt}} c_f(t)\,e^{i\delta_f t}\,A_{\text{opt}} + \sum_{\text{mw}} c_f(t)\cos\big(\varphi_f(t)\big)A_{\text{mw}}$$

- **光场**：旋转系（6P 流形 +ω_opt，耦合停失谐）；光学 RWA（反旋 Bloch–Siegert
  4.4e-5 rad ≪ tol）；**多光场拍频** $e^{i\delta_f t}$（t10 验证：exp 正确、cos 错 95%）；
- **微波**：$\cos(\varphi_f(t))$ 完整载波**硬积分**（φ 任意：定频=chirp 特例，t18 统一）——
  多光子、Bloch–Siegert、任意谐波、任意时域天然全含，零近似；
- **V_dd**（两体）：IP 相位扭曲 $e^{iC(E_k-E_l)t}$，按 dE 分组；
- 误差：DOP853 自适应（rtol/atol 显式），RHS 厄米 → 范数机器精度；
- 性能：单原子 620ns **4 s**（45×numba）、两体 1500ns **84 s**（7×）。

### 5.7.4 算法 B：`evolve_two_field`（微波 Sambe 光子块）

微波进扩展空间（M=2，5 块，多光子显式）；光场 `(A_opt⊗I_P)` 块对角。620ns 三路
（exact/hard/两场）一致 ≤6.3e-5；单原子 620ns **2 s**（最快）。两体扩展（dim=2420）
为 t15 路径 A。

### 5.7.5 任意波形微波（`arbitrary_mw_group`，t18）

ArbitraryWave → hard 兼容组：Hilbert 解析分解 $a(t)=c(t)\cos\varphi(t)$（~1e-11）
+ 全偶极实振幅 + env/φ 插值——UI 任意波形无缝喂 hard。

### 5.7.6 DOP853 核心层（`dop853.py`）

`_dop853_driver(rhs, ...)`：scipy solve_ivp(DOP853) 主循环的 numba 移植（SAFETY=0.9、
order=7 初始步、RMS×complex 误差范数、contd8 稠密输出）——**rhs 参数化**（numba
first-class），任意 N 通用。系数（_A/_C/_B/_E3/_E5/_D）复用 `hard.py`（scipy 1.15.2
移植，2.9e-10 基线）。`_env_tab`/`_env_at` 包络查表（~3e-10 误差）。

### 5.7.7 hard 引擎（`hard.py`）

`evolve_hard_fast`（单原子）/ `evolve_hard_pair`（两体）：numba RHS + dop853 驱动。
`build_hard_rhs`：fields_meta → njit RHS 闭包。V_dd 过滤 `_VDD_MIN_FRAC=1e-4`
（5184→144 物理元素）。性能：620ns 3.8-4.0s、两体 1500ns 83.6s。

---

## 5.8 两体 RWA 演化（`pair_evolve.py`）

### 5.8.1 张量基组与场

$$H(t) = \big[H_A(t)\otimes I_B\big] + \big[I_A\otimes H_B(t)\big] + V_{dd}$$

两体基组 = 单体笛卡尔积（`tensor.tensor_states`，k=i·N+j）；场镜像 `tensor.tensor_fields`
（M_A⊗I / I⊗M_A）；初态 `tensor._as_product_psi`；旋转参考 `pair_rotate_frame`。

### 5.8.2 V_dd 旋转系 secular RWA

$$|E_{\text{rot},\beta} - E_{\text{rot},\alpha}| < \texttt{SECULAR\_RATIO}{=}10\cdot|V| \Rightarrow \text{保留}$$

> 判据必须用**真实旋转系能量差**（含 V_dd 对角位移）——误用参考之差会把远失谐 vdW
> 元误当 secular 保留（T5 回归钉死）。

### 5.8.3 演化主循环（`_pair_rwa`）

- RWA 分类（`_is_rwa`）；对角化精确路径（全 RWA + 常数包络 + 无振荡非 RWA 场）；
- 步长三类场分别计入 w_max；旁观态归一；numba 并行 RK4（`_pair_rwa_jit`，~45×）；
- 两体寿命耗散（NonHermitian，loss_A/B = 速率×布居积分，Σpops+lossA+lossB≡1）；
- 远失谐散射 depump（report-only）。
- **模型局限**：secular 丢远失谐 → 二阶 vdW 能移不在演化哈密顿中。

---

## 5.9 统一驱动引擎（`driver.py`）

```
静态层 pair_static → H_static（含 vdW/交换）；驱动层 driver_operator → (D,ω,Ω)
化简层 rwa_hamiltonian → H_rwa（光子数 BFS 减 n_k·ω，支持双光子级联）
演化层 evolve → 定常哈密顿一次对角化；dressed 投影含 vdW 封锁
```

**线性响应谱**（黄金规则，`response_spectrum_linear`）：

$$S(\omega) = \sum_k |\langle k|D|i\rangle|^2\,\delta(\omega - (E_k - E_i))$$

一次静态对角化 + 驱动算符投影本征基 → 全部共振一次算出（O(N²)）。`response_spectrum`
（逐频率扫）为对照。`build_system(n_body=1/2)` 统一接口；静态能移走
`static_atom.direct_ac_stark_shifts` / `dc_stark_shift_mhz`。

---

## 5.10 核心层（`sambe.py` + `tensor.py`）

### 5.10.1 Sambe 光子块几何（`sambe.photon_geometry(E, groups, Ms)`）

任意 N 通用：混合进制 stride / 量子数 / 块对角（$E_i - (n_g - M_g)\omega_g$）/ 重构相位
（$\theta_p(t)=n_{\text{phase}}[p]\,t$）/ 中央光子块（p_center）。`_SambeSetup` 与
`_PairSambeSetup` 共用（消除几何构造重复）。

### 5.10.2 两体张量化（`tensor.py`）

`tensor_states`（基组）、`tensor_fields`（场镜像）、`_as_product_psi`（初态）、
`_reduce_A/_reduce_B`（约化布居）、`_pair_loss_vectors`（寿命丢失）——全部引擎
（pair_evolve / evolve_sambe_pair / hard_pair）共用。

---

> 兼容性：全部旧模块名/函数名经 `src/__init__.py` 与 `src/core/__init__.py` 别名层保留
> ——`from src.core import core` / `from src.core.evolve_exact import ...` 等仍可用
> （T19 后内部代码已改用新名，兼容层只服务外部）。

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
