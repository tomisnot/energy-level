# 原子能级计算器（Rb-87 / 通用碱金属）

面向里德堡原子实验的**原子能级 + AC/DC-Stark + 两体耦合 + 含时演化**计算与可视化工具。
界面实时交互：磁场 / 电场 / 光场调整，直观看到能级位移、谱线、雷达图与态布居演化。
物理核心与 ARC 库解耦，可切换任意碱金属（Cs、Li、Na、K…）。

> 本文档从**当前代码视角**撰写（2026-08-31），行文按功能组织而非开发历史。
> 全部模块无 UI 依赖，可单独导入审查。

---

## 目录

- [一、能力总览](#一能力总览)
- [二、架构与代码地图](#二架构与代码地图)
- [三、安装与运行](#三安装与运行)
- [四、界面使用](#四界面使用)
- [五、核心算法（src/core）](#五核心算法srccore)
- [六、回归自检](#六回归自检)
- [七、单位与约定](#七单位与约定)
- [八、已知限制与路线图](#八已知限制与路线图)

---

# 一、能力总览

## 1.1 两条计算路径

1. **静态路径**：给定能级组 (n,l,j)、磁场 B、寻址光，实时计算 Zeeman / 超精细 /
   AC-Stark / DC-Stark 位移，绘制能级图与 off/on 镜像频谱，底部两体耦合雷达图（R×θ）。
2. **动态路径**：给定能级组与时变波形，求解含时薛定谔方程 / Lindblad 主方程，
   绘制态布居 P(t)。支持单原子与双原子（偶极-偶极 V_dd）。

## 1.2 引擎选型

| method | 引擎 | 适用 | 性能参考 |
|---|---|---|---|
| `auto` | RWA 相互作用绘景 RK4 / Lindblad | 快速预览（默认） | 20 态 400ns ~29s |
| `ip4` | 同上（显式） | 同 auto | 同 auto |
| `exact` | Sambe 光子块第一性原理 | 关键计算交叉验证 | 22 态 620ns ~36s |
| `hard` | 光场旋转系 + 微波硬积分（DOP853+numba） | **推荐**：零微波近似 + 秒级 | 22 态 620ns **~4s**；两体 1500ns ~84s |

`hard` 为当前主推引擎：**微波零近似**（多光子/Bloch–Siegert/任意波形全含）、
范数机器精度、numba 加速 45×、支持任意相位波形 `cos(φ(t))` 与自发辐射。

## 1.3 核心特性

| 特性 | 说明 |
|---|---|
| 能级结构 | fine + 超精细 + Zeeman（数值对角化），静态/动态共用 |
| AC-Stark | Floquet 矩阵对角化（第一性原理，唯一后端），实数 eigh 13× 提速 |
| DC-Stark | 二阶 SOS（全部偶极中间态） |
| 两体耦合 | V_dd / 交换 / vdW / 雷达图 |
| 含时演化 | 单/两体，5 引擎可选（见 1.2） |
| 自发辐射 | Lindblad 主方程（单原子）；NonHermitian 寿命（两体 + hard） |
| 任意波形 | ArbitraryWave（任意频谱/啁啾/多频），hard 走 cos(φ(t)) 路径 |
| 波形管理 | 高斯/方波包络、定频定强辅助、跨原子拖拽复制 |
| 存档 | 科学数据序列化（波形/参数/初态），不存显示参数 |

---

# 二、架构与代码地图

## 2.1 分层

```
┌ 界面层 src/ui ────────────────────────────────┐
│ ui_dynamic（动态页） ui_group（能级组）          │
│ ui_spectrum（频谱） ui_radar（雷达） ui_pair     │
│ ui_waves（波形面板） evolve_worker（后台线程）   │
│ fields（场面板） app_state serialization        │
└───────────────────────────────────────────────┘
┌ 核心层 src/core ──────────────────────────────┐
│ 静态: static_atom / static_pair                │
│ 演化: evolve_rwa / evolve_arbitrary /           │
│       evolve_sambe / evolve_sambe_pair /        │
│       evolve_twofield / hard / pair_evolve      │
│ 构造: atomic_levels / field_build / waveforms   │
│ 核心: sambe / dop853 / tensor / driver          │
└───────────────────────────────────────────────┘
```

## 2.2 核心模块职责（T19 功能命名）

| 文件 | 职责 | 依赖 |
|---|---|---|
| `static_atom.py` | 单原子静态（能级/偶极/AC/DC-Stark/频谱） | ARC |
| `static_pair.py` | 两体静态（V_dd/vdW/交换/雷达图） | static_atom |
| `atomic_levels.py` | 能级构造（make_levels） | static_atom |
| `field_build.py` | 场构造（field/group/real/arbitrary_field） | static_atom |
| `evolve_rwa.py` | 单原子 RWA 演化（ip4/Lindblad/auto） | static_atom |
| `evolve_arbitrary.py` | 任意波形演化（实验室系 RK4） | static_atom |
| `evolve_sambe.py` | 单原子 Sambe 第一性引擎 + reference | sambe |
| `evolve_sambe_pair.py` | 两体 Sambe 第一性引擎 | evolve_sambe + tensor |
| `evolve_twofield.py` | 两场引擎（光场旋转 + 微波 hard/Sambe） | evolve_sambe + hard |
| `hard.py` | hard 引擎内核（单/两体 numba + cos(φ(t))） | dop853 |
| `pair_evolve.py` | 两体 RWA 演化（secular） | evolve_rwa + tensor |
| `waveforms.py` | 波形数据层（Wave/Envelope/ArbitraryWave） | — |
| `dop853.py` | DOP853 核心层（rhs 参数化，任意 N） | hard（系数） |
| `sambe.py` | Sambe 光子块几何核心（任意 N） | — |
| `tensor.py` | 两体张量化核心（基组/场镜像/约化/丢失） | evolve_rwa |
| `driver.py` | 统一驱动引擎（RWA/dressed/响应谱） | static_atom |

> **命名约定**：模块名由功能决定（T19 重构，2026-08-31）。旧名经
> `src/__init__.py` 与 `src/core/__init__.py` 别名层保留，外部引用零改动。

## 2.3 核心-调度分层（重构要点）

```
核心层（与维度 N 无关，调度层共用）：
  sambe.photon_geometry  光子块几何（混合进制 stride/对角/重构相位）
  dop853._dop853_driver  DOP853 积分驱动（rhs 参数化，任意 N）
  tensor.*              两体张量化（tensor_states/场镜像/约化/丢失）

调度层（瘦壳，各自 RHS + 打包）：
  hard.evolve_hard_fast / evolve_hard_pair
  evolve_sambe._SambeSetup / evolve_sambe_pair._PairSambeSetup
  evolve_rwa（ip4/Lindblad） / pair_evolve（secular RWA）
```

好处：消除单体/两体重复（~450 行驱动 + 几何构造归一）；bug 只修一处；
新引擎（hard 单/两体）共享已验证内核。

---

# 三、安装与运行

## 3.1 环境

```
Python 3.10（项目实测版本），依赖：
  numpy scipy matplotlib PyQt5
  arc（ARC 原子数据库，Rb-87 默认）
  numba（可选，hard 加速必需；无则回退 numpy）
```

## 3.2 启动

```bash
python 能级.py        # 开发（薄包装）
python app_entry.py   # 等价
main.py               # PyInstaller 打包入口
```

## 3.3 自检

```bash
python checks/selftest_all.py    # 全量（~2 分钟）
python checks/hard_numba_selfcheck.py   # hard 专项（9 项）
python checks/ui_selftest.py            # GUI 冒烟（35 项）
```

---

# 四、界面使用

## 4.1 动态页（含时演化）

1. 配置能级组（`ui_group`）与波形（`ui_waves`：高斯/方波/任意波形）
2. 选 method（auto/ip4/exact/hard——见 1.2）
3. 设 t_max / 点数 / 自发辐射档位 / 初态
4. 运行 → 上波形图（单轴归一化，与布居 x 对齐）+ 下布居图
5. 计算中可取消（hard 亦支持）

**双原子模式**：设 A/B 波形、R/θ/φ → V_dd 主线程预计算 → 演化。

## 4.2 静态页

能级组 + 磁场 + 寻址光 → 能级图（含 AC/DC 位移）、off/on 频谱、雷达图。
AC-Stark 光子块阶数 / 偶极窗口面板可调（n_photon 默认 4，window 默认 3）。

## 4.3 存档

文件 → 保存/恢复状态：只存科学数据（波形/参数/初态），恢复为**全量覆盖** +
自动刷新（groups_changed 信号触发重算），无需手动刷新。

---

# 五、核心算法（src/core）

> 本节精细记录物理模型、数值算法、公式与实现映射。公式用 LaTeX（GUI 已启用
> KaTeX 渲染）。

## 5.1 单原子静态（`static_atom.py`）

### 5.1.1 能级

**里德堡态**（n ≥ 20）：

$$E(n,l,j,m) = E_0(n,l,j) + g_J(l,j)\,\mu_B B\,m/\hbar$$

**低激发态超精细 + Zeeman**（|m_J,m_I⟩ 基按 m_F 分块对角化）：

$$H = A\,\mathbf{I}\!\cdot\!\mathbf{J} + B\,Q + \mu_B B\,(g_J J_z + g_I I_z)$$

实现：`atomic_levels.make_levels` → `_base_level_rows` / `get_hyperfine_levels`。

### 5.1.2 偶极与 Rabi

$$d_q = \langle b|d_q|a\rangle \ (\text{a.u. 复}),\qquad
\Omega = \frac{E_0 |d|}{\hbar},\qquad \Omega_{\text{MHz}} = \frac{\Omega}{2\pi}\times10^{-6}$$

共振 π 脉冲 = ½Ω⁻¹ µs（Ω/2π 周频约定）。四分支选择定则（mJ-mJ/F-mJ/mJ-F/F-F）见
`dipole_amplitude`。

### 5.1.3 AC-Stark（Floquet 对角化，唯一后端）

$$H_F = \sum_n (H_0 + n\hbar\omega)\otimes|n\rangle\langle n| + \frac{V}{2}\otimes(|n{+}1\rangle\langle n|+|n\rangle\langle n{-}1|)$$

- 光子块 M=2·n_photon+1（默认 4 → 9 块）；**实数 eigh**（H 实对称 → 13× 提速，2026-08-31）
- 两级基组收敛（window）：window=1 粗算 / 2-3 精算，两级差即截断上界
- 位移归属：本征态对光子块 0 投影重叠最大者；共振劈裂取离裸能最近分支

### 5.1.4 DC-Stark（二阶 SOS）

$$\Delta E_i = \sum_k \frac{|\langle k|d|i\rangle|^2 E^2}{E_i - E_k}$$

### 5.1.5 频谱

$$A = \frac{\omega^3}{3\pi\varepsilon_0\hbar c^3}\,|\langle a|d_q|b\rangle|^2\,(ea_0)^2$$

## 5.2 两体静态（`static_pair.py`）

$$V_{dd} = \frac{1}{R^3}\sum_{qq'} d_{1q} T_{qq'} d_{2q'},\qquad
T = e_q^{\mathsf T}(1-3\hat n\hat n^{\mathsf T})e_{q'}$$

- `vdd_sparse`：pair 基 k=i·N+j 稀疏矩阵（LRU 缓存）
- `exchange_coupling`：一阶 ⟨a|V_dd|b⟩（默认 60S↔60P）
- `vdw_shift`：对角化后从 |ref⊗ref⟩ 支提取（∝1/R⁶）

## 5.3 RWA 演化（`evolve_rwa.py`）

$$H(t) = H_0 - \sum_i \text{ref}_i P_i + \sum_k \tfrac{\Omega_k(t)}{2} A_k \quad(\text{RWA})$$

- **级联旋转参考**：从最低能组沿耦合对传播 ref（处理共享能级多驱动）
- **相互作用绘景 RK4**：步长三重约束（载波/耦合/稳定）+ numba 稀疏内核
- **Lindblad**：dρ/dt = −i[H,ρ] + Σ(LρL† − ½{L†L,ρ})；`lindblad_eig`（H 恒定→本征分解）
  与 `lindblad`（任意包络，矩阵 RK4，numba 42×）
- `auto` 调度：无耗散→ip4；有耗散→lindblad_eig/lindblad

## 5.4 任意波形演化（`evolve_arbitrary.py`）

$$H(t) = H_0 + \sum_k \text{amp}_k(t)\,\hat V_k \quad(\text{实验室系})$$

任意频谱/啁啾/多频；步长受 Nyquist 约束（慢但第一性原理）。

## 5.5 Sambe 第一性引擎（`evolve_sambe.py` / `evolve_sambe_pair.py`）

$$H(t) = H_0 + \sum_k \Omega_k\,c_k(t)\cos(\omega_k t)\,A_k$$

- **光子块扩展空间**（`sambe.photon_geometry`）：每载频一维，块对角承载载波相位
- **逐载频 M**：$(r_g/4)^{2(M_g-1)} \le \text{tol}$（光学 M=1、微波 M=2）
- **冻结段 + Duhamel 缺陷修正**：`δ(t) = −iC∫U_f(t,s)[K(s)−K(c_mid)]ψ(s)ds`
- **两体扩展**：idx=k+N²·p，V_dd 块对角 + 寿命 NonHermitian；大基组走稀疏 Krylov
- 误差：单旋钮 tol（默认 1e-6），error_report 可见

## 5.6 两场引擎与 hard（`evolve_twofield.py` / `hard.py` / `dop853.py`）

### 5.6.1 核心思想（2026-08-29）

电磁场按频段拆开：**光场旋转系吸收载波，微波硬积分/光子块**——消除 THz 步长爆炸，
只解析 17 GHz 微波载波。

### 5.6.2 `evolve_hard`（微波硬算，零近似，主推）

$$H(t) = \text{diag}(E_{\text{rot}}) + \sum_{\text{opt}} c_f(t)\,e^{i\delta_f t}A_{\text{opt}} + \sum_{\text{mw}} c_f(t)\cos(\varphi_f(t))\,A_{\text{mw}}$$

- **光场**：旋转系 + 光学 RWA（Bloch–Siegert 4.4e-5 rad ≪ tol）+ 多光场 exp 拍频
- **微波**：cos(φ_f(t)) 完整载波硬积分——多光子/任意谐波/任意时域全含，零近似
- **自发辐射**：NonHermitian（寿命，落点=丢失）
- **性能**：单原子 620ns ~4s（45×）、两体 1500ns ~84s（7×）
- **支持**：任意相位波形（t18）、多光场（t16）、取消、GUI 接线（t20/t21/t24）

### 5.6.3 `evolve_two_field`（微波 Sambe 光子块）

微波进扩展空间（M=2，多光子显式）；光场块对角。620ns 三路一致 ≤6.3e-5，单原子 2s。

### 5.6.4 任意波形微波（`arbitrary_mw_group`）

ArbitraryWave → hard 兼容组：Hilbert 分解 $a(t)=c(t)\cos\varphi(t)$ + 全偶极振幅。

## 5.7 两体 RWA（`pair_evolve.py`）

$$H(t) = H_A(t)\otimes I + I\otimes H_B(t) + V_{dd}$$

secular 判据（保留 |ΔE_rot| < 10·|V| 的元）+ numba RK4（~45×）+ NonHermitian 寿命。

## 5.8 驱动引擎（`driver.py`）

RWA 化简（光子数 BFS）+ dressed 投影 + 线性响应谱（黄金规则）：

$$S(\omega) = \sum_k |\langle k|D|i\rangle|^2\,\delta(\omega - (E_k - E_i))$$

---

# 六、回归自检

| 自检 | 项数 | 覆盖 |
|---|---|---|
| `core_selfcheck.py` | 22 | 静态物理（能级/AC/DC/频谱） |
| `evolve_dissipative_selfcheck.py` | 18 | Lindblad/耗散 |
| `evolve_exact_selfcheck.py` | 11 | Sambe 单体 |
| `evolve_exact_pair_selfcheck.py` | 12+SKIP | Sambe 两体 |
| `hard_numba_selfcheck.py` | 9 | hard 单/两体（含 620ns 性能） |
| `pair_evolve_selfcheck.py` | 12 | 两体 RWA |
| `two_field_selfcheck.py` | 4 | 两场引擎 |
| `ui_selftest.py` | 35 | GUI 冒烟 |
| `julia_kernel_selfcheck.py` | — | Julia 内核对照（需 julia.exe） |

> **⚠️ numba 缓存纪律**（2026-08-31）：`cache=True` 的缓存键基于字节码哈希，
> 改名等操作可能"复用"旧缓存掩盖回归。**物理/内核变更后必须清
> `__pycache__/*.nbc` 再自检**；基线刷新后 fresh 编译复验。

---

# 七、单位与约定

| 量 | 单位 | 备注 |
|---|---|---|
| 能级/耦合/失谐/Rabi | MHz | float64 |
| 偶极矩阵元 | a.u. | ARC |
| 光强 | mW/cm² | →W/m² ×10 |
| 时间 | ns | 角频率 rad/ns |
| 几何 | μm / ° | |

- `_CONV = 2π×1e-3`（MHz·ns → rad）
- 失谐约定：**正 = 蓝失谐**
- Rabi：Ω/2π 周频，π 脉冲 = ½Ω⁻¹ µs

---

# 八、已知限制与路线图

## 8.1 已知限制

- `exact`/`hard` 的 ArbitraryWave：hard 支持（t18），Sambe 暂不支持
- 两体 Sambe 大基组（N²·P>700）：稀疏 Krylov 慢（>25min 不可行）——待两体两场变体
- NonHermitian 耗散仅支持落点=丢失（显式落点/纯退相 NotImplementedError）
- pair hard 的 V_dd 需主线程预计算（ARC SQLite 线程约束）
- ip4/RWA 有 P1 静默失效风险（远失谐旁观者混叠）——关键计算用 exact/hard 交叉验证

## 8.2 路线图

| 项 | 状态 |
|---|---|
| hard 引擎全栈（耗散/取消/任意波形/GUI） | ✅ 完成 |
| 核心-调度重构（dop853/sambe/tensor） | ✅ 完成 |
| 静态 AC-Stark 实数化 13× | ✅ 完成 |
| GUI 画布对齐（单轴归一化） | ✅ 完成 |
| 两体两场变体（光旋转+微波-only 光子块，dim=2420） | ⏳ 待 Sambe 两体扩展 |
| 多光场拍频在 Sambe K 的时变相位 | ⏳ 后续 |
| 1500ns 全场景正式验收 | ⏳ 待跑 |
