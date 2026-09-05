# Rb-87 里德堡原子仿真软件 — 使用 · 原理 · 开发

> 面向三种读者的完整技术文档：
>
> - **第 1 部分 · 使用指南**：实验人员/一般用户——软件能做什么、界面如何操作。
> - **第 2 部分 · 引擎原理**：物理/算法研究者——hard / sambe / mixed 三引擎的严格推导、
>   公式与背后的物理图景（波 vs 粒子、低频 vs 高频的互补）。
> - **第 3 部分 · 开发指南**：开发者——代码架构、模块地图、如何调用核心算法组装
>   自己的仿真程序。
>
> 配套理论文档（docs/ 下）：`混合引擎理论推导-光Sambe微波当场.md`、`t58光子块截断判据.md`、
> `引擎能力边界-终稿.md`、`认知固化II-AC化与混合引擎回归.md`、`auto路由方案-动态引擎选择.md`
> 等（见第 3 部分 §9 文档索引）。

---

# 第 1 部分 · 使用指南

## 1.1 软件是什么

**Rb-87 里德堡原子仿真软件**是一个图形化的原子物理计算平台，围绕铷-87（⁸⁷Rb）的
**里德堡态**（主量子数 n≈30–100 的高激发态）展开两类计算：

1. **静态计算**：给定能级组、磁场、外加光/电场，算出
   - 里德堡能级的 Zeeman 分裂、DC-Stark 位移；
   - 寻址光（AC-Stark / 光频移）作用下各子能级的频移；
   - 跃迁谱线（频率、偏振、强度）与耦合强度；
   - 双原子偶极-偶极（V_dd）耦合强度的角度/距离依赖（雷达图）。
2. **含时动力学**：给两个（或一个）原子施加一串**微波脉冲 + 光脉冲**，从某个初态出发，
   数值求解含时薛定谔方程，输出各能级布居随时间的演化——这是软件最核心也最强大的部分，
   可模拟 π 脉冲、拉比振荡、受激拉曼绝热过程（STIRAP）、双原子共振能量交换（Förster 交换）
   等真实实验过程。

计算引擎底层使用 ARC（Alkali Rydberg Calculator）提供偶极矩阵元、寿命等原子数据，
动力学内核全部自主实现（不依赖 ARC 的演化例程）。

## 1.2 启动与界面总览

### 启动方式（三者等价）

| 命令 | 场景 |
|---|---|
| `python 能级.py` | 日常开发 |
| `python app_entry.py` | 直接运行（本模块自带 `__main__` 守卫） |
| `main.py`（PyInstaller 打包后双击） | 分发给无 Python 环境的用户 |

启动后出现主窗口（标题 **Rb-87 能级计算**），布局如下：

```
┌─────────────────────────────────────────────────────────────┐
│ 菜单：文件 ─ 保存状态 / 恢复状态                    状态栏     │
├───────────────────────────────┬─────────────────────────────┤
│ 左侧控制面板（可滚动）        │  中央画布区                  │
│  ┌─────────────────────────┐  │  [静态] [动态]  ← 两个 Tab  │
│  │ 能级组管理（n/l/j）      │  │                             │
│  ├─────────────────────────┤  │  静态：能级图/镜像频谱/雷达  │
│  │ 场面板（B/E/光场表）     │  │  动态：上=波形包络           │
│  ├─────────────────────────┤  │        下=态布居演化曲线     │
│  │ 页面专用参数（随 Tab 变）│  │                             │
│  └─────────────────────────┘  └─────────────────────────────┘
```

- 左侧**右侧滚动面板**：能级组面板 → 场面板 → 页面堆栈（随中央 Tab 切换显示静态/动态参数）。
- 中央 **「静态」/「动态」** 两个 Tab：静态页做能级/光谱/耦合强度分析；动态页做含时演化。
- 菜单 **文件 → 保存状态 / 恢复状态**：把当前所有参数（能级组/场/波形/初态/方法）存成
  JSON 存档或从存档恢复。

### 通用状态栏

窗口底部状态栏是所有计算日志的统一出口（运行进度、路由理由、耗散通道数、V_dd 非零元数
等都会显示在那里）。

## 1.3 能级组管理（两页共用）

在左侧最上方 **能级组** 区域：

1. 用 **n / l / j** 三个输入框指定主量子数、轨道角动量、总角动量（如里德堡 60S₁/₂：
   n=60, l=0, j=0.5）；
2. 点 **「添加」**——能级组加入列表（自动配色并生成标签）；
3. 列表中每行可右键 **「删除此能级组」**；
4. 下方数值表列出该组全部 mJ 子能级与能量（MHz）。

> 注意：**动态演化的基组 = 这里添加的所有能级组的全部子能级**（再加 6P 等中间态就
> 能算完整的光学寻址过程）。要模拟"光场 + 微波"的多能级过程，请把涉及的组都加进来。

## 1.4 静态计算页

### 1.4.1 场面板（三页共用同一份"场"配置）

静态页顶部常驻以下全局参数：

- **磁场 B (G)**：量子化轴 z 方向的外磁场（默认 21.7 G），三页共用。
- **电场 E (V/cm)**：直流电场（DC-Stark 二阶位移）。
- **窗口**：AC-Stark 计算中偶极伙伴基组的圈数（1–5，默认 3）。
- **光子块**：Floquet 光子块阶数（1–12，默认 4）——AC-Stark 计算的精度旋钮。
- **＋添加光**：为"静态分析"添加一束寻址光。光场表格每行：
  - **频率**（GHz，可双击编辑；右键"从跃迁对导出频率…"按 跃迁对 + 失谐 Δ 自动定频）；
  - **光强**（mW/cm²，右键"扫描光强导出强度…"可按 目标能级 + 目标频移反推光强）；
  - **偏振**（σ+ / π / σ− / 椭圆偏——椭圆偏会弹出 α/θ/ε 参数对话框）；
  - **作用范围**（both / atom1 / atom2——双原子时各打各的）。

### 1.4.2 能级图与镜像频谱

- **能级图**：横轴为 mJ，水平线为各子能级；被寻址光作用的子能级画箭头并标注
  AC-Stark 频移（自动 GHz/MHz/kHz 单位）。
- **镜像频谱**：y=0 上方为"无光"谱、下方为"有光"谱（受 AC 位移影响），方便对比
  光频移导致的谱线移动。点击谱峰可高亮并在能级图上画跃迁连线、标注频率。
- 工具栏：**能级图模式**（单一 / off/on 双图）、**y 范围**上下限下拉（筛选可见能级）、
  **参考频率**（频谱参考竖线）、**谱 x 范围**上下限编辑（空=自动）。
- 下方两张谱表（off/on）：**Freq (GHz) | From | To | Pol**，点击行与画布高亮联动，
  选中峰显示 **Δ MHz**（off/on 频率差 = 光频移的谱学表现）。

### 1.4.3 耦合强度雷达图

- 参数栏：**R (μm) / θ (°) / φ (°)** 原子间距与取向；**J (MHz)**（只读，当前点耦合强度）。
- 画布：|V_c| 耦合强度随 θ/φ 的极坐标半圆热图（inferno 色标）；鼠标移动实时更新参数栏，
  点击画圈/再点清除。
- 跃迁对由频谱表点击自动注入；配合"交换耦合/雷达图"可快速找到给定几何下 V_dd 交换
  最强的共振对——这是双原子动力学实验设计的静态预演。

## 1.5 动态演化页（核心）

### 1.5.1 单原子 / 双原子模式

页面顶部单选 **「单原子」/「双原子」**：

- **单原子**：一个原子（A）受波形驱动；顶部画布只显示 A 的波形包络。
- **双原子**：两个原子 A、B 各自受独立波形驱动，原子间有偶极-偶极相互作用 V_dd：
  - 出现 **V_dd 参数行：R (μm)、θ (°)、φ (°)**——原子间距与相对取向；
  - 波形区变成 **「原子 A」「原子 B」两个页签**，各自独立配波形；波形可从 A 拖到 B 复制
    （按住标题拖动）；
  - 初态区变成 **A / B 两个下拉**，分别选两原子的初态。

### 1.5.2 波形配置

在波形区用 **「＋高斯」「＋方波」「＋任意」** 添加脉冲，每个波形是一个可折叠参数组：

| 参数 | 说明 |
|---|---|
| **频率** | 载频（THz/GHz/MHz/kHz/Hz 单位可切，内部全精度 MHz）——>1e5 MHz 即"光场"，≤1e5 即"微波" |
| **光强** | 峰值光强（W/mW/μW/nW 单位可切）——与频率共同决定真实驱动强度（Rabi 频率） |
| **偏振** | σ+ / π / σ− / 椭圆偏 |
| 包络参数 | 高斯：中心 τ_c、σ、峰值；方波：起点、上升沿、高度、下降起点（raised-cosine 缓变沿） |
| **偏移 (ns)** | 波形绝对起点相对演化 t=0 的偏移 |
| **辅助**（可折叠） | 选 g/e 态对后：**定光强**（反推使该对 Rabi=目标 Ω 的光强写回）、**定频率**（频率=裸差+目标失谐 Δ）——只辅助定参，不参与结算 |

**任意波形**（ArbitraryWave）：无单一载频概念，载入用户自己的时域信号文件
（txt/csv/dat/npy，两列"时间 ns 幅值"，或单列+等间距 dt；载入时峰值自动归一，负幅值
= 电场反向）；参数只剩光强/偏振/偏移，可随时「重新载入波形文件…」替换样品数据；
软件按第一性原理（实验室系，无旋转波近似）求解，适合啁啾、多频叠加等任意波形。
> 注意：任意波形演化较慢，软件会提示 Nyquist 上限（f_max）与建议最小步长。

**光强量级参考**（同偏振/频率下 Rabi ∝ √I）：60S↔60P 微波（17 GHz）偶极巨大，~10⁻⁴
mW/cm² 即 MHz 级 Rabi；60S↔6P 光场（295.7 THz）需 ~10⁷–10⁸ mW/cm² 才到几十 MHz
Rabi。设置时优先用波形辅助区"定光强/定频率"按目标 Ω/Δ 反推，避免盲目试数。

### 1.5.3 演化方法与引擎选择（method_cb）

| 方法 | 含义 | 何时用 |
|---|---|---|
| **sambe** | 第一性原理 Sambe 光子块（实验室系、无 RWA、无旋转系） | 强光近共振需最高精度；STIRAP/拉曼 |
| **hard** | 光场旋转系 RWA + 微波 cos 硬积分 DOP853（numba 加速） | 纯微波/变频/任意波形 |
| **mixed**（默认） | 混合引擎：光场 Sambe 光子块 + 微波/V_dd 当场（φ-帧） | 光+微波混合/双原子交换（5 倍降维）；**软件默认**；无光场时自动降级 hard |

> 旧版 ip4（RWA 快速粗算）与 auto（智能路由）档已从选项移除：引擎层仍可通过脚本
> 调用（`evolve_rwa.evolve(..., method='ip4')` / `route_auto`），GUI 统一为上述三档。
> **mixed 需要光场**：若波形只有微波（或含任意波形），运行时会自动降级 hard 并在
> 状态栏提示——纯微波场景无需手动切换。

- **sambe 高级参数**（仅 双原子 + sambe 时显示）：
  - **frozen_h (ns)**：全频段 Sambe 纯冻结段长（0=默认缺陷修正路径；推荐 8）；
  - **per_freq_M**：逐载频光子块阶数（如 `[1,1]`，空=自动探针）。

### 1.5.4 初态与自发辐射

- **初态** 下拉：基组全部子能级（低激发态按超精细 F/mF 列；里德堡按 mJ 列）。
- **自发辐射**（decay_mode）：
  - **自动（ARC 寿命）**：按 ARC 寿命 Γ=1/τ 自动建衰变通道（τ≤100 µs 的低激发态，
    落点=丢失——离开建模空间）；
  - **禁用**；
  - **自定义**：手动加通道——从态下拉 + Γ (MHz) + 落点（含"（丢失）"），
    **「添加通道」/「删除选中」**；同态同落点=纯退相通道会提示。

### 1.5.5 显示选项

- **显示模式**：「态布居」（单原子 |⟨i|ψ⟩|² / 双原子 product 态布居）或 **「荧光读出」**
  （按读出率加权：单原子 |⟨i|ψ⟩|² 加权 / 双原子 A、B 各自约化密度 ρ_A、ρ_B）。
- **topn**：按最近结果平均布居取前 N 态显示（默认 8）——与荧光读出单选互斥。
- **荧光读出 → 复杂配置…**：弹出非模态窗口，为每个能级设 0–1 读出率，改动实时生效。

### 1.5.6 运行、进度与取消

1. 设置好波形/初态/时间范围（**t_max (ns)** 与 **点数**）/方法/显示选项；
2. 点 **「运行演化」**；
3. 弹出进度对话框：阶段标签（**准备 → 积分 → 收尾**）+ 进度条 + **「取消」**。
   进度条按阶段映射（准备 0–5% / 积分 5–95% / 收尾 95–100%），且每个阶段
   内部都有细粒度进度（t72）：
   - **准备**（主线程预处理 + JIT）：基组构建逐能级组、驱动场逐原子、
     双原子 V_dd 预计算**逐行**上报并可取消；numba JIT 编译/装载本身无
     内部进度，标签附带的**墙钟耗时**即活性指示；
   - **积分**：DOP853 内核逐步上报。进度按**预测剩余积分工作量**映射
     （平滑的局部求值密度 × 剩余模拟时间）——自适应步长下比裸 t/T 更
     贴近真实墙钟（stiff 脉冲区步长可比自由区小 1–2 个量级）；积分内核
     全程释放 GIL（nogil），进度与「取消」在积分期间实时响应（修复前
     njit 持有 GIL，整段积分进度条纹丝不动、取消按钮无响应）；
   - **收尾**：跨扇区投影 / loss / 约化布居逐点上报；
4. 完成自动关闭；失败会弹「演化失败」并建议降低光强/调整失谐/减小 t_max；
5. 结果即时画在顶部（波形包络）与底部（布居/荧光曲线；有耗散时叠加 loss 虚线）。

## 1.6 典型操作流程示例

### 示例 A：单原子 π 脉冲（微波 Rabi 振荡）

1. 静态页加入能级组 60S₁/₂（n=60, l=0, j=0.5）与 60P₃/₂（n=60, l=1, j=1.5）；
2. B=21.7 G（或按需要）；
3. 切「动态」Tab，保持「单原子」；
4. 波形区「＋高斯」：频率 17.236 GHz（微波，偏振 σ−），光强 ~5×10⁻⁵ mW/cm²
   （60S↔60P 偶极极大，μW 级光强已有 MHz 级 Rabi；也可用辅助区"定光强"：
   选 g=60S mJ、e=60P mJ，给定目标 Ω 自动反推光强），中心 70 ns、σ 20 ns、偏移 0；
5. 初态选 60S₁/₂ 某 mJ；t_max=300 ns、点数 301；方法保持默认 mixed 即可
   （纯微波会自动降级 hard——也可直接选 hard）；显示"态布居"；
6. 「运行演化」→ 底部应看到 60S 与 60P 之间的 Rabi 振荡；若振荡周期 ≠ 2/Ω 或转移
   不完全，用辅助区把目标 Ω 设准再跑。

### 示例 B：双原子共振能量交换（Förster 交换，需寻址光调共振）

1. 加入能级组：60S₁/₂、60P₃/₂、以及**中间态 6P₃/₂**（光场寻址必须，否则 60S 的
   AC 频移算错）；
2. 动态页选「双原子」，设 R/θ/φ（如 R=12 μm, θ=30°, φ=0）；
3. 「原子 A」配：微波 σ− 高斯脉冲（60S→60P 泵浦）+ 光场 π 方波（295.7 THz，I~1e8，
   时间窗 [120,280]）+ 光场 π 方波（[280,1200]，调 60S− 的 AC 位移把交换对拉共振）；
4. 「原子 B」配同样的微波脉冲（A 有光、B 无光——用光只调 A 侧共振，这是设计意图）；
5. A/B 初态都选 60S₁/₂；t_max=620 ns；方法用默认 mixed（光+微波混合正是 mixed 的
   核心适用域）；
6. 「运行演化」→ 观察 375 ns 附近两体态 |60P−,60S+⟩ ↔ |60S−,60P+1.5⟩ 的共振交换。

### 示例 C：存档复现

「文件 → 保存状态」存 JSON → 换机器/重开软件 → 「文件 → 恢复状态」→ 所有参数复原，
再点「运行演化」即可复现。

## 1.7 使用注意事项（物理提醒）

- **基组要完整**：光场耦合的中间态（如 6P）必须加入能级组，否则 AC 频移/失谐算错——
  例如 60S 的光频移主源是 60S↔6P 近对，漏 6P 会把 60S 频移从 −53 MHz 算成 +8 MHz。
- **强光下微波被抑制是物理**：强寻址光给里德堡能级 MHz 级 AC 位移，会把微波驱动推出
  共振（"光频移阻断微波"）——这是真实效应，不是 bug；调光强/失谐可恢复。
- **RWA 的边界**：hard 的光场部分用旋转波近似（引擎层 ip4 亦同），弱场远失谐误差
  ~(Ω/Δ)² 可忽略；强光近共振（STIRAP 等）请用 sambe 或 mixed（无 RWA）。
- **采样点数**：点数只管输出曲线密度；积分步长由 DOP853/自适应内部决定，不必担心
  点数太少导致不精确（只要别少到看不见振荡）。
- **双原子 V_dd 预计算**：运行前软件会在主线程预计算 V_dd 稀疏矩阵并打印非零元数——
  大基组下这一步要等一会儿属正常（t72 起该阶段在进度对话框内逐行显示进度，且可取消）。

---

# 第 2 部分 · 引擎原理

> **出发点：为什么不用 RWA**（先立靶，再讲三引擎怎么绕开它）
>
> 在介绍任何引擎之前，先回答一个根本问题：**为什么整套动态引擎不以教科书上最优雅的
> 方案——旋转波近似（RWA）——为出发点？**
>
> 单电磁场下，RWA 确实无懈可击：把 $\cos(\omega t)=\tfrac12(e^{i\omega t}+e^{-i\omega t})$
> 拆成正负频率后取旋转系，含时哈密顿变成**不含时的常数矩阵**，矩阵元物理意义一目了然
> （对角 = 失谐，非对角 = Rabi/2），解析解（Rabi 公式）随手可得。但我们的场景从来不是
> 单场：**光场（295.7 THz，驱动 60S↔6P）与微波（17 GHz，驱动 60S↔60P）同时挂在
> 60S 这一个态上**。旋转坐标系在这里遭遇结构性失败，原因有四层：
>
> **① 一个态只能有一个参考频率，两个场却要求两个（参考冲突）。** RWA 的本质是给每个态
> 选一个旋转参考 ref，使被它驱动的对"停"在慢失谐。单场时这个选择唯一；多场时同一个
> 60S：按光场平移 +295.7 THz → 光对停失谐，但 60S↔60P 微波对的旋转系差被拖到
> ≈295.7 THz，微波 Rabi 变 THz 振荡直接失效；按微波平移 → 光对差回到 295.7 THz，
> 光耦合失效。**一个态无法同时满足两个场的旋转要求。**
>
> **② 链式参考的补救引入"旁观态污染"（问题.json 现象①）。** 早期 ip4 用级联参考补救
> （ref[高] = ref[低] + ω 沿驱动图传播），使每个**驱动对**各自共振。但不在光场驱动对
> 里的态（如 60P）不参与平移 → 60S 与 60P 之间凭空出现 **−295.7 THz 的旋转系假差**，
> 直接杀死本应发生的 V_dd 交换。当年"B 侧加 1e-6 微光就恢复"的假象，实为 B 侧恰好被
> 平移了同样的 295.7 THz、两边假差巧合抵消——不是那点微光的物理。
>
> **③ 跨流形链结构性不可用（STIRAP 盲区，t33/t34 实测）。** 5S↔6P↔60S 两光场时，每光场
> 独立取低流形旋转（5S+ω₁、6P+ω₂）→ 光 1 的耦合对旋转系残差 **−295.7 THz**（实测值），
> RWA 彻底失效。这不是精度问题，是框架盲区，无放宽空间。
>
> **④ 即使参考选对，RWA 仍是近似。** 反旋项被丢弃，弱场远失谐误差 ~(Ω/Δ)² 可忽略，
> 但强光近共振（STIRAP 光 2 主脉冲区）实测中段误差达 **2.2e-2**（t39：frozen 收敛
> 3.5e-7 作证是 RWA 的错）。
>
> **结论——整套引擎的出发点 = 否定 RWA 作为通用主算法**，但**不是全盘抛弃**：RWA 被
> 降级为"低频/弱场/单场"域的专用快路径（hard 的光场部分），而通用架构彻底绕开"旋转
> 参考选择"这个病根——用 Sambe 光子块/φ-帧把载波**代数化**（当粒子），让参考坐标系
> **根本不存在**。§2.0 起从共同物理问题出发，看三种算法如何各司其职。

---

## 2.0 总览：一个物理问题，三种算法，一段波粒二象性

软件要解的问题在数学上只有一个：

$$
i\hbar\,\frac{\partial}{\partial t}|\psi(t)\rangle
= H(t)\,|\psi(t)\rangle,
\qquad
H(t)=H_{\text{atom}}+\sum_f \Omega_f\,\text{env}_f(t)\cos(\omega_f t)\,\hat A_f
$$

其中 $H_{\text{atom}}$ 是原子（含 Zeeman/Stark）哈密顿，$\hat A_f$ 是第 $f$ 个场的
偶极耦合算符（其矩阵元即 Rabi 频率 $\Omega_f$ 的归一化骨架），$\text{env}_f(t)$ 是脉冲包络。
真正的困难不在方程本身，而在**频率尺度的跨度**：

| 场 | 载频 | 周期 | 620 ns 窗内的周期数 |
|---|---|---|---|
| 光场（6P↔60S 寻址） | 295.7 THz | ≈3.4 fs | ≈1.8×10⁸ |
| 微波（60S↔60P） | 17 GHz | ≈0.058 ns | ≈10⁴ |

任何数值积分器的步长都必须远小于被积函数最快振荡的周期：若要"老老实实"积分光场
载波，步长要 ~10⁻⁶ ns，620 ns 需要 ~10⁹ 步——不可行。三引擎的差异，本质上是**如何
处理载波 $\cos(\omega t)$ 这个快振荡因子**的三种策略，它们恰好对应量子光学里
"把光看成什么"的两种互补图像：

- **hard（旋转波近似 + 硬积分）把光当作经典波**：载波是真实的 $\cos(\omega t)$ 振荡，
  但我们用一个旋转坐标系把它"转慢"（RWA），把可解析的快振荡变成慢包络；对转不慢的
  部分（微波载波）就让积分器硬扛——微波周期 0.058 ns，DOP853 轻松解析。
- **sambe（Sambe/Floquet 光子块）把光当作粒子**：把 $\cos(\omega t)=\tfrac12(e^{i\omega t}+e^{-i\omega t})$
  展开成"吸收/发射一个光子"的边带跃迁，在扩展的**光子数空间**里载波变成**静态的
  对角项** $-\omega\, n$——振荡被代数化，彻底消失，积分成本与载波频率无关。
- **mixed 两者结合**：高频的光场"当粒子"（进光子块，免 RWA），低频的微波/V_dd
  "当波"（当场硬积分），在统一的 φ-帧里单次积分——这是目前最实用的引擎：两体维度
  从全 Sambe 的 7260 降到 1452（5 倍），同时保持无 RWA 的精度。

> **物理图景小结**：频率越低，波的图像越自然（载波可解析、当场积分高效、RWA 误差小）；
> 频率越高，粒子的图像越必要（载波不可解析，必须 Floquet 化）。数值算法在频率轴上的
> 分工，暗合波粒二象性：**低频用波、高频用粒子，mix 在中间把它们缝起来**。

下面三节分别给出三引擎的严格推导与物理理解。记号统一：能量/频率用 MHz，时间用 ns，
$C=2\pi\times10^{-3}$ rad/(MHz·ns)；Rabi 频率用周频 $\Omega$（$\pi$ 脉冲时长 $=1/(2\Omega)$ µs）。

## 2.1 共同起点：实验室系哈密顿与场的离散化

设单原子基组 $\{|i\rangle\}_{i=1}^{N}$，$E_i$ 为无场能级（含磁场 Zeeman 项，单位 MHz）。
外加 $F$ 个场，第 $f$ 个场载频 $\omega_f$、包络 $\text{env}_f(t)\in[0,1]$、偏振决定的
Rabi 矩阵 $A_f$（$A_f[i,k]=\Omega_{ik}$ 峰值光强下跃迁 $|i\rangle\leftrightarrow|k\rangle$
的 Rabi 频率，MHz；含选择定则）。实验室系（Schrödinger 绘景）哈密顿（单位 $\hbar=1$，
频率单位 MHz）：

$$
H(t)=H_{\text{atom}}+\sum_{f}\text{env}_f(t)\cos(\omega_f t)\,A_f .
\tag{2.1}
$$

微波项（17 GHz）与光场项（295.7 THz）在 (2.1) 中地位完全对称——唯一的差别是 $\omega_f$
的量级。两体时，原子 A、B 各带自己的场，加上原子间偶极-偶极相互作用：

$$
H_{2\text{b}}(t)=\big(H_A(t)\otimes I+I\otimes H_B(t)\big)+V_{dd}(R,\theta,\phi),
\tag{2.2}
$$

$V_{dd}$ 由 ARC 偶极矩阵元严格计算（软件不硬编码 $C_3/C_6$）。

> **包络离散**：实际计算中每个场只有有限个"波形"（高斯/方波/任意），包络是已知的
> 慢变函数。除任意波形外，$\text{env}_f(t)$ 变化特征时间 ≥ ns 级，远慢于积分步长。

### 2.1.1 双原子基组与 V_dd 的相位问题（先铺垫）

两体基取张量积 $|i,j\rangle=|i\rangle_A\otimes|j\rangle_B$，扁平索引 $k=i\cdot N+j$，
$N^2$ 维。A 侧场 $A_f\otimes I$、B 侧场 $I\otimes A_f$。$V_{dd}$ 在实验室系是**常数**
（与时间无关）相互作用矩阵，其矩阵元连接不同的两体态 $|i,j\rangle\leftrightarrow|k,l\rangle$，
能量差 $\Delta E=E_i+E_j-E_k-E_l$（MHz 量级）。

**关键的相位问题**：当我们对某个场做旋转系变换后，不同态获得不同的时间相位
$e^{-iC\cdot\text{ref}\cdot t}$；如果 $V_{dd}$ 连接的两个态属于不同旋转流形，$V_{dd}$
项就会带上旋转系残差频率。这是多频 RWA 最容易出错的点（t34 核查的"疑点 2"），
hard 与 mixed 用了两种不同的正确解法，见 §2.2.3 与 §2.4.3。

## 2.2 hard 引擎：光 = 经典波（旋转系 RWA + 微波硬积分）

### 2.2.1 单场 RWA 推导

先看单个场 $\text{env}(t)\cos(\omega t)A$ 驱动一对近共振能级 $|g\rangle$（低）、
$|e\rangle$（高），裸失谐 $\Delta_0=\omega-(E_e-E_g)$。把 $\cos$ 写成正负频率：

$$
\cos(\omega t)=\tfrac12\big(e^{+i\omega t}+e^{-i\omega t}\big).
$$

取**旋转系** $\;|\tilde\psi(t)\rangle=e^{+iC\,\omega_{\text{ref}}\,t}\,|\psi(t)\rangle$，
其中参考频率 $\omega_{\text{ref}}$ 只加在**被光场耦合的低能流形**上（见 §2.2.2）。
对共振项（$e^{i\omega t}$ 使 $|g\rangle$ 抬升的能量与 $|e\rangle$ 拉平），在旋转系里
耦合变成**常数** $\tfrac12 A$；对反旋项（$e^{-i\omega t}$），旋转系残差频率
$\approx 2\omega$（THz），在一阶近似下平均为零——**旋转波近似（RWA）**：

$$
\text{env}(t)\cos(\omega t)A
\;\xrightarrow{\;\text{旋转系}+\text{RWA}\;}\;
\frac12\,\text{env}(t)\,A_{\text{res}},
\tag{2.3}
$$

其中 $A_{\text{res}}$ 只保留近共振对（软件用带宽 $|\,|E_i-E_j|-\omega\,|<2$ GHz 筛选），
振幅取 $\tfrac12$（t17 裁决：$\cos=\tfrac12(e^{i\omega t}+e^{-i\omega t})$ 旋转系只留
共振项 → 耦合 = ½·Rabi；这也与 Sambe 子空间耦合 $\Omega/2$ 的自洽约定一致）。
反旋项贡献（Bloch–Siegert 位移）在 THz 失谐下 ~$10^{-14}$ 相对量级，可安全丢弃。

**旋转参考的选择（rot_pick）**：对每个光场，找它耦合的"流形对"（如 60S ↔ 6P），
把其中**能量较低**的流形（6P 的全部 16 个超精细态）整体平移参考
$\text{ref}=\omega_{\text{opt}}$，于是旋转系里

$$
E^{\text{rot}}_i = E_i + \text{ref}_i,
\qquad \text{ref}_i=
\begin{cases}
\omega_f, & i\in\text{低流形（6P）}\\
0, & \text{其它（60S/60P…）}
\end{cases}
$$

60S↔6P 光学对在旋转系停在**几百 MHz 的慢失谐**（295.7 THz 被 ref 吸收），RWA 有效；
而 60S/60P 保持实验室能量，微波对不产生 THz 假差（t34 疑点 1 的结论：hard 路径
天然绕开 ip4 链式参考污染）。同流形多光场共享参考，第 2+ 个场带拍频相位
$e^{iC(\omega_f-\omega_{\text{ref}})t}$（t10 验证：指数拍频正确）。

### 2.2.2 微波硬积分：为什么微波"不当 RWA"

微波（17 GHz）驱动 60S↔60P。若对它也做 RWA，旋转系残差是 2ω_mw≈34 GHz 的反旋项与
载波本身的拍频——虽然不至于 THz，但 RWA 会丢弃真实的多光子/AC 结构。更本质的是：
**微波周期 0.058 ns，DOP853 自适应步长（~0.005–0.01 ns）完全能解析载波**，没必要做
任何近似。因此 hard 对微波**不做 RWA**，直接在实验室系积分 $\cos(\omega_m t)$：

$$
\frac{dy}{dt}=-iC\,u(t)\,\Big[\underbrace{\tfrac12\text{env}_{\text{opt}}(t)A_{\text{opt}}}_{\text{光场：旋转系常数项}}
+\underbrace{\text{env}_{\text{mw}}(t)\cos(C\omega_m t)\,A_{\text{mw}}}_{\text{微波：当场 cos 硬积分}}\Big]\,\big(u^\dagger(t)\,y\big),
\tag{2.4}
$$

其中 $u(t)=\exp\big(iC\,(E^{\text{rot}}-\min E^{\text{rot}})\,t\big)$ 是对角旋转因子
（相互作用绘景，IP）。展开后每项耦合带 $e^{iC(E^{\text{rot}}_i-E^{\text{rot}}_k)t}$ 的
IP 相位——光场项因 ref 吸收已停在慢失谐，微波项与 cos 载波拍频后共振对也变慢。
整个 RHS 无 THz 振荡，DOP853 可以大步长推进；数值内核全部 numba 编译 + 包络查表
线性插值（载波 cos 因精度要求由内核直算不查表，见 hard.py 头注释）。

### 2.2.3 两体 hard 与 V_dd 相位（E_lab ≡ E_rot）

两体推广（evolve_hard_pair）：态 $|i,j\rangle$ 的旋转因子
$u_{ij}=u_i u_j$（A/B 各自己的 $E^{\text{rot}}$），场项照 §2.2.2 组装成
$A\otimes I$ / $I\otimes A$ 三元组。

$V_{dd}$ 项的处理是 t34 核查的重点结论：**用实验室系能量差**做相位扭曲
$e^{iC(E^{\text{lab}}_k+E^{\text{lab}}_l-E^{\text{lab}}_i-E^{\text{lab}}_j)t}$ 而不是旋转系差。
为什么对？因为 hard 的旋转只落在 6P 等低流形上，而 $V_{dd}$ 的**存活元素**（经
$|V|\ge10^{-4}\max|V|$ 过滤后）全部连接里德堡态（60S/60P，ref=0）——在这些元素上
$E^{\text{lab}}=E^{\text{rot}}$，两种写法严格等价；会连接 6P 的 THz-dE 元素全被过滤
（其净效应 $V^2/\Delta\le10^{-8}$ MHz 可忽略）。数学上：场项相位
$e^{iC\,E^{\text{rot}}t}$ 与 $V_{dd}$ 项相位 $e^{iC\,E^{\text{lab}}t}$ 的差为
$e^{iC(\text{ref}_k-\text{ref}_l)t}$，存活元素 ref 差 = 0 → 等价。

### 2.2.4 hard 的物理图景与适用域

**图景**：hard 里光场是一个**经典相干波**——我们明确写出它的 $\cos(\omega t)$ 振荡，
再用旋转系"骑到波上"（RWA），把波带来的快振荡变成我们随波运动的慢变化；微波则干脆
让计算机逐振荡点积分。这个图像里光没有"光子数"，只有连续波的振幅与相位。

**适用域**（实测锚）：微波主导场景（π 脉冲/变频/任意波形）快且精确（≡ reference
1.93e-9）；弱光远失谐 + 微波（RWA 误差 ~(Ω/Δ)² 可忽略）；**唯一支持变频/任意相位场**
的引擎（t18：cos(φ_f(t)) 任意相位路径；ArbitraryWave 硬路由到 hard）。

**局限**：
1. RWA 是近似——强光近共振时 AC/dressed 处理有误差（t39：STIRAP 中段 2.2e-2，
   末态 ~9e-4）；
2. **跨流形光场链结构性不可用**（STIRAP：5S↔6P↔60S 两光场）：rot_pick 每光场独立
   取低流形，光 1 的耦合对在旋转系残差 −295.7 THz → RWA 完全失效。这是框架盲区，
   不是精度问题——纯光场链请用 sambe/mixed（φ-帧无参考选择）。

## 2.3 sambe 引擎：光 = 粒子（Sambe 光子块，无 RWA）

### 2.3.1 从 Floquet 到光子块：把载波变成静态对角

sambe 不设旋转系、不做 RWA，直接处理实验室系 (2.1)。核心思想来自 **Floquet 理论**：
周期哈密顿 $H(t+T)=H(t)$（载波周期 $T=2\pi/\omega$）的解可写成
$|\psi(t)\rangle=e^{-i\varepsilon t}\sum_n e^{-in\omega t}|c_n\rangle$（Floquet 态）。
把 $e^{-in\omega t}$ 吸收进扩展基，等价于把时间周期变成**一个额外的"光子数"量子数**：

定义**扩展空间**（Sambe 或 (t,t') 空间）基 $|i,n\rangle$（$n\in\{-M..M\}$ 光子数），
并令扩展态 $X$ 满足**时不变**方程

$$
i\frac{\partial X}{\partial t}=K\,X,
\qquad
K=\underbrace{\text{diag}\big(E_i-n\,\omega\big)}_{K_0:\ \text{静态块对角}}
+\underbrace{K_c}_{\text{光子数 } \pm1 \text{ 耦合}},
\tag{2.5}
$$

其中 $K_c$ 的矩阵元来自 $\tfrac12\text{env}(t)A$：$\cos(\omega t)A$ 的 $e^{+i\omega t}$
部分把 $|i,n\rangle$ 连到 $|k,n+1\rangle$（吸收一个光子），$e^{-i\omega t}$ 部分连到
$|k,n-1\rangle$（发射），振幅各 $\tfrac12\Omega_{ik}$。**载波消失了**——它变成了
$K_0$ 里的静态对角项 $-n\omega$。物理态 $|i\rangle$ 被嵌入**中央光子块**（n=0），
输出时按

$$
\psi_i(t)=\sum_n e^{-iC\,n\,\omega\, t}\,X_{i,n}(t)
\tag{2.6}
$$

重构（跨扇区相干叠加）。

**为什么可行（粒子图景）**：光场与原子交换的能量是 $\hbar\omega$ 的整数倍——
$\cos(\omega t)A$ 在频域就是两个 $\delta$ 峰（±ω），对应"吸收一个光子/发射一个光子"。
把光子数显式写进 Hilbert 空间，THz 振荡被代数化为静态能移 $-n\omega$，**积分成本与
载波频率完全无关**。这正是把光当作**粒子**（光子数 Fock 空间）的计算实现。

### 2.3.2 光子块截断 M：r=Ω/ω 判据

光子块必须截断到有限阶 $M$（$n\in[-M,M]$，$P=2M+1$ 个扇区）。截断误差来自被丢弃的
最高阶谐波。Floquet 谐波振幅按 Bessel 类展开 ~$(\Omega/\omega)^n$ 衰减，因此

$$
\text{截断误差}\;\sim\;\Big(\frac{\Omega}{\omega}\Big)^{2M},
\qquad
M_{\text{need}}=\Big\lceil\frac{\log(\text{tol}/C)}{2\log r_{\max}}\Big\rceil,
\quad r_{\max}=\max_{\text{近共振对}}\frac{\Omega_{ij}}{\omega},
\tag{2.7}
$$

（t58 判据库扫描实证：**M 由全偶极对最大 r=Ω/ω 决定，与失谐无关**——同 r 跨
ω/Δ=1~10⁵ 得到相同 M_need；收敛律 drift ~ C·r^{2M}）。量级：
光学近共振对 $\Omega\sim100$ MHz、$\omega=295.7$ THz → $r\sim3\times10^{-7}$ → **M=1**
即单光子阶（误差 ~10⁻¹⁴）；微波 $r\sim3\times10^{-4}$ → M=1..2。

软件按**逐载频**独立给 M（`per_freq_M` / 自动探针 `_m_analytic`），避免统一 M 把
光学拖到微波阶数造成扇区爆炸（如 22 态 3 载频：逐载频 M=(1,1,2) → 45 扇区，统一
M=2 则 125 扇区）。扩展维 $\dim=N\prod_g(2M_g+1)$，≤2200 走稠密 eigh，否则稀疏
expm_multiply。

### 2.3.3 时变包络：冻结段 + 缺陷修正 / frozen 纯冻结

包络 env(t) 使 $K$ 含时。两条传播路径：

1. **缺陷修正路径**（默认）：按包络曲率自适应分段，段内冻结中点哈密顿
   $K(\text{mid})$ → 精确指数 $e^{-iC K(\text{mid})\Delta t}$（一次 eigh 对角化），
   再对"包络偏离冻结值"的部分做 **Duhamel 缺陷修正**（Gauss–Legendre 节点 + 链上
   Richardson 外推 O(dt⁴) 提阶，误差判据 = 单步 vs 双子步差）。
2. **frozen 纯冻结**（`frozen_h_ns`，推荐用于全频段/THz 场景）：固定 h 内部分段，
   每段只做中点对角化 + 精确指数（无缺陷修正/无 Gauss 节点）。误差 = 段内包络变化
   （σ 缓变场景 h=1 ns 布居偏差 ~10⁻⁷），与 THz 解耦——THz 光子块本征跨度大时缺陷
   修正的 Gauss 核会不收敛，frozen 反而干净。

全常数包络退化为**单次对角化**，成本与总时长无关。耗散（寿命通道）以 NonHermitian
对角 $-\tfrac{i}{2}\Gamma$ 加入（显式落点/纯退相拒绝——离开建模空间即"丢失"）。

### 2.3.4 远失谐强对的 AC 化（sambe 的重要修正）

基组里有一类特殊对：**远失谐但 Rabi 极大**。典型：60S↔60P 能级差仅 17 GHz，但在
295.7 THz 光场下偶极矩阵元巨大 → $\Omega=13.6$ THz、$r=\Omega/\omega=0.046$——按
(2.7) 需要 M=4（dim 13068 不可行）。t55-t58 的 norm 漂移调查证明这是 M=1 截断不足
的根源（漂移 3.2e-3，M=4 收敛到 1.7e-7）。

物理上这类对**不转移布居**（远失谐），净效应 = **AC-Stark 位移**（二阶）。其正确
处理是"AC 化"：按 $r>r_{\text{crit}}=0.0135$ 判据把远强对从光子块耦合里剔除，同时
把它的位移注入物理能级：

$$
\delta E_i=-\frac{|\Omega_{ik}|^2}{4}\Big(\frac{1}{E_k-E_i-\omega}
+\frac{1}{E_k-E_i+\omega}\Big),
\qquad E_i\leftarrow E_i+\delta E_i
\tag{2.8}
$$

（far_sos 双向公式；t60 仲裁：±18.4 MHz 注入复现 M=4 锚，单向 Ω²/4Δ 的 ±156 GHz 是伪值；
t61 两体 AC 化后 620ns norm 从 1.08e-2 降到 1.75e-5，600× 改善。）注意近共振对
**必须保留在光子块动态处理**（其 dressed 位移由动力学自然涌现），只 AC 化远强对。

### 2.3.5 sambe 的物理图景与适用域

**图景**：sambe 里光是一束**粒子流**——每个 $\cos(\omega t)$ 被拆成"吸收/发射一个
$\hbar\omega$ 光子"的跃迁，原子与场的能量交换用光子数显式记账。载波振荡不是被"转慢"
而是被**彻底代数化**。这个图像在 THz 光场下是唯一实用的：不需要 RWA 假设，dressed
态、AC 位移、多光子过程全部自然涌现。

**适用域**：STIRAP/拉曼等多光场无 RWA 精确场景（t39 仲裁正确引擎，收敛 3.5e-7）；
强光近共振；全频段两体（AC 化降维后 dim 4356，GPU frozen ~2.4 min）。

**局限**：慢（frozen eigh/段，单原子 STIRAP 3400 ns ~25 s，两体分钟级）；**变频/任意
相位不支持**（光子块周期性假设——ArbitraryWave 硬路由到 hard）；稀疏自适应路径
（非 frozen 的 Volterra–GMRES）在 dim>700 真实场景卡死（EP6，frozen 不受影响）。

## 2.4 mixed 引擎：粒子与波的缝合（φ-帧统一架构）

### 2.4.1 为什么需要第三个引擎

- hard 快但光场有 RWA；sambe 精确但慢、且微波也占光子块导致两体维度爆炸
  （$\dim=N^2\cdot P_{\text{opt}}\cdot P_{\text{mw}}=7260$ @22 态）。
- 现实场景（交换全过程）是**光场 + 微波混合**：光场强近共振需要无 RWA，微波需要
  精确且要省维度。
- mixed 的策略：**光场进光子块（当粒子，M=1 足够），微波与 V_dd 不进光子块
  （当场，当波）**。两体 $\dim=N^2\cdot 3=1452$——5 倍降维。

### 2.4.2 φ-帧方程（核心推导）

设光场已进光子块：扩展态 $X$，$K_0=\text{diag}(E_i-n\omega_f)$（THz 级、静态）。
扩展空间方程

$$
\frac{dX}{dt}=-iC\big[K_0+K_c(t)+M(t)\big]X,
\tag{2.9}
$$

$K_c(t)=\sum_f \text{env}_f(t)K_{c,f}$ 光场耦合（光子数 ±1），
$M(t)=\sum_m \text{env}_m(t)\cos(\omega_m t)(I_P\otimes A_m)$ 微波（块内同 n）。

定义 **φ-帧** $\;\phi=e^{+iC\,D\,t}X$，**关键取 $D=K_0$ 全对角（不只去 E）**：

$$
\frac{d\phi}{dt}=iC\,D\,\phi+e^{iCDt}(-iC)(K_0+K_c+M)e^{-iCDt}\phi
=-iC\big[(K_0-D)+K_c\circ e^{iC\Delta K_0 t}+M\circ e^{iC\Delta K_0 t}\big]\phi .
$$

因为 $D=K_0$，第一项 $K_0-D=0$ **自动消对角**，只剩耦合项带相位：

$$
\boxed{\;\frac{d\phi}{dt}=-iC\Big[\sum_f \text{env}_f(t)\,K_{c,f}\circ e^{iC\,\Delta K_0\,t}
+\sum_m \text{env}_m(t)\cos(\omega_m t)\,A_m\circ e^{iC\,\Delta K_0\,t}\Big]\phi\;}
\tag{2.10}
$$

其中 $\circ$ 表示逐元素乘（Hadamard），$\Delta K_0[a,b]=K_0[a]-K_0[b]$。逐项看
$\Delta K_0$ 的频率量级：

| 耦合项 | 连接的扩展态 | $\Delta K_0$ | 频率 | 处理 |
|---|---|---|---|---|
| 光场近共振对 | $\|i,n\rangle\!\leftrightarrow\!\|j,n{+}1\rangle$ | $E_i{-}E_j{+}\omega_f\approx$失谐 | MHz 慢 | DOP853 积分 ✓ |
| 光场反旋对 | 反向 | $\approx-2\omega_f$ | THz | M=1 截断丢弃（~10⁻¹⁴）|
| 光场远失谐强对 | $\|i,n\rangle\!\leftrightarrow\!\|j,n{\pm}1\rangle$ | $\approx\pm295.7$ THz | THz | **AC 化**（§2.3.4）|
| 微波共振对（同 n）| $\|i,n\rangle\!\leftrightarrow\!\|j,n\rangle$ | $E_i{-}E_j\approx-\omega_m$ | cos 拍频 → 慢 | DOP853 积分 ✓ |
| 微波反旋对 | 同 n | $\approx+\omega_m$ | $2\omega_m=34$ GHz | DOP853 积分 ✓ |
| V_dd（两体，同扇区）| $\|i,j,n\rangle\!\leftrightarrow\!\|k,l,n\rangle$ | $E_i{+}E_j{-}E_k{-}E_l$（n 消掉）| MHz | DOP853 积分 ✓ |

**微波的关键性质**：微波不驱动光场载波（不改变 n）→ $\Delta K_0=E_i-E_j$ 与光子块
无关 → $\cos(\omega_m t)e^{iC(E_i-E_j)t}$ 的共振对拍频为慢变化——**微波的相互作用绘景
相位由 φ-帧自动给出，无需手工 E-IP**（对比 hard 要手工 $u$ 因子）。V_dd 同理：
同扇区 $\Delta D$ 与 n 无关，$e^{iC\,\Delta D\,t}$ 自动涌现（呼应 t34 疑点 2 的等价性结论，
但这里无需"E_lab≡E_rot"的存活元素论证——φ-帧天然正确，STIRAP 链式也天然正确）。

**输出重构**：$\psi(t)=e^{-iC K_0 t}\phi(t)$——THz 相位在输出时刻精确乘回。

**为什么 RHS 全慢**：近对失谐 MHz + 微波 GHz/34 GHz 反旋，无 THz → 单次 DOP853 贯穿
（无 Strang 分裂、无 eigh 冻结——t41 证明分裂是多余结构，t42 定案单 DOP853）。

### 2.4.3 mixed 的"正确组合"与历史教训

t40–t44 五连败逐一归因（理论推导文档 §6）：t40 无 φ-帧丢光场共振；t41 φ-帧但
Strang+eigh 多余（平台 1.5e-2）；t42 单 DOP853 但远对没 AC 化（60S↔60P THz 被过滤丢
AC 位移 2.36e-2）；t43/44 AC 化但伪值（单向 Ω²/4Δ、ac_diag 扩展对角——都错）。
"正确组合" = **φ-帧（D=K0 全对角）+ 远强对 AC 化注入物理 E + 微波/V_dd 当场**，
三者缺一不可。另有三个实现级陷阱（都已被修复并写入回归）：
- 反旋对必须 $|\Delta D|<20$ GHz 过滤（漏滤 → DOP853 步长被 THz 钉死）；
- RHS 打包分组键必须含驱动类型 (is_mw, ω, env 对象)——光/微波共用同一 env 函数对象时
  若只按对象分组，微波项会丢 cos 载波（用例 1：mixed 60S 不转移假象）；
- 两体 RHS 累加须与单原子同语义（单侧累加隐含 Hermitian，全矩阵对+双侧累加会 2× 加倍）。

### 2.4.4 mixed 的物理图景与适用域

**图景**：mixed 是"波粒二象性"在数值算法里的自觉运用——**同一个物理系统，高频部分
当粒子（光子块），低频部分当波（当场 cos 硬积分），V_dd 是常数相互作用当场处理**。
频率尺度的分工不是近似而是精确分解：光子块代数化 THz、当场积分解析 GHz 载波、
φ-帧把两者缝进同一个慢变量方程。

**适用域**（已实测 ≡ 锚）：光+微波混合（强光近共振需无 RWA + 微波精确）——核心适用域；
STIRAP 链（φ-帧无参考，hard rot_pick 盲区的解法，用例 4 ≡ frozen 1.56e-5）；近光+
微波共存（≡ frozen 4.35e-8）；两体 V_dd 交换（dim 1452 5 倍降维，620 ns norm ~5.7e-7）。

**局限**：光场须同载频（异频多光子块未实现，NotImplementedError）；光子块需固定载频
（变频/任意波形请用 hard）；远失谐微波对须 AC 化/当场（不当场会 THz 钉死）。

## 2.5 三引擎对比与自动路由

### 2.5.1 对照总表

| | **hard** | **sambe frozen** | **mixed** |
|---|---|---|---|
| 光的图像 | **经典波**（cos 载波 + RWA） | **粒子**（光子数边带） | 高频当粒子 + 低频当波 |
| 光场处理 | 旋转系 RWA（近共振 ½，反旋丢） | 无 RWA（光子块，dressed 全含） | 无 RWA（光子块 M=1 + 远强对 AC 化） |
| 微波处理 | cos 硬积分（无 RWA） | 光子块 M=2 | cos 当场（不进光子块） |
| 参考选择 | rot_pick（低流形 +ω）——STIRAP 链盲区 | 无（实验室系） | 无（φ-帧） |
| 两体 dim（22 态场景） | ~2420 级 | 7260（AC 化后 4356）/GPU | **1452（5 倍降维）** |
| 速度（22 态 620 ns） | ~2 s | 分钟级 | ~120 s（620 ns 两体） |
| 变频/任意波形 | **唯一支持** | 不支持 | 不支持 |
| 权威锚 | ≡ reference 1.93e-9 | 收敛 3.5e-7（t39） | ≡ frozen 1e-5~1e-8 |

### 2.5.2 auto 路由判据（引擎层 `route_auto`，脚本可用；GUI 不再暴露 auto 档）

> t71：GUI 方法选项收敛为 sambe/hard/mixed 三档后，auto 档不再出现在界面。
> 引擎层的 `route_auto` 保留供脚本调用（开发者想"按场配置自动选引擎"时），
> 其判据如下——GUI 默认 mixed 对"纯微波/任意波形"的自动降级（§1.5.3）即内联了
> 其中第 1、2 条：

`route_auto` 按场配置返回 (method, reason)：

1. 有 **ArbitraryWave/变频** → hard（唯一支持）；
2. **纯微波**（无光场）→ hard（微波硬积分精确且快 100×）；
3. **弱光**（RWA 误差 ~(Ω/Δ)² 可忽略）→ hard；
4. **强光近共振 + 微波** → mixed（无 RWA + 微波当场）；
5. **强光近共振纯光** → sambe（frozen 无 RWA）；
6. 两体时强光+微波 → mixed（5 倍降维）；强光纯光 → sambe。

"强光"判据：近共振失谐 <2 GHz 且光强 >10³ mW/cm²，或光强 >5×10⁴ mW/cm²（保守）。

---

# 第 3 部分 · 开发指南

## 3.1 总体架构

```
src/
├── core/                 # 纯计算核心（无 UI 依赖）
│   ├── static_atom.py    # 单原子静态：能级/Zeeman/DC/AC-Stark/偶极/谱线
│   ├── static_pair.py    # 两原子静态：V_dd/交换耦合/vdW/耦合强度地图
│   ├── atomic_levels.py  # 能级构造 make_levels（states 列表）
│   ├── field_build.py    # 场构造（field/group_field/real_field/arbitrary_field）
│   ├── waveforms.py      # 波形数据层（Wave/GaussianEnv/SquareEnv/ArbitraryWave）
│   ├── sambe.py          # Sambe 引擎总模块（t74：核心层+单体引擎+两体引擎
│   │                     #   合一；原 evolve_sambe/evolve_sambe_pair 并入，
│   │                     #   旧文件名留一行兼容 shim）
│   ├── evolve_reference.py # 参考积分器（H0 相互作用绘景 + Lindblad BDF；
│   │                     #   sambe 含耗散时的执行路径 + 全库权威参照）
│   ├── evolve_util.py    # 演化公用工具单源（decay/auto_decays/_as_vector/
│   │                     #   _decay_ops/_single_loss_rates/_loss_and_norm/
│   │                     #   _CONV 常量——原散落 evolve_rwa/hard，经 shim 借用）
│   ├── evolve_rwa.py     # ip4/Lindblad 遗留引擎（数值权威参照；工具已迁出）
│   ├── evolve_arbitrary.py # 任意波形第一性原理演化（实验室系 RK4）
│   ├── evolve_sambe.py   # 兼容 shim → sambe（过渡期后删）
│   ├── evolve_sambe_pair.py # 兼容 shim → sambe（含两场名惰性转发）
│   ├── evolve_sambe_pair_gpu.py # GPU 传播（研究原型）
│   ├── evolve_twofield.py # 两场引擎家族（单/两体团聚：光旋转系+微波光子块）
│   ├── hard.py           # hard 引擎单/两体同一内核（单原子=N_B=1 旁观维）
│   │                     #   + 输入格式构造 build_two_field（t74 归位）
│   ├── pair_evolve.py    # 两体 RWA（ip4，secular V_dd；遗留，权威自检）
│   ├── evolve_mixed.py   # 混合引擎单原子（φ-帧；依赖 sambe+acize+coupling）
│   ├── evolve_mixed_pair.py # 混合引擎两体（φ-帧）
│   ├── evolve_engine.py  # 统一入口 evolve_engine(method, n_body)
│   ├── route.py          # auto 路由 route_auto
│   ├── dop853.py         # DOP853 核心驱动（njit RHS 参数化）
│   ├── coupling.py       # RHS 公共层（t73）：pack_rhs + φ-帧统一内核
│   │                     #   rhs_phi_pairs（mixed 单/两体同一 RHS）
│   │                     #   + 三槽进度协议 stage_progress
│   ├── acize.py          # AC 化公共层（classify/far_sos/acize_physical）
│   ├── project.py        # 跨扇区投影公共层 project_pops
│   ├── tensor_fields.py  # 两体场提升（lift_single_A/lift_side_A/lift_fields）
│   ├── tensor.py         # 两体张量化（tensor_states/_reduce_*/_pair_loss_vectors）
│   └── __init__.py       # 兼容导出层（旧名 → 新模块 sys.modules 注册）
├── ui/                   # 界面
│   ├── fields.py         # 场面板（B/E/光场表/椭圆偏）
│   ├── ui_group.py       # 能级组管理
│   ├── ui_single.py      # 能级图画布
│   ├── ui_spectrum.py    # 静态分析面板
│   ├── ui_radar.py       # 耦合强度雷达图
│   ├── ui_waves.py       # 波形面板
│   ├── ui_dynamic.py     # 动态演化页（核心）
│   ├── evolve_worker.py  # 演化后台线程（EvolveWorker/Pair）
│   ├── app_state.py      # 能级组/场状态 store
│   ├── serialization.py  # 存档助手
│   └── ui_theme.py       # QSS 主题
├── _archive/             # 退役模块（floquet/pwc/ttl 等）
└── __init__.py           # ARC 数据目录重定向 + 顶层兼容导出
```

三层结构：
1. **静态物理层**（static_atom / static_pair）：能级、偶极、频移、V_dd——不含时；
2. **引擎家族**（evolve_*）：含时演化，返回契约统一（§3.3）；
3. **统一入口**（evolve_engine + route_auto）：按 method/n_body 分派；
   中间穿插 **t63 公共层**（sambe/dop853 与 N 无关核心 + coupling/acize/project/
   tensor_fields 单两体共用），避免每个引擎各写一遍语义漂移。

## 3.2 数据模型（写仿真脚本前必读）

### 3.2.1 states：能级列表

`states = make_levels(cache, specs)`，元素 dict：

| 字段 | 含义 |
|---|---|
| n, l, j | 主量子数 / 轨道角动量 / 总角动量 |
| kind | `'mJ'`（里德堡，纯 |J,mJ⟩）或 `'F'`（低激发，超精细混合） |
| F, mF, mj, m | 超精细量子数（F 态）/ mJ（mJ 态） |
| amps | F 态在 |mJ,mI⟩ 基的展开（复振幅 dict） |
| E_mhz | 能量（MHz） |
| label | 显示标签（"60S₁/₂ m_J=+0.5" / "6P₃/₂ F=2.0, m_F=+1.0"） |
| index | 列表序号 |

### 3.2.2 场：三种格式

| 格式 | 生产者 | 消费引擎 |
|---|---|---|
| `kind='exact'`（全偶极 A 矩阵 + 载频 + 包络） | `build_fields_exact(cache, states, waves)` | sambe / mixed / hard(两场 meta 内) |
| `kind='pair'/'real'/'arb'`（单对/组/任意波形） | `field / group_field / real_field / arbitrary_field` | ip4 系（evolve_rwa） |
| `two_field meta`（dict: E_rot + opt_terms + mw_groups） | `build_two_field(cache, states, waves)` | hard / evolve_two_field |

`kind='exact'` 场 dict：`{'kind':'exact', 'A': N×N 实对称 Rabi(MHz), 'omega_L_mhz': 载频,
'waveform': t→[0,1] callable, 'label'}`。

### 3.2.3 波形对象（Wave）

`Wave(name, freq_mhz, envelope, pol, offset_ns, intensity_mwcm2, drive)`；
包络 `GaussianEnv(center, sigma, peak)` / `SquareEnv(start, edge, height, fall_start)` /
`ArbitraryWave(t_ns, amp, ...)`。序列化：`serialize_waves / restore_waves`。

### 3.2.4 psi0 三形态

int（states 索引）| dict(n,l,j,m) | 复向量（长度 N，自动归一化）。

### 3.2.5 单位约定（全库统一）

能量/耦合/Rabi：MHz；时间：ns；角频率：rad/ns（$C=2\pi\times10^{-3}$）；
偶极：a.u.；光强：mW/cm²；Rabi 周频（π 脉冲 = 1/(2Ω) µs）。

## 3.3 引擎调用（核心 API）

### 3.3.1 统一入口

```python
from src.core.evolve_engine import evolve_engine

res = evolve_engine(
    method='mixed',          # 'mixed' | 'sambe' | 'hard'
    states=states,           # make_levels 输出
    fields=fields,           # n_body=1 时；kind='exact' 列表（mixed/sambe）
                             #   或两场 meta（hard）
    psi0=0,
    t_grid=np.linspace(0, 620, 621),
    n_body=1,                # 1 | 2
    # n_body=2 时改用：
    # fields_A=..., fields_B=..., psi0_A=..., psi0_B=..., vdd_matrix=V,
    # **kwargs 透传：frozen_h_ns / per_freq_M / cache / B_gauss ...
)
```

自动选择：

```python
from src.core.route import route_auto
method, reason = route_auto(states, waves, two_body=False, cache=cache)
```

### 3.3.2 返回契约（所有引擎同构）

**单原子**：`dict(t, pops, states, om_mhz, method, backend, loss[, psi_amp|rho,
error_report])`
**两体**：`dict(t, pops_2d, pops_A, pops_B, loss_A, loss_B, states_2d, method,
backend[, error_report])`

### 3.3.3 典型自写脚本（最小示例）

```python
import numpy as np
from arc import Rubidium87
from src.core import static_atom as core
from src.core.atomic_levels import make_levels
from src.core.waveforms import Wave, GaussianEnv
from src.core.evolve_sambe import build_fields_exact
from src.core.evolve_engine import evolve_engine

cache = core.ArcCache(Rubidium87())
# 1) 能级组：里德堡 mJ 态手动给；6P 超精细 F 态用权威 API 生成
B = 21.7
specs = [
    dict(n=60, l=0, j=0.5, kind='mJ', m=0.5),
    dict(n=60, l=0, j=0.5, kind='mJ', m=-0.5),
    dict(n=60, l=1, j=1.5, kind='mJ', m=-0.5),
]
for F, mF, _e, _w, amps in core.get_hyperfine_levels(cache, 6, 1, 1.5, B):
    specs.append(dict(n=6, l=1, j=1.5, kind='F', F=F, mF=mF,
                      amps=amps, E_mhz=float(_e)))
states = make_levels(cache, specs)
# 2) 波形 → 场（微波 σ−，17.236 GHz）
w = Wave('mw', 17236.743, GaussianEnv(70, 20), pol='σ-',
         intensity_mwcm2=5.5e-5)
fields = build_fields_exact(cache, states, [w])
# 3) 演化（hard：微波硬积分精确）
res = evolve_engine('hard', states, fields=fields, psi0=0,
                    t_grid=np.linspace(0, 300, 301), n_body=1)
pops = res['pops']          # (301, N)
```

两体脚本在示例 B（§1.6）基础上加 `pair_core.vdd_sparse` 预计算：

```python
from src.core.static_pair import vdd_sparse
V = vdd_sparse(cache, [dict(s) for s in states], R_um=12.0,
               theta_deg=30.0, phi_deg=0.0)
res = evolve_engine('mixed', states,
                    fields_A=fieldsA, fields_B=fieldsB,
                    psi0_A=0, psi0_B=0, t_grid=t, n_body=2,
                    vdd_matrix=V, cache=cache, B_gauss=21.7)
```

## 3.4 引擎内部结构速查

| 引擎 | 模块 | 主函数 | 内部核心 |
|---|---|---|---|
| ip4 | evolve_rwa.py | `evolve()` | `_rwa_rk4_core`（相互作用绘景 RK4）|
| sambe 单 | evolve_sambe.py | `evolve_sambe` | `_propagate_adaptive`（M 判定/探针 + 冻结/缺陷修正；结构走 sambe.SambeSetup——t73 单/两体同一实现） |
| sambe 两体 | evolve_sambe_pair.py | `evolve_sambe_pair` | `_pair_propagate`（稠密 eigh / 稀疏 Volterra-GMRES；结构走 sambe.SambeSetup）|
| hard 单/两体 | hard.py | `evolve_hard_fast`（别名 `evolve_hard`）/ `evolve_hard_pair` | 同一 `_pack_pair_hard` + `_rhs_hard_pair`（单原子 N_B=1 旁观维，t73b）+ `dop853._dop853_driver` |
| mixed 单 | evolve_mixed.py | `evolve_mixed` | `_MixedSetup` + φ-帧统一内核 coupling.rhs_phi_pairs（t73 与两体同一 RHS）|
| mixed 两体 | evolve_mixed_pair.py | `evolve_mixed_pair` | `_build_pair_geometry` + φ-帧统一内核 coupling.rhs_phi_pairs |

> **backend 标识**（结果 dict 里可查实际走的路径）：`'sambe-dense'/'sambe-sparse'`
> （单原子 Sambe）、`'reference-dop853'/'reference-bdf'`、`'hard-numba-dop853'`、
> `'mixed-phi-dop853'`、`'mixed-phi-dop853-pair'`、`'hard-numba-pair'`、`'ip4'` 等。

> **两场变体与 GPU**：`evolve_sambe_pair.build_two_field_pair(meta_A, meta_B)` +
> `evolve_two_field_pair(...)`（光场旋转系、只微波占光子维）与
> `evolve_sambe_pair_gpu.gpu_propagate_pair(...)`（CUDA，研究原型）已实现但未进
> GUI 主路径；一般开发用上表的六组入口即可。

## 3.5 公共层（t63，单/两体共用——新引擎优先复用）

- **`sambe.photon_geometry(E, groups, Ms)`**：光子块几何（P/dim/strides/nvecs/diag/
  nphase/p_center）——所有光子块引擎的地基。
- **`sambe.SambeSetup`（t73 单/两体合并）**：扩展空间结构唯一实现——块对角
  （含 V_dd 块对角嵌入与 −iΓ/2 NonHermitian 对角）、光子邻接耦合
  kron(T, ½K)+转置厄米、非光子常数项（两场变体光场）、hamiltonian_seg(mid)
  段装配、dense/sparse 按扩展维自动分流；`frozen_seg_propagate` 纯冻结段
  传播（含 eigh 缓存，键含全部波形）与 embed/reconstruct/physical_pops/
  dense_decomp 同收此处。单体（evolve_sambe/2200）与两体（700/两场 3000）
  仅差 dense_max 与构造参数。
- **`coupling.rhs_phi_pairs`（t73 mixed 单/两体合并）**：φ-帧统一 RHS 内核
  （a<b 单侧打包 + 内核厄米补齐；is_mw 载波门控；逐波时变 AC 对角
  ac_base×env²；NonHermitian gamma）——mixed 单/两体唯一差别只剩打包
  维度（N·P vs N²·P）；重构统一走 `sambe.sector_sum_pops`（相干扇区和
  Σ_p φ_p——光子相位在 φ-帧解析对消，旧单原子非相干版带 O(Ω/ω) 系统误差，
  t73b 修复）。
- **`hard`（t73b 单/两体合并）**：单原子 hard 合成 N_B=1 旁观维
  （E_B=[0]→uB≡1）直接走两体的 `_pack_pair_hard`+`_rhs_hard_pair`——
  旧单体内核/打包删除；`evolve_engine` 单原子 `method='hard'` 的
  `evolve_hard` 断链就此修复。
- **`dop853._dop853_driver`**：DOP853 全编译驱动（njit first-class RHS）。
- **`coupling.pack_rhs(pairs, t_end)`**：把 (a,b,v,dD,is_mw,w,env) 对打包成 RHS 数组；
  分组键 (is_mw, ω, id(env))。
- **`acize.acize_physical(E, fields_opt, r_crit)`**：AC 化唯一实现——r 判据分类 +
  远强对剔除 + per-field far_sos 位移。
- **`project.project_pops(X_t, P, N_phys, ns, ω, t)`**：跨扇区投影。
- **`tensor_fields.lift_side_A / lift_fields`**：两体场提升（A⊗I / I⊗A）。
- **`tensor.*`**：tensor_states / _reduce_A/B / _pair_loss_vectors。

## 3.6 静态 API（能级/偶极/频移/V_dd）

```python
from src.core import static_atom as core
cache = core.ArcCache(Rubidium87())
# 偶极与 Rabi
E0_sq = core.field_e0_sq(intensity_mwcm2)
d = core.dipole_amplitude(cache, row_i, row_k, q)     # q=0(π)/±1(σ±)
om = core.rabi_mhz(d, E0_sq)
# AC-Stark（权威后端：Floquet 直接对角化）
cfg = dict(enabled=True, frequency_hz=f*1e6, intensity=I,
           polarization='π', elliptical=False)
sh = core.direct_ac_stark_shifts(cache, groups, cfg, B_gauss=21.7)
# keys: (n,l,j,"m_J=±x",0)
# V_dd
from src.core.static_pair import vdd_sparse
V = vdd_sparse(cache, singles, R_um, theta_deg, phi_deg)
```

## 3.7 耗散（寿命通道，落点=丢失）

- 单原子：`evolve.decay(from_idx, gamma_mhz, None)`、`evolve.auto_decays(cache, states)`
  （ARC 寿命，τ≤100 µs）。
- 两体：`tensor._pair_loss_vectors(states, decays_A, decays_B, N, K)` → NonHermitian
  对角 −iΓ/2。显式落点/纯退相 → NotImplementedError（语义=离开建模空间即丢失）。

## 3.8 checks 回归

`python checks/selftest_all.py`（约 2 分钟）跑：core（静态 22 项）、dissipative
（Lindblad 18 项）、exact（单原子 Sambe 11 项）、exact_pair（两体第一性 10 项）、
two_field_pair、ui_selftest、pair_evolve。独立基准在 checks/ 其余脚本。

## 3.9 docs 文档索引

| 文档 | 内容 |
|---|---|
| `引擎能力边界-终稿.md` | 三引擎适用/局限总表（目标 3）|
| `混合引擎理论推导-光Sambe微波当场.md` | φ-帧 + AC 化完整推导 |
| `混合引擎两体理论方案-用例8设计.md` | 两体 V_dd φ-帧相位 + 当场同扇区 |
| `t58光子块截断判据.md` | r=Ω/ω → M 判据 |
| `t34核查结论.md` | 多频 RWA 参考污染核查（E_lab≡E_rot 等价性）|
| `认知固化II-AC化与混合引擎回归.md` | t55-t62 认知链（norm 漂移根因/仲裁）|
| `t62f认知固化.md`、`t66认知固化-单两体RHS语义不对称.md` | RHS 语义收敛教训 |
| `M选择指南-全频段两体Sambe.md` | per_freq_M 选择与性能档 |
| `auto路由方案-动态引擎选择.md` | route_auto 判据设计 |
| `AC化正式化设计-目标6.md` | ac_ize 正式路径 |
| `AI队友任务书-两场引擎.md` | hard/sambe/mixed 攻关史 |

---

## 附录 A：本次重大修复记录（t70）

**现象**：mixed 两体 375 ns 交换态 0.012 vs hard 0.558（差 ~50 倍）。

**根因 1（Hermitian 双向双计）**：`_build_pair_geometry` 用 `np.nonzero(Kcw)` 遍历全
矩阵无三角过滤——Kcw 是厄米矩阵（构造时 `K[c,r]+=conj`），每个物理耦合 (a,b) 与
(b,a) 都进 pairs_opt（2112 对中 1056 互为反向）；RHS 里每对又双向补齐 → Rabi 翻倍 →
AC 位移 ∝Rabi² 失真 4× → 交换对永不共振。V_dd/微波打包均有 `a0>=b0`/`ia>=ka` 过滤，
唯光场漏（T4/T6 同款 bug 第四次）。

**根因 2（注入语义错误）**：`_ac_ize_pair` 注入 direct_ac_stark_shifts 全量静态位移
（含近对 60S↔6P 贡献），但近对已保留在光子块 M=1 动态自然涌现位移（对角化 M=1/2/3/4
收敛 −18.87 ≡ 静态 −16.47）→ 注入 = 双计过补偿。

**修复**：pairs_opt 加 `a>=b` 三角过滤（反向由 RHS 厄米补齐）；`_ac_ize_pair`/
`evolve_mixed_pair` 加 `ac_inject=False` 默认不注入（True 保留旧语义）。
**验收**：375 ns max|Δpops|=8.19e-5；620 ns 全程 1.39e-4 ≤ 1e-2；norm ~5.7e-7。

---

*文档版本：2026-09（对应代码 commit 8d6bd2c 之后）。*
