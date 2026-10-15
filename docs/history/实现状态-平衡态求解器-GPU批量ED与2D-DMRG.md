# 实现状态 · 平衡态求解器（GPU 批量 ED + TeNPy 2D 柱形 DMRG）+ 标杆判据

> **本轮（用户离开时的自主实现）完成**：把"通用多体引擎·平衡态"的两条主力**落地为可运行代码并自测**——
> ① **GPU 批量 ED/Lanczos**，② **TeNPy 2D 柱形 DMRG**；各配**独立、交叉一致的标杆判据**。
> 全部接入 `run_checks` 分档、`--core`/`--fast` 全绿。单位/基序沿用 manybody 家族（MHz、bit=1↔|r⟩、MSB）。

## 0. 结论（TL;DR）
- 本轮（含“前沿普查→选一个实现”）在 GPU 批量 ED + TeNPy 2D 柱形 DMRG 之外，又**新实现 KPM/Chebyshev 谱方法**（general-H 有限温 + 态密度，非 stoquastic 无符号问题）。7 条 equibench 判据 `run_checks --only equibench` 全绿。
- 环境实测：**tenpy 1.1.1 已装**、**CuPy 可用**、**torch 2.7.1+cu128 (cuda True)**、GPU=**RTX 4050 Laptop**、CPU 16 核。
- GPU 批量 ED（`equilibrium.batched_ground_state`）：与逐点精确、numpy 批量化 **ΔE₀~1e-14 一致**；真机吞吐 **2.4–7.1×**（见 §4）。
- TeNPy 2D 柱形 DMRG（`dmrg.dmrg_ising_ladder`）：2D 阶梯基态 **DMRG ≡ TeNPy ExactDiag ≡ 我们的 ED ≡ GPU 批量 ≡ 独立 Kron ≡ 经典解析**，四方/六路对齐 <1e-6（ExactDiag 比对达 1e-15）。
- 判据政策同步：`requirements.txt` **删除一切"禁用库"文本**、`tenpy` 入依赖、GPU 库标"可选按需"；更新记忆"技术栈"去禁令、记"求解器禁 NQS/只用精确可控方法"、"energy-level 助手直接实现代码"。

## 1. 新增代码（文件地图）
| 文件 | 内容 |
|---|---|
| `src/core/manybody/equilibrium.py` | 平衡态求解器：`ground_state`(稠密 eigh / **Lanczos eigsh**)、`ground_state_imagtime`(虚时幂法)、`thermal_from_h`(精确配分函数 E/F/lnZ/C/⟨n⟩_β/χ)、`detect_gpu` + **`batched_ground_state`**(CuPy/torch/numpy 批量化本征，VRAM 分块、自动回退、确定性) |
| `src/core/manybody/dmrg.py` | **TeNPy 2D 柱形（2-leg Ladder）DMRG 后端**：`ising_ladder_model`(CouplingMPOModel)、`ising_ladder_full_hamiltonian`(稠密 2^N，供交叉验证)、`dmrg_ising_ladder`(基态 E0/键维/纠缠)；tepy 缺失时 `ImportError`，判据自动 SKIP |
| `checks/equibench_engine_ed_selfcheck.py` | 引擎 ED ≡ 独立 Kron ≡ 手推显式（三路径同谱）+ XX 二聚体/blockade·Fibonacci + 非厄米对角约定 |
| `checks/equibench_gs_solvers_selfcheck.py` | 稠密eigh ≡ Lanczos ≡ 虚时 ≡ 独立 Kron（随机含复厄米，N=5–7） |
| `checks/equibench_analytic_thermo_selfcheck.py` | XX 自由费米子 E₀=Σmin(0,μ)、二聚体 E/lnZ/C、N=1 Curie、β→∞ 趋基态 |
| `checks/equibench_heisenberg2d_selfcheck.py` | 引擎参数映射 ≡ 独立 2D XXX + **Lieb-Mattis 总自旋**严格值 + Lanczos≡稠密 |
| `checks/equibench_gpu_batched_selfcheck.py` | **GPU 批量 ED**：cupy/torch/numpy-batched ≡ 逐点 ≡ 独立参照 |
| `checks/equibench_dmrg_selfcheck.py` | **2D DMRG**：DMRG ≡ 我们 ED ≡ GPU 批量 ≡ 独立 Kron ≡ 经典解析；truncerr 收敛 |
| `src/core/manybody/kpm.py` | **KPM/Chebyshev 谱方法**（general-H 有限温 E/C/F + 态密度 DOS；稀疏 matvec、复/非 stoquastic 无符号问题、Hutchinson 随机迹）| ← 本轮前沿普查新增 |
| `checks/equibench_kpm_selfcheck.py` | **KPM 标杆**：有限温(精确矩)≡谱分解、随机迹确定性+nvec 收敛、DOS θ-测度矩 | 全档 |
| `src/core/manybody/dmrg.py`（①） | 新增 `rydberg_ladder_model`/`dmrg_rydberg_ladder`/`rydberg_ladder_full_hamiltonian`：**一般复耦合(V/J/A+σ⁺σ⁺)经注册自定义格点算符直喂 TeNPy MPO**（零分解零常数，与引擎逐字同） |
| `checks/equibench_dmrg_general_selfcheck.py`（①） | **MPO≡引擎 H 谱(含复 J/A)+DMRG≡精确** → 证 DMRG 后端能解一般（非 Ising）里德堡 H | --fast |
| `src/core/manybody/eq_route.py`（②） | **算法路由+统一入口**：`route_equilibrium`（按 维数/want/is_ladder/batch/gpu 选 ed/gpu-batch/dmrg/lanczos/kpm）+`solve_equilibrium` 统一 Result | 核心层 |
| `checks/equibench_router_selfcheck.py`（②） | 路由决策正确 + 跨后端(ed/lanczos/dmrg/gpu)基态一致 + thermal/dos ≡ 精确 + 契约 | --fast |
| `src/core/manybody/eq_engine.py`（②） | **产品引擎**：`build_equilibrium`(快照→calib→spec)/`run_equilibrium`(路由→统一Result)/`solve_equilibrium`(一步入口)；经 `manybody/__init__` 导出 |
| `src/core/runspec.py`（②） | `_ENGINE_TABLE` 加 `gs/thermal/dos×many`；`evolve_run` 与 `build_run(mode==2)` 各加 **equilibrium 旁路**（按 `manybody.want`），含时路径/契约不动 |
| `checks/equibench_dispatch_selfcheck.py`（②） | 端到端：假 calib spec→`evolve_run`→各后端一致+契约（无 ARC） | --fast |
| `checks/equibench_product_selfcheck.py`（②） | 封顶：真 ARC 快照→`build_run`→`evolve_run`→gs/thermal Result（产品/AI/GUI 共用入口可调用） | 全档 |
| `ui/ui_manybody.py`（②） | 多体页新增 **`请求量` 下拉**（''=含时/gs/thermal/dos）写 `manybody.want`→同一运行按钮经 `build_run` 走平衡态 | gui_manybody 7/7 |

## 2. 如何运行 / 分档 / 实测耗时
```bash
python checks/equibench_dmrg_selfcheck.py           # 单条（DMRG，需 tenpy）
python run_checks.py --core     # 22 条 fail=0（4 条秒级 equibench 在内）
python run_checks.py --fast     # 38 条 fail=0（含 equibench_gpu_batched 12.2s）
```
| 判据 | 单跑耗时 | 落档 |
|---|---|---|
| engine_ed / gs_solvers / analytic_thermo / heisenberg2d | 1.2–2.2s | **--core**（秒级内循环） |
| gpu_batched | 12.2s | --fast（含 GPU/CUDA 初始化开销） |
| dmrg | 16.5s | 全档（tenpy+GPU 导入 >15s；无 tenpy 自动 SKIP） |

## 3. 正确性（实测数字）
- **GPU 批量 ≡ CPU**：B=32,N=8 与 B=64,N=10 网格，`max|ΔE₀| = 7.1e-14 / 6.4e-14`；cupy ≡ torch ≡ numpy-batched 三者同能。
- **2D DMRG**：L∈{3,4,5} 阶梯、含 hz/复… 情形，`DMRG` 与 `TeNPy ExactDiag` 差 **~1e-15**，与我们 ED/GPU/独立 Kron 全 <1e-6（DMRG 截断容差）。经典极限 hx=hz=0：`E₀ = −Jz/4·(3L−2)` 解析锚命中 <1e-8；truncerr 1e-6→1e-12 收敛 <1e-5（N=16，ED 不可行，验纯 DMRG）。
- 装配层（engine_ed/heisenberg2d/Lieb-Mattis）此前已锁：引擎 ≡ 独立构造 Δ<1e-15、总自旋严格值精确。

## 4. 性能（RTX 4050 实测）
| 批量(点数 B × 系统 N) | CPU 批量化 | GPU 批量化(auto=torch) | 加速 | 精度 |
|---|---|---|---|---|
| 32 × N=8 (dim 256) | 3.15s | 1.32s | **2.4×** | ΔE₀ 7e-14 |
| 64 × N=10 (dim 1024) | 77.9s | 11.0s | **7.1×** | ΔE₀ 6e-14 |

> 规模越大（矩阵越厚、点数越多），GPU 批量本征的收益越显著——这正对应"**蒸馏造标签 = 参数网格上海量独立基态**"。
> 与前述判断一致：**张量网络(DMRG) 的 GPU 收益低**（顺序小算子、CPU/BLAS 为主），故 GPU 用在**批量精确对角化**最划算；DMRG 负责 **ED 够不到的 2D 尺寸**（键维可控、走 CPU）。

## 5. 加速版图（已就位，供后续用满）
- **GPU**：`batched_ground_state` 自动选 cupy>torch>numpy，VRAM 分块（默认预算 1.2GB，适配 6GB 卡）。
- **numba**：既定 `@njit nogil cache`（11 内核）；`parallel=True`/`prange` 可用未启用（下一轮热循环可上）。
- **BLAS 多线程**：sweep 已管控 OMP/MKL/OPENBLAS/NUMBA 线程；worker 内强制单线程防超订。
- **进程并行**：`src/sweep/runner.py` 的 `ProcessPoolExecutor + plan_workers` 是**批量造标签的天然骨干**（多进程×单线程）。

## 6. 已知边界 / 下一轮（诚实）
1. ~~DMRG 模型仅均匀 Ising~~ → **已做到底**：一般复耦合(逐键 V/J/A + σ⁺σ⁺ 对产生)经**注册自定义格点算符**直喂 TeNPy MPO，MPO≡引擎H（机器精度）。
   → 又补上 **任意点云（多腿/无周期）**：`dmrg.cylinder_order`（蛇形排序，分行尺度=最近邻距离/4）
   + `dmrg.terms_from_calib`（引擎 calib → 逐项 (coeff,a,opA,b,opB)，按 MPS 下标归一）
   + `dmrg.rydberg_cluster_model`（**自写 autoMPO**：单胞含全部 N 格点、1 run ⇒ 单胞下标==MPS 下标，
   任意 (i,j) 对用 `n=0` 表达）+ `dmrg.dmrg_rydberg_cluster`（返 E0/⟨n_i⟩/chi/truncerr/order）。
   验证：`equibench_dmrg_cluster`（MPO≡引擎 H **按置换逐元素** 3.6e-15、DMRG≡ED≡Lanczos、键维收敛、产品接线）。
   ❗ 实踩坑：蛇形反向行使原子对下标颠倒，**只换下标不换算符位置**会把复 J 的 σ⁺σ⁻ 共轭错（偏差 2e-1，谱看着“差不多”却全错）——已把正确做法写成 `two_site()` 并用专例钉住。
2. ~~未接产品分派~~ → **已全部完成**：`eq_route`(路由+统一入口) + `eq_engine`(build/run/solve) + `runspec` 双旁路(`build_run` mode==2 & `evolve_run` 按 `manybody.want`) + 分派表 + **GUI 「模式按钮 + 后端下拉」** + schema(`manybody.want`/`betas`/`eq_backend`)。`equibench_dispatch`/`equibench_product`/`gui_manybody` 均绿 → **ED/GPU批量/Lanczos/KPM/DMRG 全部能经产品/AI/GUI 入口正常调用**。
   旧桩 `eq_engine._geom_is_ladder() ≡ False`（使 DMRG 在产品路径上永不可达）已删，换成 `_dmrg_ok()`：按**跨过切割的键数**判“DMRG 值不值得”（⚠ 不能用最大排程：蛇形下 2-leg 梯子的首个梯级键天然跳整条链，会误拒好几何）。
3. **真·大 2D（本征 2D，非柱）**：iPEPS/TNR 未做；GPU 批量 ED 尺寸天花板 N≈20–22（2^N），DMRG 柱负责更"长"、iPEPS 负责更"宽"。
4. **SSE-QMC**：仍走"将来 Julia 桥或纯 Python 重写"；本仓已精读其论文（sign-free 边界）；对 stoquastic 子集是大规模 2D 热力学之王，但受限于非稳态性/平衡态。
5. **GPU 后端选择**：当前 `auto` 命中 torch（cupy 探测 `Device(0).asnumpy()` 未过）；两者都可用，若偏好 CuPy 需修探测或显式 `backend='cupy'`。
6. **多体页能级衔接（口径 A，已完成）**：多体的 g/r **不再来自原子表**，而是直接取「能级组面板裁剪后的全部能级」（`_levels()`→`runspec.build_states`，与动态基组同源）。**恰好 2 个** → g=第1、r=第2；**≠2**（含 >2 或 0）→ 参数页顶部常驻一行红警告，且“添加原子/画布点击”被守卫拦下（非阻塞 `main.log`，不弹模态）。能级组变动经 `main.groups_changed` 自动把已存原子 g/r 同步为当前 2 能级。同步删旧 `_default_gr`/“取能级组”按钮死路。判据 `gui_manybody` 新增 g/r 来源断言 + “≠2→不加原子且警告”守卫（PASS=7/35 子）。

## 7. 多体页：演化 / 平衡态 模式分家（已完成）

用户判断“演化和平衡态求解压根是两件事”⇒ 页顶一个互斥模式按钮（时间演化 / 平衡态），
**每个模式只显示对应组件**：

| | 时间演化（`manybody.want=''`） | 平衡态（`manybody.want∈gs/thermal/dos`） |
|---|---|---|
| 参数分区 | 含时求解器 `manybody.solver`(auto/ed/tebd)、初态 ψ0、ED 维度上限 | 求什么(gs/thermal/dos)、后端 `manybody.eq_backend`(auto/ed/gpu-batch/lanczos/kpm/dmrg)、betas |
| 中央结果区 | `res_canvas` 三格：产物态布居 / ⟨n_i⟩(t) / C_ij | `eq_canvas`：gs→⟨n_i⟩ 柱+E₀/gap、thermal→E(β)&C(β)、dos→ρ(E)，右格读数 + 顶部摘要行 |
| 运行按钮 | 「运行多体演化」 | 「求解平衡态」 |
| 共用 | 能级状态警告、几何点云表、二阶/n_window/guard、成本预估、NOT_YET/DECIDED_NO | 同左 |

模式真值就是 `manybody.want`（经 choke point 写，actor=gui）；`refresh_from_session` 反向驱动
界面（AI/存档写 `want=dos` ⇒ 页面自动切到平衡态并选中 dos）。

**顺带修掉四个“平衡态从 GUI/AI 路径调不通”的真 bug**（旧判据只直调 `build_run/evolve_run`，没经过下面这些消费点，故一直没暴露）：
1. `runspec.check_run` 取 `run['t']` → 平衡态无时间轴，`ops.prepare` 直接 KeyError。现平衡态专属分支：查原子数/want/betas 为正，dmrg 无 ladder 时点名报错。
2. `runspec.summarize_run` 取 `run['t_max']`/`npts`/`psi0` → KeyError。现报 equilibrium/want/backend/dim/betas/kpm。
3. `runspec.result_summary` 只认 `pops`，否则报“引擎契约被破坏”。现新增 `_equilibrium_summary`（ok=True + E₀/gap/n_atom 或 E/C–β 首尾或 ρ 极值）。
4. `mecha_v2.bridge.ElOps._land` 白名单只含 `pops/t/…` ⇒ **平衡态数组全部静默丢失**（GUI 无图可画）。现含 `betas/E/C/F/energies/rho/n_atom`。
另：`eq_route` 的 ed/lanczos 基态现补回 `n_atom`（逐原子 ⟨n_i⟩，与引擎同 MSB 基序），否则 gs 结果区无内容可画。

判据：`gui_manybody` 新增 H 组（两分区互斥显隐、控件不重叠、want 互斥写回、三种平衡态结果渲染）
→ **PASS=8 / 44 子**；`equibench_product` 新增一组钉住上述四个消费点 + `_land` artifact 往返回
→ 2 组 / 8 子全部通过。`--core`(22) / `contract`(4) / `manybody`(4) / `equibench`(11) / `ai`(7) / `hub`(1) / `sweep`(2) **均 fail=0**。

## 8. 多体页状态保存/导入 + 演示存档（已完成）

### 8.1 查出的四个真缺陷（均已修 + 判据钉住）
会话是真值源，但**三类状态住在会话外**，而旧代码只在“启动 seed”时同步一次：
1. **能级组不回灌**：`groups.list` 写进会话后，`main.groups`（住在 `LevelGroupStore`）仍为旧值
   ⇒ 多体页 `_levels()` 拿到 0 个能级，误报“需恰好 2 个”。修：`LevelGroupStore.replace()`（原样保留 ms/Fs 裁剪）。
2. **多体页面板不刷新**：`_load_state` 只调 `refresh()`（雷达/绘图），不碰原子表/模式/betas ⇒
   导入后表是空的、模式停在旧值。修：`MainWindow._resync_from_session()`。
3. **登记型控件不回灌（全局问题，不只多体页）**：`binder` 只在构造时 seed，任何**外部写**
   （恢复存档、AI/MCP `set_config`）都不重填 ⇒ 实测导入 `eq_backend='dmrg'` 的档后下拉仍是 auto。
   修：`Binder.pull_all()`（与已有 `push_all` 对称；全程 `_updating` 防反写，会话值为 None 的键不动控件）。
4. **`push_all` 定义了却从没人调**（其注释写明“存档/运行前同步用”）⇒ `push_all()` 接进 `_save_state`。

另：旧 v1 存档（`save_state/*.json` 顶层 `{format,mode,params}`）在 v2 下会被 Gate 以
`UnknownKey 'format'` **整批拒绝**（桥的文档明写不兼容 v1）。不在 GUI 里默默拆包（拆包后部分生效更难查），
而是**明确报出**“这是 v1 旧存档，请在当前版本重存一份”。

### 8.2 8 份演示存档（`scripts/make_demo_states.py` 生成）
全部经**真 Gate** 写入再回读校验（键名/取值一定合法），每份自带“能导入”证明：

| 档 | 几何 | 请求/后端 | 要你看的东西 |
|---|---|---|---|
| 基态-链8-ED | 1×8 (n=8) | gs / auto→ed | 基线，秒级 |
| 基态-2D-3x4-ED | 3×4 (n=12) | gs / ed | 2D 仍能精确算（dim=4096）|
| 基态-2D-4x4-Lanczos ┐ | 4×4 (n=16) | gs / lanczos ↔ dmrg | **同一哈氏顿两个独立算法**：换后端各跑一次，E₀/⟨n_i⟩ 必一致（稠密 ED 在此要 68 GB）|
| 基态-2D-4x4-DMRG   ┘ | | | |
| 基态-2D-2x10-DMRG超ED | 2×10 (n=20) | gs / dmrg χ=80 | ED/Lanczos 撞内存墙，**只有张量网络能算** |
| 有限温KPM-3x3 | 3×3 (n=9) | thermal / kpm | E(β)、C(β)；β→∞ 端应趋基态能 |
| 态密度DOS-2x6 | 2×6 (n=12) | dos / kpm | ρ(E) 整条谱形状 |
| 时间演化-阻塞链8 | 1×8 | want=''（演化） | 另一条路：布居/⟨n_i⟩(t)/C_ij；导入后模式按钮自动落在「时间演化」 |

实测（RTX 无关，CPU；标定+求解）：n=8 ≈ 1s；n=12(r_cut=7) 装配 0.6s/解 8s；n=16 DMRG(χ=80) 十几秒；
**n=20 从 121s 降到 17s** —— 差别在 `manybody.r_cut_um=7`：190 键→28 键、梯级最大程 19→2
（`cylinder_order` 改为**梯级优先**后才能吃到这个好处，蛇形排序会让首个梯级键跳整条链）。
新增 GUI 旋钮：`manybody.chi_max`（χ）、`r_cut_um`（已有）、`eq_backend`。

### 8.3 显示侧又查出三个缺陷（用户实跑 GUI 反馈）
5. **平衡态结果图完全空白 + 红字**“The truth value of an array with more than one element
   is ambiguous”——根因是我写了 `np.asarray(res.get('n_atom') or [], float)`：artifact 里
   `n_atom` 是 **numpy 数组**，`arr or []` 直接抛错，被自己的 except 接住 ⇒ 图上只剩一行红字。
   修：统一走 `ManyBodyPanel._arr(res, key)`（显式判 None）。**教训：数组绝不能用 `or` 默认值。**
6. **图上中文全是方框**——`app_entry` 模块级确实设了 YaHei，但**`arc` 包在 import 时把
   `rcParams['font.family']` 改成 `'serif'`**（`arc/calculations_atom_pairstate.py:82`），而 ARC 是在
   `MainWindow()` 建 cache 时才导入 ⇒ 设定时机被事后覆盖。用 monkeypatch `RcParams.__setitem__`
   抓调用栈定位。修：`setup_cjk_fonts()` 前先主动 `import arc`（把覆盖提前完吃），建完窗口再兜底一次。
   另：YaHei **缺 `⟨⟩`(U+27E8/9) 与 `₀`(U+2080) 字形** ⇒ 图上文字改用 mathtext
   (`$\langle n_i\rangle$`、`$E_0$`)，QLabel 保留纯文本。
7. **能级组左侧列表空**：`EnergyGroupPanel._refresh_rows()` 只被“添加/裁剪/删除”**显式**调用，
   没订阅 `store.changed` ⇒ 恢复存档直接换 store 时列表空着。修：`bind_store` 里
   `groups_store.changed.connect(group_panel.refresh_list)`（收拢到信号，不再散布显式调用）。
   附带修正：**右侧能级表原来自己 `arange(-j, j)` 枚举 mJ、无视 ms/Fs 裁剪** ⇒ 表里显示 5 个
   能级而多体页按裁剪后 2 个取 g/r（两边自相矛盾）。现改为直接遍历已含裁剪的 `build_level_rows` 结果。

### 8.4 新判据 `checks/mb_state_roundtrip_selfcheck.py`（32s，全档）
5 组：A 逐份导入→查能级/表行数/模式/请求量/后端/betas 全同步；
B 档里故意写错的 g/r 会被能级组纠正（单一真源）；C 真跑（链8 基态 + 含时）；
**D 字体守卫**（`font.family==sans-serif` + 画一次结果图零缺字形告警 + 柱子真画出来）。
**守耗时纪律**（用户要求全测 <3min）：大尺寸跨后端一致性归 `equibench_dmrg_cluster`，不在此重复烧算力。

## 9. 给回来后的你
- 一切**可复跑验证**：`run_checks --core/--fast` 全绿即代表两求解器与所有标杆自洽；DMRG/GPU 判据在全档/需 GPU·tenpy 环境。
- 最自然的下一步（按你"通用第一性 + 蒸馏"目标）：**(a)** 把**逐键复 V/J/A + mJ** 通过矩阵算符喂进 TeNPy MPO 并用现有多方对齐判据验；**(b)** 把 `batched_ground_state` 接进 **sweep 造数据集**（蒸馏数据工厂）；**(c)** 把 `equilibrium`/`dmrg` 接进 **runspec/GUI** 的 `mode==2` 分派与路由（1D→DMRG/TDVP、2D-stoquastic→QMC、2D-general→DMRG柱/批量ED）。

*实测数字为 RTX 4050 Laptop / torch-cuda 后端一次性测量，换机/换后端会变，但"GPU 批量≡CPU、DMRG≡精确"的正确性结论与实现无关。*
