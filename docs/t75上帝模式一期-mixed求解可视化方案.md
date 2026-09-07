# t75 方案：用户完全控制（上帝模式）· 第一期——mixed 引擎求解过程可视化

> 方向（船长 2026-09-07 定调）：现有 UI 暴露的参数归为**简单模式**；开发**上帝模式**，
> 允许更细粒度的操作与"人为指定而非自然模拟"。本期只做第一层地基：
> **把 mixed 引擎的求解过程变成可视、可查、可导出的**——哈密顿量的解析与数值显示、
> 各处理步骤中间结果、求解过程实时显示。后续 L2/L3 期再逐级解锁参数覆写与手动管线。
>
> **非目标（本期不做）**：不改变任何数值路径与默认结果；不动 sambe/hard 引擎内部；
> 不做参数覆写 UI（L2）与分步手动执行（L3）。

---

## 1. 现状盘点：mixed 求解链与可观测缺口

单原子 `evolve_mixed`（src/core/evolve_mixed.py:165）与两体 `evolve_mixed_pair`
（同文件 :420）的实际处理链：

```
输入 states/fields/psi0/t_grid
 ① 能量基向量 E = E_mhz − min          （bare，纯物理）
 ② 场分流 f_opt / f_mw                 （MW_MAX_MHZ = 1e5）
 ③ AC 化 ac_ize_physical_E / _ac_ize_pair
    classify_pairs：r = Ω/ω vs R_CRIT → 保留/剔除
    far_sos_pair_shift：远强对 → 位移（单原子注入 E2；两体 t70 定案不注入）
 ④ 载频分组 _group_by_freq → grps_opt/grps_mw
 ⑤ _MixedSetup / _build_pair_geometry：
    _m_analytic → 逐载频 M（r 判据）→ _SambeSetup → P/dim/D_full/Kc_dense
    耦合对打包 pairs（光场 a<b 且 |ΔD|<BW_OPT；微波逐扇区且 ||ΔD|−ω|<MW_NRES_BW）
 ⑥ AC 时变对角 ac_base/ac_env_tab（单原子；两体 t73b 恒关）
 ⑦ pack_rhs → njit 数组（iis/jjs/vals/dDs/is_mw/wcos/env_tab）
 ⑧ _dop853_driver(rhs_phi_pairs, …) 单次贯穿积分
    prog 三槽进度协议；stat = [nfev, n_acc, n_rej, —]
 ⑨ sector_sum_pops 跨扇区相干投影 → pops / loss / error_report
```

**现状可观测点**：状态栏 log 数行（AC 位移范围/P/dim/耦合对数）、进度条（t72
三槽协议）、结果 error_report（nfev/norm_dev/P/dim）、完整轨迹 `y_out (len(t), dim)`
（用完即弃，只留投影后的 pops）。

**缺口**：①③④⑤⑥⑦ 的全部中间产物（分类表、M 判定、扇区结构、K0、逐波 Kcw、
pairs 清单）算完即丢；⑧ 只有进度百分比，无步长/收敛历史；哈密顿量从未以
矩阵形态示人；诊断信息是无结构的 log 字符串，UI 无法消费。

## 2. 总体设计：diag 快照 + 求解检视器 + 上帝模式分层

三条设计原则：

1. **引擎产出结构化诊断，UI 只做展示**——引擎侧新增"诊断快照"对象随结果返回
   （`out['diag']`），采集逻辑全部挂在现有分支上，数值路径零改动。
2. **重建而非插桩**——哈密顿矩阵不在 njit 内核里采集，而是事后用独立纯函数
   从 diag 数据**重组**（诊断正确性顺带成为内核的交叉验证）。
3. **上帝模式分层渐进**——L1 查看（本期）/ L2 覆写 / L3 手动管线，UI 用一个
   总开关逐级解锁，简单模式完全不受影响。

```
┌ GUI 简单模式（现状不动）───────────────────────────┐
│ 动态页 → 运行演化 → 进度条 → 曲线                    │
└──────────────┬─────────────────────────────────────┘
               │ 上帝模式开关 ON（动态页顶部）
┌ L1 求解检视器（本期交付）──────────────────────────┐
│ 求解器检视面板：七个标签页（§3）+ 导出                │
│ 数据源：out['diag']（结构化快照）+ dt 历史 + H 重建  │
└──────────────┬─────────────────────────────────────┘
   L2 参数覆写（下期）：M/R_CRIT/BW_OPT 会话级覆写、psi0 复向量…
   L3 手动管线（远期）：装配→导出→外部改→载入续算
```

## 3. 检视器标签页设计（对应求解链 ①–⑨）

| # | 标签页 | 内容 | 数据来源（新增 diag 键） |
|---|--------|------|--------------------------|
| A | 能级与场 | bare 能级表；场表（ω/I/偏振/包络）；Rabi 矩阵 A 热图+数值表 | `diag['E_bare']`、`diag['fields']` |
| B | AC 化 | 逐对分类表（i, k, Ω_ik, ω, r=Ω/ω, 判定 keep/AC, far_sos 位移）；E→E2 前后对比条形图；两体显示"位移不注入"标记 | `diag['acize']`（pairs 表 + shifts + r_crit） |
| C | 光子块 | M 判定过程（逐载频 r 值 → 公式 (r/4)^{2(M−1)}≤tol → M 结果）；P/dim/扇区结构（扇区 p → (nvecs, E−n·ω 能段)）；K0 对角热图 | `diag['photon']`（Ms/P/dim/strides/nvecs/D_full） |
| D | 耦合对打包 | pairs 全表（a, b, \|v\|, ΔD, is_mw, ω, 分组键）；**ΔD 直方图**（MHz→THz 对数轴——一眼看出有无 THz 残留对，T4/t70 类 bug 的直接诊断）；按载频分组计数 | `diag['pairs']`（数组化摘要，不复制波形对象） |
| E | 哈密顿量 | **解析区**：φ-帧方程逐项公式 + 颜色图例（光场共振/微波 cos/V_dd/AC 对角/耗散）；**数值区**：给定 t 采样的 H(t) 快照——\|H\| 热图 + Re/Im 切换 + 逐项分矩阵（H_opt/H_mw/H_vdd/H_ac/gamma）开关叠加；dim≤1452 显示热图，更大只给分块统计 + 导出 | `assemble_H_mixed(diag, t)` 重建（§4 t75b） |
| F | 求解过程 | **运行中**：现有进度条增强——dt(t) 步长曲线实时滚动、n_acc/n_rej 计数（driver 每步写共享数组）；**运行后**：完整 dt(t) 曲线（stiff 段自动对齐波形边沿标注）、nfev/接受/拒绝、norm 偏差 | `diag['solver']`（dt 环形缓冲 + stat） |
| G | 扇区与重构 | 扩展空间逐扇区幅值 \|X_{i,n}(t)\| 曲线（从保留的 y_out 采样）；扇区和 vs 投影布居一致性校验曲线；φ-帧 vs 物理帧范数 | `diag['trajectory']`（下采样幅值） |

导出：每页「导出」按钮 → `.npz`（数值全量）+ `.csv`（表格）+ `.json`（元数据），
落 `tempsimu/diag_export/`（或存档目录）。

## 4. 实施拆解（t75a–t75e，顺序依赖）

### t75a 引擎诊断采集（core，约 1 天）
- `evolve_mixed` / `evolve_mixed_pair` 新增 kwargs：`diag=False, diag_matrices=False`。
- `diag=True`：沿现有分支把 ①–⑦ 中间产物收集进 `MixedDiag`（dataclass 或 dict），
  附到 `out['diag']`；**默认 False，GUI 简单模式零开销**。
- `diag_matrices=True` 才附带 K0/逐波 Kcw 全矩阵（大内存）。
- pairs 打包处同步记 ΔD 直方图分桶（对数桶），避免 UI 侧二次扫描。
- 顺手清理：`evolve_mixed` 中 `return out` 之后的死代码块（:311–329，历史残留）。
- **验收**：用 `save_state/交换全过程.json`（两体）与 `拉曼.json`（单原子）跑通，
  diag 内容与状态栏 log 数值逐一吻合；`diag=False` 时结果与现状逐位一致。

### t75b H(t) 重建纯函数（core，约 0.5 天）
- `assemble_H_mixed(diag, t, terms=None) -> dict(term -> dim×dim 复矩阵)`：
  H(t) = K_static + Σ_光场 env(t)·Kc∘e^{iC·ΔD·t} + Σ_微波 env(t)·cos(Cωt)·A∘e^{iC·ΔD·t}
  + AC 对角 + −iΓ/2，terms 可选分项返回。
- 单/两体同构（两体 dim=N²·P）；与 `rhs_phi_pairs` 的数学逐项对齐，但不进 njit。
- **验收**：随机 t 上 ∥H(t)−H(t)†∥≈0（无耗散时）；对 3 态 toy 场景与手工组装矩阵
  逐元一致；每项分矩阵与 rhs 内核逐项语义对照（is_mw 门控/env² 等）。

### t75c dop853 dt 历史缓冲（core 唯一内核改动，约 0.5 天）
- `_dop853_driver` 新增可选 `dt_hist`（预分配 float64 数组 + 计数索引），每接受步写
  (t, dt)；nogil 兼容（定长数组写入，无分配）。None 时完全跳过——**向后兼容**。
- **验收**：现有 checks/selftest_all 全绿；hard 路径不受影响；dt_hist 记录数 =
  n_acc。

### t75d GUI 求解检视器（ui，约 2–3 天，主体工作量）
- 动态页顶部新增**「上帝模式」开关**（QSS 高亮，开启才生效）。
- 新模块 `src/ui/solver_inspector.py`：QDialog/QDock 检视面板，七标签页（§3 表），
  全部 matplotlib FigureCanvas 复用现有画布惯例 + `ui_theme.QSS`。
- 接线：`_run_single_body/_run_two_body` 传 `diag=上帝模式`；
  `_on_evolve_done/_on_pair_evolve_done` 在上帝模式下打开检视器；
  `_poll_progress` 增强：F 页 dt 曲线/n_acc/n_rej 实时刷新（读共享数组，模式不变）。
- EvolveWorker/EvolveWorkerPair 透传 kwargs（evolve_engine **kwargs 已支持，仅加参数）。
- **验收**：ui_selftest 通过；简单模式 UI 行为逐像素不变；上帝模式两体 22 态
  620 ns 场景七个标签页全部有数据、可导出。

### t75e 回归与文档（约 0.5 天）
- `checks/` 新增 `mixed_diag_selfcheck.py`：diag 完整性断言（键齐全、ΔD 直方图
  无 >BW_OPT 残留桶、扇区和校验 ≤1e-10、dt 记录数=n_acc）。
- docs 新增 `t75mixed求解可视化.md`（交付记录 + 截图）。

## 5. 上帝模式路线图（后续期，本期只铺轨）

| 层级 | 内容 | 依赖 |
|---|---|---|
| **L1 查看**（本期） | §3 七标签页 + 导出 | diag 快照 + H 重建 + dt 历史 |
| **L2 覆写**（下期） | 引擎参数会话级覆写：逐载频 M 手动指定、R_CRIT/BW_OPT/MW_NRES_BW_MHZ/max_step/tol；psi0 复向量粘贴；AC 化逐对手动 keep/AC；哈密顿注入项（人为指定而非自然模拟） | params.py 常量改"默认+会话覆盖"函数；evolve kwargs 透传（已有） |
| **L3 手动管线**（远期） | 分步执行：只跑装配（出 diag/H）→ 暂停 → 用户改参数/矩阵 → 载入状态继续积分；多引擎并跑对比 | diag 可序列化 + 积分状态导出/载入 |

## 6. 风险与约束

1. **njit 内核改动仅 t75c 一处**，且为可选参数——把数值路径风险压到最低。
2. **内存**：两体 y_out (621, 1452) complex ≈ 14 MB，采样后存 diag；Kcw 全矩阵仅
   diag_matrices=True；全 Sambe dim 7260 场景 E 页只给分块统计+导出。
3. **线程**：F 页实时刷新沿用 t72 模式（主线程轮询共享数组，内核 nogil），
   不新增跨线程信号。
4. **语义红线**：H 重建必须与 rhs 内核逐项同语义（t62f/t70 的教训都在打包分组与
   三角过滤上）——t75b 验收里做逐项对照，防止"检视器显示的 H 不是引擎用的 H"。
5. **简单模式零回归**：diag 默认关、检视器不实例化、进度协议向后兼容
   （coupling.stage_progress 已有 len(arr)>2 分支惯例）。

## 7. 待敲定问题（请船长拍板）

1. 检视器形态：**模态对话框**（运行完弹出）还是**停靠面板**（主窗口右侧常驻）？
   倾向停靠面板——上帝模式下可与曲线对照着看。
2. E 页 H(t) 快照的采样点：固定几个预设 t（t=0 / 脉冲峰值 / 任意拖动滑块）？
   倾向滑块 + 播放按钮（逐帧扫 t，观察耦合对相位旋转）。
3. L2 的"人为指定"边界：允许直接编辑 pairs 表（删/改某个耦合对）吗？还是只允许
   参数级覆写？前者威力大但等于让用户手改物理。
4. dt 历史要不要顺带在 hard 引擎也开（driver 是共用的，顺手）？倾向开，零成本。
5. 两体 G 页扇区幅值：按 3 扇区折叠展示（每扇区 N² 条）还是只画与初态拓扑最近的
   若干态？22 态场景 N²·3=1452 条曲线必须配 topn 过滤。
