# t72：动态分析进度协议改造与 GIL 释放

> 现象（用户报告）：动态分析进度条与计算进度不吻合；长耗时环节没有每小步
> 的进度反馈。诊断与修复全程实地测量（探针 `tempsimu/probe_progress.py`）。

## 1. 根因（两个，都在测量中实锤）

**根因 1（主因）：njit 积分内核持有 GIL → 整段积分 GUI 假死。**
`_dop853_driver` 及 4 个 RHS 内核原为 `@njit(cache=True)`（无 nogil）。
numba nopython 代码**不释放 GIL**——混合两体 620ns 实测：30.2s 的运行中
采样线程（50ms 一拍）在 9.4s→30.1s 出现 **20.7s 完全空洞**。GUI 的 100ms
轮询 QTimer 与「取消」按钮同样被饿死：进度条在整个积分期间死在
"JIT 编译/准备… 5%"，直到结束才跳变。DOP853 驱动里按步写 `prog[0]` 的
代码一直在正确执行，只是**没人能读**。

**根因 2（次因）：进度映射与阶段语义粗糙。**
- 主线程预处理（基组/DC-Stark/驱动场/`vdd_sparse`）发生在 worker 启动前，
  对话框 400ms 定时器此时尚未触发 → 完全无反馈（首次触碰新基组时可达
  秒-分钟级）。
- 自适应步长下 `prog[0] = t/T` 与墙钟严重不符：实测 mixed 两体（微波高斯
  + 双光方波）积分段前 27% 模拟时间耗掉 ~70% 墙钟、末段瞬间冲满。
- numba 磁盘缓存装载（warm 进程首调驱动 ~8s）发生在驱动调用内部，
  引擎此前已置"积分"阶段 → 显示"积分 0%" 8s，标签失真。

## 2. 修复

### 2.1 内核 nogil（治根因 1）
`@njit(cache=True)` → `@njit(cache=True, nogil=True)`：dop853 的
`_rms_norm/_env_at/_dop853_step/_dop853_dense_interp/_dop853_driver` +
hard 的 `_env_at/_rhs_hard/_rhs_hard_pair` + mixed 的 `_rhs_phi`/
`_rhs_mixed_pair`，共 10 处。兼容性三连测
（`tempsimu/_nogil_compat_test.py`）：① njit 一等函数参数（rhs 实参）
可与 nogil 共用；② 内核运行期间轮询线程最大采样间隔 47ms（GIL 真释放）；
③ nogil 内 `raise RuntimeError` + 跨线程取消标志生效。

### 2.2 三槽阶段协议（治根因 2）
`progress_arr` 语义升级（`coupling.stage_progress` 助手，GUI
`_poll_progress` 消费）：

| 槽 | 含义 |
|---|---|
| `[0]` | **阶段内**进度 0..1（旧脚本 len-2 用法只写此槽，语义不变） |
| `[1]` | 取消标志（不变） |
| `[2]` | 阶段：0 准备（预处理/JIT 装载）、1 积分、2 收尾 |

- 各引擎（mixed×2 / hard×2 / sambe×2 / two_field）在装配、积分、收尾
  边界写阶段；积分映射改为**预测剩余工作量**：
  `p = nfev / (nfev + rate·(T−t))`，rate 为局部求值密度（RHS 次数/模拟
  时间）的 EMA(0.7/0.3)，p 单调钳制、封顶 0.999——nfev ∝ 墙钟（nogil
  下每次 RHS 开销恒定），条速跟随真实墙钟；
- 阶段 1 由**驱动在首个接受步**置位——调用内的 numba 装载/编译期保持
  "准备"标签；
- 收尾（跨扇区投影/loss/约化）逐点上报。

### 2.3 GUI（ui_dynamic）
- 轮询/对话框定时器提前到 `_run` 开头：主线程预处理期间对话框即可显示；
  `_prep(label, frac)` 泵 `QApplication.processEvents()` + 取消检查；
- `_current_states(pump=)` 逐能级组上报；`vdd_sparse(progress=)` 逐行上报
  （`static_pair._dipole_sparse` 每行回调，张量收缩占 0.6→1.0；缓存命中
  不回调）；
- `_poll_progress` 按阶段映射（准备 0–5% 附墙钟耗时 / 积分 5–95% /
  收尾 95–100%）；运行期间 `run_btn` 禁用防重入；`_finish_run_ui` 统一收尾
  （完成/失败/取消/预处理中止四路径）。

## 3. 实测验收（探针，修复前后同场景）

mixed 两体（22 态 620ns，README 示例 B 同构）：

| 指标 | 修复前 | 修复后 |
|---|---|---|
| 采样空洞（GIL 被占=GUI 冻结） | 9.4s→30.1s（20.7s） | 无 |
| 进度到达分布 | 全部 @结束瞬间 | 0.008@0.6s → 0.15@10.5s → 0.28@15.5s → 0.55@17.4s → 1.0@18s |
| >1s 平坦段 | 9.4s（p0=0） | 无 |
| nfev | 163412 | 163412（不变） |

**数值不变性**：hard 单原子（301×6）与 mixed 两体（621×484）布居，
HEAD vs 改动后 `max|Δ| = 0.000e+00`（逐位一致）。全量回归
`checks/selftest_all.py` 6 套件全 PASS（含 ui_selftest 57/57）。

## 4. 教训

1. **nogil 是 PyQt + numba 工作线程组合的必需项**，不是优化项：任何
   "工作线程跑 njit、主线程轮询共享数组"的模式，内核不带 nogil 就等于
   没有进度/取消。（evolve_arbitrary 此前已带 nogil——evolve_rwa 的
   "numba 内核 nogil，主线程可读"注释描述的是意图而非事实。）
2. Qt 槽内未捕获异常 → qFatal/abort（exit 127、无 Python 回溯）：重构时
   **槽读取的属性必须在启动定时器前初始化**（本次 `_pbar` 缺失导致
   ui_selftest 原生崩溃，与 nogil 无关——二分定位排除后才暴露）。
3. 自适应积分器的 t/T 不是进度；"预测剩余工作量"映射在 stiff 区间把
   条速拉回真实墙钟节奏，且端点严格对齐（p→1 当 t→T）。
