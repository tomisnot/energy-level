# src/ 重命名方案（2026-08-31 船长）——纯粹功能驱动

> 目标：摆脱开发历史（命名被"何时写的"而非"干什么的"决定），
> 全部改为"读名字即知功能"。本文档是**方案**，经用户确认后执行。

## 一、文件级（核心争议：历史混合体拆分）

现状痛点：`evolve.py` 一个文件混合 4 类功能（能级/场/演化/Lindblad）；
`hard_numba.py` 的"numba"是历史实现细节不是功能；`core.py` 名字太泛。

### 方案 A：拆分 + 重命名（彻底，推荐）
```
src/core/
├── static_atom.py      ← core.py（单原子静态：能级/Stark/偶极/光谱）
├── static_pair.py      ← pair_core.py（两体静态：V_dd/vdW/交换/雷达）
├── atomic_levels.py    ← evolve.py 拆出（make_levels/gaussian/square/step）
├── field_build.py      ← evolve.py 拆出（field/group/real/arbitrary_field）+ waveforms.build_fields
├── evolve_rwa.py       ← evolve.py 剩余（ip4/Lindblad）
├── evolve_arbitrary.py ← evolve.py 的 arbitrary 演化
├── evolve_sambe.py     ← evolve_exact.py（单原子 Sambe 光子块）
├── evolve_sambe_pair.py← evolve_exact_pair.py（两体 Sambe）
├── evolve_twofield.py  ← evolve_two_field.py（光旋转+微波 两场引擎）
├── hard.py             ← hard_numba.py + hard_pair.py 合并（hard 单/两体）
├── waveforms.py        ← 保留（Wave/Envelope 数据层，功能纯）
├── dop853.py / sambe.py / tensor.py  ← 保留（核心层，命名已功能化）
├── driver.py           ← 保留（驱动系统 + 响应谱，功能清楚）
└── __init__.py         ← 兼容导出层（旧名 → 新名 re-export）
```

### 方案 B：不拆分，仅重命名（保守，成本低）
```
core.py → static_atom.py；pair_core.py → static_pair.py
evolve.py → evolve_rwa.py（内部保留 场/能级 但注释标注）
evolve_exact.py → evolve_sambe.py；evolve_exact_pair.py → evolve_sambe_pair.py
evolve_two_field.py → evolve_twofield.py
hard_numba.py + hard_pair.py → hard.py（合并）
```
B 不拆 evolve.py（其内部 40+ 函数仍混类，收益打折）。

**船长推荐 A**——因为"形式由功能决定"正是本次目标，拆分是根治。

## 二、函数级（私有函数大多已功能化，只改历史含糊名）

### 需要改名的（语义含糊/历史遗留）
| 现状 | 功能 | 建议 |
|---|---|---|
| `evolve_exact` | Sambe 光子块演化 | `evolve_sambe` |
| `evolve_hard` | 硬算演化 | 保留（已功能化） |
| `_ip4_*` 系列 | RWA 相互作用绘景 RK4 | `_rwa_*`（ip4 是历史名） |
| `_pair_ip4` / `_pair_rk4_jit` | 两体 RWA RK4 | `_pair_rwa_*` |
| `V_at` | 时变耦合矩阵 | `coupling_at` / `_V_t` |
| `rec2` | 段传播递归 | `_propagate_seg` |
| `K_at` / `K_seg` | 段哈密顿装配 | `hamiltonian_seg` |
| `_evolve_full` / `_evolve_full_hybrid` | 未知差异 | 调查后定（hybrid=?） |
| `_run_once` / `_run_unitary` / `_run_me` | exact 运行模式 | `_run_sambe` / `_run_unitary` / `_run_mastereq` |
| `wf`（重复定义 3 次） | 波形采样 | 合并为 `_env_sample` |
| `rhs`（多文件重复） | RHS | 加前缀 `_rhs_<engine>` |

### 保留不改的（已功能化）
`tensor_states` / `_defect_sweep` / `_embed` / `_reconstruct` / `photon_geometry` /
`vdd_sparse` / `exchange_coupling` / `direct_ac_stark_shifts` / `decay` / `auto_decays`

## 三、执行策略（关键：低风险）

1. **兼容层**：`src/__init__.py`（core）做 re-export（旧名→新名），**外部调用（UI/GUI/checks/scripts）零改动**——重命名不破坏现有接口
2. **机器替换 + 船长审**：队员用脚本做 import/引用替换，船长逐文件审 diff
3. **分批**：先文件级（大动作）→ 再函数级（每个文件内）
4. **回归**：每批后跑对应 selfcheck（单原子/两体/hard/smoke）
5. **git**：每批提交，可回滚

### 引用面调查（2026-08-31 船长，决定维护范围）

- **必须维护**：`src/core` 内部互引 + `src/ui`（8 文件）+ `src/__init__.py` +
  `checks/` 活跃自检（11 个）+ `app_entry.py`
- **不需要维护**：`tempsimu/`（100+ 历史实验脚本，已 gitignore，非活跃）——
  重命名只改核心/UI/checks/兼容层，**不动 tempsimu**
- **兼容层是关键**：`src/__init__.py` re-export 旧名（`from .core import evolve as evolve`），
  ui/checks 的 `from src.core import evolve` 零改动

## 四、风险与权衡（诚实）

- **收益**：读代码即懂功能（新队员上手快）、语义准确（"exact"→"sambe"纠正误导）
- **成本**：大 diff、需全面回归；`src/__init__.py` 兼容层要维护
- **风险**：合并 hard_numba+hard_pair 需处理内部相互引用；evolve.py 拆分需理清依赖
- 用户之前说过"命名由功能决定"是明确目标——A 方案最彻底但改动最大

**请用户拍板**：A（彻底拆分，推荐）/ B（仅重命名）/ 先做函数级（不动文件）？