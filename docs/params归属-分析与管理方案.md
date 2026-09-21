# `params.py` 的归属分析与管理方案（2026-09-18）

> ⚠ **本文是"决策前"的分析**（三个方案与代价）。
> **使用者已裁决：`params.py` 整层消灭、参数就地定义**——见第 11 刀，**该文件已不存在**。
> 现行状态与调用方式见 `docs/引擎重构总览与调用说明书.md`（§4.7 参数在哪）。
> 本文保留的是**当时怎么权衡的**，不是现状。

> 使用者判断：**「它不能叫全局注册表——它只是我曾经把所有引擎里硬编码的参数集中到一块儿
> 管理、然后遗留出来的东西。本质上它们属于不同的引擎、不同的地方。是不是可以放归原处？
> 会不会有更好的管理方式？它们不一定是很特殊的数据类型，它们只是参数。」**
>
> **本文用读者统计回答"能不能放归"，并给出三个方案与代价。**

---

## 1. 现状：32 个参数，谁在读

统计口径：`params.SPECS` 里每个名字在 `src/core/*.py` 中的出现文件数（排除 `params.py` 自身）。

| 读者数 | 参数 | 个数 |
|---|---|---|
| **1 个模块** | `M_EST_TOL`、`_M_START`、`DIRECT_SOS_N_WINDOW`、`AC_EFFECT_THRESHOLD_MHZ`、`OPT_NRES_BW_MHZ`、`AC_NEAR_CUTOFF_MHZ`、`FRAME_TOL_REL`、`_TF_DENSE_MAX`、`_AC_ENV_TAB_N`、`_ENV_PROBE_N`、`_ENV_SAMPLES`、`_DT_MIN_FRAC`、`_SPARSE_SEG_MAX`、`_SPARSE_MAX_DEPTH`、`_NODE_LADDER`、`_VOLT_MAX_ITER`、`_VOLT_PICARD_FRAC`、`_VOLT_PRE_SWEEPS` | **18** |
| 2–3 个 | `R_CRIT`、`DEFAULT_RTOL`、`DEFAULT_ATOL`、`BW_OPT`、`M_EST_BW_MHZ`、`_RHS_ENV_TAB_N`、`PHASE_TOL`、`_M_MAX`、`_DENSE_MAX` | 9 |
| 4–7 个 | `RWA_TOL`(4)、`RYDBERG_N_MIN`(5)、`_TAB_N`(5)、`MW_MAX_MHZ`(7) | 4 |
| 0 个（纯派生） | `ATOL_RTOL_RATIO` | 1 |

**按"独占它的模块"再分一次**：

| 家 | 独占参数 | 数 |
|---|---|---|
| **`sambe`** | `_M_START`、`_ENV_PROBE_N`、`_DT_MIN_FRAC`、`_SPARSE_SEG_MAX`、`_SPARSE_MAX_DEPTH`、`_NODE_LADDER`、`_VOLT_MAX_ITER`、`_VOLT_PICARD_FRAC`、`_VOLT_PRE_SWEEPS` | **9** |
| `evolve_twofield` | `OPT_NRES_BW_MHZ`、`FRAME_TOL_REL`、`_TF_DENSE_MAX` | 3 |
| `static_atom` | `DIRECT_SOS_N_WINDOW`、`AC_EFFECT_THRESHOLD_MHZ` | 2 |
| `field_build` | `AC_NEAR_CUTOFF_MHZ`、`_ENV_SAMPLES` | 2 |
| `evolve_mixed` | `M_EST_TOL`、`_AC_ENV_TAB_N` | 2 |

**⇒ `sambe` 一个模块独占 9 个**（全是稀疏路径的数值保护：Volterra/GMRES 迭代上限、
节点阶梯、二分深度、细分段数上限……）。**这些参数放在 `params.py` 里，
`sambe.py` 的读者根本看不出来它们属于自己。**

**⇒ 使用者的判断被数据完全印证：它们属于不同的引擎、不同的地方。**

---

## 2. 但有一件东西搬不走：**给 UI / AI 的那张元数据表**

`SPECS` 现在同时承担**两件不同的事**：

| 职责 | 谁需要 | 能不能跟着参数搬到使用处 |
|---|---|---|
| **值 + 读取（含覆写）** | 引擎自己 | ✅ **能**——而且本来就该在那儿 |
| **元数据**（label / unit / desc / lo / hi / log / tier / settable） | 上帝模式「引擎参数」页、AI 的 `list_params('engine.')`、`pm_setup` | ❌ **搬不走**——UI 需要**一张全表**，而引擎不该知道 UI |

**⇒ 这是方案分野的唯一原因。** 三个方案的区别就是——**那张表由谁收集、放在哪。**

**另一条硬约束**：`params.py` 现在**零依赖**（不 import 任何 `src.core` 子模块）。
这不是洁癖：`evolve_util`/`static_atom`/各引擎都 import 它，
**它一旦反向 import 引擎就成环**。所以"`params.py` 去收集各引擎"这条路**是堵死的**。

---

## 3. 三个方案

### 方案 A：只搬值，元数据表留在 `params.py`

- 参数常量搬到使用它的模块；`params.py` 保留 `SPECS` **只为供数**
- **代价**：**同一件事两处声明**（值在一处、元数据在另一处），改了值忘了元数据就会漂
- **⇒ 不推荐**：这正是"同一事实两处记录"的老病

### 方案 B：参数就地定义（值 + 极简元数据），**UI 侧收集**

```python
# evolve_mixed.py 顶部
BW_OPT_MHZ = _P.Param(2e4, '光场对过滤带宽（mixed）', unit='MHz',
                      lo=1e3, hi=1e6, log=True, tier='D')
...
bw = BW_OPT_MHZ.value          # 覆写栈优先，否则本模块默认
```

- `params.py` 提供**两样东西**：`Param` 这个极轻的声明对象 + 覆写栈（`override`/`get`）
  **它依然零依赖**（不认识任何引擎）
- **收集器在消费方**（`src/ui` / `src/ai`：它们本来就 import 全部引擎）：
  扫各引擎模块的 `Param` 实例 → 拼出那张表
- **不存在注册表**：表是**扫出来的**，不是登记进去的
- **代价**：
  1. 要写一个收集器（~20 行）
  2. `override()` 的"未知名字早失败"要换机制（`params.py` 不再知道有谁）——
     可由收集器或 `Param` 自己校验；也可保留一个**可选**的注册口（谁 import 谁注册）
  3. `effective()` 快照（扫描存档用）改为记录**本次覆写集**
     ——`RunSpec` 里**已经有 `param_overrides`** 了 ✓

### 方案 C：不动，但把「归属」写进 `params.py`

- 每个 `ParamSpec` 加一个 `home='sambe'` 字段；`SPECS` 按模块分组排列
- **代价**：零风险，但只解决"读代码时不知道属于谁"，**不解决"改引擎要动一个不相干的文件"**
- **⇒ 这是"先止血"的最小版本**

---

## 4. 我的建议

**方案 B，但分两步走**：

**第一步（试点）**：只在 **`sambe`** 上做——它独占 9 个参数，最能验证方案。
- 9 个常量搬进 `sambe.py` 顶部（带一行说明）
- `params.py` 加 `Param` 类 + `get(name, default)`，保留 `SPECS` 里的这 9 条**暂时不动**（供 UI）
- **验收**：`sambe` 内部读到的值逐位不变；UI 那张表逐字段不变
- **这一步不破坏任何东西**，且能暴露真实工作量

**第二步（铺开）**：试点通过后，把其余 23 个按同一手法搬；
UI 侧写收集器；`params.py` 退化为 `Param` + 覆写栈。

**为什么不建议一步到位**：这张表同时连着 **GUI 参数页** 和 **AI 的 `list_params`**，
一步到位等于同时动引擎、GUI、AI 三面。先在一个模块上证明手法，再铺。

---

## 5. 需要你拍板的

1. **做不做**（方案 B 第一步 / 方案 C / 先不做）
2. **`Param` 的元数据要留多少**——你说"它们只是参数"，那 `lo/hi/log/tier` 这四个
   **纯 UI 用途**的字段，是留在参数定义处（方案 B），还是**完全移到 UI 侧**（参数处只有名字+默认值+一句说明）？
   后者更符合"它们只是参数"，但 UI 的输入框范围/步长就要另找地方
3. **`_` 前缀的那 20 个**（`_TAB_N`/`_SPARSE_*`/`_VOLT_*`…）**要不要在 UI 上露面**？
   现在它们都在 `SPECS` 里、都在上帝模式页上。如果它们本质是"引擎内部的数值保护"，
   也许**根本不该出现在 UI 上**——那就更该跟着引擎走。
