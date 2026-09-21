# `runspec.py` 审查（2026-09-18）

> 使用者问：「看着好多硬编码，不过我不知道是不是这就是最终形态简化不了了。」
>
> 结论先说：**硬编码大部分是"登记册的接口"（合理），但顺着查出三个真·静默洞（不合理）；
> 另外有 3 处死代码、~26 行可消的重复。可简化，但优先修静默洞。**

---

## 一、硬编码清单：22 处登记册键

`build_run` 里 `p.get('...')` 直接写键名，共 22 处：

| 组 | 键 | 处数 |
|---|---|---|
| 能级组 | `groups.list` | 1 |
| 场（静态） | `field.B_gauss`、`field.E_vpcm` | 2 |
| 动态基本 | `dynamic.mode`、`dynamic.method`、`dynamic.t_max`、`dynamic.npts` | 6（t_max/npts 各 2） |
| 初态 | `dynamic.psi_single`、`psi_A`、`psi_B`、`psi_pair` | 4 |
| 衰变 | `dynamic.decay_index`、`dynamic.decays` | 4（各 2） |
| 两体几何 | `dynamic.pair.R_um`、`theta_deg`、`phi_deg` | 3 |
| 波形 | `dynamic.waves_%s`（模板） | 1 |
| 运行级 | `solve.FROZEN_H_NS`、`solve.PER_FREQ_M` | 2 |

**它们是不是"该消掉的硬编码"？——不是。** 这是**登记册的字段路径**，
定义点在 GUI 侧（`pm_setup` 的登记表）。`runspec` 是**消费者**，
写键名等价于"读一个外部数据结构的字段"——**这是接口，不是魔法数**。

**但它们是"隐式接口"**：`p.get('dynamic.t_max') or 0.0` 这种写法
**把"键不存在"和"值就是默认"混在一起** ⇒ **键名打错就静默用默认**。
真正的风险在这里，见第三节。

**唯一该骂的一处**：`SOLVE_KWARG`（见第二节 D）。

---

## 二、真·死代码（3 处，都是我第 11 刀留下的）

### A. `_WAVE_ATOMS = ('single', 'A', 'B')`

```python
_WAVE_ATOMS = ('single', 'A', 'B')     # 全文件只出现这 1 次 = 定义处
```
**定义后从未使用** ⇒ 死常量。

### B. `build_run` 里 4 行 `param_overrides`（两个分支各 2 行）

```python
pk = ek_all.pop('param_overrides', None)      # ← 没人再生产它
...
if pk:
    ek['param_overrides'] = pk                # ← 永远不执行
```
**第 11 刀把 `god_kwargs_from_params` 的生产删了**（它现在只收 `solve.*`），
所以这 4 行是**残留**。（产品侧 `ui_dynamic.py` 确实在生产 `param_overrides`——
那是"B1 未接线"那条已文档化的债，不是这里的问题。）

### C. `evolve_run` 的 sambe 单体分支：`_sk` 永远为空

```python
if meth == 'sambe':
    _sk = {}
    if 'DEFAULT_RTOL' in ovr:        # ← ovr 恒为空（上面刚 raise 过）
        _sk['tol'] = ovr['DEFAULT_RTOL']
```
`ovr` 在函数开头就被判空并 `raise`（非空即报错）⇒ **这个 `if` 永远不成立**。
它是"上帝模式覆写 `DEFAULT_RTOL` → sambe 拿 tol"那条旧路的残骸。

### D. `SOLVE_KWARG` —— 仍是两处手抄（第 11 刀漏的同类）

```python
# solve.* → 下游 kwarg。**必须与 pm_setup._SOLVE_SPECS 的最后一列一致**
# （……改一处要改两处——下面 checks/sweep_spec_selfcheck.py 有断言把两边钉在一起）
```
**注释自己承认是手抄**，而钉它的判据 `sweep_spec_selfcheck.py`
**不在沙盒里**（依赖 `src.sweep`/`pm_setup`）⇒ **这个不变式现在无人在守**。

第 7 刀我把"引擎参数白名单"改成了签名派生，**这一处是同一类病、没治**。

---

## 三、三个真·静默洞（**实测**，最该修）

探针：给 `build_run` 喂缺键/错键的快照，看它静默兜底还是报错。

| 场景 | 实测结果 | 判定 |
|---|---|---|
| 正常（`t_max=100, npts=101`） | 101 点，跨度 100 ns | ✓ |
| **`dynamic.t_max` 键名打错/缺失** | **101 点，跨度 0 ns** | ❌ **静默** |
| **`dynamic.npts` 缺失** | **2 点**（跨度对，分辨率塌） | ❌ **静默** |
| `dynamic.method` 缺失/打错 | 静默用默认 `mixed` | ⚠️ 尚可（有默认是合理的） |
| `dynamic.mode` 缺失 | 静默默认单体 | ⚠️ 尚可 |
| **`dynamic.decay_index=7`（非法档位）** | **静默无衰变** | ❌ **静默** |
| `method='ip4'`（未知引擎） | **`ValueError`** | ✓ 好的 |

**为什么危险**：

1. **`t_max` 缺失 → 0 ns 演化**：用户会拿到"跑完了、什么都没变"的结果，
   而且 `check_run` **抓不到**（`max(npts, 2)` 保证 `len(t) ≥ 2`）。
   **这是最坏的一种——看起来成功了。**
2. **`npts` 缺失 → 2 点**：跨度对，但曲线只有两个点。同样 `check_run` 不报。
3. **`decay_index=7` → 静默无衰变**：`resolve_decays` 的
   `if mode != 2: return []` 把"非法档位"和"档位 1（禁用）"混成一条路。

**注意这与本仓一贯的纪律冲突**：`resolve_psi` 明确写着
「**绝不静默回落 0 号态**（P1）」，`check_run` 写着「校验代替静默」——
**而这三个洞正是同一类，只是没人扫过。**

**修法（很直接）**：给**必填键**加"缺失即报错"，与 `resolve_psi` 同款：

```python
def _required(p, key):
    if key not in p or p[key] is None:
        raise ValueError('快照缺少必填键 %r（登记册路径）' % key)
    return p[key]
```
- `dynamic.t_max` / `dynamic.npts`：**必填**（0 ns / 2 点不是有意义的默认）
- `dynamic.decay_index`：**非法值报错**（只认 0/1/2）
- `dynamic.method` / `dynamic.mode`：可留默认，但**若键存在而值非法就报错**

---

## 四、重复：`build_run` 的两个分支（~26 行可消）

| | 行范围 | 行数 |
|---|---|---|
| 两体分支 | L433–491 | 59 |
| 单体分支 | L493–525 | 33 |

**逐段比对，完全重复的有**：

| 段 | 两体 | 单体 | 重复行数 |
|---|---|---|---|
| `t_max`/`npts`/`t` 网格 | L439–442 | L497–500 | 8 |
| `ek_all`→`_filter_kw`→`dropped`→`pk` | L463–470 | L506–513 | 16（**其中 4 行是死的**） |
| `ek.update(ek_extra)` | L478 | L514 | 2 |
| 返回 dict 的公共键（`version`/`two_body`/`method`/`method_in`/`t`/`t_max`/`npts`/`states`/`labels`/`n_waves`/`engine_kwargs`/`warnings`） | L480–490 | L516–524 | 11 个键 |

**⇒ 约 26 行可直接消掉**（不含 dict 字面量合并带来的额外收益）。

**改法**：公共段提一次，体相关段各自 `update`：

```python
def build_run(cache, params, *, ...):
    ...
    states, meth, notes = ...                 # 公共
    t_max, npts, t = _time_grid(p, t_grid)    # 公共
    ek, dropped = _engine_kwargs(p, meth, is_pair, diag)   # 公共
    common = dict(version=..., two_body=is_pair, method=meth, method_in=meth_in,
                  t=t, t_max=t_max, npts=npts, states=states,
                  labels=[s['label'] for s in states],
                  engine_kwargs=ek, warnings=warnings + notes)
    common.update(_pair_body(...) if is_pair else _single_body(...))
    return common
```

---

## 五、能简化到什么程度 —— 我的判断

| 项 | 能不能简化 | 为什么 |
|---|---|---|
| **22 处登记册键** | **不建议消**（但建议**加必填校验**） | 它们是接口。集中成一张表反而多一层间接，且和 GUI 的定义点仍要两处对齐 |
| **`SOLVE_KWARG`** | **能** | 改成从 `pm_setup` 侧导出、或把它变成 `solve.*` 的**唯一**声明点 |
| **`build_run` 两分支重复** | **能，~26 行** | 公共段提一次 |
| **3 处死代码** | **能，直接删** | — |
| **3 个静默洞** | **能，且应该先修** | 与 `resolve_psi`/`check_run` 的既有纪律一致 |
| **`parse_per_freq_m`（27 行）** | **可议** | 它对 6 种输入形态做防御（None/''/list/tuple/bool/int/float/str-literal）。**但它是"用户输入解析器"，防御是本职**——除非确定输入只有一种形态 |
| **`evolve_run` 的 6 个 `if meth == ...`** | **不建议再压** | 它们已经是"给引擎备料"（不是分派，分派在 `_ENGINE_TABLE`）。压成表会引入"备料钩子"，**那是套层** |

**⇒ 一句话**：**这份代码的"硬编码"大部分是接口，不是债；真正该动的是三个静默洞 + 三处死代码 + 一处重复。它不是"简化不了"，但也不该为了短而短。**

---

## 六、处置（**1–2 已做**，2026-09-18）

| # | 项 | 状态 |
|---|---|---|
| **1** | **修 3 个静默洞**（`t_max`/`npts` 必填 + 非法值报错 + `decay_index` 只认 0/1/2） | ✅ **已做**：加 `_required(p, key)` 助手，与 `resolve_psi` 的"绝不静默回落"同款；实测三个洞全部转成 `ValueError` |
| **2** | **删 3 处死代码**（`_WAVE_ATOMS`、4 行 `param_overrides`、sambe 的 `_sk` 死分支） | ✅ **已做** |
| **3** | 消 `build_run` 的重复 | ✅ **已做**——但**行数没降反升**（95+32=127 vs 原 103）：消的是**重复**（网格守卫/筛参现在只在一处），不是行数。**我先前说"~26 行可消"是错的** |
| **4** | `SOLVE_KWARG` 单源化 | ✅ **沙盒侧已做**：声明为**唯一声明点**；产品侧派生是迁移项（A9）。判据 C 包加了"每个 kwarg 都有引擎收" |

**新增判据** `checks/runspec_contract_selfcheck.py`（5 包 / 17 子条件）：

| 包 | 守什么 |
|---|---|
| A | 必填键缺失/键名打错/`t_max=0`/`npts=1` → **一律报错**，且消息点名键名 |
| B | 非法 `decay_index` → 报错；合法 0/1/2 仍可跑 |
| C | 未知 `method` → 报错，消息列出合法组合 |
| D | 合法快照的回归（101 点 / 100 ns / method / 无"已忽略"警告） |
| E | 已删的死代码不回来（`_WAVE_ATOMS` / `param_overrides` / `_required` 在） |

基线：`##RE0_SUMMARY scripts=17 fail=0 seconds=331`。

**第 1 条的价值**：它防的是"**看起来跑成功了、其实什么都没算**"——
这是最坏的一类错（用户不会去查）。而 `check_run` 原本抓不到它
（`max(npts,2)` 保证 `len(t) ≥ 2`，`t_max=0` 也通过）。

> ⚠ 上面是**当时**的基线（本文写作时）。**当前基线：18 判据 / 0 FAIL / 362 秒**（见 `README.md`）。
