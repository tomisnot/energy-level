# -*- coding: utf-8 -*-
"""就地参数声明：`Param` 描述符（**零依赖**，core 内可用）。

## 为什么有这个（解 C1：参数元数据的单一定义点）

第 11 刀消灭了全局注册表 `params.py`，把 32 个参数**就地定义**在各引擎模块，
但"值"有了家、"元数据"（unit/range/desc/tier）却**无家**——GUI 参数页、AI 的
`list_params`、离线扫描三个消费者要同一份元数据表，而那张表随 `params.py` 一起
被删了（见 `docs/语义层-我们有什么缺什么.md` §三、`docs/params归属` §2）。

`Param(...)` 把**值 + 元数据声明在同一处**（真·单一定义点）；`protocol.engine_params()`
的**收集器**扫各模块的 `Param` 实例拼成全表——表是**扫出来的，不是登记进去的**，
故不存在第二真源（守纲领 §13「不套层」判据）。

## 透明子类：为什么读点不用改（因地制宜的降险）

`Param(default, ...)` 按 `default` 的类型返回一个**透明**实例：它**同时是**
`int`/`float`/`tuple`（默认值本身），**又**携带元数据。于是：

    _M_START = Param(2, 'M 起点', tier='internal')   # 声明处：值+元数据同处
    ...
    M = _M_START - 1          # 读点**无需改**：_M_START 就是个 int
    for n in _NODE_LADDER:    # 读点**无需改**：_NODE_LADDER 就是个 tuple

这消除了"把 32 个常量包起来后要改 60–80 处读点"的风险面（实测 sambe 一个模块
就 25 处读点）。**已 AST 核实 numba 安全**：core 里 11 个 `@njit` 函数**无一**
把这些常量当全局读（它们只吃实参）⇒ 常量只在普通 Python 里用，int/float/tuple
子类在普通 Python 完全透明（`==`、算术、`range()`、`np.linspace()`、`int()` 皆可）。

**唯一例外**：默认值是 `None`（或非数值）时无法透明（`None` 是单例、不可子类化），
返回不透明的 `_ParamObj`，其读点须显式 `.value`（全仓仅 `PHASE_BUDGET` 一例）。

## 三条硬约束（缺一不可）

1. **零依赖**：本模块不 import 任何东西。各引擎模块都 import 它，它一旦反向
   import 任何引擎就成环——这正是旧 `params.py` 的约束，原样继承。
2. **不是注册表**：`Param` 不认识任何其他模块、不维护全表、无 `SPECS`。
   它只是"带元数据的值"。全表由消费方（`protocol.engine_params()`）扫描收集。
3. **core 内无覆写栈**：透明实例的值恒等于声明的 `default`。运行级覆写发生在
   **引擎之外**（mecha v2 的 Gate 快照），经既有 `engine_kwargs` 通道作为 kwarg 流入
   引擎函数（引擎签名里 `default=X` 只是兜底）⇒ core 保持纯函数 + 记忆化，
   状态管理不渗进引擎（见 `docs/引擎层该有状态管理吗.md`）。

## tier 分层（D-B）

`_` 前缀的内部数值保护参数标 `tier='internal'`，收集器收进全表但**默认不对
AI/GUI 露面**——它们本质是"引擎内部数值保护"，暴露只增噪声与误调风险；保留供深挖。
"""

# tier 取值约定（收集器/消费方据此过滤；自由字符串，下列为惯例）：
#   'run'      运行级旋钮（可经 solve.*/engine.* 覆写，已是引擎 kwarg）
#   'internal' 实现常量（`_` 前缀；就地定义、不做运行级覆写；默认隐藏）
#   'physics'  物理事实（MW_MAX_MHZ / RYDBERG_N_MIN 之类，不可调）
#   'check'    判据参数（PHASE_TOL / RWA_TOL 之类，定义在判据所属模块）
TIERS = ('run', 'internal', 'physics', 'check')

# 元数据槽（所有子类共用）
_META_SLOTS = ('desc', 'unit', 'lo', 'hi', 'log', 'tier')


class _ParamBase:
    """所有 Param 的公共基类：收集器用 `isinstance(x, _ParamBase)` 识别。

    本身不存值（值由 int/float/tuple 子类承载，或 _ParamObj 的 `_default`）；
    只提供 `meta()` / `value` 协议 / `__repr__`。子类须把 `_META_SLOTS` 列进
    自己的 `__slots__` 并在 `__new__`/`__init__` 里填好。
    """
    __slots__ = ()

    def meta(self):
        """→ dict：纯元数据（可 JSON 序列化，供 registry 拼表；不含大对象）。"""
        v = self.value
        return {
            'default': v, 'desc': self.desc, 'unit': self.unit,
            'lo': self.lo, 'hi': self.hi, 'log': self.log, 'tier': self.tier,
            'type': type(v).__name__,
        }

    def __repr__(self):
        return 'Param(%r, tier=%r, unit=%r)' % (
            self.value, self.tier, self.unit)


class _ParamInt(_ParamBase, int):
    # int/float/tuple 子类**不支持非空 __slots__**（TypeError）→ 用 __dict__；
    # 全仓仅 32 个 Param，这点开销可忽略。

    def __new__(cls, value, desc='', unit=None, lo=None, hi=None,
                log=False, tier='run'):
        self = int.__new__(cls, value)
        self.desc, self.unit, self.lo, self.hi = desc, unit, lo, hi
        self.log, self.tier = bool(log), tier
        return self

    @property
    def value(self):
        return int(self)


class _ParamFloat(_ParamBase, float):

    def __new__(cls, value, desc='', unit=None, lo=None, hi=None,
                log=False, tier='run'):
        self = float.__new__(cls, value)
        self.desc, self.unit, self.lo, self.hi = desc, unit, lo, hi
        self.log, self.tier = bool(log), tier
        return self

    @property
    def value(self):
        return float(self)


class _ParamTuple(_ParamBase, tuple):

    def __new__(cls, value, desc='', unit=None, lo=None, hi=None,
                log=False, tier='run'):
        self = tuple.__new__(cls, value)
        self.desc, self.unit, self.lo, self.hi = desc, unit, lo, hi
        self.log, self.tier = bool(log), tier
        return self

    @property
    def value(self):
        return tuple(self)


class _ParamObj(_ParamBase):
    """默认值为 `None` 或其它非数值——**不透明**，读点须用 `.value`。

    全仓仅 `coupling.PHASE_BUDGET = None` 一例（None 是单例，不可子类化）。
    """
    __slots__ = ('_default',) + _META_SLOTS

    def __init__(self, value, desc='', unit=None, lo=None, hi=None,
                 log=False, tier='run'):
        self._default = value
        self.desc, self.unit, self.lo, self.hi = desc, unit, lo, hi
        self.log, self.tier = bool(log), tier

    @property
    def value(self):
        return self._default


def Param(default, desc='', unit=None, lo=None, hi=None, log=False, tier='run'):
    """就地参数声明工厂：按 `default` 类型返回**透明** Param（读点免改）。

    位置参数 `default` 是**唯一的值来源**；其余是纯元数据：

      desc : 一句人话说明（错误消息/列表用）
      unit : 单位字符串（None = 无量纲）
      lo/hi: 值域（None = 不限）；用于 session.set 的范围校验（契约锁死）
      log  : 是否对数刻度（GUI 滑块/输入框用）
      tier : 分层（见上；'internal' 默认不露面）

    int/float/tuple → 对应透明子类（**它本身就是那个值**）；
    None/其它 → 不透明 `_ParamObj`（读点用 `.value`）。
    """
    kw = dict(desc=desc, unit=unit, lo=lo, hi=hi, log=log, tier=tier)
    if isinstance(default, bool):            # bool 是 int 子类，先判
        return _ParamInt(int(default), **kw)
    if isinstance(default, int):
        return _ParamInt(default, **kw)
    if isinstance(default, float):
        return _ParamFloat(default, **kw)
    if isinstance(default, tuple):
        return _ParamTuple(default, **kw)
    return _ParamObj(default, **kw)
