# -*- coding: utf-8 -*-
"""参数语义的**通用查询/校验口**（领域无关）——schema 从注入的 engine 收，本模块**零领域**。

三个消费者（GUI 参数页 / AI `list_params` / 离线扫描）要同一份东西：每个参数的名字/类型/
单位/范围/说明/默认值 + 它对应哪个引擎形参。本模块**不声明任何领域 schema**——那是
`core/protocol.py` 的职责（经 `adapter.EngineAdapter` 注入）；这里只做**领域无关**的：
渐进披露（分页/前缀/tier 过滤）、可教学错误（did_you_mean）、通用类型/值域校验。

**依赖方向**：本模块只 import `mecha.runtime` + `mecha.errors`，**不 import core**（提炼守卫）。
"""
from . import runtime
from .errors import InvalidValue, UnknownKey, did_you_mean

__all__ = ['engine_params', 'solve_map', 'solve_schema', 'config_schema',
           'list_params', 'describe', 'all_settable_keys', 'validate', 'families']


# =====================================================================
# schema：全部从注入的 engine 收（core 是唯一定义点，这里零副本）
# =====================================================================
def engine_params():
    """→ {name: record}：引擎就地常量元数据（family='engine', readonly=True）。"""
    return runtime.engine().engine_params()


def solve_map():
    """`solve.*` → 引擎 kwarg 的唯一映射。"""
    return runtime.engine().solve_map()


def solve_schema():
    """→ {'solve.MAX_STEP': record, ...}。"""
    return runtime.engine().solve_schema()


def config_schema():
    """→ {'field.B_gauss': record, ...}：登记册配置键 schema。"""
    return runtime.engine().config_schema()


def families():
    """三家族合并视图 → {key: record}（config / solve / engine）。"""
    out = {}
    out.update(config_schema())
    out.update(solve_schema())
    out.update(engine_params())
    return out


def all_settable_keys():
    """可写键全集 = config + solve（**不含 engine.***，它只读）。"""
    return runtime.engine().settable_keys()


# =====================================================================
# 查询口（分页 / 前缀 / tier 过滤 —— R1 渐进披露，省 context）
# =====================================================================
def list_params(prefix=None, tier=None, include_internal=False, family=None,
                page=None, page_size=60):
    """渐进披露的参数表（轻量优先）。

    - `include_internal=False`（默认）→ 隐藏 tier='internal' **与整个 engine.* 只读族**
      （引擎参数不是 AI 调用重点、别干扰使用者；要看用 include_internal=True 或 family='engine'）。
    - `prefix` / `family` / `tier` 过滤；`page` 分页。截断时**明确告知截了多少 + 怎么拿全量**（I4）。
    """
    items = list(families().values())
    if not include_internal:
        items = [r for r in items
                 if r.get('tier') != 'internal' and r.get('family') != 'engine']
    if family:
        items = [r for r in items if r.get('family') == family]
    if tier:
        items = [r for r in items if r.get('tier') == tier]
    if prefix:
        items = [r for r in items if r['name'].startswith(prefix)]
    items.sort(key=lambda r: (r.get('family', ''), r['name']))
    total = len(items)
    start = 0 if page is None else max(int(page), 0) * int(page_size)
    end = start + int(page_size)
    page_items = items[start:end]
    return {
        'count': total, 'returned': len(page_items),
        'page': 0 if page is None else int(page), 'page_size': int(page_size),
        'truncated': end < total,
        'how_to_get_more': ('翻下一页 page=%d，或用 prefix=/family= 收窄；'
                            'include_internal=true 才含引擎内部参数'
                            % (0 if page is None else int(page) + 1))
        if end < total else None,
        'params': [_slim(r) for r in page_items],
    }


def _slim(rec):
    """列表视图的精简记录（去掉 None 字段，省 token）。"""
    keep = ('name', 'family', 'type', 'unit', 'desc', 'default', 'tier',
            'lo', 'hi', 'enum', 'required', 'home', 'engine_kwarg', 'readonly')
    return {k: rec[k] for k in keep if rec.get(k) is not None}


def describe(key):
    """单个键的完整记录；未知键 → UnknownKey（带 did_you_mean，D2 可教学）。"""
    fam = families()
    if key in fam:
        return dict(fam[key])
    raise UnknownKey(
        '未知参数键 %r（不在契约内；I6：契约外字段永远是 None）' % key,
        hint='调 list_params() 看全量，或用下面的建议',
        suggest=did_you_mean(key, list(fam)))


# =====================================================================
# 校验（P0.4：契约自由度锁死）——session.set 用；通用类型/值域在此，
# wave_list 的逐波真契约是领域知识 → 委托 engine.validate_wave_list。
# =====================================================================
def _type_ok(value, t):
    if t == 'float':
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if t == 'int':
        return isinstance(value, int) and not isinstance(value, bool)
    if t == 'int_or_list':
        return (isinstance(value, int) and not isinstance(value, bool)
                or isinstance(value, (list, tuple)))
    if t == 'str':
        return isinstance(value, str)
    if t == 'bool':
        return isinstance(value, bool)
    if t == 'list':
        return isinstance(value, (list, tuple))
    if t == 'dict':
        return isinstance(value, dict)
    if t == 'identity_key':
        return isinstance(value, (list, tuple)) and len(value) >= 1
    if t == 'pair_sup':
        return (isinstance(value, (list, tuple)) and len(value) == 2
                and value[0] == 'pair_sup')
    return True                       # 未知类型不挡（但 schema 应覆盖）


def validate(key, value):
    """校验 `value` 合 `key` 的契约；非法 → InvalidValue（**报错不静默**，D5）。

    返回**归一化后的值**。None 视为"清除/未设"，直接放行（build_run 对必填键自会报错）。
    """
    fam = families()
    if key not in fam:
        raise UnknownKey(
            '未知参数键 %r——写不进去（I6）' % key,
            hint='调 list_params() 或 all_settable_keys() 看合法键',
            suggest=did_you_mean(key, list(fam)))
    rec = fam[key]
    if rec.get('family') == 'engine':
        raise InvalidValue(
            '%s 是引擎就地常量、运行级只读（Re0 第11刀已移除引擎参数覆写）'
            '——不给 AI/人改，用默认值' % key,
            hint='要调精度/成本用 solve.*（如 solve.MAX_STEP_HARD / solve.COST_BUDGET）',
            suggest=[k for k in sorted(fam) if k.startswith('solve.')])
    if value is None:
        return None
    t = rec.get('type')
    if t == 'wave_list':
        # 逐波真契约是领域知识（认识 POLS / env 元组结构）→ 委托 engine（注入 InvalidValue）
        return runtime.engine().validate_wave_list(value)
    # 归一化：int 字段收整数值浮点（JSON 常把 201 发成 201.0）——**先归一再校验**
    if (t == 'int' and isinstance(value, float)
            and not isinstance(value, bool) and float(value).is_integer()):
        value = int(value)
    if t and not _type_ok(value, t):
        raise InvalidValue(
            '%s 需要类型 %s，得到 %r（%s）' % (key, t, value, type(value).__name__),
            hint='单位/类型见 describe(%r)' % key)
    if 'enum' in rec and value not in rec['enum']:
        raise InvalidValue('%s 只认 %s，得到 %r' % (key, rec['enum'], value),
                           hint='合法取值是枚举里的其中之一',
                           suggest=[str(e) for e in rec['enum']])
    num = isinstance(value, (int, float)) and not isinstance(value, bool)
    if num and 'lo' in rec and rec['lo'] is not None:
        if rec.get('exclusive_lo') and not value > rec['lo']:
            raise InvalidValue('%s 必须 > %r（得到 %r）' % (key, rec['lo'], value),
                               hint='这是 runspec 的硬约束（如 t_max>0：0ns 演化无意义）')
        if not rec.get('exclusive_lo') and not value >= rec['lo']:
            raise InvalidValue('%s 必须 >= %r（得到 %r）' % (key, rec['lo'], value))
    if num and rec.get('hi') is not None and not value <= rec['hi']:
        raise InvalidValue('%s 必须 <= %r（得到 %r）' % (key, rec['hi'], value))
    return value
