# -*- coding: utf-8 -*-
"""扫描规格：**声明式轴展开** + 登记册路径寻址（无 Qt，纯函数）。

## 为什么是"轴"而不是"让 AI 枚举网格"

AI 每写一次扫描脚本，80% 的代码在干两件与物理无关的事：把网格展开成
N 个配置、再把 N 个结果归并回一张表。这两件事完全可以由软件做——AI 只需
声明"哪个参数取哪些值"。

轴键就是**登记册全路径**（`list_params()` 的那套名字），可选带索引路径：

    'dynamic.t_max'                 → 标量参数
    'field.lights[0].intensity'     → 结构化参数的第 0 项字段
    'dynamic.waves_A[0].env'        → 波形包络（整条替换）
    'groups.list'                   → 整表替换

轴值形态：

    [4, 5, 6]                         显式取值
    {'linspace': [7300, 7420, 25]}    含端点的等间隔 25 点
    {'arange': [0, 100, 10]}          步长
    {'logspace': [1, 1e4, 12]}        对数等间隔
    {'value': [1, 2]}                 整个列表当**一个**取值（消歧义）
    5                                 单个取值（=1 个点）

多轴做**笛卡尔积**；顺序确定：轴名排序后依次展开，**最后一个轴变化最快**
（与 numpy 的 C 序一致，结果表好读）。
"""
import copy
import itertools

import numpy as np

__all__ = ['parse_path', 'get_in', 'set_in', 'expand_axis', 'build_points',
           'apply_delta', 'describe_spec']


# ---------------------------------------------------------------------
# 路径寻址
# ---------------------------------------------------------------------
def parse_path(path):
    """'a.b[2].c' → ('a.b', [2, 'c'])。

    登记册的键**本身含点**（`dynamic.pair.R_um`），所以不能按点切：
    规则 = 「第一个 `[` 之前是登记册键，之后是索引/字段路径」。
    """
    if not isinstance(path, str) or not path:
        raise ValueError('轴键必须是非空字符串，收到 %r' % (path,))
    i = path.find('[')
    if i < 0:
        return path, []
    key, rest = path[:i], path[i:]
    segs = []
    while rest:
        if rest[0] != '[':
            raise ValueError('轴键路径非法（期望 [索引] 或 .字段）：%r' % path)
        j = rest.find(']')
        if j < 0:
            raise ValueError('轴键路径非法（方括号未闭合）：%r' % path)
        try:
            segs.append(int(rest[1:j]))
        except ValueError:
            raise ValueError('轴键路径里的索引必须是整数：%r' % path)
        rest = rest[j + 1:]
        if rest.startswith('.'):
            k = rest.find('[', 1)
            if k < 0:
                segs.append(rest[1:])
                rest = ''
            else:
                segs.append(rest[1:k])
                rest = rest[k:]
        elif rest:
            raise ValueError('轴键路径非法（索引后只能是 .字段）：%r' % path)
    return key, segs


def get_in(params, path):
    """按轴键从 params 快照取值（KeyError 带可读信息）。"""
    key, segs = parse_path(path)
    if key not in params:
        raise KeyError('轴键 %r 不在登记册快照里（键 = %r）。'
                       '用 list_params() 看合法名字' % (path, key))
    cur = params[key]
    for s in segs:
        if isinstance(s, int):
            if not isinstance(cur, (list, tuple)) or not (0 <= s < len(cur)):
                raise KeyError('轴键 %r 的索引 %d 越界（当前长度 %s）'
                               % (path, s, len(cur)
                                  if isinstance(cur, (list, tuple)) else '—'))
            cur = cur[s]
        else:
            if not isinstance(cur, dict) or s not in cur:
                raise KeyError('轴键 %r 的字段 %r 不存在（现有键：%s）'
                               % (path, s,
                                  sorted(cur) if isinstance(cur, dict)
                                  else type(cur).__name__))
            cur = cur[s]
    return cur


def set_in(params, path, value):
    """按轴键写 params（**就地**；调用方负责先深拷贝）。"""
    key, segs = parse_path(path)
    if key not in params:
        raise KeyError('轴键 %r 不在登记册快照里（键 = %r）' % (path, key))
    if not segs:
        params[key] = value
        return
    cur = params[key]
    for n, s in enumerate(segs):
        last = (n == len(segs) - 1)
        if isinstance(s, int):
            if not isinstance(cur, (list, tuple)) or not (0 <= s < len(cur)):
                raise KeyError('轴键 %r 的索引 %d 越界' % (path, s))
            if last:
                cur[s] = value
            else:
                cur = cur[s]
        else:
            if not isinstance(cur, dict):
                raise KeyError('轴键 %r：%r 不是字典，无法取字段 %r'
                               % (path, type(cur).__name__, s))
            if last:
                cur[s] = value
            else:
                if s not in cur:
                    raise KeyError('轴键 %r 的字段 %r 不存在' % (path, s))
                cur = cur[s]


# ---------------------------------------------------------------------
# 轴展开
# ---------------------------------------------------------------------
def expand_axis(value):
    """轴值 → 取值列表（见模块 docstring 的四种形态）。"""
    if isinstance(value, dict):
        if 'value' in value:
            return [value['value']]
        if 'linspace' in value:
            a, b, n = value['linspace']
            return [float(x) for x in np.linspace(float(a), float(b),
                                                  int(n))]
        if 'arange' in value:
            a, b, step = value['arange']
            step = float(step)
            if step == 0:
                raise ValueError('arange 步长不能为 0')
            out, x = [], float(a)
            if step > 0:
                while x <= float(b) + 1e-12:
                    out.append(x)
                    x += step
            else:
                while x >= float(b) - 1e-12:
                    out.append(x)
                    x += step
            return out
        if 'logspace' in value:
            a, b, n = value['logspace']
            return [float(x) for x in np.logspace(np.log10(float(a)),
                                                  np.log10(float(b)),
                                                  int(n))]
        raise ValueError('轴值 dict 只认 value/linspace/arange/logspace：%r'
                         % (sorted(value),))
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


def build_points(axes=None, points=None):
    """(axes, points) → [delta_dict, …]（笛卡尔积，顺序确定）。

    delta_dict = {轴键: 取值}，可直接喂 `apply_delta`。
    """
    out = []
    axes = dict(axes or {})
    names = sorted(axes)                      # 排序 ⇒ 结果表确定、可复现
    if names:
        grids = [expand_axis(axes[k]) for k in names]
        for combo in itertools.product(*grids):
            out.append(dict(zip(names, combo)))
    for pt in (points or []):
        if not isinstance(pt, dict):
            raise ValueError('points 每项必须是 dict（轴键→取值），收到 %r'
                             % (pt,))
        out.append(dict(pt))
    if not out:
        raise ValueError('扫描点为空：请给 axes 或 points')
    return out


def apply_delta(params, delta):
    """params 深拷贝 + 应用 delta → 新 params（不改原快照）。"""
    p = copy.deepcopy(params)
    for path, value in (delta or {}).items():
        set_in(p, path, value)
    return p


# ---------------------------------------------------------------------
# 描述（回执给 AI/人看）
# ---------------------------------------------------------------------
def describe_spec(spec):
    """扫描规格 → 回执摘要（点数 / 各轴取值数 / 前几个点）。"""
    axes = spec.get('axes') or {}
    points = spec.get('points') or []
    deltas = build_points(axes, points)
    per_axis = {}
    for k in sorted(axes):
        vals = expand_axis(axes[k])
        per_axis[k] = {'n': len(vals),
                       'first': vals[0] if vals else None,
                       'last': vals[-1] if vals else None}
    return {
        'n_points': len(deltas),
        'axes': per_axis,
        'n_explicit_points': len(points),
        'head': deltas[:3],
    }
