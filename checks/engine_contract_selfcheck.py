# -*- coding: utf-8 -*-
"""引擎接口契约自洽（这一组不变式守的是"接口说的话是真的"）。

A. **分派表唯一**：`runspec._ENGINE_TABLE` 覆盖 6 个 (method, two_body) 组合，
   每个解析到真实可调用的引擎函数，未知组合明确报错。
B. **参数白名单 = 签名本身**：`_filter_kw` 只留引擎签名收的键；
   且**每个 `solve.*` 映射出来的 kwarg 至少有一个引擎签名接收**——否则那个
   `solve.*` 是死的（`max_step_hard` 就曾这样静默失效）。
C. **`torch` 不在 core 的导入链里**：`import src.core` 不得拉入 GPU 模块
   （它只有一个 torch 入口，且是函数体内惰性导入）。
D. **诊断归属可见**：只有 `mixed` 家族的引擎签名带 `diag`；`summarize(None)`
   的消息要点名引擎，不能再说"diag=True 才有"（那会让人以为是自己没传）。
"""
import inspect
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

from checks.checklib import Checker                            # noqa: E402
from src.core import diag as _diag                             # noqa: E402
from src.core import runspec as RS                             # noqa: E402

C = Checker('engine_contract')

# ---------------- A. 分派表唯一且自洽 ----------------
EXPECT = {('mixed', False), ('mixed', True), ('sambe', False), ('sambe', True),
          ('hard', False), ('hard', True)}
resolved = {}
for key in EXPECT:
    try:
        resolved[key] = RS.engine_fn(*key)
    except Exception as e:                                      # noqa: BLE001
        resolved[key] = None
        print('  [warn] engine_fn%s 失败：%s' % (key, e))

bad_unknown = False
try:
    RS.engine_fn('ip4', False)
except ValueError:
    bad_unknown = True
except Exception:                                               # noqa: BLE001
    pass

C.group('A 分派表唯一（`runspec._ENGINE_TABLE`）', [
    ('覆盖 6 个 (method, two_body) 组合：%d'
     % len([k for k in EXPECT if resolved.get(k)]),
     all(resolved.get(k) is not None for k in EXPECT)),
    ('每个都解析到可调用对象',
     all(callable(resolved.get(k)) for k in EXPECT)),
    ('未知 method 明确报错（不静默）', bad_unknown),
    ('六个函数互不重复（没有第二个同名实现）',
     len({id(f) for f in resolved.values() if f}) == 6),
])

# ---------------- B. 白名单 = 签名 ----------------
POSSIBLE = ['max_step', 'max_step_hard', 'per_freq_M', 'frozen_h_ns',
            'phase_budget', 'tol']
mismatch = []
for key, fn in resolved.items():
    if fn is None:
        continue
    sig = set(inspect.signature(fn).parameters)
    for k in POSSIBLE:
        # `_engine_accepts` 是唯一判据；它必须等于"签名里有这个形参"
        if RS._engine_accepts(fn, k) != (k in sig):
            mismatch.append('%s/%s' % (key, k))

# 每个 solve.* 映射出的 kwarg 至少被一个引擎接收
solve_kwargs = set(RS.SOLVE_KWARG.values())
dead_solve = sorted(
    kw for kw in solve_kwargs
    if not any(RS._engine_accepts(fn, kw)
               for fn in resolved.values() if fn is not None))

# `_filter_kw` 实测：塞一个签名收的 + 一个不收的
_fn = resolved[('mixed', False)]
_keep, _drop = RS._filter_kw(_fn, {'tol': 1.0, 'frozen_h_ns': 2.0,
                                   '__不存在__': 3.0})
C.group('B 参数白名单 = 引擎签名（不是手抄表）', [
    ('`_engine_accepts` 与签名逐键一致（%d 项）' % (len(resolved) * len(POSSIBLE)),
     not mismatch),
    ('`_filter_kw` 实测：收 tol、丢 frozen_h_ns/不存在键',
     set(_keep) == {'tol'} and _drop == {'frozen_h_ns', '__不存在__'}),
    ('每个 solve.* 至少被一个引擎接收（死键：%s）' % (dead_solve or '无'),
     not dead_solve),
])

# ---------------- C. torch 不在 core 的导入链里 ----------------
C.group('C torch 惰性（core 不拉 GPU 模块）', [
    ('`src.core.evolve_sambe_pair_gpu` 未被导入',
     'src.core.evolve_sambe_pair_gpu' not in sys.modules),
    ('`torch` 未被拉入', 'torch' not in sys.modules),
])

# ---------------- D. 诊断归属可见 ----------------
diag_engines = sorted(
    '%s/%s' % k for k, fn in resolved.items()
    if fn is not None and 'diag' in inspect.signature(fn).parameters)
_msg_hard = _diag.summarize(None, {'method': 'hard'}).get('msg', '')
_msg_bare = _diag.summarize(None).get('msg', '')
C.group('D 诊断归属可见（只有 mixed 家族产出）', [
    ('带 diag 形参的引擎恰为 mixed 家族：%s' % diag_engines,
     diag_engines == ['mixed/False', 'mixed/True']),
    ('消息点名引擎（method=hard）', 'hard' in _msg_hard),
    ('消息不再说"diag=True 才有"（那是在怪调用方）',
     'diag=True 才有' not in _msg_hard and 'diag=True 才有' not in _msg_bare),
])

# ---------------- E. 无跨文件重名（同名不同物是 evolve_hard 的温床） ----------------
import ast as _ast                                             # noqa: E402

_CORE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                     'src', 'core')
_names = {}
for _f in sorted(os.listdir(_CORE)):
    if not _f.endswith('.py'):
        continue
    _src = open(os.path.join(_CORE, _f), encoding='utf-8').read()
    for _nd in _ast.parse(_src).body:
        if isinstance(_nd, (_ast.FunctionDef, _ast.ClassDef)) \
                and not _nd.name.startswith('__'):
            _names.setdefault(_nd.name, []).append(_f)
_dups = sorted('%s(%s)' % (n, ','.join(v)) for n, v in _names.items()
               if len(v) > 1)
C.group('E 无跨文件重名（`evolve_hard` 那一类温床）', [
    ('重名数 = %d（%s）' % (len(_dups), '、'.join(_dups) if _dups else '干净'),
     not _dups),
    ('顶层定义总数 = %d（非空）' % len(_names), len(_names) > 100),
])

# ---------------- F. 判据对 src.core 的私有依赖有界 ----------------
# 判据里允许存在的**白盒依赖**（内部件，不是产品 API）。**新增一项都要改这里**——
# 这是刻意的摩擦：私有耦合必须是显式决定，不能顺手长出来。
_WHITELIST = {
    ('src.core.evolve_mixed', '_MixedSetup'),
    ('src.core.evolve_mixed', '_assemble_H_snapshot'),
    ('src.core.hard', '_pack_pair_hard'),
    # C5（第 14 刀）：hard/mixed 共用的耗散落点校验器，无更轻的公开入口；
    # dissip_contract_selfcheck 走它验证报错文案。刻意显式放行。
    ('src.core.evolve_util', '_single_loss_rates'),
}
_CHK = os.path.join(os.path.dirname(os.path.abspath(__file__)))
_priv_imports, _dots = set(), set()
for _f in sorted(os.listdir(_CHK)):
    if not _f.endswith('.py') or _f == 'checklib.py':
        continue
    _t = _ast.parse(open(os.path.join(_CHK, _f), encoding='utf-8').read())
    _locals = {}
    for _nd in _ast.walk(_t):
        if isinstance(_nd, _ast.ImportFrom) and _nd.module \
                and _nd.module.startswith('src.core'):
            for _a in _nd.names:
                if _a.name.startswith('_') and not _a.name.startswith('__'):
                    _priv_imports.add((_nd.module, _a.name))
                    _locals[_a.asname or _a.name] = True
    # 只有**真正被调用**的才算依赖（死 import 不该进白名单）
    _called = {_nd.func.id for _nd in _ast.walk(_t)
               if isinstance(_nd, _ast.Call) and isinstance(_nd.func, _ast.Name)}
    _priv_imports = {k for k in _priv_imports
                     if k[1] in _called or k[1] not in _locals}
    for _nd in _ast.walk(_t):
        if isinstance(_nd, _ast.Attribute) and _nd.attr.startswith('_') \
                and not _nd.attr.startswith('__') \
                and isinstance(_nd.value, _ast.Name) \
                and _nd.value.id in ('core', 'pair_core', 'SA', 'SP'):
            _dots.add(_nd.attr)

_extra = sorted('%s:%s' % k for k in (_priv_imports - _WHITELIST))
_missing = sorted('%s:%s' % k for k in (_WHITELIST - _priv_imports))
C.group('F 判据的私有依赖有界（白名单 %d 条）' % len(_WHITELIST), [
    ('白名单之外没有新增私有 import（新增：%s）' % (_extra or '无'), not _extra),
    ('白名单里没有失效项（失效：%s）' % (_missing or '无'), not _missing),
    ('无点号直取的私有名（残留：%s）' % (_dots or '无'), not _dots),
])

sys.exit(0 if C.summary() else 1)
