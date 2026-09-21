# -*- coding: utf-8 -*-
"""CodeAct 命名空间（纲领 D1：调语义层，不调裸引擎；P0.4：表达自由拉满、契约锁死）。

一次调用多步编排：AI 写一段脚本，在**受限命名空间**里 exec。命名空间只暴露
`registry / session / ops` 的**门面**（facade）+ 安全内建，**不暴露裸 core**
（runspec/build_run/evolve_run/static_atom/np…）——于是自由脚本仍**继承全部契约**
（单位/值域/可逆性/回执/事件），组合自由拉满而契约自由度锁死。

**这不是安全沙箱**（CPython exec 无法真正沙箱化）；它是"守纪律的便利层"：
- 堵住**显而易见**的侧门：门面无 `_snapshot`/`_write`（改状态只能经 `session.set`）；
  内建无 `__import__`/`open`/`eval`（拿不到裸引擎、碰不到文件系统）。
- 真正的强制在别处：`session.set` 的校验 + 总线 append-only 日志 + `mecha_kernel_
  selfcheck` 的 I1 侧门 AST 守卫。
- **D3 显式边界**：`NOT_EXPOSED` 列出"不在命名空间里的重要名字"，AI 不必靠撞错发现。
"""
import builtins as _builtins
import io
import math                      # 纯函数数学（无副作用），科学编排必备（sin/cos/pi…）

from .errors import MechaError

__all__ = ['run_codeact', 'NOT_EXPOSED', 'SAFE_BUILTINS', 'namespace']

# D3：显式声明**不在**命名空间里的重要名字（AI 该知道边界，而非执行后撞错）
NOT_EXPOSED = [
    'runspec', 'build_run', 'evolve_run', 'result_summary',      # 裸分派/引擎
    'static_atom', 'sambe', 'evolve_mixed', 'evolve_twofield', 'hard',
    'coupling', 'field_build', 'atomic_levels',                    # 裸 core 模块
    'numpy', 'np', 'arc', 'torch', 'scipy',                        # 裸数值/物理库
    'waveforms', 'Wave', 'ArbitraryWave', 'GaussianEnv', 'SquareEnv',
    'FileEnv', 'serialize_waves', 'restore_waves',                 # 裸波形类（走 fields 构造器，不手搓）
    'os', 'sys', 'subprocess', 'socket',                           # 系统面
    '__import__', 'open', 'eval', 'exec', 'compile', 'input',      # 逃逸口
    'globals', 'locals', 'vars', 'breakpoint',
]

# 安全内建白名单（**无** __import__/open/eval/exec/compile/反射逃逸口）
_SAFE_NAMES = (
    'abs', 'all', 'any', 'bool', 'dict', 'divmod', 'enumerate', 'filter',
    'float', 'format', 'frozenset', 'int', 'isinstance', 'issubclass', 'iter',
    'len', 'list', 'map', 'max', 'min', 'next', 'pow', 'range', 'repr',
    'reversed', 'round', 'set', 'slice', 'sorted', 'str', 'sum', 'tuple', 'zip',
    'True', 'False', 'None', 'Exception', 'ValueError', 'TypeError', 'KeyError',
)
# 从 builtins **模块**取（不用 __builtins__：它在不同上下文可能是 dict，getattr 取不到）
SAFE_BUILTINS = {n: getattr(_builtins, n) for n in _SAFE_NAMES
                 if hasattr(_builtins, n)}


class _Facade:
    """只暴露白名单方法的门面对象；任何其它属性（尤其 `_snapshot`/`_write`）
    都触发**可教学**的 AttributeError（堵显而易见的侧门，D1）。"""

    def __init__(self, name, methods):
        object.__setattr__(self, '_facade_name', name)
        object.__setattr__(self, '_facade_methods', dict(methods))

    def __getattr__(self, key):
        methods = object.__getattribute__(self, '_facade_methods')
        name = object.__getattribute__(self, '_facade_name')
        if key in methods:
            return methods[key]
        raise AttributeError(
            "CodeAct 命名空间里 %s 只提供 %s；%r 不可达——改状态请经 "
            "session.set()（D1：调语义层，不碰内部/裸引擎）"
            % (name, sorted(methods), key))

    def __dir__(self):
        return sorted(object.__getattribute__(self, '_facade_methods'))

    def __repr__(self):
        return '<facade %s: %s>' % (
            object.__getattribute__(self, '_facade_name'),
            ', '.join(sorted(object.__getattribute__(self, '_facade_methods'))))


def namespace(session, ops, jobs=None, fields=None, printer=None):
    """构造受限命名空间：registry/session/ops（+ 可选 jobs/fields）的**门面** + 安全内建。

    门面只转发公开方法；私有实现（session._snapshot/_write、ops.session）够不着。
    jobs 给了就暴露异步作业面；fields 给了就暴露建场层（写方法注入 actor='codeact'）。
    """
    from . import registry as _reg

    sess = _Facade('session', {
        'set': lambda key, value, reason: session.set(
            key, value, actor='codeact', reason=reason),
        'set_many': lambda patch, reason: session.set_many(
            patch, actor='codeact', reason=reason),
        'get': session.get,
        'effective': session.effective,
        'undo': lambda reason='undo': session.undo(actor='codeact', reason=reason),
        'activity': session.activity,
        'orientation': session.orientation,
    })
    opsf = _Facade('ops', {
        'run': lambda reason, diag=False: ops.run(actor='codeact', reason=reason,
                                                  diag=diag),
        'estimate': ops.estimate,
        'preview': ops.preview,
        'rerun': lambda aid, reason: ops.rerun(aid, actor='codeact', reason=reason),
        'propose': lambda reason: ops.propose(actor='codeact', reason=reason),
        'autonomy_level': ops.autonomy_level,
        'state': ops.state,
        'activity': ops.activity,
    })
    regf = _Facade('registry', {
        'list_params': _reg.list_params,
        'describe': _reg.describe,
        'families': _reg.families,
        'validate': _reg.validate,
        'all_settable_keys': _reg.all_settable_keys,
        'solve_map': _reg.solve_map,
    })
    ns = {
        '__builtins__': dict(SAFE_BUILTINS),
        '__name__': 'codeact',
        'registry': regf, 'session': sess, 'ops': opsf,
        'math': math,                    # 纯数学（sin/cos/sqrt/pi…），无副作用故安全
        'NOT_EXPOSED': list(NOT_EXPOSED),
        'result': None,
    }
    if jobs is not None:
        ns['jobs'] = _Facade('jobs', {
            'submit': lambda reason: jobs.submit(reason=reason, actor='codeact'),
            'poll': jobs.poll, 'cancel': jobs.cancel, 'list_jobs': jobs.list_jobs,
        })
    if fields is not None:
        ns['fields'] = _fields_facade(fields)
    if printer is not None:
        ns['__builtins__']['print'] = printer
    return ns


# fields 门面：写方法注入 actor='codeact'（M1 归因），读方法直通
_F_WRITE = ('set_drive', 'add_wave', 'edit_wave', 'remove_wave', 'clear_waves',
            'set_sequence', 'set_waves', 'pi_pulse', 'pi2_pulse', 'square_mw',
            'ramsey', 'stirap', 'arp')
_F_READ = ('get_waves', 'get_envelope', 'freq_for', 'intensity_for_rabi',
           'rabi_of', 'describe_drive', 'suggest_freq')


def _fields_facade(fields):
    fm = {}
    for nm in _F_WRITE:
        def _mk(name):
            m = getattr(fields, name)

            def w(*a, **k):
                k['actor'] = 'codeact'
                k.setdefault('reason', 'codeact %s' % name)
                return m(*a, **k)
            return w
        fm[nm] = _mk(nm)
    for nm in _F_READ:
        fm[nm] = getattr(fields, nm)
    return _Facade('fields', fm)


def run_codeact(code, session, ops, jobs=None, fields=None, *, require_reason=True):
    """在受限命名空间里 exec 一段编排代码。

    约定：脚本把最终产物赋给 `result`（可 JSON 化的标量/dict）；`print(...)` 被捕获。
    返回 `{ok, result, printed, events, error?}`——events 是本次编排经 choke point
    落进总线的 seq 区间（可归因、可回放）。jobs 给了则命名空间多一个 `jobs` 门面。

    require_reason：set/run 都要 reason（M1）；门面已把 actor 固定为 'codeact'。
    """
    buf = io.StringIO()

    def _printer(*a, **k):
        k.pop('file', None)
        print(*a, file=buf, **k)

    seq0 = session.bus.last_seq
    ns = namespace(session, ops, jobs, fields, printer=_printer)
    try:
        exec(compile(code, '<codeact>', 'exec'), ns)   # noqa: S102（受限命名空间）
    except MechaError as e:
        return {'ok': False, 'error': e.to_dict(), 'printed': buf.getvalue(),
                'events': [ev['seq'] for ev in session.bus.since(seq0)]}
    except Exception as e:                              # noqa: BLE001
        return {'ok': False,
                'error': {'kind': type(e).__name__, 'message': str(e)},
                'printed': buf.getvalue(),
                'events': [ev['seq'] for ev in session.bus.since(seq0)]}
    new_events = session.bus.since(seq0)
    return {'ok': True, 'result': ns.get('result'), 'printed': buf.getvalue(),
            'events': [ev['seq'] for ev in new_events],
            'n_events': len(new_events)}
