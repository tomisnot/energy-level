# -*- coding: utf-8 -*-
"""机甲框架 · 语义内核（`mecha`）——引擎之外的状态与纪律机器。

定位（纲领 §2 / §13）：**住在引擎之外**，`mecha → core` 单向依赖，`core ↛ mecha`。
它不重写物理、不造并行第二真源，只把"同一意义收归一处定义"并做纪律：

    registry.py   参数语义的单一查询口（收集 core 的 Param 声明，非复制）
    bus.py        append-only 事件总线（唯一真值日志；大载荷指针+sha1；体积闸）
    session.py    会话态 + 写 choke point（set → 校验 → 发事件；no side doors）
    errors.py     可教学错误（kind/hint/suggest；D2）
    ops.py        语义操作（run/estimate/preview/undo）经 choke point → runspec（P2）
    server.py     MCP 写通道 + 只读订阅端点（P2/P3）
    codeact.py    CodeAct 命名空间（只暴露 registry/session/ops，不暴露裸 core）（P4）

**刻意不 eager-import 子模块**：`registry` 会拉起 `core`（含 ARC，~2s）。用 PEP 562
惰性属性，让 `import src.mecha` 保持廉价，消费方（server/codeact/selfcheck）按需取。
"""

__all__ = ['Session', 'Bus', 'MISSING', 'registry', 'session', 'bus', 'errors',
           'MechaError']

_LAZY_ATTR = {
    'Session': ('.session', 'Session'),
    'MISSING': ('.session', 'MISSING'),
    'Bus': ('.bus', 'Bus'),
    'MechaError': ('.errors', 'MechaError'),
}
_LAZY_MOD = ('registry', 'session', 'bus', 'errors', 'ops', 'server',
             'codeact', 'jobs', 'autonomy', 'subscribe')


def __getattr__(name):
    import importlib
    if name in _LAZY_ATTR:
        mod, attr = _LAZY_ATTR[name]
        return getattr(importlib.import_module(mod, __name__), attr)
    if name in _LAZY_MOD:
        return importlib.import_module('.' + name, __name__)
    raise AttributeError('module %r has no attribute %r' % (__name__, name))


def __dir__():
    return sorted(list(globals()) + list(_LAZY_ATTR) + list(_LAZY_MOD))
