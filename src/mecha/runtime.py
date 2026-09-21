# -*- coding: utf-8 -*-
"""引擎注入点（服务定位）——`mecha` 经此拿到领域引擎，自身**零领域 import**。

为什么用服务定位而非到处传参：mecha 的 registry/session/ops 被 server/codeact/判据多处
调用；若改成 DI 逐个传 engine，签名 churn 面极大且易漏。改为**进程内注入一次**
（app 启动 / 判据开头 `set_engine(EngineAdapter())`），mecha 内部 `engine()` 取用。
单引擎单进程，无歧义；提炼守卫仍成立（mecha 只 import 本模块，不 import core/adapter）。

**依赖方向**：`adapter/app_entry → runtime.set_engine`；`mecha.* → runtime.engine()`。
"""
__all__ = ['set_engine', 'engine', 'has_engine']

_ENGINE = None


def set_engine(e):
    """注入领域引擎（实现 `mecha.protocol.EngineProtocol` 的对象）。返回 e 便于链式。"""
    global _ENGINE
    _ENGINE = e
    return e


def engine():
    """取当前注入的引擎；未注入 → 明确报错（不静默）。"""
    if _ENGINE is None:
        raise RuntimeError(
            'mecha 引擎未注入——先 mecha.runtime.set_engine(EngineAdapter())'
            '（app_entry / adapter.serve 启动时做；判据在开头做）')
    return _ENGINE


def has_engine():
    return _ENGINE is not None
