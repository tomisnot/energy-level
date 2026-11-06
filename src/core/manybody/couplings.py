# -*- coding: utf-8 -*-
"""从 calib 的**数值**推导活跃的相互作用通道。

多体模型只有一张 ``pairs`` 表（V/J/A）与一张 ``atoms`` 表；"哪些通道开着"
不是一个需要用户声明的选项，而是数据本身的属性：

    V ≠ 0 → I1（密度-密度）
    J ≠ 0 → I2（交换）
    A ≠ 0 → I3（双激发）

构造器负责把不用的通道写成精确的 ``0.0``，因此这里用**精确非零**判定，
不引入阈值。引擎内部的 ``terms`` 参数仍然存在（供直接调用的研究 API 用），
但省略时必须走这个推导，而不是回落到"全开"。
"""

__all__ = ['active_terms']


def _nonzero(value):
    try:
        return abs(complex(value or 0.0)) > 0.0
    except (TypeError, ValueError):
        return False


def active_terms(calib):
    """``calib`` → ``['I1', …]``。

    优先用 ``calib['terms']``（研究 API 的显式选择）；**没有**该键时按 V/J/A 的
    数值推导（RunSpec 的 calib 由 ``validate_calib`` 生成，两者一致）。
    """
    given = (calib or {}).get('terms')
    if given is not None:
        return list(given)
    pairs = (calib or {}).get('pairs') or []
    out = []
    for name, key in (('I1', 'V'), ('I2', 'J'), ('I3', 'A')):
        if any(_nonzero(p.get(key)) for p in pairs):
            out.append(name)
    return out
