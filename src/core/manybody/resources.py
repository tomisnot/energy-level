"""Conservative numerical workspace bounds, expressed in MiB."""
import math


def budget_bytes(memory_budget_mb):
    if memory_budget_mb is None:
        return None
    value = float(memory_budget_mb)
    if not math.isfinite(value) or value <= 0:
        raise ValueError('memory_budget_mb 必须是有限正数')
    return int(value * 1024**2)


def dense_limit(dense_max, memory_budget_mb, copies=4):
    budget = budget_bytes(memory_budget_mb)
    return int(dense_max) if budget is None else min(int(dense_max), int(math.sqrt(budget/(16*copies))))


def require_workspace(size, memory_budget_mb, description):
    budget = budget_bytes(memory_budget_mb)
    if budget is not None and size > budget:
        raise ValueError('%s 预计至少 %.2f MiB，超过预算 %.2f MiB' %
                         (description, size/1024**2, budget/1024**2))
