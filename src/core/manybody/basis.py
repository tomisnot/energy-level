"""Two-level basis identity: site zero is the most significant bit."""
from functools import lru_cache
import numpy as np


def basis_bits(n, states):
    states = np.asarray(states, dtype=np.int64)
    return np.asarray([((states >> (int(n) - 1 - i)) & 1)
                       for i in range(int(n))], dtype=np.int8)


def validate_edges(n, edges):
    out = set()
    for edge in edges or []:
        if len(edge) != 2 or any(isinstance(i, (bool, np.bool_)) or int(i) != i for i in edge):
            raise ValueError('blockade edge 必须是两个整数下标')
        i, j = map(int, edge)
        if not 0 <= i < n or not 0 <= j < n or i == j:
            raise ValueError('blockade edge 越界或自环：%r' % (edge,))
        out.add(tuple(sorted((i, j))))
    return sorted(out)


def independent_states(n, edges, max_states=1 << 20):
    """Enumerate only independent sets, with an explicit output-size limit."""
    n = int(n)
    if not 0 <= n <= 63:
        raise ValueError('位编码要求 0 <= N <= 63')
    edges = validate_edges(n, edges)
    blocked = [0] * n
    for i, j in edges:
        blocked[i] |= 1 << j
        blocked[j] |= 1 << i
    out = []

    def visit(site, occupied, state):
        if site == n:
            if len(out) >= int(max_states):
                raise ValueError('受限基维度超过 max_states=%d；请减小系统或调整资源预算' % max_states)
            out.append(state)
            return
        visit(site + 1, occupied, state)
        if not blocked[site] & occupied:
            visit(site + 1, occupied | (1 << site), state | (1 << (n - 1 - site)))

    visit(0, 0, 0)
    return out


def independent_set_count(n, edges, max_nodes=100000):
    """Deletion/contraction count; no list of 2**N configurations is built."""
    n = int(n)
    adj = [0] * n
    for i, j in validate_edges(n, edges):
        adj[i] |= 1 << j
        adj[j] |= 1 << i

    visited = 0

    @lru_cache(maxsize=100000)
    def count(mask):
        nonlocal visited
        visited += 1
        if visited > max_nodes:
            raise ValueError('约束图精确计数超过预算；需装配后确定有效维度')
        if not mask:
            return 1
        # Isolated vertices factor out, including an empty graph of large N.
        active = [i for i in range(n) if mask & (1 << i)]
        site = max(active, key=lambda i: (adj[i] & mask).bit_count())
        if not adj[site] & mask:
            return 1 << len(active)
        rest = mask & ~(1 << site)
        return count(rest) + count(rest & ~adj[site])

    return count((1 << n) - 1)
