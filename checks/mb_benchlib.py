# -*- coding: utf-8 -*-
"""多体判据的共享基准基座（support，不是判据）。

只放三件与物理模型无关的事：

* ``random_calib``：非对称位置 + 复 J/A + 可选环，专门用来暴露位序/符号错误；
* ``dense_H``：引擎稀疏装配的稠密化（作为独立参照的入口）；
* ``random_state``：固定种子的复态矢。

判据不要各自再写一套 setup；新判据优先从这里取。
"""
import numpy as np

__all__ = ['random_calib', 'dense_H', 'random_state']


def random_calib(n, seed=0, *, ring=False, complex_couplings=True):
    """随机但**非对称**的二能级点云 calib（位置链/环、复 J/A、V 实）。

    ``n`` 至少 2；返回的 atoms 只带 id/pos，pairs 带 V/J/A。
    """
    n = int(n)
    if n < 2:
        raise ValueError('random_calib 需要 n >= 2')
    rng = np.random.default_rng(int(seed))
    atoms = [{'id': i, 'pos': [float(i), 0.0, 0.0]} for i in range(n)]
    pairs = []
    for i in range(n if ring else n - 1):
        j = (i + 1) % n
        J = 0.4 + 0.2 * rng.normal()
        A = 0.03 * rng.normal()
        if complex_couplings:
            J = J + 1j * 0.15 * rng.normal()
            A = A + 1j * 0.02 * rng.normal()
        pairs.append({'i': i, 'j': j, 'V': 0.2 * rng.normal(),
                      'J': complex(J), 'A': complex(A)})
    return {'atoms': atoms, 'pairs': pairs}


def dense_H(calib, Om=None, delta=None, terms=None, gamma=None):
    """引擎稀疏 H → 稠密数组（只用于小系统参照）。"""
    from src.core.manybody.assemble import build_H_sparse
    return build_H_sparse(calib, Om, delta, terms, gamma).toarray()


def random_state(dim, seed=0):
    """固定种子的归一化复态矢。"""
    rng = np.random.default_rng(int(seed))
    v = rng.normal(size=int(dim)) + 1j * rng.normal(size=int(dim))
    return v / np.linalg.norm(v)
