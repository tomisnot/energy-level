# -*- coding: utf-8 -*-
"""轻量回归：球基通道分解能无损重构一阶 4×4 块。"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from scipy import sparse as sp

from src.core.manybody.assemble import calibrate_pair, decompose_pair_channels


single = [
    dict(n=1, l=0, j=0.5, kind='mJ', mj=0.5, E_mhz=0.0),
    dict(n=1, l=1, j=0.5, kind='mJ', mj=1.5, E_mhz=10.0),
]
D = {
    1: sp.csr_matrix(np.array([[0, 0], [1.0 + 0.2j, 0]], complex)),
    0: sp.csr_matrix((2, 2), dtype=complex),
    -1: sp.csr_matrix(np.array([[0, 0], [0.3 - 0.1j, 0]], complex)),
}
E = np.array([0.0, 10.0])
pair4 = ((0, 0), (1, 0), (0, 1), (1, 1))

out = decompose_pair_channels(single, D, pair4, 7.0, 47.0, 23.0)
ref = calibrate_pair(None, single, D, E, pair4, 7.0, 47.0, 23.0,
                     second_order=False)['delta1']
err = float(np.max(np.abs(out['total'] - ref)))
assert err < 1e-12, err
assert len(out['channels']) > 0
assert all({'q1', 'q2', 'tensor', 'matrix'} <= set(c) for c in out['channels'])
print('PASS channel decomposition reconstruction; max error=%.3e; channels=%d'
      % (err, len(out['channels'])))
