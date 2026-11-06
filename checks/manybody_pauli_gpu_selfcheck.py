# -*- coding: utf-8 -*-
"""GPU MPS 的 Pauli 串：CPU TDVP 对拍（R3）。

无 CUDA 时诚实 SKIP（退出码 0），不伪造通过。
覆盖两种 GPU 路由：最近邻 ``tebd`` 与任意键 ``swap-tebd``。
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from checks.checklib import Checker
from src.core.manybody.engine_mps import evolve_mps
from src.core.manybody.engine_mps_gpu import evolve_mps_gpu
from src.core.manybody.schema_mb import validate_pauli_strings

C = Checker('manybody_pauli_gpu')


def _calib(n, ring=False):
    pairs = ([{'i': i, 'j': (i + 1) % n, 'V': 0.0, 'J': 0.5, 'A': 0.0}
              for i in range(n)] if ring else
             [{'i': i, 'j': i + 1, 'V': 0.0, 'J': 0.5, 'A': 0.0}
              for i in range(n - 1)])
    return {'atoms': [{'id': i, 'pos': [float(i), 0.0, 0.0]} for i in range(n)],
            'pairs': pairs}


def main():
    try:
        import torch
        if not torch.cuda.is_available():
            C.skip('GPU MPS Pauli', 'CUDA 不可用')
            C.summary()
            return 0
    except Exception as exc:                                  # noqa: BLE001
        C.skip('GPU MPS Pauli', 'torch 不可用：%s' % exc)
        C.summary()
        return 0

    n = 4
    specs = validate_pauli_strings(
        [[[0, 'z']], [[0, 'x'], [1, 'x']], [[1, 'y'], [2, 'y']]], n)
    t = np.linspace(0.0, 1.5, 4)
    common = dict(psi0='rggg', t_grid=t, Om=np.zeros(n, complex),
                  delta=np.zeros(n), terms=['I2'], want_observables=True)
    for ring, algorithm, label in ((False, 'tebd', 'tebd-gpu'),
                                   (True, 'swap-tebd', 'swap-tebd-gpu')):
        cal = _calib(n, ring=ring)
        cpu = evolve_mps(cal, algorithm='tdvp', chi_max=128, truncerr=1e-12,
                         max_step_ns=0.1, pauli_strings=specs, **common)
        gpu = evolve_mps_gpu(cal, algorithm=algorithm, chi_max=128,
                             truncerr=1e-12, max_step_ns=0.1,
                             dtype='float64', device='cuda', order=2,
                             pauli_strings=specs, **common)
        a = np.asarray(cpu['observables']['pauli_strings'], complex)
        b = np.asarray(gpu['observables']['pauli_strings'], complex)
        C.group('%s 与 CPU TDVP 一致' % label, [
            ('形状一致 %s' % (b.shape,), a.shape == b.shape),
            ('max|Δ| < 1e-8（实测 %.2e）' % np.max(np.abs(a - b)),
             np.max(np.abs(a - b)) < 1e-8),
            ('labels 回传',
             gpu['observables']['pauli_string_labels']
             == [s['label'] for s in specs]),
            ('error_report 标注 backend/order',
             gpu['error_report'].get('mps_order') is not None),
        ])
    return 0 if C.summary() else 1


if __name__ == '__main__':
    sys.exit(main())
