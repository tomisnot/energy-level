# -*- coding: utf-8 -*-
"""CPU MPS 含时 Pauli 串：ED ↔ TDVP 对拍（R2）。

同一 calib、同一初态、同一 t_grid：
  · ED（全空间精确）作为参考；
  · TDVP（两站点、足够 chi）逐点比较 ``observables.pauli_strings``；
  · 同时验 ``pauli_strings=None`` 时结果不新增字段（默认逐位不变）。
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from checks.checklib import Checker
from checks.mb_benchlib import random_calib
from src.core.manybody.engine_ed import evolve_manybody
from src.core.manybody.engine_mps import evolve_mps
from src.core.manybody.schema_mb import validate_pauli_strings

C = Checker('manybody_pauli_mps')


def main():
    n = 4
    cal = random_calib(n, seed=5)
    specs = validate_pauli_strings(
        [[[0, 'z']], [[0, 'x'], [1, 'x']], [[1, 'y'], [2, 'y']]], n)
    t = np.linspace(0.0, 2.0, 5)
    common = dict(psi0='rggg', t_grid=t, Om=np.zeros(n, complex),
                  delta=np.zeros(n), terms=['I2'],
                  want_observables=True)
    ed = evolve_manybody(cal, pauli_strings=specs, **common)
    mps = evolve_mps(cal, algorithm='tdvp', chi_max=64, truncerr=1e-12,
                     max_step_ns=0.1, pauli_strings=specs, **common)
    a = np.asarray(ed['observables']['pauli_strings'], complex)
    b = np.asarray(mps['observables']['pauli_strings'], complex)
    C.group('TDVP 与 ED 的 Pauli 串逐点一致', [
        ('形状一致 %s' % (b.shape,), a.shape == b.shape),
        ('max|Δ| < 1e-6（实测 %.2e）' % np.max(np.abs(a - b)),
         np.max(np.abs(a - b)) < 1e-6),
        ('labels 回传',
         mps['observables']['pauli_string_labels']
         == [s['label'] for s in specs]),
        ('结果挂在 observables 下（与 ED 同形）',
         'pauli_strings' in (mps.get('observables') or {})),
    ])

    # 默认不启用时：不出现 Pauli 字段，且原有观测仍在。
    plain = evolve_mps(cal, algorithm='tdvp', chi_max=32, truncerr=1e-10,
                       max_step_ns=0.1, **common)
    obs = plain.get('observables') or {}
    C.group('pauli_strings=None 时默认结果不变', [
        ('无 pauli_strings 键', 'pauli_strings' not in obs),
        ('无 pauli_string_labels 键', 'pauli_string_labels' not in obs),
        ('sx/sz 仍在', 'sx' in obs and 'sz' in obs),
    ])

    # TEBD 在最近邻链上也应接受（同一条 evolve_mps 接线）。
    tebd = evolve_mps(cal, algorithm='tebd', chi_max=64, truncerr=1e-12,
                      max_step_ns=0.1, pauli_strings=specs, **common)
    c = np.asarray(tebd['observables']['pauli_strings'], complex)
    C.group('TEBD 走同一条测量入口', [
        ('max|Δ vs ED| < 1e-6（实测 %.2e）' % np.max(np.abs(a - c)),
         np.max(np.abs(a - c)) < 1e-6),
    ])
    return 0 if C.summary() else 1


if __name__ == '__main__':
    sys.exit(main())
