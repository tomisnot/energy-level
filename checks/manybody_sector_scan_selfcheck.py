# -*- coding: utf-8 -*-
"""sector_scan（逐扇区扫描）自检：结果 == 全空间精确基态；契约与拒绝路径正确。"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from checks.checklib import Checker                     # noqa: E402
from src.core.manybody.assemble import build_H_sparse   # noqa: E402
from src.core.manybody.sector_scan import scan_ground_sectors  # noqa: E402

C = Checker('sector_scan')

n = 6
calib = {'atoms': [{'id': i, 'pos': (float(i), 0.0, 0.0)} for i in range(n)],
         'terms': ['I1', 'I2'],
         'pairs': [{'i': i, 'j': i + 1, 'V': 0.0, 'J': -0.7, 'A': 0.0} for i in range(n - 1)]}
delta = np.array([0.6, -0.4, 0.2, 0.9, -0.7, 0.1])

r = scan_ground_sectors(calib, delta=delta, Om=np.zeros(n), terms=['I1', 'I2'],
                        device='cpu', dense_max=64)

H = build_H_sparse(calib, Om=np.zeros(n), delta=delta)
w = np.linalg.eigvalsh(H.toarray())
C.close('E0 == 全空间精确基态', r['E0'], w[0], atol=1e-10, rtol=0.)

# 扇区侧：逐扇区稠密块的最小值应能复现同一个 E0，且 argmin 一致
best_k, best_e = None, None
for k in range(n + 1):
    idx = np.array([c for c in range(1 << n) if int(c).bit_count() == k], dtype=int)
    ek = np.linalg.eigvalsh(H.toarray()[np.ix_(idx, idx)])[0]
    if best_e is None or ek < best_e:
        best_k, best_e = k, ek
C.close('逐扇区 argmin == 全空间基态', r['E0'], best_e, atol=1e-10, rtol=0.)
C.check('sector 与稠密块一致', r['sector'] == best_k, '%d vs %d' % (r['sector'], best_k))
C.check('总基 = 2^N', r['dimension_sum'] == 1 << n, str(r['dimension_sum']))
C.check('每扇区都带残差/收敛标记',
        all(('residual' in row and 'converged' in row) for row in r['per_sector']))
C.check('全部扇区收敛', r['converged'] is True)

# 非数守恒（Ω≠0）：必须拒绝，而不是静默扫描
try:
    scan_ground_sectors(calib, delta=delta, Om=np.full(n, 0.1), terms=['I1', 'I2'],
                        device='cpu')
    C.check('Ω≠0 必须拒绝', False, '未抛异常')
except ValueError as exc:
    C.check('Ω≠0 必须拒绝', 'charge_sector' in str(exc) or 'Omega' in str(exc), str(exc)[:60])

# 扇区越界
try:
    scan_ground_sectors(calib, delta=delta, Om=np.zeros(n), terms=['I1', 'I2'],
                        sectors=[0, n + 1], device='cpu')
    C.check('越界扇区必须拒绝', False, '未抛异常')
except ValueError:
    C.check('越界扇区必须拒绝', True)

sys.exit(0 if C.summary() else 1)
