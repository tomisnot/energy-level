# -*- coding: utf-8 -*-
"""验收（非零情形）：同一个**复数**模型上，ED 与 DMRG 的三体手性必须一致。

为什么需要这一步：实哈密顿量的基态（时间反演对称）必有 χ≡0，只能证明"能算"，
不能证明"算对"。这里给 kagome 加一个**Landau 规范**的相位
``J_ij → J_ij·exp(iφ(x_j−x_i))``（每个 plaquette 有净通量），基态变复、χ≠0，
再用 12 点 ED（精确）与 12 点 DMRG 交叉验证。

用法：python scripts/verify_pauli_chirality_dmrg.py
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.core.manybody.eq_engine import solve_equilibrium        # noqa: E402
from src.core.manybody.observables import normalize_pauli_strings  # noqa: E402

CALIB = Path(r"D:\code-nosync\energy-level\.mecha-v2\calib\kagome_h030_n12.json")
TRIS = [(0, 1, 2), (3, 4, 5), (6, 7, 8), (9, 10, 11)]
EVEN = (('x', 'y', 'z'), ('y', 'z', 'x'), ('z', 'x', 'y'))
ODD = (('y', 'x', 'z'), ('x', 'z', 'y'), ('z', 'y', 'x'))


def flux_calib(phi=0.3, delta=0.05):
    cal = json.loads(CALIB.read_text(encoding='utf-8'))
    x = {a['id']: a['pos'][0] for a in cal['atoms']}
    for pr in cal['pairs']:
        # Landau 规范：沿 x 取相位，保证 σ⁺σ⁻ + h.c. 厄米（反向取共轭）
        ph = np.exp(1j * phi * (x[pr['j']] - x[pr['i']]))
        pr['J'] = complex(pr['J']) * ph
    # ⚠ 必须打破简并：无场时 12 点 k=4 扇区的 gap 只有 5.7e-5，ED 与 DMRG 会落到
    #   简并子空间的不同组合上，χ 的差可达 1e-4——那是"简并基态上单态期望值
    #   不唯一"，不是算错。加一个小的交错失谐把双重简并劈开（gap ~5e-2）。
    for i, a in enumerate(cal['atoms']):
        a['delta_mhz'] = delta * (1.0 if i % 3 == 0 else -0.5)
    return cal


def specs():
    out = []
    for t, tri in enumerate(TRIS):
        for a, b, c in EVEN + ODD:
            out.append({'label': 'chi%d_%s%s%s' % (t, a, b, c),
                        'terms': [[tri[0], a], [tri[1], b], [tri[2], c]]})
    return out


def combos(vals):
    chi = []
    for t in range(len(TRIS)):
        blk = np.asarray(vals[6 * t:6 * t + 6])
        chi.append((blk[:3].sum() - blk[3:].sum()) / 8.0)
    return chi


def main():
    cal = flux_calib(0.3)
    n = len(cal['atoms'])
    sp = normalize_pauli_strings(specs(), n)
    base = {'manybody.calib': cal, 'manybody.pauli_strings': sp,
            'manybody.charge_sector': 4}
    ed = solve_equilibrium(None, dict(base, **{'manybody.eq_backend': 'ed'}), want='gs')
    dg = solve_equilibrium(None, dict(base, **{'manybody.eq_backend': 'dmrg',
                                               'manybody.mps_order': list(range(n)),
                                               'manybody.chi_max': 128,
                                               'manybody.max_sweeps': 40}), want='gs')
    a = combos(ed['pauli_strings'])
    b = combos(dg['pauli_strings'])
    print('gap(ED) = %.3e   E0(ED) = %.12f   E0(DMRG) = %.12f'
          % (ed['gap'], ed['E0'], dg['E0']))
    print('χ(ED)   =', ['%+.9f' % v.real for v in a])
    print('χ(DMRG) =', ['%+.9f' % v.real for v in b])
    print('max|χ| = %.3e   max|Δχ| = %.3e'
          % (max(abs(v) for v in a), max(abs(x - y) for x, y in zip(a, b))))
    ok = (ed['gap'] > 1e-3                      # 非简并，χ 才有唯一值
          and max(abs(v) for v in a) > 1e-6     # 真的非零
          and max(abs(x - y) for x, y in zip(a, b)) < 1e-8)
    print('PASS' if ok else 'FAIL')
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
