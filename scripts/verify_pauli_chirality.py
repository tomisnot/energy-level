# -*- coding: utf-8 -*-
"""验收：用 EL 新增的 `manybody.pauli_strings` 复算项目里独立算过的手性。

对照值是 `results/el_ramp_h030_n12_metrics.json`——那份数由
`src/el_artifact_analysis.py` 从 artifact 的 `psi_amp` 用**独立实现**算出来
（稀疏 Pauli 积 + 显式 1/8 组合）。现在换成引擎自己算，两者必须逐位吻合。

用法：python scripts/verify_pauli_chirality.py [--n24]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.core.manybody.engine_ed import evolve_manybody           # noqa: E402
from src.core.manybody.observables import normalize_pauli_strings  # noqa: E402

EL_CALIB = Path(r"D:\code-nosync\energy-level\.mecha-v2\calib")
PROJ = Path(r"D:\desktop\数值尝试\05_Rb_呼吸Kagome_CSL_实验可行性")

TRIS_12 = [(0, 1, 2), (3, 4, 5), (6, 7, 8), (9, 10, 11)]
# χ = (1/8) Σ_{abc} ε_abc σᵃσᵇσᶜ：3 个偶置换 +、3 个奇置换 −
EVEN = (('x', 'y', 'z'), ('y', 'z', 'x'), ('z', 'x', 'y'))
ODD = (('y', 'x', 'z'), ('x', 'z', 'y'), ('z', 'y', 'x'))


def chirality_specs(tris):
    specs = []
    for t, tri in enumerate(tris):
        for a, b, c in EVEN + ODD:
            specs.append({'label': 'chi%d_%s%s%s' % (t, a, b, c),
                          'terms': [[tri[0], a], [tri[1], b], [tri[2], c]]})
    return specs


def combine(vals, n_tri):
    """6 条/三角 → χ_project = (Σ偶 − Σ奇)/8。"""
    out = []
    for t in range(n_tri):
        block = np.asarray(vals[t * 6:(t + 1) * 6])
        out.append(float((block[:3].sum() - block[3:].sum()).real / 8.0))
    return out


def ramp_segments(n, delta0=7.7, tau=779.0, total=7790.0, nseg=12, pattern=(0, 1, 1)):
    segs = []
    for k in range(nseg):
        t0 = total * k / nseg
        f = np.exp(-t0 / tau)
        segs.append({'t0': t0,
                     'delta': [float(delta0 * pattern[i % 3] * f) for i in range(n)]})
    return segs


def main():
    ap = argparse.ArgumentParser()
    # 对照值在 JSON 里按 round(·, 8) 存，故容差取 1e-8（不是引擎精度）
    ap.add_argument('--tol', type=float, default=1e-8)
    args = ap.parse_args()

    calib = json.loads((EL_CALIB / 'kagome_h030_n12.json').read_text(encoding='utf-8'))
    n = len(calib['atoms'])
    specs = normalize_pauli_strings(chirality_specs(TRIS_12), n)
    t = np.linspace(0.0, 7790.0, 13)
    res = evolve_manybody(calib, psi0='rggrggrggrgg', t_grid=t,
                          Om=np.zeros(n, complex), delta=np.zeros(n),
                          terms=['I2'], segments=ramp_segments(n),
                          want_observables=True, pauli_strings=specs)
    series = np.asarray(res['observables']['pauli_strings'])
    got = combine(series[-1], len(TRIS_12))

    ref = json.loads((PROJ / 'results' / 'el_ramp_h030_n12_metrics.json').read_text(encoding='utf-8'))
    ref_labels = list(ref['chirality_up_triangles'].keys())
    want = [ref['chirality_up_triangles'][k] for k in ref_labels]
    # 项目脚本里三角顺序与这里的 (0,1,2),(3,4,5)… 一致；逐项比
    diff = float(np.max(np.abs(np.array(got) - np.array(want))))
    print('engine  χ =', [round(x, 10) for x in got])
    print('独立算  χ =', [round(x, 10) for x in want])
    print('max|Δ| = %.3e' % diff)
    ok = diff <= args.tol
    print('PASS' if ok else 'FAIL')
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
