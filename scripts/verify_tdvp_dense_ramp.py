"""tdvp-dense 新接口验收：分段撤场 + 三体 Pauli 串，与 ED 逐点对拍。

用法：python scripts/verify_tdvp_dense_ramp.py
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.core.manybody.engine_ed import evolve_manybody          # noqa: E402
from src.core.manybody.tdvp_dense import evolve_tdvp_dense      # noqa: E402
from src.core.manybody.runspec_mb import drive_from_calib       # noqa: E402

CALIB = ROOT / 'checks' / 'fixtures' / 'kagome_n12.json'
TRIS = [(0, 1, 2), (3, 4, 5), (6, 7, 8), (9, 10, 11)]
EVEN = (('x', 'y', 'z'), ('y', 'z', 'x'), ('z', 'x', 'y'))
ODD = (('y', 'x', 'z'), ('x', 'z', 'y'), ('z', 'y', 'x'))
J1, TAU, TOT, NSEG = 0.77, 779.0, 7790.0, 12


def specs():
    return [{'label': 'chi%d_%s%s%s' % (t, a, b, c),
             'terms': [[tri[0], a], [tri[1], b], [tri[2], c]]}
            for t, tri in enumerate(TRIS) for a, b, c in EVEN + ODD]


def segments(n):
    # Δ(t) = 2·Δ0·exp(-t/τ) 作用在 η=(+1,-1,-1) 子格上（与 §gap 扫描同一定义）
    eta = []
    for u in range(2):
        for v in range(2):
            sig = 1 if v % 2 == 0 else -1
            eta += [sig, -sig, -sig]
    eta = np.array(eta)
    out = []
    for k in range(NSEG):
        t0 = TOT * k / NSEG
        d = 2.0 * 7.7 * np.exp(-t0 / TAU)
        out.append({'t0': t0, 'delta': [float(d * e) for e in eta]})
    return out


def chi(vals, n_tri):
    r = []
    for t in range(n_tri):
        b = np.asarray(vals[6 * t:6 * t + 6])
        r.append((b[:3].sum() - b[3:].sum()) / 8.0)
    return r


def main():
    calib = json.loads(CALIB.read_text(encoding='utf-8'))
    n = len(calib['atoms'])
    Om, de, ga = drive_from_calib(calib)
    t = np.linspace(0.0, TOT, NSEG + 1)
    sp = specs()
    segs = segments(n)
    kw = dict(psi0='rggrggrggrgg', t_grid=t, Om=Om, delta=de, terms=['I2'],
              segments=segs, pauli_strings=sp, want_observables=True)
    dense_kw = dict(kw, chi_max=128, truncerr=1e-10, max_step_ns=10.0,
                    mps_order=[0, 1, 2, 6, 7, 8, 3, 4, 5, 9, 10, 11])

    t0 = time.perf_counter()
    ed = evolve_manybody(calib, **kw)
    t_ed = time.perf_counter() - t0

    t0 = time.perf_counter()
    de_res = evolve_tdvp_dense(calib, **dense_kw)
    t_de = time.perf_counter() - t0

    a = np.asarray(ed['observables']['pauli_strings'])
    b = np.asarray(de_res['observables']['pauli_strings'])
    print('ED        wall=%.2f s  shape=%s' % (t_ed, a.shape))
    print('tdvp-dense wall=%.2f s  shape=%s   (%.1fx faster)'
          % (t_de, b.shape, t_ed / t_de))
    print('max|Δ pauli| = %.3e   (max|ED| = %.3e)'
          % (np.max(np.abs(a - b)), np.max(np.abs(a))))
    for k in (0, len(t) // 2, len(t) - 1):
        print('  t=%7.1f ns  χ_ED=%s' % (t[k], ['%+.6e' % x.real for x in chi(a[k], 4)]))
        print('              χ_TD=%s' % ['%+.6e' % x.real for x in chi(b[k], 4)])
    ok = np.max(np.abs(a - b)) < 1e-5
    print('PASS' if ok else 'FAIL')
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
