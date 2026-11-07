"""三个 CPU TDVP 后端的同条件对拍（墙钟 + 逐原子布居）。

用法：python scripts/bench_tdvp_variants_local.py [--t-max 20] [--dt 2] [--chi 64]
"""
import argparse, json, sys, time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.core.manybody.engine_mps import evolve_mps          # noqa: E402
from src.core.manybody.mpo_tdvp import evolve_mpo_tdvp      # noqa: E402
from src.core.manybody.tdvp_dense import evolve_dense       # noqa: E402
from src.core.manybody.runspec_mb import drive_from_calib   # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--calib', default='checks/fixtures/kagome_n12.json')
    ap.add_argument('--t-max', type=float, default=20.0)
    ap.add_argument('--dt', type=float, default=2.0)
    ap.add_argument('--chi', type=int, default=64)
    a = ap.parse_args()

    calib = json.load(open(ROOT / a.calib, encoding='utf-8'))
    n = len(calib['atoms'])
    Om, de, ga = drive_from_calib(calib)
    t = np.arange(0.0, a.t_max + 1e-12, a.dt)
    psi0 = 'r' + 'g' * (n - 1)
    kw = dict(psi0=psi0, t_grid=t, Om=Om, delta=de, terms=['I2'],
              chi_max=a.chi, truncerr=1e-10, max_step_ns=a.dt)

    out = {}
    for name, fn in (('tdvp', lambda: evolve_mps(calib, algorithm='tdvp', **kw)),
                     ('tdvp-mpo', lambda: evolve_mpo_tdvp(calib, **kw)),
                     ('tdvp-dense', lambda: evolve_dense(calib, **kw))):
        try:
            t0 = time.perf_counter()
            r = fn()
            wall = time.perf_counter() - t0
            pops = np.asarray(r.get('pops_atom'), float)
            out[name] = pops
            print('%-11s wall=%7.2f s   steps=%3d  pops_sum[last]=%.6f  max_chi=%s'
                  % (name, wall, len(t) - 1, pops[-1].sum(), r.get('error_report', {}).get('max_chi', '?')))
        except Exception as exc:                                  # noqa: BLE001
            print('%-11s ERROR %s: %s' % (name, type(exc).__name__, str(exc)[:160]))

    if len(out) > 1:
        ref_name = 'tdvp' if 'tdvp' in out else list(out)[0]
        ref = out[ref_name][-1]
        for name, pops in out.items():
            if name == ref_name:
                continue
            print('max|Δpop| %-11s vs %-11s = %.3e' % (name, ref_name,
                                                       float(np.max(np.abs(pops[-1] - ref)))))


if __name__ == '__main__':
    main()
