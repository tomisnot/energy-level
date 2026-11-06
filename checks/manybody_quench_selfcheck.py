# -*- coding: utf-8 -*-
"""淬火初态（`manybody.psi0={"kind":"ground_of",…}`）的独立判据。

判据只用**精确时间演化**做对照：初态取模型 A 的基态，在模型 B 下演化，比较
逐站点占据。两条准备路都要过：
- ED：初态 → 全空间向量；
- MPS：初态 → DMRG 原生态（TDVP 演化）。
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scipy.linalg import expm                                      # noqa: E402
from src.core import protocol                                      # noqa: E402
from src.core.evolve_util import CONV                              # noqa: E402
from src.core.manybody.assemble import build_H_sparse              # noqa: E402

FAILED = []


def check(name, cond, detail=''):
    print(('  [PASS] ' if cond else '  [FAIL] ') + name + (('  ' + str(detail)) if detail and not cond else ''))
    if not cond:
        FAILED.append(name)


def chain(n, J):
    return {"atoms": [{"id": i, "pos": [float(i), 0.0, 0.0]} for i in range(n)],
            "pairs": [{"i": i, "j": i + 1, "J": J} for i in range(n - 1)]}


def exact_pops(n, calA, calB, taus):
    HA = build_H_sparse(calA).toarray()
    HB = build_H_sparse(calB).toarray()
    w, V = np.linalg.eigh(HA)
    psi0 = V[:, 0].astype(complex)
    bits = np.array([[(s >> (n - 1 - i)) & 1 for s in range(1 << n)] for i in range(n)])
    out = []
    for tau in taus:
        p = np.abs(expm(-1j * CONV * HB * tau) @ psi0) ** 2
        out.append(bits @ p)
    return np.asarray(out)          # (n_t, n)


def run_quench(n, solver, extra):
    calA, calB = chain(n, 1.0), chain(n, 2.0)
    snap = {"dynamic.mode": 2, "dynamic.t_max": 1.0, "dynamic.npts": 11,
            "manybody.calib": calB, "manybody.solver": solver,
            "manybody.psi0": {"kind": "ground_of", "calib": calA}}
    snap.update(extra)
    run = protocol.build_run(protocol.make_cache(), snap)
    res = protocol.evolve_run(run)
    return run, res, calA, calB


def main():
    # ① ED：初态被准备成向量，占据与精确演化一致
    run, res, calA, calB = run_quench(4, "ed", {})
    check('ED：psi0 落地为全空间向量', isinstance(run["psi0"], np.ndarray)
          and run["psi0"].size == 16, type(run["psi0"]))
    check('ED：quench 元数据记录初态模型能量',
          (run.get("quench") or {}).get("prepared_by") == "ed"
          and abs((run["quench"] or {}).get("E0", 0) + 2.23606797749979) < 1e-9,
          run.get("quench"))
    taus = np.asarray(res["t"])
    err = float(np.max(np.abs(np.asarray(res["pops_atom"]) - exact_pops(4, calA, calB, taus))))
    check('ED：逐站点占据与精确演化一致（<1e-10）', err < 1e-10, err)

    # ② MPS：DMRG 准备初态 + TDVP 演化，与 ED 精确解一致
    n = 6
    run_m, res_m, cA, cB = run_quench(n, "tdvp",
                                      {"manybody.chi_max": 32, "manybody.max_sweeps": 10,
                                       "manybody.mps_order": list(range(n))})
    check('MPS：psi0 落地为原生态（kind=mps）',
          isinstance(run_m["psi0"], dict) and run_m["psi0"].get("kind") == "mps"
          and (run_m.get("quench") or {}).get("prepared_by") == "dmrg",
          run_m.get("quench"))
    err_m = float(np.max(np.abs(np.asarray(res_m["pops_atom"])
                                - exact_pops(n, cA, cB, np.asarray(res_m["t"])))))
    check('MPS：TDVP 占据与精确演化一致（<1e-6）', err_m < 1e-6, err_m)

    # ③ 同一条初态也能以**态矢**直接给（Python API 路径，脉冲初态用）
    calA, calB = chain(4, 1.0), chain(4, 2.0)
    HA = build_H_sparse(calA).toarray()
    w, V = np.linalg.eigh(HA)
    snap_v = {"dynamic.mode": 2, "dynamic.t_max": 1.0, "dynamic.npts": 11,
              "manybody.calib": calB, "manybody.solver": "ed",
              "manybody.psi0": V[:, 0].astype(complex)}
    res_v = protocol.evolve_run(protocol.build_run(protocol.make_cache(), snap_v))
    check('态矢初态：与 ground_of 结果一致',
          float(np.max(np.abs(np.asarray(res_v["pops_atom"])
                              - np.asarray(res["pops_atom"])))) < 1e-12)

    # ④ 拒绝路径：初态模型与演化模型的原子数不一致
    bad = {"dynamic.mode": 2, "dynamic.t_max": 1.0, "dynamic.npts": 3,
           "manybody.calib": chain(4, 2.0), "manybody.solver": "ed",
           "manybody.psi0": {"kind": "ground_of", "calib": chain(5, 1.0)}}
    try:
        protocol.build_run(protocol.make_cache(), bad)
        check('拒绝：原子数不一致', False)
    except ValueError as exc:
        check('拒绝：原子数不一致', '原子数' in str(exc), str(exc))

    # ④ 拒绝路径：缺 calib
    bad2 = dict(bad, **{"manybody.psi0": {"kind": "ground_of"}})
    try:
        protocol.build_run(protocol.make_cache(), bad2)
        check('拒绝：缺 calib', False)
    except ValueError as exc:
        check('拒绝：缺 calib', 'calib' in str(exc), str(exc))

    print('[manybody_quench] %s  FAIL=%d' % ('PASS' if not FAILED else 'FAIL', len(FAILED)))
    return 0 if not FAILED else 1


if __name__ == '__main__':
    raise SystemExit(main())
