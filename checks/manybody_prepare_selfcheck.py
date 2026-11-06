# -*- coding: utf-8 -*-
"""制备态初态（`permutation_sector`）与文件引用模型（`calib_file`）的独立判据。

- 置换扇区：ED 与 MPS 两条路都要给出 `<P>=±1`、权重=(1±⟨P⟩)/2、演化后仍留在扇区；
- 文件引用：与行内 calib 结果一致；越界路径/缺文件/两键并给都要明确拒绝。
"""
import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.core import protocol                                      # noqa: E402
from src.core.manybody import runspec_mb                           # noqa: E402

FAILED = []


def check(name, cond, detail=''):
    print(('  [PASS] ' if cond else '  [FAIL] ') + name + (('  ' + str(detail)) if detail and not cond else ''))
    if not cond:
        FAILED.append(name)


def chain(n, J=1.0):
    return {"atoms": [{"id": i, "pos": [float(i), 0.0, 0.0]} for i in range(n)],
            "pairs": [{"i": i, "j": i + 1, "J": J} for i in range(n - 1)]}


def mirror(n):
    return [n - 1 - i for i in range(n)]


def build(calib, psi0, solver='ed', extra=None):
    snap = {"dynamic.mode": 2, "dynamic.t_max": 0.4, "dynamic.npts": 4,
            "manybody.calib": calib, "manybody.solver": solver,
            "manybody.psi0": psi0}
    snap.update(extra or {})
    return protocol.build_run(protocol.make_cache(), snap)


def overlap_of(psi, n, permutation):
    idx = np.arange(psi.size, dtype=np.int64)
    shifts = np.arange(n - 1, -1, -1)
    bits = (idx[:, None] >> shifts) & 1
    target = bits[:, np.asarray(permutation)] @ (1 << shifts)
    permuted = np.empty_like(psi)
    permuted[target] = psi
    return float(np.vdot(psi, permuted).real)


def main():
    n = 6
    calib = chain(n)

    # ① ED：± 扇区
    for sign in (1, -1):
        run = build(calib, {"kind": "permutation_sector", "base": "rgrgrg",
                            "permutation": mirror(n), "sign": sign})
        meta = run["quench"]
        psi = np.asarray(run["psi0"], complex)
        ov = overlap_of(psi, n, mirror(n))
        check('ED sign=%+d：<P> = %+d' % (sign, sign), abs(ov - sign) < 1e-12, ov)
        check('ED sign=%+d：投影权重 = (1+P)/2 = 0.5（乘积态基）' % sign,
              abs(meta["projection_weight"] - 0.5) < 1e-12, meta["projection_weight"])
        res = protocol.evolve_run(run)
        pops = np.asarray(res["pops_atom"])
        check('ED sign=%+d：演化后仍对称（pops 前后镜像相等）' % sign,
              np.max(np.abs(pops - pops[:, ::-1])) < 1e-12)

    # ② MPS：± 扇区（乘积态 → MPS 投影）
    run_m = build(calib, {"kind": "permutation_sector", "base": "rgrgrg",
                          "permutation": mirror(n), "sign": -1},
                  solver='tdvp', extra={"manybody.chi_max": 32,
                                        "manybody.mps_order": list(range(n))})
    meta_m = run_m["quench"]
    check('MPS：prepared_by=mps 且 <P>=-1',
          meta_m.get("prepared_by") == "mps" and abs(meta_m["permutation_overlap"] + 1) < 1e-9,
          meta_m)

    # ②b MPS：显式 mps_order 必须写进初态（否则 DMRG 报 Warm-start order mismatch）
    custom = [n - 1 - i for i in range(n)]          # 反向顺序，与默认不同
    run_o = build(calib, {"kind": "permutation_sector", "base": "rgrgrg",
                          "permutation": mirror(n), "sign": 1},
                  solver='tdvp', extra={"manybody.chi_max": 16,
                                        "manybody.mps_order": custom})
    check('MPS：初态 order 跟随 manybody.mps_order',
          list(run_o["psi0"].get("order") or []) == custom,
          run_o["psi0"].get("order"))
    try:
        protocol.evolve_run(run_o)
        check('MPS：自定义 order 能直接演化（不报 order mismatch）', True)
    except ValueError as exc:
        check('MPS：自定义 order 能直接演化（不报 order mismatch）', False, str(exc))

    # ③ 拒绝路径
    for bad, why in (({"kind": "permutation_sector", "base": "rgrgrg",
                       "permutation": [1, 0, 2, 3, 4, 5], "sign": 0}, 'sign 非法'),
                     ({"kind": "permutation_sector", "base": "rgrgrg",
                       "permutation": [1, 2, 0, 3, 4, 5], "sign": 1}, '非对合'),
                     ({"kind": "permutation_sector", "base": "rgrgrg",
                       "permutation": mirror(4), "sign": 1}, '长度不符')):
        try:
            build(calib, bad)
            check('拒绝：%s' % why, False)
        except ValueError:
            check('拒绝：%s' % why, True)

    # ④ 构造器（文件 → calib）：与行内 calib 得到同一基态能量
    tmp = Path(tempfile.mkdtemp())
    os.environ["EL_CALIB_DIR"] = str(tmp)
    (tmp / "m.json").write_text(json.dumps(calib), encoding="utf-8")
    ref = protocol.evolve_run(build(calib, None, extra={"manybody.want": "gs"}))
    built = runspec_mb.build_calib_from_file("m.json")
    check('构造器：文件 → calib 与行内 calib 指纹一致',
          runspec_mb.calib_fingerprint(built) == runspec_mb.calib_fingerprint(
              runspec_mb.validate_calib(calib)),
          (runspec_mb.calib_fingerprint(built),
           runspec_mb.calib_fingerprint(runspec_mb.validate_calib(calib))))
    run_f = protocol.build_run(protocol.make_cache(),
                               {"dynamic.mode": 2, "manybody.calib": built,
                                "manybody.solver": "ed", "manybody.want": "gs"})
    res_f = protocol.evolve_run(run_f)
    check('构造器：文件模型的 E0 与行内一致',
          abs(float(res_f["E0"]) - float(ref["E0"])) < 1e-9,
          (res_f["E0"], ref["E0"]))

    # ⑤ calib 只认行内形态 + 文件读取的拒绝路径
    for bad, why in (({"file": "m.json"}, 'calib 里写 file'),
                     ({"arc": {"B_gauss": 1.0}}, 'calib 里写 arc')):
        try:
            runspec_mb.validate_calib(bad)
            check('拒绝：%s' % why, False)
        except ValueError:
            check('拒绝：%s' % why, True)
    for bad_name, why in (("../evil.json", '越界'), ("/etc/passwd", '绝对越界'), ("nope.json", '不存在')):
        try:
            runspec_mb.load_calib_file(bad_name)
            check('拒绝：构造器文件 %s' % why, False)
        except ValueError:
            check('拒绝：构造器文件 %s' % why, True)

    print('[manybody_prepare] %s  FAIL=%d' % ('PASS' if not FAILED else 'FAIL', len(FAILED)))
    return 0 if not FAILED else 1


if __name__ == '__main__':
    raise SystemExit(main())
