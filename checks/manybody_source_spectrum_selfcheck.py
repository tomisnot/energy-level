# -*- coding: utf-8 -*-
"""源态 Lanczos 谱（want=spectrum）的独立判据。

判据只用**精确对角化**做对照，不引用被测实现：
- 4 站点 I2 链、固定激发数 2 的 6 维子空间；
- 源算符取对角（n₀+n₁），于是精确谱 = |⟨n|D|0⟩|² 落在 E_n−E0 上；
- 核对求积节点/权重、展宽谱峰位、和规则、以及 full_sparse 的扇区投影泄漏。
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.core import protocol                                      # noqa: E402
from src.core.manybody.basis import basis_bits                     # noqa: E402
from src.core.manybody.matrix_free import (                        # noqa: E402
    number_sector_basis, hamiltonian_linear_operator)

FAILED = []


def check(name, cond, detail=''):
    print(('  [PASS] ' if cond else '  [FAIL] ') + name + (('  ' + str(detail)) if detail and not cond else ''))
    if not cond:
        FAILED.append(name)


def exact(n=4, sector=2, eta=0.02):
    calib = {"atoms": [{"id": i, "pos": [float(i), 0.0, 0.0]} for i in range(n)],
             "pairs": [{"i": i, "j": i + 1, "J": 1.0} for i in range(n - 1)]}
    idx = number_sector_basis(n, sector)
    bits = basis_bits(n, idx)
    diag = bits[0].astype(float) + bits[1].astype(float)
    H = hamiltonian_linear_operator(calib, None, None, None, ["I2"], basis_indices=idx)
    Hd = np.column_stack([H @ np.eye(1, len(idx), j).ravel() for j in range(len(idx))])
    w, V = np.linalg.eigh(Hd)
    psi0 = V[:, 0]
    phi = diag.astype(complex) * psi0
    nodes, weights = [], []
    for m in range(len(w)):
        amp = np.vdot(V[:, m], phi)
        if abs(amp) ** 2 > 1e-12:
            nodes.append(float(w[m] - w[0]))
            weights.append(float(abs(amp) ** 2))
    order = np.argsort(nodes)
    return calib, idx, diag, phi, np.asarray(nodes)[order], np.asarray(weights)[order]


def run(spec_patch):
    snap = {"dynamic.mode": 2, "manybody.want": "spectrum",
            "manybody.eq_backend": "ed", "manybody.charge_sector": 2,
            "manybody.spectrum": {"eta": 0.02, "n_krylov": 6, "n_omega": 1201,
                                  "omega_min": -0.4, "omega_max": 4.0}}
    snap.update(spec_patch)
    cache = protocol.make_cache()
    return protocol.evolve_run(protocol.build_run(cache, snap))


def main():
    calib, idx, diag, phi, nodes_x, weights_x = exact()
    base = {"manybody.calib": calib}

    # ① 对角源：节点/权重逐项对精确值
    res = run(dict(base, **{"manybody.source": {"kind": "diagonal",
                                                 "values": diag.tolist()}}))
    nodes = np.asarray(res["evals"], float)
    weights = np.asarray(res["spectrum_weights"], float)
    check('对角源：求积节点 = 精确激发能',
          nodes.shape == nodes_x.shape and np.max(np.abs(nodes - nodes_x)) < 1e-9,
          (nodes, nodes_x))
    check('对角源：求积权重 = 精确权重',
          weights.shape == weights_x.shape and np.max(np.abs(weights - weights_x)) < 1e-9,
          (weights, weights_x))
    check('和规则：Σw = ‖φ‖²',
          abs(weights.sum() - res["source_norm"]) < 1e-10,
          (weights.sum(), res["source_norm"]))

    # ② 展宽谱：峰位与精确激发能的差应在 η 之内
    omega = np.asarray(res["energies"], float)
    s = np.asarray(res["spectral"], float)
    peaks = [omega[i] for i in range(1, len(s) - 1)
             if s[i] > s[i - 1] and s[i] >= s[i + 1] and s[i] > 0.05 * s.max()]
    hit = all(min(abs(p - x) for x in nodes_x) < 2.0 * res["eta"] for p in peaks) if peaks else False
    check('展宽谱峰位落在精确激发能附近（≤2η）', bool(peaks) and hit, peaks)

    # ③ 数值/契约
    check('契约：descriptors 含 spectral/source_norm（权重留在结果里不绘）',
          {'spectral', 'source_norm'} <=
          set((res.get('data_schema') or {}).get('quantities', {})) and
          'spectrum_weights' in res and
          'spectrum_weights' not in (res.get('data_schema') or {}).get('quantities', {}))
    check('诊断：Krylov 记录与积分比可用',
          (res.get('krylov') or {}).get('n', 0) >= 1 and
          (res.get('error_report') or {}).get('integral_ratio') is not None)
    check('扇区：dim_sub = C(4,2) = 6', res.get('dim_sub') == 6 and len(idx) == 6)

    # ④ vector 源 == diagonal 作用后的 φ
    res_v = run(dict(base, **{"manybody.source": {"kind": "vector",
                                                   "values": phi.real.tolist()}}))
    check('vector 源：权重与对角源一致',
          np.max(np.abs(np.asarray(res_v["spectrum_weights"], float) - weights_x)) < 1e-9)

    # ④b psi0 作为 |0⟩：source 省略时默认走 psi0，和显式 vector 源一致
    psi_vec = np.zeros(1 << 4, dtype=complex)
    psi_vec[idx] = phi
    res_p = run(dict(base, **{"manybody.psi0": psi_vec.tolist()}))
    wv = np.asarray(res_v["spectrum_weights"], float) / res_v["source_norm"]
    wp = np.asarray(res_p["spectrum_weights"], float) / res_p["source_norm"]
    check('psi0 作为默认源：归一化权重 = 显式 vector 源',
          np.max(np.abs(wp - wv)) < 1e-9)
    check('psi0 作为源：source_kind 如实标注',
          (res_p.get('error_report') or {}).get('source_kind') == 'psi0',
          (res_p.get('error_report') or {}).get('source_kind'))

    # ⑤ full_sparse：保扇区的对角算符 → 泄漏 0，结果一致
    n_full = 1 << 4
    rows = list(range(n_full))
    vals = []
    for s_idx in range(n_full):
        bits = [(s_idx >> (4 - 1 - i)) & 1 for i in range(4)]
        vals.append(float(bits[0] + bits[1]))
    res_f = run(dict(base, **{"manybody.source": {
        "kind": "full_sparse", "rows": rows, "cols": rows, "values": vals, "dim": n_full}}))
    leak = (res_f.get('error_report') or {}).get('projected_leak')
    check('full_sparse：保扇区算符的投影泄漏 ≈ 0', leak is not None and abs(leak) < 1e-12, leak)
    check('full_sparse：与对角源结果一致',
          np.max(np.abs(np.asarray(res_f["spectrum_weights"], float) - weights_x)) < 1e-9)

    # ⑥ 截断：n_krylov=2 只给 2 个节点（诊断可见）
    res_k = run(dict(base, **{"manybody.source": {"kind": "diagonal", "values": diag.tolist()},
                              "manybody.spectrum": {"eta": 0.2, "n_krylov": 2,
                                                    "n_omega": 201}}))
    check('截断：n_krylov=2 ⇒ 2 个求积节点，krylov.n 如实',
          len(res_k["evals"]) == 2 and res_k["krylov"]["n"] == 2)

    # ⑦ 拒绝路径
    for bad, why in (({"kind": "nope"}, '未知 kind'),
                     ({"kind": "diagonal", "values": [1.0]}, '长度不匹配'),
                     ({"kind": "sparse", "rows": [0], "cols": [1], "values": [1.0]}, '形状不匹配')):
        try:
            run(dict(base, **{"manybody.source": bad}))
            check('拒绝路径：' + why, False)
        except ValueError:
            check('拒绝路径：' + why, True)

    print('[manybody_source_spectrum] %s  FAIL=%d' %
          ('PASS' if not FAILED else 'FAIL', len(FAILED)))
    return 0 if not FAILED else 1


if __name__ == '__main__':
    raise SystemExit(main())
