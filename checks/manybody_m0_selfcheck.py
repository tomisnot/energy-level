# -*- coding: utf-8 -*-
"""M0 自检：4 维对空间 V_dd 全精确标定 + 装配（锚 A0a–A0f）。

运行：python checks/manybody_m0_selfcheck.py
（需 ARC 数据；首跑建表可能 10–30 s，之后走缓存。目标 <1 min。）

A0a 一阶 ≡ static_pair.vdd_sparse（同基组同几何，逐元素 <1e-12；
    注意 vdd_sparse 存储序=⟨q|V|p⟩ 转置序，H4 为标准序 ⟨p|V|q⟩）
A0b **φ 相位协变**（用户点名）：V4[p,q](θ,φ) = e^{i(M_p−M_q)φ}·V4[p,q](θ,0)
    ——偶极相互作用对方向角 φ 的相位依赖全程复数保留
A0c ED 装配约定 ≡ calibrate_pair（J/A/V 矩阵元逐位）
A0d 模型卡 JSON 往返逐位一致
A0e θ=0 总 m 守恒：⟨gg|V|rr⟩ ∝ δ_{2(m_r−m_g),0}（m 差不为 0 ⇒ 严格 0）
A0f 二阶（gg 支 vdW 位移）≡ static_pair.vdw_shift 全对角化提取
    （gg 远离中间态，二阶微扰精确到 O(V³/Δ²)；J/A 支与 m 简并伙伴态近简并，
     downfolding 不适用——guard 剔除数如实上报，这是模型适用域的诚实边界）
A0g 项开关：关 I2/I3 ⇒ 对应 raw 保留、生效值为 0，dropped_amp 报告非零
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from arc import Rubidium87

from src.core import core
from src.core import static_pair as sp
from src.manybody.model import ManyBodyModel
from src.manybody import assemble as asm

cache = core.ArcCache(Rubidium87())
G = (60, 0, 0.5, 0.5)          # 60S_1/2 m=+1/2
R_ = (60, 1, 1.5, 1.5)         # 60P_3/2 m=+3/2
GROUPS = [dict(n=G[0], l=G[1], j=G[2]), dict(n=R_[0], l=R_[1], j=R_[2])]
_results = []


def _check(name, ok, detail=''):
    _results.append((name, ok))
    print(f'  [{"PASS" if ok else "FAIL"}] {name}  {detail}')


_SINGLE = sp.single_atom_basis(cache, GROUPS, 0.0, 1)
_E = np.array([s['E_mhz'] for s in _SINGLE], float)
_D = {q: sp._dipole_sparse(cache, _SINGLE, q) for q in (-1, 0, 1)}


def _basis_and_D():
    return _SINGLE, _E, _D


def _pair4(single, E, g=G, r=R_):
    ig, ir = sp.find_sublevel(single, *g), sp.find_sublevel(single, *r)
    N = len(single)
    return ((ig, ig), (ir, ig), (ig, ir), (ir, ir)), ig, ir


def test_A0a_first_order():
    print('== A0a 一阶 ≡ vdd_sparse ==')
    single, E, D = _basis_and_D()
    pair4, ig, ir = _pair4(single, E)
    N = len(single)
    R, th, ph = 12.0, 30.0, 37.0
    res = asm.calibrate_pair(cache, single, D, E, pair4, R, th, ph,
                             second_order=False)
    V = sp.vdd_sparse(cache, single, R, th, ph)
    k = np.array([a * N + b for a, b in pair4])
    V4 = np.asarray(V[np.ix_(k, k)].todense()).conj()   # 转置序→标准序
    d = np.abs(res['delta1'] - V4).max()
    _check('A0a 一阶 4×4 ≡ vdd_sparse', d < 1e-12, f'max|Δ|={d:.2e}')


def test_A0b_phi_covariance():
    print('== A0b φ 相位协变（复数保留） ==')
    single, E, D = _basis_and_D()
    pair4, ig, ir = _pair4(single, E)
    Mg, Mr = G[3], R_[3]
    M = np.array([Mg + Mg, Mr + Mg, Mg + Mr, Mr + Mr])   # 各对态总 m
    for th, ph0 in ((0.0, 0.523), (30.0, 37.0), (90.0, -1.2)):
        r0 = asm.calibrate_pair(cache, single, D, E, pair4, 12.0, th, 0.0,
                                second_order=True)
        r1 = asm.calibrate_pair(cache, single, D, E, pair4, 12.0, th, ph0,
                                second_order=True)
        ph = np.deg2rad(ph0)
        phase = np.exp(1j * (M[:, None] - M[None, :]) * ph)
        pred = r0['H4'] * phase
        d = np.abs(r1['H4'] - pred).max()
        nz = np.abs(r0['H4']).max()
        # 双激发元 ⟨rr|V|gg⟩（ΔM=2）应真的带 e^{+2iφ}（非平凡相位验证）
        a0 = r0['H4'][3, 0]
        a1 = r1['H4'][3, 0]
        phA = (abs(a1 - a0 * np.exp(2j * ph)) < 1e-12 * max(abs(a0), 1e-30)
               if abs(a0) > 1e-15 else None)
        # 交换元 ⟨rg|V|gr⟩（ΔM=0）应 φ-不变（模不变、相位不绕）
        j0, j1 = r0['H4'][2, 1], r1['H4'][2, 1]
        phJ = abs(j1 - j0) < 1e-9 * max(abs(j0), 1.0)
        _check(f'A0b (θ={th:g},φ={ph0:g}) 全 4×4 协变', d < 1e-9 * max(nz, 1.0),
               f'max|Δ|={d:.2e}；A 元 e^2iφ: '
               + ('N/A(=0)' if phA is None else ('✓' if phA else '✗'))
               + f'；J 元 ΔM=0: {"✓" if phJ else "✗"}')


def test_A0c_ed_conventions():
    print('== A0c ED 装配约定 ==')
    mdl = ManyBodyModel(terms=('I1', 'I2', 'I3'), n_window=1)
    mdl.add_atom(G, R_, (0.0, 0.0, 0.0))
    mdl.add_atom(G, R_, (6.0, 9.0, 4.0))              # 一般 3D 方向 ⇒ A 复
    cal = asm.calibrate(cache, mdl, second_order=True)
    p = cal['pairs'][0]
    H = asm.build_H_ed(cal)                          # δ=0, Ω=0
    # 基序 bit i ↔ 原子 i（MSB）：|gg⟩=00, |g_i r_j⟩=01, |r_i g_j⟩=10, |rr⟩=11
    # 一般 φ 下 A=⟨rr|V|gg⟩（ΔM=2）携带复相位（A0b 的 e^{2iφ} 缠绕）；
    # J（ΔM=0）一阶恒实（T_{q,−q} 实对称）——两者都要装配正确。
    assert abs(p['raw_A'].imag) > 1e-3 * abs(p['raw_A']), \
        '一般几何下 A 应为复数（φ 相位依赖），否则 A0c 失去区分力'
    _check('A0c A 元携带复相位（一般 φ）', True,
           f"A={p['raw_A']:.4g}")
    _check('A0c H[|r_i g_j⟩,|g_i r_j⟩] ≡ raw_J',
           abs(H[2, 1] - p['raw_J']) < 1e-12,
           f"J={p['raw_J']:.4g}")
    _check('A0c 共轭元 H[1,2] ≡ conj(raw_J)',
           abs(H[1, 2] - np.conj(p['raw_J'])) < 1e-12)
    _check('A0c ⟨rr|H|gg⟩ ≡ raw_A',
           abs(H[3, 0] - p['raw_A']) < 1e-12,
           f"A={p['raw_A']:.4g}")
    _check('A0c H[rr,rr] ≡ raw_V（δ=0）',
           abs(H[3, 3] - p['raw_V']) < 1e-12,
           f"V={p['raw_V']:.4g} MHz")
    _check('A0c 厄米', np.abs(H - H.conj().T).max() < 1e-12)
    # Ω≠0 的驱动两支都要填（回归：曾因只填 |r⟩⟨g| 漏 h.c. 破坏厄米）
    H2 = asm.build_H_ed(cal, Om=[2 * np.pi * 1.0] * 2)
    _check('A0c Ω≠0 厄米（驱动含 h.c.）',
           np.abs(H2 - H2.conj().T).max() < 1e-12,
           f'max|Δ|={np.abs(H2 - H2.conj().T).max():.1e}')
    return cal


def test_A0d_roundtrip(cal):
    print('== A0d JSON 往返 ==')
    import json
    d = asm.to_json_dict(cal)
    s = json.dumps(d, ensure_ascii=False)
    cal2 = json.loads(s)
    ok = True
    for p1, p2 in zip(cal['pairs'], cal2['pairs']):
        ok &= abs(complex(*p2['raw_J']) - p1['raw_J']) == 0.0
        ok &= abs(complex(*p2['raw_A']) - p1['raw_A']) == 0.0
        ok &= p2['raw_V'] == p1['raw_V']
    _check('A0d 标定 JSON 往返复数无损', ok)
    m0 = ManyBodyModel('ising', ('I1', 'I2', 'I3'), B_gauss=1.0,
                       r_cut_um=25.0)
    m0.add_atom(G, R_, (0.0, 1.0, 2.0))
    m0.add_atom(G, R_, (3.0, 4.0, 5.0))
    d0 = m0.to_dict()
    m2 = ManyBodyModel.from_dict(d0)
    _check('A0d 模型卡 dict 往返', m2.to_dict() == d0)


def test_A0e_m_conservation():
    print('== A0e θ=0 总 m 守恒 ==')
    single, E, D = _basis_and_D()
    pair4, ig, ir = _pair4(single, E)
    res = asm.calibrate_pair(cache, single, D, E, pair4, 12.0, 0.0, 0.0,
                             second_order=True)
    # Δm = m_r − m_g = +1 ⇒ A 元（ΔM=2）在 θ=0 严格 0；
    # J 元（ΔM=0）允许非零
    a = abs(res['H4'][3, 0])
    j = abs(res['H4'][1, 2])
    _check('A0e θ=0 ⇒ A=0（ΔM=2 禁戒）', a < 1e-12, f'|A|={a:.2e}')
    _check('A0e θ=0 ⇒ J 可非零（ΔM=0 允许）', j > 0, f'|J|={j:.4g} MHz')


def test_A0f_second_order(cal):
    print('== A0f 二阶（gg 支 vdW）≡ vdw_shift 全对角化提取 ==')
    single, E, D = _basis_and_D()
    pair4, ig, ir = _pair4(single, E)
    R, th, ph = 30.0, 45.0, 20.0                      # 远距弱耦合
    res = asm.calibrate_pair(cache, single, D, E, pair4, R, th, ph,
                             second_order=True)
    # gg=|60S⊗60S⟩：一阶 ⟨gg|V|gg⟩=0（确定宇称无永久偶极）⇒ H4[0,0] 纯二阶
    sh_df = float(res['H4'][0, 0].real)
    sh_exact, _ = sp.vdw_shift(cache, GROUPS, R, th, ph, n_window=1,
                               ref_ket=G)
    d = abs(sh_df - sh_exact)
    # 二阶 Löwdin vs 全对角化：差 = 四阶 PT 截断（密集中间态流形），~2% 量级
    _check('A0f gg 支 downfolding ≡ 全对角化',
           d < max(1e-6, 5e-2 * abs(sh_exact)),
           f'downfold={sh_df:.6e}  exact={sh_exact:.6e}  Δ={d:.2e} MHz')
    d2 = np.abs(res['delta2']).max()
    _check('A0f 二阶修正非零（C6 涌现）', d2 > 0, f'|δ2|max={d2:.3g} MHz')
    info = res['info']
    _check('A0f guard 计数如实上报',
           'n_guarded' in info and 'n_intermediate' in info,
           f"n_inter={info['n_intermediate']} "
           f"n_guarded={info['n_guarded']} "
           f"min_gap={info['min_gap_mhz']:.4g}")


def test_A0g_term_switch():
    print('== A0g 项开关与丢项报告 ==')
    mdl = ManyBodyModel(terms=('I1',), n_window=1)     # 只开 I1
    mdl.add_atom(G, R_, (0.0, 0.0, 0.0))
    mdl.add_atom(G, R_, (12.0, 30.0, 37.0))
    cal = asm.calibrate(cache, mdl)
    p = cal['pairs'][0]
    _check('A0g 关 I2/I3 ⇒ 生效值 0、raw 保留',
           p['J'] == 0.0j and p['A'] == 0.0j
           and p['raw_J'] != 0 and p['raw_A'] != 0)
    _check('A0g dropped_amp 报告非零',
           cal['dropped_amp']['I2'] > 0
           and cal['dropped_amp']['I3'] > 0,
           str({k: f'{v:.3g}' for k, v in cal['dropped_amp'].items()}))


def main():
    t0 = time.time()
    test_A0a_first_order()
    test_A0b_phi_covariance()
    cal = test_A0c_ed_conventions()
    test_A0d_roundtrip(cal)
    test_A0e_m_conservation()
    test_A0f_second_order(cal)
    test_A0g_term_switch()
    n_ok = sum(1 for _, ok in _results if ok)
    print(f'\n== {n_ok}/{len(_results)} 通过，用时 {time.time() - t0:.1f}s ==')
    return 0 if n_ok == len(_results) else 1


if __name__ == '__main__':
    sys.exit(main())
