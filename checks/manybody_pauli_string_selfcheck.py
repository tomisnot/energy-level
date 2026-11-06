# -*- coding: utf-8 -*-
"""`manybody.pauli_strings`（任意阶 Pauli 字符串）的独立判据。

判据原则：**实现与独立的稠密参考必须给出同一个数**，并覆盖全空间/受限基/MPS。

- 全空间、受限基：`PauliPlan`；
- MPS：`pauli_strings_mps_batch`（局域算符之积展开）；
- 稠密参考：把 2×2 Pauli 直接 kron 成 2ⁿ×2ⁿ 矩阵，`v†Ov`，不共享实现。

另验：
  · 受限基（固定激发数）在**守恒算符**上与全空间嵌入值一致；
  · MPS 在 `conserve='Sz'` 下可用；
  · 含时 ED 的时间序列对得上解析解、且 wiring（时间下标/字符串顺序）没错；
  · DMRG 与 ED 在同一（复）模型上给出同一组期望；
  · 非法规格当场被拒（不静默改写）。
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from checks.checklib import Checker
from src.core.manybody.observables import (
    PAULI_LOCAL, normalize_pauli_strings, pauli_strings_mps_batch,
    measure_pauli_strings, PauliPlan)
from src.core.manybody.schema_mb import validate_pauli_strings
from src.core.manybody.engine_ed import evolve_manybody
from src.core.manybody.eq_engine import solve_equilibrium

C = Checker('manybody_pauli_string')


# ----------------------------------------------------------------------
def dense_reference(vec, n, terms):
    """不共享代码的参考：kron 出 2ⁿ×2ⁿ 矩阵再取期望。"""
    op = np.ones((1, 1), complex)
    by_site = {int(s): a for s, a in terms}
    for i in range(n):
        op = np.kron(op, PAULI_LOCAL[by_site[i]] if i in by_site else np.eye(2))
    return complex(np.vdot(vec, op @ vec))


def random_vector(n, seed):
    rng = np.random.default_rng(seed)
    v = rng.normal(size=1 << n) + 1j * rng.normal(size=1 << n)
    return v / np.linalg.norm(v)


def engine_site(conserve=None):
    """与 `rydberg_cluster_model` 同款站点（SZ/P/M 是固定矩阵，与 conserve 无关）。"""
    from tenpy.networks.site import SpinHalfSite
    site = SpinHalfSite(conserve=conserve)
    site.add_op('Nhat', np.array([[0, 0], [0, 1]], dtype=complex))
    site.add_op('P', np.array([[0, 0], [1, 0]], dtype=complex))
    site.add_op('M', np.array([[0, 1], [0, 0]], dtype=complex))
    site.add_op('SZ', np.array([[1, 0], [0, -1]], dtype=complex))
    if conserve is None:
        site.add_op('SX', np.array([[0, 1], [1, 0]], dtype=complex))
        site.add_op('SY', np.array([[0, -1j], [1j, 0]], dtype=complex))
    return site


def tenpy_mps(vec, n, conserve=None, seq=None):
    from tenpy.networks.mps import MPS
    from tenpy.linalg import np_conserved as npc
    site = engine_site(conserve)
    if conserve is None:
        arr = npc.Array.from_ndarray_trivial(
            vec.reshape((2,) * n), labels=['p%d' % i for i in range(n)])
        return MPS.from_full([site] * n, arr, bc='finite', form='B')
    arr = npc.Array.from_ndarray(vec.reshape((2,) * n), [site.leg] * n,
                                 qtotal=[0],
                                 labels=['p%d' % i for i in range(n)])
    return MPS.from_full([site] * n, arr, bc='finite', form='B')


# ----------------------------------------------------------------------
def t_three_paths():
    """同一随机复态上，三条路径 + 稠密参考必须一致（1/2/3/4 体、连续/非连续）。"""
    subs = []
    for n, seed in ((4, 3), (5, 8)):
        v = random_vector(n, seed)
        avail = list(range(n))
        cand = [avail[:1], avail[:2], avail[:3], avail[::-1][:4],
                [0, 2] if n > 2 else avail[:2],
                [0, 2, 4] if n > 4 else avail[:3]]
        specs = normalize_pauli_strings(
            [[[s, ax] for s, ax in zip(tri, 'xyz')] for tri in cand
             if len(tri) <= 4], n)
        specs += normalize_pauli_strings([[[0, 'y']], [[1, 'x'], [3, 'y']]], n)
        mps = tenpy_mps(v, n)
        got = pauli_strings_mps_batch(mps, specs)
        full = PauliPlan(n, specs).expect(v)
        basis = PauliPlan(n, specs, basis_states=np.arange(1 << n)).expect(v)
        for q, d in enumerate(specs):
            ref = dense_reference(v, n, d['terms'])
            subs.append(('N%d %s: full≈ref' % (n, d['label']), abs(full[q] - ref) < 1e-12))
            subs.append(('N%d %s: basis≈ref' % (n, d['label']), abs(basis[q] - ref) < 1e-12))
            subs.append(('N%d %s: mps≈ref' % (n, d['label']), abs(got[q] - ref) < 1e-10))
    C.group('三条路径 + 稠密参考一致', subs)


def t_restricted_basis():
    """固定激发数子空间：**守恒算符**上与全空间嵌入值必须相同。"""
    n, k = 5, 2
    codes = np.array([s for s in range(1 << n)
                      if bin(s).count('1') == k], dtype=np.int64)
    rng = np.random.default_rng(11)
    sub = rng.normal(size=len(codes)) + 1j * rng.normal(size=len(codes))
    sub /= np.linalg.norm(sub)
    full = np.zeros(1 << n, complex)
    full[codes] = sub
    # σᶻσᶻ 不改变总激发数 → 两个空间必须逐位一致
    terms = [(0, 'z'), (2, 'z')]
    a = PauliPlan(n, [{'terms': terms}], basis_states=codes).expect(sub)[0]
    b = PauliPlan(n, [{'terms': terms}]).expect(full)[0]
    C.close('受限基（k=%d）× σᶻσᶻ 与全空间一致' % k, a.real, b.real, atol=1e-12)
    C.check('受限基下 σˣ（非守恒）被正确截断为"无分量"',
            abs(PauliPlan(n, [{'terms': [(0, 'x')]}],
                          basis_states=codes).expect(sub)[0]) < 1e-12,
            '单 σˣ 改变激发数，扇区内应为 0')


def t_mps_conserve():
    """MPS 在 conserve='Sz' 下可用——这正是 TeNPy 自带算符做不到的那条。"""
    from tenpy.networks.mps import MPS
    n = 5
    seq = ['up', 'down', 'up', 'down', 'up']
    psi = MPS.from_product_state([engine_site('Sz')] * n, seq, bc='finite')
    specs = normalize_pauli_strings(
        [[[0, 'z']], [[0, 'z'], [2, 'z']], [[0, 'x']], [[1, 'x'], [3, 'y']],
         [[0, 'x'], [2, 'y'], [4, 'z']]], n)
    got = pauli_strings_mps_batch(psi, specs)
    # 引擎约定：index0='up'=g(n=0)、index1='down'=r(n=1)；σᶻ=diag(+1,−1)
    v = np.zeros(1 << n, complex)
    v[0b01010] = 1.0                       # g r g r g（x 位置是 index 0/1 的位）
    ref = [dense_reference(v, n, d['terms']) for d in specs]
    subs = [('conserve=Sz %s' % d['label'], abs(g - r) < 1e-12)
            for d, g, r in zip(specs, got, ref)]
    C.group("conserve='Sz' 的 MPS 上可用", subs)


def t_validation():
    """非法规格当场拒（不静默改写）。"""
    bad = []
    for label, spec in (
            ('轴名非法', [[[0, 'q']]]),
            ('同站点重复', [[[1, 'x'], [1, 'z']]]),
            ('站点越界', [[[5, 'x']]]),
            ('不是 list', {'a': 1}),
            ('空字符串', [[]]),
            ('缺 terms', [{'label': 'x'}])):
        try:
            validate_pauli_strings(spec, n_atoms=4)
            bad.append((label, False))
        except Exception:                                   # noqa: BLE001
            bad.append((label, True))
    C.group('非法规格被拒', bad)

    ok = validate_pauli_strings([[[0, 'x'], [1, 'y'], [2, 'z']]], n_atoms=4)
    C.check('自动生成 label', ok[0]['label'] == 'x@0 y@1 z@2', ok[0]['label'])
    ok2 = validate_pauli_strings([{'label': 'chi', 'terms': [[0, 'σx'], [1, 'sy']]}], 4)
    C.check('接受 σx/sy 别名并归一化',
            ok2[0]['label'] == 'chi' and ok2[0]['terms'] == [(0, 'x'), (1, 'y')],
            str(ok2))


def t_time_evolution():
    """含时 ED：时间序列对上解析解，且 wiring（时间下标/字符串顺序）正确。"""
    J = 0.77
    cal = {'atoms': [{'id': 0, 'pos': [0, 0, 0]}, {'id': 1, 'pos': [1, 0, 0]}],
           'pairs': [{'i': 0, 'j': 1, 'V': 0, 'J': J, 'A': 0}]}
    t = np.array([0.0, 0.25, 0.5, 0.75]) / J * 1000.0        # 单位 ns
    specs = normalize_pauli_strings([[[0, 'z']], [[1, 'z']], [[0, 'x'], [1, 'x']]], 2)
    res = evolve_manybody(cal, psi0='rg', t_grid=t, Om=np.zeros(2, complex),
                          delta=np.zeros(2), terms=['I2'],
                          want_observables=True, pauli_strings=specs,
                          record_psi=True)
    series = np.asarray(res['observables']['pauli_strings'])
    C.check('形状 (n_t, n_strings)', series.shape == (len(t), len(specs)),
            str(series.shape))
    C.check('labels 回传', res['observables']['pauli_string_labels']
            == [d['label'] for d in specs])
    # ⟨σᶻ₀⟩(t)：从 |rg⟩（原子0 激发）出发，σᶻ = diag(+1,−1) on (g,r) ⇒ 初值 −1，
    # 交换给出 ⟨σᶻ₀⟩ = −cos(4πJt)（在 t=1/(4J) 处完全交换 → +1）
    expect = -np.cos(4 * np.pi * J * t * 1e-3)
    subs = [('t=%.1fns σᶻ₀' % t[i], abs(series[i, 0].real - expect[i]) < 1e-9)
            for i in range(len(t))]
    subs.append(('σᶻ₁ = −σᶻ₀', np.max(np.abs(series[:, 1].real + series[:, 0].real)) < 1e-12))
    C.group('两站点 ⟨σᶻ₀⟩(t) 对上解析 cos(4πJt)', subs)
    # wiring：末时刻的值必须等于末态上的直接计算
    psi_last = np.asarray(res['psi_amp'], complex).ravel()
    direct = [dense_reference(psi_last, 2, d['terms']) for d in specs]
    C.close('末时刻 vs 末态直接计算', float(np.max(np.abs(series[-1] - np.array(direct)))),
            0.0, atol=1e-12)


def t_equilibrium_ed_and_dmrg():
    """同一**复**模型上，ED 与 DMRG 给出同一组期望（复耦合 → 期望可非零非纯实）。"""
    # 4 位环、一条键带通量 → 基态复、期望值非平凡
    n = 4
    Js = [1.0, 1.0, 1.0, np.exp(1j * np.pi / 4)]
    cal = {'atoms': [{'id': i, 'pos': [float(i), 0.0, 0.0]} for i in range(n)],
           'pairs': [{'i': i, 'j': (i + 1) % n, 'V': 0, 'J': Js[i], 'A': 0}
                     for i in range(n)]}
    specs = validate_pauli_strings(
        [[[0, 'x'], [1, 'y'], [2, 'z']], [[0, 'z']], [[1, 'z'], [3, 'z']],
         [[0, 'x'], [2, 'x']]], n)
    base = {'manybody.calib': cal, 'manybody.pauli_strings': specs}
    red = solve_equilibrium(None, dict(base, **{'manybody.eq_backend': 'ed',
                                                'manybody.charge_sector': 1}), want='gs')
    rg = solve_equilibrium(None, dict(base, **{'manybody.eq_backend': 'dmrg',
                                               'manybody.charge_sector': 1,
                                               'manybody.mps_order': list(range(n)),
                                               'manybody.chi_max': 64,
                                               'manybody.max_sweeps': 20}), want='gs')
    a = np.array(red['pauli_strings'], complex)
    b = np.array(rg['pauli_strings'], complex)
    C.close('ED/DMRG 基态期望一致', float(np.max(np.abs(a - b))), 0.0, atol=1e-8)
    # 非 z 轴的字符串非零 → 真的走到了 P/M（σˣ、σʸ）分支，而不是只算了 σᶻ。
    # （注：该模型基态仍是**实**的，所以虚部恒为 0——这不是 bug；复相位路径由
    #  三条路径随机复态那一组覆盖。）
    nonzero_xy = max(abs(a[0]), abs(a[3]))
    C.check('含 x/y 的字符串给出非零期望（走到 P/M 分支）',
            nonzero_xy > 1e-3, 'max|⟨xy·⟩| = %.3g' % nonzero_xy)
    C.check('ED 结果带 labels', red.get('pauli_string_labels') == [d['label'] for d in specs])


def t_contract():
    """RunSpec 层：键进 schema、后端不支持时明确拒。"""
    from src.core.manybody.backends import validate_time_capabilities
    from src.core.manybody.schema_mb import CONFIG_SCHEMA_MB
    C.check('manybody.pauli_strings 在 schema 里',
            'manybody.pauli_strings' in CONFIG_SCHEMA_MB)
    rejects = []
    for solver in ('peps-real-time',):
        try:
            validate_time_capabilities(solver, pauli_strings=[{'label': 'x', 'terms': [(0, 'x')]}])
            rejects.append((solver, False))
        except ValueError:
            rejects.append((solver, True))
    C.group('不支持的后端明确拒绝（不静默忽略）', rejects)
    # CPU MPS / 轨迹 / 经典自旋已接同一套测量入口；这里只验能力声明，不跑演化。
    supports = []
    for solver in ('tdvp', 'tebd', 'tdvp-single', 'tdvp-adaptive',
                   'tdvp-trajectories', 'classical-spin'):
        try:
            validate_time_capabilities(solver, pauli_strings=[{'label': 'x', 'terms': [(0, 'x')]}])
            supports.append((solver, True))
        except ValueError:
            supports.append((solver, False))
    C.group('已接入的后端通过能力声明', supports)
    try:
        validate_time_capabilities('ed', pauli_strings=[{'label': 'x', 'terms': [(0, 'x')]}])
        C.check('ed 接受 pauli_strings', True)
    except Exception as exc:                                # noqa: BLE001
        C.check('ed 接受 pauli_strings', False, str(exc))


def t_unified_entry():
    """R1：统一入口与 PauliPlan 一致，且都对得上独立稠密参考。"""
    rng = np.random.default_rng(7)
    n = 3
    dim = 1 << n
    v = rng.normal(size=dim) + 1j * rng.normal(size=dim)
    v /= np.linalg.norm(v)
    specs = normalize_pauli_strings(
        [[[0, 'z']], [[0, 'y'], [2, 'y']], [[0, 'x'], [1, 'y'], [2, 'z']]], n)
    ref = [dense_reference(v, n, s['terms']) for s in specs]
    new = measure_pauli_strings(v, specs, n=n)
    plan = PauliPlan(n, specs).expect(v)
    C.group('统一入口 / PauliPlan 与稠密参考一致', [
        ('measure == ref', np.allclose(new, ref, atol=1e-14)),
        ('plan == ref', np.allclose(plan, ref, atol=1e-14)),
    ])
    codes = np.array([1, 2, 3, 4, 5, 6], np.int64)
    vb = rng.normal(size=len(codes)) + 1j * rng.normal(size=len(codes))
    vb /= np.linalg.norm(vb)
    full = np.zeros(dim, complex)
    full[codes] = vb
    # 受限基只在**守恒算符**上与全空间嵌入等价；σˣ/σʸ 的截断另由
    # t_restricted_basis 的"非守恒 → 0"断言覆盖。
    specs_b = normalize_pauli_strings([[[0, 'z']], [[1, 'z'], [2, 'z']]], n)
    ref_b = [dense_reference(full, n, s['terms']) for s in specs_b]
    new_b = measure_pauli_strings(vb, specs_b, basis_states=codes, n=n)
    C.group('统一入口与受限基一致', [
        ('measure == 嵌入全空间后的稠密参考',
         np.allclose(new_b, ref_b, atol=1e-14)),
    ])


def main():
    t_three_paths()
    t_restricted_basis()
    t_mps_conserve()
    t_validation()
    t_time_evolution()
    t_equilibrium_ed_and_dmrg()
    t_contract()
    t_unified_entry()
    return 0 if C.summary() else 1


if __name__ == '__main__':
    sys.exit(main())
