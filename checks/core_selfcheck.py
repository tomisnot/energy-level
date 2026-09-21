# -*- coding: utf-8 -*-
"""core.py 精简自检（6 个**不变式包** / 22 个子条件）：手写算法 vs ARC。

计数模型（2026-09-14 起）：按 T 段打包成不变式包 —— 整包 1 项、包内子条件
单独累计（`SUB`）。每个原断言都原样保留为某个包的子条件，不许删/放宽。

覆盖（T1-T11 全功能面）：
  T1 原子参数/超精细   T2 超精细能量 J=1/2   T3 超精细能量 J=3/2(含B)
  T4 超精细偶极矩阵元  T5 sum rule           T6 Einstein A 系数
  T7 基本功能回归      T8 远失谐 SOS         T9 共振 dressed
  T10 F↔mJ 合法/禁忌   T11 双后端一致
"""
import os
import sys
import importlib.util
import numpy as np
if sys.platform == 'win32' and hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)     # 项目根（core.py 所在）

# ---- 应用 ARC 数据目录补丁（src/__init__ 把 AlkaliAtom.dataFolder
#      指向项目内 .arcdata/，Rubidium87() 实例化才无需外部写权限） ----
sys.path.insert(0, _ROOT)
import src            # noqa: F401

# ---- 加载 core（src.core 包无 UI 依赖，import 即安全） ----
from src.core import core

from arc.alkali_atom_data import Rubidium87, Caesium
from scipy.constants import c, epsilon_0, hbar
from checklib import Checker

ch = Checker('core_selfcheck')


def _hf_min_E(cache, n, l, j, B):
    """{F: 该 F 最低子能级能量 MHz}（相对 fine 态中心）。"""
    hf = core.get_hyperfine_levels(cache, n, l, j, B)
    out = {}
    for F, mF, e, w, a in hf:
        out.setdefault(round(F), []).append(e)
    return {F: min(v) for F, v in out.items()}


# ----------------------------------------------------------------------
# 每个 t*() 返回 [(子条件标签, 条件), ...]，由 main() 归入对应的不变式包。
def t1():
    print('T1 原子参数 / 超精细结构')
    subs = []
    cache = core.ArcCache(Rubidium87())
    subs.append(('Rb87 I/gI 来自 ARC',
                 abs(cache.I - 1.5) < 1e-9 and abs(cache.gI + 0.0009951414) < 1e-12))
    cache = core.ArcCache(Caesium())
    hf = core.get_hyperfine_levels(cache, 6, 0, 0.5, 0.0)
    E = {}
    for F, mF, e, w, a in hf:
        E.setdefault(round(F), []).append(e)
    E = {F: min(v) for F, v in E.items()}
    dE = max(E.values()) - min(E.values())
    subs.append(('Cs133 6S 分裂 = 9192.6318 MHz（铯原子钟）',
                 abs(dE - 9192.631770) <= 1e-6 * abs(9192.631770)))
    return subs


def t2():
    print('T2 超精细能量 J=1/2 vs ARC getHFSEnergyShift')
    subs = []
    cache = core.ArcCache(Rubidium87())
    ours = _hf_min_E(cache, 5, 0, 0.5, 0.0)
    A_hz, B_hz = cache.atom.getHFSCoefficients(5, 0, 0.5)
    ref = {F: cache.atom.getHFSEnergyShift(0.5, F, A_hz, B_hz) / 1e6 for F in (1, 2)}
    for F in sorted(ours):
        subs.append((f'5S F={F}', abs(ours[F] - ref[F]) <= 1e-6))
    return subs


def t3():
    print('T3 超精细能量 J=3/2（含 B 四极项）vs ARC')
    subs = []
    cache = core.ArcCache(Rubidium87())
    ours = _hf_min_E(cache, 6, 1, 1.5, 0.0)
    A_hz, B_hz = cache.atom.getHFSCoefficients(6, 1, 1.5)
    ref = {F: cache.atom.getHFSEnergyShift(1.5, F, A_hz, B_hz) / 1e6 for F in (0, 1, 2, 3)}
    for F in sorted(ours):
        subs.append((f'6P3/2 F={F}', abs(ours[F] - ref[F]) <= 1e-4))
    return subs


def t4():
    print('T4 超精细态间偶极矩阵元（B=0）全量 vs ARC')
    subs = []
    cache = core.ArcCache(Rubidium87())
    rows = core.build_level_rows(
        cache, 0.0, [], {},
        [dict(n=5, l=0, j=0.5, label='5S', color='r'),
         dict(n=6, l=1, j=1.5, label='6P3/2', color='b')])
    maxdev, n_tot = 0.0, 0
    for a in rows:
        if a['n'] != 5:
            continue
        for b in rows:
            if b['n'] != 6:
                continue
            Fa = float(a['qstr'].split('F=')[1].split(',')[0])
            Fb = float(b['qstr'].split('F=')[1].split(',')[0])
            for q in (-1, 0, 1):
                ours = core.transition_strength_q(cache, a, b, q)
                arc = cache.atom.getDipoleMatrixElementHFS(
                    5, 0, 0.5, Fa, a['mF'], 6, 1, 1.5, Fb, b['mF'], q) ** 2
                maxdev = max(maxdev, abs(ours - arc))
                n_tot += 1
    subs.append((f'5S×6P3/2 全部 {n_tot} 项一致（含 mF 守恒）', maxdev < 1e-12))
    return subs


def t5():
    print('T5 Σ|d_pair|² == getReducedMatrixElementJ²')
    subs = []
    cache = core.ArcCache(Rubidium87())
    n1, l1, j1, n2, l2, j2 = 60, 1, 0.5, 60, 0, 0.5
    S = 0.0
    for m1 in np.arange(-j1, j1 + 1, 1.0):
        for m2 in np.arange(-j2, j2 + 1, 1.0):
            q = int(round(m2 - m1))
            S += abs(cache.dme(n1, l1, j1, m1, n2, l2, j2, m2, q)) ** 2
    red = cache.atom.getReducedMatrixElementJ(n1, l1, j1, n2, l2, j2) ** 2
    subs.append(('Σ(60P1/2→60S) / reduced²', abs(S / red - 1.0) <= 1e-9))
    return subs


def t6():
    print('T6 Einstein A：spectrum_lines vs 标准公式')
    subs = []
    cache = core.ArcCache(Rubidium87())
    rows = core.build_level_rows(
        cache, 0.0, [], {},
        [dict(n=60, l=0, j=0.5, label='60S', color='r'),
         dict(n=60, l=1, j=0.5, label='60P1/2', color='b')])
    lines = core.spectrum_lines(cache, rows, False)
    maxdev, maxA = 0.0, 0.0
    for nu, A, la, lb, qa, qb, col, pol in lines:
        maxA = max(maxA, A)
        a = next(r for r in rows if r['label'] == la and r['qstr'] == qa)
        b = next(r for r in rows if r['label'] == lb and r['qstr'] == qb)
        S = sum(core.transition_strength_q(cache, a, b, q) for q in (-1, 0, 1))
        A_ref = ((2 * np.pi * nu) ** 3 / (3 * np.pi * epsilon_0 * hbar * c ** 3)
                 * S * (core.E_CHARGE * core.A0) ** 2)
        maxdev = max(maxdev, abs(A - A_ref))
    subs.append(('每条线 A == 标准公式', maxdev < 1e-9 * maxA))
    A_sum = sum(ln[1] for ln in lines)
    A_arc = cache.atom.getTransitionRate(60, 1, 0.5, 60, 0, 0.5)
    subs.append(('Σ_A / getTransitionRate = 2.0（ARC asymmetric 约定）',
                 abs(A_sum / A_arc - 2.0) < 0.01))
    return subs


def t7():
    print('T7 基本功能回归（行/谱线/collect_transitions，direct 后端）')
    subs = []
    cache = core.ArcCache(Rubidium87())
    cfg = {'enabled': True, 'intensity': 10000.0,
           'frequency_hz': 295723722000000.0, 'polarization': 'π'}
    groups = [dict(n=60, l=0, j=0.5, label='60S', color='r'),
              dict(n=60, l=1, j=0.5, label='60P1/2', color='b')]
    rows = core.build_level_rows(cache, 0.0, groups, cfg, groups)
    off = core.spectrum_lines(cache, rows, False)
    on = core.spectrum_lines(cache, rows, True)
    subs.append(('行数=4 且 off/on 线数=4/4',
                 len(rows) == 4 and len(off) == 4 and len(on) == 4))
    # 60S 两 m 子能级 π 偏振位移近似对称（direct 一次对角化结果）。
    # 默认 n_photon=4：高阶光子块对 Zeeman 失谐差敏感 → 允许 ~0.5% 物理容差。
    a_ = rows[0]['ac']
    b_ = rows[1]['ac']
    subs.append(('60S m=±0.5 位移近似相等（π，±0.5%）',
                 abs(a_ - b_) <= 5e-3 * max(abs(a_), abs(b_)) + 1e-12))
    nu = sorted(set(round(ln[0] / 1e9, 4) for ln in off))
    subs.append(('谱线频率 ≈ 16.8267 GHz', abs(nu[0] - 16.8267) < 1e-3))
    pairs = core.collect_transitions(cache, groups, cfg)
    subs.append(('collect_transitions 1 对且 qstr=m_J',
                 len(pairs) == 1 and 'm_J=' in pairs[0][1]))
    groups2 = [dict(n=5, l=0, j=0.5, label='5S', color='r'),
               dict(n=6, l=1, j=1.5, label='6P3/2', color='b')]
    cfg2 = dict(cfg, frequency_hz=core.transition_frequency_hz(
        cache.atom, 5, 0, 0.5, 6, 1, 1.5))
    pairs2 = core.collect_transitions(cache, groups2, cfg2)
    subs.append(('超精细 5S↔6P3/2 qstr 含 F/m_F',
                 len(pairs2) >= 1 and 'F=' in pairs2[0][1]))
    return subs


def t8():
    print('T8 far_sos_shift_mhz（非 RWA SOS）ω→0 == DC-Stark/2（单对解析）')
    subs = []
    cache = core.ArcCache(Rubidium87())
    n, l, j, m = 6, 0, 0.5, 0.5
    d_au = cache.dme(n, l, j, m, 6, 1, 0.5, m, 0)        # π：6S m → 6P1/2 m
    E0_sq = 2.0 * (1.0 * core.MW_PER_CM2_TO_W_PER_M2) / (epsilon_0 * c)
    nu_hz = (cache.E_mhz(6, 1, 0.5) - cache.E_mhz(6, 0, 0.5)) * 1e6
    nu_rad = 2.0 * np.pi * nu_hz
    omega = nu_rad * 1e-6                                 # ω→0（远低于跃迁）
    sos = core.far_sos_shift_mhz(d_au, E0_sq, omega, nu_rad)
    # DC 单对解析：ΔE = −d²E0²/(2ħν)（J）→ MHz；far_sos(ω→0) 应与其相等
    d_cm = abs(d_au) * core.E_CHARGE * core.A0
    dc_single = -(d_cm ** 2 * E0_sq) / (2.0 * hbar * nu_rad) / (2.0 * np.pi * hbar) / 1e6
    subs.append(('far_sos(ω→0) == DC 单对（ω→0 极限，比值=1）',
                 abs(sos / dc_single - 1.0) <= 1e-6))
    return subs


def t9():
    print('T9 共振 dressed（±Ω/2，与显示一致）')
    subs = []
    cache = core.ArcCache(Rubidium87())
    B = 21.7
    groups = [dict(n=6, l=1, j=1.5, label='6P_{3/2}', color='r'),
              dict(n=60, l=0, j=0.5, label='60S_{1/2}', color='b')]
    b63 = next(r for r in core.base_sublevels(cache, [dict(n=6, l=1, j=1.5)], B)
               if r['qstr'] == 'F=3.0, m_F=+3.0')
    a60 = next(r for r in core.base_sublevels(cache, [dict(n=60, l=0, j=0.5)], B)
               if r['qstr'] == 'm_J=+0.5')
    cfg = {'enabled': True, 'intensity': 10000.0,
           'frequency_hz': (a60['E_mhz'] - b63['E_mhz']) * 1e6, 'polarization': 'σ-'}
    rows = core.build_level_rows(cache, B, groups, cfg, groups)
    r63 = next(r for r in rows if r['group_label'] == '6P_{3/2}'
               and r['qstr'] == 'F=3.0, m_F=+3.0')
    subs.append(('共振位移 > 100 kHz（≈Ω/2，非 kHz 微扰）', abs(r63['ac']) > 0.1))
    p = min(core.collect_transitions(cache, groups, cfg, B), key=lambda x: abs(x[4]))
    # direct 完整对角化（含超精细多态/反旋项）→ 位移 ≈ −Ω/2（源态共振下移，
    # 与 legacy 两能级 +Ω/2 符号约定不同，2025-08 起只保留 direct）
    _ref = p[5] / (2.0 * np.pi) / 1e6 / 2.0
    subs.append(('|ac| ≈ Ω/2（direct 自洽，rtol 10%）',
                 abs(abs(r63['ac']) - _ref) <= 0.10 * abs(_ref)))
    return subs


def t10():
    print('T10 F↔mJ 矩阵元 vs ARC（合法 + 禁忌）')
    subs = []
    cache = core.ArcCache(Rubidium87())
    rows_6p = core.base_sublevels(cache, [dict(n=6, l=1, j=1.5)], 0.0)
    rows_60s = core.base_sublevels(cache, [dict(n=60, l=0, j=0.5)], 0.0)
    for qs_a, qs_b, q in (('F=3.0, m_F=+3.0', 'm_J=+0.5', -1),   # σ- 合法
                          ('F=3.0, m_F=-1.0', 'm_J=+0.5', -1)):  # σ- 禁忌
        a = next(r for r in rows_6p if r['qstr'] == qs_a)
        b = next(r for r in rows_60s if r['qstr'] == qs_b)
        ours = core.transition_strength_q(cache, a, b, q)
        Fa = float(qs_a.split(',')[0].split('=')[1])
        mFa = float(qs_a.split('m_F=')[1])
        mjb = float(qs_b.split('m_J=')[1])
        arc = cache.atom.getDipoleMatrixElementHFStoFS(
            6, 1, 1.5, Fa, mFa, 60, 0, 0.5, mjb, q) ** 2
        subs.append((f'{qs_a} → {qs_b} q={q:+d}', abs(ours - arc) <= 1e-14))
    return subs


# ----------------------------------------------------------------------
def main():
    print('=' * 64)
    print('core.py 精简自检（手写算法 vs ARC 官方函数）')
    print('=' * 64)
    # T11（direct vs legacy 双后端对比）已于 2025-08 删除：
    # legacy 后端移除，AC-Stark 唯一后端 = direct（第一性原理）。
    # 每个 T 段 = 1 个**不变式包**（整包 1 项；包内子条件累计进 SUB）。
    ch.group('T1–T3 原子参数与超精细能量（Rb87 I/gI、Cs 钟频、'
             'J=1/2、J=3/2 含 B 四极项）', t1() + t2() + t3())
    ch.group('T4–T5 偶极矩阵元 / sum rule（HFS 全量 vs ARC、Σ|d|² vs reduced²）',
             t4() + t5())
    ch.group('T6 Einstein A 系数（逐线标准公式 + ARC asymmetric 约定 2.0）', t6())
    ch.group('T7 基本功能回归（行/谱线/collect_transitions，direct 后端）', t7())
    ch.group('T8–T9 AC-Stark / dressed 位移（远失谐 DC 极限、共振 ±Ω/2）',
             t8() + t9())
    ch.group('T10 F↔mJ 矩阵元 vs ARC（合法 + 禁忌）', t10())
    ok = ch.summary()
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
