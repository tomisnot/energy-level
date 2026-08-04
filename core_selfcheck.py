# -*- coding: utf-8 -*-
"""core.py 自检 / 回归测试：手写算法 vs ARC 官方函数交叉验证。

用法:
    & "D:\python310\python.exe" core_selfcheck.py

全部通过时打印各 [PASS] 并以退出码 0 结束；任何失败打印 [FAIL] 并以
退出码 1 结束，便于 CI / 手动回归。

覆盖（每项都对应一个容易出错的物理量）:
    T1 核自旋 I / gI 来自 ARC 原子实例（Rb87、Cs133）
    T2 超精细能量 J=1/2：ours == breitRabi == getHFSEnergyShift
    T3 超精细能量 J=3/2：ours（含 B 电四极项）== getHFSEnergyShift(含B)
       （breitRabi 对 J>1/2 为近似，只报告差异，不作断言）
    T4 超精细态间偶极矩阵元 (B=0)：ours == getDipoleMatrixElementHFS
       （覆盖 5S×6P3/2 全部 (F,mF)×(F,mF)×q）
    T5 矩阵元求和：Σ|d_pair|² == getReducedMatrixElementJ²（比值=1）
    T6 Einstein A：spectrum_lines 每条线与标准公式一致；并报告
       vs getTransitionRate 的 ARC asymmetric 约定因子
    T7 基本功能回归：行数/线数/π 对称/谱线频率/跃迁对
    T8 远失谐 SOS vs DynamicPolarizability：j=1/2 态比值恒为 -0.5
       （ARC 约定 ΔE=-½·alpha·E0²，alpha0 为标量=2×单 m π 值）
       此项曾在 2026-08 发现并修复"缺 ħ" bug（低估 ~1e34 倍）。
"""

import os
import sys
import importlib.util
import numpy as np

# Windows 控制台默认 GBK，无法打印 ² 等字符；强制 UTF-8 输出避免崩溃。
if sys.platform == 'win32' and hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

_HERE = os.path.dirname(os.path.abspath(__file__))

# ---- 加载 core（不触发 GUI / 不执行 UI 代码） ----
spec = importlib.util.spec_from_file_location('core', os.path.join(_HERE, 'core.py'))
core = importlib.util.module_from_spec(spec)
spec.loader.exec_module(core)

from arc.alkali_atom_data import Rubidium87, Caesium
from arc.calculations_atom_single import DynamicPolarizability
from scipy.constants import c, epsilon_0, hbar

_PASS = 0
_FAIL = 0


def check(name, cond, detail=''):
    global _PASS, _FAIL
    if cond:
        _PASS += 1
        print(f'  [PASS] {name}')
    else:
        _FAIL += 1
        print(f'  [FAIL] {name}   {detail}')


def check_close(name, ours, ref, rtol=1e-6, atol=0.0):
    global _PASS, _FAIL
    if abs(ours - ref) <= atol + rtol * abs(ref):
        _PASS += 1
        print(f'  [PASS] {name}  ({ours:.6g} vs {ref:.6g})')
    else:
        _FAIL += 1
        print(f'  [FAIL] {name}  ours={ours:.8g} ref={ref:.8g}')


def _hf_min_E(cache, n, l, j, B):
    """返回 {F: 该 F 最低子能级能量 MHz}（相对 fine 态中心）。"""
    hf = core.get_hyperfine_levels(cache, n, l, j, B)
    out = {}
    for F, mF, e, w, a in hf:
        out.setdefault(round(F), []).append(e)
    return {F: min(v) for F, v in out.items()}


# ----------------------------------------------------------------------
def t1():
    print('T1 核自旋 I / gI 来自 ARC 原子实例')
    cache = core.ArcCache(Rubidium87())
    check('Rb87 I = 1.5', abs(cache.I - 1.5) < 1e-9, cache.I)
    check('Rb87 gI = -0.0009951414', abs(cache.gI + 0.0009951414) < 1e-12, cache.gI)
    cache = core.ArcCache(Caesium())
    check('Cs133 I = 3.5', abs(cache.I - 3.5) < 1e-9, cache.I)
    hf = core.get_hyperfine_levels(cache, 6, 0, 0.5, 0.0)
    check('Cs133 6S 16 个子能级', len(hf) == 16, len(hf))
    E = {}
    for F, mF, e, w, a in hf:
        E.setdefault(round(F), []).append(e)
    E = {F: min(v) for F, v in E.items()}
    dE = max(E.values()) - min(E.values())
    check_close('Cs133 6S 分裂 = 9192.6318 MHz（铯原子钟）', dE, 9192.631770, rtol=1e-6)


def t2():
    print('T2 超精细能量 J=1/2: ours == breitRabi == getHFSEnergyShift')
    cache = core.ArcCache(Rubidium87())
    ours = _hf_min_E(cache, 5, 0, 0.5, 0.0)
    E, Fv, mFv = cache.atom.breitRabi(5, 0, 0.5, np.array([0.0]))
    arc_br = {int(Fv[i]): E[0, i] / 1e6 for i in range(len(Fv))}   # Hz -> MHz
    A_hz, B_hz = cache.atom.getHFSCoefficients(5, 0, 0.5)
    arc_hfs = {F: cache.atom.getHFSEnergyShift(0.5, F, A_hz, B_hz) / 1e6 for F in (1, 2)}
    for F in sorted(ours):
        check_close(f'5S F={F} vs breitRabi', ours[F], arc_br[F], atol=1e-6)
        check_close(f'5S F={F} vs getHFSEnergyShift', ours[F], arc_hfs[F], atol=1e-6)


def t3():
    print('T3 超精细能量 J=3/2: ours（含 B 四极项）== getHFSEnergyShift')
    cache = core.ArcCache(Rubidium87())
    ours = _hf_min_E(cache, 6, 1, 1.5, 0.0)
    A_hz, B_hz = cache.atom.getHFSCoefficients(6, 1, 1.5)
    arc_hfs = {F: cache.atom.getHFSEnergyShift(1.5, F, A_hz, B_hz) / 1e6 for F in (0, 1, 2, 3)}
    for F in sorted(ours):
        check_close(f'6P3/2 F={F} vs getHFSEnergyShift(含B)', ours[F], arc_hfs[F], atol=1e-4)
    # breitRabi 对 J>1/2 为近似：仅报告偏差（不作为断言）
    E, Fv, mFv = cache.atom.breitRabi(6, 1, 1.5, np.array([0.0]))
    arc_br = {int(Fv[i]): E[0, i] / 1e6 for i in range(len(Fv))}
    maxdev = max(abs(ours[F] - arc_br.get(F, 0.0)) for F in ours)
    print(f'  [info] 6P3/2 breitRabi 近似偏差最大 {maxdev:.3f} MHz（ours 为准）')


def t4():
    print('T4 超精细态间偶极矩阵元 (B=0): ours == getDipoleMatrixElementHFS')
    cache = core.ArcCache(Rubidium87())
    rows = core.build_level_rows(
        cache, 0.0, [], {},
        [dict(n=5, l=0, j=0.5, label='5S', color='r'),
         dict(n=6, l=1, j=1.5, label='6P3/2', color='b')])
    maxdev = 0.0
    n_tot = 0
    n_nz = 0
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
                if arc > 1e-12:
                    n_nz += 1
                n_tot += 1
    check(f'5S×6P3/2 全部 {n_tot} 项一致（含 mF 守恒）', maxdev < 1e-12, f'maxdev={maxdev:.3e}')
    print(f'  [info] 非零矩阵元 {n_nz} 项 / {n_tot} 项')


def t5():
    print('T5 Σ|d_pair|² == getReducedMatrixElementJ²')
    cache = core.ArcCache(Rubidium87())
    for (n1, l1, j1, n2, l2, j2) in ((60, 1, 0.5, 60, 0, 0.5), (6, 1, 1.5, 6, 0, 0.5)):
        S = 0.0
        for m1 in np.arange(-j1, j1 + 1, 1.0):
            for m2 in np.arange(-j2, j2 + 1, 1.0):
                q = int(round(m2 - m1))
                S += abs(cache.dme(n1, l1, j1, m1, n2, l2, j2, m2, q)) ** 2
        red = cache.atom.getReducedMatrixElementJ(n1, l1, j1, n2, l2, j2) ** 2
        check_close(f'Σ({n1}{l1}{j1}→{n2}{l2}{j2}) / reduced²', S / red, 1.0, rtol=1e-9)


def t6():
    print('T6 Einstein A：spectrum_lines 与标准公式逐条一致')
    cache = core.ArcCache(Rubidium87())
    rows = core.build_level_rows(
        cache, 0.0, [], {},
        [dict(n=60, l=0, j=0.5, label='60S', color='r'),
         dict(n=60, l=1, j=0.5, label='60P1/2', color='b')])
    lines = core.spectrum_lines(cache, rows, False)
    maxdev = 0.0
    maxA = 0.0
    for nu, A, la, lb, qa, qb, col, pol in lines:
        maxA = max(maxA, A)
        # 独立重算：该子能级对的 S（各 q 求和）。注意 qstr 在跨组间
        # 会重复（如 'm_J=-0.5' 60S 和 60P 都有），必须用 (label, qstr) 联合匹配。
        a = next(r for r in rows if r['label'] == la and r['qstr'] == qa)
        b = next(r for r in rows if r['label'] == lb and r['qstr'] == qb)
        S = sum(core.transition_strength_q(cache, a, b, q) for q in (-1, 0, 1))
        A_ref = ((2 * np.pi * nu) ** 3 / (3 * np.pi * epsilon_0 * hbar * c ** 3)
                 * S * (core.E_CHARGE * core.A0) ** 2)
        maxdev = max(maxdev, abs(A - A_ref))
    check('每条线 A == 标准公式', maxdev < 1e-9 * maxA, f'maxdev={maxdev:.3e}')
    # 报告 ARC getTransitionRate 的约定因子（asymmetric 约化矩阵元，非错误）
    A_ours_sum = sum(ln[1] for ln in lines)
    A_arc = cache.atom.getTransitionRate(60, 1, 0.5, 60, 0, 0.5)
    r = A_ours_sum / A_arc
    check('Σ_A / getTransitionRate = 2.0（ARC asymmetric 约定）', abs(r - 2.0) < 0.01, f'r={r:.4f}')
    print(f'  [info] Σ_A_ours={A_ours_sum:.4f} s^-1, ARC={A_arc:.4f} s^-1, 比值={r:.4f}')


def t7():
    print('T7 基本功能回归')
    cache = core.ArcCache(Rubidium87())
    cfg = {'enabled': True, 'intensity': 10000.0,
           'frequency_hz': 295723722000000.0, 'polarization': 'π'}
    groups = [dict(n=60, l=0, j=0.5, label='60S', color='r'),
              dict(n=60, l=1, j=0.5, label='60P1/2', color='b')]
    rows = core.build_level_rows(cache, 0.0, groups, cfg, groups, backend='legacy')
    check('60S/60P 行数 = 4', len(rows) == 4, len(rows))
    off = core.spectrum_lines(cache, rows, False)
    on = core.spectrum_lines(cache, rows, True)
    check('off/on 线数 = 4/4', len(off) == 4 and len(on) == 4, (len(off), len(on)))
    a_ = core.ac_stark_shift_mhz(cache, groups[0], -0.5, groups, cfg)
    b_ = core.ac_stark_shift_mhz(cache, groups[0], 0.5, groups, cfg)
    check_close('60S m=±0.5 位移相等（π）', a_, b_, atol=1e-12)
    nu = sorted(set(round(ln[0] / 1e9, 4) for ln in off))
    check('谱线频率 ≈ 16.8267 GHz', abs(nu[0] - 16.8267) < 1e-3, nu)
    pairs = core.collect_transitions(cache, groups, cfg)
    check('collect_transitions 返回 1 对', len(pairs) == 1, len(pairs))
    if len(pairs) == 1:
        p0 = pairs[0]
        check('collect_transitions 6 元组', len(p0) == 6, len(p0))
        check('里德堡跃迁对 qstr 为 m_J', 'm_J=' in p0[1] and 'm_J=' in p0[3],
              (p0[1], p0[3]))
    # 超精细场景：5S↔6P3/2 的 qstr 应含 F / m_F
    groups2 = [dict(n=5, l=0, j=0.5, label='5S', color='r'),
               dict(n=6, l=1, j=1.5, label='6P3/2', color='b')]
    cfg2 = dict(cfg, frequency_hz=core.transition_frequency_hz(
        cache.atom, 5, 0, 0.5, 6, 1, 1.5))
    pairs2 = core.collect_transitions(cache, groups2, cfg2)
    check('5S/6P3/2 返回 1 对', len(pairs2) == 1, len(pairs2))
    if len(pairs2) == 1:
        p0 = pairs2[0]
        check('超精细跃迁对 qstr 含 F/m_F', 'F=' in p0[1] and 'm_F=' in p0[3],
              (p0[1], p0[3]))
        # 失谐最小：把频率设为某子能级对共振，应选中该对（Det≈0）
        a = next(r for r in core.build_level_rows(
            cache, 0.0, [], {}, groups2) if r['label'] == p0[0] and r['qstr'] == p0[1])
        b = next(r for r in core.build_level_rows(
            cache, 0.0, [], {}, groups2) if r['label'] == p0[2] and r['qstr'] == p0[3])
        nu_pair = abs(b['E0'] - a['E0']) * 1e6
        cfg3 = dict(cfg, frequency_hz=nu_pair)
        p3 = core.collect_transitions(cache, groups2, cfg3)[0]
        check('失谐最小选中同一对（共振）',
              (p3[1], p3[3]) == (p0[1], p0[3]), (p3[1], p3[3]))
        check('共振时 Det≈0', abs(p3[4]) < 1e-3, p3[4])


def t8():
    print('T8 远失谐 SOS vs DynamicPolarizability（j=1/2，比值=-0.5）')
    cache = core.ArcCache(Rubidium87())
    n, l, j, m = 6, 0, 0.5, 0.5
    I_mw = 1.0
    E0_sq = 2.0 * (I_mw * core.MW_PER_CM2_TO_W_PER_M2) / (epsilon_0 * c)
    dp = DynamicPolarizability(cache.atom, n, l, j)
    dp.defineBasis(5, 19)
    nu0 = core.transition_frequency_hz(cache.atom, 6, 0, 0.5, 6, 1, 0.5)
    for det_ghz in (30.0, 500.0):
        freq_hz = nu0 + det_ghz * 1e9
        cfg = {'enabled': True, 'intensity': I_mw,
               'frequency_hz': freq_hz, 'polarization': 'π'}
        group = dict(n=n, l=l, j=j, label='6S', color='r')
        ours = core.ac_stark_shift_mhz(cache, group, m, [], cfg)   # MHz @1mW
        alpha0 = dp.getPolarizability(c / freq_hz, units='SI', mj=m)[0]
        dp_mhz = 0.5 * alpha0 * E0_sq / 1e6                        # MHz @1mW
        ratio = ours / dp_mhz
        check_close(f'6S m=0.5 det={det_ghz:5.0f}GHz 比值=-0.5', ratio, -0.5, rtol=0.02)


def t9():
    print('T9 AC-Stark dressed 用于能级失谐（共振 → ±Ω/2，与显示一致）')
    cache = core.ArcCache(Rubidium87())
    B = 21.7
    groups = [dict(n=6, l=1, j=1.5, label='6P_{3/2}', color='r'),
              dict(n=60, l=0, j=0.5, label='60S_{1/2}', color='b')]
    rows_6p = core._group_sublevel_rows(cache, 6, 1, 1.5, B)
    rows_60s = core._group_sublevel_rows(cache, 60, 0, 0.5, B)
    b63 = next(r for r in rows_6p if r['qstr'] == 'F=3.0, m_F=+3.0')
    a60 = next(r for r in rows_60s if r['qstr'] == 'm_J=+0.5')
    freq_hz = (a60['E_mhz'] - b63['E_mhz']) * 1e6
    cfg = {'enabled': True, 'intensity': 10000.0,
           'frequency_hz': freq_hz, 'polarization': 'σ-'}
    rows = core.build_level_rows(cache, B, groups, cfg, groups, backend='legacy')
    r63 = next(r for r in rows if r['label'] == '6P_{3/2}'
               and r['qstr'] == 'F=3.0, m_F=+3.0')
    print(f'  6P3/2 F=3 mF=3 ac = {r63["ac"] * 1000:.1f} kHz')
    check('共振下 6P3/2 F=3 mF=3 位移 > 100 kHz（≈Ω/2，非 kHz 微扰）',
          abs(r63['ac']) > 0.1, r63['ac'])
    pairs = core.collect_transitions(cache, groups, cfg, B)
    p = min(pairs, key=lambda x: abs(x[4]))
    omega_hz = p[5]
    # collect_transitions 的 Rabi 与 AC 位移一致：ac = +Ω/2（源 6P3/2 低能）
    check_close('ac ≈ +Ω/2（Rabi 取自 collect_transitions）',
                r63['ac'], omega_hz / (2.0 * np.pi) / 1e6 / 2.0, rtol=0.05)


def t10():
    print('T10 F↔mJ 矩阵元 vs ARC getDipoleMatrixElementHFStoFS（含 mJ 匹配）')
    cache = core.ArcCache(Rubidium87())
    rows_6p = core._group_sublevel_rows(cache, 6, 1, 1.5, 0.0)
    rows_60s = core._group_sublevel_rows(cache, 60, 0, 0.5, 0.0)
    cases = [
        ('F=3.0, m_F=+3.0', 'm_J=+0.5', -1),   # σ- 合法（mJ 1.5→0.5）
        ('F=3.0, m_F=-1.0', 'm_J=+0.5', -1),   # σ- 禁忌（F=3 mF=-1 无 mJ=1.5）
        ('F=3.0, m_F=-1.0', 'm_J=-0.5', -1),   # σ- 合法（mJ 0.5→-0.5）
        ('F=3.0, m_F=+3.0', 'm_J=+0.5', +1),   # σ+ 禁忌（1.5→2.5 超范围）
    ]
    for qs_a, qs_b, q in cases:
        a = next(r for r in rows_6p if r['qstr'] == qs_a)
        b = next(r for r in rows_60s if r['qstr'] == qs_b)
        ours = core.transition_strength_q(cache, a, b, q)
        Fa = float(qs_a.split(',')[0].split('=')[1])
        mFa = float(qs_a.split('m_F=')[1])
        mjb = float(qs_b.split('m_J=')[1])
        arc = cache.atom.getDipoleMatrixElementHFStoFS(
            6, 1, 1.5, Fa, mFa, 60, 0, 0.5, mjb, q) ** 2
        check_close(f'{qs_a} → {qs_b} q={q:+d}', ours, arc, atol=1e-14)
    # 用户 bug 场景：σ- 下 F=3 mF=-1 ↔ 60S mJ=+0.5 必须为 0
    a = next(r for r in rows_6p if r['qstr'] == 'F=3.0, m_F=-1.0')
    b = next(r for r in rows_60s if r['qstr'] == 'm_J=+0.5')
    s = core.transition_strength_q(cache, a, b, -1)
    check('σ- 下 F=3 mF=-1 ↔ 60S mJ=+0.5 强度=0（禁忌）', s == 0.0, s)


def t11():
    print('T11 双频移后端：direct（矩阵对角化）vs legacy（解析/微扰）')
    cache = core.ArcCache(Rubidium87())
    check('SHIFT_BACKEND 默认 = direct', core.SHIFT_BACKEND == 'direct',
          core.SHIFT_BACKEND)
    groups = [dict(n=60, l=0, j=0.5, label='60S', color='r'),
              dict(n=60, l=1, j=0.5, label='60P1/2', color='b')]
    base = core.transition_frequency_hz(cache.atom, 60, 0, 0.5, 60, 1, 0.5)
    maxdev = 0.0
    n_pairs = 0
    for det in (-500.0, -200.0, 200.0, 500.0):
        cfg = dict(enabled=True, intensity=1e-4,
                   frequency_hz=base + det * 1e6, polarization='π')
        rows_d = core.build_level_rows(cache, 0.0, groups, cfg, groups,
                                       backend='direct')
        rows_l = core.build_level_rows(cache, 0.0, groups, cfg, groups,
                                       backend='legacy')
        check(f'T11 行数一致（direct/legacy，Δ={det:+.0f} MHz）',
              len(rows_d) == len(rows_l), (len(rows_d), len(rows_l)))
        for rd, rl in zip(rows_d, rows_l):
            if rd['qstr'] == rl['qstr']:
                maxdev = max(maxdev, abs(rd['ac'] - rl['ac']))
                n_pairs += 1
    check(f'T11 direct/legacy 位移一致（{n_pairs} 对，亚 MHz）',
          maxdev < 0.2, f'maxdev={maxdev:.4f} MHz')


def main():
    print('=' * 64)
    print('core.py 自检（手写算法 vs ARC 官方函数）')
    print('=' * 64)
    for fn in (t1, t2, t3, t4, t5, t6, t7, t8, t9, t10, t11):
        fn()
        print('-' * 64)
    print(f'汇总: PASS={_PASS}  FAIL={_FAIL}')
    sys.exit(1 if _FAIL else 0)


if __name__ == '__main__':
    main()
