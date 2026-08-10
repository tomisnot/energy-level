# -*- coding: utf-8 -*-
"""统一计算层 selfcheck（U 系列）：build_system + drive + response_spectrum。

核心命题：单体/两体共用同一引擎，仅影响因素不同。U 系列逐项验证统一接口：
    U1  单体寻址光驱动 vs 解析两能级 Rabi（6P3/2 F=3 mF=3 ↔ 60S mJ=+0.5）
    U2  单体微波能移进 H == ac_stark_shift_mhz 逐态
    U3  build_system(n_body=2).single/H == pair_core.pair_hamiltonian 直接调用
    U4  两体 drive == pair_drive_dressed 直接调用
    U5  response_spectrum 扫频在共振处初态流失最大
    U6  无相互作用极限：两体 P(PP) == [单原子 Rabi]²（sin⁴）
"""
import os
import sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))
import numpy as np
from src import core
from src import driver
from src import pair_core
from checklib import Checker

ch = Checker('unified_selfcheck')


def _cache():
    from arc import Rubidium87
    return core.ArcCache(Rubidium87())


def u_single_drive():
    """U1: 单体寻址光驱动 vs 解析 Rabi。"""
    cache = _cache()
    groups = [dict(n=6, l=1, j=1.5), dict(n=60, l=0, j=0.5),
              dict(n=60, l=1, j=0.5)]
    rows_6p = core._group_sublevel_rows(cache, 6, 1, 1.5, 0.0)
    rows_60s = core._group_sublevel_rows(cache, 60, 0, 0.5, 0.0)
    b63 = next(r for r in rows_6p if r['qstr'] == 'F=3.0, m_F=+3.0')
    a60 = next(r for r in rows_60s if r['qstr'] == 'm_J=+0.5')
    f_ad = (a60['E_mhz'] - b63['E_mhz']) * 1e6
    om = 8.0   # MHz
    ad = dict(kind='ad', group_a=dict(n=6, l=1, j=1.5),
              group_b=dict(n=60, l=0, j=0.5), pol='σ-', om_mhz=om,
              omega_rad_ns=2 * np.pi * 1e-3 * f_ad / 1e6)
    sys0 = driver.build_system(cache, groups, 1)
    t_ns = np.linspace(0, 125.0, 501)          # 两个 π 周期
    res = driver.drive(sys0, ad, t_ns=t_ns, psi0_ket=(6, 1, 1.5, 3.0))
    i60 = next(k for k, s in enumerate(res['single'])
               if s['n'] == 60 and s['l'] == 0 and s['qstr'] == 'm_J=+0.5')
    pb = res['pops'][i60]
    om_rad = 2 * np.pi * 1e-3 * om             # MHz → rad/ns
    pb_ref = np.sin(om_rad * t_ns / 2) ** 2
    err = np.max(np.abs(pb - pb_ref))
    ch.check('U1 单体寻址光 Rabi == 解析 sin²(Ωt/2)', err < 2e-3,
             f'max|Δ|= {err:.2e} @ Ω={om} MHz')
    # 归一化 + 只有 i0/i60 两态参与（其余态布居≈0）
    norm = np.sum(res['pops'], axis=0)
    ch.check('U1b 每时刻总布居归一', np.max(np.abs(norm - 1.0)) < 1e-9,
             np.max(np.abs(norm - 1.0)))
    i0 = next(k for k, s in enumerate(res['single'])
              if s['qstr'] == 'F=3.0, m_F=+3.0')
    other = np.delete(res['pops'], [i0, i60], axis=0)
    ch.check('U1c 除驱动对两态外无布居', np.max(other) < 1e-9, np.max(other))


def u_ac_stark_into_H():
    """U2: 单体微波能移进 H == ac_stark_shift_mhz 逐态。"""
    cache = _cache()
    groups = [dict(n=60, l=0, j=0.5), dict(n=60, l=1, j=0.5)]
    f_mw0 = core.transition_frequency_hz(cache.atom, 60, 0, 0.5, 60, 1, 0.5)
    # 远失谐 20GHz（≫ 60S↔60P Rabi ~GHz），保证纯二阶能移区
    mw = dict(frequency_hz=f_mw0 + 20e9, intensity=200.0, polarization='π',
              enabled=True)
    s1 = driver.build_system(cache, groups, 1, static_cfg=dict(ac_stark=mw))
    s0 = driver.build_system(cache, groups, 1)
    d0 = s1['H'].diagonal().real - s0['H'].diagonal().real
    # 对 60S mJ=+0.5 对照直接 ac_stark_shift_mhz（同一 cfg，含 enabled）
    s = next(x for x in s1['single']
             if x['n'] == 60 and x['l'] == 0 and x['qstr'] == 'm_J=+0.5')
    g = next(g for g in groups if (g['n'], g['l'], g['j']) == (60, 0, 0.5))
    ref = core.ac_stark_shift_mhz(cache, g, 0.5, groups, mw, 0.0)
    i = next(k for k, x in enumerate(s1['single']) if x is s)
    ch.close('U2 微波能移入 H == ac_stark_shift_mhz', d0[i], ref,
             rtol=1e-9, atol=1e-6)
    # 位移应显著且远小于失谐（纯远失谐能移 ≈ Ω²/4Δ）
    ch.check('U2b 位移显著且 ≪ 失谐（纯远失谐能移）',
             1.0 < abs(d0[i]) < 500.0, d0[i])


def u_build_pair_consistency():
    """U3: build_system(n_body=2) == pair_core.pair_hamiltonian 直接调用。"""
    cache = _cache()
    groups = [dict(n=60, l=0, j=0.5), dict(n=60, l=1, j=0.5)]
    R = 6.0
    sys2 = driver.build_system(cache, groups, 2, R_um=R, m_total=1.0)
    res = pair_core.pair_hamiltonian(cache, groups, R, 0.0, 0.0, 0.0, 3,
                                     m_total=1.0)
    same_single = all(s == t for s, t in zip(sys2['single'], res['single']))
    same_pairs = np.array_equal(sys2['states'], res['pairs'])
    same_H = np.allclose(sys2['H'], res['H'], atol=1e-9)
    ch.check('U3 build_system(2).single/H == pair_hamiltonian',
             same_single and same_pairs and same_H,
             (same_single, same_pairs, same_H))


def u_pair_drive_vs_dressed():
    """U4: 两体 drive == pair_drive_dressed 直接调用。"""
    cache = _cache()
    groups = [dict(n=60, l=0, j=0.5), dict(n=60, l=1, j=0.5)]
    R = 6.0
    mw = dict(kind='mw', group_a=dict(n=60, l=0, j=0.5),
              group_b=dict(n=60, l=1, j=0.5), pol='π', m_a=0.5,
              om_mhz=8.0, omega_rad_ns=2 * np.pi * 1e-3 *
              (core.transition_frequency_hz(cache.atom, 60, 0, 0.5,
                                            60, 1, 0.5) / 1e6))
    t_ns = np.linspace(0, 200.0, 501)
    sys2 = driver.build_system(cache, groups, 2, R_um=R, m_total=1.0)
    res_u = driver.drive(sys2, mw, t_ns=t_ns, psi0_ket=(60, 0, 0.5, 0.5))
    res_d = driver.pair_drive_dressed(cache, groups, R, 0.0, mw, None,
                                      (60, 0, 0.5, 0.5), t_ns, m_total=1.0)
    same_pops = np.allclose(res_u['pops'], res_d['pops'], atol=1e-9)
    ch.check('U4 两体 drive == pair_drive_dressed', same_pops,
             np.max(np.abs(res_u['pops'] - res_d['pops'])))
    # 驱动有实际效果（非零激发）
    ch.check('U4b 两体 drive 有布居转移', np.max(res_u['pops'][0]) < 0.999,
             np.max(res_u['pops'][0]))


def u_response_spectrum():
    """U5: response_spectrum 扫频在共振处初态流失最大。"""
    cache = _cache()
    groups = [dict(n=6, l=1, j=1.5), dict(n=60, l=0, j=0.5),
              dict(n=60, l=1, j=0.5)]
    rows_6p = core._group_sublevel_rows(cache, 6, 1, 1.5, 0.0)
    rows_60s = core._group_sublevel_rows(cache, 60, 0, 0.5, 0.0)
    b63 = next(r for r in rows_6p if r['qstr'] == 'F=3.0, m_F=+3.0')
    a60 = next(r for r in rows_60s if r['qstr'] == 'm_J=+0.5')
    f_ad = (a60['E_mhz'] - b63['E_mhz']) * 1e6
    om = 8.0
    ad = dict(kind='ad', group_a=dict(n=6, l=1, j=1.5),
              group_b=dict(n=60, l=0, j=0.5), pol='σ-', om_mhz=om)
    sys0 = driver.build_system(cache, groups, 1)
    t_ns = np.linspace(0, 62.5, 201)          # π/2 附近：共振时流失最大
    scan = f_ad + np.linspace(-3.0, 3.0, 31) * 1e6
    freqs, resp = driver.response_spectrum(sys0, ad, scan, t_ns=t_ns,
                                           psi0_ket=(6, 1, 1.5, 3.0))
    imax = int(np.argmax(resp))
    ch.close('U5 共振频率 == f_ad', freqs[imax], f_ad, rtol=0.02)
    ch.check('U5b 共振处初态流失大（>0.9）', resp[imax] > 0.9, resp[imax])


def u_independent_rabi_limit():
    """U6: 无相互作用极限（R 大）两体 P(PP) == [单原子 Rabi]²。"""
    cache = _cache()
    groups = [dict(n=60, l=0, j=0.5), dict(n=60, l=1, j=0.5)]
    om = 8.0
    mw = dict(kind='mw', group_a=dict(n=60, l=0, j=0.5),
              group_b=dict(n=60, l=1, j=0.5), pol='π', m_a=0.5, om_mhz=om,
              omega_rad_ns=2 * np.pi * 1e-3 *
              (core.transition_frequency_hz(cache.atom, 60, 0, 0.5,
                                            60, 1, 0.5) / 1e6))
    t_ns = np.linspace(0, 200.0, 401)
    R = 100.0                                  # vdW/交换 ~ 可忽略
    sys2 = driver.build_system(cache, groups, 2, R_um=R, m_total=1.0)
    res = driver.drive(sys2, mw, t_ns=t_ns, psi0_ket=(60, 0, 0.5, 0.5))
    # PP 对索引（60P mJ=+0.5 ⊗ 60P mJ=+0.5）
    ip = next(k for k, s in enumerate(res['single'])
              if s['n'] == 60 and s['l'] == 1 and s['qstr'] == 'm_J=+0.5')
    N = len(res['single'])
    pp = next(k for k, (i, j) in enumerate(res['pairs'])
              if i == ip and j == ip)
    ppp = res['pops'][pp]
    om_rad = 2 * np.pi * 1e-3 * om            # MHz → rad/ns
    p_ref = np.sin(om_rad * t_ns / 2) ** 4     # 两独立原子同时激发
    err = np.max(np.abs(ppp - p_ref))
    ch.check('U6 R→∞ P(PP) == sin⁴(Ωt/2)', err < 2e-3,
             f'max|Δ|={err:.2e}')


def u_single_atom_addressing():
    """U7: 单原子寻址（scope='atom1'）：只驱动原子1，原子2 不动。"""
    cache = _cache()
    groups = [dict(n=60, l=0, j=0.5), dict(n=60, l=1, j=0.5)]
    om = 8.0
    mw = dict(kind='mw', group_a=dict(n=60, l=0, j=0.5),
              group_b=dict(n=60, l=1, j=0.5), pol='π', m_a=0.5, om_mhz=om,
              scope='atom1', omega_rad_ns=2 * np.pi * 1e-3 *
              (core.transition_frequency_hz(cache.atom, 60, 0, 0.5,
                                            60, 1, 0.5) / 1e6))
    t_ns = np.linspace(0, 200.0, 401)
    R = 100.0
    sys2 = driver.build_system(cache, groups, 2, R_um=R, m_total=1.0)
    res = driver.drive(sys2, mw, t_ns=t_ns, psi0_ket=(60, 0, 0.5, 0.5))
    single = res['single']
    is_ = next(k for k, s in enumerate(single)
               if s['n'] == 60 and s['l'] == 0 and s['qstr'] == 'm_J=+0.5')
    ip = next(k for k, s in enumerate(single)
              if s['n'] == 60 and s['l'] == 1 and s['qstr'] == 'm_J=+0.5')
    idx = {k: (i, j) for k, (i, j) in enumerate(res['pairs'])}
    p1p = sum(res['pops'][k] for k, (i, j) in idx.items() if i == ip)
    p2p = sum(res['pops'][k] for k, (i, j) in idx.items() if j == ip)
    ch.check('U7 原子1 被完全驱动（P1P→1）', np.max(p1p) > 0.99,
             np.max(p1p))
    # 原子2 不动的残差：dressed 投影 + eigh 数值噪声（R=100 时 vdW/交换
    # ~1e-3 MHz，投影残差 ~1e-6 量级），阈值 1e-4 为物理不动
    ch.check('U7b 原子2 不动（P2P≈0）', np.max(p2p) < 1e-4,
             np.max(p2p))
    # 原子1 的 Rabi 周期 == 单原子（首个 P 峰在 π/Ω）
    t_pi = t_ns[np.argmax(p1p)]
    ch.close('U7c 原子1 Rabi 周期 == π/Ω', t_pi, 62.5, rtol=0.02)


def u_linear_spectrum_main_peak():
    """U8: 单体线性响应谱主峰 = 寻址光共振 f_ad，强度 == 真实 |d|²。"""
    cache = _cache()
    groups = [dict(n=6, l=1, j=1.5), dict(n=60, l=0, j=0.5),
              dict(n=60, l=1, j=0.5)]
    rows_6p = core._group_sublevel_rows(cache, 6, 1, 1.5, 0.0)
    rows_60s = core._group_sublevel_rows(cache, 60, 0, 0.5, 0.0)
    b63 = next(r for r in rows_6p if r['qstr'] == 'F=3.0, m_F=+3.0')
    a60 = next(r for r in rows_60s if r['qstr'] == 'm_J=+0.5')
    f_ad = (a60['E_mhz'] - b63['E_mhz']) * 1e6
    ad = dict(kind='ad', group_a=dict(n=6, l=1, j=1.5),
              group_b=dict(n=60, l=0, j=0.5), pol='σ-', om_mhz=8.0)
    sys0 = driver.build_system(cache, groups, 1)
    lin = driver.response_spectrum_linear(sys0, ad,
                                          psi0_ket=(6, 1, 1.5, 3.0))
    imax = int(np.argmax(lin['strength']))
    ch.close('U8 线性谱主峰 == f_ad', lin['freq_hz'][imax], f_ad, rtol=0.02)
    # 强度 == 真实 |⟨60S+0.5|d_{-1}|F3mF3⟩|²
    d2 = abs(core.dipole_amplitude(cache, b63, a60, -1)) ** 2
    ch.close('U8b 主峰强度 == |d|²', lin['strength'][imax], d2, rtol=0.02)
    # 主峰显著强于次峰（物理：σ- 从 F3mF3 主要只到 60S+0.5）
    second = np.sort(lin['strength'])[-2]
    ch.check('U8c 主峰强度 ≫ 次峰', lin['strength'][imax] > 20 * second,
             (lin['strength'][imax], second))


def u_linear_vs_time_evolution():
    """U9: 线性谱主峰 vs 逐频率含时演化共振峰位一致（两种方法交叉）。"""
    cache = _cache()
    groups = [dict(n=6, l=1, j=1.5), dict(n=60, l=0, j=0.5),
              dict(n=60, l=1, j=0.5)]
    rows_6p = core._group_sublevel_rows(cache, 6, 1, 1.5, 0.0)
    rows_60s = core._group_sublevel_rows(cache, 60, 0, 0.5, 0.0)
    b63 = next(r for r in rows_6p if r['qstr'] == 'F=3.0, m_F=+3.0')
    a60 = next(r for r in rows_60s if r['qstr'] == 'm_J=+0.5')
    f_ad = (a60['E_mhz'] - b63['E_mhz']) * 1e6
    om = 8.0
    ad = dict(kind='ad', group_a=dict(n=6, l=1, j=1.5),
              group_b=dict(n=60, l=0, j=0.5), pol='σ-', om_mhz=om)
    sys0 = driver.build_system(cache, groups, 1)
    lin = driver.response_spectrum_linear(sys0, ad,
                                          psi0_ket=(6, 1, 1.5, 3.0))
    f_lin = lin['freq_hz'][int(np.argmax(lin['strength']))]
    # 逐频率含时（π/2 处初态流失，11 点粗扫即可）
    t_ns = np.linspace(0, 62.5, 201)
    scan = f_ad + np.linspace(-3.0, 3.0, 11) * 1e6
    freqs, resp = driver.response_spectrum(sys0, ad, scan, t_ns=t_ns,
                                           psi0_ket=(6, 1, 1.5, 3.0))
    f_te = freqs[int(np.argmax(resp))]
    ch.close('U9 线性谱 vs 含时演化峰位一致', f_lin, f_te, rtol=0.03)


def main():
    u_single_drive()
    u_ac_stark_into_H()
    u_build_pair_consistency()
    u_pair_drive_vs_dressed()
    u_response_spectrum()
    u_independent_rabi_limit()
    u_single_atom_addressing()
    u_linear_spectrum_main_peak()
    u_linear_vs_time_evolution()
    ok = ch.summary()
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
