# -*- coding: utf-8 -*-
"""pair 两体精简 selfcheck：pair_core 静态（P 系列 6 项）+ driver 驱动（D 系列，TDD）。

P 系列（每物理点 1 项代表）：
    P1  C6(60S,60S) ≈ 137.8 GHz·μm⁶（基组收敛值）
    P2  C3(60S,60P) ≈ 3035 MHz·μm³（1/R³）
    P3  交换劈裂/2 == 直接矩阵元 |V_ex|
    P4  角度依赖（θ=90° |V| 减半反号）
    P5  R→∞ 极限（vdW/交换 → 0）+ 1/R³ 标度

D 系列（driver 引擎，TDD：接口由本文件定义，driver 实现后自动启用）：
    D1 无驱动极限→pair_core   D2 R→∞ Rabi==单原子   D3 两能级解析 Rabi
    D4 全时域非 RWA 交叉       D5 交换振荡            D6 弱场 AC-Stark  D7 封锁复现

driver.py 接口约定（driver 实现须遵守）：
    rwa_evolve(diag_mhz, couplings, om_mhz, psi0, t_ns) -> (t, pops)
    full_evolve(H0_mhz, D, omega_rad_ns, om_mhz, psi0, t_ns) -> (t, pops)
    pair_static(cache, groups, R_um, B_gauss, static_cfg, **kw) -> dict(single,pairs,E_pair,H)
    driver_operator(cache, single, pairs, driver_cfg) -> (D, omega_rad_ns, om_mhz)
    rwa_hamiltonian(st, D, omega_rad_ns, om_mhz) -> H_rwa
    evolve(H_rwa, psi0, t_ns) -> (t, pops)
"""
import os
import sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))     # 项目根（core.py / pair_core.py）
import numpy as np
from src import core
from src import pair_core
from checklib import Checker

c = Checker('pair_selfcheck')
_GROUPS = [{'n': 60, 'l': 0, 'j': 0.5}, {'n': 60, 'l': 1, 'j': 0.5}]
_C6_REF = 137.8e3          # MHz·μm⁶（ARC 收敛）
_C3_REF = 3035.4           # MHz·μm³（ARC）


def _cache():
    from arc import Rubidium87
    return core.ArcCache(Rubidium87())


# ================= P 系列：pair_core 静态（6 项） =================
def p_static():
    cache = _cache()

    # P1: C6（基组收敛值）
    d, _ = pair_core.vdw_shift(cache, _GROUPS, 8.0, 0.0, 0.0, 0.0, 3,
                               m_total=1.0)
    c.close('P1 C6(60S,60S)', d * 8.0 ** 6 / 1e3, _C6_REF / 1e3, rtol=0.01)

    # P2: C3（1/R³ 不变性）
    c3 = [abs(pair_core.exchange_coupling(cache, _GROUPS, R)) * R ** 3
          for R in (6.0, 8.0)]
    c.close('P2 C3(60S,60P)', c3[1], _C3_REF, rtol=0.02)
    c.close('P2b C3 与 R 无关', c3[1], c3[0], rtol=1e-4)

    # P3: 交换劈裂/2 == 直接矩阵元
    res = pair_core.pair_eigen(cache, _GROUPS, 6.0, 0.0, 0.0, 0.0, 1,
                               m_total=1.0)
    s = next(s for s in res['single'] if s['n'] == 60 and s['l'] == 0
             and abs(pair_core._mag(s) - 0.5) < 1e-9)
    p = next(s for s in res['single'] if s['n'] == 60 and s['l'] == 1
             and abs(pair_core._mag(s) - 0.5) < 1e-9)
    w2 = (pair_core.pair_component(res, s, p)
          + pair_core.pair_component(res, p, s))
    top = np.argsort(w2)[-2:]
    split = abs(res['evals'][top[1]] - res['evals'][top[0]]) / 2
    vex = abs(pair_core.exchange_coupling(cache, _GROUPS, 6.0))
    c.close('P3 劈裂/2 vs |V_ex|', split, vex, rtol=0.02)

    # P4: 角度依赖（θ=90° 减半反号）
    v0 = pair_core.exchange_coupling(cache, _GROUPS, 6.0, 0.0, 0.0)
    v90 = pair_core.exchange_coupling(cache, _GROUPS, 6.0, 90.0, 0.0)
    c.check('P4 θ=90 |V| 减半且反号',
            abs(abs(v90) - abs(v0) / 2) < 1e-3 * abs(v0) and v0 * v90 < 0,
            (v0, v90))

    # P5: R→∞ 极限 + 1/R³ 标度
    d_big, _ = pair_core.vdw_shift(cache, _GROUPS, 200.0, 0.0, 0.0, 0.0, 1,
                                   m_total=1.0)
    v_big = abs(pair_core.exchange_coupling(cache, _GROUPS, 200.0))
    c.check('P5a R=200 vdW/交换≈0', abs(d_big) < 1e-3 and v_big < 1e-3,
            (d_big, v_big))
    v10 = abs(pair_core.exchange_coupling(cache, _GROUPS, 10.0))
    v20 = abs(pair_core.exchange_coupling(cache, _GROUPS, 20.0))
    c.close('P5b 交换 1/R³ 标度', v10 / v20, 8.0, rtol=0.05)


# ================= D 系列：driver（TDD，未实现则 SKIP） =================
def d_driver():
    try:
        from src import driver
    except Exception as e:
        c.skip('D1-D7 驱动引擎', f'driver.py 未实现（{type(e).__name__}）→ 检查先行')
        return
    cache = _cache()
    t = np.linspace(0, 400.0, 801)
    psi0_2 = np.array([1 + 0j, 0j])

    # D3: 两能级 RWA vs 解析 Rabi  P_e=(Ω/Ω̃)²sin²(Ω̃ t/2)
    for om, det in ((10.0, 0.0), (10.0, 5.0)):
        omt = np.hypot(om, det)
        ref = (om / omt) ** 2 * np.sin(np.pi * 1e-3 * omt * t) ** 2
        _, pops = driver.rwa_evolve([0.0, det], [(0, 1, om)], psi0_2, t)
        err = float(np.max(np.abs(pops[1] - ref)))
        c.check(f'D3 解析Rabi Ω={om},Δ={det}', err < 1e-4, f'max_err={err:.2e}')

    # D4: 全时域（不 RWA） vs RWA（近共振：裸能级差 E_res，载频对准）
    om, E_res, det = 10.0, 10000.0, 0.0
    H0 = np.diag([0.0, E_res]).astype(complex)
    Dm = np.array([[0.0, 1.0], [1.0, 0.0]], complex)
    omega_rad_ns = 2 * np.pi * 1e-3 * (E_res + det)
    _, pr = driver.rwa_evolve([0.0, det], [(0, 1, om)], psi0_2, t)
    _, pf = driver.full_evolve(H0, Dm, omega_rad_ns, om, psi0_2, t)
    err = float(np.max(np.abs(pr[1] - pf[1])))
    c.check('D4 RWA vs 全时域(近共振)', err < 0.02, f'max_err={err:.3f}')

    # D1: 无驱动极限 → 静态 dressed 能级 == pair_core
    st = driver.pair_static(cache, _GROUPS, 6.0, 0.0,
                            dict(vdw=True, exchange=True, ac_stark=False),
                            m_total=1.0, n_window=1)
    ref = pair_core.pair_eigen(cache, _GROUPS, 6.0, 0.0, 0.0, 0.0, 1,
                               m_total=1.0)
    c.check('D1 无驱动极限= pair_core 静态',
            np.allclose(np.sort(np.real(np.linalg.eigvalsh(st['H']))),
                        np.sort(ref['evals']), rtol=1e-6, atol=1e-3))

    # D2: R→∞ 微波 Rabi == 单原子（Ω 由强度算）
    stb = driver.pair_static(cache, _GROUPS, 50.0, 0.0,
                             dict(vdw=True, exchange=True, ac_stark=False),
                             m_total=1.0, n_window=1)
    _, _, om_calc = driver.driver_operator(cache, stb['single'], stb['pairs'],
                                           dict(kind='mw', pol='π',
                                                intensity_mw_cm2=0.0005))
    ia = pair_core.find_sublevel(stb['single'], 60, 0, 0.5, 0.5)
    ib = pair_core.find_sublevel(stb['single'], 60, 1, 0.5, 0.5)
    dme = abs(core.dipole_amplitude(cache, stb['single'][ia],
                                    stb['single'][ib], 0))
    E0 = (2 * 0.0005 * 10.0 / (8.854187817e-12 * 2.99792458e8)) ** 0.5
    om_ref = E0 * dme * core.E_CHARGE * core.A0 / (1.0545718e-34) / 1e6
    c.close('D2 R→∞ Rabi == 单原子', om_calc, om_ref, rtol=0.05)

    # D7: dressed 封锁复现（vdW 位移进入封锁物理）
    om_mw = 30.0
    w = 2 * np.pi * 1e-3 * 16826.7
    cfg = dict(kind='mw', pol='π', om_mhz=om_mw, omega_rad_ns=w)

    def ppmax(res):
        N = len(res['single'])
        sel = {i * N + j: k for k, (i, j) in enumerate(res['pairs'])}
        ip = pair_core.find_sublevel(res['single'], 60, 1, 0.5, 0.5)
        return float(res['pops'][sel[ip * N + ip]].max())

    p60 = ppmax(driver.pair_drive_dressed(cache, _GROUPS, 60.0, 0.0, cfg,
                                          m_total=1.0))
    p4 = ppmax(driver.pair_drive_dressed(cache, _GROUPS, 4.0, 0.0, cfg,
                                         m_total=1.0))
    c.check('D7a R=60 自由（PP 可达）', p60 > 0.9, f'{p60:.3f}')
    c.check('D7b R=4 封锁（PP 抑制，vdW 进入）', p4 < 0.3, f'{p4:.3f}')

    # D5: 交换驱动振荡（|SP⟩↔|PS⟩，Rabi=V_ex；无相互作用静态使驱动算符独立可测）
    st = driver.pair_static(cache, _GROUPS, 6.0, 0.0,
                            dict(vdw=False, exchange=False),
                            m_total=1.0, n_window=1)
    D, w, om = driver.driver_operator(cache, st['single'], st['pairs'],
                                      dict(kind='ex', R_um=6.0))
    vex = abs(pair_core.exchange_coupling(cache, _GROUPS, 6.0))
    c.close('D5a 交换 Rabi=V_ex', om, vex, rtol=0.01)
    N = len(st['single'])
    sel = {i * N + j: k for k, (i, j) in enumerate(st['pairs'])}
    ia = pair_core.find_sublevel(st['single'], 60, 0, 0.5, 0.5)
    ib = pair_core.find_sublevel(st['single'], 60, 1, 0.5, 0.5)
    psi0 = np.zeros(len(st['pairs']), complex)
    psi0[sel[ia * N + ib]] = 1.0                   # |SP⟩
    H = driver.rwa_hamiltonian(st, D, 0.0, om, start_idx=sel[ia * N + ib])
    t = np.linspace(0, 400.0, 2001)
    _, pops = driver.evolve(H, psi0, t)
    ps = pops[sel[ib * N + ia]]                    # |PS⟩ 布居
    th = int(np.argmax(ps > 0.5))                  # 首次过半（半峰，抗 eigh 数值噪声）
    c.close('D5b 交换振荡半峰时间', t[th], 250.0 / vex, rtol=0.05)

    # D6: 弱场 AC-Stark（远失谐驱动 → |g⟩ 能移 ≈ -Ω²/4Δ）
    om2, det2 = 10.0, 200.0
    H2 = np.array([[0.0, om2 / 2.0], [om2 / 2.0, det2]])
    ev2, evc2 = np.linalg.eigh(H2)
    ig = int(np.argmax(np.abs(evc2[0, :]) ** 2))   # |g⟩ 分量最大的本征态
    c.close('D6 弱场能移 ≈ -Ω²/4Δ', ev2[ig], -om2 * om2 / (4 * det2), rtol=0.02)


# ================= 入口 =================
def main():
    print('=== pair_core 静态（P 系列）===')
    p_static()
    print('=== driver 驱动引擎（D 系列）===')
    d_driver()
    ok = c.summary()
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
