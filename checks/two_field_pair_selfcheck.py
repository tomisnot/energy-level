# -*- coding: utf-8 -*-
"""两体两场引擎自检（evolve_sambe_pair.build_two_field_pair + evolve_two_field_pair）。

场景（TP 系列）：
  TP1 无光场退化：mw_M=2 时 ≡ evolve_sambe_pair（同 groups/E/M，共传播核心）
  TP2 直积极限：V_dd=0 + 仅 A 侧微波 → 约化 A ≡ 单原子两场引擎逐点一致
  TP3 光场旋转系（第一性对拍）：人工 THz 光场 + 微波 + V_dd（含"6P 占据数
      变化"元素）≡ 两体实验室系稠密 DOP853 参考（短窗硬积分，无旋转系/
      无 RWA/无光子块）；并核对 vdd_rot_filter 的丢弃统计
  TP4 拍频拒绝：多光场 δ≠0 → NotImplementedError（静默错不如显式拒绝）
  TP5 寿命耗散：NonHermitian 通道 Σpops+loss=1

TP3 是本引擎的核心验收：旋转系吸收光场载波 + 微波-only 光子块 + V_dd 旋转
筛选的全部构造，与不做任何变换的第一性参考对拍。

运行：D:\\python310\\python.exe -u checks\\two_field_pair_selfcheck.py
"""
import os
import sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import scipy.sparse as sp
from scipy.integrate import solve_ivp

from checklib import Checker

ck = Checker('two_field_pair_selfcheck')

_CONV = 2 * np.pi * 1e-3


def _mJ_state(n, l, j, m, E, label):
    return dict(n=n, l=l, j=j, kind='mJ', mj=m, m=m, E_mhz=float(E),
                label=label, index=0, amps=None, F=None, mF=None)


# ============================================================
# TP1 无光场退化 ≡ evolve_sambe_pair（同 M、同 groups）
# ============================================================
def tp1():
    from src.core.sambe import evolve_sambe_pair
    from src.core.evolve_twofield import (build_two_field_pair,
                                          evolve_two_field_pair)
    states = [_mJ_state(60, 0, 0.5, -0.5, 0.0, 'g'),
              _mJ_state(60, 0, 0.5, +0.5, 1000.0, 'e')]
    N = 2
    A_drv = np.zeros((N, N))
    A_drv[0, 1] = A_drv[1, 0] = 20.0
    env = lambda tt: np.exp(-((np.asarray(tt, float) - 100.0) ** 2)
                            / (2 * 30.0 ** 2))
    meta_A = dict(N=N, E_rot=np.array([0.0, 1000.0]), opt_terms=[],
                  mw_groups=[dict(omega_mhz=1000.0, As=[A_drv],
                                  envs=[env], labels=['drv'])])
    meta_B = dict(N=N, E_rot=np.array([0.0, 1000.0]), opt_terms=[],
                  mw_groups=[])
    V = sp.lil_matrix((N * N, N * N), dtype=complex)
    V[0 * N + 1, 1 * N + 0] = 10.0
    V[1 * N + 0, 0 * N + 1] = 10.0
    V = V.tocsr()
    t = np.linspace(0, 200.0, 101)
    fldA = [dict(kind='exact', A=A_drv, omega_L_mhz=1000.0,
                 waveform=env, label='drv')]
    r_ex = evolve_sambe_pair(states, fldA, [], 0, 0, t,
                             vdd_matrix=V, tol=1e-7)
    # M 对齐：evolve_sambe_pair 用 M 探针（本场景 r=0.02 → M=3），两场侧
    # 显式传同一 M，保证对比的是传播而非截断阶数
    pb = r_ex['error_report']['photon_blocks']
    M_ex = (next(iter(pb.values())) - 1) // 2
    r_tf = evolve_two_field_pair(states, meta_A, meta_B, 0, 0, t,
                                 vdd_matrix=V, tol=1e-7, mw_M=M_ex)
    d = float(np.max(np.abs(r_tf['pops_2d'] - r_ex['pops_2d'])))
    ck.check(f'TP1 无光场退化 ≡ evolve_sambe_pair（M={M_ex}，'
             f'Δ={d:.2e} ≤ 1e-9）', d <= 1e-9, f'Δ={d}')
    ck.check('TP1b 契约：method/backend',
             r_tf['method'] == 'two_field'
             and r_tf['backend'] == 'sambe-pair-twofield'
             and r_tf['error_report']['photon_blocks']
             == {'1000MHz': 2 * M_ex + 1},
             f"{r_tf['method']}/{r_tf['backend']}/"
             f"{r_tf['error_report']['photon_blocks']}")
    return states, meta_A, meta_B, env, A_drv


# ============================================================
# TP2 V_dd=0 直积 ≡ 单原子两场引擎
# ============================================================
def tp2(states, meta_A, meta_B):
    from src.core.evolve_twofield import evolve_two_field_pair
    from src.core.evolve_twofield import evolve_two_field
    t = np.linspace(0, 200.0, 101)
    # 两侧都用解析判据选 M（单原子两场与两体两场同一公式 → 同 M）
    r_pair = evolve_two_field_pair(states, meta_A, meta_B, 0, 0, t,
                                   vdd_matrix=np.zeros((4, 4)), tol=1e-7,
                                   mw_M=None)
    r_single = evolve_two_field(states, meta_A, 0, t, tol=1e-7)
    dA = float(np.max(np.abs(r_pair['pops_A'] - r_single['pops'])))
    dB = float(np.max(np.abs(r_pair['pops_B'][:, 0] - 1.0)))
    ck.check(f'TP2a V_dd=0：约化 A ≡ 单原子两场（Δ={dA:.2e} ≤ 1e-6）',
             dA <= 1e-6, f'Δ={dA}')
    # B 侧完全无场，偏差为 tol 量级传播噪声累积（同 EP1b 理由，放宽）
    ck.check(f'TP2b V_dd=0：B 恒初态（Δ={dB:.2e} ≤ 1e-5）', dB <= 1e-5,
             f'Δ={dB}')


# ============================================================
# TP3 光场旋转系 + V_dd ≡ 实验室系两体稠密 DOP853（第一性对拍）
# ============================================================
def tp3():
    from src.core.evolve_twofield import (build_two_field_pair,
                                            evolve_two_field_pair)
    # 人工三态：L（低能流形，被光场旋转）、S、P（里德堡，微波驱动）
    W_OPT, DET, W_MW = 3.0e5, 200.0, 2000.0     # MHz
    E = np.array([0.0, W_OPT, W_OPT + W_MW])
    states = [_mJ_state(70, 1, 1.5, -0.5, E[0], 'L'),
              _mJ_state(60, 0, 0.5, -0.5, E[1], 'S'),
              _mJ_state(60, 1, 0.5, -0.5, E[2], 'P')]
    N = 3
    # 光场：L↔S，全 Rabi Ω=400（旋转系 meta 内为 RWA 共振项 ½Ω=200）
    OMEGA_OPT = 400.0
    A_opt_full = np.zeros((N, N))
    A_opt_full[0, 1] = A_opt_full[1, 0] = OMEGA_OPT
    A_opt_meta = 0.5 * A_opt_full
    env_opt = lambda tt: np.exp(-((np.asarray(tt, float) - 2.0) ** 2)
                                / (2 * 1.0 ** 2))
    # 微波：S↔P，全 Rabi Ω=100（Sambe 惯例 A=全 Rabi，Kc 内 ½）
    OMEGA_MW = 100.0
    A_mw = np.zeros((N, N))
    A_mw[1, 2] = A_mw[2, 1] = OMEGA_MW
    env_mw = lambda tt: np.exp(-((np.asarray(tt, float) - 3.0) ** 2)
                               / (2 * 1.0 ** 2))

    rot = E + (W_OPT + DET) * np.array([1.0, 0.0, 0.0])
    # ⚠ 旋转参考 = 光场**载频**（carrier），非跃迁频率——build_two_field 的
    # ref_add 用 w.freq_mhz；写错会把旋转系失谐平移一个 DET。
    meta = dict(N=N, E_rot=rot,
                opt_terms=[(A_opt_meta, env_opt, W_OPT + DET, 'W',
                            W_OPT + DET, np.array([1, 0, 0]))],
                mw_groups=[dict(omega_mhz=W_MW, As=[A_mw], envs=[env_mw],
                                labels=['mw'])])
    t = np.linspace(0, 5.0, 51)
    psi0 = 1                                       # |S⟩⊗|S⟩

    # V_dd：保留类（|SP⟩↔|PS⟩ 共振交换；|SL⟩↔|PL⟩ 失谐 2 GHz）
    #       + 丢弃类（|SS⟩↔|LS⟩：实验室系失谐 3e5，两侧旋转流形占据数差 1）
    V = sp.lil_matrix((N * N, N * N), dtype=complex)
    V[1 * N + 2, 2 * N + 1] = 8.0                  # |S,P⟩↔|P,S⟩
    V[2 * N + 1, 1 * N + 2] = 8.0
    V[1 * N + 0, 2 * N + 0] = 8.0                  # |S,L⟩↔|P,L⟩
    V[2 * N + 0, 1 * N + 0] = 8.0
    V[1 * N + 1, 0 * N + 1] = 4.0                  # |S,S⟩↔|L,S⟩（应被 RWA 丢弃）
    V[0 * N + 1, 1 * N + 1] = 4.0
    V = V.tocsr()

    log_lines = []
    r = evolve_two_field_pair(states, meta, meta, psi0, psi0, t,
                              vdd_matrix=V, tol=1e-7,
                              log=lambda *a: log_lines.append(' '.join(a)))
    info = r['error_report']['vdd_rwa']
    ck.check(f"TP3a V_dd 旋转筛选统计：保留 {info['n_keep']} 丢弃 "
             f"{info['n_drop']}（期望 4/2），净效应 "
             f"{info['max_effect_mhz']:.2e} ≤ 1e-4 MHz",
             info['n_keep'] == 4 and info['n_drop'] == 2
             and info['max_effect_mhz'] <= 1e-4, f'{info}')

    # ---- 实验室系两体稠密参考（无旋转系/无 RWA/无光子块，DOP853 硬积分）----
    E_pair = np.array([E[i] + E[j] for i in range(N) for j in range(N)])
    Vd = V.toarray()
    psi0_2d = np.zeros(N * N, complex)
    psi0_2d[psi0 * N + psi0] = 1.0
    A_opt_2d = np.kron(A_opt_full, np.eye(N)) + np.kron(np.eye(N), A_opt_full)
    A_mw_2d = np.kron(A_mw, np.eye(N)) + np.kron(np.eye(N), A_mw)

    def rhs(tt, y):
        H = np.diag(E_pair) + Vd \
            + env_opt(tt) * np.cos(_CONV * (W_OPT + DET) * tt) * A_opt_2d \
            + env_mw(tt) * np.cos(_CONV * W_MW * tt) * A_mw_2d
        return -1j * _CONV * (H @ y)

    sol = solve_ivp(rhs, (0.0, float(t[-1])), psi0_2d, method='DOP853',
                    t_eval=t, rtol=1e-10, atol=1e-13)
    assert sol.success, sol.message
    ref = np.abs(sol.y).T ** 2
    d = float(np.max(np.abs(r['pops_2d'] - ref)))
    # 残差来源：光学 RWA 反旋项（Bloch–Siegert ~ Ω²/4ω·C·t ~ 1e-3）+ Sambe tol
    ck.check(f'TP3b 两场 ≡ 实验室系第一性参考（Δ={d:.2e} ≤ 5e-3）',
             d <= 5e-3, f'Δ={d}')
    nd = float(np.max(np.abs(r['pops_2d'].sum(axis=1) - 1.0)))
    # 范数偏差 ~ tol×段数量级累积（缺陷单级修正），非物理错误
    # （TP3b 对第一性参考 2.3e-4 已证物理正确）
    ck.check(f'TP3c 幺正范数（|Σp−1|={nd:.2e} ≤ 1e-4）', nd <= 1e-4,
             f'nd={nd}')
    dim = r['error_report']
    ck.check('TP3d 扩展维 = N²×P_mw（光场不占光子块；两侧各 1 个光场项；'
             'r=0.05 → M=3 → 7 块）',
             dim['photon_blocks'] == {'2000MHz': 7} and dim['opt_terms'] == 2,
             f"{dim['photon_blocks']}/{dim['opt_terms']}")


# ============================================================
# TP4 多光场拍频 → 显式拒绝（单原子与两体一致）
# ============================================================
def tp4():
    from src.core.evolve_twofield import (build_two_field_pair,
                                            evolve_two_field_pair)
    from src.core.evolve_twofield import evolve_two_field
    N = 2
    A = np.zeros((N, N))
    A[0, 1] = A[1, 0] = 10.0
    env = lambda tt: np.ones_like(np.asarray(tt, float))
    mask = np.array([1, 0])
    meta = dict(N=N, E_rot=np.array([0.0, 1.0e5]),
                opt_terms=[(0.5 * A, env, 1.0002e5, 'W1', 1.0002e5, mask),
                           (0.5 * A, env, 1.0004e5, 'W2', 1.0002e5, mask)],
                mw_groups=[])
    rejected = {'single': False, 'pair': False}
    try:
        evolve_two_field(states=[dict(n=60, l=0, j=0.5, kind='mJ', mj=-0.5,
                                      m=-0.5, E_mhz=0.0, label='g', index=0,
                                      amps=None, F=None, mF=None),
                                 dict(n=60, l=0, j=0.5, kind='mJ', mj=0.5,
                                      m=0.5, E_mhz=1.0e5, label='e',
                                      index=1, amps=None, F=None, mF=None)],
                         fields_meta=meta, psi0=0,
                         t_grid=np.linspace(0, 1.0, 11))
    except NotImplementedError:
        rejected['single'] = True
    try:
        build_two_field_pair(meta, meta)
    except NotImplementedError:
        rejected['pair'] = True
    ck.check('TP4 多光场拍频 δ≠0 → NotImplementedError（单原子 + 两体）',
             rejected['single'] and rejected['pair'], f'{rejected}')


# ============================================================
# TP5 寿命耗散：NonHermitian 守恒
# ============================================================
def tp5(states, meta_A, meta_B):
    """寿命耗散：无场场景（有微波驱动时 e 布居还受 Rabi 混合，不能对 e^{-Γt}）。"""
    from src.core.evolve_util import decay
    from src.core.evolve_twofield import evolve_two_field_pair
    G = 10.0
    meta0_A = dict(N=2, E_rot=np.array([0.0, 1000.0]), opt_terms=[],
                   mw_groups=[])
    meta0_B = dict(N=2, E_rot=np.array([0.0, 1000.0]), opt_terms=[],
                   mw_groups=[])
    t = np.linspace(0, 400.0, 201)
    r = evolve_two_field_pair(states, meta0_A, meta0_B, 1, 0, t,
                              vdd_matrix=np.zeros((4, 4)), tol=1e-8,
                              decays_A=[decay(1, G)])
    exp_dec = np.exp(-G * 1e-3 * t)
    ddec = float(np.max(np.abs(r['pops_A'][:, 1] - exp_dec)))
    cons = float(np.max(np.abs(r['pops_2d'].sum(axis=1)
                               + r['loss_A'] + r['loss_B'] - 1.0)))
    ck.check(f'TP5a 寿命衰变 ≡ e^{{−Γt}}（Δ={ddec:.2e} ≤ 1e-6）',
             ddec <= 1e-6, f'Δ={ddec}')
    ck.check(f'TP5b 守恒 Σpops+loss=1（偏差 {cons:.2e} ≤ 1e-3）',
             cons <= 1e-3, f'cons={cons}')


def main():
    states, meta_A, meta_B, env, A_drv = tp1()
    tp2(states, meta_A, meta_B)
    tp3()
    tp4()
    tp5(states, meta_A, meta_B)
    ok = ck.summary()
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
