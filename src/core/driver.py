# -*- coding: utf-8 -*-
"""两体驱动引擎（driver.py）——"关注跃迁"(driver) 驱动的 RWA 动力学。

设计（driver-centric，见 _misc/two_body_design.md §5）：
  用户选"关注跃迁"(driver)，软件自动分层：
    静态层 pair_static       ：非 driver 因素 → H_static（完整 pair 基，含 vdW/交换）
    驱动层 driver_operator   ：driver → (D, ω, Ω)
    化简层 rwa_hamiltonian   ：旋转系 RWA → H_rwa（高能侧减 ω，加 Ω/2·D）
    演化层 evolve            ：薛定谔方程 → 布居时间演化

单位：能级/耦合 MHz，时间 ns，角频率 rad/ns。
复用 core（ArcCache/dipole_amplitude/偏振/强度）+ pair_core（pair 基/V_dd/交换）。
"""
import sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
import numpy as np
from scipy.integrate import solve_ivp
from scipy.constants import epsilon_0, c as C_LIGHT, hbar
from . import core
from . import pair_core

_CONV = 2 * np.pi * 1e-3            # MHz·ns → 弧度


# ================= 演化层（通用，与物理无关） =================
def rwa_evolve(diag_mhz, couplings, psi0, t_ns):
    """多能级 RWA 演化：H = diag(diag_mhz) + Σ (i<j, Ω/2)(|i⟩⟨j|+h.c.)（MHz）。

    couplings: [(i, j, Ω)]——**Ω 为完整 Rabi 频率（MHz）**，内部取 Ω/2
    （RWA 旋转波项），与 rwa_hamiltonian 的 (Ω/2)·D 约定一致。
    ⚠ 不要把"半耦合 Ω/2"传进来：内部会再 ×0.5 → Rabi 减半（常见坑）。
    返回 (t_ns, pops)：pops[k] 为第 k 态布居数组。
    """
    n = len(diag_mhz)
    H = np.diag(diag_mhz).astype(complex)
    for i, j, om in couplings:
        H[i, j] += 0.5 * om
        H[j, i] += 0.5 * om
    t, psi_t = _evolve_amp(H, psi0, t_ns)
    return t, np.abs(psi_t) ** 2


def full_evolve(H0_mhz, D, omega_rad_ns, om_mhz, psi0, t_ns):
    """完整含时积分（不 RWA）：H(t) = H0 + (Ω/2)(D e^{-iωt} + D† e^{iωt})。

    系数 Ω/2 与 RWA 耦合约定一致（V=Ωcos(ωt)·D → 旋转波项 Ω/2）。
    最无偏基准，用于交叉验证 RWA 实现。
    """
    H0 = np.asarray(H0_mhz, complex)
    Dm = np.asarray(D, complex)
    Dd = Dm.conj().T

    def rhs(t, psi):
        ph = np.exp(-1j * omega_rad_ns * t)
        Ht = H0 + (om_mhz / 2.0) * (Dm * ph + Dd * np.conj(ph))
        return (-1j * _CONV) * (Ht @ psi)

    sol = solve_ivp(rhs, (t_ns[0], t_ns[-1]), np.asarray(psi0, complex),
                    t_eval=t_ns, rtol=1e-8, atol=1e-10)
    return t_ns, np.abs(sol.y) ** 2


# ================= 静态层（pair 静态） =================
def pair_static(cache, groups, R_um, B_gauss=0.0, static_cfg=None, **kw):
    """pair 静态哈密顿（完整基，含 vdW/交换相互作用）。

    static_cfg: dict(vdw=True, exchange=True, ac_stark=False)
      vdw/exchange 均开 → 完整 V_dd（复用 pair_core.pair_hamiltonian）
      ac_stark：非 driver 光位移（暂未实现，预留）
    kw 透传：theta_deg/phi_deg/n_window/m_total/energy_window_mhz/ref_energy_mhz
    返回 dict(single, pairs, E_pair, H)。
    """
    cfg = dict(vdw=True, exchange=True, ac_stark=False)
    if static_cfg:
        cfg.update(static_cfg)
    if not (cfg['vdw'] or cfg['exchange']):
        single = pair_core.single_atom_basis(cache, groups, B_gauss,
                                             kw.get('n_window', 3))
        pairs, E_pair = pair_core.pair_indices(
            single, kw.get('m_total'), kw.get('energy_window_mhz'),
            kw.get('ref_energy_mhz'))
        return dict(single=single, pairs=pairs, E_pair=E_pair,
                    H=np.diag(E_pair))
    return pair_core.pair_hamiltonian(
        cache, groups, R_um, kw.get('theta_deg', 0.0), kw.get('phi_deg', 0.0),
        B_gauss, kw.get('n_window', 3), kw.get('m_total'),
        kw.get('energy_window_mhz'), kw.get('ref_energy_mhz'))


# ================= 驱动层（driver 算符 + Rabi） =================
def driver_operator(cache, single, pairs, driver_cfg):
    """构建驱动算符 D（pair 子空间）+ 载频 + Rabi。

    返回 (D, omega_rad_ns, om_mhz)。
    driver_cfg:
      kind='mw' : 微波驱动 group_a↔group_b（默认 60S↔60P），
                  om 由 intensity_mw_cm2（软件算）或 om_mhz（用户给）
      kind='ex' : 交换驱动 |a⊗b⟩↔|b⊗a⟩，om=V_ex(R)
    """
    kind = driver_cfg['kind']
    if kind == 'mw':
        return _mw_operator(cache, single, pairs, driver_cfg)
    if kind == 'ex':
        return _ex_operator(cache, single, pairs, driver_cfg)
    raise ValueError(f'未知 driver kind: {kind}')


def _rabi_mhz(cache, single, idx_a, idx_b, cfg, pol_q):
    """从光强算 Rabi（MHz）：Ω = E0·|d|/ħ。cfg 含 om_mhz 则直接返回。"""
    if 'om_mhz' in cfg:
        return float(cfg['om_mhz'])
    I = float(cfg['intensity_mw_cm2'])
    dmax = 0.0
    for ia in idx_a:
        for ib in idx_b:
            dmax = max(dmax, abs(core.dipole_amplitude(
                cache, single[ia], single[ib], pol_q)))
    E0 = (2.0 * I * core.MW_PER_CM2_TO_W_PER_M2 / (epsilon_0 * C_LIGHT)) ** 0.5
    return E0 * dmax * core.E_CHARGE * core.A0 / hbar / 1e6


def _mw_operator(cache, single, pairs, cfg):
    N = len(single)
    ga = cfg.get('group_a', dict(n=60, l=0, j=0.5))
    gb = cfg.get('group_b', dict(n=60, l=1, j=0.5))
    idx_a = [i for i, s in enumerate(single)
             if (s['n'], s['l'], s['j']) == (ga['n'], ga['l'], ga['j'])]
    idx_b = [i for i, s in enumerate(single)
             if (s['n'], s['l'], s['j']) == (gb['n'], gb['l'], gb['j'])]
    q = core.POLARIZATION_Q.get(cfg.get('pol', 'π'), 0)
    # 单原子跃迁算符 A：只耦合 dipole_amplitude 非零的对（选择定则，
    # 含 F↔mJ 的 CG 叠加与核自旋守恒，不能只比 _mag 数值差）
    A = np.zeros((N, N), complex)
    for ia in idx_a:
        for ib in idx_b:
            if abs(core.dipole_amplitude(cache, single[ia], single[ib],
                                         q)) > 1e-12:
                A[ia, ib] = 1.0
                A[ib, ia] = 1.0
    # 作用范围：both=两原子都打；atom1/atom2=单原子寻址（局域驱动）
    scope = cfg.get('scope', 'both')
    if scope == 'both':
        D = np.kron(A, np.eye(N)) + np.kron(np.eye(N), A)
    elif scope == 'atom1':
        D = np.kron(A, np.eye(N))
    elif scope == 'atom2':
        D = np.kron(np.eye(N), A)
    else:
        raise ValueError(f'未知 scope: {scope}')
    sel = np.array([i * N + j for i, j in pairs], dtype=int)
    D = D[np.ix_(sel, sel)]
    om = _rabi_mhz(cache, single, idx_a, idx_b, cfg, q)
    omega_rad_ns = float(cfg.get('omega_rad_ns', 0.0))
    return D, omega_rad_ns, om


def _ex_operator(cache, single, pairs, cfg):
    N = len(single)
    ga = cfg.get('group_a', dict(n=60, l=0, j=0.5))
    gb = cfg.get('group_b', dict(n=60, l=1, j=0.5))
    ma = cfg.get('m_a', 0.5)
    mb = cfg.get('m_b', 0.5)
    ia = pair_core.find_sublevel(single, ga['n'], ga['l'], ga['j'], ma)
    ib = pair_core.find_sublevel(single, gb['n'], gb['l'], gb['j'], mb)
    sel = {i * N + j: k for k, (i, j) in enumerate(pairs)}
    D = np.zeros((len(pairs), len(pairs)), complex)
    if ia is not None and ib is not None:
        k1, k2 = ia * N + ib, ib * N + ia
        if k1 in sel and k2 in sel:
            D[sel[k1], sel[k2]] = 1.0
            D[sel[k2], sel[k1]] = 1.0
    if 'om_mhz' in cfg:
        om = float(cfg['om_mhz'])
    else:
        om = abs(pair_core.exchange_coupling(cache, [ga, gb], float(cfg['R_um'])))
    return D, 0.0, om


# ================= 化简层（RWA 哈密顿） =================
def _photon_numbers(D, start, n_st):
    """沿驱动算符 D 从初始态 start 出发 BFS，推断每态光子数。

    微波驱动（每原子 S↔P）下：SS=0、SP/PS=1、PP=2 → 每态减 n_k·ω，
    正确支持双光子级联（SS→SP→PP）。D 连通分量外未触及的态记 0。
    """
    from collections import deque
    dist = {int(start): 0}
    q = deque([int(start)])
    while q:
        i = q.popleft()
        for j in np.nonzero(np.abs(D[i]) > 1e-12)[0]:
            if j not in dist:
                dist[j] = dist[i] + 1
                q.append(j)
    return np.array([dist.get(i, 0) for i in range(n_st)])


def rwa_hamiltonian(st, D, omega_rad_ns, om_mhz, photon_num=None,
                    start_idx=None):
    """静态 + RWA 驱动 → H_rwa（MHz，pair 子空间）。

    旋转系：每态按光子数减 n_k·ω（photon_num 缺省从 start_idx 沿 D BFS
    推断）；再加 (Ω/2)·D。
    """
    H = np.array(st['H'], complex)
    E = np.real(np.diag(H))
    omega_mhz = omega_rad_ns / (2 * np.pi * 1e-3)     # rad/ns → MHz
    if abs(omega_mhz) > 1e-12:
        if photon_num is None:
            if start_idx is None:
                start_idx = int(np.argmin(E))
            photon_num = _photon_numbers(D, start_idx, len(E))
        for k, nk in enumerate(photon_num):
            if nk:
                H[k, k] -= nk * omega_mhz
    H = H + (om_mhz / 2.0) * D
    return H


# ================= 演化（定常哈密顿） =================
def _evolve_amp(H_rwa, psi0, t_ns):
    """定常哈密顿演化：一次对角化 + 向量化，返回复振幅矩阵 (n_states, n_t)。"""
    H = np.asarray(H_rwa, complex)
    evals, evecs = np.linalg.eigh(H)
    psi0 = np.asarray(psi0, complex)
    c = evecs.conj().T @ psi0
    t_ns = np.asarray(t_ns)
    ph = np.exp(-1j * _CONV * evals[:, None] * t_ns[None, :])
    psi_t = evecs @ (c[:, None] * ph)
    return t_ns, psi_t


def evolve(H_rwa, psi0, t_ns):
    """定常哈密顿布居演化（MHz）：一次对角化 + 向量化。返回 (t_ns, pops)。"""
    t, psi_t = _evolve_amp(H_rwa, psi0, t_ns)
    return t, np.abs(psi_t) ** 2


# ================= dressed 投影（vdW 位移进入封锁） =================
def _dressed_hamiltonian(st, D, omega_rad_ns, om_mhz, start_idx=None):
    """静态全对角化 → dressed 能级（含 vdW）→ 驱动投影 → RWA H_dressed。

    返回 (H_dressed, evals, evecs, i0)：i0 为初始态在 dressed 基的索引
    （|初始 raw 态⟩ 投影最大的本征态，光子数 BFS 起点）。
    """
    evals, evecs = np.linalg.eigh(st['H'])
    Dd = evecs.conj().T @ D @ evecs
    if start_idx is None:
        start_idx = int(np.argmin(np.real(np.diag(st['H']))))
    w0 = np.abs(evecs[start_idx, :]) ** 2      # 初始 raw 态在 dressed 基的权重
    i0 = int(np.argmax(w0))
    omega_mhz = omega_rad_ns / (2 * np.pi * 1e-3)     # rad/ns → MHz
    if abs(omega_mhz) > 1e-12:
        photon = _photon_numbers(Dd, i0, len(evals))
        Hd = np.diag(evals - omega_mhz * photon)
    else:
        Hd = np.diag(evals)
    Hd = Hd + (om_mhz / 2.0) * Dd
    return Hd, evals, evecs, i0


# ================= 一站式入口 =================
def pair_drive(cache, groups, R_um, B_gauss, driver_cfg, static_cfg=None,
               psi0_ket=None, t_ns=None, **kw):
    """完整驱动动力学：静态 → 驱动 → RWA → 布居演化。

    psi0_ket: (n,l,j,m) 初始态（两原子同态）；None → |group_a(ma)⊗group_a(ma)⟩。
    t_ns: 时间数组（默认 0..400ns，2001 点）。
    返回 dict(single, pairs, E_pair, H, D, om_mhz, omega_rad_ns, t, pops)。
    """
    st = pair_static(cache, groups, R_um, B_gauss, static_cfg, **kw)
    D, omega_rad_ns, om = driver_operator(cache, st['single'], st['pairs'],
                                          driver_cfg)
    if t_ns is None:
        t_ns = np.linspace(0, 400.0, 2001)
    N = len(st['single'])
    sel = {i * N + j: k for k, (i, j) in enumerate(st['pairs'])}
    if psi0_ket is None:
        ga = driver_cfg.get('group_a', dict(n=60, l=0, j=0.5))
        ma = driver_cfg.get('m_a', 0.5)
        ia = pair_core.find_sublevel(st['single'], ga['n'], ga['l'], ga['j'], ma)
    else:
        ia = pair_core.find_sublevel(st['single'], *psi0_ket)
    psi0 = np.zeros(len(st['pairs']), complex)
    k0 = sel[ia * N + ia]
    psi0[k0] = 1.0
    H_rwa = rwa_hamiltonian(st, D, omega_rad_ns, om, start_idx=k0)
    t, pops = evolve(H_rwa, psi0, t_ns)
    return dict(single=st['single'], pairs=st['pairs'], E_pair=st['E_pair'],
                H=st['H'], D=D, om_mhz=om, omega_rad_ns=omega_rad_ns,
                t=t, pops=pops)


def pair_drive_dressed(cache, groups, R_um, B_gauss, driver_cfg,
                       static_cfg=None, psi0_ket=None, t_ns=None, **kw):
    """dressed 投影驱动（含 vdW 位移）：静态对角化 → 驱动投影 → RWA → 演化。

    与 pair_drive 接口一致，返回 raw 基布居 pops（并附 dressed 信息
    evals/evecs/D_dressed）。vdW（C6 二阶位移）通过静态对角化进入
    dressed 能级，封锁物理完整。
    """
    st = pair_static(cache, groups, R_um, B_gauss, static_cfg, **kw)
    D, omega_rad_ns, om = driver_operator(cache, st['single'], st['pairs'],
                                          driver_cfg)
    if t_ns is None:
        t_ns = np.linspace(0, 400.0, 2001)
    N = len(st['single'])
    sel = {i * N + j: k for k, (i, j) in enumerate(st['pairs'])}
    if psi0_ket is None:
        ga = driver_cfg.get('group_a', dict(n=60, l=0, j=0.5))
        ma = driver_cfg.get('m_a', 0.5)
        ia = pair_core.find_sublevel(st['single'], ga['n'], ga['l'], ga['j'], ma)
    else:
        ia = pair_core.find_sublevel(st['single'], *psi0_ket)
    psi0 = np.zeros(len(st['pairs']), complex)
    psi0[sel[ia * N + ia]] = 1.0
    Hd, evals, evecs, i0 = _dressed_hamiltonian(st, D, omega_rad_ns, om,
                                                sel[ia * N + ia])
    t, psi_d = _evolve_amp(Hd, evecs.conj().T @ psi0, t_ns)
    psi_raw = evecs @ psi_d
    pops = np.abs(psi_raw) ** 2
    return dict(single=st['single'], pairs=st['pairs'], E_pair=st['E_pair'],
                evals=evals, evecs=evecs,
                D_dressed=evecs.conj().T @ D @ evecs,
                om_mhz=om, omega_rad_ns=omega_rad_ns, t=t, pops=pops)


# ================= 统一系统（单体/两体同一接口） =================
def build_system(cache, groups, n_body=1, R_um=None, B_gauss=0.0,
                 static_cfg=None, theta_deg=0.0, phi_deg=0.0,
                 m_total=None, energy_window_mhz=None, ref_energy_mhz=None,
                 n_window=3):
    """统一系统构建：单体(1) / 两体(2) 同一接口，仅影响因素不同。

    返回 system(dict)：
      n_body : 1 或 2
      states : 单体=单原子行列表；两体=pair (i,j) 数组
      H      : 静态哈密顿（含全部能移源）
      cache/groups/R_um/B_gauss/static_cfg/_kw 供 drive 复用

    static_cfg：dict(vdw, exchange, ac_stark)
      ac_stark = 能移光配置 dict(frequency_hz, intensity, polarization)
      —— 单体时对该光逐态算 AC 位移加对角（如"微波做能移"）。
    """
    sys0 = dict(cache=cache, groups=groups, n_body=n_body, R_um=R_um,
                B_gauss=B_gauss, static_cfg=static_cfg or {},
                _kw=dict(theta_deg=theta_deg, phi_deg=phi_deg, m_total=m_total,
                         energy_window_mhz=energy_window_mhz,
                         ref_energy_mhz=ref_energy_mhz, n_window=n_window))
    if n_body == 1:
        single = core._direct_basis(cache, groups, B_gauss, n_window)
        H = np.diag([s['E_mhz'] for s in single]).astype(complex)
        shifts = _static_shifts(cache, single, groups, static_cfg, B_gauss)
        if shifts is not None:
            H = H + np.diag(shifts)
        sys0.update(states=single, H=H, single=single)
        return sys0
    if n_body == 2:
        res = pair_core.pair_hamiltonian(
            cache, groups, R_um, theta_deg, phi_deg, B_gauss, n_window,
            m_total, energy_window_mhz, ref_energy_mhz)
        H = res['H']
        shifts = _static_shifts(cache, res['single'], groups, static_cfg,
                                B_gauss)
        if shifts is not None:
            # 能移是单原子性质：pair 态 (i,j) 的能移 = shift_i + shift_j
            for k, (i, j) in enumerate(res['pairs']):
                H[k, k] += shifts[i] + shifts[j]
        sys0.update(states=res['pairs'], H=H, single=res['single'])
        return sys0
    raise ValueError(f'n_body 须为 1 或 2，got {n_body}')


def _static_shifts(cache, single, groups, static_cfg, B_gauss):
    """每单原子态的总静态能移（MHz 数组）；无返回 None。

    static_cfg['ac_stark']：AC 能移光（单份 dict 或多份列表，全部作能移）；
    static_cfg['dc_stark']：静电场强度 E (V/cm)。均线性叠加。
    """
    if not static_cfg:
        return None
    shifts = np.zeros(len(single))
    # AC 光能移
    ac_list = static_cfg.get('ac_stark')
    if ac_list:
        if isinstance(ac_list, dict):
            ac_list = [ac_list]
        for ac in ac_list:
            ac = dict(ac)
            ac.setdefault('enabled', True)   # 传入即视为启用
            for i, s in enumerate(single):
                g = next((g for g in groups
                          if (g['n'], g['l'], g['j']) == (s['n'], s['l'], s['j'])),
                         None)
                if g is None:
                    continue
                m = s['mj'] if s['kind'] == 'mJ' else s['mF']
                shifts[i] += core.ac_stark_shift_mhz(cache, g, m, groups, ac,
                                                     B_gauss)
    # 静电场能移（DC-Stark，二阶 SOS）
    e_vpcm = static_cfg.get('dc_stark')
    if e_vpcm:
        for i, s in enumerate(single):
            g = next((g for g in groups
                      if (g['n'], g['l'], g['j']) == (s['n'], s['l'], s['j'])),
                     None)
            if g is None:
                continue
            m = s['mj'] if s['kind'] == 'mJ' else s['mF']
            shifts[i] += core.dc_stark_shift_mhz(
                cache, g['n'], g['l'], g['j'], m, groups, float(e_vpcm),
                B_gauss)
    if np.all(shifts == 0.0):
        return None
    return shifts


def _single_operator(cache, single, driver_cfg):
    """单体驱动算符（寻址光/微波，单原子基）：D = A（group_a↔group_b）。"""
    N = len(single)
    ga = driver_cfg.get('group_a', dict(n=6, l=1, j=1.5))
    gb = driver_cfg.get('group_b', dict(n=60, l=0, j=0.5))
    idx_a = [i for i, s in enumerate(single)
             if (s['n'], s['l'], s['j']) == (ga['n'], ga['l'], ga['j'])]
    idx_b = [i for i, s in enumerate(single)
             if (s['n'], s['l'], s['j']) == (gb['n'], gb['l'], gb['j'])]
    q = core.POLARIZATION_Q.get(driver_cfg.get('pol', 'π'), 0)
    A = np.zeros((N, N), complex)
    for ia in idx_a:
        for ib in idx_b:
            if abs(core.dipole_amplitude(cache, single[ia], single[ib],
                                         q)) > 1e-12:
                A[ia, ib] = 1.0
                A[ib, ia] = 1.0
    om = _rabi_mhz(cache, single, idx_a, idx_b, driver_cfg, q)
    omega_rad_ns = float(driver_cfg.get('omega_rad_ns', 0.0))
    return A, omega_rad_ns, om


def drive(system, driver_cfg, t_ns=None, psi0_ket=None):
    """统一驱动演化（单体/两体同一入口）。

    返回 dict(t, pops, om_mhz, omega_rad_ns, n_body, single[, pairs])。
    psi0_ket: (n,l,j,m) 初始态；None → driver group_a 的 m_a。
    """
    cache = system['cache']
    if t_ns is None:
        t_ns = np.linspace(0, 400.0, 2001)
    if system['n_body'] == 1:
        single = system['single']
        D, w, om = _single_operator(cache, single, driver_cfg)
        ga = driver_cfg.get('group_a', dict(n=6, l=1, j=1.5))
        if psi0_ket is None:
            psi0_ket = (ga['n'], ga['l'], ga['j'], driver_cfg.get('m_a', 0.5))
        i0 = pair_core.find_sublevel(single, *psi0_ket)
        if i0 is None:
            raise ValueError(f'初始态 {psi0_ket} 不在基中')
        psi0 = np.zeros(len(single), complex)
        psi0[i0] = 1.0
        H_rwa = rwa_hamiltonian(system, D, w, om, start_idx=i0)
        t, pops = evolve(H_rwa, psi0, t_ns)
        return dict(t=t, pops=pops, om_mhz=om, omega_rad_ns=w, n_body=1,
                    single=single)
    # 两体：dressed（含 vdW）
    res = pair_drive_dressed(cache, system['groups'], system['R_um'],
                             system['B_gauss'], driver_cfg,
                             system['static_cfg'], psi0_ket, t_ns,
                             **system['_kw'])
    return dict(t=res['t'], pops=res['pops'], om_mhz=res['om_mhz'],
                omega_rad_ns=res['omega_rad_ns'], n_body=2,
                single=res['single'], pairs=res['pairs'])


def response_spectrum(system, driver_cfg, freq_scan_hz, t_ns=None,
                      psi0_ket=None):
    """驱动响应谱（统一）：扫驱动频率 → 初始态布居流失量（激发概率）。

    返回 (freq_hz 数组, 响应数组)。单体/两体同一引擎。
    psi0_ket: (n,l,j,m)；None → drive 默认（driver group_a 的 m_a）。
    """
    out = []
    for f in freq_scan_hz:
        cfg = dict(driver_cfg, omega_rad_ns=2 * np.pi * 1e-3 * (f / 1e6))
        res = drive(system, cfg, t_ns=t_ns, psi0_ket=psi0_ket)
        out.append(1.0 - float(res['pops'][0][-1]))   # 初态末布居流失
    return np.asarray(freq_scan_hz), np.asarray(out)


def _single_dipole_matrix(cache, single, driver_cfg):
    """单原子驱动算符（**真实偶极矩阵元模**）：A[i,j] = |⟨j|d_q|i⟩|（a.u.）。

    与 `_single_operator` 的拓扑 0/1 不同：线性响应谱需要真实相对强度。
    """
    N = len(single)
    ga = driver_cfg.get('group_a', dict(n=6, l=1, j=1.5))
    gb = driver_cfg.get('group_b', dict(n=60, l=0, j=0.5))
    idx_a = [i for i, s in enumerate(single)
             if (s['n'], s['l'], s['j']) == (ga['n'], ga['l'], ga['j'])]
    idx_b = [i for i, s in enumerate(single)
             if (s['n'], s['l'], s['j']) == (gb['n'], gb['l'], gb['j'])]
    q = core.POLARIZATION_Q.get(driver_cfg.get('pol', 'π'), 0)
    A = np.zeros((N, N), complex)
    for ia in idx_a:
        for ib in idx_b:
            d = core.dipole_amplitude(cache, single[ia], single[ib], q)
            if abs(d) > 1e-12:
                A[ia, ib] = abs(d)
                A[ib, ia] = abs(d)
    return A


def response_spectrum_linear(system, driver_cfg, psi0_ket=None):
    """线性响应谱（黄金规则，第一性原理）：**一次静态对角化**提取全部共振。

        S(ω) = Σ_k |⟨k|D|i⟩|² δ(ω - (E_k - E_i))

    静态 H（含 vdW/交换/AC-Stark 能移）对角化一次 → 全部本征能级 E_k 与
    本征态 |k⟩；驱动算符 D（**真实偶极矩阵元**，含 scope 单原子寻址）投影
    到本征基，初态 |i⟩ 的所有共振频率与相对强度一次算出（O(N²)）。
    比逐频率含时演化快 ~N_freq 倍，峰位严格、强度反映真实相对矩阵元。

    返回 dict(freq_hz, strength, om_mhz, i0)：
      freq_hz  : 共振频率 (E_k-E_i)，Hz（排除近零频弹性散射）
      strength : 相对强度 |⟨k|D|i⟩|²（D 为真实矩阵元，a.u.）
      om_mhz   : 驱动 Rabi（MHz），供 UI 标度绝对强度
      i0       : 初态在基组索引
    psi0_ket: (n,l,j,m)；None → driver group_a 的 m_a。
    """
    cache = system['cache']
    single = system['single']
    A = _single_dipole_matrix(cache, single, driver_cfg)
    ga = driver_cfg.get('group_a', dict(n=6, l=1, j=1.5))
    gb = driver_cfg.get('group_b', dict(n=60, l=0, j=0.5))
    q = core.POLARIZATION_Q.get(driver_cfg.get('pol', 'π'), 0)
    idx_a = [i for i, s in enumerate(single)
             if (s['n'], s['l'], s['j']) == (ga['n'], ga['l'], ga['j'])]
    idx_b = [i for i, s in enumerate(single)
             if (s['n'], s['l'], s['j']) == (gb['n'], gb['l'], gb['j'])]
    om = _rabi_mhz(cache, single, idx_a, idx_b, driver_cfg, q)
    if system['n_body'] == 1:
        D = A
        if psi0_ket is None:
            psi0_ket = (ga['n'], ga['l'], ga['j'],
                        driver_cfg.get('m_a', 0.5))
        i0 = pair_core.find_sublevel(single, *psi0_ket)
        if i0 is None:
            raise ValueError(f'初始态 {psi0_ket} 不在基中')
    else:
        N = len(single)
        scope = driver_cfg.get('scope', 'both')
        if scope == 'both':
            D = np.kron(A, np.eye(N)) + np.kron(np.eye(N), A)
        elif scope == 'atom1':
            D = np.kron(A, np.eye(N))
        elif scope == 'atom2':
            D = np.kron(np.eye(N), A)
        else:
            raise ValueError(f'未知 scope: {scope}')
        sel = np.array([i * N + j for i, j in system['states']], dtype=int)
        D = D[np.ix_(sel, sel)]
        if psi0_ket is None:
            psi0_ket = (ga['n'], ga['l'], ga['j'],
                        driver_cfg.get('m_a', 0.5))
        ia = pair_core.find_sublevel(single, *psi0_ket)
        if ia is None:
            raise ValueError(f'初始态 {psi0_ket} 不在基中')
        i0 = {k: v for v, k in enumerate(sel)}[ia * N + ia]
    H = np.asarray(system['H'], complex)
    evals, evecs = np.linalg.eigh(H)
    Di = evecs.conj().T @ D[:, i0]           # D|i⟩ 投影到本征基
    # 初态能量 = 本征展开期望（evals 是升序本征值，不能按基组索引 i0 取）
    E_i = np.dot(np.abs(evecs[i0, :]) ** 2, evals)
    f0 = (evals - E_i) * 1e6                 # MHz → Hz
    strength = np.abs(Di) ** 2
    mask = np.abs(f0) > 1e-3                 # 排除初态自身（近零频）
    return dict(freq_hz=f0[mask], strength=strength[mask], om_mhz=om,
                i0=i0)
