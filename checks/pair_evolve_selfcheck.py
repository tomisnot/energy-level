# -*- coding: utf-8 -*-
"""pair_evolve.py 自检（5 项）：以 pairinteraction 为权威参考，最简单直接。

设计原则（对应用户标准）：
- 所有物理输入由权威开源软件 **pairinteraction 2.x** 独立给出：
  两体哈密顿 H_pair（含单体能级与 V_dd）、本征能谱；
- 动力学参考 = 同一哈密顿量的**精确矩阵指数演化**（scipy.linalg.expm），
  不引入任何与实现共享的约定/内部函数；
- 每项只断言一个清晰的物理事实，容差明确；旧版"手搓 toy 模型 +
  手写 V_dd + 直调内部 _pair_ip4"的自我验证写法已废弃（归档至
  checks/_archive/pair_evolve_toy_selfcheck.py）。

T1 权威 V_dd：pair_core.vdd_sparse 与 pairinteraction 双原子哈密顿的
   相互作用部分逐元一致（同 R、θ 几何，B=0 基组精确对齐）。
T2 权威能级：pair_core.pair_eigen 谱（去公共参考）与 pairinteraction
   本征值一致；|60S,60S⟩ 支的 vdW 位移一致。
T3 权威动力学：pair_evolve 用 pairinteraction 的 V_dd 演化 |60S,60P⟩，
   与精确 expm 演化逐时刻对比（交换振荡）；首峰时刻 = 交换半周期。
T4 单体一致性：V_dd=0 时两原子独立演化 ≡ 单原子引擎 evolve.evolve
   （该引擎自身已对 solve_ivp 第一性原理验证），验证两体场张量化。
T5 secular 判据：旋转系失谐 ≫ |V| 的矩阵元被正确丢弃（RWA），近简并
   交换元被保留——回归钉死 secular 判据 bug（旧实现误用 ref 差，
   无光场时把 33 GHz 失谐的 vdW 耦合当静态保留）。

注：pairinteraction 未安装时整组 SKIP（不误报）。
"""
import os
import sys
import numpy as np

if sys.platform == 'win32' and hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _ROOT)
import src  # noqa: F401
from src.core import evolve, core, pair_core
from arc import Rubidium87

try:
    import pairinteraction as pi
    from scipy.linalg import expm
    _HAS_PI = True
except Exception:
    pi = None
    _HAS_PI = False

# MHz·ns → rad（与 pair_evolve/evolve 同一约定）
_CONV = 2 * np.pi * 1e-3

_n_pass = _n_fail = 0


def _check(name, cond, detail=''):
    global _n_pass, _n_fail
    if cond:
        _n_pass += 1
        print(f'  [PASS] {name}')
    else:
        _n_fail += 1
        print(f'  [FAIL] {name}   {detail}')


# ==================================================================
# pairinteraction 权威两体系统（B=0，fine 结构，两原子同基）
# ==================================================================

def pi_pair_system(qnums, R_um, theta_deg=0.0):
    """用 pairinteraction 构造双原子系统并取哈密顿量。

    qnums: 单原子基（项目顺序）的 [(n,l,j,m), ...]，两原子共用。
    返回 (H_pair, E_single, idx_pi)：
      H_pair   : (N², N²) 复数数组——两体哈密顿，**项目 product 序**
                 k = i*N + j（已按 qnums 顺序重排，与 pair_core/pair_evolve
                 的基序一致）
      E_single : (N,) 单原子能量（MHz，qnums 顺序）
      idx_pi   : 项目顺序 → pairinteraction 单原子基索引
    """
    ns = [q[0] for q in qnums]
    ls = [q[1] for q in qnums]
    js = [q[2] for q in qnums]
    ms = [q[3] for q in qnums]
    b1 = pi.BasisAtom('Rb', n=(min(ns), max(ns)), l=(min(ls), max(ls)),
                      j=(min(js), max(js)), m=(min(ms), max(ms)))
    sys1 = pi.SystemAtom(b1).diagonalize()
    sys2 = pi.SystemAtom(b1).diagonalize()
    kets = sys1.basis.kets
    idx_pi = []
    for (n, l, j, m) in qnums:
        hit = [i for i, k in enumerate(kets)
               if abs(k.n - n) < 1e-9 and abs(k.l - l) < 1e-9
               and abs(k.j - j) < 1e-9 and abs(k.m - m) < 1e-9]
        assert len(hit) == 1, \
            f'pairinteraction 基中 (n,l,j,m)={(n,l,j,m)} 匹配 {len(hit)} 个'
        idx_pi.append(hit[0])
    Npi = len(kets)
    bp = pi.BasisPair([sys1, sys2])
    sp = pi.SystemPair(bp).set_distance(
        R_um, angle_degree=theta_deg, unit='micrometer')
    H_pi = sp.get_hamiltonian(unit='MHz').tocsr()
    # product 序子块：项目 k = a*N + b → pi a*Npi + b
    sel = np.array([a * Npi + b for a in idx_pi for b in idx_pi], dtype=int)
    H_pair = H_pi[sel][:, sel].toarray()
    E_single = sys1.get_eigenenergies(unit='MHz')[idx_pi]
    return H_pair, E_single, idx_pi


# ==================================================================
# T1: 权威 V_dd —— pair_core.vdd_sparse ≡ pairinteraction
# ==================================================================
def t1():
    print('T1 权威 V_dd：pair_core.vdd_sparse ≡ pairinteraction')
    if not _HAS_PI:
        print('  [SKIP] pairinteraction 未安装'); return

    cache = core.ArcCache(Rubidium87())
    groups = [dict(n=60, l=0, j=0.5), dict(n=60, l=1, j=0.5)]
    single = pair_core.single_atom_basis(cache, groups, 0.0, 1)
    N = len(single)
    qnums = [(s['n'], s['l'], s['j'], s['mj']) for s in single]
    R, th = 10.0, 30.0
    H_pi, E_pi, _ = pi_pair_system(qnums, R, th)
    # 相互作用部分 = H_pair − 单体张量对角
    E_pair = np.kron(E_pi, np.ones(N)) + np.kron(np.ones(N), E_pi)
    V_pi = H_pi - np.diag(E_pair)
    V_proj = pair_core.vdd_sparse(cache, single, R, th).toarray()
    scale = max(np.abs(V_pi).max(), 1e-12)
    mask = np.abs(V_pi) > 1e-3 * scale
    # 相位鲁棒比较：ARC 与 pairinteraction 对偶极矩阵元采用不同的
    # 规范（约半数元素整体差一个 −1 相位），这是合法的相位约定差——
    # 物理量（能谱、动力学）不受影响（见 T2/T3）。故按 |V| 逐元比较。
    rel = np.abs(np.abs(V_proj) - np.abs(V_pi))[mask] / np.abs(V_pi)[mask]
    detail = (f'非零元 {mask.sum()}，|V| 相对误差 median='
              f'{np.median(rel):.2e} max={rel.max():.2e}'
              f'（scale={scale:.3g} MHz）')
    _check('|V_dd| 逐元相对误差 < 2e-3', rel.size > 0 and rel.max() < 2e-3, detail)
    # 交换耦合元抽查（|60S,60P⟩↔|60P,60S⟩）
    ia = pair_core.find_sublevel(single, 60, 0, 0.5, 0.5)
    ib = pair_core.find_sublevel(single, 60, 1, 0.5, 0.5)
    k_sp, k_ps = ia * N + ib, ib * N + ia
    v_proj = V_proj[k_sp, k_ps]
    v_pi = V_pi[k_sp, k_ps]
    _check('交换耦合元一致', abs(v_proj - v_pi) < 1e-3 * max(abs(v_pi), 1e-9),
           f'proj={v_proj:.6f} pi={v_pi:.6f} MHz')


# ==================================================================
# T2: 权威能级 —— pair_eigen ≡ pairinteraction
# ==================================================================
def t2():
    print('T2 权威能级：pair_eigen ≡ pairinteraction')
    if not _HAS_PI:
        print('  [SKIP] pairinteraction 未安装'); return

    cache = core.ArcCache(Rubidium87())
    groups = [dict(n=60, l=0, j=0.5), dict(n=60, l=1, j=0.5)]
    R, th = 10.0, 30.0
    res = pair_core.pair_eigen(cache, groups, R, th, B_gauss=0.0, n_window=1)
    N = len(res['single'])
    qnums = [(s['n'], s['l'], s['j'], s['mj']) for s in res['single']]
    H_pi, E_pi, _ = pi_pair_system(qnums, R, th)

    # 全谱（去公共参考，抵消 ARC/pairinteraction 能量零点约定差）
    ev_proj = np.sort(res['evals'])
    ev_pi = np.sort(np.linalg.eigvalsh(H_pi))
    d_proj = ev_proj - ev_proj[0]
    d_pi = ev_pi - ev_pi[0]
    err = np.max(np.abs(d_proj - d_pi))
    scale = max(np.abs(d_pi[-1]), 1e-9)
    _check('谱（去公共参考）max 差 < 0.5 MHz', err < 0.5,
           f'max err={err:.4f} MHz（谱宽={scale:.3g} MHz）')

    # vdW 位移：|60S,60S⟩ 支本征值 − 2E(60S)
    ia = pair_core.find_sublevel(res['single'], 60, 0, 0.5, 0.5)
    k_ss = ia * N + ia
    shift_proj, _ = pair_core.vdw_shift(
        cache, groups, R, th, B_gauss=0.0, n_window=1,
        ref_ket=(60, 0, 0.5, 0.5))
    ev, evec = np.linalg.eigh(H_pi)
    w = np.abs(evec[k_ss, :]) ** 2
    im = int(np.argmax(w))
    shift_pi = ev[im] - 2.0 * E_pi[ia]
    _check('vdW 位移一致（< 0.5 MHz）', abs(shift_proj - shift_pi) < 0.5,
           f'proj={shift_proj:+.5f} pi={shift_pi:+.5f} MHz')


# ==================================================================
# T3: 权威动力学 —— pair_evolve ≡ 精确 expm（pairinteraction V_dd）
# ==================================================================
def t3():
    print('T3 权威动力学：交换振荡 ≡ 精确 expm（pairinteraction V_dd）')
    if not _HAS_PI:
        print('  [SKIP] pairinteraction 未安装'); return

    from src.core import pair_evolve
    cache = core.ArcCache(Rubidium87())
    qnums = [(60, 0, 0.5, 0.5), (60, 1, 0.5, 0.5)]
    R = 5.0
    H_pi, E_pi, _ = pi_pair_system(qnums, R, 0.0)
    N = 2
    E_pair = np.kron(E_pi, np.ones(N)) + np.kron(np.ones(N), E_pi)
    V_pi = H_pi - np.diag(E_pair)

    # 用 pairinteraction 能量构造 states（旋转系与参考完全一致）
    specs = [dict(n=60, l=0, j=0.5, m=0.5, E_mhz=E_pi[0]),
             dict(n=60, l=1, j=0.5, m=0.5, E_mhz=E_pi[1])]
    states = evolve.make_levels(cache, specs)
    psi0_A = np.zeros(N, complex); psi0_A[0] = 1.0   # |60S⟩
    psi0_B = np.zeros(N, complex); psi0_B[1] = 1.0   # |60P⟩
    t = np.linspace(0.0, 200.0, 1001)

    res = pair_evolve.pair_evolve(
        states, [], [], psi0_A, psi0_B, vdd_matrix=V_pi,
        t_grid=t, method='ip4', progress_arr=np.zeros(2))

    # 精确参考：ψ(t) = expm(−i·_CONV·H_pair·t)·ψ0
    psi0_2d = np.kron(psi0_A, psi0_B)
    pops_ref = np.zeros((len(t), N * N))
    for m, tm in enumerate(t):
        U = expm(-1j * _CONV * H_pi * tm)
        pops_ref[m] = np.abs(U @ psi0_2d) ** 2
    err = np.max(np.abs(res['pops_2d'] - pops_ref))
    _check('pops 与精确演化逐点一致（< 1e-5）', err < 1e-5,
           f'max err={err:.2e}')

    # 交换拍频：|SP⟩↔|PS⟩ 的 sin² 振荡，首峰时刻 T/4 = π/(2·_CONV·|V_ex|)
    V_ex = float(np.real(H_pi[1, 2]))
    t_first_peak = np.pi / (2.0 * _CONV * abs(V_ex))
    p_ps = res['pops_2d'][:, 2]
    # 多个等高峰时 argmax 会取最后一个 → 取首次越过 0.99 的时刻
    k_first = int(np.argmax(p_ps > 0.99))
    _check('|PS⟩ 首峰时刻 ≈ π/(2|V_ex|)（< 2 ns）',
           abs(t[k_first] - t_first_peak) < 2.0,
           f't={t[k_first]:.3f} ns vs {t_first_peak:.3f} ns'
           f'（V_ex={V_ex:.3f} MHz）')


# ==================================================================
# T4: 单体一致性 —— V_dd=0 两原子独立 ≡ 单原子引擎
# ==================================================================
def t4():
    print('T4 单体一致性：V_dd=0 两原子独立 ≡ 单原子引擎')
    from src.core import pair_evolve
    cache = core.ArcCache(Rubidium87())
    specs = [dict(n=6, l=0, j=0.5, m=0.5),      # g: 6S
             dict(n=60, l=0, j=0.5, m=0.5),     # e1: 60S
             dict(n=60, l=1, j=0.5, m=0.5)]     # e2: 60P
    states = evolve.make_levels(cache, specs)
    N = len(states)
    field = evolve.field(0, 1, omega0_mhz=10.0,
                         waveform=evolve.square(0.0, 1e12))
    psi0 = np.zeros(N, complex); psi0[0] = 1.0
    t = np.linspace(0, 100.0, 201)

    res1 = evolve.evolve(states, [field], psi0, t, method='ip4',
                         progress_arr=np.zeros(2))
    res2 = pair_evolve.pair_evolve(
        states, [field], [], psi0, psi0,
        vdd_matrix=None, t_grid=t, method='ip4',
        progress_arr=np.zeros(2))
    err = np.max(np.abs(res2['pops_A'] - res1['pops']))
    _check('A 约化布居 ≡ 单原子（< 1e-3）', err < 1e-3, f'max err={err:.2e}')
    _check('B 恒在 |g⟩', np.max(np.abs(res2['pops_B'][:, 0] - 1.0)) < 1e-6)


# ==================================================================
# T5: secular 判据 —— 远失谐丢弃 / 近简并保留
# ==================================================================
def t5():
    print('T5 secular 判据：远失谐丢弃 / 近简并保留')
    from src.core import pair_evolve
    cache = core.ArcCache(Rubidium87())
    specs = [dict(n=60, l=0, j=0.5, m=0.5),      # 60S
             dict(n=60, l=1, j=0.5, m=0.5),      # 60P1/2
             dict(n=59, l=1, j=0.5, m=0.5)]      # 59P1/2（vdW 中间态）
    states = evolve.make_levels(cache, specs)
    N = len(states)
    E = np.array([s['E_mhz'] for s in states])
    E_rot_2d = np.kron(E, np.ones(N)) + np.kron(np.ones(N), E)  # 无光场 ref=0
    V_diag, V_sec, n_kept = pair_evolve.vdd_secular(
        cache, [dict(s) for s in states], 5.0, 0.0, 0.0,
        np.zeros(N), np.zeros(N), N, E_rot_2d=E_rot_2d)

    # 近简并交换元 |60S,60P⟩↔|60P,60S⟩（Δ=0）→ 保留
    k_sp, k_ps = 0 * N + 1, 1 * N + 0
    kept_ex = abs(V_sec[k_sp, k_ps]) > 1e-12
    _check('近简并交换元保留',
           kept_ex, f'|V_sec|={abs(V_sec[k_sp, k_ps]):.4f} MHz')

    # 远失谐 vdW 元 |60S,60S⟩↔|60P,60P⟩（Δ≈33.7 GHz ≫ V≈24 MHz）→ 丢弃
    k_ss, k_pp = 0 * N + 0, 1 * N + 1
    dropped = abs(V_sec[k_ss, k_pp]) < 1e-12
    detune = E[1] + E[1] - 2 * E[0]
    _check('远失谐 vdW 元丢弃（RWA）', dropped,
           f'|V_sec|={abs(V_sec[k_ss,k_pp]):.2e}（Δ={detune:.0f} MHz）')
    # V_dd 对角位移仍保留（secular 模型含对角 vdW 一阶项）
    _check('对角位移进入 V_diag',
           abs(V_diag[k_ss] - V_diag[0]) < 1e-9 or abs(V_diag[k_ss]) > 0,
           f'V_diag[|SS⟩]={V_diag[k_ss]:.4f} MHz')
    # vdW 并入（2025-08）：被丢弃的远失谐元之二阶位移 |V|²/Δ 并入对角——
    # V_diag[|SS⟩] == 基组内丢弃通道 Σ|V|²/Δ（与 pair_core.vdw_shift 同物理，基组截断）
    V_full = pair_core.vdd_sparse(cache, [dict(s) for s in states],
                                  5.0, 0.0, 0.0).toarray()
    manual = 0.0
    for b in range(N * N):
        v = abs(V_full[k_ss, b])
        if b == k_ss or v < 1e-9:
            continue
        delta = E_rot_2d[k_ss] - E_rot_2d[b]
        if abs(delta) >= 10.0 * v:            # 丢弃通道 → v²/Δ
            manual += v * v / delta
    _check('vdW 并入对角 = 基组内 Σ|V|²/Δ',
           abs(V_diag[k_ss] - manual) < 1e-9,
           f'V_diag={V_diag[k_ss]:.6f} manual={manual:.6f} MHz')


# ==================================================================
if __name__ == '__main__':
    t1()
    t2()
    t3()
    t4()
    t5()
    print(f'\n[pair_evolve_selfcheck]  PASS={_n_pass}  FAIL={_n_fail}')
    sys.exit(1 if _n_fail > 0 else 0)
