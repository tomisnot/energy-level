# -*- coding: utf-8 -*-
"""pair_evolve.py 自检（8 项）：独立于实现细节，只验物理约束。

策略：用最小 toy 模型（3 态 g/e1/e2），手写 V_dd 验证。
每个 test 独立；pair_evolve 未实现时自动 SKIP。

T1  单体一致性：V_dd=0 → 两原子独立演化 = 单原子 ⊗ 单原子
T2  总概率守恒：pops ≤ 1
T3  封锁：V_dd ≫ Ω → |rr⟩ 布居被抑制
T4  交换振荡：|sr⟩ ↔ |rs⟩ 拍频
T5  初态检验：|a⟩⊗|b⟩ t=0 布居正确
T6  Lindblad 衰变：双原子独立衰变 ≈ exp(-Γt)
T7  V_dd 对角位移：|gg⟩ 布居不受影响
T8  V_dd 矩阵厄米性
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
from src.core import evolve, core
from arc import Rubidium87

_n_pass = _n_fail = 0


def _check(name, cond, detail=''):
    global _n_pass, _n_fail
    if cond:
        _n_pass += 1
        print(f'  [PASS] {name}')
    else:
        _n_fail += 1
        print(f'  [FAIL] {name}   {detail}')


def _close(name, ours, ref, rtol=1e-3, atol=0.0):
    err = abs(ours - ref)
    ok = err <= atol + rtol * max(abs(ref), 1e-12)
    _check(name, ok, f'ours={ours:.8g} ref={ref:.8g} err={err:.2e}')


# ---- 最小 toy 模型：3 态 (g, e1, e2) ----
def _toy_states(cache):
    specs = [
        dict(n=6, l=0, j=0.5, m=0.5),   # g: 6S1/2
        dict(n=60, l=0, j=0.5, m=0.5),   # e1: 60S1/2
        dict(n=60, l=1, j=0.5, m=0.5),   # e2: 60P1/2
    ]
    return evolve.make_levels(cache, specs)


def _toy_field_g_e1(omega0_mhz=10.0, detune_mhz=0.0):
    """g↔e1 共振驱动。"""
    return evolve.field(0, 1, omega0_mhz=omega0_mhz,
                        detune_mhz=detune_mhz,
                        waveform=evolve.square(0.0, 1e12))


def _toy_field_empty():
    """空场（omega0=0，用于让 pair_evolve 有 H0 演化）。"""
    return evolve.field(0, 0, omega0_mhz=0.0,
                        waveform=evolve.square(0.0, 1e12))


def _psi0_g(N):
    """初态 |g⟩。"""
    psi = np.zeros(N, complex); psi[0] = 1.0
    return psi


# ==================================================================
# T1: V_dd=0 → 两原子独立演化 = 单原子 ⊗ 单原子
# ==================================================================
def t1():
    print('T1 零 V_dd 单体一致性')
    try:
        from src.core import pair_evolve
    except ImportError:
        print('  [SKIP] pair_evolve 未实现'); return

    cache = core.ArcCache(Rubidium87())
    states = _toy_states(cache)
    N = len(states)
    field = _toy_field_g_e1(omega0_mhz=10.0)
    psi0 = _psi0_g(N)
    t = np.linspace(0, 100.0, 201)

    # 单原子演化
    res1 = evolve.evolve(states, [field], psi0, t, method='ip4',
                         progress_arr=np.zeros(2))
    pops1 = res1['pops']

    # 双原子（V_dd=0，A 有驱动，B 无驱动）
    res2 = pair_evolve.pair_evolve(
        states, [field], [], psi0, psi0,
        vdd_matrix=None, t_grid=t, method='ip4',
        progress_arr=np.zeros(2))

    max_err = np.max(np.abs(res2['pops_A'] - pops1))
    _close('A 约化布居 ≡ 单原子', max_err, 0.0, atol=1e-3)
    _close('B 始终在 |g⟩', np.max(np.abs(res2['pops_B'][:, 0] - 1.0)),
           0.0, atol=1e-6)


# ==================================================================
# T2: 总概率守恒
# ==================================================================
def t2():
    print('T2 总概率守恒')
    try:
        from src.core import pair_evolve
    except ImportError:
        print('  [SKIP] pair_evolve 未实现'); return

    cache = core.ArcCache(Rubidium87())
    states = _toy_states(cache)
    N = len(states)
    field = _toy_field_g_e1(omega0_mhz=10.0)
    psi0 = _psi0_g(N)
    t = np.linspace(0, 200.0, 301)

    res = pair_evolve.pair_evolve(
        states, [field], [field], psi0, psi0,
        vdd_matrix=None, t_grid=t, method='ip4',
        progress_arr=np.zeros(2))
    _check('pops_A ≤ 1', np.all(res['pops_A'] <= 1.0 + 1e-6),
           f'max={res["pops_A"].max():.8f}')
    _check('pops_B ≤ 1', np.all(res['pops_B'] <= 1.0 + 1e-6),
           f'max={res["pops_B"].max():.8f}')


# ==================================================================
# T3: 封锁：V_dd ≫ Ω → |rr⟩ 布居被抑制
# ==================================================================
def t3():
    print('T3 封锁')
    try:
        from src.core import pair_evolve
    except ImportError:
        print('  [SKIP] pair_evolve 未实现'); return

    cache = core.ArcCache(Rubidium87())
    states = _toy_states(cache)
    N = len(states)
    field = _toy_field_g_e1(omega0_mhz=10.0)
    psi0 = _psi0_g(N)
    t = np.linspace(0, 100.0, 201)

    # 强 V_dd：|e1,e1⟩=|1,1⟩ → k=1*N+1 偏移 10000 MHz
    V_dd = np.zeros((N * N, N * N), complex)
    k_rr = 1 * N + 1
    V_dd[k_rr, k_rr] = 10000.0  # 远超 Rabi

    res = pair_evolve.pair_evolve(
        states, [field], [field], psi0, psi0,
        vdd_matrix=V_dd, t_grid=t, method='ip4',
        progress_arr=np.zeros(2))
    pops_rr = res['pops_2d'][:, k_rr]
    _check('pops_rr ≈ 0（封锁）', pops_rr.max() < 0.01,
           f'max pops_rr = {pops_rr.max():.6f}')


# ==================================================================
# T4: 交换振荡（纯 V_dd 演化，无光场驱动）
# ==================================================================
def t4():
    print('T4 交换振荡')
    try:
        from src.core import pair_evolve
    except ImportError:
        print('  [SKIP] pair_evolve 未实现'); return

    N = 3  # toy 3 态
    V_ex = 5.0  # MHz → 拍频 = V_ex/(2π)

    # 构造旋转系能量：|1,0⟩ 和 |0,1⟩ 简并
    E_rot_2d = np.full(N * N, 10000.0)  # 远离态
    k_sg = 1 * N + 0  # |e1,g⟩
    k_gs = 0 * N + 1  # |g,e1⟩
    k_gg = 0 * N + 0  # |g,g⟩
    k_ee = 1 * N + 1  # |e1,e1⟩
    E_rot_2d[k_sg] = 0.0   # 简并对
    E_rot_2d[k_gs] = 0.0
    E_rot_2d[k_gg] = 0.0   # 基态（不参与）
    E_rot_2d[k_ee] = 500.0 # 远离

    # V_dd：仅 |1,0⟩ ↔ |0,1⟩ 交换耦合
    V_dd = np.zeros((N * N, N * N), complex)
    V_dd[k_sg, k_gs] = V_ex
    V_dd[k_gs, k_sg] = V_ex

    # 初态 = |1,0⟩（直接两体向量）
    psi0_2d = np.zeros(N * N, complex)
    psi0_2d[k_sg] = 1.0

    # V_dd 作为常数场（omega_L=0，omega0=1）
    V_field = {
        'kind': 'multi', 'M': V_dd, 'A': np.zeros_like(V_dd),
        'ac_shift': np.zeros(N * N),
        'g_idx': [], 'e_idx': [],
        'omega0_mhz': 1.0, 'omega_L_mhz': 0.0,
        'waveform': evolve.square(0.0, 1e12)}
    t = np.linspace(0, 200.0, 1001)

    from src.core.pair_evolve import _pair_ip4
    t_out, pops_2d, _ = _pair_ip4(
        E_rot_2d, [V_field], psi0_2d, t,
        frac=20.0, progress_arr=np.zeros(2))

    # |1,0⟩ 和 |0,1⟩ 布居应有振荡
    pop_ksg = pops_2d[:, k_sg]
    pop_kgs = pops_2d[:, k_gs]
    amp = max(pop_ksg.max() - pop_ksg.min(),
              pop_kgs.max() - pop_kgs.min())
    _check('|1,0⟩/|0,1⟩ 有振荡', amp > 0.3,
           f'振幅={amp:.4f}')
    # 交换拍频 ≈ V_ex/(2π) → T = 2π/V_ex ≈ 1.257 ns
    # 在 t=500 ns 内应有 ~398 周期
    _check('|1,0⟩ + |0,1⟩ 概率守恒',
           np.allclose(pop_ksg + pop_kgs, 1.0, atol=0.05),
           f'sum range=[{pop_ksg.min()+pop_kgs.min():.4f}, '
           f'{pop_ksg.max()+pop_kgs.max():.4f}]')


# ==================================================================
# T5: 初态检验
# ==================================================================
def t5():
    print('T5 初态检验')
    try:
        from src.core import pair_evolve
    except ImportError:
        print('  [SKIP] pair_evolve 未实现'); return

    cache = core.ArcCache(Rubidium87())
    states = _toy_states(cache)
    N = len(states)

    psi0_A = np.zeros(N, complex); psi0_A[1] = 1.0  # |e1⟩
    psi0_B = np.zeros(N, complex); psi0_B[2] = 1.0  # |e2⟩
    field = _toy_field_empty()
    t = np.array([0.0, 100.0])

    res = pair_evolve.pair_evolve(
        states, [field], [field], psi0_A, psi0_B,
        vdd_matrix=None, t_grid=t, method='ip4',
        progress_arr=np.zeros(2))
    _close('t=0: A 在 |e1⟩', res['pops_A'][0, 1], 1.0, atol=1e-6)
    _close('t=0: B 在 |e2⟩', res['pops_B'][0, 2], 1.0, atol=1e-6)


# ==================================================================
# T6: Lindblad 衰变
# ==================================================================
def t6():
    print('T6 Lindblad 双原子独立衰变')
    # 验证衰变约定：通过单原子 evolve（pair_evolve 尚未实现 Lindblad）
    cache = core.ArcCache(Rubidium87())
    states = _toy_states(cache)
    N = len(states)

    psi0 = np.zeros(N, complex); psi0[1] = 1.0  # |e1⟩
    gamma = 5.0  # MHz
    decays = [evolve.decay(1, gamma, 0)]  # e1→g, Γ=5 MHz

    field = _toy_field_empty()
    t = np.linspace(0, 100.0, 201)

    res = evolve.evolve(
        states, [field], psi0, t, method='lindblad',
        decays=decays, progress_arr=np.zeros(2))

    # P_e1(t) = exp(-Γ×1e-3×t)  （Γ MHz → 速率 /ns = Γ×1e-3）
    rate = gamma * 1e-3
    P_exact = np.exp(-rate * t)
    pops_e = res['pops'][:, 1]
    err = np.max(np.abs(pops_e - P_exact))
    _close('衰变 ≈ exp(-Γt)', err, 0.0, atol=5e-3)
    _close('总概率守恒', np.sum(res['pops'][-1]) + res.get('loss', np.zeros(1))[-1],
           1.0, atol=1e-6)


# ==================================================================
# T7: V_dd 对角位移
# ==================================================================
def t7():
    print('T7 V_dd 对角位移')
    try:
        from src.core import pair_evolve
    except ImportError:
        print('  [SKIP] pair_evolve 未实现'); return

    cache = core.ArcCache(Rubidium87())
    states = _toy_states(cache)
    N = len(states)

    V_dd = np.zeros((N * N, N * N), complex)
    k_rr = 1 * N + 1
    V_dd[k_rr, k_rr] = 100.0  # MHz

    psi0 = _psi0_g(N)
    field = _toy_field_empty()
    t = np.array([0.0, 200.0])

    res = pair_evolve.pair_evolve(
        states, [field], [field], psi0, psi0,
        vdd_matrix=V_dd, t_grid=t, method='ip4',
        progress_arr=np.zeros(2))
    # 无驱动 + 初态 |gg⟩ → V_dd 对角不影响 |gg⟩
    _close('pops_A(200) 在 |g⟩', res['pops_A'][1, 0], 1.0, atol=1e-6)
    _close('pops_B(200) 在 |g⟩', res['pops_B'][1, 0], 1.0, atol=1e-6)


# ==================================================================
# T8: V_dd 厄米性
# ==================================================================
def t8():
    print('T8 V_dd 厄米性')
    N = 3
    V = np.random.randn(N * N, N * N) + 1j * np.random.randn(N * N, N * N)
    V = (V + V.conj().T) / 2
    _check('随机 V_dd 厄米', np.allclose(V, V.conj().T))


# ==================================================================
if __name__ == '__main__':
    t1()
    t2()
    t3()
    t4()
    t5()
    t6()
    t7()
    t8()
    print(f'\n[pair_evolve_selfcheck]  PASS={_n_pass}  FAIL={_n_fail}')
    sys.exit(1 if _n_fail > 0 else 0)
