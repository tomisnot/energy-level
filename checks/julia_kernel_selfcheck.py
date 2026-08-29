# -*- coding: utf-8 -*-
"""Julia 内核自检（julia/ 包，分阶段验证门 G0..G6）。

验证阶梯（每阶段先复现此前全部 gate）：
  G0 数据管道：NPZ schema 往返（Python→Julia 解析→回显）逐位一致
  G1 单段传播子 vs expm / QO.jl（S1）
  G2 稠密路径端到端 EP1–EP5（S2）
  G3 稀疏基线 gen-3 vs Python 现状（S3）
  G4/G5 Wosc 权重与段传播子（S4/S5）
  G6 bench_n20 ≤60s（S6）

无 Julia 时整组 SKIP（照 pairinteraction 模式，不误报）。
运行：D:\\python310\\python.exe -u checks\\julia_kernel_selfcheck.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

import _julia_bridge as jb
from checklib import Checker

ck = Checker('julia_kernel')
WORK = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..',
                    'julia', 'work')
os.makedirs(WORK, exist_ok=True)

_CONV = 2 * np.pi * 1e-3          # MHz·ns → rad（与引擎一致）


def _run_engine(name, req_kwargs, tol=None, timeout=900):
    """构造请求 → 跑 Julia 引擎 → 返回 (resp dict | None, elapsed)。"""
    import time as _time
    req_path = os.path.join(WORK, f'{name}_req.npz')
    resp_path = os.path.join(WORK, f'{name}_resp.npz')
    if os.path.exists(resp_path):
        os.remove(resp_path)
    jb.write_request(req_path, **req_kwargs)
    t0 = _time.perf_counter()
    try:
        code, out = jb.run_cli(req_path, resp_path, timeout=timeout)
    except Exception as e:                       # noqa: BLE001
        ck.check(f'{name} 运行', False, f'异常: {e}')
        return None, 0.0
    el = _time.perf_counter() - t0
    if code != 0:
        d = np.load(resp_path) if os.path.exists(resp_path) else {}
        msg = d.get('err_msg', b'?') if len(d) else out[-300:]
        ck.check(f'{name} 运行', False, f'code={code} msg={msg}')
        return None, el
    d = dict(np.load(resp_path))
    if int(d['ok'][0]) != 1:
        ck.check(f'{name} 运行', False,
                 f"err={bytes(d['err_msg']).decode('utf-8', 'replace')}")
        return None, el
    return d, el


def _py_fields(A, omega, env_callable):
    """合成 kind='exact' 场 dict（供 Python 引擎/单原子参考用）。"""
    return {'kind': 'exact', 'A': np.asarray(A, float),
            'omega_L_mhz': float(omega), 'waveform': env_callable,
            'label': 'synthetic'}


def _g2_states(E):
    return [{'E_mhz': float(e), 'label': f's{i}'}
            for i, e in enumerate(E)]


# ============================================================
# G2：稠密路径端到端 EP1–EP5（Python 引擎 = 权威参考）
# ============================================================

def g2_ep1_single_atom_limit():
    """EP1：V_dd=0 ≡ 单原子 exact 引擎直积（pops_A 逐点一致）。"""
    from src.core import evolve_exact as ex
    from src.core import evolve_exact_pair as pep
    N, N2 = 2, 4
    E = [0.0, 1000.0]
    A_A = [[0.0, 20.0], [20.0, 0.0]]
    gauss = lambda t: np.exp(-(np.asarray(t, float) - 100.0) ** 2
                             / (2 * 30.0 ** 2))          # noqa: E731
    t = np.linspace(0.0, 200.0, 201)
    E_pair = [E[i] + E[j] for i in range(N) for j in range(N)]
    # Python 单原子参考（合成场，无 ARC 依赖）
    states = _g2_states(E)
    fA = [_py_fields(A_A, 1000.0, gauss)]
    rA = ex.evolve_exact(states, fA, 0, t, tol=1e-8)
    d, _ = _run_engine('g2_ep1', dict(
        N=N, E_pair=E_pair, vdd=None,
        groups=[(1000.0, np.kron(A_A, np.eye(2)))],
        fields=[(0, 0, (100.0, 30.0, 1.0, 0.0), None, None)],
        psi0=(0, 0), t_grid=t, tol=1e-8))
    if d is None:
        return
    diff = float(np.max(np.abs(d['pops_A'] - rA['pops'])))
    ck.check('EP1a V_dd=0 ≡ 单原子（pops_A ≤5e-6）', diff <= 5e-6,
             f'max|Δ|={diff:.2e}')
    stuck = float(np.max(np.abs(d['pops_B'][:, 0] - 1.0)))
    # 门限说明：微波场景（r=Ω/ω=0.02）M=3 光子截断底 ≈7.3e-7，两实现
    # （Python/Julia）逐位一致；原 5e-7 对应原版光频场景（r~1e-8）。
    ck.check('EP1b B 恒初态（≤1e-6，截断底校准）', stuck <= 1e-6,
             f'{stuck:.2e}')
    # 最强门：Julia ≡ Python pair 引擎直接对拍（同 tol 同场景）
    rP = pep.evolve_exact_pair(states, fA, [], 0, 0, t, vdd_matrix=None,
                               tol=1e-8)['pops_2d']
    cross = float(np.max(np.abs(d['pops_2d'] - rP)))
    ck.check('EP1c Julia≡Python pair 逐位（≤1e-9）', cross <= 1e-9,
             f'{cross:.2e}')


def g2_ep2_exchange():
    """EP2：无场交换振荡 vs 解析 2×2（π/(2|V|)）。"""
    N, N2 = 2, 4
    E = [0.0, 1000.0]
    V = 10.0
    vdd = np.zeros((N2, N2), complex)
    vdd[1, 2] = vdd[2, 1] = V                 # |g,e⟩ ↔ |e,g⟩
    t_peak = np.pi / (2 * _CONV * V)          # ≈ 25.0 ns
    t = np.linspace(0.0, 2.5 * t_peak, 401)
    E_pair = [E[i] + E[j] for i in range(N) for j in range(N)]
    d, _ = _run_engine('g2_ep2', dict(
        N=N, E_pair=E_pair, vdd=vdd, groups=[], fields=[],
        psi0=(0, 1), t_grid=t, tol=1e-7))
    if d is None:
        return
    p_ps = d['pops_2d'][:, 2]                 # 0-based col2 = k1=3 = |e,g⟩
    i_pk = int(np.argmax(p_ps))
    dt = abs(t[i_pk] - t_peak)
    ck.check('EP2a 交换首峰时刻（2%）', dt <= 0.02 * t_peak,
             f't={t[i_pk]:.3f} vs {t_peak:.3f}')
    ck.check('EP2b 交换幅度（≥0.99）', float(p_ps[i_pk]) >= 0.99,
             f'{p_ps[i_pk]:.6f}')


def g2_ep3_vdw():
    """EP3：vdW 远失谐通道 vs 稠密 DOP853 + 相干相位提取。"""
    from scipy.integrate import solve_ivp
    N, N2 = 3, 9
    E = [0.0, 1000.0, 0.0]                    # g, e, u
    W, Ddet = 50.0, 2000.0
    E_pair = np.array([E[i] + E[j] for i in range(N) for j in range(N)])
    vdd = np.zeros((N2, N2), complex)
    vdd[0, 4] = vdd[4, 0] = W                 # |gg⟩(k0=0) ↔ |ee⟩(k0=1·3+1=4)
    t = np.linspace(0.0, 800.0, 401)
    # Python 权威参考：稠密 DOP853（无场），初态 |gg⟩（schema psi0 为直积）
    H = np.diag(E_pair) + vdd

    def rhs(tt, y):
        return -1j * _CONV * (H @ y)
    psi0b = np.zeros(N2, complex)
    psi0b[0] = 1.0
    solb = solve_ivp(rhs, (t[0], t[-1]), psi0b, t_eval=t,
                     method='DOP853', rtol=1e-10, atol=1e-12)
    refb = np.abs(solb.y.T) ** 2
    d, _ = _run_engine('g2_ep3', dict(
        N=N, E_pair=E_pair, vdd=vdd, groups=[], fields=[],
        psi0=(0, 0), t_grid=t, tol=1e-7))
    if d is None:
        return
    diff = float(np.max(np.abs(d['pops_2d'] - refb)))
    ck.check('EP3a vdW 通道 vs DOP853（≤1e-5）', diff <= 1e-5,
             f'max|Δ|={diff:.2e}')

    # EP3b 相位提取（AC 干涉法，record_amp）
    def run_amp(t_end, psiA3):
        tt = np.linspace(0.0, t_end, 11)
        dd, _ = _run_engine(f'g2_ep3b_{int(t_end)}', dict(
            N=N, E_pair=E_pair, vdd=vdd, groups=[], fields=[],
            psi0=(psiA3, np.array([1.0, 0, 0], complex)),
            t_grid=tt, tol=1e-8, record_amp=True))
        return dd
    psiA3 = np.zeros(3, complex); psiA3[0] = psiA3[2] = 1 / np.sqrt(2)
    d1, d2 = run_amp(100.0, psiA3), run_amp(300.0, psiA3)
    if d1 is None or d2 is None:
        return
    ph1 = np.angle(d1['psi_amp'][0] * np.conj(d1['psi_amp'][6]))  # |gg⟩ vs |ug⟩
    ph2 = np.angle(d2['psi_amp'][0] * np.conj(d2['psi_amp'][6]))
    dphi = (ph2 - ph1 + np.pi) % (2 * np.pi) - np.pi
    delta_fit = -dphi / (_CONV * (300.0 - 100.0))
    delta_th = -W ** 2 / Ddet
    ck.check('EP3b 相干相位 δ（≤0.125 MHz）',
             abs(delta_fit - delta_th) <= max(0.1 * abs(delta_th), 0.02),
             f'δ_fit={delta_fit:.4f} vs {delta_th:.4f} MHz')


def g2_ep4_decay():
    """EP4：寿命衰变 vs 解析 e^{−Γt} + 守恒。"""
    N, N2 = 2, 4
    E = [0.0, 1000.0]
    G_mhz = 10.0
    rate = G_mhz * 1e-3                       # /ns（Γ MHz → 速率 Γ×1e-3）
    E_pair = [E[i] + E[j] for i in range(N) for j in range(N)]
    gamma_A = np.zeros(N2); gamma_A[2] = gamma_A[3] = rate   # 0-based k=i·N+j：|e,g⟩=2,|e,e⟩=3
    gamma_B = np.zeros(N2)
    t = np.linspace(0.0, 400.0, 201)
    d, _ = _run_engine('g2_ep4', dict(
        N=N, E_pair=E_pair, vdd=None, groups=[], fields=[],
        psi0=(1, 0), t_grid=t, tol=1e-8,
        gamma_A=gamma_A, gamma_B=gamma_B))
    if d is None:
        return
    ref = np.exp(-rate * t)
    diff = float(np.max(np.abs(d['pops_A'][:, 1] - ref)))
    ck.check('EP4a 寿命指数衰变（≤1e-6）', diff <= 1e-6, f'{diff:.2e}')
    cons = float(np.max(np.abs(d['pops_2d'].sum(axis=1) + d['loss_A']
                               + d['loss_B'] - 1.0)))
    ck.check('EP4b 守恒（≤1e-3）', cons <= 1e-3, f'{cons:.2e}')


def g2_ep5_tol_convergence():
    """EP5：tol 收敛性 + 幺正范数（Python 引擎 = 权威）。"""
    from src.core import evolve_exact_pair as pep
    N, N2 = 2, 4
    E = [0.0, 1000.0]
    A_A = [[0.0, 20.0], [20.0, 0.0]]
    gauss = lambda t: np.exp(-(np.asarray(t, float) - 100.0) ** 2
                             / (2 * 30.0 ** 2))          # noqa: E731
    t = np.linspace(0.0, 200.0, 201)
    E_pair = [E[i] + E[j] for i in range(N) for j in range(N)]
    vdd = np.zeros((N2, N2), complex)
    vdd[1, 2] = vdd[2, 1] = 10.0
    states = _g2_states(E)
    fA = [_py_fields(A_A, 1000.0, gauss)]
    # Python 深收敛参考（1e-9）作为权威
    r_deep = pep.evolve_exact_pair(states, fA, [], 0, 0, t,
                                   vdd_matrix=vdd, tol=1e-9)['pops_2d']
    refs = {}
    errs = {}
    for tol in (1e-4, 1e-7):
        d, el = _run_engine(f'g2_ep5_{tol:.0e}'.replace('+', '').replace(
            '-', ''), dict(
            N=N, E_pair=E_pair, vdd=vdd,
            groups=[(1000.0, np.kron(A_A, np.eye(2)))],
            fields=[(0, 0, (100.0, 30.0, 1.0, 0.0), None, None)],
            psi0=(0, 0), t_grid=t, tol=tol))
        if d is None:
            return
        errs[tol] = float(np.max(np.abs(d['pops_2d'] - r_deep)))
        if tol == 1e-7:
            r7 = pep.evolve_exact_pair(states, fA, [], 0, 0, t,
                                       vdd_matrix=vdd, tol=1e-7)['pops_2d']
            cross = float(np.max(np.abs(d['pops_2d'] - r7)))
            ck.check('EP5 Julia≡Python 直接对拍（≤1e-9）', cross <= 1e-9,
                     f'{cross:.2e}')
            cons = float(np.max(np.abs(d['pops_2d'].sum(axis=1) - 1.0)))
            ck.check('EP5b 幺正范数（≤5e-6）', cons <= 5e-6, f'{cons:.2e}')
            print(f"    [G2] EP5 耗时 tol=1e-7: {el:.1f}s, "
                  f"段数={int(d['n_segments'][0])}")
    ck.check('EP5a tol 收敛单调（err(1e-7)<err(1e-4) 且 ≤2e-5）',
             errs[1e-7] < errs[1e-4] and errs[1e-7] <= 2e-5,
             f'err(1e-4)={errs[1e-4]:.2e} err(1e-7)={errs[1e-7]:.2e}')


def g55_arbitrary_wave():
    """G5.5：任意波形 → FFT 主音分解 → multi-tone exact。

    (a) 双音合成波分解：载频/包络 vs 解析真值；
    (b) 分解表示 vs 解析 multi-tone 请求的引擎输出一致；
    (c) 啁啾（密集谱）→ 明确报错指引 ip4。
    """
    dt = 0.01
    t = np.arange(0.0, 200.0 + dt / 2, dt)
    f1, f2 = 1000.0, 2500.0          # MHz
    g1 = np.exp(-(t - 100.0) ** 2 / (2 * 30.0 ** 2))
    edge = 20.0
    rise = 0.5 * (1 - np.cos(np.pi * np.clip((t - 40.0) / edge, 0, 1)))
    fall = 1 - 0.5 * (1 - np.cos(np.pi * np.clip((t - 120.0) / edge, 0, 1)))
    sq = np.where((t >= 40.0) & (t <= 120.0 + edge), rise * fall, 0.0)
    amp = g1 * np.cos(2 * np.pi * f1 * 1e-3 * t) +         0.5 * sq * np.cos(2 * np.pi * f2 * 1e-3 * t)
    # (a) 分解
    try:
        tones, resid = jb.decompose_tones(t, amp, max_tones=4)
    except ValueError as e:
        ck.check('G5.5a 双音分解', False, str(e))
        return
    ck.check('G5.5a 分解出 2 音', len(tones) == 2,
             f'{[round(tt[0], 3) for tt in tones]}')
    freqs = sorted(tt[0] for tt in tones)
    ck.check('G5.5a 载频（质心，≤0.05 MHz）',
             abs(freqs[0] - f1) <= 0.05 and abs(freqs[1] - f2) <= 0.05,
             str(freqs))
    ck.check('G5.5a 分解残差（≤2e-2）', resid <= 2e-2, f'{resid:.2e}')
    # 包络 vs 解析（在采样点上，容差含带通边缘效应）
    env_err = 0.0
    for f_mhz, te, ev in tones:
        ana = g1 if abs(f_mhz - f1) < 1 else sq * 0.5
        ana_i = np.interp(te, t, ana)
        env_err = max(env_err, float(np.max(np.abs(ev - ana_i))))
    ck.check('G5.5a 包络还原（≤5e-3）', env_err <= 5e-3, f'{env_err:.2e}')

    # (b) 分解表示 vs 解析请求，引擎输出对拍
    N, N2 = 2, 4
    E = [0.0, 1000.0]
    E_pair = [E[i] + E[j] for i in range(N) for j in range(N)]
    A_A = [[0.0, 20.0], [20.0, 0.0]]
    A_B2 = [[0.0, 10.0], [10.0, 0.0]]
    K2_1 = np.kron(A_A, np.eye(2))
    K2_2 = np.kron(np.eye(2), A_B2)
    t_grid = np.linspace(0, 200, 201)
    # 解析请求（高斯+方波两载频）
    kw_ana = dict(N=N, E_pair=E_pair, vdd=None,
                  groups=[(f1, K2_1), (f2, K2_2)],
                  fields=[(0, 0, (100.0, 30.0, 1.0, 0.0), None, None),
                          (1, 1, (40.0, edge, 0.5, 120.0), None, None)],
                  psi0=(0, 0), t_grid=t_grid, tol=1e-6, dense_max=0)
    # 分解请求（采样包络 f_kind=2）
    fld = []
    for f_mhz, te, ev in tones:
        gi = 0 if abs(f_mhz - f1) < 1 else 1
        fld.append((gi, 2, (0.0, 0.0, 0.0, 0.0), te, ev))
    kw_dec = dict(N=N, E_pair=E_pair, vdd=None,
                  groups=[(f1, K2_1), (f2, K2_2)],
                  fields=fld, psi0=(0, 0), t_grid=t_grid, tol=1e-6,
                  dense_max=0)
    d_ana, _ = _run_engine('g55_ana', kw_ana)
    d_dec, _ = _run_engine('g55_dec', kw_dec)
    if d_ana is None or d_dec is None:
        return
    cross = float(np.max(np.abs(d_ana['pops_2d'] - d_dec['pops_2d'])))
    ck.check('G5.5b 分解表示 ≡ 解析请求（≤5e-3）', cross <= 5e-3,
             f'{cross:.2e}')

    # (c) 啁啾（密集谱）→ 拒绝并指引 ip4
    f0 = 1000.0
    chirp = 2.0                    # MHz/ns 扫频速率
    phase = 2 * np.pi * (f0 * 1e-3 * t + 0.5 * chirp * 1e-3 * t ** 2)
    amp_c = np.cos(phase) * g1
    try:
        jb.decompose_tones(t, amp_c, max_tones=6)
        ck.check('G5.5c 啁啾拒绝', False, '未报错')
    except ValueError as e:
        ck.check('G5.5c 啁啾拒绝并指引 ip4', 'ip4' in str(e), str(e)[:80])


def g3_sparse_ep6():
    """G3：强制稀疏（gen-3 IP Volterra–GMRES）vs 稠密，镜像 EP6 门限。"""
    N, N2 = 2, 4
    E = [0.0, 1000.0]
    E_pair = [E[i] + E[j] for i in range(N) for j in range(N)]
    A_A = [[0.0, 20.0], [20.0, 0.0]]
    kwargs = dict(
        N=N, E_pair=E_pair, vdd=None,
        groups=[(1000.0, np.kron(A_A, np.eye(2)))],
        fields=[(0, 0, (100.0, 30.0, 1.0, 0.0), None, None)],
        psi0=(0, 0), t_grid=np.linspace(0, 200, 101), tol=1e-7)
    d_dense, _ = _run_engine('g3_dense', dict(kwargs, dense_max=700))
    d_sparse, _ = _run_engine('g3_sparse', dict(kwargs, dense_max=0))
    if d_dense is None or d_sparse is None:
        return
    diff = float(np.max(np.abs(d_dense['pops_2d'] - d_sparse['pops_2d'])))
    ck.check('EP6a 稀疏vs稠密（≤5e-4）', diff <= 5e-4, f'{diff:.2e}')
    cons = float(np.max(np.abs(d_sparse['pops_2d'].sum(axis=1) - 1.0)))
    ck.check('EP6b 稀疏范数（≤2e-3）', cons <= 2e-3, f'{cons:.2e}')


def g2_cancel():
    """取消语义：cancel 文件存在 → 退出码 2 / ok=2。
    场景带光场（走自适应路径——快速路径不设取消检查，与 Python 一致）。"""
    req_path = os.path.join(WORK, 'g2_cancel_req.npz')
    resp_path = os.path.join(WORK, 'g2_cancel_resp.npz')
    cancel_path = os.path.join(WORK, 'g2_cancel.flag')
    N, N2 = 2, 4
    E = [0.0, 1000.0]
    E_pair = [E[i] + E[j] for i in range(N) for j in range(N)]
    A_A = [[0.0, 20.0], [20.0, 0.0]]
    jb.write_request(req_path, N=N, E_pair=E_pair, vdd=None,
                     groups=[(1000.0, np.kron(A_A, np.eye(2)))],
                     fields=[(0, 0, (100.0, 30.0, 1.0, 0.0), None, None)],
                     psi0=(0, 0), t_grid=np.linspace(0, 100, 101),
                     tol=1e-6)
    with open(cancel_path, 'w', encoding='utf-8') as f:
        f.write('1')
    code, _ = jb.run_cli(req_path, resp_path, cancel_file=cancel_path)
    os.remove(cancel_path)
    d = dict(np.load(resp_path)) if os.path.exists(resp_path) else {}
    ok = int(d['ok'][0]) if 'ok' in d else -1
    ck.check('G2 取消语义（rc=2, ok=2）', code == 2 and ok == 2,
             f'code={code} ok={ok}')


def main():
    exe = jb.find_julia()
    if exe is None:
        ck.skip('julia_kernel 全部', '未找到 julia.exe')
        sys.exit(0 if ck.summary() else 1)
    print(f'[julia] {exe}')
    g0_roundtrip()
    g2_ep1_single_atom_limit()
    g2_ep2_exchange()
    g2_ep3_vdw()
    g2_ep4_decay()
    g2_ep5_tol_convergence()
    g2_cancel()
    g3_sparse_ep6()
    g55_arbitrary_wave()
    # ---- 后续阶段 gate 随开发追加（G4..G6）----
    sys.exit(0 if ck.summary() else 1)


# ============================================================
# G0：NPZ 往返（EP5/EP6 合成场景 + 多组/方波/采样包络覆盖）
# ============================================================
def _g0_scenario():
    """EP5/EP6 合成模型 + schema 覆盖增强。

    N=2（g E=0, e E=1000 MHz）；V_dd 交换 10 MHz（|g,e⟩↔|e,g⟩）；
    组 0：ω=1000 MHz，A 侧 Rabi 20 MHz 高斯（center 100 σ 30）+
    B 侧 Rabi 8 MHz 采样包络（同频合并覆盖）；
    组 1：ω=2000 MHz，B 侧 Rabi 5 MHz 方波（start 10 edge 5 height 0.8）。
    """
    N = 2
    N2 = N * N
    E = np.array([0.0, 1000.0])
    E_pair = np.array([E[i] + E[j] for i in range(N) for j in range(N)])
    vdd = np.zeros((N2, N2), complex)
    vdd[0 * N + 1, 1 * N + 0] = 10.0     # |g,e⟩ ↔ |e,g⟩
    vdd[1 * N + 0, 0 * N + 1] = 10.0
    # 组 0 K2：A 侧 kron(A_A, I)；B 侧 kron(I, A_B)（同频合并）
    A_A = np.array([[0.0, 20.0], [20.0, 0.0]])
    A_B = np.array([[0.0, 8.0], [8.0, 0.0]])
    I2 = np.eye(2)
    K2_0 = np.kron(A_A, I2) + np.kron(I2, A_B)
    # 组 1 K2：B 侧方波场
    A_B2 = np.array([[0.0, 5.0], [5.0, 0.0]])
    K2_1 = np.kron(I2, A_B2)
    groups = [(1000.0, K2_0), (2000.0, K2_1)]
    # 采样包络（kind=2）：B 侧 8 MHz 场，梯形采样
    samp_t = np.linspace(0.0, 200.0, 41)
    samp_v = np.clip(np.minimum((samp_t - 20.0) / 15.0,
                                (180.0 - samp_t) / 15.0), 0.0, 0.9)
    fields = [
        (0, 0, (100.0, 30.0, 1.0, 0.0), None, None),      # A 高斯
        (0, 2, (0.0, 0.0, 0.0, 0.0), samp_t, samp_v),     # B 采样（同频组 0）
        (1, 1, (10.0, 5.0, 0.8, 20.0), None, None),       # B 方波（组 1）
    ]
    return dict(N=N, E_pair=E_pair, vdd=vdd, groups=groups, fields=fields,
                psi0=(0, 0), t_grid=np.linspace(0.0, 200.0, 201),
                tol=1e-6, gamma_pair=None,
                samp_t=samp_t, samp_v=samp_v)


def g0_roundtrip():
    req_path = os.path.join(WORK, 'g0_req.npz')
    resp_path = os.path.join(WORK, 'g0_resp.npz')
    sc = _g0_scenario()
    jb.write_request(req_path, sc['N'], sc['E_pair'], vdd=sc['vdd'],
                     groups=sc['groups'], fields=sc['fields'],
                     psi0=sc['psi0'], t_grid=sc['t_grid'], tol=sc['tol'])
    code, out = jb.run_cli(req_path, resp_path, echo=True)
    ck.check('G0 CLI 退出码', code == 0, f'code={code}\n{out[-400:]}')
    if code != 0:
        return
    d = np.load(resp_path)
    ck.check('G0 ok 标志', int(d['ok'][0]) == 1, str(d['ok']))
    ck.check('G0 N', int(d['e_N'][0]) == sc['N'])
    ck.check('G0 E_pair 逐位一致',
             np.array_equal(d['e_E_pair'], sc['E_pair']),
             f"max|Δ|={np.max(np.abs(d['e_E_pair'] - sc['E_pair'])) if 'e_E_pair' in d else '?'}")
    ck.check('G0 t_grid 一致', np.array_equal(d['e_t_grid'], sc['t_grid']))
    ck.check('G0 组数', int(d['e_n_groups'][0]) == 2)
    ck.check('G0 载频一致',
             np.allclose(d['e_group_omega'], [1000.0, 2000.0],
                         rtol=0, atol=0))
    # V_dd COO 逐元一致（按 (r,c) 排序对比；CSC findnz 顺序可不同）
    key_r = np.array([1, 2], np.int64)      # (1,2),(2,1) 的 0-based 坐标
    key_c = np.array([2, 1], np.int64)
    key_v = np.array([10.0 + 0j, 10.0 + 0j], complex)
    gr, gc, gv = d['e_vdd_row'], d['e_vdd_col'], d['e_vdd_data']
    ordj = np.lexsort((gc, gr))
    ordk = np.lexsort((key_c, key_r))
    ck.check('G0 V_dd 形状一致', len(gr) == len(key_r),
             f'{len(gr)} vs {len(key_r)}')
    if len(gr) == len(key_r):
        ck.check('G0 V_dd 元素逐位一致',
                 np.array_equal(gr[ordj], key_r[ordk])
                 and np.array_equal(gc[ordj], key_c[ordk])
                 and np.array_equal(gv[ordj], key_v[ordk]))
    # 包络参数一致（含绝对平移）
    ck.check('G0 高斯包络参数', np.array_equal(d['e_f_p_0'],
                                               [100.0, 30.0, 1.0, 0.0]))
    ck.check('G0 采样包络回显',
             np.array_equal(d['e_f_samp_t_1'], sc['samp_t'])
             and np.allclose(d['e_f_samp_v_1'], sc['samp_v'],
                             rtol=0, atol=0))
    ck.check('G0 方波包络参数', np.array_equal(d['e_f_p_2'],
                                               [10.0, 5.0, 0.8, 20.0]))
    ck.check('G0 场-组归属', np.array_equal(d['e_f_group'], [0, 0, 1]))
    ck.check('G0 旋钮默认', int(d['e_m_max'][0]) == 8
             and int(d['e_dense_max'][0]) == 700
             and np.array_equal(d['e_node_ladder'], [16, 32, 64, 128]))


if __name__ == "__main__":
    main()
