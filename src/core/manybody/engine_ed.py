# -*- coding: utf-8 -*-
"""多体 ED 引擎（M1）：稀疏装配 → 含时演化 → 统一 Result 契约。

角色：manybody 家族的演化执行体，经 `runspec.evolve_run`（dynamic.mode==2 /
method='ed'）唯一分派点被调用；GUI 与 AI 走同一段代码（I5）。

数值分档（按实际有效基维度，兼顾显式内存预算）：
- dim ≤ `_MB_DENSE_MAX`（默认 2^12）且厄米 → 稠密 `numpy.linalg.eigh` 全时刻
  向量化传播 `psi(t)=V·exp(-i·CONV·λ·t)·V†psi0`（精确）；总激发数守恒时
  （无驱动 + 无双激发 A）自动按激发数扇区块对角化（A2 锚 + 提速），并**当场验证**
  块对角性，不满足即回退全空间（绝不静默给出错物理）。
- 否则 → 稀疏 `scipy.sparse.linalg.expm_multiply` **逐区间步进**（内存安全：一次
  只持一个态矢，支持非厄米 −iΓ/2 与大 dim）。⚠ expm_multiply **不接受 tol 参数**。

单位：H/Ω/δ/Γ=MHz，t=ns，相位 = CONV·(MHz)·(ns)，CONV=2π×1e-3（同各引擎）。
耗散（约定）：非厄米对角 −(i/2)Γ_i n_i 为**接缝**，默认 gamma=0（厄米）；
完整 Lindblad 交叉验证留 M4。

Result 契约（与六引擎同键，result_labels/result_summary 零改动可用）：
    t / pops / states / method='manybody' / backend / loss / error_report / diag
  多体附加：pops_atom(n_t,N) 逐原子激发、S(n_t) 中切纠缠熵、corr(N,N) 末态关联。
  dim > `_MB_FULLPOP_MAX`（默认 2^16）时**不物化全联合 pops**：pops 退化为逐原子
  边缘（M=N）并明确写进 error_report（不静默截断）。
"""
from .results import time_result
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import expm_multiply

from ..evolve_util import CONV
from .couplings import active_terms
from ..param import Param
from .state import state_vector
from .basis import basis_bits
from .resources import dense_limit, budget_bytes, require_workspace
from .runtime import (time_grid, check_cancel, segment_intervals, segment_drive)
from .assemble import build_H_sparse
from .observables import (bit_matrix, popcount, corr_matrix, entropy_at_cut,
                          product_label, spin_observables, entropy_in_basis, loop_observables,
                          link_loop_observables)
from .observables import spin_observables_basis
from .observables import PauliPlan

# 内部数值保护（tier='internal'，默认不对 AI/GUI 露面；可经 solve.* 覆写）
_MB_DENSE_MAX = Param(4096, 'ED 稠密 eigh 维度上限（超过走稀疏 Krylov 步进）',
                      tier='internal')
_MB_FULLPOP_MAX = Param(65536, 'ED 全联合布居物化维度上限（超过只出逐原子边缘）',
                        tier='internal')
_MB_MAX_DIM = Param(1 << 20, 'ED 硬维度上限 2^N（超过 ED 不可行，改用 MPS）',
                    tier='internal')


# =====================================================================
# 初态解析（2^N 产物态）
# =====================================================================
def psi0_vector(n, dim, spec):
    """Compatibility entry for the shared full-space state parser."""
    if int(dim) != 1 << int(n):
        raise ValueError('dim 必须等于 2**N')
    return state_vector(spec, n)


def ground_state(H, k=1):
    """Compatibility entry using the common robust small-dimension solver."""
    from .equilibrium import ground_state as solve_ground_state
    result = solve_ground_state(H, k=k, return_evecs=True)
    # Existing callers request the lowest vector through this helper.
    return result['evals'], result['evecs']


# =====================================================================
# 对称扇区判定（总激发数守恒）
# =====================================================================
def _is_excitation_conserving(calib, Om, use):
    """H 是否守恒总激发数：无驱动（Ω=0）且无双激发 A（I3 关或全 A=0）。

    对角项（δ n_i、V n_i n_j）与交换 J（σ⁺σ⁻）都守恒；σˣ 驱动与 σ⁺σ⁺ 破坏。
    """
    if np.any(np.abs(np.asarray(Om, complex)) > 0.0):
        return False
    if 'I3' in use and any(abs(p['A']) > 0.0 for p in calib['pairs']):
        return False
    return True


# =====================================================================
# 传播子
# =====================================================================
_check_cancel = check_cancel


def _propagate_dense(H, psi0, t, n, dim, use_sectors):
    """稠密 eigh 全时刻向量化传播 → (psi_all(dim,n_t), used_sectors)。"""
    Hd = H.toarray() if sp.issparse(H) else np.asarray(H)
    if use_sectors:
        pc = popcount(n, dim)
        rows, cols = np.nonzero(Hd)
        if rows.size and not np.all(pc[rows] == pc[cols]):
            use_sectors = False          # 非块对角 → 回退全空间（不给错物理）
    if use_sectors:
        psi_all = np.zeros((dim, len(t)), complex)
        for k in range(n + 1):
            idx = np.where(pc == k)[0]
            if idx.size == 0 or not np.any(psi0[idx]):
                continue
            evals, evecs = np.linalg.eigh(Hd[np.ix_(idx, idx)])
            c = evecs.conj().T @ psi0[idx]
            phase = np.exp(-1j * CONV * np.outer(evals, t-t[0]))
            psi_all[idx, :] = evecs @ (phase * c[:, None])
        return psi_all, True
    evals, evecs = np.linalg.eigh(Hd)
    c = evecs.conj().T @ psi0
    phase = np.exp(-1j * CONV * np.outer(evals, t-t[0]))
    return evecs @ (phase * c[:, None]), False


def _propagate_sparse(H, psi0, t, reduce_k, progress_arr, stage_progress):
    """稀疏 expm_multiply 逐区间步进（内存安全，支持非厄米）；流式归约。"""
    A = (-1j * CONV) * H
    psi = psi0.astype(complex).copy()
    reduce_k(0, psi)
    n_t = len(t)
    for k in range(1, n_t):
        _check_cancel(progress_arr)
        psi = expm_multiply(A * (t[k] - t[k - 1]), psi)   # 无 tol 参数（坑）
        reduce_k(k, psi)
        stage_progress(progress_arr, 1, (k + 1) / n_t)


def _propagate_segments(calib, segments, psi0, t, n, gamma, terms,
                        reduce_k, progress_arr, stage_progress, basis_indices=None):
    """Integrate piecewise-constant drives at their exact jump boundaries."""
    matrices = {}
    psi = psi0.astype(complex).copy()
    reduce_k(0, psi)
    for k in range(1, len(t)):
        for a, b in segment_intervals(t[k-1], t[k], segments):
            check_cancel(progress_arr)
            om, de = segment_drive((a+b)/2, segments, n)
            key = (tuple(om), tuple(de))
            if key not in matrices:
                matrices[key] = build_H_sparse(calib, om, de, terms, gamma,
                                                basis_indices=basis_indices)
            psi = expm_multiply((-1j*CONV*(b-a))*matrices[key], psi)
        reduce_k(k, psi)
        stage_progress(progress_arr, 1, (k+1)/len(t))


# =====================================================================
# 主入口
# =====================================================================
@time_result(loss_kind=lambda r: 'physical_loss' if not r['error_report']['hermitian'] else 'norm_drift')
def evolve_manybody(calib, psi0=None, t_grid=None, Om=None, delta=None,
                    gamma=None, terms=None, segments=None, progress_arr=None,
                    log=None, want_corr=True, want_entropy=True,
                    use_sectors='auto', dense_max=None, fullpop_max=None,
                    max_dim=None, entropy_cut=None, record_psi=False,
                    diag=False, plan=None, want_observables=True, loops=None,
                    want_xy_corr=True, link_pairs=None, pauli_strings=None,
                    memory_budget_mb=None):
    """多体 ED 含时演化 → 统一 Result 契约 dict。

    calib   ：`assemble.calibrate` 产物（atoms/pairs[V,J,A]/terms/single_basis）。
    psi0    ：初态 spec（None=全基态；见 psi0_vector；'ground'=先 Lanczos 求基态）。
    Om/delta/gamma：逐原子 Rabi/失谐/衰减率（MHz；缺省 0）。gamma≠0 → 非厄米。
    segments：分段常数驱动 [{t0,Om,delta}, …]（缺省 None=静态 Om/delta）。
    use_sectors：True/False/'auto'——守恒时按激发数扇区块对角化（auto=守恒则用）。
    dense_max/fullpop_max/max_dim：数值分档上限（缺省用本模块 Param）。
    pauli_strings：``[{'label','terms':[(site,axis),…]}]``（经
        ``schema_mb.validate_pauli_strings`` 规范化）→ 每个输出时刻算一遍期望，
        结果在 ``observables.pauli_strings``（复数组 (n_t, n_strings)）与
        ``observables.pauli_string_labels``。
    """
    from ..coupling import stage_progress
    log = log or (lambda *a: None)
    if progress_arr is None:
        progress_arr = np.zeros(2)
    dense_max = int(_MB_DENSE_MAX if dense_max is None else dense_max)
    fullpop_max = int(_MB_FULLPOP_MAX if fullpop_max is None else fullpop_max)
    max_dim = int(_MB_MAX_DIM if max_dim is None else max_dim)

    atoms = calib.get('atoms') or []
    n = len(atoms)
    if n == 0:
        raise ValueError('多体演化：calib 没有原子（先 assemble.calibrate）')
    if link_pairs is not None:
        link_pairs = [(int(a), int(b)) for a, b in link_pairs]
        if any(a < 0 or a >= n or b < 0 or b >= n for a, b in link_pairs):
            raise ValueError('link_pairs 含越界原子下标')
    projected = plan is not None and plan.get('projected')
    basis_indices = np.asarray(plan['states'], np.int64) if projected else None
    dim = len(basis_indices) if projected else 1 << n
    if dim > max_dim:
        raise ValueError('多体 ED 有效维度 %d 超上限 %d；请调整系统或后端' % (dim, max_dim))
    t = time_grid(t_grid)
    check_cancel(progress_arr)
    if projected:
        Om = plan['Om'] if Om is None else Om
        delta = plan['delta'] if delta is None else delta
        gamma = plan['gamma'] if gamma is None else gamma
        terms = plan['terms'] if terms is None else terms
    n_t = int(t.size)
    dense_max = dense_limit(dense_max, memory_budget_mb)
    require_workspace(dim * (n+4) * 16 + n_t*n*8*12, memory_budget_mb,
                      'ED 基空间、观测量和传播向量')
    budget = budget_bytes(memory_budget_mb)
    if budget is not None:
        # Joint output and all-time dense propagation share the workspace.
        fullpop_max = min(fullpop_max, budget // max(1, n_t*8*4))
        if dim*n_t*16*2 > budget:
            dense_max = 0

    Om = np.zeros(n, complex) if Om is None else np.asarray(Om, complex)
    delta = np.zeros(n) if delta is None else np.asarray(delta, float)
    gamma = np.zeros(n) if gamma is None else np.asarray(gamma, float)
    use = set(active_terms(calib) if terms is None else terms)
    hermitian = not bool(np.any(np.abs(gamma) > 0.0))

    if isinstance(psi0, str) and psi0 == 'ground':
        initial_drive = segment_drive(t[0], segments, n)
        om0, de0 = (Om, delta) if initial_drive is None else initial_drive
        H0 = build_H_sparse(calib, om0, de0, terms, None, basis_indices=basis_indices)
        _, vecs = ground_state(H0, k=1)
        psi0v = vecs[:, 0].astype(complex)
    else:
        psi0v = state_vector(psi0, n, basis_indices)

    bits = basis_bits(n, basis_indices) if projected else bit_matrix(n, dim)
    cut = int(entropy_cut if entropy_cut is not None else max(1, n // 2))
    # entropy_cut 的**入口校验**：越界必须在传播前失败。若拖到 `_reduce` 才由
    # entropy_at_cut 返 0，会把"参数错"伪装成"没纠缠"。n<2 时 S 恒 0 是物理事实。
    if want_entropy and n >= 2 and not (1 <= cut <= n - 1):
        raise ValueError('多体：entropy_cut=%d 越界（n_atoms=%d 时须 ∈[1,%d]）'
                         % (cut, n, n - 1))
    S_note = (None if (not want_entropy or n >= 2)
              else 'n_atoms=%d < 2：没有双部划分，S 恒 0（物理事实，非缺数据）' % n)
    pops_atom = np.zeros((n_t, n))
    S_mid = np.full(n_t, np.nan) if want_entropy else None
    norm = np.zeros(n_t)
    sx = np.zeros((n_t, n)); sy = np.zeros((n_t, n))
    sz = np.zeros((n_t, n)); spv = np.zeros((n_t, n), complex)
    smv = np.zeros((n_t, n), complex); phase = np.full((n_t, n), np.nan)
    link_series = (np.full((n_t, len(link_pairs)), np.nan + 0j, complex)
                   if link_pairs is not None else None)
    if pauli_strings and not want_observables:
        raise ValueError('manybody.pauli_strings 需要 want_observables=true'
                         '（它挂在公共观测契约的 observables 下，不会单独开一份结果）')
    pauli_series = (np.zeros((n_t, len(pauli_strings)), complex)
                    if pauli_strings else None)
    pauli_plan = (PauliPlan(n, pauli_strings, basis_states=basis_indices)
                  if pauli_strings else None)
    xy_last = None
    joint = np.zeros((n_t, dim)) if dim <= fullpop_max else None
    holder = {'p': None, 'psi': None}

    def _reduce(k, psi):
        nonlocal xy_last
        p = np.abs(psi) ** 2
        norm[k] = float(p.sum())
        pops_atom[k] = bits @ p
        if want_entropy:
            value = (entropy_in_basis(psi, basis_indices, n, cut) if projected
                     else entropy_at_cut(psi, n, cut))
            S_mid[k] = np.nan if value is None else value
        if joint is not None:
            joint[k] = p
        holder['p'] = p
        holder['psi'] = psi
        if want_observables:
            ob = spin_observables_basis(psi, basis_indices, n) if projected else spin_observables(psi, n)
            sx[k], sy[k], sz[k] = ob['sx'], ob['sy'], ob['sz']
            spv[k], smv[k], phase[k] = ob['sp'], ob['sm'], ob['phase']
            if link_pairs is not None:
                link_series[k] = [ob['xy_corr'][a, b] for a, b in link_pairs]
            if k == n_t - 1 and want_xy_corr:
                xy_last = ob['xy_corr']
        if pauli_series is not None:
            pauli_series[k] = pauli_plan.expect(psi)

    stage_progress(progress_arr, 1, 0.0)
    used_sectors = False
    if segments:
        _propagate_segments(calib, segments, psi0v, t, n, gamma, terms,
                            _reduce, progress_arr, stage_progress, basis_indices=basis_indices)
    elif hermitian and dim <= dense_max:
        H = build_H_sparse(calib, Om, delta, terms, gamma, basis_indices=basis_indices)
        want_sec = (not projected and bool(_is_excitation_conserving(calib, Om, use))
                    if use_sectors == 'auto' else (not projected and bool(use_sectors)))
        if use_sectors is True and not want_sec:
            log('[manybody] use_sectors=True 但 H 不守恒总激发数（有驱动/双激发）'
                '→ 用全空间')
            want_sec = False
        psi_all, used_sectors = _propagate_dense(H, psi0v, t, n, dim, want_sec)
        for k in range(n_t):
            _check_cancel(progress_arr)
            _reduce(k, psi_all[:, k])
            stage_progress(progress_arr, 1, (k + 1) / n_t)


    else:
        H = build_H_sparse(calib, Om, delta, terms, gamma, basis_indices=basis_indices)
        _propagate_sparse(H, psi0v, t, _reduce, progress_arr, stage_progress)
    stage_progress(progress_arr, 2, 1.0)

    corr = (corr_matrix(bits, holder['p'])
            if (want_corr and holder['p'] is not None) else None)
    if hermitian:
        loss = 1.-norm
        norm_dev = float(np.max(np.abs(norm - 1.0)))
    else:
        loss = 1.0 - norm                       # 非厄米范数衰减 = 损失通道
        norm_dev = 0.0
    if want_entropy and np.isnan(S_mid).any():
        S_note = '部分输出时刻未计算纠缠熵（Schmidt 矩阵预算或零范数）；缺失值为 NaN'
    capped = joint is None or projected
    if not capped:
        pops = joint
        states = [{'label': product_label(n, s), 'index': s}
                  for s in range(dim)]
    else:
        pops = pops_atom                        # 退化：逐原子边缘作 pops（M=N）
        states = [{'label': 'atom%d@%s' % (a['id'], _fmt_pos(a['pos'])),
                   'atom_id': a['id']} for a in atoms]
        log('[manybody] pops 使用逐原子占据；联合基概率见 basis_probabilities（若已物化）')

    backend = ('ed-plan-%s' % plan['constraint']['kind'] if projected else
               'ed' if hermitian else 'ed-nonhermitian')
    if used_sectors:
        backend += '-sectors'
    loop_phase, loop_winding = loop_observables(xy_last, loops) if (
        want_observables and xy_last is not None and loops is not None) else ([], [])
    loop_phase_series, loop_winding_series = link_loop_observables(
        link_series, link_pairs, loops) if (
            want_observables and link_series is not None and loops is not None) else (
                np.empty((0, 0), float), np.empty((0, 0), float))
    obs = (dict(sx=sx, sy=sy, sz=sz, sp=spv, sm=smv, phase=phase,
                xy_corr=xy_last, loop_phase=loop_phase,
                loop_winding=loop_winding, loop_phase_series=loop_phase_series,
                loop_winding_series=loop_winding_series, link_corr=link_series,
                link_pairs=link_pairs,
                pauli_strings=pauli_series,
                pauli_string_labels=([d['label'] for d in pauli_strings]
                                     if pauli_strings else [])) if want_observables else None)
    return dict(
        t=t, pops=pops, states=states, method='manybody', backend=backend,
        pops_atom=pops_atom, S=S_mid, corr=corr, loss=loss, norms=norm,
        basis_probabilities=joint,
        basis_states=(basis_indices if projected else None),
        basis=dict(kind='projected' if projected else 'full', bit_order='site0-msb',
                   n_atoms=n, dim=dim, full_dim=1 << n),
        error_report=dict(
            backend=backend, solver=('eigh' if (hermitian and dim <= dense_max
                                                and not segments)
                                     else 'expm_multiply'),
            dim=dim, n_atoms=n, n_t=n_t, hermitian=hermitian,
            used_sectors=used_sectors, full_pops=(not capped),
            projected=bool(projected),
            entropy_cut=(cut if want_entropy else None),
            entropy_note=S_note,
            max_norm_dev=norm_dev),
        diag=({'dim': dim, 'n_atoms': n, 'backend': backend,
               'used_sectors': used_sectors, 'hermitian': hermitian,
               'terms': sorted(use), 'n_pairs': len(calib.get('pairs') or []),
               'dropped_amp': calib.get('dropped_amp'),
               'entropy_cut': (cut if want_entropy else None),
               'full_pops': (not capped)} if diag else None),
        observables=obs,
        **({'psi_amp': holder['psi']} if record_psi else {}))



def _fmt_pos(pos):
    return '(%.1f,%.1f,%.1f)' % tuple(float(x) for x in pos)


def evolve_tebd_stub(*args, **kwargs):
    """兼容旧调用点：转发到 TeNPy TEBD 实现。"""
    from .engine_mps import evolve_mps
    kwargs['algorithm'] = 'tebd'
    return evolve_mps(*args, **kwargs)
