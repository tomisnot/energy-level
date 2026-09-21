# -*- coding: utf-8 -*-
"""多体 ED 引擎（M1）：稀疏装配 → 含时演化 → 统一 Result 契约。

角色：manybody 家族的演化执行体，经 `runspec.evolve_run`（dynamic.mode==2 /
method='ed'）唯一分派点被调用；GUI 与 AI 走同一段代码（I5）。

数值分档（维度 dim=2^N 是唯一主约束）：
- dim ≤ `_MB_DENSE_MAX`（默认 2^12）且厄米 → 稠密 `numpy.linalg.eigh` 全时刻
  向量化传播 `psi(t)=V·exp(-i·CONV·λ·t)·V†psi0`（精确）；总激发数守恒时
  （无驱动 + 无双激发 A）自动按激发数扇区块对角化（A2 锚 + 提速），并**当场验证**
  块对角性，不满足即回退全空间（绝不静默给出错物理）。
- 否则 → 稀疏 `scipy.sparse.linalg.expm_multiply` **逐区间步进**（内存安全：一次
  只持一个态矢，支持非厄米 −iΓ/2 与大 dim）。⚠ expm_multiply **不接受 tol 参数**。

单位：H/Ω/δ/Γ=MHz，t=ns，相位 = CONV·(MHz)·(ns)，CONV=2π×1e-3（同各引擎）。
耗散（裁决③）：非厄米对角 −(i/2)Γ_i n_i 为**接缝**，默认 gamma=0（厄米）；
完整 Lindblad 交叉验证留 M4。

Result 契约（与六引擎同键，result_labels/result_summary 零改动可用）：
    t / pops / states / method='manybody' / backend / loss / error_report / diag
  多体附加：pops_atom(n_t,N) 逐原子激发、S(n_t) 中切纠缠熵、corr(N,N) 末态关联。
  dim > `_MB_FULLPOP_MAX`（默认 2^16）时**不物化全联合 pops**：pops 退化为逐原子
  边缘（M=N）并明确写进 error_report（不静默截断）。
"""
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import expm_multiply, eigsh

from ..evolve_util import CONV
from ..param import Param
from .assemble import build_H_sparse
from .observables import (bit_matrix, popcount, corr_matrix, entropy_at_cut,
                          product_label)

# 内部数值保护（tier='internal'，默认不对 AI/GUI 露面；可经 solve.* 覆写）
_MB_DENSE_MAX = Param(4096, 'ED 稠密 eigh 维度上限（超过走稀疏 Krylov 步进）',
                      tier='internal')
_MB_FULLPOP_MAX = Param(65536, 'ED 全联合布居物化维度上限（超过只出逐原子边缘）',
                        tier='internal')
_MB_MAX_DIM = Param(1 << 20, 'ED 硬维度上限 2^N（超过 ED 不可行，等 M3 MPS）',
                    tier='internal')


# =====================================================================
# 初态解析（2^N 产物态）
# =====================================================================
def _from_pattern(n, dim, pat):
    """'|grg>'/'grg' → 产物态 one-hot（原子 0 在左/MSB；r=1,g=0）。"""
    pat = str(pat).replace('|', '').replace('>', '').replace(' ', '')
    if len(pat) != n:
        raise ValueError('多体初态串长度须=%d（原子数），得 %r' % (n, pat))
    idx = 0
    for ch in pat:
        if ch == 'r':
            idx = (idx << 1) | 1
        elif ch == 'g':
            idx = idx << 1
        else:
            raise ValueError('多体初态串只认 g/r，得字符 %r' % ch)
    v = np.zeros(dim, complex)
    v[idx] = 1.0
    return v


def _is_gr(x):
    return x in ('g', 'r', 0, 1)


def _norm_or_raise(v, what='多体初态'):
    nrm = float(np.linalg.norm(v))
    if nrm <= 0.0:
        raise ValueError('%s零范数（各分量幅度全为零）' % what)
    return v / nrm


def psi0_vector(n, dim, spec):
    """多体初态 spec → 2^N 复向量（Σ|amp|²=1 归一）。

    支持：None（全基态 |g…g⟩=0）/ int（产物态索引）/ str（'grg' 模式）/
    逐原子 list（['g','r',…] 或 [0,1,…]）/ 复数组(dim,)/
    叠加 `['sup', [[模式或索引, amp], …]]`。非法**明确报错**，绝不静默回落。
    """
    if spec is None:
        v = np.zeros(dim, complex)
        v[0] = 1.0
        return v
    if isinstance(spec, (int, np.integer)) and not isinstance(spec, bool):
        i = int(spec)
        if not 0 <= i < dim:
            raise ValueError('多体初态索引 %d 越界（dim=%d）' % (i, dim))
        v = np.zeros(dim, complex)
        v[i] = 1.0
        return v
    if isinstance(spec, str):
        return _from_pattern(n, dim, spec)
    if isinstance(spec, np.ndarray):
        v = np.asarray(spec, complex)
        if v.shape != (dim,):
            raise ValueError('多体初态向量形状应 (%d,)，得 %s' % (dim, v.shape))
        return _norm_or_raise(v)
    if isinstance(spec, (list, tuple)):
        if spec and spec[0] == 'sup':
            v = np.zeros(dim, complex)
            for comp in (spec[1] if len(spec) > 1 else []):
                if (isinstance(comp, (list, tuple)) and len(comp) == 2
                        and not isinstance(comp[0], (list, tuple))):
                    pat, amp = comp
                    if isinstance(amp, (list, tuple)) and len(amp) == 2:
                        amp = complex(float(amp[0]), float(amp[1]))
                    v = v + complex(amp) * psi0_vector(n, dim, pat)
                else:
                    v = v + psi0_vector(n, dim, comp)
            return _norm_or_raise(v, '多体叠加初态')
        if len(spec) == n and all(_is_gr(x) for x in spec):
            pat = ''.join('r' if x in ('r', 1) else 'g' for x in spec)
            return _from_pattern(n, dim, pat)
        if len(spec) == dim:
            return _norm_or_raise(np.asarray(spec, complex))
    raise ValueError('无法解析多体初态 psi0=%r' % (spec,))


# =====================================================================
# 基态（Lanczos）
# =====================================================================
def ground_state(H, k=1):
    """厄米 H（稀疏/稠密）→ 最低 k 个 (本征值, 本征矢)（scipy Lanczos）。"""
    Hs = sp.csr_matrix(H)
    dim = Hs.shape[0]
    kk = max(1, min(int(k), dim - 1))
    w, v = eigsh(Hs.astype(complex), k=kk, which='SA')
    order = np.argsort(w)
    return w[order], v[:, order]


# =====================================================================
# 对称扇区判定（总激发数守恒）
# =====================================================================
def _is_excitation_conserving(calib, Om, use):
    """H 是否守恒总激发数：无驱动（Ω=0）且无双激发 A（I3 关或全 A=0）。

    对角项（δ n_i、V n_i n_j）与交换 J（σ⁺σ⁻）都守恒；σˣ 驱动与 σ⁺σ⁺ 破坏。
    """
    if np.any(np.abs(np.asarray(Om, float)) > 0.0):
        return False
    if 'I3' in use and any(abs(p['A']) > 0.0 for p in calib['pairs']):
        return False
    return True


# =====================================================================
# 传播子
# =====================================================================
def _check_cancel(progress_arr):
    if progress_arr is not None and float(progress_arr[1]) != 0.0:
        raise RuntimeError('用户取消')


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
            phase = np.exp(-1j * CONV * np.outer(evals, t))
            psi_all[idx, :] = evecs @ (phase * c[:, None])
        return psi_all, True
    evals, evecs = np.linalg.eigh(Hd)
    c = evecs.conj().T @ psi0
    phase = np.exp(-1j * CONV * np.outer(evals, t))
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
                        reduce_k, progress_arr, stage_progress):
    """分段常数驱动：逐段建 H，区间中点选段，段内 expm_multiply 连乘。

    段边界与输出网格对齐时精确（每段 H 恒定）；否则为区间中点取值。
    """
    segs = sorted(segments, key=lambda s: float(s.get('t0', 0.0)))
    if not segs:
        raise ValueError('segments 为空')
    Hcache = {}

    def _H_for(tm):
        j = 0
        for i, s in enumerate(segs):
            if float(s.get('t0', 0.0)) <= tm:
                j = i
        if j not in Hcache:
            Omj = np.asarray(segs[j].get('Om', np.zeros(n)), float)
            dj = np.asarray(segs[j].get('delta', np.zeros(n)), float)
            Hcache[j] = build_H_sparse(calib, Omj, dj, terms, gamma)
        return Hcache[j]

    psi = psi0.astype(complex).copy()
    reduce_k(0, psi)
    n_t = len(t)
    for k in range(1, n_t):
        _check_cancel(progress_arr)
        Hk = _H_for(0.5 * (t[k - 1] + t[k]))
        psi = expm_multiply((-1j * CONV * (t[k] - t[k - 1])) * Hk, psi)
        reduce_k(k, psi)
        stage_progress(progress_arr, 1, (k + 1) / n_t)


# =====================================================================
# 主入口
# =====================================================================
def evolve_manybody(calib, psi0=None, t_grid=None, Om=None, delta=None,
                    gamma=None, terms=None, segments=None, progress_arr=None,
                    log=None, want_corr=True, want_entropy=True,
                    use_sectors='auto', dense_max=None, fullpop_max=None,
                    max_dim=None, entropy_cut=None, record_psi=False,
                    diag=False):
    """多体 ED 含时演化 → 统一 Result 契约 dict。

    calib   ：`assemble.calibrate` 产物（atoms/pairs[V,J,A]/terms/single_basis）。
    psi0    ：初态 spec（None=全基态；见 psi0_vector；'ground'=先 Lanczos 求基态）。
    Om/delta/gamma：逐原子 Rabi/失谐/衰减率（MHz；缺省 0）。gamma≠0 → 非厄米。
    segments：分段常数驱动 [{t0,Om,delta}, …]（缺省 None=静态 Om/delta）。
    use_sectors：True/False/'auto'——守恒时按激发数扇区块对角化（auto=守恒则用）。
    dense_max/fullpop_max/max_dim：数值分档上限（缺省用本模块 Param）。
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
    dim = 1 << n
    if dim > max_dim:
        raise ValueError('多体 ED 维度 2^%d=%d 超上限 %d——ED 不可行；'
                         '大 N 等 M3 的 MPS/TEBD 后端' % (n, dim, max_dim))
    t = np.asarray(t_grid, float)
    if t.size < 2:
        raise ValueError('多体演化：时间网格须 >= 2 点（得到 %d）' % t.size)
    n_t = int(t.size)

    Om = np.zeros(n) if Om is None else np.asarray(Om, float)
    delta = np.zeros(n) if delta is None else np.asarray(delta, float)
    gamma = np.zeros(n) if gamma is None else np.asarray(gamma, float)
    use = set(calib['terms'] if terms is None else terms)
    hermitian = not bool(np.any(np.abs(gamma) > 0.0))

    if isinstance(psi0, str) and psi0 == 'ground':
        H0 = build_H_sparse(calib, Om, delta, terms, None)
        _, vecs = ground_state(H0, k=1)
        psi0v = vecs[:, 0].astype(complex)
    else:
        psi0v = psi0_vector(n, dim, psi0)

    bits = bit_matrix(n, dim)
    cut = int(entropy_cut if entropy_cut is not None else max(1, n // 2))
    pops_atom = np.zeros((n_t, n))
    S_mid = np.zeros(n_t)
    norm = np.zeros(n_t)
    joint = np.zeros((n_t, dim)) if dim <= fullpop_max else None
    holder = {'p': None, 'psi': None}

    def _reduce(k, psi):
        p = np.abs(psi) ** 2
        norm[k] = float(p.sum())
        pops_atom[k] = bits @ p
        if want_entropy:
            S_mid[k] = entropy_at_cut(psi, n, cut)
        if joint is not None:
            joint[k] = p
        holder['p'] = p
        holder['psi'] = psi

    stage_progress(progress_arr, 1, 0.0)
    used_sectors = False
    if segments:
        _propagate_segments(calib, segments, psi0v, t, n, gamma, terms,
                            _reduce, progress_arr, stage_progress)
    elif hermitian and dim <= dense_max:
        H = build_H_sparse(calib, Om, delta, terms, gamma)
        want_sec = (bool(_is_excitation_conserving(calib, Om, use))
                    if use_sectors == 'auto' else bool(use_sectors))
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
        H = build_H_sparse(calib, Om, delta, terms, gamma)
        _propagate_sparse(H, psi0v, t, _reduce, progress_arr, stage_progress)
    stage_progress(progress_arr, 2, 1.0)

    corr = (corr_matrix(bits, holder['p'])
            if (want_corr and holder['p'] is not None) else None)
    if hermitian:
        loss = np.zeros(n_t)
        norm_dev = float(np.max(np.abs(norm - 1.0)))
    else:
        loss = 1.0 - norm                       # 非厄米范数衰减 = 损失通道
        norm_dev = 0.0
    capped = joint is None
    if not capped:
        pops = joint
        states = [{'label': product_label(n, s), 'index': s}
                  for s in range(dim)]
    else:
        pops = pops_atom                        # 退化：逐原子边缘作 pops（M=N）
        states = [{'label': 'atom%d@%s' % (a['id'], _fmt_pos(a['pos'])),
                   'atom_id': a['id']} for a in atoms]
        log('[manybody] dim=2^%d=%d > 全布居上限 %d：pops 退化为逐原子边缘'
            '（M=N），全联合分布未物化' % (n, dim, fullpop_max))

    backend = ('ed' if hermitian else 'ed-nonhermitian')
    if used_sectors:
        backend += '-sectors'
    return dict(
        t=t, pops=pops, states=states, method='manybody', backend=backend,
        pops_atom=pops_atom, S=S_mid, corr=corr, loss=loss,
        error_report=dict(
            backend=backend, solver=('eigh' if (hermitian and dim <= dense_max
                                                and not segments)
                                     else 'expm_multiply'),
            dim=dim, n_atoms=n, n_t=n_t, hermitian=hermitian,
            used_sectors=used_sectors, full_pops=(not capped),
            entropy_cut=(cut if want_entropy else None),
            max_norm_dev=norm_dev),
        diag=({'dim': dim, 'n_atoms': n, 'backend': backend,
               'used_sectors': used_sectors, 'hermitian': hermitian,
               'terms': sorted(use), 'n_pairs': len(calib.get('pairs') or []),
               'dropped_amp': calib.get('dropped_amp'),
               'entropy_cut': (cut if want_entropy else None),
               'full_pops': (not capped)} if diag else None),
        **({'psi_amp': holder['psi']} if record_psi else {}))


def _fmt_pos(pos):
    return '(%.1f,%.1f,%.1f)' % tuple(float(x) for x in pos)


def evolve_tebd_stub(*args, **kwargs):
    """M3 桩：TeNPy/MPS（TEBD）后端未实现。

    正常路径不会到此（resolve_solver 在 build 阶段就拦 tebd）；本桩仅
    为唯一分派表 ('tebd','many') 条目兼防御（直接调也不静默）。
    """
    raise NotImplementedError(
        '多体 solver=tebd（TeNPy/MPS 后端）属 M3，尚未实现；'
        '现请用 ed（dim=2^N ≤ manybody.ed_max_dim）')
