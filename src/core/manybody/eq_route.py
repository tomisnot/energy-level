# -*- coding: utf-8 -*-
"""与具体实验协议无关的平衡态求解编排。

solve_equilibrium 规范化输入、校验能力并路由；各任务由独立的 gs/thermal/
DOS/ramp 辅助函数执行。输入支持物理 calib/Plan 或明确的 H/H_builder；专用
模型后端不能忽略显式矩阵。结果统一附带数据语义与基身份。

实验波形和论文协议留在模型/应用层。后端声明见 backends.py。
"""
import time
from .basis import basis_bits
from .backends import normalize_equilibrium_backend, require_equilibrium_features
from .results import describe_result
from .resources import dense_limit, require_workspace
from functools import wraps

import numpy as np
import scipy.sparse as sp

from . import assemble as _asm
from . import equilibrium as _eq
from . import kpm as _kpm
from . import qmc as _qmc
from .hamiltonian_plan import compile_sparse
from .matrix_free import (hamiltonian_linear_operator, number_sector_basis,
                          check_number_conserving)

__all__ = ['route_equilibrium', 'solve_equilibrium']

_DENSE_MAX = 4096          # 2^n ≤ 此 → 稠密 eigh 可接受（约 N≤12）
#: 基态关联矩阵 C_ij 的计算上限（维度）：代价 ~ N²·dim，2^18 约 0.3 s。
#: 超出就不给 corr（不是填零），需要它的调用方应改走 MPS 侧测量。
CORR_MAX_DIM = 1 << 18


def route_equilibrium(n_sites, want='gs', *, is_ladder=False, batch=1,
                      gpu=None, dense_max=_DENSE_MAX, dmrg_ok=None):
    """纯决策函数：→ backend 名。可单独测（不跑真解）。

    `dmrg_ok`：能不能上 DMRG。缺省跟随 `is_ladder`（向后兼容）；新架构下任意点云
    都能 DMRG（自写 autoMPO），由调用方按“顺序质量”给 True/False。
    """
    if want in ('thermal', 'dos'):
        return 'kpm'
    if want == 'spectrum':
        # 源态谱走 Krylov（ED/Lanczos）；DMRG 没有对应的源态 Lanczos 入口。
        return 'ed' if (1 << int(n_sites)) <= dense_max else 'lanczos'
    if gpu is None:
        gpu = _eq.detect_gpu() is not None
    can_dmrg = is_ladder if dmrg_ok is None else bool(dmrg_ok)
    dim = 1 << int(n_sites)
    if batch and batch > 1 and gpu:
        return 'gpu-batch'
    if dim <= dense_max:
        return 'ed'
    if can_dmrg:
        return 'dmrg'
    return 'lanczos'


def _engine_H(system, dense=False):
    Hs = _asm.build_H_sparse(system['calib'], system.get('Om'), system.get('delta'),
                              terms=system.get('terms'), gamma=system.get('gamma'))
    return np.asarray(Hs.todense()) if dense else Hs


def _system_H(system, om=None, de=None, dense=False):
    """system → 哈密顿量：'H' 直传优先，其次 'H_builder'(Om, delta)，再次 calib。

    `dense=True` → 三种口径（Plan/H/H_builder/calib）都返回稠密数组
    （`want='dos'` 的 `eigvalsh` 需要）。
    """
    if system.get('plan') is not None and system['plan'].get('projected'):
        H = compile_sparse(system['plan'])
    elif 'H' in system and system['H'] is not None:
        H = system['H']
    elif 'H_builder' in system and system['H_builder'] is not None:
        H = system['H_builder'](om, de)
    else:
        return _engine_H(system, dense=dense)
    return np.asarray(H.todense()) if (dense and sp.issparse(H)) else H


def _check_sector_conserving(H, labels, seed=0, max_check=20000):
    """激发数守恒的**抽样校验**：非零元 (i,j) 必须满足 labels[i]==labels[j]。

    抽 max(1000, 1%)（上限 max_check）个非零元；不守恒即 raise——扇区切片对
    非守恒 H 是静默算错（跨界耦合被丢弃），不许伪装成"没约束"。
    """
    labels = np.asarray(labels)
    Hs = H if sp.issparse(H) else sp.csr_matrix(H)
    coo = Hs.tocoo()
    nz = coo.data != 0
    rows, cols = coo.row[nz], coo.col[nz]
    if len(rows) == 0:
        return
    n_check = int(min(max_check, max(1000, len(rows) // 100), len(rows)))
    rng = np.random.default_rng(seed)
    sel = rng.choice(len(rows), size=n_check, replace=False)
    bad = labels[rows[sel]] != labels[cols[sel]]
    n_bad = int(np.count_nonzero(bad))
    if n_bad:
        raise ValueError('sector 模式要求 H 激发数守恒：抽样 %d 个非零元发现 %d 个'
                         '跨界耦合（如 Ω 驱动项）——请先剥离非守恒项或关闭扇区'
                         % (n_check, n_bad))


def _gs_observables(vec, n_sites):
    """基态向量 → 逐原子⟨n_i⟩ + 连通占据关联 C_ij（与引擎同 MSB 基序：bit=1↔|r⟩）。

    关联矩阵只在 ``2^N ≤ CORR_MAX_DIM`` 时算（ED 语境；cost ~ N²·dim），
    超出就不给——**不静默填零**，结果里干脆没有这个字段。
    """
    n = int(n_sites)
    dim = 1 << n
    p = np.abs(np.asarray(vec).ravel()) ** 2
    bits = _eq.bit_matrix(n, dim)
    out = dict(n_atom=bits @ p)
    if n and dim <= CORR_MAX_DIM:
        from .observables import corr_matrix
        out['corr'] = corr_matrix(bits, p)
    return out


def _pauli_observables(system, vec, n_sites, states=None):
    """``manybody.pauli_strings`` → 结果字段（基态）。

    ``states`` 为 None 表示 ``vec`` 是**全空间** 2^N 态矢；否则是这些位编码
    张成的子空间里的向量（固定荷扇区 / 投影基）。引擎只算
    ``⟨Π σ^{a_i}_{s_i}⟩``，物理含义（是否乘 1/8、是否连通减法）由调用方组合。
    """
    specs = system.get('pauli_strings')
    if not specs:
        return {}
    from .observables import measure_pauli_strings
    n = int(n_sites)
    if states is None:
        vals = measure_pauli_strings(vec, specs, n=n)
    else:
        vals = measure_pauli_strings(vec, specs, basis_states=states, n=n)
    return {'pauli_strings': [complex(v) for v in vals],
            'pauli_string_labels': [d['label'] for d in specs]}


def _equilibrium_result(fn):
    @wraps(fn)
    def solve(system, *args, **kwargs):
        result = fn(system, *args, **kwargs)
        result.setdefault('requested_backend', kwargs.get('backend', 'auto'))
        result.setdefault('resolved_backend', result.get('backend'))
        if result.get('vec') is not None:
            indices = result.get('sector_idx')
            if indices is None and (system.get('plan') or {}).get('projected'):
                indices = system['plan']['states']
            result['basis'] = dict(bit_order='site0-msb', indices=indices,
                                   dim=len(result['vec']), n_sites=result.get('n_sites'))
        return describe_result(result, system.get('calib'),
                               approximation=('finite-beta QMC' if result.get('backend') == 'qmc'
                                              and result.get('want') == 'gs' else None))
    return solve


def _resolve_source(source, psi0, n_sites, idx=None):
    """源态规格 → 与 ``|0⟩`` 同基的 ``φ``。返回 ``(phi, info)``。

    四种**都在"当前基"里**的声明（当前基=扇区基或全空间，由调用方口径决定）：

    ``{"kind":"vector","values":[…]}}``
        直接给 φ（长度 = 当前基维度）。
    ``{"kind":"diagonal","values":[…]}``
        对角算符 D，φ = D⊙ψ0（占据/计数这类探针的标准写法）。
    ``{"kind":"sparse","rows":[…],"cols":[…],"values":[…]}``
        COO 稀疏矩阵 O，φ = O ψ0（单粒子/键算符）。
    ``{"kind":"full_sparse", …}``
        全 ``2^N`` 空间的 COO；扇区模式下会投影回扇区并如实报告被丢弃的权重
        （``projected_leak``）——**不静默截断**。

    ⚠ 这里不解释"哪个算符是质量算符"：源由调用方给，引擎只做线性代数。
    """
    if not isinstance(source, dict):
        raise ValueError('manybody.source 须是映射（kind=vector/diagonal/sparse/full_sparse）')
    kind = str(source.get('kind') or '')
    n_sites = int(n_sites)
    dim = len(idx) if idx is not None else (1 << n_sites)
    psi = np.asarray(psi0, dtype=complex).reshape(-1)
    if psi.size != dim:
        raise ValueError('初态维度 %d 与当前基维度 %d 不一致' % (psi.size, dim))

    def _coo(block):
        rows = np.asarray(block.get('rows', ()), dtype=int).ravel()
        cols = np.asarray(block.get('cols', ()), dtype=int).ravel()
        vals = np.asarray(block.get('values', ()), dtype=complex).ravel()
        if not (len(rows) == len(cols) == len(vals)):
            raise ValueError('source 的 rows/cols/values 长度不一致')
        if len(rows) == 0:
            raise ValueError('source 的稀疏结构为空')
        return sp.coo_matrix((vals, (rows, cols)),
                             shape=(int(block.get('dim') or (rows.max() + 1)),
                                    int(block.get('dim') or (cols.max() + 1)))).tocsr()

    if kind == 'vector':
        v = np.asarray(source.get('values', ()), dtype=complex).reshape(-1)
        if v.size != dim:
            raise ValueError('source vector 长度 %d ≠ 当前基维度 %d' % (v.size, dim))
        return v, {'kind': kind, 'dim': dim}
    if kind == 'diagonal':
        d = np.asarray(source.get('values', ()), dtype=complex).reshape(-1)
        if d.size != dim:
            raise ValueError('source diagonal 长度 %d ≠ 当前基维度 %d' % (d.size, dim))
        return d * psi, {'kind': kind, 'dim': dim}
    if kind in ('site_weights', 'weights'):
        # 逐站点权重的**对角**算符 O = Σ wᵢ nᵢ：在当前基里展开成对角元。
        # 这是"给一个物理量"的最短写法（N 个数，而不是基维度个数）。
        w = np.asarray(source.get('weights', ()), dtype=float).reshape(-1)
        if w.size != n_sites:
            raise ValueError('source site_weights 长度 %d ≠ 原子数 %d' % (w.size, n_sites))
        from .basis import basis_bits
        codes = (np.arange(1 << n_sites, dtype=np.int64) if idx is None
                 else np.asarray(idx, dtype=np.int64))
        diag = basis_bits(n_sites, codes).T @ w          # (dim,) 实对角
        info = {'kind': 'site_weights', 'dim': dim,
                'sum_weights': float(w.sum()),
                'weight_sq': float((w * w).sum())}
        d = diag.astype(complex)
        if source.get('subtract_mean'):
            # 连通谱：φ = (O − ⟨O⟩)|0⟩，扣掉弹性峰（ω=0 的 |⟨O⟩|² 分量）。
            mean = complex(np.vdot(psi, d * psi))
            d = d - mean
            info['subtracted_mean'] = mean.real
        return d * psi, info
    if kind == 'sparse':
        O = _coo(source)
        if O.shape != (dim, dim):
            raise ValueError('source sparse 形状 %r ≠ 当前基 (%d,%d)' % (O.shape, dim, dim))
        return O @ psi, {'kind': kind, 'dim': dim}
    if kind == 'full_sparse':
        full = 1 << n_sites
        if full > (1 << 16):
            raise ValueError('full_sparse 需要 2^N ≤ 65536；大系统请给当前基（扇区）的 source')
        O = _coo(source)
        if O.shape != (full, full):
            raise ValueError('source full_sparse 形状 %r ≠ (%d,%d)' % (O.shape, full, full))
        if idx is None:
            return O @ psi, {'kind': kind, 'dim': full}
        expand = np.zeros(full, dtype=complex)
        expand[np.asarray(idx, int)] = psi
        phi_full = O @ expand
        phi = phi_full[np.asarray(idx, int)]
        norm_full = float(np.linalg.norm(phi_full))
        leak = 0.0 if norm_full == 0 else 1.0 - float(np.linalg.norm(phi)) ** 2 / norm_full ** 2
        return phi, {'kind': kind, 'dim': dim, 'projected_leak': max(0.0, leak)}
    raise ValueError('未知 source.kind=%r（可选 vector/diagonal/sparse/full_sparse）' % kind)


def _solve_spectrum(system, base, *, sector, dense_max, memory_budget_mb,
                    progress_arr):
    """源态 Lanczos 谱（``want='spectrum'``）。"""
    from . import spectrum as _spec
    from .runtime import check_cancel, report_progress
    want = base.get('want')
    n_sites = base.get('n_sites')
    if n_sites is None:
        raise ValueError('want=spectrum 需要 n_sites（或 calib）')
    n_sites = int(n_sites)
    opts = dict(system.get('spectrum') or {})
    source = system.get('source')
    if source is None:
        raise ValueError('want=spectrum 需要 system["source"]（vector/diagonal/sparse/full_sparse）')
    # 当前基：扇区（固定激发数）或全空间
    idx = None
    if sector is not None:
        if 'calib' not in system:
            raise ValueError('扇区谱需要 system["calib"]')
        check_number_conserving(system['calib'], system.get('Om'), system.get('terms'))
        idx = number_sector_basis(n_sites, sector)
        H = hamiltonian_linear_operator(
            system['calib'], system.get('Om'), system.get('delta'),
            system.get('gamma'), system.get('terms'), basis_indices=idx)
    else:
        H = _system_H(system)
    dim = len(idx) if idx is not None else (1 << n_sites)
    # |0⟩：显式态矢优先，否则就地求基态
    psi0 = system.get('psi0')
    if isinstance(psi0, np.ndarray) and psi0.ndim == 1:
        gs = {'vec': np.asarray(psi0, dtype=complex), 'E0': float('nan')}
    else:
        gs = _eq.ground_state(H, dense_max=dense_max, k=1, v0=None, seed=0)
    check_cancel(progress_arr)
    vec = np.asarray(gs['vec'], dtype=complex).reshape(-1)
    if vec.size != dim:
        raise ValueError('基态维度 %d ≠ 当前基 %d' % (vec.size, dim))
    phi, source_info = _resolve_source(source, vec, n_sites, idx)
    report_progress(progress_arr, 0.35)
    out = _spec.source_spectrum(
        H, phi, shift=float(gs['E0']) if np.isfinite(gs.get('E0', np.nan)) else 0.0,
        eta=float(opts.get('eta') or 0.05),
        n_krylov=int(opts.get('n_krylov') or 0) or None,
        omega_min=opts.get('omega_min'), omega_max=opts.get('omega_max'),
        n_omega=int(opts.get('n_omega') or 800),
        reortho=bool(opts.get('reortho', True)),
        memory_budget_mb=memory_budget_mb)
    report_progress(progress_arr, 1.0)
    errors = dict(out.pop('error_report') or {})
    errors.update({'source_kind': source_info.get('kind'),
                   'sector': None if sector is None else int(sector),
                   'dim_basis': dim})
    # 源侧的其余诊断（投影泄漏、扣掉的期望值、权重统计…）如实转出，不丢。
    for key, value in source_info.items():
        if key not in ('kind', 'dim'):
            errors.setdefault(key, value)
    return dict(base, E0=float(gs['E0']) if np.isfinite(gs.get('E0', np.nan)) else None,
                gap=None,          # 谱任务没有"能隙"这个标量；别把 0.0 塞给卡片
                evals=out['evals'], spectrum_weights=out['spectrum_weights'],
                energies=out['omega'], spectral=out['spectral'],
                source_norm=out['source_norm'], eta=out['eta'],
                krylov=out['krylov'], error_report=errors,
                **({'sector_idx': idx, 'dim_sub': dim} if idx is not None else {}))


@_equilibrium_result
def solve_equilibrium(system, want='gs', *, betas=None, backend='auto',
                      sector=None, k=2, v0=None, ramp=None, log=None,
                      sector_check=True, chi_max=200, num=400, nvec=200,
                      seed=0, dense_max=_DENSE_MAX, exact=False,
                      truncerr=1e-10, max_sweeps=60, qmc_sweeps=2000,
                      qmc_burn_in=500, qmc_measure_every=5, qmc_M=None,
                      memory_budget_mb=None, dtype='float64', return_state=False,
                      mps_dtype='auto', mps_diagnostics=None, progress_arr=None):
    """统一平衡态求解。

    system：{'calib':..,'Om':..,'delta':.., 'ladder':dict(...), 'is_ladder':bool,
             'n_sites':int, 'batch':int}
      - ladder: 传入则 DMRG 可用（含 V_r/J_r/A_r/V_l/J_l/A_l/delta/Omega）。
      - **模型后端解耦**：'H'（矩阵直传）或 'H_builder'(Om, delta)
        优先于 calib——三能级 V 型（rydberg_fci）等任何模型后端可直接进门面。
        直传口径下**不做逐原子观测**（bit 序约定属于模型层），Result 只含
        能量/本征矢。'sector_labels'：1 维逐基矢激励数（sector 模式必需）。
    want：'gs' | 'thermal' | 'dos' | 'ramp'。
    sector：None=全空间；int=固定激励数（仅 gs）。calib 模型直接生成组合基；
      直传 H 需 sector_labels 并校验守恒。system['charge_sector'] 是产品配置入口，
      同时支持固定荷 DMRG。尚未实现该约束的后端明确拒绝。
    return_state：显式返回 CPU/GPU 态矢 vec 或有限 DMRG 的原生 MPS state。
    k：gs 请求的本征值个数（evals 同序返回）。
    v0：Lanczos 初值（warm-start；dense 分支忽略）。
    ramp（want='ramp' 必填）：dict(Om_t, de_t, L[, n_tot, v0_warm, dense_max,
      taus])——Om_t/de_t 逐 τ 驱动序列；L=密度归一格点数（截断模型 dim≠2^L
      时无法从维度反推，必须显式）；n_tot=1 维逐基矢总激发数对角算符（模型
      侧知识，用于 ns 观测）；v0_warm 缺省 True。
    返回统一 dict：{'want','backend', + n_sites/E0/gap/evals/S（gs，按口径）或
      E/C/F(betas)（thermal）或 rho(dos) 或 taus/gaps/ns（ramp）}。
    """
    system = dict(system)
    declared_plan = system.get('plan') or {}
    for key in ('Om', 'delta', 'gamma', 'terms'):
        if system.get(key) is None and key in declared_plan:
            system[key] = declared_plan[key]
        elif declared_plan.get('projected') and key in declared_plan and not np.array_equal(
                np.asarray(system[key]), np.asarray(declared_plan[key])):
            raise ValueError('system.%s 与 projected Plan 冲突' % key)
    if declared_plan.get('projected') and (system.get('H') is not None or system.get('H_builder') is not None):
        raise ValueError('不能同时指定 projected Plan 与显式 H/H_builder')
    backend = normalize_equilibrium_backend(backend, want) if want != 'ramp' else backend
    dense_max = dense_limit(dense_max, memory_budget_mb)
    if want in ('thermal','dos') and system.get('gamma') is not None and np.any(np.asarray(system['gamma']) != 0):
        raise ValueError('平衡态要求 gamma=0 的厄米 Hamiltonian')
    if want == 'thermal':
        betas = np.atleast_1d(np.asarray([1.] if betas is None else betas, float))
        if betas.ndim != 1 or not len(betas) or not np.isfinite(betas).all() or np.any(betas <= 0):
            raise ValueError('betas 须为非空、有限的正数数组')
    explicit_h = system.get('H') is not None or system.get('H_builder') is not None
    if explicit_h and backend in ('dmrg','qmc','free-fermion','gpu-lanczos'):
        raise ValueError('%s 需要物理 calib；不能忽略显式 H/H_builder' % backend)
    n_sites = system.get('n_sites')
    if n_sites is None and 'calib' in system:
        n_sites = len(system['calib']['atoms'])
    batch = system.get('batch', 1)
    is_ladder = system.get('is_ladder', 'ladder' in system)
    dmrg_ok = system.get('dmrg_ok', is_ladder)
    plan = system.get('plan') or {}
    native_initial=isinstance(system.get('psi0'),dict) and system['psi0'].get('kind')=='mps'
    needs_mps=(native_initial or system.get('mps_order') is not None or
               mps_diagnostics is not None or mps_dtype!='auto')
    if backend=='auto' and needs_mps:
        if want!='gs' or explicit_h or plan.get('projected'):
            raise ValueError('Requested finite-MPS features require an unprojected calib ground-state problem')
        backend='dmrg'
    if return_state and want != 'gs':
        raise ValueError('return_state 当前仅用于基态或实时 MPS 演化')
    charge = system.get('charge_sector')
    if charge is not None:
        if isinstance(charge, (bool, np.bool_)) or int(charge) != charge:
            raise ValueError('charge_sector 必须是整数')
        if sector is not None and int(sector) != int(charge):
            raise ValueError('sector 与 charge_sector 冲突')
        if want not in ('gs', 'spectrum'):
            raise ValueError('charge_sector 当前仅支持基态/源态谱；不能静默忽略热态/DOS 的扇区')
        if backend not in ('dmrg', 'free-fermion'):
            sector = int(charge)
    if (backend == 'auto' and plan.get('projected')
            and want == 'gs'):
        # The reduced basis is the routing input; never send a projected plan
        # to a backend which would silently rebuild the unconstrained MPO.
        backend = ('ed' if int(system['plan']['dim']) <= dense_max else 'lanczos')
    if want in ('gs', 'thermal', 'dos', 'spectrum') and backend == 'auto' and sector is None:
        if n_sites is None:
            raise ValueError('system 需要 n_sites（或 calib）以自动路由；'
                             "直传 H 时可显式 backend='ed'/'lanczos'")
        backend = route_equilibrium(n_sites, want, is_ladder=is_ladder,
                                    batch=batch, dense_max=dense_max,
                                    dmrg_ok=dmrg_ok)
    # `n_sites` 进 Result：契约声明了它，所有有格点语义的分支都要给。
    # 直传 H 且无 calib 时不放——那种口径下格点数是模型侧知识，塞 None 是假装知道。
    backend = normalize_equilibrium_backend(backend, want) if want != 'ramp' else backend
    if exact and backend == 'gpu-kpm':
        raise ValueError('gpu-kpm 不提供 exact 谱迹；请选择 CPU kpm 或 ed')
    if backend=='dmrg' and sector is not None:
        if isinstance(sector,(bool,np.bool_)) or int(sector)!=sector:
            raise ValueError('sector must be an integer excitation number')
        system['charge_sector']=int(sector)
        sector=None
    from .mps_dtype import validate_policy
    from .runtime import check_cancel
    validate_policy(mps_dtype);check_cancel(progress_arr)
    features=[]
    if mps_dtype!='auto':features.append('mps_dtype')
    if mps_diagnostics is not None:features.append('mps_analysis')
    if system.get('mps_order') is not None:features.append('mps_order')
    if system.get('psi0') is not None and backend=='dmrg':features.append('mps_initial_state')
    if isinstance(system.get('psi0'),dict) and system['psi0'].get('kind')=='mps':features.append('mps_initial_state')
    require_equilibrium_features(backend,features)
    if features and want!='gs':raise ValueError('Finite-MPS options require want=gs')
    base = dict(want=want, backend=backend)
    if n_sites is not None:
        base['n_sites'] = int(n_sites)

    qmc_options = dict(sweeps=qmc_sweeps, burn_in=qmc_burn_in,
                       measure_every=qmc_measure_every, M=qmc_M, seed=seed)
    if want == 'ramp':
        return _solve_ramp(system, base, ramp, v0, seed, log)
    if want == 'gs':
        return _solve_ground(system, base, sector=sector, k=k, v0=v0,
            sector_check=sector_check, chi_max=chi_max, seed=seed, dense_max=dense_max,
            truncerr=truncerr, max_sweeps=max_sweeps, betas=betas,
            qmc_options=qmc_options, memory_budget_mb=memory_budget_mb,
            dtype=dtype, return_state=return_state,mps_dtype=mps_dtype,
            mps_diagnostics=mps_diagnostics,progress_arr=progress_arr)
    if want == 'spectrum':
        return _solve_spectrum(system, base, sector=sector, dense_max=dense_max,
                               memory_budget_mb=memory_budget_mb,
                               progress_arr=progress_arr)
    if want == 'thermal':
        return _solve_thermal(system, base, betas=betas, num=num, nvec=nvec,
            seed=seed, exact=exact, qmc_options=qmc_options,
            memory_budget_mb=memory_budget_mb, dtype=dtype)
    if want == 'dos':
        return _solve_dos(system, base, num=num, nvec=nvec, seed=seed, exact=exact,
                          memory_budget_mb=memory_budget_mb, dtype=dtype)
    raise ValueError('want 需为 gs/thermal/dos/ramp，实=%r' % want)


def _solve_ramp(system, base, ramp, v0, seed, log):
    if ramp is None:
        raise ValueError("want='ramp' 需要 ramp=dict(Om_t, de_t, L, ...)")
    hb = system.get('H_builder')
    if hb is None:
        raise ValueError("want='ramp' 需要 system['H_builder'](Om, delta)")
    taus = np.asarray(ramp.get('taus',
                               np.linspace(0.0, 1.0, len(ramp['Om_t']))),
                      float)
    Om_t = np.asarray(ramp['Om_t'], float)
    de_t = np.asarray(ramp['de_t'], float)
    L = ramp.get('L')
    if L is None:
        raise ValueError('ramp 需要 L（密度归一格点数）')
    n_tot = (np.asarray(ramp['n_tot'], float)
             if ramp.get('n_tot') is not None else None)
    v0_warm = bool(ramp.get('v0_warm', True))
    dmax = int(ramp.get('dense_max', 2048))
    n_tau = len(taus)
    gaps, ns = [], []
    v0_i = v0
    t_start = time.time()
    for it in range(n_tau):
        Hi = hb(float(Om_t[it]), float(de_t[it]))
        try:
            r = _eq.ground_state(Hi, dense_max=dmax, k=2, v0=v0_i, seed=seed)
        except sp.linalg.ArpackNoConvergence:
            # 大维个别点 ARPACK 不收敛：换初值重试一次（仍失败则向上抛，
            # 不许静默吞错——静默吞错 = ramp 曲线里埋一段错的）。
            r = _eq.ground_state(Hi, dense_max=dmax, k=2, v0=None,
                                 seed=seed + 1)
        v0_i = r['vec'] if v0_warm else None
        gaps.append(r['gap'])
        if n_tot is not None:
            ns.append(float(n_tot @ (np.abs(r['vec']) ** 2)) / L)
        else:
            ns.append(float('nan'))
        if log is not None and (it % max(1, n_tau // 10) == 0
                                or it == n_tau - 1):
            log('ramp %d/%d Ω=%.3f δ=%.3f gap=%.4f n=%.4f (%.0fs)'
                % (it + 1, n_tau, Om_t[it], de_t[it], gaps[-1], ns[-1],
                   time.time() - t_start))
    return dict(base, taus=taus, gaps=np.asarray(gaps, float),
                ns=np.asarray(ns, float), Om_t=Om_t, de_t=de_t)



def _solve_ground(system, base, *, sector, k, v0, sector_check, chi_max,
                  seed, dense_max, truncerr, max_sweeps, betas, qmc_options,
                  memory_budget_mb, dtype, return_state,mps_dtype,mps_diagnostics,progress_arr):
    backend = base['backend']
    plan = system.get('plan') or {}
    charge = system.get('charge_sector')
    n_sites = base.get('n_sites')
    explicit_h = system.get('H') is not None or system.get('H_builder') is not None
    if return_state and backend in ('qmc', 'gpu-batch', 'free-fermion'):
        raise ValueError('所选基态后端尚未实现 return_state')
    if system.get('pauli_strings'):
        from .backends import equilibrium_pauli_backends
        alts = equilibrium_pauli_backends()
        if backend not in tuple(alts) + ('auto',):
            raise ValueError(
                '后端 %s 只有统计量/谱，不提供多体态，无法计算 manybody.pauli_strings；'
                '请改 eq_backend=%s。该键不会被静默忽略'
                % (backend, ' / '.join(alts)))
    if system.get('gamma') is not None and np.any(np.asarray(system['gamma']) != 0):
        raise ValueError('基态求解要求厄米 Hamiltonian（gamma=0）')
    if sector is not None and backend in ('qmc', 'gpu-batch', 'kpm', 'gpu-kpm'):
        raise ValueError('该后端尚未实现固定 sector；请选择 ed/lanczos/gpu-lanczos/dmrg')
    if backend == 'qmc':
        if 'calib' not in system:
            raise ValueError('qmc 后端需要 system[\'calib\']')
        b = betas if betas is not None else [10.0]
        qr = _run_qmc(system, b, qmc_options)
        return dict(base, E0=float(qr['E'][0]), gap=np.nan,
                    n_atom=qr['n_atom'][0], E_err=float(qr['E_err'][0]),
                    average_sign=qr['average_sign'][0], samples=qr['samples'][0],
                    qmc_diagnostics=qr['diagnostics'][0])
    if sector is not None:
        if backend == 'dmrg':
            raise ValueError('sector 模式暂不支持 dmrg 后端'
                             '（TeNPy charge 等价能力后续接入）')
        if ('calib' in system and system.get('H') is None
                and system.get('H_builder') is None and not plan.get('projected')):
            check_number_conserving(system['calib'], system.get('Om'), system.get('terms'))
            idx = number_sector_basis(n_sites, sector)
            if backend == 'auto':
                backend = 'ed' if len(idx) <= dense_max else 'lanczos'
                base['backend'] = backend
            if backend == 'gpu-lanczos':
                out = _eq.torch_lanczos_ground_state(
                    system['calib'], system.get('Om'), system.get('delta'),
                    system.get('gamma'), system.get('terms'), basis_indices=idx,
                    m=system.get('krylov_dim', 48), seed=seed, dtype=dtype,
                    device=system.get('device', 'cuda'), return_state=return_state,
                    memory_budget_mb=memory_budget_mb)
                return dict(base, sector=int(sector), dim_sub=len(idx), sector_idx=idx, **out)
            if backend not in ('ed', 'lanczos'):
                raise ValueError('该后端不支持固定 sector')
            Hsub = hamiltonian_linear_operator(
                system['calib'], system.get('Om'), system.get('delta'),
                system.get('gamma'), system.get('terms'), basis_indices=idx)
            if backend == 'ed' and len(idx) <= dense_max:
                Hsub = np.column_stack([Hsub @ np.eye(1, len(idx), j).ravel()
                                       for j in range(len(idx))])
            r = _eq.ground_state(Hsub, dense_max=dense_max if backend == 'ed' else 0,
                                 k=k, v0=v0, seed=seed)
            bits = np.array([((idx >> (int(n_sites)-1-i)) & 1)
                             for i in range(int(n_sites))])
            p_sub = np.abs(r['vec']) ** 2
            obs = dict(n_atom=bits @ p_sub)
            if len(idx) <= CORR_MAX_DIM:
                # 固定扇区内的连通占据关联（= 连通 S^zS^z）；与全空间同一定义
                from .observables import corr_matrix
                obs['corr'] = corr_matrix(bits, p_sub)
            return dict(base, sector=int(sector), charge_sector=int(sector),
                        dim_sub=len(idx), sector_idx=idx, E0=r['E0'], gap=r['gap'],
                        evals=r['evals'], **({'vec': r['vec']} if return_state else {}),
                        **_pauli_observables(system, r['vec'], n_sites, states=idx),
                        **obs)
        if backend not in ('auto', 'ed', 'lanczos'):
            raise ValueError('直传 Hamiltonian 的 sector 仅支持 CPU ed/lanczos')
        labels = system.get('sector_labels')
        projected_codes = None
        if labels is None and plan.get('projected'):
            projected_codes = np.asarray(plan['states'], np.int64)
            labels = basis_bits(n_sites, projected_codes).sum(axis=0)
        if labels is None:
            raise ValueError("sector 模式需要 system['sector_labels']")
        labels = np.asarray(labels)
        Hfull = _system_H(system)
        r = _eq.ground_state(Hfull, sector=int(sector),
                             sector_labels=labels, sector_check=sector_check,
                             dense_max=dense_max, k=k, v0=v0, seed=seed)
        # vec 为**子空间**本征矢；全局矢需调用方按 sector_idx 自行展开。
        codes = r['sector_idx'] if projected_codes is None else projected_codes[r['sector_idx']]
        if backend == 'auto':
            base['backend'] = 'ed' if r['backend'] == 'eq-eigh' else 'lanczos'
        return dict(base, sector=int(sector), dim_sub=int(r['dim_sub']),
                    sector_idx=codes, E0=r['E0'], gap=r['gap'],
                    **({'n_atom': basis_bits(n_sites,codes) @ abs(r['vec'])**2}
                       if projected_codes is not None else {}),
                    evals=r['evals'], **({'vec': r['vec']} if return_state else {}))
    if backend == 'dmrg':
        if plan.get('projected'):
            raise ValueError('DMRG 当前未接入 projected HamiltonianPlan；请先选择 ed 或 qmc')
        from . import dmrg as _dm
        if system.get('ladder'):
            if (mps_dtype!='auto' or mps_diagnostics is not None or system.get('mps_order') is not None or system.get('psi0') is not None):
                raise ValueError('Finite-MPS extensions require generic calib, not the legacy ladder shortcut')
            if charge is not None or return_state:
                raise ValueError('ladder 快捷入口不支持 charge_sector/return_state；请使用通用 calib')
            # 均匀 2-leg 阶梯快捷入口（equibench_dmrg* 钉的就是这条）
            out = _dm.dmrg_rydberg_ladder(chi_max=chi_max,
                                          **_ladd_to_kwargs(system['ladder']))
            return dict(base, E0=out['E0'], gap=out.get('gap'),
                        S=out.get('S_mid'), chi=out.get('chi'))
        if 'calib' not in system:
            raise ValueError("dmrg 后端需 system['calib']（任意点云）"
                             "或 system['ladder']（均匀阶梯）")
        selected_calib=system['calib']
        if system.get('terms') is not None:
            selected_calib=dict(selected_calib,terms=list(system['terms']))
        out = _dm.dmrg_rydberg_cluster(
            selected_calib, system.get('Om'), system.get('delta'),
            chi_max=chi_max, truncerr=truncerr, max_sweeps=max_sweeps,
            charge_sector=charge, return_state=return_state,psi0=system.get('psi0'),order=system.get('mps_order'),
            mps_dtype=mps_dtype,diagnostics=mps_diagnostics,memory_budget_mb=memory_budget_mb,
            pauli_strings=system.get('pauli_strings'),
            progress_arr=progress_arr)
        return dict(base, E0=out['E0'], n_atom=out['n_atom'],
                    S=out['S_mid'], chi=out['chi'], order=out['order'],
                    entropy_cut=out.get('entropy_cut'),
                    n_legs=out['n_legs'], mpo_bond_dim=out['mpo_bond_dim'],
                    **({'state': out['state']} if return_state else {}),
                    charge_sector=out.get('charge_sector'),
                    conserve=('Sz' if out.get('charge_sector') is not None else None),
                    error_report=out.get('error_report',{}),used_options=out.get('used_options',{}),
                    **{key:out[key] for key in ['energy_variance','permutation_overlap',
                                                'diagnostics','corr','pauli_strings',
                                                'pauli_string_labels'] if key in out})
    if backend == 'free-fermion':
        if plan.get('projected'):
            raise ValueError('free-fermion 后端不支持 projected HamiltonianPlan')
        if 'calib' not in system:
            raise ValueError('free-fermion 后端需要 system[\'calib\']')
        from .free_fermion import solve_ground_state
        return dict(base, **solve_ground_state(
            system['calib'], system.get('delta'),
            charge_sector=system.get('charge_sector'),
            terms=system.get('terms'), Om=system.get('Om')))
    if (backend == 'lanczos' and 'calib' in system and not plan.get('projected')
            and system.get('H') is None and system.get('H_builder') is None):
        H = hamiltonian_linear_operator(system['calib'], system.get('Om'),
                                        system.get('delta'), system.get('gamma'),
                                        system.get('terms'))
    elif backend == 'gpu-lanczos':
        if plan.get('projected'):
            raise ValueError('gpu-lanczos 尚不支持 projected Plan；请选择 CPU lanczos')
        if 'calib' not in system:
            raise ValueError('gpu-lanczos 后端需要 system[\'calib\']')
        return dict(base, **_eq.torch_lanczos_ground_state(
            system['calib'], system.get('Om'), system.get('delta'),
            system.get('gamma'), system.get('terms'),
            m=system.get('krylov_dim', 48), seed=seed,
            dtype=dtype, device=system.get('device', 'cuda'),
            return_state=return_state, memory_budget_mb=memory_budget_mb))
    else:
        H = _system_H(system)
    if backend == 'gpu-batch':
        Hp = H if isinstance(H, np.ndarray) else H.toarray()
        # A single system describes one Hamiltonian, even if batch is a
        # routing hint. Repeating it B times only wastes host/GPU memory.
        require_workspace(Hp.size*Hp.itemsize*4, memory_budget_mb, '批量 ED')
        rb = _eq.batched_ground_state(Hp[None])
        return dict(base, E0=float(rb['E0'][0]), gap=float(rb['gap'][0]),
                    n_atom=rb['n_atom'][0])
    if backend == 'lanczos':
        r = _eq.ground_state(H if isinstance(H, sp.linalg.LinearOperator)
                             else H if sp.issparse(H) else sp.csr_matrix(H),
                             dense_max=0, k=k, v0=v0, seed=seed)
    else:  # ed
        r = _eq.ground_state(H if isinstance(H, np.ndarray)
                             else sp.csr_matrix(H),
                             dense_max=dense_max, k=k, v0=v0, seed=seed)
    if plan.get('projected'):
        st = np.asarray(system['plan']['states'], dtype=int)
        obs = dict(n_atom=basis_bits(n_sites, st) @ (np.abs(r['vec']) ** 2))
        obs.update(_pauli_observables(system, r['vec'], n_sites, states=st))
    elif 'calib' in system and not explicit_h:
        obs = _gs_observables(r['vec'], int(n_sites))
        obs.update(_pauli_observables(system, r['vec'], n_sites))
    else:
        obs = _pauli_observables(system, r['vec'], n_sites)
    return dict(base, E0=r['E0'], gap=r['gap'], evals=r['evals'],
                **({'vec': r['vec']} if return_state else {}), **obs)



def _solve_thermal(system, base, *, betas, num, nvec, seed, exact, qmc_options,
                   memory_budget_mb, dtype):
    backend = base['backend']
    if backend == 'qmc':
        if 'calib' not in system:
            raise ValueError('qmc 后端需要 system[\'calib\']')
        qr = _run_qmc(system, betas, qmc_options)
        return dict(base, betas=qr['betas'], E=qr['E'], C=qr['C'], F=qr['F'],
                    E_err=qr['E_err'], n_atom=qr['n_atom'],
                    average_sign=qr['average_sign'], samples=qr['samples'],
                    qmc_diagnostics=qr['diagnostics'])
    H = _system_H(system)
    if exact or backend == 'ed':
        require_workspace(H.shape[0]**2*16*4, memory_budget_mb, '精确热态谱分解')
    if backend == 'ed':
        spectrum = np.linalg.eigvalsh(H.toarray() if sp.issparse(H) else np.asarray(H))
        return dict(base, **_eq.thermal_from_spectrum(spectrum,betas))
    if backend in ('gpu-kpm', 'kpm-gpu'):
        kp = _kpm.kpm_moments_gpu(H, num=num, nvec=nvec, seed=seed,
                                  chunk=system.get('kpm_chunk', 32),
                                  memory_budget_mb=memory_budget_mb, device=system.get('device','cuda'), dtype=dtype)
        # Reuse the numerically identical post-processing path.
        th = _kpm._thermal_from_moments(H, betas if betas is not None else [1.0], kp, num)
        return dict(base, betas=th['betas'], E=th['E'], C=th['C'], F=th['F'])
    th = _kpm.kpm_thermal(H, betas if betas is not None else [1.0], num=num,
                          nvec=nvec, seed=seed, exact=(exact or backend == 'ed'))
    return dict(base, betas=th['betas'], E=th['E'], C=th['C'], F=th['F'])



def _solve_dos(system, base, *, num, nvec, seed, exact, memory_budget_mb, dtype):
    backend = base['backend']
    H = _system_H(system)
    if exact or backend == 'ed':
        require_workspace(H.shape[0]**2*16*4, memory_budget_mb, '精确 DOS 谱分解')
    lo, hi, _, _ = _kpm._bounds(H)
    span = max(hi-lo, 1e-9)
    energies = np.linspace(lo+span*1e-6, hi-span*1e-6, 400)
    if backend == 'gpu-kpm':
        moments = _kpm.kpm_moments_gpu(H, num=num, nvec=nvec, seed=seed,
            device=system.get('device','cuda'), dtype=dtype,
            chunk=system.get('kpm_chunk',32), memory_budget_mb=memory_budget_mb)
        rho = _kpm._dos_from_moments(*moments, energies, num)
    else:
        rho = _kpm.kpm_dos(H, energies, num=num, nvec=nvec, seed=seed,
                           exact=(exact or backend == 'ed'))
    return dict(base, energies=energies, rho=rho)



def _run_qmc(system, betas, options):
    return _qmc.sse_run(system['calib'], betas=betas, Om=system.get('Om'),
                        delta=system.get('delta'), terms=system.get('terms'),
                        plan=system.get('plan'), **options)


def _ladd_to_kwargs(lp):
    """ladder dict → dmrg_rydberg_ladder kwargs（校验键）。"""
    need = {'length', 'V_r', 'J_r', 'A_r', 'V_l', 'J_l', 'A_l', 'delta', 'Omega'}
    return {k: lp[k] for k in need if k in lp}
