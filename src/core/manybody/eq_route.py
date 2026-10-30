# -*- coding: utf-8 -*-
"""与具体实验协议无关的平衡态求解编排。

solve_equilibrium 规范化输入、校验能力并路由；各任务由独立的 gs/thermal/
DOS/ramp 辅助函数执行。输入支持物理 calib/Plan 或明确的 H/H_builder；专用
模型后端不能忽略显式矩阵。结果统一附带数据语义与基身份。

实验波形和论文协议留在模型/应用层。后端声明见 backends.py。
"""
import time
from .basis import basis_bits
from .backends import normalize_equilibrium_backend
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


def route_equilibrium(n_sites, want='gs', *, is_ladder=False, batch=1,
                      gpu=None, dense_max=_DENSE_MAX, dmrg_ok=None):
    """纯决策函数：→ backend 名。可单独测（不跑真解）。

    `dmrg_ok`：能不能上 DMRG。缺省跟随 `is_ladder`（向后兼容）；新架构下任意点云
    都能 DMRG（自写 autoMPO），由调用方按“顺序质量”给 True/False。
    """
    if want in ('thermal', 'dos'):
        return 'kpm'
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

    `dense=True` → 把结果稠密化（`want='dos'` 要拿 `eigvalsh` 定谱界，它不吃稀疏）。
    ⚠ 2026-10-02 修：本函数原先**没有** `dense` 形参，而 dos 分支一直按
    `_system_H(system, dense=True)` 调用 ⇒ `want='dos'` **从未跑通过**（无条件
    TypeError，与 system 用哪种口径无关）。三种口径现在统一支持 dense。
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
    """基态向量 → 逐原子⟨n_i⟩（与引擎同 MSB 基序：bit=1↔|r⟩）。拿不到就当空。"""
    p = np.abs(np.asarray(vec).ravel()) ** 2
    return dict(n_atom=_eq.bit_matrix(int(n_sites), 1 << int(n_sites)) @ p)


def _equilibrium_result(fn):
    @wraps(fn)
    def solve(system, *args, **kwargs):
        result = fn(system, *args, **kwargs)
        result.setdefault('requested_backend', kwargs.get('backend', 'auto'))
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


@_equilibrium_result
def solve_equilibrium(system, want='gs', *, betas=None, backend='auto',
                      sector=None, k=2, v0=None, ramp=None, log=None,
                      sector_check=True, chi_max=200, num=400, nvec=200,
                      seed=0, dense_max=_DENSE_MAX, exact=False,
                      truncerr=1e-10, max_sweeps=60, qmc_sweeps=2000,
                      qmc_burn_in=500, qmc_measure_every=5, qmc_M=None,
                      memory_budget_mb=None, dtype='float64', return_state=False):
    """统一平衡态求解。

    system：{'calib':..,'Om':..,'delta':.., 'ladder':dict(...), 'is_ladder':bool,
             'n_sites':int, 'batch':int}
      - ladder: 传入则 DMRG 可用（含 V_r/J_r/A_r/V_l/J_l/A_l/delta/Omega）。
      - **模型后端解耦**（2026-10-02）：'H'（矩阵直传）或 'H_builder'(Om, delta)
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
    if return_state and want != 'gs':
        raise ValueError('return_state 当前仅用于基态或实时 MPS 演化')
    charge = system.get('charge_sector')
    if charge is not None:
        if isinstance(charge, (bool, np.bool_)) or int(charge) != charge:
            raise ValueError('charge_sector 必须是整数')
        if sector is not None and int(sector) != int(charge):
            raise ValueError('sector 与 charge_sector 冲突')
        if want != 'gs':
            raise ValueError('charge_sector 当前仅支持基态求解；不能静默忽略热态/DOS 的扇区')
        if backend not in ('dmrg', 'free-fermion'):
            sector = int(charge)
    if (backend == 'auto' and plan.get('projected')
            and want == 'gs'):
        # The reduced basis is the routing input; never send a projected plan
        # to a backend which would silently rebuild the unconstrained MPO.
        backend = ('ed' if int(system['plan']['dim']) <= dense_max else 'lanczos')
    if want in ('gs', 'thermal', 'dos') and backend == 'auto' and sector is None:
        if n_sites is None:
            raise ValueError('system 需要 n_sites（或 calib）以自动路由；'
                             "直传 H 时可显式 backend='ed'/'lanczos'")
        backend = route_equilibrium(n_sites, want, is_ladder=is_ladder,
                                    batch=batch, dense_max=dense_max,
                                    dmrg_ok=dmrg_ok)
    # `n_sites` 进 Result（2026-10-02 补）：契约里声明了它，但各分支一直没给
    # ⇒ 两条"Result 契约键齐"判据常年红。直传 H 且无 calib 时**不放**——
    # 那种口径下格点数是模型侧知识，塞个 None 是"假装知道"。
    backend = normalize_equilibrium_backend(backend, want) if want != 'ramp' else backend
    if exact and backend == 'gpu-kpm':
        raise ValueError('gpu-kpm 不提供 exact 谱迹；请选择 CPU kpm 或 ed')
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
            dtype=dtype, return_state=return_state)
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
                  memory_budget_mb, dtype, return_state):
    backend = base['backend']
    plan = system.get('plan') or {}
    charge = system.get('charge_sector')
    n_sites = base.get('n_sites')
    explicit_h = system.get('H') is not None or system.get('H_builder') is not None
    if return_state and backend in ('qmc', 'gpu-batch', 'free-fermion'):
        raise ValueError('所选基态后端尚未实现 return_state')
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
            return dict(base, sector=int(sector), charge_sector=int(sector),
                        dim_sub=len(idx), sector_idx=idx, E0=r['E0'], gap=r['gap'],
                        evals=r['evals'], **({'vec': r['vec']} if return_state else {}),
                        n_atom=bits @ np.abs(r['vec'])**2)
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
            if charge is not None or return_state:
                raise ValueError('ladder 快捷入口不支持 charge_sector/return_state；请使用通用 calib')
            # 均匀 2-leg 阶梯参数表（旧路：equibench_dmrg* 钉的就是这条）
            out = _dm.dmrg_rydberg_ladder(chi_max=chi_max,
                                          **_ladd_to_kwargs(system['ladder']))
            return dict(base, E0=out['E0'], gap=out.get('gap'),
                        S=out.get('S_mid'), chi=out.get('chi'))
        if 'calib' not in system:
            raise ValueError("dmrg 后端需 system['calib']（任意点云）"
                             "或 system['ladder']（均匀阶梯）")
        out = _dm.dmrg_rydberg_cluster(
            system['calib'], system.get('Om'), system.get('delta'),
            chi_max=chi_max, truncerr=truncerr, max_sweeps=max_sweeps,
            charge_sector=charge, return_state=return_state)
        return dict(base, E0=out['E0'], n_atom=out['n_atom'],
                    S=out['S_mid'], chi=out['chi'], order=out['order'],
                    n_legs=out['n_legs'], mpo_bond_dim=out['mpo_bond_dim'],
                    **({'state': out['state']} if return_state else {}),
                    charge_sector=system.get('charge_sector'),
                    conserve=('Sz' if system.get('charge_sector') is not None else None))
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
    elif 'calib' in system and not explicit_h:
        obs = _gs_observables(r['vec'], int(n_sites))
    else:
        obs = {}
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
