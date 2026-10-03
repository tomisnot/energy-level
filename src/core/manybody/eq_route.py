# -*- coding: utf-8 -*-
"""平衡态**算法路由 + 统一入口**：按问题特征挑 ED / GPU 批量 / Lanczos / DMRG / KPM。

把已实现的三条主力（`equilibrium` ED/批量/Lanczos、`dmrg` TeNPy 2D 柱、`kpm` 谱方法）
收敛到一个门面 `solve_equilibrium(system, want=...)` 与纯函数 `route_equilibrium(...)`，
返回**统一 Result 字典**。这是"接进 runspec/GUI"的核心可测层——分派/GUI 绑定是后续薄层
（本模块不改动 `runspec.evolve_run` 单分派点，以免破坏 `runspec_contract` 判据）。

路由规则（want='gs' 基态）：
  批点数>1 且有 GPU      → 'gpu-batch'（海量独立点一次批量化，蒸馏数据工厂主力）
  维数 2^n ≤ dense_max   → 'ed'（稠密 eigh，机器精度、可给全套观测）
  2-leg 阶梯且大          → 'dmrg'（TeNPy，ED 够不到的 2D 尺寸）
  其余大稀疏              → 'lanczos'（scipy eigsh）
want='thermal'/'dos'      → 'kpm'（有限温/态密度唯一可控、无符号问题）

**模型后端解耦**（2026-10-02）：system 支持 `'H'`（矩阵直传）与 `'H_builder'(Om, delta)`
（参数化构造）两种口径，摆脱对 `assemble.build_H_sparse` 的硬绑定——三能级 V 型
（`rydberg_fci`）等任何模型后端都能直接进本门面。注意：`'H'` 直传口径下**不做逐原子
观测**（bit 序约定属于模型层），Result 只含能量/本征矢。

**求解动词**（2026-10-02 收敛）：
  want='gs' + sector=int  → 固定激励数扇区切片求解（守恒抽样校验，不守恒即 raise）
  want='ramp'             → 参数 ramp 逐点瞬时基态（warm-start 编排）

**边界**（2026-10-02 晚裁决）：本模块只放**与具体模型/实验协议无关**的求解编排。
`calibrate_mu`（E(N)→化学势→ramp δ 端点）与 `sweep_profile`（三段式 ramp 波形）
已摘出——它们是 arXiv:2202.00699 Fig.4 的**实验协议选择**，不是求解动词：
δ 端点取"µ_c(0→1)−offset"与"平台中央"、Ω 的三段升降划分，都是该论文的做法，
换一篇论文（乃至换一个扫描协议）就全变了，通用引擎不该持有唯一的 ramp 波形。
去向：`sweep_profile` → `rydberg_fci`（论文模型侧，零依赖纯函数）；
`calibrate_mu` → 论文脚本本地（循环调本门面的 `want='gs', sector=N` 通用动词）。
"""
import time

import numpy as np
import scipy.sparse as sp

from . import assemble as _asm
from . import equilibrium as _eq
from . import kpm as _kpm

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
    Hs = _asm.build_H_sparse(system['calib'], system.get('Om'), system.get('delta'))
    return np.asarray(Hs.todense()) if dense else Hs


def _system_H(system, om=None, de=None, dense=False):
    """system → 哈密顿量：'H' 直传优先，其次 'H_builder'(Om, delta)，再次 calib。

    `dense=True` → 把结果稠密化（`want='dos'` 要拿 `eigvalsh` 定谱界，它不吃稀疏）。
    ⚠ 2026-10-02 修：本函数原先**没有** `dense` 形参，而 dos 分支一直按
    `_system_H(system, dense=True)` 调用 ⇒ `want='dos'` **从未跑通过**（无条件
    TypeError，与 system 用哪种口径无关）。三种口径现在统一支持 dense。
    """
    if 'H' in system and system['H'] is not None:
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
    try:
        p = np.abs(np.asarray(vec).ravel()) ** 2
        bits = _eq.bit_matrix(int(n_sites), 1 << int(n_sites))
        return dict(n_atom=bits @ p)
    except Exception:                                   # noqa: BLE001
        return {}


def solve_equilibrium(system, want='gs', *, betas=None, backend='auto',
                      sector=None, k=2, v0=None, ramp=None, log=None,
                      sector_check=True, chi_max=200, num=400, nvec=200,
                      seed=0, dense_max=_DENSE_MAX, exact=False,
                      truncerr=1e-10, max_sweeps=60):
    """统一平衡态求解。

    system：{'calib':..,'Om':..,'delta':.., 'ladder':dict(...), 'is_ladder':bool,
             'n_sites':int, 'batch':int}
      - ladder: 传入则 DMRG 可用（含 V_r/J_r/A_r/V_l/J_l/A_l/delta/Omega）。
      - **模型后端解耦**（2026-10-02）：'H'（矩阵直传）或 'H_builder'(Om, delta)
        优先于 calib——三能级 V 型（rydberg_fci）等任何模型后端可直接进门面。
        直传口径下**不做逐原子观测**（bit 序约定属于模型层），Result 只含
        能量/本征矢。'sector_labels'：1 维逐基矢激励数（sector 模式必需）。
    want：'gs' | 'thermal' | 'dos' | 'ramp'。
    sector：None=全空间（缺省）；int=固定激励数扇区切片（仅 'gs'；首次调用做
      守恒抽样校验，循环调用方后续可用 sector_check=False 关闭）。
      dmrg 后端不支持扇区（TeNPy charge 等价能力后续接入，不静默）。
    k：gs 请求的本征值个数（evals 同序返回）。
    v0：Lanczos 初值（warm-start；dense 分支忽略）。
    ramp（want='ramp' 必填）：dict(Om_t, de_t, L[, n_tot, v0_warm, dense_max,
      taus])——Om_t/de_t 逐 τ 驱动序列；L=密度归一格点数（截断模型 dim≠2^L
      时无法从维度反推，必须显式）；n_tot=1 维逐基矢总激发数对角算符（模型
      侧知识，用于 ns 观测）；v0_warm 缺省 True。
    返回统一 dict：{'want','backend', + n_sites/E0/gap/evals/S（gs，按口径）或
      E/C/F(betas)（thermal）或 rho(dos) 或 taus/gaps/ns（ramp）}。
    """
    n_sites = system.get('n_sites')
    if n_sites is None and 'calib' in system:
        n_sites = len(system['calib']['atoms'])
    batch = system.get('batch', 1)
    is_ladder = system.get('is_ladder', 'ladder' in system)
    dmrg_ok = system.get('dmrg_ok', is_ladder)
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
    base = dict(want=want, backend=backend)
    if n_sites is not None:
        base['n_sites'] = int(n_sites)

    if want == 'ramp':
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

    if want == 'gs':
        if sector is not None:
            if backend == 'dmrg':
                raise ValueError('sector 模式暂不支持 dmrg 后端'
                                 '（TeNPy charge 等价能力后续接入）')
            labels = system.get('sector_labels')
            if labels is None:
                raise ValueError("sector 模式需要 system['sector_labels']")
            labels = np.asarray(labels)
            Hfull = _system_H(system)
            if sector_check:
                _check_sector_conserving(Hfull, labels, seed=seed)
            idx = np.where(labels == int(sector))[0]
            if len(idx) == 0:
                raise ValueError('扇区 %d 为空（labels 范围 %d..%d）'
                                 % (sector, labels.min(), labels.max()))
            Hfull = Hfull if sp.issparse(Hfull) else sp.csr_matrix(Hfull)
            Hs = Hfull[np.ix_(idx, idx)].tocsr()
            dim_sub = Hs.shape[0]
            r = _eq.ground_state(Hs,
                                 dense_max=(dim_sub if dim_sub <= dense_max
                                            else 0),
                                 k=k, v0=v0, seed=seed)
            # vec 为**子空间**本征矢；全局矢需调用方按 sector_idx 自行展开。
            return dict(base, sector=int(sector), dim_sub=int(dim_sub),
                        sector_idx=idx, E0=r['E0'], gap=r['gap'],
                        evals=r['evals'], vec=r['vec'])
        if backend == 'dmrg':
            from . import dmrg as _dm
            if system.get('ladder'):
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
                chi_max=chi_max, truncerr=truncerr, max_sweeps=max_sweeps)
            return dict(base, E0=out['E0'], n_atom=out['n_atom'],
                        S=out['S_mid'], chi=out['chi'], order=out['order'],
                        n_legs=out['n_legs'], mpo_bond_dim=out['mpo_bond_dim'])
        H = _system_H(system)
        if backend == 'gpu-batch':
            Hp = H if isinstance(H, np.ndarray) else H.toarray()
            B = int(batch)
            rb = _eq.batched_ground_state(np.repeat(Hp[None], B, 0))
            return dict(base, E0=float(rb['E0'][0]), gap=float(rb['gap'][0]),
                        n_atom=rb['n_atom'][0])
        if backend == 'lanczos':
            r = _eq.ground_state(H if sp.issparse(H) else sp.csr_matrix(H),
                                 dense_max=0, k=k, v0=v0, seed=seed)
        else:  # ed
            r = _eq.ground_state(H if isinstance(H, np.ndarray)
                                 else sp.csr_matrix(H),
                                 dense_max=dense_max, k=k, v0=v0, seed=seed)
        if 'calib' in system and 'H' not in system:
            obs = _gs_observables(r['vec'], int(n_sites))
        else:
            obs = {}
        return dict(base, E0=r['E0'], gap=r['gap'], evals=r['evals'], **obs)

    if want == 'thermal':
        H = _system_H(system)
        th = _kpm.kpm_thermal(H, betas if betas is not None else [1.0], num=num,
                              nvec=nvec, seed=seed, exact=exact)
        return dict(base, betas=th['betas'], E=th['E'], C=th['C'], F=th['F'])

    if want == 'dos':
        H = _system_H(system, dense=True)
        ev = np.linalg.eigvalsh(H)
        Eg = np.linspace(ev[0] + 1e-9, ev[-1] - 1e-9, 400)
        rho = _kpm.kpm_dos(_system_H(system), Eg, num=num, nvec=nvec, seed=seed,
                           exact=exact)
        return dict(base, energies=Eg, rho=rho)

    raise ValueError('want 需为 gs/thermal/dos/ramp，实=%r' % want)


def _ladd_to_kwargs(lp):
    """ladder dict → dmrg_rydberg_ladder kwargs（校验键）。"""
    need = {'length', 'V_r', 'J_r', 'A_r', 'V_l', 'J_l', 'A_l', 'delta', 'Omega'}
    return {k: lp[k] for k in need if k in lp}
