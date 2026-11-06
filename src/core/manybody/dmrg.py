# -*- coding: utf-8 -*-
"""2D 柱形（ladder/cylinder）DMRG 后端 —— 包 TeNPy（CPU·BLAS，通用 2D 平衡态主力之一）。

角色：manybody 家族的**平衡态**求解后端之一（与 `equilibrium` 的 ED/批量 ED、将来的
SSE-QMC 并列）。吃 TeNPy 的 `CouplingMPOModel`+`TwoSiteDMRGEngine`，专攻 ED 够不着的
2D 相关多体基态；键维 chi / truncerr 给可控误差。⚠ TeNPy 是 CPU/BLAS，DMRG 顺序扫描
⇒ **GPU 收益低**（大 2D 用 GPU 批量 ED 更值；本模块负责"超出 ED 尺寸的通用 2D 基态"）。

模型：横场(+纵向) Ising 柱（= 深阻塞里德堡 Ising 在赝自旋表象下的等价形式）：
    H = Σ_<ij> Jz Ŝz_i Ŝz_j − Σ_i (hx Ŝx_i + hz Ŝz_i)
几何：`Ladder` 二维柱（2 腿=周界，沿链方向长度 `length`）；`pairs['nearest_neighbors']`
自动含 rung+leg 近邻键。约定：up=|g⟩、dw=|r⟩，Ŝz=(Î/2−n̂)，Ŝx=σ̂x/2。

判据 `checks/equibench_dmrg_selfcheck.py` 把 `full_H`（稠密 2^N）同时喂 DMRG / TeNPy
ExactDiag / 我们的 ED / 我们的 GPU 批量 —— 四方对齐即证明本后端正确（与约定映射无关）。
"""
import numpy as np

from .observables import bit_matrix
from .couplings import active_terms

__all__ = ['dmrg_ising_ladder', 'ising_ladder_model', 'ising_ladder_full_hamiltonian',
           'cylinder_order', 'terms_from_calib', 'rydberg_cluster_model',
           'dmrg_rydberg_cluster', 'rydberg_cluster_full_hamiltonian', 'dmrg_run']

_HAS_TENPY = None


def _require_tenpy():
    global _HAS_TENPY
    if _HAS_TENPY is False:
        raise ImportError('需要 TeNPy（pip install tenpy）作为 2D 柱形 DMRG 后端')
    _HAS_TENPY = False   # 先置 False，成功导入后置 True
    import tenpy  # noqa: F401
    _HAS_TENPY = True


def ising_ladder_model(length, Jz=1.0, hx=0.5, hz=0.0):
    """构造 2 腿 × length 的 Ising 柱 TeNPy 模型（CouplingMPOModel 子类）。"""
    _require_tenpy()
    from tenpy.models.model import CouplingMPOModel
    from tenpy.networks.site import SpinHalfSite

    class _IsingLadder(CouplingMPOModel):
        def init_sites(self, params):
            return SpinHalfSite(conserve=None)

        def init_terms(self, params):
            Jz_ = params.get('Jz', 1.0)
            hx_ = params.get('hx', 0.5)
            hz_ = params.get('hz', 0.0)
            for u in range(len(self.lat.unit_cell)):        # unit-cell 下标，非 mps 下标
                self.add_onsite(-hx_, u, 'Sx')
                self.add_onsite(-hz_, u, 'Sz')
            for u1, u2, dx in self.lat.pairs['nearest_neighbors']:
                self.add_coupling(Jz_, u1, 'Sz', u2, 'Sz', dx)

    return _IsingLadder(dict(lattice='Ladder', L=int(length), Jz=Jz, hx=hx, hz=hz,
                             bc_MPS='finite'))


def ising_ladder_full_hamiltonian(length, Jz=1.0, hx=0.5, hz=0.0):
    """→ (dense 2^N×2^N H, N_sites)：TeNPy ExactDiag 的全空间 H，供交叉验证。"""
    _require_tenpy()
    from tenpy.algorithms.exact_diag import ExactDiag
    M = ising_ladder_model(length, Jz, hx, hz)
    ed = ExactDiag(M)
    ed.build_full_H_from_mpo()
    fh = ed.full_H
    if hasattr(fh, 'to_ndarray'):                       # tenpy np_conserved.Array
        H = fh.to_ndarray()
    elif hasattr(fh, 'todense'):                        # scipy sparse
        H = np.asarray(fh.todense())
    else:
        H = np.asarray(fh)
    n = M.lat.N_sites
    return np.asarray(H, dtype=complex), n


def dmrg_ising_ladder(length, Jz=1.0, hx=0.5, hz=0.0, chi_max=200,
                      truncerr=1e-10, max_sweeps=60):
    """2D 柱形 DMRG 求基态。→ dict(E0, gap?, n_sites, chi, S_mid, backend)。

    chi=chi_max、truncerr 控误差；返回 E0（MHz 无关，按模型无量纲单位）。
    """
    _require_tenpy()
    M = ising_ladder_model(length, Jz, hx, hz)
    out = dmrg_run(M, chi_max=chi_max, truncerr=truncerr, max_sweeps=max_sweeps)
    out['backend'] = 'tenpy-dmrg-ladder'
    return out


# =====================================================================
# 一般复耦合 Rydberg 阶梯（逐键 V·nn + J·σ⁺σ⁻ + A·σ⁺σ⁺ + 位δn + Ω/2 σˣ）
#   与 assemble.build_H_sparse 逐字同（n=½Î−Ŝz, σˣ=2Ŝx, 引擎σ⁺=TeNPy 'Sm'）。
# =====================================================================
def rydberg_ladder_model(length, V_r=0.0, J_r=0.0, A_r=0.0,
                         V_l=0.0, J_l=0.0, A_l=0.0, delta=0.0, Omega=0.0):
    """2-leg Rydberg 阶梯，rung/leg 分型复耦合；返回 TeNPy CouplingMPOModel。

    基矢 up=|g⟩(bit0)、down=|r⟩(bit1)，与引擎一致。δ/Ω 均匀，V 实、J/A 复。
    """
    _require_tenpy()
    from tenpy.models.model import CouplingMPOModel
    from tenpy.networks.site import SpinHalfSite

    class _RydbergLadder(CouplingMPOModel):
        def init_sites(self, params):
            # 注册与引擎逐字同的单格点算符（基 up=g=[1,0], down=r=[0,1]）：
            #   Nhat=|r><r|, P=σ⁺=|r><g|, M=σ⁻=|g><r|, SX=σˣ
            site = SpinHalfSite(conserve=None)
            site.add_op('Nhat', np.array([[0, 0], [0, 1]], dtype=complex))
            site.add_op('P', np.array([[0, 0], [1, 0]], dtype=complex))
            site.add_op('M', np.array([[0, 1], [0, 0]], dtype=complex))
            site.add_op('SX', np.array([[0, 1], [1, 0]], dtype=complex))
            site.add_op('SY', np.array([[0, -1j], [1j, 0]], dtype=complex))
            return site

        def init_terms(self, params):
            # onsite: δ n̂ + (Ω σ⁺ + Ω* σ⁻)/2
            for u in range(len(self.lat.unit_cell)):
                self.add_onsite(delta, u, 'Nhat')
                self.add_onsite(Omega / 2.0, u, 'P')
                self.add_onsite(np.conj(Omega) / 2.0, u, 'M')

            def bond(typ, V, J, A):
                if typ not in self.lat.pairs:
                    return
                for u1, u2, dx in self.lat.pairs[typ]:
                    if V != 0:
                        self.add_coupling(V, u1, 'Nhat', u2, 'Nhat', dx)
                    if J != 0:            # J σ⁺_iσ⁻_j + h.c.
                        self.add_coupling(J, u1, 'P', u2, 'M', dx)
                        self.add_coupling(np.conj(J), u1, 'M', u2, 'P', dx)
                    if A != 0:            # A σ⁺_iσ⁺_j + h.c.
                        self.add_coupling(A, u1, 'P', u2, 'P', dx)
                        self.add_coupling(np.conj(A), u1, 'M', u2, 'M', dx)

            bond('rung_NN', V_r, J_r, A_r)
            bond('leg_NN', V_l, J_l, A_l)

    return _RydbergLadder(dict(lattice='Ladder', L=int(length), bc_MPS='finite'))


def _run_dmrg(M, chi_max=200, truncerr=1e-10, max_sweeps=60):
    """均匀阶梯旧入口：收敛到 `dmrg_run`（单一引擎调用实现）。"""
    out = dmrg_run(M, chi_max=chi_max, truncerr=truncerr, max_sweeps=max_sweeps)
    out['backend'] = 'tenpy-dmrg-ladder'
    return out


def dmrg_rydberg_ladder(length, V_r=0.0, J_r=0.0, A_r=0.0, V_l=0.0, J_l=0.0,
                        A_l=0.0, delta=0.0, Omega=0.0, chi_max=200,
                        truncerr=1e-10, max_sweeps=60):
    """一般复耦合 Rydberg 阶梯的 DMRG 基态（含 σ⁺σ⁻/σ⁺σ⁺/复耦合）。"""
    _require_tenpy()
    M = rydberg_ladder_model(length, V_r, J_r, A_r, V_l, J_l, A_l, delta, Omega)
    return _run_dmrg(M, chi_max, truncerr, max_sweeps)


def rydberg_ladder_full_hamiltonian(length, V_r=0.0, J_r=0.0, A_r=0.0,
                                    V_l=0.0, J_l=0.0, A_l=0.0, delta=0.0, Omega=0.0):
    """一般阶梯的稠密 2^N H（TeNPy ExactDiag）——验证 MPO=引擎 H 与 DMRG。"""
    _require_tenpy()
    from tenpy.algorithms.exact_diag import ExactDiag
    M = rydberg_ladder_model(length, V_r, J_r, A_r, V_l, J_l, A_l, delta, Omega)
    ed = ExactDiag(M)
    ed.build_full_H_from_mpo()
    fh = ed.full_H
    if hasattr(fh, 'to_ndarray'):
        H = fh.to_ndarray()
    elif hasattr(fh, 'todense'):
        H = np.asarray(fh.todense())
    else:
        H = np.asarray(fh)
    return np.asarray(H, dtype=complex), M.lat.N_sites


# =====================================================================
# 任意点云（多腿/无周期）→ 自写 MPO：Chain + strength 数组 ⇒ 逐键复耦合
#   不再要求平移不变/2-leg：逐位 δ/Ω 与同一程长的逐键 V/J/A 都由数组给出，
#   因而直接吃引擎 calib（与 assemble.build_H_sparse 同式）。
#   MPS 顺序由 `cylinder_order` 给出（梯级优先）——顺序只影响 DMRG 速度，不影响正确性。
# =====================================================================
def cylinder_order(positions, tol=1e-6):
    """点云 → 柱形（**梯级优先**）MPS 顺序。→ (order, info)

    按 y 分行（行=圆柱的一腿，分行尺度 = 最近邻距离的 1/4），行内按 x；
    输出顺序为“列优先、腿最快”：index = col·n_legs + row。于是
      · 梯级键（同列跨腿）程 = 1；腿键（同腿邻列）程 = 腿数 → **最大程与链长无关**。
    ❌ 不用“蛇形反向行”：那会让首个梯级键跳整条链（2×10 实测程达 19 →
    每条长键各占一个 MPO 通道，DMRG 从秒级涨到 200s）。蛇形只适合
    “腿向为短轴”的宽柱，而这里腿数才是短轴。
    任意腿数、两方向均开边界（无周期）；非矩形（行不等长）按列补齐。
    `order[k]` = 第 k 个 MPS 位的原子下标。
    """
    pts = [tuple(float(v) for v in (p or (0.0, 0.0, 0.0))[:2]) for p in positions]
    n = len(pts)
    if n == 0:
        return [], dict(n_legs=0, leg_len=[], perm=[])
    ys = np.array([p[1] for p in pts])
    xs = np.array([p[0] for p in pts])
    # 分行阈值拿“晶格常数”当尺度：取几何最近邻距离中位数的 1/4。
    # （用“y 间隔中位数”会被行内微抖主导而行被拆散；实测踩到。分行错不影响
    #   正确性，只影响 MPS 顺序质量 → 见 terms_from_calib 的 two_site 注）
    if n >= 2:
        P = np.array([[float(a), float(b)] for a, b in pts])
        d = np.linalg.norm(P[:, None, :] - P[None, :, :], axis=2)
        np.fill_diagonal(d, np.inf)
        nn = np.median(d.min(axis=1))
        thr = max(tol, 0.25 * float(nn))
    else:
        thr = tol
    rows = []
    for i in np.argsort(ys, kind='stable'):
        y = float(ys[i])
        if rows and abs(y - rows[-1][0]) <= thr:
            rows[-1][1].append(int(i))
        else:
            rows.append((y, [int(i)]))
    for r in rows:                       # 行内按 x 升序
        r[1].sort(key=lambda a: float(xs[a]))
    order = []
    width = max(len(r[1]) for r in rows)
    for c in range(width):               # 列优先：同一列的各腿相邻
        for _y, idxs in rows:
            if c < len(idxs):
                order.append(int(idxs[c]))
    order = [int(a) for a in order]
    perm = [0] * n
    for k, a in enumerate(order):
        perm[int(a)] = k
    return order, dict(n_legs=len(rows), leg_len=[len(r[1]) for r in rows],
                       perm=perm)


def terms_from_calib(calib, Om=None, delta=None, gamma=None, *, order=None):
    """引擎 calib（逐键复 V/J/A + 逐位 δ/Ω）→ (order, onsite, couplings, info)。

    与 `assemble.build_H_sparse` 同式：H = Σ δ_i n_i + Σ Ω_i/2 σˣ_i
      + Σ_{i<j} [V n_i n_j + J σ⁺_iσ⁻_j + A σ⁺_iσ⁺_j + h.c.]；只取 calib['terms'] 已开的项。

    `couplings` = [(coeff, a, opA, b, opB), …]，已按 **a < b（MPS 下标）** 归一：
    归一时是“交换整条 (算符,格点) 对”而非只换下标——复 J/A 的 σ⁺/σ⁻ 不对称，
    只换下标会把 J 项共轭错（实测陷阱：蛇形反向行的 permErr 达 2e-1）。
    """
    atoms = calib['atoms']
    n = len(atoms)
    use = set(active_terms(calib))
    Om = np.zeros(n, complex) if Om is None else np.asarray(Om, complex)
    delta = np.zeros(n) if delta is None else np.asarray(delta, float)
    gamma = np.zeros(n) if gamma is None else np.asarray(gamma, float)
    default_order, info = cylinder_order([a.get('pos') for a in atoms])
    if order is None:
        order = default_order
    else:
        if (len(order) != n or any(isinstance(i,(bool,np.bool_)) or not isinstance(i,(int,np.integer)) for i in order)
                or sorted(order) != list(range(n))):
            raise ValueError('order 必须是原子下标的完整整数排列')
        order = list(map(int,order))
        info = dict(info, perm={a:k for k,a in enumerate(order)}, ordering='explicit')
    perm = info['perm']
    # The imaginary onsite part is used by the MPS time evolvers for the same
    # effective no-jump loss convention as engine_ed.  DMRG callers leave
    # gamma=None and therefore retain the real Hamiltonian exactly.
    onsite = [(complex(delta[order[k]], -0.5 * gamma[order[k]]),
               complex(Om[order[k]]) / 2.0) for k in range(n)]
    couplings = []

    def two_site(coeff, s1, op1, s2, op2):
        if coeff == 0:
            return
        if s1 <= s2:
            couplings.append((complex(coeff), int(s1), op1, int(s2), op2))
        else:
            couplings.append((complex(coeff), int(s2), op2, int(s1), op1))

    for p in calib.get('pairs') or []:
        a, b = perm[int(p['i'])], perm[int(p['j'])]
        if a == b:
            continue
        if 'I1' in use:
            two_site(p['V'], a, 'Nhat', b, 'Nhat')
        if 'I2' in use:
            two_site(p['J'], a, 'P', b, 'M')
            two_site(np.conj(p['J']), a, 'M', b, 'P')
        if 'I3' in use:
            two_site(p['A'], a, 'P', b, 'P')
            two_site(np.conj(p['A']), a, 'M', b, 'M')
    return order, onsite, couplings, info


def rydberg_cluster_model(onsite, couplings, conserve=None, extra_onsite=None, *, mps_dtype='complex'):
    """任意点云的 Rydberg MPO 模型（自写 autoMPO：逐位 δ/Ω + 逐键复耦合 + 任意程）。

    onsite: [(δ_i, Ω_i/2), …] 按 MPS 顺序；couplings: [(coeff, a, opA, b, opB), …]（a<b）。

    手法：`Chain(L=n)`（单胞只有 1 格点）+ **strength 传数组**：
      · 逐位 δ/Ω → 长度 n 的数组；
      · 同一 (opA, opB, 程 m=b−a) 的一批键 → 长度 n−m 的数组（下标=左端位）。
    ✅ 不用“N 格点单胞”：那会让 TeNPy 把整条链当成一个环，n≥~18 就被
    `max_N_sites_per_ring` 守卫拦下（实测 n=20 直接报错）；本写法无此限且处同精度。
    """
    _require_tenpy()
    from tenpy.models.model import CouplingMPOModel
    from tenpy.models.lattice import Chain
    from tenpy.networks.site import SpinHalfSite

    n = len(onsite)
    if n == 0:
        raise ValueError('rydberg_cluster_model：没有格点')

    # 与引擎逐字同的单格点算符（up=g=[1,0], down=r=[0,1]）
    site = SpinHalfSite(conserve=conserve)
    site.add_op('Nhat', np.array([[0, 0], [0, 1]], dtype=complex))
    site.add_op('P', np.array([[0, 0], [1, 0]], dtype=complex))
    site.add_op('M', np.array([[0, 1], [0, 0]], dtype=complex))
    site.add_op('SZ', np.array([[1, 0], [0, -1]], dtype=complex))
    if conserve is None:
        site.add_op('SX', np.array([[0, 1], [1, 0]], dtype=complex))
        site.add_op('SY', np.array([[0, -1j], [1j, 0]], dtype=complex))
    valid = ('Nhat', 'P', 'M', 'SX')

    delta = np.array([complex(d) for (d, _o) in onsite], dtype=complex)
    # TeNPy drops an exactly zero MPO and cannot construct its terminal leg; retain
    # a numerically null diagonal anchor for the legitimate H=0 model.
    if not np.any(delta) and not np.any([complex(o) for (_d,o) in onsite]):
        delta[0] = 1e-15
    om2 = np.array([complex(o) for (_d, o) in onsite], dtype=complex)
    groups = {}
    for (c, a, opA, b, opB) in couplings:
        assert opA in valid and opB in valid, (opA, opB)
        a, b = int(a), int(b)
        assert a < b, (a, b)                        # 调用方已按 MPS 下标归一
        m = b - a
        arr = groups.get((opA, opB, m))
        if arr is None:
            arr = np.zeros(n - m, dtype=complex)
            groups[(opA, opB, m)] = arr
        arr[a] += complex(c)

    class _RydbergCluster(CouplingMPOModel):
        def init_sites(self, params):
            return site

        def init_lattice(self, params):
            return Chain(L=n, site=site, bc='open', bc_MPS='finite')

        def init_terms(self, params):
            self.add_onsite(delta, 0, 'Nhat')
            for coeff, op in (extra_onsite or []):
                self.add_onsite(np.asarray(coeff, complex), 0, op)
            # Keep zero-valued P/M channels in the MPO graph.  TeNPy cannot infer the
            # terminal charge leg for a pure diagonal one-site model otherwise.
            self.add_onsite(om2, 0, 'P')
            self.add_onsite(np.conj(om2), 0, 'M')
            for (opA, opB, m), arr in groups.items():
                if np.any(arr):
                    self.add_coupling(arr, 0, opA, 0, opB, m)

    from .mps_dtype import select_mpo_dtype
    model = _RydbergCluster(dict())
    model.H_MPO = select_mpo_dtype(model.H_MPO, mps_dtype)
    return model


def _mpo_dense(M):
    from tenpy.algorithms.exact_diag import ExactDiag
    ed = ExactDiag(M)
    ed.build_full_H_from_mpo()
    fh = ed.full_H
    if hasattr(fh, 'to_ndarray'):
        H = fh.to_ndarray()
    elif hasattr(fh, 'todense'):
        H = np.asarray(fh.todense())
    else:
        H = np.asarray(fh)
    return np.asarray(H, dtype=complex)


def rydberg_cluster_full_hamiltonian(onsite, couplings):
    """→ (稠密 2^n H, n_sites)（MPS 顺序）——验「自写 MPO ≡ 引擎 H」。"""
    _require_tenpy()
    M = rydberg_cluster_model(onsite, couplings)
    return _mpo_dense(M), M.lat.N_sites


_UNSET = object()     # 区分“不传（用 TeNPy 严格默认）”与“显式 None（降级为警告）”


def dmrg_run(M, chi_max=200, truncerr=1e-10, max_sweeps=60, want_state=False,
             max_trunc_err=_UNSET, product_state=None, *, initial_state=None,
             mps_dtype='auto', progress_arr=None):
    """在任意 TeNPy 模型上跑 TwoSiteDMRG → dict(E0, chi, S_mid, truncerr, n_sites, [psi])。

    `max_trunc_err`：TeNPy 的可靠性守卫——它比对的是**全部 sweep 的最大**截断误差
    （`np.max(trunc_err_list)`）。不传 → 用严格默认（小 chi 做收敛扫描会直接
    `TenpyInconsistencyError`，不静默降级）；传 `None` → 降为警告（故意看低键维结果时用）；
    传数值 → 自定义阈值。无论哪种，实际误差均回在 `truncerr` 里。TeNPy 自己的
    建议就是：不信任单一阈值，拿另一个算法交叉验（本仓 `equibench_dmrg_cluster` 干的就是这个）。
    """
    from tenpy.networks.mps import MPS
    from tenpy.algorithms.dmrg import TwoSiteDMRGEngine
    n = M.lat.N_sites
    from .mps_dtype import select_mps_dtype
    from .runtime import check_cancel, report_progress
    check_cancel(progress_arr)
    if initial_state is None:
        psi = MPS.from_product_state(M.lat.mps_sites(),
                                     [0] * n if product_state is None else list(product_state),
                                     bc='finite', dtype=complex)
    else:
        if (initial_state.bc != 'finite' or initial_state.L != n or
                any(a.leg != b.leg for a,b in zip(initial_state.sites,M.lat.mps_sites()))):
            raise ValueError('Warm-start MPS physical bases/charges do not match model')
        psi = initial_state
    psi = select_mps_dtype(psi,mps_dtype,complex_required=np.issubdtype(M.H_MPO.dtype,np.complexfloating))
    opts = dict(max_sweeps=max_sweeps, min_sweeps=2, mixer=True,
                trunc_params=dict(chi_max=chi_max, trunc_cut=truncerr))
    if max_trunc_err is not _UNSET:
        opts['max_trunc_err'] = (None if max_trunc_err is None
                                 else float(max_trunc_err))
    class CancellableDMRG(TwoSiteDMRGEngine):
        def run_iteration(self):
            check_cancel(progress_arr)
            result = super().run_iteration()
            check_cancel(progress_arr)
            report_progress(progress_arr,min(.99,self.sweeps/max(2,max_sweeps+2)))
            return result
    eng = CancellableDMRG(psi, M, opts)
    E0, psi = eng.run()
    # Sweep-limited/mixer outputs can retain a noncanonical gauge. Measure a
    # canonical copy, preserving the physical norm and the optimizer's state.
    psi = psi.copy()
    psi.canonical_form(renormalize=False, cutoff=0.)
    cut = n // 2
    S_mid = (float(psi.entanglement_entropy(bonds=[cut])[0]) if n >= 2 else 0.0)
    try:
        terr = float(np.max(np.asarray(eng.trunc_err_list, dtype=float)))
    except Exception:
        terr = float('nan')
    out = dict(E0=float(E0), E0_per_site=float(E0) / n, n_sites=n,
               chi=int(max(psi.chi)) if len(psi.chi) else 1, S_mid=S_mid,
               truncerr=terr, entropy_cut=(cut if n >= 2 else None),
               error_report=dict(solver_converged=bool(eng.is_converged()),sweeps=int(eng.sweeps),
                   max_trunc_err=terr,final_trunc_err=float(eng.sweep_stats['max_trunc_err'][-1]),
                   tensor_dtype=str(psi.dtype),mpo_dtype=str(M.H_MPO.dtype),
                   canonical_measurement_copy=True))
    if want_state:
        out['psi'] = psi
    report_progress(progress_arr,1.)
    return out


def dmrg_rydberg_cluster(calib, Om=None, delta=None, chi_max=200,
                         truncerr=1e-10, max_sweeps=60, max_trunc_err=_UNSET,
                         charge_sector=None, return_state=False, *, psi0=None, order=None,
                         mps_dtype='auto', diagnostics=None, memory_budget_mb=None,
                         pauli_strings=None, progress_arr=None):
    """任意点云（多腿/无周期）的 Rydberg DMRG 基态 —— 直接吃引擎 calib。

    ``pauli_strings``：``[{'label','terms':[(原子下标, 轴),…]}]``（原子**原序**）→
    每条的期望值进 ``pauli_strings`` 列表。与 ``conserve`` 无关（走约化密度矩阵）。

    → dict(E0, n_atom(按**原子原序**), chi, S_mid, truncerr, order, n_legs,
           mpo_bond_dim, backend)
    """
    _require_tenpy()
    from .mps_dtype import validate_policy
    from .state import validate_mps_state, product_vectors
    from .mps_analysis import analyze_mps, partition_model_terms, variance_workspace_bytes, validate_diagnostics
    from .resources import budget_bytes
    from .runtime import progress_range
    validate_policy(mps_dtype);budget_bytes(memory_budget_mb)
    native = isinstance(psi0,dict) and psi0.get('kind')=='mps'
    initial = None
    if native:
        validate_mps_state(psi0,len(calib['atoms']))
        if order is None:order=list(psi0['order'])
        if charge_sector is None:charge_sector=psi0.get('charge_sector')
        elif psi0.get('charge_sector') != charge_sector:raise ValueError('Warm-start charge metadata mismatch')
        initial=psi0['psi']
    order, onsite, couplings, info = terms_from_calib(calib, Om, delta, order=order)
    if native and list(psi0['order'])!=order:raise ValueError('Warm-start order mismatch; no implicit reordering')
    n = len(order)
    diagnostic_options=validate_diagnostics(diagnostics,n)
    conserve = None
    product = None
    if charge_sector is not None:
        if any(abs(o) > 1e-14 for _d, o in onsite):
            raise ValueError('DMRG Sz 守恒扇区要求 Omega=0')
        if any(opA == 'P' and opB == 'P' or opA == 'M' and opB == 'M'
               for _c, _a, opA, _b, opB in couplings):
            raise ValueError('DMRG Sz 守恒扇区不支持 I3 双激发项')
        if isinstance(charge_sector, (bool, np.bool_)) or int(charge_sector) != charge_sector:
            raise ValueError('charge_sector 必须是整数')
        k = int(charge_sector)
        if not (0 <= k <= n):
            raise ValueError('charge_sector 越界')
        conserve = 'Sz'
        product = [1] * k + [0] * (n - k)
    if not native and not (psi0 is None or isinstance(psi0,str) and psi0=='ground'):
        vectors=product_vectors(psi0,n)
        if charge_sector is not None:
            if any(np.count_nonzero(v)!=1 for v in vectors) or abs(sum(abs(v[1])**2 for v in vectors)-charge_sector)>1e-8:
                raise ValueError('Product initial state must lie entirely in charge_sector')
        product=[vectors[i] for i in order]
    if native and charge_sector is not None and abs(np.real(np.sum(initial.expectation_value('Nhat')))-charge_sector)>1e-8:
        raise ValueError('Warm-start state occupation does not match charge metadata')
    M = rydberg_cluster_model(onsite, couplings, conserve=conserve,mps_dtype=mps_dtype)
    r = dmrg_run(M, chi_max, truncerr, max_sweeps, want_state=True,
                 max_trunc_err=max_trunc_err, product_state=product,initial_state=initial,
                 mps_dtype=mps_dtype,progress_arr=progress_range(progress_arr,0.,.8 if diagnostic_options else 1.))
    psi = r.pop('psi')
    n_mps = np.asarray(psi.expectation_value('Nhat'), float)      # MPS 顺序
    n = len(order)
    n_atom = np.zeros(n)
    for k, a in enumerate(order):                                 # 映回原子原序
        n_atom[int(a)] = n_mps[k]
    from .state import make_mps_state
    state=(make_mps_state(psi,order,charge_sector) if return_state or diagnostic_options else None)
    if charge_sector is not None and abs(n_mps.sum()-charge_sector)>1e-8:
        raise ValueError('DMRG initial/final occupation does not match charge_sector')
    analysis={}
    if diagnostic_options:
        parts=None
        if diagnostic_options.get('variance'):
            limit=512. if memory_budget_mb is None else memory_budget_mb
            groups=1
            while True:
                parts=partition_model_terms(onsite,couplings,conserve=conserve,groups=groups,mps_dtype=mps_dtype)
                estimate=max(variance_workspace_bytes(psi,a,b) for i,a in enumerate(parts) for b in parts[i:])
                if estimate<=budget_bytes(limit):break
                if groups>=n:raise ValueError('Exact variance cannot fit budget even with individual term groups')
                groups=min(n,groups*2)
        # variance/permutation/blocks 才需要 MPO 分析；只求 corr 时跳过 analyze_mps。
        if parts is not None or diagnostic_options.get('permutation') is not None \
                or diagnostic_options.get('blocks') is not None:
            analysis=analyze_mps(state,mpos=parts,permutation=diagnostic_options.get('permutation'),
                blocks=diagnostic_options.get('blocks'),memory_budget_mb=512. if memory_budget_mb is None else memory_budget_mb,
                progress_arr=progress_range(progress_arr,.8,1.))
        if diagnostic_options.get('corr'):
            # 连通占据关联 C_ij = ⟨n_i n_j⟩ − ⟨n_i⟩⟨n_j⟩。代价 O(N²) 次收缩，
            # 只有显式声明 mps_diagnostics={'corr': True} 才算（不默认跑）。
            cc = np.asarray(psi.correlation_function('Nhat', 'Nhat',
                            sites1=list(range(n)), sites2=list(range(n)),
                            hermitian=True), dtype=complex).real
            corr_mps = cc - np.outer(n_mps, n_mps)
            corr = np.zeros((n, n), float)
            corr[np.ix_(order, order)] = corr_mps        # MPS 顺序 → 原子原序
            analysis['corr'] = corr
    pauli_out = {}
    if pauli_strings:
        # 任意阶 Pauli 字符串：走**约化密度矩阵**而不是 expectation_value_term，
        # 因为 conserve='Sz' 的格点上 σˣ/σʸ 不注册（只有 Sp/Sm/Sz/Id），
        # 而那正是含 U(1) 时的默认路径——不绕开它，手性类算符根本算不了。
        # ⚠ 站点下标是**原子原序**（与 n_atom 同口径），内部按 order 映射到 MPS 序。
        from .observables import measure_pauli_strings, pauli_specs_to_mps_order
        mapped = pauli_specs_to_mps_order(pauli_strings, order)
        vals = measure_pauli_strings(psi, mapped)
        pauli_out = {'pauli_strings': [complex(v) for v in vals],
                     'pauli_string_labels': [d['label'] for d in pauli_strings]}
    return dict(E0=r['E0'], n_atom=n_atom, chi=r['chi'], S_mid=r['S_mid'],
                truncerr=r['truncerr'],
                entropy_cut=r['entropy_cut'],
                **({'state': state} if return_state else {}),
                **({'corr': analysis['corr']} if 'corr' in analysis else {}),
                error_report=r['error_report'],charge_sector=charge_sector,
                used_options=dict(mps_dtype=mps_dtype,memory_budget_mb=memory_budget_mb,diagnostics=diagnostics),
                **({'diagnostics':analysis} if analysis else {}),
                **{k:analysis[k] for k in ['energy_variance','permutation_overlap'] if k in analysis},
                **pauli_out,
                n_sites=n, order=order, n_legs=info['n_legs'],
                leg_len=info['leg_len'],
                mpo_bond_dim=max(int(W.shape[1]) for W in M.H_MPO._W),
                backend='tenpy-dmrg-cluster')
