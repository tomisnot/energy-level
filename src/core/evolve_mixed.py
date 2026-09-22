# -*- coding: utf-8 -*-
"""混合引擎（单体 + 两体）：光场 Sambe 光子块（免 RWA）+ 微波 cos 硬积分。

φ-帧变换 φ = e^{iC·D_full·t}·X（D_full = 光子块全对角 = E − n·ω）把 THz 对角
吸进全局相位，剩下的项都可控（光场共振对慢；微波同扇区 cos 干涉）→ **单
DOP853 直接积分**即可，无需 Strang 分裂 + eigh 冻结（那会引入结构误差）。

    dφ/dt = −iC·[Σ_光场 env·Kc∘e^{iC·ΔD·t} + Σ_微波 env·cos(ωt)·A∘e^{iC·ΔD·t}]·φ

- 光场对：`BW_OPT` 过滤反旋（±2ω 振幅可弃）
- 微波对：成本窗 `mw_cap_mhz(T)` + 精度判据 Θ≥`PHASE_TOL`
- 线性 RHS 双向（Hermitian 天然，无 K 构造双计问题）
- 全程 φ-帧连续积分，输出时 X = e^{−iC·D·t}·φ 投影
- **无光场合法**：光子维退化 P=1 → 等价于实验室系微波/V_dd 当场精确积分

理论推导见 `docs/混合引擎理论推导-光Sambe微波当场.md`，
判据与历史动因见 `docs/参数与判据决策史.md`。

判据白盒接口（内部件，**不是**产品 API；改名/改签名要同步 checks/）：
  · `_MixedSetup`        —— mixed 单体 setup（`mw_pair_keep_selfcheck`、
                            `params_registry_selfcheck` 构造它验 T 自适应/成本上限）
  · `_assemble_H_snapshot` —— H 快照（`ai_h_snapshot_guard_selfcheck` 验尺寸闸）
"""
import numpy as np

from .evolve_util import (as_vector, _single_loss_rates, _loss_and_norm,
                          pair_psi0_vector, split_field_loss, CONV)
from . import tensor as _tensor
from .diag import h_snapshot_ok as _h_snap_ok
from .diag import h_snapshot_note as _h_snap_note
from .coupling import (rhs_phi_pairs, pack_rhs, stage_progress,
                       static_phase_keep, mw_pair_keep, mw_cap_mhz,
                       dop853_driver_mixed)  # 薄包装：让磁盘缓存命中
from .field_build import group_by_freq, feature_max_step
from .sambe import (_SambeSetup, _m_analytic, embed,
                    sector_sum_pops, sector_sum_amp)

from .static_atom import MW_MAX_MHZ
from .coupling import PHASE_TOL, R_CRIT, _RHS_ENV_TAB_N
from .param import Param   # 就地参数声明（值+元数据同处；mecha.registry 扫出全表）


# ==== 本模块的参数（就地定义；Param 声明值+元数据同处，透明子类读点免改）====
ATOL_RTOL_RATIO = Param(1e-2, 'atol/rtol 比（派生 DEFAULT_ATOL 用；非独立旋钮）', tier='internal')
BW_OPT = Param(2e4, '光场对过滤带宽（mixed）', unit='MHz', tier='run')
DEFAULT_RTOL = Param(1e-8, '积分相对容差（mixed 入口默认 rtol）', tier='run')
DEFAULT_ATOL = Param(DEFAULT_RTOL * ATOL_RTOL_RATIO, '积分绝对容差（派生 = DEFAULT_RTOL × ATOL_RTOL_RATIO，非独立旋钮）', tier='internal')
M_EST_TOL = Param(1e-6, '光子块截断容差', tier='run')
_AC_ENV_TAB_N = Param(4001, 'mixed AC 表点数；AC 时变对角的独立包络表点数（AC 只跟随慢变光强）', tier='internal')
# ---- AC 化层（唯一消费者家族就是 mixed）----


def classify_pairs(A, omega, r_crit=None):
    """按 r=Ω/ω 分类：True = 保留（近/弱对——光子块动态）；False = AC 化。"""
    r_crit = R_CRIT if r_crit is None else r_crit
    A = np.asarray(A, float)
    r = np.abs(A) / omega
    keep = (r <= r_crit) & (np.abs(A) > 1e-9)
    return keep


def far_sos_pair_shift(A_ij, E_k, E_i, omega):
    """单对 (i,k) 对源态 i 的 A² 等价 far_sos 位移（MHz）。

    = −A_ij²·(E_k−E_i)/(2·((E_k−E_i)²−ω²))——与
    static_atom.far_sos_shift_mhz（d²·w/(w²−ω²) 结构）同语义；A² ∝ d²·E0²
    的 Rabi 形式。t60 仲裁：±18.23 与 M=4 锚一致（C1≈A 6.87e-3）。
    """
    nu = float(E_k) - float(E_i)
    return -(float(A_ij) ** 2) * nu / (nu ** 2 - omega ** 2) / 2.0


def acize_physical(E, fields_opt, r_crit=None):
    """远强对 AC 化（唯一实现）：剔除 + per-field 位移。

    fields_opt: kind='exact' 光场列表（A 全偶极矩阵 + omega_L_mhz）。
    返回 (A_new_list, shifts_list)：
      A_new_list[f]：剔除远强对后的光场 A（近对保留）
      shifts_list[f]：第 f 个光场的 N 维位移向量（env² 全量时变语义——
        K0 用纯 E；RHS 时变对角 Σ_f shifts_f·env_f(t)² 由调用侧组装）
    """
    r_crit = R_CRIT if r_crit is None else r_crit
    N = len(E)
    shifts_list = []
    A_new = []
    for f in fields_opt:
        A = np.asarray(f['A'], float).copy()
        w = float(f['omega_L_mhz'])
        sh_f = np.zeros(N)
        keep = classify_pairs(A, w, r_crit)
        for i in range(N):
            for k in range(i + 1, N):
                if not keep[i, k]:
                    if abs(A[i, k]) < 1e-9:
                        continue
                    dE = far_sos_pair_shift(A[i, k], E[k], E[i], w)
                    sh_f[i] += dE
                    sh_f[k] -= dE
                    A[i, k] = A[k, i] = 0.0
        A_new.append(A)
        shifts_list.append(sh_f)
    return A_new, shifts_list
def ac_ize_physical_E(E, f_opt, r_crit=None):
    """远强对 AC 化入口（唯一消费者家族是 mixed）。"""
    r_crit = R_CRIT if r_crit is None else r_crit
    A_new, shifts_list = acize_physical(E, f_opt, r_crit)
    return E, A_new, shifts_list




def _assemble_H_snapshot(D_full, pairs, t, C, ac_base=None, ac_waves=None,
                         gamma=None, tags=None):
    """t75 诊断：重组 t 时刻的扩展空间哈密顿 H（MHz，数值显示用）。

    H(t) = diag(K0) + Σ_光场 env·Kc∘e^{iCΔD·t} + Σ_微波 env·cos(Cωt)·A∘e^{iCΔD·t}
           + AC 对角 env² + 耗散 −iΓ/2——与 rhs_phi_pairs 逐项同语义
    （v 已含光子块 ½ 约定；不在 njit 内核里插桩，事后重建即交叉验证）。
    tags：与 pairs 等长的分项名列表（两体用，区分 V_dd/光场/微波）；
    None 时按 is_mw 二分。返回分项 dict（键按实际内容懒建）。
    """
    t = float(t)
    dim = np.asarray(D_full).size
    # 尺寸保护：策略的唯一所有者在 core/diag.py（阈值/判定/说明同源）
    if not _h_snap_ok(dim):
        return _h_snap_note(dim)
    terms = dict(
        K0=np.diag(np.asarray(D_full, complex)),
        AC对角=np.zeros((dim, dim), complex),
        耗散=np.zeros((dim, dim), complex),
    )
    for k, p in enumerate(pairs):
        a, b, v, dD, is_mw, w, env = p[:7]
        e = float(np.atleast_1d(env(np.array([t])))[0])
        if e == 0.0:
            continue
        ph = np.exp(1j * C * dD * t)
        cw = e * (np.cos(C * w * t) if is_mw else 1.0)
        name = tags[k] if tags is not None else ('微波' if is_mw else '光场')
        if name not in terms:
            terms[name] = np.zeros((dim, dim), complex)
        val = cw * v * ph
        terms[name][a, b] += val
        terms[name][b, a] += np.conj(val)
    if ac_base is not None and ac_waves:
        for wi, (sh, wf) in enumerate(ac_waves):
            dev = float(np.atleast_1d(wf(np.array([t])))[0]) ** 2
            if dev != 0.0:
                terms['AC对角'] += np.diag(dev * np.asarray(ac_base[wi], float))
    if gamma is not None:
        g = np.asarray(gamma, float)
        if np.any(g):
            terms['耗散'] = np.diag(-0.5j * g)
    return terms


def _fmt_budget(v):
    """成本预算的日志显示（None/空 = 不限制）。"""
    return "不限制" if v is None or float(v) <= 0.0 else f"{float(v):g} rad"


def _cost_report(setup, states, T, phase_budget, alts=(1e4, 4e4)):
    """**运行前成本报告**（不积分）：保留集多大 / 最远失谐 / 预计步数，
    以及"若改上某个成本档"会删掉什么、能快多少。

    为什么要它：成本只该"换速度"，不该默默改模型。默认不限制 → 保留全部
    真实耦合，但用户有权知道这要付多少，并**在运行前**决定要不要省。
    步数标定沿用 t80 的锚：最坏总步数 ≈ C·Δ_max·T/0.535。
    """
    Cv = CONV
    mw = [p for p in setup.pairs if int(p[4]) == 1]
    dets = [abs(abs(float(p[3])) - float(p[5])) for p in mw]
    dmax = max(dets) if dets else 0.0
    rep = {
        'ok': True, 'T_ns': float(T), 'budget': phase_budget,
        'n_pairs': len(setup.pairs), 'n_mw': len(mw),
        'max_det_mhz': dmax,
        'mw_cap_mhz': float(setup.mw_cap),
        'est_steps': (Cv * dmax * float(T) / 0.535) if dmax else 0.0,
        'n_drop_phase': int(setup.mw_pair_stats.get('drop_phase', 0)),
        'alt': {},
    }
    if dets:
        i = int(np.argmax(dets))
        a, b = int(mw[i][0]), int(mw[i][1])
        rep['worst'] = {
            'a': a, 'b': b,
            'label_a': states[a].get('label', '') if a < len(states) else '',
            'label_b': states[b].get('label', '') if b < len(states) else '',
            'abs_v_mhz': float(abs(complex(mw[i][2]))),
            'det_mhz': float(dets[i]),
        }
    for b in alts:
        cap = float(b) / (Cv * max(float(T), 1e-12))
        keep = [d for d in dets if d <= cap]
        nd = len(dets) - len(keep)
        dnew = max(keep) if keep else 0.0
        rep['alt'][b] = {
            'cap_mhz': cap, 'n_drop': nd, 'max_det_mhz': dnew,
            'est_steps': (Cv * dnew * float(T) / 0.535) if dnew else 0.0,
            'speedup': (dmax / dnew) if dnew > 0 else float('inf'),
        }
    return rep


class _MixedSetup:
    """混合引擎预计算：光场光子块 + φ-帧耦合对统一打包。"""

    def __init__(self, E, grps_opt, grps_mw, N, M_override=None,
                 bw_opt=None, mw_cap=None, mw_arb=None, t_end_ns=None,
                 phase_budget=None):
        self.N = N
        self.E = np.asarray(E, float)
        self.grps_opt = grps_opt
        self.grps_mw = grps_mw
        # t75：M 覆写（上帝模式自定义）——int 全组 / list 逐组；None=自动探针
        if not grps_opt:
            Ms = []                 # 无光场：光子维退化，无组可覆写（t77）
        elif M_override is None:
            Ms = _m_analytic(grps_opt, M_EST_TOL, self.E)
        elif np.isscalar(M_override):
            Ms = [int(M_override)] * len(grps_opt)
        else:
            Ms = [int(m) for m in M_override]
            if len(Ms) != len(grps_opt):
                raise ValueError(
                    f'per_freq_M 长度 {len(Ms)} != 载频组数 {len(grps_opt)}')
        self.M_used = list(Ms)
        self.T = float(t_end_ns) if t_end_ns else None
        # t75/t80：过滤带宽覆写——None = 用本模块顶部的 BW_OPT
        self.bw = float(bw_opt) if bw_opt is not None else BW_OPT
        # 微波候选窗 = 相预算派生（mw_cap_mhz(T)），再夹绝对上界 MW_MAX_MHZ
        # —— 与 `build_two_field` 的 `min(mw_cap_mhz, MW_MAX_MHZ)` 同构，
        # 保证两引擎保留集一致、且 MW_MAX_MHZ 这个旋钮在任何路径都真的生效。
        # 显式 mw_cap 覆写 = 绝对 MHz；无 T（脚本直调）退回绝对上界。
        if mw_cap is not None:
            self.mw_cap = float(mw_cap)
        elif self.T is not None:
            self.mw_cap = min(mw_cap_mhz(self.T, phase_budget),
                              MW_MAX_MHZ)
        else:
            self.mw_cap = MW_MAX_MHZ
        self.mw_pair_stats = {'keep': 0, 'drop_cap': 0, 'drop_phase': 0,
                               'drop_cost_loud': 0, 'loud_max_om': 0.0,
                               'loud_max_det': 0.0}
        self.setup = _SambeSetup(self.E, grps_opt, Ms)
        self.P = self.setup.P
        self.dim = self.setup.dim
        self.D_full = np.real(np.asarray(self.setup.diag))  # t73: K0→diag

        # 波清单（光场在前 + 微波）——log / AC 映射用
        self._env_list = []
        self._is_mw = []
        self._wcos = []
        for gi, g in enumerate(grps_opt):
            for wf in g['envs']:
                self._env_list.append(wf)
                self._is_mw.append(0)
                self._wcos.append(0.0)
        for g in grps_mw:
            w = float(g['omega_mhz'])
            for A, env in zip(g['As'], g['envs']):
                self._env_list.append(env)
                self._is_mw.append(1)
                self._wcos.append(w)
        for g in (mw_arb or []):          # 任意波形微波（t79）：无单一载频
            for A, env in zip(g['As'], g['envs']):
                self._env_list.append(env)
                self._is_mw.append(1)
                self._wcos.append(0.0)    # 载波由 ph_fn 查表承担
        # 耦合对 7 元组（a<b 单侧——反向由共享内核厄米补齐，与两体同契约；
        # t70 三角过滤：Kcw 为 Hermitian 全矩阵，双向打包会把同一耦合驱动两次）
        self.pairs = []
        # 光场对因 BW_OPT 被丢弃的记账（2026-09-22，手册 §8.3）：原先**完全静默**
        # （同族的成本窗剔除在两体侧有记账 + diag + 告警，单体侧什么都没有）。
        self.bw_drop_stats = {'n': 0, 'max_abs_v_mhz': 0.0}
        for gi, g in enumerate(grps_opt):
            Kcw_list = self.setup.kc_lists(gi)   # 唯一入口（稠密缺失→同源稀疏）
            for wi_local, wf in enumerate(g['envs']):
                if wi_local >= len(Kcw_list):
                    continue            # AC 化后零耦合波（t62e：仅占波行）
                Kcw = self.setup.densify(Kcw_list[wi_local])
                ii, jj = np.nonzero(np.abs(Kcw) > 1e-12)
                for a, b in zip(ii, jj):
                    if a >= b:
                        continue
                    if abs(self.D_full[a] - self.D_full[b]) < self.bw:
                        self.pairs.append(
                            (int(a), int(b), Kcw[a, b],
                             float(self.D_full[a] - self.D_full[b]),
                             0, 0.0, wf))
                    else:
                        # BW_OPT 名义是"双峰分离器"（共振对 ~0 vs 反旋 ~2ω），但它
                        # 同时会删掉任何落在 bw~2ω 之间的光场对 ⇒ 改了模型必须交代
                        self.bw_drop_stats['n'] += 1
                        self.bw_drop_stats['max_abs_v_mhz'] = max(
                            self.bw_drop_stats['max_abs_v_mhz'],
                            abs(complex(Kcw[a, b])))
        for g in grps_mw:
            w = float(g['omega_mhz'])
            for A, env in zip(g['As'], g['envs']):
                A = np.asarray(A, float)
                for sa in range(self.P):
                    for i in range(N):
                        for k in range(i + 1, N):
                            om = abs(A[i, k])
                            if om < 1e-12:
                                continue
                            a = sa * N + i
                            b = sa * N + k
                            dE = float(self.D_full[a] - self.D_full[b])
                            det = abs(abs(dE) - w)
                            # t79/t80：主判据 = 累积相位 Θ=C·T·Ω²/max(Δ,Ω)（与 V_dd
                            # 同一判据，随 Ω² 自适应：强场保留远失谐多光子通道）；
                            # mw_cap 是**成本预算**（相预算派生，t80）。无 T 退回上限判。
                            if det > self.mw_cap:
                                self.mw_pair_stats['drop_cap'] += 1
                                # t83：**成本剔除但要被大声说出来**。相预算窗宽
                                # = PHASE_BUDGET/(C·T) 随 T 收窄 ⇒ 只是把演化窗口
                                # 拉长，就可能把远失谐梯级悄悄删掉、物理变成 0
                                # （实测：freeze 三光子路径 T=400 ns 有 0.87 布居，
                                # T=1200 ns 变 0.0000——第三级 Δ=7.08 GHz 被剔）。
                                # 故此处在"Θ 判据本来会保留"时记一条吵闹统计，
                                # 让日志/诊断能立刻回答"路径为什么消失"。
                                if self.T is not None:
                                    _th = CONV * self.T * om ** 2 / max(det, om)
                                    if _th >= PHASE_TOL:
                                        self.mw_pair_stats['drop_cost_loud'] += 1
                                        if om > self.mw_pair_stats['loud_max_om']:
                                            self.mw_pair_stats['loud_max_om'] = om
                                            self.mw_pair_stats['loud_max_det'] = det
                                continue
                            if self.T is not None and not mw_pair_keep(
                                    om, det, self.T):
                                self.mw_pair_stats['drop_phase'] += 1
                                continue
                            self.mw_pair_stats['keep'] += 1
                            self.pairs.append((a, b, complex(A[i, k]), dE,
                                               1, w, env))
        # 任意波形微波（t79）：8 元组带 ph_fn → 内核走 cos(φ(t)) 查表。
        # 选对判据不能用单一 w（信号无单载频），改用**信号瞬时频率带宽**
        # [band0, band1]（arbitrary_mw_group 给出）+ 同一成本预算 mw_cap 容差。
        # 为什么不照 hard 那样全收：mixed 的 φ-帧里光学对会带 THz 相位把
        # 积分器钉死，且与光场光子块双重计入；hard 无光子块故可全收。
        for g in (mw_arb or []):
            ph_fn = g.get('phase_fn')
            band = g.get('band_mhz')
            for A, env in zip(g['As'], g['envs']):
                A = np.asarray(A, float)
                for sa in range(self.P):
                    for i in range(N):
                        for k in range(i + 1, N):
                            if abs(A[i, k]) < 1e-12:
                                continue
                            a = sa * N + i
                            b = sa * N + k
                            dE = float(self.D_full[a] - self.D_full[b])
                            if band is not None:
                                ad = abs(dE)
                                if not (band[0] - self.mw_cap
                                        <= ad <= band[1] + self.mw_cap):
                                    continue
                            self.pairs.append((a, b, complex(A[i, k]), dE,
                                               1, 0.0, env, ph_fn))
        # 远对净 AC 位移一律由 far_sos 注入物理能级，此处不建 eigh 位移块
        self._is_mw = np.asarray(self._is_mw, np.int64)
        self._wcos = np.asarray(self._wcos, np.float64)

    def pack(self, T, n_tab=None):
        """pairs → coupling.pack_rhs 数组（t73：与两体 mixed 同一打包层）。"""
        n_tab = _RHS_ENV_TAB_N if n_tab is None else int(n_tab)
        self.pack = pack_rhs(self.pairs, T, n_tab=n_tab)
        return self.pack


def evolve_mixed(states, fields, psi0, t_grid, tol=None,
                 max_step=0.05, max_step_hard=None,
                 progress_arr=None, log=None, record_amp=False,
                 driver='numba', decays=None,
                 per_freq_M=None, r_crit=None, diag=False,
                 bw_opt=None, mw_cap=None, mw_max=None, mw_arb=None,
                 phase_budget=None, dry_run=False, field_loss=None):
    """混合引擎主入口：φ-帧单 DOP853（光场 Sambe 结构 + 微波 cos 硬积分）。

    fields : kind='exact' 场列表（build_fields_exact 输出；光场 + 微波混合）。
             **无光场合法**（纯微波，或光波被 AC 化/选择定则全部剔除）：
             光子维退化 P=1，自动等价于实验室系微波当场精确积分（t77）。
    tol    : DOP853 相对容差
    max_step : DOP853 最大步长（ns；微波 17GHz 载波自动限步，0.05 保守）
    driver : 'numba'（dop853 核心层）| 'scipy'（对照）
    decays : t62 耗散——寿命通道列表（落点=丢失，NonHermitian 对角；
            与 hard 引擎同语义，_single_loss_rates；显式落点/纯退相拒绝）
    per_freq_M : t75 上帝模式覆写——逐载频光子块阶数（int 全组/list 逐组；
            None=自动探针 _m_analytic）
    r_crit : t75 上帝模式覆写——AC 化 r=Ω/ω 阈值（None=默认 R_CRIT）
    diag   : t75 诊断快照——True 时 out['diag'] 附结构化中间产物
            （能级/AC 化/光子块几何/耦合对/H(t) 快照/步长历史/扇区轨迹）
    bw_opt / mw_cap / mw_max : t75/t80 上帝模式覆写——φ-帧光场过滤带宽 /
            微波候选窗**绝对**上界（MHz）/ 微波-光场分界（None=默认）
    phase_budget : t80 上帝模式覆写——微波候选窗的**无量纲相预算**（rad），
            默认 params.PHASE_BUDGET（微波候选窗的主旋钮）
    """
    tol = DEFAULT_RTOL if tol is None else tol
    log = log or (lambda *a: None)
    t = np.asarray(t_grid, float)
    if progress_arr is None:
        progress_arr = np.zeros(3)
    stage_progress(progress_arr, 0, 0.0)     # 阶段 0：装配/JIT 装载
    N = len(states)
    E = np.array([s['E_mhz'] for s in states], float)
    E = E - float(E.min())
    psi0v = as_vector(states, psi0)
    C = CONV

    # t62 耗散（落点=丢失）：Gamma_ns（/ns）→ gamma（MHz）传给内核；
    # Gamma_ns 收尾的 loss 曲线复用，不重复计算（t77 清理）
    gamma = None
    Gamma_ns = np.zeros(N)
    if decays:
        Gamma_ns = _single_loss_rates(states, decays)
        if np.any(Gamma_ns):
            gamma = Gamma_ns / CONV
            log(f'[mixed] 耗散 {int((Gamma_ns > 0).sum())} 态'
                f'（落点=丢失，NonHermitian）')
    mm = float(mw_max) if mw_max is not None else MW_MAX_MHZ
    f_opt = [f for f in fields if f['omega_L_mhz'] > mm]
    f_mw = [f for f in fields if f['omega_L_mhz'] <= mm]
    # t62 AC 化（正确组合）+ t62e 时变化：远强对剔除 + per-field far_sos 位移
    # t75：r_crit 可由上帝模式覆写（None=R_CRIT 默认）
    rc = float(r_crit) if r_crit is not None else R_CRIT
    E2, A_opt_new, shifts_list = ac_ize_physical_E(E, f_opt, rc)
    f_opt_ac = [dict(f, A=A) for f, A in zip(f_opt, A_opt_new)]
    grps_opt = group_by_freq(f_opt_ac)
    grps_mw = group_by_freq(f_mw) if f_mw else []
    if not grps_opt:
        # 无光场（纯微波，或光波被 AC 化/偏振选择定则全部剔除）——光子维
        # 退化 P=1：φ-帧回到实验室系，微波/V_dd 当场精确积分（t77）
        log('[mixed] 无光场 → 光子维退化 P=1，等价于实验室系'
            ' 微波/V_dd 当场精确积分（与 hard 微波路径数值等价）')
    # t79/t80：微波对判据需要演化窗口 T（精度 Θ=C·T·Ω²/max(Δ,Ω) 与成本
    # 预算 mw_cap_mhz(T) 都随 T 变），故先取 T 再建 setup。
    T = float(t[-1])
    setup = _MixedSetup(E2, grps_opt, grps_mw, N, M_override=per_freq_M,
                        bw_opt=bw_opt, mw_cap=mw_cap, mw_arb=mw_arb,
                        t_end_ns=T, phase_budget=phase_budget)
    # t62e/t73：AC 时变对角——per-wave (n_ac, dim) + 独立 env 表（与两体
    # mixed 同款；按波形对象同一性映射）
    ac_waves = []
    for gi, g in enumerate(grps_opt):
        for wf in g['envs']:
            for f_idx, f in enumerate(f_opt):
                if f['waveform'] is wf and np.any(shifts_list[f_idx]):
                    ac_waves.append((shifts_list[f_idx], wf))
    # 耗散 gamma（N）→ 扩展空间（N·P，每扇区同衰减）
    if gamma is not None:
        gamma = np.tile(gamma, setup.P)
    E_shift = E2 - E
    n_ac = int(np.count_nonzero(np.abs(E_shift) > 1e-9))
    log(f'[mixed] AC 化位移范围 {E_shift.min():+.1f}..{E_shift.max():+.1f} MHz'
        f'（{n_ac} 态）；P={setup.P} dim={setup.dim}')
    # （T 已在建 setup 前取得——微波对判据需要它）
    ac_base = None
    ac_env_tab = None
    ac_env_dt = 1.0
    if ac_waves:
        n_acv = len(ac_waves)
        n_ac_tab = _AC_ENV_TAB_N
        ac_base = np.zeros((n_acv, setup.dim), float)
        ac_env_tab = np.zeros((n_acv, n_ac_tab), float)
        ts_ac = np.linspace(0.0, T, n_ac_tab)
        for wi, (sh_f, wf) in enumerate(ac_waves):
            for i in range(N):
                v = sh_f[i]
                if v != 0.0:
                    for pn in range(setup.P):
                        ac_base[wi, pn * N + i] = v
            ac_env_tab[wi] = np.asarray(wf(ts_ac), float)
        ac_env_dt = T / (n_ac_tab - 1)
    # 场致损耗（opt-in）：逐波 env²×逐态时变对角，与 ac 块同构（实衰减而非相位）。
    # loss_base 以 /ns 存（out_mhz×1e-3），跨光子扇区平铺；field_loss=None → 不建。
    loss_base = None
    loss_env_tab = None
    loss_env_dt = 1.0
    if field_loss is not None and getattr(field_loss, 'channels', None):
        _ls = [ch for ch in field_loss.channels if float(ch['out_mhz']) > 0.0]
        if _ls:
            n_ls = len(_ls)
            n_ls_tab = _AC_ENV_TAB_N
            loss_base = np.zeros((n_ls, setup.dim), float)
            loss_env_tab = np.zeros((n_ls, n_ls_tab), float)
            ts_ls = np.linspace(0.0, T, n_ls_tab)
            for wi, ch in enumerate(_ls):
                rate_ns = float(ch['out_mhz']) * 1e-3        # MHz → /ns（env=1 峰值）
                src = int(ch['src'])
                for pn in range(setup.P):
                    loss_base[wi, pn * N + src] = rate_ns
                loss_env_tab[wi] = np.asarray(ch['env_fn'](ts_ls), float)
            loss_env_dt = T / (n_ls_tab - 1)
            log(f'[mixed] 场致损耗 {n_ls} 通道（env²×逐态，NonHermitian）')
    pack = setup.pack(T)
    _st = setup.mw_pair_stats
    log(f'[mixed] 波 {len(setup._wcos)}'        f'（mw {int(setup._is_mw.sum())}）；耦合对 {len(setup.pairs)}'
        f'（微波对：保留 {_st["keep"]}，'
        f'成本预算剔除 {_st["drop_cap"]}，累积相位剔除 {_st["drop_phase"]}）'
        f'；微波候选窗 {setup.mw_cap:.4g} MHz'
        f'（成本预算 {_fmt_budget(phase_budget)} / T={T:g} ns）')
    if len(setup.pairs) == 0 and int(setup._is_mw.sum()) > 0:
        # S4a（2026-09-12）：多光子通道由基组内**真实单光子矩阵元**串起来；
        # 缺中间态（如踢 60P→58G 却没有 59D/58F）时配对集为空，
        # 表现为"频率对、强度对，却一点都踢不动"，此前完全静默。
        log('⚠ [mixed] 微波波形在基组内找不到任何候选耦合对：'
            '多半是**基组缺中间态**（多光子通道需要真实中间态），'
            '或频率与任何能级间隔都不匹配 —— 该波不会产生任何转移。')
    if _st.get('drop_cost_loud'):
        # 成本剔除但 Θ 判据本来会保留 → 这是"拉长窗口反而丢路径"的典型信号，
        # 必须吵闹（否则只有"布居不动"这一个无从下手的现象）。
        log(f'[mixed] ⚠ 有 {_st["drop_cost_loud"]} 对**仅因成本窗被剔除**'
            f'（Θ 判据本会保留，最强一对 |Ω|={_st["loud_max_om"]:.3g} MHz /'
            f' Δ={_st["loud_max_det"]:.3g} MHz）。窗宽 = PHASE_BUDGET/(C·T)，'
            f'此处 T={T:g} ns 给出 {setup.mw_cap:.4g} MHz；'
            f'若该远失谐通道是物理的一部分（通常是），请把成本预算留空'
            f'（= 不限制，当前 {_fmt_budget(phase_budget)}）。'
            f'成本只该"换速度"，不该默默改模型。')

    if dry_run:
        # 运行前预检：装配（成本窗/保留集）到此为止，**不积分**。
        # GUI 用它给出"这次要付多少"的报告，由用户决定是否上成本档。
        return {'dry_run': True, 'method': 'mixed',
                'cost': _cost_report(setup, states, T, phase_budget)}

    X0 = embed(psi0v, setup.setup)
    prog = (np.zeros(3) if progress_arr is None
            else np.asarray(progress_arr, float))
    stat = np.zeros(6) if diag else np.zeros(4)
    dt_hist = (np.zeros((100000, 2)) if diag else None)
    stage_progress(prog, 0, 0.9)             # 装配完成（JIT 装载在驱动首调）

    if driver == 'numba':
        # 阶段 1（积分）由驱动在首个接受步置位——调用内的 numba 装载/
        # 编译期保持阶段 0（"准备"标签 + 耗时），GUI 显示不失真
        #
        # ⚠ 2026-09-22 修 C1b（手册 §8.1 同族，**默认引擎静默给错**）：
        # `max_step` 只在**起步**参与步长选择（dop853.py:239-243 自述：
        # `h_abs = min(100·h0, h1, interval, max_step)`，之后按误差控制自由
        # 增长），**不是上限**。而包络在脉冲前后**恰为 0**（|t−c| ≳ 19σ 时
        # exp 下溢到 0.0）⇒ 误差估计看到 0 误差 ⇒ 步长一路涨到 ~60 ns ⇒
        # **整个短脉冲被跨过去**。实测（σ=2 ns / T=620 ns / 共振 1000 MHz /
        # Ω=1 MHz）：默认 nfev=131（10 个接受步！）给 7.7e-94（解析 2.48e-4）；
        # σ=0.62 ns 给**恰好 0**。且**与 tol 无关、与 max_step 无关**（后者只管
        # 起步）——这正是"调 tol/调 max_step 都没反应"的原因。
        # 修法：未显式给 max_step_hard 时，由包络最窄特征时长导出每步硬上限
        # （`feature_max_step`，深混叠区才生效 ⇒ 宽脉冲场景 no-op）。
        _msh = max_step_hard
        if _msh is None:
            _msh = feature_max_step(getattr(setup, '_env_list', None), T)
        y_out = dop853_driver_mixed(X0, t, T, float(tol), DEFAULT_ATOL,
                                    max_step, prog, stat, dt_hist,
                                    float(_msh or 0.0),
                                    pack['iis'], pack['jjs'],
                                    pack['vals'], pack['dDs'], pack['is_mw'],
                                    pack['wcos'], pack['n_pairs_w'],
                                    pack['env_tab'], pack['env_dt'],
                                    pack['has_ph'], pack['ph_tab'], float(C),
                                    gamma, ac_base, ac_env_tab, ac_env_dt,
                                    loss_base, loss_env_tab, loss_env_dt)
    else:
        from scipy.integrate import solve_ivp
        stage_progress(prog, 1, 0.0)         # 阶段 1：积分（scipy 无内部钩子）

        def rhs(tv, phi):
            return rhs_phi_pairs(tv, phi, pack['iis'], pack['jjs'],
                                 pack['vals'], pack['dDs'], pack['is_mw'],
                                 pack['wcos'], pack['n_pairs_w'],
                                 pack['env_tab'], pack['env_dt'],
                                 pack['has_ph'], pack['ph_tab'], C,
                                 gamma, ac_base, ac_env_tab, ac_env_dt,
                                 loss_base, loss_env_tab, loss_env_dt)
        sol = solve_ivp(rhs, (0.0, T), X0, method='DOP853', t_eval=t,
                        rtol=tol, atol=DEFAULT_ATOL, max_step=max_step)
        if not sol.success:
            raise RuntimeError(f'混合引擎积分失败: {sol.message}')
        y_out = sol.y.T       # (len(t), dim) 统一布局

    # 输出：φ → X = e^{−iC·D·t}·φ → 投影物理布居
    # y_out: (len(t), dim)（numba driver 与 scipy 均此布局）
    stage_progress(prog, 2, 0.0)             # 阶段 2：收尾（φ→物理投影）
    pops = np.zeros((len(t), N))
    for i in range(len(t)):
        # t73b：相干扇和大（解析精确——光子相位恰对消；旧非相干和带
        # O(Ω/ω) 系统误差，22 态场景实测 ~4e-3）
        pops[i] = sector_sum_pops(y_out[i], setup.P, N)
        stage_progress(prog, 2, i / max(len(t) - 1, 1))
    stage_progress(prog, 2, 1.0)
    # t62 耗散：loss 曲线（梯形积分）+ 含 loss 范数偏差（同 hard）
    comp = None
    if field_loss is not None and getattr(field_loss, 'channels', None):
        comp = split_field_loss(t, pops, Gamma_ns, field_loss,
                                trap=getattr(field_loss, 'trap', None))
        loss = comp['spont'] + comp['scatter'] + comp['photoion']
        norm_dev = float(np.max(np.abs(pops.sum(axis=1) + loss - 1.0)))
    else:
        loss, norm_dev = _loss_and_norm(pops, Gamma_ns, t)
    # ---- t75 诊断快照（diag=True 才采集；数值路径零改动）----
    if diag:
        pr = setup.pairs
        diag_pairs = dict(
            a=np.array([p[0] for p in pr], int),
            b=np.array([p[1] for p in pr], int),
            v=np.array([p[2] for p in pr], complex),
            dD=np.array([p[3] for p in pr], float),
            is_mw=np.array([p[4] for p in pr], int),
            w=np.array([p[5] for p in pr], float))
        acize_fields = []
        for f0, f1, sh in zip(f_opt, A_opt_new, shifts_list):
            n0 = int((np.abs(np.asarray(f0['A'], float)) > 1e-9).sum() // 2)
            n1 = int((np.abs(np.asarray(f1, float)) > 1e-9).sum() // 2)
            acize_fields.append(dict(
                omega=float(f0['omega_L_mhz']), n_before=n0, n_kept=n1,
                n_acized=n0 - n1,
                shift_min=float(sh.min()), shift_max=float(sh.max())))
        idx = np.linspace(0, len(t) - 1, min(len(t), 240)).astype(int)
        sector_amps = {p: np.abs(y_out[idx][:, p * N:(p + 1) * N])
                       for p in range(setup.P)}
        ts_snap = sorted(set(float(x) for x in
                             np.linspace(0.0, T, 5)))
        snaps = ({t0: _assemble_H_snapshot(
                setup.D_full, pr, t0, C, ac_base=ac_base,
                ac_waves=ac_waves if ac_base is not None else None,
                gamma=gamma) for t0 in ts_snap}
        if _h_snap_ok(setup.dim) else
        {'skipped': _h_snap_note(setup.dim)})
        diag_dt = (dt_hist[:int(stat[4])] if dt_hist is not None
                   else np.zeros((0, 2)))
        # 2026-09-22（手册 §8.4）：`diag.summarize/warnings` 读的
        # `meta['keep'/'n_drop_cap'/'n_drop_phase']`、`pairs['n'/'max_abs_v_mhz']`、
        # `acize['n_shifted']` 此前**全仓无写入端** ⇒ 那些字段恒 None、"成本窗剔除
        # =改了模型"的告警**永不触发**（哑告警）。此处一并补齐。
        _vmax = (float(np.abs(diag_pairs['v']).max())
                 if diag_pairs['v'].size else 0.0)
        out_diag = dict(
            meta=dict(method='mixed', backend='mixed-phi-dop853', N=N,
                      P=setup.P, dim=setup.dim, t_end=T,
                      M=list(setup.M_used), M_user=(per_freq_M is not None),
                      r_crit=float(rc), n_pairs=len(pr), tol=float(tol),
                      max_step=float(max_step), bw=float(setup.bw),
                      mw_cap=float(setup.mw_cap), mw_max=float(mm),
                      keep=int(setup.mw_pair_stats.get('keep', 0)),
                      n_drop_cap=int(setup.mw_pair_stats.get('drop_cap', 0)),
                      n_drop_phase=int(setup.mw_pair_stats.get('drop_phase', 0)),
                      n_drop_bw=int(setup.bw_drop_stats.get('n', 0)),
                      max_drop_bw_abs_v_mhz=float(
                          setup.bw_drop_stats.get('max_abs_v_mhz', 0.0)),
                      max_abs_v_mhz=_vmax,
                      labels=[s['label'] for s in states], stride=N),
            photon=dict(Ms=list(setup.M_used), P=setup.P, dim=setup.dim,
                        strides=list(setup.setup.strides),
                        nvecs=np.asarray(setup.setup.nvecs),
                        D_full=np.asarray(setup.D_full, float).copy()),
            acize=dict(r_crit=float(rc), fields=acize_fields,
                       n_shifted=int(np.count_nonzero(
                           np.abs(np.asarray(E2 - E, float)) > 1e-9)),
                       shift=np.asarray(E2 - E, float).copy()),
            pairs=dict(diag_pairs, n=int(diag_pairs['v'].size),
                       max_abs_v_mhz=_vmax),
            solver=dict(t_steps=diag_dt[:, 0].copy(),
                        dt=diag_dt[:, 1].copy(),
                        nfev=int(stat[0]), n_acc=int(stat[1]),
                        n_rej=int(stat[2]), norm_dev=norm_dev),
            trajectory=dict(t=t[idx].copy(), sector_amps=sector_amps),
            H_snapshots=snaps,
        )
    out = dict(t=t, pops=pops, states=states,
               om_mhz=[float(np.abs(np.asarray(g['As'][0])).max())
                       for g in grps_mw] if grps_mw else [],
               method='mixed', backend='mixed-phi-dop853',
               loss=loss,
               diag=out_diag if diag else None,
               error_report=dict(n_segments=int(stat[1]),
                                 nfev=int(stat[0]), n_acc=int(stat[1]),
                                 n_rej=int(stat[2]), dt_min_hit=False,
                                 max_norm_dev=norm_dev,
                                 P=setup.P, dim=setup.dim))
    if comp is not None:
        out['loss_components'] = comp
    if record_amp:
        # ⚠ 2026-09-22 修（手册 §8.8）：原为
        #     Xf = exp(-1j*C*setup.D_full*T) * y_out[-1];  out['psi_amp'] = Xf
        # 两个错：① 长度是**扩展维** N·P（其余单体引擎 hard/sambe/reference/
        # twofield 一律回 N 维物理振幅）；② 给的是 Sambe 的 X，不是物理振幅。
        # `sector_sum_amp` 正是本引擎算 pops 用的同一个重构（见上方
        # `pops[i] = sector_sum_pops(y_out[i], setup.P, N)`）⇒ 与 pops 同源、长度 N。
        out['psi_amp'] = sector_sum_amp(y_out[-1], setup.P, N)
    return out


# ============================================================
# 两体混合引擎
# ============================================================


def _ac_ize_pair(E, A_o, omega_opt, r_crit=None):
    """AC 化（t70 定案）：剔除远强对；**位移不注入**。

    近对（如 60S↔6P）保留在光子块 M=1 动态、AC 位移自然涌现——与 hard 的
    RWA 动态同物理（M=1 对角化涌现 −18.87 ≡ 静态引擎 −16.47）。再注入全量
    静态位移 = 与动态双计（375ns 交换 0.012 vs 0.558 的根因，t70 实验定案；
    旧 inject=True 考古语义已随 t73b 深度清理移除）。
    """
    r_crit = R_CRIT if r_crit is None else r_crit
    if A_o is None or not np.any(np.abs(A_o) > 1e-9):
        return None
    A_new, _ = acize_physical(
        E, [dict(kind='exact', A=np.asarray(A_o, float),
                 omega_L_mhz=float(omega_opt))], r_crit)
    return A_new[0]


def _build_pair_geometry(E, optA_ac, optB_ac, omega_opt, bw_opt=None):
    """两体光子块结构（t73b 单次装配）：全部光波（已按侧提升）入同一载频组
    → 一个 _SambeSetup 同时给出 K0（纯 E）与逐波 Kcw（t69 逐波形态：
    交叉项不可能出现；a<b 三角过滤 + |ΔD|<BW_OPT 反旋过滤）。

    optA_ac / optB_ac 元素 (A_new, env)——A 侧 lift_side_A(A⊗I)、B 侧 I⊗A。
    """
    from .tensor_fields import lift_side_A
    bw_opt = BW_OPT if bw_opt is None else float(bw_opt)
    N = len(E)
    N2 = N * N

    def _one(t):
        return np.ones_like(np.atleast_1d(t), float)

    E_pair = np.array([E[i] + E[j]
                       for i in range(N) for j in range(N)])
    flds = []
    envs_out = []
    for side, lst in (('A', optA_ac), ('B', optB_ac)):
        for A_new, env in lst:
            if A_new is None or not np.any(np.abs(A_new) > 1e-9):
                continue
            if env is None:
                env = _one
            flds.append(dict(kind='exact',
                             A=lift_side_A(np.asarray(A_new, float), N, side),
                             omega_L_mhz=float(omega_opt), waveform=env))
            envs_out.append(env)
    # 无光场合法（t77）：全光波被 AC 化/选择定则剔除时没有光子组可挂，
    # 光子维退化 P=1（setup.P 自动给出）→ 微波/V_dd 当场精确积分。
    # Ms 必须随组数走：固定 [1] 会在零组时触发 photon_geometry 的长度校验。
    grps = group_by_freq(flds)
    setup = _SambeSetup(E_pair, grps, [1] * len(grps))
    P = setup.P
    dim = N2 * P
    K0 = np.real(np.asarray(setup.diag))
    pairs_opt = []
    # ⚠ 2026-09-12 修「静默丢物理」（E1 真凶）：光场近对**必须**同时支持 setup 的
    # 两种表示。`SambeSetup` 在 `dim > _DENSE_MAX`(2200) 时把 `Kc_dense` 置 None ——
    # 那**只是性能选择**（稀疏 vs 稠密），但这里原来写成
    #     `setup.Kc_dense[0] if setup.Kc_dense else []`
    # 于是超阈值时**整批光场耦合化为乌有**、光场变成"没接线"：
    #   实测 3 组（dim=1452 ≤ 2200）→ n_pairs=1206、频移光有效（B:60P=0.896）；
    #   ＋58G_7/2（dim=2700 > 2200）→ n_pairs 掉到 150、频移光**完全失效**
    #   （与"不加光"逐位相同）——这就是 E1"加一个远处能级配对数 1206→150"的真因，
    #   与 Θ/PHASE_TOL/候选窗无关。
    # 修法：稠密没有就用同源的**稀疏** `Kc_ext`（同矩阵、少一次 toarray）。
    Kc_lists = setup.kc_lists(0)
    import scipy.sparse as _sp
    for Kcw, env in zip(Kc_lists, envs_out):
        if _sp.issparse(Kcw):
            _co = Kcw.tocoo()
            _m = np.abs(_co.data) > 1e-12
            ii, jj, vv = _co.row[_m], _co.col[_m], _co.data[_m]
        else:
            Kcw = np.asarray(Kcw, complex)
            ii, jj = np.nonzero(np.abs(Kcw) > 1e-12)
            vv = Kcw[ii, jj]
        for a, b, val in zip(ii, jj, vv):
            # a<b 三角过滤（t70）：Kcw 为 Hermitian 全矩阵，双向打包会把
            # 同一耦合驱动两次（Rabi 翻倍 → AC 位移失真 4×）；
            # 反向由 RHS 厄米补齐自动。
            if a >= b:
                continue
            if abs(K0[a] - K0[b]) < bw_opt:
                pairs_opt.append((int(a), int(b), complex(val),
                                  float(K0[a] - K0[b]), env))
            else:
                # D1：**不静默**——记账（数量 + 被剔项最大 |v|），供 diag 告警
                # N3 修：`val` 是 complex（Kcw[a,b]），`float(val)` 会刷 ComplexWarning
                # 且语义是"取实部"（将来出现复耦合会静默算错）⇒ 用 complex 再取模。
                _BW_STATS['n_drop_bw'] += 1
                _BW_STATS['max_drop_abs_v_mhz'] = max(
                    _BW_STATS['max_drop_abs_v_mhz'], abs(complex(val)))
    return K0, pairs_opt, P, dim


# ---- 光场对 BW_OPT 丢弃统计（2026-09-15，核查报告 D1）----
# 背景：`|ΔD| ≥ BW_OPT` 的光场对原先是**静默丢弃**（同族的成本窗剔除却有 loud 统计
# 与诊断告警）——违反项目自定的"改物理必须交代"。几何构建与 diag 装配不在同一函数
# （前者返回 4 元组，不宜改签名），故用模块级统计传递；引擎单次运行、非重入，够用。
_BW_STATS = {'n_drop_bw': 0, 'max_drop_abs_v_mhz': 0.0}


def _take_bw_stats():
    """**读取并清零**（N2 修：模块级计数器必须每次运行独立——原来跨运行累积，
    实测同一进程连跑两次两体 mixed：run1 `n_drop_bw=16` → run2 `=32`，
    会让诊断虚高、归因错；GUI 连续运行 / 脚本扫描 / MCP 多次 run_compute 都会中）。

    ⚠ 2026-09-22 补（手册 §8.2）：**这个"读并清零"只在 `diag=True` 时被调用**
    （唯一调用点在 `if diag:` 块内）⇒ `diag=False` 的运行仍然累积，于是随后任一
    `diag=True` 的运行报出的是"本进程至今所有运行之和"（实测 diag=False 连跑 3 次
    ⇒ 16/32/48）。真正的修法在入口：`evolve_mixed_pair` 开头调 `_reset_bw_stats()`。
    """
    n = int(_BW_STATS.get('n_drop_bw', 0))
    m = float(_BW_STATS.get('max_drop_abs_v_mhz', 0.0))
    _BW_STATS['n_drop_bw'] = 0
    _BW_STATS['max_drop_abs_v_mhz'] = 0.0
    return n, m


def _reset_bw_stats():
    """每次**运行入口**清零（与 diag 无关）——见 `_take_bw_stats` 的 ⚠ 段。"""
    _BW_STATS['n_drop_bw'] = 0
    _BW_STATS['max_drop_abs_v_mhz'] = 0.0


def _split_waves(fields, mw_max=None):
    """fields（kind='exact'）→ (opt_waves, mw_waves)：**逐波列表，不合并**。

    每波形独立 (A, omega, env) 元组——最终形态（t69 船长裁定）：废除 t68 的
    同频合并模型（(ΣA)(Σenv) 展开有交叉项 A₁·env₂+A₂·env₁——顺序脉冲如
    波形1@70+副本@200 在 t=70 给 (A₁+A₂)·1=2A₁ → Rabi 翻倍 → π 提前 + 振荡
    回流——与 sambe T4/T6 修过的同频合并 bug 同类）。逐波后每波独立驱动，
    交叉项不可能出现；pack_rhs 按 (is_mw, w, id(env)) 分组天然支持同频多波。
    """
    mw_max = MW_MAX_MHZ if mw_max is None else mw_max
    opt, mw = [], []
    for f in fields or []:
        A = np.asarray(f['A'], float)
        om = float(f['omega_L_mhz'])
        (opt if om > mw_max else mw).append((A, om, f.get('waveform')))
    return opt, mw


def evolve_mixed_pair(states, fields_A, fields_B, psi0_A, psi0_B, t_grid,
                      vdd_matrix=None, R_um=None, theta_deg=0.0,
                      phi_deg=0.0, cache=None, tol=None, max_step=0.05,
                      progress_arr=None, log=None, record_amp=False,
                      decays_A=None, decays_B=None, driver='numba',
                      r_crit=None, diag=False,
                      bw_opt=None, mw_cap=None, mw_max=None,
                      mw_arb_A=None, mw_arb_B=None, phase_budget=None,
                      psi0_pair=None, max_step_hard=None):
    """两体混合引擎（正式版）：光场光子块 + 微波/V_dd 当场，φ-帧单 DOP853。

    fields_A/B : kind='exact' 场列表（build_fields_exact 输出——光场 + 微波）。
             **无光场合法**（两侧都无光波，或光波被 AC 化/选择定则剔除）：
             光子维退化 P=1，自动等价于微波/V_dd 当场精确积分（t77）。
    psi0_pair  : **两体纠缠初态** spec（['pair_sup', […]]；见
             evolve_util.as_pair_vector）——非 None 时完全取代
             psi0_A/psi0_B（逐原子直积）。交换后的结果就是这种态。
    vdd_matrix  : V_dd 稀疏/稠密矩阵（N²×N²）；None 且给 R_um/cache 时算
    r_crit : t75 上帝模式覆写——AC 化 r=Ω/ω 阈值（None=默认 R_CRIT）
    diag   : t75 诊断快照——True 时 out['diag'] 附结构化中间产物
    bw_opt / mw_cap / mw_max : t75/t80 上帝模式覆写——φ-帧过滤带宽 /
            微波候选窗**绝对**上界（MHz）/ 微波-光场分界（None=默认）
    phase_budget : t80 上帝模式覆写——微波候选窗的无量纲相预算（rad）
    返回 dict(t, pops, pops_A, pops_B, states, loss_A, loss_B,
              method='mixed', backend='mixed-phi-dop853-pair', error_report)。

    注意：V_dd 在 φ-帧同扇区 → ΔD=E_pair 差自动涌现（与 n 无关）；
    耗散（t62 实现）：寿命通道（落点=丢失）NonHermitian 对角——与
    hard_pair 同语义（_pair_loss_vectors；显式落点/纯退相拒绝）。
    """
    tol = DEFAULT_RTOL if tol is None else tol
    log = log or (lambda *a: None)
    if progress_arr is None:
        progress_arr = np.zeros(3)
    stage_progress(progress_arr, 0, 0.0)     # 阶段 0：装配/JIT 装载
    _reset_bw_stats()                        # 每次运行独立（手册 §8.2）
    t = np.asarray(t_grid, float)
    T = float(t[-1])
    N = len(states)
    N2 = N * N
    E = np.array([s['E_mhz'] for s in states], float)

    # t62 耗散（落点=丢失）：GammaA/B（/ns）→ gamma_MHz（N²）传给内核
    GammaA = GammaB = None
    gamma = None
    if decays_A or decays_B:
        GammaA, GammaB, has = _tensor._pair_loss_vectors(
            states, decays_A, decays_B, N, N2)
        if has:
            gamma = (GammaA + GammaB) / CONV
            log(f'[mixed_pair] 耗散 {int((gamma > 0).sum())} 两体态'
                f'（落点=丢失，NonHermitian）')

    # ---- t69 最终形态：逐波拆分（每波形独立 A/ω/env——废除合并模型）----
    mm = float(mw_max) if mw_max is not None else MW_MAX_MHZ
    optA, mwA = _split_waves(fields_A, mm)
    optB, mwB = _split_waves(fields_B, mm)

    # 光场载频一致性（共享光子块要求所有光波同频；异频 → 各自光子块未实现）
    # 无光场（om_o 保持 None）合法：光子维退化 P=1，见 _build_pair_geometry（t77）
    om_o = None
    for _A, om, _env in (optA + optB):
        if om_o is None:
            om_o = om
        elif abs(om_o - om) > 1e-6:
            raise NotImplementedError(
                f'光场载频不同（{om_o:.0f} vs {om:.0f} MHz）——共享光子块'
                '要求同频；异频多光子块为后续扩展')

    # ---- V_dd（同扇区当场；预计算/传入）----
    from . import pair_core as _pc
    if vdd_matrix is not None:
        V = vdd_matrix
    elif R_um is not None:
        if cache is None:
            raise ValueError('两体混合需要 cache 算 V_dd 或显式 vdd_matrix')
        V = _pc.vdd_sparse(cache, [dict(s) for s in states],
                           R_um, theta_deg, phi_deg)
    else:
        V = None
    if V is not None:
        import scipy.sparse as sp
        if sp.issparse(V):
            V = V.tocsr()
        else:
            V = sp.csr_matrix(np.asarray(V, complex))
    V_sp = V

    # ---- AC 化逐波（每波独立剔除远强对；位移不注入——t70 定案，见
    # _ac_ize_pair docstring。t73b：shifts 全 0 → 时变 AC 对角路径移除）----
    # t75：r_crit 可由上帝模式覆写
    rc = float(r_crit) if r_crit is not None else R_CRIT
    optA_ac = [(A_new, env) for (A, om, env) in optA
               if (A_new := _ac_ize_pair(E, A, om, rc)) is not None]
    optB_ac = [(A_new, env) for (A, om, env) in optB
               if (A_new := _ac_ize_pair(E, A, om, rc)) is not None]
    if optA_ac or optB_ac:
        log(f'[mixed_pair] AC 化：剔除远强对'
            f'（{len(optA_ac)}+{len(optB_ac)} 光波逐波，位移不注入）')

    # ---- 光子块结构（K0 纯 E——t62f 语义；共享扇区 M=1）----
    stage_progress(progress_arr, 0, 0.2)     # AC 化完成 → 光子块几何
    bw = float(bw_opt) if bw_opt is not None else BW_OPT
    K0, pairs_opt, P, dim = _build_pair_geometry(
        E, optA_ac, optB_ac, om_o, bw_opt=bw)
    has_opt = P > 1                      # t77：P=1 即无光场退化
    # 耗散 gamma（N²）→ 扩展空间（N²·P，每扇区同衰减）
    if gamma is not None:
        gamma = np.tile(gamma, P)
    if has_opt:
        log(f'[mixed_pair] 光场 P={P} dim={N2}×{P}={dim}'
            f'（5 倍降维参考：全 Sambe N²·15）')
    else:
        log(f'[mixed_pair] 无光场 → 光子维退化 P=1 dim={dim}，'
            '等价于微波/V_dd 当场精确积分')

    # ---- pairs 打包（7 元组 a,b,v,dD,is_mw,w,env）+ 分项标签（t75 诊断）----
    # t80：微波候选窗 = 相预算派生（同一 T 同时喂精度判据与成本预算）
    mw_cap = (float(mw_cap) if mw_cap is not None
              else mw_cap_mhz(T, phase_budget))
    pairs = []
    pair_tags = []
    # t79：微波对保留统计（主判据=累积相位，mw_cap 是成本预算）
    # `drop_cost_loud`（2026-09-22）：**仅因成本窗**被剔除、而 Θ 判据本来会保留的
    # 对数——单体侧早有此记账 + 吵闹日志（见 `_MixedSetup` 与 `drop_cost_loud`），
    # 两体侧一直没有 ⇒ "把演化窗口拉长反而丢掉远失谐梯级"在两体上完全静默。
    mw_stats = {'keep': 0, 'drop_cap': 0, 'drop_phase': 0,
                'drop_cost_loud': 0, 'loud_max_om': 0.0, 'loud_max_det': 0.0}

    def _one(tv):
        return np.ones_like(np.atleast_1d(tv))

    # V_dd：同扇区保留（env=恒 1——修复原型误乘光包络）。
    # **过滤（t78）**：vdd_sparse 含跨尺度对（60S/60P↔6P，|Δ|≈591THz）——若全
    # 打包进 φ-帧 RHS，DOP853 步长被钉死 ~1e-6ns → 620ns 需 1e9+ 步（半小时+）。
    # 判据自 t78 起改用**与 hard 共用的累积相位** Λ=C·T·|V|²/max(|ΔE|,|V|)：
    # 它天然剔除 THz 远失谐对（Θ→0），同时**保留**原先被误删的 34.6 GHz 双翻转
    # Förster 通道（|ΔE| 大但只是 2×17.3 GHz，Θ 仍达标）
    # 与 |V| 无关地一刀切，是 hard↔mixed V_dd 结果差 ~1e-5 的根因。
    # dE 与扇区无关（同扇区 ΔD = 纯物理能级差），故先在基态对上筛、再复制。
    E_pair = np.array([E[i] + E[j] for i in range(N) for j in range(N)])
    if V_sp is not None:
        coo = V_sp.tocoo()
        _vals = np.asarray(coo.data)
        _keep, _info = static_phase_keep(
            _vals, E_pair[coo.row] - E_pair[coo.col], T)
        log(f'[mixed_pair] V_dd 累积相位判据：保留 {_info["n_keep"]}/'
            f'{_vals.size}（θ_tol={PHASE_TOL:.0e} rad；被剔项 max|V|='
            f'{_info["max_v_drop"]:.3g} MHz，保留项最快相位 '
            f'{_info["max_rate_keep"]:.3g} rad/ns）')
        for a0, b0, vv in zip(coo.row[_keep], coo.col[_keep], _vals[_keep]):
            if a0 >= b0 or abs(vv) < 1e-12:
                continue
            for pn in range(P):
                a = a0 + pn * N2
                b = b0 + pn * N2
                dD = K0[a] - K0[b]
                pairs.append((int(a), int(b), complex(vv), dD, 0, 0.0,
                              _one))
                pair_tags.append('V_dd')

    # 微波当场逐波：A 侧 (i,k) 同 B 态 j；B 侧 (j,l) 同 A 态 i（修复残留 j bug）
    # t69：每波形独立 (A_mw, om_m, env_m)——不合并（交叉项不可能出现）
    for side, mw_list in (('A', mwA), ('B', mwB)):
        for A_mw, om_m, env_m in mw_list:
            if A_mw is None or not np.any(np.abs(A_mw) > 1e-9):
                continue
            if env_m is None:
                env_m = _one
            for ia in range(N):
                for ka in range(N):
                    if ia >= ka or abs(A_mw[ia, ka]) < 1e-9:
                        continue
                    for jb in range(N):
                        if side == 'A':
                            # t62f：dE 用纯 E（C 点——位移由 RHS 时变对角承担；
                            # 位移不注入（t70）——配对共振不被静态位移污染）
                            dE = (E[ia] + E[jb]) - (E[ka] + E[jb])
                            a0 = ia * N + jb
                            b0 = ka * N + jb
                        else:
                            dE = (E[jb] + E[ia]) - (E[jb] + E[ka])
                            a0 = jb * N + ia
                            b0 = jb * N + ka
                        om = abs(A_mw[ia, ka])
                        det = abs(abs(dE) - om_m)
                        # t79/t80：主判据 = 累积相位 Θ=C·T·Ω²/max(Δ,Ω)（与 V_dd 同
                        # 一判据，随 Ω² 自适应）；mw_cap 是相预算派生的成本上限。
                        if det > mw_cap:
                            mw_stats['drop_cap'] += 1
                            # 成本剔除但 Θ 判据本来会保留 ⇒ 这是"拉长窗口反而丢
                            # 路径"的信号，必须吵闹（与单体侧同口径）
                            if mw_pair_keep(om, det, T):
                                mw_stats['drop_cost_loud'] += 1
                                if om > mw_stats['loud_max_om']:
                                    mw_stats['loud_max_om'] = om
                                    mw_stats['loud_max_det'] = det
                            continue
                        if not mw_pair_keep(om, det, T):
                            mw_stats['drop_phase'] += 1
                            continue
                        mw_stats['keep'] += 1
                        for pn in range(P):
                            pairs.append((a0 + pn * N2, b0 + pn * N2,
                                          complex(A_mw[ia, ka]), dE, 1, om_m,
                                          env_m))
                            pair_tags.append('微波')

    # 光场近对逐波并入（pairs_opt 为 5 元组 (a,b,v,dD,env)——各波独立 env）
    for (a, b, v, dD, env) in pairs_opt:
        if env is None:
            env = _one
        pairs.append((int(a), int(b), v, dD, 0, 0.0, env))
        pair_tags.append('光场')

    log(f'[mixed_pair] 微波对累积相位判据：保留 {mw_stats["keep"]}'
        f'（θ_tol={PHASE_TOL:.0e} rad；成本预算剔除 '
        f'{mw_stats["drop_cap"]}，累积相位剔除 {mw_stats["drop_phase"]}）')
    if mw_stats['drop_cost_loud']:
        log(f'[mixed_pair] ⚠ 有 {mw_stats["drop_cost_loud"]} 对**仅因成本窗被剔除**'
            f'（Θ 判据本会保留，最强一对 |Ω|={mw_stats["loud_max_om"]:.3g} MHz /'
            f' Δ={mw_stats["loud_max_det"]:.3g} MHz）。窗宽 = PHASE_BUDGET/(C·T)，'
            f'此处 T={T:g} ns 给出 {mw_cap:.4g} MHz；若该远失谐通道是物理的一部分，'
            f'请把成本预算留空（= 不限制）。成本只该"换速度"，不该默默改模型。')

    # 任意波形微波（t79）：8 元组带 ph_fn → 内核走 cos(φ(t)) 查表。
    # 选对判据用**信号瞬时频率带宽**（任意波形无单一 om_m），不能用
    # |dE|-om_m；也不能像 hard 那样全收——mixed 的 φ-帧里光学对会带 THz
    # 相位钉死积分器，且与光场光子块双重计入。
    for side, arb_list in (('A', mw_arb_A), ('B', mw_arb_B)):
        for g in (arb_list or []):
            ph_fn = g.get('phase_fn')
            band = g.get('band_mhz')
            for A_mw, env_m in zip(g['As'], g['envs']):
                A_mw = np.asarray(A_mw, float)
                if not np.any(np.abs(A_mw) > 1e-9):
                    continue
                for ia in range(N):
                    for ka in range(N):
                        if ia >= ka or abs(A_mw[ia, ka]) < 1e-9:
                            continue
                        for jb in range(N):
                            if side == 'A':
                                dE = (E[ia] + E[jb]) - (E[ka] + E[jb])
                                a0 = ia * N + jb
                                b0 = ka * N + jb
                            else:
                                dE = (E[jb] + E[ia]) - (E[jb] + E[ka])
                                a0 = jb * N + ia
                                b0 = jb * N + ka
                            if band is not None:
                                ad = abs(dE)
                                if not (band[0] - mw_cap
                                        <= ad <= band[1] + mw_cap):
                                    continue
                            for pn in range(P):
                                pairs.append((a0 + pn * N2, b0 + pn * N2,
                                              complex(A_mw[ia, ka]), dE, 1,
                                              0.0, env_m, ph_fn))
                                pair_tags.append('微波')

    # ---- 初态嵌入（中心扇区 n=0）+ 积分 ----
    # psi0_pair（两体 spec）非 None → 纠缠/任意两体叠加；否则逐原子直积。
    psi0_2d = pair_psi0_vector(states, psi0_A, psi0_B, psi0_pair)
    X0 = np.zeros(dim, complex)
    pc = P // 2                      # t77：中心扇区随 P 走（有光场 3→1；无光场 1→0）
    X0[pc * N2: (pc + 1) * N2] = psi0_2d

    pack = pack_rhs(pairs, T)
    Cv = CONV
    # 窄脉冲每步硬上限（2026-09-22，C1b 同族修复）：与单体 mixed 同一处事故
    # （`max_step` 只管起步；包络在脉冲外恰为 0 ⇒ 步长暴涨跨过脉冲）。
    _msh = max_step_hard
    if _msh is None:
        _msh = feature_max_step([p[6] for p in pairs], T)
    stat = np.zeros(6) if diag else np.zeros(4)
    dt_hist = (np.zeros((100000, 2)) if diag else None)
    stage_progress(progress_arr, 0, 0.9)     # 打包/env 表完成（JIT 在驱动首调）
    ac_base = None          # t73b：位移不注入（t70 定案）→ AC 对角恒关；
    ac_env_tab = None       # 内核 AC 分支保留供单原子 mixed 使用
    ac_env_dt = 1.0

    def rhs(tv, phi):
        return rhs_phi_pairs(tv, phi, pack['iis'], pack['jjs'],
                               pack['vals'], pack['dDs'], pack['is_mw'],
                               pack['wcos'], pack['n_pairs_w'],
                               pack['env_tab'], pack['env_dt'],
                               pack['has_ph'], pack['ph_tab'], Cv, gamma,
                               ac_base, ac_env_tab, ac_env_dt)

    pops = np.zeros((len(t), N2))
    pops[0] = np.abs(psi0_2d) ** 2
    # driver='scipy': solve_ivp 逐区间（对照路径）
    # 阶段 1（积分）由驱动在首个接受步置位——调用内的 numba 装载/编译期
    # 保持阶段 0（"准备"标签 + 墙钟耗时），GUI 显示不失真
    if driver == 'numba':
        y_out = dop853_driver_mixed(
            X0, t, T, float(tol), 1e-11, max_step,
            progress_arr, stat, dt_hist, float(_msh or 0.0),
            pack['iis'], pack['jjs'],
            pack['vals'], pack['dDs'], pack['is_mw'], pack['wcos'],
            pack['n_pairs_w'], pack['env_tab'], pack['env_dt'],
            pack['has_ph'], pack['ph_tab'], Cv, gamma,
            ac_base, ac_env_tab, ac_env_dt)
        stage_progress(progress_arr, 2, 0.0)  # 阶段 2：收尾（跨扇区投影）
        for i in range(1, len(t)):
            # t73b：与单原子同一重构（= 旧 Xf/ns 相位式——e^{-iC·E·t} 全局
            # 相位后光子相位恰对消，布居相同）
            pops[i] = sector_sum_pops(y_out[i], P, N2)
            stage_progress(progress_arr, 2, i / max(len(t) - 1, 1))
        n_dop = int(stat[0]) if len(stat) > 0 else 0
    else:
        from scipy.integrate import solve_ivp
        X = X0.copy()
        tprev = 0.0
        n_dop = 0
        # 逐扇区重构系数（随 P：有光场 [-1,0,1]；无光场 [0]）
        ns = np.arange(P, dtype=float) - (P // 2)
        om_rec = float(om_o) if om_o is not None else 0.0
        stage_progress(progress_arr, 1, 0.0)  # 阶段 1：积分（scipy 逐区间）
        for i in range(1, len(t)):
            if progress_arr[1] != 0.0:
                raise RuntimeError('用户取消')
            tb = float(t[i])
            sol = solve_ivp(rhs, (tprev, tb), X, method='DOP853', rtol=tol,
                            atol=1e-11, max_step=max_step)   # scipy 对照路径
            # （刻意比生产路径 DEFAULT_ATOL=1e-10 更紧：这条路的用途就是
            #   与 numba 驱动逐位对照，误差预算留给数值路径本身）
            if not sol.success:
                raise RuntimeError(f'两体混合积分失败: {sol.message}')
            n_dop += int(sol.nfev)
            X = sol.y[:, -1]
            Xf = np.exp(-1j * Cv * K0 * tb) * X
            psi = np.zeros(N2, complex)
            for pn in range(P):
                psi += np.exp(-1j * Cv * ns[pn] * om_rec * tb) \
                    * Xf[pn * N2: (pn + 1) * N2]
            pops[i] = np.abs(psi) ** 2
            tprev = tb
            progress_arr[0] = min(tprev / T, 0.999)
    stage_progress(progress_arr, 2, 1.0)

    # t62 耗散：loss_A/loss_B 曲线 + 含 loss 范数（同 sambe.evolve_sambe_pair）
    loss_A = np.zeros(len(t))
    loss_B = np.zeros(len(t))
    norm_dev = float(np.max(np.abs(pops.sum(axis=1) - 1.0)))
    if GammaA is not None:
        dt_arr = np.diff(t)
        rate_A = pops @ (GammaA * 1.0)          # (n_t,)，GammaA 已 /ns
        rate_B = pops @ GammaB
        loss_A[1:] = np.cumsum(0.5 * (rate_A[1:] + rate_A[:-1]) * dt_arr)
        loss_B[1:] = np.cumsum(0.5 * (rate_B[1:] + rate_B[:-1]) * dt_arr)
        norm_dev = float(np.max(np.abs(
            pops.sum(axis=1) + loss_A + loss_B - 1.0)))
    pops_A = _tensor._reduce_A(pops, N)
    pops_B = _tensor._reduce_B(pops, N)
    # ---- t75 诊断快照（两体；diag=True 才采集）----
    if diag:
        diag_pairs = dict(
            a=np.array([p[0] for p in pairs], int),
            b=np.array([p[1] for p in pairs], int),
            v=np.array([p[2] for p in pairs], complex),
            dD=np.array([p[3] for p in pairs], float),
            is_mw=np.array([p[4] for p in pairs], int),
            w=np.array([p[5] for p in pairs], float))
        acize_fields = []
        for (A0, om0, _env0) in (optA + optB):
            A1 = _ac_ize_pair(E, A0, om0, rc)
            n0 = int((np.abs(np.asarray(A0, float)) > 1e-9).sum() // 2)
            n1 = (int((np.abs(np.asarray(A1, float)) > 1e-9).sum() // 2)
                  if A1 is not None else 0)
            acize_fields.append(dict(
                omega=float(om0), n_before=n0, n_kept=n1, n_acized=n0 - n1,
                shift_min=0.0, shift_max=0.0))
        idx = np.linspace(0, len(t) - 1, min(len(t), 240)).astype(int)
        sector_amps = {p: np.abs(y_out[idx][:, p * N2:(p + 1) * N2])
                       for p in range(P)}
        ts_snap = sorted(set(float(x) for x in np.linspace(0.0, T, 5)))
        snaps = ({t0: _assemble_H_snapshot(K0, pairs, t0, Cv, gamma=gamma,
                                           tags=pair_tags)
                  for t0 in ts_snap}
                 if _h_snap_ok(dim) else {'skipped': _h_snap_note(dim)})
        diag_dt = (dt_hist[:int(stat[4])] if dt_hist is not None
                   else np.zeros((0, 2)))
        _vmax = (float(np.abs(diag_pairs['v']).max())
                 if diag_pairs['v'].size else 0.0)
        out_diag = dict(
            meta=dict(method='mixed', backend='mixed-phi-dop853-pair', N=N,
                      P=P, dim=dim, t_end=T, M=[1] if has_opt else [],
                      r_crit=float(rc), n_pairs=len(pairs), tol=float(tol),
                      max_step=float(max_step), bw=bw, mw_cap=mw_cap,
                      mw_max=float(mm),
                      # 2026-09-15（核查报告 D1）：光场对因 BW_OPT 被丢弃的数量——
                      # 与成本窗同款的"改了模型必须交代"；原来完全静默、且
                      # diag 侧读的 n_drop_cap/n_drop_phase 根本没有写入端（哑告警）。
                      # 2026-09-22（手册 §8.4）：把 keep/n_drop_cap/n_drop_phase 也
                      # 补上——它们此前仍无写入端，故"成本窗剔除=改了模型"的告警
                      # 一直不触发（D1 只补了 n_drop_bw）。
                      n_drop_bw=(_bw := _take_bw_stats())[0],
                      max_drop_bw_abs_v_mhz=float(_bw[1]),
                      keep=int(mw_stats['keep']),
                      n_drop_cap=int(mw_stats['drop_cap']),
                      n_drop_phase=int(mw_stats['drop_phase']),
                      n_drop_cost_loud=int(mw_stats['drop_cost_loud']),
                      max_abs_v_mhz=_vmax,
                      labels=[f"({states[i]['label']})×({states[j]['label']})"
                              for i in range(N) for j in range(N)],
                      stride=N2),
            photon=dict(Ms=[1] if has_opt else [], P=P, dim=dim,
                        strides=[1] if has_opt else [],
                        nvecs=np.arange(P), D_full=np.asarray(K0, float).copy()),
            acize=dict(r_crit=float(rc), fields=acize_fields,
                       n_shifted=0, shift=np.zeros(N2)),
            pairs=dict(diag_pairs, n=int(diag_pairs['v'].size),
                       max_abs_v_mhz=_vmax),
            solver=dict(t_steps=diag_dt[:, 0].copy(), dt=diag_dt[:, 1].copy(),
                        nfev=int(stat[0]), n_acc=int(stat[1]),
                        n_rej=int(stat[2]), norm_dev=norm_dev),
            trajectory=dict(t=t[idx].copy(), sector_amps=sector_amps),
            H_snapshots=snaps,
        )
    report = dict(nfev=n_dop, max_norm_dev=norm_dev,
                  P=P, dim=dim,
                  photon_blocks=({f'{om_o:.6g}MHz': P} if has_opt else {}),
                  ac_shifts=None)
    out = dict(
        t=t, pops=pops, pops_A=pops_A, pops_B=pops_B,
        loss_A=loss_A, loss_B=loss_B,
        states=[{
            'label': f"({states[i]['label']}) x ({states[j]['label']})",
            'E_mhz': E_pair[i * N + j],
            'ket_i': i, 'ket_j': j,
        } for i in range(N) for j in range(N)],
        method='mixed', backend='mixed-phi-dop853-pair',
        error_report=report,
        diag=out_diag if diag else None,
    )
    if record_amp:
        # ⚠ 2026-09-22 修（手册 §8.8 同族）：原为 `sector_sum_amp(X, P, N2)`——
        # `X` 只在 **scipy** 分支（`driver != 'numba'`）里被赋值，生产路径
        # （driver='numba'）走 `y_out` ⇒ `record_amp=True` **必崩**
        # （实测 `UnboundLocalError: local variable 'X'`）。改用 `y_out[-1]`
        # （与上方 `pops[i] = sector_sum_pops(y_out[i], P, N2)` 同源）。
        out['psi_amp'] = sector_sum_amp(y_out[-1], P, N2)
    log(f'[mixed_pair] 完成：norm {norm_dev:.2e}，DOP853 求值 {n_dop}')
    return out


