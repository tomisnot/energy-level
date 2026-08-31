# -*- coding: utf-8 -*-
"""hard 路线加速内核（numba 三件套第 1、2 件）：evolve_hard 的 JIT RHS + 包络查表。

背景（docs/AI队友任务书-两场引擎.md §12 hard 路线，harddev 负责）：
- 目标：单原子 22 态 620ns ≤5 s（实测基线 173.2 s @ rtol=1e-8）；
  零近似不变——DOP853 硬积分是 hard 的身份，禁止解析化载波（那是变成 Sambe）。
- Julia 经验迁移（docs/Julia引擎交接文档.md §14.1/§15.3）：
  ① JIT 内核化（=> numba njit）：RHS 全量编译，消除每次 RHS 求值的 Python
     解释开销（基线 560057 次 RHS / 173.2 s ≈ 309 µs/次 → 内核 ~3-5 µs/次）；
  ② 包络查表化（_pack_pair_terms 模式，pair_evolve.py:290）：均匀网格预采样
     + 线性插值。包络慢变（高斯 σ=40ns / 方波沿 20ns），620ns 上 2^17 点
     → h≈0.0047ns → 插值误差 ~3e-10 ≪ rtol=1e-8；
     —— 载波 cos **不查表**：线性插值表误差 ~h²ω²/8，要达 1e-9 需
        h ≤ sqrt(8e-9)/ω（ω=216.6 rad/ns 反旋项）≈ 8.2e-7 ns → 7.5e8 点/载波，
        不可行；内核直接 cos()（libm ~10ns/次）是唯一正确选择。
- 数学与 evolve_two_field.evolve_hard 逐项一致：
    Ṽ(t)  = Σ_opt c(t)·A_opt + Σ_mw env(t)·cos(C·ω·t)·A_mw
    y'    = −iC·u·(Ṽ·(conj(u)∘y))，u = exp(iC·(E_rot−min)·t)
  （原版 u·Ṽ·u† @ y 的结合律等价分解，浮点舍入差异 ~1e-15 级，
   对 DOP853 轨迹的影响远小于 rtol。）

2026-08-30 harddev 首版。
"""

# ============================================================
# hard 单/两体合并（t19：hard_numba + hard_pair → hard.py）。
# 单原子：_rhs_hard / _pack_hard / build_hard_rhs / evolve_hard_fast；
# 两体：_rhs_hard_pair / _pack_pair_hard / evolve_hard_pair（k=i·N+j 张量
# + V_dd 相位扭曲 + njit DOP853）。系数 _A.._D 供 dop853 核心层 import。
# ============================================================

import numpy as np
from numba import njit

from . import evolve as _evolve
from . import pair_evolve as _pe
from .evolve_sambe import _CONV

# 包络查表点数（620ns → h≈0.0047ns；插值误差 ~3e-10）
_TAB_N = 2 ** 17


@njit(cache=True)
def _env_at(tab, t, t0, dt):
    """均匀网格线性插值（端点 clamp）。"""
    x = (t - t0) / dt
    i0 = int(x)
    n = tab.shape[0]
    if i0 < 0:
        return tab[0]
    if i0 >= n - 1:
        return tab[n - 1]
    f = x - i0
    return tab[i0] * (1.0 - f) + tab[i0 + 1] * f


@njit(cache=True)
def _rhs_hard(t, y, E, opt_As, opt_tabs, opt_deltas, opt_masks,
              mw_As, mw_ws, mw_tabs, mw_phases, t0, dt, CONV):
    """单原子 hard RHS（光场旋转系 + 微波 cos(φ_f(t)) 硬积分，零近似）。

    y' = −iC·u·(Ṽ·(conj(u)∘y))，u = exp(iC·E·t)。
    光场拍频（§2.3/§8）：δ=ω_f−ω_ref ≠ 0 的光场耦合项乘
    exp(iC·δ·t·(m[i]−m[k]))（m=旋转流形掩码，t10 验证 exp 拍频正确）；
    δ=0（首场/旧元组）逐位退化为 c·A 原行为。
    微波（t18 统一 cos(φ_f(t)) 任意相位）：相位表 mw_phases 线性插值求值
    ——定频 φ=C·ω·t 为线性表（插值精确，逐位≈直算）；变频/任意相位由
    phase_fn 采样。步长随 φ' 自适应（物理必然）。
    所有矩阵按非零元跳过（A 稀疏，光学只耦合 6P↔60S 等少数对）。
    """
    N = E.shape[0]
    V = np.zeros((N, N), np.complex128)
    n_opt = opt_As.shape[0]
    for oi in range(n_opt):
        c = _env_at(opt_tabs[oi], t, t0, dt)
        if c != 0.0:
            A = opt_As[oi]
            d = opt_deltas[oi]
            if d != 0.0:
                p1 = np.exp(1j * CONV * d * t)
                p2 = p1.conjugate()
                mask = opt_masks[oi]
                for i in range(N):
                    mi = mask[i]
                    for k in range(N):
                        a = A[i, k]
                        if a != 0.0:
                            ph = p1 if (mi - mask[k]) > 0 else p2
                            V[i, k] += c * a * ph
            else:
                for i in range(N):
                    for k in range(N):
                        a = A[i, k]
                        if a != 0.0:
                            V[i, k] += c * a
    n_mw = mw_As.shape[0]
    for mi in range(n_mw):
        env = _env_at(mw_tabs[mi], t, t0, dt)
        if env != 0.0:
            cw = env * np.cos(_env_at(mw_phases[mi], t, t0, dt))
            A = mw_As[mi]
            for i in range(N):
                for k in range(N):
                    a = A[i, k]
                    if a != 0.0:
                        V[i, k] += cw * a
    u = np.exp(1j * CONV * E * t)
    z = np.empty(N, np.complex128)
    for k in range(N):
        z[k] = u[k].conjugate() * y[k]
    out = np.empty(N, np.complex128)
    for i in range(N):
        acc = 0.0j
        for k in range(N):
            vik = V[i, k]
            if vik != 0.0:
                acc += vik * z[k]
        out[i] = -1j * CONV * u[i] * acc
    return out


def _env_tab(_env_sample, T, n_pts):
    """包络在 [0,T] 均匀网格上的采样表（_env_sample 为绝对时间 callable）。"""
    ts = np.linspace(0.0, float(T), n_pts)
    return np.asarray(_env_sample(ts), float)


def _pack_hard(fields_meta, T, n_tab=_TAB_N):
    """fields_meta（build_two_field 输出）→ 内核打包数组。

    返回 (E, opt_As, opt_tabs, opt_deltas, opt_masks,
          mw_As, mw_ws, mw_tabs, t0, dt, CONV)，
    供 njit 内核直接消费（打包/查表在热循环外做一次——Julia 经验）。
    opt_deltas：光场拍频 δ=ω_f−ω_ref（首场/旧 4 元组 = 0 → 逐位退化）；
    opt_masks：旋转流形掩码（拍频相位符号 m[i]−m[k]）。
    """
    E_rot = np.asarray(fields_meta['E_rot'], float)
    E = E_rot - float(E_rot.min())
    opt_terms = fields_meta['opt_terms']
    mw_groups = fields_meta['mw_groups']
    n0 = E.size
    if opt_terms:
        opt_As = np.stack([np.asarray(t[0], float) for t in opt_terms])
        opt_tabs = np.stack([_env_tab(t[1], T, n_tab) for t in opt_terms])
        opt_deltas = np.array(
            [float(t[2]) - float(t[4]) if len(t) > 4 else 0.0
             for t in opt_terms], float)
        opt_masks = np.stack(
            [np.asarray(t[5], int) if len(t) > 5
             else np.zeros(n0, int)
             for t in opt_terms])
    else:
        opt_As = np.zeros((0, n0, n0))
        opt_tabs = np.zeros((0, n_tab))
        opt_deltas = np.zeros(0)
        opt_masks = np.zeros((0, n0), int)
    mw_As_list, mw_ws_list, mw_tabs_list, mw_phase_list = [], [], [], []
    for g in mw_groups:
        w = float(g['omega_mhz'])
        # t18：hard 统一 cos(φ_f(t)) 任意相位路径——定频 φ=C·ω·t（线性，
        # 插值精确退化）；变频/任意相位由 mw_groups 的 phase_fn 提供
        # （如 ArbitraryWave 解析信号相位；无则默认定频）。
        ph_fn = g.get('phase_fn')
        for A, wf in zip(g['As'], g['envs']):
            mw_As_list.append(np.asarray(A, float))
            mw_ws_list.append(w)
            mw_tabs_list.append(_env_tab(wf, T, n_tab))
            if ph_fn is None:
                ts = np.linspace(0.0, float(T), n_tab)
                mw_phase_list.append(_CONV * w * ts)
            else:
                mw_phase_list.append(_env_tab(ph_fn, T, n_tab))
    if mw_As_list:
        mw_As = np.stack(mw_As_list)
        mw_ws = np.asarray(mw_ws_list, float)
        mw_tabs = np.stack(mw_tabs_list)
        mw_phases = np.stack(mw_phase_list)
    else:
        mw_As = np.zeros((0, n0, n0))
        mw_ws = np.zeros(0)
        mw_tabs = np.zeros((0, n_tab))
        mw_phases = np.zeros((0, n_tab))
    t0 = 0.0
    dt = float(T) / (n_tab - 1) if n_tab > 1 else 1.0
    CONV = float(_CONV)
    return (E, opt_As, opt_tabs, opt_deltas, opt_masks,
            mw_As, mw_ws, mw_tabs, mw_phases, t0, dt, CONV)


def build_hard_rhs(fields_meta, T, n_tab=_TAB_N):
    """fields_meta（build_two_field 输出）→ njit RHS 闭包（供 solve_ivp 使用）。

    数组在闭包外预打包（Julia 经验：打包/查表在热循环外做一次）。
    """
    (E, opt_As, opt_tabs, opt_deltas, opt_masks,
     mw_As, mw_ws, mw_tabs, t0, dt, CONV) = \
        _pack_hard(fields_meta, T, n_tab=n_tab)

    def _rhs_hard_pack(tv, y):
        return _rhs_hard(tv, y, E, opt_As, opt_tabs, opt_deltas, opt_masks,
                         mw_As, mw_ws, mw_tabs, t0, dt, CONV)
    return _rhs_hard_pack


# ============================================================
# numba DOP853 驱动（scipy _ivp/rk.py + dop853_coefficients.py 的逐行移植）
# 理由：scipy solve_ivp 的步进循环是纯 Python（~40-80µs/步 × ~47k 步），
# 在 RHS 内核化后成为剩余瓶颈；整循环 JIT 后 620ns 从 ~4-6s 降到 ~1.5s。
# 系数直接取自 scipy 1.15.2（dop853_coefficients.py，Hairer DOP853）。
# ============================================================
import numpy as _np
_N_STAGES = 12
_N_EXT = 16

_A = _np.array([
    [0.0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [5.26001519587677318785587544488e-2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [1.97250569845378994544595329183e-2, 5.91751709536136983633785987549e-2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [2.95875854768068491816892993775e-2, 0, 8.87627564304205475450678981324e-2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [2.41365134159266685502369798665e-1, 0, -8.84549479328286085344864962717e-1, 9.24834003261792003115737966543e-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [3.7037037037037037037037037037e-2, 0, 0, 1.70828608729473871279604482173e-1, 1.25467687566822425016691814123e-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [3.7109375e-2, 0, 0, 1.70252211019544039314978060272e-1, 6.02165389804559606850219397283e-2, -1.7578125e-2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [3.70920001185047927108779319836e-2, 0, 0, 1.70383925712239993810214054705e-1, 1.07262030446373284651809199168e-1, -1.53194377486244017527936158236e-2, 8.27378916381402288758473766002e-3, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [6.24110958716075717114429577812e-1, 0, 0, -3.36089262944694129406857109825, -8.68219346841726006818189891453e-1, 2.75920996994467083049415600797e1, 2.01540675504778934086186788979e1, -4.34898841810699588477366255144e1, 0, 0, 0, 0, 0, 0, 0, 0],
    [4.77662536438264365890433908527e-1, 0, 0, -2.48811461997166764192642586468, -5.90290826836842996371446475743e-1, 2.12300514481811942347288949897e1, 1.52792336328824235832596922938e1, -3.32882109689848629194453265587e1, -2.03312017085086261358222928593e-2, 0, 0, 0, 0, 0, 0, 0],
    [-9.3714243008598732571704021658e-1, 0, 0, 5.18637242884406370830023853209, 1.09143734899672957818500254654, -8.14978701074692612513997267357, -1.85200656599969598641566180701e1, 2.27394870993505042818970056734e1, 2.49360555267965238987089396762, -3.0467644718982195003823669022, 0, 0, 0, 0, 0, 0],
    [2.27331014751653820792359768449, 0, 0, -1.05344954667372501984066689879e1, -2.00087205822486249909675718444, -1.79589318631187989172765950534e1, 2.79488845294199600508499808837e1, -2.85899827713502369474065508674, -8.87285693353062954433549289258, 1.23605671757943030647266201528e1, 6.43392746015763530355970484046e-1, 0, 0, 0, 0, 0],
    [5.42937341165687622380535766363e-2, 0, 0, 0, 0, 4.45031289275240888144113950566, 1.89151789931450038304281599044, -5.8012039600105847814672114227, 3.1116436695781989440891606237e-1, -1.52160949662516078556178806805e-1, 2.01365400804030348374776537501e-1, 4.47106157277725905176885569043e-2, 0, 0, 0, 0],
    [5.61675022830479523392909219681e-2, 0, 0, 0, 0, 0, 2.53500210216624811088794765333e-1, -2.46239037470802489917441475441e-1, -1.24191423263816360469010140626e-1, 1.5329179827876569731206322685e-1, 8.20105229563468988491666602057e-3, 7.56789766054569976138603589584e-3, -8.298e-3, 0, 0, 0],
    [3.18346481635021405060768473261e-2, 0, 0, 0, 0, 2.83009096723667755288322961402e-2, 5.35419883074385676223797384372e-2, -5.49237485713909884646569340306e-2, 0, 0, -1.08347328697249322858509316994e-4, 3.82571090835658412954920192323e-4, -3.40465008687404560802977114492e-4, 1.41312443674632500278074618366e-1, 0, 0],
    [-4.28896301583791923408573538692e-1, 0, 0, 0, 0, -4.69762141536116384314449447206, 7.68342119606259904184240953878, 4.06898981839711007970213554331, 3.56727187455281109270669543021e-1, 0, 0, 0, -1.39902416515901462129418009734e-3, 2.9475147891527723389556272149, -9.15095847217987001081870187138, 0],
], dtype=_np.float64)

_B = _A[_N_STAGES, :_N_STAGES]
_C = _np.array([0.0, 0.526001519587677318785587544488e-01,
                0.789002279381515978178381316732e-01,
                0.118350341907227396726757197510,
                0.281649658092772603273242802490,
                0.333333333333333333333333333333, 0.25,
                0.307692307692307692307692307692,
                0.651282051282051282051282051282, 0.6,
                0.857142857142857142857142857142, 1.0, 1.0, 0.1, 0.2,
                0.777777777777777777777777777778], dtype=_np.float64)
_E3 = _np.zeros(_N_STAGES + 1)
_E3[:-1] = _B.copy()
_E3[0] -= 0.244094488188976377952755905512
_E3[8] -= 0.733846688281611857341361741547
_E3[11] -= 0.220588235294117647058823529412e-1
_E5 = _np.zeros(_N_STAGES + 1)
_E5[0] = 0.1312004499419488073250102996e-1
_E5[5] = -0.1225156446376204440720569753e+1
_E5[6] = -0.4957589496572501915214079952
_E5[7] = 0.1664377182454986536961530415e+1
_E5[8] = -0.3503288487499736816886487290
_E5[9] = 0.3341791187130174790297318841
_E5[10] = 0.8192320648511571246570742613e-1
_E5[11] = -0.2235530786388629525884427845e-1
_D = _np.zeros((4, _N_EXT))
_D[0, 0] = -0.84289382761090128651353491142e+1
_D[0, 5] = 0.56671495351937776962531783590
_D[0, 6] = -0.30689499459498916912797304727e+1
_D[0, 7] = 0.23846676565120698287728149680e+1
_D[0, 8] = 0.21170345824450282767155149946e+1
_D[0, 9] = -0.87139158377797299206789907490
_D[0, 10] = 0.22404374302607882758541771650e+1
_D[0, 11] = 0.63157877876946881815570249290
_D[0, 12] = -0.88990336451333310820698117400e-1
_D[0, 13] = 0.18148505520854727256656404962e+2
_D[0, 14] = -0.91946323924783554000451984436e+1
_D[0, 15] = -0.44360363875948939664310572000e+1
_D[1, 0] = 0.10427508642579134603413151009e+2
_D[1, 5] = 0.24228349177525818288430175319e+3
_D[1, 6] = 0.16520045171727028198505394887e+3
_D[1, 7] = -0.37454675472269020279518312152e+3
_D[1, 8] = -0.22113666853125306036270938578e+2
_D[1, 9] = 0.77334326684722638389603898808e+1
_D[1, 10] = -0.30674084731089398182061213626e+2
_D[1, 11] = -0.93321305264302278729567221706e+1
_D[1, 12] = 0.15697238121770843886131091075e+2
_D[1, 13] = -0.31139403219565177677282850411e+2
_D[1, 14] = -0.93529243588444783865713862664e+1
_D[1, 15] = 0.35816841486394083752465898540e+2
_D[2, 0] = 0.19985053242002433820987653617e+2
_D[2, 5] = -0.38703730874935176555105901742e+3
_D[2, 6] = -0.18917813819516756882830838328e+3
_D[2, 7] = 0.52780815920542364900561016686e+3
_D[2, 8] = -0.11573902539959630126141871134e+2
_D[2, 9] = 0.68812326946963000169666922661e+1
_D[2, 10] = -0.10006050966910838403183860980e+1
_D[2, 11] = 0.77771377980534432092869265740
_D[2, 12] = -0.27782057523535084065932004339e+1
_D[2, 13] = -0.60196695231264120758267380846e+2
_D[2, 14] = 0.84320405506677161018159903784e+2
_D[2, 15] = 0.11992291136182789328035130030e+2
_D[3, 0] = -0.25693933462703749003312586129e+2
_D[3, 5] = -0.15418974869023643374053993627e+3
_D[3, 6] = -0.23152937917604549567536039109e+3
_D[3, 7] = 0.35763911791061412378285349910e+3
_D[3, 8] = 0.93405324183624310003907691704e+2
_D[3, 9] = -0.37458323136451633156875139351e+2
_D[3, 10] = 0.10409964950896230045147246184e+3
_D[3, 11] = 0.29840293426660503123344363579e+2
_D[3, 12] = -0.43533456590011143754432175058e+2
_D[3, 13] = 0.96324553959188282948394950600e+2
_D[3, 14] = -0.39177261675615439165231486172e+2
_D[3, 15] = -0.14972683625798562581422125276e+3


# ============================================================
# DOP853 系数（scipy 1.15.2 逐行移植，已验证基线）——保留在此作为
# 单一来源，dop853 核心层从本模块 import（禁止手写重建）。
# 驱动（步进/误差/步长/稠密输出主循环）已上收至 src/core/dop853.py，
# 以 njit first-class 函数参数化 RHS（_numba_funcarg_test.py + t11 实验
# 验证：与旧驱动逐位一致 max|Δy|=0，nfev/stat 全同）。
# 注意：dop853 的驱动导入放在系数之后（dop853 反向 import 本模块系数，
# 模块级互引在系数定义前会成环）。
# ============================================================
from .dop853 import _dop853_driver


def _mw_terms_of(fields_meta):
    mw_terms = []
    for g in fields_meta['mw_groups']:
        for A, wf in zip(g['As'], g['envs']):
            mw_terms.append((np.asarray(A, float),
                             float(g['omega_mhz']), wf))
    return mw_terms


def evolve_hard_fast(states, fields_meta, psi0, t_grid, rtol=1e-9, atol=1e-11,
                     progress_arr=None, log=None, record_amp=False,
                     n_tab=_TAB_N, driver='numba'):
    """evolve_hard 的 numba 加速版：同数学、同 DOP853、同返回契约。

    driver='numba'（默认）：复用 dop853 核心层全编译 DOP853 主循环
      （dop853._dop853_driver(_rhs_hard, ...)，njit first-class RHS 参数
      化——t11 验证与旧驱动逐位一致）；RHS 在内核内直接调用——无 Python
      派发开销（~10µs/RHS 而非 ~59µs）；
    driver='scipy'：njit RHS + scipy solve_ivp（对照用）。
    backend='hard-numba-dop853'。唯一差异：包络按查表线性插值求值
    （误差 ~3e-10 ≪ rtol），其余逐位同构。
    """
    log = log or (lambda *a: None)
    t = np.asarray(t_grid, float)
    N = fields_meta['N']
    E_rot = np.asarray(fields_meta['E_rot'], float)
    opt_terms = fields_meta['opt_terms']
    mw_terms = _mw_terms_of(fields_meta)
    psi0v = _evolve._as_vector(states, psi0)
    T = float(t[-1])
    E = E_rot - float(E_rot.min())
    pack = _pack_hard(fields_meta, T, n_tab=n_tab)

    if driver == 'numba':
        # 函数级 import：dop853 从本模块 import 系数（_A.._D），模块级
        # 互引会成环——驱动只在运行期需要，函数级加载零开销（sys.modules）。
        from .dop853 import _dop853_driver
        prog = (np.zeros(2) if progress_arr is None
                else np.asarray(progress_arr, float))
        stat = np.zeros(4)
        y_out = _dop853_driver(_rhs_hard, psi0v, t, T, float(rtol),
                               float(atol), np.inf, prog, stat, *pack)
        pops = np.abs(y_out) ** 2
        nfev = int(stat[0])
        n_acc = int(stat[1])
        n_rej = int(stat[2])
        if progress_arr is not None:
            progress_arr[0] = 1.0
        log(f'[hard-numba] DOP853 完成：{len(t)} 输出点，{nfev} 次 RHS 求值'
            f'（{n_acc} 接受步 / {n_rej} 拒绝步）'
            + (f'（微波载波 {max((abs(w) for _, w, _ in mw_terms), default=0):.4g}'
               ' MHz 硬积分，零微波近似）'))
        report = dict(n_segments=n_acc, max_richardson_diff=0.0,
                      dt_min_hit=False,
                      max_norm_dev=float(np.max(np.abs(pops.sum(axis=1) - 1.0))))
        if report['max_norm_dev'] > 1e-6:
            log(f'[hard-numba] ⚠ 范数偏差 {report["max_norm_dev"]:.2e}'
                '（DOP853 容差不足或 RHS 非厄米）')
        out = dict(t=t, pops=pops, states=states,
                   om_mhz=[float(np.abs(A).max()) for A, _, _ in mw_terms]
                   + [float(np.abs(A_opt).max()) for A_opt, *_ in opt_terms],
                   method='hard', backend='hard-numba-dop853',
                   loss=np.zeros(len(t)), error_report=report)
        if record_amp:
            uT = np.exp(1j * _CONV * E * T)
            out['psi_amp'] = np.conj(uT) * y_out[-1]
        return out

    # ---- driver='scipy'：njit RHS + scipy solve_ivp ----
    from scipy.integrate import solve_ivp
    rhs = build_hard_rhs(fields_meta, T, n_tab=n_tab)
    if progress_arr is not None:
        progress_arr[0] = 0.15
    sol = solve_ivp(_rhs_hard_pack, (0.0, T), psi0v, method='DOP853',
                    t_eval=t, rtol=rtol, atol=atol)
    if not sol.success:
        raise RuntimeError(f'hard(numba) 积分失败: {sol.message}')
    pops = np.abs(sol.y).T ** 2
    if progress_arr is not None:
        progress_arr[0] = 1.0
    nfev = getattr(sol, 'nfev', None)
    log(f'[hard-numba(scipy)] DOP853 完成：{len(t)} 输出点'
        + (f'，{nfev} 次 RHS 求值' if nfev else '')
        + (f'（微波载波 {max((abs(w) for _, w, _ in mw_terms), default=0):.4g}'
           ' MHz 硬积分，零微波近似）'))
    report = dict(n_segments=1, max_richardson_diff=0.0, dt_min_hit=False,
                  max_norm_dev=float(np.max(np.abs(pops.sum(axis=1) - 1.0))))
    if report['max_norm_dev'] > 1e-6:
        log(f'[hard-numba] ⚠ 范数偏差 {report["max_norm_dev"]:.2e}'
            '（DOP853 容差不足或 RHS 非厄米）')
    out = dict(t=t, pops=pops, states=states,
               om_mhz=[float(np.abs(A).max()) for A, _, _ in mw_terms]
               + [float(np.abs(A_opt).max()) for A_opt, *_ in opt_terms],
               method='hard', backend='hard-numba-dop853',
               loss=np.zeros(len(t)), error_report=report)
    if record_amp:
        uT = np.exp(1j * _CONV * E * T)
        out['psi_amp'] = np.conj(uT) * sol.y[:, -1]
    return out



# ============================================================
# 打包：两体场项（A⊗I / I⊗A 三元组）+ V_dd（按 dE 分组）
# ============================================================
# V_dd 保留阈值（相对 max|V|）：|V| < 此比例的元素（Forster 型 |60S,60P>↔|6P,6P>
# 与 THz 型 60S↔6P/60P↔6P，实测占 5184 中 5040 个、|V| ≤ 1e-4 MHz）其离共振净
# 效应 ≤ V²/Δ ≤ 1e-8 MHz，集体位移 ≤ ~1e-7 MHz ≪ rtol——硬算表示下省略
# （每元素逐个 exp 相位是 RHS 主开销：5184 MACs ≈ 59µs/RHS）。144 个物理活性
# 元素全保留（|dE|·C ≤ 218 rad/ns，精确相位）。
_VDD_MIN_FRAC = 1e-4


def _embed_triplets(A, side, N, mask=None):
    """单原子 A（N×N）→ 两体嵌入三元组 (rows, cols, vals[, signs])。

    side='A'：A⊗I（k=(i,j)→i·N+j，仅 i 变化）；side='B'：I⊗A（仅 j 变化）。
    返回按 row 排序（写顺序化，numba 标量循环更快）。
    mask 提供时（光场拍频）：signs[每元素] = mask[单原子行]−mask[单原子列]
    ∈{±1,0}——旋转系内拍频相位 exp(iC·δ·t·sign) 的符号（t10：exp 非 cos）。
    """
    ii, jj = np.nonzero(np.abs(np.asarray(A, float)) > 1e-15)
    rows, cols, vals = [], [], []
    has_mask = mask is not None
    signs = [] if has_mask else None
    if side == 'A':
        for a, b in zip(ii, jj):
            s = mask[a] - mask[b] if has_mask else 0
            for j in range(N):
                rows.append(a * N + j)
                cols.append(b * N + j)
                vals.append(float(A[a, b]))
                if has_mask:
                    signs.append(s)
    else:
        for a, b in zip(ii, jj):
            s = mask[a] - mask[b] if has_mask else 0
            for i in range(N):
                rows.append(i * N + a)
                cols.append(i * N + b)
                vals.append(float(A[a, b]))
                if has_mask:
                    signs.append(s)
    rows = np.asarray(rows, np.int64)
    cols = np.asarray(cols, np.int64)
    vals = np.asarray(vals, float)
    if rows.size > 1:
        o = np.argsort(rows, kind='stable')
        rows = rows[o]
        cols = cols[o]
        vals = vals[o]
        if has_mask:
            signs = np.asarray(signs, np.int8)[o]
    if has_mask:
        return rows, cols, vals, np.asarray(signs, np.int8)
    return rows, cols, vals


def _pack_pair_hard(fields_meta_A, fields_meta_B, states, T, vdd_sp, n_tab):
    """→ (E_A, E_B, fld_rows, fld_cols, fld_vals, fld_envs, fld_phase_tabs,
         fld_ptr, fld_deltas, fld_sgns,
         vdd_rows, vdd_cols, vdd_vals, vdd_grp_start, vdd_grp_dE, t0, dt, CONV)
    E_A/E_B：每原子 IP 能量（各自减 min）；fld_*：全部场项三元组
    （光场相位表=0 → cos(0)=1，微波相位表=C·ω·t 或 phase_fn——t18 统一
    cos(φ_f(t)) 任意相位）；fld_deltas：光场拍频 δ=ω_f−ω_ref（微波/首场=0）；
    fld_sgns：每三元组相位符号 m[i]−m[k]（δ=0 项为 0）；
    vdd_*：V_dd 按 dE 排序分组。
    """
    N = len(states)
    E_lab = np.array([s['E_mhz'] for s in states], float)
    E_A = np.asarray(fields_meta_A['E_rot'], float)
    E_B = np.asarray(fields_meta_B['E_rot'], float)
    E_A = E_A - float(E_A.min())
    E_B = E_B - float(E_B.min())
    E_lab_pair = np.array([E_lab[i] + E_lab[j]
                           for i in range(N) for j in range(N)])

    # ---- 场项 ----
    fld_rows_l, fld_cols_l, fld_vals_l, fld_env_l = [], [], [], []
    fld_phase_l, fld_delta_l, fld_sgn_l = [], [], []
    for side, meta in (('A', fields_meta_A), ('B', fields_meta_B)):
        for tup in meta['opt_terms']:
            A_opt, wf = tup[0], tup[1]
            omega_f = float(tup[2])
            omega_ref = float(tup[4]) if len(tup) > 4 else omega_f
            mask = tup[5] if len(tup) > 5 else None
            delta = omega_f - omega_ref
            if delta != 0.0 and mask is not None:
                r, c, v, sg = _embed_triplets(A_opt, side, N, mask)
                fld_sgn_l.append(sg)
            else:
                r, c, v = _embed_triplets(A_opt, side, N)
                fld_sgn_l.append(np.zeros(r.size, np.int8))
            fld_rows_l.append(r)
            fld_cols_l.append(c)
            fld_vals_l.append(v)
            fld_env_l.append(_env_tab(wf, T, n_tab))
            fld_phase_l.append(np.zeros(n_tab))        # 光场：cos(0)=1
            fld_delta_l.append(delta)
        for g in meta['mw_groups']:
            w = float(g['omega_mhz'])
            ph_fn = g.get('phase_fn')
            for A, wf in zip(g['As'], g['envs']):
                r, c, v = _embed_triplets(A, side, N)
                fld_rows_l.append(r)
                fld_cols_l.append(c)
                fld_vals_l.append(v)
                fld_env_l.append(_env_tab(wf, T, n_tab))
                if ph_fn is None:
                    ts = np.linspace(0.0, float(T), n_tab)
                    fld_phase_l.append(_CONV * w * ts)   # 定频 φ=C·ω·t
                else:
                    fld_phase_l.append(_env_tab(ph_fn, T, n_tab))
                fld_delta_l.append(0.0)
                fld_sgn_l.append(np.zeros(r.size, np.int8))
    n_terms = len(fld_rows_l)
    if n_terms:
        fld_rows = np.concatenate(fld_rows_l)
        fld_cols = np.concatenate(fld_cols_l)
        fld_vals = np.concatenate(fld_vals_l)
        fld_envs = np.stack(fld_env_l)
        fld_phase_tabs = np.stack(fld_phase_l)
        fld_deltas = np.asarray(fld_delta_l, float)
        fld_sgns = np.concatenate(fld_sgn_l)
        ptr = np.zeros(n_terms + 1, np.int64)
        for k, r in enumerate(fld_rows_l):
            ptr[k + 1] = ptr[k] + r.size
    else:
        fld_rows = np.zeros(0, np.int64)
        fld_cols = np.zeros(0, np.int64)
        fld_vals = np.zeros(0, float)
        fld_envs = np.zeros((0, n_tab))
        fld_phase_tabs = np.zeros((0, n_tab))
        fld_deltas = np.zeros(0)
        fld_sgns = np.zeros(0, np.int8)
        ptr = np.zeros(1, np.int64)

    # ---- V_dd：相位扭曲 + 按 dE 分组 ----
    # 分组：dE 精确量化（δ=1e-6 MHz）——过滤后元素全为物理活性（|V|≥1e-4·max，
    # 实测 |dE|·C ≤ 218 rad/ns），相位须精确；组内按 row 排序（顺序写）。
    # 性能：过滤后 5184→144 元素 → V_dd MAC 从 ~59µs 降到 ~4µs/RHS。
    if vdd_sp is None:
        vdd_rows = np.zeros(0, np.int64)
        vdd_cols = np.zeros(0, np.int64)
        vdd_vals = np.zeros(0, complex)
        grp_start = np.zeros(1, np.int64)
        grp_dE = np.zeros(0, float)
    else:
        V = vdd_sp.tocsr().astype(complex)
        rows, cols = V.nonzero()
        vals = np.asarray(V.data, complex)
        vmax = np.max(np.abs(vals)) if vals.size else 0.0
        if vmax > 0.0:
            keep = np.abs(vals) >= _VDD_MIN_FRAC * vmax
            rows = rows[keep]
            cols = cols[keep]
            vals = vals[keep]
        if rows.size == 0:
            vdd_rows = np.zeros(0, np.int64)
            vdd_cols = np.zeros(0, np.int64)
            vdd_vals = np.zeros(0, complex)
            grp_start = np.zeros(1, np.int64)
            grp_dE = np.zeros(0, float)
        else:
            dE = E_lab_pair[rows] - E_lab_pair[cols]
            key = np.round(dE * 1e6)          # δ=1e-6 MHz
            order = np.lexsort((rows, key))   # 按 (组, row) 排序
            rows = rows[order]
            cols = cols[order]
            vals = vals[order]
            key = key[order]
            bounds = np.flatnonzero(np.diff(key) != 0.0)
            grp_start = np.concatenate([[0], bounds + 1, [rows.size]])
            grp_dE = dE[order][grp_start[:-1]]
            vdd_rows = rows
            vdd_cols = cols
            vdd_vals = vals
    t0 = 0.0
    dt = float(T) / (n_tab - 1) if n_tab > 1 else 1.0
    CONV = float(_CONV)
    return (E_A, E_B, fld_rows, fld_cols, fld_vals, fld_envs,
            fld_phase_tabs, ptr,
            fld_deltas, fld_sgns,
            vdd_rows, vdd_cols, vdd_vals, grp_start, grp_dE, t0, dt, CONV)


# ============================================================
# njit 内核（_env_at 查表来自 dop853 核心层）
# ============================================================
@njit(cache=True)
def _rhs_hard_pair(t, y, E_A, E_B, fld_rows, fld_cols, fld_vals, fld_envs,
                   fld_phase_tabs, fld_ptr, fld_deltas, fld_sgns,
                   vdd_rows, vdd_cols, vdd_vals,
                   vdd_grp_start, vdd_grp_dE, t0t, dtt, CONV):
    """两体 hard RHS：y' = −iC·[u∘(Ṽ·(conj(u)∘y)) + V_dd_twisted·y]。

    光场拍频：δ=ω_f−ω_ref ≠ 0 的项乘 exp(iC·δ·t·sgn)（sgn=每三元组
    相位符号，m[i]−m[k]；δ=0 逐位退化——t10 验证 exp 拍频正确）。
    微波（t18）：cos(φ_f(t)) 相位表插值——定频 φ=C·ω·t（光场相位表=0
    → cos(0)=1），变频/任意相位由 phase_fn 采样。
    """
    N = E_A.size
    N2 = N * N
    uA = np.exp(1j * CONV * E_A * t)
    uB = np.exp(1j * CONV * E_B * t)
    # zu[k] = conj(uA[i])·conj(uB[j])·y[k]
    zu = np.empty(N2, np.complex128)
    for i in range(N):
        ui = uA[i].conjugate()
        for j in range(N):
            zu[i * N + j] = ui * uB[j].conjugate() * y[i * N + j]
    # 场项累加器（后续 ×u_pair）与 V_dd 累加器（相位已含 u·conj(u)，
    # **不再**乘 u_pair）分开——混用会把 V_dd 项多乘一个 u[k]。
    dy_f = np.zeros(N2, np.complex128)
    dy_v = np.zeros(N2, np.complex128)
    # 场项（逐项三元组；光场相位表=0 → cos(0)=1，微波相位表=φ_f(t)）
    n_terms = fld_ptr.size - 1
    for ti in range(n_terms):
        c = _env_at(fld_envs[ti], t, t0t, dtt)
        cw = c * np.cos(_env_at(fld_phase_tabs[ti], t, t0t, dtt))
        if cw != 0.0:
            d = fld_deltas[ti]
            if d != 0.0:
                p1 = np.exp(1j * CONV * d * t)
                p2 = p1.conjugate()
                for m in range(fld_ptr[ti], fld_ptr[ti + 1]):
                    ph = p1 if fld_sgns[m] > 0 else p2
                    dy_f[fld_rows[m]] += fld_vals[m] * cw * ph * zu[fld_cols[m]]
            else:
                for m in range(fld_ptr[ti], fld_ptr[ti + 1]):
                    dy_f[fld_rows[m]] += fld_vals[m] * cw * zu[fld_cols[m]]
    # V_dd（相位扭曲按 dE 分组：每组一个 exp；相位 e^{iC·(E_lab[k]−E_lab[l])t}
    # 已含旋转系 Ω-twist 与 IP u-twist 的合并，无额外 u 因子）
    n_grp = vdd_grp_dE.size
    for g in range(n_grp):
        ph = np.exp(1j * CONV * vdd_grp_dE[g] * t)
        for m in range(vdd_grp_start[g], vdd_grp_start[g + 1]):
            dy_v[vdd_rows[m]] += vdd_vals[m] * ph * y[vdd_cols[m]]
    # out = −iC·[u_pair∘dy_f + dy_v]
    out = np.empty(N2, np.complex128)
    for i in range(N):
        ui = uA[i]
        for j in range(N):
            k = i * N + j
            out[k] = -1j * CONV * (ui * uB[j] * dy_f[k] + dy_v[k])
    return out

# ============================================================
# DOP853 驱动已上收至 src/core/dop853.py（first-class RHS 参数化，
# t11 实验验证与旧两体驱动逐位一致）。本模块只保留内核与打包。
# ============================================================



# ============================================================
# 主入口
# ============================================================
def evolve_hard_pair(states, fields_meta_A, fields_meta_B, psi0_A, psi0_B,
                     t_grid, vdd_matrix=None, R_um=None, theta_deg=0.0,
                     phi_deg=0.0, cache=None, rtol=1e-9, atol=1e-11,
                     progress_arr=None, log=None, record_amp=False,
                     n_tab=_TAB_N, decays_A=None, decays_B=None,
                     driver='numba'):
    """两体 hard（第一性原理 DOP853 硬积分，零微波近似）。

    返回契约与 evolve_exact_pair 同键：t, pops_2d, pops_A, pops_B,
    loss_A, loss_B, states_2d, method='hard', backend='hard-numba-pair',
    error_report。
    """
    from . import pair_core as _pc
    log = log or (lambda *a: None)
    t = np.asarray(t_grid, float)
    T = float(t[-1])
    N = len(states)
    N2 = N * N

    # ---- V_dd ----
    if vdd_matrix is not None:
        V = vdd_matrix
    elif R_um is not None:
        if cache is None:
            raise ValueError('两体 hard 需要 cache（core.ArcCache 实例）'
                             '来计算 V_dd；请传入 cache 或显式 vdd_matrix。')
        V = _pc.vdd_sparse(cache, [dict(s) for s in states],
                           R_um, theta_deg, phi_deg)
    else:
        V = None

    # ---- 耗散（寿命通道 → NonHermitian 对角，同 evolve_exact_pair）----
    GammaA = GammaB = None
    has_decay = bool(decays_A or decays_B)
    if has_decay:
        GammaA, GammaB, _ = _pe._pair_loss_vectors(
            states, decays_A, decays_B, N, N2)
    gamma_MHz = np.zeros(N2)
    if has_decay:
        gamma_MHz = (np.asarray(GammaA, float)
                     + np.asarray(GammaB, float)) / _CONV

    # ---- 初态 ----
    psi0_A_vec = _evolve._as_vector(states, psi0_A)
    psi0_B_vec = _evolve._as_vector(states, psi0_B)
    psi0_2d = _pe._as_product_psi(psi0_A_vec, psi0_B_vec, N)

    pack = _pack_pair_hard(fields_meta_A, fields_meta_B, states, T, V, n_tab)
    E_A, E_B = pack[0], pack[1]
    E_lab = np.array([s['E_mhz'] for s in states], float)
    E_pair = np.array([E_lab[i] + E_lab[j]
                       for i in range(N) for j in range(N)])

    if driver == 'numba':
        prog = (np.zeros(2) if progress_arr is None
                else np.asarray(progress_arr, float))
        stat = np.zeros(4)
        # 复用 dop853 核心层（first-class RHS 参数化，t11 验证逐位一致）
        y_out = _dop853_driver(_rhs_hard_pair, psi0_2d, t, T,
                               float(rtol), float(atol), np.inf,
                               prog, stat, *pack)
        nfev = int(stat[0])
        n_acc = int(stat[1])
        n_rej = int(stat[2])
        if progress_arr is not None:
            progress_arr[0] = 1.0
    else:
        # driver='scipy'：njit RHS + scipy solve_ivp（对照用）
        from scipy.integrate import solve_ivp

        def _rhs_hard_pack(tv, y):
            return _rhs_hard_pair(tv, y, *pack)
        sol = solve_ivp(_rhs_hard_pack, (0.0, T), psi0_2d, method='DOP853',
                        t_eval=t, rtol=rtol, atol=atol)
        if not sol.success:
            raise RuntimeError(f'hard-pair(scipy) 积分失败: {sol.message}')
        y_out = sol.y.T
        nfev = int(getattr(sol, 'nfev', 0))
        n_acc = -1

    pops_2d = np.abs(y_out) ** 2

    # ---- loss（NonHermitian 速率积分，同 pair_evolve 语义）----
    loss_A = np.zeros(len(t))
    loss_B = np.zeros(len(t))
    if has_decay:
        rate_A = pops_2d @ GammaA
        rate_B = pops_2d @ GammaB
        dt_arr = np.diff(t)
        loss_A[1:] = np.cumsum(0.5 * (rate_A[1:] + rate_A[:-1]) * dt_arr)
        loss_B[1:] = np.cumsum(0.5 * (rate_B[1:] + rate_B[:-1]) * dt_arr)
        norm_dev = float(np.max(np.abs(
            pops_2d.sum(axis=1) + loss_A + loss_B - 1.0)))
    else:
        norm_dev = float(np.max(np.abs(pops_2d.sum(axis=1) - 1.0)))

    pops_A = _pe._reduce_A(pops_2d, N)
    pops_B = _pe._reduce_B(pops_2d, N)
    report = dict(n_segments=n_acc, max_richardson_diff=0.0, dt_min_hit=False,
                  max_norm_dev=norm_dev, nfev=nfev)
    if report['max_norm_dev'] > 1e-6:
        log(f'[hard-pair] ⚠ 范数偏差 {report["max_norm_dev"]:.2e}')
    log(f'[hard-pair] DOP853 完成：{len(t)} 输出点，{nfev} 次 RHS 求值'
        f'（N²={N2} 维，硬积分零微波近似）')
    out = dict(
        t=t, pops_2d=pops_2d, pops_A=pops_A, pops_B=pops_B,
        loss_A=loss_A, loss_B=loss_B,
        states_2d=[{
            'label': f"({states[i]['label']}) x ({states[j]['label']})",
            'E_mhz': E_pair[i * N + j],
            'ket_i': i, 'ket_j': j,
        } for i in range(N) for j in range(N)],
        method='hard', backend='hard-numba-pair',
        error_report=report)
    if record_amp:
        uT = np.exp(1j * _CONV * (E_A[:, None] + E_B[None, :]).ravel() * T)
        out['psi_amp'] = np.conj(uT) * y_out[-1]
    return out
