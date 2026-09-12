# -*- coding: utf-8 -*-
"""DOP853 核心层（与态空间维度 N 无关）——重构第 2a 步。

从 hard_numba.py 抽取共享内核：两处驱动（单体/两体）的步进/误差/步长/
稠密输出逻辑逐行同构，仅 RHS 调用不同。numba 支持 njit 函数作为 first-class
参数（_numba_funcarg_test.py 已验证），故统一为 _dop853_driver(rhs, ...)，
rhs 为任意 njit 编译的 RHS 回调（单体/两体通用）。

⚠ 缓存约束（t76，踩过坑，勿随意回退）：
  把 jit 函数当**实参**传给另一个 jit 函数，会让 numba 的缓存键含 Dispatcher
  类型（WeakType，键 = weakref ≈ 对象 id）→ 跨进程永不命中；而"把 jit 函数
  当实参传递给未内联的 jit 函数"又会把调用方标记为 has_dynamic_globals →
  调用方**完全不可缓存**。故本模块三个内核全部 `inline='always'`、附加参数
  走**元组**（内联器不支持 `*args`），并由引擎侧的薄包装静态绑定各自 RHS
  （coupling.dop853_driver_mixed / hard.dop853_driver_hard）——包装签名只剩
  数组/标量 → 键稳定 → 真正命中。详见那两个包装的注释。

- 系数（_A/_C/_B/_E3/_E5/_D）：scipy 1.15.2 逐行移植、2.9e-10 基线验证；
  t76 从 hard.py 上收至本模块（原先反向 import 造成 dop853 ⇄ hard 循环依赖）。
- _rms_norm                 ：范数（纯工具；_env_at/_env_tab 单源在 hard.py）
- _dop853_step / _dop853_dense_interp：单步与稠密插值（与 RHS 参数化解耦）
- _dop853_driver                       ：主循环（rhs 参数化）

使用方：coupling.dop853_driver_mixed（mixed 单/两体）、
        hard.dop853_driver_hard（hard 单/两体）。
"""
import numpy as np
from numba import njit

from .params import _TAB_N  # t74c 参数单源

_np = np

# ============================================================
# DOP853 系数（Butcher 表）：scipy 1.15.2 dop853_coefficients.py 逐行移植。
# 与积分器同源，故与驱动同居本模块——原先借住在 hard.py，造成
# dop853 ⇄ hard 循环导入，只能靠"导入顺序"维持（t76 上收）。
# 禁止手写重建（曾因手写系数 54 处错误）。
# ============================================================
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


@njit(cache=True, nogil=True)
def _rms_norm(x):
    """RMS 范数 sqrt(Σ|x|²/n)（scipy norm 移植）。"""
    n = x.size
    s = 0.0
    for i in range(n):
        s += x[i].real * x[i].real + x[i].imag * x[i].imag
    return _np.sqrt(s / n)


@njit(cache=True, nogil=True, inline='always')
def _dop853_step(t, y, f, h, K, rhs, rargs):
    """单步：12 级显式 RK（rhs 为 njit 回调，rargs 为其附加参数元组）。"""
    n = y.size
    for i in range(n):
        K[0, i] = f[i]
    for s in range(1, 13):
        dy = _np.zeros(n, _np.complex128)
        for j in range(s):
            a = _A[s, j]
            if a != 0.0:
                for i in range(n):
                    dy[i] += a * K[j, i]
        for i in range(n):
            dy[i] *= h
        K[s] = rhs(t + _C[s] * h, y + dy, *rargs)
    y_new = _np.empty(n, _np.complex128)
    for i in range(n):
        acc = y[i]
        for j in range(12):
            b = _B[j]
            if b != 0.0:
                acc += h * b * K[j, i]
        y_new[i] = acc
    f_new = rhs(t + h, y_new, *rargs)
    for i in range(n):
        K[12, i] = f_new[i]
    return y_new, f_new


@njit(cache=True, nogil=True, inline='always')
def _dop853_dense_interp(y_old, y_new, f_old, f_new, K, t_old, h, x,
                         rhs, rargs):
    """Hairer contd8 稠密输出：补算 13..15 级 + 7 次多项式插值（scipy 移植，
    与 hard_numba 原版逐行等价；rhs 参数化）。"""
    n = y_old.size
    for s in range(13, 16):
        dy = _np.zeros(n, _np.complex128)
        for j in range(s):
            a = _A[s, j]
            if a != 0.0:
                for i in range(n):
                    dy[i] += a * K[j, i]
        for i in range(n):
            dy[i] *= h
        K[s] = rhs(t_old + _C[s] * h, y_old + dy, *rargs)
    F = _np.zeros((7, n), _np.complex128)
    for i in range(n):
        d = y_new[i] - y_old[i]
        F[0, i] = d
        F[1, i] = h * f_old[i] - d
        F[2, i] = 2.0 * d - h * (f_new[i] + f_old[i])
    for r in range(4):
        for j in range(16):
            dij = _D[r, j]
            if dij != 0.0:
                for i in range(n):
                    F[3 + r, i] += h * dij * K[j, i]
    y = _np.zeros(n, _np.complex128)
    for i in range(6, -1, -1):
        for k in range(n):
            y[k] += F[i, k]
        if (6 - i) % 2 == 0:
            for k in range(n):
                y[k] *= x
        else:
            for k in range(n):
                y[k] *= (1.0 - x)
    for k in range(n):
        y[k] += y_old[k]
    return y


# inline='always' 是**缓存正确性**要求，不是性能偏好：
# 本函数把 rhs 当参数，numba 会把"把 jit 函数当实参传递"记成动态全局
# （has_dynamic_globals）→ 任何调用它的非内联函数都会被拒绝缓存
# （实测警告：Cannot cache ... as it uses dynamic globals）。强制内联后，
# 调用方（coupling.dop853_driver_mixed / hard.dop853_driver_hard）里的
# rhs 实参变成对模块级 jit 函数的**直接全局调用**，于是可缓存且键稳定。
# 本函数自身因此不再产生缓存条目（它被完整内联进薄包装）。
@njit(cache=True, nogil=True, inline='always')
def _dop853_driver(rhs, y0, t_eval, T, rtol, atol, max_step, prog, stat,
                   dt_hist, rargs, max_step_hard=0.0):
    """scipy solve_ivp(DOP853) 主循环的 numba 移植（rhs 参数化，任意 N）。

    rhs(t, y, *rargs) -> dydt（njit 编译）。rargs 承载场矩阵/查表等附加参数。
    返回 y_out (n_eval, n) 复数；stat[0..3] = nfev/n_acc/n_rej/n_dense。
    dt_hist：None 或预分配 (M,2) float64——每接受步写 (t_new, h)，行数记
    stat[4]（t75 诊断；调用方传 dt_hist 时 stat 须 ≥6 行）。None=不记录。
    步进/误差/步长控制与 scipy rk.py 逐行一致（SAFETY=0.9 等）。

    max_step_hard（2026-09-12 加，运行级 `solve.MAX_STEP_HARD`）：
    **每个**接受步都复核的硬上限（ns）；0.0 = 不限制 → 与改前逐位一致。
    为什么要单开一个键：`max_step` 只在**起步**参与
    `h_abs = min(100·h0, h1, interval, max_step)`，之后 `h_abs *= factor`
    按误差控制自由增长（实测：请求 0.05 ns 的 10 ns 演化里出现 **2.2 ns** 的
    接受步）——它是"初始步长提示"，不是上限。需要真压步长（例如怀疑欠采样）
    时显式给 max_step_hard，代价是步数/耗时上升。
    """
    n = y0.size
    n_eval = t_eval.size
    SAFETY = 0.9
    MIN_FACTOR = 0.2
    MAX_FACTOR = 10.0
    EXP = -1.0 / 8.0
    t0 = 0.0
    K = _np.zeros((16, n), _np.complex128)
    y_out = _np.zeros((n_eval, n), _np.complex128)
    nfev = 0
    n_acc = 0
    n_rej = 0
    n_dense = 0

    y = y0.copy()
    f = rhs(t0, y, *rargs)
    nfev += 1
    i_eval = 0
    while i_eval < n_eval and t_eval[i_eval] <= t0:
        for k in range(n):
            y_out[i_eval, k] = y[k]
        i_eval += 1
    if i_eval >= n_eval or t0 >= T:
        stat[0] = nfev
        stat[1] = n_acc
        stat[2] = n_rej
        stat[3] = n_dense
        return y_out

    # ---- select_initial_step（scipy common.py，order=7）----
    scale = _np.empty(n)
    for i in range(n):
        scale[i] = atol + _np.abs(y[i]) * rtol
    d0 = _rms_norm(y / scale)
    d1 = _rms_norm(f / scale)
    interval = T - t0
    if d0 < 1e-5 or d1 < 1e-5:
        h0 = 1e-6
    else:
        h0 = 0.01 * d0 / d1
    if h0 > interval:
        h0 = interval
    y1 = y + h0 * f
    f1 = rhs(t0 + h0, y1, *rargs)
    nfev += 1
    d2 = _rms_norm((f1 - f) / scale) / h0
    if d1 <= 1e-15 and d2 <= 1e-15:
        h1 = max(1e-6, h0 * 1e-3)
    else:
        h1 = (0.01 / max(d1, d2)) ** (1.0 / 8.0)
    h_abs = min(100.0 * h0, h1, interval, max_step)
    if max_step_hard > 0.0 and h_abs > max_step_hard:
        h_abs = max_step_hard          # 硬上限：起步步长也压住

    # ---- 主循环 ----
    t = t0
    # t72 进度映射：自适应步长下 t/T 与墙钟严重不符（stiff 脉冲区步长比
    # 自由区小 1-2 个量级 → 条在 stiff 区爬、末段瞬间冲满）。nogil 内核
    # 中每次 RHS 求值开销恒定 → nfev ∝ 墙钟。用平滑的局部求值密度 rate
    # （RHS 次数/模拟时间，EMA 0.7/0.3）外推剩余求值数：
    #     p = nfev / (nfev + rate·(T−t))
    # 条的移动速度跟随真实墙钟速度，未来变贵/变便宜时自动减速/加速；
    # p 单调不减（钳制）、封顶 0.999（收尾阶段接管最后一段）。
    rate_est = 0.0
    prev_t = 0.0
    prev_nfev = 1
    p_max = 0.0
    stage_set = False
    while t < T and i_eval < n_eval:
        step_accepted = False
        step_rejected = False
        while not step_accepted:
            h = h_abs
            t_new = t + h
            if t_new > T:
                t_new = T
                h = t_new - t
            y_new, f_new = _dop853_step(t, y, f, h, K, rhs, rargs)
            nfev += 12
            # 误差估计（scipy _estimate_error_norm）
            err5_n2 = 0.0
            err3_n2 = 0.0
            for i in range(n):
                sc = atol + max(_np.abs(y[i]), _np.abs(y_new[i])) * rtol
                e5 = 0.0j
                e3 = 0.0j
                for j in range(13):
                    e5 += _E5[j] * K[j, i]
                    e3 += _E3[j] * K[j, i]
                e5 = e5 / sc
                e3 = e3 / sc
                err5_n2 += e5.real * e5.real + e5.imag * e5.imag
                err3_n2 += e3.real * e3.real + e3.imag * e3.imag
            if err5_n2 == 0.0 and err3_n2 == 0.0:
                error_norm = 0.0
            else:
                error_norm = (_np.abs(h) * err5_n2
                              / _np.sqrt((err5_n2 + 0.01 * err3_n2) * n))
            if error_norm < 1.0:
                if error_norm == 0.0:
                    factor = MAX_FACTOR
                else:
                    factor = min(MAX_FACTOR, SAFETY * error_norm ** EXP)
                if step_rejected:
                    factor = min(1.0, factor)
                h_abs *= factor
                if max_step_hard > 0.0 and h_abs > max_step_hard:
                    h_abs = max_step_hard      # 接受步后立即压回硬上限
                step_accepted = True
            else:
                h_abs *= max(MIN_FACTOR, SAFETY * error_norm ** EXP)
                if max_step_hard > 0.0 and h_abs > max_step_hard:
                    h_abs = max_step_hard      # 拒绝分支同样压住
                step_rejected = True
                n_rej += 1
        h_step = h
        n_acc += 1
        # t75：步长历史（诊断）——接受步写 (t_new, h)，行数记 stat[4]
        if dt_hist is not None:
            m = int(stat[4])
            if m < dt_hist.shape[0]:
                dt_hist[m, 0] = t_new
                dt_hist[m, 1] = h_step
                stat[4] = m + 1
        # t72：首个接受步后置阶段=1（积分）——驱动调用内的 numba 装载/
        # 编译期保持调用侧的阶段 0（"准备"标签 + 墙钟耗时），GUI 不失真
        if (not stage_set) and prog.shape[0] > 2:
            prog[2] = 1.0
            stage_set = True
        # t72：预测剩余工作量的进度映射（见循环前注释）
        dt_loc = t_new - prev_t
        if dt_loc > 0.0:
            inst = (nfev - prev_nfev) / dt_loc
            if rate_est == 0.0:
                rate_est = inst
            else:
                rate_est = 0.7 * rate_est + 0.3 * inst
        prev_t = t_new
        prev_nfev = nfev
        denom = nfev + rate_est * (T - t_new)
        p_now = nfev / denom if denom > 0.0 else 0.0
        if p_now > 0.999:
            p_now = 0.999
        if p_now < p_max:
            p_now = p_max
        p_max = p_now
        prog[0] = p_now
        # t21：取消支持——GUI 置 prog[1]≠0 时抛 RuntimeError（numba njit 内
        # raise 已验证兼容），上层（worker）按 progress_arr[1]≠0 → 'cancelled'
        if prog[1] != 0.0:
            raise RuntimeError('cancelled')
        # 稠密输出：记录 (t, t_new] 内的 t_eval 点
        if i_eval < n_eval and t_eval[i_eval] <= t_new:
            dense_done = False
            while i_eval < n_eval and t_eval[i_eval] <= t_new:
                x = (t_eval[i_eval] - t) / h_step
                if x <= 0.0:
                    yq = y
                elif x >= 1.0:
                    yq = y_new
                else:
                    if not dense_done:
                        dense_done = True
                        n_dense += 1
                        nfev += 3
                    yq = _dop853_dense_interp(y, y_new, f, f_new, K, t,
                                              h_step, x, rhs, rargs)
                for k in range(n):
                    y_out[i_eval, k] = yq[k]
                i_eval += 1
        t = t_new
        y = y_new
        f = f_new
    stat[0] = nfev
    stat[1] = n_acc
    stat[2] = n_rej
    stat[3] = n_dense
    return y_out
