# -*- coding: utf-8 -*-
"""多体配置契约（`manybody.*` 命名空间）+ 逐原子真校验。

与 core/protocol.py 的 CONFIG_SCHEMA 同角色，但只声明多体页的键；由 protocol
合并进 config_schema()（键在多体、单/两体互不打扰）。值域/必填投影自 build_run
自身的解析，`validate_pauli_strings` / `validate_calib` 在 session.set **当场**
把橡皮图章换成真报错。

单位与全库一致：位置 µm、Ω/δ/Γ/V/J/A MHz、时间 ns、B Gauss。
"""
from ..param import Param
from .backends import time_backend_names, equilibrium_backend_names, MPS_DTYPES

__all__ = ['CONFIG_SCHEMA_MB', 'SOLVE_META_MB', 'mb_get',
           'validate_drive_segments', 'validate_loops', 'validate_link_pairs',
           'validate_pauli_strings', 'MB_N_WINDOW', 'MB_GUARD_MHZ',
           'MB_ED_MAX_DIM']

# 标定/维度默认（就地 Param：值+元数据同处，被 protocol.engine_params 扫出；
# 同时作为 CONFIG_SCHEMA_MB 各键 default 的单一来源）。
MB_N_WINDOW = Param(3, '单原子基伙伴态窗口 |Δn|≤n_window（补 vdW 中间态）',
                    tier='run')
MB_GUARD_MHZ = Param(1.0, 'Löwdin downfolding 近简并中间态剔除阈',
                     unit='MHz', lo=0.0, tier='run')
MB_ED_MAX_DIM = Param(1 << 20, 'ED 有效基维度上限（solver=auto 据此判可行）',
                      lo=2, tier='run')

# `manybody.*` → build_run 读的登记册快照键。type='mb_psi' 是多体专有类型：
# 结构灵活，由 resolve_mb_psi 在 build 阶段校验（需原子数上下文，set 阶段拿不到）。
CONFIG_SCHEMA_MB = {
    'manybody.calib': dict(
        type='dict', default=None,
        desc='**多体唯一的模型输入**：{atoms:[{id, pos:(x,y,z)µm, 可选 Omega_mhz/'
             'Omega_phase_rad/delta_mhz/gamma_mhz}], '
             'pairs:[{i,j,V,J,A}](MHz)}。'
             '活跃通道由 pairs 的数值推导（V≠0→I1，J≠0→I2，A≠0→I3），不接受 terms；'
             '也不接受 file/arc——文件名或 ARC 参数请先用 el_build_calib 转成这里的数。'),
    'manybody.build': dict(
        type='dict', default=None,
        desc='**装配入参**（与 el_build_calib 的参数**同一份**，不是备注）：'
             '{source:"file", file:"x.json"} 或 '
             '{source:"arc", atoms:[{g,r,pos,…}], arc:{B_gauss,n_window,r_cut_um,'
             'second_order,guard_mhz,periods_um}}。'
             '构造器按它产出 calib；build_run/evolve_run 一行都不读。'
             '回执（指纹/尾界/丢对…）由 calib 现算，不另存，避免与真实产物漂移。'),
    'manybody.label': dict(
        type='str', default=None,
        desc='一句话任务名（**只用于卡片标题/检索**，不参与运行）。别塞长文本，'
             '细节写进 manybody.notes 或 artifact。'),
    'manybody.notes': dict(
        type='str', default=None,
        desc='简短备注（**只用于卡片折叠区**，不参与运行）：为什么跑这一组、引用等。'),
    'manybody.source': dict(
        type='dict', default=None,
        desc='源态谱的源（want=spectrum；省略时默认用 manybody.psi0 作 |0⟩，'
             'psi0 也为空则用基态），在**当前基**里给 φ 或算符：'
             '{kind:"vector",values} / {kind:"psi0"}（|0⟩ 本身） / {kind:"diagonal",values} / '
             '{kind:"site_weights",weights:[…N…],subtract_mean?:bool}'
             '（O=Σwᵢnᵢ 的对角展开；subtract_mean=true 扣掉 ⟨O⟩，得连通谱） / '
             '{kind:"sparse",rows,cols,values} / {kind:"full_sparse",…}（扇区模式下投影回扇区'
             '并在 error_report.projected_leak 报告被丢弃权重）。引擎不解释算符的物理含义'),
    'manybody.spectrum': dict(
        type='dict', default=None,
        desc='源态谱参数：{eta(MHz 洛伦兹半宽), n_krylov, n_omega, '
             'omega_min, omega_max, reortho}；窗与展宽是显示参数，不是物理结论'),
    'manybody.psi0': dict(
        type='mb_psi', default=None,
        desc="初态：None=全基态|g…g⟩ / int=产物态索引 / 'grg'=模式串 / "
             "'ground'=Lanczos 基态 / ['sup',[[模式或索引,amp],…]]=叠加；"
             "一般局部乘积态可用 {kind:'product', amplitudes:[[a_g,a_r],…]}，"
             "振幅可为复数或 [实部,虚部]；PEPS 可用 {kind:'peps', tensors:[…]} 提供五阶张量；"
             "淬火/制备态用 {kind:'ground_of', calib:{…}, terms:[…]}（初态是另一个模型的基态，"
             "ED 给全空间向量、MPS 给同 order/charge 的原生态）；"
             "分支初态用 {kind:'permutation_sector', base:<任意 psi0>, permutation:[…], sign:±1}"
             "（对合站点置换的 ± 本征扇区，返回投影权重与 <P>）；"
             "Python RunSpec 可用返回的 kind=mps 原生状态接续，跨会话请使用 HDF5 checkpoint"),
    'manybody.return_state': dict(
        type='bool', default=False,
        desc='显式返回基态向量或有限 MPS 状态以供后续演化（tdvp-dense 返回原生 dense MPS）；可能占用额外内存，原始态不放入摘要'),
    'manybody.solver': dict(
        type='str', enum=time_backend_names(), default='auto',
        desc='**含时演化**求解器：auto=小系统 ED、大系统 TDVP / ed=精确对角化 / '
             'tebd=最近邻 MPS / tebd-gpu=CUDA 最近邻 MPS / swap-tebd-gpu=CUDA 任意键 SWAP-TEBD / tdvp=两站点 TDVP / tdvp-single=单站点 TDVP / '
             'tdvp-adaptive=两站点到单站点自适应 TDVP / tdvp-dense=CPU 稠密两站点 TDVP（有限开链，支持 segments/Pauli/原生 MPS return_state；无 charge；与 tdvp 同阶对拍）。'
             '平衡态不看本键，看 `manybody.eq_backend`。'),
    'manybody.collapse_ops': dict(type='list', default=None, desc='开放系统局部跳跃通道：[{site, rate, operator|matrix}]，供 tdvp-trajectories'),
    'manybody.n_trajectories': dict(type='int', default=32, lo=1, hi=100000, desc='量子轨迹条数'),
    'manybody.seed': dict(type='int', default=0, desc='量子轨迹、QMC 与 KPM 的随机种子'),
    'manybody.max_step_ns': dict(type='float', default=0.1, lo=1e-6,
                                 desc='MPS/PEPS/轨迹/经典自旋/tdvp-dense 内部最大步长（ns）；ED 使用指数传播，无此步长旋钮'),
    'manybody.classical_damping': dict(type='float', default=0.0, lo=0.0, desc='classical-spin Landau-Lifshitz 阻尼系数'),
    'manybody.grid_shape': dict(type='list', default=None, desc='PEPS 矩形网格 [rows,cols]'),
    'manybody.idmrg': dict(
        type='dict', default=None,
        desc='无限 MPS / iDMRG 的通用模型规格：{cell:[{...}], terms:[{kind:"onsite",site,op,coeff} 或 {kind:"pair",i,j,dx,op_i,op_j,coeff}], mixer?:bool}。'
             'dx 是 j 相对 i 的单胞偏移；只支持一个平移方向、有限程、局部维数 2。'
             'Rydberg 单元胞可经 from_rydberg_cell 生成；本键与 manybody.calib 互斥。'),
    'manybody.eq_backend': dict(
        type='str', enum=equilibrium_backend_names(),
        default='auto',
        desc='**平衡态**求解后端：auto=按规模路由 / ed=稠密对角 / gpu-batch=批量本征 / '
             'gpu-lanczos=CUDA 矩阵自由 Lanczos / lanczos=稀疏迭代 / kpm=Chebyshev 谱 / dmrg=TeNPy 有限 MPS DMRG / idmrg=无限 MPS iDMRG / '
             'gpu-kpm=CUDA 分块 KPM / free-fermion=适用范围内二次费米子 / qmc=stoquastic SSE（带统计误差）'
             '（**任意点云**：多腿/无周期/逐键复耦合，自写 autoMPO）。'),
    'manybody.want': dict(
        type='str', enum=['', 'gs', 'thermal', 'dos', 'spectrum'], default='',
        desc="''=含时演化（默认）；gs=基态、thermal=有限温（配 manybody.betas）、"
             "dos=态密度、spectrum=源态谱（配 manybody.source）——"
             "非空即走平衡态路径（runspec 委托 eq_engine 路由）。"),
    'manybody.kpm_num': dict(type='int', default=400, lo=2,
        desc='KPM Chebyshev 阶数'),
    'manybody.kpm_nvec': dict(type='int', default=1500, lo=1,
        desc='KPM 随机迹探针数'),
    'manybody.betas': dict(
        type='list', default=None,
        desc='thermal 的逆温度清单（1/MHz），如 [0.1,1,10]；None=[1.0]。'),
    # 平衡态 DMRG 截断控制（只有 backend=dmrg 时消费；不像引擎内部常量，这些是人/AI
    # 要当场权衡精度↔时长的旋钮，故上 UI）。
    'manybody.chi_max': dict(
        type='int', default=200, lo=2, hi=4000,
        desc='MPS/PEPS/量子轨迹最大键维 χ（越大越准越慢；键数 ~χ³）'),
    'manybody.truncerr': dict(
        type='float', default=1e-10, lo=0.0, hi=1e-2,
        desc='MPS 单次截断误差目标（保得越紧越慢）'),
    'manybody.krylov_dim': dict(
        type='int', default=20, lo=4, hi=200,
        desc='TDVP Krylov 子空间维数（越大越准越慢）'),
    'manybody.krylov_tol': dict(
        type='float', default=1e-10, lo=1e-14, hi=1e-2,
        desc='TDVP Krylov 误差容限'),
    'manybody.tdvp_adaptive_switch_chi': dict(
        type='int', default=16, lo=2, hi=4000,
        desc='自适应 TDVP 达到该键维后从两站点切换到单站点'),
    'manybody.tdvp_adaptive_min_steps': dict(
        type='int', default=1, lo=0,
        desc='自适应 TDVP 至少运行多少个两站点内部步后允许切换'),
    'manybody.max_sweeps': dict(
        type='int', default=60, lo=2, hi=500,
        desc='DMRG 最大 sweep 数（TEBD/TDVP 忽略）'),
    'manybody.mps_dtype': dict(type='str',enum=list(MPS_DTYPES),default='auto',
        desc='通用 calib DMRG 静态张量策略：auto=严格实数时 float64，否则 complex128；real 拒绝任何非零虚部；complex 保留复数'),
    'manybody.mps_order': dict(type='list',default=None,
        desc='通用 calib DMRG / CPU MPS 演化 / tdvp-dense 的显式顺序，列出全部原子下标；不改变原子身份或模型耦合'),
    'manybody.mps_diagnostics': dict(type='dict',default=None,
        desc='通用 calib DMRG 可选诊断：{variance:bool, corr:bool, permutation:[目标原子下标], blocks:[连续块长度]}；corr 给连通占据关联 C_ij=⟨n_i n_j⟩−⟨n_i⟩⟨n_j⟩（O(N²) 次收缩）；精确收缩受 memory_budget_mb 预算约束'),
    'manybody.qmc_sweeps': dict(
        type='int', default=2000, lo=10, hi=10_000_000,
        desc='SSE-QMC 测量 sweep 数（越大统计误差越小）'),
    'manybody.qmc_burn_in': dict(
        type='int', default=500, lo=0, hi=10_000_000,
        desc='SSE-QMC 热化步数'),
    'manybody.qmc_measure_every': dict(
        type='int', default=5, lo=1, hi=100_000,
        desc='SSE-QMC 每隔多少步采样'),
    'manybody.qmc_M': dict(
        type='int', default=None, lo=4,
        desc='SSE operator string 长度；None=按原子数自动设置'),
    'manybody.ed_max_dim': dict(
        type='int', default=int(MB_ED_MAX_DIM), lo=2,
        desc='ED 维度上限（solver=auto 据此在 ED 与 TDVP 间路由）'),
    'manybody.drive_segments': dict(
        type='list', default=None,
        desc='分段常数驱动 [{t0,Om:[…],phase:[…],delta:[…]},…]（缺省=静态，用 atoms 的 Ω/δ）。'
             '结构在 build 阶段当场校验（schema_mb.validate_drive_segments）：'
             'Om 为幅值、phase 为弧度，Om/delta/phase 为长度=原子数的数值序列'),
    'manybody.drive_interp': dict(
        type='str', enum=['left', 'midpoint'], default='left',
        desc='分段驱动取值规则：left=每段左端点常数值（默认，逐位兼容旧结果）；'
             'midpoint=把段条目视为线性 ramp 的控制点，取该段中点值（二阶 ramp 精度）。'
             '仅 ED / CPU MPS / tdvp-dense 消费；其它后端会在 build 阶段显式拒绝。'),
    'manybody.distance_cutoff_um': dict(
        type='float', default=None, lo=0.0,
        desc='可选的距离截断（µm，按 calib 显式 pos 的欧氏距离，不做周期最小像）：只保留 R≤cutoff 的原子对，丢弃项写进 calib.dropped/'
             'warnings；这是近似，会改变 ξ/χ_bulk，必须由调用方自查。None=不截断。'),
    'manybody.constraint': dict(
        type='dict', default=None,
        desc='Hamiltonian 约束：{kind:none|blockade|pxp, mode:full|project, '
             'threshold_mhz:..., edges:auto|[[i,j],...]}；在装配层生成 HamiltonianPlan'),
    'manybody.want_observables': dict(
        type='bool', default=True,
        desc='含时结果是否输出公共观测契约：逐时 X/Y/Z、升降算符、局部相位、'
             '请求的 link_corr 与 loop_phase/winding'),
    'manybody.loops': dict(
        type='list', default=None,
        desc='要计算的闭环原子下标列表；输出 loop_phase 与 loop_winding，None=不计算'),
    'manybody.link_pairs': dict(
        type='list', default=None,
        desc='逐时链路关联的原子对 [[i,j],…]；输出 link_corr。若同时给 loops，'
             '闭环边会自动并入测量集合'),
    'manybody.pauli_strings': dict(
        type='list', default=None,
        desc='**任意 Pauli 字符串**的期望值（通用高阶观测原语）：'
             '[[[0,"x"],[1,"y"],[2,"z"]], …] 或 '
             '[{"label":"chi_A","terms":[[0,"x"],[1,"y"],[2,"z"]]}, …]。'
             '每条 = 若干 [站点, 轴]（轴 x/y/z，可写 σx），同一字符串内站点不得重复。'
             '单格点矩阵是标准 Pauli，基为 (|g⟩,|r⟩)；本仓 sp/sm 是 |r><g|/|g><r|，'
             '与 σ± 的教科书编号相反。引擎**不解释**物理含义（手性要不要 /8、'
             '要不要连通减法，由调用方组合）。支持：含时 ed（全空间/投影基）、'
             'CPU MPS 四档（tebd/tdvp/tdvp-single/tdvp-adaptive）+ tdvp-dense、GPU MPS 两档'
             '（tebd-gpu/swap-tebd-gpu）、轨迹（系综平均）、经典自旋（因子化）；'
             '平衡态 ed/lanczos 的 gs、dmrg/idmrg 的 gs。PEPS/QMC/KPM/free-fermion/gpu-batch'
             '**明确拒绝**，不静默忽略。'),
    'manybody.charge_sector': dict(
        type='int', default=None, lo=0,
        desc='DMRG/ED 可选固定总激发数扇区；仅守恒 Hamiltonian 生效'),
    'manybody.memory_budget_mb': dict(
        type='float', default=None, lo=1,
        desc='ED/GPU 工作区或 PEPS 中间张量内存预算（MiB，保守估计）'),
    'manybody.dtype': dict(
        type='str', enum=['float32', 'float64'], default='float64',
        desc='GPU 多体计算精度'),
    'manybody.device': dict(
        type='str', default='cuda',
        desc='GPU 多体后端设备（如 cuda、cuda:0）；CPU 后端忽略'),
    'manybody.svd_driver': dict(
        type='str', enum=['auto', 'gesvdj', 'gesvd'], default='auto',
        desc='GPU MPS SVD 驱动；auto 使用 PyTorch 默认稳定路径，gesvdj 可尝试加速'),
}

# M1 无多体 solve.* 运行旋钮：所有多体配置走 manybody.*（含标定/维度旋钮）。
# 数值内部保护（_MB_DENSE_MAX/_MB_FULLPOP_MAX/_MB_MAX_DIM）是 engine_ed 的 Param，
# 经 engine_kwargs 编程覆写，不经 solve.*（保持 solve.* 单/两体语义不膨胀）。
SOLVE_META_MB = {}


def _num(x):
    import math
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def validate_drive_segments(segments, n_atoms=None, err=None):
    """`manybody.drive_segments` 的**结构**真契约。

    这个键的 schema 类型是通用 `list`，`session.set` 阶段不做结构校验；消费方
    `engine_ed._propagate_segments` 在工作线程里才 `segs[j].get('t0')` /
    `np.asarray(segs[j].get('Om', np.zeros(n)), float)`。元素不是 dict / 缺 t0 /
    `Om` 长度不对，会在 execute 才炸（AttributeError 或广播错，不指向病因）。

    `build_run_manybody` 在 build 阶段（主线程）调它 ⇒ 早失败 + 点名"第几段 /
    哪个键 / 期望多长"。`set` 阶段要前移需适配器把 `'drive_segments'` 加进
    `custom_types()` 并转调本函数。
    """
    if err is None:
        err = ValueError
    if segments is None:
        return None
    if not isinstance(segments, (list, tuple)):
        raise err('manybody.drive_segments 须是 list（得到 %s）'
                  % type(segments).__name__)
    if not segments:
        raise err('manybody.drive_segments 不能是空 list'
                  '（要静态驱动请设为 None / 不设）')
    n = None if n_atoms is None else int(n_atoms)
    for i, s in enumerate(segments):
        tag = 'segment[%d]' % i
        if not isinstance(s, dict):
            raise err('%s 须是 dict（{t0, Om, phase, delta}，得到 %s）'
                      % (tag, type(s).__name__))
        t0 = s.get('t0')
        if not _num(t0) or float(t0) < 0.0:
            raise err('%s 缺 t0 或非法（须 ≥0 的数值，得到 %r）' % (tag, t0))
        for key in ('Om', 'phase', 'delta'):
            v = s.get(key)
            if v is None:
                continue
            if _num(v):
                raise err('%s %s 是标量 %r——须给长度 = 原子数的序列：引擎按 '
                          '%s[i] 逐原子取值，标量会退化成 0 维数组、到演化时才崩'
                          % (tag, key, v, key))
            if not isinstance(v, (list, tuple)) and not hasattr(v, '__len__'):
                raise err('%s %s 须是序列（得到 %s）'
                          % (tag, key, type(v).__name__))
            if n is not None and len(v) != n:
                raise err('%s %s 长度 %d ≠ 原子数 %d' % (tag, key, len(v), n))
            if not all(_num(x) for x in v):
                raise err('%s %s 须全为数值（得到 %r）' % (tag, key, v))
    return list(segments)


def validate_loops(loops, n_atoms=None, err=None):
    """Validate user-specified loops without assigning lattice meaning."""
    if loops is None:
        return None
    if err is None:
        err = ValueError
    if not isinstance(loops, (list, tuple)):
        raise err('manybody.loops 须是闭环列表')
    n = None if n_atoms is None else int(n_atoms)
    out = []
    for i, loop in enumerate(loops):
        if not isinstance(loop, (list, tuple)) or len(loop) < 2:
            raise err('manybody.loops[%d] 须至少含两个原子下标' % i)
        q = []
        for x in loop:
            if not isinstance(x, int) or isinstance(x, bool):
                raise err('manybody.loops[%d] 含非整数下标 %r' % (i, x))
            if n is not None and not (0 <= int(x) < n):
                raise err('manybody.loops[%d] 下标 %d 越界（N=%d）' % (i, x, n))
            q.append(int(x))
        out.append(q)
    return out


def validate_link_pairs(pairs, n_atoms=None, err=None):
    """Validate directed link pairs for the common observable contract."""
    if pairs is None:
        return None
    if err is None:
        err = ValueError
    if not isinstance(pairs, (list, tuple)):
        raise err('manybody.link_pairs 须是 [[i,j],…]')
    n = None if n_atoms is None else int(n_atoms)
    out = []
    for q, pair in enumerate(pairs):
        if not isinstance(pair, (list, tuple)) or len(pair) != 2:
            raise err('manybody.link_pairs[%d] 须包含两个原子下标' % q)
        a, b = pair
        if (not isinstance(a, int) or isinstance(a, bool) or
                not isinstance(b, int) or isinstance(b, bool)):
            raise err('manybody.link_pairs[%d] 须全为整数' % q)
        if n is not None and not (0 <= int(a) < n and 0 <= int(b) < n):
            raise err('manybody.link_pairs[%d] 下标越界（N=%d）' % (q, n))
        out.append((int(a), int(b)))
    return out


def validate_pauli_strings(spec, n_atoms=None, err=None):
    """`manybody.pauli_strings` 的**结构真契约**（build 阶段当场判）。

    消费方（ED 的逐时刻归约、DMRG 的约化密度矩阵）要到工作线程才 `terms`
    解包；元素不是 [site, axis]、轴名写错、同站点重复、下标越界，都会在那时
    炸成广播错/KeyError 而不指向病因。这里提前变成"第几条 / 哪个站点"。
    """
    from .observables import normalize_pauli_strings
    return normalize_pauli_strings(spec, n_atoms=n_atoms, err=err)


def mb_get(p, key):
    """manybody.* 配置值：快照有则用，否则回落 CONFIG_SCHEMA_MB 的 default（单一来源）。"""
    v = (p or {}).get(key)
    if v is not None:
        return v
    return CONFIG_SCHEMA_MB.get(key, {}).get('default')
