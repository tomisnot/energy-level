# -*- coding: utf-8 -*-
"""多体配置契约（`manybody.*` 命名空间）+ 逐原子真校验。

与 core/protocol.py 的 CONFIG_SCHEMA 同角色，但只声明多体页的键；由 protocol
合并进 config_schema()（键在多体、单/两体互不打扰）。值域/必填投影自 build_run
自身的解析，validate_atom_list 在 session.set **当场**把橡皮图章换成真报错
（仿 protocol.validate_wave_list）。

单位与全库一致：位置 µm、Ω/δ/Γ/V/J/A MHz、时间 ns、B Gauss。
"""
from ..param import Param

__all__ = ['CONFIG_SCHEMA_MB', 'SOLVE_META_MB', 'validate_atom_list', 'mb_get',
           'validate_drive_segments', 'MB_N_WINDOW', 'MB_GUARD_MHZ',
           'MB_ED_MAX_DIM']

# 标定/维度默认（就地 Param：值+元数据同处，被 protocol.engine_params 扫出；
# 同时作为 CONFIG_SCHEMA_MB 各键 default 的单一来源）。
MB_N_WINDOW = Param(3, '单原子基伙伴态窗口 |Δn|≤n_window（补 vdW 中间态）',
                    tier='run')
MB_GUARD_MHZ = Param(1.0, 'Löwdin downfolding 近简并中间态剔除阈',
                     unit='MHz', lo=0.0, tier='run')
MB_ED_MAX_DIM = Param(1 << 20, 'ED 维度上限 2^N（solver=auto 据此判可行）',
                      lo=2, tier='run')

# `manybody.*` → build_run 读的登记册快照键。type='atom_list'/'mb_psi' 是多体
# 专有类型：atom_list 经 validate_atom_list 当场校验；mb_psi 结构灵活，由
# resolve_mb_psi 在 build 阶段校验（需原子数上下文，set 阶段拿不到）。
CONFIG_SCHEMA_MB = {
    'manybody.template': dict(
        type='str', enum=['ising', 'xy', 'potts3'], default='ising',
        desc='局部空间模板：ising/xy=每原子 2 能级（M1 ED 支持）；potts3=3 能级（未实现）'),
    'manybody.terms': dict(
        type='list', default=['I1'],
        desc='相互作用项开关：I1=密度-密度 V·n_i n_j / I2=交换 J(σ⁺σ⁻) / I3=双激发 A(σ⁺σ⁺)'),
    'manybody.atoms': dict(
        type='atom_list', default=[],
        desc='任意点云：逐原子 {g:(n,l,j,m), r:(n,l,j,m), pos:(x,y,z)µm, '
             'Omega_mhz, delta_mhz, gamma_mhz}'),
    'manybody.psi0': dict(
        type='mb_psi', default=None,
        desc="初态：None=全基态|g…g⟩ / int=产物态索引 / 'grg'=模式串 / "
             "'ground'=Lanczos 基态 / ['sup',[[模式或索引,amp],…]]=叠加"),
    'manybody.B_gauss': dict(
        type='float', unit='Gauss', default=None,
        desc='静磁场（Zeeman）；None=复用 field.B_gauss'),
    'manybody.n_window': dict(
        type='int', default=int(MB_N_WINDOW), lo=0,
        desc='单原子基伙伴态窗口 |Δn|≤n_window（补 vdW 中间态）'),
    'manybody.r_cut_um': dict(
        type='float', unit='um', default=None, lo=0.0,
        desc='相互作用截断半径（None=不截断；被丢对计入 dropped 报告，不静默）'),
    'manybody.energy_window_mhz': dict(
        type='float', unit='MHz', default=None,
        desc='能量窗口截断（可选）'),
    'manybody.solver': dict(
        type='str', enum=['auto', 'ed', 'tebd'], default='auto',
        desc='求解器：auto=按维度选 / ed=精确对角化（M1）/ tebd=TeNPy MPS（M3，未实现）'),
    'manybody.second_order': dict(
        type='bool', default=True,
        desc='二阶 Löwdin downfolding（vdW C6 从伙伴态涌现）；关=只一阶 V_dd'),
    'manybody.guard_mhz': dict(
        type='float', unit='MHz', default=float(MB_GUARD_MHZ), lo=0.0,
        desc='近简并中间态剔除阈（|E−E_m|<guard 剔除并计数上报，不许静默）'),
    'manybody.ed_max_dim': dict(
        type='int', default=int(MB_ED_MAX_DIM), lo=2,
        desc='ED 维度上限（solver=auto 据此判可行；超过明确报错，等 M3 的 MPS）'),
    'manybody.seed': dict(
        type='int', default=None, desc='随机种子（阵列抖动/量子轨迹，可复现）'),
    'manybody.drive_segments': dict(
        type='list', default=None,
        desc='分段常数驱动 [{t0,Om:[…],delta:[…]},…]（缺省=静态，用 atoms 的 Ω/δ）'),
}

# M1 无多体 solve.* 运行旋钮：所有多体配置走 manybody.*（含标定/维度旋钮）。
# 数值内部保护（_MB_DENSE_MAX/_MB_FULLPOP_MAX/_MB_MAX_DIM）是 engine_ed 的 Param，
# 经 engine_kwargs 编程覆写，不经 solve.*（保持 solve.* 单/两体语义不膨胀）。
SOLVE_META_MB = {}


def _num(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def validate_drive_segments(segments, n_atoms=None, err=None):
    """`manybody.drive_segments` 的**结构**真契约（2026-09-22，工单 P2-2）。

    为什么需要：这个键的 schema 类型是通用 `list`，`session.set` 阶段**不做任何
    结构校验**；而消费方 `engine_ed._propagate_segments` 要等到**工作线程**里才
    `segs[j].get('t0')` / `np.asarray(segs[j].get('Om', np.zeros(n)), float)`
    ⇒ 元素不是 dict / 缺 t0 / `Om` 长度不对，都要到 execute 才炸，而且报的是
    AttributeError 或广播错，**不指向病因**。

    这里把它变成**当场可判**：`build_run_manybody` 在 build 阶段（主线程）调它
    ⇒ 早失败 + 点名"第几段 / 哪个键 / 期望多长"。`set` 阶段的前移需适配器把
    `'drive_segments'` 加进 `custom_types()` 并转调本函数（跨模块，见工单 P2-2）。
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
            raise err('%s 须是 dict（{t0, Om, delta}，得到 %s）'
                      % (tag, type(s).__name__))
        t0 = s.get('t0')
        if not _num(t0) or float(t0) < 0.0:
            raise err('%s 缺 t0 或非法（须 ≥0 的数值，得到 %r）' % (tag, t0))
        for key in ('Om', 'delta'):
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


def mb_get(p, key):
    """manybody.* 配置值：快照有则用，否则回落 CONFIG_SCHEMA_MB 的 default（单一来源）。"""
    v = (p or {}).get(key)
    if v is not None:
        return v
    return CONFIG_SCHEMA_MB.get(key, {}).get('default')


def validate_atom_list(atoms, err=None):
    """逐原子真校验（session.set 当场，I6/D5）；非法 → 抛 `err`（默认 ValueError）。

    校验结构：g/r=(n,l,j,m) 四元组数值、pos=(x,y,z) µm 三元组数值、Ω/δ/Γ 数值。
    重合原子（R=0）留给 calibrate 的 geometry() 报（build 阶段，消息更具体）。
    """
    if err is None:
        err = ValueError
    if not isinstance(atoms, (list, tuple)):
        raise err('manybody.atoms 须是 list（得到 %s）' % type(atoms).__name__)
    if not atoms:
        raise err('manybody.atoms 不能为空（至少摆 1 个原子）')
    for i, a in enumerate(atoms):
        tag = 'atom[%d]' % i
        if not isinstance(a, dict):
            raise err('%s 须是 dict（{g,r,pos,…}）' % tag)
        for key in ('g', 'r'):
            q = a.get(key)
            if not isinstance(q, (list, tuple)) or len(q) != 4:
                raise err('%s %s 须是 (n,l,j,m) 四元组（得到 %r）' % (tag, key, q))
            if not all(_num(x) for x in q):
                raise err('%s %s=(n,l,j,m) 须全数值（得到 %r）' % (tag, key, q))
        pos = a.get('pos')
        if not isinstance(pos, (list, tuple)) or len(pos) != 3:
            raise err('%s pos 须是 (x,y,z) µm 三元组（得到 %r）' % (tag, pos))
        if not all(_num(x) for x in pos):
            raise err('%s pos 须全数值（得到 %r）' % (tag, pos))
        for key in ('Omega_mhz', 'delta_mhz', 'gamma_mhz'):
            v = a.get(key)
            if v is not None and not _num(v):
                raise err('%s %s 须是数值 MHz（得到 %r）' % (tag, key, v))
    return list(atoms)
