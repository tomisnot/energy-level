# -*- coding: utf-8 -*-
"""多体产品配置：完整物理模型 → 标定快照 → 可执行 RunSpec。

产品固定包含 I1/I2/I3 全相互作用；按项选择的研究模型使用 ManyBodyModel。
标定在 ARC cache 所属线程完成，执行阶段只消费数值快照。后端名单、能力和
参数映射来自 backends；默认值来自 schema_mb；基空间决定后再选择后端。

单位：位置 µm，能量/Ω/δ/Γ MHz，时间 ns，B Gauss。
"""
import hashlib

import numpy as np

from .model import ManyBodyModel, TERMS
from . import assemble as _asm
from .backends import TIME_BACKENDS, normalize_time_backend, validate_time_capabilities
from .state import validate_psi_spec
from .hamiltonian_plan import build_plan, plan_summary
from .schema_mb import (mb_get, validate_drive_segments, validate_loops,
                        validate_link_pairs)

__all__ = ['model_from_params', 'drive_from_model', 'resolve_mb_psi',
           'resolve_solver', 'calibrate_memo', 'build_run_manybody']

_CALIB_MEMO = {}


def _fmt_pos(pos):
    return '(%.1f,%.1f,%.1f)' % tuple(float(x) for x in pos)


def _calib_key(model, second_order, guard_mhz):
    """calib 记忆化键：几何+能级+terms+B+n_window+r_cut+标定旋钮（**不含驱动**）。

    与两体 V_dd 缓存键同理：calib 只依赖这些 ⇒ 改 Ω/δ/Γ 复用同一 calib（不重碰 ARC）。
    """
    h = hashlib.sha1()
    h.update(repr((model.template, tuple(model.terms),
                   round(float(model.B_gauss), 9), int(model.n_window),
                   model.r_cut_um, model.periods_um,
                   bool(second_order), round(float(guard_mhz), 9))).encode())
    for a in model.atoms:
        h.update(repr((tuple(a['g']), tuple(a['r']),
                       tuple(round(float(x), 9) for x in a['pos']))).encode())
    return h.hexdigest()[:16]


def calibrate_memo(cache, model, second_order=True, guard_mhz=1.0):
    """带记忆的 `assemble.calibrate`（内容寻址；同几何+能级+标定旋钮 → 复用）。

    返回**共享**只读 calib（下游不得原地改，同 runspec._VDD_MEMO 纪律）。
    """
    key = _calib_key(model, second_order, guard_mhz)
    hit = _CALIB_MEMO.get(key)
    if hit is not None:
        return hit
    calib = _asm.calibrate(cache, model, second_order=second_order,
                           guard_mhz=guard_mhz)
    if len(_CALIB_MEMO) > 8:                     # 别让缓存无界长
        _CALIB_MEMO.clear()
    _CALIB_MEMO[key] = calib
    return calib


def model_from_params(p, warnings=None):
    """产品快照 → 全相互作用二能级模型；旧 terms/template 仅迁移提示。"""
    p = p or {}
    B = p.get('manybody.B_gauss')
    if B is None:
        B = p.get('field.B_gauss')               # 未设 → 复用共享 Zeeman 场
    retired = sorted(k for k in p if k in ('manybody.' + 'template', 'manybody.' + 'terms'))
    if retired and warnings is not None:
        warnings.append('[多体迁移] %s 已退役；产品模型固定使用 I1/I2/I3 全相互作用。'
                        '选择项的研究模型请使用 ManyBodyModel Python API。' % ', '.join(retired))
    m = ManyBodyModel(
        template='full', terms=tuple(TERMS), B_gauss=float(B or 0.0),
        n_window=int(mb_get(p, 'manybody.n_window')),
        r_cut_um=(None if p.get('manybody.r_cut_um') is None
                  else float(p.get('manybody.r_cut_um'))),
        periods_um=p.get('manybody.periods_um'))
    for a in (p.get('manybody.atoms') or []):
        m.add_atom(a['g'], a['r'], a['pos'],
                   Omega_mhz=a.get('Omega_mhz', 0.0),
                   Omega_phase_rad=a.get('Omega_phase_rad', 0.0),
                   delta_mhz=a.get('delta_mhz', 0.0),
                   gamma_mhz=a.get('gamma_mhz', 0.0))
    return m


def drive_from_model(model):
    """模型卡逐原子 Ω/δ/Γ → arrays（MHz；运行级，不进 calib 键）。

    The external model stores a real amplitude and a phase so it remains
    JSON-compatible.  The engines receive the physical complex coefficient
    ``Omega = amplitude * exp(1j*phase)``.
    """
    Om = np.array([
        float(a.get('Omega_mhz', 0.0))
        * np.exp(1j * float(a.get('Omega_phase_rad', 0.0)))
        for a in model.atoms
    ], complex)
    delta = np.array([a.get('delta_mhz', 0.0) for a in model.atoms], float)
    gamma = np.array([a.get('gamma_mhz', 0.0) for a in model.atoms], float)
    return Om, delta, gamma


def resolve_mb_psi(model, spec):
    """多体初态 spec 校验（回紧凑形态，**不物化** 2^N 向量）。

    与 runspec.resolve_psi 同纪律：非法**明确报错**，绝不静默回落全基态。
    None → None（engine 解释为全基态 |g…g⟩）；'ground' → engine 侧先 Lanczos 求基态。
    """
    if spec is None:
        return None
    if isinstance(spec, str) and spec == 'ground':
        return 'ground'
    n = len(model.atoms)
    # Validate the compact grammar without allocating a 2**N vector.  The
    # execution backend materializes it only when ED actually needs it; MPS
    # consumes the local vectors directly.
    validate_psi_spec(spec, n)
    return spec


def resolve_solver(solver, n_atoms, ed_max_dim, warnings, effective_dim=None):
    """Choose ED/TDVP using the effective basis dimension, then normalize aliases."""
    dim = (1 << int(n_atoms)) if effective_dim is None else int(effective_dim)
    name = normalize_time_backend(solver)
    if name in ('auto', 'ed'):
        if dim > int(ed_max_dim):
            if name == 'ed':
                raise ValueError('多体 ED 有效维度 %d 超 ed_max_dim=%d' % (dim, ed_max_dim))
            name = 'tdvp'
        else:
            name = 'ed'
        if normalize_time_backend(solver) == 'auto':
            warnings.append('[多体] solver=auto → %s（有效维度=%d）' % (name, dim))
    return name


def _backend_options(p, solver, ed_max_dim):
    """Map only the declared options, using schema defaults including zero values."""
    backend = TIME_BACKENDS[solver]
    options = {}
    for suffix, kw in backend.options:
        value = mb_get(p, 'manybody.' + suffix)
        if value is not None:
            options[kw] = value
    if 'svd_driver' in options and options['svd_driver'] == 'auto':
        options['svd_driver'] = None
    if backend.module in ('engine_mps', 'engine_mps_gpu'):
        options['algorithm'] = backend.algorithm or solver
        options['return_state'] = bool(mb_get(p, 'manybody.return_state'))
    if backend.charge:
        options['charge_sector'] = mb_get(p, 'manybody.charge_sector')
    if solver == 'ed':
        options['max_dim'] = ed_max_dim
        options['record_psi'] = bool(mb_get(p, 'manybody.return_state'))
    options['want_observables'] = bool(mb_get(p, 'manybody.want_observables'))
    return options


def build_run_manybody(cache, params, *, pump=None, method=None, diag=False,
                       t_grid=None):
    """登记册快照 + cache → 多体 RunSpec（纯容器，可 pickle 跨进程）。

    与 runspec.build_run 对称：读 manybody.* + dynamic.t_max/npts，标定（cache 所属线程），
    产出携带紧凑 calib 的 RunSpec。**calibrate 碰 ARC/SQLite ⇒ 必须在 ops.prepare
    （cache 所属线程）里调**；evolve（execute/工作线程）只用 calib 装 H，不碰 cache。
    """
    from ..runspec import _time_grid, god_kwargs_from_params, _filter_kw, engine_fn
    p = params or {}
    warnings = []
    model = model_from_params(p, warnings=warnings)
    n = len(model.atoms)
    if n == 0:
        raise ValueError('多体：manybody.atoms 为空——先摆原子'
                         '（set_param("manybody.atoms", [{g,r,pos,…}, …])）')
    # 时间网格：复用 dynamic.t_max/npts（与单/两体同一 _time_grid 语义 + 报错不静默）
    t_max, npts, t = _time_grid(p, t_grid)

    # 标定（cache 所属线程碰 ARC）——记忆化
    second_order = bool(mb_get(p, 'manybody.second_order'))
    guard_mhz = float(mb_get(p, 'manybody.guard_mhz'))
    ed_max_dim = int(mb_get(p, 'manybody.ed_max_dim'))
    calib = calibrate_memo(cache, model, second_order=second_order,
                           guard_mhz=guard_mhz)

    solver_in = method or mb_get(p, 'manybody.solver')
    psi0 = resolve_mb_psi(model, p.get('manybody.psi0'))
    solver_request = solver_in
    if isinstance(psi0, dict) and psi0.get('kind') == 'mps':
        if str(solver_in or 'auto').lower() == 'auto':
            solver_request = 'tdvp' if psi0.get('backend') == 'tenpy' else 'swap-tebd-gpu'
            warnings.append('[多体] solver=auto 按 MPS 初态选择 '+solver_request)
    Om, delta, gamma = drive_from_model(model)
    plan = build_plan(calib, Om=Om, delta=delta, gamma=gamma,
                      terms=model.terms, constraint=p.get('manybody.constraint'),
                      max_states=ed_max_dim)
    solver = resolve_solver(solver_request, n, ed_max_dim, warnings,
                            effective_dim=plan['dim'])
    segments = validate_drive_segments(p.get('manybody.drive_segments'), n)
    validate_time_capabilities(
        solver, segments=segments, plan=plan, gamma=gamma, psi0=psi0,
        return_state=mb_get(p, 'manybody.return_state'),
        charge_sector=mb_get(p, 'manybody.charge_sector'),
        collapse_ops=mb_get(p, 'manybody.collapse_ops'))
    if plan['projected']:
        warnings.append('[多体装配] %s：dim %d → %d' %
                        (plan['constraint']['kind'], plan['full_dim'], plan['dim']))
    ek = _backend_options(p, solver, ed_max_dim)
    extras, dropped = _filter_kw(engine_fn(solver, 'many'),
                                  god_kwargs_from_params(p, pair=False, diag=diag))
    ek.update(extras)
    if dropped:
        warnings.append('[多体引擎参数] %s 不接收：%s' % (solver, ', '.join(sorted(dropped))))

    # 标定的"被丢项"报告（关项/截断不许静默，对齐 static_phase_keep 口径）
    off = {k: v for k, v in (calib.get('dropped_amp') or {}).items() if v}
    if off:
        warnings.append('[多体标定] 关闭项的最大被丢幅值(MHz)：%s'
                        % {k: round(float(v), 6) for k, v in off.items()})
    ndrop = len(calib.get('dropped') or [])
    if ndrop:
        warnings.append('[多体标定] r_cut 截断 %d 对（尾项未计入 H）' % ndrop)
    guarded = sum(int((pr.get('info') or {}).get('n_guarded', 0))
                  for pr in calib.get('pairs', []))
    if guarded:
        warnings.append('[多体标定] %d 个近简并中间态被 guard 剔除'
                        '（downfolding 在该支不适用，见 M0 诚实边界）' % guarded)

    states = [{'label': 'atom%d@%s' % (a['id'], _fmt_pos(a['pos'])),
               'atom_id': a['id']} for a in calib['atoms']]
    # 分段常数驱动的**结构校验**（2026-09-22，工单 P2-2）：原先只到工作线程的
    # `_propagate_segments` 才炸（AttributeError/广播错，不指向病因）。现在在
    # build（cache 所属线程）早失败并点名第几段/哪个键/期望多长。
    loops = validate_loops(p.get('manybody.loops'), n)
    link_pairs = validate_link_pairs(p.get('manybody.link_pairs'), n)
    # A requested loop is itself a set of directed links.  Include those links
    # automatically so every real-time backend can return a loop phase/winding
    # time series without a solver-specific special case.
    if loops is not None:
        required = [(a, b) for loop in loops
                    for a, b in zip(loop, loop[1:] + loop[:1])]
        link_pairs = list(link_pairs or [])
        for pair in required:
            if pair not in link_pairs and (pair[1], pair[0]) not in link_pairs:
                link_pairs.append(pair)
    if 'manybody.want_observables' in p:
        ek['want_observables'] = bool(p.get('manybody.want_observables'))
    if loops is not None:
        ek['loops'] = loops
    if link_pairs is not None:
        ek['link_pairs'] = link_pairs
    return {
        'version': 1, 'two_body': False, 'nbody': 'many',
        'method': solver, 'method_in': solver_in,
        't': t, 't_max': t_max, 'npts': npts,
        'calib': calib, 'hamiltonian_plan': plan, 'plan_summary': plan_summary(plan),
        'n_atoms': n, 'dim': plan['dim'], 'full_dim': 1 << n,
        'psi0': psi0, 'Om': Om, 'delta': delta, 'gamma': gamma,
        'segments': segments,
        'collapse_ops': p.get('manybody.collapse_ops') or [],
        'n_trajectories': int(p.get('manybody.n_trajectories') or 32),
        'terms': list(model.terms), 'template': model.template,
        'engine_kwargs': ek, 'warnings': warnings,
        # 契约兼容字段（让 summarize_run/check_run 的通用路径不 KeyError）
        'states': states, 'labels': [s['label'] for s in states],
        'fields': None, 'decays': [], 'n_waves': {},
        'vdd_matrix': None, 'pair': None, 'field_loss': None,
    }
