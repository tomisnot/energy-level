# -*- coding: utf-8 -*-
"""多体运行配置解析（对齐 core/runspec.py 模式）：登记册快照 → 多体 RunSpec。

与单体/两体 build_run 对称，产物交 `runspec.evolve_run` 唯一分派点
（dynamic.mode==2 / method='ed'）⇒ GUI 与 AI 走同一段引擎代码（I5）。

关键分工（守 B-1① ARC SQLite **线程亲和**铁律）：
- **calibrate 在 build 阶段**（主线程，碰 ARC/SQLite）；RunSpec 只携带**紧凑 calib**
  （V/J/A + atoms + terms），**不携带 H 矩阵**——H 在 execute（工作线程）由 engine_ed
  装配，纯数值、不碰 cache。故多体 async job 的 prepare/execute 拆分天然安全。
- calib 按内容键**记忆化**（几何+能级+terms+B+n_window+r_cut+标定旋钮，**不含驱动**
  ⇒ 改 Ω/δ 复用同一 calib），仿 runspec._VDD_MEMO。

单位与全库一致：位置 µm、Ω/δ/Γ/V/J/A MHz、时间 ns、B Gauss。
"""
import hashlib

import numpy as np

from .model import ManyBodyModel, TEMPLATE_TERMS
from . import assemble as _asm
from . import engine_ed as _ed
from .schema_mb import mb_get, validate_drive_segments

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
                   model.r_cut_um,
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
    """登记册快照 → ManyBodyModel（从 manybody.* 键装配；类比 runspec.build_states）。

    `terms` 与 `template` 的关系（2026-09-22，工单 P2-1）：**显式 terms 优先**；
    terms 为 None（未设）时由 `template` 给预设（ising→I1 / xy→I2）。若两者**都显式
    给出且不一致**，写一条 warning（不静默——否则用户"选了 xy 却得到 I1"无从察觉）。
    """
    p = p or {}
    B = p.get('manybody.B_gauss')
    if B is None:
        B = p.get('field.B_gauss')               # 未设 → 复用共享 Zeeman 场
    tmpl = mb_get(p, 'manybody.template')
    terms_raw = mb_get(p, 'manybody.terms')      # None = 跟随 template 预设
    if warnings is not None and terms_raw is not None:
        preset = TEMPLATE_TERMS.get(tmpl)
        if preset is not None and tuple(terms_raw) != tuple(preset):
            warnings.append(
                '[多体] manybody.terms=%s 与 manybody.template=%s 的预设 %s 不一致'
                '——**以显式 terms 为准**（相互作用内容只有一个真源）；'
                '若想用模板预设，请清空 manybody.terms'
                % (list(terms_raw), tmpl, list(preset)))
    m = ManyBodyModel(
        template=tmpl,
        terms=(None if terms_raw is None else tuple(terms_raw)),
        B_gauss=float(B or 0.0),
        n_window=int(mb_get(p, 'manybody.n_window')),
        r_cut_um=(None if p.get('manybody.r_cut_um') is None
                  else float(p.get('manybody.r_cut_um'))))
    for a in (p.get('manybody.atoms') or []):
        m.add_atom(a['g'], a['r'], a['pos'],
                   Omega_mhz=a.get('Omega_mhz', 0.0),
                   delta_mhz=a.get('delta_mhz', 0.0),
                   gamma_mhz=a.get('gamma_mhz', 0.0))
    return m


def drive_from_model(model):
    """模型卡逐原子 Ω/δ/Γ → (Om, delta, gamma) 数组（MHz；运行级，不进 calib 键）。"""
    Om = np.array([a.get('Omega_mhz', 0.0) for a in model.atoms], float)
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
    _ed.psi0_vector(n, 1 << n, spec)             # 借引擎解析器校验（非法会 raise）
    return spec


def resolve_solver(solver, n_atoms, ed_max_dim, warnings):
    """manybody.solver → 实际后端 'ed'；auto 记理由；tebd → M3 未实现（明确报错）。"""
    dim = 1 << int(n_atoms)
    s = str(solver or 'auto').lower()
    if s in ('auto', 'ed'):
        if dim > int(ed_max_dim):
            raise ValueError('多体：dim=2^%d=%d 超 ed_max_dim=%d——ED 不可行；'
                             '大 N 需 M3 的 MPS/TEBD 后端（未实现）。请减原子数或'
                             '调大 manybody.ed_max_dim（内存自负）'
                             % (n_atoms, dim, ed_max_dim))
        if s == 'auto':
            warnings.append('[多体] solver=auto → ed（dim=2^%d=%d ≤ ed_max_dim=%d；'
                            'MPS/TEBD 后端在 M3）' % (n_atoms, dim, ed_max_dim))
        return 'ed'
    if s in ('tebd', 'mps', 'mps_mpo'):
        raise NotImplementedError('多体 solver=%r 属 M3（TeNPy/MPS 后端，未实现）；'
                                  '现请用 ed（dim ≤ ed_max_dim）' % s)
    raise ValueError('多体：未知 solver=%r（合法 auto/ed/tebd）' % solver)


def build_run_manybody(cache, params, *, pump=None, method=None, diag=False,
                       t_grid=None):
    """登记册快照 + cache → 多体 RunSpec（纯容器，可 pickle 跨进程）。

    与 runspec.build_run 对称：读 manybody.* + dynamic.t_max/npts，标定（主线程），
    产出携带紧凑 calib 的 RunSpec。**calibrate 碰 ARC/SQLite ⇒ 必须在 ops.prepare
    （主线程）里调**；evolve（execute/工作线程）只用 calib 装 H，不碰 cache。
    """
    from ..runspec import _time_grid, god_kwargs_from_params, _filter_kw
    p = params or {}
    warnings = []
    model = model_from_params(p, warnings=warnings)
    n = len(model.atoms)
    if n == 0:
        raise ValueError('多体：manybody.atoms 为空——先摆原子'
                         '（set_param("manybody.atoms", [{g,r,pos,…}, …])）')
    if model.template == 'potts3':
        raise NotImplementedError('多体 template=potts3（3 能级/原子）M1 未实现；'
                                  'ED 装配现为每原子 2 能级（ising/xy）')

    # 时间网格：复用 dynamic.t_max/npts（与单/两体同一 _time_grid 语义 + 报错不静默）
    t_max, npts, t = _time_grid(p, t_grid)

    # 标定（主线程碰 ARC）——记忆化
    second_order = bool(mb_get(p, 'manybody.second_order'))
    guard_mhz = float(mb_get(p, 'manybody.guard_mhz'))
    ed_max_dim = int(mb_get(p, 'manybody.ed_max_dim'))
    calib = calibrate_memo(cache, model, second_order=second_order,
                           guard_mhz=guard_mhz)

    solver_in = method or mb_get(p, 'manybody.solver')
    solver = resolve_solver(solver_in, n, ed_max_dim, warnings)
    Om, delta, gamma = drive_from_model(model)
    psi0 = resolve_mb_psi(model, p.get('manybody.psi0'))

    # 运行级 engine_kwargs：多体无专有 solve.*；通用 solve.* 按 evolve_manybody 签名筛
    ek_all = god_kwargs_from_params(p, pair=False, diag=diag)
    ek, dropped = _filter_kw(_ed.evolve_manybody, ek_all)
    if dropped:
        warnings.append('[多体引擎参数] evolve_manybody 不接收这些键，已忽略：%s'
                        % ', '.join(sorted(dropped)))
    ek['max_dim'] = ed_max_dim      # evolve 用同一维度上限（与 resolve_solver 一致，免双标）

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
    # build（主线程）早失败并点名第几段/哪个键/期望多长。
    segments = validate_drive_segments(p.get('manybody.drive_segments'), n)
    return {
        'version': 1, 'two_body': False, 'nbody': 'many',
        'method': solver, 'method_in': solver_in,
        't': t, 't_max': t_max, 'npts': npts,
        'calib': calib, 'n_atoms': n, 'dim': 1 << n,
        'psi0': psi0, 'Om': Om, 'delta': delta, 'gamma': gamma,
        'segments': segments,
        'terms': list(model.terms), 'template': model.template,
        'engine_kwargs': ek, 'warnings': warnings,
        # 契约兼容字段（让 summarize_run/check_run 的通用路径不 KeyError）
        'states': states, 'labels': [s['label'] for s in states],
        'fields': None, 'decays': [], 'n_waves': {},
        'vdd_matrix': None, 'pair': None, 'field_loss': None,
    }
