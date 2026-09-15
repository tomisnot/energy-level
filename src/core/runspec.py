# -*- coding: utf-8 -*-
"""运行配置解析公共层（**无 Qt**）：登记册快照 → RunSpec。

## 为什么需要这个模块（P5 单一真源）

"配置 → (基组, 场, 初态, 衰变, 引擎参数)" 这套解析原先**只存在于**
`ui_dynamic._run_single_body` / `_run_two_body` 里。于是任何想批量跑的人
（离线脚本、并行扫描）都只能自己再拼一套基组装配——`scripts/tmp/_scan2d_par.py`
的 `build_level_rows_direct` 就是这么来的。后果是两条通道的基组定义会**静默
分叉**（SKILL 坑 18 / 待办 §5 第 8 条：6P 展开成 4 个 mJ 还是 16 个 (F,m_F)）。

本模块把这段解析提成无 Qt 的纯函数：

    params（登记册快照）+ cache → RunSpec（纯容器，可 pickle）

GUI 的单跑与并行扫描的 worker **都调它** ⇒ "扫描"在构造上严格等于
"把人点运行的那份配置跑 N 次"。这也是 `src/sweep/` 能成立的前提：
worker 子进程不 import PyQt5。

## 与 GUI 的等价性怎么保证

逐分支照抄 `ui_dynamic._run_single_body/_run_two_body` 的解析顺序（含
sambe+任意波形 → hard 的改道、mixed 的 mw_arb 通道、两体 V_dd 的
R/θ/φ 与「仅丢失落点」约束）。`checks/sweep_parity_selfcheck.py` 拿真实
GUI 面板跑同一配置，逐位对拍 states/fields/psi/decays。

## 单位

与软件其余部分一致：B=Gauss、E=V/cm、t=ns、微波 freq=MHz、光频=Hz、
Γ=MHz。本模块不改单位，只搬运。
"""
import copy
import hashlib

import numpy as np

from . import static_atom as core
from .atomic_levels import make_levels, find_state_index, state_key
from . import waveforms as _wf
from .field_build import build_fields_exact
from .evolve_twofield import build_two_field, arbitrary_mw_group
from .evolve_util import decay, auto_decays
from . import static_pair as pair_core

SPEC_VERSION = 1

_WAVE_ATOMS = ('single', 'A', 'B')

# solve.* → 下游 kwarg。**必须与 pm_setup._SOLVE_SPECS 的最后一列一致**
# （那是 GUI 侧的定义点；本表是无 Qt 侧的镜像，改一处要改两处——下面
# `checks/sweep_spec_selfcheck.py` 有断言把两边钉在一起）。
SOLVE_KWARG = {
    'MAX_STEP': 'max_step',
    'MAX_STEP_HARD': 'max_step_hard',
    'PER_FREQ_M': 'per_freq_M',
    'FROZEN_H_NS': 'frozen_h_ns',
    'COST_BUDGET': 'phase_budget',
}

# 各引擎**实际收**的运行级 kwarg 白名单（照签名抄）。
# 为什么要显式白名单：登记册里 solve.* 的**读回值**（GUI 控件当前值）会一起
# 进 kwargs，而它们并不都是每个引擎认得的形参——例如 `frozen_h_ns` 是 sambe
# 两体的旋钮，`evolve_mixed`/`evolve_mixed_pair` 的签名里**没有**它。
# 不筛就会 `TypeError: unexpected keyword argument`（实测过这条）。GUI 侧靠
# "只传用户显式改过的键"绕开了，runner 传的是登记册全量 ⇒ 必须显式筛。
_MIXED_PAIR_KW = ('tol', 'max_step', 'max_step_hard', 'r_crit', 'diag',
                  'bw_opt', 'mw_cap', 'mw_max', 'phase_budget')
_MIXED_SINGLE_KW = _MIXED_PAIR_KW + ('per_freq_M',)


# =====================================================================
# 基组
# =====================================================================
def groups_from_params(p):
    """登记册快照 → 能级组列表（**原样**保留 color/ms/Fs 等附加字段）。"""
    return [dict(g) for g in ((p or {}).get('groups.list') or [])]


def build_states(cache, groups, B_gauss=0.0, E_vpcm=0.0, pump=None):
    """动态基组（= `ui_dynamic._current_states` 的同源实现）。

    低激发态展开超精细 F/m_F（用场面板磁场 B），里德堡态保持 mJ；
    能量 = 组中心 + Zeeman(B) + DC-Stark(E)，与静态页/两体页同基准。
    逐子能级裁剪（组条目的 ms / Fs）与静态页共用 `static_atom` 判据。

    pump：可选 callable(frac)——每个能级组完成后上报。
    """
    B = float(B_gauss or 0.0)
    E_vpcm = float(E_vpcm or 0.0)
    specs = []
    n_groups = max(len(groups), 1)
    for gi, g in enumerate(groups):
        n, l, j = int(g['n']), int(g['l']), float(g['j'])
        _ms, _Fs = core.trim_of(g)
        _keep = core.sublevel_trim(_ms, _Fs)
        hf = core.get_hyperfine_levels(cache, n, l, j, B)
        if hf is None:
            for m in np.arange(-j, j + 1e-9, 1.0):
                if _keep is not None and not _keep(
                        dict(kind='mJ', mj=float(m))):
                    continue
                E = core._sublevel_energy(cache, n, l, j, m, B)
                if abs(E_vpcm) > 1e-9:
                    E += core.dc_stark_shift_mhz(
                        cache, n, l, j, m, groups, E_vpcm, B)
                specs.append(dict(n=n, l=l, j=j, kind='mJ', m=float(m),
                                  E_mhz=float(E)))
        else:
            for F, mF, e_hf, _w, amps in hf:
                if _keep is not None and not _keep(
                        dict(kind='F', F=float(F), mF=float(mF))):
                    continue
                E0 = cache.E_mhz(n, l, j)
                if abs(E_vpcm) > 1e-9:
                    E0 += core.dc_stark_shift_mhz(
                        cache, n, l, j, mF, groups, E_vpcm, B)
                specs.append(dict(n=n, l=l, j=j, kind='F', F=F, mF=mF,
                                  E_hf=float(e_hf), E_mhz=float(E0),
                                  amps=amps))
        if pump is not None:
            pump((gi + 1) / n_groups)
    return make_levels(cache, specs)


def states_fingerprint(states):
    """基组指纹（用于 worker 侧缓存 V_dd 等昂贵预计算）。"""
    h = hashlib.sha1()
    for s in states:
        h.update(repr(tuple(state_key(s))).encode('utf-8'))
    return h.hexdigest()[:16]


# =====================================================================
# 初态
# =====================================================================
def resolve_psi(states, spec, which='single'):
    """初态解析：叠加态 spec 优先，否则身份键 → 索引。

    与 `ui_dynamic._psi0_for` 同语义——**绝不静默回落 0 号态**（P1）。
    返回 int（基组索引）或 list（`['sup', …]`，交引擎按 (n,l,j,m) 解析）。
    """
    if (isinstance(spec, (list, tuple)) and len(spec) == 2
            and spec[0] == 'sup'):
        return list(spec)
    if spec is None:
        raise ValueError(
            '初态未选定（%s）：快照里 dynamic.psi_%s 为空'
            '（先添加能级组，或用 set_initial_state / set_superposition）'
            % (which, which))
    i = find_state_index(states, spec)
    if i < 0:
        raise ValueError(
            '初态身份键不在当前基组内（%s）：%r —— 基组可能已被改动；'
            '用 list_levels() 拿合法身份键后重设' % (which, spec))
    return int(i)


def resolve_pair_psi(spec):
    """两体纠缠初态 spec（`dynamic.psi_pair`）——非法/未设 → None。"""
    if (isinstance(spec, (list, tuple)) and len(spec) == 2
            and spec[0] == 'pair_sup'
            and isinstance(spec[1], (list, tuple)) and spec[1]):
        return list(spec)
    return None


# =====================================================================
# 衰变（自发辐射）
# =====================================================================
def resolve_decays(cache, states, decay_index, decay_cfgs, two_body=False):
    """自发辐射通道 → `evolve_util.decay` 列表。

    档位（`dynamic.decay_index`）：0 自动（ARC 寿命）/ 1 禁用 / 2 自定义。
    两体 v1 只支持「丢失」落点（GUI 同约束，此处同样**显式报错**不静默）。
    """
    mode = int(decay_index or 0)
    if mode == 1:
        return []
    labels = [s['label'] for s in states]
    if mode == 0:
        return auto_decays(cache, states)
    if mode != 2:
        return []
    out = []
    for c in (decay_cfgs or []):
        frm = c.get('from')
        if frm not in labels:
            continue
        gamma = float(c.get('gamma_mhz') or 0.0)
        if two_body:
            if c.get('to'):
                raise NotImplementedError(
                    '双原子自定义衰变 v1 仅支持「丢失」落点'
                    '（显式落点请用单原子模式）：%r → %r' % (frm, c.get('to')))
            out.append(decay(labels.index(frm), gamma, None))
        else:
            to = c.get('to')
            ti = labels.index(to) if (to and to in labels) else None
            out.append(decay(labels.index(frm), gamma, ti))
    return out


# =====================================================================
# 引擎参数覆写（engine.* / solve.*）
# =====================================================================
def god_kwargs_from_params(p, pair=False, diag=False):
    """登记册快照里的 `engine.*` / `solve.*` → 引擎 kwargs。

    与 `pm_setup.god_split(MainWindow.engine_overrides)` **行为等价**：
    登记册的 engine./solve. 读回值就是"覆写优先"的当前生效值，故这里用
    全部生效值做覆写，结果与 GUI 逐位一致（且顺带把默认值冻结进扫描存档，
    可复现性更好）。

    pair=True：剔除 `per_freq_M`（两体 P 固定，GUI 也把它走独立形参）。
    """
    po, ek = {}, {}
    for k, v in (p or {}).items():
        if not isinstance(k, str) or v is None:
            continue
        if k.startswith('engine.'):
            name = k.split('.', 1)[1]
            if name in _ENGINE_NAMES:
                po[name] = v
        elif k.startswith('solve.'):
            kw = SOLVE_KWARG.get(k.split('.', 1)[1])
            if kw is not None:
                ek[kw] = v
    if pair:
        ek.pop('per_freq_M', None)
    if diag:
        ek['diag'] = True
    out = {}
    if po:
        out['param_overrides'] = po
    out.update(ek)
    return out


def _filter_kw(d, allow):
    """kwargs → (只留 allow 里的, 被丢掉的键集合)。"""
    keep = {k: v for k, v in (d or {}).items() if k in allow}
    drop = {k for k in (d or {}) if k not in allow}
    return keep, drop


def _engine_names():
    try:
        from . import params as _P
        return set(_P.SPEC_BY_NAME)
    except Exception:                              # noqa: BLE001
        return set()


_ENGINE_NAMES = _engine_names()


def parse_per_freq_m(v):
    """`solve.PER_FREQ_M` → int 或 [int,…]（对齐 `_sambe_adv_params` 的解析）。"""
    if v is None or v == '':
        return None
    if isinstance(v, (list, tuple)):
        try:
            return [int(x) for x in v]
        except (TypeError, ValueError):
            return None
    if isinstance(v, bool):
        return None
    if isinstance(v, (int, float)):
        return int(v)
    try:
        import ast
        x = ast.literal_eval(str(v))
    except Exception:                              # noqa: BLE001
        return None
    if isinstance(x, list):
        try:
            return [int(i) for i in x]
        except (TypeError, ValueError):
            return None
    try:
        return int(x)
    except (TypeError, ValueError):
        return None


# =====================================================================
# 波形与场
# =====================================================================
def waves_for(p, atom):
    """登记册快照 → Wave/ArbitraryWave 列表（atom ∈ single/A/B）。"""
    return _wf.restore_waves((p or {}).get('dynamic.waves_%s' % atom) or [])


def has_arb(waves):
    return any(isinstance(w, _wf.ArbitraryWave) for w in (waves or []))


def effective_method(method, wave_lists):
    """改道规则（与 `ui_dynamic._run_*` 同一条）。

    sambe 的光子块扩展空间需要**单一载频**，故遇任意波形（完整时域驱动、
    无载频）改道 hard；mixed 自 t79 起自己支持任意波形微波，**不改道**
    （否则会连光场一起从光子块降级到 RWA）。光强包络（FileEnv）是普通
    Wave，三者都不受影响。
    """
    if method == 'sambe' and any(has_arb(w) for w in wave_lists):
        return 'hard', ('sambe 不支持任意波形（完整驱动无载频）→ 自动改用 '
                        'hard；若文件只是光强形状，请按「＋任意 → 作为光强'
                        '包络」导入')
    return method, None


def _split_arb(waves):
    reg = [w for w in waves if not isinstance(w, _wf.ArbitraryWave)]
    arb = [w for w in waves if isinstance(w, _wf.ArbitraryWave)]
    return reg, arb


def build_fields_single(cache, method, states, waves, t_end_ns):
    """单体场容器（逐引擎分支与 GUI 同源）。

    返回 (fields, extra_kwargs, notes)。
    """
    ek, notes = {}, []
    if method == 'sambe':
        return build_fields_exact(cache, states, waves), ek, notes
    if method == 'mixed':
        reg, arb = _split_arb(waves)
        fields = build_fields_exact(cache, states, reg)
        if arb:
            ek['mw_arb'] = [arbitrary_mw_group(cache, states, w)
                            for w in arb]
            notes.append('mixed：任意波形微波 %d 条 → arbitrary_mw_group'
                         '（Hilbert 分解 env/φ）→ 微波项 cos(φ(t)) 当场积分'
                         % len(arb))
        notes.append('mixed：光场 Sambe 光子块 + 微波当场（φ-帧）')
        return fields, ek, notes
    # hard：两场 meta（光旋转 + 微波硬积分）
    reg, arb = _split_arb(waves)
    meta = build_two_field(cache, states, reg, t_end_ns=float(t_end_ns))
    for w in arb:
        meta['mw_groups'].append(arbitrary_mw_group(cache, states, w))
    return meta, ek, notes


def build_fields_pair(cache, method, states, waves_A, waves_B, t_end_ns):
    """两体场容器（逐引擎分支与 GUI 同源）。返回 (fA, fB, extra, notes)。"""
    ek, notes = {}, []
    if method == 'mixed':
        notes.append('mixed 两体：光场 Sambe 光子块 + 微波/V_dd 当场（φ-帧）')
        regA, arbA = _split_arb(waves_A)
        regB, arbB = _split_arb(waves_B)
        fA = build_fields_exact(cache, states, regA)
        fB = build_fields_exact(cache, states, regB)
        if arbA:
            ek['mw_arb_A'] = [arbitrary_mw_group(cache, states, w)
                              for w in arbA]
        if arbB:
            ek['mw_arb_B'] = [arbitrary_mw_group(cache, states, w)
                              for w in arbB]
        if arbA or arbB:
            notes.append('mixed 两体：任意波形微波 A=%d B=%d 条 → '
                         'arbitrary_mw_group（Hilbert 分解 env/φ）'
                         % (len(arbA), len(arbB)))
        return fA, fB, ek, notes
    if method == 'sambe':
        return (build_fields_exact(cache, states, waves_A),
                build_fields_exact(cache, states, waves_B), ek, notes)

    def _meta(wvs):
        reg, arb = _split_arb(wvs)
        m = build_two_field(cache, states, reg, t_end_ns=float(t_end_ns))
        for w in arb:
            m['mw_groups'].append(arbitrary_mw_group(cache, states, w))
        return m

    return _meta(waves_A), _meta(waves_B), ek, notes


# =====================================================================
# 两体 V_dd
# =====================================================================
_VDD_MEMO = {}


def vdd_matrix(cache, states, R_um, theta_deg=0.0, phi_deg=0.0, progress=None,
               use_memo=True):
    """两体 V_dd 稀疏矩阵（`static_pair.vdd_sparse` 的带缓存包装）。

    V_dd 只依赖 (基组, R, θ, φ)，**与频率/强度/包络无关** ⇒ 扫描里若轴不碰
    这三者，各点可复用同一个矩阵。缓存放在**进程内**（每个 worker 一份），
    所以并行扫描从"每点重算"降到"每 worker 算一次"。

    ⚠ 缓存键含基组指纹 + 几何，任何一项变了就重算，不会串味。
    """
    key = (states_fingerprint(states), round(float(R_um), 9),
           round(float(theta_deg), 9), round(float(phi_deg), 9))
    if use_memo and key in _VDD_MEMO:
        return _VDD_MEMO[key]
    m = pair_core.vdd_sparse(cache, [dict(s) for s in states],
                             float(R_um), float(theta_deg), float(phi_deg),
                             progress=progress)
    if use_memo:
        if len(_VDD_MEMO) > 8:                     # 别让缓存无界长
            _VDD_MEMO.clear()
        _VDD_MEMO[key] = m
    return m


# =====================================================================
# 总入口
# =====================================================================
def build_run(cache, params, *, pump=None, vdd=True, method=None, diag=False,
              t_grid=None, two_body=None):
    """登记册快照 + cache → RunSpec（纯容器，可 pickle 跨进程）。

    参数
      vdd     两体时是否就地算 V_dd（GUI 主线程路径 True；
              runner worker 传 False 后自行调 `vdd_matrix`，以便走进程内缓存）
      method  覆盖方法（None = 用快照里的 dynamic.method）
      diag    是否让引擎采集运行诊断
      t_grid  显式时间网格（None = 由 t_max/npts 生成）

    返回 dict，键见模块 docstring 与本函数末尾的 `spec` 组装。
    """
    p = params or {}
    mode = int(p.get('dynamic.mode') or 0)
    is_pair = (mode == 1) if two_body is None else bool(two_body)

    warnings = []
    groups = groups_from_params(p)
    if not groups:
        raise ValueError('基组为空：快照里 groups.list 没有能级组')
    states = build_states(cache, groups, p.get('field.B_gauss'),
                          p.get('field.E_vpcm'), pump=pump)
    if not states:
        raise ValueError('基组为空：能级组解析后没有任何子能级')

    meth_in = method or (p.get('dynamic.method') or 'mixed')
    if is_pair:
        waves_A = waves_for(p, 'A')
        waves_B = waves_for(p, 'B')
        meth, note = effective_method(meth_in, [waves_A, waves_B])
        if note:
            warnings.append(note)
        t_max = float(p.get('dynamic.t_max') or 0.0)
        npts = int(p.get('dynamic.npts') or 0)
        t = (np.asarray(t_grid, float) if t_grid is not None
             else np.linspace(0.0, t_max, max(npts, 2)))
        fA, fB, ek_extra, notes = build_fields_pair(
            cache, meth, states, waves_A, waves_B, float(t[-1]))
        psi0_A = resolve_psi(states, p.get('dynamic.psi_A'), 'A')
        psi0_B = resolve_psi(states, p.get('dynamic.psi_B'), 'B')
        psi0_pair = resolve_pair_psi(p.get('dynamic.psi_pair'))
        dcfg = p.get('dynamic.decays') or []
        d_index = p.get('dynamic.decay_index')
        decays_A = resolve_decays(cache, states, d_index, dcfg, two_body=True)
        decays_B = decays_A
        R = p.get('dynamic.pair.R_um')
        TH = p.get('dynamic.pair.theta_deg')
        PH = p.get('dynamic.pair.phi_deg')
        R = 0.0 if R is None else float(R)
        TH = 0.0 if TH is None else float(TH)
        PH = 0.0 if PH is None else float(PH)
        vdd_m = None
        if vdd:
            vdd_m = vdd_matrix(cache, states, R, TH, PH)
        # 引擎参数：与 `_collect_god_kwargs(pair=True)` + `_run_two_body` 同源
        # ——非 mixed 的运行级 kwargs **静默无效**，所以这里明确报出来
        # （核查报告 B1 的口径：不许"以为生效了"）。
        ek_all = god_kwargs_from_params(p, pair=False, diag=diag)
        pk = ek_all.pop('param_overrides', None)
        ek = {}
        if meth == 'mixed':
            ek, dropped = _filter_kw(ek_all, _MIXED_PAIR_KW)
            if dropped:
                warnings.append('[引擎参数] mixed 两体不接收这些键，已忽略：%s'
                                % ', '.join(sorted(dropped)))
        elif ek_all:
            warnings.append('[引擎参数] 两体 method=%s 不接收这些运行级参数，'
                            '已忽略：%s（仅 mixed 生效）'
                            % (meth, ', '.join(sorted(ek_all))))
        if pk:
            ek['param_overrides'] = pk
        frozen_h = per_M = None
        if meth == 'sambe':
            fh = p.get('solve.FROZEN_H_NS')
            frozen_h = float(fh) if fh else None
            per_M = parse_per_freq_m(p.get('solve.PER_FREQ_M'))
            ek.pop('frozen_h_ns', None)
            ek.pop('per_freq_M', None)
        ek.update(ek_extra)
        return {
            'version': SPEC_VERSION, 'two_body': True, 'method': meth,
            'method_in': meth_in, 't': t, 't_max': float(t_max), 'npts': npts,
            'states': states, 'labels': [s['label'] for s in states],
            'fields_A': fA, 'fields_B': fB,
            'psi0_A': psi0_A, 'psi0_B': psi0_B, 'psi0_pair': psi0_pair,
            'decays_A': decays_A, 'decays_B': decays_B,
            'vdd_matrix': vdd_m,
            'pair': {'R_um': R, 'theta_deg': TH, 'phi_deg': PH},
            'frozen_h_ns': frozen_h, 'per_freq_M': per_M,
            'n_waves': {'A': len(waves_A), 'B': len(waves_B)},
            'engine_kwargs': ek, 'warnings': warnings + notes,
        }

    waves = waves_for(p, 'single')
    meth, note = effective_method(meth_in, [waves])
    if note:
        warnings.append(note)
    t_max = float(p.get('dynamic.t_max') or 0.0)
    npts = int(p.get('dynamic.npts') or 0)
    t = (np.asarray(t_grid, float) if t_grid is not None
         else np.linspace(0.0, t_max, max(npts, 2)))
    fields, ek_extra, notes = build_fields_single(
        cache, meth, states, waves, float(t[-1]))
    psi0 = resolve_psi(states, p.get('dynamic.psi_single'), 'single')
    decays = resolve_decays(cache, states, p.get('dynamic.decay_index'),
                            p.get('dynamic.decays') or [], two_body=False)
    ek_all = god_kwargs_from_params(p, pair=False, diag=diag)
    pk = ek_all.pop('param_overrides', None)
    ek = {}
    if meth == 'mixed':
        ek, dropped = _filter_kw(ek_all, _MIXED_SINGLE_KW)
        if dropped:
            warnings.append('[引擎参数] mixed 不接收这些键，已忽略：%s'
                            % ', '.join(sorted(dropped)))
    elif ek_all:
        warnings.append('[引擎参数] method=%s 不接收这些运行级参数，已忽略：%s'
                        '（仅 mixed 生效；hard 的 max_step_hard 请经库 API '
                        '直调 hard.evolve_hard）'
                        % (meth, ', '.join(sorted(ek_all))))
    if pk:
        ek['param_overrides'] = pk
    ek.update(ek_extra)
    return {
        'version': SPEC_VERSION, 'two_body': False, 'method': meth,
        'method_in': meth_in, 't': t, 't_max': float(t_max), 'npts': npts,
        'states': states, 'labels': [s['label'] for s in states],
        'fields': fields,
        'psi0': psi0, 'psi0_pair': None,
        'decays': decays,
        'vdd_matrix': None, 'pair': None,
        'n_waves': {'single': len(waves)},
        'engine_kwargs': ek, 'warnings': warnings + notes,
    }


def summarize_run(run):
    """RunSpec → 人类/AI 可读摘要（不含大数组）。"""
    st = run['states']
    out = {
        'two_body': bool(run['two_body']),
        'method': run['method'],
        'n_states': len(st),
        'states_head': [s['label'] for s in st[:6]],
        't_max_ns': run['t_max'], 'npts': run['npts'],
        'n_decays': len(run.get('decays') or run.get('decays_A') or []),
        'psi0': (run.get('psi0') if not isinstance(run.get('psi0'), list)
                 else 'sup(%d)' % len(run['psi0'][1]))
        if not run['two_body'] else None,
        'psi0_A': (run.get('psi0_A') if not isinstance(run.get('psi0_A'), list)
                   else 'sup(%d)' % len(run['psi0_A'][1]))
        if run['two_body'] else None,
        'psi0_pair': ('%d 分量' % len(run['psi0_pair'][1])
                      if run.get('psi0_pair') else None),
        'engine_kwargs_keys': sorted(run.get('engine_kwargs') or {}),
        'warnings': list(run.get('warnings') or []),
    }
    if run['two_body']:
        out['pair'] = run.get('pair')
        out['vdd_nnz'] = (None if run.get('vdd_matrix') is None
                          else int(run['vdd_matrix'].nnz))
    return out


# =====================================================================
# 引擎调度（GUI worker 与并行 runner 的**同一条**引擎调用路径）
# =====================================================================
def evolve_run(run, progress_arr=None, log=None, cache=None):
    """RunSpec → 引擎结果 dict。

    逐分支照抄 `src/ui/evolve_worker.py` 的 EvolveWorker / EvolveWorkerPair
    （含 `params.override(param_overrides)` 的容差覆写、mixed 的 cache 透传、
    sambe 单体的"仅显式覆写 DEFAULT_RTOL 才传 tol"）。GUI 与 runner 都调它，
    所以"扫描"和"人点运行"走的是同一段引擎代码。

    progress_arr：[阶段内进度, 取消标志, 阶段] 三元数组（可为共享内存视图，
                  并行时父进程据此显示进度、写 [1] 即取消）。
    """
    from . import params as _P
    ek = dict(run.get('engine_kwargs') or {})
    ovr = dict(ek.pop('param_overrides', None) or {})
    meth = run['method']
    states = run['states']
    t = run['t']
    with _P.override(ovr):
        if run['two_body']:
            if meth == 'hard':
                from .hard import evolve_hard_pair
                return evolve_hard_pair(
                    states, run['fields_A'], run['fields_B'],
                    run['psi0_A'], run['psi0_B'],
                    t_grid=t, vdd_matrix=run['vdd_matrix'],
                    decays_A=run.get('decays_A'), decays_B=run.get('decays_B'),
                    progress_arr=progress_arr, psi0_pair=run.get('psi0_pair'),
                    log=log)
            if meth == 'sambe':
                from .sambe import evolve_sambe_pair
                return evolve_sambe_pair(
                    states, run['fields_A'], run['fields_B'],
                    run['psi0_A'], run['psi0_B'],
                    t_grid=t, vdd_matrix=run['vdd_matrix'],
                    decays_A=run.get('decays_A'), decays_B=run.get('decays_B'),
                    progress_arr=progress_arr, log=log,
                    frozen_h_ns=run.get('frozen_h_ns'),
                    per_freq_M=run.get('per_freq_M'),
                    psi0_pair=run.get('psi0_pair'))
            if meth == 'mixed':
                from .evolve_mixed import evolve_mixed_pair
                _ek = dict(ek)
                _ek.setdefault('tol', _P.get('DEFAULT_RTOL'))
                if run.get('psi0_pair') is not None:
                    _ek['psi0_pair'] = run['psi0_pair']
                return evolve_mixed_pair(
                    states, run['fields_A'], run['fields_B'],
                    run['psi0_A'], run['psi0_B'],
                    t_grid=t, vdd_matrix=run['vdd_matrix'],
                    decays_A=run.get('decays_A'), decays_B=run.get('decays_B'),
                    progress_arr=progress_arr, cache=cache, log=log, **_ek)
            raise ValueError('两体未知引擎 %r' % meth)
        if meth == 'hard':
            from .evolve_twofield import evolve_hard
            return evolve_hard(
                states, run['fields'], run['psi0'], t,
                rtol=_P.get('DEFAULT_RTOL'), atol=_P.get('DEFAULT_ATOL'),
                numba=True, progress_arr=progress_arr, log=log,
                decays=run.get('decays'))
        if meth == 'mixed':
            from .evolve_mixed import evolve_mixed
            _ek = dict(ek)
            _ek.setdefault('tol', _P.get('DEFAULT_RTOL'))
            return evolve_mixed(
                states, run['fields'], run['psi0'], t,
                progress_arr=progress_arr, log=log,
                decays=run.get('decays'), **_ek)
        if meth == 'sambe':
            from .sambe import evolve_sambe
            _sk = {}
            if 'DEFAULT_RTOL' in ovr:
                _sk['tol'] = ovr['DEFAULT_RTOL']
            return evolve_sambe(
                states, run['fields'], run['psi0'], t,
                progress_arr=progress_arr, log=log,
                decays=run.get('decays'), **_sk)
    raise ValueError('未知引擎 %r' % meth)


# =====================================================================
# 结果摘要（扫描的每点回执；与 server._res_summary 同口径）
# =====================================================================
def result_labels(run, res):
    """结果布居的态标签（单体取基组标签；两体取成对标签）。

    两体**必须**用 `res['states_2d']`——直接照搬单体标签会把 N² 维布居
    标错（server._res_labels 的 2026-09-10 教训）。
    """
    if res is None:
        return []
    pops2 = res.get('pops_2d')
    if pops2 is not None:
        s2 = res.get('states_2d')
        if s2 is None:
            return []
        out = []
        for x in np.asarray(s2).ravel():
            out.append(str(x.get('label', '?')) if isinstance(x, dict)
                       else str(x))
        return out
    return [s.get('label', '?') for s in (run.get('states') or [])]


def result_summary(run, res, top_n=8, want_labels=None):
    """结果 → 紧凑摘要（不搬大数组）：top-N 平均布居 + 终态布居。

    want_labels：可选标签片段列表 → 额外回传匹配项的**终态布居**（A6 口径）。
    """
    if res is None:
        return {'ok': False, 'msg': '无结果（可能被取消）'}
    labels = result_labels(run, res)
    arr = res.get('pops')
    if arr is None:
        arr = res.get('pops_2d')
    arr = np.asarray(arr)
    if arr.ndim < 2:
        arr = arr.reshape(1, -1)
    avg = arr.mean(axis=0)
    order = np.argsort(-avg)[:max(int(top_n), 1)]
    lab = (lambda i: labels[i] if i < len(labels) else '#%d' % i)
    out = {
        'ok': True,
        'n_t': int(arr.shape[0]),
        'n_states': int(arr.shape[1]),
        'top_states_avg_pop': {lab(int(i)): round(float(avg[i]), 6)
                               for i in order},
        'final_top_pop': {lab(int(i)): round(float(arr[-1][i]), 6)
                          for i in order},
    }
    if want_labels:
        sel = {}
        for frag in want_labels:
            hits = [i for i, L in enumerate(labels) if frag in L]
            if hits:
                sel[frag] = {labels[i]: round(float(arr[-1][i]), 6)
                             for i in hits}
        out['selected_final_pop'] = sel
    return out


def check_run(run):
    """RunSpec 的**运行前**自检（返回问题列表）。

    对应 S4 的"校验代替静默"：把已知会静默错物理的情形明确报出来。
    判据与 GUI `_run` 的校验同源（零波形 + 零耗散通道 = 空演化，GUI 也拒绝）。
    """
    problems = list(run.get('warnings') or [])
    t = np.asarray(run['t'], float)
    if len(t) < 2:
        problems.append('时间网格少于 2 点，演化无意义')
    n_w = run.get('n_waves') or {}
    if not sum(int(v or 0) for v in n_w.values()):
        has_decay = bool(run.get('decays') or run.get('decays_A'))
        if not has_decay:
            problems.append(
                '既无波形也无耗散通道 → 空演化（先加波形，或开自发辐射/'
                '自定义衰变通道）')
    return problems


def _all_waves_of(run):
    """从 RunSpec 的场容器里反推波形条数（只作诊断用）。

    ⚠ "有没有驱动"的判据**不用它**（改用 build_run 记下的 `n_waves`：
    场容器按引擎不同形态不同，反推容易误判）。
    """
    out = []
    for f in (run.get('fields'), run.get('fields_A'), run.get('fields_B')):
        if f is None:
            continue
        if isinstance(f, dict):                    # two-field meta
            out.extend(f.get('mw_groups') or [])
            if f.get('opt') is not None:
                out.append(f.get('opt'))
        else:
            out.extend(f or [])
    return out


# =====================================================================
# 深拷贝工具（扫描里每点都要基于同一份 params 派生）
# =====================================================================
def clone_params(p):
    """params 快照深拷贝（list/dict/标量；numpy 数组按需）。"""
    return copy.deepcopy(p)
