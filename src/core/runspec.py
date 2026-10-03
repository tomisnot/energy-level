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
import inspect

import numpy as np

from . import static_atom as core
from .atomic_levels import make_levels, find_state_index, state_key
from . import waveforms as _wf
from .field_build import build_fields_exact, build_field_loss
from .evolve_twofield import build_two_field, arbitrary_mw_group
from .evolve_util import decay, auto_decays
from . import static_pair as pair_core

from .evolve_mixed import DEFAULT_ATOL
from .evolve_mixed import DEFAULT_RTOL

SPEC_VERSION = 1

# solve.* → 下游 kwarg。**本表是唯一声明点**——`protocol.solve_schema()` 的
# solve.* 记录与 GUI 检视器都从本表派生，没有第二份手抄。
# （v1 时代 GUI 侧 `pm_setup._SOLVE_SPECS` 曾手抄一份、靠判据钉镜像；
#   pm_setup 已退役，单一真源后那面守卫随 v1 清洗移除——见 sweep_spec H 段注释。）
SOLVE_KWARG = {
    'MAX_STEP': 'max_step',
    'MAX_STEP_HARD': 'max_step_hard',
    'PER_FREQ_M': 'per_freq_M',
    'FROZEN_H_NS': 'frozen_h_ns',
    'COST_BUDGET': 'phase_budget',
}

def _required(p, key):
    """快照里的**必填**登记册键；缺失/为 None → 报错（**不静默兜底**）。

    与 `resolve_psi` 的"绝不静默回落"同一纪律：`p.get(key) or 默认` 会把
    "键不存在"和"值就是默认"混成一件事——键名打错就静默跑一个空演化。
    """
    if not isinstance(p, dict) or p.get(key) is None:
        raise ValueError(
            '快照缺少必填键 %r（登记册路径；不静默兜底）——'
            '若这是旧存档，请先补齐该字段' % key)
    return p[key]


# (method, nbody) → 引擎函数。**唯一分派表**：`build_run` 按它的**签名**筛参数，
# `evolve_run` 按它调用——"哪个 method 走哪个函数"只有这一个事实源。
# nbody ∈ {'single','pair','many'}（旧 two_body:bool 经 _as_nbody 兼容映射）。
# manybody 走惰性 import ⇒ core 顶层不 import manybody（无环）。
_ENGINE_TABLE = {
    ('mixed', 'single'): ('.evolve_mixed', 'evolve_mixed'),
    ('mixed', 'pair'): ('.evolve_mixed', 'evolve_mixed_pair'),
    ('sambe', 'single'): ('.sambe', 'evolve_sambe'),
    ('sambe', 'pair'): ('.sambe', 'evolve_sambe_pair'),
    ('hard', 'single'): ('.hard', 'evolve_hard'),
    ('hard', 'pair'): ('.hard', 'evolve_hard_pair'),
    ('ed', 'many'): ('.manybody.engine_ed', 'evolve_manybody'),
    ('tebd', 'many'): ('.manybody.engine_ed', 'evolve_tebd_stub'),
    # 平衡态产品入口（manybody.eq_engine）：method 即请求量，走 eq_route 路由到
    # ed/gpu-batch/lanczos/kpm/dmrg。spec['equilibrium']=True 时由 evolve_run 旁路。
    ('gs', 'many'): ('.manybody.eq_engine', 'run_equilibrium'),
    ('thermal', 'many'): ('.manybody.eq_engine', 'run_equilibrium'),
    ('dos', 'many'): ('.manybody.eq_engine', 'run_equilibrium'),
}

_NBODY_OF_BOOL = {False: 'single', True: 'pair'}


def _as_nbody(nbody):
    """two_body:bool（旧调用）| 'single'/'pair'/'many' → 规范 nbody 串。"""
    if isinstance(nbody, bool):
        return _NBODY_OF_BOOL[nbody]
    return str(nbody)


def engine_fn(method, nbody):
    """(method, nbody) → 引擎函数（唯一分派表的查询口，惰性 import）。

    nbody 收 bool（旧：False=single/True=pair）或 'single'/'pair'/'many'。
    """
    key = (str(method), _as_nbody(nbody))
    if key not in _ENGINE_TABLE:
        raise ValueError('未知引擎 method=%r nbody=%s；合法：%s'
                         % (method, key[1], sorted(_ENGINE_TABLE)))
    mod, name = _ENGINE_TABLE[key]
    import importlib
    return getattr(importlib.import_module(mod, __package__), name)


# =====================================================================
# 基组
# =====================================================================
def groups_from_params(p):
    """登记册快照 → 能级组列表（**原样**保留 color/ms/Fs 等附加字段）。"""
    return [dict(g) for g in ((p or {}).get('groups.list') or [])]


def build_states(cache, groups, B_gauss=0.0, E_vpcm=0.0, pump=None,
                 with_static=False):
    """登记册里的能级组 → 动态基组（`make_levels` 的 states）。

    枚举、能量、逐子能级裁剪**全部走 `static_atom.base_sublevels`**——
    与静态页同一实现，两页的能级基准不会漂移。这里只把它投影成
    `make_levels` 要的 specs，并**把 `base_sublevels` 算出的 `label` 显式传下去**
    （不让 `make_levels` 自己再拼一次）⇒ 两页的态名**同源同字**。
    `E_hf` 留 0，因为行里的 `E_mhz` 已含超精细项。

    pump：可选 callable(frac)——每个能级组完成后上报。
    `with_static=True`（2026-09-22，P1-6）：把 `qstr` / `dc_shift_mhz` /
    `group_label` / `color` 一并透传进 states（供 `protocol.list_levels` 的
    静态列用）。默认 False ⇒ 与改前**逐位相同**。
    """
    B = float(B_gauss or 0.0)
    Ev = float(E_vpcm or 0.0)
    specs = []
    for r in core.base_sublevels(cache, groups, B, Ev, pump=pump,
                                 with_dc=with_static):
        common = dict(n=r['n'], l=r['l'], j=r['j'], E_mhz=r['E_mhz'],
                      label=r['label'])
        if with_static:
            for k in ('qstr', 'dc_shift_mhz', 'group_label', 'color'):
                if k in r:
                    common[k] = r[k]
        if r['kind'] == 'F':
            specs.append(dict(common, kind='F', F=r['F'], mF=r['mF'],
                              E_hf=0.0, amps=r['amps']))
        else:
            specs.append(dict(common, kind='mJ', m=float(r['mj'])))
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
    if mode not in (0, 1, 2):
        raise ValueError(
            'dynamic.decay_index 只认 0（自动）/ 1（禁用）/ 2（自定义），'
            '得到 %r——不静默当成"无衰变"' % decay_index)
    if mode == 1:
        return []
    if mode == 0:
        return auto_decays(cache, states)
    labels = [s['label'] for s in states]
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
    """登记册快照里的 `solve.*` → 引擎 kwargs。

    ⚠ **`engine.*`（引擎参数）本阶段不接线**：参数已就地定义在各引擎模块里
    （全局注册表已删除），运行级覆写还没有"映射到引擎形参"的通道。
    快照里若带 `engine.*` 键，`build_run` 会**明确写进 warnings**——不静默。

    pair=True：剔除 `per_freq_M`（两体 P 固定，GUI 也把它走独立形参）。
    """
    ek = {}
    for k, v in (p or {}).items():
        if not isinstance(k, str) or v is None:
            continue
        if k.startswith('solve.'):
            kw = SOLVE_KWARG.get(k.split('.', 1)[1])
            if kw is not None:
                ek[kw] = v
    if pair:
        ek.pop('per_freq_M', None)
    if diag:
        ek['diag'] = True
    return ek


def _engine_accepts(fn, key):
    """引擎签名是否收该 kwarg。

    六个引擎都没有 `**kwargs`，所以就是查形参名。登记册里 `solve.*` 的**读回值**
    （GUI 控件当前值）会一起进 kwargs，而它们不都是每个引擎认得的形参
    （`frozen_h_ns` 是 sambe 的旋钮，`evolve_mixed*` 签名里没有它）；不筛就会
    `TypeError: unexpected keyword argument`。
    """
    return key in inspect.signature(fn).parameters


def _filter_kw(fn, d):
    """kwargs → (只留**引擎签名**收的, 被丢掉的键集合)。

    白名单 = 签名本身，不是手抄表——引擎改签名，筛子自动跟。
    """
    keep, drop = {}, set()
    for k, v in (d or {}).items():
        if _engine_accepts(fn, k):
            keep[k] = v
        else:
            drop.add(k)
    return keep, drop




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
def _time_grid(p, t_grid):
    """时间网格 → `(t_max, npts, t)`。

    `dynamic.t_max` / `dynamic.npts` 是**必填**（`_required`）：
    `p.get(k) or 默认` 会把"键不存在"和"值就是默认"混成一件事——
    键名打错就静默跑一个 0 ns 的空演化，而且 `check_run` 抓不到。
    """
    t_max = float(_required(p, 'dynamic.t_max'))
    npts = int(_required(p, 'dynamic.npts'))
    if t_grid is None and t_max <= 0.0:
        raise ValueError(
            'dynamic.t_max 必须 > 0：0 ns 演化没有意义（得到 %r）' % t_max)
    if npts < 2:
        raise ValueError('dynamic.npts 必须 >= 2（得到 %r）' % npts)
    t = (np.asarray(t_grid, float) if t_grid is not None
         else np.linspace(0.0, t_max, npts))
    return t_max, npts, t


def _engine_kwargs(p, meth, is_pair, diag, warnings):
    """运行级 kwargs：按**引擎签名**筛（白名单 = 签名，不是手抄表）。

    被丢掉的键写进 `warnings`——不许"以为生效了"（核查报告 B1 的口径）。
    """
    ek_all = god_kwargs_from_params(p, pair=False, diag=diag)
    ek, dropped = _filter_kw(engine_fn(meth, is_pair), ek_all)
    if dropped:
        warnings.append('[引擎参数] %s%s不接收这些键，已忽略：%s'
                        % (meth, ' 两体' if is_pair else '',
                           ', '.join(sorted(dropped))))
    return ek


def _f0(v):
    """None → 0.0，否则 float(v)（两体几何的缺省）。"""
    return 0.0 if v is None else float(v)


def build_run(cache, params, *, pump=None, vdd=True, method=None, diag=False,
              t_grid=None, two_body=None):
    """登记册快照 + cache → RunSpec（纯容器，可 pickle 跨进程）。

    参数
      vdd     两体时是否就地算 V_dd（GUI 主线程路径 True；
              runner worker 传 False 后自行调 `vdd_matrix`，以便走进程内缓存）
      method  覆盖方法（None = 用快照里的 dynamic.method）
      diag    是否让引擎采集运行诊断
      t_grid  显式时间网格（None = 由 t_max/npts 生成）

    结构：**公共四步**（基组 → 波形/方法 → 时间网格 → 引擎参数）
    然后按单体/两体各自组装。两体的差异只在
    `fields_A/B`、`psi0_A/B/psi0_pair`、`vdd_matrix`、`pair`、`frozen_h_ns`。
    """
    p = params or {}
    mode = int(p.get('dynamic.mode') or 0)
    if mode == 2:
        # 多体（N 原子阵列）：同一 RunSpec 契约；calibrate 在此主线程碰 ARC/SQLite。
        # 平衡态请求 manybody.want∈gs/thermal/dos → 委托 eq_engine（不走时间轴，
        # 经 eq_route 路由 ed/gpu-batch/lanczos/kpm/dmrg）；否则走含时 ED 演化。
        _w = str(p.get('manybody.want') or '').lower()
        if _w in ('gs', 'thermal', 'dos'):
            from .manybody.eq_engine import build_equilibrium
            return build_equilibrium(cache, p, want=_w,
                                     method=(p.get('manybody.eq_backend')
                                             or p.get('manybody.solver')),
                                     betas=p.get('manybody.betas'))
        from .manybody.runspec_mb import build_run_manybody
        return build_run_manybody(cache, p, pump=pump, method=method,
                                  diag=diag, t_grid=t_grid)
    is_pair = (mode == 1) if two_body is None else bool(two_body)

    warnings = []
    groups = groups_from_params(p)
    if not groups:
        raise ValueError('基组为空：快照里 groups.list 没有能级组')
    states = build_states(cache, groups, p.get('field.B_gauss'),
                          p.get('field.E_vpcm'), pump=pump)
    if not states:
        raise ValueError('基组为空：能级组解析后没有任何子能级')

    # ---- 公共 ①：波形 + 实际方法（改道规则）----
    meth_in = method or (p.get('dynamic.method') or 'mixed')
    wl = [waves_for(p, a) for a in (('A', 'B') if is_pair else ('single',))]
    meth, note = effective_method(meth_in, wl)
    if note:
        warnings.append(note)

    # ---- 公共 ②：时间网格 ----
    t_max, npts, t = _time_grid(p, t_grid)

    # ---- 公共 ③：场（体相关）----
    if is_pair:
        fA, fB, ek_extra, notes = build_fields_pair(
            cache, meth, states, wl[0], wl[1], float(t[-1]))
    else:
        fields, ek_extra, notes = build_fields_single(
            cache, meth, states, wl[0], float(t[-1]))

    # ---- 公共 ④：引擎参数（按签名筛）+ 返回骨架 ----
    ek = _engine_kwargs(p, meth, is_pair, diag, warnings)
    ek.update(ek_extra)
    run = {
        'version': SPEC_VERSION, 'two_body': is_pair, 'method': meth,
        'method_in': meth_in, 't': t, 't_max': t_max, 'npts': npts,
        'states': states, 'labels': [s['label'] for s in states],
        'engine_kwargs': ek, 'warnings': warnings + notes,
    }

    # ---- 体相关组装 ----
    d_index = p.get('dynamic.decay_index')
    dcfgs = p.get('dynamic.decays') or []
    if is_pair:
        decays_A = resolve_decays(cache, states, d_index, dcfgs, two_body=True)
        R = _f0(p.get('dynamic.pair.R_um'))
        TH = _f0(p.get('dynamic.pair.theta_deg'))
        PH = _f0(p.get('dynamic.pair.phi_deg'))
        frozen_h = per_M = None
        if meth == 'sambe':
            fh = p.get('solve.FROZEN_H_NS')
            frozen_h = float(fh) if fh else None
            per_M = parse_per_freq_m(p.get('solve.PER_FREQ_M'))
            ek.pop('frozen_h_ns', None)
            ek.pop('per_freq_M', None)
        run.update({
            'fields_A': fA, 'fields_B': fB,
            'psi0_A': resolve_psi(states, p.get('dynamic.psi_A'), 'A'),
            'psi0_B': resolve_psi(states, p.get('dynamic.psi_B'), 'B'),
            'psi0_pair': resolve_pair_psi(p.get('dynamic.psi_pair')),
            'decays_A': decays_A, 'decays_B': decays_A,
            'vdd_matrix': (vdd_matrix(cache, states, R, TH, PH) if vdd
                           else None),
            'pair': {'R_um': R, 'theta_deg': TH, 'phi_deg': PH},
            'frozen_h_ns': frozen_h, 'per_freq_M': per_M,
            'n_waves': {'A': len(wl[0]), 'B': len(wl[1])},
            'field_loss': None,               # 两体场致损耗 → H3b
        })
        return run

    decays = resolve_decays(cache, states, d_index, dcfgs, two_body=False)
    run.update({
        'fields': fields,
        'psi0': resolve_psi(states, p.get('dynamic.psi_single'), 'single'),
        'psi0_pair': None,
        'decays': decays,
        'vdd_matrix': None, 'pair': None,
        'n_waves': {'single': len(wl[0])},
    })
    # 场致加热与损失（opt-in）：登记册无 dynamic.field_loss → None（默认关，
    # 引擎侧 H3 才消费）。此处只装配、不改变任何现有演化数值。
    fl_spec = p.get('dynamic.field_loss')
    run['field_loss'] = (build_field_loss(cache, states, fields, decays,
                                          fl_spec) if fl_spec else None)
    return run


def summarize_run(run):
    """RunSpec → 人类/AI 可读摘要（不含大数组）。"""
    if run.get('equilibrium'):
        # 平衡态请求（gs/thermal/dos）：**无时间轴/无初态**，不套用含时的
        # t_max/npts/psi0 字段（旧写法会在 `run['t_max']` KeyError）。
        return {
            'two_body': False, 'nbody': run.get('nbody', 'many'),
            'equilibrium': True, 'want': run.get('want'),
            'method': run.get('method'), 'solver': run.get('method'),
            'backend_hint': run.get('method_in'),
            'n_atoms': run.get('n_atoms'), 'dim': run.get('dim'),
            'terms': run.get('terms'),
            'betas': (list(run.get('betas')) if run.get('betas') else None),
            'kpm': dict(run.get('kpm') or {}),
            'states_head': [s['label'] for s in (run.get('states') or [])[:6]],
            'warnings': list(run.get('warnings') or []),
        }
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
    if run.get('nbody') == 'many':
        out.update(nbody='many', n_atoms=run.get('n_atoms'),
                   dim=run.get('dim'), terms=run.get('terms'),
                   template=run.get('template'), solver=run.get('method'))
        return out
    if run['two_body']:
        out['pair'] = run.get('pair')
        out['vdd_nnz'] = (None if run.get('vdd_matrix') is None
                          else int(run['vdd_matrix'].nnz))
    return out


# =====================================================================
# 引擎调度 —— **唯一分派点**
# =====================================================================
def evolve_run(run, progress_arr=None, log=None, cache=None):
    """RunSpec → 引擎结果 dict。

    这是全仓**唯一**的 method×n_body 分派点（引擎函数由 `_ENGINE_TABLE` 给出）。
    分支只负责"给这个引擎备料"（引擎函数来自唯一分派表 `engine_fn`）：
    mixed 的 cache 透传、sambe 单体的"仅显式覆写 DEFAULT_RTOL 才传 tol"。
    GUI 与并行 runner 都调它 ⇒ "扫描"与"人点运行"走的是同一段引擎代码。

    progress_arr：[阶段内进度, 取消标志, 阶段] 三元数组（可为共享内存视图，
                  并行时父进程据此显示进度、写 [1] 即取消）。
    """
    ek = dict(run.get('engine_kwargs') or {})
    ovr = dict(ek.pop('param_overrides', None) or {})
    if ovr:
        raise ValueError(
            '运行级引擎参数覆写尚未接线（参数已就地定义在各引擎模块）：%s'
            % ', '.join(sorted(ovr)))  # 早失败，不静默
    meth = run['method']
    if run.get('nbody') == 'many':
        # 平衡态旁路（want ∈ gs/thermal/dos）：不走时间轴，直接由 eq_engine 路由求解。
        # 分派键用 want（→run_equilibrium）；run['method'] 存的是后端（ed/kpm/dmrg/...）。
        if run.get('equilibrium'):
            fn = engine_fn(run.get('want', 'gs'), 'many')
            return fn(run, progress_arr=progress_arr, log=log)
        # 多体含时：唯一分派表取 engine_fn(meth,'many')；分支只备料（透传 calib/驱动/psi0）
        fn = engine_fn(meth, 'many')
        return fn(run['calib'], run.get('psi0'), run['t'],
                  Om=run.get('Om'), delta=run.get('delta'),
                  gamma=run.get('gamma'), terms=run.get('terms'),
                  segments=run.get('segments'),
                  progress_arr=progress_arr, log=log, **ek)
    tb = bool(run['two_body'])
    # 引擎函数来自**唯一分派表**；下面的分支只负责"给这个引擎备料"
    # （mixed 的 cache 透传、sambe 单体的"仅显式覆写 DEFAULT_RTOL 才传 tol"等）。
    fn = engine_fn(meth, tb)
    states = run['states']
    t = run['t']
    if tb:
        if meth == 'hard':
            return fn(
                states, run['fields_A'], run['fields_B'],
                run['psi0_A'], run['psi0_B'],
                t_grid=t, vdd_matrix=run['vdd_matrix'],
                decays_A=run.get('decays_A'), decays_B=run.get('decays_B'),
                progress_arr=progress_arr, psi0_pair=run.get('psi0_pair'),
                max_step_hard=ek.get('max_step_hard'),
                log=log)
        if meth == 'sambe':
            return fn(
                states, run['fields_A'], run['fields_B'],
                run['psi0_A'], run['psi0_B'],
                t_grid=t, vdd_matrix=run['vdd_matrix'],
                decays_A=run.get('decays_A'), decays_B=run.get('decays_B'),
                progress_arr=progress_arr, log=log,
                frozen_h_ns=run.get('frozen_h_ns'),
                per_freq_M=run.get('per_freq_M'),
                psi0_pair=run.get('psi0_pair'))
        if meth == 'mixed':
            _ek = dict(ek)
            _ek.setdefault('tol', DEFAULT_RTOL)
            if run.get('psi0_pair') is not None:
                _ek['psi0_pair'] = run['psi0_pair']
            return fn(
                states, run['fields_A'], run['fields_B'],
                run['psi0_A'], run['psi0_B'],
                t_grid=t, vdd_matrix=run['vdd_matrix'],
                decays_A=run.get('decays_A'), decays_B=run.get('decays_B'),
                progress_arr=progress_arr, cache=cache, log=log, **_ek)
    if meth == 'hard':
        return fn(
            states, run['fields'], run['psi0'], t,
            rtol=DEFAULT_RTOL, atol=DEFAULT_ATOL,
            numba=True, progress_arr=progress_arr, log=log,
            decays=run.get('decays'),
            max_step_hard=ek.get('max_step_hard'))
    if meth == 'mixed':
        _ek = dict(ek)
        _ek.setdefault('tol', DEFAULT_RTOL)
        return fn(
            states, run['fields'], run['psi0'], t,
            progress_arr=progress_arr, log=log,
            decays=run.get('decays'), field_loss=run.get('field_loss'), **_ek)
    if meth == 'sambe':
        return fn(
            states, run['fields'], run['psi0'], t,
            progress_arr=progress_arr, log=log,
            decays=run.get('decays'))
    raise ValueError('未知引擎 %r' % meth)


# =====================================================================
# 结果摘要（扫描的每点回执；与 server._res_summary 同口径）
# =====================================================================
def result_labels(run, res):
    """结果布居的态标签。

    **统一契约**：所有引擎的 `res['states']` 就是 `res['pops']` 各列的身份
    （单体 N 个、两体 N² 个），每个含 `label`。所以这里**不再按单体/两体分支**。

    旧写法要特判 `res['states_2d']`，漏判就会把 N² 维布居标错
    （`server._res_labels` 的 2026-09-10 教训）——契约统一后那个分支消失了。
    """
    if res is None:
        return []
    st = res.get('states') or (run.get('states') or [])
    out = []
    for x in np.asarray(st).ravel():
        out.append(str(x.get('label', '?')) if isinstance(x, dict)
                   else str(x))
    return out


def result_summary(run, res, top_n=8, want_labels=None):
    """结果 → 紧凑摘要（不搬大数组）：top-N 平均布居 + 终态布居。

    want_labels：可选标签片段列表 → 额外回传匹配项的**终态布居**（A6 口径）。
    """
    if res is None:
        return {'ok': False, 'msg': '无结果（可能被取消）'}
    if run.get('equilibrium') or (res.get('pops') is None and res.get('want')):
        return _equilibrium_summary(run, res)
    labels = result_labels(run, res)
    arr = res.get('pops')
    if arr is None:
        return {'ok': False, 'msg': '结果里没有 pops（引擎契约被破坏）'}
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


def _equilibrium_summary(run, res):
    """平衡态 Result → 紧凑摘要（数组走 artifact，这里只给读数）。

    旧 `result_summary` 只要 `pops`，平衡态结果无 pops ⇒ 会错报“引擎契约被破坏”。
    """
    want = str(res.get('want') or run.get('want') or '')
    n = int(run.get('n_atoms') or res.get('n_sites') or 0)
    out = {'ok': True, 'equilibrium': True, 'want': want,
           'backend': res.get('backend'), 'n_atoms': n,
           'dim': int(run.get('dim') or (1 << n) if n else 0)}
    r6 = lambda v: round(float(v), 6)          # noqa: E731
    if want == 'gs':
        if res.get('E0') is not None:
            out['E0'] = round(float(res['E0']), 9)
        for k in ('gap', 'chi', 'S'):
            if res.get(k) is not None:
                out[k] = round(float(res[k]), 9)
        na = res.get('n_atom')
        if na is not None:
            out['n_atom'] = [r6(v) for v in np.asarray(na).ravel()]
        return out
    if want == 'thermal':
        b = np.asarray(res.get('betas') or [], float)
        out['n_betas'] = int(b.size)
        if b.size:
            out['betas'] = [r6(v) for v in b[:8]]
            for key in ('E', 'C', 'F'):
                a = np.asarray(res.get(key) or [], float)
                if a.size == b.size:
                    out['%s_first_last' % key] = [r6(a[0]), r6(a[-1])]
        return out
    if want == 'dos':
        e = np.asarray(res.get('energies') or [], float)
        rho = np.asarray(res.get('rho') or [], float)
        out['n_bins'] = int(e.size)
        if e.size and rho.size:
            out['E_range'] = [r6(e[0]), r6(e[-1])]
            out['rho_max'] = r6(rho.max())
            out['E_at_rho_max'] = r6(float(e[int(np.argmax(rho))]))
        return out
    return {'ok': False, 'msg': '平衡态 want=%r 不认识（应为 gs/thermal/dos）' % want}


def check_run(run):
    """RunSpec 的**运行前**自检（返回问题列表）。

    对应 S4 的"校验代替静默"：把已知会静默错物理的情形明确报出来。
    判据与 GUI `_run` 的校验同源（零波形 + 零耗散通道 = 空演化，GUI 也拒绝）。
    """
    problems = list(run.get('warnings') or [])
    if run.get('equilibrium'):
        # 平衡态（gs/thermal/dos）**不走时间轴** ⇒ 不能套用含时的时网格/空演化判据
        #（旧写法在 `run['t']` 直接 KeyError，GUI/AI 经 ops.prepare 就挂）。
        if int(run.get('n_atoms') or 0) <= 0:
            problems.append('多体平衡态：没有原子（manybody.atoms 为空）')
        w = str(run.get('want') or '')
        if w not in ('gs', 'thermal', 'dos'):
            problems.append('多体平衡态：want 需为 gs/thermal/dos，实=%r' % w)
        if w == 'thermal':
            bt = [float(b) for b in (run.get('betas') or [])]
            if not bt:
                problems.append('thermal 需要 betas（逆温度清单，1/MHz，如 [0.1,1,10]）')
            elif any(b <= 0 for b in bt):
                problems.append('betas 必须全为正数（1/MHz）')
        if str(run.get('method') or '') == 'dmrg' and not run.get('ladder') \
                and run.get('dmrg_ok') is False:
            # 自写 autoMPO 后任意几何都能 DMRG；这里只提醒“排序不好可能很慢”（非阻断）。
            problems.append('dmrg：点云蛇形排序后排键较长（长程键多），MPS 键数会较大；'
                            '如太慢可改 auto/lanczos')
        return problems
    t = np.asarray(run['t'], float)
    if len(t) < 2:
        problems.append('时间网格少于 2 点，演化无意义')
    if run.get('nbody') == 'many':
        # 多体无"波形"概念（驱动来自 atoms 的 Ω/δ）：只查原子数，不套用
        # 单/两体的"既无波形也无耗散=空演化"判据（对多体是误报）
        if int(run.get('n_atoms') or 0) <= 0:
            problems.append('多体：没有原子（manybody.atoms 为空）')
        return problems
    n_w = run.get('n_waves') or {}
    if not sum(int(v or 0) for v in n_w.values()):
        has_decay = bool(run.get('decays') or run.get('decays_A'))
        if not has_decay:
            problems.append(
                '既无波形也无耗散通道 → 空演化（先加波形，或开自发辐射/'
                '自定义衰变通道）')
    # 场致损耗的**唯一消费者是单体 mixed**（`evolve_run` 的 mixed 单体分支）。
    # 其余路径（hard / sambe / reference / 两体 / 多体）会**静默忽略**它 ⇒ 明确报出来
    # （2026-09-22，工单 P2-5；D5：不许"以为生效了"）。
    if run.get('field_loss') is not None and not (
            run.get('method') == 'mixed' and not run.get('two_body')):
        problems.append(
            'dynamic.field_loss 已配置，但当前路径（method=%s%s）**不消费**场致损耗'
            '——它只在「单体 mixed」生效；请改用 mixed 单体，或清空 '
            'dynamic.field_loss（不静默忽略）'
            % (run.get('method'), ' 两体' if run.get('two_body') else ''))
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
