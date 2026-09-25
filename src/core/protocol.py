# -*- coding: utf-8 -*-
"""引擎调用协议（core 侧）——「引擎接受什么配置、怎么跑、能预估/预览什么、波形怎么校验」
的**唯一领域声明点**。

"领域知识回 core"的落点：原先散在 v1 `mecha/registry`（CONFIG_SCHEMA/SOLVE_META/
engine_params 扫描/wave_list 校验/runspec_read_keys）与 `mecha/ops`（estimate/preview/
list_levels）里的**物理/引擎知识**，全部收归此处。`mecha` v2 是**领域无关框架**，
经本仓 `mecha_v2/engine.py` 的 ElEngine（import 本协议）消费；
本协议**不 import mecha**（守 `core ↛ mecha`）。

单位与 core 一致：B=Gauss、E=V/cm、t=ns、微波 freq=MHz、光频=Hz、Γ=MHz。
"""
import ast
import importlib
import os

import numpy as np          # 静态查询（P1-6）要回列数组（让库的体积闸无损转指针）

from . import runspec as _runspec
from .manybody import schema_mb as _mb_schema   # 多体配置契约（schema_mb 只 import param，无环）
# P1-7（2026-09-22）：AC-Stark **静态**精度旋钮的引擎默认。CONFIG_SCHEMA 的
# `field.window` / `field.n_photon` 默认值**派生**自这里（不手抄数字 ⇒ 不会漂）。
# 二者只被静态通路消费（`static_atom._ac_precision`），动态侧 `runspec.py` 零命中。
from .static_atom import AC_N_PHOTON, AC_SOS_N_WINDOW
from . import static_atom as SA          # 静态查询（P1-6）要 build_level_rows_direct 等

__all__ = ['CONFIG_SCHEMA', 'SOLVE_META', 'config_schema', 'solve_schema',
           'solve_map', 'engine_params', 'engine_param_names', 'settable_keys',
           'validate_wave_list', 'validate_atom_list', 'build_run', 'check_run',
           'evolve_run', 'result_summary', 'summarize_run', 'make_cache',
           'estimate', 'preview', 'list_levels', 'count_waves',
           'wave_acfg', 'address_cfg_from_snapshot',
           'static_spectrum', 'static_ac_sweep', 'static_pair',
           'runspec_read_keys']

_HERE = os.path.dirname(os.path.abspath(__file__))

# =====================================================================
# 配置键 schema（groups.* / field.* / dynamic.*）——build_run 读的登记册快照键。
# 值域/必填**投影自 runspec 自身的校验**；`runspec_read_keys()` AST 交叉防漂（I6）。
# =====================================================================
CONFIG_SCHEMA = {
    'groups.list': dict(type='list', required=True,
                        desc='能级组列表（每项 n/l/j + 可选子能级裁剪 ms/Fs）'),
    'field.B_gauss': dict(type='float', unit='Gauss', desc='静磁场（Zeeman）'),
    'field.E_vpcm': dict(type='float', unit='V/cm',
                         desc='静电场（DC-Stark）；n=60 的 Inglis-Teller 约 25 V/cm'),
    # ---- P1-7（2026-09-22）：AC-Stark **静态**精度旋钮 ----
    # 「窗口 / 光子块」两个控件原先**静默失能**（人改了无反应、AI 也改不到），根因是
    # 引擎侧 `AC_N_PHOTON`/`AC_SOS_N_WINDOW` 是**裸模块常量**、且没有可写配置键。
    # 现在：① 两者包成 `Param`（registry 收集得到，family='engine' 只读）；
    #       ② 这两个键给出**可写**入口（default 派生自引擎 Param，不手抄）。
    # ⚠ **只对静态生效**：消费方是 `static_atom._ac_precision`（经 build_level_rows /
    #   direct_ac_stark_shifts 的**显式形参**），**动态侧 runspec.py 零命中**——
    #   AC 位移是静态显示量、不参与演化（`static_atom.py` 的 519-522 行）。接线时
    #   不要把这两个键引进 build_run/evolve_run，否则会破坏"两页同基准"的前提。
    'field.n_photon': dict(type='int', default=int(AC_N_PHOTON), lo=1, hi=12,
                           desc='AC-Stark 光子块阶数（块数=2n+1）；静态页显示精度，'
                                '不参与演化。n 越大越准越慢（n=8 时 ~61s/次）'),
    'field.window': dict(type='int', default=int(AC_SOS_N_WINDOW), lo=1, hi=5,
                         desc='AC-Stark 偶极伙伴窗 |Δn|≤本值（Floquet 基组）；'
                              '静态页显示精度，不参与演化'),
    # ---- P1-6（2026-09-22）：静态列的 opt-in 开关 ----
    # 为什么是**快照键**而不是函数形参：`list_levels` 这条链（server/ops/protocol/
    # adapter）**四处都是无参的** ⇒ 没有 `**opts` 可透传。走快照键 ⇒ 零 mecha-sdk
    # 改动，且本键会出现在 list_params 里（带代价说明）⇒ 对 AI **可发现**。
    'static.columns': dict(
        type='str', enum=['off', 'basic', 'ac'], default='off',
        desc='list_levels 是否附静态列：off（默认，逐位不变）/ '
             'basic（+ E0_mhz/dc_shift_mhz/qstr/group_label/color/amps；'
             '**无额外代价**，与 base_sublevels 同一遍）/ '
             'ac（再 + ac_shift_mhz/ac_affected；**需一次 Floquet 对角化，'
             '实测 n_photon=4/window=3 时 ~13 s/次**，window=5 更大）。'
             '**只影响静态查询，不参与演化**'),
    'dynamic.mode': dict(type='int', enum=[0, 1, 2],
                         desc='0=单体 / 1=两体 / 2=多体（N 原子阵列，走 manybody.*）'),
    'dynamic.method': dict(type='str', enum=['mixed', 'sambe', 'hard'],
                           desc='引擎；sambe 遇耗散转 reference、遇任意波形改道 hard'),
    'dynamic.t_max': dict(type='float', unit='ns', required=True, lo=0.0,
                          exclusive_lo=True, desc='演化总时长（必须 > 0）'),
    'dynamic.npts': dict(type='int', required=True, lo=2,
                         desc='时间网格点数（必须 >= 2）'),
    'dynamic.psi_single': dict(type='identity_key',
                               desc='单体初态身份键（atomic_levels.state_key；绝不静默回落）'),
    'dynamic.psi_A': dict(type='identity_key', desc='两体 A 初态身份键'),
    'dynamic.psi_B': dict(type='identity_key', desc='两体 B 初态身份键'),
    'dynamic.psi_pair': dict(type='pair_sup', desc='两体纠缠叠加初态（["pair_sup", [...]]）'),
    'dynamic.decay_index': dict(type='int', enum=[0, 1, 2],
                                desc='衰变档：0 自动(ARC寿命)/1 禁用/2 自定义'),
    'dynamic.decays': dict(type='list', desc='自定义衰变通道（decay_index=2 时用）'),
    'dynamic.pair.R_um': dict(type='float', unit='um', desc='两体间距 R'),
    'dynamic.pair.theta_deg': dict(type='float', unit='deg', desc='两体极角 θ'),
    'dynamic.pair.phi_deg': dict(type='float', unit='deg', desc='两体方位角 φ'),
    'dynamic.waves_single': dict(type='wave_list', desc='单体波形（serialize_waves 同构；建场层构造）'),
    'dynamic.waves_A': dict(type='wave_list', desc='两体 A 波形'),
    'dynamic.waves_B': dict(type='wave_list', desc='两体 B 波形'),
    'dynamic.field_loss': dict(type='dict',
                               desc='场致加热与损失配置（opt-in；缺省 None=关，逐位不变）'),
}

# `solve.*` 元数据（键**从 runspec.SOLVE_KWARG 派生**，此处只附单位/值域/说明）
SOLVE_META = {
    'MAX_STEP': dict(type='float', unit='ns', lo=0.0,
                     desc='DOP853 起步最大步长（只管起步）'),
    'MAX_STEP_HARD': dict(type='float', unit='ns', lo=0.0,
                          desc='每步复核的硬步长上限（真压步长；只对 mixed/hard 生效）'),
    'PER_FREQ_M': dict(type='int_or_list', desc='sambe 每频率光子数 M（int 或 [int,...]）'),
    'FROZEN_H_NS': dict(type='float', unit='ns', desc='sambe 冻结段长'),
    'COST_BUDGET': dict(type='float', unit='rad', lo=0.0,
                        desc='微波候选窗相预算（对应 coupling.PHASE_BUDGET）'),
}

_ENGINE_CACHE = None


# =====================================================================
# 1. schema 收集（engine.* 扫 core 自己的 Param；solve.* 派生自 SOLVE_KWARG）
# =====================================================================
def _scan_param_defs():
    """AST 扫 core（含 manybody/ 子包）各模块**顶层** `NAME = Param(...)` → [(mod, name, lineno)]。"""
    defs = []
    scan = [(_HERE, '')]
    mb = os.path.join(_HERE, 'manybody')
    if os.path.isdir(mb):
        scan.append((mb, 'manybody.'))
    for d, prefix in scan:
        for fn in sorted(os.listdir(d)):
            if not fn.endswith('.py') or fn in ('param.py', 'protocol.py'):
                continue
            mod = prefix + fn[:-3]
            try:
                tree = ast.parse(open(os.path.join(d, fn), encoding='utf-8').read())
            except (SyntaxError, OSError):
                continue
            for node in tree.body:                       # 只扫顶层
                if (isinstance(node, ast.Assign) and len(node.targets) == 1
                        and isinstance(node.targets[0], ast.Name)
                        and isinstance(node.value, ast.Call)
                        and isinstance(node.value.func, ast.Name)
                        and node.value.func.id == 'Param'):
                    defs.append((mod, node.targets[0].id, node.lineno))
    return defs


def engine_params():
    """→ {name: record}：引擎就地常量元数据（family='engine', **readonly=True**）。

    engine.* 只读（Re0 第11刀已移除运行级覆写；调精度/成本用 solve.*）。
    表是**扫出来的**，core 是唯一定义点。
    """
    global _ENGINE_CACHE
    if _ENGINE_CACHE is not None:
        return _ENGINE_CACHE
    defs = _scan_param_defs()
    mods = {}
    out = {}
    for mod, name, lineno in defs:
        if mod not in mods:
            mods[mod] = importlib.import_module('.%s' % mod, __package__)
        p = getattr(mods[mod], name, None)
        if p is None or not hasattr(p, 'meta'):
            continue
        rec = dict(p.meta())
        rec.update(name=name, home=mod, family='engine', lineno=lineno,
                   readonly=True)
        out[name] = rec
    _ENGINE_CACHE = out
    return out


def engine_param_names():
    return sorted(engine_params())


def solve_map():
    return dict(_runspec.SOLVE_KWARG)


def solve_schema():
    """→ {'solve.MAX_STEP': record, ...}：键从 SOLVE_KWARG 派生（不手抄）。"""
    out = {}
    for key, kwarg in _runspec.SOLVE_KWARG.items():
        meta = dict(SOLVE_META.get(key, {}))
        rec = dict(meta)
        rec.update(name='solve.' + key, family='solve', engine_kwarg=kwarg,
                   tier='run', desc=meta.get('desc', '运行级旋钮 → %s' % kwarg))
        out['solve.' + key] = rec
    return out


def config_schema():
    out = {}
    for src_schema in (CONFIG_SCHEMA, _mb_schema.CONFIG_SCHEMA_MB):
        for key, meta in src_schema.items():
            rec = dict(meta)
            rec.update(name=key, family='config', tier=meta.get('tier', 'run'))
            out[key] = rec
    return out


def settable_keys():
    """可写键全集 = config（含 manybody.*）+ solve（**不含 engine.***，它只读）。"""
    return sorted(set(CONFIG_SCHEMA) | set(_mb_schema.CONFIG_SCHEMA_MB)
                  | set('solve.' + k for k in _runspec.SOLVE_KWARG))


# =====================================================================
# 2. wave_list 逐波真契约（建场层 §3）——把橡皮图章换成当场报错。
#    err 由 adapter 注入 mecha 的 InvalidValue（core 不认识 mecha 错误类型）。
# =====================================================================
def _is_ptr(x):
    return isinstance(x, dict) and x.get('$pointer') is True


def _num_gt0(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool) and x > 0


def _check_series(t, a, tag, err):
    if _is_ptr(t) or _is_ptr(a):
        return
    if t is None or a is None:
        raise err('%s file/arb 须给 t_ns 与 amp（或走指针）' % tag)
    try:
        tn = [float(x) for x in t]
        an = [float(x) for x in a]
    except (TypeError, ValueError):
        raise err('%s t_ns/amp 须是数值序列' % tag)
    if len(tn) != len(an):
        raise err('%s t_ns(%d) 与 amp(%d) 不等长' % (tag, len(tn), len(an)))
    if len(tn) < 2:
        raise err('%s t_ns/amp 须 ≥ 2 点' % tag)
    if not all(tn[k + 1] > tn[k] for k in range(len(tn) - 1)):
        raise err('%s t_ns 须严格递增' % tag)


def validate_wave_list(waves, err=None):
    """逐波校验（在 session.set 当场，I6/D5）；非法 → 抛 `err`（默认 ValueError）。"""
    from .waveforms import POLS
    if err is None:
        err = ValueError
    if not isinstance(waves, (list, tuple)):
        raise err('wave_list 须是 list（得到 %s）' % type(waves).__name__)
    kinds = ('gauss', 'square', 'file', 'arb')
    for i, w in enumerate(waves):
        tag = 'wave[%d]' % i
        if not isinstance(w, dict):
            raise err('%s 须是 dict（serialize_waves 同构）' % tag)
        pol = w.get('pol', 'π')
        if pol not in POLS:
            raise err('%s pol=%r 非法（合法 %s）' % (tag, pol, list(POLS)))
        if pol == '椭圆偏':
            ell = w.get('elliptical')
            if not (isinstance(ell, dict)
                    and {'alpha_deg', 'theta_deg', 'eps_deg'} <= set(ell)):
                raise err("%s pol='椭圆偏' 必须带 elliptical={alpha_deg,theta_deg,eps_deg}"
                          % tag)
        inten = w.get('intensity', w.get('intensity_mwcm2'))
        if inten is not None and not _num_gt0(inten):
            raise err('%s intensity 须 > 0（得到 %r）' % (tag, inten))
        if w.get('type') == 'arb':
            _check_series(w.get('t_ns'), w.get('amp'), tag, err)
            continue
        env = w.get('env')
        if not isinstance(env, (list, tuple)) or not env:
            raise err('%s 缺 env（或该设 type=arb）' % tag)
        kind = env[0]
        if kind not in kinds:
            raise err('%s env kind=%r 非法（合法 %s）' % (tag, kind, list(kinds)))
        f = w.get('freq_mhz')
        if kind != 'arb' and not _num_gt0(f):
            raise err('%s freq_mhz 须 > 0（得到 %r）；无载频请设 type=arb' % (tag, f))
        if kind == 'gauss':
            if len(env) < 4:
                raise err('%s gauss env 须 (kind,center_ns,sigma_ns,peak)' % tag)
            if not _num_gt0(env[2]):
                raise err('%s gauss sigma_ns 须 > 0' % tag)
            if not (isinstance(env[3], (int, float)) and 0 < env[3] <= 1):
                raise err('%s gauss peak 须 ∈ (0,1]' % tag)
        elif kind == 'square':
            if len(env) < 4:
                raise err('%s square env 须 (kind,start_ns,edge_ns,height[,fall_start])' % tag)
            if not (isinstance(env[2], (int, float)) and env[2] >= 0):
                raise err('%s square edge_ns 须 ≥ 0' % tag)
            if not (isinstance(env[3], (int, float)) and 0 < env[3] <= 1):
                raise err('%s square height 须 ∈ (0,1]' % tag)
        elif kind == 'file':
            if len(env) == 2 and _is_ptr(env[1]):
                continue
            if len(env) < 3:
                raise err('%s file env 须 (kind,t_ns,amp) 或 (kind,指针)' % tag)
            _check_series(env[1], env[2], tag, err)
    return waves


def validate_atom_list(atoms, err=None):
    """多体 `manybody.atoms` 逐原子真校验（委托 schema_mb；adapter 注入 InvalidValue）。

    与 validate_wave_list 同角色：把“橡皮图章”换成当场报错（g/r 四元组 /
    pos 三元组 / Ω・δ・Γ 数值）。core 不认识 mecha 错误类型，故 err 由 adapter 注入。
    """
    return _mb_schema.validate_atom_list(atoms, err=err)


# =====================================================================
# 3. 运行（复用 runspec 唯一分派）+ cache 工厂 + 波形计数
# =====================================================================
build_run = _runspec.build_run
check_run = _runspec.check_run
evolve_run = _runspec.evolve_run
result_summary = _runspec.result_summary
summarize_run = _runspec.summarize_run


def make_cache(atom='Rubidium87'):
    """ARC 缓存工厂（领域知识：mecha 不该认识 arc/ArcCache）。"""
    import arc
    from .static_atom import ArcCache
    return ArcCache(getattr(arc, atom)())


def count_waves(snap):
    """快照里的活跃驱动条数（单体或两体）——供 run 旁注/preview（重叠检测在建场层）。"""
    n = 0
    for atom in ('single', 'A', 'B'):
        n += len(snap.get('dynamic.waves_%s' % atom) or [])
    return n


# =====================================================================
# 寻址光 → address_cfg（**单一定义点**，2026-09-22 P1-6）
# =====================================================================
def wave_acfg(w, field_B=None):
    """一个 `Wave` → `static_atom` 系列要的 `address_cfg`。

    这是"寻址光参数从哪来"的**唯一答案**：**从 `dynamic.waves_*` 的波形现造**
    （静态页那套 GUI 本地态不入 registry）⇒ "AI 设的寻址光"与"人设的波形"是
    **同一个真源**，且零新增 schema 键。
    原先这段逻辑内联在 `preview` 里；现在 preview 与静态查询**共用本函数**。
    """
    acfg = {'enabled': True,
            'intensity': getattr(w, 'intensity_mwcm2', None),
            'frequency_hz': float(getattr(w, 'freq_mhz')) * 1e6}
    ell = getattr(w, 'elliptical', None)
    if getattr(w, 'pol', None) == '椭圆偏' and ell:
        acfg['elliptical'] = ell
    else:
        acfg['polarization'] = getattr(w, 'pol', None)
    return acfg


def address_cfg_from_snapshot(snap, atom=None):
    """快照 → (address_cfg | None, note)。

    取**第一条有单一载频**的波（任意波形无载频 ⇒ 定不了 AC 失谐）。
    返回 None 时 `note` 说明原因（**不静默给 0 位移**，D5）。
    """
    atoms = ((atom,) if atom else
             (('A', 'B') if int((snap or {}).get('dynamic.mode') or 0) == 1
              else ('single',)))
    n_arb = 0
    for a in atoms:
        for w in _runspec.waves_for(snap, a):
            if getattr(w, 'freq_mhz', None) is None:
                n_arb += 1
                continue
            return wave_acfg(w), None
    if n_arb:
        return None, ('寻址光只有任意波形（无单一载频）——定不了 AC 失谐；'
                      '要算 AC 位移请用「光强包络」（Wave + 载频）')
    return None, ('未配置寻址光（dynamic.waves_%s 为空）→ 无 AC 位移'
                  % '/'.join(atoms))


# =====================================================================
# 4. 查询：estimate / preview / list_levels（纯查询，不跑演化）
#    多体（dynamic.mode==2）走 _mb_* 分支（委托 manybody 的 cost_mb/runspec_mb）
# =====================================================================
def _is_many(snap):
    return int((snap or {}).get('dynamic.mode') or 0) == 2


def _mb_estimate(cache, snap):
    """多体 dry-run 成本预估：dim=2^N + ED 可行性 + cost_mb 外推（不标定、不演化）。"""
    from .manybody import cost_mb, runspec_mb
    model = runspec_mb.model_from_params(snap)
    n = len(model.atoms)
    solver = (snap or {}).get('manybody.solver') or 'auto'
    out = {'backend': 'sim', 'method': solver, 'two_body': False,
           'nbody': 'many', 'n_atoms': n, 'warnings': []}
    if n == 0:
        out['warnings'].append('manybody.atoms 为空——先摆原子才能预估')
        return out
    dim = 1 << n
    out['dim'] = dim
    ed_max = int(_mb_schema.mb_get(snap, 'manybody.ed_max_dim'))
    out['ed_feasible'] = cost_mb.ed_feasible(n, ed_max)
    t_max = (snap or {}).get('dynamic.t_max')
    npts = (snap or {}).get('dynamic.npts')
    if t_max and npts:
        out['est_sec'] = round(cost_mb.estimate_mb_cost(n, float(t_max),
                                                        int(npts)), 1)
        gb = cost_mb.estimate_mb_peak_gb(n, int(npts))
        out['est_peak_gb'] = round(gb, 3) if gb else None
    out['note'] = ('多体 ED 成本 ∝ 2^N（N=%d, dim=%d）；est_sec 粗估，'
                   '决定要不要走 async' % (n, dim))
    if not out['ed_feasible']:
        out['warnings'].append(
            'dim=2^%d=%d 超 ed_max_dim=%d——ED 不可行（等 M3 的 MPS/TEBD）'
            % (n, dim, ed_max))
    return out


def _mb_preview(cache, snap):
    """多体行为预览：标定（碰 ARC，主线程；记忆化预热）→ 耦合对/blockade/被丢项。

    标定失败**不致命**：降级为几何-only 预览 + note（preview 是纯查询，不该因
    ARC 问题整体失败）。
    """
    from .manybody import runspec_mb
    model = runspec_mb.model_from_params(snap)
    n = len(model.atoms)
    out = {'backend': 'sim', 'nbody': 'many', 'n_atoms': n, 'dim': 1 << n,
           'template': model.template, 'terms': list(model.terms),
           'waves': [], 'driven': [], 'notes': []}
    if n == 0:
        out['notes'].append('manybody.atoms 为空——先摆原子')
        out['basis_enough'] = False
        return out
    Oms = [abs(float(a.get('Omega_mhz', 0.0) or 0.0)) for a in model.atoms]
    out['max_Omega_mhz'] = round(max(Oms), 6) if Oms else 0.0
    try:
        second_order = bool(_mb_schema.mb_get(snap, 'manybody.second_order'))
        guard = float(_mb_schema.mb_get(snap, 'manybody.guard_mhz'))
        calib = runspec_mb.calibrate_memo(cache, model,
                                          second_order=second_order,
                                          guard_mhz=guard)
    except Exception as e:                          # noqa: BLE001
        out['notes'].append('标定失败（降级为几何预览，不影响规模报告）：%s' % e)
        out['basis_enough'] = True
        return out
    pairs = calib.get('pairs') or []
    out['n_pairs'] = len(pairs)
    out['n_dropped'] = len(calib.get('dropped') or [])
    out['dropped_amp'] = calib.get('dropped_amp')
    Vs = [abs(float(p['V'])) for p in pairs]
    out['max_V_mhz'] = round(max(Vs), 6) if Vs else 0.0
    out['n_guarded'] = sum(int((p.get('info') or {}).get('n_guarded', 0))
                           for p in pairs)
    if out['max_V_mhz'] and out['max_Omega_mhz']:
        out['blockade_note'] = ('max|V|/max|Ω|=%.3g（≫1 为 blockade 区）'
                                % (out['max_V_mhz'] / out['max_Omega_mhz']))
    out['basis_enough'] = True
    out['notes'].append('多体预览：%d 原子、%d 耦合对、dim=2^%d=%d（标定碰 ARC，'
                        '已记忆化预热）' % (n, len(pairs), n, 1 << n))
    return out


def _mb_list_levels(cache, snap):
    """多体：逐原子 (g,r) 身份 + 产物态寻址说明（供 psi0 选择；不碰 cache）。"""
    from .manybody import runspec_mb
    model = runspec_mb.model_from_params(snap)
    n = len(model.atoms)
    atoms = [{'id': i, 'g': list(a['g']), 'r': list(a['r']),
              'pos': list(a['pos']), 'Omega_mhz': a.get('Omega_mhz', 0.0),
              'delta_mhz': a.get('delta_mhz', 0.0)}
             for i, a in enumerate(model.atoms)]
    return {'n_atoms': n, 'dim': 1 << n, 'nbody': 'many', 'atoms': atoms,
            'psi0_hint': ('设 manybody.psi0：None=全基态|g…g⟩ / "grg"模式串 / '
                          'int产物态索引 / "ground"(Lanczos基态) / '
                          '["sup",[["grg",amp],…]]叠加'),
            'note': ('多体基是 2^%d=%d 个产物态 |g/r⟩^N（原子 0 在左/MSB）；'
                     '布居/关联/纠缠熵见结果 pops_atom/corr/S' % (n, 1 << n))}


def estimate(cache, snap):
    """dry-run 成本预估（复用 cost.py + build_states）。**不跑演化、不算 V_dd**。"""
    if _is_many(snap):
        return _mb_estimate(cache, snap)
    from . import cost
    groups = _runspec.groups_from_params(snap)
    method = snap.get('dynamic.method') or 'mixed'
    two_body = int(snap.get('dynamic.mode') or 0) == 1
    t_max = snap.get('dynamic.t_max')
    npts = snap.get('dynamic.npts')
    out = {'backend': 'sim', 'method': method, 'two_body': two_body,
           'n_groups': len(groups), 't_max_ns': t_max, 'npts': npts,
           'warnings': []}
    if not groups:
        out['warnings'].append('groups.list 为空——先加能级组才能预估')
        return out
    try:
        states = _runspec.build_states(cache, groups, snap.get('field.B_gauss'),
                                       snap.get('field.E_vpcm'))
        out['n_states'] = len(states)
    except Exception as e:                          # noqa: BLE001
        out['warnings'].append('基组枚举失败（无法预估规模）：%s' % e)
        return out
    if two_body:
        dim = len(states) ** 2
        out['pair_dim'] = dim
        if t_max and npts:
            out['est_sec'] = round(cost.estimate_pair_cost(
                len(states), float(t_max), int(npts)), 1)
            gb = cost.estimate_pair_peak_gb(dim)
            out['est_peak_gb'] = round(gb, 3) if gb else None
            out['note'] = ('两体成本 ∝ N⁴·t_max（cost.py 实测标定外推）；'
                           'est_sec 只是粗估，决定要不要走 async')
        else:
            out['warnings'].append('缺 dynamic.t_max/npts——两体成本无法外推')
    else:
        out['note'] = ('单体演化通常秒级；N=%d、t_max=%s、npts=%s。'
                       % (len(states), t_max, npts))
    return out


def preview(cache, snap):
    """纯查询的**行为预览**（基线）：驱动哪些跃迁/各多强、基组够不够。

    重叠告警（detect_wave_overlap）由 adapter 叠加（那是建场层的领域逻辑）。
    """
    if _is_many(snap):
        return _mb_preview(cache, snap)
    from . import static_atom as SA
    groups = _runspec.groups_from_params(snap)
    B = float(snap.get('field.B_gauss') or 0.0)
    out = {'backend': 'sim', 'n_groups': len(groups),
           'waves': [], 'driven': [], 'missing_from_basis': [], 'notes': []}
    if not groups:
        out['notes'].append('groups.list 为空——先加能级组')
        return out
    states = _runspec.build_states(cache, groups, B, snap.get('field.E_vpcm'))
    labels = [s['label'] for s in states]
    basis_nlj = {(s['n'], s['l'], s['j']) for s in states}
    out['n_states'] = len(states)
    out['basis_head'] = labels[:8]
    _SPDF = 'SPDFGHIKLMNO'
    lab_of, g_lab = {}, []
    for i, g in enumerate(groups):
        gg = dict(g)
        lab = (gg.get('label') or gg.get('group_label')
               or '%d%s' % (gg['n'], _SPDF[int(gg['l'])]
                            if int(gg['l']) < len(_SPDF) else '?'))
        if lab in lab_of:
            lab = '%s#%d' % (lab, i)
        gg['label'] = lab
        lab_of[lab] = (gg['n'], gg['l'], gg['j'])
        g_lab.append(gg)
    atoms = ('A', 'B') if int(snap.get('dynamic.mode') or 0) == 1 else ('single',)
    for atom in atoms:
        for w in _runspec.waves_for(snap, atom):
            wd = {'atom': atom, 'name': getattr(w, 'name', '?'),
                  'pol': getattr(w, 'pol', '?'),
                  'intensity_mwcm2': getattr(w, 'intensity_mwcm2', None)}
            freq = getattr(w, 'freq_mhz', None)
            if freq is None:
                wd['note'] = '任意波形（无单一载频）——不能定失谐，需实跑'
                out['waves'].append(wd)
                continue
            wd['freq_mhz'] = freq
            out['waves'].append(wd)
            acfg = wave_acfg(w, B)
            try:
                for (ls, qs, lt, qt, det, om) in SA.collect_transitions(
                        cache, g_lab, acfg, field_B=B):
                    tgt_nlj = lab_of.get(lt)
                    in_basis = (tgt_nlj in basis_nlj) if tgt_nlj else False
                    out['driven'].append(
                        {'src': ls, 'tgt': lt, 'qstr_tgt': qt,
                         'det_mhz': round(float(det) / 1e6, 4),
                         'rabi_rad_s': float(om), 'tgt_in_basis': in_basis})
                    if not in_basis:
                        out['missing_from_basis'].append(lt)
            except Exception as e:          # noqa: BLE001
                out['notes'].append('波 %s 的跃迁预览失败（不影响基组报告）：%s'
                                    % (wd['name'], e))
    out['missing_from_basis'] = sorted(set(out['missing_from_basis']))
    out['basis_enough'] = not out['missing_from_basis']
    if not out['driven']:
        out['notes'].append('没有波驱动到组间跃迁（查波频率/偏振，或加能级组）')
    if out['missing_from_basis']:
        out['notes'].append(
            '基组缺 %d 个被驱动到的组：%s——加对应能级组，否则这些跃迁的布居'
            '无处可去（结果会失真）' % (len(out['missing_from_basis']),
                                        out['missing_from_basis'][:4]))
    out['n_drives'] = count_waves(snap)
    out['n_waves'] = out['n_drives']
    return out


def list_levels(cache, snap):
    """列出当前基组的态（label + 身份键 + 能量）——语义寻址，不猜索引。

    ## 静态列（P1-6，2026-09-22）——**显式 opt-in，默认关**

    由快照键 `static.columns` 控制（`off` 默认 / `basic` / `ac`）：
      · `basic`：额外回 `E0_mhz`(= `E_mhz`，同一行) / `dc_shift_mhz` / `qstr` /
        `group_label` / `color` / `amps`。**便宜**——这些量本就在
        `base_sublevels` 的那一遍里算出来了（`with_dc` 顺手记下 DC 位移）。
      · `ac`：再额外回 `ac_shift_mhz` / `ac_affected`（AC-Stark）。**贵**——
        要一次 Floquet 对角化：`n_photon=4 / window=3` 时 **~13 s/次**
        （`src/ui/fields.py:82` 实测原文"window=3 全伙伴（N~400，n=4 时 ~13s/次）"），
        `window=5` 更大。故**绝不默认开**，且回 `cost_note` 说明代价（D5）。

    为什么用**快照键**而不是函数形参：`list_levels` 这条链
    （`server.list_levels()` → `ops.list_levels()` → `protocol` → `adapter`）
    **四处都是无参的**（实测 `server.py:202` / `ops.py:199` / `mecha/protocol.py:81`
    / `adapter/__init__.py:261`）⇒ 没有 `**opts` 可透传。走快照键 ⇒
    **零 `mecha-sdk` 改动**，且该键会出现在 `list_params` 里、带 desc 说明代价
    ——对 AI 反而**可发现**。（若将来库侧加了形参，本键可退回默认关。）
    """
    if _is_many(snap):
        return _mb_list_levels(cache, snap)
    from .atomic_levels import state_key
    groups = _runspec.groups_from_params(snap)
    cols = str((snap or {}).get('static.columns') or 'off')
    if cols not in ('off', 'basic', 'ac'):
        raise ValueError('static.columns 只认 off/basic/ac，得到 %r' % cols)
    if not groups:
        return {'n_states': 0, 'states': [],
                'note': '基组为空——先 set_param("groups.list", [{n,l,j}, ...])'}
    states = _runspec.build_states(cache, groups, snap.get('field.B_gauss'),
                                   snap.get('field.E_vpcm'),
                                   with_static=(cols != 'off'))
    out = [{'label': s['label'], 'key': list(state_key(s)),
            'n': s['n'], 'l': s['l'], 'j': s['j'], 'kind': s['kind'],
            'm': s.get('m'), 'E_mhz': round(float(s['E_mhz']), 6)}
           for s in states]
    if cols != 'off':
        for d, s in zip(out, states):
            d['E0_mhz'] = round(float(s['E_mhz']), 6)
            d['qstr'] = s.get('qstr')
            d['group_label'] = s.get('group_label')
            d['color'] = s.get('color')
            d['dc_shift_mhz'] = round(float(s.get('dc_shift_mhz', 0.0)), 6)
            d['amps'] = ({str(k): [float(np.real(v)), float(np.imag(v))]
                          for k, v in (s.get('amps') or {}).items()}
                         if s.get('amps') else None)
    res = {'n_states': len(out), 'states': out,
           'hint': 'set_config("dynamic.psi_single", <key>) 设初态；'
                   'set_drive 的 src/tgt 用 query_levels 的身份键（label 不认）'}
    if cols == 'off':
        res['static_hint'] = (
            '要 DC-Stark 位移 / qstr / amps：set_config("static.columns","basic")；'
            '再加 AC-Stark（**~13 s/次**，n_photon=4/window=3）："ac"')
        return res
    res['static_columns'] = cols
    if cols == 'basic':
        res['cost_note'] = ('basic 列来自 base_sublevels 同一遍计算，**无额外代价**；'
                            'AC-Stark 需另设 static.columns="ac"（~13 s/次）')
        return res
    # ---- ac 档：一次 Floquet 对角化（贵）----
    B = float(snap.get('field.B_gauss') or 0.0)
    acfg, note = address_cfg_from_snapshot(snap)
    if acfg is None:
        for d in out:
            d['ac_shift_mhz'] = None
            d['ac_affected'] = None
        res['ac_note'] = note
        res['cost_note'] = 'AC 列未计算（%s）' % note
        return res
    n_ph = (snap or {}).get('field.n_photon')
    win = (snap or {}).get('field.window')
    rows = SA.build_level_rows_direct(
        cache, B, groups, acfg, groups, float(snap.get('field.E_vpcm') or 0.0),
        n_ph, win)
    # 按 (n,l,j,qstr) 分桶、桶内按 occ 顺序消费 —— 与 `_occ_keys_direct` 同语义，
    # 故重复组（同 n,l,j 加两次）也能逐一对上，不靠位置猜。
    buckets = {}
    for r in rows:
        k = (r['n'], r['l'], r['j'], r['qstr'])
        buckets.setdefault(k, []).append(r)
    occ = {}
    for d, s in zip(out, states):
        k = (s['n'], s['l'], s['j'], s.get('qstr'))
        i = occ.get(k, 0)
        occ[k] = i + 1
        bucket = buckets.get(k) or []
        r = bucket[i] if i < len(bucket) else None
        d['ac_shift_mhz'] = (None if r is None
                             else round(float(r.get('ac', 0.0)), 6))
        d['ac_affected'] = (None if r is None
                            else bool(r.get('affected', False)))
    res['ac_precision'] = {'n_photon': n_ph, 'window': win}
    res['cost_note'] = ('ac 列需一次 Floquet 对角化（实测 n_photon=4/window=3 时 '
                        '~13 s；window=5 更大）——故**只在显式请求时**算。')
    return res


# =====================================================================
# 4b. 静态查询（P1-6，2026-09-22）：谱线 / AC 光强扫描 / 两体静态
#
# 三条**纯查询**（不跑演化、不写状态、幂等），与 list_levels/preview/estimate 同类
# ⇒ 走 `EngineProtocol`（引擎自报、库只搬运），**不在库里注册领域工具名**。
#
# ⚠ **关于"大数组走指针"**：`$pointer` 机制住在库里（`mecha/bus.py` 的
# `payload_pointer` / `gate_value`），**core 拿不到 mecha root** ⇒ core **不产生
# 指针**，只负责**把大结果做成 numpy 数组**（而不是大 list）——因为库的体积闸
# 对**数组**是无损转指针（带 sha1 可校验），对**大 list/dict** 只会截断成
# `$truncated` 头 512 字节（**会丢数据**）。这是本设计对"走指针"的准确说法。
# =====================================================================
def _static_rows(cache, snap, acfg):
    """快照 → (rows, coeffs, note)：静态能级行（AC 唯一后端 direct）。"""
    groups = _runspec.groups_from_params(snap)
    if not groups:
        raise ValueError('基组为空：先 set_param("groups.list", [{n,l,j}, ...])'
                         '——静态量无从算起')
    B = float((snap or {}).get('field.B_gauss') or 0.0)
    Ev = float((snap or {}).get('field.E_vpcm') or 0.0)
    n_ph = (snap or {}).get('field.n_photon')
    win = (snap or {}).get('field.window')
    cfg = acfg if acfg is not None else {'enabled': False}
    rows = SA.build_level_rows_direct(cache, B, groups, cfg, groups, Ev,
                                      n_ph, win)
    coeffs = SA.address_pol_coeffs(cfg) if cfg.get('enabled') else None
    note = None
    if not cfg.get('enabled'):
        note = '未配置寻址光 → 行内无 AC 位移（ac_on 对本结果无影响）'
    return rows, coeffs, note


def static_spectrum(cache, snap, top_n=50, full=False, ac_on=True):
    """子能级间全部偶极允许跃迁的谱线（频率 Hz + 爱因斯坦 A + 两端身份 + 偏振）。

    实现 = `static_atom.spectrum_lines`（与 GUI 静态页**同一函数**）。
    `top_n`：按 A 降序取前 N 条（默认 50）。`full=True` 时**额外**回
    `arrays`（numpy 列数组）——**大结果做成数组而非大 list**，这样库的体积闸
    会无损转 `$pointer`（见上方 ⚠）。
    """
    acfg, ac_note = address_cfg_from_snapshot(snap)
    rows, coeffs, note = _static_rows(cache, snap, acfg)
    lines = SA.spectrum_lines(cache, rows, bool(ac_on), coeffs)
    lines = sorted(lines, key=lambda t: -float(t[1]))
    top = [{'freq_hz': float(nu), 'A_s1': float(A),
            'label_a': la, 'label_b': lb, 'qstr_a': qa, 'qstr_b': qb,
            'pol': pol}
           for (nu, A, la, lb, qa, qb, _color, pol) in lines[:max(int(top_n), 0)]]
    out = {'n_lines': len(lines), 'ac_on': bool(ac_on),
           'n_rows': len(rows), 'top_n': int(top_n),
           'truncated': len(lines) > len(top), 'lines': top}
    notes = [n for n in (ac_note, note) if n]
    if notes:
        out['notes'] = notes
    if full and lines:
        out['arrays'] = {
            'freq_hz': np.asarray([float(t[0]) for t in lines]),
            'A_s1': np.asarray([float(t[1]) for t in lines]),
            'label_a': np.asarray([t[2] for t in lines]),
            'label_b': np.asarray([t[3] for t in lines]),
            'qstr_a': np.asarray([t[4] for t in lines]),
            'qstr_b': np.asarray([t[5] for t in lines]),
            'pol': np.asarray([t[7] for t in lines]),
        }
        out['arrays_note'] = ('arrays 是**列数组**（不是 list）⇒ 库的体积闸可无损转 '
                              '$pointer；大 list 只会被截断')
    return out


def static_ac_sweep(cache, snap, intensities_mwcm2):
    """AC-Stark 位移 vs 光强（一次 Floquet 结构、批量对角化）。

    实现 = `static_atom.direct_ac_stark_sweep`（与 GUI 静态页扫描**同一函数**）。
    ⚠ **形状转换**：原函数回 `dict[tuple_key → ndarray]`，**tuple 键无法 JSON 化**
    ⇒ 这里转成 `{keys: [{n,l,j,qstr,occ}], shifts_mhz: 2D 数组}`（行=态、列=光强）。
    """
    inten = [float(x) for x in (intensities_mwcm2 or [])]
    if not inten:
        raise ValueError('static_ac_sweep 需要 intensities_mwcm2（非空数值序列）')
    acfg, ac_note = address_cfg_from_snapshot(snap)
    if acfg is None:
        return {'ok': False, 'n_states': 0, 'keys': [], 'intensities_mwcm2': inten,
                'note': ac_note}
    groups = _runspec.groups_from_params(snap)
    if not groups:
        raise ValueError('基组为空：先 set_param("groups.list", [{n,l,j}, ...])')
    B = float((snap or {}).get('field.B_gauss') or 0.0)
    n_ph = (snap or {}).get('field.n_photon')
    win = (snap or {}).get('field.window')
    keys, shifts = SA.direct_ac_stark_sweep(cache, groups, acfg, B, inten,
                                            n_ph, win)
    klist = [{'n': int(k[0]), 'l': int(k[1]), 'j': float(k[2]),
              'qstr': k[3], 'occ': int(k[4])} for k in keys]
    mat = (np.asarray([shifts[k] for k in keys], float)
           if keys else np.zeros((0, len(inten))))
    return {'ok': True, 'n_states': len(keys), 'n_intensities': len(inten),
            'intensities_mwcm2': inten, 'keys': klist,
            'shifts_mhz': mat,
            'ac_precision': {'n_photon': n_ph, 'window': win},
            'shape_note': 'shifts_mhz 形状 = (n_states, n_intensities)，'
                          '行序与 keys 一一对应'}


def static_pair(cache, snap, what='exchange', with_evecs=False, n_window=None,
                R_grid=None, theta_grid=None, trans1=None, trans2=None):
    """两体静态：本征能级 / vdW 位移 / 交换耦合 / 耦合强度雷达图。

    `what` ∈ `{'exchange','evals','vdw','map','all'}`，**默认 `'exchange'`（唯一便宜档）**。
    `evals`/`vdw`/`all` 都要稠密 `eigh`（矩阵 (N²)² ⇒ 内存 **∝ N⁴**）⇒ 受维度闸保护。
    `evals` 只回**本征值**；`with_evecs=True` 才回本征矢（同尺寸 ⇒ N=126 时 ~4 GB）
    ⇒ 做成数组让库转指针。**稠密 H 不暴露**。

    ★ `n_window` **默认 1**（不是 `static_pair` 模块的 3）：实测 60S+60P 下
      n_window=1 → N=54 / dim=2916 / 136 MB；**n_window=3 → N=126 / dim=15876 /
      4.0 GB**（不可行）。而 `exchange` 在 nw=1 与 nw=3 给出**同一个值**
      （-3.705294 MHz，实测——一阶矩阵元与伙伴态无关）⇒ AI 面默认取便宜档，
      要 vdW 精度自己显式加 `n_window`。

    ⚠ 顺带消掉两个**静默点**（设计 §1.4）：
      · `coupling_strength_map` 对"非里德堡交换对"**静默返全零 Z** ⇒ 这里回
        `exchange_valid=False` + `note`（物理上确实无交换，但**必须说**）；
      · 对"基组缺源/目标态"**静默返 None** ⇒ 这里**明确报错**并点名。
    """
    from . import static_pair as _sp
    groups = _runspec.groups_from_params(snap)
    if not groups:
        raise ValueError('基组为空：先 set_param("groups.list", [{n,l,j}, ...])')
    B = float((snap or {}).get('field.B_gauss') or 0.0)
    # 两体几何**只认 registry 的三个扁平键**（与 `runspec.py:555-557` 同一真源）——
    # 旧版读 `snap.get('dynamic.pair')` 字典键，而该键**不在 registry 里**
    # （`set_param("dynamic.pair", …)` 会被 UnknownKey 拒）⇒ 经 MCP/AI 调本查询
    # 时 R 恒为 0、必报"需要 R_um > 0"；只有绕过 registry 直调 core 的合成快照
    # 能喂到那个字典键（判据因此假绿）。2026-09-22 修。
    R = float((snap or {}).get('dynamic.pair.R_um') or 0.0)
    TH = float((snap or {}).get('dynamic.pair.theta_deg') or 0.0)
    PH = float((snap or {}).get('dynamic.pair.phi_deg') or 0.0)
    nw = int(1 if n_window is None else n_window)
    out = {'R_um': R, 'theta_deg': TH, 'phi_deg': PH, 'B_gauss': B,
           'n_window': nw,
           'n_window_note': ('默认 1（便宜档）：exchange 与 n_window=3 同值，'
                             '但 pair 维小得多；evals/vdw 要精度请显式加 n_window')}
    if R <= 0.0:
        raise ValueError('两体静态需要 dynamic.pair.R_um > 0（当前 %r）' % R)
    # ---- 维度闸：eigh 路径（evals/vdw/all）先算 N² 再决定能不能做 ----
    need_eig = what in ('all', 'evals', 'vdw')
    if need_eig:
        N = len(_sp.single_atom_basis(cache, groups, B, nw))
        dim = N * N
        cap = int(_sp.PAIR_EIGEN_MAX_DIM)
        if dim > cap:
            raise ValueError(
                '两体静态本征分解**不可行**：n_window=%d 时单原子基 N=%d ⇒ '
                'pair 维 N²=%d > 闸值 %d（稠密 eigh 矩阵 ≈ %.2f GB，还要同尺寸 '
                'evecs）。请调小 n_window（实测 n_window=1 时 N 小得多）或减少'
                '能级组；exchange 是**一阶矩阵元**，不需要本征分解，可直接用 '
                'what="exchange"' % (nw, N, dim, cap, dim * dim * 16 / 1e9))
        out['pair_dim'] = dim
        out['eigh_matrix_mb'] = round(dim * dim * 16 / 1e6, 1)
    if what in ('all', 'evals'):
        res = _sp.pair_eigen(cache, groups, R, TH, PH, B_gauss=B, n_window=nw)
        out['evals'] = {'evals_mhz': np.asarray(res['evals'], float),
                        'n_pair_states': int(len(res['evals'])),
                        'E_pair_mhz': np.asarray(res['E_pair'], float)}
        if with_evecs:
            out['evals']['evecs'] = np.asarray(res['evecs'])
            out['evals']['evecs_note'] = (
                'evecs 形状 (N²,N²)——体积 ∝ N⁴；做成数组以便库无损转指针')
    if what in ('all', 'vdw'):
        sh, e_ref = _sp.vdw_shift(cache, groups, R, TH, PH, B_gauss=B,
                                  n_window=nw)
        out['vdw'] = {'shift_mhz': float(sh), 'E_ref_pair_mhz': float(e_ref)}
    if what in ('all', 'exchange'):
        out['exchange'] = {'J_mhz': float(_sp.exchange_coupling(
            cache, groups, R, TH, PH, B_gauss=B, n_window=nw))}
    if what in ('all', 'map'):
        if R_grid is None or theta_grid is None:
            raise ValueError('what="map" 需要 R_grid 与 theta_grid')
        # 顺序有意：**先查"态在不在"（配置错 → 报错），再查"是不是里德堡"（物理事实 → note）**。
        # 反过来的话，一个既有非里德堡态、又缺另一个态的场景会只报"非里德堡"，
        # 把真正的配置错（缺态）掩盖掉。
        m = _sp.coupling_strength_map(
            cache, groups, R_grid, theta_grid, phi_deg=PH,
            trans1=trans1, trans2=trans2, B_gauss=B, n_window=nw)
        if m is None:
            raise ValueError(
                '耦合强度雷达图：基组缺少指定的源/目标态'
                '（trans1=%r, trans2=%r）——请先加对应能级组，或改用 '
                'list_levels 的 label/qstr 指定' % (trans1, trans2))
        Rg, Tg, Z = m
        ok, why = _exchange_domain_ok(groups, trans1, trans2)
        out['map'] = {'exchange_valid': bool(ok),
                      'R_grid': np.asarray(Rg, float),
                      'theta_grid': np.asarray(Tg, float),
                      'Z_mhz': np.asarray(Z, float),
                      'shape_note': 'Z_mhz 形状 = (len(R_grid), len(theta_grid))'}
        if not ok:
            # 物理上确实无交换 ⇒ Z 恒 0 是**正确**的，但必须明说（否则读成"算出来了"）
            out['map']['note'] = why
    return out


def _exchange_domain_ok(groups, trans1, trans2):
    """交换对是否在里德堡态之间（`static_pair` 的静默全零判据）。

    返回 (ok, why)。**只做判据**——真正的"无交换"物理结论仍由
    `coupling_strength_map` 给（它对非里德堡对返全零，`static_pair.py:411-414`）。
    """
    keys = []
    for t in (trans1, trans2):
        if not t:
            continue
        for k in t:
            if isinstance(k, (list, tuple)) and len(k) == 4:
                keys.append(k)
    if not keys:
        return True, None
    low = [k for k in keys if float(k[0]) < float(SA.RYDBERG_N_MIN)]
    if low:
        return False, ('交换对涉及非里德堡态（n=%s < RYDBERG_N_MIN=%d）——'
                       '交换耦合只在里德堡态之间才有物理意义，故 Z 恒 0；'
                       '这不是"算不出来"，是**物理上不存在**'
                       % ([int(k[0]) for k in low], int(SA.RYDBERG_N_MIN)))
    return True, None


# =====================================================================
# 5. I6 交叉校验：AST 扫 runspec 实读的键（core 侧自检，防 schema 漂）
# =====================================================================
def runspec_read_keys():
    """AST 扫 `runspec.py` + `manybody/runspec_mb.py` 对快照 `p` 的读取
    → (literal_keys, dynamic_patterns)。I6：这些键都须在 config_schema（防漂）。

    识别 `p.get('lit')` / `_required(p,'lit')` / `mb_get(p,'lit')`（多体默认回落口）。
    """
    literals, dynamic = set(), set()
    _nodes = []
    for _path in (os.path.join(_HERE, 'runspec.py'),
                  os.path.join(_HERE, 'manybody', 'runspec_mb.py')):
        if os.path.exists(_path):
            _nodes.extend(
                ast.walk(ast.parse(open(_path, encoding='utf-8').read())))

    def _str_of(node):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            return node.value
        return None

    def _recv_is_p(node):
        return any(isinstance(x, ast.Name) and x.id == 'p' for x in ast.walk(node))

    for nd in _nodes:
        if not isinstance(nd, ast.Call):
            continue
        f = nd.func
        if (isinstance(f, ast.Attribute) and f.attr == 'get' and nd.args
                and _recv_is_p(f.value)):
            s = _str_of(nd.args[0])
            if s is not None:
                literals.add(s)
            elif isinstance(nd.args[0], ast.BinOp):
                dynamic.add(ast.unparse(nd.args[0]))
        if (isinstance(f, ast.Name) and f.id in ('_required', 'mb_get')
                and len(nd.args) >= 2
                and isinstance(nd.args[0], ast.Name) and nd.args[0].id == 'p'):
            s = _str_of(nd.args[1])
            if s is not None:
                literals.add(s)
    return sorted(literals), sorted(dynamic)
