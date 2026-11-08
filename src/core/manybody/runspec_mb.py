# -*- coding: utf-8 -*-
"""多体产品配置：完整物理模型 → 标定快照 → 可执行 RunSpec。

产品固定包含 I1/I2/I3 全相互作用；按项选择的研究模型使用 ManyBodyModel。
标定在 ARC cache 所属线程完成，执行阶段只消费数值快照。后端名单、能力和
参数映射来自 backends；默认值来自 schema_mb；基空间决定后再选择后端。

单位：位置 µm，能量/Ω/δ/Γ MHz，时间 ns，B Gauss。
"""
import hashlib
import json
import os
from pathlib import Path

import numpy as np

from .model import ManyBodyModel, TERMS
from . import assemble as _asm
from .backends import (TIME_BACKENDS, ConfigConflict, normalize_time_backend,
                       validate_time_capabilities)
from .state import validate_psi_spec
from .hamiltonian_plan import build_plan, plan_summary
from .schema_mb import (mb_get, validate_drive_segments, validate_loops,
                        validate_link_pairs, validate_pauli_strings)

__all__ = ['model_from_arc_spec', 'build_calib_from_arc', 'build_calib_from_file',
           'resolve_mb_psi',
           'resolve_solver', 'calibrate_memo', 'calib_summary', 'calib_fingerprint', 
           'build_calib_from_arc', 'build_calib_from_file', 'build_run_manybody',
           'validate_calib', 'calib_from_params', 'calib_dir', 'load_calib_file',
           'resolve_model',
           'drive_from_calib', 'atom_count', 'prepare_initial_state',
           'CALIB_TERMS']

_CALIB_MEMO = {}

#: 引擎认识的三类相互作用项（I1=密度-密度 V，I2=交换 J，I3=双激发 A）。
CALIB_TERMS = tuple(TERMS)


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


def model_from_arc_spec(spec):
    """**构造器内部**：``{"atoms":[…g/r…], "arc":{…}}`` → ManyBodyModel。

    只被 ``el_build_calib`` 调用，不读配置快照；RunSpec 永远见不到这个形态。
    """
    spec = spec or {}
    raw = spec.get('atoms')
    if not isinstance(raw, (list, tuple)) or not raw:
        raise ValueError('el_build_calib(atoms=…)：原子表为空')
    arc = spec.get('arc') or {}
    if not isinstance(arc, dict):
        raise ValueError('el_build_calib：arc 须是映射')
    model = ManyBodyModel(
        template='full', terms=tuple(TERMS),
        B_gauss=float(arc.get('B_gauss', 0.0) or 0.0),
        n_window=int(arc.get('n_window', 3)),
        r_cut_um=(None if arc.get('r_cut_um') is None
                  else float(arc['r_cut_um'])),
        periods_um=arc.get('periods_um'))
    for a in raw:
        model.add_atom(a['g'], a['r'], a['pos'],
                       Omega_mhz=a.get('Omega_mhz', 0.0),
                       Omega_phase_rad=a.get('Omega_phase_rad', 0.0),
                       delta_mhz=a.get('delta_mhz', 0.0),
                       gamma_mhz=a.get('gamma_mhz', 0.0))
    return model


def build_calib_from_arc(cache, spec):
    """**构造器**：ARC 参数 → calib（走 ``calibrate_memo``，含 ARC/cache）。"""
    arc = (spec or {}).get('arc') or {}
    model = model_from_arc_spec(spec)
    return calibrate_memo(cache, model,
                          second_order=bool(arc.get('second_order', True)),
                          guard_mhz=float(arc.get('guard_mhz', 1.0)))


def build_calib_from_file(name):
    """**构造器**：读 ``<calib_dir>/<name>.json`` → 校验后的 calib。"""
    return validate_calib(load_calib_file(name))


def _num(value):
    """耦合系数：保留复数（Peierls 相位），零值统一成 0.0（抹掉 −0.0）。"""
    if value is None:
        return 0.0
    c = complex(value)
    return (c.real + 0.0) if c.imag == 0.0 else complex(c.real + 0.0, c.imag + 0.0)


def _real(value):
    """实耦合系数（V 只取实部；虚部非零是调用方错误）。"""
    if value is None:
        return 0.0
    c = complex(value)
    if c.imag != 0.0:
        raise ValueError('manybody.calib.pairs[].V 必须是实数（收到 %r）' % (value,))
    return c.real + 0.0


def validate_calib(calib):
    """校验并规范一份**显式 calib**（引擎的模型货币），返回新 dict，不改入参。

    calib 是多体引擎唯一认的模型形态：``assemble.build_H_sparse``、
    ``eq_route``、``dmrg.terms_from_calib`` 全都只吃它。ARC 标定只是**生产**
    calib 的一条途径；给定成对耦合的理论模型（任意点云、任意 J/V/A）走这里，
    不需要 cache、也不需要原子能级身份。

    结构::

        {"atoms": [{"id": 0, "pos": [x,y,z],          # µm
                    # 可选、运行级驱动（不进标定缓存键）：
                    "Omega_mhz": 0.0, "Omega_phase_rad": 0.0,
                    "delta_mhz": 0.0, "gamma_mhz": 0.0}, ...],
         "pairs": [{"i": 0, "j": 1, "V": 0.0, "J": 0.0, "A": 0.0}, ...],  # MHz
         }                                             # terms 由 pairs 推导，不接受

    位置单位与全库一致（µm）；``Id`` 只是标签，可缺省（默认下标）。
    **不接受** ``file``/``arc``（那是构造器的参数，不是 RunSpec 的写法），
    也**不接受** ``terms``（活跃通道由数值推导：V≠0→I1，J≠0→I2，A≠0→I3；
    不用某通道就把它的系数写成 0.0）。
    """
    if not isinstance(calib, dict):
        raise ValueError('manybody.calib 须是映射，收到 %s'
                         % type(calib).__name__)
    if 'file' in calib or 'arc' in calib:
        raise ValueError(
            'manybody.calib 只接受行内模型 {atoms, pairs}。'
            '文件请用 el_build_calib({"file": …})，ARC 请用 '
            'el_build_calib({"atoms": […g/r…], "arc": {…}})：它们会把转换结果写进本键。')
    if 'terms' in calib:
        pass        # 允许回读（validate 的输出可再次喂回来）；一致性在下面按数值核对
    raw_atoms = calib.get('atoms')
    if not isinstance(raw_atoms, (list, tuple)) or not raw_atoms:
        raise ValueError('manybody.calib.atoms 为空——calib 至少要有一个原子')
    atoms = []
    for i, a in enumerate(raw_atoms):
        if not isinstance(a, dict):
            raise ValueError('manybody.calib.atoms[%d] 须是映射' % i)
        pos = a.get('pos')
        if not isinstance(pos, (list, tuple)) or len(pos) != 3:
            raise ValueError('manybody.calib.atoms[%d].pos 须是 (x,y,z) µm' % i)
        # `+ 0.0` 把 −0.0 规范成 0.0：DSH 的参数守卫拒绝 `Object.is(x, -0)`
        # （dsh-api-session-controller client.js:1311）⇒ 生成器算出的 `-0.0`
        # 会让整次工具调用被判"不可无损序列化"。这里在入库前就抹平。
        item = {'id': int(a.get('id', i)), 'pos': [float(v) + 0.0 for v in pos]}
        for key in ('Omega_mhz', 'Omega_phase_rad', 'delta_mhz', 'gamma_mhz'):
            item[key] = float(a.get(key) or 0.0) + 0.0
        atoms.append(item)
    n = len(atoms)

    pairs = []
    for k, pr in enumerate(calib.get('pairs') or []):
        if not isinstance(pr, dict):
            raise ValueError('manybody.calib.pairs[%d] 须是映射' % k)
        try:
            i, j = int(pr['i']), int(pr['j'])
        except (KeyError, TypeError, ValueError):
            raise ValueError('manybody.calib.pairs[%d] 缺 i/j（原子下标）' % k) from None
        if i == j or not (0 <= i < n) or not (0 <= j < n):
            raise ValueError('manybody.calib.pairs[%d] 的 (i,j)=(%d,%d) 不在 0..%d 内或自耦合'
                             % (k, i, j, n - 1))
        pairs.append({'i': i, 'j': j,
                      'V': _real(pr.get('V')),
                      'J': _num(pr.get('J')),
                      'A': _num(pr.get('A'))})

    # 活跃通道由**数值**推导：某通道全零就是关的。构造器负责把不用的通道写成 0.0，
    # 所以这里用精确非零判定（不做阈值猜测）。
    terms = [name for name, key in (('I1', 'V'), ('I2', 'J'), ('I3', 'A'))
             if any(abs(pr[key]) > 0.0 for pr in pairs)]
    given = calib.get('terms')
    if given is not None and sorted(str(t) for t in given) != sorted(terms):
        raise ValueError(
            'manybody.calib 的 terms=%s 与数值推导 %s 不一致：活跃通道由 pairs 的数值决定'
            '（V≠0→I1，J≠0→I2，A≠0→I3）。不要手写 terms；不用某通道就把它的系数写成 0.0。'
            % (list(given), terms))
    # 其余键（single_basis/template/J0/boundary/…）原样带过去：calib 是数据容器，
    # 与 ARC 标定产物保持同形，不因校验丢掉调用方的元数据。
    out = dict(calib)
    out.update({'atoms': atoms, 'pairs': pairs, 'terms': terms,
                'source': str(calib.get('source') or 'explicit calib'),
                'dropped': list(calib.get('dropped') or []),
                'dropped_amp': dict(calib.get('dropped_amp') or {})})
    return out


def _fp_num(value):
    c = complex(value or 0.0)
    return round(c.real, 12) if c.imag == 0.0 else [round(c.real, 12), round(c.imag, 12)]


def calib_fingerprint(calib):
    """模型内容指纹：规范 JSON 的 sha256 前 12 位（只认数，不认来源）。"""
    payload = {
        'atoms': [[int(a['id']), [_fp_num(v) for v in a['pos']],
                   _fp_num(a.get('delta_mhz', 0.0)),
                   _fp_num(a.get('Omega_mhz', 0.0)),
                   _fp_num(a.get('Omega_phase_rad', 0.0)),
                   _fp_num(a.get('gamma_mhz', 0.0))]
                  for a in calib.get('atoms') or []],
        'pairs': [[int(p['i']), int(p['j']),
                   _fp_num(p.get('V', 0.0)),
                   _fp_num(p.get('J', 0.0)),
                   _fp_num(p.get('A', 0.0))]
                  for p in calib.get('pairs') or []],
    }
    blob = json.dumps(payload, sort_keys=True, separators=(',', ':')).encode()
    return hashlib.sha256(blob).hexdigest()[:12]


def calib_summary(calib):
    """进 RunSpec 的模型摘要（卡片与复现核对用）。**不含通道**。"""
    pairs = calib.get('pairs') or []
    js = [abs(complex(p.get('J') or 0.0)) for p in pairs]
    return {'n_atoms': len(calib.get('atoms') or []),
            'n_pairs': len(pairs),
            'fingerprint': calib_fingerprint(calib),
            'sum_abs_J': round(sum(js), 9),
            'max_abs_J': round(max(js, default=0.0), 9)}


def calib_receipt(calib):
    """**装配回执**：只放能由 calib **现算**的事实（不存副本，不会漂）。

    进 RunSpec 的 ``build_receipt``，供卡片阅读；运行路径不使用。
    """
    pairs = calib.get('pairs') or []
    js = [abs(complex(p.get('J') or 0.0)) for p in pairs]
    boundary = calib.get('boundary') or {}
    tail = calib.get('tail_norm_bound', boundary.get('H_tail_norm_bound'))
    return {
        'fingerprint': calib_fingerprint(calib),
        'n_atoms': len(calib.get('atoms') or []),
        'n_pairs': len(pairs),
        'sum_abs_J': round(sum(js), 9),
        'max_abs_J': round(max(js, default=0.0), 9),
        'n_dropped': len(calib.get('dropped') or []),
        'n_guarded': sum(int((p.get('info') or {}).get('n_guarded', 0))
                         for p in pairs),
        'tail_norm_bound': None if tail is None else float(tail),
    }


def calib_dir():
    """大模型输入的有界目录：``$EL_CALIB_DIR`` 或 ``<cwd>/.mecha-v2/calib``。

    只允许从这里读 calib 文件——不开放任意路径（配置面不该能读宿主任意文件）。
    """
    base = os.environ.get('EL_CALIB_DIR')
    root = Path(base) if base else Path.cwd() / '.mecha-v2' / 'calib'
    return root.resolve()


def load_calib_file(name):
    """读 ``el_build_calib(file=…)`` 指向的 JSON。

    相对名解析到 :func:`calib_dir`；绝对路径必须仍在该目录内（穿越/越界直接拒绝）。
    """
    if not isinstance(name, str) or not name.strip():
        raise ValueError('el_build_calib(file=…)：文件名须是非空字符串')
    base = calib_dir()
    raw = Path(name.strip())
    path = (raw if raw.is_absolute() else base / raw).resolve()
    if path != base and base not in path.parents:
        raise ValueError('el_build_calib(file=…)：文件必须位于 %s 之内，收到 %s' % (base, path))
    if not path.is_file():
        raise ValueError('el_build_calib(file=…)：文件不存在：%s（目录 %s）' % (path, base))
    try:
        data = json.loads(path.read_text(encoding='utf-8'))
    except json.JSONDecodeError as exc:
        raise ValueError('el_build_calib(file=…)：不是合法 JSON：%s（%s）' % (path, exc)) from None
    if not isinstance(data, dict):
        raise ValueError('el_build_calib(file=…)：顶层须是对象（calib）')
    return data


def calib_from_params(p):
    """快照里的模型：**只有** ``manybody.calib`` 一处。不校验内容（在 resolve_model 里做）。

    文件/ARC 不是 RunSpec 的写法——它们由 ``el_build_calib`` 转成 calib 后写进同一个键。
    """
    p = p or {}
    return p.get('manybody.calib')


def atom_count(p):
    """快照的原子数：来自 ``manybody.calib``（没有模型就是 0）。"""
    explicit = calib_from_params(p)
    if isinstance(explicit, dict) and isinstance(explicit.get('atoms'), (list, tuple)):
        return len(explicit['atoms'])
    return 0


def resolve_model(p, cache=None, *, warnings=None):
    """快照 → ``(calib, None, terms)``。多体**只有一条装配路**：行内 calib。

    ``cache`` 保留在签名里只为兼容既有调用点；行内 calib 不需要它。
    """
    p = p or {}
    explicit = calib_from_params(p)
    if explicit is None:
        raise ValueError('多体需要 manybody.calib（行内模型）。'
                         '文件请用 el_build_calib({"file": …})，'
                         'ARC 请用 el_build_calib({"atoms": […g/r…], "arc": {…}})。')
    calib = validate_calib(explicit)
    cutoff = p.get('manybody.distance_cutoff_um')
    if cutoff is not None:
        cutoff = float(cutoff)
        if not np.isfinite(cutoff) or cutoff <= 0:
            raise ValueError('manybody.distance_cutoff_um 必须为有限正数')
        atoms = calib['atoms']
        kept, dropped = [], list(calib.get('dropped') or [])
        for pr in calib['pairs']:
            pi = np.asarray(atoms[int(pr['i'])]['pos'], float)
            pj = np.asarray(atoms[int(pr['j'])]['pos'], float)
            R = float(np.linalg.norm(pi - pj))
            if R <= cutoff:
                kept.append(pr)
            else:
                dropped.append(dict(i=int(pr['i']), j=int(pr['j']), R_um=R,
                                    reason='r_cut_product'))
        calib['pairs'] = kept
        calib['dropped'] = dropped
        calib['terms'] = [name for name, key in (('I1', 'V'), ('I2', 'J'), ('I3', 'A'))
                          if any(abs(pr[key]) > 0.0 for pr in kept)]
        if dropped and warnings is not None:
            warnings.append('[多体标定] distance_cutoff_um=%.6g 丢弃 %d 对（近似）'
                            % (cutoff, len(dropped) - len(explicit.get('dropped') or [])))
    return calib, None, list(calib['terms'])


def drive_from_calib(calib):
    """calib 逐原子 Ω/δ/Γ → arrays（MHz；缺省 0，与 drive_from_calib 同口径）。"""
    atoms = calib.get('atoms') or []
    Om = np.array([float(a.get('Omega_mhz', 0.0))
                   * np.exp(1j * float(a.get('Omega_phase_rad', 0.0)))
                   for a in atoms], complex)
    delta = np.array([float(a.get('delta_mhz', 0.0)) for a in atoms], float)
    gamma = np.array([float(a.get('gamma_mhz', 0.0)) for a in atoms], float)
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
    # 允许直接给原子数：calib 直通路没有 ManyBodyModel 对象。
    n = int(model) if isinstance(model, int) else len(model.atoms)
    # Validate the compact grammar without allocating a 2**N vector.  The
    # execution backend materializes it only when ED actually needs it; MPS
    # consumes the local vectors directly.
    validate_psi_spec(spec, n)
    return spec


PREPARE_KINDS = ('ground_of', 'permutation_sector')


def prepare_initial_state(spec, *, n_sites, solver, warnings=None,
                          mps_order=None, charge_sector=None,
                          chi_max=200, max_sweeps=20, memory_budget_mb=None,
                          calib=None, terms=None):
    """``manybody.psi0`` 的**制备态**语法 → 具体初态（向量或原生 MPS）。

    支持两种通用形态（都不含领域知识）：

    - ``{"kind":"ground_of","calib":{…},"terms":[…]}}``：初态是*另一个模型*的基态；
    - ``{"kind":"permutation_sector","base":<psi0>,"permutation":[…],"sign":±1}``：
      把 ``base`` 投影到显式站点置换的本征扇区（要求对合），用于分支初态。

    返回 ``(psi0, meta)``；``meta`` 进 RunSpec 的 ``quench`` 字段。
    """
    kind = spec.get('kind') if isinstance(spec, dict) else None
    if kind == 'ground_of':
        return _prepare_ground_of(spec, n_sites=n_sites, solver=solver,
                                  warnings=warnings, mps_order=mps_order,
                                  charge_sector=charge_sector, chi_max=chi_max,
                                  max_sweeps=max_sweeps,
                                  memory_budget_mb=memory_budget_mb)
    if kind == 'permutation_sector':
        return _prepare_permutation_sector(
            spec, n_sites=n_sites, solver=solver, warnings=warnings,
            mps_order=mps_order, charge_sector=charge_sector, chi_max=chi_max,
            max_sweeps=max_sweeps, memory_budget_mb=memory_budget_mb,
            calib=calib, terms=terms)
    raise ValueError('未知的制备态 kind=%r；支持 %s' % (kind, ', '.join(PREPARE_KINDS)))


def _prepare_ground_of(spec, *, n_sites, solver, warnings, mps_order,
                       charge_sector, chi_max, max_sweeps, memory_budget_mb):
    from .assemble import build_H_sparse
    from .equilibrium import ground_state
    if not isinstance(spec, dict) or not spec.get('calib'):
        raise ValueError('psi0.kind=ground_of 需要 calib（初态模型）')
    calib0 = validate_calib(spec['calib'])
    if len(calib0['atoms']) != int(n_sites):
        raise ValueError('psi0 的 calib 原子数 %d ≠ 演化模型 %d'
                         % (len(calib0['atoms']), int(n_sites)))
    terms0 = list(spec.get('terms') or calib0['terms'])
    module = TIME_BACKENDS[normalize_time_backend(solver)].module
    if module in ('engine_mps', 'engine_mps_gpu'):
        from . import dmrg as _dm
        out = _dm.dmrg_cluster(calib0, None, None, chi_max=int(chi_max),
                                       max_sweeps=int(max_sweeps), return_state=True,
                                       charge_sector=charge_sector, order=mps_order,
                                       memory_budget_mb=memory_budget_mb)
        if warnings is not None:
            warnings.append('[初态] ground_of 由 DMRG 准备：E0=%r，chi=%r'
                            % (out.get('E0'), out.get('chi')))
        return out['state'], {'kind': 'ground_of', 'prepared_by': 'dmrg',
                              'E0': float(out['E0']), 'terms': terms0}
    H = build_H_sparse(calib0, terms=terms0)
    if H.shape[0] > (1 << 20):
        raise ValueError('ED 初态维度 %d 过大；请指定 MPS 后端（solver=tdvp）'
                         % H.shape[0])
    r = ground_state(H, dense_max=2048, k=1)
    vec = np.asarray(r['vec'], complex).reshape(-1)
    if warnings is not None:
        warnings.append('[初态] ground_of 由 ED 准备：E0=%r，dim=%d'
                        % (r.get('E0'), vec.size))
    return vec, {'kind': 'ground_of', 'prepared_by': 'ed',
                 'E0': float(r['E0']), 'terms': terms0}


def _validate_involution(permutation, n_sites):
    n = int(n_sites)
    if not isinstance(permutation, (list, tuple)) or len(permutation) != n:
        raise ValueError('permutation 必须是长度 %d 的下标列表' % n)
    p = [int(x) for x in permutation]
    if sorted(p) != list(range(n)):
        raise ValueError('permutation 必须是 0..%d-1 的置换' % (n - 1))
    if any(p[p[i]] != i for i in range(n)):
        raise ValueError('permutation 必须是对合（U²=I）才能定义 ±1 扇区')
    return p


def _project_vector(vec, n_sites, permutation, sign):
    """把全空间态矢投影到置换算符的 ±1 本征空间。返回 (归一化向量, 权重, 重叠)。"""
    n = int(n_sites)
    vec = np.asarray(vec, complex).reshape(-1)
    if vec.size != (1 << n):
        raise ValueError('初态向量长度 %d ≠ 2^N=%d' % (vec.size, 1 << n))
    idx = np.arange(vec.size, dtype=np.int64)
    shifts = np.arange(n - 1, -1, -1)
    bits = (idx[:, None] >> shifts) & 1                      # (dim, n)，site0=MSB
    target = bits[:, np.asarray(permutation)] @ (1 << shifts)
    permuted = np.empty_like(vec)
    permuted[target] = vec
    overlap = float(np.vdot(vec, permuted).real)
    weight = float(np.clip((1.0 + sign * overlap) / 2.0, 0.0, 1.0))
    if weight <= 1e-12:
        raise ValueError('该扇区投影权重 %.3g 太小：base 落在相反本征空间'
                         '（<P>=%.6g，sign=%+d）' % (weight, overlap, sign))
    out = (vec + sign * permuted) / 2.0
    out = out / np.linalg.norm(out)
    return out, weight, overlap


def _product_mps_state(calib, base, order, charge_sector):
    """乘积态初态（模式串/索引/None）→ 原生 TeNPy 态（与演化同 order/charge）。"""
    from . import dmrg as _dm
    from .state import product_vectors, make_mps_state
    from tenpy.networks.mps import MPS
    order = list(order)
    n = len(calib['atoms'])
    vectors = product_vectors(None if base is None else base, n)
    p_state = [int(round(abs(v[1]) ** 2)) for v in vectors]   # 0=|g⟩(up) 1=|r⟩(down)
    _, onsite, couplings, _info = _dm.terms_from_calib(calib, None, None)
    conserve = 'Sz' if charge_sector is not None else None
    model = _dm.cluster_model(onsite, couplings, conserve=conserve)
    psi = MPS.from_product_state(model.lat.mps_sites(),
                                 [p_state[int(a)] for a in order], bc='finite')
    return make_mps_state(psi, order, charge_sector)


def _prepare_permutation_sector(spec, *, n_sites, solver, warnings, mps_order,
                                charge_sector, chi_max, max_sweeps,
                                memory_budget_mb, calib, terms):
    from .state import state_vector
    sign = spec.get('sign', 1)
    if isinstance(sign, bool) or int(sign) not in (-1, 1):
        raise ValueError('permutation_sector 的 sign 取 +1 或 -1')
    sign = int(sign)
    permutation = _validate_involution(spec.get('permutation'), n_sites)
    base = spec.get('base')
    if base is None:
        raise ValueError('permutation_sector 需要 base（任意 psi0 语法；'
                         '如 None/模式串/{kind:"ground_of",…}）')
    module = TIME_BACKENDS[normalize_time_backend(solver)].module
    if module == 'engine_ed':
        if isinstance(base, dict) and base.get('kind') in PREPARE_KINDS:
            base_vec, base_meta = prepare_initial_state(
                base, n_sites=n_sites, solver=solver, warnings=warnings,
                mps_order=mps_order, charge_sector=charge_sector,
                chi_max=chi_max, max_sweeps=max_sweeps,
                memory_budget_mb=memory_budget_mb, calib=calib, terms=terms)
        else:
            base_vec, base_meta = np.asarray(state_vector(base, n_sites), complex), None
        vec, weight, overlap = _project_vector(base_vec, n_sites, permutation, sign)
        if warnings is not None:
            warnings.append('[初态] permutation_sector(sign=%+d) 由 ED 投影：'
                            '权重=%.9g，<P>_base=%.9g' % (sign, weight, overlap))
        return vec, {'kind': 'permutation_sector', 'prepared_by': 'ed', 'sign': sign,
                     'projection_weight': weight,
                     'sector_overlap': float(sign),
                     'base_overlap': float(overlap),
                     'permutation_overlap': float(sign),
                     'base': base_meta}
    # MPS 后端：先拿 base 的原生态，再走 mps_symmetry 的投影
    from . import mps_symmetry
    if isinstance(base, dict) and base.get('kind') == 'mps':
        state, base_meta = base, None
    elif isinstance(base, dict) and base.get('kind') in PREPARE_KINDS:
        state, base_meta = prepare_initial_state(
            base, n_sites=n_sites, solver=solver, warnings=warnings,
            mps_order=mps_order, charge_sector=charge_sector, chi_max=chi_max,
            max_sweeps=max_sweeps, memory_budget_mb=memory_budget_mb,
            calib=calib, terms=terms)
    else:
        if calib is None:
            raise ValueError('MPS 的 permutation_sector 需要 calib（用来建乘积态）')
        from . import dmrg as _dm
        # 初态的 MPS 顺序必须与**本次运行**一致：DMRG/演化不会隐式重排
        # （否则报 Warm-start order mismatch）。
        default_order, _o, _c, _i = _dm.terms_from_calib(calib, None, None)
        order = list(mps_order) if mps_order is not None else default_order
        state = _product_mps_state(calib, base, order, charge_sector)
        base_meta = {'kind': 'product', 'order': order}
    out = mps_symmetry.project_permutation_sector(
        state, permutation, sign, chi_max=chi_max,
        memory_budget_mb=memory_budget_mb)
    weight = float(out.get('projection_weight'))
    sector_overlap = float(np.real(out.get('permutation_overlap', 0.0)))
    if warnings is not None:
        warnings.append('[初态] permutation_sector(sign=%+d) 由 MPS 投影：'
                        '权重=%.9g，<P>=%.9g' % (sign, weight, sector_overlap))
    return out['state'], {'kind': 'permutation_sector', 'prepared_by': 'mps',
                          'sign': sign,
                          'projection_weight': weight,
                          'sector_overlap': sector_overlap,
                          # 权重 → 基态在扇区上的重叠：w=(1+sign·<P>)/2
                          'base_overlap': float((2.0 * weight - 1.0) / sign),
                          'permutation_overlap': sector_overlap,
                          'base': base_meta}


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
    if backend.module == 'tdvp_dense':
        options['return_state'] = bool(mb_get(p, 'manybody.return_state'))
    if backend.segments and backend.module in ('engine_ed', 'engine_mps', 'tdvp_dense'):
        options['drive_interp'] = mb_get(p, 'manybody.drive_interp') or 'left'
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
    ed_max_dim = int(mb_get(p, 'manybody.ed_max_dim'))
    # 唯一装配路：行内 manybody.calib（文件/ARC 已由 el_build_calib 转好）
    calib, model, terms = resolve_model(p, cache, warnings=warnings)
    n = len(calib['atoms'])
    if n == 0:
        raise ValueError('多体：manybody.calib.atoms 为空——先给模型'
                         '（文件/ARC 用 el_build_calib 转成 calib）')
    # 时间网格：复用 dynamic.t_max/npts（与单/两体同一 _time_grid 语义 + 报错不静默）
    t_max, npts, t = _time_grid(p, t_grid)

    solver_in = method or mb_get(p, 'manybody.solver')
    psi0_raw = p.get('manybody.psi0')
    quench = isinstance(psi0_raw, dict) and psi0_raw.get('kind') in PREPARE_KINDS
    # `ground_of` 要先知道后端（ED 出向量 / MPS 出原生态）才能准备，故留到
    # solver 解析之后再落地；其余初态语法在这里照旧校验。
    psi0 = None if quench else resolve_mb_psi(model if model is not None else n, psi0_raw)
    solver_request = solver_in
    if isinstance(psi0, dict) and psi0.get('kind') == 'mps':
        if str(solver_in or 'auto').lower() == 'auto':
            native = psi0.get('backend')
            solver_request = {
                'tenpy': 'tdvp',
                'tdvp-dense': 'tdvp-dense',
                'torch': 'swap-tebd-gpu',
            }.get(native)
            if solver_request is None:
                raise ConfigConflict(
                    '未知的 MPS 初态 backend=%r；支持 tenpy/tdvp-dense/torch'
                    % (native,), keys=('manybody.psi0',))
            warnings.append('[多体] solver=auto 按 MPS 初态选择 '+solver_request)
    Om, delta, gamma = drive_from_calib(calib)
    plan = build_plan(calib, Om=Om, delta=delta, gamma=gamma,
                      terms=terms, constraint=p.get('manybody.constraint'),
                      max_states=ed_max_dim)
    solver = resolve_solver(solver_request, n, ed_max_dim, warnings,
                            effective_dim=plan['dim'])
    quench_meta = None
    if quench:
        # 制备态初态（另一个模型的基态 / 置换扇区投影）。按已解析的后端落地：
        # ED 给全空间向量，MPS 给原生 TeNPy 态（同 order/charge）。
        psi0, quench_meta = prepare_initial_state(
            psi0_raw, n_sites=n, solver=solver, warnings=warnings,
            mps_order=mb_get(p, 'manybody.mps_order'),
            charge_sector=mb_get(p, 'manybody.charge_sector'),
            chi_max=int(mb_get(p, 'manybody.chi_max')),
            max_sweeps=int(mb_get(p, 'manybody.max_sweeps')),
            memory_budget_mb=p.get('manybody.memory_budget_mb'),
            calib=calib, terms=terms)
    segments = validate_drive_segments(p.get('manybody.drive_segments'), n)
    drive_interp = mb_get(p, 'manybody.drive_interp') or 'left'
    if drive_interp not in ('left', 'midpoint'):
        raise ConfigConflict('manybody.drive_interp 须为 left/midpoint，得到 %r'
                             % (drive_interp,),
                             keys=('manybody.drive_interp',))
    if segments and drive_interp != 'left':
        module = TIME_BACKENDS[solver].module
        if module not in ('engine_ed', 'engine_mps', 'tdvp_dense'):
            raise ConfigConflict(
                '%s 尚不支持 drive_interp=%s；只有 ED/CPU MPS/tdvp-dense 支持中点规则'
                % (solver, drive_interp),
                keys=('manybody.drive_interp',))
    pauli_specs = validate_pauli_strings(p.get('manybody.pauli_strings'), n_atoms=n)
    validate_time_capabilities(
        solver, segments=segments, plan=plan, gamma=gamma, psi0=psi0,
        return_state=mb_get(p, 'manybody.return_state'),
        charge_sector=mb_get(p, 'manybody.charge_sector'),
        collapse_ops=mb_get(p, 'manybody.collapse_ops'), mps_order=mb_get(p,'manybody.mps_order'),
        pauli_strings=pauli_specs)
    if plan['projected']:
        warnings.append('[多体装配] %s：dim %d → %d' %
                        (plan['constraint']['kind'], plan['full_dim'], plan['dim']))
    ek = _backend_options(p, solver, ed_max_dim)
    if pauli_specs:
        # 规范化后的 [{'label','terms'}] 才是引擎要的形状（原始快照里可能是
        # 纯列表/带 label 的 dict 混写），所以在这里覆盖，不走 options 直通。
        ek['pauli_strings'] = pauli_specs
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
        warnings.append('[多体标定] 标定截断/丢弃 %d 对（尾项未计入 H）' % ndrop)
    guarded = sum(int((pr.get('info') or {}).get('n_guarded', 0))
                  for pr in calib.get('pairs', []))
    if guarded:
        warnings.append('[多体标定] %d 个近简并中间态被 guard 剔除'
                        '（downfolding 在该支不适用，见 M0 诚实边界）' % guarded)

    states = [{'label': 'atom%d@%s' % (a['id'], _fmt_pos(a['pos'])),
               'atom_id': a['id']} for a in calib['atoms']]
    # 分段常数驱动的**结构校验**必须在 build 阶段失败并点名第几段/哪个键/期望多长；
    # 不能等到工作线程 `_propagate_segments` 才炸（报错不指向病因）。
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
        'calib': calib, 'calib_summary': calib_summary(calib),
        'build': p.get('manybody.build'),
        'label': p.get('manybody.label'),
        'notes': p.get('manybody.notes'),
        'build_receipt': calib_receipt(calib),
        'hamiltonian_plan': plan, 'plan_summary': plan_summary(plan),
        'n_atoms': n, 'dim': plan['dim'], 'full_dim': 1 << n,
        'psi0': psi0, 'Om': Om, 'delta': delta, 'gamma': gamma,
        'quench': quench_meta,
        'segments': segments,
        'collapse_ops': p.get('manybody.collapse_ops') or [],
        'n_trajectories': int(p.get('manybody.n_trajectories') or 32),
        'terms': list(terms),
        'template': (model.template if model is not None else 'calib'),
        'engine_kwargs': ek, 'warnings': warnings,
        # 契约兼容字段（让 summarize_run/check_run 的通用路径不 KeyError）
        'states': states, 'labels': [s['label'] for s in states],
        'fields': None, 'decays': [], 'n_waves': {},
        'vdd_matrix': None, 'pair': None, 'field_loss': None,
    }
