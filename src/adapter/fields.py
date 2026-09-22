# -*- coding: utf-8 -*-
"""建场层（`fields`）——波形/含时场的**语义构造**适配层（§11「不可提炼的适配」）。

把"建含时电磁场"这块唯一还退化成裸结构的语义面（`dynamic.waves_*` 曾是橡皮图章
`type='list'`），补成让 AI 得心应手的适配层：AI 说"在 60S→60P 上打一个 Ω/2π=50 MHz
的 π 脉冲"就能建好，改一个参数只发一个 patch，建完即 preview 验证。

**不套层（§9 / 纲领 §13）**：波形的唯一真值仍是快照里 `serialize_waves` 同构的 dict；
本层只"正确拼它 + 校验它（registry 的 wave_list 契约）+ 记一条事件"，**不引入并行
表示**（`fields_selfcheck` 逐位对拍 `restore_waves`）。`src/core` **一行不改**，只调用
`waveforms`/`static_atom`(rabi_mhz/intensity_mwcm2_from_rabi/dipole_amplitude/field_e0_sq)/
`atomic_levels`(find_state_index/state_key) 既有函数（I5 人机同源）。

**因地制宜（对 §4 的一处调整）**：设计稿说"快照里只存指针"，但 core 的 `restore_waves`
读不了指针、且 core 不能改。故指针化落在**事件/回执层**——大数组经 `bus._shrink` 自动
转指针、`get_waves` 只回**摘要**、`get_envelope` 单独取曲线并过体积闸。快照仍存可
restore 的物理量。这才是 I3 的真关切（"数组永不内联进事件或回程"），且不破 core。

三层 API（都经同一 choke point 写 `dynamic.waves_<atom>`）：
  L2 意图级：set_drive / pi_pulse / pi2_pulse / square_mw / ramsey / stirap / arp
  L1 逐波级：add_wave / edit_wave / remove_wave / get_waves / set_sequence
  L0 原始级：set_waves（整表替换，serialize 同构）
辅助换算（§2，写回物理量、不参与结算）：freq_for / intensity_for_rabi / describe_drive /
  suggest_freq。
"""
import numpy as np

from mecha.errors import MechaError, InvalidValue, did_you_mean

__all__ = ['Fields', 'pi_ns', 'pi2_ns']


def pi_ns(rabi_mhz):
    """π 脉冲时长（ns）= 500/Ω_MHz（布居 ∝ sin²(π·Ω_MHz·t_µs)，见 build_fields_exact 周频约定）。"""
    return 500.0 / float(rabi_mhz)


def pi2_ns(rabi_mhz):
    """π/2 脉冲时长（ns）= 250/Ω_MHz。"""
    return 250.0 / float(rabi_mhz)


def _is_ptr(x):
    return isinstance(x, dict) and x.get('$pointer') is True


def _fnum(v, what):
    """数值字段校验：非数值→InvalidValue（当场、可教学），不拖到 float() 的 ValueError。"""
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        raise InvalidValue('%s 须是数值（得到 %r）' % (what, v))
    return float(v)


# ---------------------------------------------------------------------
# 重叠驱动检测（P0：把"扫描忘了清"这类**静默给错答案**变成大声告警，D5）。
# 纯函数，供 fields/ops 共用；cockpit 只读事件里的标志（不重算，不破独立）。
# ---------------------------------------------------------------------
def wave_span(w):
    """波的时间支撑 [t0,t1]（ns，绝对时间含 offset）——重叠检测用（近似即可）。"""
    off = float(w.get('offset_ns', 0.0) or 0.0)
    if w.get('type') == 'arb':
        t = w.get('t_ns') or []
        if len(t) >= 2 and not _is_ptr(t):
            return (off + float(t[0]), off + float(t[-1]))
        return (off, off)
    env = w.get('env') or ()
    kind = env[0] if env else None
    try:
        if kind == 'square':
            start, edge = float(env[1]), float(env[2])
            fall = float(env[4]) if len(env) > 4 else start + 2 * edge
            return (off + start, off + fall + edge)
        if kind == 'gauss':
            center, sigma = float(env[1]), float(env[2])
            return (off + center - 3 * sigma, off + center + 3 * sigma)
        if kind == 'file':
            t = env[1] if len(env) > 1 else []
            if len(t) >= 2 and not _is_ptr(t):
                return (off + float(t[0]), off + float(t[-1]))
    except (TypeError, ValueError, IndexError):
        pass
    return (off, off)


def _same_transition(a, b, rtol=1e-6):
    """两波是否驱动同一跃迁（同 drive 身份键，或同载频）。"""
    da, db = a.get('drive'), b.get('drive')
    if da and db and da.get('g') == db.get('g') and da.get('e') == db.get('e'):
        return True
    fa, fb = a.get('freq_mhz'), b.get('freq_mhz')
    if fa and fb and abs(float(fa) - float(fb)) <= rtol * max(1.0, abs(float(fa))):
        return True
    return False


def detect_wave_overlap(waves):
    """检测"同一跃迁、时间重叠"的多条驱动（扫描忘清的签名）→ dict 或 None。

    **不误报合法多波**：Ramsey（同跃迁但不重叠）、STIRAP（重叠但不同跃迁）都不触发。
    只报"同跃迁 + 时间重叠"——那几乎必是意外累积（没人故意在同一跃迁上叠两个重脉充）。
    """
    ws = [w for w in (waves or []) if isinstance(w, dict)]
    for i in range(len(ws)):
        for k in range(i + 1, len(ws)):
            a, b = ws[i], ws[k]
            if not _same_transition(a, b):
                continue
            (a0, a1), (b0, b1) = wave_span(a), wave_span(b)
            if a0 < b1 and b0 < a1:                    # 时间区间相交
                return {'n_waves': len(ws),
                        'overlap': [a.get('name'), b.get('name')],
                        'warning': '同一跃迁有 ≥2 条时间重叠的驱动（%s / %s，共 %d 条）'
                                   '——多半是扫描忘了清空；用 clear_waves 或 remove_wave '
                                   '清掉多余的（否则你测的是多驱叠加，不是单脉充）'
                                   % (a.get('name'), b.get('name'), len(ws))}
    return None


class Fields:
    """建场层：持有 session（配置快照 + 惰性 cache + 写 choke point）。"""

    def __init__(self, session):
        self.session = session
        self._basis_key = None       # 基组记忆化（同配置同输出，纯记忆化非状态）
        self._basis_val = None

    # ================= 基组 / 身份解析 =================
    def _basis(self):
        """从当前配置建基组（build_states，与动态页同源）→ (states, snapshot)。

        按 (groups, B, E_vpcm) 记忆化：set_drive 会多次解析基组（freq/intensity/rabi），
        大基组下重复 build_states 太慢；同配置必同输出（纯记忆化，不是状态）。
        """
        import json
        from ..core import runspec as RS
        snap = self.session.effective()
        groups = RS.groups_from_params(snap)
        if not groups:
            raise MechaError('基组为空——先设 groups.list 才能按跃迁建场',
                             hint='set_param("groups.list", [{n,l,j}, ...])')
        key = (json.dumps(groups, sort_keys=True, default=str),
               snap.get('field.B_gauss'), snap.get('field.E_vpcm'))
        if self._basis_key == key and self._basis_val is not None:
            return self._basis_val, snap
        states = RS.build_states(self.session.cache(), groups,
                                 snap.get('field.B_gauss'), snap.get('field.E_vpcm'))
        self._basis_key, self._basis_val = key, states
        return states, snap

    def _resolve(self, states, spec, which='态'):
        """身份键元组 / label 字符串 / 索引 int → (state_dict, index)；找不到**报错**（绝不静默）。"""
        from ..core.atomic_levels import find_state_index
        if isinstance(spec, bool):
            raise InvalidValue('%s 不能是 bool' % which)
        if isinstance(spec, int):
            if 0 <= spec < len(states):
                return states[spec], spec
            raise InvalidValue('%s 索引 %r 越界（基组 %d 个态）' % (which, spec, len(states)))
        if isinstance(spec, str):
            hits = [i for i, s in enumerate(states) if s.get('label') == spec]
            if len(hits) == 1:
                return states[hits[0]], hits[0]
            if not hits:
                raise InvalidValue('%s label %r 不在基组' % (which, spec),
                                   hint='用 get_waves 前先看基组 label；或给身份键',
                                   suggest=did_you_mean(spec, [s.get('label') for s in states]))
            raise InvalidValue('%s label %r 有 %d 个匹配（请给身份键消歧）'
                               % (which, spec, len(hits)))
        if isinstance(spec, (list, tuple)):
            i = find_state_index(states, spec)
            if i < 0:
                raise InvalidValue('%s 身份键 %r 不在基组' % (which, tuple(spec)))
            return states[i], i
        raise InvalidValue('%s 须是 身份键/label/索引，得到 %r' % (which, spec))

    def _q_of(self, pol):
        from ..core.static_atom import POLARIZATION_Q
        q = POLARIZATION_Q.get(pol)
        if q is None:
            raise InvalidValue("偏振 %r 无单 q 映射（椭偏请给主 q 或直接给强度）" % pol,
                               suggest=list(POLARIZATION_Q))
        return q

    # ================= §2 辅助换算（写回物理量，不参与结算） =================
    def freq_for(self, src, tgt, detuning_mhz=0.0):
        """「定频率」：freq_mhz = |E_mhz[tgt] − E_mhz[src]| + Δ（相对裸能级差，不含动态 AC）。"""
        states, _ = self._basis()
        gs, _ = self._resolve(states, src, '源态')
        ge, _ = self._resolve(states, tgt, '目标态')
        bare = abs(float(ge['E_mhz']) - float(gs['E_mhz']))
        return {'freq_mhz': bare + float(detuning_mhz), 'bare_mhz': bare,
                'detuning_mhz': float(detuning_mhz),
                'src': gs['label'], 'tgt': ge['label']}

    def intensity_for_rabi(self, src, tgt, pol, rabi_mhz):
        """「定光强」：目标 Rabi(Ω/2π, MHz) → intensity_mwcm2（core 逆换算，不绕平方律）。"""
        from ..core.static_atom import dipole_amplitude, intensity_mwcm2_from_rabi
        states, _ = self._basis()
        gs, _ = self._resolve(states, src, '源态')
        ge, _ = self._resolve(states, tgt, '目标态')
        q = self._q_of(pol)
        d = dipole_amplitude(self.session.cache(), gs, ge, q)
        if abs(d) < 1e-12:
            raise InvalidValue('该对在偏振 %r 下无偶极耦合（Δm 不匹配）——不静默建零效果波'
                               % pol, hint='换偏振或换跃迁对')
        inten = intensity_mwcm2_from_rabi(abs(d), float(rabi_mhz))
        return {'intensity_mwcm2': inten, 'dipole_au': abs(d), 'rabi_mhz': float(rabi_mhz)}

    def rabi_of(self, src, tgt, pol, intensity_mwcm2):
        """强度 → Rabi（describe_drive / 时长计算用；core 正向换算）。"""
        from ..core.static_atom import dipole_amplitude, rabi_mhz, field_e0_sq
        states, _ = self._basis()
        gs, _ = self._resolve(states, src, '源态')
        ge, _ = self._resolve(states, tgt, '目标态')
        q = self._q_of(pol)
        d = dipole_amplitude(self.session.cache(), gs, ge, q)
        return rabi_mhz(abs(d), field_e0_sq(float(intensity_mwcm2)))

    def describe_drive(self, atom, name_or_index):
        """§2.5 实时反馈（M2 场版）：给一条波的当前物理量，反算并报出 Ω / Δ。"""
        w = self._find(self._waves(atom), name_or_index)
        out = {'name': w.get('name'), 'freq_mhz': w.get('freq_mhz'),
               'intensity_mwcm2': w.get('intensity'), 'pol': w.get('pol')}
        dr = w.get('drive')
        if not dr:
            out['note'] = '这条波没记 drive（直接填的物理量）——无法反算 Δ；Ω 需跃迁对'
            return out
        try:
            g, e = dr['g'], dr['e']
            out['rabi_mhz'] = self.rabi_of(tuple(g), tuple(e), w.get('pol', 'π'),
                                           w.get('intensity'))
            f = self.freq_for(tuple(g), tuple(e))
            out['detuning_mhz'] = float(w.get('freq_mhz')) - f['bare_mhz']
            out['bare_mhz'] = f['bare_mhz']
        except MechaError as ex:
            out['note'] = '反算失败（drive 的态可能已不在基组）：%s' % ex
        return out

    def suggest_freq(self, pol):
        """§2.1 默认载频建议：该偏振下**偶极最强的跨流形**跃迁（刻意排除同流形）。"""
        from ..core.static_atom import dipole_amplitude
        states, _ = self._basis()
        q = self._q_of(pol)
        cache = self.session.cache()
        best = None
        for i in range(len(states)):
            for k in range(i + 1, len(states)):
                a, b = states[i], states[k]
                if (a['n'], a['l'], a['j']) == (b['n'], b['l'], b['j']):
                    continue                      # 同流形（Zeeman/超精细劈裂）——静默坑，排除
                d = abs(dipole_amplitude(cache, a, b, q))
                if d < 1e-12:
                    continue
                if best is None or d > best[0]:
                    bare = abs(float(b['E_mhz']) - float(a['E_mhz']))
                    best = (d, bare, a['label'], b['label'])
        if best is None:
            raise MechaError('偏振 %r 下找不到跨流形偶极跃迁（基组只有一个流形？）' % pol)
        return {'freq_mhz': best[1], 'dipole_au': best[0],
                'src': best[2], 'tgt': best[3], 'pol': pol}

    # ================= 波形 dict 构造（serialize_waves 同构） =================
    def _samples(self, spec):
        t, a = spec.get('t_ns'), spec.get('amp')
        path = spec.get('env_path')
        if path and (t is None or a is None):
            z = np.load(path)
            t = z['t_ns'] if 't_ns' in z.files else z[z.files[0]]
            a = z['amp'] if 'amp' in z.files else z[z.files[1]]
        if t is None or a is None:
            raise InvalidValue('file/arb 须给 t_ns+amp 或 env_path')
        return [float(x) for x in t], [float(x) for x in a]

    def _build_wave(self, spec):
        """友好 spec → serialize_waves 同构 dict（供 add_wave/set_sequence/set_drive 用）。

        **spec 键校验**（P0：未知键不再静默忽略→用错默认，D5）：先验 kind，再验键属
        于该 kind 的合法集；未知键 → InvalidValue + did_you_mean（第一次就打对）。
        """
        if not isinstance(spec, dict):
            raise InvalidValue('wave spec 须是 dict（得到 %s）' % type(spec).__name__)
        kind = spec.get('kind', 'square')
        common = {'name', 'kind', 'freq_mhz', 'pol', 'intensity_mwcm2',
                  'intensity', 'offset_ns', 'elliptical', 'drive'}
        per = {'square': {'start_ns', 'edge_ns', 'height', 'fall_start_ns'},
               'gauss': {'center_ns', 'sigma_ns', 'peak'},
               'file': {'t_ns', 'amp', 'env_path'},
               'arb': {'t_ns', 'amp', 'env_path'}}
        if kind not in per:
            raise InvalidValue('kind=%r 非法' % kind, suggest=list(per))
        allowed = common | per[kind]
        unknown = sorted(set(spec) - allowed)
        if unknown:
            raise InvalidValue(
                '%s 波 spec 有未知键 %s（会被静默忽略→用错默认，D5 不许）'
                % (kind, unknown),
                hint='该 kind 合法键：%s' % sorted(allowed),
                suggest=[(did_you_mean(u, allowed) or [u])[0] for u in unknown])
        name = spec.get('name', 'wave')
        pol = spec.get('pol', 'π')
        intensity = _fnum(spec.get('intensity_mwcm2', spec.get('intensity', 100.0)),
                          'intensity_mwcm2')
        offset = _fnum(spec.get('offset_ns', 0.0), 'offset_ns')
        ell = spec.get('elliptical')
        if kind == 'arb':
            t, a = self._samples(spec)
            return {'type': 'arb', 'name': name, 't_ns': t, 'amp': a, 'pol': pol,
                    'intensity': intensity, 'elliptical': ell, 'offset_ns': offset}
        freq = spec.get('freq_mhz')
        if freq is None:
            raise InvalidValue('%s 波须给 freq_mhz（arb 无载频例外）' % kind)
        freq = _fnum(freq, 'freq_mhz')
        if kind == 'square':
            start = float(spec.get('start_ns', offset))
            edge = float(spec.get('edge_ns', 1.0))
            height = float(spec.get('height', 1.0))
            fall = spec.get('fall_start_ns')
            fall = float(fall) if fall is not None else start + 2 * edge
            env = ('square', start, edge, height, fall)
        elif kind == 'gauss':
            if spec.get('center_ns') is None or spec.get('sigma_ns') is None:
                raise InvalidValue('gauss 须给 center_ns + sigma_ns',
                                   hint='gauss env = (center_ns, sigma_ns, peak)')
            env = ('gauss', float(spec['center_ns']), float(spec['sigma_ns']),
                   float(spec.get('peak', 1.0)))
        elif kind == 'file':
            t, a = self._samples(spec)
            env = ('file', t, a)
        else:
            raise InvalidValue('kind=%r 非法' % kind, suggest=['square', 'gauss', 'file', 'arb'])
        return {'name': name, 'freq_mhz': float(freq), 'intensity': intensity,
                'pol': pol, 'elliptical': ell, 'offset_ns': offset, 'env': env,
                'drive': spec.get('drive')}

    def _f_max(self, t_ns):
        from ..core.waveforms import _f_max_mhz
        try:
            t = [float(x) for x in t_ns]
            if len(t) < 2:
                return None
            dt = min(t[i + 1] - t[i] for i in range(len(t) - 1))
            return round(float(_f_max_mhz(dt, t[-1] - t[0])), 4)
        except (ValueError, TypeError):
            return None

    def _summary(self, w):
        """§4：每波只回**摘要**（绝不内联大数组）。"""
        if w.get('type') == 'arb':
            t = w.get('t_ns') or []
            return {'name': w.get('name'), 'kind': 'arb', 'pol': w.get('pol'),
                    'intensity_mwcm2': w.get('intensity'),
                    'offset_ns': w.get('offset_ns', 0.0),
                    'N': len(t) if not _is_ptr(t) else 'ptr',
                    'span_ns': (round(float(t[-1]) - float(t[0]), 4)
                                if len(t) >= 2 and not _is_ptr(t) else None),
                    'f_max_mhz': self._f_max(t) if not _is_ptr(t) else None,
                    'has_drive': False}
        env = w.get('env') or ('?',)
        kind = env[0]
        out = {'name': w.get('name'), 'kind': kind, 'freq_mhz': w.get('freq_mhz'),
               'pol': w.get('pol'), 'intensity_mwcm2': w.get('intensity'),
               'offset_ns': w.get('offset_ns', 0.0), 'has_drive': w.get('drive') is not None}
        if kind == 'gauss':
            out['env'] = {'center_ns': env[1], 'sigma_ns': env[2], 'peak': env[3]}
        elif kind == 'square':
            out['env'] = {'start_ns': env[1], 'edge_ns': env[2], 'height': env[3],
                          'fall_start_ns': env[4] if len(env) > 4 else None}
        elif kind == 'file':
            t = env[1] if len(env) > 1 else []
            out['env'] = {'N': len(t) if not _is_ptr(t) else 'ptr',
                          'f_max_mhz': self._f_max(t) if not _is_ptr(t) else None}
        return out

    # ================= 读写（都经 choke point） =================
    def _atom_key(self, atom):
        if atom not in ('single', 'A', 'B'):
            raise InvalidValue("atom 须是 'single'/'A'/'B'（得到 %r）" % atom,
                               suggest=['single', 'A', 'B'])
        return 'dynamic.waves_%s' % atom

    def _waves(self, atom):
        return list(self.session.get(self._atom_key(atom)) or [])

    def _find(self, waves, name_or_index):
        if isinstance(name_or_index, int) and not isinstance(name_or_index, bool):
            if 0 <= name_or_index < len(waves):
                return waves[name_or_index]
            raise InvalidValue('波索引 %r 越界（共 %d 条）' % (name_or_index, len(waves)))
        for w in waves:
            if w.get('name') == name_or_index:
                return w
        raise InvalidValue('找不到波 %r' % name_or_index,
                           suggest=did_you_mean(name_or_index, [w.get('name') for w in waves]))

    def _write(self, atom, waves, actor, reason):
        rec = self.session.set(self._atom_key(atom), waves, actor=actor, reason=reason)
        return rec

    def _arb_note(self, waves, snap):
        """arb（无载频）只 hard 支持；sambe/mixed 会改道（对齐 runspec.effective_method）。"""
        if not any(w.get('type') == 'arb' for w in waves):
            return None
        meth = snap.get('dynamic.method') or 'mixed'
        if meth == 'sambe':
            return 'arb 无单一载频：sambe 会**改道 hard**（runspec.effective_method）'
        return 'arb 无单一载频：仅 hard 完整支持（mixed 走 arbitrary_mw_group）'

    # ---- L0 原始级 ----
    def set_waves(self, atom, waves, *, actor='ai', reason=None):
        """整表替换（收 serialize 同构 list）——存档恢复 / 批量导入。不要求已有基组。"""
        rec = self._write(atom, list(waves), actor, reason or 'set_waves(%s)' % atom)
        snap = self.session.effective()
        return {'ok': True, 'n_waves': len(waves), 'seq': rec['seq'],
                'waves': [self._summary(w) for w in waves],
                'arb_note': self._arb_note(waves, snap)}

    def get_waves(self, atom='single'):
        """只读：每波摘要（不内联数组，§4）+ 活跃驱动数 + 重叠告警。

        `count`/`n_waves`/`n_drives` 是**同一值的三个名**（历史别名，与 preview 一致）——
        不管你看哪个都拿得到“几条驱动”，防“文档承诺的字段实际没有”的漂移。
        """
        waves = self._waves(atom)
        n = len(waves)
        return {'atom': atom, 'count': n, 'n_waves': n, 'n_drives': n,
                'overlap_warning': detect_wave_overlap(waves),
                'waves': [self._summary(w) for w in waves]}

    def get_envelope(self, atom, name_or_index):
        """单独取一条波的包络曲线（过体积闸；大数组由 MCP 回程 gate 指针化）。"""
        w = self._find(self._waves(atom), name_or_index)
        if w.get('type') == 'arb':
            return {'name': w.get('name'), 'kind': 'arb',
                    't_ns': w.get('t_ns'), 'amp': w.get('amp')}
        env = w.get('env')
        if env and env[0] == 'file':
            return {'name': w.get('name'), 'kind': 'file', 't_ns': env[1], 'amp': env[2]}
        return {'name': w.get('name'), 'kind': env[0] if env else None,
                'note': 'gauss/square 是解析包络，无需取样；参数见 get_waves 摘要',
                'env': env}

    # ---- L1 逐波级 ----
    def add_wave(self, atom, spec, *, actor='ai', reason=None):
        """加一条波（友好 spec → serialize 同构）。同名则替换（幂等）。"""
        wd = self._build_wave(spec)
        waves = [w for w in self._waves(atom) if w.get('name') != wd.get('name')]
        waves.append(wd)
        rec = self._write(atom, waves, actor,
                          reason or 'add_wave %s' % wd.get('name'))
        return {'ok': True, 'n_waves': len(waves), 'seq': rec['seq'],
                'wave': self._summary(wd),
                'overlap_warning': detect_wave_overlap(waves),
                'arb_note': self._arb_note(waves, self.session.effective())}

    def edit_wave(self, atom, name_or_index, patch, *, actor='ai', reason=None):
        """§5.1 增量 patch：改一字段只发一个 patch（不重发整表）；其余波/字段逐位不变。"""
        if not isinstance(patch, dict):
            raise InvalidValue('patch 须是 dict（{字段: 新值}）')
        waves = self._waves(atom)
        w = self._find(waves, name_or_index)
        idx = waves.index(w)
        new = dict(w)
        for k, v in patch.items():
            if k in ('intensity_mwcm2',):
                new['intensity'] = float(v)
            elif k == 'env':
                new['env'] = v
            elif k in ('name', 'freq_mhz', 'pol', 'offset_ns', 'elliptical', 'drive', 'type'):
                new[k] = v
            else:
                raise InvalidValue('patch 键 %r 不是可直接改的波形字段' % k,
                                   hint='可改：name/freq_mhz/intensity_mwcm2/pol/offset_ns/env/drive',
                                   suggest=did_you_mean(k, ['name', 'freq_mhz', 'intensity_mwcm2',
                                                              'pol', 'offset_ns', 'env', 'drive']))
        waves[idx] = new
        rec = self._write(atom, waves, actor,
                          reason or 'edit_wave %s %s' % (new.get('name'), sorted(patch)))
        return {'ok': True, 'seq': rec['seq'], 'wave': self._summary(new),
                'patched': sorted(patch)}

    def remove_wave(self, atom, name_or_index, *, actor='ai', reason=None):
        waves = self._waves(atom)
        w = self._find(waves, name_or_index)
        waves = [x for x in waves if x is not w]
        rec = self._write(atom, waves, actor,
                          reason or 'remove_wave %s' % w.get('name'))
        return {'ok': True, 'n_waves': len(waves), 'seq': rec['seq'],
                'removed': w.get('name')}

    def clear_waves(self, atom, *, actor='ai', reason=None):
        """清空某原子的波形表（扫描前一键清，防累积）。一条 set 事件、可 undo。"""
        rec = self._write(atom, [], actor, reason or 'clear_waves(%s)' % atom)
        return {'ok': True, 'n_waves': 0, 'seq': rec['seq']}

    def set_sequence(self, atom, specs, *, actor='ai', reason=None):
        """§5.3 一次表达多波（声明式建整套，如 STIRAP/Ramsey）——**一条事件、原子写入**。"""
        waves = [self._build_wave(s) for s in specs]   # 先全建（任一非法则整体失败前不写）
        rec = self._write(atom, waves, actor, reason or 'set_sequence(%d 波)' % len(waves))
        return {'ok': True, 'n_waves': len(waves), 'seq': rec['seq'],
                'waves': [self._summary(w) for w in waves],
                'overlap_warning': detect_wave_overlap(waves),
                'arb_note': self._arb_note(waves, self.session.effective())}

    # ---- L2 意图级 ----
    def set_drive(self, atom, src, tgt, *, pol='π', rabi_mhz=None,
                  intensity_mwcm2=None, detuning_mhz=0.0, pulse=None, dur_ns=None,
                  edge_ns=1.0, height=1.0, offset_ns=0.0, name=None,
                  actor='ai', reason=None):
        """「在某跃迁上打个驱动」：给跃迁 + Rabi/强度（+ 脉冲类型/时长）→ 建方波。

        freq/intensity/duration 全由 §2 换算得出（写回物理量）；drive 记跃迁身份键
        （§2.4，不参与结算）。经 choke point 写 dynamic.waves_<atom>，一条事件、可 undo。
        """
        from ..core.atomic_levels import state_key
        states, snap = self._basis()
        gs, _ = self._resolve(states, src, '源态')
        ge, _ = self._resolve(states, tgt, '目标态')
        if rabi_mhz is None and intensity_mwcm2 is None:
            raise InvalidValue('需给 rabi_mhz（捷径）或 intensity_mwcm2（自由度）之一',
                               hint='给目标 Rabi 走 §2.2 逆换算；或直接给强度')
        bare = abs(float(ge['E_mhz']) - float(gs['E_mhz']))
        freq = bare + float(detuning_mhz)
        if intensity_mwcm2 is None:
            intensity_mwcm2 = self.intensity_for_rabi(
                state_key(gs), state_key(ge), pol, rabi_mhz)['intensity_mwcm2']
        else:
            intensity_mwcm2 = float(intensity_mwcm2)
            if rabi_mhz is None:
                rabi_mhz = self.rabi_of(state_key(gs), state_key(ge), pol,
                                        intensity_mwcm2)
        if dur_ns is None:
            if pulse == 'pi':
                dur_ns = pi_ns(rabi_mhz)
            elif pulse == 'pi2':
                dur_ns = pi2_ns(rabi_mhz)
            else:
                raise InvalidValue('需给 dur_ns 或 pulse（"pi"/"pi2"）',
                                   hint='π 脉冲时长 = 500/Ω_MHz ns（§2.3）')
        plateau = max(float(dur_ns) - 2 * float(edge_ns), 0.0)
        env = ('square', float(offset_ns), float(edge_ns), float(height),
               float(offset_ns) + plateau)
        nm = name or '%s→%s' % (gs['label'], ge['label'])
        wd = {'name': nm, 'freq_mhz': float(freq), 'intensity': float(intensity_mwcm2),
              'pol': pol, 'elliptical': None, 'offset_ns': float(offset_ns), 'env': env,
              'drive': {'g': state_key(gs), 'e': state_key(ge)}}
        waves = [w for w in self._waves(atom) if w.get('name') != nm] + [wd]
        rec = self._write(atom, waves, actor,
                          reason or 'set_drive %s (%s)' % (nm, pulse or 'dur=%sns' % dur_ns))
        return {'ok': True, 'seq': rec['seq'], 'name': nm,
                'freq_mhz': float(freq), 'bare_mhz': bare,
                'detuning_mhz': float(detuning_mhz),
                'intensity_mwcm2': float(intensity_mwcm2), 'rabi_mhz': rabi_mhz,
                'dur_ns': float(dur_ns), 'wave': self._summary(wd),
                'n_waves': len(waves), 'overlap_warning': detect_wave_overlap(waves),
                'arb_note': self._arb_note(waves, snap),
                'hint': '调 preview_behavior 看这条驱动了什么、基组够不够（D7）'}

    def pi_pulse(self, atom, src, tgt, *, rabi_mhz, pol='π', detuning_mhz=0.0,
                 edge_ns=1.0, offset_ns=0.0, name=None, actor='ai', reason=None):
        return self.set_drive(atom, src, tgt, pol=pol, rabi_mhz=rabi_mhz,
                              detuning_mhz=detuning_mhz, pulse='pi', edge_ns=edge_ns,
                              offset_ns=offset_ns, name=name, actor=actor,
                              reason=reason or 'π 脉冲')

    def pi2_pulse(self, atom, src, tgt, *, rabi_mhz, pol='π', detuning_mhz=0.0,
                  edge_ns=1.0, offset_ns=0.0, name=None, actor='ai', reason=None):
        return self.set_drive(atom, src, tgt, pol=pol, rabi_mhz=rabi_mhz,
                              detuning_mhz=detuning_mhz, pulse='pi2', edge_ns=edge_ns,
                              offset_ns=offset_ns, name=name, actor=actor,
                              reason=reason or 'π/2 脉冲')

    def square_mw(self, atom, freq_mhz, dur_ns, *, intensity_mwcm2=1.0, pol='π',
                  edge_ns=1.0, offset_ns=0.0, name='mw', actor='ai', reason=None):
        """最常见的纯微波方波（直接给频率/时长，不经跃迁解析）。"""
        spec = {'kind': 'square', 'name': name, 'freq_mhz': freq_mhz, 'pol': pol,
                'intensity_mwcm2': intensity_mwcm2, 'start_ns': offset_ns,
                'edge_ns': edge_ns, 'height': 1.0,
                'fall_start_ns': offset_ns + max(float(dur_ns) - 2 * float(edge_ns), 0.0),
                'offset_ns': offset_ns}
        return self.add_wave(atom, spec, actor=actor, reason=reason or 'square_mw')

    # ---- §5.4 协议预设 ----
    def ramsey(self, atom, src, tgt, sep_ns, *, rabi_mhz, pol='π', detuning_mhz=0.0,
               edge_ns=1.0, actor='ai', reason=None):
        """Ramsey：两个 π/2 + 自由演化间隔 sep_ns。"""
        t2 = pi2_ns(rabi_mhz)
        inten = self.intensity_for_rabi(src, tgt, pol, rabi_mhz)['intensity_mwcm2']
        specs = []
        for i, off in enumerate((0.0, t2 + float(sep_ns))):
            specs.append(self._drive_spec(atom, src, tgt, pol, inten, detuning_mhz,
                                          t2, edge_ns, off, name='ramsey_%d' % i))
        return self.set_sequence(atom, specs, actor=actor,
                                 reason=reason or 'Ramsey（sep=%sns）' % sep_ns)

    def stirap(self, atom, pump, stokes, *, rabi_mhz, sigma_ns, pump_center_ns,
               stokes_center_ns, pol='π', actor='ai', reason=None):
        """STIRAP：counter-intuitive 时序（Stokes 先于 pump），两条高斯。"""
        specs = []
        for tag, (s, t), center in (('stokes', stokes, stokes_center_ns),
                                    ('pump', pump, pump_center_ns)):
            inten = self.intensity_for_rabi(s, t, pol, rabi_mhz)['intensity_mwcm2']
            fr = self.freq_for(s, t)['freq_mhz']
            specs.append({'kind': 'gauss', 'name': tag, 'freq_mhz': fr, 'pol': pol,
                          'intensity_mwcm2': inten, 'center_ns': float(center),
                          'sigma_ns': float(sigma_ns), 'peak': 1.0,
                          'drive': self._drive_of(s, t)})
        return self.set_sequence(atom, specs, actor=actor,
                                 reason=reason or 'STIRAP（counter-intuitive）')

    def arp(self, atom, src, tgt, chirp_mhz, dur_ns, *, rabi_mhz, pol='π',
            n_samples=2001, offset_ns=0.0, actor='ai', reason=None):
        """啁啾扫频（ARP）：线性扫频的**任意波形**（arb，无单一载频 → 只 hard 支持）。

        amp(t) = cos(2π·(f0·t + (rate/2)·t²)·1e-3)（f 为 MHz、t 为 ns ⇒ 相位换算 1e-3）。
        ⚠ 载频须在采样网格可表达（Nyquist）——微波啁啾可行；光学载频不可直接采样，
        那种情形应改用 file 包络 + 引擎的载波处理，本方法**只适合微波 ARP**。
        """
        fr = self.freq_for(src, tgt)
        f0 = fr['freq_mhz']
        inten = self.intensity_for_rabi(src, tgt, pol, rabi_mhz)['intensity_mwcm2']
        t = np.linspace(float(offset_ns), float(offset_ns) + float(dur_ns), int(n_samples))
        rate = float(chirp_mhz) / float(dur_ns)                 # MHz/ns
        phase = 2 * np.pi * (f0 * (t - offset_ns) + 0.5 * rate * (t - offset_ns) ** 2) * 1e-3
        amp = np.cos(phase)
        amp = amp / (np.max(np.abs(amp)) or 1.0)                # 峰值归一（ArbitraryWave 约定）
        spec = {'kind': 'arb', 'name': 'arp', 'pol': pol, 'intensity_mwcm2': inten,
                't_ns': t.tolist(), 'amp': amp.tolist(), 'offset_ns': float(offset_ns)}
        out = self.add_wave(atom, spec, actor=actor,
                            reason=reason or 'ARP 啁啾 %s MHz/%s ns' % (chirp_mhz, dur_ns))
        out['arp_note'] = ('微波 ARP（载频可采样）；光学啁啾请改用 file 包络。arb 仅 hard 支持'
                           '（sambe 改道 hard）')
        return out

    # ---- 内部助手 ----
    def _drive_of(self, src, tgt):
        from ..core.atomic_levels import state_key
        states, _ = self._basis()
        gs, _ = self._resolve(states, src, '源态')
        ge, _ = self._resolve(states, tgt, '目标态')
        return {'g': state_key(gs), 'e': state_key(ge)}

    def _drive_spec(self, atom, src, tgt, pol, intensity, detuning_mhz, dur_ns,
                    edge_ns, offset_ns, name):
        fr = self.freq_for(src, tgt, detuning_mhz)
        plateau = max(float(dur_ns) - 2 * float(edge_ns), 0.0)
        return {'kind': 'square', 'name': name, 'freq_mhz': fr['freq_mhz'], 'pol': pol,
                'intensity_mwcm2': intensity, 'start_ns': float(offset_ns),
                'edge_ns': float(edge_ns), 'height': 1.0,
                'fall_start_ns': float(offset_ns) + plateau,
                'offset_ns': float(offset_ns), 'drive': self._drive_of(src, tgt)}
