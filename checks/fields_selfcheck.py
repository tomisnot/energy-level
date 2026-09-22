# -*- coding: utf-8 -*-
"""建场层自检（fields_selfcheck）——波形/含时场语义构造的契约（§8 九包，带致死性）。

  A 换算对拍：set_drive(detuning=0) 的 freq_mhz **逐位** == |ΔE_mhz|（裸能级差）。
     致死性：换一个目标子能级（B≠0 已 Zeeman 分裂）→ freq 必须不同（证明真按能级算）。
  B Rabi 反推：给 rabi_mhz=Ω → intensity；rabi_of 往返 == Ω；且 collect_transitions
     独立复算该波 Ω 一致（两条 core 路径互证，非自说自话）。
  C π 时长：t_π == 500/Ω_MHz（ns）、t_{π/2} == 250/Ω_MHz。
  D 校验致死：pol 非法 / gauss 缺 sigma / freq 非数值 → **set() 当场** InvalidValue
     （证明不是拖到 build_run 才炸；I6/D5）。
  E 指针化：12001 点 file 包络 → 事件/回执**无内联数组**、get_waves 只回摘要（N=12001）；
     对照小包络事件不被截（I3/I4）。
  F patch 隔离：edit_wave 改一字段，其余波/字段**逐位不变**。
  G preview 联动：无波→driven=0；set_drive 后→该跃迁 driven 且 basis_enough（D7）。
  H 不套层：构造器产出的存档 **逐位 ==** 手搓 serialize_waves 同构结果，且 restore_waves 能吃。
  I arb 改道：建 arb 波→回执提示改道 hard；与 runspec.effective_method 一致（不静默，D5）。

用临时 root；真建基组（B=21.7 Gauss 使子能级 Zeeman 分裂，供 A 的致死性）。
"""
import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

from checks.checklib import Checker                        # noqa: E402
from arc import Rubidium87                                 # noqa: E402
from src.core.static_atom import ArcCache                  # noqa: E402
from src.core import runspec as RS                         # noqa: E402
from src.core import waveforms as WF                       # noqa: E402
from src.core import static_atom as SA                     # noqa: E402
from src.core.atomic_levels import state_key               # noqa: E402
from mecha.session import Session                      # noqa: E402
from mecha.ops import Ops                              # noqa: E402
from src.adapter.fields import Fields, pi_ns, pi2_ns       # noqa: E402
from mecha.errors import InvalidValue                  # noqa: E402
from mecha import runtime                              # noqa: E402
from src.adapter import EngineAdapter                      # noqa: E402

runtime.set_engine(EngineAdapter())   # 注入领域引擎（mecha 零领域 import）

C = Checker('fields')

_TMP = tempfile.mkdtemp(prefix='fields_chk_')
_cache = ArcCache(Rubidium87())
_G = [dict(n=60, l=0, j=0.5), dict(n=60, l=1, j=1.5)]
_GLAB = [dict(n=60, l=0, j=0.5, label='60S'), dict(n=60, l=1, j=1.5, label='60P')]
_B = 21.7                                     # Zeeman 分裂子能级（供 A 致死性）
_st = RS.build_states(_cache, _G, _B, 0.0)


def _mk_session(method='mixed'):
    s = Session(root=_TMP)
    s.set('groups.list', _G, actor='setup', reason='基组')
    s.set('field.B_gauss', _B, actor='setup', reason='B')
    s.set('dynamic.method', method, actor='setup', reason='引擎')
    return s


_sess = _mk_session()
F = Fields(_sess)
_sf = F.suggest_freq('π')
S_LAB, P_LAB = _sf['src'], _sf['tgt']        # π 最强耦合对（Δm=0，保证有偶极）
_gs = next(x for x in _st if x['label'] == S_LAB)
_ge = next(x for x in _st if x['label'] == P_LAB)

# ---------------- A. 换算对拍（freq == 裸能级差） ----------------
fr = F.freq_for(S_LAB, P_LAB)
bare_manual = abs(float(_ge['E_mhz']) - float(_gs['E_mhz']))
# 致死性：换一个目标子能级（B≠0 已分裂）→ freq 必须不同
_other_p = next(x['label'] for x in _st
                if x['l'] == 1 and x['label'] != P_LAB
                and abs(x['E_mhz'] - _ge['E_mhz']) > 1e-6)
fr_other = F.freq_for(S_LAB, _other_p)
C.group('A freq_for 逐位 == 裸能级差（致死性：错能级→freq 不同）', [
    ('detuning=0 时 freq_mhz == bare_mhz', fr['freq_mhz'] == fr['bare_mhz']),
    ('freq_mhz 逐位 == 手算 |ΔE_mhz|', abs(fr['freq_mhz'] - bare_manual) == 0.0),
    ('加失谐 → freq = 裸差 + Δ', abs(F.freq_for(S_LAB, P_LAB, 5.0)['freq_mhz']
                                    - (bare_manual + 5.0)) < 1e-9),
    ('**致死性**：换目标子能级 → freq 不同（证明真按能级算，非常量）',
     abs(fr_other['freq_mhz'] - fr['freq_mhz']) > 1e-6),
])

# ---------------- B. Rabi 反推（两条 core 路径互证） ----------------
RABI = 50.0
ir = F.intensity_for_rabi(S_LAB, P_LAB, 'π', RABI)
rb = F.rabi_of(S_LAB, P_LAB, 'π', ir['intensity_mwcm2'])
# 独立路径：collect_transitions 用该强度复算 Ω
_acfg = {'enabled': True, 'intensity': ir['intensity_mwcm2'],
         'frequency_hz': fr['freq_mhz'] * 1e6, 'polarization': 'π'}
_tr = SA.collect_transitions(_cache, _GLAB, _acfg, field_B=_B)
_om_mhz = [abs(t[5]) / (2 * 3.141592653589793) / 1e6 for t in _tr]
_closest = min(_om_mhz, key=lambda x: abs(x - RABI)) if _om_mhz else -1
C.group('B Rabi↔强度往返 + collect_transitions 独立复算一致', [
    ('intensity_for_rabi 给出正强度', ir['intensity_mwcm2'] > 0),
    ('rabi_of 往返 == 目标 Ω（%.6f vs %.1f）' % (rb, RABI), abs(rb - RABI) < 1e-6),
    ('collect_transitions 独立复算 Ω ≈ 目标（%.4f vs %.1f，另一条 core 路径）'
     % (_closest, RABI), abs(_closest - RABI) < 0.5),
])

# ---------------- C. π 时长 ----------------
rp = F.set_drive('single', S_LAB, P_LAB, pol='π', rabi_mhz=RABI, pulse='pi',
                 name='pi', actor='ai', reason='π')
rp2 = F.set_drive('single', S_LAB, P_LAB, pol='π', rabi_mhz=RABI, pulse='pi2',
                  name='pi2', actor='ai', reason='π/2')
C.group('C π/π2 时长 == 500/Ω、250/Ω（§2.3）', [
    ('t_π == 500/Ω_MHz（%.4f ns）' % rp['dur_ns'], abs(rp['dur_ns'] - pi_ns(RABI)) < 1e-9),
    ('t_π/2 == 250/Ω_MHz（%.4f ns）' % rp2['dur_ns'],
     abs(rp2['dur_ns'] - pi2_ns(RABI)) < 1e-9),
    ('500/50 == 10 ns（数值锚）', abs(pi_ns(50.0) - 10.0) < 1e-12),
])

# ---------------- D. 校验致死（set 当场报错，不拖到 build_run） ----------------
def _raises(fn):
    try:
        fn(); return False
    except InvalidValue:
        return True
    except Exception:                                        # noqa: BLE001
        return 'other'


_sd = _mk_session(); Fd = Fields(_sd)
bad_pol = _raises(lambda: Fd.add_wave('single', {
    'kind': 'square', 'name': 'x', 'freq_mhz': 2e4, 'pol': 'x',
    'intensity_mwcm2': 1.0, 'start_ns': 0, 'edge_ns': 1, 'fall_start_ns': 8},
    actor='ai', reason='bad pol'))
bad_gauss = _raises(lambda: Fd.add_wave('single', {
    'kind': 'gauss', 'name': 'g', 'freq_mhz': 2e4, 'pol': 'π',
    'intensity_mwcm2': 1.0, 'center_ns': 5.0}, actor='ai', reason='gauss 缺 sigma'))
bad_freq = _raises(lambda: Fd.add_wave('single', {
    'kind': 'square', 'name': 'f', 'freq_mhz': 'abc', 'pol': 'π',
    'intensity_mwcm2': 1.0, 'start_ns': 0, 'edge_ns': 1, 'fall_start_ns': 8},
    actor='ai', reason='freq 非数值'))
C.group('D 非法波形在 set() **当场**报错（不拖到 build_run；I6/D5）', [
    ("pol='x' → InvalidValue（当场）", bad_pol is True),
    ('gauss 缺 sigma_ns → InvalidValue（当场）', bad_gauss is True),
    ('freq_mhz 非数值 → InvalidValue（当场）', bad_freq is True),
    ('被拒后快照未写入该波（自愈回退）',
     _sd.get('dynamic.waves_single') in (None, [])),
])

# ---------------- E. 指针化（I3/I4） ----------------
_se = _mk_session(); Fe = Fields(_se)
_t = np.linspace(0, 100, 12001)
_a = np.cos(2 * np.pi * _t / 50.0)
Fe.add_wave('single', {'kind': 'file', 'name': 'big', 'freq_mhz': 2e4, 'pol': 'π',
                       'intensity_mwcm2': 1.0, 't_ns': _t.tolist(),
                       'amp': _a.tolist()}, actor='ai', reason='大 file 包络')
_ev = [e for e in _se.activity(0) if e.get('target') == 'dynamic.waves_single'][-1]
_after = _ev['after']
_sum = Fe.get_waves('single')['waves'][0]
_env = _sum.get('env', {})
C.group('E 大数组走指针/摘要，不内联进事件/回执（I3/I4）', [
    ('set 事件的 after 未内联 12001 点（被截断/指针化）',
     isinstance(_after, dict) and (_after.get('$truncated') is True
                                    or _after.get('$pointer') is True)),
    ('截断告知 kept<total + where（I4）',
     (not isinstance(_after, dict)) or _after.get('$truncated') is not True
     or (_after.get('total', 0) > _after.get('kept', 0) and 'where' in _after)),
    ('get_waves 只回摘要（file 波 N=12001 在 env，无 t_ns/amp 数组）',
     _env.get('N') == 12001 and 't_ns' not in _sum and 'amp' not in _sum
     and 't_ns' not in _env and 'amp' not in _env),
    ('摘要带带宽预检 f_max_mhz（D7）', _env.get('f_max_mhz') is not None),
])

# ---------------- F. patch 隔离 ----------------
_sf2 = _mk_session(); Ff = Fields(_sf2)
Ff.set_drive('single', S_LAB, P_LAB, pol='π', rabi_mhz=50.0, pulse='pi',
             name='w1', actor='ai', reason='w1')
Ff.square_mw('single', 2e4, 30.0, name='w2', intensity_mwcm2=2.0,
             actor='ai', reason='w2')
before = [dict(w) for w in _sf2.get('dynamic.waves_single')]
Ff.edit_wave('single', 'w2', {'intensity_mwcm2': 9.0}, actor='ai', reason='改 w2 强度')
after = _sf2.get('dynamic.waves_single')
w1_unchanged = after[0] == before[0]
w2 = next(w for w in after if w['name'] == 'w2')
w2b = next(w for w in before if w['name'] == 'w2')
C.group('F edit_wave patch 隔离：只改的字段变，其余逐位不变', [
    ('w2 的 intensity 改成 9.0', w2['intensity'] == 9.0),
    ('w2 的其它字段（freq/pol/env）逐位不变',
     all(w2[k] == w2b[k] for k in ('freq_mhz', 'pol', 'env', 'offset_ns'))),
    ('另一条波 w1 **逐位不变**', w1_unchanged),
    ('波数不变（edit 不增删）', len(after) == len(before) == 2),
])

# ---------------- G. preview 联动（D7） ----------------
_sp = _mk_session(); Fp = Fields(_sp); Op = Ops(_sp)
drv0 = len(Op.preview().get('driven', []))
Fp.set_drive('single', S_LAB, P_LAB, pol='π', rabi_mhz=50.0, pulse='pi',
             name='pv', actor='ai', reason='建驱动')
pv = Op.preview()
C.group('G preview 联动：建场前 driven=0，建场后报该跃迁 driven（D7）', [
    ('无波时 preview driven=0', drv0 == 0),
    ('set_drive 后 preview driven>=1', len(pv.get('driven', [])) >= 1),
    ('preview 报 basis_enough（基组够）', pv.get('basis_enough') is True),
    ('preview 与建场同源（driven 的 tgt 是 60P 组）',
     any('60P' in str(d.get('tgt')) for d in pv.get('driven', []))),
])

# ---------------- H. 不套层（逐位 == serialize_waves） ----------------
_sh = _mk_session(); Fh = Fields(_sh)
Fh.set_drive('single', S_LAB, P_LAB, pol='π', rabi_mhz=RABI, pulse='pi',
             edge_ns=1.0, offset_ns=0.0, name='iso', actor='ai', reason='不套层对拍')
got = _sh.get('dynamic.waves_single')[0]
_inten = Fh.intensity_for_rabi(S_LAB, P_LAB, 'π', RABI)['intensity_mwcm2']
_fr = Fh.freq_for(S_LAB, P_LAB)['freq_mhz']
_hand = WF.Wave('iso', freq_mhz=_fr, envelope=WF.SquareEnv(0.0, 1.0, 1.0, 8.0),
                pol='π', offset_ns=0.0, intensity_mwcm2=_inten,
                drive={'g': state_key(_gs), 'e': state_key(_ge)})
_ser = WF.serialize_waves([_hand])[0]
_restored = WF.restore_waves(_sh.get('dynamic.waves_single'))
C.group('H 不套层：构造器存档逐位 == 手搓 serialize_waves（§13/§9）', [
    ('键集合相同', set(got) == set(_ser)),
    ('逐字段逐位相同（无并行表示）', got == _ser),
    ('restore_waves 能吃构造器产出（同一数据模型）',
     len(_restored) == 1 and abs(_restored[0].freq_mhz - _fr) < 1e-9),
    ('drive 往返保留（身份键）', _restored[0].drive is not None),
])

# ---------------- I. arb 改道（不静默，D5） ----------------
_sa = _mk_session(method='sambe'); Fa = Fields(_sa)
ra = Fa.add_wave('single', {'kind': 'arb', 'name': 'chirp', 'pol': 'π',
                            'intensity_mwcm2': 1.0,
                            't_ns': np.linspace(0, 50, 501).tolist(),
                            'amp': np.cos(np.linspace(0, 20, 501)).tolist()},
                 actor='ai', reason='arb 波')
_restored_arb = WF.restore_waves(_sa.get('dynamic.waves_single'))
_meth, _note = RS.effective_method('sambe', [_restored_arb])
C.group('I arb 改道 hard：回执主动提示，与 runspec.effective_method 一致（D5）', [
    ('建 arb 波回执带改道提示（含 hard）',
     ra.get('arb_note') and 'hard' in ra['arb_note']),
    ('runspec.effective_method(sambe, [arb]) 确实改道 hard', _meth == 'hard'),
    ('提示与引擎实际改道一致（不静默）', _meth == 'hard' and 'hard' in ra['arb_note']),
])

# ---------------- J. AI 面接线（MCP 工具 + CodeAct 门面） ----------------
import json                                                 # noqa: E402
from mecha import server as SVR                          # noqa: E402
from mecha import codeact as CA                          # noqa: E402

_sj = _mk_session(); _fj = Fields(_sj); _oj = Ops(_sj)
_srv, _tools = SVR.create_server(_sj, _oj, fields=_fj, actor='ai-test',
                                 log_level='ERROR')


def _call(n, **kw):
    return json.loads(_tools[n](**kw))


_sd = _call('set_drive', reason='MCP set_drive', src=S_LAB, tgt=P_LAB,
            rabi_mhz=50.0, pulse='pi', name='mcp_pi')
_gw = _call('get_waves', atom='single')
_ca = CA.run_codeact(
    "r = fields.pi_pulse('single', %r, %r, rabi_mhz=50.0, name='ca_pi')\n"
    "result = {'n': len(fields.get_waves('single')['waves']), 'freq': r['freq_mhz']}"
    % (S_LAB, P_LAB), _sj, _oj, fields=_fj)
C.group('J AI 面接线：MCP set_drive/get_waves + CodeAct fields 门面（功能真够得到）', [
    ('MCP set_drive 工具建场成功（freq>0）',
     _sd.get('ok') is True and _sd.get('freq_mhz', 0) > 0),
    ('MCP get_waves 工具读回摘要', _gw.get('count', 0) >= 1),
    ('CodeAct fields 门面可 pi_pulse + get_waves',
     _ca.get('ok') is True and _ca.get('result', {}).get('n', 0) >= 1),
    ('CodeAct 建的场归因 actor=codeact（M1）',
     any(e['actor'] == 'codeact' for e in _sj.activity(0))),
])

# ---------------- K. 重叠驱动告警 + clear_waves + spec 键校验（P0） ----------------
_sk = _mk_session(); Fk = Fields(_sk)
Fk.set_drive('single', S_LAB, P_LAB, pol='π', rabi_mhz=50.0, pulse='pi',
             name='sw1', actor='ai', reason='扫描点1')
_r2 = Fk.set_drive('single', S_LAB, P_LAB, pol='π', rabi_mhz=55.0, pulse='pi',
                   name='sw2', actor='ai', reason='扫描点2')
_ovl = _r2['overlap_warning']
_cl = Fk.clear_waves('single', actor='ai', reason='扫描前清')
_after_clear = Fk.get_waves('single')
# 不误报：Ramsey（同跃迁但时间不重叠）
_rr = _mk_session(); Fr = Fields(_rr)
_rm = Fr.ramsey('single', S_LAB, P_LAB, sep_ns=200.0, rabi_mhz=50.0,
                actor='ai', reason='ramsey')
# spec 未知键（startns 是 start_ns 的 typo）→ 不静默忽略
_sk2 = _mk_session(); Fk2 = Fields(_sk2)
_spec_rej = _raises(lambda: Fk2.add_wave('single', {
    'kind': 'square', 'name': 'x', 'freq_mhz': 2e4, 'pol': 'π',
    'intensity_mwcm2': 1.0, 'startns': 0, 'edge_ns': 1, 'fall_start_ns': 8},
    actor='ai', reason='typo key'))
# 字段一致性（防“文档承诺 n_drives、实现只有 n_waves”这类漂移）
_pk = _mk_session(); Fp2 = Fields(_pk); Op2 = Ops(_pk)
Fp2.set_drive('single', S_LAB, P_LAB, pol='π', rabi_mhz=50.0, pulse='pi',
              name='d1', actor='ai', reason='d1')
_gw = Fp2.get_waves('single')
_pvk = Op2.preview()
C.group('K 重叠驱动大声告警（P0，防静默给错答）+ clear_waves + spec 键校验', [
    ('**致死性**：同跃迁时间重叠的两条驱动 → overlap_warning 非空',
     _ovl is not None and _ovl.get('n_waves') == 2),
    ('告警文案点名“扫描忘了清空”', bool(_ovl) and '清' in _ovl['warning']),
    ('clear_waves 清空后 n_waves=0、无告警',
     _cl['n_waves'] == 0 and _after_clear['overlap_warning'] is None),
    ('**不误报**：Ramsey（同跃迁但时间不重叠）无告警',
     _rm.get('overlap_warning') is None and _rm['n_waves'] == 2),
    ('spec 未知键（startns）→ InvalidValue（不静默忽略→用错默认）',
     _spec_rej is True),
    ('get_waves 与 preview 计数字段一致（n_waves==n_drives，两处都有，防漂移）',
     _gw.get('n_waves') == _gw.get('n_drives') == _pvk.get('n_waves')
     == _pvk.get('n_drives') == 1),
])

sys.exit(0 if C.summary() else 1)
