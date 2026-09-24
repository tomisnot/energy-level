# -*- coding: utf-8 -*-
"""驱动手感物理守卫（v2 drive_wave_selfcheck）——接替 v1 fields_selfcheck 的换算契约。

v1 的 Fields 门面（src/adapter/fields.py）随 adapter 退役；驱动换算的真身现在
在 `ElEngine.drive_wave` + 工具层 `set_drive`（ADR 2026-09-23-EL驱动手感）。本判据
经**真门**（V2Bridge + build_tool_registry 的 set_drive 工具）验换算物理不漂移：

  A 换算对拍：set_drive(detuning=Δ) 的 freq_mhz == |ΔE_mhz| + Δ（逐位）；
     致死性：换一个目标子能级（B≠0 已 Zeeman 分裂）→ freq 必须不同。
  B Rabi↔光强往返：给 rabi_mhz=Ω → 落库 intensity；用 SA 独立复算 Ω' == Ω
     （两条路径互证：drive_wave 的换算 vs static_atom 的正向公式）。
  C π 时长：env 平台段 == 500/Ω（π）/ 250/Ω（π2）ns。
  D 非法波形**写门当场**被拒（validate_wave_list 经 Gate）：pol 非法 / square
     env 缺第 5 元素 / freq 非数值；被拒后快照未写入（不拖到 build_run 才炸）。
  E append/replace 语义：默认追加、replace 整组替换（扫描忘清空的 P0 坑的机制面）。
  F get_waves 摘要不内联大数组：file 包络（12001 点）经查询面只回形状摘要，
     回执里无 t_ns/amp（I3 大载荷不内联）。

真建基组（B=21.7 Gauss 使子能级 Zeeman 分裂，供 A 的致死性）；临时数据根。
"""
import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

from checks.checklib import Checker                        # noqa: E402
from src.core import static_atom as SA                     # noqa: E402
from src.core import atomic_levels as AL                   # noqa: E402
from mecha.authority import Mode                           # noqa: E402
from mecha.errors import GateDenied, MechaError            # noqa: E402
from mecha.gate import Channel                             # noqa: E402

from examples.energy_level.bridge import V2Bridge, _build_gui_tools  # noqa: E402

EL_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
C = Checker('drive_wave')

_TMP = tempfile.mkdtemp(prefix='drivewave_')
br = V2Bridge.for_gui(el_root=EL_ROOT, data_root=_TMP)
br.write_many({'groups.list': [dict(n=60, l=0, j=0.5), dict(n=60, l=1, j=1.5)],
               'field.B_gauss': 21.7, 'field.E_vpcm': 0.0,
               'dynamic.mode': 0, 'dynamic.method': 'mixed',
               'dynamic.t_max': 50.0, 'dynamic.npts': 51,
               'dynamic.decay_index': 0},
              reason='判据基组', actor='gui')

# 工具面（AI 通道 + AI 模式——set_drive 走真门）
ai = Channel("ai")
tools = _build_gui_tools(br._engine, br._gate, br._authority, ai, br.jobs)
br.switch_mode(Mode.AI, actor='human')


def _call(tool, **kw):
    r = tools.execute(tool, kw)
    if r.get('is_error'):
        raise AssertionError('工具 %s 失败：%s' % (tool, r.get('error')))
    return r['value']


# 态与身份键（π 最强耦合对：Δm=0 且偶极非零）
_lv = br._engine.list_levels(br.effective())


def _as_state(s):
    """list_levels 行 → dipole_amplitude 要的 mJ 态（kind/n/l/j/mj）。"""
    return {'kind': 'mJ', 'n': s['n'], 'l': s['l'], 'j': s['j'], 'mj': s['m']}


_S = [s for s in _lv['states'] if s['kind'] == 'mJ' and s['l'] == 0]   # 60S
_P_STATES = [s for s in _lv['states'] if s['kind'] == 'mJ' and s['l'] == 1]   # 60P
_src, _tgt = None, None
for _a in _S:
    for _b in _P_STATES:
        if abs(_a['m'] - _b['m']) < 1e-9:                    # Δm=0（π）
            _d = abs(SA.dipole_amplitude(br._engine._cache,
                                         _as_state(_a), _as_state(_b), 0))
            if _d > 0:
                _src, _tgt = _a, _b
                break
    if _src:
        break
assert _src and _tgt, '找不到 π 耦合对'
SRC_K, TGT_K = _src['key'], _tgt['key']
_D = abs(SA.dipole_amplitude(br._engine._cache, _as_state(_src),
                             _as_state(_tgt), 0))

# ---------------- A. 换算对拍（freq == 裸能级差 + 失谐） ----------------
w0 = _call('set_drive', src=SRC_K, tgt=TGT_K, rabi_mhz=50.0, pulse='pi',
           mode='replace')['wave']
# list_levels 的 E_mhz 按 6 位小数舍入（协议层展示口径）；drive_wave 用未舍入
# 态能 ⇒ 对拍容差取舍入误差界（2×5e-7）。
bare = abs(float(_tgt['E_mhz']) - float(_src['E_mhz']))
_alt = next((s for s in _P_STATES if s is not _tgt
             and abs(s['E_mhz'] - _tgt['E_mhz']) > 1e-6), None)
C.group('A set_drive 频率 == |ΔE_mhz| + 失谐（致死性：换能级→freq 不同）', [
    ('detuning=0 时 freq == 手算裸能级差（舍入容差内）',
     abs(w0['freq_mhz'] - bare) < 2e-6),
    ('加失谐 Δ=5 → freq == 裸差 + 5',
     abs(_call('set_drive', src=SRC_K, tgt=TGT_K, rabi_mhz=50.0,
               pulse='pi', detuning_mhz=5.0, mode='replace')['wave']['freq_mhz']
         - (bare + 5.0)) < 2e-6),
    ('**致死性**：换目标子能级 → freq 不同（真按能级算，非常量）',
     _alt is not None and abs(
         _call('set_drive', src=SRC_K, tgt=_alt['key'], intensity_mwcm2=1.0,
               duration_ns=10.0, mode='replace')['wave']['freq_mhz']
         - w0['freq_mhz']) > 1e-6),
])

# ---------------- B. Rabi↔光强往返（SA 独立复算） ----------------
RABI = 50.0
wd = _call('set_drive', src=SRC_K, tgt=TGT_K, rabi_mhz=RABI, pulse='pi',
           mode='replace')['wave']
# 独立路径：从落库光强用 static_atom 正向公式复算 Ω
_om_back = SA.rabi_mhz(_D, SA.field_e0_sq(wd['intensity']), 1.0)
# 反向入口：给 intensity 应得同一 Ω 的波
wi = _call('set_drive', src=SRC_K, tgt=TGT_K, intensity_mwcm2=wd['intensity'],
           duration_ns=10.0, mode='replace')['wave']
C.group('B Rabi↔光强往返 + SA 独立复算一致（两条路径互证）', [
    ('落库 intensity > 0', wd['intensity'] > 0),
    ('SA 复算 Ω 往返 == 目标（%.6f vs %.1f）' % (_om_back, RABI),
     abs(_om_back - RABI) < 1e-6),
    ('intensity 入口与 rabi 入口落库逐位一致', wi['intensity'] == wd['intensity']),
])

# ---------------- C. π/π2 时长 == 500/Ω、250/Ω ----------------
wpi = _call('set_drive', src=SRC_K, tgt=TGT_K, rabi_mhz=RABI, pulse='pi',
            mode='replace')['wave']
wpi2 = _call('set_drive', src=SRC_K, tgt=TGT_K, rabi_mhz=RABI, pulse='pi2',
             mode='replace')['wave']
C.group('C π/π2 平台时长 == 500/Ω、250/Ω ns', [
    ('t_π == 500/Ω（env 第 5 元素）', abs(wpi['env'][4] - 500.0 / RABI) < 1e-9),
    ('t_π/2 == 250/Ω', abs(wpi2['env'][4] - 250.0 / RABI) < 1e-9),
    ('数值锚：Ω=50 → t_π=10 ns', abs(wpi['env'][4] - 10.0) < 1e-12),
])

# ---------------- D. 非法波形写门当场被拒（不拖到 build_run） ----------------
def _gate_rejects(wave):
    try:
        br.write('dynamic.waves_single', [wave], reason='判据非法波', actor='ai')
        return False
    except GateDenied:
        return True


_bad_pol = _gate_rejects({'name': 'x', 'freq_mhz': 2e4, 'intensity': 1.0,
                          'pol': 'x', 'offset_ns': 0.0,
                          'env': ['square', 0.0, 1.0, 1.0, 8.0]})
_bad_env = _gate_rejects({'name': 'y', 'freq_mhz': 2e4, 'intensity': 1.0,
                          'pol': 'π', 'offset_ns': 0.0,
                          'env': ['square', 0.0, 1.0]})       # 元素不足（须 ≥4）
_bad_freq = _gate_rejects({'name': 'z', 'freq_mhz': 'abc', 'intensity': 1.0,
                           'pol': 'π', 'offset_ns': 0.0,
                           'env': ['square', 0.0, 1.0, 1.0, 8.0]})
C.group('D 非法波形在 Gate **当场**被拒（validate_wave_list；I6/D5）', [
    ("pol='x' → 当场拒", _bad_pol),
    ('square env 缺第 5 元素 → 当场拒', _bad_env),
    ('freq_mhz 非数值 → 当场拒', _bad_freq),
    ('被拒后快照未写入非法波（一字节没动）',
     all(w.get('name') not in ('x', 'y', 'z')
         for w in (br.read('dynamic.waves_single') or []))),
])

# ---------------- E. append / replace 语义 ----------------
_call('set_drive', src=SRC_K, tgt=TGT_K, rabi_mhz=50.0, pulse='pi',
      mode='replace', name='e1')
r_app = _call('set_drive', src=SRC_K, tgt=TGT_K, rabi_mhz=55.0, pulse='pi',
              name='e2')
r_rep = _call('set_drive', src=SRC_K, tgt=TGT_K, rabi_mhz=60.0, pulse='pi',
              mode='replace', name='e3')
C.group('E append（默认）/ replace（整组替换）语义（扫描 P0 坑的机制面）', [
    ('append 后 n_waves=2', r_app['n_waves'] == 2),
    ('replace 后 n_waves=1 且只剩 e3',
     r_rep['n_waves'] == 1 and [w['name'] for w in
                                br.read('dynamic.waves_single')] == ['e3']),
])

# ---------------- F. get_waves 摘要不内联大数组（I3） ----------------
_t = np.linspace(0, 100, 12001)
_a = np.cos(2 * np.pi * _t / 50.0)
br.write('dynamic.waves_single',
         [{'name': 'big', 'freq_mhz': 2e4, 'intensity': 1.0, 'pol': 'π',
           'offset_ns': 0.0, 'env': ['file', _t.tolist(), _a.tolist()]}],
         reason='判据大 file 包络', actor='ai')
gw = tools.execute('run_actions', {'actions': [{'op': 'get_waves',
                                                'args': {'atom': 'single'}}]})
assert not gw.get('is_error'), gw
_sum = gw['value']['outputs'][0]['waves'][0]
C.group('F get_waves 摘要不内联大数组（file 包络只回形状，I3）', [
    ('摘要 envelope.kind=file', _sum['envelope']['kind'] == 'file'),
    ('摘要无 t_ns/amp 大数组', 't_ns' not in _sum and 'amp' not in _sum
     and 't_ns' not in _sum['envelope'] and 'amp' not in _sum['envelope']),
    ('摘要带 name/freq/强度（可概览）',
     _sum['name'] == 'big' and _sum['freq_mhz'] == 2e4
     and _sum['intensity_mwcm2'] == 1.0),
])

br.close()
sys.exit(0 if C.summary() else 1)
