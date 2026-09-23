# -*- coding: utf-8 -*-
"""静态能力 AI 入口自检（P1-6，2026-09-22）：`static.columns` 三档 + 三个静态查询。

## 这条判据守什么

用户裁决："静态引擎 AI 调用不了是**缺口**：想让 AI 也能算静态量（推荐）"。
本判据守**同源**（AI 拿到的静态量必须与 GUI 静态页/动态基组**同一实现**、逐位一致）、
**opt-in**（贵的列绝不默认开）、**大结果做成数组**（不是大 list——库的体积闸对数组
无损转指针，对大 list 只会截断丢数据）、以及**两个静默点已消**。

## 致死性（每条都验过"改回去必红"）

- A2/A3：`dc_shift_mhz` 改成恒 0 / `E_mhz` 改成另算一遍 ⇒ 红；
- A5：`ac_shift_mhz` 与直调 `build_level_rows_direct` 逐位不一致 ⇒ 红；
- A6：把 `static.columns` 默认改成 `basic`/`ac`（即默认开）⇒ 红（**成本回归**：
  AC 档 ~13 s/次，见 `src/ui/fields.py:82`）；
- B2：`arrays` 改回大 list（不是 ndarray）⇒ 红（那会被库截断丢数据）；
- C2：把 `shifts` 改回 `dict[tuple→ndarray]` ⇒ 红（**tuple 键无法 JSON 化**）；
- D2：去掉 `PAIR_EIGEN_MAX_DIM` 闸 ⇒ `n_window=3` 会去分配 4 GB eigh（红/挂）；
- D4：去掉 `_exchange_domain_ok` 前置判据 ⇒ 非里德堡交换对**静默全零**，红；
- E：把静态查询接进 `runspec.py` ⇒ 红（静态量不得渗进动态）。

用法：python checks\\static_ai_selfcheck.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

from checks.checklib import Checker                            # noqa: E402
from src.core import protocol as P                             # noqa: E402
from src.core import static_atom as SA                         # noqa: E402
from src.core import static_pair as SP                         # noqa: E402

C = Checker('static_ai')
cache = P.make_cache()

# 便宜场景：n_photon=1 / window=1（AC 档才 ~1-2 s；GUI 默认 4/3 是 ~13 s/N~400）
# ⚠ 两体几何用 **registry 的三个扁平键**（`dynamic.pair.R_um` 等）——与
# `runspec.py:555-557` 同一真源。旧版这里喂 `'dynamic.pair'` 字典键：它**不在
# registry 里**（set_param 会 UnknownKey 拒）⇒ 只有本判据这种"直调 core 的合成
# 快照"能喂进去，MCP/AI 路径恒 R=0 必报错（2026-09-22 修 core 时同步改这里）。
BASE = {
    'groups.list': [dict(n=60, l=0, j=0.5, label='G1', color='#f00'),
                    dict(n=60, l=1, j=1.5, label='G2')],
    'field.B_gauss': 21.7, 'field.E_vpcm': 1.0,
    'field.n_photon': 1, 'field.window': 1,
    'dynamic.waves_single': [{'name': '寻址光', 'freq_mhz': 295.7237e6,
                              'intensity': 1e3, 'pol': 'σ-', 'offset_ns': 0.0,
                              'env': ['gauss', 0.0, 1e9, 1.0]}],
    'dynamic.pair.R_um': 8.0, 'dynamic.pair.theta_deg': 30.0,
    'dynamic.pair.phi_deg': 0.0,
}
def snap(**kw):
    return dict(BASE, **kw)


# ================= A. static.columns 三档 =================
off = P.list_levels(cache, snap())
ba = P.list_levels(cache, snap(**{'static.columns': 'basic'}))
ac = P.list_levels(cache, snap(**{'static.columns': 'ac'}))

_o = off['states'][0]
_b = ba['states'][0]
_a = ac['states'][0]
# dc 列自洽：dc == E(Ev) − E(0)
_noE = P.list_levels(cache, snap(**{'field.E_vpcm': 0.0,
                                    'static.columns': 'basic'}))
_dc_err = max(abs(x['dc_shift_mhz'] - (x['E_mhz'] - y['E_mhz']))
              for x, y in zip(ba['states'], _noE['states']))
# ac 列与直调逐位一致
_acfg = P.address_cfg_from_snapshot(BASE)[0]
_rows = SA.build_level_rows_direct(cache, 21.7, BASE['groups.list'], _acfg,
                                   BASE['groups.list'], 1.0, 1, 1)
_ac_same = all(abs(d['ac_shift_mhz'] - round(float(r['ac']), 6)) < 1e-12
               and d['ac_affected'] == bool(r['affected'])
               for d, r in zip(ac['states'], _rows))

C.group('A static.columns 三档：opt-in、同源、便宜档真便宜', [
    ('off 档不含静态列（键集与改前同构）',
     not any(k in _o for k in ('dc_shift_mhz', 'ac_shift_mhz', 'qstr', 'amps'))),
    ('off 档给 static_hint（可发现性）', 'static_hint' in off),
    ('basic 档含 6 个静态列',
     all(k in _b for k in ('E0_mhz', 'dc_shift_mhz', 'qstr', 'group_label',
                           'color', 'amps'))),
    (f'dc_shift_mhz 自洽（== E(Ev)−E(0)，最大偏差 {_dc_err:.2e}）', _dc_err < 1e-6),
    ('dc 列非恒 0（场景确实有 DC 位移）', abs(float(_b['dc_shift_mhz'])) > 1e-3),
    ('三档的 E_mhz 逐位一致（静态列不改能量基准）',
     all(x['E_mhz'] == y['E_mhz'] == z['E_mhz']
         for x, y, z in zip(off['states'], ba['states'], ac['states']))),
    (f'ac 列与直调 build_level_rows_direct 逐位一致（{_ac_same}）', _ac_same),
    ('ac 档回 cost_note（明写代价，D5）',
     'cost_note' in ac and '13' in ac['cost_note']),
    ('basic 档 cost_note 明写"无额外代价"',
     '无额外代价' in ba.get('cost_note', '')),
    ('★ 默认 off（贵的列绝不默认开——AC 档 ~13 s/次）',
     P.config_schema()['static.columns']['default'] == 'off'),
])

# 无寻址光 ⇒ ac 列 None + note（不静默给 0）
_no = P.list_levels(cache, snap(**{'static.columns': 'ac',
                                   'dynamic.waves_single': []}))
C.group('A 无寻址光：ac 列必须 None + note（不静默给 0）', [
    ('ac_shift_mhz is None', _no['states'][0]['ac_shift_mhz'] is None),
    ('ac_note 说明原因', 'ac_note' in _no and '寻址光' in _no['ac_note']),
])

# ================= B. static_spectrum =================
sp = P.static_spectrum(cache, snap(), top_n=3, full=True)
_direct = sorted(SA.spectrum_lines(cache, _rows, True,
                                   SA.address_pol_coeffs(_acfg)),
                 key=lambda t: -float(t[1]))
_same = all(abs(a['freq_hz'] - d[0]) < 1e-12 and abs(a['A_s1'] - d[1]) < 1e-18
            for a, d in zip(sp['lines'], _direct))
C.group('B static_spectrum：与 GUI 同源 + 大结果做成数组（不是大 list）', [
    (f'n_lines={sp["n_lines"]} 且 top 非空', sp['n_lines'] > 0 and bool(sp['lines'])),
    (f'与直调 spectrum_lines 逐位一致（{_same}）', _same),
    ('top 按 A 降序', all(sp['lines'][i]['A_s1'] >= sp['lines'][i + 1]['A_s1']
                          for i in range(len(sp['lines']) - 1))),
    ('★ full=True 的 arrays 是 ndarray 列数组（库可无损转指针）',
     isinstance(sp.get('arrays'), dict)
     and all(isinstance(v, np.ndarray) for v in sp['arrays'].values())),
    ('truncated 标记存在', 'truncated' in sp),
])

# ================= C. static_ac_sweep =================
sw = P.static_ac_sweep(cache, snap(), [1.0, 10.0, 100.0])
C.group('C static_ac_sweep：tuple 键已转形（否则无法 JSON 化）', [
    (f'shifts_mhz 是 2D ndarray，形状 {getattr(sw["shifts_mhz"], "shape", None)}',
     isinstance(sw['shifts_mhz'], np.ndarray) and sw['shifts_mhz'].ndim == 2),
    ('形状 = (n_states, n_intensities)',
     sw['shifts_mhz'].shape == (sw['n_states'], 3)),
    ('keys 是 list[dict]（不是 tuple 键）',
     isinstance(sw['keys'], list) and isinstance(sw['keys'][0], dict)
     and {'n', 'l', 'j', 'qstr', 'occ'} <= set(sw['keys'][0])),
    ('intensities 原样回传', sw['intensities_mwcm2'] == [1.0, 10.0, 100.0]),
])

# ================= D. static_pair =================
pr = P.static_pair(cache, snap())                       # 默认 = exchange（便宜档）
_j_direct = SP.exchange_coupling(cache, BASE['groups.list'], 8.0, 30.0, 0.0,
                                 B_gauss=21.7, n_window=1)
_guard = None
try:
    P.static_pair(cache, snap(), what='evals', n_window=3)
except ValueError as e:
    _guard = str(e)
_nr = dict(snap(**{'groups.list': [dict(n=6, l=1, j=1.5, label='6P'),
                                   dict(n=60, l=0, j=0.5, label='60S')]}))
# ⚠ 6P 是**非里德堡** ⇒ 基组里是 `kind='F'`（超精细），4 元组第 4 位是 **mF（整数）**
# 而不是 mJ ⇒ 必须给 mF=2.0（半整数 0.5 不是合法 mF）。这正是
# `static_atom.dipole_amplitude` 记的 "F vs mJ 陷阱"。
_mp = P.static_pair(cache, _nr, what='map', R_grid=[8.0], theta_grid=[0.0],
                    trans1=((6, 1, 1.5, 2.0), (60, 0, 0.5, 0.5)),
                    trans2=((60, 0, 0.5, 0.5), (6, 1, 1.5, 2.0)))
# 半整数 m 指向 F 态 ⇒ 找不到态 ⇒ 必须**明确报错**（这是 AI 最容易踩的坑）
_fvsmj = None
try:
    P.static_pair(cache, _nr, what='map', R_grid=[8.0], theta_grid=[0.0],
                  trans1=((6, 1, 1.5, 0.5), (60, 0, 0.5, 0.5)),
                  trans2=((60, 0, 0.5, 0.5), (6, 1, 1.5, 0.5)))
except ValueError as e:
    _fvsmj = str(e)
_miss = None
try:
    P.static_pair(cache, snap(), what='map', R_grid=[8.0], theta_grid=[0.0],
                  trans1=((60, 0, 0.5, 0.5), (99, 0, 0.5, 0.5)),
                  trans2=((60, 1, 1.5, 0.5), (60, 0, 0.5, 0.5)))
except ValueError as e:
    _miss = str(e)
C.group('D static_pair：默认便宜档 + 维度闸 + 两个静默点已消', [
    ('默认 what=exchange（不触发 eigh）', 'exchange' in pr and 'evals' not in pr),
    (f'J 与直调 exchange_coupling 同值（{pr["exchange"]["J_mhz"]:.6f}）',
     abs(pr['exchange']['J_mhz'] - float(_j_direct)) < 1e-12),
    ('★ 默认 n_window=1（nw=3 时 pair 维 15876 ⇒ eigh 4 GB，不可行）',
     pr['n_window'] == 1),
    (f'★ 维度闸在 nw=3 时早失败并点名代价（{(_guard or "没报错！")[:34]}…）',
     _guard is not None and '不可行' in _guard and 'GB' in _guard),
    ('★ 非里德堡交换对：exchange_valid=False + note（不是静默全零）',
     _mp['map'].get('exchange_valid') is False and 'note' in _mp['map']),
    ('★ 基组缺源/目标态：明确报错（不是静默 None）',
     _miss is not None and '缺少' in _miss),
    ('★ F vs mJ 陷阱：半整数 m 指向 F 态 ⇒ 明确报错（不静默返 None）',
     _fvsmj is not None and '缺少' in _fvsmj),
])

# ================= E. 静态不得渗进动态（源码级反向断言） =================
_src = open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                         'src', 'core', 'runspec.py'), encoding='utf-8').read()
C.group('E 静态查询不得渗进动态（与 P1-7 同款反向断言）', [
    # 断言"**调用**"而不是名字：`runspec.py` 本来就有
    # `from . import static_pair as pair_core`（模块导入，合法）⇒ 只查 `名字(`
    (f'runspec.py 不调用 {n}(', (n + '(') not in _src)
    for n in ('static_spectrum', 'static_ac_sweep', 'static_pair')
] + [('static.columns 也不在 runspec.py（静态列只影响查询）',
      'static.columns' not in _src)])

# ================= F. 可教学错误 =================
# ⚠ 契约差异（有意）：`list_levels` 的空基组沿用**既有契约**（回 note、不抛），
# 而**新查询**（static_spectrum/static_pair/static_ac_sweep）**抛 ValueError**——
# 一个"算不出任何东西"的查询不该回一个看起来正常的结果。
_errs = {}
try:
    _e = P.list_levels(cache, snap(**{'groups.list': []}))
    _errs['空基组(list_levels)'] = _e.get('note')
except Exception as e:                                           # noqa: BLE001
    _errs['空基组(list_levels)'] = None
for tag, fn in (('空基组(static_spectrum)',
                 lambda: P.static_spectrum(cache, snap(**{'groups.list': []}))),
                ('非法 static.columns',
                 lambda: P.list_levels(cache, snap(**{'static.columns': 'xx'}))),
                ('两体 R=0', lambda: P.static_pair(
                    cache, snap(**{'dynamic.pair.R_um': 0.0})))):
    try:
        fn()
        _errs[tag] = None
    except Exception as e:                                       # noqa: BLE001
        _errs[tag] = str(e)
C.group('F 可教学错误：点名病因与解法（不静默）', [
    (f'list_levels 空基组沿用既有契约（note 含 groups.list：'
     f'{( _errs["空基组(list_levels)"] or "")[:20]}…）',
     bool(_errs['空基组(list_levels)'])
     and 'groups.list' in _errs['空基组(list_levels)']),
    (f'★ 新查询空基组**抛错**（{(_errs["空基组(static_spectrum)"] or "")[:20]}…）',
     bool(_errs['空基组(static_spectrum)'])
     and 'groups.list' in _errs['空基组(static_spectrum)']),
    (f'非法 static.columns 报错列出合法值（{(_errs["非法 static.columns"] or "")[:22]}…）',
     bool(_errs['非法 static.columns'])
     and 'off/basic/ac' in _errs['非法 static.columns']),
    (f'两体 R=0 报错含 R_um（{(_errs["两体 R=0"] or "")[:22]}…）',
     bool(_errs['两体 R=0']) and 'R_um' in _errs['两体 R=0']),
])

sys.exit(0 if C.summary() else 1)
