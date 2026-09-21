# -*- coding: utf-8 -*-
"""静态页与动态页的能级基准一致性（能级枚举已收成 `static_atom.base_sublevels`）。

守三件事：
  A. 两页的态集合与能量**逐位相同**（同一 `groups + B + E_vpcm`）；
  B. DC-Stark 两页都生效且相等（`E_vpcm≠0` 时能量确实移动，两页移动量相同）；
  C. 超精细态的 DC 按 mJ 分量加权——**不能**把 mF 当 mJ 传（那会恒返回 0）。
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

from checks.checklib import Checker                            # noqa: E402
from arc import Rubidium87                                     # noqa: E402
from src.core import runspec as RS                             # noqa: E402
from src.core import static_atom as core                       # noqa: E402

C = Checker('levels_twopage')
cache = core.ArcCache(Rubidium87())

B = 21.7
GROUPS = [dict(n=60, l=0, j=0.5, color='#1f77b4', label='60S'),
          dict(n=60, l=1, j=1.5, color='#ff7f0e', label='60P'),
          dict(n=6, l=1, j=1.5, color='#2ca02c', label='6P')]
NO_LIGHT = dict(enabled=False, frequency_hz=0.0, intensity=0.0,
                polarization='π', elliptical={})


def _qkey(s):
    if s['kind'] == 'F':
        return (s['n'], s['l'], s['j'],
                'F=%.1f, m_F=%+.1f' % (s['F'], s['mF']))
    return (s['n'], s['l'], s['j'], 'm_J=%+.1f' % s['mj'])


def _pair(E_vpcm):
    """同一配置下取 (静态 rows, 动态 states)。"""
    rows = core.build_level_rows(cache, B, GROUPS, NO_LIGHT, GROUPS,
                                 E_vpcm=E_vpcm)
    st = RS.build_states(cache, GROUPS, B, E_vpcm)
    return rows, st


# ---------------- A. 态集合 + 能量逐位一致 ----------------
for Ev in (0.0, 1.0, 10.0):
    rows, st = _pair(Ev)
    sm = {_qkey(r): r['E0'] for r in rows}
    dm = {_qkey(s): s['E_mhz'] for s in st}
    same_keys = set(sm) == set(dm)
    dmax = (max((abs(sm[k] - dm[k]) for k in sm), default=0.0)
            if same_keys else float('nan'))
    C.group('A 两页同基准（E_vpcm=%.0f V/cm，%d 个态）' % (Ev, len(sm)), [
        ('态集合相同（静态 %d / 动态 %d）' % (len(sm), len(dm)), same_keys),
        ('能量逐位相同 max|Δ| = %.1e' % dmax, same_keys and dmax == 0.0),
    ])


# ---------------- A2. 两页 label 同字（2026-09-18 统一到 auto_label） ----------------
def _key(kind, n, l, j, mj, mF):
    return (kind, n, l, j, round(float(mj if kind == 'mJ' else mF), 6))


for Ev in (0.0, 10.0):
    rows, st = _pair(Ev)
    sm = {_key(r['kind'], r['n'], r['l'], r['j'], r.get('mj'), r.get('mF')):
          r['label'] for r in rows}
    dm = {_key(s['kind'], s['n'], s['l'], s['j'], s.get('mj'), s.get('mF')):
          s['label'] for s in st}
    ok_set = set(sm) == set(dm)
    ok_lab = ok_set and all(sm[k] == dm[k] for k in sm)
    gl = sorted({r.get('group_label') for r in rows})
    C.group('A2 两页 label 同字（E_vpcm=%.0f）' % Ev, [
        ('态集合相同（%d 个）' % len(sm), ok_set),
        ('**label 逐项相同**（统一到 auto_label）', ok_lab),
        ('人工名仍在 `group_label`（供显示层）：%s'
         % ','.join(str(x) for x in gl), all(bool(g) for g in gl)),
    ])

# ---------------- B. DC-Stark 两页都生效 ----------------
r0, s0 = _pair(0.0)
r1, s1 = _pair(10.0)
d_static = max(abs(a['E0'] - b['E0']) for a, b in zip(r0, r1))
d_dyn = max(abs(a['E_mhz'] - b['E_mhz']) for a, b in zip(s0, s1))
C.group('B DC-Stark 两页都生效（E_vpcm 0→10 V/cm）', [
    ('静态页能量确实移动 %.3f MHz' % d_static, d_static > 1.0),
    ('动态基组能量确实移动 %.3f MHz' % d_dyn, d_dyn > 1.0),
    ('两页移动量逐位相同（%.6f / %.6f）' % (d_static, d_dyn),
     d_static == d_dyn),
])

# ---------------- C. 超精细态 DC 按 mJ 加权 ----------------
hf = core.get_hyperfine_levels(cache, 6, 1, 1.5, B)
F0, mF0, _e, _w, amps0 = hf[0]
wrong = core.dc_stark_shift_mhz(cache, 6, 1, 1.5, float(mF0), GROUPS, 100.0, B)
right = core.dc_stark_hf(cache, 6, 1, 1.5, amps0, GROUPS, 100.0, B)
rows100, _st = _pair(100.0)
rows0, _st0 = _pair(0.0)


def _p6p(rows):
    return [r for r in rows
            if r['n'] == 6 and r['kind'] == 'F'
            and abs(r['F'] - F0) < 1e-9 and abs(r['mF'] - mF0) < 1e-9]


_row_shift = _p6p(rows100)[0]['E0'] - _p6p(rows0)[0]['E0']
C.group('C 超精细 DC 按 mJ 分量加权（非 mF 直传）', [
    ('mF 直传恒为 0（旧行为的静默失效）：%.3e MHz' % wrong, wrong == 0.0),
    ('加权后非零：%.6e MHz' % right, abs(right) > 1e-9),
    ('静态页行位移 ≈ 加权值（%.6e vs %.6e）' % (_row_shift, right),
     abs(_row_shift - right) < 1e-6),
])

sys.exit(0 if C.summary() else 1)
