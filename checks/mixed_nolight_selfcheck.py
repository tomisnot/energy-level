# -*- coding: utf-8 -*-
"""mixed 无光场退化自检（t77）：引擎层不再要求光场。

背景：mixed = 光场光子块 + 微波/V_dd 当场。光场缺席时（纯微波，或光波被
AC 化/偏振选择定则全部剔除）光子维退化 P=1，φ-帧回到实验室系，等价于
微波/V_dd 当场精确积分。本自检钉住这条契约，防止回归成"必须光场"。

覆盖：
  1. 零耦合偏振光波被丢弃后仍可运行（原 ValueError 场景）
  2. 单体纯微波：mixed ≡ hard（≤1e-6）
  3. 两体无 V_dd：mixed_pair ≡ hard_pair（≤1e-6）
  4. 两体含 V_dd：≤1e-6（t78 起 hard/mixed 共用累积相位判据 → 存活集一致）
  5. 有光场回归：光子块仍为 P=3

用法：python checks\\mixed_nolight_selfcheck.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

from checks.checklib import Checker                            # noqa: E402
from arc import Rubidium87                                     # noqa: E402
from src.core import static_atom as core                       # noqa: E402
from src.core.atomic_levels import make_levels                 # noqa: E402
from src.core.evolve_mixed import evolve_mixed, evolve_mixed_pair  # noqa: E402
from src.core.field_build import build_fields_exact            # noqa: E402
from src.core.hard import (build_two_field, evolve_hard,       # noqa: E402
                           evolve_hard_pair)
from src.core.static_pair import vdd_sparse                    # noqa: E402
from src.core.waveforms import Wave, GaussianEnv               # noqa: E402

B = 21.7
C = Checker('mixed_nolight')
cache = core.ArcCache(Rubidium87())


def _levels(ms_s, ms_p, with_6p=False):
    specs = [dict(n=60, l=0, j=0.5, kind='mJ', m=m) for m in ms_s]
    specs += [dict(n=60, l=1, j=1.5, kind='mJ', m=m) for m in ms_p]
    if with_6p:
        specs += [dict(n=6, l=1, j=1.5, kind='F', F=F, mF=mF,
                       E_mhz=cache.E_mhz(6, 1, 1.5), E_hf=float(e), amps=a)
                  for F, mF, e, _w, a in core.get_hyperfine_levels(
                      cache, 6, 1, 1.5, B)]
    return make_levels(cache, specs)


def _mw(freq, pol='σ-', I=1.2e-3):
    return Wave('微波', freq, GaussianEnv(70, 20), pol=pol, intensity_mwcm2=I)


# ---- 1. 零耦合偏振光波（Δm=0 基组 + σ− 光波）被丢弃后仍可运行 ----
s1 = _levels([+0.5], [+0.5])
nu1 = s1[1]['E_mhz'] - s1[0]['E_mhz']
f1 = build_fields_exact(cache, s1, [
    Wave('寻址光', 295.7237e6, GaussianEnv(0.0, 1e9), pol='σ-',
         intensity_mwcm2=1e4),                      # 零耦合 → 被丢弃
    _mw(nu1, pol='π')])
C.check('零耦合光波被丢弃（只剩微波）',
        [x['label'] for x in f1] == ['微波'],
        f'实际 {[x["label"] for x in f1]}')
t1 = np.linspace(0.0, 120.0, 121)
r1 = evolve_mixed(s1, f1, 0, t1)
C.check('原 ValueError 场景现可运行',
        r1['backend'] == 'mixed-phi-dop853')
C.check('无光场 → P=1 且维度退化到 N',
        r1['error_report']['P'] == 1 and r1['error_report']['dim'] == len(s1),
        f'P={r1["error_report"]["P"]} dim={r1["error_report"]["dim"]}')
C.check('确有动力学（非空演化）',
        float(np.ptp(r1['pops'][:, 1])) > 0.1,
        f'60P 摆幅 {float(np.ptp(r1["pops"][:, 1])):.3f}')

# ---- 2. 单体纯微波：mixed ≡ hard ----
s2 = _levels([+0.5], [-0.5])
nu2 = s2[1]['E_mhz'] - s2[0]['E_mhz']
f2 = [_mw(nu2)]
t2 = np.linspace(0.0, 200.0, 201)
r_m = evolve_mixed(s2, build_fields_exact(cache, s2, f2), 0, t2)
r_h = evolve_hard(s2, build_two_field(cache, s2, f2), 0, t2)
C.check('单体共振 π 脉冲确实发生',
        r_m['pops'][:, 1].max() > 0.99,
        f'最大 60P {r_m["pops"][:, 1].max():.4f}')
C.close('单体 mixed ≡ hard', float(np.abs(r_m['pops'] - r_h['pops']).max()),
        0.0, rtol=0.0, atol=1e-6)

# ---- 3/4. 两体：无 V_dd / 含 V_dd ----
s3 = _levels([+0.5, -0.5], [+1.5, +0.5, -0.5])
nu3 = s3[3]['E_mhz'] - s3[0]['E_mhz'] if s3[3]['E_mhz'] > 0 else nu2
f3 = [_mw(nu3)]
fA = build_fields_exact(cache, s3, f3)
metaA = build_two_field(cache, s3, f3)
t3 = np.linspace(0.0, 200.0, 201)
i0 = 0

p_nv_m = evolve_mixed_pair(s3, fA, fA, i0, i0, t3)
p_nv_h = evolve_hard_pair(s3, metaA, metaA, i0, i0, t3)
C.close('两体（无 V_dd）mixed_pair ≡ hard_pair',
        float(np.abs(p_nv_m['pops'] - p_nv_h['pops']).max()),
        0.0, rtol=0.0, atol=1e-6)

V = vdd_sparse(cache, [dict(x) for x in s3], R_um=12.0,
               theta_deg=30.0, phi_deg=0.0)
p_v_m = evolve_mixed_pair(s3, fA, fA, i0, i0, t3, vdd_matrix=V)
p_v_h = evolve_hard_pair(s3, metaA, metaA, i0, i0, t3, vdd_matrix=V)
C.check('两体 P=1 且含 V_dd 时仍收敛',
        p_v_m['error_report']['P'] == 1
        and p_v_m['error_report']['max_norm_dev'] < 1e-5,
        f'norm={p_v_m["error_report"]["max_norm_dev"]:.2e}')
C.close('两体（含 V_dd）mixed_pair ≡ hard_pair（t78 统一判据后）',
        float(np.abs(p_v_m['pops'] - p_v_h['pops']).max()),
        0.0, rtol=0.0, atol=1e-6)

# ---- 5. 有光场回归：光子块仍是 P=3 ----
s5 = _levels([+0.5], [-0.5], with_6p=True)
f5 = build_fields_exact(cache, s5, [
    Wave('寻址光', 295.7237e6, GaussianEnv(0.0, 1e9), pol='σ-',
         intensity_mwcm2=1e4),
    _mw(nu2)])
r5 = evolve_mixed(s5, f5, 0, t1)
C.check('有光场时光子块未被误退化（P=3）',
        r5['error_report']['P'] == 3,
        f'P={r5["error_report"]["P"]}')

sys.exit(0 if C.summary() else 1)
