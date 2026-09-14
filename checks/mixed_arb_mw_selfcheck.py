# -*- coding: utf-8 -*-
"""mixed 任意相位微波自检（t79）：env·cos(ωt) → env·cos(φ(t)) 推广的物理正确性。

用途: python -u checks/mixed_arb_mw_selfcheck.py

背景：mixed 的微波项原为 `env(t)·cos(Cv·w·t)`（w 单一标量），故 GUI 遇到
ArbitraryWave（无单一载频）只能整体改道 hard —— 但那会连**光场**一起从
Sambe 光子块（无 RWA）降级到旋转系 RWA。t79 把微波项推广为 `env·cos(φ(t))`
（φ 走查表插值，numba 内核不能调 Python callable）。

物理正确性怎么钉：
  ① 常数频率下两者**逐位等价**（φ=ωt 线性 → 线性插值零误差）；自检直接
     对拍解析载波分支与查表相位分支的 RHS。
  ② 任意（啁啾）相位下 **mixed ≡ hard**（同一条任意波形、同一强度）——
     hard 的 cos(φ_f(t)) 是项目里已确立的权威任意相位路径，作锚。
  ③ 无光场时 mixed 光子维退化 P=1，与 hard 数值等价（既有 t77 契约）。

用低载波跃迁（60P −1.5 ↔ −0.5，~40.5 MHz）以便 ArbitraryWave 采样能
真实表达载波（17 GHz 会超 Nyquist）；且二级纯微波无光子块，是最干净的对照。
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
from src.core.coupling import pack_rhs, rhs_phi_pairs          # noqa: E402
from src.core.evolve_engine import evolve_engine               # noqa: E402
from src.core.evolve_twofield import arbitrary_mw_group        # noqa: E402
from src.core.evolve_util import _CONV                         # noqa: E402
from src.core.field_build import build_fields_exact            # noqa: E402
from src.core.hard import build_two_field                      # noqa: E402
from src.core.waveforms import ArbitraryWave                   # noqa: E402

B = 21.7
C = Checker('mixed_arb_mw')
# ------------------- 计数模型：不变式包（见 checks/checklib.py）-------------------
# group(...) 开一个包，其后相邻的 check(...) 全部变成该包的**子条件**：整包只计 1
# 项，子条件逐个进 SUB=（覆盖面不缩水：SUB == 改前逐条计数的项数）。原断言名与判据
# 表达式一律原样保留——只是不再各自计 1 项；不过时整包自动展开子条件。
_PKG = None


class _Pkg:
    """一个不变式包（先缓冲子条件，关闭时一次性交给 C.group 登记）。"""

    def __init__(self, name, detail=''):
        self.name, self.detail, self.subs = name, detail, []

    def flush(self):
        C.group(self.name, self.subs, self.detail)


def group(name, detail=''):
    """开新的不变式包：此后 check() 的断言都记入它（直到下一个 group）。"""
    global _PKG
    if _PKG is not None:
        _PKG.flush()
    _PKG = _Pkg(name, detail)


def flush():
    """关闭最后一个包（本文件收尾）。"""
    global _PKG
    if _PKG is not None:
        _PKG.flush()
        _PKG = None


def check(name, cond, detail=''):
    """薄包装：判据表达式原样，只登记为**当前包的子条件**（detail 作现场值）。"""
    if _PKG is None:
        raise RuntimeError(f'check() 出现在任何 group() 之前：{name}')
    _PKG.subs.append((name + (('  | ' + str(detail)) if detail else ''),
                      bool(cond)))
    return bool(cond)


def close(name, ours, ref, rtol=1e-6, atol=0.0):
    """薄包装：原 C.close 的签名不变，登记为当前包的子条件。"""
    return check(name, abs(ours - ref) <= atol + rtol * abs(ref),
                 f'ours={ours:.8g} ref={ref:.8g}')
cache = core.ArcCache(Rubidium87())

# ---- 二级：60S(mJ=+0.5) → 60P(mJ=-0.5)（Δl=1、Δm=-1，σ- 电偶极允许）----
# 刻意把上能级放在 40.5 MHz：真实失谐 17.3 GHz 会超出 ArbitraryWave 的
# 采样 Nyquist（要表达 17 GHz 载波需 dt≲0.01 ns）。对拍只需要一个合法的
# 二能级电偶极系统，绝对能量不影响 mixed≡hard 这一结论。
ES = float(cache.E_mhz(60, 0, 0.5))
NU_SET = 40.519
s2 = make_levels(cache, [
    dict(n=60, l=0, j=0.5, kind='mJ', m=+0.5, E_mhz=ES),
    dict(n=60, l=1, j=1.5, kind='mJ', m=-0.5, E_mhz=ES + NU_SET)])
NU = float(s2[1]['E_mhz'] - s2[0]['E_mhz'])
group('A 二级系统与常数频率下 RHS 逐位等价（解析载波 ≡ 查表相位）')
check('二级跃迁落在微波段（便于采样）', 5.0 < NU < 200.0, f'{NU:.3f} MHz')
POL = 'σ-'


def _arb(tag, chirp_mhz_per_ns=0.0, T=600.0, dt=0.5, I=1.2e-3):
    """造一条任意微波波形：载波 NU + 线性啁啾 + 平滑包络。"""
    tn = np.arange(0.0, T + 1e-9, dt)
    ph = _CONV * (NU * tn + 0.5 * chirp_mhz_per_ns * tn ** 2)
    amp = np.sin(np.pi * tn / T) ** 2 * np.cos(ph)
    return ArbitraryWave(tag, tn, amp, pol=POL, intensity_mwcm2=I,
                         offset_ns=0.0)


# ---- ① 解析载波 vs 查表相位：RHS 逐位等价（常数频率）----
def _rhs_probe(use_table):
    t_end, ntab = 50.0, 2001
    env = lambda t: 1.0 + 0.1 * np.sin(0.01 * np.asarray(t, float))
    ts = np.linspace(0.0, t_end, ntab)
    p = [(0, 1, 5.0 + 0j, NU, 1, NU, env)]
    if use_table:
        phl = _CONV * NU * ts
        p = [(0, 1, 5.0 + 0j, NU, 1, NU, env,
              lambda t: np.interp(np.asarray(t, float), ts, phl))]
    pk = pack_rhs(p, t_end, n_tab=ntab)
    phi = np.array([1.0 + 0j, 0.0 + 0j])

    def rhs(tv):
        return rhs_phi_pairs(tv, phi, pk['iis'], pk['jjs'], pk['vals'],
                             pk['dDs'], pk['is_mw'], pk['wcos'],
                             pk['n_pairs_w'], pk['env_tab'], pk['env_dt'],
                             pk['has_ph'], pk['ph_tab'], _CONV)
    return np.array([rhs(tv) for tv in (0.137, 7.3, 23.71, 49.93)])


d_rhs = float(np.abs(_rhs_probe(False) - _rhs_probe(True)).max())
check('① 解析载波 ≡ 查表相位（RHS 逐位）', d_rhs < 1e-12, f'Δ={d_rhs:.3e}')

# ---- ② 任意（啁啾）相位：mixed ≡ hard ----
T = 600.0
tg = np.linspace(0.0, T, 301)
group('B 任意（啁啾）相位：mixed ≡ hard（两组各 4 子条件）')
for tag, chirp in (('常数载频', 0.0), ('啁啾 0.01 MHz/ns', 0.01)):
    w = _arb(tag, chirp_mhz_per_ns=chirp, T=T)
    grp = arbitrary_mw_group(cache, s2, w)
    check(f'{tag}：带宽与载频自洽',
            grp['band_mhz'][0] <= NU <= grp['band_mhz'][1] + 1e-9,
            f"band={grp['band_mhz']}, NU={NU:.3f}")

    meta = build_two_field(cache, s2, [])
    meta['mw_groups'].append(grp)
    r_h = evolve_engine('hard', s2, fields=meta, psi0=0, t_grid=tg, n_body=1,
                        rtol=1e-10, atol=1e-12)

    fields_m = build_fields_exact(cache, s2, [])       # 无普通波
    r_m = evolve_engine('mixed', s2, fields=fields_m, psi0=0, t_grid=tg,
                        n_body=1, mw_arb=[grp], tol=1e-10)

    amp_m = float(np.ptp(r_m['pops'][:, 1]))
    check(f'{tag}：确有动力学', amp_m > 0.05, f'60P 摆幅 {amp_m:.3f}')
    check(f'{tag}：mixed P=1（无光场退化）',
            r_m['error_report']['P'] == 1,
            f"P={r_m['error_report']['P']}")
    close(f'② {tag}：mixed ≡ hard',
            float(np.abs(r_m['pops'] - r_h['pops']).max()), 0.0,
            rtol=0.0, atol=1e-6)

# ---- ③ 回归：普通（单一载频）微波仍 mixed ≡ hard ----
from src.core.waveforms import Wave, GaussianEnv                  # noqa: E402
wv = [Wave('普通微波', NU, GaussianEnv(150, 40), pol=POL,
           intensity_mwcm2=1.2e-3)]
r_om = evolve_engine('mixed', s2, fields=build_fields_exact(cache, s2, wv),
                     psi0=0, t_grid=tg, n_body=1, tol=1e-10)
r_oh = evolve_engine('hard', s2, fields=build_two_field(cache, s2, wv),
                     psi0=0, t_grid=tg, n_body=1, rtol=1e-10, atol=1e-12)
group('C 回归：普通（单一载频）微波仍 mixed ≡ hard')
check('③ 回归组确有动力学（防空洞对照）',
        float(np.ptp(r_om['pops'][:, 1])) > 0.05,
        f"60P 摆幅 {float(np.ptp(r_om['pops'][:, 1])):.3f}")
close('③ 回归：普通微波 mixed ≡ hard',
        float(np.abs(r_om['pops'] - r_oh['pops']).max()), 0.0,
        rtol=0.0, atol=1e-6)

flush()
sys.exit(0 if C.summary() else 1)
