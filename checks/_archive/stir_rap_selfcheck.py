# -*- coding: utf-8 -*-
"""协议自检（STIRAP + 拉曼跃迁，S/R 系列）：

  S6  STIRAP（field() 干净 3 能级，B=21.7G）：
       5S F=2 mF=2 --σ+泵浦(Δ=+10MHz)--> 6P3/2 F=3 mF=3 --σ-斯托克斯(Δ=-10)--> 60S mJ=+1/2
       · 反直觉时序（斯托克斯先）→ 60S 保真 >0.99，且 6P 峰≈0（暗态中间态真空）
       · 直觉时序 → 保真明显更低、6P 峰显著（~0.5）
       · 反直觉 + 自动衰减（6P Γ=8.46MHz）仍 >0.99（STIRAP 抗自发辐射）
  S6b real_field（GUI 路径，B=300G 拉开 Zeeman，默认 5GHz 截断）：
       无衰减 >0.99；加衰减 >0.7
  R1  对称 Λ 拉曼（Δ=300，Ω=20/20）：Ω_eff=Ω²/2Δ=0.667MHz → π 时刻解析对照
  R2  对称 Λ 拉曼（Δ=100，Ω=20/20）：Ω_eff=2.0MHz → π 时刻解析对照（Ω_eff 一般式）
  R3  非对称 Λ 拉曼（Ω_p=20/Ω_s=40）：差频光移 δ2=(Ωs²-Ωp²)/4Δ=1.0MHz →
       转移幅度 ≈ Ω_eff²/(Ω_eff²+δ2²)=0.64（引擎正确捕捉差频光移）

运行：D:\\python310\\python.exe -u checks\\stir_rap_selfcheck.py
"""
import os
import sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
_CHECKS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(_CHECKS))    # 项目根（src/）
sys.path.insert(0, _CHECKS)                     # checks/（checklib）

import numpy as np
from arc import Rubidium87

from src import core, evolve
from checklib import Checker

ck = Checker('stir_rap_selfcheck')
cache = core.ArcCache(Rubidium87())


def build(B):
    specs = []
    for F, mF, e_hf, _w, amps in core.get_hyperfine_levels(cache, 5, 0, 0.5, B):
        specs.append(dict(n=5, l=0, j=0.5, kind='F', F=F, mF=mF,
                          E_hf=float(e_hf), amps=amps, E_mhz=cache.E_mhz(5, 0, 0.5)))
    for F, mF, e_hf, _w, amps in core.get_hyperfine_levels(cache, 6, 1, 1.5, B):
        specs.append(dict(n=6, l=1, j=1.5, kind='F', F=F, mF=mF,
                          E_hf=float(e_hf), amps=amps, E_mhz=cache.E_mhz(6, 1, 1.5)))
    specs.append(dict(n=60, l=0, j=0.5, kind='mJ', m=+0.5,
                      E_mhz=core._sublevel_energy(cache, 60, 0, 0.5, 0.5, B)))
    return evolve.make_levels(cache, specs)


def idx(states, F=None, mF=None, m=None):
    for i, s in enumerate(states):
        if F is not None and (s.get('F') != F or abs(s.get('mF', 9) - mF) > 1e-9):
            continue
        if m is not None and abs(s['m'] - m) > 1e-9:
            continue
        return i
    raise KeyError((F, mF, m))


# ---------------- S6 STIRAP（干净 3 能级，B=21.7G） ----------------
st = build(21.7)
i5 = idx(st, F=2, mF=2); i6 = idx(st, F=3, mF=3); i60 = idx(st, m=0.5)
SIG, SEP, DELTA, OM = 400.0, 800.0, 10.0, 50.0
t_p, t_s = 2.5 * SIG + SEP, 2.5 * SIG
TMAX = 2.5 * SIG + SEP + 4 * SIG
t = np.linspace(0, TMAX, int(TMAX / 4.0) + 1)


def run_stir(order, dec=None):
    pc, sc = (t_p, t_s) if order == 'counter' else (t_s, t_p)
    pump = evolve.field(i5, i6, omega0_mhz=OM, waveform=evolve.gaussian(pc, SIG),
                        detune_mhz=DELTA)
    stok = evolve.field(i6, i60, omega0_mhz=OM, waveform=evolve.gaussian(sc, SIG),
                        detune_mhz=-DELTA)
    r = evolve.evolve(st, [pump, stok], i5, t, method='auto', decays=dec)
    return r['pops'][-1, i60], r['pops'][:, i6].max()


c, p6c = run_stir('counter')
it, p6i = run_stir('intuitive')
cd, p6d = run_stir('counter', evolve.auto_decays(cache, st))
ck.check(f'S6 STIRAP 反直觉保真 {c:.4f}（6P峰={p6c:.3f}≈0 暗态）',
         c > 0.99 and p6c < 0.01, f'c={c} p6={p6c}')
ck.check(f'S6 STIRAP 直觉对照保真 {it:.4f}（6P峰={p6i:.3f}，时序可区分）',
         it < c - 0.005 and p6i > 0.3, f'it={it} p6={p6i}')
ck.check(f'S6 STIRAP 反直觉+衰减 {cd:.4f}（抗自发辐射）',
         cd > 0.99, f'cd={cd}')

# ---------------- S6b real_field（GUI 路径，B=300G） ----------------
st2 = build(300.0)
i5b = idx(st2, F=2, mF=2); i6b = idx(st2, F=3, mF=3); i60b = idx(st2, m=0.5)
d_p = core.dipole_amplitude(cache, evolve._sublevel_row(st2[i5b]),
                            evolve._sublevel_row(st2[i6b]), 1)
nu_p = st2[i6b]['E_mhz'] - st2[i5b]['E_mhz']
I_p = core.intensity_mwcm2_from_rabi(abs(d_p), OM)
d_s = core.dipole_amplitude(cache, evolve._sublevel_row(st2[i6b]),
                            evolve._sublevel_row(st2[i60b]), -1)
nu_s = st2[i60b]['E_mhz'] - st2[i6b]['E_mhz']
I_s = core.intensity_mwcm2_from_rabi(abs(d_s), OM)
fld_p, _ = evolve.real_field(cache, st2, nu_p + DELTA, I_p, 'σ+',
                             waveform=evolve.gaussian(t_p, SIG))
fld_s, _ = evolve.real_field(cache, st2, nu_s - DELTA, I_s, 'σ-',
                             waveform=evolve.gaussian(t_s, SIG))
r_b = evolve.evolve(st2, [fld_p, fld_s], i5b, t, method='auto')
r_bd = evolve.evolve(st2, [fld_p, fld_s], i5b, t, method='auto',
                     decays=evolve.auto_decays(cache, st2))
ck.check(f'S6b real_field@B=300G 无衰减 {r_b["pops"][-1, i60b]:.4f}（GUI 路径）',
         r_b['pops'][-1, i60b] > 0.99, f'p={r_b["pops"][-1, i60b]}')
ck.check(f'S6b real_field@B=300G 加衰减 {r_bd["pops"][-1, i60b]:.4f}',
         r_bd['pops'][-1, i60b] > 0.7, f'p={r_bd["pops"][-1, i60b]}')

# ---------------- R1/R2 对称 Λ 拉曼（Ω_eff 解析对照） ----------------
t_r = np.linspace(0, 3000.0, 601)


def raman_peak(om_p, om_s, delta):
    pump = evolve.field(i5, i6, omega0_mhz=om_p, detune_mhz=delta)
    stok = evolve.field(i6, i60, omega0_mhz=om_s, detune_mhz=-delta)
    r = evolve.evolve(st, [pump, stok], i5, t_r, method='auto')
    p = r['pops'][:, i60]
    # π 时刻 = 首次布居 ≥0.999（满转移穿越；峰顶 sin²≈1−ε 宽平，argmax/局部
    # 极大会因浮点/采样微差偏移；0.999 穿越点距真 π 时刻 <2%）
    idx = np.where(p[:int(len(t_r) * 0.8)] > 0.999)[0]
    return p.max(), (t_r[idx[0]] if len(idx) else np.nan)


oe1 = 20.0 ** 2 / (2.0 * 300.0)
amp1, tp1 = raman_peak(20.0, 20.0, 300.0)
T1 = np.pi / (2.0 * np.pi * oe1 * 1e-3)
ck.check(f'R1 对称Λ Ω_eff={oe1:.3f}MHz π时刻 {tp1:.0f} vs 解析 {T1:.0f}',
         amp1 > 0.99 and abs(tp1 / T1 - 1.0) < 0.05,
         f'amp={amp1} t={tp1}')
oe2 = 20.0 ** 2 / (2.0 * 100.0)
amp2, tp2 = raman_peak(20.0, 20.0, 100.0)
T2 = np.pi / (2.0 * np.pi * oe2 * 1e-3)
ck.check(f'R2 对称Λ Ω_eff={oe2:.3f}MHz π时刻 {tp2:.0f} vs 解析 {T2:.0f}',
         amp2 > 0.99 and abs(tp2 / T2 - 1.0) < 0.05,
         f'amp={amp2} t={tp2}')

# ---------------- R3 非对称 Λ（差频光移） ----------------
oe3 = 20.0 * 40.0 / (2.0 * 300.0)
d2 = (40.0 ** 2 - 20.0 ** 2) / (4.0 * 300.0)          # δ2=1.0MHz 差频光移
pred = oe3 ** 2 / (oe3 ** 2 + d2 ** 2)                # 转移幅度预测
amp3, _ = raman_peak(20.0, 40.0, 300.0)
ck.check(f'R3 非对称Λ 转移幅度 {amp3:.3f} vs 光移预测 {pred:.3f}',
         abs(amp3 - pred) < 0.05, f'amp={amp3} pred={pred}')

ok = ck.summary()
sys.exit(0 if ok else 1)
