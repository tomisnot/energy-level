# -*- coding: utf-8 -*-
"""最终验收：1) 真实保存状态的 <1min + 正确性；2) 标准场景回归。"""
import os, sys, json, time
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ['QT_QPA_PLATFORM'] = 'offscreen'

import numpy as np
from PyQt5.QtWidgets import QApplication
from arc import Rubidium87
from src import core, evolve, waveforms
import mw_bench_firstprinciples as mb

cache = core.ArcCache(Rubidium87())
ok = True

# ================= 1) 真实保存状态验收 =================
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data = json.load(open(os.path.join(ROOT, 'save_state', 'rb87_state.json'), encoding='utf-8'))
app = QApplication.instance() or QApplication([])
B = float(data['B_gauss'])
specs = []
for g in data['groups']:
    n, l, j = g['n'], g['l'], g['j']
    hf = core.get_hyperfine_levels(cache, n, l, j, B)
    if hf is None:
        for m in np.arange(-j, j + 1e-9, 1.0):
            specs.append(dict(n=n, l=l, j=j, kind='mJ', m=float(m)))
    else:
        for F, mF, e_hf, _w, amps in hf:
            specs.append(dict(n=n, l=l, j=j, kind='F', F=F, mF=mF,
                              E_hf=float(e_hf), amps=amps))
states = evolve.make_levels(cache, specs)
waves = waveforms.restore_waves(data.get('waves', []))
fields = waveforms.build_fields(waves, states, cache)
t = np.linspace(0, float(data['dynamic']['t_max']), int(data['dynamic']['npts']))
psi0 = int(data['dynamic']['psi'])
print(f'===== 验收 1：真实保存状态（N={len(states)}，光场 {len(fields)}，'
      f't_max={t[-1]:.0f} ns，auto）=====')
t0 = time.perf_counter()
r = evolve.evolve(states, fields, psi0, t, method='auto', backend='auto')
dt_auto = time.perf_counter() - t0
norm_err = np.max(np.abs(r['pops'].sum(axis=1) - 1.0))
print(f'[auto → {r["method"]}] 耗时 {dt_auto:.1f}s；归一误差 {norm_err:.2e}')
# 收敛参考：ip4 frac=80
r80 = evolve.evolve(states, fields, psi0, t, method='ip4', ip4_frac=80.0)
diff = np.max(np.abs(r['pops'] - r80['pops']))
print(f'[ip4 frac=80 参考] 耗时 {time.perf_counter()-t0-dt_auto:.0f}s；'
      f'auto vs 参考 max|Δpops| = {diff:.3e}')
ok &= (dt_auto <= 60.0 and norm_err < 1e-4 and diff < 1e-4)
print(f'  验收1：{"PASS" if (dt_auto<=60 and norm_err<1e-4 and diff<1e-4) else "FAIL"}'
      f'（<1min={dt_auto<=60}，归一={norm_err<1e-4}，收敛={diff<1e-4}）')

# ================= 2) 标准场景回归 =================
print('\n===== 验收 2：标准场景回归 =====')
# S1 方波 ip4（floquet 已封存至 src/_archive/evolve_floquet_pwc.py，用其交叉验证）
from src._archive.evolve_floquet_pwc import evolve_floquet  # noqa: E402
st1, i01, E1, fld1, _, om1, w1, _, _, _ = mb.build_s1(cache, 22.0)
t1 = np.linspace(0, 400.0, 2001)
ac1 = E1 - np.array([s['E_mhz'] for s in st1])
ref1 = mb.floquet_evolve_w(np.diag(E1).astype(float), fld1['A'].astype(float),
                           w1, evolve._as_vector(st1, i01), t1, n_sub=4000)
r1 = evolve_floquet(st1, [fld1], i01, t1, ac_shift_mhz=ac1)
d1 = mb.maxdiff(r1['pops'], ref1)
print(f'  S1 floquet(封存): {d1:.2e}  {"PASS" if d1<1e-4 else "FAIL"}')
ok &= d1 < 1e-4
# S2 高斯 ip4
env = evolve.gaussian(200.0, 100.0)
st2, i02, E2, fld2, _, om2, w2, _, _, _ = mb.build_s1(cache, 22.0, env)
t2 = np.linspace(0, 400.0, 2001)
ac2 = E2 - np.array([s['E_mhz'] for s in st2])
w = mb._CONV * w2
s_g = lambda tt: (np.asarray(env(np.asarray(tt, float)), float)
                  * np.cos(w * np.asarray(tt, float)))
ref2 = mb.integrate_rk4_ip(E2, [(fld2['A'].astype(float), s_g)],
                           evolve._as_vector(st2, i02), t2,
                           2 * np.pi / w / 40.0)
r2 = evolve.evolve(st2, [fld2], i02, t2, method='ip4', ac_shift_mhz=ac2)
d2 = mb.maxdiff(r2['pops'], ref2)
print(f'  S2 ip4: {d2:.2e}  {"PASS" if d2<1e-4 else "FAIL"}')
ok &= d2 < 1e-4
# S3 阶梯短窗 ip4 vs hybrid
st3, i03, E3, fields3 = mb.build_s3(cache, 22.0)
t3 = np.linspace(0, 20.0, 201)
ac3 = E3 - np.array([s['E_mhz'] for s in st3])
_, pops_hy, _ = evolve._evolve_full_hybrid(
    E3, st3, fields3, evolve._as_vector(st3, i03), t3, False)
r3 = evolve.evolve(st3, fields3, i03, t3, method='ip4', ac_shift_mhz=ac3)
d3 = mb.maxdiff(r3['pops'], pops_hy)
print(f'  S3 ip4: {d3:.2e}  {"PASS" if d3<1e-4 else "FAIL"}')
ok &= d3 < 1e-4

print(f'\n===== 最终：{"全部 PASS ✅" if ok else "存在 FAIL ❌"} =====')
sys.exit(0 if ok else 1)
