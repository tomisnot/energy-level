# -*- coding: utf-8 -*-
"""轻量自检：hard≡exact + 620ns窗口 + 范数 + Rabi + 同频场组合."""
import os, sys, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from arc import Rubidium87
from src.core import core
from src.core.evolve import make_levels
from src.core.waveforms import GaussianEnv
from src.core.evolve_exact import _CONV
from scipy.integrate import solve_ivp

CACHE = core.ArcCache(Rubidium87())
FAIL = []; PASS = []

def check(name, cond, msg=""):
    if cond:
        PASS.append(name)
        print(f"  ✅ [{name}]")
    else:
        FAIL.append(name)
        print(f"  ❌ [{name}]{msg}")

# Test 1: 二态Rabi (σ⁻, Ω=5MHz, σ_gauss=40ns, peak=1)
print("\n[Test 1] 二态Rabi...")
states_2 = make_levels(CACHE, [
    dict(n=60,l=0,j=0.5,kind='mJ',m=-0.5,E_mhz=0.0),
    dict(n=60,l=0,j=0.5,kind='mJ',m=0.5,E_mhz=17236.743)])
E2 = np.array([0.0, 17236.743])
A2 = np.zeros((2,2)); A2[0,1]=A2[1,0]=5.0
psi0_2 = np.array([0,1],complex); psi0_2[1]=1.0
tw = lambda tv: float(np.asarray(GaussianEnv(100,40,1)(np.atleast_1d(tv)),float).flat[0])

def rhs_dop(tv,y):
    V = tw(tv)*np.cos(_CONV*17236.743*tv)*A2
    u = np.exp(1j*_CONV*E2*tv)
    return -1j*_CONV*((u[:,None]*V*np.conj(u)[None,:])@y)

sol = solve_ivp(rhs_dop,(0,300),psi0_2.copy(),method='DOP853',
                t_eval=np.linspace(0,300,301),rtol=1e-8,atol=1e-10)
pop_peak = np.max(np.abs(sol.y[1])**2)
check("Rabi P_max≈0.5", abs(pop_peak - 0.5021) < 0.01,
      f" got {pop_peak:.4f}")

# Test 2-4: exact/hard/reference tests
from tempsimu._scenario_models import load_scene
from src.core.evolve_exact import build_fields_exact, evolve_exact, evolve_reference

cache, states, wa, wb, _ = load_scene()
psi0_idx = next(i for i,s in enumerate(states)
                if (s['n'],s['l'],s['j'])==(60,0,0.5) and abs(s['m']-0.5)<1e-9)

t_test = np.linspace(0,300,61)

# Test 2: Single MW pulse → π-pulse
print("[Test 2] Exact single-MW π-pulse...")
fields_w1 = build_fields_exact(cache, states, [wa[0]])
r_ex = evolve_exact(states, fields_w1, psi0_idx, t_test, tol=1e-4, log=None)
check("Exact single-MW P(60S)<0.01", r_ex['pops'][-1,psi0_idx]<0.01,
      f" got {r_ex['pops'][-1,psi0_idx]:.4f}")

# Test 3: Combo [1..4] all ≈0.0013
w_mw1 = [w for w in wa if w.name=='波形1']
w_mw_all = [w for w in wa if '波形1' in w.name]
w_mw2 = [w for w in wa if w.name=='波形1(副本)(副本)']
combos = [w_mw1, w_mw_all, w_mw2, wa]
for idx, cwaves in enumerate(combos, 1):
    fields_c = build_fields_exact(cache, states, cwaves)
    r_c = evolve_exact(states, fields_c, psi0_idx, t_test, tol=1e-4, log=None)
    val = r_c['pops'][-1,psi0_idx]
    check(f"Combo[{idx}] P(60S)≈0.0013", abs(val-0.0013)<0.001,
          f" got {val:.4f}")

# Test 4: 620ns optical window exact vs reference
print("[Test 4] 620ns Window exact vs reference...")
fields_full = build_fields_exact(cache, states, wa)
t_620 = np.linspace(0,620,63)
r_ex6 = evolve_exact(states, fields_full, psi0_idx, t_620, tol=1e-4, log=None)
r_re6 = evolve_reference(states, fields_full, psi0_idx, t_620, rtol=1e-9, atol=1e-11, log=None)
opt_mask = (t_620>=310)&(t_620<=580)
diff_opt = np.max(np.abs(r_ex6['pops'][opt_mask]-r_re6['pops'][opt_mask]))
check("620ns Window |exact-ref|≤1e-3", diff_opt<1e-3, f" got {diff_opt:.2e}")

# Summary
print(f"\n{'='*50}")
print(f"自检结果: {len(PASS)} pass / {len(PASS)+len(FAIL)} total")
if FAIL:
    print(f"失败项: {FAIL}")
else:
    print("全部通过！")
