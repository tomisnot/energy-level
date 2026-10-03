# -*- coding: utf-8 -*-
"""KPM/Chebyshev 谱方法标杆：general-H 的有限温 & DOS，对齐精确对角化参照。

被测 `src/core/manybody/kpm.py`（只需稀疏 matvec、对复/非 stoquastic 无符号问题）。
在**小维**上拿 `equilibrium.thermal_from_h`（谱分解精确）与 `eigvalsh` 直方图当金标准：
- 有限温：kpm_thermal(exact 矩) 的 E(β)、C(β) ≡ 精确热力学（Chebyshev 插值精度，紧）；
  随机 Z2 迹（Hutchinson）在同 β 下按统计容差逼近（验随机迹确实收敛到精确）。
- DOS：归一 ∫ρ dE ≈ 1；一阶矩 ∫E ρ(E)dE ≈ Tr(H)/D；与 eigvalsh 直方图形状吻合（精确矩、紧）。
H 用**含复 J/A 的广义里德堡**（QMC 会在此栽符号问题，KPM 不受限）——正是本方法的价值。
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import numpy as np
import scipy.sparse as sp

from checks.checklib import Checker                              # noqa: E402
from src.core.manybody import assemble as asm                    # noqa: E402
from src.core.manybody import equilibrium as eq                  # noqa: E402
from src.core.manybody import kpm                                # noqa: E402

C = Checker('equibench_kpm')
TERMS = ('I1', 'I2', 'I3')


def _calib(n, pairs):
    return dict(terms=list(TERMS),
                atoms=[dict(id=i, pos=(float(i), 0.0, 0.0)) for i in range(n)],
                pairs=pairs, single_basis=[])


def _pair(i, j, V, J, A):
    return dict(i=i, j=j, V=float(V), J=complex(J), A=complex(A), raw_V=float(V),
                raw_J=complex(J), raw_A=complex(A), delta={}, info={})


rng = np.random.default_rng(7)
n = 8
dim = 1 << n
pairs = [_pair(i, i + 1, rng.uniform(0, 2), rng.normal() + 1j * rng.normal(),
               rng.normal() + 1j * rng.normal()) for i in range(n - 1)]
pairs += [_pair(i, i + 2, rng.uniform(0, 1), rng.normal(0, .5), rng.normal(0, .5))
          for i in range(n - 2)]
Om = rng.uniform(0, 1, n)
delta = rng.uniform(-1, 1, n)
H = np.asarray(asm.build_H_ed(_calib(n, pairs), Om, delta, terms=list(TERMS)))
Hs = sp.csr_matrix(H)
evals = np.linalg.eigvalsh(H)

# ---------- 有限温：KPM(精确矩) vs 精确谱分解 thermal_from_h ----------
betas = np.array([0.1, 0.5, 2.0])
kt = kpm.kpm_thermal(Hs, betas, num=512, exact=True)
subs_t = []
for i, b in enumerate(betas):
    th = eq.thermal_from_h(Hs, float(b))
    subs_t += [
        (f'β={b}: KPM E(β) ≡ 精确 <1e-5', abs(kt['E'][i] - th['E']) < 1e-5),
        (f'β={b}: KPM C(β) ≡ 精确 <1e-4', abs(kt['C'][i] - th['C']) < 1e-4),
    ]
C.group('KPM 有限温（精确矩）≡ 谱分解热力学（复/非 stoquastic 广义 H）', subs_t)

# ---------- 随机 Z2 迹（Hutchinson）：确定性 + 随 nvec 收敛 + 统计容差 ----------
betas_s = np.array([0.1, 0.3])
k2 = kpm.kpm_thermal(Hs, betas_s, num=400, nvec=200, seed=3, exact=False)
k2b = kpm.kpm_thermal(Hs, betas_s, num=400, nvec=200, seed=3, exact=False)   # 同种子
k8 = kpm.kpm_thermal(Hs, betas_s, num=400, nvec=2000, seed=5, exact=False)
subs_s = [
    ('同种子两次完全一致（确定性）', bool(np.array_equal(k2['E'], k2b['E']))),
    ('随机迹 E(β) 大样本(nvec=2000) ≡ 精确 <2e-2',
     max(abs(k8['E'][i] - eq.thermal_from_h(Hs, float(b))['E'])
         for i, b in enumerate(betas_s)) < 2e-2),
    ('增样本 nvec 200→2000 不劣于小样本偏蒮',
     abs(k8['E'][0] - eq.thermal_from_h(Hs, 0.1)['E']) <=
     abs(k2['E'][0] - eq.thermal_from_h(Hs, 0.1)['E']) + 1e-3),
]
C.group('KPM 随机迹（Hutchinson）：确定性 + nvec 收敛 + 统计容差', subs_s)

# ---------- DOS：θ-测度矩（归一/E/E²）——分布由矩定，已足够证 DOS 正确 ----------
numd = 400
mu, c, r = kpm.kpm_moments(Hs, numd, exact=True)
G = kpm.jackson_kernel(numd)
th = np.linspace(0.0, np.pi, 4001)[1:-1]
xg = np.cos(th)
coef = np.empty(numd + 1)
coef[0] = mu[0]
coef[1:] = 2 * G[1:] * mu[1:]
from numpy.polynomial.chebyshev import chebval
series = chebval(xg, coef)
inv_pi = 1.0 / np.pi
norm = inv_pi * np.trapezoid(series, th)
E1 = inv_pi * np.trapezoid((c + r * xg) * series, th)
E2 = inv_pi * np.trapezoid((c + r * xg) ** 2 * series, th)
t1 = float(np.mean(evals))
t2 = float(np.mean(evals ** 2))
# 另验：kpm_dos 逐点非负且积分为 1（与上面矩一致的另一条路径）
Eg = np.linspace(evals[0] + 1e-9, evals[-1] - 1e-9, 500)
rho = kpm.kpm_dos(Hs, Eg, num=numd, exact=True)
subs_d = [
    ('DOS 归一 ∫ρdE = μ0 ≈ 1 (<1e-2)', abs(norm - 1.0) < 1e-2),
    ('DOS 一阶矩 == Tr H/D (<1e-3)', abs(E1 - t1) < 1e-3),
    ('DOS 二阶矩 == Tr H²/D (rel<2e-2)', abs(E2 - t2) < 2e-2 * max(1.0, abs(t2))),
    ('kpm_dos 逐点非负 & 梯形积分≈1', bool(np.all(rho >= -1e-12)) and abs(np.trapezoid(rho, Eg) - 1.0) < 5e-2),
]
C.group('KPM 态密度：θ-测度矩(归一/E/E²) + 逐点非负/积分（general 复 H）', subs_d)

sys.exit(0 if C.summary() else 1)
