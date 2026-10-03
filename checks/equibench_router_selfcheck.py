# -*- coding: utf-8 -*-
"""② 算法路由 + 统一入口 标杆：路由决策正确 + 跨后端一致 + Result 契约齐。

- `route_equilibrium` 纯决策：按 (n_sites, want, is_ladder, batch, gpu) 选后端，逐条核。
- `solve_equilibrium` 跨后端一致：同一 2-leg 阶梯，强制 ed/lanczos/dmrg/(gpu-batch) → E0 相符；
  thermal(KPM) ≡ 精确谱分解热力学；dos 归一。
- 统一 Result 契约键齐（want/backend/n_sites + E0 或 E/C/F 或 energies/rho）。
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
from src.core.manybody import eq_route as R                      # noqa: E402

C = Checker('equibench_router')

# 1) 纯路由决策（不跑解，确定性；gpu 显式传入）
subs_r = [
    ('thermal→kpm', R.route_equilibrium(6, 'thermal', gpu=False) == 'kpm'),
    ('dos→kpm', R.route_equilibrium(6, 'dos', gpu=False) == 'kpm'),
    ('小维gs→ed', R.route_equilibrium(6, 'gs', gpu=False) == 'ed'),
    ('批点+GPU→gpu-batch', R.route_equilibrium(6, 'gs', batch=32, gpu=True) == 'gpu-batch'),
    ('大阶梯gs→dmrg', R.route_equilibrium(20, 'gs', is_ladder=True, gpu=False) == 'dmrg'),
    ('大非阶梯gs→lanczos', R.route_equilibrium(20, 'gs', is_ladder=False, gpu=False) == 'lanczos'),
    ('gpu但批点=1不抢gpu-batch', R.route_equilibrium(6, 'gs', batch=1, gpu=True) == 'ed'),
]
C.group('② route_equilibrium 决策正确', subs_r)

# 2) 跨后端一致（同一 2-leg 阶梯 length=3 → n=6）
L = 3
n = 2 * L
V_r, J_r, V_l, J_l, delta, Omega = 0.8, 0.3, 0.5, -0.4, 0.6, 1.1


def _pair(i, j, V, J):
    return dict(i=i, j=j, V=float(V), J=complex(J), A=0j, raw_V=float(V),
                raw_J=complex(J), raw_A=0j, delta={}, info={})


pairs = [_pair(2 * x, 2 * x + 1, V_r, J_r) for x in range(L)]
pairs += [_pair(2 * x, 2 * x + 2, V_l, J_l) for x in range(L - 1)]
pairs += [_pair(2 * x + 1, 2 * x + 3, V_l, J_l) for x in range(L - 1)]
calib = dict(terms=['I1', 'I2', 'I3'],
             atoms=[dict(id=i, pos=(float(i % 2), float(i // 2), 0.0)) for i in range(n)],
             pairs=pairs, single_basis=[])
ladder = dict(length=L, V_r=V_r, J_r=J_r, A_r=0.0, V_l=V_l, J_l=J_l, A_l=0.0,
              delta=delta, Omega=Omega)
system = dict(calib=calib, Om=[Omega] * n, delta=[delta] * n, ladder=ladder,
              is_ladder=True, n_sites=n)

E_ed = R.solve_equilibrium(system, want='gs', backend='ed')['E0']
subs_c = [('Result 契约键齐(gs)',
           {'want', 'backend', 'n_sites', 'E0'} <= set(R.solve_equilibrium(system, 'gs', backend='ed')))]
subs_c.append(('lanczos ≡ ed E0 <1e-8',
               abs(R.solve_equilibrium(system, 'gs', backend='lanczos')['E0'] - E_ed) < 1e-8))
HAS_TENPY = True
try:
    import tenpy  # noqa: F401
except Exception:
    HAS_TENPY = False
if HAS_TENPY:
    subs_c.append(('dmrg ≡ ed E0 <1e-6',
                   abs(R.solve_equilibrium(system, 'gs', backend='dmrg')['E0'] - E_ed) < 1e-6))
else:
    C.skip('dmrg 后端一致性', 'tenpy 不可用')
if eq.detect_gpu() is not None:
    subs_c.append(('gpu-batch ≡ ed E0 <1e-8',
                   abs(R.solve_equilibrium({**system, 'batch': 4}, 'gs',
                                           backend='gpu-batch')['E0'] - E_ed) < 1e-8))
else:
    C.skip('gpu-batch 后端一致性', '无 GPU')
C.group('② solve_equilibrium 跨后端基态一致（ed/lanczos/dmrg/gpu）', subs_c)

# 3) thermal(KPM) ≡ 精确谱分解；dos 归一；Result 契约
Hs = asm.build_H_sparse(calib, [Omega] * n, [delta] * n)
betas = [0.3, 1.0]
th = R.solve_equilibrium(system, want='thermal', betas=betas, num=400, nvec=1500, seed=3)
subs_th = [
    ('thermal Result 键齐', {'want', 'backend', 'E', 'C', 'F'} <= set(th)),
    ('KPM E(β) ≡ 精确热力学 <2e-2',
     max(abs(th['E'][i] - eq.thermal_from_h(Hs, b)['E']) for i, b in enumerate(betas)) < 2e-2),
]
dos = R.solve_equilibrium(system, want='dos', num=400, exact=True)
norm = float(np.trapezoid(dos['rho'], dos['energies'])) if hasattr(np, 'trapezoid') \
    else float(np.trapz(dos['rho'], dos['energies']))
subs_th.append(('dos 契约键齐 & rho 非负 & 粗略归一(0.7~1.3)',
                {'want', 'backend', 'energies', 'rho'} <= set(dos)
                and bool(np.all(dos['rho'] >= -1e-12)) and 0.7 < norm < 1.3))
C.group('② solve_equilibrium 有限温/态密度一致 + 契约', subs_th)

sys.exit(0 if C.summary() else 1)
