# -*- coding: utf-8 -*-
"""② 端到端**分派调用**集成判据：请求经唯一分派点 runspec.evolve_run → 平衡态后端。

证明"加进来的算法都能正常调用"：用**假 calib**（不碰 ARC）造一个 equilibrium RunSpec，
塞进 `runspec.evolve_run`（全仓唯一分派点），断言 method=gs/thermal/dos 被路由到
eq_engine → eq_route → ed/lanczos/kpm，返回带契约键的 Result，且与**直接调** eq_route 一致。
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import numpy as np

from checks.checklib import Checker                              # noqa: E402
from src.core import runspec                                     # noqa: E402
from src.core.manybody import eq_route                           # noqa: E402
from src.core.manybody import assemble as asm                    # noqa: E402

C = Checker('equibench_dispatch')

# 假 calib：n=4 链，含复 J → 一般 H。small → 路由默认走 ed。
n = 4
rng = np.random.default_rng(11)
pairs = [dict(i=i, j=i + 1, V=float(rng.uniform(1, 5)), J=complex(rng.normal(), rng.normal()),
              A=0j, raw_V=0.0, raw_J=0j, raw_A=0j, delta={}, info={}) for i in range(n - 1)]
calib = dict(terms=['I1', 'I2', 'I3'],
             atoms=[dict(id=i, pos=(float(i), 0.0, 0.0)) for i in range(n)],
             pairs=pairs, single_basis=[])
Om = rng.uniform(0, 3, n)
delta = rng.uniform(-2, 2, n)
states = [{'label': 'atom%d' % i, 'atom_id': i} for i in range(n)]
system = dict(calib=calib, Om=Om, delta=delta, n_sites=n, is_ladder=False, batch=1)

# 分派表可达性
subs = [('engine_fn(gs,many) 可导入', runspec.engine_fn('gs', 'many').__name__ == 'run_equilibrium'),
        ('engine_fn(thermal,many) 可导入', runspec.engine_fn('thermal', 'many').__name__ == 'run_equilibrium'),
        ('manybody.solve_equilibrium 暴露', 'solve_equilibrium' in __import__('src.core.manybody', fromlist=['x']).__all__)]
C.group('② 平衡态后端经唯一分派表可达', subs)

# evolve_run：gs 路由（method=ed / auto / lanczos）与直接调 eq_route 一致
def _spec(**over):
    s = dict(version=1, nbody='many', equilibrium=True, two_body=False, want='gs',
             method='ed', calib=calib, n_atoms=n, Om=Om, delta=delta, gamma=None,
             states=states, labels=[x['label'] for x in states], kpm={}, warnings=[],
             terms=['I1', 'I2', 'I3'], template='ising')
    s.update(over)
    return s

E_direct = eq_route.solve_equilibrium(system, 'gs', backend='ed')['E0']
subs_g = []
for be in ('ed', 'auto', 'lanczos'):
    res = runspec.evolve_run(_spec(method=be), log=lambda *a: None)
    subs_g += [
        (f'evolve_run(gs,{be}) 契约键齐',
         {'want', 'backend', 'n_sites', 'E0', 'method', 'states'} <= set(res)),
        (f'evolve_run(gs,{be}) E0 == 直接调 eq_route', abs(res['E0'] - E_direct) < 1e-9),
    ]
C.group('② evolve_run(gs) → eq_engine → ed/auto/lanczos 一致 + 契约', subs_g)

# evolve_run：thermal / dos
rt = runspec.evolve_run(_spec(method='kpm', want='thermal', betas=[1.0]), log=lambda *a: None)
rd = runspec.evolve_run(_spec(method='kpm', want='dos'), log=lambda *a: None)
Hs = asm.build_H_sparse(calib, Om, delta)
E_exact_th = __import__('src.core.manybody.equilibrium', fromlist=['x']).thermal_from_h(Hs, 1.0)['E']
subs_th = [
    ('evolve_run(thermal) 返回 E/C/F 且≡精确(随机迹<3e-2)',
     'E' in rt and abs(float(rt['E'][0]) - E_exact_th) < 3e-2),
    ('evolve_run(dos) 返回 energies/rho', 'energies' in rd and 'rho' in rd and len(rd['rho']) == len(rd['energies'])),
]
C.group('② evolve_run(thermal/dos) → KPM 可达且数值合理', subs_th)

sys.exit(0 if C.summary() else 1)
