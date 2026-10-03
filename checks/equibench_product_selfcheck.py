# -*- coding: utf-8 -*-
"""② 封顶集成：真实 cache + 登记册快照 → runspec.build_run → evolve_run → 平衡态 Result。

证明"从产品/AI/GUI 共用的入口能正常调用新算法"：`manybody.want∈{gs,thermal,dos}` 使
`build_run(dynamic.mode==2)` 委托 eq_engine，`evolve_run` 唯一分派点路由到
ed/gpu-batch/lanczos/kpm（dmrg 需显式 ladder spec）。用真 Rubidium87 标定（首跑建表数秒）。
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import numpy as np

from checks.checklib import Checker                              # noqa: E402
from src.core import runspec as RS                               # noqa: E402
from src.core import static_atom as core                         # noqa: E402

C = Checker('equibench_product')

try:
    from arc import Rubidium87
    cache = core.ArcCache(Rubidium87())
    HAS_ARC = True
except Exception as e:
    HAS_ARC = False
    print('  [info] ARC 不可用（%r）→ SKIP 产品路径' % e)

G = (60, 0, 0.5, 0.5)
R = (60, 1, 1.5, 1.5)


def _atom(pos, Om=2 * np.pi * 2.0, dl=0.0):
    return {'g': list(G), 'r': list(R), 'pos': list(pos),
            'Omega_mhz': Om, 'delta_mhz': dl}


if not HAS_ARC:
    C.skip('产品路径 build_run→evolve_run(平衡态)', '无 ARC/cache')
    sys.exit(0 if C.summary() else 1)


def _params(want, betas=None):
    p = {'dynamic.mode': 2, 'dynamic.t_max': 50.0, 'dynamic.npts': 51,
         'field.B_gauss': 21.7, 'manybody.template': 'ising',
         'manybody.terms': ['I1', 'I2'],
         'manybody.atoms': [_atom((0., 0., 0.)), _atom((6., 0., 0.))],
         'manybody.solver': 'auto', 'manybody.want': want}
    if betas is not None:
        p['manybody.betas'] = betas
    return p


# gs：走平衡态分支 → ed/lanczos（任意几何），Result 含 E0/method/want
try:
    spec = RS.build_run(cache, _params('gs'))
    ok_build = (spec.get('nbody') == 'many' and spec.get('equilibrium') is True
                and spec.get('want') == 'gs')
    res = RS.evolve_run(spec)
    E0 = res.get('E0')
    ok_res = (res.get('method') == 'gs' and res.get('backend') in
              ('ed', 'lanczos', 'gpu-batch', 'dmrg') and np.isfinite(E0))
except Exception:
    import traceback; traceback.print_exc()
    ok_build = ok_res = False
    E0 = float('nan')

# thermal：KPM 路径
try:
    spec_t = RS.build_run(cache, _params('thermal', [1.0]))
    res_t = RS.evolve_run(spec_t)
    ok_th = (res_t.get('backend') == 'kpm' and 'E' in res_t
             and np.isfinite(np.asarray(res_t['E']).astype(float)).all())
except Exception:
    import traceback; traceback.print_exc()
    ok_th = False

subs = [
    ('manybody.want=gs → build_run 走平衡态 spec', ok_build),
    ('evolve_run(gs) → Result(method/backend/E0 有限)', ok_res),
    ('evolve_run(thermal) → KPM E(β) 有限', ok_th),
]
C.group('② 产品路径：快照→build_run→evolve_run→平衡态 Result（真 ARC）', subs)

# ---------------- 平衡态走通 GUI/AI 的三个消费点 + artifact 落地 ----------------
# 旧代码里这三处都只认含时 RunSpec：check_run 取 run['t'] KeyError、
# summarize_run 取 run['t_max'] KeyError、result_summary 只要 pops（否则报“契约被破坏”），
# 而 _land 只落 pops/t ⇒ 平衡态数组全丢。本组把四个坑钉住。
try:
    probs_gs = RS.check_run(spec)
    ok_chk = (isinstance(probs_gs, list)
              and not any('时间网格' in str(x) or '空演化' in str(x) for x in probs_gs))
except Exception:
    probs_gs, ok_chk = ['EXC'], False
try:
    smry = RS.summarize_run(spec)
    ok_sum = (smry.get('equilibrium') is True and smry.get('want') == 'gs'
              and smry.get('n_atoms') == 2)
except Exception:
    smry, ok_sum = {}, False
try:
    rsum = RS.result_summary(spec, res)
    ok_rsum = (rsum.get('ok') is True and rsum.get('equilibrium') is True
               and rsum.get('E0') is not None)
except Exception:
    rsum, ok_rsum = {}, False
# thermal 的 check_run 必须拦住空 betas（不静默用默认值）
try:
    probs_nb = RS.check_run(RS.build_run(cache, _params('thermal', [])))
    ok_beta_guard = any('betas' in str(x) for x in probs_nb)
except Exception:
    ok_beta_guard = False
# _land：平衡态数组必须进 artifact 并能取回
try:
    import tempfile
    from pathlib import Path
    from mecha_v2.bridge import ElOps

    class _Stub:
        root = Path(tempfile.mkdtemp(prefix='eq_land_'))

    _res_arr = dict(res)
    _res_arr.update(betas=np.array([0.5, 1.0]), E=np.array([1.0, 0.5]),
                    energies=np.array([-1.0, 0.0, 1.0]), rho=np.array([0.1, 0.9, 0.2]))
    aid = ElOps._land(_Stub(), spec, _res_arr)
    loaded = ElOps.load_artifact(_Stub(), aid)
    arrs = loaded[1] if loaded else {}
    ok_land = all(k in arrs for k in ('betas', 'E', 'energies', 'rho', 'n_atom'))
except Exception:
    import traceback; traceback.print_exc()
    ok_land = False
C.group('平衡态走通 GUI/AI 消费点（check_run/summarize_run/result_summary/_land）', [
    ('check_run(平衡态) 不报 KeyError/不误报含时判据（%s）' % (probs_gs,), ok_chk),
    ('summarize_run(平衡态) 报 equilibrium/want/n_atoms（%s）'
     % {k: smry.get(k) for k in ('equilibrium', 'want', 'n_atoms')}, ok_sum),
    ('result_summary(平衡态 gs) ok 且带 E₀（%s）'
     % {k: rsum.get(k) for k in ('ok', 'backend', 'E0')}, ok_rsum),
    ('thermal 空 betas 被 check_run 拦住（不静默用默认）', ok_beta_guard),
    ('_land 把平衡态数组落进 artifact 且可取回', ok_land),
])

sys.exit(0 if C.summary() else 1)
