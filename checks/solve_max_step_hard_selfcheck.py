# -*- coding: utf-8 -*-
"""自检：`solve.MAX_STEP_HARD` —— 真正硬的步长上限（2026-09-12 用户裁决 #4）。

背景：`solve.MAX_STEP` 只是**初始**步长
（`dop853.py` 只在起步 `min(100·h0, h1, interval, max_step)`，之后
`h_abs *= factor` 无复核）——实测请求 0.05 ns 的 10 ns 演化里出现 **2.2 ns**
的接受步。裁决：**不改** MAX_STEP 的既有语义（改了会动全部已有数值结果），
另开一个默认关的硬键 `MAX_STEP_HARD`（空/0 = 不限制、与改前逐位一致）。

覆盖：
  A. 默认关：None / 0.0 / 不传 → 三者**逐位一致**（"默认不影响"的硬要求）
  B. 硬上限真的生效：`diag.solver.dt` 的**最大步长 ≤ 设定值**
  C. 代价：步数/求值数上升（压步长不是免费的）
  D. 精度：压步长后布居与默认路径一致到 1e-3 内（若差很多说明别的东西坏了）
  E. 接线：`solve.MAX_STEP_HARD` 在册 + god_split 映射到 kwarg `max_step_hard`
     + mixed/hard 两体入口都有该入参

用法：python checks/solve_max_step_hard_selfcheck.py（秒级，无需 ARC）
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

from checks.checklib import Checker                              # noqa: E402
from src.core.evolve_mixed import evolve_mixed_pair              # noqa: E402

C = Checker('solve_max_step_hard')
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

_Q = [(60, 0, 0.5, +0.5), (60, 0, 0.5, -0.5), (60, 1, 1.5, +1.5),
      (60, 1, 1.5, +0.5), (60, 1, 1.5, -0.5)]
S = [dict(n=n, l=l, j=j, m=m, kind='mJ', mj=m, mF=None, E_mhz=0.0,
          label='L%d' % i) for i, (n, l, j, m) in enumerate(_Q)]
N = len(S)
K1, K2 = 4 * N + 0, 0 * N + 4
import scipy.sparse as sp                                        # noqa: E402

V = np.zeros((N * N, N * N))
V[K1, K2] = V[K2, K1] = 50.0
Vsp = sp.csr_matrix(V)
t = np.linspace(0.0, 10.0, 11)
# 有相对相位 → 布居真的摆（对称组合是暗态，压步长也看不出差别）
spec = ['pair_sup', [[60, 1, 1.5, -0.5, 60, 0, 0.5, +0.5, 1.0, 0.0],
                     [60, 0, 0.5, +0.5, 60, 1, 1.5, -0.5, 0.0, 1.0]]]


def run(**extra):
    return evolve_mixed_pair(S, None, None, 0, 0, t, vdd_matrix=Vsp,
                             psi0_pair=spec, diag=True, **extra)


print('A) 默认关（None / 0.0 / 不传）三者逐位一致')
r0 = run()                                  # 不传
r1 = run(max_step_hard=None)
r2 = run(max_step_hard=0.0)
group('A 默认关：三者逐位一致 + 默认路径步长确实超 max_step')
check('A1 不传 ≡ None（逐位）',
        float(np.max(np.abs(r0['pops'] - r1['pops']))) == 0.0)
check('A2 不传 ≡ 0.0（逐位，= 与改前行为一致）',
        float(np.max(np.abs(r0['pops'] - r2['pops']))) == 0.0)
dt0 = np.asarray(r0['diag']['solver']['dt'], float)
check('A3 默认路径的步长确实**超过** max_step=0.05（证明"只管起步"）',
        float(dt0.max()) > 0.05,
        'max dt=%.3g ns（步数 %d）' % (float(dt0.max()), len(dt0)))

print('B) 硬上限生效：dt_max ≤ 设定值')
CAP = 0.1
rc = run(max_step_hard=CAP)
dtc = np.asarray(rc['diag']['solver']['dt'], float)
group('B 硬上限生效且非免费（dt_max / 步数 / nfev）')
check('B1 dt_max ≤ MAX_STEP_HARD', float(dtc.max()) <= CAP * (1 + 1e-9),
        'max dt=%.4g ≤ %.4g' % (float(dtc.max()), CAP))
check('B2 步数变多（压步长不是免费的）',
        len(dtc) > len(dt0),
        '%d 步 vs 默认 %d 步' % (len(dtc), len(dt0)))
check('B3 nfev 上升', float(rc['diag']['solver']['nfev'])
        > float(r0['diag']['solver']['nfev']),
        'nfev %d vs %d' % (rc['diag']['solver']['nfev'],
                           r0['diag']['solver']['nfev']))

print('C) 精度：压步长不改变物理')
group('C 精度：压步长不改变物理（<1e-3 / <1e-4）')
check('C1 压步长后布居与默认路径差 < 1e-3',
        float(np.max(np.abs(rc['pops'] - r0['pops']))) < 1e-3,
        'max|Δ|=%.2e' % float(np.max(np.abs(rc['pops'] - r0['pops']))))
check('C2 更紧的硬上限（0.02 ns）也一致（< 1e-4）',
        float(np.max(np.abs(run(max_step_hard=0.02)['pops']
                             - r0['pops']))) < 1e-4)

print('D) 接线：solve.MAX_STEP_HARD 在册 + 映射 + 入参')
from src.core import runspec as _rs                             # noqa: E402
from src.core import protocol as _proto                         # noqa: E402
from src.core.evolve_mixed import evolve_mixed_pair as _emp      # noqa: E402
from src.core.hard import evolve_hard_pair as _ehp               # noqa: E402
import inspect                                                   # noqa: E402


def _rs_solve():
    """v2 派生面：solve schema（键从 SOLVE_KWARG 派生，不手抄；无需建 cache）。"""
    return dict(_proto.solve_schema())

group('D 接线：在册 + 映射 + 两体入口入参 + 说明书')
check('D1 SOLVE_KWARG 含 MAX_STEP_HARD',
        'MAX_STEP_HARD' in _rs.SOLVE_KWARG)
check('D2 god_kwargs_from_params 映射到 kwarg max_step_hard',
        _rs.god_kwargs_from_params({'solve.MAX_STEP_HARD': 0.05})
        == {'max_step_hard': 0.05})
check('D3 mixed 两体入口有 max_step_hard',
        'max_step_hard' in inspect.signature(_emp).parameters)
check('D4 hard 两体入口有 max_step_hard',
        'max_step_hard' in inspect.signature(_ehp).parameters)
check('D5 MAX_STEP 说明含"起步"（只约束起步、非全程硬上限；说明书即 prompt）',
        '起步' in (_rs_solve().get('solve.MAX_STEP') or {}).get('desc', ''))

flush()
ok = C.summary()
sys.exit(0 if ok else 1)
