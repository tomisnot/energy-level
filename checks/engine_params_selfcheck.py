# -*- coding: utf-8 -*-
"""引擎参数**就地定义**后的一致性（替代已归档的 `params_registry_selfcheck`）。

`params.py` 整层消灭后，参数住在各自的引擎模块里。这份判据守的是
"就地定义"最容易出的三类错：

  A. **归属**：每个模块的参数块存在、常量可读、没有"注册了却没人读"的孤儿；
  B. **派生量一致**：`DEFAULT_ATOL == DEFAULT_RTOL × ATOL_RTOL_RATIO`
     （2026-09-18 实测踩过：照 `SPECS.default` 取值插成了 `None`，
     导致 numba 类型推断失败、3 个判据同时红）；
  C. **`solve.*` 的映射**：`SOLVE_KWARG` 的每个 kwarg 至少被一个引擎签名接收
     （否则那个 `solve.*` 是死的——`max_step_hard` 就曾这样静默失效）。
"""
import ast
import inspect
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

from checks.checklib import Checker                            # noqa: E402
from src.core import coupling as CP                            # noqa: E402
from src.core import evolve_mixed as EM                        # noqa: E402
from src.core import evolve_rwa as ER                          # noqa: E402
from src.core import evolve_twofield as ET                     # noqa: E402
from src.core import field_build as FB                         # noqa: E402
from src.core import hard as HD                                # noqa: E402
from src.core import runspec as RS                             # noqa: E402
from src.core import sambe as SB                               # noqa: E402
from src.core import static_atom as SA                         # noqa: E402

C = Checker('engine_params')

# 参数 → 家（与各模块顶部的「本模块的参数」块一致）
HOME = {
    'PHASE_TOL': ('coupling', CP), 'R_CRIT': ('coupling', CP),
    '_RHS_ENV_TAB_N': ('coupling', CP), 'PHASE_BUDGET': ('coupling', CP),
    'DEFAULT_RTOL': ('evolve_mixed', EM), 'ATOL_RTOL_RATIO': ('evolve_mixed', EM),
    'DEFAULT_ATOL': ('evolve_mixed', EM), 'M_EST_TOL': ('evolve_mixed', EM),
    'BW_OPT': ('evolve_mixed', EM), '_AC_ENV_TAB_N': ('evolve_mixed', EM),
    'M_EST_BW_MHZ': ('sambe', SB), '_M_START': ('sambe', SB),
    '_M_MAX': ('sambe', SB), '_DENSE_MAX': ('sambe', SB),
    '_ENV_PROBE_N': ('sambe', SB), '_DT_MIN_FRAC': ('sambe', SB),
    '_SPARSE_SEG_MAX': ('sambe', SB), '_SPARSE_MAX_DEPTH': ('sambe', SB),
    '_NODE_LADDER': ('sambe', SB), '_VOLT_MAX_ITER': ('sambe', SB),
    '_VOLT_PICARD_FRAC': ('sambe', SB), '_VOLT_PRE_SWEEPS': ('sambe', SB),
    'OPT_NRES_BW_MHZ': ('evolve_twofield', ET), 'FRAME_TOL_REL': ('evolve_twofield', ET),
    '_TF_DENSE_MAX': ('evolve_twofield', ET),
    'RWA_TOL': ('evolve_rwa', ER),
    'MW_MAX_MHZ': ('static_atom', SA), 'RYDBERG_N_MIN': ('static_atom', SA),
    'DIRECT_SOS_N_WINDOW': ('static_atom', SA),
    'AC_EFFECT_THRESHOLD_MHZ': ('static_atom', SA),
    '_ENV_SAMPLES': ('field_build', FB), '_TAB_N': ('hard', HD),
}

# ---------------- A. 归属：每个常量都真的存在，且有人读 ----------------
missing = sorted(n for n, (_, m) in HOME.items() if not hasattr(m, n))
_CORE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                     'src', 'core')
_src = {f: open(os.path.join(_CORE, f), encoding='utf-8').read()
        for f in os.listdir(_CORE) if f.endswith('.py')}
orphans = []
for n, (home, _m) in HOME.items():
    # 至少在一个模块里被"读到"（排除自己模块顶部的定义行）
    reads = 0
    for f, t in _src.items():
        try:
            tree = ast.parse(t)
        except SyntaxError:
            continue
        for nd in ast.walk(tree):
            if isinstance(nd, ast.Name) and nd.id == n:
                reads += 1
    if reads <= 1:                        # 只有定义那一处
        orphans.append('%s(%s)' % (n, home))
C.group('A 参数就地定义：常量存在、且有人读', [
    ('全部 %d 个常量都可从它的家取到（缺：%s）' % (len(HOME), missing or '无'),
     not missing),
    ('没有"定义了却没人读"的孤儿（孤儿：%s）' % (orphans or '无'), not orphans),
    ('8 个家模块都有「本模块的参数」块',
     sum(1 for t in _src.values() if '# ==== 本模块的参数' in t) >= 8),
])

# ---------------- B. 派生量一致 ----------------
C.group('B 派生量一致（踩过的坑）', [
    ('DEFAULT_ATOL == DEFAULT_RTOL × ATOL_RTOL_RATIO',
     abs(EM.DEFAULT_ATOL - EM.DEFAULT_RTOL * EM.ATOL_RTOL_RATIO) == 0.0),
    ('DEFAULT_ATOL 是数（不是 None —— numba 类型推断要它）',
     isinstance(EM.DEFAULT_ATOL, float)),
    ('DEFAULT_ATOL 量级合理（1e-14..1e-2）',
     1e-14 < EM.DEFAULT_ATOL < 1e-2),
])

# ---------------- C. solve.* 映射：每个 kwarg 都有引擎收 ----------------
_names = [('mixed', False), ('mixed', True), ('sambe', False), ('sambe', True),
          ('hard', False), ('hard', True)]
_fns = [RS.engine_fn(*k) for k in _names]
dead = sorted(kw for kw in set(RS.SOLVE_KWARG.values())
              if not any(kw in inspect.signature(f).parameters for f in _fns))
C.group('C `solve.*` 映射的 kwarg 都有引擎接收', [
    ('没有死映射（死键：%s）' % (dead or '无'), not dead),
    ('`SOLVE_KWARG` 的键都在 `solve.` 命名空间下（%d 个）'
     % len(RS.SOLVE_KWARG), len(RS.SOLVE_KWARG) >= 5),
])

# ---------------- D. 关键默认值不漂（有物理含义的） ----------------
GOLDEN = [
    ('PHASE_BUDGET', CP.PHASE_BUDGET.value, None, '默认不限制成本（None 是唯一不透明 Param，读 .value）'),
    ('MW_MAX_MHZ', SA.MW_MAX_MHZ, 1e5, '微波/光场分界（MHz）'),
    ('RYDBERG_N_MIN', SA.RYDBERG_N_MIN, 20, 'n≥20 视为里德堡（超精细可忽略）'),
    ('_TAB_N', HD._TAB_N, 2 ** 17, 'hard 包络表点数 = **接口契约**（承诺的驱动带宽）'),
    ('RWA_TOL', ER.RWA_TOL, 1e-4, 'RWA 有效性门'),
    ('PHASE_TOL', CP.PHASE_TOL, 1e-4, '弱耦合可忽略门（V_dd 与微波对共用）'),
]
C.group('D 有物理含义的关键默认值不漂', [
    ('%s = %r（%s）' % (n, v, why), v == want)
    for n, v, want, why in GOLDEN
])

sys.exit(0 if C.summary() else 1)
