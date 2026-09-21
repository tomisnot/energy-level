# -*- coding: utf-8 -*-
"""hard 家族去重 + 运行级参数可达性（2026-09-18 引擎接口规范化第一/二刀）。

## 这条判据守什么

【病症 B】`src/core/` 里曾有两个**同名的单体 hard 引擎**：
  - `hard.py::evolve_hard`                      ← 正本（有 `max_step_hard`）
  - `evolve_twofield.py::evolve_hard`           ← 滞后拷贝（无 `max_step_hard`，
                                                   且 `from .hard import _single_loss_rates`，
                                                   是**依赖 hard.py 的旧副本**）

两个分派器还各指一个（`evolve_engine.py:36` → hard.py；`runspec.py:633` → evolve_twofield.py），
⇒ 同一个 `method='hard'` 走的是两个不同实现。

**裁决（2026-09-18，实测）**：两函数在"单体 + 纯微波"输入下终态布居
`max|Δpops| = 0.000e+00`（逐位相同），且都与 `evolve_reference` 一致
（4.8e-13）。逐行 diff 只有 11 行差异，全是"hard.py 多 `max_step_hard`" +
"拷贝反向 import hard 的私有助手"。⇒ **evolve_twofield.py 那份是滞后拷贝，已删。**

## 本判据的断言

A 包：`hard.evolve_hard` 是唯一实现 —— `evolve_twofield` 里**不许**再出现同名函数。
B 包：`solve.MAX_STEP_HARD` **可经 `build_run`/`evolve_run` 下发**（旧文案写的是
     "请经库 API 直调 hard.evolve_hard"，即入口不通；现在通了）。
"""
import inspect
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

from checks.checklib import Checker                            # noqa: E402
from arc import Rubidium87                                     # noqa: E402
from src.core import static_atom as core                       # noqa: E402
from src.core import evolve_twofield as _etf                   # noqa: E402
from src.core import hard as _hard                             # noqa: E402
from src.core.atomic_levels import make_levels                 # noqa: E402
from src.core.evolve_mixed import evolve_mixed                 # noqa: E402
from src.core.field_build import build_fields_exact            # noqa: E402
from src.core.hard import build_two_field                      # noqa: E402
from src.core.waveforms import GaussianEnv, Wave               # noqa: E402

C = Checker('hard_dedup')
cache = core.ArcCache(Rubidium87())

# ---------------- A. 同名双实现已去重 ----------------
C.group('A hard 单体只有一个实现（滞后拷贝已删）', [
    ('hard.py 有 evolve_hard',
     callable(getattr(_hard, 'evolve_hard', None))),
    ('evolve_twofield.py **没有** evolve_hard',
     not hasattr(_etf, 'evolve_hard')),
    ('hard.evolve_hard 收 max_step_hard（正本是功能超集）',
     'max_step_hard' in inspect.signature(_hard.evolve_hard).parameters),
    ('evolve_twofield 的核心函数还在（没误删）',
     all(hasattr(_etf, n) for n in
         ('evolve_two_field', 'evolve_two_field_pair', 'build_two_field',
          'build_two_field_pair', 'arbitrary_mw_group'))),
])

# ---------------- B. max_step_hard 可经 RunSpec 下发 ----------------
B_GAUSS = 21.7


def _levels(ms_s, ms_p):
    specs = [dict(n=60, l=0, j=0.5, kind='mJ', m=m) for m in ms_s]
    specs += [dict(n=60, l=1, j=1.5, kind='mJ', m=m) for m in ms_p]
    return make_levels(cache, specs)


s = _levels([+0.5], [-0.5])
nu = s[1]['E_mhz'] - s[0]['E_mhz']
w = Wave('微波', nu, GaussianEnv(70, 20), pol='π', intensity_mwcm2=1.2e-3)
meta = build_two_field(cache, s, [w])
tg = np.linspace(0.0, 60.0, 61)
psi0 = 0

r_hard = _hard.evolve_hard(s, meta, psi0, tg, rtol=1e-9, atol=1e-11)
r_mixed = evolve_mixed(s, build_fields_exact(cache, s, [w]), psi0, tg)

# `build_run` 的入参形态：登记册快照（只给 hard 关心的那几个键）
from src.core import runspec as RS                             # noqa: E402
from src.core.atomic_levels import state_key                   # noqa: E402

# ⚠ `dynamic.psi_single` 收的是**身份键**（`state_key()` 的 6 元组），
#   不是能级组的 label——喂 label 会被 `resolve_psi` 明确拒绝（不静默回落 0 号态）。
#   这是本仓"初态必须可复现"的设计，不是坑；但**调用方第一次一定会踩**。
p = {'groups.list': [dict(n=60, l=0, j=0.5), dict(n=60, l=1, j=1.5)],
     'dynamic.method': 'hard', 'dynamic.mode': 0,
     'dynamic.t_max': 60.0, 'dynamic.npts': 61,
     'dynamic.psi_single': state_key(s[0]),
     'field.B_gauss': 0.0, 'field.E_vpcm': 0.0,
     'solve.MAX_STEP_HARD': 1.0}
run = RS.build_run(cache, p, vdd=False)


def _no_ignore_warning(rn):
    return not any('已忽略' in str(x) for x in (rn.get('warnings') or []))


C.group('B solve.MAX_STEP_HARD 不再是"收了又丢"', [
    ('build_run 不报"max_step_hard 已忽略"', _no_ignore_warning(run)),
    ('engine_kwargs 里带上了 max_step_hard',
     (run.get('engine_kwargs') or {}).get('max_step_hard') == 1.0),
    ('method 落到 hard', run['method'] == 'hard'),
])

_err = ''
try:
    r_run = RS.evolve_run(run, cache=cache)
    ok_run = True
except Exception as e:                                          # noqa: BLE001
    ok_run, _err = False, '%s: %s' % (type(e).__name__, e)
C.group('B 经 RunSpec 真能跑出结果（与直调逐位同源）', [
    ('evolve_run 跑通（hard 单体）', ok_run),
    ('返回契约统一：有 pops / states 两键',
     ok_run and 'pops' in r_run and 'states' in r_run),
    ('两体键名 pops_2d / states_2d 已废弃',
     ok_run and 'pops_2d' not in r_run and 'states_2d' not in r_run),
    ('states 长度 == pops 列数',
     ok_run and len(r_run['states']) == r_run['pops'].shape[1]),
], '默认步长（无 max_step_hard 覆写）' if ok_run else ('跑失败：' + _err))
if not ok_run:
    sys.exit(1)

# 压步长：物理必须不变（这条同时验"参数真下发到了引擎"）
p2 = dict(p, **{'solve.MAX_STEP_HARD': 0.05})
run2 = RS.build_run(cache, p2, vdd=False)
r_tight = RS.evolve_run(run2, cache=cache)
d = float(np.max(np.abs(r_tight['pops'] - r_run['pops'])))
C.close('压步长（max_step_hard=0.05）不改变物理', d, 0.0, rtol=0.0, atol=1e-3)

# 更紧的硬上限也一致（< 1e-4）——从源仓 solve_max_step_hard_selfcheck 的 C2 迁来
# （那条判据整体是"接线判据"、在沙盒里被跳过，它的物理部分在这里补上）
p3 = dict(p, **{'solve.MAX_STEP_HARD': 0.02})
r_tight2 = RS.evolve_run(RS.build_run(cache, p3, vdd=False), cache=cache)
d2 = float(np.max(np.abs(r_tight2['pops'] - r_run['pops'])))
C.close('更紧的硬上限（max_step_hard=0.02）也一致', d2, 0.0, rtol=0.0, atol=1e-4)

sys.exit(0 if C.summary() else 1)
