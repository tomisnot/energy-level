# -*- coding: utf-8 -*-
"""引擎参数注册表自检（t81）：默认值 / 覆写快照 / 隔离 / 引擎真的吃到覆写。

计数模型（2026-09-14 改为**不变式包**，见 checklib）：默认摘要只数包（项数），
包内子条件单独累计为 `SUB`。本文件按自身分节 A–F 打包，每节 1 包，
判据表达式原样搬进包内 `subs`（覆盖面不变，`SUB` == 改前项数）。

覆盖：
  A. 注册表一致性：DEFAULTS 与 SPECS 对齐、无重复名、别名转发、
     派生量（DEFAULT_ATOL）随依赖变、coerce 类型化与报错；
  B. 覆写语义：线程本地栈、嵌套就近优先、退出还原、未知名 KeyError；
  C. 线程隔离：另一个线程看不到本线程的覆写；
  D. 引擎真的吃到覆写（端到端，不是"只改了字典"）：
       · PHASE_BUDGET 覆写 → mixed 的微波候选窗与保留对数变化；
       · _RHS_ENV_TAB_N 覆写 → pack_rhs 表尺寸变化；
       · PHASE_TOL 覆写 → 判据阈值变化（保留集变化）；
       · _TAB_N 覆写 → hard 建表的点数变化（走 hard 建场路径）。
  E. UI 侧接口：params.specs() 覆盖全部可调参数、tier 分组非空。
  F. 控制链路契约（源码级）：引擎页由 SPECS 生成、覆写只有一个来源。

用法：python checks\\params_registry_selfcheck.py
"""
import os
import sys
import threading

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

from checks.checklib import Checker                              # noqa: E402
from src.core import params as P                                 # noqa: E402

C = Checker('params_registry')

# ---------------------------------------------------------------
# A. 注册表一致性
# ---------------------------------------------------------------
_names = [s.name for s in P.SPECS]
_A = []
_A.append(('A1 SPECS 无重复名', len(_names) == len(set(_names))))
_A.append(('A2 DEFAULTS 覆盖 SPECS（+派生量）',
           set(_names) <= set(P.DEFAULTS) and 'DEFAULT_ATOL' in P.DEFAULTS))
_A.append(('A3 每个可调 spec 的 default 与其 DEFAULTS 项一致',
           all(P.DEFAULTS[s.name] == s.default for s in P.SPECS if s.settable)))
_A.append(('A4 别名转发：POP_PHASE_TOL ≡ PHASE_TOL',
           P.get('POP_PHASE_TOL') == P.get('PHASE_TOL')
           and P.POP_PHASE_TOL == P.PHASE_TOL))
_A.append(('A5 模块属性回落（PEP 562）：params.X == get(X)',
           all(getattr(P, s.name) == P.get(s.name) for s in P.SPECS)))
_A.append(('A6 派生量：DEFAULT_ATOL = DEFAULT_RTOL × ATOL_RTOL_RATIO',
           abs(P.get('DEFAULT_ATOL')
               - P.get('DEFAULT_RTOL') * P.get('ATOL_RTOL_RATIO')) < 1e-18))
with P.override(DEFAULT_RTOL=1e-6):
    _A.append(('A7 派生量随依赖变（覆写 rtol → atol 跟着变）',
               abs(P.get('DEFAULT_ATOL') - 1e-6 * P.get('ATOL_RTOL_RATIO'))
               < 1e-18))
_A.append(('A8 覆写退出后派生量还原',
           abs(P.get('DEFAULT_ATOL')
               - P.get('DEFAULT_RTOL') * P.get('ATOL_RTOL_RATIO')) < 1e-18))

_v, _e = P.coerce('_M_MAX', '12')
_A.append(('A9 coerce int：容忍 "1e4" 写法',
           P.coerce('_DENSE_MAX', '1e4')[0] == 10000))
_A.append(('A10 coerce float', P.coerce('PHASE_TOL', '2.5e-5')[0] == 2.5e-5))
_A.append(('A11 coerce tuple',
           P.coerce('_NODE_LADDER', '16,32,64')[0] == (16, 32, 64)))
_A.append(('A12 coerce 空串 → (None, None)（= 用默认）',
           P.coerce('PHASE_TOL', '  ') == (None, None)))
_A.append(('A13 coerce 报错带名字',
           P.coerce('_M_MAX', 'abc')[0] is None
           and '_M_MAX' in P.coerce('_M_MAX', 'abc')[1]))
_A.append(('A14 coerce 未知名报错', P.coerce('NOT_A_PARAM', '1')[0] is None))
C.group('A 注册表一致性（SPECS / DEFAULTS / 别名 / 派生量 / coerce）', _A)

# ---------------------------------------------------------------
# B. 覆写语义
# ---------------------------------------------------------------
_d0 = P.get('PHASE_TOL')
with P.override(PHASE_TOL=1e-2):
    _in1 = P.get('PHASE_TOL')
    with P.override({'PHASE_TOL': 1e-3, 'R_CRIT': 0.5}):
        _in2 = P.get('PHASE_TOL')
        _rc = P.get('R_CRIT')
    _back1 = P.get('PHASE_TOL')
_B = [
    ('B1 覆写生效', _in1 == 1e-2),
    ('B2 嵌套就近优先', _in2 == 1e-3 and _rc == 0.5),
    ('B3 内层退出还原到外层', _back1 == 1e-2),
    ('B4 全部退出还原默认', P.get('PHASE_TOL') == _d0),
]
try:
    with P.override(NOT_A_PARAM=1):
        pass
    _kerr = False
except KeyError:
    _kerr = True
_B.append(('B5 未知名 → KeyError（早失败，不静默无效）', _kerr))
with P.override(PHASE_TOL=1e-5):
    _co = P.current_overrides()
_B.append(('B6 current_overrides 反映栈内容（退出后为空）',
           _co.get('PHASE_TOL') == 1e-5 and P.current_overrides() == {}))
C.group('B 覆写语义（栈 / 嵌套就近 / 还原 / 未知名）', _B)

# ---------------------------------------------------------------
# C. 线程隔离
# ---------------------------------------------------------------
_seen = {}


def _worker():
    _seen['other'] = P.get('PHASE_TOL')


with P.override(PHASE_TOL=1e-1):
    t = threading.Thread(target=_worker)
    t.start()
    t.join()
    _seen['main'] = P.get('PHASE_TOL')
C.group('C 线程隔离（别的线程看不到本线程覆写）',
        [(f'C1 别的线程={_seen.get("other")} / 本线程={_seen.get("main")}'
          f'（期望 {_d0} / 0.1）',
          _seen.get('other') == _d0 and _seen.get('main') == 1e-1)])

# ---------------------------------------------------------------
# D. 引擎端到端：覆写真的改变了引擎行为
# ---------------------------------------------------------------
from arc import Rubidium87                                       # noqa: E402
from src.core import static_atom as core                         # noqa: E402
from src.core.atomic_levels import make_levels                   # noqa: E402
from src.core.waveforms import Wave, GaussianEnv                 # noqa: E402
from src.core.field_build import build_fields_exact, _group_by_freq  # noqa: E402
from src.core.evolve_mixed import _MixedSetup                    # noqa: E402
from src.core.coupling import pack_rhs, mw_cap_mhz               # noqa: E402

cache = core.ArcCache(Rubidium87())
states = make_levels(cache, [
    dict(n=60, l=0, j=0.5, kind='mJ', m=+0.5),
    dict(n=60, l=0, j=0.5, kind='mJ', m=-0.5),
    dict(n=60, l=1, j=1.5, kind='mJ', m=+0.5),
    dict(n=60, l=1, j=1.5, kind='mJ', m=-0.5),
    dict(n=59, l=2, j=2.5, kind='mJ', m=-0.5),
    dict(n=59, l=2, j=2.5, kind='mJ', m=-1.5),
    dict(n=58, l=3, j=3.5, kind='mJ', m=-2.5),
    dict(n=58, l=4, j=4.5, kind='mJ', m=-3.5),
])
E = np.array([s['E_mhz'] for s in states], float)
w = Wave('微波', 7500.0, GaussianEnv(200, 40), pol='σ-', intensity_mwcm2=1.0)
fields = build_fields_exact(cache, states, [w])
grps_mw = _group_by_freq(fields)
T = 400.0


def _setup():
    return _MixedSetup(E, [], grps_mw, len(states), t_end_ns=T)


_D = []
_D.append(('D1 默认（成本预算空）→ 绝对上界：不因成本删真实耦合',
           abs(mw_cap_mhz(T) - float(P.get('MW_MAX_MHZ'))) < 1e-6))
_s_lo = _setup()
_cap_hi = mw_cap_mhz(T, 1e6)                # 显式成本档（运行级参数）
_s_hi = _MixedSetup(E, [], grps_mw, len(states), t_end_ns=T,
                    phase_budget=1e6)
_D.append((f'D2 显式成本档 → 候选窗更宽（成本是运行级选择，不是引擎默认）'
           f'（{_cap_hi:.0f} vs {_s_lo.mw_cap:.0f}）',
           _cap_hi > _s_lo.mw_cap))
_D.append((f'D3 显式成本档 → 保留对数 ≥ 默认（窗更宽不丢对）'
           f'（{_s_hi.mw_pair_stats} vs {_s_lo.mw_pair_stats}）',
           _s_hi.mw_pair_stats['keep'] >= _s_lo.mw_pair_stats['keep']))
with P.override(MW_MAX_MHZ=1e3):            # 绝对上界压到 1 GHz → 几乎全丢
    _s_tiny = _setup()
_D.append((f'D4 MW_MAX_MHZ 覆写 → 上界收紧后保留对数下降'
           f'（{_s_tiny.mw_pair_stats} vs {_s_lo.mw_pair_stats}）',
           _s_tiny.mw_pair_stats['keep'] < _s_lo.mw_pair_stats['keep']))

_pairs = [(0, 1, 1.0j, 0.0, 1, 7500.0, w.waveform)]
_n_def = pack_rhs(_pairs, 10.0)['env_tab'].shape[1]
with P.override(_RHS_ENV_TAB_N=1234):
    _n_ovr = pack_rhs(_pairs, 10.0)['env_tab'].shape[1]
_D.append((f'D5 _RHS_ENV_TAB_N 覆写 → pack_rhs 表尺寸跟随'
           f'（{_n_def} / {_n_ovr}）',
           _n_def == 20001 and _n_ovr == 1234))

from src.core.coupling import mw_pair_keep                         # noqa: E402
_strong = mw_pair_keep(1e3, 0.0, 400.0, cap_mhz=1e6)
with P.override(PHASE_TOL=1e6):        # 阈值拉到天上 → 本该留的强对也丢
    _all_drop = mw_pair_keep(1e3, 0.0, 400.0, cap_mhz=1e6)
with P.override(PHASE_TOL=1e-30):      # 阈值压到地下 → 极弱对也留
    _all_keep = mw_pair_keep(1e-6, 1e3, 400.0, cap_mhz=1e6)
_D.append((f'D6 PHASE_TOL 覆写生效于判据（双向）'
           f'（默认={_strong} 抬高={_all_drop} 压低={_all_keep}）',
           _strong is True and _all_drop is False and _all_keep is True))

# 覆写不泄漏：D 段里用了 4 个 with，退出后必须全部还原
_D.append((f'D7 覆写不泄漏（D 段退出后无残留 {P.current_overrides()}）',
           P.current_overrides() == {}))
C.group('D 引擎真的吃到覆写（端到端：候选窗 / 表尺寸 / 判据 / 无泄漏）', _D)

# ---------------------------------------------------------------
# E. UI 侧接口
# ---------------------------------------------------------------
_sett = P.specs()
_n_settable = sum(1 for s in P.SPECS if s.settable)
_E = [
    (f'E1 specs() = 全部可调参数（排除派生/别名）（{len(_sett)} 项 / '
     f'可调 {_n_settable} / 总 {len(P.SPECS)}）',
     len(_sett) == _n_settable == len(P.SPECS) - 1),
    ('E2 每个 tier 都有参数（UI 分组不留空组）',
     all(P.specs(t) for t, _ in P.TIERS)),
    ('E3 派生量不进 UI（DEFAULT_ATOL 只读）',
     'DEFAULT_ATOL' not in {s.name for s in _sett}),
    ('E4 别名不进 UI', 'POP_PHASE_TOL' not in {s.name for s in _sett}),
    ('E5 kind 只用 float/int/tuple',
     {s.kind for s in P.SPECS} <= {'float', 'int', 'tuple'}),
]
C.group('E UI 侧接口（specs / tier / 派生与别名不进 UI / kind）', _E)

# ---------------------------------------------------------------
# F. 控制链路契约（源码级断言——防止有人把"注册表驱动"改回硬编码）
# ---------------------------------------------------------------
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _read(rel):
    return open(os.path.join(_ROOT, rel), encoding='utf-8').read()


_insp = _read(os.path.join('src', 'ui', 'solver_inspector.py'))
_wk = _read(os.path.join('src', 'ui', 'evolve_worker.py'))
_dyn = _read(os.path.join('src', 'ui', 'ui_dynamic.py'))
_F = [
    ('F1 引擎参数页由 params.SPECS 生成（不是手写清单）',
     '_P.specs(' in _insp and '_E_PARAMS' not in _insp),
    ('F2 检视器有 param_overrides()/重置接口',
     'def param_overrides(' in _insp
     and 'def reset_param_overrides(' in _insp),
    ('F3 引擎覆写只有一个来源（会话真值；god_values 不再自己塞）',
     "ek['param_overrides'] = po" not in _insp
     and "engine_overrides" in _dyn
     and "def param_errors(" in _insp),
    ('F3b 非法输入不弹模态（只标红 + 计数 + 状态栏）',
     'QMessageBox.warning(self, \'引擎参数\'' not in _insp),
    ('F4 worker 在 params.override() 快照里跑引擎（单/两体都吃）',
     _wk.count('params as _P') >= 1
     and _wk.count("pop('param_overrides'") == 2
     and _wk.count('with _P.override(ovr):') == 2),
    ('F5 ui_dynamic 对**所有**引擎都传 param_overrides（不再只管 mixed）',
     _dyn.count("gk.pop('param_overrides'") == 2
     and _dyn.count("engine_kwargs['param_overrides'] = pk") == 2),
    ('F6 硬编码容差已从 worker 移除（走注册表）',
     'rtol=1e-8, atol=1e-10' not in _wk
     and "ek.setdefault('tol', 1e-8)" not in _wk),
]
C.group('F 控制链路契约（源码级：SPECS 驱动 / 单一覆写来源 / 无硬编码容差）', _F)

sys.exit(0 if C.summary() else 1)
