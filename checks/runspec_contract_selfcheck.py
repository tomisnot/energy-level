# -*- coding: utf-8 -*-
"""`runspec` 的接口契约：**缺键/非法值不静默兜底**（与 `resolve_psi` 同一纪律）。

背景（2026-09-18 审查实测）：`p.get(key) or 默认` 把"键不存在"和"值就是默认"
混成一件事 ⇒ 键名打错就静默跑一个空演化。实测过的三个洞：

  ① `dynamic.t_max` 缺失 → t 网格跨度 **0 ns**（看起来跑成功了）
  ② `dynamic.npts`  缺失 → **2 点**（跨度对、分辨率塌）
  ③ `dynamic.decay_index=7` → 静默**无衰变**

`check_run` 抓不到① ②（`max(npts,2)` 保证 `len(t) ≥ 2`）。
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

from checks.checklib import Checker                            # noqa: E402
from arc import Rubidium87                                     # noqa: E402
from src.core import runspec as RS                             # noqa: E402
from src.core import static_atom as core                       # noqa: E402
from src.core.atomic_levels import state_key                   # noqa: E402

C = Checker('runspec_contract')
cache = core.ArcCache(Rubidium87())

GROUPS = [dict(n=60, l=0, j=0.5), dict(n=60, l=1, j=1.5)]
_st = RS.build_states(cache, GROUPS, 21.7, 0.0)
BASE = {'groups.list': GROUPS,
        'field.B_gauss': 21.7, 'field.E_vpcm': 0.0,
        'dynamic.mode': 0, 'dynamic.method': 'mixed',
        'dynamic.t_max': 100.0, 'dynamic.npts': 101,
        'dynamic.psi_single': state_key(_st[0]), 'dynamic.decay_index': 1}


def raises(patch):
    """给快照打补丁后 build_run —— 返回 (是否报错, 消息)。"""
    p = dict(BASE)
    p.update(patch)
    try:
        RS.build_run(cache, p, vdd=False)
        return False, ''
    except Exception as e:                                      # noqa: BLE001
        return True, '%s: %s' % (type(e).__name__, str(e))


# ---------------- A. 必填键缺失/打错 → 报错 ----------------
bad_t, msg_t = raises({'dynamic.t_max': None})
bad_t2, _ = raises({'dynamic.tmax': 100.0, 'dynamic.t_max': None})
bad_n, msg_n = raises({'dynamic.npts': None})
bad_zero, msg_zero = raises({'dynamic.t_max': 0.0})
bad_one, _ = raises({'dynamic.npts': 1})
C.group('A 必填键缺失/非法 → 报错（不静默跑空演化）', [
    ('t_max 缺失 → 报错', bad_t),
    ('t_max 键名打错（dynamic.tmax）→ 报错', bad_t2),
    ('npts 缺失 → 报错', bad_n),
    ('t_max = 0 → 报错（0 ns 演化没意义）', bad_zero),
    ('npts = 1 → 报错（网格 < 2 点）', bad_one),
    ('报错信息点名键名', 'dynamic.t_max' in msg_t),
])

# ---------------- B. 非法 decay_index → 报错 ----------------
bad_d, msg_d = raises({'dynamic.decay_index': 7})
ok_d0, _ = raises({'dynamic.decay_index': 0})
ok_d1, _ = raises({'dynamic.decay_index': 1})
ok_d2, _ = raises({'dynamic.decay_index': 2})
C.group('B 非法 decay_index → 报错（合法 0/1/2 仍可跑）', [
    ('decay_index = 7 → 报错', bad_d),
    ('消息点名合法取值', '0' in msg_d and '1' in msg_d and '2' in msg_d),
    ('0（自动）/ 1（禁用）/ 2（自定义）都可跑',
     not ok_d0 and not ok_d1 and not ok_d2),
])

# ---------------- C. 未知 method → 报错 ----------------
bad_m, msg_m = raises({'dynamic.method': 'ip4'})
C.group('C 未知 method → 报错（不静默改道）', [
    ('method=ip4 → 报错', bad_m),
    ('消息列出合法组合', 'mixed' in msg_m and 'hard' in msg_m),
])

# ---------------- D. 合法快照的回归 ----------------
run = RS.build_run(cache, BASE, vdd=False)
res_ok = (len(run['t']) == 101 and abs(float(run['t'][-1]) - 100.0) < 1e-9
          and run['method'] == 'mixed' and not run['two_body'])
C.group('D 合法快照不受影响（回归）', [
    ('t 网格 101 点、跨度 100 ns', len(run['t']) == 101
     and abs(float(run['t'][-1]) - 100.0) < 1e-9),
    ('method / two_body 正确', run['method'] == 'mixed' and not run['two_body']),
    ('warnings 里没有"已忽略"（本快照不该有）',
     not any('已忽略' in str(w) for w in run['warnings'])),
])

# ---------------- E. 死代码不再回来 ----------------
_SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    'src', 'core', 'runspec.py')
_src = open(_SRC, encoding='utf-8').read()
C.group('E 已删的死代码不回来', [
    ('`_WAVE_ATOMS` 不存在', '_WAVE_ATOMS' not in _src),
    ('`build_run` 不再搬 `param_overrides`', '_src.count' and
     _src.count("ek['param_overrides']") == 0),
    ('`_required` 助手在', 'def _required(' in _src),
])

sys.exit(0 if C.summary() else 1)
