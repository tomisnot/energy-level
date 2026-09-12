# -*- coding: utf-8 -*-
"""自检：二期 A 语义状态面（声明式组操作 + 语义初态 + 配置快照）。

headless 可测的部分：
  · groups_delta 的判定正确性与确定性（这是"删错组/漏加组"的根治点）
  · SimState 暴露了 set_groups / list_levels / set_initial_state / get_config
  · CodeAct 命名空间注入了这四个语义操作（且旧 state_mod 缺失时不报错）
跑法：python checks/ai_semantic_state_selfcheck.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

from src.ai import codeact                                      # noqa: E402
from src.ai.sim_state import groups_delta                       # noqa: E402

FAIL = []


def chk(cond, msg):
    print(('  OK   ' if cond else '  FAIL ') + msg)
    if not cond:
        FAIL.append(msg)


print('1) 声明式组操作的判定核心 groups_delta')
cur = [(60, 0, 0.5), (60, 1, 1.5), (6, 1, 1.5), (58, 4, 3.5)]
want = [(60, 1, 1.5), (58, 4, 3.5), (58, 4, 4.5), (59, 2, 2.5)]
a, r = groups_delta(cur, want)
chk(a == [(58, 4, 4.5), (59, 2, 2.5)], '待加 = %s' % (a,))
chk(r == sorted([(60, 0, 0.5), (6, 1, 1.5)]), '待删确定有序 = %s' % (r,))
a2, r2 = groups_delta(cur, cur)
chk(a2 == [] and r2 == [], '幂等：同集合 → 无增删')
a3, r3 = groups_delta([], want)
chk(a3 == [(60, 1, 1.5), (58, 4, 3.5), (58, 4, 4.5), (59, 2, 2.5)],
    '空基组 → 全部新增且保持 want 顺序')
a4, r4 = groups_delta(cur, [(60, 1, 1.5), (60, 1, 1.5)])
chk(a4 == [] and sorted(r4) == sorted([(60, 0, 0.5), (6, 1, 1.5),
                                       (58, 4, 3.5)]),
    '重复项去重 + 其余移除')
# 本会话的真实事故：想要 {60S,60P_3/2,58G_7/2,58G_9/2,59D_5/2,58F_5/2}，
# 旧实现按 '60P' 前缀先匹配到 60P_3/2 而删错组
cur_acc = [(60, 0, 0.5), (60, 1, 1.5), (6, 1, 1.5), (58, 4, 3.5),
           (59, 2, 2.5), (58, 3, 2.5), (60, 1, 0.5)]
target = [(60, 0, 0.5), (60, 1, 1.5), (58, 4, 3.5), (58, 4, 4.5),
          (59, 2, 2.5), (58, 3, 2.5)]
a5, r5 = groups_delta(cur_acc, target)
chk((60, 1, 1.5) not in r5 and (58, 3, 2.5) not in r5,
    '事故回归：60P_3/2 与 58F_5/2 **不被误删**（待删 %s）' % (r5,))
chk((60, 1, 0.5) in r5 and (6, 1, 1.5) in r5 and a5 == [(58, 4, 4.5)],
    '事故回归：只删 60P_1/2 与 6P，只加 58G_9/2')

print('2) SimState 暴露语义操作')
from src.ai.sim_state import SimState                            # noqa: E402
for name in ('set_groups', 'list_levels', 'set_initial_state', 'get_config'):
    chk(callable(getattr(SimState, name, None)), 'SimState.%s 存在' % name)

print('3) CodeAct 命名空间注入这四个语义操作')


class _S:
    def get_state(self, scope=None):
        return {}

    def summary(self):
        return {}

    def set_param(self, *a, **k):
        return {}

    def add_level_group(self, *a, **k):
        return {}

    def remove_level_group(self, *a, **k):
        return {}

    def set_waves(self, *a, **k):
        return {}

    def to_dict(self):
        return {}

    def set_groups(self, *a, **k):
        return {'ok': True}

    def list_levels(self, *a, **k):
        return {'ok': True}

    def set_initial_state(self, *a, **k):
        return {'ok': True}

    def get_config(self):
        return {'ok': True}


ns = codeact._make_namespace(_S(), '.', {'seq': [0]}, lambda j: {}, lambda j: {},
                             lambda j, s=False: {})
for name in ('set_groups', 'list_levels', 'set_initial_state', 'get_config'):
    chk(name in ns, '命名空间含 %s' % name)
class _Old:
    """只有旧接口的 state_mod（无二期语义操作）——注入时不得报错。"""
    def get_state(self, scope=None):
        return {}

    def summary(self):
        return {}

    def set_param(self, *a, **k):
        return {}

    def add_level_group(self, *a, **k):
        return {}

    def remove_level_group(self, *a, **k):
        return {}

    def set_waves(self, *a, **k):
        return {}

    def to_dict(self):
        return {}


ns_old = codeact._make_namespace(_Old(), '.', {'seq': [0]}, lambda j: {},
                                 lambda j: {}, lambda j, s=False: {})
chk('set_groups' not in ns_old, '旧 state_mod（无这些方法）不注入、不报错')


print('4) 二期 B：事前成本粗估 estimate_pair_cost（纯函数，可 headless 测）')
from src.ai.sim_state import estimate_pair_cost                  # noqa: E402
base = estimate_pair_cost(36, 300.0, 201)
chk(abs(base - 120.0) < 1e-6, '标定锚点：N=36/t=300ns ≈ 120 s（得 %.1f s）' % base)
chk(estimate_pair_cost(48, 300.0, 201) > 3.0 * base,
    'dim∝N² → 成本∝N⁴：N=48 至少 3× 于 N=36')
chk(estimate_pair_cost(36, 600.0, 201) > 1.9 * base,
    '对 t_max 近似线性：600 ns ≈ 2× 300 ns')
chk(estimate_pair_cost(18, 300.0, 201) < 0.2 * base,
    'N 减半 → 成本降 ~16×')

print('5) CodeAct 注入 list_jobs（服务端统一作业视图）')
ns2 = codeact._make_namespace(_S(), '.', {'seq': [0]}, lambda j: {},
                              lambda j: {}, lambda j, s=False: {},
                              t_jobs=lambda limit=20: {'ok': True, 'jobs': []})
chk('list_jobs' in ns2, '命名空间含 list_jobs')
chk(ns2['list_jobs']()['ok'], 'list_jobs 可调用')

print('\n结果：%s' % ('全部通过' if not FAIL else '%d 项失败' % len(FAIL)))
sys.exit(1 if FAIL else 0)
