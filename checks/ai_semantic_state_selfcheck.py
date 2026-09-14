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

from checks.checklib import Checker                             # noqa: E402
from src.ai import codeact                                      # noqa: E402
from src.ai.sim_state import groups_delta                       # noqa: E402

C = Checker('ai_semantic_state')


print('1) 声明式组操作的判定核心 groups_delta')
cur = [(60, 0, 0.5), (60, 1, 1.5), (6, 1, 1.5), (58, 4, 3.5)]
want = [(60, 1, 1.5), (58, 4, 3.5), (58, 4, 4.5), (59, 2, 2.5)]
a, r = groups_delta(cur, want)
a2, r2 = groups_delta(cur, cur)
a3, r3 = groups_delta([], want)
a4, r4 = groups_delta(cur, [(60, 1, 1.5), (60, 1, 1.5)])
# 本会话的真实事故：想要 {60S,60P_3/2,58G_7/2,58G_9/2,59D_5/2,58F_5/2}，
# 旧实现按 '60P' 前缀先匹配到 60P_3/2 而删错组
cur_acc = [(60, 0, 0.5), (60, 1, 1.5), (6, 1, 1.5), (58, 4, 3.5),
           (59, 2, 2.5), (58, 3, 2.5), (60, 1, 0.5)]
target = [(60, 0, 0.5), (60, 1, 1.5), (58, 4, 3.5), (58, 4, 4.5),
          (59, 2, 2.5), (58, 3, 2.5)]
a5, r5 = groups_delta(cur_acc, target)
# 不变式包："声明式组操作绝不删错/漏加"是**一个**不变式：增删判定、有序性、
# 幂等、去重、事故回归都是它的子条件。
C.group('1 groups_delta 判定核心（增删/有序/幂等/去重/事故回归）', [
    ('待加 = %s' % (a,), a == [(58, 4, 4.5), (59, 2, 2.5)]),
    ('待删确定有序 = %s' % (r,), r == sorted([(60, 0, 0.5), (6, 1, 1.5)])),
    ('幂等：同集合 → 无增删', a2 == [] and r2 == []),
    ('空基组 → 全部新增且保持 want 顺序',
     a3 == [(60, 1, 1.5), (58, 4, 3.5), (58, 4, 4.5), (59, 2, 2.5)]),
    ('重复项去重 + 其余移除',
     a4 == [] and sorted(r4) == sorted([(60, 0, 0.5), (6, 1, 1.5),
                                        (58, 4, 3.5)])),
    ('事故回归：60P_3/2 与 58F_5/2 **不被误删**（待删 %s）' % (r5,),
     (60, 1, 1.5) not in r5 and (58, 3, 2.5) not in r5),
    ('事故回归：只删 60P_1/2 与 6P，只加 58G_9/2',
     (60, 1, 0.5) in r5 and (6, 1, 1.5) in r5 and a5 == [(58, 4, 4.5)]),
])

print('2) SimState 暴露语义操作')
from src.ai.sim_state import SimState                            # noqa: E402
C.group('2 SimState 暴露语义操作', [
    ('SimState.%s 存在' % name, callable(getattr(SimState, name, None)))
    for name in ('set_groups', 'list_levels', 'set_initial_state', 'get_config')
])

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
C.group('3 CodeAct 命名空间注入四个语义操作（旧 state_mod 不报错）', [
    ('命名空间含 %s' % name, name in ns)
    for name in ('set_groups', 'list_levels', 'set_initial_state', 'get_config')
] + [
    ('旧 state_mod（无这些方法）不注入、不报错', 'set_groups' not in ns_old),
])


print('4) 二期 B：事前成本粗估 estimate_pair_cost（纯函数，可 headless 测）')
from src.ai.sim_state import estimate_pair_cost                  # noqa: E402
base = estimate_pair_cost(36, 300.0, 201)
# 不变式包：cost 模型 = "标定锚点 + 三条标度律"，同属一个事前成本不变式。
C.group('4 estimate_pair_cost 事前成本粗估（锚点 + 标度律）', [
    ('标定锚点：N=36/t=300ns ≈ 120 s（得 %.1f s）' % base,
     abs(base - 120.0) < 1e-6),
    ('dim∝N² → 成本∝N⁴：N=48 至少 3× 于 N=36',
     estimate_pair_cost(48, 300.0, 201) > 3.0 * base),
    ('对 t_max 近似线性：600 ns ≈ 2× 300 ns',
     estimate_pair_cost(36, 600.0, 201) > 1.9 * base),
    ('N 减半 → 成本降 ~16×',
     estimate_pair_cost(18, 300.0, 201) < 0.2 * base),
])

print('5) CodeAct 注入 list_jobs（服务端统一作业视图）')
ns2 = codeact._make_namespace(_S(), '.', {'seq': [0]}, lambda j: {},
                              lambda j: {}, lambda j, s=False: {},
                              t_jobs=lambda limit=20: {'ok': True, 'jobs': []})
C.group('5 CodeAct 注入 list_jobs（服务端统一作业视图）', [
    ('命名空间含 list_jobs', 'list_jobs' in ns2),
    ('list_jobs 可调用', bool(ns2['list_jobs']()['ok'])),
])


print('6) 二期 C/A10：变更清单 _diff_changes（P3 可归因，纯函数）')
from src.ai.codeact import _diff_changes                          # noqa: E402
before = {'params': {'dynamic.t_max': 100.0, 'groups.list': [
    {'n': 60, 'l': 0, 'j': 0.5}, {'n': 6, 'l': 1, 'j': 1.5}]}}
after = {'params': {'dynamic.t_max': 250.0, 'groups.list': [
    {'n': 60, 'l': 0, 'j': 0.5}, {'n': 58, 'l': 4, 'j': 3.5}]}}
d = _diff_changes(before, after, ['job7'])
d2 = _diff_changes(before, before, [])
# 不变式包：变更清单的"可归因"是一个不变式（参数 / 组增删 / 作业 / 无变更全空）。
C.group('6 _diff_changes 变更清单可归因（参数/组增删/作业/无变更）', [
    ('参数变更被识别：%s' % (d['params'],),
     d['params'] == [{'param': 'dynamic.t_max', 'old': 100.0, 'new': 250.0}]),
    ('组增删被识别：+%s −%s' % (d['groups_added'], d['groups_removed']),
     d['groups_added'] == [[58, 4, 3.5]]
     and d['groups_removed'] == [[6, 1, 1.5]]),
    ('本脚本起的作业被列出', d['jobs'] == ['job7']),
    ('无变更 → 全空',
     d2['params'] == [] and d2['groups_added'] == []
     and d2['groups_removed'] == []),
])

print('7) run_code 返回 changes[]（本脚本改了什么，P3）')
import tempfile                                                 # noqa: E402
import os as _os                                                 # noqa: E402


class _S2:
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
        return {'params': {'dynamic.t_max': 123.0}}

    def restore_snapshot(self, snap):
        return None


_root = tempfile.mkdtemp(prefix='ai_chk_')
out = codeact.run_code('x = 1 + 1', _S2(), _root, {'seq': [0]},
                       lambda jid: {'ok': True}, lambda jid: {'done': True},
                       lambda jid, s=False: {'ok': True})
C.group('7 run_code 返回 changes[]（本脚本改了什么，P3）', [
    ('changes 结构齐备：%s' % sorted((out.get('changes') or {}).keys()),
     isinstance(out.get('changes'), dict) and 'params' in out['changes']
     and 'jobs' in out['changes']),
])

ok = C.summary()
sys.exit(0 if ok else 1)
