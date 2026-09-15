# -*- coding: utf-8 -*-
"""自检：并行扫描的 **AI 可及面接线**（工具注册 / CodeAct 白名单 / 归因）。

为什么单独测这一层：坑 19 的教训是"写入侧通了 ≠ 引擎侧通了"，反过来也成立
——**引擎侧通了 ≠ AI 够得着**。本轮踩过的同款：`set_waves` 支持 arb 但 AI 面
没入口、`_pending_psi_keys` 写进去了运行期却读不到。所以接线要单独钉住。

覆盖（每段 1 个不变式包）：
  A MCP 工具面：`submit_sweep / sweep_status / sweep_result / cancel_sweep /
    list_sweeps` 五个名字都在，描述里含关键用法词（axes/登记册路径/workers/
    record），且工具总数只增不减
  B CodeAct 白名单：五个名字都注入命名空间；`run_sweep_and_wait` 是 callable；
    **未接线时 `submit_sweep` 返回 ok=False 而不是抛**（旧服务端兼容）
  C `run_code` 真透传：在受限命名空间里 exec 一段代码，能看见 sweep_* 名字
  D 归因（P3）：提交/取消走 `_sweep_op` → 进 journal；**无快照 ⇒ 不参与 undo**
    （否则一次扫描会让「撤销AI」回滚到无关的旧状态）

用法：python checks/sweep_ai_wiring_selfcheck.py（毫秒级；无窗口、无 ARC）
"""
import os
import shutil
import sys
import tempfile

os.environ.setdefault('EL_NO_AI_SERVER', '1')
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

from checks.checklib import Checker                              # noqa: E402
from src.ai import server as S                                   # noqa: E402
from src.ai import codeact as CA                                 # noqa: E402
from src.ai.journal import get_journal                           # noqa: E402

C = Checker('sweep_ai_wiring')
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SWEEP_TOOLS = ('submit_sweep', 'sweep_status', 'sweep_result',
               'cancel_sweep', 'list_sweeps')


class _StubMCP:
    """只记录 @mcp.tool 注册的名字与描述（不起真 server）。"""

    def __init__(self):
        self.tools = {}

    def tool(self, description=''):
        def deco(fn):
            self.tools[fn.__name__] = description
            return fn
        return deco


class _StubState:
    """SimState 的最小替身（命名空间构造只按属性取用）。"""

    def get_state(self, scope=None):
        return {'params': {}}

    def summary(self):
        return {'ok': True}

    def set_param(self, *a, **k):
        return {'ok': True}

    def add_level_group(self, *a, **k):
        return {'ok': True}

    def remove_level_group(self, *a, **k):
        return {'ok': True}

    def set_waves(self, *a, **k):
        return {'ok': True}

    def set_groups(self, *a, **k):
        return {'ok': True}

    def set_initial_state(self, *a, **k):
        return {'ok': True}

    def list_levels(self):
        return {'levels': []}

    def get_config(self):
        return {'config': {}}

    def estimate_cost(self):
        return {'ok': True}

    def set_superposition(self, *a, **k):
        return {'ok': True}

    def set_pair_superposition(self, *a, **k):
        return {'ok': True}

    def to_dict(self):
        return {'params': {}}


# ---------------------------------------------------------------
print('A) MCP 工具面（注册 + 描述可读）')
m = _StubMCP()
S._register_tools(m)
_desc = {t: (m.tools.get(t) or '') for t in SWEEP_TOOLS}
C.group('A 五个并行扫描工具都注册且描述含关键用法词', [
    ('%s 已注册' % t, t in m.tools) for t in SWEEP_TOOLS
] + [
    ('工具总数 ≥ 36（只增不减）', len(m.tools) >= 36),
    ('submit_sweep 描述讲清 axes/路径/workers/record',
     all(k in _desc['submit_sweep']
         for k in ('axes', 'field.lights', 'workers', 'record'))),
    ('sweep_status 描述点明"不经 Qt 主线程"（长扫描期间可查）',
     'Qt 主线程' in _desc['sweep_status']),
    ('sweep_result 描述给出汇总表口径',
     'selected_table' in _desc['sweep_result']),
])

# ---------------------------------------------------------------
print('B) CodeAct 白名单（命名空间注入）')
ns = CA._make_namespace(
    _StubState(), ROOT, {'seq': [0]},
    lambda jid, diag=True: {'ok': True},
    lambda jid: {'done': True}, lambda jid, *a, **k: {'summary': {}},
    t_sweep_submit=lambda **k: {'ok': True, 'sweep_id': 'swX'},
    t_sweep_status=lambda sid=None, **k: {'ok': True, 'running': False},
    t_sweep_result=lambda sid=None, *a, **k: {'ok': True},
    t_sweep_cancel=lambda sid=None, **k: {'ok': True},
    t_sweep_list=lambda limit=20: {'ok': True, 'sweeps': []})
ns_bare = CA._make_namespace(
    _StubState(), ROOT, {'seq': [0]},
    lambda jid, diag=True: {'ok': True},
    lambda jid: {'done': True}, lambda jid, *a, **k: {'summary': {}})
C.group('B CodeAct 命名空间（注入 + 未接线不抛）', [
    ('submit_sweep/sweep_status/sweep_result/cancel_sweep/list_sweeps 全在',
     all(k in ns for k in
         ('submit_sweep', 'sweep_status', 'sweep_result', 'cancel_sweep',
          'list_sweeps'))),
    ('run_sweep_and_wait 是 callable', callable(ns.get('run_sweep_and_wait'))),
    ('未接线时 submit_sweep 返回 ok=False 而非抛',
     ns_bare['submit_sweep'](axes={'a': [1]}) ==
     {'ok': False, 'msg': '本进程不支持并行扫描（服务端未接线）'}),
    ('未接线时不注入 sweep_status（兼容旧服务端）',
     'sweep_status' not in ns_bare),
])

# ---------------------------------------------------------------
print('C) run_code 真透传（在受限命名空间里 exec）')
tmp = tempfile.mkdtemp(prefix='sweep_wire_')
try:
    out = CA.run_code(
        "result = sorted(k for k in list(globals()) if 'sweep' in k)",
        _StubState(), tmp, {'seq': [0]},
        lambda jid, diag=True: {'ok': True},
        lambda jid: {'done': True}, lambda jid, *a, **k: {'summary': {}},
        t_sweep_submit=lambda **k: {'ok': True},
        t_sweep_status=lambda sid=None, **k: {'ok': True},
        t_sweep_result=lambda sid=None, *a, **k: {'ok': True},
        t_sweep_cancel=lambda sid=None, **k: {'ok': True},
        t_sweep_list=lambda limit=20: {'ok': True})
    got = out.get('result') or []
    C.group('C run_python 的脚本里真能用 sweep_* 名字', [
        ('run_code 执行成功', out.get('ok') is True),
        ('脚本看见了 submit_sweep/sweep_status/sweep_result',
         {'submit_sweep', 'sweep_status', 'sweep_result'} <= set(got)),
        ('脚本看见了 cancel_sweep/list_sweeps/run_sweep_and_wait',
         {'cancel_sweep', 'list_sweeps', 'run_sweep_and_wait'} <= set(got)),
    ], 'got=%s' % got)
finally:
    shutil.rmtree(tmp, ignore_errors=True)

# ---------------------------------------------------------------
print('D) 归因：提交/取消进流水但不参与 undo（P3）')
j = get_journal(ROOT)
seq_before = (j.since(0) or [{}])[-1].get('seq', 0)
_, snap_before = j.last_snapshot('ai')
S._sweep_op('submit_sweep', {'sweep': 'swTEST', 'n_points': 3}, '自检')
ops = [o for o in j.since(seq_before) if o.get('kind') == 'submit_sweep']
_, snap_after = j.last_snapshot('ai')
C.group('D 扫描提交可归因、且不污染 undo 栈', [
    ('流水里出现 submit_sweep 且有原因', bool(ops) and
     ops[-1].get('reason') == '自检'),
    ('detail 带 sweep_id/n_points',
     (ops[-1].get('detail') or {}).get('sweep') == 'swTEST'),
    ('**无快照** ⇒ undo 目标未变（不会被扫描顶掉）',
     snap_after is snap_before),
])

ok = C.summary()
sys.exit(0 if ok else 1)
