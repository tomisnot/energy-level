# -*- coding: utf-8 -*-
"""自检：AI 通道两个易用性修复（A1 白名单 / A5 npz 顶层键）。

不依赖 Qt 主窗口与运行中的应用——只测 `src/ai/codeact._make_namespace`
与 `run_compute_and_wait` 这两个纯函数行为。

跑法：python checks/ai_codeact_tools_selfcheck.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

from src.ai import codeact                                      # noqa: E402

FAIL = []


def chk(cond, msg):
    print(('  OK   ' if cond else '  FAIL ') + msg)
    if not cond:
        FAIL.append(msg)


class _State:
    """最小 state_mod 替身：只需要 _make_namespace 用到的几个名字。"""
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


def mk(**kw):
    return codeact._make_namespace(
        _State(), '.', {'seq': [0]},
        lambda jid: {'ok': True}, lambda jid: {'done': True},
        lambda jid, save=False: {'ok': True}, **kw)


print('A1：只读/取消工具进 CodeAct 命名空间')
ns = mk(t_list_params=lambda prefix='': {'params': {}},
        t_diag=lambda **k: {'ok': True},
        t_cancel=lambda jid: {'ok': True})
for key in ('list_params', 'get_diag', 'cancel_job', 'job_status',
            'job_result'):
    chk(key in ns, '命名空间含 %s' % key)
ns0 = mk()
chk('list_params' not in ns0 and 'get_diag' not in ns0
    and 'cancel_job' not in ns0,
    '未传入时不注入（避免 None 可调用导致 TypeError）')
chk('job_status' in ns0 and 'job_result' in ns0, 'job_status/job_result 恒在')

print('A5：run_compute_and_wait 把 npz 提到顶层（单体/两体一致）')
fn = ns0['run_compute_and_wait']
codeact._make_namespace.__globals__  # noqa: B018 - 仅确认可调用
r = codeact._make_namespace(
    _State(), '.', {'seq': [0]},
    lambda jid: {'ok': True}, lambda jid: {'done': True},
    lambda jid, save=False: {'ok': True, 'summary': {'npz': 'X.npz'}},
)['run_compute_and_wait'](timeout=1, poll=0.01, save_npz=True)
chk(r.get('npz') == 'X.npz', "顶层 npz = %r" % r.get('npz'))
chk(r.get('job_id') == 'job1', 'job_id 仍回传（%r）' % r.get('job_id'))

print('\n结果：%s' % ('全部通过' if not FAIL else '%d 项失败' % len(FAIL)))
sys.exit(1 if FAIL else 0)
