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

from checks.checklib import Checker                          # noqa: E402

from src.ai import codeact                                      # noqa: E402

C = Checker('ai_codeact_tools')
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


def chk(cond, msg):
    """薄包装（沿用本文件原签名）：登记为**当前包的子条件**，不再各自计 1 项。"""
    return check(str(msg), bool(cond))


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
group('A 只读/取消工具进 CodeAct 命名空间（t_list_params/t_diag/t_cancel）')
for key in ('list_params', 'get_diag', 'cancel_job', 'job_status',
            'job_result'):
    chk(key in ns, '命名空间含 %s' % key)
group('B 未传入不注入 + job_status / job_result 恒在')
ns0 = mk()
chk('list_params' not in ns0 and 'get_diag' not in ns0
    and 'cancel_job' not in ns0,
    '未传入时不注入（避免 None 可调用导致 TypeError）')
chk('job_status' in ns0, 'job_status 恒在')
chk('job_result' in ns0, 'job_result 恒在')

print('A5：run_compute_and_wait 把 npz 提到顶层（单体/两体一致）')
group('C run_compute_and_wait：npz 提到顶层、job_id 仍回传')
fn = ns0['run_compute_and_wait']
codeact._make_namespace.__globals__  # noqa: B018 - 仅确认可调用
r = codeact._make_namespace(
    _State(), '.', {'seq': [0]},
    lambda jid: {'ok': True}, lambda jid: {'done': True},
    lambda jid, save=False: {'ok': True, 'summary': {'npz': 'X.npz'}},
)['run_compute_and_wait'](timeout=1, poll=0.01, save_npz=True)
chk(r.get('npz') == 'X.npz', "顶层 npz = %r" % r.get('npz'))
chk(r.get('job_id') == 'job1', 'job_id 仍回传（%r）' % r.get('job_id'))

flush()
sys.exit(0 if C.summary() else 1)
