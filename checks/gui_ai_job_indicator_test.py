# -*- coding: utf-8 -*-
"""B3 自检：AI 作业在 GUI **可见、可取消**（G3「人可随时监控与接管」）。

## 口径变更（2026-09-16 用户裁决）

用户原话："旧的「取消AI任务」已经被新的「取消计算」按钮代替，**请删除，统一渠道**。"

于是 B3 的原实现（状态栏常驻「AI 作业中… [取消AI作业]」指示器 + 自己的 QTimer）
整个退役：运行中「运行演化」按钮本身就变成「取消计算」，**人与 AI 的作业都走它**
（同一条 `_on_run_clicked → _cancel_run`）。本文件因此从"测那个指示器"
改为测**统一渠道这条不变式**：

- 不再存在 AI 专属的取消入口（按钮/指示器/安装函数 + 源码里那串死代码）；
- 但 B3 要保住的两件事一件不少：**归因**（`list_jobs` 里 owner=ai）
  与**可接管**（统一渠道能取消 AI 作业；MCP 侧 `cancel_job` 同路径仍在）。

## 顺带钉住一个静默故障（登记区 #15）

`_init_ai` 尾部原有 5 行**引用未定义名字**的死代码（`_ai_job_cancel` / `_QT` /
`_ai_job_tick`，全库无定义），NameError 被本函数兜底 except 吞掉
⇒ 它**后面**的 op 流水订阅与 `return port` 从未执行（AI 端点其实起来了，
状态栏却报"端点启动失败"）。本文件用"stub 一个假 port → 断言 `_init_ai`
真的返回它 + 订阅真的装上"把这条钉住。

覆盖（每段 1 个不变式包）：
  A 旧入口已彻底删除（属性 + 源码都不许再有残留）
  B 归因不变：owner=ai 的作业在 list_jobs 里可见且 running
  C **统一渠道**：运行中按钮即"取消计算"，`_on_run_clicked` 分派正确
  D AI 作业可被取消（MCP 同路径 `_t_cancel_job` 仍可用）
  E `_init_ai` 收尾不被 NameError 打断（登记区 #15 的回归）

用法：python checks/gui_ai_job_indicator_test.py
"""
import ast
import inspect
import os
import sys
import time

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
os.environ.setdefault('EL_NO_AI_SERVER', '1')

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

from checks.checklib import Checker                            # noqa: E402
from PyQt5.QtWidgets import QApplication, QLabel                # noqa: E402
import app_entry as M                                          # noqa: E402
from src.ai import server as ai_server                         # noqa: E402
from src.ai.journal import get_journal                         # noqa: E402

C = Checker('gui_ai_job_indicator')
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_app = QApplication.instance() or QApplication(sys.argv)
win = M.MainWindow()
win.show()          # `server._win()` 只认 **isVisible()** 的 MainWindow（取消路径要用）

# ---------------------------------------------------------------
print('A) AI 专属取消入口已删除（统一渠道）')
#  扫描必须打**AST**，不能扫源码文本：本文件与 app_entry 的注释里会**引用**
# 那些已删的名字来解释历史（第一次就因此假红）。注释不在 AST 里，代码才在。
_src = inspect.getsource(M)
_tree = ast.parse(_src)
_names = {n.id for n in ast.walk(_tree) if isinstance(n, ast.Name)}
_attrs = {n.attr for n in ast.walk(_tree) if isinstance(n, ast.Attribute)}
_strs = {n.value for n in ast.walk(_tree)
         if isinstance(n, ast.Constant) and isinstance(n.value, str)}
_defs = {n.name for n in ast.walk(_tree) if isinstance(n, ast.FunctionDef)}
C.group('A 旧入口彻底退役（属性 + **AST** 双查）', [
    ('无 _install_ai_job_indicator', not hasattr(win, '_install_ai_job_indicator')),
    ('无 _ai_job_btn', not hasattr(win, '_ai_job_btn')),
    ('无 _ai_job_ind', not hasattr(win, '_ai_job_ind')),
    ('AST 里无 _install_ai_job_indicator 定义',
     '_install_ai_job_indicator' not in _defs),
    ('AST 里无 _ai_job_btn / _ai_job_ind 属性访问',
     not ({'_ai_job_btn', '_ai_job_ind'} & _attrs)),
    ('AST 里无 _ai_job_cancel / _ai_job_tick / _QT 名字引用（登记区 #15 根因）',
     not ({'_ai_job_cancel', '_ai_job_tick', '_QT'} & _names)),
    ('AST 里无 "取消AI作业" 字面量', '取消AI作业' not in _strs),
])

# ---------------------------------------------------------------
print('B) 归因不变：owner=ai 的作业可见')
jid = 'jobTEST'
ai_server._jobs[jid] = {'started': time.time() - 3.0, 'done': False,
                        'error': None, 'res': None, 'owner': 'ai',
                        'kind': 'single', 't_max': None}
st = ai_server.ui_job_status()
J = ai_server._t_list_jobs()
row = next((r for r in J['jobs'] if r['job_id'] == jid), None)
C.group('B AI 作业的归因与可见性（B3 要保住的一半）', [
    ('ui_job_status 能看到它', (st or {}).get('job_id') == jid),
    ('list_jobs 给出 owner=ai / status=running',
     bool(row) and row['owner'] == 'ai' and row['status'] == 'running'),
], 'status=%s row=%s' % (st, row))

# ---------------------------------------------------------------
print('C) 统一渠道：运行中按钮 = 取消计算')


def _run_btn_labels(src):
    """AST 取 `run_btn.setText('…')` 的**真实文案**（不硬编码进自检）。"""
    out = []
    for node in ast.walk(ast.parse(src)):
        if (isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == 'setText'
                and isinstance(node.func.value, ast.Attribute)
                and node.func.value.attr == 'run_btn'
                and node.args and isinstance(node.args[0], ast.Constant)):
            out.append(node.args[0].value)
    return out


d = win.dynamic
_labels_run = _run_btn_labels(inspect.getsource(type(d)))
_called = {}
d._cancel_run = lambda *a, **k: _called.__setitem__('cancel', True)
d._run = lambda *a, **k: _called.__setitem__('run', True)
d._run_active = True
d.run_btn.setText('取消计算')            # 模拟"运行中"人眼所见
_txt_running = d.run_btn.text()
d._on_run_clicked()
_first = dict(_called)
_called.clear()
d._run_active = False
d._on_run_clicked()
_second = dict(_called)
C.group('C 一个按钮两个职能（人与 AI 同一条路）', [
    ("运行中按钮文案由 _run() 写成「取消计算」",
     '取消计算' in _labels_run),
    ("空闲文案「运行演化」也在（同一个按钮）",
     '运行演化' in _labels_run),
    ('运行中点它 → _cancel_run（不是再起一个）',
     _first == {'cancel': True}),
    ('空闲时点它 → _run', _second == {'run': True}),
], 'labels=%s running_txt=%s' % (_labels_run, _txt_running))

# ---------------------------------------------------------------
print('D) AI 作业可取消（MCP 侧同路径仍可用）')
# 真实取消路径 `_t_cancel_job` 要三样东西齐全，自检如实补齐（都是"运行中"必有的）：
#   ① 可见的 MainWindow（`_win()` 只认 isVisible）
#   ② 主线程派发器（生产上由 start_server 装；本文件 stub 了 start_server）
#   ③ `progress_arr` 与 `_dlg`（`ui_dynamic._run()` 起步时建的运行态字段；
#      "取消计算"按钮也只在运行中才可点 ⇒ 补齐它们就是"运行中"的真实前置。
#      ⚠ 顺带一个观察：`_cancel_run` 对这三个字段是硬依赖，若在**没跑过**的
#      面板上被调到会 AttributeError（GUI 路径不可达；MCP `cancel_job` 也先
#      判 `job['done']`，故当前不可达）。这里不修 UI，只如实补齐前置。
import numpy as np                                            # noqa: E402
from src.ai import dispatch as _dispatch                      # noqa: E402
_dispatch.init()
for _n in ('_cancel_run', '_run'):       # 撤掉 C 包打的方法桩，走真实现
    if _n in d.__dict__:
        delattr(d, _n)
if getattr(d, 'progress_arr', None) is None:
    d.progress_arr = np.zeros(3)
d._dlg = None
d.progress_arr[1] = 0.0
_r = ai_server.ui_cancel_job(jid)
_st2 = ai_server.ui_job_status()
_flag = float(d.progress_arr[1])
_unknown = ai_server.ui_cancel_job('不存在')
C.group('D 取消能力强于"有个按钮"（MCP 同源，且**落到引擎标志**）', [
    ('ui_cancel_job 走通并报 ok', bool(_r) and _r.get('ok') is True),
    ('取消意图立刻落到作业登记（cancelled=True）',
     bool(ai_server._jobs.get(jid, {}).get('cancelled'))),
    ('**引擎取消标志被置位**（progress_arr[1]==1.0，真正停算的那一下）',
     _flag == 1.0),
    ('取消后不再算 running',
     (_st2 or {}) == {} or _st2 is None),
    ('未知 job 明确报错（与 MCP cancel_job 同契约）',
     _unknown.get('ok') is False),
], 'r=%s flag=%s unknown=%s' % (_r, _flag, _unknown))
ai_server._jobs.pop(jid, None)

# ---------------------------------------------------------------
print('E) _init_ai 收尾不被 NameError 打断（登记区 #15 回归）')
_j = get_journal(ROOT)
_n_before = len(getattr(_j, '_listeners', []))
_orig_start = ai_server.start_server
ai_server.start_server = lambda _win: 8765          # 不起真 server
try:
    _port = win._init_ai()
finally:
    ai_server.start_server = _orig_start
_labels = [w.text() for w in win.statusBar().findChildren(QLabel)]
C.group('E AI 端点收尾真的跑完了（不再被死代码吞掉）', [
    ('_init_ai 返回了 port（不再是被 except 吃掉后的 None）', _port == 8765),
    ('状态栏装上「● AI 在线」', any('AI 在线' in t for t in _labels)),
    ('op 流水订阅装上了（死代码之后的语句确实执行了）',
     len(getattr(_j, '_listeners', [])) == _n_before + 1),
], 'port=%s labels=%s' % (_port, _labels[:6]))

try:
    win.close()
except Exception:                                # noqa: BLE001
    pass

ok = C.summary()
sys.exit(0 if ok else 1)
