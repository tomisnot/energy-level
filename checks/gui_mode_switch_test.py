# -*- coding: utf-8 -*-
"""模式切换 + 检视器常驻 + 引擎参数双向绑定自检（t82）。

为什么单列：上帝模式从"动态页的一个复选框"升级成**凌驾各页的模式**
（菜单栏行的分段开关），并把引擎参数接成"会话真值 ↔ 检视器页 ↔ 参数注册表"
三向绑定。这类接线的失效方式是静默的（切了模式没生效 / 显示不同步 /
AI 改了页面不更新），必须真建 MainWindow 才能验。

覆盖：
  1. 菜单栏行有互斥的「极简模式 / 上帝模式」按钮；默认极简；
  2. 切上帝模式 → 检视器自动显示 + 标题/徽标标记；
  3. 极简模式 → 检视器隐藏但**对象仍在**（登记跟真值源走，不跟显隐走）；
  4. 双向绑定：页面改值 → 会话真值 → 注册表 engine.* 读到同一份；
     注册表 set（= AI 路径）→ 会话真值 → 页面回填；
  5. 运行 kwargs：覆写**与模式无关**都生效；diag 只在上帝模式采集；
  6. engine.* 全部在册（= params.SPECS 可调数）。

用法：python checks\\gui_mode_switch_test.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

from checks.checklib import Checker                          # noqa: E402
from PyQt5.QtWidgets import QApplication                     # noqa: E402
import app_entry as M                                        # noqa: E402
from src.core import params as P                             # noqa: E402

C = Checker('gui_mode_switch')
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
_app = QApplication.instance() or QApplication(sys.argv)
win = M.MainWindow()

group('A 模式开关接线（互斥按钮 / 默认极简 / 检视器惰性创建 + 跟随）')
check('1 菜单栏行有互斥模式按钮 + 默认极简',
        set(win._mode_btns) == {'simple', 'god'}
        and win.menuBar().cornerWidget() is not None
        and win.god_mode is False
        and win._mode_btns['simple'].isChecked())
check('2 检视器惰性创建（未打开前不建）', win.dynamic._inspector is None)

win.set_mode(god=True)
insp = win.dynamic._inspector
check('3 切上帝模式 → 检视器自动显示', insp is not None and insp.isVisible())
check('4 标题与徽标标记模式',
        '上帝模式' in win.windowTitle()
        and not win.dynamic.mode_badge.isHidden())
check('5 按钮状态跟随', win._mode_btns['god'].isChecked()
        and not win._mode_btns['simple'].isChecked())

# 双向绑定之一：控件 → 后端（用 PHASE_TOL；PHASE_BUDGET 已迁到运行级
# solve.COST_BUDGET，不再是引擎参数页的旋钮）
insp._e_edits['PHASE_TOL'].setText('2e-4')
group('B 双向绑定：控件 ↔ 会话真值 ↔ 注册表（engine.* 全在册）')
check('6 页面改值 → 会话真值',
        float(win.engine_overrides.get('engine.PHASE_TOL', 0)) == 2e-4,
        str(win.engine_overrides))
check('7 注册表 engine.* 读到同一份',
        win.param_reg.read('engine.PHASE_TOL')
        == win.engine_overrides['engine.PHASE_TOL'])

# 双向绑定之二：后端（= AI 路径）→ 控件
win.param_reg.set('engine.PHASE_TOL', 3e-4, source='ai', reason='自检')
check('8 注册表 set → 会话真值',
        win.engine_overrides.get('engine.PHASE_TOL') == 3e-4)
check('9 注册表 set → 页面回填', insp._e_edits['PHASE_TOL'].text() == '0.0003',
        insp._e_edits['PHASE_TOL'].text())
check('10 engine.* 全部在册（AI 可及面 = 可调参数）',
        len([n for n in win.param_reg.names() if n.startswith('engine.')])
        == len(P.specs()))

kw = win.dynamic._collect_god_kwargs()
group('C 运行 kwargs：覆写照旧生效、diag 只随上帝模式（含清空）')
check('11 上帝模式：kwargs 带覆写 + diag',
        kw.get('diag') is True
        and kw['param_overrides'].get('PHASE_TOL') == 3e-4, str(kw.keys()))

win.set_mode(god=False)
check('12 切回极简 → 检视器隐藏但对象仍在',
        win.dynamic._inspector is insp and not insp.isVisible()
        and win.dynamic.mode_badge.isHidden())
kw2 = win.dynamic._collect_god_kwargs()
check('13 极简模式：覆写照旧生效、diag 不采',
        'diag' not in kw2
        and kw2['param_overrides'].get('PHASE_TOL') == 3e-4, str(kw2))

win.set_engine_overrides({})
check('14 清空覆写 → 页面同步清空',
        insp._e_edits['PHASE_TOL'].text() == ''
        and win.param_reg.read('engine.PHASE_TOL') == P.DEFAULTS['PHASE_TOL'])

# 15/16：用户报过的场景——在输入框里留了个打错的字（如 _M_MAX='abc'）
# 然后点运行：**不许弹模态框**，也不许把其它合法覆写一起丢掉。
_logs = []
_orig_log = win.log
win.log = lambda msg, timeout=8000: _logs.append(str(msg))
try:
    insp._e_edits['PHASE_TOL'].setText('2e-4')
    ed_max = insp._e_edits['_M_MAX']
    ed_max.setText('abc')
    kw3 = win.dynamic._collect_god_kwargs()
finally:
    win.log = _orig_log
group('D 非法输入：不弹模态、不牵连合法覆写、自愈可见')
check('15 非法项被忽略但不牵连合法覆写（无模态框）',
        kw3['param_overrides'] == {'PHASE_TOL': 2e-4},
        str(kw3.get('param_overrides')))
_healed = None
try:
    _healed = int(ed_max.text())
except (TypeError, ValueError):
    _healed = None
check('16 非法输入非模态可见（编辑期自愈到有效值 + 计数行点名，不弹框）',
        _healed is not None and '非法' in insp._e_count.text(),
        f'text={ed_max.text()!r} count={insp._e_count.text()!r}')
insp._e_edits['_M_MAX'].setText('')
insp.reset_param_overrides()

flush()
sys.exit(0 if C.summary() else 1)
