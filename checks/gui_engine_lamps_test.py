# -*- coding: utf-8 -*-
"""engine.* 指示灯与双向绑定自检（t83）。

背景（实测踩过的三个坑，本文件就是它们的回归闸）：
  1. **只 register 不绑控件 = 永远没有指示灯**——框架 `ParamRegistry._paint()`
     在 `binding.widget is None` 时直接 return，所以必须 `bind_engine_widgets`
     把检视器页的输入框绑到 `engine.*`；
  2. 框架 `seed_baseline()`/`load()` 会对每个 binding 调一次 `write_fn(当前值)`
     建基准 → 若 write_fn 不判等，窗口构造期就把 32 个参数写进 store
     （"生效 32 项"的假象 + 存档臃肿）→ 故 `engine_write` 只存**偏离默认**的项；
  3. 非法/半成品输入**不许牵连其它覆写**（页面走逐参数增量信号，不是整批替换）。

灯的约定（沿用框架/项目既有语义）：人输入 = 黄灯（touched），提交后清灯
（人改不高亮，防自闪）；AI/程序写入 = 绿灯 + 来源紫。

用法：python checks\\gui_engine_lamps_test.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

from checks.checklib import Checker                          # noqa: E402
from PyQt5.QtWidgets import QApplication                     # noqa: E402
import app_entry as M                                        # noqa: E402
from src.core import params as P                             # noqa: E402

C = Checker('gui_engine_lamps')
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
reg = win.param_reg

group('A 构造期契约：store 干净 + 输入框绑定就位')
check('1 构造期不产生假覆写（store 干净）',
        win.engine_overrides == {}, str(win.engine_overrides))

win.set_mode(god=True)
insp = win.dynamic._inspector
ed = insp._e_edits['PHASE_TOL']
b = reg._bindings['engine.PHASE_TOL']
check('2 输入框已绑到 engine.*（绑定数 = 可调数）',
        b.widget is ed and insp._engine_bound == len(P.specs()),
        f'{insp._engine_bound} / {len(P.specs())}')

# ① AI / 程序写入路径
reg.set('engine.PHASE_TOL', 5e-4, source='ai', reason='自检')
group('B AI/程序写入路径：真值 / 控件 / 灯 / read + engine.* 全在册')
check('3 AI set → 会话真值更新', win.engine_overrides.get('engine.PHASE_TOL') == 5e-4)
check('4 AI set → 控件文本同步（不是只亮灯不改数）',
        ed.text() == '0.0005', ed.text())
check('5 AI set → 指示灯亮（绿 + 来源紫）', bool(ed.styleSheet()),
        repr(ed.styleSheet())[:60])
check('6 reg.read 给数值而非输入框文本',
        isinstance(reg.read('engine.PHASE_TOL'), float))
check('7 engine.* 全部在册', len([n for n in reg.names()
                                    if n.startswith('engine.')])
        == len(P.specs()))

reg._clear_highlights()
# ② 人输入路径
ed.setText('2.5e-4')
group('C 人输入路径：即时更新 / 黄灯 / 提交后清灯')
check('8 输入 → 会话真值立即更新（不等回车）',
        win.engine_overrides.get('engine.PHASE_TOL') == 2.5e-4,
        str(win.engine_overrides.get('PHASE_TOL')))
check('9 输入 → 黄灯（touched）', bool(ed.styleSheet()))
reg.commit_pending()
check('10 提交后值不变、pending 清空（人改不高亮=既有约定）',
        win.engine_overrides.get('engine.PHASE_TOL') == 2.5e-4
        and not reg.has_pending()
        and reg.last_error('engine.PHASE_TOL') in (None, ''))

# ③ 非法输入：只忽略该项
before = dict(win.engine_overrides)
ed.setText('abc')
group('D 非法输入只忽略该项（自愈可见）+ 批量清空')
check('11 非法 → 不写 store、不牵连其它覆写',
        win.engine_overrides == before, str(win.engine_overrides))
check('12 非法 → 自愈回退 + 可见提示（不写 store）',
        ed.text() == '0.00025' and '非法' in insp._e_count.text()
        and insp.param_errors() == [],
        f'text={ed.text()!r} count={insp._e_count.text()!r}')
# 清掉非法项，确认页面能恢复
ed.setText('')
check('13 非法项清空后恢复（该项覆写清除、read 回默认）',
        'engine.PHASE_TOL' not in win.engine_overrides
        and reg.read('engine.PHASE_TOL') == P.DEFAULTS['PHASE_TOL'])

# ④ 批量清空
win.set_engine_overrides({})
check('14 批量清空 → 控件全清、store 空',
        win.engine_overrides == {}
        and all(not e.text() for e in insp._e_edits.values()))

flush()
sys.exit(0 if C.summary() else 1)