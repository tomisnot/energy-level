# -*- coding: utf-8 -*-
"""椭圆偏参数行自检（t85）——钉住用户反馈："选椭圆偏后没有配套参数框"。

要求：参数**平时隐藏**；某行偏振选「椭圆偏」后在表格正下方冒出来一行
（α/θ/ε 三个输入框 + 哪一行的选择）；切回标准偏振则收起并清除参数；
参数要真的进 `light_cfgs()`（下游 fields/app_state 就靠这个键）。

用法：python checks\\gui_ellip_row_test.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

from checks.checklib import Checker                          # noqa: E402
from PyQt5.QtWidgets import QApplication                      # noqa: E402

import app_entry as M                                         # noqa: E402

C = Checker('gui_ellip_row')
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
win.field.set_lights_visible(True)
f = win.field

group('A 平时隐藏 / 新行默认 σ+ / 选椭圆偏冒出（三参数框 + 下拉列该行）')
check('1 平时隐藏（没有椭圆偏就不出现）',
        f.ellip_row.isHidden(), f'visible={f.ellip_row.isVisible()}')

f.add_light()
check('2 新光行默认 σ+，参数行仍隐藏', f.ellip_row.isHidden())

# 选「椭圆偏」→ 参数行冒出来
pol = f.table.cellWidget(0, 3)
pol.setCurrentText('椭圆偏')
_app.processEvents()
check('3 选「椭圆偏」→ 参数行出现', not f.ellip_row.isHidden())
check('4 三个参数框齐备且默认 0°',
        all(w.value() == 0.0 for w in (f.ellip_a, f.ellip_t, f.ellip_e)),
        f'{f.ellip_a.value()}/{f.ellip_t.value()}/{f.ellip_e.value()}')
check('5 下拉列出该行（光1）', f.ellip_sel.currentText() == '光1',
        f.ellip_sel.currentText())

# 改参数 → 进存储 + 进 light_cfgs
f.ellip_a.setValue(15.0)
f.ellip_t.setValue(30.0)
f.ellip_e.setValue(-5.0)
f._on_ellip_box()
_cfg = f.light_cfgs()[0]
group('B 参数进存储与 light_cfgs + tooltip + 切回标准偏振即收起清除')
check('6 参数写进存储与 light_cfgs（下游就吃这个键）',
        _cfg.get('polarization') == '椭圆偏'
        and _cfg.get('elliptical') == {'alpha_deg': 15.0, 'theta_deg': 30.0,
                                       'eps_deg': -5.0},
        str(_cfg.get('elliptical')))
check('7 偏振单元格 tooltip 带上具体角度',
        'α=15' in (pol.toolTip() or ''), str(pol.toolTip()))

# 切回标准偏振 → 收起 + 清除
pol.setCurrentText('σ-')
_app.processEvents()
check('8 切回 σ- → 参数行收起、椭圆参数被清除',
        f.ellip_row.isHidden() and 'elliptical' not in f.light_cfgs()[0],
        str(f.light_cfgs()[0]))

# 两行同时椭圆偏：下拉能切换
pol.setCurrentText('椭圆偏')
f.add_light()
f.table.cellWidget(1, 3).setCurrentText('椭圆偏')
_app.processEvents()
group('C 两行互不串 + 程序化载入（存档/AI）回填角度')
check('9 两行椭圆偏 → 下拉两项可切换',
        f.ellip_sel.count() == 2 and not f.ellip_row.isHidden(),
        f'count={f.ellip_sel.count()}')
f.ellip_sel.setCurrentIndex(0)
f.ellip_a.setValue(7.0)
f._on_ellip_box()
f.ellip_sel.setCurrentIndex(1)
_app.processEvents()
_a2 = f.ellip_a.value()
f.ellip_sel.setCurrentIndex(0)
_app.processEvents()
check('10 两行参数互不串（切回去还是各自的值）',
        f.ellip_sel.currentText() == '光1' and _a2 == 0.0
        and f.ellip_a.value() == 7.0,
        f'row2={_a2} row1={f.ellip_a.value()} / {f._ellip}')


# 11. 程序化写偏振（存档/AI set_param 走 blockSignals）也要冒出来 —— 实测漏过一次
f.set_lights([{'frequency_hz': 7.5e9, 'intensity': 1.0, 'polarization': '椭圆偏',
               'scope': 'both',
               'elliptical': {'alpha_deg': 12.0, 'theta_deg': 34.0,
                              'eps_deg': 5.0}}])
_app.processEvents()
check('11 程序化载入椭圆偏 → 参数行出现且回填角度',
        not f.ellip_row.isHidden() and f.ellip_sel.count() == 1
        and f.ellip_a.value() == 12.0 and f.ellip_t.value() == 34.0
        and f.ellip_e.value() == 5.0,
        f'hidden={f.ellip_row.isHidden()} a={f.ellip_a.value()} '
        f't={f.ellip_t.value()} e={f.ellip_e.value()}')
flush()
sys.exit(0 if C.summary() else 1)
