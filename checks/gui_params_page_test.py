# -*- coding: utf-8 -*-
"""引擎参数页 GUI 接线自检（t81）：页面 ↔ 覆写快照 ↔ 引擎 的契约。

为什么单列：本页是**控制引擎参数的正式入口**，它的失效方式是"静默无效"
（填了值但没进引擎 / 存档回填丢了 / 非法输入被吞掉），单元测试抓不到——
必须真的把 QDialog 建起来、真的调它的读写接口。

覆盖：
  1. 页面由 params.SPECS 生成：每个可调参数一行、分层组数与非空层数一致；
  2. 填值 → param_overrides() 类型化快照；计数刷新；
  3. 非法输入：**只忽略该项**（标红 + 计数 + param_errors()），
     不弹模态框、不牵连其它覆写；
  4. god_values() 只管场景级旋钮 + diag（引擎覆写唯一来源 = 会话真值）；
  5. 重置清空；
  6. 存档回填：新格式（param_overrides）与旧格式（kwarg 平铺）都认。

用法：python checks\\gui_params_page_test.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

from checks.checklib import Checker                          # noqa: E402
from PyQt5.QtWidgets import QApplication                     # noqa: E402
from src.core import params as P                             # noqa: E402
from src.ui.solver_inspector import SolverInspector          # noqa: E402

C = Checker('gui_params_page')

_app = QApplication.instance() or QApplication(sys.argv)
insp = SolverInspector()

C.check('1 页面已建', insp._tab_e is not None)
C.check('2 每个可调参数一行（由 SPECS 生成）',
        len(insp._e_edits) == len(P.specs()),
        f'{len(insp._e_edits)} vs {len(P.specs())}')
C.check('3 分层组数 == 非空层数',
        sum(1 for t, _ in P.TIERS if P.specs(t)) == 4)
C.check('4 初始无覆写',
        insp.param_overrides() == {} and '无覆写' in insp._e_count.text())

insp._e_edits['PHASE_TOL'].setText('1e-3')
insp._e_edits['_M_MAX'].setText('5')
C.check('5 填值 → 类型化快照',
        insp.param_overrides() == {'PHASE_TOL': 1e-3, '_M_MAX': 5},
        str(insp.param_overrides()))
C.check('6 计数刷新', '2 项' in insp._e_count.text(),
        insp._e_count.text())

insp._e_edits['_M_MAX'].setText('abc')
C.check('7 非法输入：自愈回退该项、不弹模态、不牵连其它覆写',
        insp.param_overrides() == {'PHASE_TOL': 1e-3}
        and insp.param_errors() == []
        and '非法' in insp._e_count.text()
        and insp._e_edits['_M_MAX'].toolTip() != '',
        f"{insp.param_overrides()} / {insp._e_edits['_M_MAX'].text()!r}")
C.check('7b 非法原因可见（tooltip 含类型与"留空"提示，非模态）',
        '应为 int' in insp._e_edits['_M_MAX'].toolTip()
        and '留空' in insp._e_edits['_M_MAX'].toolTip(),
        insp._e_edits['_M_MAX'].toolTip())
insp._e_edits['_M_MAX'].setText('5')
C.check('7c 改回合法后提示消失、值进快照',
        insp._e_edits['_M_MAX'].toolTip() == ''
        and insp.param_overrides().get('_M_MAX') == 5,
        f"{insp._e_edits['_M_MAX'].toolTip()!r} / {insp.param_overrides()}")

_ek = insp.god_values()
C.check('8 god_values 不再自带引擎覆写（唯一来源=会话真值）',
        'param_overrides' not in _ek and _ek.get('diag') is True,
        str(_ek.keys()))

insp.reset_param_overrides()
C.check('9 重置清空', insp.param_overrides() == {})

insp.set_god_params({'param_overrides': {'PHASE_TOL': 1e-5}})
C.check('10 回填：新格式（引擎参数全路径）',
        insp.param_overrides() == {'PHASE_TOL': 1e-5},
        str(insp.param_overrides()))
insp.reset_param_overrides()
insp.set_god_params({'bw_opt': 1234.0})
C.check('11 回填：旧格式兼容（bw_opt → BW_OPT）',
        insp.param_overrides() == {'BW_OPT': 1234.0},
        str(insp.param_overrides()))
insp.reset_param_overrides()

sys.exit(0 if C.summary() else 1)
