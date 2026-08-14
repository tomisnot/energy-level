# -*- coding: utf-8 -*-
"""统一 selfcheck 入口：依次运行全部检查脚本。

    core_selfcheck.py       单原子 44 项（手写 vs ARC）
    core_ac_directcheck.py  Floquet 直接对角化交叉验证（13 项）
    pair_selfcheck.py       pair 两体：pair_core 静态 + driver 驱动
    unified_selfcheck.py    统一计算层（build_system/drive/response_spectrum）
    evolve_selfcheck.py     单原子动态模拟（evolve.py：独立 solve_ivp 参考 + 归一/多光场/组驱动/大基组）
    pulse_selfcheck.py      pulse.py 独立模块（包络形状/TTL 触发定位/field 集成）
    ui_selftest.py          UI 回归（主窗口/布局/编辑器/TTL/动态页/进度取消）

运行：D:\python310\python.exe -u selftest_all.py
"""
import os
import subprocess
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_SCRIPTS = ['core_selfcheck.py', 'core_ac_directcheck.py',
            'pair_selfcheck.py', 'unified_selfcheck.py',
            'evolve_selfcheck.py', 'pulse_selfcheck.py',
            'ui_selftest.py']


def main():
    rc = 0
    for s in _SCRIPTS:
        print(f'\n========== {s} ==========')
        r = subprocess.run([sys.executable, '-u', os.path.join(_HERE, s)])
        rc |= r.returncode
    print('\n==== 全部检查完成 =====')
    sys.exit(rc)


if __name__ == '__main__':
    main()
