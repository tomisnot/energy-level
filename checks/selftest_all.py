# -*- coding: utf-8 -*-
"""统一 selfcheck 入口：依次运行全部检查脚本。

    core_selfcheck.py       单原子 44 项（手写 vs ARC）
    core_ac_directcheck.py  Floquet 直接对角化交叉验证（13 项）
    pair_selfcheck.py       pair 两体：pair_core 静态 + driver 驱动
    unified_selfcheck.py    统一计算层（build_system/drive/response_spectrum）

运行：D:\\python310\\python.exe -u selftest_all.py
"""
import os
import subprocess
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_SCRIPTS = ['core_selfcheck.py', 'core_ac_directcheck.py',
            'pair_selfcheck.py', 'unified_selfcheck.py']


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
