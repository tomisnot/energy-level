# -*- coding: utf-8 -*-
"""精简回归入口：只跑 3 个核心自检（约 1 分钟）。

    core_selfcheck.py              单原子静态（能级/偶极/AC 位移，24 项）
    evolve_dissipative_selfcheck.py  Lindblad 耗散引擎（解析衰变/时变/退相/
                                     ac 约定回归/numba≡numpy，18 项）
    ui_selftest.py                 UI 回归（主窗口/编辑器/TTL/动态页/存档恢复）

其余自检（两体 pair/unified、STIRAP/拉曼、pulse 模块、Floquet 交叉验证、
单原子 evolve）已归档到 checks/_archive/，需要时单独运行，不进日常回归。

用法：
  python checks\\selftest_all.py       # 全量（约 1 分钟）
"""
import os
import sys
import time
import subprocess

sys.stdout.reconfigure(encoding='utf-8', errors='replace')

_HERE = os.path.dirname(os.path.abspath(__file__))
_SCRIPTS = ['core_selfcheck.py', 'evolve_dissipative_selfcheck.py',
            'ui_selftest.py']


def main():
    rc = 0
    t_all = time.perf_counter()
    for s in _SCRIPTS:
        print(f'\n===== {s} =====')
        t0 = time.perf_counter()
        r = subprocess.run([sys.executable, '-u', os.path.join(_HERE, s)])
        rc |= r.returncode
        print(f'—— {s} 完成，{time.perf_counter() - t0:.0f}s')
    print(f'\n==== 回归完成，{time.perf_counter() - t_all:.0f}s，'
          f'{"全部通过 ✅" if rc == 0 else "有失败 ❌"} ====')
    sys.exit(rc)


if __name__ == '__main__':
    main()
