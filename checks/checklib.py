# -*- coding: utf-8 -*-
"""检查公共设施（checklib）：供 core_selfcheck / pair_selfcheck 等复用。"""
import sys
if sys.platform == 'win32' and hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')


class Checker:
    """PASS/FAIL/SKIP 计数器 + 汇总（退出码 = FAIL 数）。"""

    def __init__(self, title='selftest'):
        self.title = title
        self.n_pass = 0
        self.n_fail = 0
        self.n_skip = 0

    def check(self, name, cond, detail=''):
        if cond:
            self.n_pass += 1
            print(f'  [PASS] {name}')
        else:
            self.n_fail += 1
            print(f'  [FAIL] {name}   {detail}')

    def close(self, name, ours, ref, rtol=1e-6, atol=0.0):
        if abs(ours - ref) <= atol + rtol * abs(ref):
            self.n_pass += 1
            print(f'  [PASS] {name}  ({ours:.6g} vs {ref:.6g})')
        else:
            self.n_fail += 1
            print(f'  [FAIL] {name}  ours={ours:.8g} ref={ref:.8g}')

    def skip(self, name, why=''):
        self.n_skip += 1
        print(f'  [SKIP] {name}   {why}')

    def summary(self):
        print(f'[{self.title}]  PASS={self.n_pass}  FAIL={self.n_fail}'
              f'  SKIP={self.n_skip}')
        return self.n_fail == 0
