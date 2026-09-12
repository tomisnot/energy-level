# -*- coding: utf-8 -*-
"""自检：两体**峰值内存**粗估（#11/B 假说的事后加固，2026-09-12）。

事故背景：两体 job20（dim=7056）跑完时进程被 Qt fail-fast 杀掉（根因是 QThread
生命周期，见 `checks/gui_worker_lifetime_test.py`）。**内存是待排除的并发因素**：
本机 15.8 GB、崩溃时段可用约 5–6 GB（另有一个离线扫描进程），而使用者实测
dim=4624 的两体作业**私有内存 1.36 GB** ⇒ 按 dim² 外推 dim=7056 ≈ 3.2 GB。
所以 `estimate_cost()` 现在同时给**时间**与**内存**粗估，供"要不要降规模"决策。

覆盖（纯函数，无 Qt / 无 ARC / 毫秒级）：
  A. 标定锚点：dim=4624 → 1.36 GB（使用者实测）；dim=7056 ≈ 3.2 GB（×2.33）
  B. 单调性与标度：dim 翻倍 → 4×；factor 线性
  C. 边界：dim=None/0/负 → None（不抛）
  D. `avail_ram_gb()` 可调用且给出正数（取不到 → None 也算过）
  E. `estimate_cost()` 源码接线（含 'est_peak_gb' / 'mem_advice' 键与提示语）

用法：python checks/pair_mem_estimate_selfcheck.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

from checks.checklib import Checker                              # noqa: E402
from src.ai.sim_state import (estimate_pair_peak_gb,             # noqa: E402
                              avail_ram_gb)

C = Checker('pair_mem_estimate')
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

print('A) 标定锚点与外推')
g4624 = estimate_pair_peak_gb(4624)
C.check('A1 dim=4624 → 1.36 GB（使用者实测锚点）',
        abs(g4624 - 1.36) < 1e-9, '%.3f GB' % g4624)
g7056 = estimate_pair_peak_gb(7056)
C.check('A2 dim=7056 ≈ 3.17 GB（dim² 外推 ×%.2f）'
        % ((7056 / 4624.0) ** 2), abs(g7056 - 1.36 * (7056 / 4624.0) ** 2) < 1e-9,
        '%.2f GB' % g7056)
C.check('A3 dim=2700（3 组两体）< 0.5 GB —— 小作业不该被吓到',
        estimate_pair_peak_gb(2700) < 0.5,
        '%.2f GB' % estimate_pair_peak_gb(2700))

print('B) 标度与单调性')
C.check('B1 dim 翻倍 → 峰值 ×4',
        abs(estimate_pair_peak_gb(2000) / estimate_pair_peak_gb(1000) - 4.0)
        < 1e-9)
C.check('B2 factor 线性', abs(estimate_pair_peak_gb(4624, factor=2.0)
                              - 2.72) < 1e-9)
C.check('B3 单调不减',
        all(estimate_pair_peak_gb(a) <= estimate_pair_peak_gb(b)
            for a, b in ((1000, 2000), (2000, 4624), (4624, 7056))))

print('C) 边界')
C.check('C1 dim=None/0/负 → None（不抛）',
        estimate_pair_peak_gb(None) is None
        and estimate_pair_peak_gb(0) is None
        and estimate_pair_peak_gb(-5) is None)

print('D) 可用内存（取不到也算过）')
a = avail_ram_gb()
C.check('D1 avail_ram_gb() 返回正 float 或 None',
        a is None or (isinstance(a, float) and a > 0.0), str(a))
if a:
    print('     本机当前可用内存 ≈ %.2f GB；dim=7056 预计峰值 %.2f GB（占 %.0f%%）'
          % (a, g7056, 100 * g7056 / a))

print('E) estimate_cost 接线（源码级）')
src = open(os.path.join(ROOT, 'src', 'ai', 'sim_state.py'),
           encoding='utf-8').read()
for key in ("'est_peak_gb'", "'avail_ram_gb'", "'mem_advice'",
            'estimate_pair_peak_gb(dim)',
            '崩溃时是**整进程消失**'):
    C.check('E1 源码含 %s' % key, key in src)

ok = C.summary()
sys.exit(0 if ok else 1)
