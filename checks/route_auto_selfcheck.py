# -*- coding: utf-8 -*-
"""route_auto 轻量自检：构造假 waves 场景验证路由逻辑（不跑演化）。

场景覆盖：
1. 含 ArbitraryWave → hard
2. 纯微波 → hard
3. 弱光 + 微波 → hard
4. 强光 + 微波（单原子）→ mixed
5. 强光纯光场 → sambe
6. 强光 + 微波 两体 → mixed（t62 两体混合正式化——5 倍降维）
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
from src.core.route import route_auto
from src.core.waveforms import Wave, ArbitraryWave, GaussianEnv


def w(name, freq, I=100.0, kind='gauss'):
    return Wave(name, freq, GaussianEnv(100, 20), intensity_mwcm2=I)


# 假能级（任意占位——route 不强依赖）
states = [dict(n=60, l=0, j=0.5), dict(n=60, l=1, j=1.5), dict(n=6, l=1, j=1.5)]

cases = [
    ('含 ArbitraryWave → hard',
     [w('光1', 295.7e6), ArbitraryWave('arb', [0, 10, 20], [0, 1, 0])],
     False, 'hard'),
    ('纯微波 → hard',
     [w('微波', 17236.743)], False, 'hard'),
    ('弱光 + 微波 → hard',
     [w('光弱', 295.7e6, I=1.0), w('微波', 17236.743)], False, 'hard'),
    ('强光 + 微波 单原子 → mixed',
     [w('光强', 295.7e6, I=5e7), w('微波', 17236.743)], False, 'mixed'),
    ('强光纯光场 单原子 → sambe',
     [w('光强', 295.7e6, I=5e7)], False, 'sambe'),
    ('强光 + 微波 两体 → mixed（5 倍降维）',
     [w('光强', 295.7e6, I=5e7), w('微波', 17236.743)], True, 'mixed'),
]

npass = 0
for desc, waves, tb, expect in cases:
    m, reason = route_auto(states, waves, two_body=tb)
    ok = m == expect
    npass += ok
    print(f'[{"PASS" if ok else "FAIL"}] {desc}\n       → {m}（期望 {expect}）— {reason}')
print(f'\nroute_auto 自检: PASS={npass} FAIL={len(cases)-npass}')
