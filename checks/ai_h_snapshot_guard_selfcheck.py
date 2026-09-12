# -*- coding: utf-8 -*-
"""自检：诊断 H 快照的尺寸闸（headless；不分配大矩阵）。"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

FAIL = []


def chk(cond, msg):
    print(('  OK   ' if cond else '  FAIL ') + msg)
    if not cond:
        FAIL.append(msg)


from src.core.evolve_mixed import (                              # noqa: E402
    _assemble_H_snapshot, _H_SNAPSHOT_MAX_MB)

print('1) 大维数（8112）→ 只回元信息，不分配')
big = _assemble_H_snapshot(np.zeros(8112), [], 0.0, 0.157)
chk(isinstance(big, dict) and big.get('too_large') is True,
    '返回 too_large=True（dim=%s）' % big.get('dim'))
chk(abs(big.get('term_mb_each', 0) - 1003.7) < 2.0,
    '单项 MB 与弹窗数字一致：%.1f MB（8112²×16B = 1004 MiB）'
    % big.get('term_mb_each', 0))
chk(big.get('limit_mb') == _H_SNAPSHOT_MAX_MB, '报出阈值 %s' % big.get('limit_mb'))
chk('note' in big and 'H 快照跳过' in big['note'], '给出可读说明')

print('2) 小维数 → 行为不变（仍建分项矩阵）')
small = _assemble_H_snapshot(np.array([1.0, 2.0, 3.0, 4.0]), [], 0.0, 0.157)
chk(isinstance(small, dict) and not small.get('too_large'),
    '未触发尺寸闸')
chk('K0' in small and 'AC对角' in small and '耗散' in small,
    '分项齐全：%s' % sorted(small)[:5])
chk(np.asarray(small['AC对角']).shape == (4, 4), 'AC 对角为 4×4 稠密（小维数正常）')

print('3) 阈值语义：恰好等于阈值不触发、略超即触发')
n_ok = int((_H_SNAPSHOT_MAX_MB * 1024 ** 2 / 16.0) ** 0.5)
r_ok = _assemble_H_snapshot(np.zeros(n_ok), [], 0.0, 0.157)
r_no = _assemble_H_snapshot(np.zeros(n_ok + 2), [], 0.0, 0.157)
chk(not r_ok.get('too_large'), 'dim=%d（≈%.0f MB）不触发' % (n_ok, _H_SNAPSHOT_MAX_MB))
chk(r_no.get('too_large') is True, 'dim=%d 触发' % (n_ok + 2))

print('\n结果：%s' % ('全部通过' if not FAIL else '%d 项失败' % len(FAIL)))
sys.exit(1 if FAIL else 0)
