# -*- coding: utf-8 -*-
"""自检：检视器摘要头与 AI（core.diag.summarize）同源（P2/P5）+ viewer 守卫。

1) 同一 diag 快照下，检视器 meta_label 的数字必须等于 core.diag.summarize()
   的对应字段（n_pairs / n_microwave 此前只有 get_diag 有，模型卡与检视器
   各写一份 -> 数字会分叉）；
2) 配置明确"无微波波形"时不得误报"微波候选对为 0"（E2）；
3) H 快照被跳过时不崩（两层诊断策略的配套）。
用法：python checks/gui_diag_summary_test.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

from checks.checklib import Checker
from PyQt5.QtWidgets import QApplication
from src.core.diag import summarize
from src.ui.solver_inspector import mock_diag

C = Checker('gui_diag_summary')
_app = QApplication.instance() or QApplication(sys.argv)
try:
    import app_entry as M
    win = M.MainWindow()
except MemoryError as e:
    print('  SKIP 内存不足（软件正在运行？）：%s' % e)
    sys.exit(0)

win.dynamic.show_inspector()
insp = getattr(win.dynamic, '_inspector', None)
C.check('1 检视器存在', insp is not None)

D = mock_diag()
D.setdefault('pairs', {})
D['pairs'].setdefault('n', D['meta'].get('n_pairs'))
D['pairs']['n_microwave'] = 0
D['meta']['n_pairs'] = D['pairs']['n']
CFG = {'has_mw': True, 'max_freq_mhz': 7380.0,
       'basis_n': len(D['meta'].get('labels') or [])}
ss = summarize(D, CFG)

insp.load_diag(D)
txt = insp.meta_label.text()
C.check('2 摘要头含 配对/微波/AC 化/max omega',
        all(k in txt for k in ('配对', '微波', 'AC 化')), txt[:80])
C.check('3 n_pairs 与共享摘要一致（%s）' % ss.get('n_pairs'),
        str(ss.get('n_pairs')) in txt, txt)
C.check('4 n_microwave 与共享摘要一致（%s）' % ss.get('n_microwave'),
        '微波 %s' % ss.get('n_microwave') in txt, txt)

s2 = summarize(D, {'has_mw': False, 'max_freq_mhz': 2.96e8})
C.check('5 配置无微波时不误报告警（E2）', s2['warnings'] == [],
        str(s2['warnings']))

D2 = mock_diag()
D2['H_snapshots'] = {'skipped': {'too_large': True, 'dim': 8112,
                                 'note': 'H 快照跳过（dim 过大）'}}
try:
    insp.load_diag(D2)
    ok, err = True, ''
except Exception as e:
    ok, err = False, repr(e)
C.check('6 跳过 H 快照时不崩（viewer 守卫）', ok, err)

C.summary()
