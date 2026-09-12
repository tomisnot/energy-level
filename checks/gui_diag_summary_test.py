# -*- coding: utf-8 -*-
"""自检：检视器摘要头与 AI（core.diag.summarize）**同源**（P2/P5）。

判据：同一 diag 快照下，检视器 meta_label 里出现的数字必须与
`core.diag.summarize(diag)` 的对应字段一致（尤其 n_pairs / n_microwave ——
这两项此前只有 get_diag 有、模型卡与检视器各写一份）。

用法：python checks/gui_diag_summary_test.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

from checks.checklib import Checker                            # noqa: E402
from PyQt5.QtWidgets import QApplication                       # noqa: E402
from src.core.diag import summarize                            # noqa: E402
try:
    import app_entry as M                                      # noqa: E402
except MemoryError as e:                                       # pragma: no cover
    print('  SKIP 内存不足（软件在跑？）：%s' % e)
    sys.exit(0)

C = Checker('gui_diag_summary')
_app = QApplication.instance() or QApplication(sys.argv)
try:
    win = M.MainWindow()
except MemoryError as e:                                       # pragma: no cover
    print('  SKIP 内存不足（软件在跑？）：%s' % e)
    sys.exit(0)

win.dynamic.show_inspector()      # 检视器懒创建：先显示再取
insp = getattr(win.dynamic, '_inspector', None)
C.check('1 检视器存在', insp is not None)

DIAG = {'meta': {'method': 'mixed', 'backend': 'mixed-phi-dop853-pair', 'N': 30,
                 'P': 3, 'dim': 2700, 't_end': 400.0, 'tol': 1e-8,
                 'max_step': 0.05, 'n_pairs': 150, 'M': [1],
                 'labels': ['a', 'b']},
        'pairs': {'n': 150, 'n_microwave': 0, 'max_abs_v_mhz': 200.0},
        'acize': {'n_shifted': 3, 'shift': [0.0, 1.0]},
        'photon': {'Ms': [1], 'P': 3, 'dim': 2700},
        'solver': {'nfev': 12345, 'n_acc': 678, 'n_rej': 0, 'norm_dev': 1e-7,
                   'dt': [0.01, 0.02]},
        'H_snapshots': {}}
from src.ui.solver_inspector import mock_diag             # noqa: E402
DIAG = mock_diag()          # 检视器自带的完整结构样本（UI 迭代用）
# 补两项共享摘要所需的统计（真实 diag 由引擎给出）
DIAG.setdefault('pairs', {})
DIAG['pairs'].setdefault('n', DIAG['meta'].get('n_pairs'))
DIAG['pairs']['n_microwave'] = 0
DIAG['meta']['n_pairs'] = DIAG['pairs']['n']
ss = summarize(DIAG, {'has_mw': True, 'max_freq_mhz': 7380.0,
                       'basis_n': len(DIAG['meta'].get('labels') or [])})
insp.load_diag(DIAG)      # 真正加载（此前漏了这步，标签还是占位文字）
txt = insp.meta_label.text()
C.check('2 摘要头含"配对"与共享字段', '配对' in txt and '微波' in txt, txt[:60])
C.check('3 n_pairs 与共享摘要一致（%s）' % ss.get('n_pairs'),
        str(ss.get('n_pairs')) in txt, txt)
C.check('4 n_microwave 与共享摘要一致（%s）' % ss.get('n_microwave'),
        '微波 %s' % ss.get('n_microwave') in txt, txt)
C.check('5 AC 化字段在摘要头出现', 'AC 化' in txt, txt)
        insp.meta_label.toolTip()[:50])

# 同一快照、不同配置（无微波）→ 不应出现微波零对告警
s2 = summarize(DIAG, {'has_mw': False, 'max_freq_mhz': 2.96e8})
C.check('8 配置无微波时不误报告警（E2）', s2['warnings'] == [],
        str(s2['warnings']))

C.summary()
