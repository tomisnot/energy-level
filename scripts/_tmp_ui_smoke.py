# -*- coding: utf-8 -*-
"""冒烟：EnergyPlotPanel 移入 src/ui_single.py 后，能级.py 仍正常。"""
import os, sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ['QT_QPA_PLATFORM'] = 'offscreen'

from src import core, ui_pair, ui_fields
from src.ui_single import EnergyPlotPanel, COLOR_CYCLE
print('src import OK; COLOR_CYCLE len =', len(COLOR_CYCLE))

from PyQt5.QtWidgets import QApplication
import 能级
app = QApplication(sys.argv)
w = 能级.MainWindow()
w.n_spin.setValue(60); w.l_spin.setValue(0); w.j_spin.setValue(0.5); w.add_group()
w.n_spin.setValue(60); w.l_spin.setValue(1); w.j_spin.setValue(0.5); w.add_group()
w.add_light()
w.field.table.item(0, 1).setText('16.8267')
w.field._table_changed()

# 单体 exchange → 旧版画布（EnergyPlotPanel）正常
w.field.focus_box.setCurrentIndex(1)
w.refresh_single()
print('exchange stack idx =', w.single_stack.currentIndex(),
      '| canvas type =', type(w.canvas).__name__)
print('level rows =', len(w.canvas._rows), '| off lines =', len(w.canvas._lines_off))
app.processEvents()

# 单体响应谱
w.field.focus_box.setCurrentIndex(0)
w.refresh_single()
print('single resp stack idx =', w.single_stack.currentIndex(),
      '| resp =', w.resp_status.text())
app.processEvents()

# 两体响应谱
w.canvas_tabs.setCurrentIndex(1)
print('pair resp =', w.resp_status.text())
app.processEvents()
print('SMOKE OK')
