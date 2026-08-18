# -*- coding: utf-8 -*-
"""UI 快照工具：离屏渲染主窗口两页（静态/动态）→ PNG，供人工/模型查看迭代。

用法: D:\\python310\\python.exe -u checks\\ui_shot.py
输出: _ui_shot_static.png · _ui_shot_dynamic.png
"""
import os, sys, json
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ['QT_QPA_PLATFORM'] = 'offscreen'

from PyQt5.QtWidgets import QApplication, QMessageBox
QMessageBox.information = staticmethod(lambda *a, **k: None)
QMessageBox.warning = staticmethod(lambda *a, **k: None)

import 能级 as main_mod

app = QApplication([])
win = main_mod.MainWindow()
win.resize(1400, 820)
win.show()
app.processEvents()

# 用保存状态给主窗口填充真实内容
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
state_path = os.path.join(ROOT, 'save_state', 'rb87_state.json')
if os.path.exists(state_path):
    try:
        state = json.load(open(state_path, encoding='utf-8'))
        win.dynamic._restore_dynamic(state)
        app.processEvents()
    except Exception as e:
        print(f'恢复状态跳过: {e}')

# 静态页
win.canvas_tabs.setCurrentIndex(0)
app.processEvents()
win.grab().save('_ui_shot_static.png')
print('静态页截图 → _ui_shot_static.png')

# 动态页
win.canvas_tabs.setCurrentIndex(1)
app.processEvents()
win.grab().save('_ui_shot_dynamic.png')
print('动态页截图 → _ui_shot_dynamic.png')
win.close()
