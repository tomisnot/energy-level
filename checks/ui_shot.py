# -*- coding: utf-8 -*-
"""UI 快照工具：离屏渲染主窗口 / 时序波形编辑器 → PNG，供人工/模型查看迭代。

用法: D:\\python310\\python.exe -u checks\\ui_shot.py
输出: _ui_shot_main.png（主窗口，含动态页）· _ui_shot_editor.png（编辑器）
"""
import os, sys, json
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ['QT_QPA_PLATFORM'] = 'offscreen'

import numpy as np
from PyQt5.QtWidgets import QApplication, QMessageBox
QMessageBox.information = staticmethod(lambda *a, **k: None)
QMessageBox.warning = staticmethod(lambda *a, **k: None)

import 能级 as main_mod
from src import ttl_pulse

app = QApplication([])
win = main_mod.MainWindow()
win.resize(1400, 820)
win.show()
app.processEvents()

# 用保存状态给主窗口填充真实内容
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
state_path = os.path.join(ROOT, 'save_state', 'rb87_state.json')
if os.path.exists(state_path):
    state = json.load(open(state_path, encoding='utf-8'))
    win.dynamic._restore_dynamic(state)
    app.processEvents()

# 切到动态页截图（含编辑器）
win.canvas_tabs.setCurrentIndex(2)
app.processEvents()
win.grab().save('_ui_shot_main.png')
print('主窗口截图 → _ui_shot_main.png')

# 编辑器（恢复状态已创建，含配置好的脉冲）
ed = win.dynamic.editor
if ed is not None:
    ed.show()
    app.processEvents()
    ed.grab().save('_ui_shot_editor.png')
    print('编辑器截图 → _ui_shot_editor.png')
    # 下半波形区单独截（看通道列 + 画布 + 参数面板）
    ed.wave.grab().save('_ui_shot_wave.png')
    print('波形区截图 → _ui_shot_wave.png')
    # 上半 TTL 区
    ed.ttl.grab().save('_ui_shot_ttl.png')
    print('TTL 区截图 → _ui_shot_ttl.png')
print('done')
