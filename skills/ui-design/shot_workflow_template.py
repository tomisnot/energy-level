# -*- coding: utf-8 -*-
"""离屏 UI 截图工作流模板——AI 自主 UI 迭代的核心工具。

用法：
    1. 复制到项目（如 checks/ui_shot.py），填好下方三个【适配点】；
    2. python checks/ui_shot.py
    3. AI 视觉读取输出的 PNG → 按 checklist.md 自查 → 改代码 → 重跑。

设计要点（每条都对应真实踩坑，勿删）：
- QT_QPA_PLATFORM=offscreen：不弹窗、不干扰用户桌面；
- 主模块 import 在 QApplication 之前：本项目实测顺序颠倒会原生崩溃；
- 显式加载 CJK 字体：offscreen 字体库无中文，否则全部渲染成方框；
- 模拟全屏：目标分辨率 resize（而非 showFullScreen——offscreen 虚拟屏
  只有 800×600，基于屏幕尺寸的自适应布局会被压扁）；
- QMessageBox 打补丁：离线下弹模态对话框会挂死脚本；
- 每次截图前 processEvents：让布局/信号先落地。
"""
import os
import sys

sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ['QT_QPA_PLATFORM'] = 'offscreen'

from PyQt5.QtWidgets import QApplication, QMessageBox

QMessageBox.information = staticmethod(lambda *a, **k: None)
QMessageBox.warning = staticmethod(lambda *a, **k: None)

# ==================================================================
# 【适配点 1】主模块 import —— 必须在 QApplication 之前（本项目实测
# 颠倒会 exit 127 原生崩溃）。换成你的项目主窗口模块。
# ==================================================================
import 能级 as main_mod          # noqa: E402  ← 换成你的

FULL_W, FULL_H = 1920, 1080      # 模拟用户全屏分辨率

app = QApplication([])

# ---- CJK 字体（offscreen 必需，否则中文全是方框） ----
from PyQt5.QtGui import QFontDatabase, QFont  # noqa: E402
for _fn in ('msyh.ttc', 'msyh.ttf', 'simhei.ttf', 'simsun.ttc'):
    _p = os.path.join('C:\\', 'Windows', 'Fonts', _fn)
    if os.path.exists(_p):
        _fid = QFontDatabase.addApplicationFont(_p)
        _fams = QFontDatabase.applicationFontFamilies(_fid)
        if _fams:
            app.setFont(QFont(_fams[0], 10))
            print('UI 字体:', _fams[0])
            break


def build_window():
    """【适配点 2】构造真实主窗口并模拟全屏。"""
    win = main_mod.MainWindow()
    win.resize(FULL_W, FULL_H)
    win.show()
    app.processEvents()
    # 若布局上限基于"屏幕尺寸"（offscreen 虚拟屏仅 800×600），
    # 按真实全屏比例重设——否则面板被压扁：
    # win.panel_scroll.setMinimumWidth(int(FULL_W * 0.22))
    # win.panel_scroll.setMaximumWidth(int(FULL_W * 0.34))
    # app.processEvents()
    return win


def load_scenario(win):
    """【适配点 3】用真实数据填充界面（存档/状态文件），不要空界面。"""
    import json
    state_path = os.path.join(ROOT, 'save_state', '你的存档.json')
    if os.path.exists(state_path):
        try:
            win.dynamic._restore_dynamic(json.load(open(state_path,
                                                        encoding='utf-8')))
            app.processEvents()
            print('已恢复存档')
        except Exception as e:
            print(f'恢复状态跳过: {e}')


def capture_list(win):
    """【适配点 4】返回 [(文件名, 截图 callable), ...]——要截的所有页面。"""
    out = []
    win.canvas_tabs.setCurrentIndex(0)
    app.processEvents()
    out.append(('static', win.grab))
    win.canvas_tabs.setCurrentIndex(1)
    app.processEvents()
    out.append(('dynamic', win.grab))
    return out


# ==================================================================
# 主流程（一般无需修改）
# ==================================================================
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)                                   # PNG 落项目根便于查看

win = build_window()
load_scenario(win)
app.processEvents()

for name, grab_callable in capture_list(win):
    app.processEvents()
    grab_callable().save(f'_ui_shot_{name}.png')
    print(f'截图 _ui_shot_{name}.png')

win.close()
print('完成——用视觉模态读取 PNG，按 checklist.md 自查。')
