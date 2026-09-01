# -*- coding: utf-8 -*-
"""全局 UI 主题（单一 QSS 来源）：主窗口 / 对话框 / 表格 / 状态栏统一风格。

用法：
    from src import ui_theme
    ui_theme.apply(app)          # 主程序入口调用一次
    ui_theme.QSS                  # 需要局部再注入时（如独立窗口）直接用

设计：中性灰阶 + 圆角 + 悬停态（无 emoji / 无彩色），与 TTL 编辑器原主题一致，
覆盖全部常用控件；所有字体/颜色/尺寸收敛为常量，便于统一调整。

t29：QComboBox 全局样式接管绘制后原生下拉箭头消失（与按钮难区分）——
补充 QComboBox::down-arrow 三角图像（assets/down_arrow.png，中性灰
COL_TEXT_DIM #6b7280，纯 Python 生成已入库；缺失时运行时 fallback 就地
生成，仍失败则 QSS 不设 image，不崩溃）。
"""

import os

# ---------------- 主题常量（统一调整入口） ----------------
FONT_FAMILY = 'Microsoft YaHei'
FONT_SIZE = '10pt'
FONT_SIZE_LG = '11pt'
CHANNEL_COL_W = 124       # 时序/波形编辑器左侧通道列统一宽度（容纳"＋ 添加通道"）
COL_BG = '#f2f3f5'          # 窗口背景
COL_PANEL = '#ffffff'       # 面板/输入框背景
COL_BORDER = '#c9ced6'
COL_BORDER_HOVER = '#a7b0ba'
COL_TEXT = '#2c3137'
COL_TEXT_DIM = '#6b7280'
COL_ACCENT = '#7f93a8'      # 焦点/选中
COL_HEAD = '#edf0f3'        # 表头
COL_ROW_ALT = '#f7f8fa'     # 交替行
COL_GRID = '#e4e6ea'
COL_DANGER = '#c0392b'

_ARROW_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           'assets', 'down_arrow.png')


def _down_arrow_url():
    """down-arrow PNG 绝对路径（正斜杠，QSS url 用）；缺失时 try 就地生成
    （纯 Python 生成器 scripts/make_down_arrow.py，无 Qt 依赖）；仍失败
    返回 None——QSS 不设 image（Qt 不渲染箭头，不崩溃）。"""
    if not os.path.isfile(_ARROW_PATH):
        try:
            from scripts.make_down_arrow import make_png
            data = make_png()
            os.makedirs(os.path.dirname(_ARROW_PATH), exist_ok=True)
            with open(_ARROW_PATH, 'wb') as f:
                f.write(data)
        except Exception:
            return None
    return _ARROW_PATH.replace('\\', '/')


_DOWN_ARROW_URL = _down_arrow_url()
_DOWN_ARROW_QSS = (
    f'QComboBox::down-arrow {{ image: url("{_DOWN_ARROW_URL}"); '
    f'width: 12px; height: 8px; }}'
    if _DOWN_ARROW_URL else '')

QSS = f"""
* {{ font-family: '{FONT_FAMILY}'; font-size: {FONT_SIZE}; }}
QMainWindow, QDialog, QWidget {{ background: {COL_BG}; color: {COL_TEXT}; }}
QToolBar {{ background: {COL_PANEL}; border: none; border-bottom: 1px solid {COL_BORDER};
           padding: 5px 8px; spacing: 6px; }}
QToolBar QToolButton {{ background: transparent; border: 1px solid transparent;
           border-radius: 5px; padding: 4px 10px; color: {COL_TEXT}; }}
QToolBar QToolButton:hover {{ background: #eceef1; border-color: {COL_BORDER}; }}
QToolBar QToolButton:pressed {{ background: #e0e3e7; }}
QToolBar::separator {{ background: {COL_BORDER}; width: 1px; margin: 4px 2px; }}
QPushButton {{ background: {COL_PANEL}; border: 1px solid {COL_BORDER}; border-radius: 5px;
              padding: 5px 14px; color: {COL_TEXT}; }}
QPushButton:hover {{ background: #eef1f4; border-color: {COL_BORDER_HOVER}; }}
QPushButton:pressed {{ background: #e0e4e9; }}
QPushButton:disabled {{ color: #aab0b8; background: #f0f1f3; }}
QDoubleSpinBox, QSpinBox, QComboBox, QLineEdit, QTextEdit, QPlainTextEdit {{
    background: {COL_PANEL}; border: 1px solid {COL_BORDER}; border-radius: 4px;
    padding: 3px 6px; color: {COL_TEXT}; selection-background-color: #d4dbe3; }}
QDoubleSpinBox:focus, QSpinBox:focus, QComboBox:focus, QLineEdit:focus,
QTextEdit:focus, QPlainTextEdit:focus {{ border-color: {COL_ACCENT}; }}
QComboBox::drop-down {{ border: none; width: 18px; }}
{_DOWN_ARROW_QSS}
QComboBox QAbstractItemView {{ background: {COL_PANEL}; border: 1px solid {COL_BORDER};
    selection-background-color: #d4dbe3; selection-color: {COL_TEXT}; }}
QTableWidget, QTableView {{
    background: {COL_PANEL}; alternate-background-color: {COL_ROW_ALT};
    gridline-color: {COL_GRID}; border: 1px solid {COL_BORDER}; border-radius: 5px; }}
QTableWidget::item, QTableView::item {{ padding: 2px 4px; }}
QHeaderView::section {{ background: {COL_HEAD}; color: {COL_TEXT}; border: none;
    border-bottom: 1px solid {COL_BORDER}; border-right: 1px solid {COL_BORDER};
    padding: 5px 6px; font-weight: 400; }}
QListWidget {{ background: {COL_PANEL}; border: 1px solid {COL_BORDER}; border-radius: 5px; }}
QListWidget::item {{ padding: 3px 6px; }}
QListWidget::item:selected {{ background: #d4dbe3; color: {COL_TEXT}; }}
QScrollArea {{ border: 1px solid {COL_BORDER}; border-radius: 6px; background: {COL_BG}; }}
QSplitter::handle {{ background: {COL_BORDER}; }}
QSplitter::handle:hover {{ background: #c3c9d1; }}
QScrollBar:vertical {{ background: transparent; width: 10px; margin: 0; }}
QScrollBar::handle:vertical {{ background: #c3c9d1; border-radius: 5px; min-height: 24px; }}
QScrollBar::handle:vertical:hover {{ background: #aab3bd; }}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height: 0; }}
QScrollBar:horizontal {{ background: transparent; height: 10px; margin: 0; }}
QScrollBar::handle:horizontal {{ background: #c3c9d1; border-radius: 5px; min-width: 24px; }}
QScrollBar::handle:horizontal:hover {{ background: #aab3bd; }}
QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {{ width: 0; }}
QGraphicsView {{ background: {COL_PANEL}; border: 1px solid {COL_BORDER}; border-radius: 6px; }}
QToolButton {{ background: transparent; border: 1px solid transparent;
               border-radius: 5px; padding: 3px 9px; }}
QToolButton:hover {{ background: #eceef1; border-color: {COL_BORDER}; }}
QToolButton:checked {{ background: #e2e7ed; border-color: #c3c9d1; }}
QToolButton::menu-indicator {{ image: none; }}
QMenuBar {{ background: {COL_PANEL}; border-bottom: 1px solid {COL_BORDER}; }}
QMenuBar::item {{ padding: 5px 10px; background: transparent; }}
QMenuBar::item:selected {{ background: #eceef1; border-radius: 4px; }}
QMenu {{ background: {COL_PANEL}; border: 1px solid {COL_BORDER}; border-radius: 6px; padding: 4px; }}
QMenu::item {{ padding: 5px 22px; border-radius: 4px; }}
QMenu::item:selected {{ background: #d4dbe3; }}
QStatusBar {{ background: {COL_PANEL}; border-top: 1px solid {COL_BORDER}; color: {COL_TEXT_DIM}; }}
QStatusBar QLabel {{ color: {COL_TEXT_DIM}; }}
QProgressDialog {{ background: {COL_PANEL}; border: 1px solid {COL_BORDER}; }}
QMessageBox {{ background: {COL_PANEL}; }}
QToolTip {{ background: #3a3f45; color: #ffffff; border: none; padding: 4px 6px; }}
QGroupBox {{ border: 1px solid {COL_BORDER}; border-radius: 6px; margin-top: 8px;
             padding-top: 4px; }}
QGroupBox::title {{ subcontrol-origin: margin; left: 8px; padding: 0 4px; }}
QLabel {{ background: transparent; }}
QTabWidget::pane {{ border: 1px solid {COL_BORDER}; border-radius: 5px; top: -1px; }}
QTabBar::tab {{ background: #e8eaee; border: 1px solid {COL_BORDER}; border-bottom: none;
                border-top-left-radius: 5px; border-top-right-radius: 5px;
                padding: 5px 14px; margin-right: 2px; }}
QTabBar::tab:selected {{ background: {COL_PANEL}; font-weight: 600; }}
QTabBar::tab:hover:!selected {{ background: #f0f2f5; }}
"""


def apply(app):
    """把全局主题应用到 QApplication（幂等）。"""
    app.setStyleSheet(QSS)
    return QSS
