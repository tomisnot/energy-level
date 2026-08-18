# -*- coding: utf-8 -*-
"""Rb-87 能级计算软件 UI（物理核心见 core.py，可单独审查）。

本文件只负责装配与编排：
- 创建页面组件（src/pages/static_page.py 静态页、src/ui_dynamic.py 动态页）
- 注册到 Page 表（画布 tab + 右侧面板 + 切页钩子），统一切页循环
- 菜单（文件→保存/恢复）、全屏布局适配

页面内部业务逻辑由各页面组件自包含，主程序不做逐页特例。
"""

import sys
import json
import matplotlib
from PyQt5.QtCore import Qt, QTimer, pyqtSignal
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QTabWidget, QStackedWidget,
                             QScrollArea, QSizePolicy, QFileDialog,
                             QMessageBox)

from arc import Rubidium87
from src import core
from src import fields
from src import ui_dynamic
from src import ui_theme
from src.ui.pages.static_page import StaticPage

matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False


class Page:
    """一个页面的统一注册记录：左侧画布 tab + 右侧特有面板 panel。
    on_show：切到该页时统一刷新；on_light：光场变化时统一刷新（None=不响应）；
    use_field：本页是否显示共用场面板（能级组区始终共用）；
    field_lights：是否显示场面板里的光场表格/关注跃迁/初态区
    （B/E 行恒显，两页共用同一实例，天然互通）。
    加新页只需追加一条记录，切页/布局/tab 全部由统一循环驱动，无逐页特例。"""
    __slots__ = ('name', 'tab', 'panel', 'on_show', 'on_light', 'use_field',
                 'field_lights')

    def __init__(self, name, tab, panel, on_show=None, on_light=None,
                 use_field=True, field_lights=True):
        self.name = name
        self.tab = tab
        self.panel = panel
        self.on_show = on_show
        self.on_light = on_light
        self.use_field = use_field
        self.field_lights = field_lights


class MainWindow(QMainWindow):
    # 共用能级组增删事件（事件驱动，页面订阅刷新，不散布同步调用）
    groups_changed = pyqtSignal()

    def __init__(self):
        super().__init__()
        self.setWindowTitle('Rb-87 能级计算')
        self.resize(1400, 800)
        app = QApplication.instance()
        if app is not None:
            app.setStyleSheet(ui_theme.QSS)
        self.statusBar().showMessage('就绪', 3000)

        self.atom = Rubidium87()
        self.cache = core.ArcCache(self.atom)
        self._build_ui()

    def log(self, msg, timeout=8000):
        """统一信息出口：状态栏显示。"""
        if hasattr(self, 'statusBar'):
            self.statusBar().showMessage(str(msg), timeout)

    def closeEvent(self, ev):
        super().closeEvent(ev)

    def _build_ui(self):
        main_widget = QWidget(self)
        self.setCentralWidget(main_widget)
        root = QVBoxLayout(main_widget)
        top = QHBoxLayout()
        root.addLayout(top, 1)

        # ---- 页面组件（画布 + 右侧面板自包含）----
        self.static = StaticPage(self)
        self.dynamic = ui_dynamic.DynamicPanel(self, self)
        # 静态页能级组增删 → 透传给动态页（共用能级组）
        self.static.groups_changed.connect(self.groups_changed)
        self.static.groups_changed.connect(self.dynamic.refresh_groups)

        # ---- 定义场：静磁场 + 静电场 + 添加光（两页共用；按页显隐）----
        self.field = fields.FieldPanel(self)
        self.field.changed.connect(self._on_light_changed)
        # ---- 共享状态层：能级组 + 场（单一数据源，页面注入）----
        from src.ui.app_state import LevelGroupStore, FieldState
        self.groups_store = LevelGroupStore(self.cache, self)
        self.field_state = FieldState(self.field, self)
        self.field_state.sync_address_from_lights()
        self.static.bind_store(self.groups_store, self.field_state)
        self._on_light_changed()

        # ---- 右侧共用布局：能级组区 → 场面板 → 页面栈（静态面板/动态面板切换）----
        control_widget = QWidget()
        control_widget.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        control_layout = QVBoxLayout(control_widget)
        control_layout.setSpacing(2)     # 紧凑：组件间留 2px
        control_layout.addWidget(self.static.group_widget)   # ① 能级组区（恒显，内容高）
        control_layout.addWidget(self.field)                 # ② 场面板（按页显隐）
        self.page_stack = QStackedWidget()
        control_layout.addWidget(self.page_stack, 1)         # ③ 页面特有区（吃满剩余，
                                                             #    内部 wave_panel/雷达图占满）

        # ---- 页面注册 ----
        self.canvas_tabs = QTabWidget()
        self.pages = [
            Page('静态', self.static.canvas, self.static.panel_widget,
                 on_show=self.static.show, on_light=self.static.on_light),
            Page('动态', self.dynamic, self.dynamic.dynamic_widget,
                 on_show=self.dynamic.refresh_groups,
                 on_light=self.dynamic.refresh_groups,
                 use_field=True, field_lights=False),
        ]
        for p in self.pages:
            self.canvas_tabs.addTab(p.tab, p.name)
            self.page_stack.addWidget(p.panel)
        self.canvas_tabs.currentChanged.connect(self._on_canvas_tab_changed)
        top.addWidget(self.canvas_tabs, 3)

        self.panel_scroll = QScrollArea()
        self.panel_scroll.setWidgetResizable(True)
        self.panel_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.panel_scroll.setWidget(control_widget)
        self.control_widget = control_widget
        top.addWidget(self.panel_scroll, 2)

        # 初始切到静态页
        self._on_canvas_tab_changed(0)
        # 菜单栏：文件 → 保存/恢复状态
        menubar = self.menuBar()
        fm = menubar.addMenu('文件')
        fm.addAction('保存状态', self._save_state)
        fm.addAction('恢复状态', self._load_state)
        QTimer.singleShot(0, self._apply_fullscreen_layout)

    # ---------------- 切页 ----------------
    def _on_canvas_tab_changed(self, idx):
        page = self.pages[idx]
        self.page_stack.setCurrentIndex(idx)
        self.field.setVisible(page.use_field)
        self.field.set_lights_visible(page.field_lights)
        self.panel_scroll.setVisible(True)
        if page.on_show is not None:
            page.on_show()

    def _on_light_changed(self, *_):
        """场/光变化：同步 field_state + 当前页刷新钩子。"""
        if hasattr(self, 'field_state'):
            self.field_state.sync_address_from_lights()
        if not hasattr(self, 'pages'):
            return
        page = self.pages[self.canvas_tabs.currentIndex()]
        if page.on_light is not None:
            page.on_light()

    # ---------------- 状态管理（菜单：文件 → 保存/恢复） ----------------
    def _save_state(self):
        path, _ = QFileDialog.getSaveFileName(
            self, '保存状态', 'rb87_state.json', 'JSON (*.json)')
        if not path:
            return
        try:
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(self.dynamic._serialize_dynamic(), f,
                          ensure_ascii=False, indent=1)
        except Exception as e:
            QMessageBox.warning(self, '保存失败', str(e))
            return
        QMessageBox.information(self, '保存状态', f'已保存 → {path}')

    def _load_state(self):
        path, _ = QFileDialog.getOpenFileName(
            self, '恢复状态', '', 'JSON (*.json)')
        if not path:
            return
        try:
            with open(path, encoding='utf-8') as f:
                state = json.load(f)
        except Exception as e:
            QMessageBox.warning(self, '恢复失败', str(e))
            return
        try:
            self.dynamic._restore_dynamic(state)
        except Exception as e:
            QMessageBox.warning(self, '恢复失败', str(e))
            return
        QMessageBox.information(self, '恢复状态', f'已恢复 ← {path}')

    # ---------------- 全屏布局适配 ----------------
    def _apply_fullscreen_layout(self):
        screen = QApplication.primaryScreen()
        if screen is None:
            return
        geo = screen.availableGeometry()
        w, h = geo.width(), geo.height()
        self.panel_scroll.setMinimumWidth(int(w * 0.22))
        self.panel_scroll.setMaximumWidth(int(w * 0.34))
        st = self.static
        st.spec_off_table.setMinimumHeight(max(140, int(h * 0.05)))
        st.spec_on_table.setMinimumHeight(max(140, int(h * 0.05)))
        st.radar.setMinimumHeight(max(150, int(h * 0.18)))
        dpi = max(100, min(int(h / 9), 150))
        for fig in (st.canvas.level_fig, st.canvas.level_off_fig,
                    st.canvas.level_on_fig, st.canvas.spec_mirror_fig):
            fig.set_dpi(dpi)
        st.refresh_plot()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    # 最大化而非真全屏：保留右上角最小化/最大化/关闭三个窗口按钮
    window.showMaximized()
    sys.exit(app.exec_())
