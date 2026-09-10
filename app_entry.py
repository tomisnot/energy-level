# -*- coding: utf-8 -*-
"""Rb-87 能级计算软件本体（MainWindow + 启动 main）。

本模块承载原 能级.py 的全部功能（物理核心见 core.py，可单独审查），
只负责装配与编排：
- 创建模块组件（静态模块 / 动态模块）
- 组装 tab + 右侧面板 + 切页钩子，统一切换循环
- 菜单（文件→保存/恢复）、全屏布局适配

模块内部业务逻辑由各自组件自行维护，主程序不做逐模块特例。

入口关系（三者等价，命令不同）：
- 开发：python 能级.py   （薄包装，转发本模块 main）
- 直接：python app_entry.py（本模块自带 __main__ 守卫）
- 打包：main.py（PyInstaller 入口，冻结模块方式 import 本模块后调 main；
  冻结环境下磁盘脚本的 import 解析与常规模块不同，故不用 runpy 执行）
"""

import os
import sys

# ── 冻结 windowed 模式标准守卫 ─────────────────────────────────────────
# PyInstaller 窗口程序（console=False）下 sys.stdout/stderr 为 None，任何
# 顶层 sys.stdout.reconfigure()/write 都会 AttributeError。启动即补成
# devnull，之后所有模块（含 pair_core/driver 的 utf-8 重配置）安全无脑。
if sys.stdout is None:
    sys.stdout = open(os.devnull, 'w', encoding='utf-8', errors='replace')
if sys.stderr is None:
    sys.stderr = open(os.devnull, 'w', encoding='utf-8', errors='replace')

import json
import numpy as np
import matplotlib
from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QObject
from PyQt5.QtGui import QColor, QIcon, QPixmap
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QTabWidget, QStackedWidget,
                             QScrollArea, QSizePolicy, QFileDialog,
                             QMessageBox, QLabel, QSpinBox, QDoubleSpinBox,
                             QComboBox, QPushButton, QListWidget,
                             QListWidgetItem, QTableWidget, QTableWidgetItem,
                             QHeaderView, QLineEdit, QMenu, QAbstractItemView)

from arc import Rubidium87
from src.core import core
from src import archive
from src import fields
from src import ui_dynamic
from src import ui_theme
from src.ui.ui_group import EnergyGroupPanel
from src.ui.ui_spectrum import LevelAnalysisPanel

matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False


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
        self.groups = []
        self.groups_store = None
        self.field_state = None
        self._sorted_groups = []
        self._y_view_user_set = False
        self._last_lines_off = None
        self._last_lines_on = None
        self._selected_spec_table = 'off'
        self._selected_spec_row = -1
        self._level_all = []
        self._level_by_key = {}
        self._build_ui()

    def log(self, msg, timeout=8000):
        """统一信息出口：状态栏显示。"""
        if hasattr(self, 'statusBar'):
            self.statusBar().showMessage(str(msg), timeout)

    def closeEvent(self, ev):
        try:
            from src.ai import server as ai_server
            ai_server.stop_server()
        except Exception:
            pass
        super().closeEvent(ev)

    def _init_ai(self):
        """启动进程内 MCP server（AI 协同端点）+ journal → 状态栏联动。

        失败不影响软件本体使用（如无 mcp SDK / 端口占用），仅日志提示。
        """
        try:
            from src.ai import server as ai_server
            from src.ai.journal import get_journal
            port = ai_server.start_server(self)
            if port:
                # P3：状态栏常驻 AI 在线指示（人随时知道 AI 通道开着）
                from PyQt5.QtWidgets import QLabel
                ind = QLabel(f'● AI 在线 :{port}')
                ind.setToolTip(f'MCP 端点 http://127.0.0.1:{port}/mcp\n'
                               'AI 的每次操作会显示在这里，并可一键撤销')
                ind.setStyleSheet('color:#a855f7; padding:0 8px;')
                self.statusBar().addPermanentWidget(ind)
                # P3：人一键撤销最近一个 AI 操作（可逆与可归因的落地）
                from PyQt5.QtWidgets import QPushButton
                undo_btn = QPushButton('撤销AI')
                undo_btn.setToolTip('撤销最近一个 AI 操作（回滚其变更前状态）')
                undo_btn.setStyleSheet('color:#a855f7;')

                def _ui_undo():
                    r = ai_server.ui_undo_last_ai()
                    self.log(r.get('msg', ''), 5000)
                undo_btn.clicked.connect(_ui_undo)
                self.statusBar().addPermanentWidget(undo_btn)
                # P3 协同 UI：AI 来源的 op 滚动到状态栏（人自己的不刷屏）
                j = get_journal(os.path.dirname(os.path.abspath(__file__)))

                def _on_op(op):
                    if op.actor.startswith('ai'):
                        d = op.detail
                        txt = f"[AI] {op.kind}: {d.get('path') or d.get('job') or ''}" \
                              f"{'（' + op.reason + '）' if op.reason else ''}"
                        self.log(txt, 6000)
                j.subscribe(_on_op)
                # 人机写操作统一落账由 param_framework op→journal 桥完成
                # （server.start_server 内 reg.on_op(_journal_bridge) 接线）
            return port
        except Exception as e:
            self.log(f'[AI] 端点启动失败（软件功能不受影响）: {e}')
            return None

    def bind_store(self, groups_store, field_state):
        self.groups_store = groups_store
        self.field_state = field_state
        self.groups = self.groups_store.groups
        self.groups_store.changed.connect(self._on_store_groups_changed)

    def _on_store_groups_changed(self):
        self.groups = self.groups_store.groups
        self.analysis_panel.rebuild_y_combos()
        self.refresh_plot()
        self.groups_changed.emit()

    def _sync_address_from_lights(self):
        if self.field_state is not None:
            self.field_state.sync_address_from_lights()

    def refresh(self):
        self.analysis_panel.radar.refresh()
        self._sync_address_from_lights()
        self.refresh_plot()

    def refresh_plot(self):
        try:
            self.analysis_panel.refresh_view()
        except Exception as e:
            self.log(f'能级图刷新失败: {e}')
            QMessageBox.warning(self, '刷新失败', str(e))

    def _fill_level_table(self):
        fs = self.field_state
        B = fs.B_gauss if fs is not None else 0.0
        ac = fs.address_cfg if fs is not None else {'enabled': False}
        self._level_all = core.build_level_rows(self.cache, B, self.groups, ac, self.groups)
        self._level_by_key = {(r['n'], r['l'], r['j'], r['mj']): r for r in self._level_all if r['kind'] == 'mJ'}
        ref = min((r['E0'] for r in self._level_all), default=0.0)
        lowest = {}
        for r in self._level_all:
            key = (r['n'], r['l'], r['j'])
            if key not in lowest or r['E0'] < lowest[key]['E0']:
                lowest[key] = r
        self.group_panel.level_table.setRowCount(0)
        for g in self.groups:
            key = (g['n'], g['l'], g['j'])
            pix = QPixmap(12, 12)
            pix.fill(QColor(g['color']))
            lo = lowest.get(key)
            for m in np.arange(-g['j'], g['j'] + 1e-9, 1.0):
                r = self.group_panel.level_table.rowCount()
                self.group_panel.level_table.insertRow(r)
                self.group_panel.level_table.setItem(r, 0, QTableWidgetItem(QIcon(pix), g['label']))
                it_m = QTableWidgetItem(f'{m:g}')
                it_m.setTextAlignment(Qt.AlignCenter)
                self.group_panel.level_table.setItem(r, 1, it_m)
                rm = self._level_by_key.get((key[0], key[1], key[2], float(m)))
                if rm is None:
                    rm = lo
                if rm is not None:
                    self.group_panel.level_table.setItem(r, 2, QTableWidgetItem(f"{rm['E0'] - ref:.3f}"))
                else:
                    self.group_panel.level_table.setItem(r, 2, QTableWidgetItem(''))

    def _build_ui(self):
        # 参数注册表先建（面板构造时标准件就地登记 = 定义即注册）
        from src.pm_setup import new_registry
        self.param_reg = new_registry(self)
        main_widget = QWidget(self)
        self.setCentralWidget(main_widget)
        root = QVBoxLayout(main_widget)
        top = QHBoxLayout()
        root.addLayout(top, 1)

        self.group_panel = EnergyGroupPanel(self)
        self.analysis_panel = LevelAnalysisPanel(self)
        self.dynamic = ui_dynamic.DynamicPanel(self, self)
        self.analysis_panel.canvas.peakPicked.connect(self.analysis_panel.on_peak_picked)
        self.analysis_panel.canvas.peakFreqsChanged.connect(self.analysis_panel.update_peak_pair_label)
        self.groups_changed.connect(self.refresh)

        self.field = fields.FieldPanel(self)
        self.field.data_changed.connect(self._on_light_changed)

        from src.ui.app_state import LevelGroupStore, FieldState
        self.groups_store = LevelGroupStore(self.cache, self)
        self.field_state = FieldState(self.field, self)
        self.field_state.sync_address_from_lights()
        self.bind_store(self.groups_store, self.field_state)
        self._on_light_changed()

        control_widget = QWidget()
        control_widget.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        control_layout = QVBoxLayout(control_widget)
        control_layout.setSpacing(2)
        control_layout.addWidget(self.group_panel)
        control_layout.addWidget(self.field)

        self.page_stack = QStackedWidget()
        control_layout.addWidget(self.page_stack, 1)

        self.canvas_tabs = QTabWidget()
        self.modules = [
            ('静态', self.analysis_panel.canvas, self.analysis_panel, self.refresh, self.refresh, True, True),
            ('动态', self.dynamic, self.dynamic.dynamic_widget, self.dynamic.refresh_groups, self.dynamic.refresh_groups, True, False),
        ]
        for name, tab_widget, panel_widget, on_show, on_light, use_field, field_lights in self.modules:
            self.canvas_tabs.addTab(tab_widget, name)
            self.page_stack.addWidget(panel_widget)
        self.canvas_tabs.currentChanged.connect(self._on_canvas_tab_changed)
        top.addWidget(self.canvas_tabs, 3)

        self.panel_scroll = QScrollArea()
        self.panel_scroll.setWidgetResizable(True)
        self.panel_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.panel_scroll.setWidget(control_widget)
        self.control_widget = control_widget
        top.addWidget(self.panel_scroll, 2)

        self._on_canvas_tab_changed(0)
        # 参数登记收尾：标准件已在面板创建处工厂就地登记（定义即注册），
        # 这里补登外部/非标参数（光场表/能级组/波形/psi/衰变/模式）+ 播种。
        # 存档/恢复/AI 三条路都吃这一张册（登记即存档）。
        from src.pm_setup import register_panel_params
        for _panel in ('field', 'group', 'dynamic'):
            register_panel_params(self.param_reg, self, _panel)
        self.param_reg.seed_baseline()
        menubar = self.menuBar()
        fm = menubar.addMenu('文件')
        fm.addAction('保存状态', self._save_state)
        fm.addAction('恢复状态', self._load_state)
        QTimer.singleShot(0, self._apply_fullscreen_layout)

    def _on_canvas_tab_changed(self, idx):
        self.page_stack.setCurrentIndex(idx)
        _, _, _, on_show, _, use_field, field_lights = self.modules[idx]
        self.field.setVisible(use_field)
        self.field.set_lights_visible(field_lights)
        self.panel_scroll.setVisible(True)
        if on_show is not None:
            on_show()

    def _on_light_changed(self, *_):
        if hasattr(self, 'field_state'):
            self.field_state.sync_address_from_lights()
        if not hasattr(self, 'modules'):
            return
        idx = self.canvas_tabs.currentIndex()
        _, _, _, _, on_light, _, _ = self.modules[idx]
        if on_light is not None:
            on_light()

    def _save_state(self):
        path, _ = QFileDialog.getSaveFileName(self, '保存状态', 'rb87_state.json', 'JSON (*.json)')
        if not path:
            return
        snapshot = archive.snapshot(self.param_reg)   # v3：登记册即存档
        try:
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(snapshot, f, ensure_ascii=False, indent=1)
        except Exception as e:
            QMessageBox.warning(self, '保存失败', str(e))
            return
        QMessageBox.information(self, '保存状态', f'已保存 → {path}')

    def _load_state(self):
        path, _ = QFileDialog.getOpenFileName(self, '恢复状态', '', 'JSON (*.json)')
        if not path:
            return
        try:
            with open(path, encoding='utf-8') as f:
                state = json.load(f)
        except Exception as e:
            QMessageBox.warning(self, '恢复失败', str(e))
            return
        try:
            unknown = archive.restore(self.param_reg, state)
            if unknown:
                self.log(f'存档含 {len(unknown)} 个未识别键（已忽略）', 6000)
        except Exception as e:
            QMessageBox.warning(self, '恢复失败', str(e))
            return
        QMessageBox.information(self, '恢复状态', f'已恢复 ← {path}')

    def _apply_fullscreen_layout(self):
        screen = QApplication.primaryScreen()
        if screen is None:
            return
        geo = screen.availableGeometry()
        w, h = geo.width(), geo.height()
        self.panel_scroll.setMinimumWidth(int(w * 0.22))
        self.panel_scroll.setMaximumWidth(int(w * 0.34))
        self.analysis_panel.spec_off_table.setMinimumHeight(max(140, int(h * 0.05)))
        self.analysis_panel.spec_on_table.setMinimumHeight(max(140, int(h * 0.05)))
        self.analysis_panel.radar.setMinimumHeight(max(150, int(h * 0.18)))
        dpi = max(100, min(int(h / 9), 150))
        for fig in (self.analysis_panel.canvas.level_fig, self.analysis_panel.canvas.level_off_fig, self.analysis_panel.canvas.level_on_fig, self.analysis_panel.canvas.spec_mirror_fig):
            fig.set_dpi(dpi)
        self.refresh_plot()


def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.showMaximized()
    window._init_ai()
    return app.exec_()


if __name__ == '__main__':
    sys.exit(main())