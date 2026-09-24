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
from PyQt5.QtWidgets import (QButtonGroup, QPushButton, QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QTabWidget, QStackedWidget,
                             QScrollArea, QSizePolicy, QFileDialog,
                             QMessageBox, QLabel, QSpinBox, QDoubleSpinBox,
                             QComboBox, QPushButton, QListWidget,
                             QListWidgetItem, QTableWidget, QTableWidgetItem,
                             QHeaderView, QLineEdit, QMenu, QAbstractItemView)

from src.core import core
from src.ui import fields
from src.ui import ui_dynamic
from src.ui import ui_manybody
from src.ui import ui_theme
from src.ui import _mecha_bind
from src.ui.ui_group import EnergyGroupPanel
from src.ui.ui_spectrum import LevelAnalysisPanel

# mecha v2：GUI 与 AI（MCP 服务器线程）共一 Gate、一份 History（v1 门面已退役）。
from src.v2_bind import make_v2_bridge

matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False


_MODE_QSS = (
    'QPushButton { padding: 3px 12px; border: 1px solid #d6d9de;'
    ' background: #ffffff; color: #5b6270; font-size: 9pt; }'
    'QPushButton:checked { background: #eef2ff; color: #2b3a67;'
    ' font-weight: 600; border-color: #8fa6e8; }'
    'QPushButton:hover { background: #f3f5f9; }'
)

# 项目根（= 本文件所在目录）：v2 数据根（`.mecha-v2/` journal+artifacts）与
# `.mcp-port` / `.human-token` 的**唯一定义点**——GUI 进程内装配（make_v2_bridge）
# 与 run_ai() 等的 .mcp-port 必须同源，否则"起得来但看不见"（假绿）。
PROJ_ROOT = os.path.dirname(os.path.abspath(__file__))


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

        # mecha v2 门面：GUI 与 AI（本进程后台 MCP 服务器线程）**共享一个 Gate、
        # 一份 History**；另起人工控制端点供夺权/交权（actor 服务端固定 human）。
        # 根 = 模块级 `PROJ_ROOT`（唯一定义点，与 run_ai() 等的 .mcp-port 同源）。
        self.app_bridge = make_v2_bridge(PROJ_ROOT)
        self.cache = self.app_bridge.cache
        # 面板控件 ↔ session。**notify=self.log**：写权被拒等"必须让人看见"的反馈
        # 走状态栏（P2-4）——没有它，`Binder._reject` 就只是"声明了没人接"的空机制。
        # `self.log(msg, timeout)` 与 `notify(msg, timeout)` 签名对齐（见 `_reject` 的超时语义）。
        self.binder = _mecha_bind.Binder(self.app_bridge, notify=self.log)
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
        # 视图模式位（极简/上帝 = 检视器显隐）；写权（human/AI）是另一维，走
        # app_bridge.switch_mode。engine.* 只读（点1），故不再有引擎覆写状态。
        self.god_mode = False
        self._build_ui()
        # 第一交付物（总纲 §1.2 事实 2）：人监控 AI 的侧边栏，两层信任。
        # 须在 _build_ui 之后挂（menuBar 那时才存在）。
        from src.ui.monitor_panel import attach_monitor_dock
        attach_monitor_dock(self)

    def log(self, msg, timeout=8000):
        """统一信息出口：状态栏显示。"""
        if hasattr(self, 'statusBar'):
            self.statusBar().showMessage(str(msg), timeout)

    def closeEvent(self, ev):
        super().closeEvent(ev)


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
        # P1-7：AC-Stark 精度（窗口/光子块）——静态页唯一来源 = 场面板 ac_kwargs()；
        # 无场面板时 None=引擎默认（与接线前逐位一致）。
        _ac_kw = self.field.ac_kwargs() if getattr(self, 'field', None) else None
        self._level_all = core.build_level_rows(self.cache, B, self.groups, ac,
                                                self.groups, **(_ac_kw or {}))
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
        # 面板控件经 _mecha_bind 就地绑到 session（定义即注册）；binder 已在 __init__ 建。
        main_widget = QWidget(self)
        self.setCentralWidget(main_widget)
        root = QVBoxLayout(main_widget)
        top = QHBoxLayout()
        root.addLayout(top, 1)

        self.group_panel = EnergyGroupPanel(self)
        self.analysis_panel = LevelAnalysisPanel(self)
        self.dynamic = ui_dynamic.DynamicPanel(self, self)
        # 多体页（P3-1）：单独一页，自定版式（裁决③）。参数面板进 page_stack
        # ⇒ 与动态页**互斥占位**，不与它抢右侧面板；既有页控件一行不动。
        self.manybody = ui_manybody.ManyBodyPanel(self, self)
        self.analysis_panel.canvas.peakPicked.connect(self.analysis_panel.on_peak_picked)
        self.analysis_panel.canvas.peakFreqsChanged.connect(self.analysis_panel.update_peak_pair_label)
        self.groups_changed.connect(self.refresh)

        self.field = fields.FieldPanel(self)
        self.field.data_changed.connect(self._on_light_changed)

        # 常驻计算进度（2026-09-14 使用者裁决）：固定在最底下一栏，文本在旁；
        # 人为与 AI 发起的运行**共用**这两个控件（同一条 _run / _poll_progress
        # / _cancel_run 路径），故人机行为完全一致。
        from PyQt5.QtWidgets import QLabel as _QL, QProgressBar as _QPB
        _rl = _QL('空闲')
        _rl.setStyleSheet('color:#666; padding:0 8px;')
        _rp = _QPB()
        _rp.setRange(0, 1000)
        _rp.setFixedWidth(180)
        _rp.setValue(0)
        _rp.setToolTip('演化进度（人与 AI 同源）；运行中"运行演化"按钮变为'
                       '"取消计算"，点它即取消')
        self.statusBar().addPermanentWidget(_rl)
        self.statusBar().addPermanentWidget(_rp)
        self._run_label, self._run_pbar = _rl, _rp
        self.dynamic._stage_label = _rl       # 动态页直接写这两个控件
        self.dynamic._pbar = _rp

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
            # 多体（P3-1）：场面板保留可见（manybody.B_gauss 缺省复用 field.B_gauss）；
            # 光场表隐藏——多体的驱动来自 atoms 的 Ω/δ，不是场面板的光场表。
            ('多体', self.manybody, self.manybody.manybody_widget, self.manybody.on_show, self.manybody.on_light, True, False),
        ]
        for name, tab_widget, panel_widget, on_show, on_light, use_field, field_lights in self.modules:
            self.canvas_tabs.addTab(tab_widget, name)
            self.page_stack.addWidget(panel_widget)
        self.canvas_tabs.currentChanged.connect(self._on_canvas_tab_changed)
        top.addWidget(self.canvas_tabs, 3)

        # 并行扫描的人侧可见面（2026-09-16 用户裁决：**不是弹窗**，而是底栏
        # 之上**向上展开**的一栏；画布顶部那行字已删除、职能并入本栏表头）。
        # 它加在 `root`（中央布局）里 `top` **之后** ⇒ 位于底栏正上方；
        # 隐藏时不占高度，展开时把画布往上挤——"向上展开"就是布局的自然行为。
        # 数据只读 src.sweep.registry.status()，与 AI 的 sweep_status 读同一份
        # status.json（P2 人机同源）；取消只写 cancel 文件、不阻塞主线程。
        from src.ui.sweep_panel import SweepStack
        self.sweep_stack = SweepStack(self)
        root.addWidget(self.sweep_stack)

        self.panel_scroll = QScrollArea()
        self.panel_scroll.setWidgetResizable(True)
        self.panel_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.panel_scroll.setWidget(control_widget)
        self.control_widget = control_widget
        top.addWidget(self.panel_scroll, 2)

        self._on_canvas_tab_changed(0)
        # 面板控件经 _mecha_bind 已就地绑到 session；非标量（能级组/波形/psi/衰变）
        # 由各自面板/run 流程写 session（见 ui_dynamic run 流）。存档=session 快照。
        menubar = self.menuBar()
        fm = menubar.addMenu('文件')
        fm.addAction('保存状态', self._save_state)
        fm.addAction('恢复状态', self._load_state)
        self._build_mode_switch(menubar)
        QTimer.singleShot(0, self._apply_fullscreen_layout)

    # ---------- 模式切换（极简 / 上帝）----------
    # t82：上帝模式是**凌驾于各页之上的模式**，不是某一页的复选框 ——
    # 开关放在菜单栏那一行（QMenuBar 右上角 corner widget），两个互斥按钮。
    # 开上帝模式：检视器自动显示 + 运行采集 diag（更重的诊断） + 标题/徽标标记；
    # 极简模式：检视器隐藏但**对象与绑定照旧存在**（登记跟真值源走，不跟显隐走）。
    def _build_mode_switch(self, menubar):
        box = QWidget()
        hl = QHBoxLayout(box)
        hl.setContentsMargins(0, 2, 8, 2)
        hl.setSpacing(0)
        self._mode_btns = {}
        grp = QButtonGroup(self)
        grp.setExclusive(True)
        for key, text, tip in (
                ('simple', '极简模式', '只做建模与算例：检视器窗口隐藏，'
                                       '其余行为不变'),
                ('god', '上帝模式', '求解过程可视化 + 引擎参数覆写：'
                                    '检视器自动显示，运行采集诊断')):
            b = QPushButton(text)
            b.setCheckable(True)
            b.setCursor(Qt.PointingHandCursor)
            b.setToolTip(tip)
            b.setStyleSheet(_MODE_QSS)
            grp.addButton(b)
            hl.addWidget(b)
            self._mode_btns[key] = b
            b.clicked.connect(lambda _c, k=key: self.set_mode(god=(k == 'god')))
        self._mode_group = grp
        menubar.setCornerWidget(box, Qt.TopRightCorner)
        self.set_mode(god=False)          # 默认极简

    def set_mode(self, god):
        """切模式（菜单栏按钮 / 代码 / 将来 AI 都走这里，唯一入口）。"""
        god = bool(god)
        if god == self.god_mode and getattr(self, '_mode_ready', False):
            return
        self.god_mode = god
        self._mode_ready = True
        self._mode_btns['god'].setChecked(god)
        self._mode_btns['simple'].setChecked(not god)
        self.setWindowTitle('Rb-87 能级计算' + ('　· 上帝模式' if god else ''))
        if hasattr(self, 'dynamic'):
            self.dynamic.set_god_mode(god)
        self.log('已切到' + ('上帝模式（检视器已打开）' if god else '极简模式'))

    def apply_solve_params(self, values):
        """把 solve.* 运行级参数写进 session（经 choke point）；engine.* 只读忽略
        （点1：引擎内核常量不开放覆写）。调用方：ui_dynamic 成本档
        （`_preflight_cost` 选"剔除远失谐对"时写 solve.COST_BUDGET）。

        **P0-7**：原来这里还挂着 `_on_engine_param_edited`（检视器改单个参数的第二个
        写入口）——它全库无任何连接点，且与 `solver_inspector._on_param_edit`
        （检视器参数页自己的写路径）重复。已删除：同一键只留**一个**人类写面。
        """
        for k, v in (values or {}).items():
            if str(k).startswith('solve.'):
                try:
                    self.app_bridge.write(k, v, reason='apply_solve_params',
                                          actor='gui')
                except Exception:                      # noqa: BLE001
                    pass

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
        try:
            self.app_bridge.save(path, reason='GUI 保存状态')   # v2 快照 dict（{键:值}）
        except Exception as e:                          # noqa: BLE001
            QMessageBox.warning(self, '保存失败', str(e))
            return
        QMessageBox.information(self, '保存状态', f'已保存 → {path}')

    def _load_state(self):
        path, _ = QFileDialog.getOpenFileName(self, '恢复状态', '', 'JSON (*.json)')
        if not path:
            return
        try:
            self.app_bridge.load(path, reason='GUI 恢复状态')   # 经 Gate 原子批量写→订阅刷新
        except Exception as e:                          # noqa: BLE001
            QMessageBox.warning(self, '恢复失败', str(e))
            return
        self.refresh()
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


def run_gui(app):
    """GUI 模式：本地操控面板（进程内自托管 mecha 内核），人自己开车。"""
    window = MainWindow()
    window.showMaximized()
    return app.exec_()


def run_ai():
    """AI 模式：后台起 GUI（v2 同进程权威装配），**前台跑 dsh**（AI harness）。

    v2 没有独立的 Hub/驾驶舱进程：GUI 进程本身就是权威装配（Gate/History +
    MCP 服务器线程 + 人工控制端点），写 `<PROJ_ROOT>/.mcp-port`；AI 经 MCP
    协议连它，与 GUI 过**同一道门**。监控 = GUI 侧边栏（驾驶舱的职责由
    Monitor 面板接替）。dsh 启动慢 → 放**前台**，它的启动输出/进度直接进本
    控制台，你能看着它起来。dsh 退出或 Ctrl+C → 收尾 GUI。
    """
    import atexit
    import shutil
    import subprocess
    import time
    here = os.path.dirname(os.path.abspath(__file__))
    py = sys.executable
    flags = getattr(subprocess, 'CREATE_NEW_CONSOLE', 0) if os.name == 'nt' else 0
    procs = []

    def _spawn(name, args):
        try:
            p = subprocess.Popen(args, cwd=here, creationflags=flags)
            procs.append((name, p))
            return p
        except Exception as e:                          # noqa: BLE001
            print('[launcher] %s 启动失败：%s' % (name, e))
            return None

    def _cleanup():
        for _n, p in procs:
            try:
                if p.poll() is None:
                    p.terminate()
            except Exception:                          # noqa: BLE001
                pass
    atexit.register(_cleanup)

    # 后台服务：GUI（--gui 跳过模式弹窗）。它就是权威本体：MCP 端点与人工
    # 控制端点都在它的进程里，journal/artifacts 落 `<PROJ_ROOT>/.mecha-v2/`。
    _spawn('GUI', [py, os.path.join(here, 'app_entry.py'), '--gui'])

    # 等 GUI 写好 .mcp-port（dsh 的 re0-mecha 插件据此发现并连 MCP 端点）——带进度点
    pf = os.path.join(here, '.mcp-port')
    print('[launcher] 等 GUI 就绪（.mcp-port）', end='', flush=True)
    ready = False
    for _ in range(160):                                 # cache 构建 + 装配，最多 ~40s
        try:
            if os.path.exists(pf) and open(pf, encoding='utf-8').read().strip():
                ready = True
                break
        except OSError:
            pass
        time.sleep(0.25)
        print('.', end='', flush=True)
    print(' 就绪' if ready else ' 超时（dsh 会退回默认 :8765）')

    dsh_exe = shutil.which('dsh')

    # 端口策略：项目实例**永不占 3080**（那是官方 dsh 的默认端口）。
    # profile 补丁里 remote-web-ui 托管块把 webserver 钉死在字面量 3080，
    # `--port` 会静默失效，且该插件每次启动把实际端口回写共享 profile（机制见
    # mecha-sdk《DSH-实例并存原理》§3）；cordis.project.patch.yml 经 `--patch`
    # （组合链最后一层）关掉该插件并恢复端口表达式 ⇒ --port 生效、官方配置
    # 不被污染、两实例并存，使用者零额外操作。
    def _pick_dsh_port(preferred=3081, tries=10):
        """从 preferred 起找第一个可绑定的回环端口（3080 永不入选）。"""
        import socket
        for p in range(preferred, preferred + tries):
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    s.bind(('127.0.0.1', p))
                return p
            except OSError:
                continue
        return preferred

    dsh_port = _pick_dsh_port()
    dsh_args = ('web --patch ./dsh/cordis.source.patch.yml'
                ' --patch ./dsh/cordis.project.patch.yml'
                ' --port %d' % dsh_port)
    dsh_dir = os.path.join(here, 'dsh')

    # 依赖预检：dsh/node_modules 缺失（新检出/清理后）→ 自动 npm install。
    # 否则 re0-mecha 插件 import @modelcontextprotocol/sdk 失败，dsh 启动即崩。
    if dsh_exe and os.path.isdir(dsh_dir) \
            and not os.path.isdir(os.path.join(dsh_dir, 'node_modules')):
        print('[launcher] dsh/node_modules 缺失 → 自动 npm install（首次约 15-60s）…',
              flush=True)
        try:
            subprocess.run('npm install', shell=True, cwd=dsh_dir)
        except Exception as e:                          # noqa: BLE001
            print('[launcher] ⚠ npm install 失败：%s（dsh 可能起不来）' % e)

    print('=' * 70)
    print('  AI 模式：GUI（含 MCP 端点 + 人工控制端点）已后台起（独立窗、可最小化）')
    print('  下面**前台启动 dsh**（较慢，启动输出直接显示在本窗，请等它起来）：')
    print('  · 本实例绑定 :%d（避开官方 dsh 的 :3080，两实例可并存）' % dsh_port)
    print('  · dsh 起来后会**自动打开浏览器** = AI 界面（URL 也在下面打印）')
    print('  · 监控不用单独开——AI 动作实时进 GUI 侧边栏（Monitor 面板）')
    print('  · 停：本窗 Ctrl+C（一并收尾 GUI）')
    print('=' * 70, flush=True)

    try:
        if dsh_exe:
            # 前台跑 dsh：输出继承本控制台（启动进度/URL 可见），阻塞到 dsh 退出
            rc = subprocess.run('"%s" %s' % (dsh_exe, dsh_args), shell=True,
                                cwd=here).returncode
            if rc not in (0, None):
                print('[launcher] ⚠ dsh 退出码 %s（崩溃？）——看上面的报错。' % rc)
                print('           常见：dsh/node_modules 缺依赖（删掉它重跑会自动 npm'
                      ' install）、patch 路径错、或端口冲突。')
        else:
            print('[launcher] ⚠ 没在 PATH 找到 dsh——请手动运行：dsh %s' % dsh_args)
            print('[launcher] Hub/驾驶舱保持后台运行；本窗回车结束（并收尾后台服务）…')
            input()
    except KeyboardInterrupt:
        print('\n[launcher] Ctrl+C —— 收尾 Hub/驾驶舱…')
    finally:
        _cleanup()
    return 0


def _choose_mode():
    """弹窗选启动模式 → 'ai' / 'gui' / None（取消/关窗）。"""
    box = QMessageBox()
    box.setWindowTitle('Rb-87 能级计算 · 选择启动模式')
    box.setIcon(QMessageBox.Question)
    box.setText('请选择本次启动的模式：')
    box.setInformativeText(
        'AI 模式：后台起 GUI（MCP 端点 + 人工控制端点都在里面）+ dsh harness，'
        'AI 在 dsh 里驱动引擎、你在 GUI 侧边栏监控。\n\n'
        'GUI 模式：本地操控面板（同一道门），人自己开车。\n\n'
        '──── 写权模式（两种启动都从这里开始）────\n'
        '· 启动即「人写」：只有你能改参数。\n'
        '· 侧边栏「交给 AI」按钮（或浏览器开人工控制台 http://127.0.0.1:8767/）'
        '→ 切到「AI 写」：界面变只读，AI 的每笔写都记 actor=ai 进史。\n'
        '· 随时「人接管」夺回；「全员禁写」是维护态。AI 不能给自己解锁。')
    b_ai = box.addButton('AI 模式', QMessageBox.AcceptRole)
    b_gui = box.addButton('GUI 模式', QMessageBox.ActionRole)
    box.addButton('取消', QMessageBox.RejectRole)
    box.setDefaultButton(b_gui)
    box.exec_()
    c = box.clickedButton()
    if c is b_ai:
        return 'ai'
    if c is b_gui:
        return 'gui'
    return None


def main():
    """入口：弹窗选 AI 模式 / GUI 模式，自动运行对应的。

    `--gui`（或 `--ai`）命令行参数跳过弹窗——run_ai() 后台起 GUI 用。
    """
    app = QApplication.instance() or QApplication(sys.argv)
    try:
        app.setStyleSheet(ui_theme.QSS)
    except Exception:                                      # noqa: BLE001
        pass
    if '--gui' in sys.argv:
        return run_gui(app)
    if '--ai' in sys.argv:
        return run_ai()
    mode = _choose_mode()
    if mode == 'gui':
        return run_gui(app)
    if mode == 'ai':
        return run_ai()
    return 0


if __name__ == '__main__':
    sys.exit(main())