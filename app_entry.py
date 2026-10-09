# -*- coding: utf-8 -*-
"""Rb-87 能级计算软件本体（MainWindow + 启动 main）。

本模块承载原 能级.py 的全部功能（物理核心见 core.py，可单独审查），
只负责装配与编排：
- 创建模块组件（静态模块 / 动态模块）
- 组装 tab + 右侧面板 + 切页钩子，统一切换循环
- 菜单（文件→保存/恢复）、全屏布局适配

模块内部业务逻辑由各自组件自行维护，主程序不做逐模块特例。

入口（本文件只起【本软件自己的一个形态】，不编排进程）：
- `--mcp`：MCP 服务形态（headless；由框架 supervisor 起）
- `--gui`：本地 GUI（人或 human adapter 起）
- 起整站（含 harness 与进程编排）用：`python -m mecha --all`
"""

import os
import queue
import threading
import sys

# ── 无控制台启动的标准守卫 ─────────────────────────────────────────────
# 以 pythonw.exe（或任何无控制台方式）启动时 sys.stdout/stderr 为 None，
# 任何顶层 sys.stdout.reconfigure()/write 都会 AttributeError。启动即补成
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
from mecha.authority import Mode
from mecha.errors import MechaError

def setup_cjk_fonts():
    """让 matplotlib 图上的中文不用方框（**必须在 ARC 导入之后再调**）。

    真因（实测）：`arc` 包在 import 时就把自己的绘图风格写进全局 rcParams
    （`arc/calculations_atom_pairstate.py:82` 设 `font.family='serif'`）⇒ 在它之后
    创建的每个 Text 都解析到 DejaVu Serif，中文全变方框。而 ARC 过去是在
    `MainWindow()` 里建 cache 时才被导入——比本模块的设定时机晚。
    所以这里**先主动 import arc**（它本来就要用，不增加成本），把它的覆盖提前完
    成，再设我们的字体；不能反过来（晚设会被晚导入的 arc 盖掉）。
    """
    matplotlib.rcParams['font.family'] = 'sans-serif'
    matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei',
                                              'DejaVu Sans']
    matplotlib.rcParams['axes.unicode_minus'] = False


try:                                        # 先让 arc 把 rcParams 改完
    import arc                              # noqa: F401
except Exception:                           # noqa: BLE001
    pass                                    # 无 ARC 时走纯逻辑路径，不影响字体
setup_cjk_fonts()


_MODE_QSS = (
    'QPushButton { padding: 3px 12px; border: 1px solid #d6d9de;'
    ' background: #ffffff; color: #5b6270; font-size: 9pt; }'
    'QPushButton:checked { background: #eef2ff; color: #2b3a67;'
    ' font-weight: 600; border-color: #8fa6e8; }'
    'QPushButton:hover { background: #f3f5f9; }'
)

# 项目根（= 本文件所在目录）：v2 数据根（`.mecha-v2/` journal+artifacts）与
# `.mcp-port` 的**唯一定义点**——任何形态（GUI / MCP 服务）都从同一处装配
# （`make_v2_bridge`）⇒ 端口文件与数据根同源，否则
# "起得来但看不见"（假绿）。
PROJ_ROOT = os.path.dirname(os.path.abspath(__file__))


class MainWindow(QMainWindow):
    # 共用能级组增删事件（事件驱动，页面订阅刷新，不散布同步调用）
    groups_changed = pyqtSignal()

    def __init__(self, bridge=None):
        super().__init__()
        self.setWindowTitle('Rb-87 能级计算')
        self.resize(1400, 800)
        app = QApplication.instance()
        if app is not None:
            app.setStyleSheet(ui_theme.QSS)
        self.statusBar().showMessage('就绪', 3000)

        # mecha v2 门面：GUI 模式 = 纯本地软件，**不起 MCP/监控端点**（AI 物理
        # 看不见这台软件；想动 AI 走看门人切到 AI 模式）。同一道 Gate、同一份
        # History 的机制照旧（写权恒 human，Binder/只读门控保留待库语义）。
        # 根 = 模块级 `PROJ_ROOT`（唯一定义点，与权威进程的 .mcp-port 同源）。
        #: ⭐ 三层架构：**窗口复用注入的桥**（软件进程内开窗 = `--mcp --with-gui`）。
        #: 没有注入 ⇒ 独立进程那一档（`--gui`）：它仍**自建**桥（纯本地、不起端点）。
        #: ⚠ 同进程里自建第二份 ⇒ 同一数据根抢写租约 ⇒ `writer_lease_held`
        #: （步骤 0 的 A4 实测；本行就是为消灭那个形态而生）。
        self.app_bridge = bridge if bridge is not None else make_v2_bridge(
            PROJ_ROOT, start_services=False)
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
        setup_cjk_fonts()               # 兜底：万一又有东西在构造期改了 rcParams
        # 两模式收拢（2026-09-24）：GUI 模式 = 纯本地软件、无 AI 连接，故**不挂
        # AI 监控侧边栏**（那是 AI 模式的事，且已迁到 dsh 右侧栏面板）。控制权

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
        # 组列表也由同一信号驱动：以前只靠面板自己显式重画，恢复存档直接换 store
        # 时列表就空着（实测）。收拢到信号 ⇒ 任何写路径都不会漏刷新。
        self.groups_store.changed.connect(self.group_panel.refresh_list)

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
        self.group_panel.level_table.setRowCount(0)
        # 行直接取自 `_level_all`——它由 `base_sublevels` 生成，**已施加逐子能级裁剪**。
        # 旧写法在这里自己 `arange(-j, j)` 枚举 mJ ⇒ 无视 ms/Fs：表里显示 5 个能级、
        # 而多体页按裁剪后的 2 个取 g/r（两边自相矛盾，用户无法判断哪个是真的）。
        for r in self._level_all:
            key = (r['n'], r['l'], r['j'])
            pix = QPixmap(12, 12)
            pix.fill(QColor(r.get('color') or '#1f77b4'))
            name = r.get('group_label') or core.auto_label(*key)
            row = self.group_panel.level_table.rowCount()
            self.group_panel.level_table.insertRow(row)
            self.group_panel.level_table.setItem(row, 0, QTableWidgetItem(QIcon(pix), name))
            if r['kind'] == 'F':
                m_txt = 'F=%g, mF=%+g' % (float(r['F']), float(r['mF']))
            else:
                m_txt = '%g' % float(r['mj'])
            it_m = QTableWidgetItem(m_txt)
            it_m.setTextAlignment(Qt.AlignCenter)
            self.group_panel.level_table.setItem(row, 1, it_m)
            self.group_panel.level_table.setItem(
                row, 2, QTableWidgetItem(f"{r['E0'] - ref:.3f}"))

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
        fm.addSeparator()
        self._build_mode_switch(menubar)
        QTimer.singleShot(0, self._apply_fullscreen_layout)

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
            # 先把“住在面板里、经 external 登记”的值推回会话（否则存档拿到的是旧值；
            # push_all 的注释就是“存档/运行前同步用”，但此前**无调用方**）。
            self.binder.push_all()
            self.app_bridge.save(path, reason='GUI 保存状态')   # v2 快照 dict（{键:值}）
        except Exception as e:                          # noqa: BLE001
            QMessageBox.warning(self, '保存失败', str(e))
            return
        QMessageBox.information(self, '保存状态', f'已保存 → {path}')

    def _resync_from_session(self):
        """恢复存档后把“住在会话外”的状态拉回同步（否则界面显示旧态）。

        会话是真值源，但三类状态不在会话里：登记型控件只在构造时 seed 一次、
        能级组住在 `LevelGroupStore`、多体页持有 `_atoms` 面板缓存。实测踩过：
        只 `load()` 不拉同步 → 后端下拉仍显示 auto（会话其实已是 dmrg）、多体页
        原子表为空、且因 `main.groups` 未更新而取不到 g/r。
        """
        self.binder.pull_all()                    # 登记控件 + external 面板
        snap = self.app_bridge.effective() or {}
        n = 0
        if self.groups_store is not None:
            n = self.groups_store.replace(snap.get('groups.list') or [])
        mb = getattr(self, 'manybody', None)
        if mb is not None:
            mb.refresh_from_session()      # 原子表/模式/请求量/betas 回显
            mb.refresh_groups()            # 再把已存原子 g/r 对到当前能级组
        return n

    def _load_state(self):
        path, _ = QFileDialog.getOpenFileName(self, '恢复状态', '', 'JSON (*.json)')
        if not path:
            return
        try:
            with open(path, encoding='utf-8') as f:
                data = json.load(f)
        except Exception as e:                          # noqa: BLE001
            QMessageBox.warning(self, '恢复失败', '文件不可读：%s' % e)
            return
        if isinstance(data, dict) and ('format' in data or 'params' in data):
            # 旧 v1 存档（顶层 {format,mode,params}）：v2 只读扁平 {键:值}快照。
            # 不在这里默默拆包：旧快照可能含已删键，而写档是**原子批量**
            #（坏一键全不写），拆包后部分生效更难查。明确报出来。
            QMessageBox.warning(
                self, '恢复失败',
                '这是 **v1 旧存档**（顶层带 %r）——v2 只读扁平 {键:值} 快照。\n'
                '请在当前版本重新保存一份。' % sorted(data.keys()))
            return
        try:
            n = self.app_bridge.load(path, reason='GUI 恢复状态')   # 经 Gate 原子批量写→订阅刷新
        except Exception as e:                          # noqa: BLE001
            QMessageBox.warning(self, '恢复失败', str(e))
            return
        n_groups = self._resync_from_session()
        self.refresh()
        QMessageBox.information(self, '恢复状态',
                                '已恢复 %d 键（能级组 %d 个）← %s' % (n, n_groups, path))

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
    """本地 GUI：纯本地操控面板，写权在人（不起任何端点）。

    这是**独立进程**那一档（三层架构：纯 GUI 用户不走 mecha，自己找入口）。
    软件进程内开窗是另一条路：`--mcp --with-gui`（窗口复用同一份桥）。
    """
    window = MainWindow()
    window.showMaximized()
    return app.exec_()


def _initial_mode():
    """读**启动者**声明的初始写权（三层架构 §5.1：框架经 `MECHA_INITIAL_MODE` 送进来）。

    ⚠ 空串 = 框架的【显式取消该变量】语义 ⇒ 同样走缺省。
    ⚠ 解析一律用 `.value`：`Mode` 是 **str-Enum** —— `str(Mode.AI)` 给的是 `'Mode.AI'`
      **不是** `'ai'`（三个仓的子代理刚都断在这上面）。
    缺省 = 本仓声明的 AI（"从 mecha 启动永远是 AI 组"）；非法值 ⇒ 回落 AI 并**说出来**。
    """
    raw = (os.environ.get('MECHA_INITIAL_MODE') or '').strip().lower()
    if not raw:
        return Mode.AI
    try:
        return Mode(raw)
    except ValueError:
        print('[mcp] ⚠ MECHA_INITIAL_MODE=%r 非法（只认 %s）⇒ 回落 AI'
              % (raw, '/'.join(m.value for m in Mode)), flush=True)
        return Mode.AI


#: ⭐ 「手动面」（GUI 窗口）在本进程内的**唯一**记载点：``None`` = 还没开。
#: 它同时是**幂等**的依据（已开 ⇒ 不重复开、不报错）。
_MANUAL = {'window': None}

#: 控制端点 → **Qt 主线程** 的搬运队列（见 `_open_on_qt_thread` 的说明）。
_CONTROL_Q = queue.Queue()

#: 主循环是否在跑（决定「能不能把开窗请求搬到 Qt 线程」）。
_MAIN_LOOP = {'running': False}


def open_manual_surface_now(bridge):
    """在当前线程打开「手动面」（GUI 窗口）—— ⚠ **必须在 Qt 主线程调用**（Qt 的硬要求）。

    ⭐ 复用**注入的**桥（``MainWindow(bridge=bridge)``）：这是步骤 0 的 A4 实测的必改点
    —— 同进程里自建第二份桥会抢同一数据根的写租约（``writer_lease_held``）。
    已经开着 ⇒ **幂等**返回 ``'already-open'``。
    """
    if _MANUAL['window'] is not None:
        return 'already-open'
    app = QApplication.instance()
    if app is None:
        raise MechaError('没有 QApplication ⇒ 本进程没有「手动面」能力',
                         kind='no_manual_surface')
    window = MainWindow(bridge=bridge)
    window.showMaximized()
    _MANUAL['window'] = window
    return 'opened'


def _open_on_qt_thread(bridge):
    """把「开窗」这件事**搬回 Qt 主线程**做（控制请求来在 MCP 服务器那条线程上）。

    ⭐ **为什么是搬回主线程、而不是在别的线程里起 Qt**：Qt 要求 GUI 对象与事件循环属于
    **创建 QApplication 的那条线程**（这里就是主线程）—— 在别的线程建窗口是未定义行为。
    而主线程此刻正阻塞在等待循环里 ⇒ 用一条队列把请求递过去，主循环取到后**在主线程**
    建窗口（`_drain_control_queue`）。
    ⚠ 这样**不动任何 native 初始化顺序**：窗口的建法与本进程 `--gui` 那一档**逐字相同**
    （不碰 EL 那条 QThread / `0xc0000409` 的历史雷）。
    """
    if _MANUAL['window'] is not None:
        return 'already-open'
    if threading.current_thread() is threading.main_thread():
        return open_manual_surface_now(bridge)          # 已经在 Qt 线程：直接开
    if not _MAIN_LOOP['running']:
        raise MechaError('当前没有在跑的 Qt 主循环，无法开窗', kind='no_manual_surface')
    box: dict = {}
    done = threading.Event()
    _CONTROL_Q.put((box, done))
    if not done.wait(30.0):
        raise MechaError('开窗请求超时（主循环没响应）', kind='open_surface_timeout')
    if 'err' in box:
        raise box['err']
    return box['result']


def _drain_control_queue(bridge):
    """主循环每拍调一次：执行「必须在 Qt 线程做」的请求（当前只有开窗）。"""
    while True:
        try:
            box, done = _CONTROL_Q.get_nowait()
        except queue.Empty:
            return
        try:
            box['result'] = open_manual_surface_now(bridge)
        except BaseException as e:                      # noqa: BLE001 —— 交给请求方响亮报错
            box['err'] = e
        finally:
            done.set()


def run_mcp(with_gui=False):
    """MCP 服务形态：给框架 supervisor 起；**可选**在同一进程里开 GUI 窗口。

    - 默认 headless：v2 栈照装（Gate/History + MCP 服务器线程 + 监控端点）；
    - `--with-gui`：⭐ **窗口复用这同一份桥**（三层架构 §3 裁决 1 / §6 义务 1）。
      ⚠ **不碰** `--gui` 那一档（独立进程形态，仍是自建桥）。

    v2 栈照装（Gate/History + MCP 服务器线程 + 监控端点），写权落 AI：
    装配出厂 HUMAN 后 `switch_mode(AI, actor='human')`——起它的那一侧（框架
    supervisor，代表人【从 mecha 启动就带 AI】的意图）开闸，不是 AI 自解锁
    （护栏仍在：AI 面改不了 mode）。

    阻塞直到 Ctrl+C / terminate；journal 写租约随进程退出释放（换腿与收尾都
    依赖【先真退出、再起下一个写宿主】这条时序）。
    """
    import threading
    bridge = make_v2_bridge(PROJ_ROOT)
    #: 初始写权：**启动者声明**（§5.1）——缺省 = 本仓声明的 AI。⚠ `Mode` 是 str-Enum，
    #: 所以传 `.value` 给桥（桥的 `switch_mode` 吃字符串）。
    bridge.switch_mode(_initial_mode().value, actor='human')
    print('[mcp] MCP 服务就绪：MCP 端点 + 只读监控端点已起，写权=%s。'
          % bridge.mode, flush=True)
    # ⭐ 控制端点（三层架构 §7.1.1）：人侧专用，**不导出给 AI**；与能力面共用同一份
    # Gate/Authority/History（同一本账）。⚠ 端口号不进任何日志、也不进给 AI 的文本。
    from mecha.software_control import start_control_endpoint
    from mecha_v2.control import CONTROL_SERVER_NAME, CONTROL_PORT_FILE, wire_control
    control_ep, _control_port = start_control_endpoint(
        wire_control(bridge, open_surface=lambda: _open_on_qt_thread(bridge)),
        port_file=os.path.join(PROJ_ROOT, CONTROL_PORT_FILE),
        server_name=CONTROL_SERVER_NAME,
        log=lambda m: print(m, flush=True))          # 收尾见 finally: control_ep.stop()
    print('[mcp] 控制端点已起（端口文件 %s；地址只给框架，不进给 AI 的文本）。'
          % CONTROL_PORT_FILE, flush=True)
    if with_gui:
        open_manual_surface_now(bridge)      # 与按需开窗走**同一处**逻辑（幂等）
        app = QApplication.instance()
        print('[mcp] 已在**同一进程**开窗（复用同一份桥；窗口随本进程生死）。',
              flush=True)
        return app.exec_() if app is not None else 0
    stop = threading.Event()
    _MAIN_LOOP['running'] = True             # ← 主循环跑起来，才可能把开窗搬回主线程
    try:
        while not stop.is_set():
            _drain_control_queue(bridge)     # 执行必须在 Qt 线程做的事（开窗）
            if _MANUAL['window'] is not None:
                app.processEvents()          # 有窗了才泵 Qt（headless 时不碰 Qt 事件）
            stop.wait(0.05)                  # 可中断：Ctrl+C / terminate 照旧
    except KeyboardInterrupt:
        print('[mcp] 收到 Ctrl+C，退出（租约释放）。')
    finally:
        _MAIN_LOOP['running'] = False
        try:
            control_ep.stop()
        except Exception:                    # noqa: BLE001 —— 收尾不掩盖退出
            pass
    return 0


def main():
    """入口：`--mcp` MCP 服务形态 / `--gui` 本地 GUI。

    ⚠ 无参数**不再静默降级**（旧【弹窗选模式】整套已作废）：明确指向
    `python -m mecha --all`（起整站）。⚠ 是 `python -m mecha`，**不是**
    `python -m mecha.launch` —— 后者是模块不是包，会**静默什么都不做**。
    """
    #: ⚠ **如实记**：这一行是**无条件**的 ⇒ **headless（`--mcp` 不带 `--with-gui`）时
    #: Qt 也已经被加载**：「`--mcp` 的 Qt 惰性化」**尚未落地**（别以为它是 lazy 的）。
    #: 反过来这也是个好消息：**按需开窗的代价很低**（Qt 已在，只是晚一点画窗口）——
    #: 但那需要一个「框架→软件」的控制通道，属下一轮，本轮**不做半成品**。
    app = QApplication.instance() or QApplication(sys.argv)
    try:
        app.setStyleSheet(ui_theme.QSS)
    except Exception:                                      # noqa: BLE001
        pass
    if '--gui' in sys.argv:
        return run_gui(app)
    if '--mcp' in sys.argv:
        return run_mcp(with_gui='--with-gui' in sys.argv)
    print('未指定形态：本文件只起【本软件自己的一个形态】（--mcp 或 --gui）。\n'
          '要起整站（含 harness 与进程编排）用：python -m mecha --all\n'
          '  python app_entry.py --mcp                起 MCP 服务端点（headless）\n'
          '  python app_entry.py --mcp --with-gui     同上，并在**同一进程**开窗\n'
          '  python app_entry.py --gui                起本地 GUI（独立进程）')
    return 2


if __name__ == '__main__':
    sys.exit(main())