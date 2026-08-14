# -*- coding: utf-8 -*-
"""Rb-87 能级计算软件 UI（物理核心见 core.py，可单独审查）。

本文件只负责显示与交互：
- 画布直接渲染 core 输出的能级行 / 谱线数据，不做任何二次换算
  （core 返回的字段即画布直接使用的字段，避免"算一遍、再换算一遍"的 bug）
- Qt 控件、画布布局、视野/高亮/参考线等纯显示逻辑
"""

import sys
import json
import numpy as np
import matplotlib
from PyQt5.QtCore import Qt, QTimer, pyqtSignal
from PyQt5.QtGui import QColor, QIcon, QPixmap
from PyQt5.QtWidgets import *

from arc import Rubidium87
from src import core
from src import ui_pair
from src import fields
from src import ui_dynamic
from src import ui_theme
from src.ui_single import EnergyPlotPanel, COLOR_CYCLE

matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False


class Page:
    """一个页面的统一注册记录：左侧画布 tab + 右侧特有面板 panel。
    on_show：切到该页时统一刷新；on_light：光场变化时统一刷新（None=不响应）；
    use_field：本页是否显示共用场面板（能级组区始终共用）；
    field_lights：是否显示场面板里的光场表格/关注跃迁/初态区
    （B/E 行恒显，三页共用同一实例，天然互通）。
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
        # 全局主题（对所有窗口/对话框/表格生效，含 TTL 编辑器）
        app = QApplication.instance()
        if app is not None:
            app.setStyleSheet(ui_theme.QSS)
        self.statusBar().showMessage('就绪', 3000)

        self.atom = Rubidium87()
        self.cache = core.ArcCache(self.atom)
        self.field_B = 0.0
        self.groups = []
        self._color_counter = 0   # 已分配颜色的计数（删除/换视野不改变颜色）
        self.address_cfg = {
            'enabled': False,
            'intensity': 0.0,        # mW/cm^2
            'frequency_hz': 0.0,     # Hz
            'polarization': 'π',
        }
        self._sorted_groups = []   # 按能量升序排列的能级组（y 轴节点顺序）
        self._y_view_user_set = False   # 用户是否手动设置过上下限节点
        self._last_lines_off = None    # 谱峰表只在线数据变化时重填
        self._last_lines_on = None
        self._selected_spec_table = 'off'   # 最近选中的能谱表（off/on）
        self._selected_spec_row = -1        # 最近选中的谱线行（-1=无）
        self._build_ui()

    def log(self, msg, timeout=8000):
        """统一信息出口：状态栏显示（GUI 里 stdout 不可见，算法提示/警告走这里）。"""
        if hasattr(self, 'statusBar'):
            self.statusBar().showMessage(str(msg), timeout)

    def closeEvent(self, ev):
        """主窗口关闭 → 联动关闭时序/波形编辑器，避免残留独立窗口。"""
        dyn = getattr(self, 'dynamic', None)
        if dyn is not None and getattr(dyn, 'editor', None) is not None:
            try:
                dyn.editor.close()
            except Exception:
                pass
        super().closeEvent(ev)

    def _build_ui(self):
        main_widget = QWidget(self)
        self.setCentralWidget(main_widget)
        root = QVBoxLayout(main_widget)
        top = QHBoxLayout()
        root.addLayout(top, 1)

        # 左侧：QTab 只装画布；右侧共用操作面板。页面以 Page 记录统一注册
        # （见 self.pages），tab/面板/切换全部由循环驱动，加新页只需追加一条记录。
        self.canvas_tabs = QTabWidget()
        # 单体页：QStackedWidget（页0=单体响应谱，页1=旧版能级图+off/on 两能谱）
        self.single_resp = ui_pair.ResponsePlot()
        self.canvas = EnergyPlotPanel()
        self.single_stack = QStackedWidget()
        self.single_stack.addWidget(self.single_resp)   # 0：响应谱
        self.single_stack.addWidget(self.canvas)        # 1：能级图/两能谱
        self.pair_canvas = ui_pair.ResponsePlot()
        self.dynamic = ui_dynamic.DynamicPanel(self, self)

        control_widget = QWidget()
        # 字体随全局主题；宽度强制跟随视图（不出现横向滚动条）
        control_widget.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        control_layout = QVBoxLayout(control_widget)
        control_layout.setSpacing(10)

        # ---------------- 共用能级组区（所有页面共用）：定义能级 → 管理能级 → 数值表 -------------
        self.shared_widget = QWidget()
        shared_lay = QVBoxLayout(self.shared_widget)
        shared_lay.setContentsMargins(0, 0, 0, 0)
        shared_lay.setSpacing(10)
        control_layout.addWidget(self.shared_widget)
        # 定义能级：n/l/j/标签 挤一行
        row_add = QHBoxLayout()
        row_add.addWidget(QLabel('n'))
        self.n_spin = QSpinBox()
        self.n_spin.setRange(1, 200)
        self.n_spin.setValue(60)
        row_add.addWidget(self.n_spin)
        row_add.addWidget(QLabel('l'))
        self.l_spin = QSpinBox()
        self.l_spin.setRange(0, 199)
        self.l_spin.setValue(0)
        row_add.addWidget(self.l_spin)
        row_add.addWidget(QLabel('j'))
        self.j_spin = QDoubleSpinBox()
        self.j_spin.setRange(0.0, 10.0)
        self.j_spin.setSingleStep(0.5)
        self.j_spin.setDecimals(1)
        self.j_spin.setValue(0.5)
        row_add.addWidget(self.j_spin)
        self.n_spin.editingFinished.connect(self.on_nlj_changed)
        self.l_spin.editingFinished.connect(self.on_nlj_changed)
        self.j_spin.editingFinished.connect(self.on_nlj_changed)
        self.label_preview = QLabel(core.auto_label(60, 0, 0.5))
        row_add.addWidget(self.label_preview, 1)
        self.add_button = QPushButton('添加')
        self.add_button.clicked.connect(self.add_group)
        row_add.addWidget(self.add_button)
        shared_lay.addLayout(row_add)
        # 能级组列表（"添加能级组"底下，右键删除）
        self.group_list = QListWidget()
        self.group_list.setContextMenuPolicy(Qt.CustomContextMenu)
        self.group_list.customContextMenuRequested.connect(self.show_context_menu)
        self.group_list.setMinimumHeight(120)      # 可随窗口拉伸
        shared_lay.addWidget(self.group_list)

        # 能级管理表（共用；全部子能级直显，E 为裸能级能量（不含能移））
        self._level_all = []         # 最近一次全部子能级行（组内最低值用）
        self._level_by_key = {}      # (n,l,j,mj) -> mJ 行（数值显示用）
        self.level_table = QTableWidget(0, 3)
        self.level_table.setHorizontalHeaderLabels(
            ['能级组', 'mJ', 'E (MHz)'])
        self.level_table.horizontalHeader().setSectionResizeMode(
            0, QHeaderView.Stretch)
        self.level_table.setMinimumHeight(150)     # 可随窗口拉伸
        self.level_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.level_table.setToolTip(
            '全部 mJ 磁子能级（每行一个）；动态演化基组 = 本表全部能级，无需选择')
        shared_lay.addWidget(self.level_table)

        # 定义场：静磁场 + 静电场 + 添加光（按页 use_field 显隐）
        self.field = fields.FieldPanel(self)
        self.field.changed.connect(self.on_light_changed)
        control_layout.addWidget(self.field)

        # ---- 单体特有控件（本页 panel）----
        self.single_widget = QWidget()
        single_lay = QVBoxLayout(self.single_widget)
        single_lay.setSpacing(6)

        # 能级图显示模式 + y 上下限 挤一排（能级图在最左）
        row_y = QHBoxLayout()
        row_y.addWidget(QLabel('能级图'))
        self.level_mode_box = QComboBox()
        self.level_mode_box.addItems(['单一', 'off/on 双图'])
        self.level_mode_box.setToolTip(
            '能级图显示模式：\n'
            '  单一：一张能级图（Zeeman + AC 位移与箭头），off/on 跃迁连线叠加\n'
            '  off/on 双图：非寻址光（裸能级）与寻址光（移后）各一张并排，\n'
            '    分别标注对应频谱（off 谱 / on 谱）选中的跃迁连线')
        self.level_mode_box.currentIndexChanged.connect(self.on_level_mode_changed)
        row_y.addWidget(self.level_mode_box)
        row_y.addWidget(QLabel('y 下限'))
        self.y_lower_combo = QComboBox()
        self.y_lower_combo.currentIndexChanged.connect(self.on_y_node_changed)
        row_y.addWidget(self.y_lower_combo, 1)
        row_y.addWidget(QLabel('上限'))
        self.y_upper_combo = QComboBox()
        self.y_upper_combo.currentIndexChanged.connect(self.on_y_node_changed)
        row_y.addWidget(self.y_upper_combo, 1)
        single_lay.addLayout(row_y)

        # 参考频率 + 频谱横轴上下限 挤一排（参考频率在最左；空=自动/不画）
        row_spec = QHBoxLayout()
        row_spec.addWidget(QLabel('参考频率'))
        self.spec_ref_edit = QLineEdit('17.2368')
        self.spec_ref_edit.setPlaceholderText('空=不画')
        self.spec_ref_edit.setMaximumWidth(140)
        self.spec_ref_edit.returnPressed.connect(self.on_spectrum_ref_changed)
        row_spec.addWidget(self.spec_ref_edit)
        row_spec.addWidget(QLabel('谱x 下限'))
        self.spec_freq_lo_edit = QLineEdit()
        self.spec_freq_lo_edit.setPlaceholderText('自动')
        self.spec_freq_lo_edit.returnPressed.connect(self.on_spectrum_range_changed)
        row_spec.addWidget(self.spec_freq_lo_edit, 1)
        row_spec.addWidget(QLabel('上限'))
        self.spec_freq_hi_edit = QLineEdit()
        self.spec_freq_hi_edit.setPlaceholderText('自动')
        self.spec_freq_hi_edit.returnPressed.connect(self.on_spectrum_range_changed)
        row_spec.addWidget(self.spec_freq_hi_edit, 1)
        single_lay.addLayout(row_spec)
        self.on_spectrum_ref_changed()   # 初始化默认参考线

        # 双谱选中峰频率：off 峰频率 | on 峰频率 | 频率差（一行，不标在图上）
        self.peak_pair_label = QLabel('')
        self.peak_pair_label.setStyleSheet(
            'font-family: Consolas; font-size: 9pt;')
        self.peak_pair_label.setWordWrap(False)
        single_lay.addWidget(self.peak_pair_label)

        # ---- 单体旧版数据表：off/on 两个谱表（还原；能级数值表已上移到能级组列表下）----
        self.spec_off_table = self._make_spec_table('off')
        single_lay.addWidget(self.spec_off_table)
        self.spec_on_table = self._make_spec_table('on')
        single_lay.addWidget(self.spec_on_table)

        # ---- 两体特有控件（单体模式隐藏）----
        self.pair_widget = QWidget()
        pair_lay = QVBoxLayout(self.pair_widget)
        pair_lay.setSpacing(6)
        row = QHBoxLayout()
        row.addWidget(QLabel('R (μm)'))
        self.pair_r = QDoubleSpinBox()
        self.pair_r.setRange(0.1, 500.0); self.pair_r.setDecimals(2)
        self.pair_r.setValue(6.0)
        row.addWidget(self.pair_r, 1)
        row.addWidget(QLabel('θ°'))
        self.pair_theta = QDoubleSpinBox()
        self.pair_theta.setRange(0.0, 180.0); self.pair_theta.setDecimals(1)
        row.addWidget(self.pair_theta, 1)
        row.addWidget(QLabel('φ°'))
        self.pair_phi = QDoubleSpinBox()
        self.pair_phi.setRange(0.0, 360.0); self.pair_phi.setDecimals(1)
        row.addWidget(self.pair_phi, 1)
        pair_lay.addLayout(row)
        row = QHBoxLayout()
        row.addWidget(QLabel('总 m'))
        self.pair_mtotal = QLineEdit('1')
        self.pair_mtotal.setToolTip('总磁量子数 m1+m2（空=全 m，基组可能很大）')
        row.addWidget(self.pair_mtotal, 1)
        row.addWidget(QLabel('窗口 MHz'))
        self.pair_window = QLineEdit('')
        self.pair_window.setToolTip('能量窗口截断（空=不限；用窗口时需同时填参考 MHz）')
        row.addWidget(self.pair_window, 1)
        row.addWidget(QLabel('参考 MHz'))
        self.pair_ref = QLineEdit('')
        self.pair_ref.setPlaceholderText('空=0')
        row.addWidget(self.pair_ref, 1)
        pair_lay.addLayout(row)
        row = QHBoxLayout()
        row.addWidget(QLabel('谱窗 ±MHz'))
        self.spectrum_window = QDoubleSpinBox()
        self.spectrum_window.setRange(0.1, 1e5)
        self.spectrum_window.setDecimals(1)
        self.spectrum_window.setValue(200.0)
        row.addWidget(self.spectrum_window, 1)
        self.render_resp_btn = QPushButton('渲染响应谱')
        self.render_resp_btn.clicked.connect(self.render_response)
        row.addWidget(self.render_resp_btn, 1)
        pair_lay.addLayout(row)
        self.resp_status = QLabel('')
        self.resp_status.setStyleSheet('color: #0b5394;')
        pair_lay.addWidget(self.resp_status)
        # 响应谱峰表（像旧版右侧的谱表，放两体显示区）
        self.peak_table = QTableWidget(0, 4)
        self.peak_table.setHorizontalHeaderLabels(
            ['光', '失谐 (MHz)', '频率 (GHz)', '强度'])
        self.peak_table.horizontalHeader().setStretchLastSection(True)
        self.peak_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.peak_table.setMinimumHeight(140)     # 可随窗口拉伸
        pair_lay.addWidget(self.peak_table)

        # ---- 动态分析特有控件（与单体/两体同级；共用能级组，特有 m/初态/光场/运行）----
        self.dynamic_widget = self.dynamic.dynamic_widget

        # ---- 页面统一注册：一个页面 = 左侧画布 tab + 右侧特有面板 panel。
        # on_show：切到该页时统一刷新；on_light：光场变化时统一刷新（None=不响应）；
        # use_field：本页是否显示共用场面板。加新页只需追加一条，切页/布局/tab 全自动。----
        self.pages = [
            Page('单体', self.single_stack, self.single_widget,
                 on_show=self._show_single, on_light=self.refresh_single),
            Page('两体', self.pair_canvas, self.pair_widget,
                 on_show=self._show_pair, on_light=self.render_response),
            Page('动态分析', self.dynamic, self.dynamic_widget,
                 on_show=self.dynamic.refresh_groups,
                 on_light=self.dynamic.refresh_groups,   # B/E 变化 → 重建基组（含 Zeeman/DC）
                 use_field=True, field_lights=False),    # 只显示 B/E 行；光场在编辑器里配
        ]
        # 左侧 tab 由注册表驱动（顺序 = pages 顺序）
        for p in self.pages:
            self.canvas_tabs.addTab(p.tab, p.name)
        self.canvas_tabs.currentChanged.connect(self.on_canvas_tab_changed)
        top.addWidget(self.canvas_tabs, 3)
        # 右侧布局：共用能级组区 → 场面板 → 页面栈（各页特有区，QStackedWidget 分页）
        self.page_stack = QStackedWidget()
        for p in self.pages:
            self.page_stack.addWidget(p.panel)
        control_layout.addWidget(self.page_stack)

        # 点击频谱峰 → 对应峰表滚动并选中该行（定位跃迁对）
        self.canvas.peakPicked.connect(self._on_peak_picked)
        # 选中峰频率变化 → 右侧面板显示 off/on 频率与频率差
        self.canvas.peakFreqsChanged.connect(self._update_peak_pair_label)

        # 面板放入滚动区（仅纵向滚动；宽度跟随视图，禁用横向滚动条）
        self.panel_scroll = QScrollArea()
        self.panel_scroll.setWidgetResizable(True)
        self.panel_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.panel_scroll.setWidget(control_widget)
        self.control_widget = control_widget     # 动态分析 tab 时隐藏右侧共用面板
        top.addWidget(self.panel_scroll, 2)

        self._rebuild_y_combos()
        # 初始磁场由 FieldPanel 提供
        self.field_B = self.field.b_gauss()
        # 初始模式：画布默认单体 tab（统一切换逻辑：右侧 = 共用区 + 单体特有区）
        self.on_canvas_tab_changed(0)
        # 菜单栏：文件 → 保存/恢复状态（名义上全部参数，先覆盖动态页）
        menubar = self.menuBar()
        fm = menubar.addMenu('文件')
        fm.addAction('保存状态', self._save_state)
        fm.addAction('恢复状态', self._load_state)
        # 启动后按全屏尺寸适配各组件（面板宽度、表高、画布 dpi、字号）
        QTimer.singleShot(0, self._apply_fullscreen_layout)

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

    # ---------------- 模式切换 / 添加光 / 响应谱 -------------
    def on_canvas_tab_changed(self, idx):
        """统一页面行为：右侧 = 共用能级组区 + 场面板（按页声明）+ 页面栈当前页。
        切换零特例：页面栈 setCurrentIndex、按 use_field 显隐场面板、调 on_show 刷新。
        加新页只需在 self.pages 注册一条，本方法无需改动。"""
        page = self.pages[idx]
        self.page_stack.setCurrentIndex(idx)
        self.field.setVisible(page.use_field)     # 光场定义：按页声明是否用场面板
        self.field.set_lights_visible(page.field_lights)   # B/E 恒显；光场区按页显隐
        self.panel_scroll.setVisible(True)       # 右侧单一面板
        if page.on_show is not None:
            page.on_show()                        # 显示时刷新（统一钩子）

    def _show_single(self):
        self.field.rebuild_focus(show_exchange=True)
        self.refresh_single()

    def _show_pair(self):
        self.field.rebuild_focus(show_exchange=False)
        self.render_response()

    def refresh_single(self):
        """单体页：关注某份光 → 单体响应谱；关注 exchange → 旧版能级+两能谱。"""
        if self.field.focus_is_exchange():
            self.single_stack.setCurrentIndex(1)   # 旧版能级图 + off/on 频谱
            self._sync_address_from_lights()
            self.refresh_plot()
        else:
            self.single_stack.setCurrentIndex(0)   # 单体响应谱
            self.render_single_response()

    def render_single_response(self):
        """单体响应谱（关注某份光，统一引擎 n_body=1）。"""
        if not self.groups:
            self.resp_status.setText('请先添加能级组')
            return
        psi = self.field.psi_ket()
        if psi is None:
            self.resp_status.setText('初态格式：n l j m')
            return
        cfgs = self.field.light_cfgs()
        if not cfgs:
            self.resp_status.setText('请先添加光')
            return
        try:
            curves, peaks, system = ui_pair.response_curves(
                self.cache, self.groups, cfgs, psi, 1,
                B_gauss=self.field.b_gauss(),
                window_mhz=float(self.spectrum_window.value()),
                drive_idx=self.field.focus_drive_idx(),
                dc_stark_vpcm=self.field.e_vpcm())
        except Exception as e:
            self.resp_status.setText(f'计算失败: {e}')
            return
        self.single_resp.render(
            curves, f'单体响应谱  B={self.field.b_gauss():g}G '
            f'E={self.field.e_vpcm():g}V/cm  '
            f'初始 {psi[0]} {psi[1]} {psi[2]:g} m={psi[3]:+.1f}')
        self._fill_peak_table(peaks)
        self.resp_status.setText(f'{len(peaks)} 个峰')

    def add_light(self):
        self.field.add_light()

    def on_light_changed(self, *_):
        """场/光变化：更新 field_B + 单体 address 同步 + 统一调当前页刷新钩子。"""
        self.field_B = self.field.b_gauss()
        self._sync_address_from_lights()
        if not hasattr(self, 'pages'):
            return                       # 构造期间（pages 尚未注册）
        page = self.pages[self.canvas_tabs.currentIndex()]
        if page.on_light is not None:
            page.on_light()

    def _sync_address_from_lights(self):
        """单体模式：第一份光同步到 address_cfg（旧版能级图/频谱计算）。"""
        c = self.field.first_cfg()
        if c:
            self.address_cfg.update(
                frequency_hz=c['frequency_hz'],
                intensity=c['intensity'],
                polarization=c['polarization'])
            if c.get('elliptical'):
                self.address_cfg['elliptical'] = dict(c['elliptical'])
            else:
                self.address_cfg.pop('elliptical', None)
            self.address_cfg['enabled'] = True
        else:
            self.address_cfg['enabled'] = False

    def _parse_pair_psi(self):
        return self.field.psi_ket()

    # scan_light_intensity 已删除：旧 AddLightRow（智能光行）接口被光场表格取代，
    # 失谐/扫描光强暂缓（以后在表格化方案下重做）。

    def render_response(self):
        """两体响应谱：统一引擎线性谱 → 画布 + 底部峰表。"""
        if not self.groups:
            self.resp_status.setText('请先添加能级组')
            return
        psi = self._parse_pair_psi()
        if psi is None:
            self.resp_status.setText('初态格式：n l j m')
            return
        cfgs = self.field.light_cfgs()
        if not cfgs:
            self.resp_status.setText('请先添加光')
            return
        try:
            m_total = (float(self.pair_mtotal.text())
                       if self.pair_mtotal.text().strip() else None)
            window = (float(self.pair_window.text())
                      if self.pair_window.text().strip() else None)
            ref = (float(self.pair_ref.text())
                   if self.pair_ref.text().strip() else None)
        except ValueError:
            self.resp_status.setText('总 m/窗口/参考 需为数字或空')
            return
        try:
            curves, peaks, system = ui_pair.response_curves(
                self.cache, self.groups, cfgs, psi, 2,
                R_um=float(self.pair_r.value()),
                B_gauss=self.field.b_gauss(),
                theta_deg=float(self.pair_theta.value()),
                phi_deg=float(self.pair_phi.value()),
                m_total=m_total, energy_window_mhz=window,
                ref_energy_mhz=ref,
                window_mhz=float(self.spectrum_window.value()),
                drive_idx=self.field.focus_drive_idx(),
                dc_stark_vpcm=self.field.e_vpcm())
        except Exception as e:
            self.resp_status.setText(f'计算失败: {e}')
            return
        title = (f'两体响应谱  R={self.pair_r.value():g}μm '
                 f'θ={self.pair_theta.value():g}° φ={self.pair_phi.value():g}°  '
                 f'B={self.field.b_gauss():g}G E={self.field.e_vpcm():g}V/cm  '
                 f'初始 {psi[0]} {psi[1]} {psi[2]:g} m={psi[3]:+.1f}')
        self.pair_canvas.render(curves, title)
        self._fill_peak_table(peaks)
        self.resp_status.setText(f'{len(peaks)} 个峰')

    def _fill_peak_table(self, peaks):
        """响应谱峰表填充。"""
        self.peak_table.setRowCount(len(peaks))
        for r, (lab, x, f, s) in enumerate(peaks):
            self.peak_table.setItem(r, 0, QTableWidgetItem(lab))
            self.peak_table.setItem(r, 1, QTableWidgetItem(f'{x:.3f}'))
            self.peak_table.setItem(r, 2, QTableWidgetItem(f'{f / 1e9:.6f}'))
            self.peak_table.setItem(r, 3, QTableWidgetItem(f'{s:.3e}'))
        self.resp_status.setText(f'{len(peaks)} 个峰')

    # ---------------- 事件处理 ----------------
    def on_nlj_changed(self, *_):
        # 实时预览自动生成的标签
        n = int(self.n_spin.value())
        l = int(self.l_spin.value())
        j = float(self.j_spin.value())
        ok, _ = core.validate_state(n, l, j)
        if ok:
            self.label_preview.setText(core.auto_label(n, l, j))

    def on_spectrum_range_changed(self):
        """频谱横轴上下限 line 控件：空=自动，填入数字即手动范围。"""
        lo = self._parse_range_edit(self.spec_freq_lo_edit)
        hi = self._parse_range_edit(self.spec_freq_hi_edit)
        self.canvas.set_spectrum_range(lo, hi)

    @staticmethod
    def _parse_range_edit(edit):
        text = edit.text().strip()
        if not text:
            return None
        try:
            return float(text)
        except ValueError:
            return None

    def on_spectrum_ref_changed(self):
        """频谱参考频率：填入数字即画竖线，清空则不画。"""
        text = self.spec_ref_edit.text().strip()
        try:
            f = float(text) if text else None
        except ValueError:
            f = None
        self.canvas.set_spectrum_ref(f)

    def _make_spec_table(self, which):
        """构造谱峰数据表（off/on 共用样式与点击行为）。"""
        table = QTableWidget(0, 4)
        table.setHorizontalHeaderLabels(['Freq (GHz)', 'From', 'To', 'Pol'])
        table.setSelectionBehavior(QAbstractItemView.SelectRows)
        table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        table.verticalHeader().setVisible(False)
        table.setMinimumWidth(0)
        table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        # Pol 列固定窄，给 Freq/From/To 留地方
        table.horizontalHeader().setSectionResizeMode(3, QHeaderView.Fixed)
        table.setColumnWidth(3, 55)
        table.setMinimumHeight(170)      # 可随窗口拉伸
        table.verticalHeader().setDefaultSectionSize(18)
        table.cellClicked.connect(
            lambda r, c: self._on_peak_table_clicked(which, r, c))
        return table

    @staticmethod
    def _populate_peak_table(table, lines):
        """把谱线数据填入峰表：频率、源子能级、目标子能级、偏振。

        表头用 Stretch 模式随面板宽度伸缩；被截断的单元格内容可通过悬浮提示查看。
        """
        table.setRowCount(len(lines))
        for i, ln in enumerate(lines):
            nu, A, la, lb, qa, qb, col, pol = ln
            table.setItem(i, 0, QTableWidgetItem(f'{nu / 1e9:.4f}'))
            it1 = QTableWidgetItem(f'{la} {qa}')
            it1.setToolTip(f'{la} {qa}')
            table.setItem(i, 1, it1)
            it2 = QTableWidgetItem(f'{lb} {qb}')
            it2.setToolTip(f'{lb} {qb}')
            table.setItem(i, 2, it2)
            table.setItem(i, 3, QTableWidgetItem(pol))

    def _on_peak_picked(self, which, idx):
        """点击频谱峰后：对应峰表滚动并选中该行（并记录为"获取自能谱"来源）。"""
        table = self.spec_off_table if which == 'off' else self.spec_on_table
        table.clearSelection()
        if idx < 0 or idx >= table.rowCount():
            return
        self._selected_spec_table = which
        self._selected_spec_row = idx
        table.selectRow(idx)
        table.scrollToItem(table.item(idx, 0))

    def _update_peak_pair_label(self, off_f, on_f):
        """右侧面板一行显示：off 峰频率 | on 峰频率 | 频率差。"""
        if off_f is None and on_f is None:
            self.peak_pair_label.setText('')
            return
        parts = []
        if off_f is not None:
            parts.append(f'off {off_f:.4f} GHz')
        if on_f is not None:
            parts.append(f'on {on_f:.4f} GHz')
        if off_f is not None and on_f is not None:
            parts.append(f'Δ {(on_f - off_f) * 1000:+.1f} MHz')
        self.peak_pair_label.setText('   |   '.join(parts))

    def on_level_mode_changed(self, *_):
        """能级图显示模式切换：单一 / off-on 双图。"""
        mode = 'dual' if self.level_mode_box.currentIndex() == 1 else 'single'
        self.canvas.set_level_mode(mode)

    def _on_peak_table_clicked(self, which, row, col):
        """能谱表点击：记录选中谱线并高亮图中对应峰。"""
        self._selected_spec_table = which
        self._selected_spec_row = row
        self.canvas.set_spectrum_highlight(which, row)

    # ---------------- y 轴显示范围（按能级组节点） ----------------
    def _rebuild_y_combos(self):
        """按能量升序重建上下限组下拉框；添加/删除能级组时调用。"""
        self._sorted_groups = sorted(
            self.groups, key=lambda g: self.cache.E_mhz(g['n'], g['l'], g['j']))
        lo = self.y_lower_combo.currentIndex()
        hi = self.y_upper_combo.currentIndex()
        self.y_lower_combo.blockSignals(True)
        self.y_upper_combo.blockSignals(True)
        self.y_lower_combo.clear()
        self.y_upper_combo.clear()
        for g in self._sorted_groups:
            self.y_lower_combo.addItem(g['label'])
            self.y_upper_combo.addItem(g['label'])
        n = len(self._sorted_groups)
        if n > 0:
            if not self._y_view_user_set:
                # 用户未手动设置：默认显示全部（下限=最低，上限=最高）
                lo, hi = 0, n - 1
            else:
                lo = max(0, min(lo, n - 1))
                hi = max(0, min(hi, n - 1))
                if hi < lo:
                    hi = lo
            self.y_lower_combo.setCurrentIndex(lo)
            self.y_upper_combo.setCurrentIndex(hi)
        self.y_lower_combo.blockSignals(False)
        self.y_upper_combo.blockSignals(False)
        self._sync_y_combos()

    def on_y_node_changed(self, *_):
        self._y_view_user_set = True
        self._sync_y_combos()
        self.refresh_plot()

    def _sync_y_combos(self):
        """保证下限索引 <= 上限索引。"""
        lo = self.y_lower_combo.currentIndex()
        hi = self.y_upper_combo.currentIndex()
        if lo >= 0 and hi >= 0 and hi < lo:
            self.y_upper_combo.blockSignals(True)
            self.y_upper_combo.setCurrentIndex(lo)
            self.y_upper_combo.blockSignals(False)

    def _visible_groups(self):
        """返回 y 轴上下限节点之间的能级组（含节点本身）。"""
        if not self._sorted_groups:
            return []
        lo = self.y_lower_combo.currentIndex()
        hi = self.y_upper_combo.currentIndex()
        if lo < 0 or hi < 0:
            return list(self._sorted_groups)
        return self._sorted_groups[max(0, lo):hi + 1]

    def _assign_color(self):
        """按添加顺序分配固定颜色（每个能级组与一个颜色绑定）。"""
        color = COLOR_CYCLE[self._color_counter % len(COLOR_CYCLE)]
        self._color_counter += 1
        return color

    def add_group(self):
        n = int(self.n_spin.value())
        l = int(self.l_spin.value())
        j = float(self.j_spin.value())
        ok, msg = core.validate_state(n, l, j)
        if not ok:
            QMessageBox.warning(self, '参数不合法', msg)
            return
        group = {
            'n': n,
            'l': l,
            'j': j,
            'label': core.auto_label(n, l, j),
            'color': self._assign_color(),
        }
        self.groups.append(group)
        # 列表项带颜色块，与图中颜色一一对应
        pix = QPixmap(12, 12)
        pix.fill(QColor(group['color']))
        item = QListWidgetItem(QIcon(pix), f"{group['label']}  (n={n}, l={l}, j={j})")
        self.group_list.addItem(item)
        self._rebuild_y_combos()
        self.refresh_plot()
        self.groups_changed.emit()   # 依赖方（动态页初态/驱动）订阅刷新

    def show_context_menu(self, pos):
        item = self.group_list.itemAt(pos)
        if item is None:
            return
        menu = QMenu(self)
        delete_action = menu.addAction('删除此能级组')
        action = menu.exec_(self.group_list.mapToGlobal(pos))
        if action == delete_action:
            row = self.group_list.row(item)
            if row >= 0:
                self.group_list.takeItem(row)
                del self.groups[row]
                self._rebuild_y_combos()
                self.refresh_plot()
                self.groups_changed.emit()

    def _apply_fullscreen_layout(self):
        """全屏/最大化后按屏幕尺寸适配：面板宽度、表最小高、画布 dpi。"""
        screen = QApplication.primaryScreen()
        if screen is None:
            return
        geo = screen.availableGeometry()
        w, h = geo.width(), geo.height()
        # 右侧面板宽度：屏宽 22% ~ 34%（超宽屏不占满）
        self.panel_scroll.setMinimumWidth(int(w * 0.22))
        self.panel_scroll.setMaximumWidth(int(w * 0.34))
        # 表格给"最小高度"（可随窗口拉伸），全屏下更大
        self.group_list.setMinimumHeight(max(90, int(h * 0.06)))
        self.level_table.setMinimumHeight(max(120, int(h * 0.08)))
        self.peak_table.setMinimumHeight(max(110, int(h * 0.08)))
        self.spec_off_table.setMinimumHeight(max(80, int(h * 0.06)))
        self.spec_on_table.setMinimumHeight(max(80, int(h * 0.06)))
        # 画布 dpi：按屏高提升分辨率，文字保持清晰
        dpi = max(100, min(int(h / 9), 150))
        for fig in (self.canvas.level_fig, self.canvas.level_off_fig,
                    self.canvas.level_on_fig, self.canvas.spec_mirror_fig,
                    self.pair_canvas.fig):
            fig.set_dpi(dpi)
        self.refresh_plot()

    def refresh_plot(self):
        """旧版能级图 + off/on 频谱 + 谱表。计算/绘制异常统一转状态栏提示。"""
        try:
            visible = self._visible_groups()
            rows = self.canvas.draw_levels(
                self.cache, self.field_B, self.groups,
                self.address_cfg, True, visible)

            # 谱峰表：仅当谱线数据更新时重填
            if self.canvas._lines_off is not self._last_lines_off:
                self._last_lines_off = self.canvas._lines_off
                self._populate_peak_table(self.spec_off_table, self.canvas._lines_off)
            if self.canvas._lines_on is not self._last_lines_on:
                self._last_lines_on = self.canvas._lines_on
                self._populate_peak_table(self.spec_on_table, self.canvas._lines_on)

            self._fill_level_table()     # 共用能级管理表（全部 mJ 能级直显）
        except Exception as e:
            self.log(f'能级图刷新失败: {e}')
            QMessageBox.warning(self, '刷新失败', str(e))

    # ---------------- 共用能级管理表（全部 mJ 能级直显） ----------------
    def _fill_level_table(self):
        """填充共用能级表：每行一个 mJ 磁子能级（全部能级，动态基组 = 全部）。
        超精细组无 mJ 行 → 数值显示组内最低子能级（mJ 为近似基组）。"""
        self._level_all = core.build_level_rows(
            self.cache, self.field_B, self.groups, self.address_cfg, self.groups)
        self._level_by_key = {
            (r['n'], r['l'], r['j'], r['mj']): r
            for r in self._level_all if r['kind'] == 'mJ'}
        ref = min((r['E0'] for r in self._level_all), default=0.0)   # 裸能（不含能移）
        lowest = {}
        for r in self._level_all:
            key = (r['n'], r['l'], r['j'])
            if key not in lowest or r['E0'] < lowest[key]['E0']:
                lowest[key] = r
        self.level_table.setRowCount(0)
        for g in self.groups:
            key = (g['n'], g['l'], g['j'])
            pix = QPixmap(12, 12)
            pix.fill(QColor(g['color']))
            lo = lowest.get(key)
            for m in np.arange(-g['j'], g['j'] + 1e-9, 1.0):
                r = self.level_table.rowCount()
                self.level_table.insertRow(r)
                self.level_table.setItem(
                    r, 0, QTableWidgetItem(QIcon(pix), g['label']))
                it_m = QTableWidgetItem(f'{m:g}')
                it_m.setTextAlignment(Qt.AlignCenter)
                self.level_table.setItem(r, 1, it_m)
                rm = self._level_by_key.get((key[0], key[1], key[2], float(m)))
                if rm is None:
                    rm = lo
                if rm is not None:
                    self.level_table.setItem(
                        r, 2, QTableWidgetItem(f"{rm['E0'] - ref:.3f}"))
                else:
                    self.level_table.setItem(r, 2, QTableWidgetItem(''))


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    # 最大化而非真全屏：保留右上角最小化/最大化/关闭三个窗口按钮
    window.showMaximized()
    sys.exit(app.exec_())
