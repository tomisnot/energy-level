# -*- coding: utf-8 -*-
"""Rb-87 能级计算软件 UI（物理核心见 core.py，可单独审查）。

本文件只负责显示与交互：
- 画布直接渲染 core 输出的能级行 / 谱线数据，不做任何二次换算
  （core 返回的字段即画布直接使用的字段，避免"算一遍、再换算一遍"的 bug）
- Qt 控件、画布布局、视野/高亮/参考线等纯显示逻辑
"""

import sys
import numpy as np
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QColor, QIcon, QPixmap
from PyQt5.QtWidgets import *

from arc import Rubidium87
from src import core
from src import ui_pair
from src import ui_fields
from src.ui_single import EnergyPlotPanel, COLOR_CYCLE





class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Rb-87 能级计算')
        self.resize(1400, 800)

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
        self._scan_pair = None   # 光强扫描目标能级对 (la, qa, lb, qb)；跟随能级对而非下标
        self._target_line = None   # 目标跃迁对（能谱表确定）: (which,row,nu_hz,la,qa,lb,qb,pol)；None=直接频率模式
        self._programmatic_freq = False   # 程序设置 freq_box 的临时标志（区分用户手动改频率）

        main_widget = QWidget(self)
        self.setCentralWidget(main_widget)
        root = QVBoxLayout(main_widget)
        top = QHBoxLayout()
        root.addLayout(top, 1)

        # 左侧：QTab 只装画布（单体能级+频谱 / 两体响应谱）；右侧共用操作面板
        self.canvas_tabs = QTabWidget()
        # 单体页：QStackedWidget（页0=单体响应谱，页1=旧版能级图+off/on 两能谱）
        self.single_resp = ui_pair.ResponsePlot()
        self.canvas = EnergyPlotPanel()
        self.single_stack = QStackedWidget()
        self.single_stack.addWidget(self.single_resp)   # 0：响应谱
        self.single_stack.addWidget(self.canvas)        # 1：能级图/两能谱
        self.canvas_tabs.addTab(self.single_stack, '单体')
        self.pair_canvas = ui_pair.ResponsePlot()
        self.canvas_tabs.addTab(self.pair_canvas, '两体')
        self.canvas_tabs.currentChanged.connect(self.on_canvas_tab_changed)
        top.addWidget(self.canvas_tabs, 3)

        control_widget = QWidget()
        # 圆润字体 + 稍大字号；宽度强制跟随视图（不出现横向滚动条）
        control_widget.setStyleSheet(
            "font-family: 'Microsoft YaHei'; font-size: 10pt;")
        control_widget.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        control_layout = QVBoxLayout(control_widget)
        control_layout.setSpacing(10)

        # ---------------- 右侧面板：定义能级 → 管理能级 → 定义场 → 模式区 -------------
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
        control_layout.addLayout(row_add)
        # 能级组列表（"添加能级组"底下，右键删除）
        self.group_list = QListWidget()
        self.group_list.setContextMenuPolicy(Qt.CustomContextMenu)
        self.group_list.customContextMenuRequested.connect(self.show_context_menu)
        self.group_list.setFixedHeight(90)
        control_layout.addWidget(self.group_list)

        # 定义场：静磁场 + 静电场 + 添加光（两页共用同一份）
        self.field = ui_fields.FieldPanel(self)
        self.field.changed.connect(self.on_light_changed)
        control_layout.addWidget(self.field)

        # ---- 单体特有控件（两体模式隐藏）----
        self.single_widget = QWidget()
        single_lay = QVBoxLayout(self.single_widget)
        single_lay.setSpacing(6)
        control_layout.addWidget(self.single_widget)

        # 频移计算后端（单体模式显示）；enable/偏振/椭偏 已下沉到智能光行
        row_backend = QHBoxLayout()
        row_backend.addWidget(QLabel('频移'))
        self.backend_box = QComboBox()
        self.backend_box.addItems(['矩阵对角化', '解析/微扰'])
        self.backend_box.setToolTip(
            '频移计算后端：\n'
            '  矩阵对角化（默认）：Floquet 直接对角化，自洽含近共振 dressed +\n'
            '    远失谐反旋项 + 多态同时耦合，精度最高（numpy，无需 GPU）\n'
            '  解析/微扰：逐伙伴 RWA dressed + 远失谐 SOS，速度最快')
        self.backend_box.setCurrentIndex(0 if core.SHIFT_BACKEND == 'direct' else 1)
        self.backend_box.currentIndexChanged.connect(self.on_backend_changed)
        row_backend.addWidget(self.backend_box, 1)
        single_lay.addLayout(row_backend)
        # 已下沉到智能光行的调光控件（属性保留供旧方法访问，隐藏）
        self.enable_box = QCheckBox('address 光')
        self.enable_box.setChecked(True)
        self.enable_box.toggled.connect(self.on_address_changed)
        self.enable_box.hide()
        self.pol_box = QComboBox()
        self.pol_box.addItems(core.POLARIZATION_LABELS + ['任意椭偏'])
        self.pol_box.currentTextChanged.connect(self.on_address_changed)
        self.pol_box.hide()
        self.hidden_host = QWidget(self)
        hidden_lay = QVBoxLayout(self.hidden_host)

        # 任意椭偏参数行：传播方向角 α / 主轴方位角 θ / 椭偏角 ε（仅椭偏时显示）
        self.ell_widget = QWidget()
        row_ell = QHBoxLayout(self.ell_widget)
        row_ell.setContentsMargins(0, 0, 0, 0)
        row_ell.addWidget(QLabel('α°'))
        self.alpha_spin = QDoubleSpinBox()
        self.alpha_spin.setRange(0.0, 90.0)
        self.alpha_spin.setDecimals(1)
        self.alpha_spin.setSingleStep(1.0)
        self.alpha_spin.setValue(0.0)
        self.alpha_spin.setToolTip('传播方向与量子化轴 z 的夹角（0°=沿 z，90°=垂直 z）')
        self.alpha_spin.editingFinished.connect(self.on_address_changed)
        row_ell.addWidget(self.alpha_spin)
        row_ell.addWidget(QLabel('θ°'))
        self.theta_spin = QDoubleSpinBox()
        self.theta_spin.setRange(0.0, 180.0)
        self.theta_spin.setDecimals(1)
        self.theta_spin.setSingleStep(1.0)
        self.theta_spin.setValue(0.0)
        self.theta_spin.setToolTip('偏振椭圆主轴在垂直传播方向平面内的方位角（0~180°）')
        self.theta_spin.editingFinished.connect(self.on_address_changed)
        row_ell.addWidget(self.theta_spin)
        row_ell.addWidget(QLabel('ε°'))
        self.eps_spin = QDoubleSpinBox()
        self.eps_spin.setRange(-45.0, 45.0)
        self.eps_spin.setDecimals(1)
        self.eps_spin.setSingleStep(1.0)
        self.eps_spin.setValue(0.0)
        self.eps_spin.setToolTip('椭偏角（0=线偏，+45=左旋圆偏，−45=右旋圆偏）')
        self.eps_spin.editingFinished.connect(self.on_address_changed)
        row_ell.addWidget(self.eps_spin)
        hidden_lay.addWidget(self.ell_widget)
        # 仅当选"任意椭偏"时才显示（on_address_changed 里切换显隐）
        self.ell_widget.setVisible(False)

        # 光强数值 + 粗调滑块 一行
        row_intensity = QHBoxLayout()
        row_intensity.addWidget(QLabel('光强 (mW/cm²)'))
        self.intensity_box = QDoubleSpinBox()
        self.intensity_box.setRange(0.0, 1000000000.0)
        # 小步长：扫描结果附近精细微调（不再一次跳 1e6）
        self.intensity_box.setSingleStep(1000)
        self.intensity_box.setDecimals(1)
        self.intensity_box.setValue(1.0)
        self.intensity_box.editingFinished.connect(self.on_address_changed)
        row_intensity.addWidget(self.intensity_box)
        self.intensity_slider = QSlider(Qt.Horizontal)
        self.intensity_slider.setRange(0, 1000000000)
        self.intensity_slider.setValue(1)
        self.intensity_slider.sliderReleased.connect(self.on_intensity_slider)
        row_intensity.addWidget(self.intensity_slider, 1)
        hidden_lay.addLayout(row_intensity)

        # 光强扫描：固定 B/偏振/频率，小步长扫描光强使选中能级对的跃迁频移=目标值
        row_scan = QHBoxLayout()
        row_scan.addWidget(QLabel('目标频移 (MHz)'))
        self.target_shift_spin = QDoubleSpinBox()
        self.target_shift_spin.setRange(-100000.0, 100000.0)
        self.target_shift_spin.setDecimals(2)
        self.target_shift_spin.setSingleStep(1.0)
        self.target_shift_spin.setValue(20.0)
        self.target_shift_spin.setToolTip(
            '想要选中能级对（From→To）的跃迁频率频移（带符号，MHz）')
        row_scan.addWidget(self.target_shift_spin)
        self.scan_button = QPushButton('扫描光强')
        self.scan_button.setToolTip(
            '先在下方能谱表选中一条谱线，设置目标频移后点此按钮：\n'
            '保持 B/偏振/频率不变，对选中能级对小步长扫描光强，\n'
            '使该能级对跃迁频率频移达到目标值（结果写入光强框）。\n'
            '目标量 = 能级差频移 Δν = AC_To − AC_From，不是单个能级位移。')
        self.scan_button.clicked.connect(self.on_scan_intensity)
        row_scan.addWidget(self.scan_button)
        hidden_lay.addLayout(row_scan)
        # 扫描目标能级对显示（选中谱线即更新，让用户确认算的是哪一对）
        self.scan_target_label = QLabel('')
        self.scan_target_label.setWordWrap(True)
        self.scan_target_label.setStyleSheet('color: #0b5394;')
        hidden_lay.addWidget(self.scan_target_label)

        # 频率一行
        row_freq = QHBoxLayout()
        row_freq.addWidget(QLabel('频率 (GHz)'))
        self.freq_box = QDoubleSpinBox()
        self.freq_box.setRange(0.0, 10000000.0)
        self.freq_box.setSingleStep(0.1)
        self.freq_box.setDecimals(3)
        self.freq_box.setValue(295724.048)
        self.freq_box.editingFinished.connect(self.on_freq_edited)
        row_freq.addWidget(self.freq_box, 1)
        hidden_lay.addLayout(row_freq)

        # 从能谱表确定频率：下方能谱表选中谱线后点此按钮，
        # address 光频率 = 谱线频率 + 失谐 Δ
        row_get = QHBoxLayout()
        self.get_from_spec_button = QPushButton('获取自能谱')
        self.get_from_spec_button.setToolTip(
            '先在下方能谱表选中一条谱线，再点此按钮：\n'
            '将 address 光频率设为 谱线频率 + 失谐 Δ')
        self.get_from_spec_button.clicked.connect(self.on_get_from_spectrum)
        row_get.addWidget(self.get_from_spec_button)
        row_get.addWidget(QLabel('失谐 Δ (MHz)'))
        self.detune_spin = QDoubleSpinBox()
        self.detune_spin.setRange(-1000000.0, 1000000.0)
        self.detune_spin.setDecimals(1)
        self.detune_spin.setSingleStep(10.0)
        self.detune_spin.setValue(0.0)
        self.detune_spin.editingFinished.connect(self.on_detune_edited)
        row_get.addWidget(self.detune_spin, 1)
        hidden_lay.addLayout(row_get)
        self._set_detune_indicator(False)   # 初始为直接频率模式：失谐未生效（红标）

        self.detuning_label = QLabel('')
        self.detuning_label.setWordWrap(True)
        hidden_lay.addWidget(self.detuning_label)

        self.relative_box = QCheckBox('相对显示')
        self.relative_box.setChecked(True)
        self.relative_box.toggled.connect(self.on_address_changed)
        # 能级图显示模式：单一 / off-on 双图
        row_display = QHBoxLayout()
        row_display.addWidget(self.relative_box)
        row_display.addWidget(QLabel('能级图'))
        self.level_mode_box = QComboBox()
        self.level_mode_box.addItems(['单一', 'off/on 双图'])
        self.level_mode_box.setToolTip(
            '能级图显示模式：\n'
            '  单一：一张能级图（Zeeman + AC 位移与箭头），off/on 跃迁连线叠加\n'
            '  off/on 双图：非寻址光（裸能级）与寻址光（移后）各一张并排，\n'
            '    分别标注对应频谱（off 谱 / on 谱）选中的跃迁连线')
        self.level_mode_box.currentIndexChanged.connect(self.on_level_mode_changed)
        row_display.addWidget(self.level_mode_box)
        single_lay.addLayout(row_display)

        # y 轴上下限组 一行
        row_y = QHBoxLayout()
        row_y.addWidget(QLabel('y 下限'))
        self.y_lower_combo = QComboBox()
        self.y_lower_combo.currentIndexChanged.connect(self.on_y_node_changed)
        row_y.addWidget(self.y_lower_combo, 1)
        row_y.addWidget(QLabel('上限'))
        self.y_upper_combo = QComboBox()
        self.y_upper_combo.currentIndexChanged.connect(self.on_y_node_changed)
        row_y.addWidget(self.y_upper_combo, 1)
        single_lay.addLayout(row_y)
        self.y_view_label = QLabel('')
        single_lay.addWidget(self.y_view_label)

        # 频率参考值单独一行（在谱范围组件上方），默认 17.8168 GHz
        row_ref = QHBoxLayout()
        row_ref.addWidget(QLabel('参考频率 (GHz)'))
        self.spec_ref_edit = QLineEdit('17.2368')
        self.spec_ref_edit.setPlaceholderText('空=不画')
        self.spec_ref_edit.returnPressed.connect(self.on_spectrum_ref_changed)
        row_ref.addWidget(self.spec_ref_edit, 1)
        single_lay.addLayout(row_ref)
        self.on_spectrum_ref_changed()   # 初始化默认参考线

        # 频谱横轴上下限 一行
        row_specx = QHBoxLayout()
        row_specx.addWidget(QLabel('谱x 下限'))
        self.spec_freq_lo_edit = QLineEdit()
        self.spec_freq_lo_edit.setPlaceholderText('自动')
        self.spec_freq_lo_edit.returnPressed.connect(self.on_spectrum_range_changed)
        row_specx.addWidget(self.spec_freq_lo_edit, 1)
        row_specx.addWidget(QLabel('上限'))
        self.spec_freq_hi_edit = QLineEdit()
        self.spec_freq_hi_edit.setPlaceholderText('自动')
        self.spec_freq_hi_edit.returnPressed.connect(self.on_spectrum_range_changed)
        row_specx.addWidget(self.spec_freq_hi_edit, 1)
        single_lay.addLayout(row_specx)

        # 双谱选中峰频率：off 峰频率 | on 峰频率 | 频率差（一行，不标在图上）
        self.peak_pair_label = QLabel('')
        self.peak_pair_label.setStyleSheet(
            'font-family: Consolas; font-size: 9pt;')
        self.peak_pair_label.setWordWrap(False)
        single_lay.addWidget(self.peak_pair_label)

        # ---- 单体旧版数据表：能级数值 + off/on 两个谱表（还原）----
        self.info_text = QPlainTextEdit()
        self.info_text.setReadOnly(True)
        self.info_text.setFixedHeight(110)
        self.info_text.setLineWrapMode(QPlainTextEdit.NoWrap)
        self.info_text.setStyleSheet('font-family: Consolas; font-size: 9pt;')
        single_lay.addWidget(self.info_text)
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
        self.peak_table.setFixedHeight(140)
        pair_lay.addWidget(self.peak_table)
        control_layout.addWidget(self.pair_widget)

        # 点击频谱峰 → 对应峰表滚动并选中该行（定位跃迁对）
        self.canvas.peakPicked.connect(self._on_peak_picked)
        # 选中峰频率变化 → 右侧面板显示 off/on 频率与频率差
        self.canvas.peakFreqsChanged.connect(self._update_peak_pair_label)

        # 面板放入滚动区（仅纵向滚动；宽度跟随视图，禁用横向滚动条）
        self.panel_scroll = QScrollArea()
        self.panel_scroll.setWidgetResizable(True)
        self.panel_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.panel_scroll.setWidget(control_widget)
        top.addWidget(self.panel_scroll, 2)

        self._rebuild_y_combos()
        # 初始磁场由 FieldPanel 提供
        self.field_B = self.field.b_gauss()
        # 初始模式：画布默认单体 tab，只显示单体特有区（两体区隐藏）
        self.single_widget.setVisible(True)
        self.pair_widget.setVisible(False)
        self.field.rebuild_focus(show_exchange=True)   # 初始为单体模式
        self.refresh_single()
        # 启动后按全屏尺寸适配各组件（面板宽度、表高、画布 dpi、字号）
        QTimer.singleShot(0, self._apply_fullscreen_layout)

    # ---------------- 模式切换 / 添加光 / 响应谱 -------------
    def on_canvas_tab_changed(self, idx):
        """切换左侧画布：单体=响应谱或能级图，两体=响应谱；面板公共区共用。"""
        single = (idx == 0)
        self.single_widget.setVisible(single)
        self.pair_widget.setVisible(not single)
        self.field.rebuild_focus(show_exchange=single)
        if single:
            self.refresh_single()
        else:
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
        """场/光变化：更新 field_B + 单体 address 同步 + 按模式刷新。"""
        self.field_B = self.field.b_gauss()
        self._sync_address_from_lights()
        if self.canvas_tabs.currentIndex() == 0:
            self.refresh_single()
        else:
            self.render_response()

    def _sync_address_from_lights(self):
        """单体模式：第一份光同步到 address_cfg（现有能级/频谱计算）。"""
        c = self.field.first_cfg()
        if c:
            self.address_cfg.update(
                frequency_hz=c['frequency_hz'],
                intensity=c['intensity'],
                polarization=c['polarization'])
            self.pol_box.setCurrentText(c['polarization'])
            self.intensity_box.setValue(c['intensity'])
            self.freq_box.setValue(c['frequency_hz'] / 1e9)
            # 最后勾选启用 + 显式设 enabled（pol_box 默认可能同值不触发刷新）
            self.enable_box.setChecked(True)
            self.address_cfg['enabled'] = True
        else:
            self.enable_box.setChecked(False)

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
    def on_field_changed(self, *_):
        # valueChanged 实时提交（带新值参数，忽略）；从 FieldPanel 读取
        self.field_B = self.field.b_gauss()
        self.refresh_plot()

    def on_backend_changed(self, *_):
        """切换频移计算后端：矩阵对角化（默认）/ 解析微扰。"""
        core.SHIFT_BACKEND = ('direct' if self.backend_box.currentIndex() == 0
                              else 'legacy')
        self.refresh_plot()

    def on_address_changed(self, *_):
        self.address_cfg['enabled'] = self.enable_box.isChecked()
        self.address_cfg['intensity'] = self.intensity_box.value()
        self.address_cfg['frequency_hz'] = self.freq_box.value() * 1e9
        self.address_cfg['polarization'] = self.pol_box.currentText()
        # 任意椭偏：写入三参数；标准偏振则清除（address_pol_coeffs 据此切换）
        is_ell = (self.address_cfg['polarization'] == '任意椭偏')
        self.ell_widget.setVisible(is_ell)
        if is_ell:
            self.address_cfg['elliptical'] = {
                'alpha_deg': self.alpha_spin.value(),
                'theta_deg': self.theta_spin.value(),
                'eps_deg': self.eps_spin.value(),
            }
        else:
            self.address_cfg.pop('elliptical', None)
        # 同步光强滑块（整数粗调；QSlider.setValue 会自动按滑块范围钳位）
        slider_val = int(round(self.intensity_box.value()))
        if self.intensity_slider.value() != slider_val:
            self.intensity_slider.blockSignals(True)
            self.intensity_slider.setValue(slider_val)
            self.intensity_slider.blockSignals(False)
        self.refresh_plot()

    def on_intensity_slider(self):
        # 滑块释放时才提交：滑块 -> 数值框 -> 刷新（sliderReleased 无参数）
        self.intensity_box.setValue(float(self.intensity_slider.value()))
        self.on_address_changed()

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
        table.setFixedHeight(170)
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
            self._scan_pair = None
            self._update_scan_target()
            return
        self._selected_spec_table = which
        self._selected_spec_row = idx
        table.selectRow(idx)
        table.scrollToItem(table.item(idx, 0))
        self._store_scan_pair(which, idx)

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
        self._store_scan_pair(which, row)

    def on_freq_edited(self):
        """用户直接修改频率 → 回到"直接频率"模式（目标对失效，失谐标红）。"""
        if not self._programmatic_freq:
            self._clear_target_mode()
        self.on_address_changed()

    def _clear_target_mode(self):
        """清除目标跃迁对模式：失谐组件标记为不生效（红边框）。"""
        self._target_line = None
        self._set_detune_indicator(False)

    def on_detune_edited(self):
        """失谐组件回车/完成编辑：目标模式下按 目标线频率+Δ 重算频率。"""
        if self._target_line is None:
            self.statusBar().showMessage(
                '当前为直接频率模式；请先用"获取自能谱"指定目标跃迁对', 5000)
            return
        nu_line = self._target_line[2]
        nu = nu_line + self.detune_spin.value() * 1e6
        self._set_freq_programmatic(nu)
        self.statusBar().showMessage(
            f'失谐 Δ={self.detune_spin.value():+.1f} MHz → 频率 {nu / 1e9:.4f} GHz', 5000)

    def _set_freq_programmatic(self, nu):
        """程序设置频率框（不触发"用户手动改频→清除目标模式"），并刷新。"""
        self._programmatic_freq = True
        self.freq_box.setValue(nu / 1e9)
        self.freq_box.editingFinished.emit()   # → on_freq_edited（跳过清除）→ 刷新
        self._programmatic_freq = False
        self._set_detune_indicator(True)

    def _set_detune_indicator(self, valid):
        """失谐组件标记：无效（直接频率模式/失谐不匹配）时红边框。"""
        self.detune_spin.setStyleSheet(
            '' if valid else 'QDoubleSpinBox { border: 1px solid #d00; }')

    def on_get_from_spectrum(self):
        """用能谱表选中谱线的频率（+失谐 Δ）确定寻址光频率，进入目标跃迁对模式。"""
        lines = (self.canvas._lines_off if self._selected_spec_table == 'off'
                 else self.canvas._lines_on)
        row = self._selected_spec_row
        if row < 0 or row >= len(lines):
            QMessageBox.information(self, '提示', '请先在下方能谱表中选中一条谱线')
            return
        nu_line, A, la, lb, qa, qb, col, pol = lines[row]
        nu = nu_line + self.detune_spin.value() * 1e6   # Hz
        # 记录目标跃迁对（模式 B：频率由 目标线频率+失谐 决定）
        self._target_line = (self._selected_spec_table, row, nu_line,
                             la, qa, lb, qb, pol)
        self._set_freq_programmatic(nu)
        self.statusBar().showMessage(
            f'已设目标跃迁对 {la} {qa} → {lb} {qb} (pol={pol})：'
            f'线频 {nu_line / 1e9:.4f} GHz + Δ{self.detune_spin.value():+.1f} MHz'
            f' = {nu / 1e9:.4f} GHz', 8000)

    # ---------------- 光强扫描（固定 B/偏振/频率，小步长迭代求目标频移） ----------------
    def _store_scan_pair(self, which, row):
        """从选中谱线记录扫描目标能级对 (la, qa, lb, qb)。

        跟随"能级对"而非行下标（与"获取自能谱"的 _target_line 同理）：
        参数变化谱线重排后仍指向用户选的那一对。未选中则清空。
        """
        lines = (self.canvas._lines_off if which == 'off' else self.canvas._lines_on)
        if row < 0 or row >= len(lines):
            self._scan_pair = None
        else:
            nu, A, la, lb, qa, qb, col, pol = lines[row]
            self._scan_pair = (la, qa, lb, qb)
        self._update_scan_target()

    def _update_scan_target(self):
        """在界面上显示当前扫描目标能级对（让用户确认算的是哪一对）。"""
        if self._scan_pair is None:
            self.scan_target_label.setText('')
            return
        la, qa, lb, qb = self._scan_pair
        src = 'off 谱' if self._selected_spec_table == 'off' else 'on 谱'
        self.scan_target_label.setText(
            f'扫描目标能级对（{src}）：{la} {qa} → {lb} {qb}')

    def _pair_shift_at_intensity(self, la, qa, lb, qb, I_mw, cfg=None):
        """给定光强下，选中能级对 (From=la,qa → To=lb,qb) 的跃迁频率频移 (MHz)。

        目标量 = 能级对之间能级差的频移（盯单个能级没有意义）：
            Δν(I) = [E_To(I) − E_From(I)] − [E_To0 − E_From0] = ac_To − ac_From
        扫描统一用 direct 后端：单次 Floquet 对角化自洽（多态/超精细相干），
        不随用户当前后端切换而改变扫描结果。能级对不在可见组中返回 None。
        cfg：光配置（频率/偏振/强度），缺省用当前 address_cfg。
        """
        cfg = dict(self.address_cfg if cfg is None else cfg)
        cfg['enabled'] = True
        cfg['intensity'] = I_mw
        rows = core.build_level_rows(
            self.cache, self.field_B, self.groups, cfg,
            self.groups, backend='direct')
        ac = {}
        for r in rows:
            if (r['label'], r['qstr']) in ((la, qa), (lb, qb)):
                ac[(r['label'], r['qstr'])] = r['ac']
        if (la, qa) not in ac or (lb, qb) not in ac:
            return None
        return ac[(lb, qb)] - ac[(la, qa)]

    def _sweep_pair_shifts(self, keys, shifts, la, qa, lb, qb):
        """从 direct_ac_stark_sweep 结果提取能级对频移数组 (MHz)；缺失返回 None。

        keys 为 [(n,l,j,qstr,occ), ...]，label 由 core.auto_label 反推，
        与选中谱线的 (label, qstr) 匹配（重复 qstr 取第一个）。
        """
        la_key = lb_key = None
        for k in keys:
            n, l, j, qstr, occ = k
            if core.auto_label(n, l, j) == la and qstr == qa and la_key is None:
                la_key = k
            if core.auto_label(n, l, j) == lb and qstr == qb and lb_key is None:
                lb_key = k
        if la_key is None or lb_key is None:
            return None
        return shifts[lb_key] - shifts[la_key]

    def on_scan_intensity(self):
        """固定 B/偏振/频率，对选中能级对小步长扫描光强，使跃迁频移=目标值。

        目标 = 选中能级对（From→To）的跃迁频率频移 Δν = ac_To − ac_From
        （不是单个能级位移——能级差才是有物理意义的量）。
        算法（direct_ac_stark_sweep：一次构建耦合、按 sqrt(I) 重标定批量对角化）：
          1) win=1 精简基组粗扫 24 点定位跨越区间；
          2) win=2 基组（与默认全基组误差 <0.3%）区间内小步长细扫 24 点 + 插值；
          3) 默认全基组（win=3）验证，超出容差则割线校正一次。
        目标区间内光强相对步长 <3%，最终以软件默认全基组结果为准。
        """
        pair = self._scan_pair
        if pair is None:
            QMessageBox.information(self, '提示', '请先在下方能谱表中选中一条谱线')
            return
        la, qa, lb, qb = pair
        target = self.target_shift_spin.value()   # MHz
        if abs(target) < 1e-9:
            QMessageBox.information(self, '提示', '目标频移不能为 0')
            return
        if not self.address_cfg.get('enabled', False):
            QMessageBox.information(self, '提示', '请先勾选 address 光（扫描需在寻址光下）')
            return
        cfg = dict(self.address_cfg)
        cfg['enabled'] = True
        SCAN_WIN, REFINE_WIN = 1, 2
        tol = max(abs(target) * 0.01, 0.05)    # MHz

        self._update_scan_target()
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            # 1) win=1 对数粗扫：定位目标频移所在区间（覆盖弱到强驱动）
            grid = np.geomspace(1e-6, 1e9, 24)     # mW/cm²
            keys, s_by_key = core.direct_ac_stark_sweep(
                self.cache, self.groups, cfg, self.field_B, grid,
                sos_n_window=SCAN_WIN)
            shifts = self._sweep_pair_shifts(keys, s_by_key, la, qa, lb, qb)
            if shifts is None:
                QMessageBox.warning(
                    self, '能级对缺失',
                    f'扫描时找不到能级对 {la} {qa} → {lb} {qb}，\n'
                    f'请重新在能谱表中选中一条谱线。')
                return
            # 找所有跨越目标值的区间（多态共振/非单调时可能有多个）
            crossings = [k for k in range(len(grid) - 1)
                         if (shifts[k] < target < shifts[k + 1])
                         or (shifts[k + 1] < target < shifts[k])]
            if not crossings:
                best = int(np.argmin(np.abs(shifts - target)))
                QMessageBox.warning(
                    self, '未收敛',
                    f'光强扫描未跨越目标频移 {target:+.2f} MHz\n'
                    f'（可能强驱动多态共振，光强→频移不再单调）。\n'
                    f'最接近点：I={grid[best]:.4g} mW/cm² → 频移 {shifts[best]:+.3f} MHz。\n'
                    f'请调整目标值，或改用失谐/频率扫描。')
                self._set_intensity_and_refresh(grid[best])
                return
            # 多个跨越区间取端点频移最接近目标值的一个（最相关段）
            idx = min(crossings,
                      key=lambda c: min(abs(shifts[c] - target),
                                        abs(shifts[c + 1] - target)))
            Ia, Ib = grid[idx], grid[idx + 1]

            # 2) win=2 区间内小步长细扫 + 插值（与默认全基组误差 <0.3%）
            fine = np.geomspace(Ia, Ib, 24)
            keys2, f_by_key = core.direct_ac_stark_sweep(
                self.cache, self.groups, cfg, self.field_B, fine,
                sos_n_window=REFINE_WIN)
            sf = self._sweep_pair_shifts(keys2, f_by_key, la, qa, lb, qb)
            fj = next((j for j in range(len(fine) - 1)
                       if (sf[j] < target < sf[j + 1])
                       or (sf[j + 1] < target < sf[j])), None)
            if fj is None:
                jb = int(np.argmin(np.abs(sf - target)))
                I_best = float(fine[jb])
            else:
                t = (target - sf[fj]) / (sf[fj + 1] - sf[fj]) if sf[fj + 1] != sf[fj] else 0.5
                t = min(max(t, 1e-9), 1 - 1e-9)
                I_best = fine[fj] + t * (fine[fj + 1] - fine[fj])
            I_best = min(max(I_best, 1e-6), 1e9)

            # 3) 默认全基组（win=3）验证；超出容差则割线校正一次
            actual = self._pair_shift_at_intensity(la, qa, lb, qb, I_best)
            if actual is None:
                QMessageBox.warning(
                    self, '能级对缺失',
                    f'最终验证找不到能级对 {la} {qa} → {lb} {qb}')
                return
            if abs(actual - target) > tol:
                I2 = I_best * (1.3 if actual < target else 0.7)
                S2 = self._pair_shift_at_intensity(la, qa, lb, qb, I2)
                if S2 is not None and abs(S2 - actual) > 1e-12:
                    I_best = I_best + (target - actual) * (I2 - I_best) / (S2 - actual)
                    I_best = min(max(I_best, 1e-6), 1e9)
                    actual = self._pair_shift_at_intensity(la, qa, lb, qb, I_best)
                    if actual is None:
                        actual = 0.0
            self._set_intensity_and_refresh(I_best)
            src = 'off 谱' if self._selected_spec_table == 'off' else 'on 谱'
            self.scan_target_label.setText(
                f'扫描目标能级对（{src}）：{la} {qa} → {lb} {qb}\n'
                f'目标频移 {target:+.3f} MHz → 光强 {I_best:.6g} mW/cm²'
                f'（实际 {actual:+.3f} MHz）')
            self.statusBar().showMessage(
                f'扫描完成：目标频移 {target:+.3f} MHz @ {la} {qa}→{lb} {qb}，'
                f'光强 {I_best:.6g} mW/cm² → 实际 {actual:+.3f} MHz'
                f'（差 {actual - target:+.3f} MHz）', 12000)
        finally:
            QApplication.restoreOverrideCursor()

    def _set_intensity_and_refresh(self, I_mw):
        """写回光强框并刷新（钳位到 spinbox 范围）。"""
        lo = self.intensity_box.minimum()
        hi = self.intensity_box.maximum()
        I_mw = min(max(I_mw, lo), hi)
        self.intensity_box.setValue(I_mw)
        self.on_address_changed()

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

    def _apply_fullscreen_layout(self):
        """全屏/最大化后按屏幕尺寸适配：面板宽度、表高、画布 dpi、字号。"""
        screen = QApplication.primaryScreen()
        if screen is None:
            return
        geo = screen.availableGeometry()
        w, h = geo.width(), geo.height()
        # 右侧面板宽度：屏宽 22% ~ 34%（超宽屏不占满）
        self.panel_scroll.setMinimumWidth(int(w * 0.22))
        self.panel_scroll.setMaximumWidth(int(w * 0.34))
        # 面板内列表/表高度按屏高比例（全屏下更大）
        self.group_list.setFixedHeight(max(90, int(h * 0.07)))
        self.peak_table.setFixedHeight(max(110, int(h * 0.1)))
        # 画布 dpi：按屏高提升分辨率，文字保持清晰
        dpi = max(100, min(int(h / 9), 150))
        for fig in (self.canvas.level_fig, self.canvas.level_off_fig,
                    self.canvas.level_on_fig, self.canvas.spec_mirror_fig,
                    self.pair_canvas.fig):
            fig.set_dpi(dpi)
        # 控制面板字号略增
        if self.panel_scroll.widget() is not None:
            self.panel_scroll.widget().setStyleSheet(
                "font-family: 'Microsoft YaHei'; font-size: 11pt;")
        self.refresh_plot()

    def refresh_plot(self):
        visible = self._visible_groups()
        rows = self.canvas.draw_levels(
            self.cache, self.field_B, self.groups,
            self.address_cfg, self.relative_box.isChecked(), visible)

        # 谱峰表：仅当谱线数据更新时重填
        if self.canvas._lines_off is not self._last_lines_off:
            self._last_lines_off = self.canvas._lines_off
            self._populate_peak_table(self.spec_off_table, self.canvas._lines_off)
        if self.canvas._lines_on is not self._last_lines_on:
            self._last_lines_on = self.canvas._lines_on
            self._populate_peak_table(self.spec_on_table, self.canvas._lines_on)

        # 更新视野标签：显示当前上下限节点范围
        if self._sorted_groups:
            lo = max(0, self.y_lower_combo.currentIndex())
            hi = min(len(self._sorted_groups) - 1, self.y_upper_combo.currentIndex())
            if lo <= hi:
                self.y_view_label.setText(
                    f'显示 {self._sorted_groups[lo]["label"]} ~ '
                    f'{self._sorted_groups[hi]["label"]}（共 {hi - lo + 1} 个组）')
        else:
            self.y_view_label.setText('')

        # 更新跃迁对 / 近共振警告：
        # - 目标模式（获取自能谱）：显示能谱表确定的跃迁对，失谐以组件为准
        # - 直接频率模式：显示最近跃迁对（core 按失谐最小选子能级对）
        if self._target_line is not None:
            which, row, nu_line, la, qa, lb, qb, pol = self._target_line
            det = self.address_cfg['frequency_hz'] - nu_line
            text = (f'目标跃迁对（能谱表确定，失谐以组件为准）：\n'
                    f'  From  {la}  {qa}\n'
                    f'  To    {lb}  {qb}\n'
                    f'  Det   {det / 1e6:+.1f} MHz\n'
                    f'  Pol   {pol}')
            self.detuning_label.setStyleSheet('')
            self.detuning_label.setText(text)
        else:
            pairs = core.collect_transitions(
                self.cache, self.groups, self.address_cfg, self.field_B)
            if pairs:
                closest = min(pairs, key=lambda p: abs(p[4]))
                text = (f'最近跃迁对（直接频率模式）：\n'
                        f'  From  {closest[0]}  {closest[1]}\n'
                        f'  To    {closest[2]}  {closest[3]}\n'
                        f'  Det   {closest[4] / 1e6:+.1f} MHz\n'
                        f'  Rabi  {closest[5] / (2 * np.pi) / 1e6:.1f} MHz')
                near = [p for p in pairs
                        if abs(p[4]) < max(core.AC_FAR_DETUNE_CUTOFF_HZ, p[5] / (2.0 * np.pi))]
                if len(near) >= 2:
                    text += (f'\n⚠ 多态近共振：{len(near)} 个跃迁对同时近共振'
                             f'（两能级模型可能失效，请调大失谐）')
                    self.detuning_label.setStyleSheet('color: #c00000;')
                else:
                    self.detuning_label.setStyleSheet('')
                self.detuning_label.setText(text)
            else:
                self.detuning_label.setText('')
                self.detuning_label.setStyleSheet('')

        # 更新数值表：能级差恒为相对最低子能级的正值（不受显示模式影响）
        if rows:
            lines = ['能级数值（MHz，相对最低子能级）：']
            ref = np.min([r['E'] for r in rows])
            for r in rows:
                flag = '*' if r['affected'] else ' '
                lines.append(
                    f"{flag}{r['label']:<12} {r['qstr']:<18} E={r['E'] - ref:12.3f}"
                    f"  (AC {r['ac']:+.3f})")
            self.info_text.setPlainText('\n'.join(lines))
        else:
            self.info_text.setPlainText('')


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    # 最大化而非真全屏：保留右上角最小化/最大化/关闭三个窗口按钮
    window.showMaximized()
    sys.exit(app.exec_())
