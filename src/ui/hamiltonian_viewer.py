# -*- coding: utf-8 -*-
"""哈密顿量可视化（重写版）：扩展空间 H(t) 的多级标注自适应热图。

设计要点（从真项目迭代沉淀）：
- 滚轮缩放 + 左键平移（pg ViewBox.PanMode，GPU blit 不卡顿）；
- **分层索引标注 v2**（三级，全部随"每格像素 ppu"自适应，见下方常量）：
  * L0 扇区 n=±k：仅 ≥2 个扇区可见时出现，色块底 + 贯穿分隔线；
  * L1 流形组（如 '6P 3/2'）：字号 10/9/8/7 取放得下的最大档，
    放不下时降级为头部（'60S 1/2'→'60S'），再放不下才隐藏；
  * L2 逐维：ppu≥11 才出现；最深档（ppu≥34）显示 '索引 · 伙伴态'，
    中档显示伙伴态名，浅档只给索引号；密度抽稀步长 = ceil(13/ppu)，
    同一轴所有标签统一字号（不出现参差）；
  * 精确身份（全名/能量/矩阵元）由底栏 hover 与右侧表负责，
    图上只做"这是哪一段、哪一维"的粗略定位；
- hover 联动：当前行/列的扇区/组/逐维标注一并染成强调色；
- 全部标注走场景层 QGraphicsSimpleTextItem，避免 pg AxisItem 隐形坑
  （长名整组消失、setTicks 格式崩溃等）；旋转定位走 QTransform，
  不用 setTransformOriginPoint（离屏 grab 下定位易偏半个字宽）；
- 扇区白线分隔 + 组背景淡色填充；
- 鼠标 hover：十字高亮当前行/列 + 浮窗显示 (a,b)/|H|/arg/物理含义；
- 右侧色条，零值=白底、非零值用 viridis 系——零元素视觉直接"消失"；
- 顶/底/左/右四侧留白在 GraphicsLayoutWidget.ci 上做，labels 全部 addItem
  到 glw.scene()（场景层像素坐标，不受 ViewBox 裁剪影响）。

测试方式：见 ui_design/shot_hamiltonian.py（offscreen 多档缩放截图）。
"""
from __future__ import annotations
import math
import re
from typing import List, Tuple, Optional, Dict, Any

import numpy as np

from PyQt5.QtCore import Qt, QRectF, QPointF, QEvent, QTimer, pyqtSignal
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QFrame,
                              QLabel, QCheckBox, QPushButton, QSlider,
                              QGraphicsSimpleTextItem, QGraphicsRectItem,
                              QGraphicsPathItem, QSizePolicy, QToolTip)
import pyqtgraph as pg


# ============================================================
# 工具
# ============================================================

# 渐变预热：pg.colormap.get 在首次调用时较慢；模块导入时触发一次
_VIRIDIS = pg.colormap.get('viridis')


def _to_unicode_sub(s: str) -> str:
    """LaTeX 下标 → 普通分数形式：60P_{3/2} → 60P 3/2。

    （注：曾尝试 Unicode 下标 ₁₂₃，但雅黑无该字形渲染成方框；
    普通形式无字体依赖、宽度相近。）
    """
    import re as _re
    return _re.sub(r'_\{([^}]*)\}', r' \1', s)


def _derive_groups(labels: List[str]) -> List[Tuple[int, int, str]]:
    """从 labels 推导连续流形组 [(start, end, name), ...]（end 开区间）。

    组名带角标：取标签开头到第一个空格的主干（'60S₁/₂ m_J=+0.5' →
    '60S₁/₂'；两体积态去外括号取首段），连续相同合并为一段。
    """
    out = []
    for i, lab in enumerate(labels):
        head = str(lab).lstrip('(').split()[0] if str(lab).split() else str(lab)
        head = _to_unicode_sub(head)
        if out and out[-1][2] == head and out[-1][1] == i:
            out[-1][1] = i + 1
        else:
            out.append([i, i + 1, head])
    return [(s, e, n) for s, e, n in out]

class HamiltonianViewer(QWidget):
    """H(t) 热图可视化：多级物理标注 + hover + 色条 + 缩放平移。

    数据契约（同 mock_diag 产出 / evolve_mixed(... diag=True).diag）：
        meta:    {dim, labels, stride, P, ...}
        photon:  {D_full, ...}  ← 用于显示能量
        H_snapshots: {t0: {term: dim×dim complex}}

    用法：
        v = HamiltonianViewer()
        v.set_data(snapshots, meta, photon)
        v.set_visible_terms(['K0', '光场', ...])     # 可选：默认全开
        v.strip_K0_diag = True                       # 可选：去K0对角
    """

    # 数据变化信号：滑块/项勾选/去对角 都会触发；外部可借此刷新附属面板
    dataChanged = pyqtSignal()

    # 配色
    COL_BG = '#ffffff'
    COL_GRID = '#e6e9ec'
    COL_GRID_MAJOR = '#cdd2d8'
    COL_DIVIDER = '#1a1a1a'   # 扇区分隔线：深色实线
    COL_BORDER = '#2c3137'    # 外框
    COL_LABEL = '#2c3137'     # 主文字
    COL_LABEL_DIM = '#7a8088' # 能量/单位 等次要文字
    COL_TOOLTIP_BG = '#21262d'
    COL_TOOLTIP_FG = '#f5f6f8'

    # 顶/左条扇区名配色（按扇区序循环）
    SECTOR_COLORS = ['#185FA5', '#0F6E56', '#854F0B', '#72243E', '#534AB7',
                     '#A32D2D']

    def __init__(self, parent=None):
        super().__init__(parent)
        self._snapshots: Dict[float, Dict[str, np.ndarray]] = {}
        self._ts_sorted: List[float] = []
        self._meta: Dict[str, Any] = {}
        self._photon: Dict[str, Any] = {}
        self._D_full: Optional[np.ndarray] = None
        self._dim = 0
        self._stride = 0
        self._P = 0
        self._labels: List[str] = []
        self._groups: List[Tuple[int, int, str]] = []
        self._strip_K0 = True
        self._term_set: Dict[str, bool] = {}

        # 场景层缓存（避免 zoom 时反复 alloc）
        self._sector_lines: List[QGraphicsPathItem] = []
        self._group_tints: List[QGraphicsRectItem] = []
        self._group_lines: List[QGraphicsLineItem] = []
        self._x_tick_items: List[QGraphicsSimpleTextItem] = []
        self._y_tick_items: List[QGraphicsSimpleTextItem] = []
        self._top_strip_items: Dict[str, QGraphicsSimpleTextItem] = {}
        self._left_strip_items: Dict[str, QGraphicsSimpleTextItem] = {}
        # 标注文本联动表：(item, 基础字体, 基础颜色, 元数据)——hover 时改色用
        self._band_items: List[Tuple[Any, QtGui.QFont, str, Dict[str, Any]]] = []
        self._band_chrome: List[Any] = []      # 带底/分隔线等非文本装饰
        self._hover_items: List[QGraphicsPathItem] = []
        self._hover_text: Optional[QGraphicsSimpleTextItem] = None
        self._last_hover_cell: Optional[Tuple[int, int]] = None
        self._deco_pending = False   # 延迟重绘合并标志（见 _on_view_changed）
        self._term_color: Dict[str, str] = {
            'K0': '#444441', '光场': '#185FA5', '微波': '#854F0B',
            'V_dd': '#0F6E56', 'AC对角': '#72243E', '耗散': '#A32D2D'}

        self._build_ui()
        self._wire()

    # ---------- 构造 ----------
    def _init_metrics(self):
        """把布局像素绑到**字体度量**上，而不是写死。

        字号是 pt，其像素高度随屏幕 DPI 缩放（125%/150% 很常见）。写死
        23px 的带高在缩放屏上装不下 10pt 文字 → `_fit_font` 返回 None →
        扇区名整条消失（用户实测：x 轴全丢、y 轴只差 3px 阈值所以还在）。
        """
        f10 = QtGui.QFont('Microsoft YaHei', 10)
        f10.setBold(True)
        h10 = math.ceil(QtGui.QFontMetricsF(f10).height())
        self.TOP_GROUP_H = h10 + 4
        self.TOP_SECTOR_H = h10 + 4
        self.LEFT_GROUP_W = h10 + 8
        self.LEFT_SECTOR_W = h10 + 8
        w7 = QtGui.QFontMetricsF(QtGui.QFont('Microsoft YaHei', 7))
        self.DIM_GUTTER = math.ceil(
            w7.boundingRect('000 · 6P 3/2 F2 +0').width()) + 14
        self.MARGIN_L = (self.DIM_GUTTER + self.BAND_GAP
                         + self.LEFT_GROUP_W + self.LEFT_SECTOR_W)
        self.MARGIN_T = (self.DIM_GUTTER + self.BAND_GAP
                         + self.TOP_GROUP_H + self.TOP_SECTOR_H)

    def _build_ui(self):
        self._init_metrics()
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(6)

        # ---- 工具条：时间滑块 + 项开关 + 视图控制 ----
        bar = QHBoxLayout()
        bar.setSpacing(8)
        self.t_label = QLabel('t = — ns')
        self.t_label.setStyleSheet(
            'QLabel{font-family:Consolas,monospace; font-size:11pt; '
            'font-weight:600; color:#2c3137;}')
        self.t_label.setMinimumWidth(130)
        bar.addWidget(self.t_label)
        self.t_slider = QSlider(Qt.Horizontal)
        self.t_slider.setRange(0, 99)
        self.t_slider.setValue(50)
        self.t_slider.setMinimumWidth(160)
        bar.addWidget(self.t_slider, 1)
        # 项开关：默认按项颜色着色文字
        self.term_boxes: Dict[str, QCheckBox] = {}
        for name, color in self._term_color.items():
            cb = QCheckBox(name)
            cb.setChecked(name in ('K0', '光场', '微波'))
            cb.setStyleSheet(
                f'QCheckBox{{color:{color}; font-weight:500;}}')
            self.term_boxes[name] = cb
            bar.addWidget(cb)
        self.strip_box = QCheckBox('去 K0 对角')
        self.strip_box.setChecked(True)
        self.strip_box.setToolTip(
            'K0 对角（THz 级能移）会淹没耦合项——勾选后隐去对角，只看耦合结构')
        bar.addWidget(self.strip_box)
        reset_btn = QPushButton('⟲ 还原视图')
        reset_btn.setToolTip('恢复全矩阵视图（0..dim）')
        bar.addWidget(reset_btn)
        self._reset_btn = reset_btn
        outer.addLayout(bar)

        # ---- 主图区：左条 + 热图 + 色条 ----
        center = QHBoxLayout()
        center.setSpacing(0)

        # 左侧的"组/扇区条" + y 轴标签区（用 QFrame 占位，不放 scene 外面）
        # 直接由 GraphicsLayoutWidget 的 ci.left margin 处理
        # 这里只用 central widget（plot）+ 一个色条小部件
        center_wrap = QWidget()
        center_wrap_lay = QHBoxLayout(center_wrap)
        center_wrap_lay.setContentsMargins(0, 0, 0, 0)
        center_wrap_lay.setSpacing(0)

        self.glw = pg.GraphicsLayoutWidget()
        self.glw.setBackground(self.COL_BG)
        # 留白（l/t/r/b）——这些像素在 scene 上是"可绘但 ViewBox 不可见"
        self.glw.ci.setContentsMargins(self.MARGIN_L, self.MARGIN_T,
                                       self.MARGIN_R, self.MARGIN_B)
        # ↑ 左/上 = 逐维槽 + 组名带 + 扇区带（x/y 同构，细标都在最外沿）
        # ↑ 右 18px 给色条留呼吸；下 18px 已无标注（细标移到上/左）
        self.plot = self.glw.addPlot(row=0, col=0)
        self.plot.setAspectLocked(True)
        # 隐藏 pg 原生轴——我们用场景层自绘
        self.plot.hideAxis('bottom')
        self.plot.hideAxis('left')
        self.vb = self.plot.getViewBox()
        self.vb.setMouseMode(self.vb.PanMode)
        self.vb.setMenuEnabled(False)
        self.vb.setAcceptHoverEvents(True)
        # 矩阵惯例：行 0 在顶部（row index 向下增长），主对角线左上→右下
        self.vb.invertY(True)
        self.img = pg.ImageItem(axisOrder='row-major')
        self.plot.addItem(self.img)
        # 自定义 LUT：白(0) → viridis(max)，让零元素在视觉上"消失"
        self._build_lut()
        self.img.setLookupTable(self._lut)
        # scene 事件过滤（hover）
        self.glw.scene().installEventFilter(self)
        self.vb.sigXRangeChanged.connect(self._on_view_changed)
        self.vb.sigYRangeChanged.connect(self._on_view_changed)
        # 布局/尺寸稳定后必须重绘装饰——否则 set_data 期间（布局未跑完）
        # 用退化变换画出的刻度会残留在最终画面上
        self.vb.sigResized.connect(self._on_view_changed)

        center_wrap_lay.addWidget(self.glw, 1)
        # 色条
        self.cbar = _ColorBar(min_label='0', max_label='|H| max',
                              lut=self._lut, parent=self)
        self.cbar.setFixedWidth(36)
        self.cbar.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Expanding)
        center_wrap_lay.addWidget(self.cbar, 0, Qt.AlignVCenter)

        center.addWidget(center_wrap, 1)
        outer.addLayout(center, 1)

        # ---- 底栏：摘要 + hover 详情 ----
        foot = QHBoxLayout()
        foot.setSpacing(8)
        self.dim_label = QLabel('dim = —')
        self.dim_label.setStyleSheet(
            'QLabel{color:#6b7280; font-size:10pt;}')
        foot.addWidget(self.dim_label)
        self.hover_label = QLabel('悬停查看矩阵元详情')
        self.hover_label.setStyleSheet(
            'QLabel{color:#6b7280; font-size:10pt; font-style:italic;}')
        self.hover_label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        # 关键：布局忽略 sizeHint（两体积态全名 40+ 字符会把本行撑开、
        # 整体重排错位）——文本超宽时省略号截断，全文放 tooltip
        self.hover_label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Fixed)
        self.hover_label.setFixedHeight(46)   # 两行：每行显示一个态
        self.hover_label.setMinimumWidth(0)
        foot.addWidget(self.hover_label, 1)
        outer.addLayout(foot)

    def _build_lut(self, n=512):
        """白(0) → viridis(max) 的 512 段 LUT（3 通道 RGB）。"""
        v = _VIRIDIS.getLookupTable(0.0, 1.0, n)  # (n, 3) uint8
        # 替换最前 8 段为白色（容许极小数值视为零——数值上的浮点精度噪声）
        n_white = max(4, n // 64)
        v[:n_white] = np.array([255, 255, 255], dtype=np.ubyte)
        self._lut = v
        self._lut_white_n = n_white

    def _wire(self):
        self.t_slider.valueChanged.connect(self._on_slider)
        for cb in self.term_boxes.values():
            cb.stateChanged.connect(self._on_term_change)
        self.strip_box.stateChanged.connect(self._on_term_change)
        self._reset_btn.clicked.connect(self._reset_view)
        # 颜色联动 term 名（公式着色/勾选项同色——以后可接）

    # ---------- 公共 API ----------
    def set_data(self, snapshots, meta, photon=None):
        """snapshots: {t0: {term: dim×dim complex}}; meta 含 dim/labels/stride;
        photon: photon dict（含 D_full）。
        """
        self._snapshots = snapshots or {}
        # 两层诊断策略（2026-09-12）：轻量层永远有；重型 H 快照在 dim 过大
        # 时被跳过（{'skipped': …}）或为空 —— 此时不画图，否则 pyqtgraph
        # 在 setImage(空) 之后 setRect 会 ZeroDivisionError。
        self._h_skipped = ((not self._snapshots)
                           or ('skipped' in self._snapshots))
        self._ts_sorted = sorted(self._snapshots.keys())
        self._meta = meta or {}
        self._photon = photon or {}
        self._dim = int(self._meta.get('dim', 0))
        self._stride = int(self._meta.get('stride', 0) or 0)
        self._P = self._dim // self._stride if self._stride else 0
        self._labels = list(self._meta.get('labels') or [])
        self._groups = _derive_groups(self._labels) if self._labels else []
        self._D_full = (self._photon.get('D_full') if self._photon
                        else None)
        if self._D_full is not None:
            self._D_full = np.asarray(self._D_full, float)
        # 同步勾选项：data 里有哪几项就启用
        present = set()
        if self._ts_sorted:
            present = set(self._snapshots[self._ts_sorted[0]].keys())
        for n, cb in self.term_boxes.items():
            cb.blockSignals(True)
            cb.setEnabled(n in present)
            cb.setChecked(cb.isChecked() and n in present)
            cb.blockSignals(False)
        # 滑块
        self.t_slider.blockSignals(True)
        self.t_slider.setRange(0, max(len(self._ts_sorted) - 1, 0))
        if self._ts_sorted:
            self.t_slider.setValue(len(self._ts_sorted) // 2)
        self.t_slider.blockSignals(False)
        self._dim_label_text()
        self._refresh_image()
        self._reset_view()

    def set_visible_terms(self, terms: List[str]):
        """手动覆写项勾选（外部控制用）；terms 内的会开，其它关。"""
        for n, cb in self.term_boxes.items():
            cb.blockSignals(True)
            cb.setChecked(n in terms)
            cb.blockSignals(False)
        self._refresh_image()

    def set_strip_K0_diag(self, on: bool):
        self.strip_box.blockSignals(True)
        self.strip_box.setChecked(bool(on))
        self.strip_box.blockSignals(False)
        self._strip_K0 = bool(on)
        self._refresh_image()

    # ---------- 信号回调 ----------
    def _on_slider(self, _v=None):
        if not self._ts_sorted:
            return
        i = self.t_slider.value()
        i = max(0, min(i, len(self._ts_sorted) - 1))
        t0 = self._ts_sorted[i]
        self.t_label.setText(f't = {t0:.3f} ns')
        self._refresh_image()
        self.dataChanged.emit()

    def _on_term_change(self, _v=None):
        self._strip_K0 = self.strip_box.isChecked()
        self._refresh_image()
        self.dataChanged.emit()

    def _on_view_changed(self, *_):
        # sigXRangeChanged 在 setRange 中途发射，此时 img 的场景变换还是
        # 旧值——立刻重绘会用退化几何。推迟到事件循环空闲时（变换已更新），
        # 并用 pending 标志合并连续缩放产生的密集重绘。
        if self._deco_pending:
            return
        self._deco_pending = True
        QTimer.singleShot(0, self._run_update_decorations)

    def _run_update_decorations(self):
        self._deco_pending = False
        self._update_decorations()
        self._fix_scene_rect()

    def resizeEvent(self, ev):
        super().resizeEvent(ev)
        self._fix_scene_rect()               # 窗口尺寸变化后重钉 sceneRect

    def _fix_scene_rect(self):
        """一次性固定 sceneRect 为大画布（4000×4000，幂等）。

        QGraphicsScene 的 sceneRect 默认随 item 并集单调扩张（不缩回）：
        aspect-lock 下 ImageItem 会溢出视口（正常，ViewBox 裁剪显示），但
        溢出部分会单调撑大 sceneRect（实测可被撑到 4567）→ 视图基准漂移、
        缩放平移交互错乱。
        ⚠ 只能钉一次固定值：若按 widget 当前尺寸钉，之后窗口 resize 变大时
        sceneRect 不会跟随（显式设置后自动扩张关闭），内容会被裁出视图。
        """
        r = self.glw.scene().sceneRect()
        if (r.width(), r.height()) != (4000.0, 4000.0):
            self.glw.scene().setSceneRect(0, 0, 4000, 4000)

    def _reset_view(self):
        if not self._dim:
            return
        self.vb.setRange(xRange=(0, self._dim), yRange=(0, self._dim),
                         padding=0)
        self._update_decorations()
        self._fix_scene_rect()

    # ---------- 数据 → 图像 ----------
    def _dim_label_text(self):
        if not self._dim:
            self.dim_label.setText('dim = —')
        else:
            n_sec = self._P if self._P else 1
            self.dim_label.setText(
                f'dim = {self._dim}   '
                f'扇区 {n_sec} × {self._dim // max(n_sec,1)}')

    def _current_H(self) -> np.ndarray:
        if not self._ts_sorted:
            return np.zeros((0, 0), complex)
        i = max(0, min(self.t_slider.value(), len(self._ts_sorted) - 1))
        t0 = self._ts_sorted[i]
        terms = self._snapshots[t0]
        present = [n for n, cb in self.term_boxes.items()
                   if cb.isChecked() and n in terms]
        H = np.zeros((self._dim, self._dim), complex)
        for n in present:
            H = H + terms[n]
        if self._strip_K0:
            H = H - np.diag(np.diag(H))
        return H

    def _refresh_image(self):
        if not self._dim or getattr(self, '_h_skipped', False):
            return
        H = self._current_H()
        A = np.abs(H)
        # 保留旧 viewRange（如果有）
        keep = self.vb.viewRange() if self.img.image is not None else None
        # row-major：A[a, b] → 图像第 a 行第 b 列；setImage(A.T) → image[j, a]
        self.img.setImage(A.T, autoLevels=False)
        vmax = float(A.max()) if A.size else 1.0
        self.img.setLevels([0.0, vmax or 1.0])
        self.img.setRect(QRectF(0, 0, self._dim, self._dim))
        if keep is not None:
            (x0, x1), (y0, y1) = keep
            x0 = min(max(x0, 0), self._dim)
            x1 = min(max(x1, 0), self._dim)
            y0 = min(max(y0, 0), self._dim)
            y1 = min(max(y1, 0), self._dim)
            if x1 > x0 and y1 > y0:
                self.vb.setRange(xRange=(x0, x1), yRange=(y0, y1),
                                 padding=0)
        # 色条同步
        self.cbar.set_range(0.0, vmax or 1.0, unit='MHz')
        # 更新 t 标签（不触发 _on_slider 自身递归）
        if self._ts_sorted:
            i = max(0, min(self.t_slider.value(), len(self._ts_sorted) - 1))
            self.t_label.setText(f't = {self._ts_sorted[i]:.3f} ns')
        self._update_decorations()

    # ============================================================
    # 场景层绘制（扇区分隔、组背景、顶/左条、主轴标签）
    # ============================================================

    def _clear_items(self, items):
        # 先复制内容、再清原容器（注意：直接 items = list(items) 会丢
        # 掉原引用，导致原 list 永远清不掉、孤儿项逐次累积）
        if isinstance(items, dict):
            vals = list(items.values())
            items.clear()
        else:
            vals = list(items)
            items.clear()
        for it in vals:
            try:
                self.glw.scene().removeItem(it)
            except Exception:
                pass

    def _full_scene_geom(self):
        """完整热图（数据 0..dim × 0..dim）的场景矩形 → (fx, fy, fw, fh)。

        注意：不能用 img.mapRectToScene——item 的场景变换在 setRange/resize
        后是"惰性"更新的，信号触发时读到的可能是滞后值。这里从一手数据
        同步推导：viewRange（信号发出时已更新）× vb 场景矩形（布局定），
        aspect-lock 下 pg 会扩展 range 使两轴比例一致，故 sx == sy。
        """
        if self._dim == 0 or self.img.image is None:
            return 0.0, 0.0, 1.0, 1.0
        (rx0, rx1), (ry0, ry1) = self.vb.viewRange()
        vb = self.vb.sceneBoundingRect()
        if rx1 <= rx0 or ry1 <= ry0 or vb.width() < 2 or vb.height() < 2:
            return 0.0, 0.0, 1.0, 1.0
        sx = vb.width() / (rx1 - rx0)
        sy = vb.height() / (ry1 - ry0)
        fx = vb.left() - rx0 * sx
        fy = vb.top() - ry0 * sy   # invertY(True)：数据 y=0 在视口顶边
        return fx, fy, self._dim * sx, self._dim * sy

    def _viewport_scene_geom(self):
        """当前视口内"可见的热图部分"的场景矩形 → (ox, oy, pw, ph)。

        供**数据类**装饰使用：扇区/分组分隔线、hover 行列高亮、命中测试
        （这些只该覆盖真正有数据的区域）。表头带/外框/轴刻度改用
        `_frame_scene_geom()`（钉在视口上，不随拖拽移动）。
        """
        if self._dim == 0 or self.img.image is None:
            return 0.0, 0.0, 1.0, 1.0
        fx, fy, fw, fh = self._full_scene_geom()
        vb_scene = self.vb.sceneBoundingRect()
        x0 = max(fx, vb_scene.left())
        y0 = max(fy, vb_scene.top())
        x1 = min(fx + fw, vb_scene.right())
        y1 = min(fy + fh, vb_scene.bottom())
        if x1 <= x0 or y1 <= y0:
            return fx, fy, fw, fh
        return x0, y0, x1 - x0, y1 - y0

    def _frame_scene_geom(self):
        """**外框 / 表头**的定位基准 → 视口矩形 (ox, oy, pw, ph)。

        与 `_viewport_scene_geom()`（可见热图 = 矩阵∩视口）的区别：
        矩阵比视口小时（缩到全景、或拖到数据边缘），交集会跟着矩阵跑，
        于是"表头和外框被一起拖走"。改用视口矩形后——外框四条边钉死在
        视口上、表头钉在视口左上，**拖拽/平移只动里面的矩阵图像**；
        表头文字仍按数据坐标映射，所以内容照样随平移/缩放更新。
        """
        vb = self.vb.sceneBoundingRect()
        if vb.width() < 2 or vb.height() < 2:
            return self._viewport_scene_geom()
        return vb.left(), vb.top(), vb.width(), vb.height()

    def _update_decorations(self):
        if not self._dim:
            return
        # 清旧
        self._clear_items(self._sector_lines)
        self._clear_items(self._group_tints)
        self._clear_items(self._group_lines)
        self._clear_items(self._x_tick_items)
        self._clear_items(self._y_tick_items)
        # dict 的 clear 方式：先复制 values 到 list
        for k in list(self._top_strip_items.keys()):
            self._clear_items([self._top_strip_items.pop(k)])
        for k in list(self._left_strip_items.keys()):
            self._clear_items([self._left_strip_items.pop(k)])
        # 带底/分隔线（非文本装饰）+ 文本联动表
        self._clear_items(self._band_chrome)
        self._band_items.clear()
        # 重画
        self._draw_sector_dividers()
        self._draw_group_dividers()
        self._draw_group_tints()
        self._draw_top_band()
        self._draw_left_band()
        self._draw_axis_ticks()
        self._draw_matrix_frame()
        # 保留 hover 元素
        if self._last_hover_cell is not None:
            self._draw_hover(self._last_hover_cell, force=True)

    # ---- 组间分隔线（每个扇区内，组段边界的深灰细线） ----
    def _draw_group_dividers(self):
        if not self._groups or not self._stride or self._P < 1:
            return
        fx, fy, fw, fh = self._full_scene_geom()
        ox, oy, pw, ph = self._viewport_scene_geom()
        (rx0, rx1), (ry0, ry1) = self.vb.viewRange()
        if fw <= 0 or fh <= 0 or self._dim == 0:
            return
        sx = fw / self._dim                       # 数据→场景（px/维）
        sy = fh / self._dim
        boundaries = [g[1] for g in self._groups[:-1]]     # 组段内边界
        pen = QtGui.QPen(QtGui.QColor('#6b7280'), 0.8)
        for b in boundaries:
            for p in range(self._P):
                d = p * self._stride + b                   # 扩展坐标
                if not (rx0 <= d <= rx1):
                    continue
                xa = fx + d * sx
                # 竖线：只画在**可见热图**的纵向范围内
                ya = max(min(fy + p * self._stride * sy,
                             fy + (p + 1) * self._stride * sy), oy)
                yb = min(max(fy + p * self._stride * sy,
                             fy + (p + 1) * self._stride * sy), oy + ph)
                if yb > ya and ox <= xa <= ox + pw:
                    ln_v = QtWidgets.QGraphicsLineItem(xa, ya, xa, yb)
                    ln_v.setPen(pen)
                    ln_v.setZValue(2.6)
                    self.glw.scene().addItem(ln_v)
                    self._group_lines.append(ln_v)
                # 横线：只画在**可见热图**的横向范围内
                y0h = fy + d * sy
                x0h = max(min(fx + p * self._stride * sx,
                              fx + (p + 1) * self._stride * sx), ox)
                x1h = min(max(fx + p * self._stride * sx,
                              fx + (p + 1) * self._stride * sx), ox + pw)
                if x1h > x0h and oy <= y0h <= oy + ph:
                    ln_h = QtWidgets.QGraphicsLineItem(x0h, y0h, x1h, y0h)
                    ln_h.setPen(pen)
                    ln_h.setZValue(2.6)
                    self.glw.scene().addItem(ln_h)
                    self._group_lines.append(ln_h)

    # ---- 扇区分隔（深色实线 + 浅色高光） ----
    def _draw_sector_dividers(self):
        """扇区分隔线：只画在**可见热图**范围内。

        旧实现用 pg.InfiniteLine——它横贯整个 ViewBox，在非方形 ViewBox 下
        会伸到方形矩阵之外的空白里（"贯穿全场"）。改为场景层有界线段，
        端点取可见热图矩形。
        """
        if not self._stride or self._P < 2:
            return
        fx, fy, fw, fh = self._full_scene_geom()
        ox, oy, pw, ph = self._viewport_scene_geom()
        (rx0, rx1), (ry0, ry1) = self.vb.viewRange()
        if fw <= 0 or fh <= 0:
            return
        sx = fw / self._dim
        sy = fh / self._dim
        for p in range(1, self._P):
            d = p * self._stride
            # 竖线（数据 x = d）
            if rx0 <= d <= rx1:
                x = fx + d * sx
                if ox <= x <= ox + pw:
                    for color, w in (('#ffffff', 1.6), ('#1a1a1a', 0.6)):
                        self._band_line(x, oy, x, oy + ph, color, w, 5.0)
            # 横线（数据 y = d）
            if ry0 <= d <= ry1:
                y = fy + d * sy
                if oy <= y <= oy + ph:
                    for color, w in (('#ffffff', 1.6), ('#1a1a1a', 0.6)):
                        self._band_line(ox, y, ox + pw, y, color, w, 5.0)

    # ---- 组背景淡色填充（扇区内，以中央扇区为模板向所有扇区复制） ----
    def _draw_group_tints(self):
        if not self._groups or not self._stride or self._P < 1:
            return
        # 颜色：按组出现顺序循环一组很淡的色调
        tint_colors = ['#F0E0C0', '#C0D8F0', '#C0E0C8', '#F0C8D0', '#D8C8F0',
                       '#F0E8C0']
        pc = self._P // 2  # 中央扇区（结构周期重复）
        for gi, (s, e, _name) in enumerate(self._groups):
            color = tint_colors[gi % len(tint_colors)]
            for p in range(self._P):
                x0 = p * self._stride + s
                x1 = p * self._stride + e
                rect = QGraphicsRectItem(x0, x0, x1 - x0, x1 - x0)
                rect.setBrush(QtGui.QColor(color))
                rect.setPen(QtGui.QPen(QtCore.Qt.NoPen))
                rect.setOpacity(0.13)   # 淡：仅作背景参照，不抢主图
                rect.setZValue(1.5)   # 在 heatmap 之上但在分隔线之下
                self.plot.addItem(rect)
                self._group_tints.append(rect)

    # ============================================================
    # 分层索引标注 v2（全部走场景层 QGraphicsSimpleTextItem）
    # ------------------------------------------------------------
    # 统一布局（x/y 完全对称，行为一致）：
    #   贴近热图 → 外沿：  组名带 → 扇区带 → 逐维槽
    #   x 轴全部在上，y 轴全部在左（不再出现"一半上一半下"）。
    # 由于 aspect lock 保证 sx == sy，两轴 ppu 恒等；再给两侧**相同**的
    # 标注预算（`_dim_avail()` 取两轴较小者），同一视野下两轴的档位/
    # 密度/字号必然一致。
    #
    # 三级层级，随"每格像素 ppu"自适应：
    #   L0 扇区 n=±k  —— 仅当 ≥2 个扇区可见；色块底 + 深色贯穿分隔线
    #   L1 流形组     —— 字号 10/9/8/7 取放得下的最大档；名字放不下时
    #                    降级为头部（'60S 1/2'→'60S'），再放不下才隐藏
    #   L2 逐维       —— ppu≥11 才出现；ppu≥34 显示 '索引 · 伙伴态'，
    #                    ≥22 显示伙伴态名，否则只给索引号；
    #                    字号 8/7/6 逐级降级；抽稀步长 = ceil(13/ppu)
    #   ⚠ 可见区只落在**一个组段**内时，L0/L1 整条带隐藏（那时它只有
    #     "你在哪个组"这一条信息），空出的槽深全部让给 L2 细标。
    # 精确身份（全名/能量/矩阵元）由底栏 hover 与右侧表负责，
    # 图上标注只承担"这是哪一段、哪一维"的粗略定位。
    # ============================================================

    # 下面这些**像素常量在 `_init_metrics()` 里按字体度量重算**（实例属性），
    # 这里只留类级兜底值。原因：字号是 pt，其像素高度随屏幕 DPI 缩放，
    # 写死 px 的带高在 125%/150% 缩放的屏幕上装不下 10pt 文字。
    DIM_GUTTER = 96        # 逐维标注槽深度（px）——x 在上 / y 在左，同宽
    TOP_GROUP_H = 23       # 顶带：贴近热图的一行 = 组名
    TOP_SECTOR_H = 23      # 顶带：外侧一行 = 扇区名
    LEFT_GROUP_W = 27      # 左带：贴近热图的一列 = 组名（竖排）
    LEFT_SECTOR_W = 26     # 左带：外侧一列 = 扇区名（竖排）
    BAND_GAP = 3           # 带与热图之间的缝隙
    DIM_MIN_PPU = 6.0      # 逐维标注出现的最小格宽（px）——全视野也留刻度尺
    DIM_MIN_GAP = 13.0     # 逐维标注的最小间距（px）→ 抽稀步长

    # 顶/左留白 = 逐维槽 + 带（两侧同构）；同样在 `_init_metrics()` 里重算
    MARGIN_L = DIM_GUTTER + BAND_GAP + LEFT_GROUP_W + LEFT_SECTOR_W
    MARGIN_T = DIM_GUTTER + BAND_GAP + TOP_GROUP_H + TOP_SECTOR_H
    MARGIN_R = 12
    MARGIN_B = 12          # 细标已移到上/左，右下只留一点呼吸

    COL_BAND_BG = '#F8FAFC'
    COL_BAND_ZEBRA = '#EDF1F6'
    COL_BAND_LINE = '#E2E8F0'
    COL_BAND_EDGE = '#CBD5E1'
    COL_TICK = '#94A3B8'
    COL_HL = '#E05A3A'

    # ---------- 小工具 ----------
    @staticmethod
    def _fit_font(text, avail_len, avail_thick=None,
                  sizes=(10, 9, 8, 7), bold=False,
                  family='Microsoft YaHei'):
        """字号阶梯里挑"放得下"的最大档；全放不下 → (None, None)。"""
        for s in sizes:
            f = QtGui.QFont(family, s)
            f.setBold(bold)
            fm = QtGui.QFontMetricsF(f)
            # 用 boundingRect 而非 width：QGraphicsSimpleTextItem 的盒宽就是
            # boundingRect，width() 在高 DPI 下会低估几个像素 → 标签越界。
            if fm.boundingRect(text).width() > avail_len:
                continue
            if avail_thick is not None and fm.height() > avail_thick:
                continue
            return s, f
        return None, None

    @staticmethod
    def _compact_partner(label):
        """两体积态 '(A)×(B)' → B 侧紧凑名（如 '6P F2 −1'）。

        单体积态/无 '×' 时退化为整体紧凑名。只服务图上 L2 粗略标注；
        完整名永远由底栏 hover 给出。
        """
        s = str(label or '')
        if ')×(' in s:
            s = s.split(')×(', 1)[1]
        s = _to_unicode_sub(s.strip().strip('()'))
        toks = s.replace(',', ' ').split()
        if not toks:
            return ''
        head = toks[0]
        if len(toks) > 1 and re.fullmatch(r'\d+/\d+', toks[1]):
            head = f'{head} {toks[1]}'
        fpart = ''
        for t in toks[1:]:
            if t.startswith('F='):
                fpart = ' F' + t.split('=', 1)[1].rstrip('0').rstrip('.')
                break
            if re.fullmatch(r'F\d+(?:\.\d+)?', t):
                fpart = ' ' + t
                break
        val = ''
        for t in reversed(toks[1:]):
            if '=' in t and not t.startswith('F='):
                val = t.split('=', 1)[1]
                break
        if not val and len(toks) > 1 and not re.fullmatch(r'\d+/\d+', toks[-1]):
            val = toks[-1]                    # 单体态：'+½' / '−1' 这类末位量子数
        if val:
            try:
                x = float(val)
                if abs(x - round(x)) < 1e-9:
                    val = f'{int(round(x)):+d}'
                else:
                    val = f'{x:+.2f}'.rstrip('0').rstrip('.')
            except ValueError:
                pass
            val = val.replace('-', '−')
        return f'{head}{fpart} {val}'.strip()

    def _dim_tick_texts(self, e, ppu):
        """第 e 维的分层候选标注（详细 → 简略），信息量随 ppu 分三档。"""
        if self._stride:
            _p, i = divmod(e, self._stride)
        else:
            i = e
        lab = self._labels[i] if 0 <= i < len(self._labels) else ''
        full = self._compact_partner(lab)
        head = full.split()[0] if full else ''
        if ppu >= 34.0:                       # 最深档：索引 + 伙伴态（可交叉查表）
            return [f'{e} · {full}', full, head, str(e)]
        if ppu >= 22.0:                       # 中档：伙伴态名
            return [full, head, str(e)]
        return [str(e)]                       # 浅档：只给索引号

    def _add_text(self, txt, font, color, cx, cy, rotate=0, z=8.0,
                  meta=None, anchor='c'):
        """放一个场景层文本：以 (cx, cy) 为锚点，可选旋转。

        anchor：c=盒中心 / n=盒顶贴 cy / s=盒底贴 cy / e=盒右贴 cx /
                w=盒左贴 cx（旋转后按"旋转后的场景盒"计算）。
        旋转走 QTransform（translate→rotate→translate(-w/2,-h/2)），
        不依赖 setTransformOriginPoint 的定位语义——后者在离屏 grab 下
        定位易偏半个字宽（旧代码踩过）。
        """
        it = QGraphicsSimpleTextItem(txt)
        it.setFont(font)
        it.setBrush(QtGui.QBrush(QtGui.QColor(color)))
        br = it.boundingRect()
        sw = br.height() if rotate else br.width()
        sh = br.width() if rotate else br.height()
        if anchor == 'c':
            bx, by = cx - sw / 2, cy - sh / 2
        elif anchor == 'n':
            bx, by = cx - sw / 2, cy
        elif anchor == 's':
            bx, by = cx - sw / 2, cy - sh
        elif anchor == 'e':
            bx, by = cx - sw, cy - sh / 2
        else:                                  # 'w'
            bx, by = cx, cy - sh / 2
        if rotate:
            t = QtGui.QTransform()
            t.translate(bx + sw / 2, by + sh / 2)
            t.rotate(rotate)
            t.translate(-br.width() / 2, -br.height() / 2)
            it.setTransform(t)
        else:
            it.setPos(bx, by)
        it.setZValue(z)
        if meta:
            it.setData(0, meta)
            self._band_items.append((it, QtGui.QFont(font), color, meta))
        self.glw.scene().addItem(it)
        return it

    def _band_rect(self, rect, color, z=7.0):
        r = QGraphicsRectItem(rect)
        r.setBrush(QtGui.QColor(color))
        r.setPen(QtGui.QPen(QtCore.Qt.NoPen))
        r.setZValue(z)
        self.glw.scene().addItem(r)
        self._band_chrome.append(r)
        return r

    def _band_line(self, x0, y0, x1, y1, color, w=1.0, z=7.4):
        ln = QtWidgets.QGraphicsLineItem(x0, y0, x1, y1)
        ln.setPen(QtGui.QPen(QtGui.QColor(color), w))
        ln.setZValue(z)
        self.glw.scene().addItem(ln)
        self._band_chrome.append(ln)
        return ln

    def _band_counts(self, axis):
        """该轴可见的（扇区段数, 组段数）。"""
        n_sec = n_seg = 0
        if not self._stride or self._P < 1 or not self._groups:
            return 0, 0
        (x0_d, x1_d), (y0_d, y1_d) = self.vb.viewRange()
        lo, hi = (x0_d, x1_d) if axis == 'x' else (y0_d, y1_d)
        for p in range(self._P):
            s0, s1 = p * self._stride, (p + 1) * self._stride
            if min(s1, hi) <= max(s0, lo):
                continue
            n_sec += 1
            for (gs, ge, _name) in self._groups:
                a, b = s0 + gs, s0 + ge
                if min(b, hi) > max(a, lo):
                    n_seg += 1
        return n_sec, n_seg

    def _band_depth(self, axis):
        """该轴标注带从热图边缘向外占的 px。两层各自独立判断：

        · **扇区层**（`n=±k`）：仅当**该轴**可见扇区 ≥2 时才出现（只看到
          一个扇区时它就是常数，极细档连它一起让位给逐维细标）；
        · **组名层**：仅当**该轴**可见组段 ≥2 时才出现。
        x/y 两轴可见跨度可以不同，所以必须分别判断。
        """
        n_sec, n_seg = self._band_counts(axis)
        if n_sec < 1:
            return 0
        depth = self.BAND_GAP
        if n_seg >= 2:
            depth += self.TOP_GROUP_H if axis == 'x' else self.LEFT_GROUP_W
        if self._P >= 2 and n_sec >= 2:
            depth += self.TOP_SECTOR_H if axis == 'x' else self.LEFT_SECTOR_W
        return depth if depth > self.BAND_GAP else 0

    def _dim_avail(self):
        """L2 标签可用长度（px）：取两轴较小者 → 两轴档位必然一致。"""
        return max(0.0, min(self.MARGIN_T - self._band_depth('x'),
                            self.MARGIN_L - self._band_depth('y')) - 8)

    # ---------- 顶带（x 轴：扇区行 + 组名行） ----------
    def _draw_top_band(self):
        if not self._dim:
            return
        n_sec, n_seg = self._band_counts('x')
        show_grp = n_seg >= 2                 # 只落在一个组里 → 组名行让位
        show_sec = self._P >= 2 and n_sec >= 2
        if not (show_grp or show_sec):
            return
        ox, oy, pw, ph = self._frame_scene_geom()
        fx, fy, fw, fh = self._full_scene_geom()
        if fw <= 0 or pw <= 0:
            return
        (x0_d, x1_d), _ = self.vb.viewRange()
        sx = fw / self._dim
        g = self.BAND_GAP
        gh, sh = self.TOP_GROUP_H, self.TOP_SECTOR_H
        x_l, x_r = ox, ox + pw
        # 由内向外排：组名行（可选）→ 扇区行（可选）
        y_cursor = oy - g
        y_g1 = y_g0 = y_cursor
        if show_grp:
            y_g1, y_g0 = y_cursor, y_cursor - gh
            y_cursor = y_g0
        y_s1 = y_s0 = y_cursor
        if show_sec:
            y_s1, y_s0 = y_cursor, y_cursor - sh
            y_cursor = y_s0
        band_top = y_cursor

        def s_x(d):
            return fx + d * sx

        def clip(a, b):
            a2, b2 = max(a, x0_d), min(b, x1_d)
            return None if b2 <= a2 else (a2, b2)

        # ---- 扇区行可见段 ----
        sec_vis = []
        if show_sec and self._stride:
            for p in range(self._P):
                sp = clip(p * self._stride, (p + 1) * self._stride)
                if sp:
                    sec_vis.append((p, sp, (sp[1] - sp[0]) * sx))

        # ---- 带底 + 边线 ----
        self._band_rect(QRectF(x_l, band_top, x_r - x_l, oy - g - band_top),
                        self.COL_BAND_BG, z=7.0)
        self._band_line(x_l, band_top, x_r, band_top, self.COL_BAND_EDGE, 1.0)
        self._band_line(x_l, oy - g, x_r, oy - g, self.COL_BAND_EDGE, 1.0)
        if show_grp and show_sec:
            self._band_line(x_l, y_g0, x_r, y_g0, self.COL_BAND_LINE, 0.8)

        # ---- 扇区行：色块底 + 扇区名 ----
        if show_sec:
            for p, (a, b), px in sec_vis:
                color = self.SECTOR_COLORS[p % len(self.SECTOR_COLORS)]
                cx0 = max(s_x(a), x_l)
                cx1 = min(s_x(b), x_r)
                if cx1 <= cx0:
                    continue
                tint = QtGui.QColor(color)
                tint.setAlpha(28)
                self._band_rect(QRectF(cx0, y_s0, cx1 - cx0, sh),
                                tint.name(QtGui.QColor.HexArgb), z=7.1)
                txt = f'n = {p - 1:+d}' if (p - 1) != 0 else 'n = 0'
                _s, font = self._fit_font(txt, cx1 - cx0 - 10, None,
                                          sizes=(10, 9, 8), bold=True)
                if font is None:
                    continue
                it = self._add_text(txt, font, color, (cx0 + cx1) / 2,
                                    (y_s0 + y_s1) / 2, z=8.0,
                                    meta={'kind': 'sector', 'axis': 'x',
                                          'lo': a, 'hi': b})
                self._top_strip_items[f'sec{p}'] = it

        # ---- 组名行：斑马底 + 组名（放不下降级为头部；单组档整行不画） ----
        for gi, (s, e, name) in enumerate(self._groups if show_grp else []):
            for p in range(self._P):
                a, b = p * self._stride + s, p * self._stride + e
                sp = clip(a, b)
                if sp is None:
                    continue
                cx0 = max(s_x(sp[0]), x_l)
                cx1 = min(s_x(sp[1]), x_r)
                if cx1 <= cx0:
                    continue
                if gi % 2 == 1:
                    self._band_rect(QRectF(cx0, y_g0, cx1 - cx0, gh),
                                    self.COL_BAND_ZEBRA, z=7.05)
                font = None
                txt = ''
                for cand in (name, name.split()[0] if name else ''):
                    if not cand:
                        continue
                    _s, font = self._fit_font(cand, cx1 - cx0 - 8, None,
                                              sizes=(10, 9, 8, 7), bold=True)
                    if font is not None:
                        txt = cand
                        break
                if font is None:
                    continue
                it = self._add_text(txt, font, self.COL_LABEL,
                                    (cx0 + cx1) / 2, (y_g0 + y_g1) / 2,
                                    z=8.2,
                                    meta={'kind': 'group', 'axis': 'x',
                                          'lo': a, 'hi': b})
                self._top_strip_items[f'grp{gi}_{p}'] = it

        # ---- 扇区边界：贯穿整带 ----
        if show_sec and self._stride:
            for p in range(1, self._P):
                d = p * self._stride
                if not (x0_d <= d <= x1_d):
                    continue
                x = s_x(d)
                if x_l <= x <= x_r:
                    self._band_line(x, band_top, x, oy, '#334155', 1.4, 7.6)
        # ---- 组边界：从组名行顶贯穿到热图上缘 ----
        if show_grp and self._groups and self._stride:
            for gseg in self._groups[:-1]:
                b = gseg[1]
                for p in range(self._P):
                    d = p * self._stride + b
                    if not (x0_d <= d <= x1_d):
                        continue
                    x = s_x(d)
                    if x_l <= x <= x_r:
                        self._band_line(x, y_g0, x, oy, self.COL_BAND_EDGE,
                                        0.9, 7.5)

        # ---- 表头左上角：补齐顶带与左带之间的空档，让表头读起来是一整张表 ----
        left_w = self._band_depth('y')
        cx0 = ox - left_w
        if cx0 < ox and band_top < oy:
            self._band_rect(QRectF(cx0, band_top, ox - cx0, oy - band_top),
                            self.COL_BAND_BG, z=6.9)
            self._band_line(cx0, band_top, ox, band_top,
                            self.COL_BAND_EDGE, 1.0, 7.3)
            self._band_line(cx0, band_top, cx0, oy,
                            self.COL_BAND_EDGE, 1.0, 7.3)

    # ---------- 左带（y 轴：组名列 + 扇区列，均竖排） ----------
    def _draw_left_band(self):
        if not self._dim:
            return
        n_sec, n_seg = self._band_counts('y')
        show_grp = n_seg >= 2
        show_sec = self._P >= 2 and n_sec >= 2
        if not (show_grp or show_sec):
            return
        ox, oy, pw, ph = self._frame_scene_geom()
        fx, fy, fw, fh = self._full_scene_geom()
        if fh <= 0 or ph <= 0:
            return
        _, (y0_d, y1_d) = self.vb.viewRange()
        sy = fh / self._dim
        g = self.BAND_GAP
        gw, sw = self.LEFT_GROUP_W, self.LEFT_SECTOR_W
        y_t, y_b = oy, oy + ph
        # 由内向外排：组名列（可选）→ 扇区列（可选）
        x_cursor = ox - g
        x_g1 = x_g0 = x_cursor
        if show_grp:
            x_g1, x_g0 = x_cursor, x_cursor - gw
            x_cursor = x_g0
        x_s1 = x_s0 = x_cursor
        if show_sec:
            x_s1, x_s0 = x_cursor, x_cursor - sw
            x_cursor = x_s0
        band_left = x_cursor

        def s_y(d):
            return fy + d * sy

        def clip(a, b):
            a2, b2 = max(a, y0_d), min(b, y1_d)
            return None if b2 <= a2 else (a2, b2)

        sec_vis = []
        if show_sec and self._stride:
            for p in range(self._P):
                sp = clip(p * self._stride, (p + 1) * self._stride)
                if sp:
                    sec_vis.append((p, sp, (sp[1] - sp[0]) * sy))

        # ---- 带底 + 边线 ----
        self._band_rect(QRectF(band_left, y_t, ox - g - band_left, y_b - y_t),
                        self.COL_BAND_BG, z=7.0)
        self._band_line(band_left, y_t, band_left, y_b, self.COL_BAND_EDGE, 1.0)
        self._band_line(ox - g, y_t, ox - g, y_b, self.COL_BAND_EDGE, 1.0)
        if show_grp and show_sec:
            self._band_line(x_g0, y_t, x_g0, y_b, self.COL_BAND_LINE, 0.8)

        # ---- 扇区列：色块底 + 扇区名（竖排） ----
        if show_sec:
            for p, (a, b), px in sec_vis:
                color = self.SECTOR_COLORS[p % len(self.SECTOR_COLORS)]
                cy0 = max(s_y(a), y_t)
                cy1 = min(s_y(b), y_b)
                if cy1 <= cy0:
                    continue
                tint = QtGui.QColor(color)
                tint.setAlpha(28)
                self._band_rect(QRectF(x_s0, cy0, sw, cy1 - cy0),
                                tint.name(QtGui.QColor.HexArgb), z=7.1)
                txt = f'n = {p - 1:+d}' if (p - 1) != 0 else 'n = 0'
                _s, font = self._fit_font(txt, cy1 - cy0 - 10, None,
                                          sizes=(10, 9, 8), bold=True)
                if font is None:
                    continue
                it = self._add_text(txt, font, color, (x_s0 + x_s1) / 2,
                                    (cy0 + cy1) / 2, rotate=-90, z=8.0,
                                    meta={'kind': 'sector', 'axis': 'y',
                                          'lo': a, 'hi': b})
                self._left_strip_items[f'sec{p}'] = it

        # ---- 组名列：斑马底 + 组名（竖排，放不下降级为头部；单组档不画） ----
        for gi, (s, e, name) in enumerate(self._groups if show_grp else []):
            for p in range(self._P):
                a, b = p * self._stride + s, p * self._stride + e
                sp = clip(a, b)
                if sp is None:
                    continue
                cy0 = max(s_y(sp[0]), y_t)
                cy1 = min(s_y(sp[1]), y_b)
                if cy1 <= cy0:
                    continue
                if gi % 2 == 1:
                    self._band_rect(QRectF(x_g0, cy0, gw, cy1 - cy0),
                                    self.COL_BAND_ZEBRA, z=7.05)
                font = None
                txt = ''
                for cand in (name, name.split()[0] if name else ''):
                    if not cand:
                        continue
                    _s, font = self._fit_font(cand, cy1 - cy0 - 10, None,
                                              sizes=(10, 9, 8, 7), bold=True)
                    if font is not None:
                        txt = cand
                        break
                if font is None:
                    continue
                it = self._add_text(txt, font, self.COL_LABEL,
                                    (x_g0 + x_g1) / 2, (cy0 + cy1) / 2,
                                    rotate=-90, z=8.2,
                                    meta={'kind': 'group', 'axis': 'y',
                                          'lo': a, 'hi': b})
                self._left_strip_items[f'grp{gi}_{p}'] = it

        # ---- 扇区边界：贯穿整带 ----
        if show_sec and self._stride:
            for p in range(1, self._P):
                d = p * self._stride
                if not (y0_d <= d <= y1_d):
                    continue
                y = s_y(d)
                if y_t <= y <= y_b:
                    self._band_line(band_left, y, ox, y, '#334155', 1.4, 7.6)
        # ---- 组边界：从组名列左缘贯穿到热图左缘 ----
        if show_grp and self._groups and self._stride:
            for gseg in self._groups[:-1]:
                b = gseg[1]
                for p in range(self._P):
                    d = p * self._stride + b
                    if not (y0_d <= d <= y1_d):
                        continue
                    y = s_y(d)
                    if y_t <= y <= y_b:
                        self._band_line(x_g0, y, ox, y,
                                        self.COL_BAND_EDGE, 0.9, 7.5)

    # ---------- 数据 ↔ 场景坐标换算（与 _full_scene_geom 同一手推导） ----------
    def _map_data_to_scene_x(self, x_data, fx, fw):
        return fx + x_data * (fw / max(self._dim, 1))

    def _map_data_to_scene_y(self, y_data, fy, fh):
        return fy + y_data * (fh / max(self._dim, 1))

    def _scene_to_data_x(self, x_scene, fx, fw):
        return (x_scene - fx) * self._dim / max(fw, 1)

    def _scene_to_data_y(self, y_scene, fy, fh):
        return (y_scene - fy) * self._dim / max(fh, 1)

    # ---------- L2 逐维标注（自适应密度/字号/内容） ----------
    def _draw_axis_ticks(self):
        ox, oy, pw, ph = self._frame_scene_geom()
        fx, fy, fw, fh = self._full_scene_geom()
        self._draw_dim_ticks_x(ox, oy, pw, ph, fx, fw)
        self._draw_dim_ticks_y(ox, oy, pw, ph, fy, fh)

    def _dim_plan(self, axis):
        """该轴 L2 标注计划 → (es, font, texts)；本档不画则 None。

        纯函数（只读几何/数据）：绘制与几何断言共用同一判据，
        避免"该不该画"在两处各写一遍后走偏。
        """
        dim = self._dim
        if not dim:
            return None
        fx, fy, fw, fh = self._full_scene_geom()
        ox, oy, pw, ph = self._viewport_scene_geom()
        if axis == 'x':
            if fw <= 0 or pw <= 0:
                return None
            ppu = fw / dim
            lo_d, hi_d = self.vb.viewRange()[0]
        else:
            if fh <= 0 or ph <= 0:
                return None
            ppu = fh / dim
            lo_d, hi_d = self.vb.viewRange()[1]
        if ppu < self.DIM_MIN_PPU:
            return None
        step = max(1, int(np.ceil(self.DIM_MIN_GAP / ppu)))
        lo = max(0, int(np.floor(lo_d)))
        hi = min(dim - 1, int(np.ceil(hi_d)) - 1)
        if hi < lo:
            return None
        lo += (step - lo % step) % step
        es = list(range(lo, hi + 1, step))
        if not es:
            return None
        chosen = self._pick_tier(es, ppu, self._dim_avail(), step * ppu - 2)
        if chosen is None:
            return None
        font, texts = chosen
        return es, font, texts

    def _draw_dim_ticks_x(self, ox, oy, pw, ph, fx, fw):
        plan = self._dim_plan('x')
        if plan is None:
            return
        es, font, texts = plan
        # x 细标在**上方**（与 y 全部在左对称）：盒底贴带顶，向上生长
        band_top = oy - self._band_depth('x')
        y_bot = band_top - 4
        for e, txt in zip(es, texts):
            x_c = self._map_data_to_scene_x(e + 0.5, fx, fw)
            if x_c < ox + 8 or x_c > ox + pw - 8:
                continue
            self._band_line(x_c, y_bot, x_c, band_top,
                            self.COL_TICK, 1.0, 7.7)      # 刻度短线
            it = self._add_text(txt, font, self.COL_LABEL_DIM, x_c, y_bot,
                                rotate=-90, z=7.8, anchor='s',
                                meta={'kind': 'dim', 'axis': 'x', 'e': e})
            self._x_tick_items.append(it)

    def _draw_dim_ticks_y(self, ox, oy, pw, ph, fy, fh):
        plan = self._dim_plan('y')
        if plan is None:
            return
        es, font, texts = plan
        x_right = ox - self._band_depth('y') - 8
        for e, txt in zip(es, texts):
            y_c = self._map_data_to_scene_y(e + 0.5, fy, fh)
            if y_c < oy + 6 or y_c > oy + ph - 6:
                continue
            self._band_line(x_right, y_c, x_right + 4, y_c,
                            self.COL_TICK, 1.0, 7.7)      # 刻度短线
            it = self._add_text(txt, font, self.COL_LABEL_DIM, x_right, y_c,
                                z=7.8, anchor='e',
                                meta={'kind': 'dim', 'axis': 'y', 'e': e})
            self._y_tick_items.append(it)

    def _pick_tier(self, es, ppu, avail_len, avail_thick):
        """给一批维度统一选"详细度 + 字号"：从最详细档开始，要求**全部**
        标签在同一字号下放得下，保证同一轴上字号与信息量一致（不出现
        '5 · 名' 与 '名' 混排的参差）。返回 (font, texts) 或 None。

        量宽一律按**加粗**字体——hover 命中时 `_highlight_band` 会把标签
        加粗，加粗后更宽；按常规字重量宽会让加粗版越出画布（高 DPI 下
        实测旋转标签顶到画布外）。绘制仍用常规字重。
        """
        tiers = self._dim_tick_texts(es[0], ppu)
        for tier in range(len(tiers)):
            texts = [self._dim_tick_texts(e, ppu)[tier] for e in es]
            if any(not t for t in texts):
                continue
            for s in (8, 7, 6):
                fb = QtGui.QFont('Microsoft YaHei', s)
                fb.setBold(True)
                fm = QtGui.QFontMetricsF(fb)
                if fm.height() > avail_thick:
                    continue
                if any(fm.boundingRect(t).width() > avail_len for t in texts):
                    continue
                return QtGui.QFont('Microsoft YaHei', s), texts
        return None

    def _draw_matrix_frame(self):
        """外框：给**视口**加一圈细边框（位置钉死在视口上，拖拽不动）。

        底边/右边没有标注带、零值区又是白底，矩阵到哪儿结束看不出来——
        这圈框把绘图区收住。上/左两条边与表头带自己的边线重合，画重了
        也看不出来；保留整圈是为了"无表头"档（单组视图）外框仍完整。
        """
        ox, oy, pw, ph = self._frame_scene_geom()
        if pw <= 0 or ph <= 0:
            return
        r = QGraphicsRectItem(QRectF(ox, oy, pw, ph))
        r.setBrush(QtGui.QBrush(QtCore.Qt.NoBrush))
        r.setPen(QtGui.QPen(QtGui.QColor(self.COL_BAND_EDGE), 1.0))
        r.setZValue(6.0)
        self.glw.scene().addItem(r)
        self._band_chrome.append(r)

    def _highlight_band(self, a, b):
        """hover 联动：把命中当前行/列的扇区/组/逐维标注染成强调色。"""
        if not self._band_items:
            return
        for it, base_font, base_color, meta in self._band_items:
            if not meta:
                continue
            hit = False
            if a is not None:
                idx = a if meta.get('axis') == 'x' else b
                if meta.get('kind') == 'dim':
                    hit = (meta.get('e') == idx)
                else:
                    hit = (meta.get('lo', 0) <= idx < meta.get('hi', 0))
            if hit:
                f = QtGui.QFont(base_font)
                f.setBold(True)
                it.setFont(f)
                it.setBrush(QtGui.QBrush(QtGui.QColor(self.COL_HL)))
            else:
                it.setFont(base_font)
                it.setBrush(QtGui.QBrush(QtGui.QColor(base_color)))

    # ============================================================
    # Hover
    # ============================================================

    def eventFilter(self, obj, ev):
        if obj is self.glw.scene():
            if ev.type() == QEvent.GraphicsSceneMouseMove:
                pt = ev.scenePos()
                self._on_hover(pt)
                return False
        return super().eventFilter(obj, ev)

    def _on_hover(self, scene_pt: QPointF):
        if not self._dim:
            return
        ox, oy, pw, ph = self._viewport_scene_geom()   # 可见区：边界
        fx, fy, fw, fh = self._full_scene_geom()       # 完整图：数据映射
        if not (ox <= scene_pt.x() <= ox + pw and
                oy <= scene_pt.y() <= oy + ph):
            self._clear_hover()
            return
        a = int(np.clip(int(self._scene_to_data_x(scene_pt.x(), fx, fw)),
                        0, self._dim - 1))
        b = int(np.clip(int(self._scene_to_data_y(scene_pt.y(), fy, fh)),
                        0, self._dim - 1))
        # 像素更接近哪个数据单元（中心）
        ax = self._map_data_to_scene_x(a + 0.5, fx, fw)
        ay = self._map_data_to_scene_y(b + 0.5, fy, fh)
        if abs(scene_pt.x() - ax) > (fw / self._dim) / 2 or \
           abs(scene_pt.y() - ay) > (fh / self._dim) / 2:
            self._clear_hover()
            return
        if (a, b) != self._last_hover_cell:
            self._draw_hover((a, b))

    def _clear_hover(self):
        if self._last_hover_cell is None:
            return
        self._last_hover_cell = None
        self._clear_items(self._hover_items)
        if self._hover_text is not None:
            try:
                self.glw.scene().removeItem(self._hover_text)
            except Exception:
                pass
            self._hover_text = None
        self._highlight_band(None, None)
        _reset = '悬停查看矩阵元详情'
        _fm = self.hover_label.fontMetrics()
        self.hover_label.setText(_fm.elidedText(
            _reset, Qt.ElideRight, max(self.hover_label.width() - 16, 120)))
        self.hover_label.setStyleSheet(
            'QLabel{color:#6b7280; font-size:10pt; font-style:italic;}')

    def _draw_hover(self, cell, force=False):
        if not force and self._last_hover_cell == cell:
            return
        self._last_hover_cell = cell
        a, b = cell
        # 旧元素清掉
        self._clear_items(self._hover_items)
        if self._hover_text is not None:
            try:
                self.glw.scene().removeItem(self._hover_text)
            except Exception:
                pass
            self._hover_text = None
        ox, oy, pw, ph = self._viewport_scene_geom()   # 可见区：带跨度
        fx, fy, fw, fh = self._full_scene_geom()       # 完整图：数据映射
        # 行/列高亮线（场景层）；单元格场景矩形用映射两端点计算，方向无关
        x_s0 = self._map_data_to_scene_x(a, fx, fw)
        x_s1 = self._map_data_to_scene_x(a + 1, fx, fw)
        y_s0 = self._map_data_to_scene_y(b, fy, fh)
        y_s1 = self._map_data_to_scene_y(b + 1, fy, fh)
        x_cell, cell_w = min(x_s0, x_s1), abs(x_s1 - x_s0)
        y_cell, cell_h = min(y_s0, y_s1), abs(y_s1 - y_s0)
        # 行（横向，覆盖第 b 行的可见部分）
        path_row = QtGui.QPainterPath()
        path_row.addRect(QRectF(ox, y_cell, pw, cell_h))
        # 列
        path_col = QtGui.QPainterPath()
        path_col.addRect(QRectF(x_cell, oy, cell_w, ph))
        for path in (path_row, path_col):
            it = QGraphicsPathItem(path)
            it.setBrush(QtGui.QBrush(QtCore.Qt.NoBrush))          # 不填充：不遮挡热图像素
            it.setPen(QtGui.QPen(QtGui.QColor('#E05A3A'), 1.6))
            it.setZValue(6)
            self.glw.scene().addItem(it)
            self._hover_items.append(it)
        # 当前 cell 边框（更亮）
        cell_path = QtGui.QPainterPath()
        cell_path.addRect(QRectF(x_cell, y_cell, cell_w, cell_h))
        it = QGraphicsPathItem(cell_path)
        it.setBrush(QtGui.QColor(0, 0, 0, 0))  # 透明
        it.setPen(QtGui.QPen(QtGui.QColor('#1a1a1a'), 1.4))
        it.setZValue(7)
        self.glw.scene().addItem(it)
        self._hover_items.append(it)
        # 数据
        H = self._current_H()
        v = H[a, b] if 0 <= a < H.shape[0] and 0 <= b < H.shape[1] else 0
        a_mag = abs(v)
        arg = np.angle(v) / np.pi if a_mag > 1e-12 else 0.0
        # 物理含义
        mean_txt = '—'
        if self._labels and self._stride:
            pa, ia = divmod(a, self._stride)
            pb, ib = divmod(b, self._stride)
            la = (self._labels[ia] if ia < len(self._labels) else str(ia))
            lb = (self._labels[ib] if ib < len(self._labels) else str(ib))
            mean_txt = f'{lb}|n={pb - 1} → {la}|n={pa - 1}'
        self.hover_label.setStyleSheet(
            'QLabel{color:#2c3137; font-size:10pt; font-family:Consolas,monospace;}')
        # 两行：每行一个态（不省略——展示全面且不占地）
        lb_txt = (f'列 {mean_txt.split(" → ")[1]}'
                  if ' → ' in mean_txt else f'列 {mean_txt}')
        la_txt = (f'行 {mean_txt.split(" → ")[0]}   '
                  f'|H| = {a_mag:.4g} MHz   arg = {arg:+.2f}π')
        fm = self.hover_label.fontMetrics()
        avail = max(self.hover_label.width() - 16, 120)
        full = lb_txt + '\n' + la_txt
        self.hover_label.setText('\n'.join(
            fm.elidedText(t, Qt.ElideRight, avail) for t in (lb_txt, la_txt)))
        self.hover_label.setToolTip(full)
        self._highlight_band(a, b)

    # ---------- 暴露给外部使用：Top-N 表 ----------
    def top_matrix_elements(self, n=30) -> List[Tuple[int, int, float, float, str]]:
        """返回当前快照的 top-n 矩阵元（按 |H| 排序）。"""
        H = self._current_H()
        if H.size == 0:
            return []
        A = np.abs(H)
        ii, jj = np.nonzero(A > 1e-12)
        if ii.size == 0:
            return []
        order = np.argsort(-A[ii, jj])[:n]
        out = []
        for k in order:
            a, b = int(ii[k]), int(jj[k])
            mag = float(A[a, b])
            arg = float(np.angle(H[a, b]) / np.pi) if mag > 1e-12 else 0.0
            mean = '—'
            if self._labels and self._stride:
                pa, ia = divmod(a, self._stride)
                pb, ib = divmod(b, self._stride)
                la = (self._labels[ia] if ia < len(self._labels) else str(ia))
                lb = (self._labels[ib] if ib < len(self._labels) else str(ib))
                mean = f'{lb}|n={pb - 1} → {la}|n={pa - 1}'
            out.append((a, b, mag, arg, mean))
        return out


# ============================================================
# 色条（独立 QWidget）
# ============================================================

class _ColorBar(QWidget):
    """垂直色条：左侧渐变 + 右侧 5 档刻度文字。"""

    def __init__(self, min_label='0', max_label='max', lut=None, parent=None):
        super().__init__(parent)
        self._lut = lut
        self._vmin = 0.0
        self._vmax = 1.0
        self._unit = 'MHz'
        self.setMinimumHeight(180)
        self.setMinimumWidth(40)

    def set_range(self, vmin: float, vmax: float, unit: str = ''):
        self._vmin = float(vmin)
        self._vmax = float(vmax or 1.0)
        if unit:
            self._unit = unit
        self.update()

    def set_lut(self, lut):
        self._lut = lut
        self.update()

    def paintEvent(self, ev):
        from PyQt5.QtGui import QPainter, QLinearGradient, QImage
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing, False)
        p.setRenderHint(QPainter.TextAntialiasing, True)
        w, h = self.width(), self.height()
        # 渐变区
        grad_x = 6
        grad_w = 14
        grad_top = 6
        grad_bot = h - 6
        if self._lut is not None and grad_bot > grad_top:
            # 构造一张渐变图（竖向：上=vmax → 下=vmin）
            n = 256
            arr = np.zeros((n, grad_w, 3), dtype=np.ubyte)
            for i in range(n):
                # 顶部（y=0）= vmax → arr[0]；底部（y=n-1）= vmin → arr[-1]
                t = 1.0 - i / (n - 1)
                idx = int(round(t * (len(self._lut) - 1)))
                arr[i, :, :] = self._lut[idx]
            img = QtGui.QImage(arr.data, grad_w, n,
                               grad_w * 3, QtGui.QImage.Format_RGB888)
            # 拉伸到 grad_top..grad_bot
            p.drawImage(QtCore.QRect(grad_x, grad_top, grad_w,
                                     grad_bot - grad_top), img)
            # 边框
            p.setPen(QtGui.QPen(QtGui.QColor('#7a8088'), 0.8))
            p.setBrush(QtCore.Qt.NoBrush)
            p.drawRect(grad_x, grad_top, grad_w, grad_bot - grad_top)
        # 刻度文字（5 档：min, 0.25, 0.5, 0.75, max）
        p.setPen(QtGui.QColor('#2c3137'))
        fnt = QtGui.QFont('Microsoft YaHei', 7)
        p.setFont(fnt)
        for i, t in enumerate((1.0, 0.75, 0.5, 0.25, 0.0)):
            y = grad_top + (1.0 - t) * (grad_bot - grad_top)
            val = self._vmin + t * (self._vmax - self._vmin)
            # 短刻度
            p.drawLine(grad_x + grad_w, int(y), grad_x + grad_w + 2, int(y))
            txt = f'{val:.3g}' if i in (0, 4) else f'{val:.2g}'
            p.drawText(grad_x + grad_w + 4, int(y) - 6, 26, 12,
                       Qt.AlignLeft | Qt.AlignVCenter, txt)
        # 单位
        p.setPen(QtGui.QColor('#6b7280'))
        p.drawText(grad_x, h - 14, grad_w + 30, 12,
                   Qt.AlignLeft | Qt.AlignVCenter, self._unit)
        p.end()
