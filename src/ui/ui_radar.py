# -*- coding: utf-8 -*-
"""两体相互作用耦合强度雷达图（左右布局：左参数栏，右热图）。

左侧为 R / θ / φ / J 四行参数输入与显示；
右侧为雷达图画布，显示耦合强度 |V_c| 的极坐标半圆热图。

- 回车提交任一参数时，会刷新对应的三参数并更新 J
- 若输入超出当前绘图范围，则重新绘制雷达图
- 选定位置时在图上画圈，不响应鼠标移动
- 再点击图像任意处，则清掉圈，恢复鼠标跟踪更新参数栏
"""
import numpy as np
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.patches import Circle

from ..core import pair_core

N_THETA = 181
N_R = 48
EPS = 1e-9


class CouplingRadarPlot(QWidget):
    """两体耦合强度雷达图：左参数栏 + 右热图。"""

    def __init__(self, page, parent=None):
        super().__init__(parent)
        self.page = page
        self._trans1 = None
        self._trans2 = None
        self._last_key = None
        self._last_data = None
        self._mouse_xy = None
        self._manual_point = None
        self._selection_active = False
        self._marker_patch = None

        self._plot_r_min = 0.1
        self._plot_r_max = 30.0
        self._plot_phi = 0.0

        CONTROL_W = 200
        FIG_W = 650
        FIG_H = 325

        self.setFixedSize(CONTROL_W + FIG_W, FIG_H)

        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(6)

        ctrl_widget = QWidget()
        ctrl_widget.setFixedWidth(CONTROL_W)
        ctrl_widget.setFixedHeight(FIG_H)
        ctrl = QVBoxLayout(ctrl_widget)
        ctrl.setContentsMargins(4, 4, 4, 4)
        ctrl.setSpacing(6)

        self.r_edit = self._make_param_row(ctrl, 'R', '10.0', 'μm')
        self.theta_edit = self._make_param_row(ctrl, 'θ', '90.0', '°')
        self.phi_edit = self._make_param_row(ctrl, 'φ', '0.0', '°')
        self.j_edit = self._make_param_row(ctrl, 'J', '0.0', 'MHz')
        self.j_edit.setReadOnly(True)
        self.j_edit.setStyleSheet('background: #f7f7f7; color: #0b5394;')

        self.r_edit.returnPressed.connect(self._commit_manual_input)
        self.theta_edit.returnPressed.connect(self._commit_manual_input)
        self.phi_edit.returnPressed.connect(self._commit_manual_input)

        lay.addWidget(ctrl_widget, 0)

        self.fig = Figure(figsize=(FIG_W / 100.0, FIG_H / 100.0), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.fig.subplots_adjust(left=0.005, right=0.995, top=0.995, bottom=0.005)
        self.canvas = FigureCanvas(self.fig)
        self.canvas.setFixedSize(FIG_W, FIG_H)
        self.canvas.mpl_connect('motion_notify_event', self._on_mouse_move)
        self.canvas.mpl_connect('button_press_event', self._on_canvas_click)
        lay.addWidget(self.canvas, 0)

        self._cb = None
        self.draw_placeholder('点击/回车更新雷达参数')
        self._sync_value_fields(10.0, 90.0, 0.0, 0.0)

    def _make_param_row(self, parent_layout, text, value, unit):
        row = QHBoxLayout()
        label = QLabel(text)
        label.setFixedWidth(20)
        line = QLineEdit(str(value))
        line.setFixedWidth(100)
        line.setToolTip(f'{text} {unit}')
        unit_label = QLabel(unit)
        unit_label.setFixedWidth(50)
        row.addWidget(label)
        row.addWidget(line)
        row.addWidget(unit_label)
        row.addStretch(1)
        parent_layout.addLayout(row)
        return line

    def _sync_value_fields(self, r, theta, phi, j):
        self.r_edit.setText(f'{r:.3g}')
        self.theta_edit.setText(f'{theta:.3g}')
        self.phi_edit.setText(f'{phi:.3g}')
        self.j_edit.setText(f'{j:.3g}')

    def _read_value(self, line, default):
        try:
            return float(line.text())
        except ValueError:
            return default

    def set_transition(self, which, src_key, tgt_key):
        """谱表/峰点击联动：更新当前用于雷达图的跃迁对。"""
        if src_key is None or tgt_key is None:
            return
        src_key = str(src_key).strip()
        tgt_key = str(tgt_key).strip()
        if not src_key or not tgt_key:
            return
        if which == 'off':
            self._trans1 = (src_key, tgt_key)
        else:
            self._trans2 = (src_key, tgt_key)
        self.refresh()

    def _radial_donut_range(self, r_min, r_max):
        """自动定范围时，保留近原点的“空洞区”，避免极小 R 让图中心全部偏黑。"""
        r_min = float(r_min)
        r_max = float(r_max)
        if not np.isfinite(r_min) or not np.isfinite(r_max) or r_max <= 0:
            return r_min, r_max
        cut = max(0.5, 0.08 * r_max)
        if r_min < cut:
            r_min = cut
        return r_min, r_max

    def _compute_j(self, r, theta):
        if self._last_data is None:
            return 0.0
        R, theta_grid, Z, _ = self._last_data
        ir = int(np.argmin(np.abs(R - r)))
        it = int(np.argmin(np.abs(theta_grid - theta)))
        return float(Z[ir, it])

    def _clear_marker(self):
        if self._marker_patch is None:
            return
        try:
            if self._marker_patch.axes is not None:
                self._marker_patch.remove()
        except (AttributeError, ValueError, NotImplementedError):
            pass
        self._marker_patch = None
        self.fig.canvas.draw_idle()

    def _draw_selection_ring(self, r, theta):
        if self._last_data is None:
            return
        R, _, _, _ = self._last_data
        rr = float(np.clip(r, float(np.min(R)), float(np.max(R))))
        th_deg = float(np.clip(theta, 0.0, 180.0))
        x = rr * np.cos(np.deg2rad(th_deg))
        y = rr * np.sin(np.deg2rad(th_deg))
        self._clear_marker()
        self._marker_patch = Circle(
            (x, y), radius=max(rr * 0.04, 0.8),
            fill=False, edgecolor='white', linewidth=1.5, linestyle='--')
        self.ax.add_patch(self._marker_patch)
        self.fig.canvas.draw_idle()

    def _commit_manual_input(self):
        if self._last_data is None:
            self.refresh()
            if self._last_data is None:
                return

        r = self._read_value(self.r_edit, self._plot_r_min)
        theta = self._read_value(self.theta_edit, 90.0)
        phi = self._read_value(self.phi_edit, self._plot_phi)

        if r <= 0:
            r = self._plot_r_min
        theta = float(np.clip(theta, 0.0, 180.0))
        phi = float(np.mod(phi, 360.0))

        if r > self._plot_r_max * 1.15 or r < self._plot_r_min * 0.85:
            self._plot_r_min = max(0.1, r * 0.7)
            self._plot_r_max = max(r * 1.3, self._plot_r_max * 1.2)
            self.refresh()

        self._plot_phi = phi
        self._manual_point = (r, theta, phi)
        self._selection_active = True
        self._draw_selection_ring(r, theta)
        self._sync_value_fields(r, theta, phi, self._compute_j(r, theta))

    def _on_canvas_click(self, event):
        if self._selection_active:
            self._selection_active = False
            self._manual_point = None
            self._clear_marker()

    def _on_mouse_move(self, event):
        if self._selection_active or event.inaxes is not self.ax or self._last_data is None:
            return
        self._mouse_xy = (event.xdata, event.ydata)
        self._update_pos_label()

    def _update_pos_label(self):
        if self._last_data is None or self._mouse_xy is None:
            return
        R, theta, Z, phi = self._last_data
        x, y = self._mouse_xy
        r = float(np.hypot(x, y))
        th_deg = float(np.rad2deg(np.arctan2(y, x)))
        if th_deg < 0:
            th_deg += 180.0
        if r <= 0 or r > float(np.max(R)):
            return
        it = int(np.argmin(np.abs(theta - th_deg)))
        ir = int(np.argmin(np.abs(R - r)))
        J = float(Z[ir, it])
        self._sync_value_fields(r, th_deg, phi, J)

    def refresh(self):
        groups = list(getattr(self.page, 'groups_store', None).groups if getattr(self.page, 'groups_store', None) is not None else [])
        if len(groups) < 1:
            self.draw_placeholder('请先添加能级组')
            return

        B = float(getattr(getattr(self.page, 'field_state', None), 'B_gauss', 0.0) or 0.0)
        phi = float(self.phi_edit.text() if self.phi_edit.text() else str(self._plot_phi))
        self._plot_phi = phi

        try:
            self._plot_r_min, self._plot_r_max = self._radial_donut_range(self._plot_r_min, self._plot_r_max)
            R_grid = np.geomspace(max(self._plot_r_min, 1e-3), self._plot_r_max, N_R)
            theta_grid = np.linspace(0.0, 180.0, N_THETA)
            res = pair_core.coupling_strength_map(
                self.page.cache,
                groups,
                R_grid,
                theta_grid,
                phi_deg=phi,
                trans1=self._trans1,
                trans2=self._trans2,
                B_gauss=B,
            )
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.draw_placeholder(f'计算失败: {e}')
            return

        if res is None:
            self.draw_placeholder('跃迁态不在能级基组中')
            return

        R, theta, Z = res
        self._last_data = (R, theta, Z, phi)
        self._draw(R, theta, Z, phi)
        if self._selection_active and self._manual_point is not None:
            r, th, ph = self._manual_point
            self._draw_selection_ring(r, th)
            self._sync_value_fields(r, th, ph, self._compute_j(r, th))

    def _draw(self, R, theta, Z, phi):
        ax = self.ax
        self._clear_marker()
        ax.clear()
        TH, RR = np.meshgrid(np.deg2rad(theta), R)
        X = RR * np.cos(TH)
        Y = RR * np.sin(TH)
        L = np.log10(np.maximum(Z, EPS))
        vmax = float(np.max(L))
        pc = ax.pcolormesh(X, Y, L, cmap='inferno',
                           vmin=vmax - 8.0, vmax=vmax, shading='gouraud',
                           clip_on=True)
        pc.set_clip_path(ax.patch)
        Rmax = float(np.max(R))
        pad = 0.08 * Rmax
        ax.set_xlim(-Rmax - pad, Rmax + pad)
        ax.set_ylim(-0.5 * pad, Rmax + 0.5 * pad)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)
        for sp in ax.spines.values():
            sp.set_visible(False)
        if self._selection_active and self._manual_point is not None:
            self._draw_selection_ring(*self._manual_point[:2])
        self.fig.canvas.draw_idle()

    def draw_placeholder(self, msg=''):
        ax = self.ax
        self._clear_marker()
        ax.clear()
        if self._cb is not None:
            self._cb.remove()
            self._cb = None
        if msg:
            ax.text(0.5, 0.5, msg, ha='center', va='center',
                    transform=ax.transAxes, fontsize=9, color='0.5')
        self._last_data = None
        self.fig.canvas.draw_idle()
