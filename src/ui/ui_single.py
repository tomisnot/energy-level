# -*- coding: utf-8 -*-
"""旧版单体画布：能级图 + off/on 镜像对比频谱。

（由 能级.py 的 EnergyPlotPanel 独立而来，纯移动，逻辑零改动。）

只负责显示与交互，直接渲染 core 返回的能级行 / 谱线数据，不做任何二次换算。
"""

import numpy as np
from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QStackedWidget)
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.lines import Line2D

from ..core import core

# 每个能级组按添加顺序绑定一个固定颜色（tab10 + tab20 色板，20 色）
COLOR_CYCLE = [
    '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
    '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
    '#aec7e8', '#ffbb78', '#98df8a', '#ff9896', '#c5b0d5',
    '#c49c94', '#f7b6d2', '#c7c7c7', '#dbdb8d', '#9edae5',
]


class EnergyPlotPanel(QWidget):
    """能级图 + 镜像对比频谱（一块画布）。

    频谱为镜像显示：off 谱向上、on 谱向下，共用一条 x 轴（频率），
    两根谱线根部在 y=0 对齐——共振的线上下对齐、非共振错开可见。

    交互：点击频谱线 → 高亮该峰、在能级图上画出该跃迁对的源↔目标
    子能级连线，并发出 peakPicked 信号（供峰表滚动定位）。
    """

    # 点击频谱峰后发出（which='off'/'on'，idx=谱线下标，-1=清除）
    peakPicked = pyqtSignal(str, int)
    # 选中峰频率变化后发出（off_freq_ghz / on_freq_ghz，None=该侧未选中）
    peakFreqsChanged = pyqtSignal(object, object)

    def __init__(self, parent=None):
        super().__init__(parent)
        # 能级图（占主导地位，保持原有比例）——单图模式（默认）
        self.level_fig = Figure(figsize=(7, 5), dpi=100)
        self.ax = self.level_fig.add_subplot(111)
        self.level_canvas = FigureCanvas(self.level_fig)
        # 双图模式：非寻址光（off）/ 寻址光（on）各一张，并排
        self.level_off_fig = Figure(figsize=(7, 5), dpi=100)
        self.ax_off = self.level_off_fig.add_subplot(111)
        self.level_off_canvas = FigureCanvas(self.level_off_fig)
        self.level_on_fig = Figure(figsize=(7, 5), dpi=100)
        self.ax_on = self.level_on_fig.add_subplot(111)
        self.level_on_canvas = FigureCanvas(self.level_on_fig)
        dual_host = QWidget()
        dual_lay = QHBoxLayout(dual_host)
        dual_lay.setContentsMargins(0, 0, 0, 0)
        dual_lay.setSpacing(1)
        dual_lay.addWidget(self.level_off_canvas, 1)
        dual_lay.addWidget(self.level_on_canvas, 1)
        # 能级图切换容器：页 0 = 单图，页 1 = off/on 双图
        self.level_stack = QStackedWidget()
        self.level_stack.addWidget(self.level_canvas)
        self.level_stack.addWidget(dual_host)
        self._level_mode = 'single'
        # 镜像对比频谱（一块画布）：off 谱向上、on 谱向下，共用一条 x 轴。
        # 两根谱线根部在 y=0（x 轴）对齐，共振的线上下对齐、非共振错开。
        self.spec_mirror_fig = Figure(figsize=(7, 2.2), dpi=100)
        self.ax_spec_mirror = self.spec_mirror_fig.add_subplot(111)
        self.spec_mirror_fig.subplots_adjust(left=0.04, right=0.98, top=0.97, bottom=0.16)
        self.spec_mirror_canvas = FigureCanvas(self.spec_mirror_fig)

        # 绝对紧凑布局：能级图占主导，镜像频谱占一块
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(1)
        lay.addWidget(self.level_stack, 5)
        lay.addWidget(self.spec_mirror_canvas, 2)

        # 镜像频谱点击：先按 y 上下判断 off/on，再按点到竖线的 display 距离定位
        self.spec_mirror_canvas.mpl_connect('button_press_event', self._on_spec_click)

        # 频谱横轴手动范围（GHz）；None 表示自动
        self._spec_range = None
        # 谱峰高亮下标（-1=无）与参考频率（GHz，None=不画竖线）
        self._hl_off = -1
        self._hl_on = -1
        self._hl_off_key = None   # 选中谱线的跃迁对标识（参数变化后保持高亮）
        self._hl_on_key = None
        self._spec_ref = None
        # 频谱重绘签名与缓存（避免无关输入触发整块重绘）
        self._sig_off = None
        self._sig_on = None
        self._lines_off = []
        self._lines_on = []
        self._freqs_off = None
        self._freqs_on = None
        self._rows = []          # 最近一次能级行数据（画跃迁连线用）
        self._last_draw = None   # 最近一次 draw_levels 参数（重绘连线用）

    @staticmethod
    def _format_ac(ac):
        """把 AC-Stark 位移格式化为带单位字符串。"""
        a = abs(ac)
        if a >= 1000.0:
            return f'{ac / 1000:+.2f} GHz'
        if a >= 1.0:
            return f'{ac:+.1f} MHz'
        return f'{ac * 1000:+.0f} kHz'

    # ---------------- 能级图 ----------------

    def _draw_level_axes(self, cache, field_B, groups, address_cfg, relative,
                         visible, ac_kwargs=None):
        """把 core.build_level_rows 算出的能级行画到当前模式对应的画布。

        'single'：画到 self.ax（单图，含 off/on 两条跃迁连线）；
        'dual'  ：非寻址光（裸能级）+ 寻址光（移后）各一张并排。

        `ac_kwargs`（P1-7）：AC-Stark 精度 `{n_photon, sos_n_window}`，来自
        `FieldPanel.ac_kwargs()`（静态页唯一来源）。**只影响静态页**——动态基组
        由 runspec 走自己的路径，从不读这两个值（用户裁决 2026-09-22）。
        """
        rows = core.build_level_rows(cache, field_B, groups, address_cfg, visible,
                                     **(ac_kwargs or {}))
        if self._level_mode == 'dual':
            return self._draw_level_dual(rows, relative, visible)

        if not rows:
            self.ax.clear()
            self.ax.set_xlim(-1.5, 1.5)
            self.ax.set_ylim(-1.0, 1.0)
            self.ax.text(0, 0, 'Add level groups to begin.', ha='center', va='center')
            self._rows = []
            self.level_fig.tight_layout()
            self.level_canvas.draw_idle()
            return rows

        x_values, y, y0 = self._level_coords(rows, relative)
        self._rows = rows
        self._render_level_plot(
            self.ax, self.level_fig, rows, x_values, y, y0, relative, visible,
            title=None, use_shifted=True, connectors='both')
        return rows

    def _draw_level_dual(self, rows, relative, visible):
        """双图模式：off（裸能级，无箭头）+ on（移后能级，带箭头）各一张。"""
        if not rows:
            for ax, fig in ((self.ax_off, self.level_off_fig),
                            (self.ax_on, self.level_on_fig)):
                ax.clear()
                ax.set_xlim(-1.5, 1.5)
                ax.set_ylim(-1.0, 1.0)
                ax.text(0, 0, 'Add level groups to begin.', ha='center', va='center')
                fig.tight_layout()
            self.level_off_canvas.draw_idle()
            self.level_on_canvas.draw_idle()
            self._rows = []
            return rows
        x_values, y, y0 = self._level_coords(rows, relative)
        self._rows = rows
        self._render_level_plot(
            self.ax_off, self.level_off_fig, rows, x_values, y, y0, relative,
            visible, title='off: no address light',
            use_shifted=False, connectors='off')
        self._render_level_plot(
            self.ax_on, self.level_on_fig, rows, x_values, y, y0, relative,
            visible, title='on: with address light',
            use_shifted=True, connectors='on')
        return rows

    @staticmethod
    def _level_coords(rows, relative):
        """返回 (x_values, y, y0)：y=移后能级，y0=裸能级（相对可选）。"""
        x_values = np.array([r['x'] for r in rows])
        energies = np.array([r['E'] for r in rows])
        energies0 = np.array([r['E0'] for r in rows])
        if relative:
            ref = np.min(np.concatenate([energies, energies0]))
            y = energies - ref
            y0 = energies0 - ref
        else:
            y = energies
            y0 = energies0
        return x_values, y, y0

    def _render_level_plot(self, ax, fig, rows, x_values, y, y0, relative, visible,
                           title, use_shifted, connectors):
        """把能级行渲染到指定 ax。

        use_shifted=True：画移后能级（y）+ 箭头（受寻址光影响子能级）；
        use_shifted=False：只画裸能级（y0），无箭头（非寻址光视图）。
        connectors：'both'/'off'/'on'——画哪一侧选中谱峰的跃迁连线。
        """
        ax.clear()
        if use_shifted:
            yy, yy0 = y, y0
            y_all = np.concatenate([y, y0])
        else:
            yy, yy0 = y0, y0
            y_all = y0

        x_min = np.min(x_values) - 0.6
        x_max = np.max(x_values) + 0.6
        ax.set_xlim(x_min, x_max)
        span_all = np.max(y_all) - np.min(y_all)
        if span_all < 1e-12:
            span_all = 1.0
        margin = max(span_all * 0.08, 10.0)
        ax.set_ylim(np.min(y_all) - margin, np.max(y_all) + margin)
        ax.grid(True, alpha=0.3)
        # 不画标题/坐标轴标题（省空间，刻度数值已足够）；
        # 双图模式用左上角极小角标区分 off/on
        if title:
            ax.text(0.005, 0.985, title, transform=ax.transAxes,
                    fontsize=6.5, va='top', color='0.45')

        for r, yyv, yy0v in zip(rows, yy, yy0):
            if r['affected'] and use_shifted:
                ax.hlines(yy0v, r['x'] - 0.18, r['x'] + 0.18, color='0.5', lw=1.5, alpha=0.7)
                ax.annotate('', xy=(r['x'], yyv), xytext=(r['x'], yy0v),
                            arrowprops=dict(arrowstyle='->', color=r['color'], lw=1.6))
                ax.hlines(yyv, r['x'] - 0.22, r['x'] + 0.22, color=r['color'], lw=4)
                ax.plot([r['x']], [yyv], 'o', color=r['color'], ms=6, zorder=5)
                ax.text(r['x'] + 0.3, yyv, self._format_ac(r['ac']), fontsize=8,
                        color=r['color'], va='center')
            else:
                ax.hlines(yy0v, r['x'] - 0.18, r['x'] + 0.18, color=r['color'],
                          lw=2.5, alpha=0.8)

        unique_x = sorted(set(np.round(x_values, 3).tolist()))
        ax.set_xticks(unique_x)
        ax.set_xticklabels([f'{x:.1f}' for x in unique_x])

        legend_handles = []
        legend_labels = []
        seen = set()
        for group in visible:
            label = group['label']
            if label not in seen:
                seen.add(label)
                legend_handles.append(Line2D([], [], color=group.get('color', COLOR_CYCLE[0]), lw=2.5))
                legend_labels.append(label)
        if use_shifted and any(r['affected'] for r in rows):
            legend_handles.append(Line2D([], [], color='0.5', lw=1.5))
            legend_labels.append('before shift')
            legend_handles.append(Line2D([], [], color='0.4', lw=4, marker='o', ms=5))
            legend_labels.append('AC-affected sublevel')
        ax.legend(legend_handles, legend_labels, loc='upper left', fontsize=8)

        self._draw_transition_connectors(ax, rows, x_values, yy, yy0, connectors)
        fig.tight_layout()
        fig.canvas.draw_idle()

    # ---------------- 频谱图 ----------------

    def _draw_spectrum_mirror(self):
        """镜像对比绘制：一条 x 轴，off 谱向上、on 谱向下。

        y 用 ±log10(A/floor) 映射（隐藏刻度），两根谱线根部在 y=0（x 轴）
        对齐：共振的线 off/on 频率相同 → 上下对齐成一条直线；
        非共振的 on 线相对 off 错开 → 肉眼可见位移。
        """
        ax = self.ax_spec_mirror
        ax.clear()
        off, on = self._lines_off, self._lines_on
        allA = [ln[1] for ln in off] + [ln[1] for ln in on]
        if not allA:
            ax.text(0.5, 0.5, 'no transitions', ha='center', va='center',
                    transform=ax.transAxes, fontsize=8)
            ax.set_xlabel('Frequency (GHz)', fontsize=8)
            ax.set_yticks([])
            ax.tick_params(axis='y', left=False)
            self._freqs_off = None
            self._freqs_on = None
            return
        maxA = max(allA)
        floor = maxA * 1e-6
        span = float(np.log10(maxA / floor)) + 0.1
        ax.set_ylim(-span, span)
        ax.axhline(0, color='0.25', lw=0.8)
        freqs_off, freqs_on = [], []
        # off 谱向上（+）、on 谱向下（−）；两段绘制逻辑完全相同，合并为一处
        for which, lines, sign, hl_idx in (
                ('off', off, 1.0, self._hl_off), ('on', on, -1.0, self._hl_on)):
            freqs = freqs_off if which == 'off' else freqs_on
            for idx, ln in enumerate(lines):
                nu, A, la, lb, qa, qb, col, pol = ln
                nu_g = nu / 1e9
                y = sign * float(np.log10(max(A, floor) / floor))
                hl = (idx == hl_idx)
                c = 'red' if hl else col
                ax.vlines(nu_g, 0, y, color=c, lw=2.5 if hl else 1.2)
                ax.plot([nu_g], [y], 'o', color=c, ms=7 if hl else 3, zorder=5)
                # 选中峰频率不再标在图上，改由右侧面板显示（peakFreqsChanged 信号）
                freqs.append(nu_g)
        if self._spec_ref is not None:
            ax.axvline(self._spec_ref, color='0.3', ls='--', lw=1.2, alpha=0.9)
        ax.set_yticks([])
        ax.tick_params(axis='y', left=False)
        ax.set_xlabel('Frequency (GHz)', fontsize=8)
        ax.text(0.005, 0.94, 'no address', transform=ax.transAxes,
                fontsize=7, va='top', color='0.35')
        ax.text(0.005, 0.06, 'with address', transform=ax.transAxes,
                fontsize=7, va='bottom', color='0.35')
        self._freqs_off = np.array(freqs_off) if freqs_off else None
        self._freqs_on = np.array(freqs_on) if freqs_on else None

    def _auto_spectrum_bounds(self):
        """根据谱线数据自动计算横轴范围 (GHz)。"""
        freqs = [f for f in (self._freqs_off, self._freqs_on) if f is not None]
        if not freqs:
            return 0.0, 1.0
        allf = np.concatenate(freqs)
        lo, hi = float(np.min(allf)), float(np.max(allf))
        span = hi - lo
        if span < 1e-9:
            span = max(abs(hi) * 0.01, 1.0)
        return lo - 0.05 * span, hi + 0.05 * span

    def _sync_spectrum_xlims(self):
        """应用镜像频谱横轴范围：手动设置优先，否则自动。"""
        if self._spec_range is not None:
            lo, hi = self._spec_range
            if hi > lo:
                self.ax_spec_mirror.set_xlim(lo, hi)
                return
        auto_lo, auto_hi = self._auto_spectrum_bounds()
        self.ax_spec_mirror.set_xlim(auto_lo, auto_hi)

    def set_spectrum_range(self, lo, hi):
        """手动设置频谱横轴范围 (GHz)；lo/hi 为 None 表示该端自动。"""
        if lo is None and hi is None:
            self._spec_range = None
        else:
            auto_lo, auto_hi = self._auto_spectrum_bounds()
            self._spec_range = (auto_lo if lo is None else lo,
                                auto_hi if hi is None else hi)
        self._sync_spectrum_xlims()
        self.spec_mirror_canvas.draw_idle()

    def set_spectrum_highlight(self, which, idx):
        """峰表点击：高亮谱峰并在能级图画跃迁连线（idx=-1 清除）。"""
        self._apply_hl(which, idx)

    # ---------------- 谱峰拾取（点击定位跃迁对） ----------------

    def _on_spec_click(self, event):
        """点击镜像频谱：先按 y 上下判断 off/on，再按点到竖线的 display 距离定位。

        竖线根部在 y=0（x 轴），off 向上、on 向下；命中 ≤ 12 px，
        未命中（空白）则清除高亮与峰表选中。
        """
        if event.inaxes is not self.ax_spec_mirror:
            return
        # 数据坐标 y>0 → 上半（off），y<0 → 下半（on）
        _, y_data = self.ax_spec_mirror.transData.inverted().transform(
            (event.x, event.y))
        which = 'off' if y_data >= 0 else 'on'
        ax = self.ax_spec_mirror
        lines = self._lines_off if which == 'off' else self._lines_on
        sign = 1.0 if which == 'off' else -1.0
        idx = -1
        if lines:
            maxA = max(ln[1] for ln in lines)
            floor = maxA * 1e-6
            best, best_d, bestA = -1, float('inf'), -1.0
            for k, ln in enumerate(lines):
                nu_g = ln[0] / 1e9
                y = sign * float(np.log10(max(ln[1], floor) / floor))
                xd, yd = ax.transData.transform((nu_g, y))
                x0, y0 = ax.transData.transform((nu_g, 0.0))
                lo, hi = min(y0, yd), max(y0, yd)
                dx = abs(event.x - xd)
                if lo <= event.y <= hi:
                    dy = 0.0
                else:
                    dy = min(abs(event.y - lo), abs(event.y - hi))
                d = (dx * dx + dy * dy) ** 0.5
                if d <= 12.0:
                    # 同频重叠（d 几乎相等，如里德堡 mJ 对）时选 A 最大（主峰）
                    if (best < 0 or d < best_d - 1e-6
                            or (abs(d - best_d) <= 1e-6 and ln[1] > bestA)):
                        best, best_d, bestA = k, d, ln[1]
            if best >= 0:
                idx = best
        if idx >= 0:
            self._apply_hl(which, idx)
            self.peakPicked.emit(which, idx)
        else:
            self._apply_hl(which, -1)
            self.peakPicked.emit(which, -1)

    def _apply_hl(self, which, idx):
        """统一应用谱峰高亮：重绘频谱 + 重绘能级图（叠加跃迁连线）。

        同时记录选中线的跃迁对标识，参数变化谱线重算后用于保持高亮。
        """
        if which == 'off':
            self._hl_off = idx
            self._hl_off_key = self._line_key(self._lines_off, idx)
        else:
            self._hl_on = idx
            self._hl_on_key = self._line_key(self._lines_on, idx)
        self._redraw_spectra()
        # 能级图叠加连线需按当前状态重绘（ArcCache 缓存保证快速）
        if self._last_draw is not None:
            cache, field_B, groups, address_cfg, relative, visible, ac_kw = \
                self._last_draw
            self.draw_levels(cache, field_B, groups, address_cfg, relative,
                             visible, ac_kw)
        # 通知右侧面板显示选中峰频率 / 频率差
        self._emit_peak_freqs()

    def _emit_peak_freqs(self):
        """发出当前 off/on 选中峰频率（GHz），None 表示该侧未选中。"""
        off_f = None
        on_f = None
        if (self._hl_off >= 0 and self._freqs_off is not None
                and self._hl_off < len(self._freqs_off)):
            off_f = float(self._freqs_off[self._hl_off])
        if (self._hl_on >= 0 and self._freqs_on is not None
                and self._hl_on < len(self._freqs_on)):
            on_f = float(self._freqs_on[self._hl_on])
        self.peakFreqsChanged.emit(off_f, on_f)

    @staticmethod
    def _line_key(lines, idx):
        """谱线 idx 的跃迁对标识 (la, qa, lb, qb)；idx 无效返回 None。"""
        if idx < 0 or idx >= len(lines):
            return None
        nu, A, la, lb, qa, qb, col, pol = lines[idx]
        return (la, qa, lb, qb)

    def _find_line(self, lines, key):
        """按跃迁对标识在谱线里找 idx；不存在返回 -1。"""
        if key is None:
            return -1
        for i, ln in enumerate(lines):
            if (ln[2], ln[4], ln[3], ln[5]) == key:
                return i
        return -1

    def _remap_hl(self):
        """谱线重算后按跃迁对 key 重新映射高亮：峰仍在则保留，消失则清除。"""
        self._hl_off = self._find_line(self._lines_off, self._hl_off_key)
        self._hl_on = self._find_line(self._lines_on, self._hl_on_key)

    def _draw_transition_connectors(self, ax, rows, x_values, y, y0, which='both'):
        """在指定 ax 上画当前选中的跃迁对连线。

        off 谱用裸能级 y0、on 谱用移后能级 y（与频谱数据一致），
        连线中点上方标注跃迁频率（GHz）。
        which='both'：off（虚线）+ on（实线）都画（单图模式）；
        which='off'/'on'：只画对应一侧（双图模式）。
        """
        if not rows:
            return
        idx_of = {(r['label'], r['qstr']): i for i, r in enumerate(rows)}
        y_lo = min(float(np.min(y)), float(np.min(y0)))
        y_hi = max(float(np.max(y)), float(np.max(y0)))
        yspan = max(y_hi - y_lo, 1.0)
        pairs = []
        if which in ('off', 'both'):
            pairs.append(('off', y0))
        if which in ('on', 'both'):
            pairs.append(('on', y))
        for w, yy in pairs:
            lines = self._lines_off if w == 'off' else self._lines_on
            idx = self._hl_off if w == 'off' else self._hl_on
            if idx < 0 or idx >= len(lines):
                continue
            nu, A, la, lb, qa, qb, col, pol = lines[idx]
            i1 = idx_of.get((la, qa))
            i2 = idx_of.get((lb, qb))
            if i1 is None or i2 is None:
                continue
            x1, x2 = float(x_values[i1]), float(x_values[i2])
            y1, y2 = float(yy[i1]), float(yy[i2])
            ax.plot([x1, x2], [y1, y2], color='red', lw=1.6,
                    ls='-' if w == 'on' else '--', zorder=6)
            ax.plot([x1, x2], [y1, y2], 'o', color='red', ms=4, zorder=6)
            ax.text((x1 + x2) / 2, max(y1, y2) + 0.04 * yspan,
                    f'{nu / 1e9:.4f} GHz', ha='center', va='bottom',
                    fontsize=8, color='red')

    def set_spectrum_ref(self, freq_ghz):
        """设置频谱参考频率 (GHz)；None 表示不画竖线。"""
        self._spec_ref = freq_ghz
        self._redraw_spectra()

    def set_level_mode(self, mode):
        """切换能级图模式：'single'（单图）/ 'dual'（off/on 双图并排）。"""
        self._level_mode = 'dual' if mode == 'dual' else 'single'
        self.level_stack.setCurrentIndex(1 if self._level_mode == 'dual' else 0)
        if self._last_draw is not None:
            cache, field_B, groups, address_cfg, relative, visible, ac_kw = \
                self._last_draw
            self.draw_levels(cache, field_B, groups, address_cfg, relative,
                             visible, ac_kw)
        self._emit_peak_freqs()

    def _redraw_spectra(self):
        """用缓存谱线重绘镜像频谱（不重算物理量）。"""
        self._draw_spectrum_mirror()
        self.spec_mirror_canvas.draw_idle()
        self._sync_spectrum_xlims()

    def draw_levels(self, cache, field_B, groups, address_cfg, relative, visible,
                    ac_kwargs=None):
        """更新画布：能级图 + 镜像对比频谱。返回能级行数据。

        能级图总是重绘；镜像频谱的 off/on 谱线各自独立签名缓存：
        无寻址光谱仅在裸能级输入变化时重算，有寻址光谱随光参数变化。
        谱线一律由 core.spectrum_lines 计算，画布只负责绘制。

        `ac_kwargs`：AC-Stark 精度（`{n_photon, sos_n_window}`；None=引擎默认）。
        """
        # 记住本次绘制参数：点击谱峰高亮时需要重绘能级图（叠加跃迁连线）
        self._last_draw = (cache, field_B, groups, address_cfg, relative, visible,
                           ac_kwargs)
        rows = self._draw_level_axes(cache, field_B, groups, address_cfg,
                                     relative, visible, ac_kwargs)
        enabled = address_cfg.get('enabled', False)

        if not rows:
            self._sig_off = None
            self._sig_on = None
            self._lines_off = []
            self._lines_on = []
            self._freqs_off = None
            self._freqs_on = None
            self._hl_off = -1
            self._hl_on = -1
            self._hl_off_key = None
            self._hl_on_key = None
            self._draw_spectrum_mirror()
            self.spec_mirror_canvas.draw_idle()
            return rows

        gkey = tuple((g['n'], g['l'], g['j']) for g in groups)
        vkey = tuple((g['n'], g['l'], g['j']) for g in visible)
        sig_off = (field_B, gkey, vkey)
        ell = address_cfg.get('elliptical')
        ell_key = (ell.get('alpha_deg', 0.0), ell.get('theta_deg', 0.0),
                   ell.get('eps_deg', 0.0)) if ell else None
        sig_on = (sig_off, address_cfg['intensity'], address_cfg['frequency_hz'],
                  address_cfg['polarization'], ell_key, enabled)

        # 谱线数据：无寻址光与裸能级输入相关，有寻址光随光参数变化；
        # 任一变化则重算对应谱线并重绘镜像画布
        redraw = False
        if sig_off != self._sig_off:
            self._lines_off = core.spectrum_lines(cache, rows, ac_on=False)
            self._sig_off = sig_off
            redraw = True
        if sig_on != self._sig_on:
            # on 谱始终显示全部偏振的跃迁线（寻址光只负责移位，不筛谱线）；
            # AC 位移本身已按寻址光偏振计算，sig_on 缓存含偏振/椭偏键
            self._lines_on = core.spectrum_lines(cache, rows, ac_on=True)
            self._sig_on = sig_on
            redraw = True
        if redraw:
            # 谱线重算后按跃迁对 key 重新映射高亮（峰仍在则保留，消失则清除）
            self._remap_hl()
            self._draw_spectrum_mirror()
            self.spec_mirror_canvas.draw_idle()

        self._sync_spectrum_xlims()
        # 谱线重算后选中峰频率可能变化，同步右侧面板
        self._emit_peak_freqs()
        return rows
