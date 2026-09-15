# -*- coding: utf-8 -*-
"""pyqtgraph 分级标注自适应热图组件（从真实项目提炼，可直接复用）。

特性：
- ImageItem 呈现矩阵：A[a, b] → a=纵轴向上、b=横轴，方格不变形；
- 滚轮缩放 + 左键平移（PanMode），GPU blit——几千维也即时响应；
- **分级标注**（核心）：按视野深度自动切换标注层级，逐层钻取不混排：
    粗（span ≥ 3×组宽）  ：流形组名（如 P3/2，居中于组）+ 扇区名 n=±1
    中（组宽 ≤ span < 3×组宽）：扇区名 + 抽稀短标签
    细（span < 组宽 或 stride=0）  ：逐维全标注（自动 ≤10 个防重叠）
  组边界画分隔线，标签与矩阵行列通过网格对齐；
- x 轴刻度文字旋转 −90°（自下而上阅读），锚定轴线向下悬挂，
  不侵入绘图区；y 轴横排多行。

依赖：pyqtgraph（pip install pyqtgraph），PyQt5。
用法见文件尾部 demo（直接运行可看效果）。
"""
import re

import numpy as np

from PyQt5.QtCore import Qt, QRectF
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QPushButton)
import pyqtgraph as pg


def derive_groups(labels):
    """从标签列表自动推导连续流形组 [(start, end, name), ...]（end 开区间）。

    规则：取标签开头的 "数字+字母" 主干（如 '60S'、'(6P' 中的 '6P'），
    连续相同者合并为一段；无法提取时退化为每标签一段。
    """
    groups = []
    for i, lab in enumerate(labels):
        m = re.match(r'\(?\s*(\d+\s*[A-Za-z])', str(lab))
        key = m.group(1).replace(' ', '') if m else str(lab)[:6]
        if groups and groups[-1][2] == key and groups[-1][1] == i:
            groups[-1][1] = i + 1
        else:
            groups.append([i, i + 1, key])
    return [(s, e, n) for s, e, n in groups]


def wrap_text(s, width=20):
    """定宽折行（pyqtgraph 隐形坑：文本 rect 超轴宽会被整组丢弃）。"""
    return '\n'.join(s[j:j + width] for j in range(0, len(s), width))


class _RotatedAxisItem(pg.AxisItem):
    """x 轴刻度文字旋转 −90°：竖排、自下而上阅读。

    锚点 = 文本块顶端贴轴线（translate 到 (cx, rect.top + 文本长/2) 后旋转），
    文本自轴线向下悬挂——若以矩形中心旋转，多行文本上半截会转入绘图区。
    """

    def boundingRect(self):
        """扩容：容纳悬挂的旋转竖排文字（pg 的 contains 检查会静默丢弃
        超出 bounding 的文字——轴高 85 装不下 tickStop 下方的文本区）。"""
        br = super().boundingRect()
        return br.adjusted(-15, -5, 15, 60)

    def drawPicture(self, p, axisSpec, tickSpecs, textSpecs):
        p.setRenderHint(p.RenderHint.Antialiasing, False)
        p.setRenderHint(p.RenderHint.TextAntialiasing, True)
        pen, p1, p2 = axisSpec
        p.setPen(pen)
        p.drawLine(p1, p2)
        for pen, p1, p2 in tickSpecs:
            p.setPen(pen)
            p.drawLine(p1, p2)
        for rect, flags, text in textSpecs:
            br = p.boundingRect(QRectF(0, 0, 100, 100),
                                QtCore.Qt.AlignmentFlag.AlignCenter, text)
            bw, bh = br.width(), br.height()
            cx = rect.center().x()
            y_top = rect.top()
            p.save()
            p.translate(cx, y_top + bw / 2.0)
            p.rotate(-90)
            p.drawText(QRectF(-bw / 2, -bh / 2, bw, bh),
                       QtCore.Qt.AlignmentFlag.AlignCenter, text)
            p.restore()


class _PlainAxisItem(pg.AxisItem):
    """左轴：显式绘制刻度文字（深色、不受 pg 内部 clip/pen 影响）。

    （pg 默认 drawPicture 的 setPen(textPen)/setClipRect(bounding) 组合在
    部分离屏/白底场景下会把文字画成不可见——自己画最可靠。）
    """

    def boundingRect(self):
        """扩容：左侧容纳多行物理标签（同 contains 静默丢弃问题）。"""
        br = super().boundingRect()
        return br.adjusted(-45, -15, 15, 15)

    def drawPicture(self, p, axisSpec, tickSpecs, textSpecs):
        p.setRenderHint(p.RenderHint.Antialiasing, False)
        p.setRenderHint(p.RenderHint.TextAntialiasing, True)
        pen, p1, p2 = axisSpec
        p.setPen(pen)
        p.drawLine(p1, p2)
        for pen, p1, p2 in tickSpecs:
            p.setPen(pen)
            p.drawLine(p1, p2)
        p.setPen(pg.mkPen('#2c3137'))
        if self.style.get('tickFont') is not None:
            p.setFont(self.style['tickFont'])
        for rect, flags, text in textSpecs:
            p.save()
            c = rect.center()
            p.translate(c.x(), c.y())
            p.drawText(QRectF(-rect.width() / 2, -rect.height() / 2,
                              rect.width(), rect.height()),
                       QtCore.Qt.AlignmentFlag.AlignCenter, text)
            p.restore()


class AdaptiveHeatmap(QWidget):
    """分级标注自适应热图。

    set_data(A, label_fn, dim, detail_fn) 后：
      A[a, b] → a=纵轴向上、b=横轴；
      label_fn(i)=短标签（大视野抽稀用）；
      detail_fn(i)=逐维全标注多行文本（细视野用，自动 ≤max_ticks 个）；
    set_label_meta(labels, stride, groups=None) 启用分级标注
      （groups=None 时自动从 labels 推导连续流形组）。

    交互：滚轮缩放、左键平移、⟲ 还原视图。
    """

    def __init__(self, parent=None, detail_span=30, max_ticks=10):
        super().__init__(parent)
        self.detail_span = detail_span
        self.max_ticks = max_ticks
        self._dim = 0
        self._labels = None
        self._stride = 0
        self._groups = None
        self._detail_fn = None
        self._label_fn = None
        self._sector_lines = []

        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        self.glw = pg.GraphicsLayoutWidget()
        self.glw.setBackground('w')
        self.glw.ci.setContentsMargins(90, 10, 40, 76)   # 左/上/右/下留白——
        # y 标签右对齐伸向左侧、x 标签+色带挂在下方，都落在这片预留区
        self.plot = self.glw.addPlot(row=0, col=0)
        self.plot.setAspectLocked(True)
        self.plot.showGrid(x=True, y=True, alpha=0.15)
        _font = QtGui.QFont('Microsoft YaHei', 7)   # 默认 ~10pt 会互相穿插
        self.vb = self.plot.getViewBox()
        self.vb.setMouseMode(self.vb.PanMode)
        self.vb.setMenuEnabled(False)
        self.img = pg.ImageItem(axisOrder='row-major')
        self.plot.addItem(self.img)
        self.cmap = pg.colormap.get('viridis')
        self.img.setLookupTable(self.cmap.getLookupTable(0.0, 1.0, 256))
        self.ax_b = _RotatedAxisItem(orientation='bottom')  # 竖排自下而上
        self.ax_l = _PlainAxisItem(orientation='left')      # 横排自左向右
        self.plot.setAxisItems({'bottom': self.ax_b, 'left': self.ax_l})
        self.ax_b.setHeight(100)
        self.ax_l.setWidth(190)
        for ax in (self.ax_b, self.ax_l):
            # 刻度文字走 pg picture 链路（分级标注 setTicks）；
            # ⚠ 勿设 showValues=False——pg 会在 generateDrawSpecs 开头
            # 直接丢弃全部文字（曾致标注整体消失的元凶）
            ax.setStyle(tickFont=_font)
        self.vb.sigXRangeChanged.connect(lambda *_: self._update_all())
        self.vb.sigYRangeChanged.connect(lambda *_: self._update_all())
        lay.addWidget(self.glw, 1)

        row = QHBoxLayout()
        btn = QPushButton('⟲ 还原视图')
        btn.clicked.connect(self.reset_view)
        row.addWidget(btn)
        row.addStretch(1)
        lay.addLayout(row)

    # ---------- 数据 ----------
    def set_data(self, A, label_fn, dim, detail_fn=None):
        keep = self.vb.viewRange() if self._dim else None
        self._dim = dim
        self._label_fn = label_fn
        self._detail_fn = detail_fn or (lambda i: self._label_fn(i))
        self.img.setImage(A.T, autoLevels=False)
        vmax = float(A.max()) if A.size else 1.0
        self.img.setLevels([0.0, vmax or 1.0])
        self.img.setRect(QRectF(0, 0, dim, dim))
        if keep is not None:
            (kx0, kx1), (ky0, ky1) = keep
            kx0 = min(max(kx0, 0), dim); kx1 = min(max(kx1, 0), dim)
            ky0 = min(max(ky0, 0), dim); ky1 = min(max(ky1, 0), dim)
            if kx1 > kx0 and ky1 > ky0:
                self.vb.setRange(xRange=(kx0, kx1), yRange=(ky0, ky1),
                                 padding=0)
        self._update_all()

    def set_label_meta(self, labels, stride, groups=None):
        """启用分级标注：labels[i]=物理名；stride=扇区大小；
        groups=[(start, end, name), ...]（phys 索引段，None=自动推导）。"""
        self._labels = labels
        self._stride = stride
        self._groups = groups if groups is not None else derive_groups(labels)

    def reset_view(self):
        if self._dim:
            self.vb.setRange(xRange=(0, self._dim), yRange=(0, self._dim),
                             padding=0)

    # ---------- 内部：装饰（扇区分隔线 + 分组色带） ----------
    def _viewport_scene_geom(self):
        """视口矩形 → 场景坐标 (ox, oy, pw, ph)。"""
        vp = self.vb.boundingRect()
        origin = self.plot.mapToScene(vp.topLeft())
        corner = self.plot.mapToScene(vp.bottomRight())
        size = corner - origin
        return origin.x(), origin.y(), abs(size.x()), abs(size.y())

    def _update_decorations(self):
        for ln in self._sector_lines:
            try:
                self.glw.scene().removeItem(ln)
            except Exception:
                pass
        self._sector_lines = []
        stride = self._stride
        if not stride or not self._dim or self._dim % stride:
            return
        P = self._dim // stride
        ox, oy, pw, ph = self._viewport_scene_geom()
        span_x = max(pw, 1e-12)

        # 扇区分隔线（白虚线，画在热图上）
        for p in range(1, P):
            pos = p * stride - 0.5
            for angle in (90, 0):
                ln = pg.InfiniteLine(pos=pos, angle=angle,
                                     pen=pg.mkPen('#ffffff', width=1,
                                                  style=Qt.DashLine))
                self.plot.addItem(ln)
                self._sector_lines.append(ln)

        # 分组色带（轴外侧：每流形组一段浅色——文字放不下时靠色带辨组）
        band_colors = ['#DEEBF7', '#FDEBD3', '#E2F0D9', '#FBE2E5', '#EDE7F6']
        if not self._groups:
            return
        span_y = max(ph, 1e-12)
        y1v, y0v = self._dim, 0.0          # 数据 y 范围（0..dim）
        pc = P // 2                         # 组段取中央扇区（结构周期重复）
        for gi, (s, e, _name) in enumerate(self._groups):
            color = band_colors[gi % len(band_colors)]
            # y 轴带
            ys = oy + (y1v - (pc * stride + e)) / (y1v - y0v) * ph
            ye = oy + (y1v - (pc * stride + s)) / (y1v - y0v) * ph
            rect_y = QtWidgets.QGraphicsRectItem(ox - 40, ys, 24,
                                                 max(abs(ye - ys), 2))
            rect_y.setBrush(QtGui.QBrush(QtGui.QColor(color)))
            rect_y.setPen(pg.mkPen(None))
            self.glw.scene().addItem(rect_y)
            self._sector_lines.append(rect_y)
            # x 轴带
            xs = ox + (pc * stride + s) / self._dim * pw
            xe = ox + (pc * stride + e) / self._dim * pw
            rect_x = QtWidgets.QGraphicsRectItem(xs, oy + ph + 30,
                                                 max(xe - xs, 2), 20)
            rect_x.setBrush(QtGui.QBrush(QtGui.QColor(color)))
            rect_x.setPen(pg.mkPen(None))
            self.glw.scene().addItem(rect_x)
            self._sector_lines.append(rect_x)

    # ---------- 分级标注 ----------
    def _update_all(self):
        """分级 = 随缩放切换标注层级（每档单级，不同屏混排——混排必重叠）：
        粗档扇区名+组名色带；中档抽稀短标签（|n= 随标签携带）；
        细档逐维全标注。扇区分隔线+分组色带恒显。"""
        self._update_decorations()
        (x0, x1), (y0, y1) = self.vb.viewRange()
        vp = self.vb.boundingRect()
        origin = self.plot.mapToScene(vp.topLeft())
        size = self.plot.mapToScene(vp.bottomRight()) - \
            self.plot.mapToScene(vp.topLeft())
        self.ax_b.setTicks(self._tick_list(x0, x1, True, abs(size.x())))
        self.ax_l.setTicks(self._tick_list(y0, y1, False, abs(size.y())))

    def _tick_list(self, lo, hi, is_x, axis_len_px=700):
        """分级：粗=扇区名(+组名若放得下)；中=扇区名+抽稀短标签；细=逐维全标注。

        阈值相对**矩阵总维度**（教训：3×stride 当粗阈值会让 stride 小时
        中等/细档几乎不可达）。
        "放得下才显示"：文字像素宽(组名≈45px/扇区名≈28px)超过该标注在轴上
        占据的像素宽度时静默跳过——组粒度小于像素最小组距时组名自然缺席，
        组身份由色带+边界线表达（表格可查名）。
        """
        dim = self._dim
        lo_i = max(int(np.floor(lo)), 0)
        hi_i = min(int(np.ceil(hi)), dim - 1)
        if hi_i < lo_i:
            return []
        span = hi_i - lo_i + 1
        stride = self._stride
        P = dim // stride if (stride and dim % stride == 0) else 0
        px_per_dim = axis_len_px / max(span, 1)
        coarse_min = max(3 * stride if stride else dim, 2 * self.detail_span)
        mid_min = max(stride if stride else 0, self.detail_span)

        if stride and P >= 2 and span >= coarse_min:
            # ---- 粗：扇区名（major）+ 组名（minor，放得下才显示）----
            sec_step = max(1, int(np.ceil(28.0 / (stride * px_per_dim))))
            sec = [(p * stride + stride / 2.0, f'n={p - 1}')
                   for p in range(0, P, sec_step)
                   if lo_i <= p * stride + stride / 2.0 <= hi_i]
            grp = []
            pc = P // 2                       # 组名只标中央扇区（结构周期重复）
            for (s, e, name) in (self._groups or []):
                pos = pc * stride + (s + e) / 2.0
                if lo_i <= pos <= hi_i and (e - s) * px_per_dim >= 45.0:
                    grp.append((pos, name))
            return [sec, grp]

        if stride and P >= 2 and span >= mid_min:
            # ---- 中：抽稀短标签（扇区已含在各标签 |n=… 中，不并排显示）----
            step = max(1, int(np.ceil(span / self.max_ticks)))
            return [[(t, self._label_fn(t))
                     for t in range(lo_i, hi_i + 1, step)]]

        # ---- 细：逐维全标注（扇区已含在各标签 |n=… 中，不并排显示）----
        if stride and P >= 2:
            step = max(1, int(np.ceil(span / self.max_ticks)))
            return [[(t, self._detail_fn(t))
                     for t in range(lo_i, hi_i + 1, step)]]
        step = max(1, int(np.ceil(span / self.max_ticks)))
        return [[(t, self._detail_fn(t))
                 for t in range(lo_i, hi_i + 1, step)]]


# ---------------------------------------------------------------
# demo：直接运行本文件（多档缩放截图由外部脚本驱动）
# ---------------------------------------------------------------
if __name__ == '__main__':
    import sys
    from PyQt5.QtWidgets import QApplication
    app = QApplication(sys.argv)
    dim, stride = 75, 5
    base = ['60S+½', '60S−½', '60P−½', '6P+1', '6P0']
    labels = [base[i % stride] for i in range(dim)]
    rng = np.random.default_rng(0)
    A = np.abs(rng.normal(0, 1, (dim, dim)))
    A[10, 25] = A[25, 10] = 12.0

    w = QWidget()
    w.setWindowTitle('分级标注自适应热图 demo')
    lay = QVBoxLayout(w)
    hm = AdaptiveHeatmap()
    lay.addWidget(hm, 1)

    def label_fn(i):
        return f'{labels[i % stride]}|n={i // stride - 1}'

    def detail_fn(i):
        p, k = divmod(i, stride)
        return f'{base[k]}\n|n={p - 1}\n{-i * 1e6:+.3g} MHz'

    hm.set_data(A, label_fn, dim, detail_fn)
    hm.set_label_meta(labels, stride)
    w.resize(1200, 800)
    w.show()
    sys.exit(app.exec_())
