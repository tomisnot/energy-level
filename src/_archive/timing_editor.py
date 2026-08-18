#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""TTL 时序控制编辑器（PyQt5，独立脚本）

模型：一切时序 = 边沿(Edge) + 父沿偏移。
  - 同通道相邻沿默认父子（保持时间）；根沿（parent=None）锚定绝对时间
  - Ctrl 点两沿 → 建立相对关系（先选 = 父）；求解 = 沿父链传播 + 同通道递增
交互：
  - 拖沿 = 改偏移；拖根沿 = 整棵子树平移（拖动中不重设场景范围防崩溃）
  - 双击沿/区间改数值；右键更多操作；Ctrl 缩放；双击空白加通道
  - 动态通道：起始 3 个 TTL，可加波形通道（绑定 TTL 沿为播放触发）
  - 时间范围按时序自动扩展；导出 evolve.py 可用的 waveform 代码

运行：D:\\python310\\python.exe scripts\\timing_editor.py
"""
import sys
from collections import namedtuple
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QGraphicsView, QGraphicsScene,
    QGraphicsPathItem, QGraphicsSimpleTextItem, QGraphicsItem, QToolBar,
    QAction, QDockWidget, QTableWidget, QTableWidgetItem, QDialog,
    QTextEdit, QVBoxLayout, QLabel, QPushButton, QMenu, QInputDialog,
    QMessageBox, QWidget, QHeaderView, QLineEdit, QSpinBox)
from PyQt5.QtCore import Qt, QRectF, QPointF, QPoint, QTimer, pyqtSignal
from PyQt5.QtGui import (
    QPen, QColor, QPainter, QPainterPath, QFont,
    QDoubleValidator)

# ================= 常量 =================
ROW_H, RULER_H, MARK, X_LEFT, SNAP_PX = 66, 38, 11, 12, 6
PALETTE = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6',
           '#1abc9c', '#e67e22', '#16a085', '#c0392b', '#2980b9']


# ================= 数据模型 =================
class Relation:
    """时序关系：父沿 → 子沿 的相对偏移。
    画布尺寸标注 ↔ 右侧表格行 ↔ 本对象一一对应。"""
    def __init__(self, parent, child, offset_ns=0.0):
        self.parent = parent
        self.child = child
        self.offset_ns = float(offset_ns)

    def desc(self):
        return f'{self.parent.desc}→{self.child.desc}'


class Edge:
    """沿。定位三态：relation(相对距离,不可拖) / anchored(根锚点,不可拖) /
    free(未定义位置,可拖，root_abs 暂存当前位置)。"""
    def __init__(self, kind='rise', relation=None, root_abs=0.0, anchored=False):
        self.kind = kind          # 'rise' | 'fall'
        self.relation = relation  # Relation | None
        self.root_abs = float(root_abs)   # 锚点/自由沿的（暂存）绝对时间
        self.anchored = bool(anchored)    # True=根锚点（数值定义，不可拖）
        self.t_abs = 0.0          # 求解后的绝对时间
        self.owner = None         # 所属通道（solve 时由 model 维护）

    @property
    def draggable(self):
        """可拖动 = 未定义位置（无父关系且非锚点）。"""
        return self.relation is None and not self.anchored

    @property
    def idx(self):
        """通道内沿序号（0,1,2,…）。"""
        return self.owner.edges.index(self) if self.owner is not None else 0

    @property
    def letter(self):
        """通道字母 A,B,C,…。"""
        return chr(ord('A') + self.owner.index) if self.owner is not None else '?'

    @property
    def desc(self):
        """唯一沿名：通道字母 + 通道内序号，如 a0、a1、b2。"""
        return f'{self.letter}{self.idx}'

    @property
    def symbol(self):
        """上升/下降三角标注（与画布 marker 一致）。"""
        return '▲' if self.kind == 'rise' else '▼'


class TTLChannel:
    def __init__(self, name, color, edges=None):
        self.name, self.color = name, color
        self.edges = edges or []  # 有序沿序列（隐含高低交替）
        self.index = 0            # 通道序号（solve 时由 model 维护）


class WfChannel:
    """波形（光场包络）通道：可绑定一个 TTL 沿作为播放触发。"""
    def __init__(self, name, color, wf_type='square', width_ns=400.0,
                 sigma_ns=80.0, trigger=None, offset_ns=0.0):
        self.name, self.color = name, color
        self.wf_type = wf_type    # 'square' | 'gaussian'
        self.width_ns, self.sigma_ns = float(width_ns), float(sigma_ns)
        self.trigger, self.offset_ns = trigger, float(offset_ns)
        self.start_ns = 0.0       # 无触发时的绝对起点

    def start(self):
        return self.trigger.t_abs + self.offset_ns if self.trigger \
            else self.start_ns


class TimingModel:
    """容器 + 求解器（约束图拓扑传播）。"""
    def __init__(self):
        self.channels, self.wf_channels = [], []
        self.t_max = 1000.0
        self.ok = True

    def solve(self):
        for i, ch in enumerate(self.channels):
            ch.index = i
            for e in ch.edges:
                e.owner = ch      # 沿知道所属通道（命名/序号需要）
        state = {}                # 0=未访问 1=栈中(环) 2=完成
        self.ok = True

        def _abs(e):
            s = state.get(id(e), 0)
            if s == 1:
                self.ok = False
            elif s == 0:
                state[id(e)] = 1
                if e.relation is None:
                    e.t_abs = e.root_abs
                else:
                    _abs(e.relation.parent)
                    e.t_abs = e.relation.parent.t_abs + e.relation.offset_ns
                state[id(e)] = 2

        for ch in self.channels:
            for e in ch.edges:
                _abs(e)
        tmax = 100.0
        for ch in self.channels:
            for k, e in enumerate(ch.edges):
                tmax = max(tmax, e.t_abs)
                if k and e.t_abs < ch.edges[k - 1].t_abs - 1e-9:
                    self.ok = False
        for wc in self.wf_channels:
            tmax = max(tmax, wc.start() + wc.width_ns * 2)
        self.t_max = tmax * 1.12
        return self.ok

    def all_edges(self):
        for ch in self.channels:
            yield from ch.edges

    def all_relations(self):
        for ch in self.channels:
            for e in ch.edges:
                if e.relation is not None:
                    yield e.relation

    # ---------- 基本操作：所有删除/解绑都收敛到这两条 ----------
    def free_edge(self, e):
        """沿脱离父关系 → 自由态（原位暂存 root_abs，可拖动）。"""
        if e.relation is not None:
            e.root_abs = e.t_abs
            e.relation = None
            e.anchored = False
            return True
        return False

    def free_children(self, e):
        """删 e 时：所有以 e 为父的沿一并转自由（位置不突变）。"""
        for oe in self.all_edges():
            if oe.relation is not None and oe.relation.parent is e:
                self.free_edge(oe)


def _with_rel(parent, child, offset_ns):
    """把关系绑定到子沿：child.relation = Relation(parent, child, offset)。"""
    child.relation = Relation(parent, child, offset_ns)
    return child


# ================= 默认初始时序 =================
def make_default_model():
    m = TimingModel()
    prev = None
    for i in range(3):
        if i == 0:
            rise = Edge('rise', anchored=True)   # 根锚点 t=0
        else:
            rise = Edge('rise')
            _with_rel(prev, rise, 200.0)
        fall = _with_rel(rise, Edge('fall'), 500.0)   # 高电平 500ns
        m.channels.append(TTLChannel(f'TTL{i + 1}', PALETTE[i], [rise, fall]))
        prev = fall
    m.solve()
    return m


# ================= 图形项 =================
RelGeo = namedtuple('RelGeo', 'x0 x1 y0 y1 dim_y cross')


class EdgeMarker(QGraphicsPathItem):
    """沿位置标记：竖线 + 顶部三角帽（rise ▲朝上绿 / fall ▼朝下红）。
    选中 → 橙色加粗；悬停 → 加深。拖动/Ctrl 多选/双击/右键交互不变。"""
    _SEL = '#f5b041'

    def __init__(self, edge, scene):
        super().__init__()
        self.edge, self.scene = edge, scene
        self.ch = edge.owner.index      # 通道序号（solve 时已维护）
        self.setFlag(QGraphicsItem.ItemIsSelectable, True)
        self.setFlag(QGraphicsItem.ItemIgnoresTransformations, True)
        # 仅 free（未定义位置）可拖动；相对距离/根锚点位置由数值决定，只可点选
        self.setCursor(Qt.SizeHorCursor if edge.draggable
                       else Qt.PointingHandCursor)
        self.setAcceptHoverEvents(True)
        self.setZValue(20)
        self._p0 = self._t0 = None
        self._hover = False
        self._path()

    def _base_color(self):
        """rise 绿 / fall 红（上升=进入高电平，下降=回到低电平）。"""
        return QColor('#2e7d32' if self.edge.kind == 'rise' else '#c0392b')

    def _path(self):
        # 竖线 + 顶部三角帽：rise 尖朝上（▲），fall 尖朝下（▼）
        up = self.edge.kind == 'rise'
        h = MARK * 0.55
        p = QPainterPath()
        p.moveTo(0, -MARK * 0.5)
        p.lineTo(0, MARK * 1.5)                       # 沿位竖线
        p.moveTo(-h, -MARK * 0.5)                      # 三角帽（基边在竖线顶部）
        p.lineTo(h, -MARK * 0.5)
        p.lineTo(0, -MARK * 0.5 - h if up else -MARK * 0.5 + h)
        p.closeSubpath()
        self.setPath(p)

    def shape(self):
        """加宽命中区：竖线+帽 → 用竖直宽条覆盖（±9px，好点）。"""
        p = QPainterPath()
        p.addRect(QRectF(-9, -MARK * 0.7, 18, MARK * 2.4))
        return p

    def paint(self, painter, option, widget=None):
        painter.setRenderHint(QPainter.Antialiasing, True)
        if self in self.scene._ctrl:
            c, w = QColor(self._SEL), 2.6
        elif self._hover:
            c, w = self._base_color().darker(140), 2.4
        else:
            c, w = self._base_color(), 2.0
        painter.setPen(QPen(c, w))
        painter.setBrush(QColor(c))          # 三角帽填充
        painter.drawPath(self.path())

    def hoverEnterEvent(self, ev):
        self._hover = True
        self.setToolTip(
            f'{self.edge.desc}  {self.edge.symbol} '
            f't={self.edge.t_abs:.1f}ns'
            + ('' if self.edge.draggable else '（数值定义，双击改值）'))
        self.update()

    def hoverLeaveEvent(self, ev):
        self._hover = False; self.update()

    def sync_pos(self):
        self.setPos(self.edge.t_abs,
                    RULER_H + self.ch * ROW_H + ROW_H * 0.38)

    def _snap(self, t):
        d = SNAP_PX / max(self.scene.view_scale(), 1e-6)
        for e in self.scene.model.all_edges():
            if e is not self.edge and abs(e.t_abs - t) < d:
                t = e.t_abs
        return t

    def _commit(self, rebuild=False):
        self.scene.commit(rebuild)

    # ---------- 交互 ----------
    def mousePressEvent(self, ev):
        if ev.modifiers() & Qt.ControlModifier:
            self.scene.ctrl_toggle(self); ev.accept(); return
        if ev.button() == Qt.LeftButton:
            self.scene.select_only(self)          # 普通点击：选中当前
            if self.edge.draggable:               # 仅未定义位置的沿可拖（数值关系优先）
                self._p0, self._t0 = ev.scenePos(), self.edge.t_abs
            ev.accept(); return
        super().mousePressEvent(ev)

    def mouseMoveEvent(self, ev):
        if self._p0 is None:
            return super().mouseMoveEvent(ev)
        t = self._snap(self._t0 + ev.scenePos().x() - self._p0.x())
        self.edge.root_abs = t          # free 沿：暂存位置（子沿跟随）
        self.scene.refresh(False)           # 拖动中不重设场景范围（防崩溃）
        self.scene.view.status_cb(f'沿 t={self.edge.t_abs:.1f}ns')
        ev.accept()

    def mouseReleaseEvent(self, ev):
        if self._p0 is not None:
            self._p0 = None
            QTimer.singleShot(0, self._after_drag)      # 延迟到事件空闲
            ev.accept(); return
        super().mouseReleaseEvent(ev)

    def _after_drag(self):
        """拖动结束：恢复场景范围 + 沿表同步（refresh 内含自动适应）。"""
        self.scene.refresh()
        self.scene.window.refresh_tables()

    def mouseDoubleClickEvent(self, ev):
        e = self.edge
        if e.relation is None:
            v, ok = QInputDialog.getDouble(None, '绝对时间', '锚定绝对时间 (ns):',
                                           e.root_abs, 0, 1e9, 1)
            if ok:
                e.root_abs = v
        else:
            v, ok = QInputDialog.getDouble(None, '偏移', '相对父沿偏移 (ns):',
                                           e.relation.offset_ns, -1e6, 1e6, 1)
            if ok:
                e.relation.offset_ns = v
        self._commit(); ev.accept()

    def contextMenuEvent(self, ev):
        e = self.edge
        m = QMenu()
        if e.anchored:
            a = m.addAction(f'根锚点 t={e.root_abs:.1f}ns'); a.setEnabled(False)
        else:
            m.addAction('设为根锚点（独立）').triggered.connect(self._make_root)
        sub = m.addMenu('以其他沿为参考…')
        for ci, ch in enumerate(self.scene.model.channels):
            for o in ch.edges:
                if o is not e:
                    sub.addAction(f'{o.desc} @{o.t_abs:.1f}ns'
                                  ).triggered.connect(
                                      lambda _, o=o: self._set_parent(o))
        m.addAction('删除此沿').triggered.connect(self._delete)
        m.addAction('其后插入沿（保持时间…）').triggered.connect(self._insert)
        m.exec_(ev.screenPos()); ev.accept()

    def _make_root(self):
        e = self.edge
        self.scene.model.free_edge(e)   # 脱离关系，原位暂存
        e.anchored = True               # 转根锚点（数值定义，不可拖）
        self._commit()

    def _set_parent(self, o):
        e = self.edge
        _with_rel(o, e, e.t_abs - o.t_abs)
        self._commit()

    def _delete(self):
        self.scene.window._delete_edge(self.edge)   # 与沿表右键共用同一删除

    def _insert(self):
        ch = self.scene.model.channels[self.ch]
        ch.edges.insert(ch.edges.index(self.edge) + 1,
                        _with_rel(self.edge, Edge(
                            'fall' if self.edge.kind == 'rise' else 'rise'),
                            200.0))
        self._commit(rebuild=True)


class _OffsetEdit(QLineEdit):
    """画布内联偏移编辑框：Esc 取消，回车/失焦应用。"""
    def __init__(self, text, on_done):
        super().__init__(text)
        self._on_done = on_done
        self.setValidator(QDoubleValidator(-1e6, 1e6, 1))
        self.setAlignment(Qt.AlignCenter)
        self.setFont(QFont('Sans Serif', 13))   # 与标注数字字号一致
        self.setFixedWidth(80)

    def keyPressEvent(self, ev):
        if ev.key() == Qt.Key_Escape:
            # 延迟到事件空闲再回收（销毁自身不能在按键处理中进行）
            QTimer.singleShot(0, lambda: self._on_done(False))
            ev.accept(); return
        super().keyPressEvent(ev)


class _RelSelectable:
    """关系标注通用交互：单击选中、右键删除（与关系表右键一致）。"""
    def mousePressEvent(self, ev):
        self.scene_ref.select_relation(self.relation)
        ev.accept()

    def contextMenuEvent(self, ev):
        self.scene_ref.select_relation(self.relation)    # 与表中一致：先选中
        m = QMenu()
        m.addAction('删除此关系（子沿变自由，位置不变）').triggered.connect(
            lambda: self.scene_ref.window._delete_rel(self.relation))
        m.exec_(ev.screenPos()); ev.accept()


class DimLabel(_RelSelectable, QGraphicsSimpleTextItem):
    """关系标注数字：单击选中关系；双击原地内联编辑偏移（不弹窗）。"""
    def __init__(self, relation, scene):
        super().__init__()
        self.relation, self.scene_ref = relation, scene
        self.setFlag(QGraphicsItem.ItemIgnoresTransformations, True)
        self.setCursor(Qt.PointingHandCursor)
        self.setZValue(7)
        self.setBrush(QColor('#8a6d3b'))
        self.setFont(QFont('Sans Serif', 13, QFont.Bold))

    def set_selected(self, sel):
        self.setBrush(QColor('#e67e22') if sel else QColor('#8a6d3b'))
        self.update()

    def mouseDoubleClickEvent(self, ev):
        self.scene_ref.start_edit_offset(self.relation)  # 双击 = 原地编辑
        ev.accept()


class DimensionItem(_RelSelectable, QGraphicsPathItem):
    """关系尺寸标注线：点击 → 选中该关系（线高亮 + 表格行高亮）。"""
    def __init__(self, relation, scene):
        super().__init__()
        self.relation, self.scene_ref = relation, scene
        pen = QPen(QColor('#8a6d3b'), 2.0); pen.setCosmetic(True)
        self.setPen(pen)
        self.setZValue(5)
        self.setCursor(Qt.PointingHandCursor)
        self.setAcceptHoverEvents(True)

    def set_selected(self, sel):
        pen = QPen(QColor('#e67e22') if sel else QColor('#8a6d3b'),
                   2.8 if sel else 2.0)
        pen.setCosmetic(True)
        self.setPen(pen)
        self.update()


# ================= 场景 =================
class _ClickText(QGraphicsSimpleTextItem):
    """可点击文本（✕ 删通道 / ＋ 加通道），点击触发回调。"""
    def __init__(self, text, on_click):
        super().__init__(text)
        self._on_click = on_click
        self.setFlag(QGraphicsItem.ItemIgnoresTransformations, True)
        self.setAcceptedMouseButtons(Qt.LeftButton)
        self.setCursor(Qt.PointingHandCursor)

    def mousePressEvent(self, ev):
        self._on_click()
        ev.accept()


class TimelineScene(QGraphicsScene):
    def __init__(self, model, window):
        super().__init__()
        self.model, self.window = model, window   # window 直接驱动表格高亮
        self.view = None                 # TimingWindow 创建 view 后补上
        self.markers = {}                # id(edge) → EdgeMarker
        self.dims = {}                   # id(relation) → DimensionItem
        self.dim_labels = {}             # id(relation) → DimLabel
        self.wf_labels = {}              # id(wc) → 触发信息文本
        self.lane_items, self.wf_items = {}, {}
        self._ctrl = []                  # 选中沿（paint 查 self in _ctrl）
        self._sel_rel = None             # 选中关系（画布↔表格联动）
        self._edit_proxy = None          # 内联偏移编辑框（QGraphicsProxyWidget）
        self._edit_rel = None
        self.label_shift = {}            # id(relation) → 防重叠垂直错开量(场景)
        self.setBackgroundBrush(QColor(255, 255, 255))

    def view_scale(self):
        return self.view.transform().m11() if self.view else 1.0

    def _total_h(self):
        return (RULER_H + (len(self.model.channels)
                           + len(self.model.wf_channels)) * ROW_H + 60)

    def _lane_y(self, i):
        return RULER_H + i * ROW_H

    def _edge_y(self, e):
        return self._lane_y(self._ch_of(e)) + ROW_H * 0.38

    def _wf_y(self, i):
        return self._lane_y(len(self.model.channels) + i)

    def _ch_of(self, e):
        return e.owner.index if e.owner is not None else 0

    def _wf_idx(self, wc):
        return self.model.wf_channels.index(wc)

    # ---------- 几何 ----------
    def _lane_path(self, i, ch):
        y = self._lane_y(i)
        hi, lo = y + ROW_H * 0.30, y + ROW_H * 0.70
        p = QPainterPath(); p.moveTo(0, lo)
        lvl = 0
        for e in ch.edges:
            p.lineTo(e.t_abs, lo if lvl == 0 else hi)
            p.lineTo(e.t_abs, hi if lvl == 0 else lo)
            lvl ^= 1
        p.lineTo(self.model.t_max, hi if lvl else lo)
        return p

    def _wf_path(self, wc):
        y0 = self._wf_y(self._wf_idx(wc)) + ROW_H * 0.75
        p = QPainterPath(); p.moveTo(0, y0)
        start, w, sig = wc.start(), wc.width_ns, wc.sigma_ns
        for k in range(61):
            tt = start + w * k / 60
            if wc.wf_type == 'square':
                env = 1.0 if start <= tt <= start + w else 0.0
            else:
                c = start + w / 2
                env = 2.71828 ** (-(tt - c) ** 2 / (2 * sig ** 2))
                env = env if env >= 0.01 else 0.0
            p.lineTo(tt, y0 - env * ROW_H * 0.5)
        p.lineTo(self.model.t_max, y0)
        return p

    # ---------- 文本辅助 ----------
    def _mk_text(self, text, color, size=11, bold=False):
        t = QGraphicsSimpleTextItem(text)
        t.setBrush(QColor(color))
        t.setFont(QFont('Sans Serif', size, QFont.Bold if bold else QFont.Normal))
        t.setFlag(QGraphicsItem.ItemIgnoresTransformations, True)  # 不随缩放压扁
        return t

    def _center_x(self, item, cx):
        return cx - item.boundingRect().width() / (2 * self.view_scale())

    # ---------- 关系标注几何（每个 Relation 一个标注，一一对应） ----------
    def _rel_geometry(self, rel):
        """尺寸标注几何 RelGeo(x0,x1,y0,y1,dim_y,cross)。cross=跨通道。"""
        p, c = rel.parent, rel.child
        cross = self._ch_of(p) != self._ch_of(c)
        x0, x1 = p.t_abs, c.t_abs
        if cross:
            y0 = self._edge_y(p) + MARK + 2
            y1 = self._edge_y(c) - MARK - 2
            dim_y = (y0 + y1) / 2
        else:                     # 同通道：尺寸线放通道底部（电平下方）
            dim_y = self._lane_y(self._ch_of(c)) + ROW_H * 0.90
            y0 = y1 = dim_y
        return RelGeo(x0, x1, y0, y1, dim_y, cross)

    def _rel_shift(self, rel):
        """该关系标注的防重叠垂直错开量（场景坐标）。"""
        return self.label_shift.get(id(rel), 0.0)

    def _dim_path(self, rel):
        """工程尺寸标注：extension 线（仅跨通道，拉长到尺寸线）+ 尺寸线 + 45° 刻度。
        整条标注随 label_shift 整体下移（防重叠时与数字一起动）。"""
        g = self._rel_geometry(rel)
        dim_y = g.dim_y + self._rel_shift(rel)
        p = QPainterPath()
        if g.cross:
            p.moveTo(g.x0, g.y0); p.lineTo(g.x0, dim_y)      # extension 线
            p.moveTo(g.x1, g.y1); p.lineTo(g.x1, dim_y)
        p.moveTo(min(g.x0, g.x1), dim_y); p.lineTo(max(g.x0, g.x1), dim_y)  # 尺寸线
        t = 6.0
        p.moveTo(g.x0 - t, dim_y - t); p.lineTo(g.x0 + t, dim_y + t)    # 45° 刻度
        p.moveTo(g.x1 - t, dim_y - t); p.lineTo(g.x1 + t, dim_y + t)
        return p

    def _set_dim_label(self, lab, rel):
        """数字垂直中心精确对齐尺寸线（用屏幕坐标 + m22 换算）。"""
        g = self._rel_geometry(rel)
        cx = (g.x0 + g.x1) / 2
        dim_y = g.dim_y + self._rel_shift(rel)
        c = self.view.mapFromScene(QPointF(cx, dim_y))
        h = lab.boundingRect().height()      # 屏幕像素（ignore-transforms）
        lab.setPos(self._center_x(lab, cx),
                   self.view.mapToScene(
                       QPoint(int(c.x()), int(c.y() - h / 2))).y())

    def layout_dim_labels(self):
        """防重叠：按 x 排序，整条标注（尺寸线+数字）x 范围重叠时整体垂直错开。
        用整条尺寸线的 x 范围判断（不只是数字宽度），长尺寸线才不会盖住其他标注。"""
        if self.view is None:
            return
        labs = list(self.dim_labels.items())
        if not labs:
            return
        s = max(self.view_scale(), 1e-6)
        m22 = self.view.transform().m22() or 1e-6   # 垂直缩放（与 m11 可能不同）
        self.label_shift = {}
        items = []
        for kid, lab in labs:
            rel = lab.relation
            g = self._rel_geometry(rel)
            br = lab.boundingRect()
            items.append(((g.x0 + g.x1) / 2, kid, lab, br.width(), br.height(),
                          min(g.x0, g.x1), max(g.x0, g.x1)))
        items.sort(key=lambda t: t[0])       # 按中心 x 排序
        placed = []   # 屏幕矩形 (x0, x1, y0, y1)：x=整条尺寸线，y=数字所在层
        for cx, kid, lab, w, h, l0, l1 in items:
            rel = lab.relation
            dim_y = self._rel_geometry(rel).dim_y
            c = self.view.mapFromScene(QPointF(cx, dim_y))
            x_l, x_r = l0 * s, l1 * s        # 尺寸线整段（屏幕像素）
            y = c.y() - h / 2
            dy = 0
            while any(x_l < q[1] and x_r > q[0]
                      and y < q[3] and y + h > q[2] for q in placed):
                dy += h * 0.7                # 向下错开一个标签高度
                y = c.y() - h / 2 + dy
            placed.append((x_l, x_r, y, y + h))
            self.label_shift[id(rel)] = dy / m22   # 场景偏移（垂直用 m22）
            lab.setPos(cx - w / (2 * s),
                       self.view.mapToScene(
                           QPoint(int(c.x()), int(y))).y())
            d = self.dims.get(id(rel))
            if d is not None:
                d.setPath(self._dim_path(rel))   # 尺寸线跟随数字一起动

    # ---------- 全量重建 ----------
    def commit(self, rebuild=True):
        """模型已改 → 求解 + 重建(结构变化)或轻量刷新(仅数值变化)。
        所有修改模型的地方都收敛到这里。"""
        self.model.solve()
        (self.rebuild if rebuild else self.refresh)()

    def rebuild(self):
        self.clear()
        self.markers, self.dims, self.dim_labels = {}, {}, {}
        self.wf_labels = {}
        self.lane_items, self.wf_items = {}, {}
        self.label_shift = {}
        self._ctrl, self._sel_rel = [], None
        self.model.solve()
        self.setSceneRect(-X_LEFT, 0, self.model.t_max + X_LEFT, self._total_h())
        for ch in self.model.channels:
            self._build_lane(ch)
            for e in ch.edges:
                self._build_edge(e)
        for rel in self.model.all_relations():
            self._build_rel(rel)
        for i, wc in enumerate(self.model.wf_channels):
            self._build_wf(i, wc)
        self.window.refresh_tables()      # 图像重建后同步表格
        self.window.view.auto_fit()       # 自动适应新时序范围
        self.layout_dim_labels()          # 标签防重叠布局

    # ---------- 通道底色条（TTL / 波形共用） ----------
    def _lane_bg(self, y, color):
        bg = QGraphicsPathItem()
        p = QPainterPath(); p.addRect(QRectF(0, y, self.model.t_max, ROW_H))
        bg.setPath(p)
        bg.setBrush(color); bg.setPen(QPen(Qt.NoPen)); bg.setZValue(0)
        self.addItem(bg)

    def _build_lane(self, ch):
        """TTL 通道轨道：只画电平波形（通道名/沿数控制都在左侧列）。"""
        i = ch.index
        y = self._lane_y(i)
        self._lane_bg(y, QColor(245, 245, 248)
                      if i % 2 == 0 else QColor(255, 255, 255))
        lane = QGraphicsPathItem(); lane.setPath(self._lane_path(i, ch))
        lane.setPen(QPen(QColor(ch.color), 2.5)); lane.setZValue(1)
        self.addItem(lane); self.lane_items[id(ch)] = lane

    def _build_edge(self, e):
        mk = EdgeMarker(e, self)
        mk.sync_pos(); mk.setZValue(20)
        self.addItem(mk); self.markers[id(e)] = mk
        if e.relation is None and e.anchored and abs(e.root_abs) > 1e-9:
            lab = self._mk_text(f't={e.root_abs:.0f}ns', '#5a5a5a', 11)
            lab.setZValue(6); self.addItem(lab)
            lab.setPos(e.t_abs + 6, self._edge_y(e) - 18)

    def _build_rel(self, rel):
        """每个关系一个尺寸标注组件（线 + 数字），与 Relation 一一对应。"""
        dim = DimensionItem(rel, self)
        dim.setPath(self._dim_path(rel))
        self.addItem(dim); self.dims[id(rel)] = dim
        lab = DimLabel(rel, self)
        lab.setText(f'{rel.offset_ns:.0f}ns')
        self._set_dim_label(lab, rel)
        self.addItem(lab); self.dim_labels[id(rel)] = lab

    def _build_wf(self, i, wc):
        y = self._wf_y(i)
        self._lane_bg(y, QColor(250, 250, 240))
        # 只画波形曲线 + 触发信息（通道名在波形编辑器左侧列）
        curve = QGraphicsPathItem(); curve.setPath(self._wf_path(wc))
        curve.setPen(QPen(QColor(wc.color), 2)); curve.setZValue(1)
        self.addItem(curve); self.wf_items[id(wc)] = curve
        if wc.trigger is not None:
            info = self._mk_text(
                f'← 触发 {wc.trigger.desc} +{wc.offset_ns:.0f}ns', '#785028', 11)
            info.setPos(wc.start() + 2, y + ROW_H * 0.30)
            info.setZValue(3); self.addItem(info)
            self.wf_labels[id(wc)] = info

    def _del_channel(self, ch):
        """删除指定通道（TTL 或波形），重建。"""
        if ch in self.model.channels:
            self.model.channels.remove(ch)
        elif ch in self.model.wf_channels:
            self.model.wf_channels.remove(ch)
        else:
            return
        self.commit()
        self.window.channels_changed.emit()

    # ---------- 轻量刷新（拖动/数值微调，不重建 item） ----------
    def refresh(self, update_extent=True):
        self.model.solve()
        if update_extent:
            self.setSceneRect(-X_LEFT, 0, self.model.t_max + X_LEFT,
                              self._total_h())
        for i, ch in enumerate(self.model.channels):
            self.lane_items[id(ch)].setPath(self._lane_path(i, ch))
        for rel in self.model.all_relations():
            dim = self.dims.get(id(rel))
            if dim is not None:
                dim.setPath(self._dim_path(rel))
            lab = self.dim_labels.get(id(rel))
            if lab is not None:
                lab.setText(f'{rel.offset_ns:.0f}ns')
                self._set_dim_label(lab, rel)
        for e in self.model.all_edges():
            self.markers[id(e)].sync_pos()
        for wc in self.model.wf_channels:
            self.wf_items[id(wc)].setPath(self._wf_path(wc))
            info = self.wf_labels.get(id(wc))
            if info is not None and wc.trigger is not None:
                info.setPos(wc.start() + 2,
                            self._wf_y(self._wf_idx(wc)) + ROW_H * 0.30)
        self.layout_dim_labels()
        if update_extent:
            self.window.view.auto_fit()   # 数值变化后也自动适应（与重建一致）

    # ---------- 沿选中管理（paint 只看 _ctrl） ----------
    def select_only(self, m):
        """普通点击：只选中该沿（清空多选），并同步沿表高亮。"""
        self._ctrl = [m]
        self._update_markers()
        self.window.highlight_edge(m.edge)

    def select_only_for(self, e):
        """沿表点击 → 画布选中该沿。"""
        m = self.markers.get(id(e))
        if m is not None:
            self.select_only(m)

    def clear_ctrl_sel(self):
        if self._ctrl:
            self._ctrl = []
            self._update_markers()

    def ctrl_toggle(self, m):
        if m in self._ctrl:
            self._ctrl.remove(m)
        else:
            self._ctrl.append(m)
            if len(self._ctrl) > 2:
                self._ctrl.pop(0)
        self._update_markers()
        if len(self._ctrl) == 2:
            self._link()

    def _update_markers(self):
        for mk in self.markers.values():
            mk.update()

    # ---------- 关系选中（画布标注 ↔ 表格行，直接互调） ----------
    def select_relation(self, rel):
        """选中关系：标注高亮 + 关系表行高亮。"""
        if self._sel_rel is rel:
            return
        if self._sel_rel is not None:
            self._set_rel_sel(self._sel_rel, False)
        self._sel_rel = rel
        self._set_rel_sel(rel, True)
        self.window.highlight_rel(rel)

    def _set_rel_sel(self, rel, sel):
        d = self.dims.get(id(rel))
        if d is not None:
            d.set_selected(sel)
        l = self.dim_labels.get(id(rel))
        if l is not None:
            l.set_selected(sel)

    def clear_rel_sel(self):
        if self._sel_rel is not None:
            self._set_rel_sel(self._sel_rel, False)
            self._sel_rel = None

    # ---------- 内联偏移编辑（双击数字，原地输入不弹窗） ----------
    def start_edit_offset(self, rel):
        self._finish_edit(apply=True)      # 先收尾上一个编辑
        lab = self.dim_labels.get(id(rel))
        if lab is None:
            return
        edit = _OffsetEdit(f'{rel.offset_ns:.0f}', self._finish_edit)
        proxy = self.addWidget(edit)
        proxy.setFlag(QGraphicsItem.ItemIgnoresTransformations, True)
        proxy.setZValue(30)
        # 屏幕坐标精确定位：编辑框中心与数字中心重合（跟随错开后的数字）
        c = self.view.mapFromScene(lab.scenePos())
        w = lab.boundingRect().width()
        h = lab.boundingRect().height()
        ew, eh = edit.width(), edit.height()
        proxy.setPos(self.view.mapToScene(QPoint(
            int(c.x() + w / 2 - ew / 2), int(c.y() + h / 2 - eh / 2))))
        self._edit_proxy, self._edit_rel = proxy, rel
        edit.selectAll()
        edit.setFocus()
        # 回收全部延迟到事件空闲：不能在 QLineEdit 按键处理中销毁它（会崩）
        edit.returnPressed.connect(
            lambda: QTimer.singleShot(0, self._finish_edit))
        edit.editingFinished.connect(
            lambda: QTimer.singleShot(0, self._finish_edit))

    def _finish_edit(self, apply=True):
        if self._edit_proxy is None:
            return
        proxy, edit = self._edit_proxy, self._edit_proxy.widget()
        txt, rel = edit.text(), self._edit_rel
        self._edit_proxy, self._edit_rel = None, None   # 先清状态防 editingFinished 重入
        self.removeItem(proxy)
        if apply and rel is not None:
            try:
                rel.offset_ns = float(txt)
            except ValueError:
                pass
            self.commit()

    def _link(self):
        p, c = self._ctrl[0].edge, self._ctrl[1].edge   # 先选 = 父沿
        self.clear_ctrl_sel()
        v, ok = QInputDialog.getDouble(None, '建立时序关系',
                                       f'“{c.desc}”相对“{p.desc}”的偏移 (ns):',
                                       c.t_abs - p.t_abs, -1e6, 1e6, 1)
        if ok:
            _with_rel(p, c, v)
            self.commit()

    def channel_at(self, scene_y):
        if scene_y < RULER_H:
            return None
        y = scene_y - RULER_H
        n = len(self.model.channels)
        if y < n * ROW_H:
            return int(y // ROW_H)
        yy = y - n * ROW_H
        if yy < len(self.model.wf_channels) * ROW_H:
            return ('wf', int(yy // ROW_H))
        return None


# ================= 视图 =================
class TimelineView(QGraphicsView):
    def __init__(self, scene, status_cb):
        super().__init__(scene)
        self.status_cb = status_cb
        self.setRenderHint(QPainter.Antialiasing)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.setViewportUpdateMode(QGraphicsView.FullViewportUpdate)

    _CLICKABLE = (EdgeMarker, DimLabel, DimensionItem)   # 沿/标注自己处理事件

    # 点击空白（非沿/标注）清除两种选中
    def mousePressEvent(self, ev):
        if not isinstance(self.itemAt(ev.pos()), self._CLICKABLE):
            sc = self.scene()
            if isinstance(sc, TimelineScene):
                sc.clear_ctrl_sel()
                sc.clear_rel_sel()
        super().mousePressEvent(ev)

    def drawForeground(self, painter, rect):
        """标尺：刻度线与数字统一用视图像素坐标，天然对齐。"""
        painter.save()
        painter.resetTransform()
        painter.fillRect(QRectF(0, 0, self.viewport().width(), RULER_H),
                         QColor(238, 238, 240))
        vp = self.mapToScene(self.viewport().rect()).boundingRect()
        step = 20 if vp.width() <= 200 else 100 if vp.width() <= 1000 \
            else 500 if vp.width() <= 5000 else 2000
        t0 = int(vp.left() / step) * step
        painter.setPen(Qt.black)
        painter.setFont(QFont('Sans Serif', 11))
        for t in range(t0, int(vp.right()) + 1, step):
            xv = self.mapFromScene(QPointF(t, 0)).x()
            painter.drawLine(QPointF(xv, RULER_H - 7), QPointF(xv, RULER_H))
            painter.drawText(QRectF(xv - 40, 0, 80, RULER_H - 8),
                             Qt.AlignCenter, f'{t:.0f} ns')
        painter.restore()

    def auto_fit(self):
        """自动适应窗口：fitInView 新 sceneRect，缩放只钳制上限。
        （m11<1 时**不**放大到 1:1：大 t_max 会严重水平溢出；文字用
        ItemIgnoresTransformations 恒可读，缩小全览即可。）"""
        sc = self.scene()
        if sc is None:
            return
        self.fitInView(sc.sceneRect(), Qt.IgnoreAspectRatio)
        m11 = self.transform().m11()
        if m11 > 4.0:
            self.scale(4.0 / m11, 1)

    def wheelEvent(self, ev):
        if ev.modifiers() & Qt.ControlModifier:
            f = 1.2 if ev.angleDelta().y() > 0 else 1 / 1.2
            self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
            self.scale(f, 1)
            QTimer.singleShot(0, self.scene().layout_dim_labels)  # 缩放后重排
        else:
            super().wheelEvent(ev)

    def mouseDoubleClickEvent(self, ev):
        # 双击沿/标注 → 交给 item 自己处理（沿编辑/内联编辑/选中）
        if isinstance(self.itemAt(ev.pos()), self._CLICKABLE):
            return super().mouseDoubleClickEvent(ev)
        sc, pos = self.scene(), self.mapToScene(ev.pos())
        if not isinstance(sc, TimelineScene):
            return super().mouseDoubleClickEvent(ev)
        hit = sc.channel_at(pos.y())
        if isinstance(hit, int):
            ch = sc.model.channels[hit]
            if not ch.edges:
                e = Edge('rise'); e.root_abs = pos.x()
                ch.edges.append(e)
            else:
                segs = [((ch.edges[k - 1], ch.edges[k]),
                         abs(pos.x() - (ch.edges[k - 1].t_abs
                                        + ch.edges[k].t_abs) / 2))
                        for k in range(1, len(ch.edges))]
                e0, e1 = min(segs, key=lambda s: s[1])[0] \
                    if segs else (ch.edges[-1], None)
                if e1 is not None and e1.relation is not None:  # 设保持时间
                    v, ok = QInputDialog.getDouble(None, '保持时间',
                        f'{ch.name} 该电平保持 (ns):',
                        e1.relation.offset_ns, 0, 1e7, 1)
                    if ok:
                        e1.relation.offset_ns = v
                elif e1 is None:                  # 通道末尾插入沿
                    v, ok = QInputDialog.getDouble(None, '插入沿',
                        '在通道末尾插入沿，偏移 (ns):', 200, 0, 1e7, 1)
                    if ok:
                        ch.edges.append(_with_rel(ch.edges[-1], Edge(
                            'fall' if ch.edges[-1].kind == 'rise' else 'rise'),
                            v))
            sc.commit()
            ev.accept(); return
        if hit is None:
            self._add_ttl()
        ev.accept()

    def contextMenuEvent(self, ev):
        if isinstance(self.itemAt(ev.pos()), EdgeMarker):
            return super().contextMenuEvent(ev)   # 沿自己处理
        m = QMenu()
        m.addAction('添加 TTL 通道').triggered.connect(self._add_ttl)
        m.addAction('添加波形通道').triggered.connect(self._add_wf)
        m.exec_(ev.globalPos())

    def _add_ttl(self):
        sc = self.scene()
        if isinstance(sc, TimelineScene):
            n = len(sc.model.channels)
            e = Edge('rise'); e.root_abs = sc.model.t_max * 0.5
            sc.model.channels.append(TTLChannel(
                f'TTL{n + 1}', PALETTE[n % len(PALETTE)],
                [e, _with_rel(e, Edge('fall'), 400.0)]))
            sc.commit()

    def _add_wf(self):
        sc = self.scene()
        if isinstance(sc, TimelineScene):
            n = len(sc.model.wf_channels)
            sc.model.wf_channels.append(WfChannel(
                f'光场{n + 1}',
                PALETTE[(len(sc.model.channels) + n) % len(PALETTE)]))
            sc.commit()


# ================= 导出 =================
def compile_code(model):
    """生成 evolve.py 可用的 waveform 代码（文本）。"""
    lines = ['# 生成的时序 → evolve.evolve 的 waveform callable',
             'from src import evolve', '']
    for i, wc in enumerate(model.wf_channels):
        s = wc.start()
        if wc.wf_type == 'square':
            lines.append(f'wf_{i} = evolve.square({s:.1f}, '
                         f'{s + wc.width_ns:.1f})   # {wc.name}')
        else:
            c = s + wc.width_ns / 2
            lines.append(f'wf_{i} = evolve.gaussian({c:.1f}, '
                         f'{wc.sigma_ns:.1f})   # {wc.name}')
        lines.append(f'#   {wc.name} 触发: '
                     f"{wc.trigger.desc if wc.trigger else '绝对'} start={s:.1f}ns")
        lines.append(f'#   fields.append(evolve.field(g, e, omega0_mhz=..., '
                     f'waveform=wf_{i}))')
        lines.append('')
    for i, ch in enumerate(model.channels):
        lines.append(f'# {ch.name}: ' + '  '.join(
            f'{e.desc}@{e.t_abs:.1f}' for e in ch.edges))
    return '\n'.join(lines)


class ExportDialog(QDialog):
    def __init__(self, code, parent=None):
        super().__init__(parent)
        self.setWindowTitle('导出时序代码')
        self.resize(620, 420)
        lay = QVBoxLayout(self)
        te = QTextEdit(); te.setPlainText(code)
        te.setFont(QFont('Consolas', 9))
        lay.addWidget(te)
        bt = QPushButton('关闭'); bt.clicked.connect(self.accept)
        lay.addWidget(bt)


# ================= 主窗口 =================
class TimingWindow(QMainWindow):
    channels_changed = pyqtSignal()      # TTL/波形通道增删后发出（左列/外部同步）

    def __init__(self):
        super().__init__()
        self.setWindowTitle('TTL 时序控制编辑器')
        self.resize(1500, 900)
        self.model = make_default_model()
        self.scene = TimelineScene(self.model, self)
        self.view = TimelineView(self.scene, self._status)
        self.scene.view = self.view
        self.setCentralWidget(self.view)
        self.scene.rebuild()
        self._build_toolbar()
        self._build_dock()
        self._status('就绪：拖动边沿改偏移，双击沿/标注数字改数值，单击标注选中，右键更多。'
                     f' 当前范围 ~{self.model.t_max:.0f} ns')

    def _status(self, text):
        self.statusBar().showMessage(str(text))

    def _build_toolbar(self):
        tb = QToolBar('主工具'); tb.setMovable(False)
        self.addToolBar(tb)
        for label, fn in (('＋TTL通道', self._add_ttl),
                          ('＋波形通道', self._add_wf),
                          ('－通道', self._del_channel)):
            a = QAction(label, self); a.triggered.connect(fn); tb.addAction(a)
        tb.addSeparator()
        for label, fn in (('导出代码', self._export),):
            a = QAction(label, self); a.triggered.connect(fn); tb.addAction(a)

    def _add_ttl(self):
        self.view._add_ttl()
        self.channels_changed.emit()

    def _add_wf(self):
        self.view._add_wf()
        self.channels_changed.emit()

    def _del_channel(self):
        if not self.model.channels:
            return
        self.model.channels.pop()
        self.scene.rebuild()
        self.channels_changed.emit()

    def _fit(self):
        self.view.auto_fit()

    def _export(self):
        ExportDialog(compile_code(self.model), self).exec_()

    def _build_dock(self):
        dock = QDockWidget('时序信息', self)
        dock.setAllowedAreas(Qt.RightDockWidgetArea | Qt.LeftDockWidgetArea)
        w = QWidget()
        v = QVBoxLayout(w)

        def _lab(txt):
            l = QLabel(txt); v.addWidget(l); return l

        _lab('沿（按绝对时间，双击 t 编辑 / 右键删除）：')
        self.edge_table = QTableWidget(0, 2)
        self.edge_table.verticalHeader().setDefaultSectionSize(24)  # 行高对应默认字体
        self.edge_table.setHorizontalHeaderLabels(['沿', 't(ns)'])
        self.edge_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.Stretch)
        self.edge_table.setContextMenuPolicy(Qt.CustomContextMenu)
        self.edge_table.customContextMenuRequested.connect(self._edge_menu)
        self.edge_table.cellClicked.connect(self._edge_clicked)
        v.addWidget(self.edge_table)
        _lab('关系（根锚点 + 相对距离，双击值编辑 / 右键删除）：')
        self.rel_table = QTableWidget(0, 3)
        self.rel_table.verticalHeader().setDefaultSectionSize(24)  # 行高对应默认字体
        self.rel_table.setHorizontalHeaderLabels(['父沿', '子沿', '偏移(ns)'])
        self.rel_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.Stretch)
        self.rel_table.setContextMenuPolicy(Qt.CustomContextMenu)
        self.rel_table.customContextMenuRequested.connect(self._rel_menu)
        self.rel_table.itemChanged.connect(self._rel_changed)
        self.rel_table.cellClicked.connect(self._rel_clicked)
        v.addWidget(self.rel_table)
        dock.setWidget(w)
        self.addDockWidget(Qt.RightDockWidgetArea, dock)
        self.refresh_tables()

    # ---------- 表数据 ----------
    def _edge_name(self, e):
        """沿名：a0 ▲ / a1 ▼（通道字母+序号，附三角上升下降标注）。"""
        return f'{e.desc} {e.symbol}'

    def _edge_rows(self):
        return [(self._edge_name(e), e)
                for ch in self.model.channels for e in ch.edges]

    def _rel_rows(self):
        """关系表行：(类型, A名, B名, 对象)。'root'=根锚点沿 / 'rel'=相对距离。
        free（未定义位置）沿不在此表。"""
        rows = []
        for ch in self.model.channels:
            for e in ch.edges:
                if e.relation is None:
                    if e.anchored:
                        rows.append(('root', self._edge_name(e), '— 锚点', e))
                else:
                    r = e.relation
                    rows.append(('rel', self._edge_name(r.parent),
                                 self._edge_name(r.child), r))
        return rows

    def refresh_tables(self):
        if not hasattr(self, 'edge_table'):   # 表格未建（scene.rebuild 先于 dock）
            return
        self._loading = True
        try:
            er = self._edge_rows()
            self.edge_table.setRowCount(len(er))
            for r, (name, e) in enumerate(er):
                for c, txt in ((0, name), (1, f'{e.t_abs:.1f}')):
                    it = QTableWidgetItem(txt)
                    it.setFlags(it.flags() & ~Qt.ItemIsEditable)  # 沿表只读
                    self.edge_table.setItem(r, c, it)
            rr = self._rel_rows()
            self.rel_table.setRowCount(len(rr))
            for r, (typ, an, bn, obj) in enumerate(rr):
                for c, txt in ((0, an), (1, bn)):
                    it = QTableWidgetItem(txt)
                    it.setFlags(it.flags() & ~Qt.ItemIsEditable)
                    self.rel_table.setItem(r, c, it)
                val = f'{obj.root_abs:.0f}' if typ == 'root' \
                    else f'{obj.offset_ns:.0f}'
                v = QTableWidgetItem(val)
                v.setData(Qt.UserRole, (typ, obj))
                self.rel_table.setItem(r, 2, v)
        finally:
            self._loading = False

    def _apply(self):
        self.scene.commit(rebuild=True)   # 结构变化：solve + 重建 + 表格/防重叠

    # ---------- 画布 ↔ 表格联动（直接互调，cellClicked 不触发 selectRow 循环） ----------
    def highlight_edge(self, e):
        """画布选中沿 → 沿表高亮 + 关系表根锚点行高亮。"""
        for r, (name, ee) in enumerate(self._edge_rows()):
            if ee is e:
                self.edge_table.selectRow(r)
                break
        else:
            self.edge_table.clearSelection()
        for r, (typ, an, bn, obj) in enumerate(self._rel_rows()):
            if typ == 'root' and obj is e:
                self.rel_table.selectRow(r)
                return

    def highlight_rel(self, rel):
        """画布选中关系 → 关系表对应行高亮。"""
        for r, (typ, an, bn, obj) in enumerate(self._rel_rows()):
            if typ == 'rel' and obj is rel:
                self.rel_table.selectRow(r)
                return
        self.rel_table.clearSelection()

    def _edge_clicked(self, row, col):
        """沿表点击 → 画布选中该沿。"""
        e = self._edge_rows()[row][1]
        self.scene.select_only_for(e)

    def _rel_clicked(self, row, col):
        """关系表点击：根锚点行 → 画布选中该沿；相对距离行 → 选中该关系。"""
        typ, an, bn, obj = self._rel_rows()[row]
        if typ == 'root':
            self.scene.select_only_for(obj)
        else:
            self.scene.select_relation(obj)

    # ---------- 表格编辑（内联，仅关系表可编辑） ----------
    def _rel_changed(self, item):
        if self._loading or item.column() != 2:
            return
        data = item.data(Qt.UserRole)
        if data is None:
            return
        typ, obj = data
        try:
            v = float(item.text())
        except ValueError:
            self.refresh_tables(); return
        if typ == 'root':
            obj.root_abs = v          # 根锚点：编辑绝对时间
        else:
            obj.offset_ns = v         # 相对距离：编辑偏移
        self._apply()

    # ---------- 右键删除 ----------
    def _edge_menu(self, pos):
        row = self.edge_table.rowAt(pos.y())
        if row < 0:
            return
        e = self._edge_rows()[row][1]
        m = QMenu(self)
        m.addAction('删除此沿').triggered.connect(lambda: self._delete_edge(e))
        m.exec_(self.edge_table.viewport().mapToGlobal(pos))

    def _delete_edge(self, e):
        for ch in self.model.channels:
            if e in ch.edges:
                if len(ch.edges) <= 1:
                    QMessageBox.information(self, '提示', '每条 TTL 至少保留一个沿。')
                    return
                ch.edges.remove(e)
                break
        self.model.free_children(e)      # 依赖被删沿的沿 → 自由（位置不突变）
        self._apply()

    def _rel_menu(self, pos):
        row = self.rel_table.rowAt(pos.y())
        if row < 0:
            return
        typ, an, bn, obj = self._rel_rows()[row]
        m = QMenu(self)
        if typ == 'root':
            m.addAction('根锚点（删除需先给它加父关系）').setEnabled(False)
        else:
            m.addAction('删除此关系（子沿变自由，位置不变）').triggered.connect(
                lambda: self._delete_rel(obj))
        m.exec_(self.rel_table.viewport().mapToGlobal(pos))

    def _delete_rel(self, rel):
        self.model.free_edge(rel.child)   # 子沿转自由：原位暂存、可拖动
        self._apply()

    # ---------- 通道高电位区间数（画布 QSpinBox 调用） ----------
    def set_high_count(self, ch, n):
        """高电位区间数 n → 沿数 2n（每高电平一对 rise+fall，从低电平开始）。"""
        target = 2 * n
        if len(ch.edges) == target:
            return
        while len(ch.edges) < target:
            if ch.edges:
                last = ch.edges[-1]
                kind = 'rise' if len(ch.edges) % 2 == 0 else 'fall'
                ch.edges.append(_with_rel(last, Edge(kind), 200.0))
            else:                       # 空通道：首沿作根锚点
                ch.edges.append(Edge('rise', anchored=True, root_abs=100.0))
        while len(ch.edges) > target:
            e = ch.edges.pop()
            self.model.free_children(e)   # 依赖被删沿的沿 → 自由（位置不突变）
        self._apply()


# ================= 运行 =================
def main():
    import traceback

    def _hook(tp, val, tb):
        traceback.print_exception(tp, val, tb)
        sys.__excepthook__(tp, val, tb)

    sys.excepthook = _hook
    app = QApplication(sys.argv)
    win = TimingWindow(); win.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
