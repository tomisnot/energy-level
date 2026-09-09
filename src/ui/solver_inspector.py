# -*- coding: utf-8 -*-
"""求解检视器（t75 L1）：mixed 引擎求解过程的可视化面板（上帝模式）。

数据驱动：`load_diag(diag)` 接收引擎诊断快照（evolve_mixed[_pair] 在
diag=True 时产出；UI 迭代可用 `mock_diag()`）。diag 结构契约：

    meta      : {method, backend, N, P, dim, t_end, M, r_crit,
                 stride, labels, n_pairs, tol, max_step}
    photon    : {Ms, P, dim, strides, nvecs, D_full}
    acize     : {r_crit, fields: [{omega, n_before, n_kept, n_acized, …}], shift}
    pairs     : {a, b, v, dD, is_mw, w}
    solver    : {t_steps, dt, nfev, n_acc, n_rej, norm_dev}
    trajectory: {t, sector_amps: {sector: (n_t, n_phys)}}
    H_snapshots: {t0: {term: dim×dim complex}}

v1.1（船长反馈四条）：
- 处理链主体富文本卡片化（分步着色徽标 + 关键数字加粗），比 AC 表更醒目；
- 四个精度旋钮从主界面移入检视器对应位置（per_freq_M/r_crit → 处理链页，
  tol/max_step → 求解过程页），「应用并重跑」就地生效；
- 哈密顿页：解析式按分项着色（与勾选项同色）+ 热图扇区分隔/标注 +
  坐标轴物理标签 + 矩阵元表「物理含义」列——索引与物理意义结合；
- 窗口扩容约一倍（1500×960）。
"""
import json

import numpy as np

from PyQt5.QtCore import Qt, pyqtSignal, QRectF
from PyQt5 import QtCore, QtGui
from PyQt5.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QLabel,
                             QTabWidget, QCheckBox, QTableWidget,
                             QTableWidgetItem, QSlider, QWidget, QHeaderView,
                             QAbstractItemView, QPushButton, QLineEdit,
                             QMessageBox, QFrame, QSpinBox, QSizePolicy)
from matplotlib.figure import Figure
from matplotlib.ticker import MaxNLocator
from matplotlib.backends.backend_qt5agg import (
    FigureCanvasQTAgg as FigureCanvas, NavigationToolbar2QT)

import pyqtgraph as pg

from . import ui_theme

_MOCK = None          # mock_diag() 惰性缓存（UI 迭代用，生产路径不触达）

# 分项 → 颜色（公式着色与勾选项一致）
TERM_COLORS = {
    'K0': '#444441', '光场': '#185FA5', '微波': '#854F0B',
    'V_dd': '#0F6E56', 'AC对角': '#72243E', '耗散': '#A32D2D',
}


class _RotatedAxisItem(pg.AxisItem):
    """x 轴刻度文字旋转 −90°：竖排、自下而上阅读（节省纵向空间）。

    pg 原生不支持刻度文字旋转——重写 drawPicture，把每个 textSpec 平移到
    刻度中心后旋转绘制（rect 用文本实际包围盒居中，避免 pg 预估的横排
    矩形在旋转后错位）。
    ——> v1.2 哈密顿页切到 HamiltonianViewer 后已无引用，保留类便于回退。
    """

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
            y_top = rect.top()          # 轴线下方（含 offset）
            p.save()
            # 锚点 = 文本块顶端贴轴线：旋转后文本自轴线向下悬挂，
            # 不侵入上方热图（下方 setHeight 预留空间正好承接）
            p.translate(cx, y_top + bw / 2.0)
            p.rotate(-90)
            p.drawText(QRectF(-bw / 2, -bh / 2, bw, bh),
                       QtCore.Qt.AlignmentFlag.AlignCenter, text)
            p.restore()


def save_diag_npz(d, path):
    """诊断快照 → npz（数值全量）。键布局：meta/*、photon/*、acize/*、
    pairs/*、solver/*、traj/*、H/<t0>/<分项>。载回 np.load(allow_pickle=False)。
    """
    flat = {}
    meta = dict(d.get('meta', {}))
    meta['labels'] = json.dumps(meta.get('labels', []), ensure_ascii=False)
    for k, v in meta.items():
        flat[f'meta/{k}'] = v if isinstance(v, (int, float, str)) else str(v)
    ph = d.get('photon', {})
    for k in ('D_full', 'nvecs', 'Ms', 'strides'):
        if k in ph:
            flat[f'photon/{k}'] = np.asarray(ph[k])
    ac = d.get('acize', {})
    if ac.get('shift') is not None:
        flat['acize/shift'] = np.asarray(ac['shift'], float)
    flat['acize/fields_json'] = json.dumps(ac.get('fields', []),
                                           ensure_ascii=False)
    pr = d.get('pairs', {})
    for k in ('a', 'b', 'v', 'dD', 'is_mw', 'w'):
        if k in pr:
            flat[f'pairs/{k}'] = np.asarray(pr[k])
    sv = d.get('solver', {})
    for k in ('t_steps', 'dt'):
        if sv.get(k) is not None:
            flat[f'solver/{k}'] = np.asarray(sv[k])
    for k in ('nfev', 'n_acc', 'n_rej', 'norm_dev'):
        if k in sv:
            flat[f'solver/{k}'] = sv[k]
    tr = d.get('trajectory', {})
    if tr.get('t') is not None:
        flat['traj/t'] = np.asarray(tr['t'])
        for sec, arr in (tr.get('sector_amps') or {}).items():
            flat[f'traj/sector_{sec}'] = np.asarray(arr)
    for t0, terms in (d.get('H_snapshots') or {}).items():
        for term, mat in terms.items():
            flat[f'H/{t0:.6f}/{term}'] = np.asarray(mat)
    np.savez_compressed(path, **flat)


def _derive_groups(labels):
    """从标签自动推导连续流形组 [(start, end, name), ...]（分级粗标注用）。

    组名带角标：取标签开头到第一个空格的主干（'60S₁/₂ m_J=+0.5' →
    '60S₁/₂'；两体积态去外括号取首段），连续相同合并为一段。
    """
    groups = []
    for i, lab in enumerate(labels):
        head = str(lab).lstrip('(').split()[0] if str(lab).split() else str(lab)
        if groups and groups[-1][2] == head and groups[-1][1] == i:
            groups[-1][1] = i + 1
        else:
            groups.append([i, i + 1, head])
    return [(s, e, n) for s, e, n in groups]

# ============================================================
# 诊断快照 mock（t75a 接线前的 UI 开发数据；结构 = 生产 diag 契约）
# ============================================================

def mock_diag(N=6, n_t=200, t_end=620.0, seed=3):
    """构造与生产 diag 同构的演示数据（small N=6, P=3）。"""
    global _MOCK
    if _MOCK is not None:
        return _MOCK
    rng = np.random.default_rng(seed)
    P = 3
    dim = N * P
    E_bare = np.sort(rng.uniform(0, 60.0, N)) * 1000.0     # MHz
    D_full = np.tile(E_bare, P) + np.repeat([-17236.7, 0.0, 17236.7], N)

    a, b, v, dD, is_mw, wv = [], [], [], [], [], []
    for i in range(N - 1):
        for s in range(P):
            A = int(s * N + i)
            B = int(s * N + i + 1)
            d = float(D_full[A] - D_full[B])
            a.append(A); b.append(B)
            v.append(complex(rng.uniform(0.5, 8.0)))
            dD.append(d); is_mw.append(0); wv.append(0.0)
    for i in range(N - 1):
        for s in range(P):
            A = int(s * N + i)
            B = int(s * N + i + 1)
            d = float(D_full[A] - D_full[B])
            a.append(A); b.append(B)
            v.append(complex(rng.uniform(0.2, 2.0)))
            dD.append(d); is_mw.append(1); wv.append(17236.7)
    pairs = dict(a=np.array(a), b=np.array(b), v=np.array(v),
                 dD=np.array(dD), is_mw=np.array(is_mw), w=np.array(wv))

    t = np.linspace(0.0, t_end, n_t)
    env = np.exp(-((t - 0.4 * t_end) / (0.08 * t_end)) ** 2)
    dt = 0.05 * (1.0 + 3.0 * np.exp(-((t - 0.4 * t_end) / (0.05 * t_end)) ** 2)) \
        + rng.uniform(0, 0.01, n_t)

    sector_amps = {s: np.abs(rng.normal(0, 1.0, (n_t, N))) * env[:, None]
                   for s in range(P)}

    ts = [0.0, 0.25 * t_end, 0.4 * t_end, 0.6 * t_end]
    H_snaps = {}
    for t0 in ts:
        terms = {}
        terms['K0'] = np.diag(D_full.astype(complex))
        opt = np.zeros((dim, dim), complex)
        mw = np.zeros((dim, dim), complex)
        for k in range(len(a)):
            A, B, vv, d = a[k], b[k], v[k], dD[k]
            ph = np.exp(1j * 2 * np.pi * 1e-3 * d * t0)
            if is_mw[k]:
                mw[A, B] = env[min(int(t0 / t_end * (n_t - 1)), n_t - 1)] \
                    * vv * ph * np.cos(2 * np.pi * 1e-3 * wv[k] * t0)
                mw[B, A] = np.conj(mw[A, B])
            else:
                opt[A, B] = 0.5 * env[min(int(t0 / t_end * (n_t - 1)),
                                          n_t - 1)] * vv * ph
                opt[B, A] = np.conj(opt[A, B])
        terms['光场'] = opt
        terms['微波'] = mw
        H_snaps[float(t0)] = terms

    # 标注统一用 LaTeX 下标形式（'60S_{1/2}'）——与真实 diag 的 labels 同约定，
    # 由 viewer 的 _to_unicode_sub 转成 '60S 1/2'；离屏截图里 Unicode 下标
    # ₁₂₃ 在雅黑缺字会渲染成豆腐块，故 mock 数据不直接用它们。
    labels = ['60S_{1/2} +½', '60S_{1/2} −½', '60P_{3/2} −½', '6P_{3/2} F2 +1',
              '6P_{3/2} F2 0', '6P_{3/2} F2 −1'][:N]
    diag = dict(
        meta=dict(method='mixed', backend='mixed-phi-dop853',
                  N=N, P=P, dim=dim, t_end=t_end,
                  M=[1], r_crit=0.0135, stride=N, labels=labels,
                  n_pairs=len(a), tol=1e-8, max_step=0.05),
        photon=dict(Ms=[1], P=P, dim=dim, strides=[1], nvecs=np.arange(P),
                    D_full=D_full),
        acize=dict(r_crit=0.0135, shift=np.zeros(N),
                   fields=[dict(omega=295723721.9, n_before=3, n_kept=3,
                                n_acized=0, shift_min=0.0, shift_max=0.0)]),
        pairs=pairs,
        solver=dict(t_steps=t, dt=dt, nfev=182043, n_acc=6041, n_rej=118,
                    norm_dev=5.7e-7),
        trajectory=dict(t=t, sector_amps=sector_amps),
        H_snapshots=H_snaps,
    )
    _MOCK = diag
    return diag


# ============================================================
# 检视器主体
# ============================================================

class SolverInspector(QDialog):
    """非模态检视面板：load_diag(diag) 填充，各标签页独立刷新。

    上帝旋钮内嵌：per_freq_M/r_crit 在处理链页，tol/max_step 在求解过程页；
    「应用并重跑」发 rerun_requested 信号，由 DynamicPanel 触发新一轮演化。
    """

    rerun_requested = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle('求解检视器（上帝模式）')
        self.setModal(False)
        self.setWindowFlags(self.windowFlags() | Qt.WindowMinMaxButtonsHint)
        self.resize(1500, 960)          # v1.1：扩容约一倍
        self.setStyleSheet(ui_theme.QSS)
        self._diag = None

        lay = QVBoxLayout(self)
        lay.setContentsMargins(10, 10, 10, 10)
        # 顶行：meta 摘要（左）+ 导出按钮（右）——导出原本是一条占满整宽、
        # 顶在窗口最底的大按钮，既浪费一整行又和各页底部的说明挤在一起。
        top = QHBoxLayout()
        top.setSpacing(8)
        self.meta_label = QLabel('（未加载数据——运行演化后自动填充）')
        self.meta_label.setStyleSheet('color: %s; font-size:12px;'
                                      % ui_theme.COL_TEXT_DIM)
        # 摘要可能很长：别撑宽窗口（超出部分省略号截断，全文进 tooltip）
        self.meta_label.setSizePolicy(QSizePolicy.Ignored,
                                      QSizePolicy.Preferred)
        self.meta_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        top.addWidget(self.meta_label, 1)
        self.export_btn = QPushButton('⬇ 导出诊断')
        self.export_btn.setToolTip('当前诊断快照 → .npz（数值全量）+ .json（元数据）')
        self.export_btn.clicked.connect(self.export_diag)
        top.addWidget(self.export_btn, 0)
        lay.addLayout(top)

        self.tabs = QTabWidget()
        lay.addWidget(self.tabs, 1)
        self._tab_l = self._make_tab_pipeline()
        self._tab_h = self._make_tab_hamiltonian()
        self._tab_p = self._make_tab_pairs()
        self._tab_s = self._make_tab_solver()
        self._tab_x = self._make_tab_sectors()
        self._tab_e = self._make_tab_params()
        self.tabs.addTab(self._tab_l, '处理链')
        self.tabs.addTab(self._tab_h, '哈密顿量')
        self.tabs.addTab(self._tab_p, '耦合对')
        self.tabs.addTab(self._tab_s, '求解过程')
        self.tabs.addTab(self._tab_x, '扇区与重构')
        self.tabs.addTab(self._tab_e, '引擎参数')

    # ---------- 上帝旋钮（内嵌各页；主界面不再占用空间） ----------
    def _make_knob(self, attr, label, tip, width=90):
        lab = QLabel(label)
        lab.setToolTip(tip)
        e = QLineEdit('')
        e.setPlaceholderText('默认')
        e.setToolTip(tip)
        e.setFixedWidth(width)
        setattr(self, attr, e)
        return lab, e

    def _make_rerun_btn(self):
        btn = QPushButton('⟳ 应用并重跑')
        btn.setToolTip('按当前旋钮值重新运行演化（diag 同步开启）')
        btn.clicked.connect(self.rerun_requested.emit)
        return btn

    def set_god_params(self, params):
        """DynamicPanel 把当前覆写值填进旋钮（params: dict 或 None）。"""
        p = params or {}
        for key, attr in (('per_freq_M', 'god_perfreq_m'),
                          ('r_crit', 'god_rcrit'),
                          ('max_step', 'god_maxstep'),
                          ('tol', 'god_tol')):
            e = getattr(self, attr, None)
            if e is None:
                continue
            val = p.get(key)
            e.setText('' if val is None else str(val))
        # 引擎参数页覆写回填
        for kw, e in (getattr(self, '_e_edits', None) or {}).items():
            val = p.get(kw)
            e.setText('' if val is None else str(val))

    def god_values(self, pair=False):
        """读取旋钮 → evolve_mixed(_pair) 覆写 kwargs（空=默认不传）。"""
        ek = {}
        if not pair:
            m = self.god_perfreq_m.text().strip()
            if m:
                try:
                    ek['per_freq_M'] = (int(m) if '[' not in m
                                        else json.loads(m))
                except Exception:
                    QMessageBox.warning(self, '上帝参数',
                                        f'per_freq_M 解析失败: {m}')
                    return {}
        for attr, key in (('god_rcrit', 'r_crit'),
                          ('god_maxstep', 'max_step'),
                          ('god_tol', 'tol')):
            s = getattr(self, attr).text().strip()
            if s:
                try:
                    ek[key] = float(s)
                except ValueError:
                    QMessageBox.warning(self, '上帝参数',
                                        f'{key} 解析失败: {s}')
                    return {}
        # 引擎参数页覆写
        for kw, e in (getattr(self, '_e_edits', None) or {}).items():
            s = e.text().strip()
            if s:
                try:
                    ek[kw] = float(s)
                except ValueError:
                    QMessageBox.warning(self, '引擎参数',
                                        f'{kw} 解析失败: {s}')
                    return {}
        ek['diag'] = True
        return ek

    # ---------- 数据入口 ----------
    def set_placeholder(self, text):
        """无 diag 时的占位说明。"""
        self._diag = None
        self.meta_label.setText(text)
        for card in getattr(self, '_l_cards', []):
            card['detail'].setText('')
            card['big'].setText('')
        self.l_table.setRowCount(0)
        # 清空 viewer（无数据时显示提示）
        if hasattr(self, 'h_viewer'):
            self.h_viewer.set_data({}, {'dim': 0, 'labels': [], 'stride': 0},
                                    {})

    def load_diag(self, diag):
        self._diag = diag
        m = diag.get('meta', {})
        M = m.get('M')
        mtxt = f'M={M} ' if M else ''
        self.meta_label.setText(
            "method=%s  backend=%s   N=%s  P=%s  dim=%s  %st_end=%s ns"
            % (m.get('method', '?'), m.get('backend', '?'), m.get('N', '?'),
               m.get('P', '?'), m.get('dim', '?'), mtxt, m.get('t_end', '?')))
        # viewer 取数据
        if hasattr(self, 'h_viewer'):
            self.h_viewer.set_data(diag.get('H_snapshots') or {},
                                   m, diag.get('photon') or {})
        self._refresh_pipeline()
        self._refresh_hamiltonian()
        self._refresh_pairs()
        self._refresh_solver()
        self._refresh_sectors()

    # ---------- Tab 引擎参数（params.py 数值阈值覆写） ----------
    _E_PARAMS = [
        ('bw_opt', 'BW_OPT',
         'mixed φ-帧：光场/V_dd 共振对过滤带宽 |ΔD| < 值保留（MHz）。'
         '调小 → 更多对进 φ-帧精确计算（更准更慢）；调大 → 反旋过滤更激进。'
         '注意：漏滤 THz 对会把步长钉死'),
        ('mw_nres_bw', 'MW_NRES_BW_MHZ',
         '微波近共振保留带宽 | |ΔD|−ω| < 值（MHz）。默认覆盖超精细/精细流形'),
        ('mw_max', 'MW_MAX_MHZ',
         '微波/光场分界：载频 ≤ 值按微波处理（MHz）。默认 1e5（100 GHz）'),
    ]

    def _make_tab_params(self):
        from ..core import params as _params
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.setContentsMargins(10, 8, 10, 8)
        lay.setSpacing(8)
        info = QLabel(
            '引擎数值收敛/过滤阈值（定义于 <b>params.py</b>，默认值实时读取）。'
            '覆写仅对本次运行生效，不写回 params.py；清空 = 恢复默认。'
            '物理含义较强的参数在各自页：r_crit / per_freq_M → 处理链，'
            'tol / max_step → 求解过程。')
        info.setWordWrap(True)
        lay.addWidget(info)

        # 每个可覆写参数一张卡（与处理链页同一套视觉语言），三张 stretch=1
        # 平分主体。原先是一张 3 行表格被拉满整页，下面空一大片白。
        self._e_edits = {}
        for (kw, const, desc), accent in zip(
                self._E_PARAMS, ('#378ADD', '#BA7517', '#1D9E75')):
            card = QFrame()
            card.setObjectName('ecard%s' % kw)
            card.setStyleSheet(
                'QFrame#ecard%s{background:#FFFFFF; border:1px solid %s; '
                'border-radius:8px;}' % (kw, ui_theme.COL_BORDER))
            cl = QHBoxLayout(card)
            cl.setContentsMargins(10, 6, 12, 6)
            cl.setSpacing(12)
            bar = QFrame()
            bar.setFixedWidth(6)
            bar.setStyleSheet('background:%s; border:none; border-radius:3px;'
                              % accent)
            cl.addWidget(bar)
            # 左栏：kwarg 名 + 默认值 + params.py 常量名
            # 用布局直接挂在卡片里（不套 QWidget 包装）——套包装的话子控件
            # 不继承卡片白底，会在卡里显示一块灰底，且竖向被拉散。
            lbox = QVBoxLayout()
            lbox.setSpacing(1)
            lbox.addStretch(1)
            name = QLabel('<b style="font-size:14px;">%s</b>' % kw)
            name.setTextFormat(Qt.RichText)
            name.setFixedWidth(150)
            lbox.addWidget(name)
            dflt = QLabel('默认 %g' % float(getattr(_params, const)))
            dflt.setStyleSheet('color:%s; font-size:9pt;'
                               % ui_theme.COL_TEXT_DIM)
            dflt.setFixedWidth(150)
            lbox.addWidget(dflt)
            cst = QLabel('params.%s' % const)
            cst.setStyleSheet('color:%s; font-size:8pt;'
                              % ui_theme.COL_TEXT_DIM)
            cst.setFixedWidth(150)
            lbox.addWidget(cst)
            lbox.addStretch(1)
            cl.addLayout(lbox)
            # 中栏：运行级覆写输入
            e = QLineEdit('')
            e.setFixedWidth(120)
            e.setPlaceholderText('空 = 默认')
            e.setToolTip('运行级覆写：仅本次运行生效，不写回 params.py')
            self._e_edits[kw] = e
            cl.addWidget(e, 0, Qt.AlignVCenter)
            # 右栏：说明（占满剩余宽度）
            d = QLabel(desc)
            d.setWordWrap(True)
            d.setStyleSheet('color:%s;' % ui_theme.COL_TEXT_DIM)
            cl.addWidget(d, 1)
            lay.addWidget(card, 1)
        return w

    # ---------- 诊断导出 ----------
    def export_diag(self):
        d = self._diag
        if not d:
            QMessageBox.information(self, '导出诊断',
                                    '当前没有诊断数据——先用上帝模式运行演化。')
            return
        from PyQt5.QtWidgets import QFileDialog
        path, _ = QFileDialog.getSaveFileName(self, '导出诊断', 'mixed_diag.npz',
                                              'NPZ (*.npz)')
        if not path:
            return
        try:
            save_diag_npz(d, path)
            QMessageBox.information(self, '导出诊断', f'已保存：\n{path}')
        except Exception as e:
            QMessageBox.warning(self, '导出失败', str(e))

    # ---------- Tab 处理链（大卡片主体） ----------
    def _make_tab_pipeline(self):
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.setContentsMargins(10, 8, 10, 8)
        lay.setSpacing(8)

        # 上帝旋钮行（r_crit ↔ ②AC 化；per_freq_M ↔ ③光子块）
        knob = QFrame()
        knob.setObjectName('godKnobs')
        knob.setStyleSheet(
            'QFrame#godKnobs{background:%s; border:1px solid %s; '
            'border-radius:6px;}' % (ui_theme.COL_HEAD, ui_theme.COL_BORDER))
        kl = QHBoxLayout(knob)
        kl.setContentsMargins(8, 6, 8, 6)
        kl.setSpacing(6)
        kl.addWidget(QLabel('<b>精度旋钮</b>'))
        for attr, label, tip in (
                ('god_rcrit', 'r_crit',
                 'AC 化阈值 r=Ω/ω（空=默认 0.0135；越小越多对进光子块精确计算）'),
                ('god_perfreq_m', 'per_freq_M',
                 '逐载频光子块阶数 M（int 或 [M1,…]；空=自动探针；两体固定 3 扇区）')):
            lab, e = self._make_knob(attr, label, tip)
            kl.addWidget(lab)
            kl.addWidget(e)
        kl.addStretch(1)
        kl.addWidget(self._make_rerun_btn())
        lay.addWidget(knob)

        # 五步大卡片（平分主体空间——处理链是本页的主角）
        self._l_cards = []
        defs = [('①', '#378ADD', '输入', '#E6F1FB', '#185FA5'),
                ('②', '#D85A30', 'AC 化', '#FAECE7', '#993C1D'),
                ('③', '#534AB7', '光子块', '#EEEDFE', '#3C3489'),
                ('④', '#BA7517', '耦合对打包', '#FAEEDA', '#854F0B'),
                ('⑤', '#1D9E75', 'φ-帧积分', '#E1F5EE', '#0F6E56')]
        for badge, accent, title, bg, big_color in defs:
            card = QFrame()
            card.setObjectName(f'lcard{badge}')
            card.setStyleSheet(
                f'QFrame#lcard{badge}{{background:{bg}; '
                f'border:1px solid {ui_theme.COL_BORDER}; border-radius:8px;}}')
            cl = QHBoxLayout(card)
            cl.setContentsMargins(10, 4, 12, 4)
            cl.setSpacing(12)
            bar = QFrame()
            bar.setFixedWidth(6)
            bar.setStyleSheet(f'background:{accent}; border:none; '
                              'border-radius:3px;')
            cl.addWidget(bar)
            t = QLabel(f'<b style="font-size:15px;">{badge} {title}</b>')
            t.setTextFormat(Qt.RichText)
            t.setFixedWidth(150)
            cl.addWidget(t)
            detail = QLabel('')
            detail.setTextFormat(Qt.RichText)
            detail.setWordWrap(True)
            cl.addWidget(detail, 1)
            big = QLabel('')
            big.setStyleSheet(f'font-size:19px; font-weight:600; '
                              f'color:{big_color};')
            big.setMinimumWidth(200)
            big.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            cl.addWidget(big)
            lay.addWidget(card, 1)             # 五卡平分主体空间
            self._l_cards.append(dict(detail=detail, big=big))

        head = QLabel('<b>AC 化明细</b>（远强对剔除，r=Ω/ω 判据）')
        lay.addWidget(head)
        self.l_table = QTableWidget(0, 5)
        self.l_table.setHorizontalHeaderLabels(
            ['载频 MHz', '剔除前对数', '保留对数', '剔除对数', '位移范围 MHz'])
        self.l_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.Stretch)
        self.l_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.l_table.setFixedHeight(140)
        lay.addWidget(self.l_table)
        return w

    def _refresh_pipeline(self):
        d = self._diag
        if not d:
            return
        m = d.get('meta', {})
        ac = d.get('acize', {})
        pr = d.get('pairs', {})
        sv = d.get('solver', {})
        n_mw = int((np.asarray(pr.get('is_mw', [])) == 1).sum()) if pr else 0
        n_all = len(np.asarray(pr.get('a', []))) if pr else 0
        dO = np.abs(np.asarray(pr.get('dD', []), float)) if pr else np.array([])
        n_thz = int((dO > 2e4).sum()) if dO.size else 0
        sh = np.asarray(ac.get('shift', []), float)
        sh_txt = (f'{sh.min():+.1f}…{sh.max():+.1f}' if sh.size else '—')
        M = m.get('M')
        is_pair = 'pair' in str(m.get('backend', ''))
        ac_txt = ('far_sos 位移不注入（t70 定案）' if is_pair
                  else 'far_sos 位移注入物理能级')
        M_src = '手动覆写' if m.get('M_user') else '自动探针'
        warn = ('  <span style="background:#FCEBEB;color:#A32D2D;">&nbsp;⚠ '
                'THz 残留：检查过滤带宽&nbsp;</span>' if n_thz else '')
        nd = sv.get('norm_dev')
        nd_txt = ('%.1e' % nd if isinstance(nd, float) else '—')
        n_ac = sum(f.get('n_acized', 0) for f in (ac.get('fields') or []))

        def kv(k, v, hi=False):
            vv = f'<b>{v}</b>' if hi else str(v)
            return f'<span style="color:#6b7280;">{k}</span> {vv}'

        details = [
            kv('method', m.get('method'), True) + '  ' +
            kv('N', m.get('N')) + '  ' +
            kv('t_end', f"{m.get('t_end')} ns") + '  ' +
            kv('tol', m.get('tol')) + '  ' + kv('max_step', m.get('max_step')),
            kv('r_crit', m.get('r_crit'), True) + '  ' +
            kv('位移', sh_txt + ' MHz') +
            f'  <span style="color:#6b7280;">{ac_txt}</span>',
            kv('M', M, True) +
            f'  <span style="color:#6b7280;">[{M_src}]</span>  ' +
            kv('扇区 P', m.get('P'), True) + '  ' +
            kv('扩展维 dim', m.get('dim'), True) +
            '  <span style="color:#6b7280;">（dim 决定计算量）</span>',
            kv('微波', n_mw) + '  ' + kv('光场/V_dd', n_all - n_mw) + '  ' +
            kv('|ΔD|>20GHz 残留', n_thz, n_thz > 0) + warn,
            kv('nfev', sv.get('nfev', '—')) + '  ' +
            kv('接受/拒绝',
               f"{sv.get('n_acc', '—')}/{sv.get('n_rej', '—')}"),
        ]
        bigs = [
            f'{m.get("N")} 态 · {m.get("t_end")} ns',
            f'剔除 {n_ac} 对',
            f'dim = {m.get("dim")}',
            f'{n_all} 对',
            f'norm {nd_txt}',
        ]
        for card, dtxt, btxt in zip(self._l_cards, details, bigs):
            card['detail'].setText(dtxt)
            card['big'].setText(btxt)
        fields = ac.get('fields', []) or []
        tab = self.l_table
        tab.setRowCount(len(fields))
        for r, f in enumerate(fields):
            smin, smax = f.get('shift_min', 0.0), f.get('shift_max', 0.0)
            tab.setItem(r, 0, QTableWidgetItem('%.6g' % f.get('omega', 0)))
            tab.setItem(r, 1, QTableWidgetItem(str(f.get('n_before', 0))))
            tab.setItem(r, 2, QTableWidgetItem(str(f.get('n_kept', 0))))
            tab.setItem(r, 3, QTableWidgetItem(str(f.get('n_acized', 0))))
            tab.setItem(r, 4, QTableWidgetItem(f'{smin:+.1f}…{smax:+.1f}'))

    # ---------- Tab 哈密顿量（HamiltonianViewer：场景层自绘标注 + hover + 色条） ----------
    def _make_tab_hamiltonian(self):
        from .hamiltonian_viewer import HamiltonianViewer
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.setContentsMargins(10, 8, 10, 8)
        lay.setSpacing(6)

        # 解析式（点击文字中的项名可切换勾选；与热图配色一致）
        self.h_formula = QLabel('')
        self.h_formula.setTextFormat(Qt.RichText)
        self.h_formula.setWordWrap(True)
        self.h_formula.setStyleSheet(
            'QLabel{font-family:Microsoft YaHei, sans-serif; font-size:19px; '
            'background:%s; border:1px solid %s; border-radius:6px; '
            'padding:6px 8px;}' % (ui_theme.COL_HEAD, ui_theme.COL_BORDER))
        lay.addWidget(self.h_formula)

        mid = QHBoxLayout()
        mid.setSpacing(8)
        # 新 viewer：自带工具条（slider + 项 + 视图控制）+ 热图 + 色条 + hover
        self.h_viewer = HamiltonianViewer(self)
        mid.addWidget(self.h_viewer, 3)
        # 右侧：矩阵元表 —— 标题/表/摘要收进一张带边框的卡片，
        # 否则三者各自浮在白底上，整页看起来"散"（用户反馈）。
        card = QFrame()
        card.setObjectName('hElemCard')
        card.setStyleSheet(
            'QFrame#hElemCard{background:%s; border:1px solid %s; '
            'border-radius:6px;}' % (ui_theme.COL_PANEL, ui_theme.COL_BORDER))
        right = QVBoxLayout(card)
        right.setContentsMargins(8, 6, 8, 6)
        right.setSpacing(5)
        _title = QLabel('矩阵元 |H|（按幅值排序）')
        _title.setStyleSheet(
            'QLabel{border:none; color:%s; font-size:10pt; font-weight:600;}'
            % ui_theme.COL_TEXT)
        right.addWidget(_title)
        self.h_table = QTableWidget(0, 4)
        self.h_table.setFrameShape(QFrame.NoFrame)      # 边框由卡片提供
        self.h_table.setStyleSheet(
            'QTableWidget{background:%s; border:none;'
            ' alternate-background-color:%s;}'
            'QHeaderView::section{background:%s; color:%s; border:none;'
            ' border-bottom:1px solid %s; padding:3px 4px;}'
            'QTableWidget::item{border-bottom:1px solid %s;}'
            % (ui_theme.COL_PANEL, ui_theme.COL_ROW_ALT, ui_theme.COL_HEAD,
               ui_theme.COL_TEXT, ui_theme.COL_BORDER, ui_theme.COL_GRID))
        self.h_table.setShowGrid(False)
        self.h_table.setAlternatingRowColors(True)
        self.h_table.setHorizontalHeaderLabels(
            ['(a,b)', '|H| MHz', 'arg H', '物理含义'])
        self.h_table.horizontalHeader().setSectionResizeMode(
            3, QHeaderView.Stretch)
        self.h_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        right.addWidget(self.h_table, 1)
        self.h_info = QLabel('')
        self.h_info.setWordWrap(True)
        self.h_info.setStyleSheet(
            'QLabel{border:none; color:%s; font-size:9pt; '
            'font-family:Consolas,monospace;}' % ui_theme.COL_TEXT_DIM)
        right.addWidget(self.h_info)
        mid.addWidget(card, 2)
        lay.addLayout(mid, 1)
        # viewer 数据变化 → 刷公式 + 表
        self.h_viewer.dataChanged.connect(self._refresh_hamiltonian)
        return w

    def _h_reset_view(self):
        """⟲ 还原视图（viewer 内部有同名按钮，这里作兼容占位）。"""
        if hasattr(self, 'h_viewer'):
            self.h_viewer._reset_view()

    def _formula_html(self, present):
        """解析式富文本：出现的分项按颜色着色，未出现的灰色。"""
        spans = {
            '光场': 'Σ<sub>光场</sub> env(t)·K<sub>c</sub>∘e<sup>iCΔD·t</sup>',
            '微波': 'Σ<sub>微波</sub> env(t)·cos(ωt)·A∘e<sup>iCΔD·t</sup>',
            'V_dd': 'V<sub>dd</sub>',
            'AC对角': 'AC 对角 env²',
            '耗散': '−iΓ/2',
        }
        parts = []
        for name, txt in spans.items():
            color = TERM_COLORS[name] if name in present else '#b4b2a9'
            parts.append(f'<span style="color:{color};">{txt}</span>')
        head = ('<span style="color:%s;">dφ/dt = −iC·[ K<sub>0</sub> 对角'
                '</span> + ' % TERM_COLORS['K0'])
        return head + ' + '.join(parts) + ' ]·φ　（项颜色 = viewer 内勾选项）'

    def _refresh_hamiltonian(self):
        """viewer 状态变化时（slider/项/去对角）→ 刷公式 + 表 + info。"""
        d = self._diag
        if d is None:
            return
        snaps = d.get('H_snapshots') or {}
        if not snaps:
            return
        # 解析式（按当前 t 快照的项着色）
        ts = self.h_viewer._ts_sorted
        i = max(0, min(self.h_viewer.t_slider.value(), len(ts) - 1))
        terms = snaps[ts[i]]
        self.h_formula.setText(self._formula_html(sorted(terms.keys())))
        # 矩阵元表
        rows = self.h_viewer.top_matrix_elements(30)
        tab = self.h_table
        tab.setRowCount(len(rows))
        for r, (a, b, mag, arg, mean) in enumerate(rows):
            tab.setItem(r, 0, QTableWidgetItem(f'({a},{b})'))
            tab.setItem(r, 1, QTableWidgetItem('%.4g' % mag))
            tab.setItem(r, 2, QTableWidgetItem(f'{arg:+.2f}π'))
            tab.setItem(r, 3, QTableWidgetItem(mean))
        # 摘要
        if rows:
            self.h_info.setText(
                f'dim={self.h_viewer._dim}  非零元=—  '
                f'最大|H|={rows[0][2]:.4g} MHz')
        else:
            self.h_info.setText('当前叠加项无矩阵元')

    # ---------- Tab 耦合对 ----------
    def _make_tab_pairs(self):
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.setContentsMargins(10, 8, 10, 8)
        mid = QHBoxLayout()
        self.p_fig = Figure(figsize=(6.2, 5.6), dpi=100,
                            constrained_layout=True)
        self.p_canvas = FigureCanvas(self.p_fig)
        mid.addWidget(self.p_canvas, 2)
        self.p_table = QTableWidget(0, 7)
        self.p_table.setHorizontalHeaderLabels(
            ['a', 'b', '|v| MHz', 'ΔD MHz', '类型', '载频 MHz', '物理含义'])
        # 前六列按内容定宽，余下宽度全给「物理含义」——原先它被挤到只剩
        # 几十像素，每个态名都以省略号截断（与哈密顿页同一类问题）。
        for c, cw in enumerate((48, 48, 78, 88, 68, 92)):
            self.p_table.setColumnWidth(c, cw)
        self.p_table.horizontalHeader().setSectionResizeMode(
            6, QHeaderView.Stretch)
        self.p_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        mid.addWidget(self.p_table, 3)
        lay.addLayout(mid, 1)
        self.p_info = QLabel('')
        self.p_info.setWordWrap(True)
        lay.addWidget(self.p_info)
        return w

    def _refresh_pairs(self):
        d = self._diag
        pr = (d or {}).get('pairs')
        if not pr:
            return
        m = (d or {}).get('meta', {})
        labels = m.get('labels') or []
        stride = int(m.get('stride') or 0)
        a, b, v, dD = pr['a'], pr['b'], pr['v'], pr['dD']
        is_mw, wv = pr['is_mw'], pr['w']
        fig = self.p_fig
        fig.clear()
        ax = fig.add_subplot(111)
        dO = np.abs(dD)
        pos = dO[dO > 0]
        if pos.size:
            lo, hi = float(pos.min()), float(pos.max())
            if hi <= lo * 10.0:              # 单一量级 → 对数区间退化，展开
                lo, hi = lo * 0.5, hi * 2.0
            bins = np.logspace(np.log10(max(lo, 1e-2)),
                               np.log10(max(hi, 1e2)), 40)
        else:
            bins = np.logspace(-2, 2, 40)
        for flag, lab, color in ((0, '光场/V_dd', '#378ADD'),
                                 (1, '微波', '#BA7517')):
            sel = is_mw == flag
            if sel.any():
                ax.hist(np.abs(dD[sel]), bins=bins, alpha=0.75,
                        label=lab, color=color)
        ax.set_xscale('log')
        # 计数直方图：量级跨度不大就用线性 + 整数刻度——对数轴会把 6/10/20
        # 这类整数计数画成 6×10⁰ / 10¹ / 2×10¹ 的怪刻度，反而难读。
        cnts, _ = np.histogram(np.abs(dD), bins=bins)
        nz = cnts[cnts > 0]
        if nz.size and float(nz.max()) / max(float(nz.min()), 1.0) > 50.0:
            ax.set_yscale('log')
        else:
            ax.set_yscale('linear')
            ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        ax.axvline(2e4, color='#A32D2D', ls='--', lw=1, label='BW_OPT=20GHz')
        ax.set_xlabel('|ΔD| MHz（对数）')
        ax.set_ylabel('耦合对数')
        ax.set_title('打包耦合对 ΔD 分布（THz 残留 = 步长钉死隐患）')
        ax.legend(fontsize=8)
        self.p_canvas.draw_idle()

        order = np.argsort(-np.abs(v))[:60]
        tab = self.p_table
        tab.setRowCount(len(order))
        for r, k in enumerate(order):
            k = int(k)
            tab.setItem(r, 0, QTableWidgetItem(str(int(a[k]))))
            tab.setItem(r, 1, QTableWidgetItem(str(int(b[k]))))
            tab.setItem(r, 2, QTableWidgetItem('%.4g' % abs(v[k])))
            tab.setItem(r, 3, QTableWidgetItem('%.4g' % abs(dD[k])))
            tab.setItem(r, 4, QTableWidgetItem(
                '微波' if is_mw[k] else '光/Vdd'))
            tab.setItem(r, 5, QTableWidgetItem(
                '%.6g' % wv[k] if wv[k] else '—'))
            mean_txt = '—'
            if labels and stride:
                pa, ia = divmod(int(a[k]), stride)
                pb, ib = divmod(int(b[k]), stride)
                la = labels[ia] if ia < len(labels) else str(ia)
                lb = labels[ib] if ib < len(labels) else str(ib)
                mean_txt = f'{lb}|n={pb - 1} ↔ {la}|n={pa - 1}'
            tab.setItem(r, 6, QTableWidgetItem(mean_txt))
        n_mw = int((is_mw == 1).sum())
        n_thz = int((dO > 2e4).sum())
        self.p_info.setText(
            '共 %d 对（微波 %d，光场/V_dd %d）；|ΔD|>BW_OPT 残留 %d 对%s'
            % (len(a), n_mw, len(a) - n_mw, n_thz,
               '（⚠ 需检查过滤带宽）' if n_thz else '——正常'))

    # ---------- Tab 求解过程 ----------
    def _make_tab_solver(self):
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.setContentsMargins(10, 8, 10, 8)
        lay.setSpacing(6)
        knob = QFrame()
        knob.setStyleSheet(
            'QFrame{background:%s; border:1px solid %s; border-radius:6px;}'
            % (ui_theme.COL_HEAD, ui_theme.COL_BORDER))
        kl = QHBoxLayout(knob)
        kl.setContentsMargins(8, 6, 8, 6)
        kl.setSpacing(6)
        kl.addWidget(QLabel('<b>积分旋钮</b>'))
        for attr, label, tip in (
                ('god_tol', 'tol',
                 'DOP853 相对容差（空=默认 1e-8；每步允许的相对误差）'),
                ('god_maxstep', 'max_step',
                 'DOP853 最大步长 ns（空=默认 0.05）')):
            lab, e = self._make_knob(attr, label, tip)
            kl.addWidget(lab)
            kl.addWidget(e)
        kl.addStretch(1)
        kl.addWidget(self._make_rerun_btn())
        lay.addWidget(knob)
        self.s_fig = Figure(figsize=(10.5, 4.4), dpi=100,
                            constrained_layout=True)
        self.s_canvas = FigureCanvas(self.s_fig)
        lay.addWidget(self.s_canvas, 1)
        self.s_info = QLabel('')
        self.s_info.setWordWrap(True)
        lay.addWidget(self.s_info)
        return w

    def _refresh_solver(self):
        d = self._diag
        sv = (d or {}).get('solver')
        if not sv:
            return
        t, dt = sv.get('t_steps'), sv.get('dt')
        fig = self.s_fig
        fig.clear()
        if t is not None and dt is not None and len(t):
            ax = fig.add_subplot(111)
            ax.semilogy(t, dt, lw=0.8, color='#185FA5')
            ax.set_xlabel('t (ns)')
            ax.set_ylabel('接受步长 dt (ns)')
            ax.set_title('DOP853 步长历史（凹谷 = stiff 段 ≈ 波形边沿/共振区）')
            ax.grid(alpha=0.3)
        self.s_canvas.draw_idle()
        self.s_info.setText(
            'nfev=%s   接受步=%s   拒绝步=%s（拒绝率 %.1f%%）   '
            'max|norm−1|=%.2e' % (
                sv.get('nfev', '?'), sv.get('n_acc', '?'),
                sv.get('n_rej', '?'),
                100.0 * sv.get('n_rej', 0) / max(sv.get('n_acc', 1), 1),
                sv.get('norm_dev', float('nan'))))

    # ---------- Tab 扇区与重构 ----------
    def _make_tab_sectors(self):
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.setContentsMargins(10, 8, 10, 8)
        lay.setSpacing(6)
        row = QHBoxLayout()
        row.addWidget(QLabel('每扇区显示前'))
        self.x_topn = QSpinBox()
        self.x_topn.setRange(0, 500)
        self.x_topn.setValue(12)
        self.x_topn.setToolTip('按平均幅值取前 N 条曲线（0=全部）；'
                               '两体场景每扇区 N² 条，建议 8~16')
        self.x_topn.valueChanged.connect(self._refresh_sectors)
        row.addWidget(self.x_topn)
        row.addWidget(QLabel('条（0=全部，按平均幅值排序）'))
        row.addStretch(1)
        lay.addLayout(row)
        self.x_fig = Figure(figsize=(10.5, 5.4), dpi=100,
                            constrained_layout=True)
        self.x_canvas = FigureCanvas(self.x_fig)
        lay.addWidget(self.x_canvas, 1)
        self.x_info = QLabel('')
        self.x_info.setWordWrap(True)
        lay.addWidget(self.x_info)
        return w

    def _refresh_sectors(self):
        d = self._diag
        tr = (d or {}).get('trajectory')
        if not tr:
            return
        t = tr['t']
        amps = tr['sector_amps']
        fig = self.x_fig
        fig.clear()
        ax = fig.add_subplot(111)
        cmap = {0: '#185FA5', 1: '#0F6E56', 2: '#854F0B', 3: '#A32D2D',
                4: '#534AB7'}
        topn = self.x_topn.value() if getattr(self, 'x_topn', None) else 0
        n_shown = 0
        for s, arr in sorted(amps.items()):
            P = len(amps)
            lab = ('n=%+d' % (s - (P - 1) // 2) if P % 2 == 1
                   else '扇区 %d' % s)
            order = np.argsort(-arr.mean(axis=0))
            cols = order[:topn] if topn > 0 else order
            first = True
            for i in cols:
                ax.plot(t, arr[:, i], lw=0.7, alpha=0.7,
                        color=cmap.get(s, '#5F5E5A'),
                        label=lab if first else None)
                first = False
                n_shown += 1
        ax.set_xlabel('t (ns)')
        ax.set_ylabel('|X| 扇区幅值')
        ax.set_title('扩展空间逐扇区幅值（相干扇和 = 物理布居）')
        # 图例放左下：右上角被 n=−1（t=0 处 ≈1.0）和 n=0 的抬升占住，
        # 原来贴在右上会压在曲线身上；左下 t<30 区间各扇区都接近 0，最空。
        ax.legend(fontsize=8, loc='lower left', framealpha=0.9)
        ax.grid(alpha=0.3)
        self.x_canvas.draw_idle()
        m = (d or {}).get('meta', {})
        self.x_info.setText(
            'P=%s 扇区 × N=%s 态，显示 %s 条；输出重构 '
            'ψ=Σ_p e^{−iC·n_p·ω·t}·X_p（t73b 相干扇和）'
            % (m.get('P', '?'), m.get('N', '?'), n_shown))
