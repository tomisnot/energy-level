# -*- coding: utf-8 -*-
"""场定义面板：静磁场 + 静电场 + 光场表格 + 关注跃迁 + 初态。

物理模型定义（单体和两体共用同一份"场"）：
- 静磁场 B (G) / 静电场 E (V/cm)：一行，产生 Zeeman / DC-Stark 能移
- 光场：QTableWidget（# | 频率 GHz | 光强 mW/cm² | 偏振），参考谱表样式，
  右键删除；双击编辑频率/光强，偏振下拉。
- 关注跃迁 combobox：选项 = 已创建光场 +（单体页）exchange。
  选某份光 → 该光为响应谱扫频对象（其余作能移）；选 exchange → 不画响应谱，
  画旧版两能谱对比。由 rebuild_focus(show_exchange) 按模式刷新。
- 初态：关注跃迁 行右侧 n/l/j/m spin（与"添加能级"同款控件），两页响应谱共用。
- 右键频率单元格 → 从跃迁对 + 失谐导出频率；右键强度单元格 → 扫描光强实现目标频移。
- 作用范围：scope 默认 'both'（表格第 5 列可选 atom1/atom2）。
"""
import numpy as np
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtWidgets import *

from . import core


class FieldPanel(QWidget):
    """静磁场 B + 静电场 E + 光场表格 + 关注跃迁 + 初态。"""

    changed = pyqtSignal()
    POLS = ['σ+', 'π', 'σ-', '椭圆偏']

    def __init__(self, page, parent=None):
        """page：MainWindow（提供 cache/groups 等）。"""
        super().__init__(parent)
        self.page = page
        self._ellip = {}        # 光场行 → 椭圆偏振参数 dict(alpha_deg, theta_deg, eps_deg)
        self._ellip_dlg = None  # 非模态椭圆参数弹窗引用（防 GC）
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(4)
        # ---- 静磁场 + 静电场 + 添加光 一行 ----
        row = QHBoxLayout()
        row.addWidget(QLabel('磁场 B (G)'))
        self.b_spin = QDoubleSpinBox()
        self.b_spin.setRange(0.0, 1000000.0); self.b_spin.setDecimals(2)
        self.b_spin.setValue(21.7)
        self.b_spin.setToolTip('静磁场（量子化轴 z）：三页共用，Zeeman 能移')
        self.b_spin.setFixedWidth(90)              # 紧凑：避免大 range 撑宽
        row.addWidget(self.b_spin)
        row.addWidget(QLabel('电场 E (V/cm)'))
        self.e_spin = QDoubleSpinBox()
        self.e_spin.setRange(-1e6, 1e6); self.e_spin.setDecimals(3)
        self.e_spin.setValue(0.0)
        self.e_spin.setToolTip('静电场（DC-Stark 二阶能移）：三页共用')
        self.e_spin.setFixedWidth(90)
        row.addWidget(self.e_spin)
        row.addStretch(1)
        self.add_btn = QPushButton('＋添加光')
        self.add_btn.setMinimumWidth(110)
        self.add_btn.setToolTip(
            '添加一份光（双击表格单元格改频率/光强；右键删除；\n'
            '作用范围：两原子都打 / 只打 1 号 / 只打 2 号）')
        self.add_btn.clicked.connect(self.add_light)
        row.addWidget(self.add_btn)
        lay.addLayout(row)
        self.b_spin.valueChanged.connect(self._changed)
        self.e_spin.editingFinished.connect(self._changed)
        # ---- 光场表格 + 关注跃迁 + 初态：独立容器，按页显隐（B/E 恒显） ----
        self.lights_box = QWidget()
        lb = QVBoxLayout(self.lights_box)
        lb.setContentsMargins(0, 0, 0, 0)
        lb.setSpacing(4)
        # ---- 光场表格 ----
        self.table = QTableWidget(0, 5)
        self.table.setHorizontalHeaderLabels(
            ['#', '频率 (GHz)', '光强 (mW/cm²)', '偏振', '作用范围'])
        self.table.verticalHeader().setVisible(False)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        # 单击选中即可编辑（AllEditTriggers：单击/双击/F2/键入都进入编辑）
        self.table.setEditTriggers(QAbstractItemView.AllEditTriggers)
        hh = self.table.horizontalHeader()
        hh.setSectionResizeMode(QHeaderView.Stretch)
        hh.setSectionResizeMode(0, QHeaderView.Fixed)
        hh.setSectionResizeMode(3, QHeaderView.Fixed)
        hh.setSectionResizeMode(4, QHeaderView.Fixed)
        self.table.setColumnWidth(0, 32)
        self.table.setColumnWidth(3, 56)
        self.table.setColumnWidth(4, 100)
        self.table.verticalHeader().setDefaultSectionSize(22)
        self.table.setMinimumHeight(150)       # 可随窗口拉伸
        self.table.itemChanged.connect(self._table_changed)
        self.table.itemSelectionChanged.connect(self._changed)
        self.table.setContextMenuPolicy(Qt.CustomContextMenu)
        self.table.customContextMenuRequested.connect(self._table_menu)
        lb.addWidget(self.table)
        # ---- 关注跃迁（独立一行：下拉自然宽度，不再与量子数抢空间）----
        row = QHBoxLayout()
        row.addWidget(QLabel('关注跃迁'))
        self.focus_box = QComboBox()
        self.focus_box.setToolTip(
            '选择关注哪个跃迁：某份光 → 响应谱扫频对象（其余光作能移）；\n'
            '单体页的 exchange → 旧版两能谱对比（不画响应谱）')
        self.focus_box.currentIndexChanged.connect(self._changed)
        row.addWidget(self.focus_box, 1)
        lb.addLayout(row)
        # ---- 初态（n/l/j/m 一行，固定宽紧凑对齐，不被下拉挤压）----
        row = QHBoxLayout()
        row.addWidget(QLabel('初态'))
        self.psi_n = QSpinBox(); self.psi_n.setRange(1, 200); self.psi_n.setValue(60)
        self.psi_l = QSpinBox(); self.psi_l.setRange(0, 199); self.psi_l.setValue(0)
        self.psi_j = QDoubleSpinBox(); self.psi_j.setRange(0.0, 10.0)
        self.psi_j.setSingleStep(0.5); self.psi_j.setDecimals(1); self.psi_j.setValue(0.5)
        self.psi_m = QDoubleSpinBox(); self.psi_m.setRange(-10.0, 10.0)
        self.psi_m.setSingleStep(0.5); self.psi_m.setDecimals(1); self.psi_m.setValue(0.5)
        for lab, w in (('n', self.psi_n), ('l', self.psi_l),
                       ('j', self.psi_j), ('m', self.psi_m)):
            w.setFixedWidth(60)                 # 固定宽：整行对齐、不随内容变宽
            row.addWidget(QLabel(lab))
            row.addWidget(w)
        row.addStretch(1)
        lb.addLayout(row)
        for w in (self.psi_n, self.psi_l, self.psi_j, self.psi_m):
            w.editingFinished.connect(self._changed)
        lay.addWidget(self.lights_box)

    def set_lights_visible(self, show):
        """控制光场表格/关注跃迁/初态区显隐（B/E 行恒显，三页共用）。"""
        self.lights_box.setVisible(show)

    # ---------------- 值接口 ----------------
    def _changed(self, *_):
        self.changed.emit()

    def _table_changed(self, *_):
        self.rebuild_focus(self._focus_exchange)
        self.changed.emit()

    def b_gauss(self):
        return float(self.b_spin.value())

    def e_vpcm(self):
        return float(self.e_spin.value())

    # ---------------- 光场表格 ----------------
    def _pol_text(self, r):
        """第 r 份光的偏振文本（兼容 QComboBox / 残留 QLineEdit 编辑器）。"""
        w = self.table.cellWidget(r, 3)
        if w is None:
            return 'π'
        if isinstance(w, QComboBox):
            return w.currentText()
        return w.text().strip() or 'π'   # QLineEdit 编辑器残留

    def add_light(self):
        r = self.table.rowCount()
        self.table.insertRow(r)
        it = QTableWidgetItem(str(r + 1))
        it.setFlags(Qt.ItemIsEnabled)          # 序号只读
        self.table.setItem(r, 0, it)
        self.table.setItem(r, 1, QTableWidgetItem('295724.048'))
        self.table.setItem(r, 2, QTableWidgetItem('126849894.0'))
        # 偏振 / 作用范围列：QComboBox 常驻，其下的 item 不可编辑
        # （否则 AllEditTriggers 单击会启动 QLineEdit 编辑器并顶掉 combo）
        noedit = Qt.ItemIsEnabled | Qt.ItemIsSelectable
        it_pol = QTableWidgetItem('')
        it_pol.setFlags(noedit)
        self.table.setItem(r, 3, it_pol)
        pol = QComboBox()
        pol.addItems(self.POLS)
        pol.setCurrentText('π')
        pol.currentTextChanged.connect(lambda t, r=r: self._on_pol_selected(r, t))
        self.table.setCellWidget(r, 3, pol)
        it_scope = QTableWidgetItem('')
        it_scope.setFlags(noedit)
        self.table.setItem(r, 4, it_scope)
        scope = QComboBox()
        scope.addItems(['both', 'atom1', 'atom2'])
        scope.setCurrentText('both')
        scope.setToolTip('作用范围：两原子都打 / 只打 1 号 / 只打 2 号')
        scope.currentTextChanged.connect(self._table_changed)
        self.table.setCellWidget(r, 4, scope)
        self._table_changed()

    # ---------------- 椭圆偏振 ----------------
    def _on_pol_selected(self, r, text):
        """偏振下拉变化：选"椭圆偏"→ 打开参数弹窗；选标准偏振 → 清除椭圆参数。"""
        if text == '椭圆偏':
            self._open_ellip_dialog(r)
        else:
            self._ellip.pop(r, None)
            self._set_pol_tooltip(r)
        self._table_changed()

    def _open_ellip_dialog(self, r):
        """打开椭圆偏振参数弹窗（非模态：改参数回车即实时触发重算）。"""
        cur = self._ellip.get(r)
        if cur is None:
            cur = dict(alpha_deg=0.0, theta_deg=0.0, eps_deg=0.0)
            self._ellip[r] = cur
        self._ellip_dlg = EllipDialog(
            self, cur,
            lambda a, t, e, r=r: self._apply_ellip(r, a, t, e))
        self._ellip_dlg.show()

    def _apply_ellip(self, r, alpha, theta, eps):
        """椭圆参数生效（回车实时）：写入存储 + tooltip + 触发重算。"""
        self._ellip[r] = dict(alpha_deg=alpha, theta_deg=theta, eps_deg=eps)
        self._set_pol_tooltip(r)
        self._table_changed()

    def _set_pol_tooltip(self, r):
        """偏振单元格悬停提示：椭圆偏时显示具体参数。"""
        w = self.table.cellWidget(r, 3)
        if not isinstance(w, QComboBox):
            return
        e = self._ellip.get(r)
        if self._pol_text(r) == '椭圆偏' and e:
            w.setToolTip(f'椭圆偏: α={e["alpha_deg"]:g}° '
                         f'θ={e["theta_deg"]:g}° ε={e["eps_deg"]:g}°')
        else:
            w.setToolTip(None)

    def _table_menu(self, pos):
        item = self.table.itemAt(pos)
        if item is None:
            return
        menu = QMenu(self)
        col = item.column()
        row = item.row()
        act_freq = menu.addAction('从跃迁对导出频率…') if col == 1 else None
        act_inten = menu.addAction('扫描光强导出强度…') if col == 2 else None
        # 椭圆偏振：右键偏振列 → 修改参数弹窗（仅当该行当前是椭圆偏）
        act_ellip = None
        if col == 3 and self._pol_text(row) == '椭圆偏':
            act_ellip = menu.addAction('修改椭圆偏振参数…')
        if act_freq or act_inten or act_ellip:
            menu.addSeparator()
        act_del = menu.addAction('删除此光')
        action = menu.exec_(self.table.mapToGlobal(pos))
        if action == act_freq:
            self._open_freq_dialog(row)
        elif action == act_inten:
            self._open_intensity_dialog(row)
        elif action == act_ellip:
            self._open_ellip_dialog(row)
        elif action == act_del:
            self.table.removeRow(row)
            self._ellip.pop(row, None)
            for r in range(self.table.rowCount()):
                self.table.item(r, 0).setText(str(r + 1))
            self._table_changed()

    def _row_cfg(self, r):
        """第 r 份光的配置 dict（频率/偏振/强度/scope）；越界返回 None。"""
        cfgs = self.light_cfgs()
        return cfgs[r] if r < len(cfgs) else None

    def _open_freq_dialog(self, row):
        """右键频率：从跃迁对 + 失谐导出频率 → 写回频率单元格。"""
        dlg = FreqFromTransitionDialog(self.page, row, self)
        if dlg.exec_() == QDialog.Accepted:
            self.table.setItem(
                row, 1, QTableWidgetItem(f'{dlg.result_freq_hz() / 1e9:.6f}'))
            self._table_changed()

    def _open_intensity_dialog(self, row):
        """右键强度：扫描光强实现目标频移 → 写回强度单元格。"""
        cfg = self._row_cfg(row)
        if cfg is None:
            return
        dlg = IntensityScanDialog(self.page, row, cfg, self)
        if dlg.exec_() == QDialog.Accepted:
            self.table.setItem(
                row, 2, QTableWidgetItem(f'{dlg.result_intensity():.6g}'))
            self._table_changed()

    # ---------------- 关注跃迁 ----------------
    def rebuild_focus(self, show_exchange):
        """刷新"关注跃迁"选项；保持当前选择。show_exchange：单体页加 exchange。"""
        self._focus_exchange = bool(show_exchange)
        cur = self.focus_box.currentIndex()
        self.focus_box.blockSignals(True)
        self.focus_box.clear()
        for r in range(self.table.rowCount()):
            it_f = self.table.item(r, 1)
            f = it_f.text() if it_f is not None else '?'
            self.focus_box.addItem(f'光{r + 1}: {f}GHz {self._pol_text(r)}')
        if show_exchange:
            self.focus_box.addItem('exchange')
        if self.focus_box.count() > 0:
            # 无选中（-1）默认选第一个；否则尽量保持原选择
            idx = 0 if cur < 0 else min(cur, self.focus_box.count() - 1)
            self.focus_box.setCurrentIndex(idx)
        self.focus_box.blockSignals(False)

    def focus_drive_idx(self):
        """关注某份光 → 光索引；关注 exchange（或空）→ -1。"""
        i = self.focus_box.currentIndex()
        if i < 0:
            return -1
        n = self.table.rowCount()
        if i < n:
            return i
        return -1                    # exchange

    def focus_is_exchange(self):
        """是否关注 exchange（选项索引 ≥ 光场数，仅单体页可出现）。"""
        return (self.focus_box.currentIndex() >= 0
                and self.focus_box.currentIndex() >= self.table.rowCount()
                and self._focus_exchange)

    # ---------------- 数据 ----------------
    def light_cfgs(self):
        """全部光的配置列表（frequency_hz/polarization/intensity/scope）。"""
        cfgs = []
        for r in range(self.table.rowCount()):
            it_f = self.table.item(r, 1)
            it_i = self.table.item(r, 2)
            if it_f is None or it_i is None:
                continue
            try:
                f_hz = float(it_f.text()) * 1e9
                inten = float(it_i.text())
            except ValueError:
                continue
            scope_w = self.table.cellWidget(r, 4)
            if isinstance(scope_w, QComboBox):
                scope = scope_w.currentText()
            elif scope_w is not None:
                scope = scope_w.text().strip() or 'both'   # QLineEdit 残留
            else:
                scope = 'both'
            pol = self._pol_text(r)
            cfg = dict(frequency_hz=f_hz, polarization=pol,
                       intensity=inten, scope=scope, enabled=True)
            if pol == '椭圆偏':
                e = self._ellip.get(r)
                if e:
                    cfg['elliptical'] = dict(alpha_deg=e['alpha_deg'],
                                             theta_deg=e['theta_deg'],
                                             eps_deg=e['eps_deg'])
            cfgs.append(cfg)
        return cfgs

    def first_cfg(self):
        """第一份光的配置（用于单体 address 同步）；无光返回 None。"""
        cfgs = self.light_cfgs()
        return cfgs[0] if cfgs else None

    def psi_ket(self):
        """当前初态 (n, l, j, m)；m 超范围（|m|>j）返回 None。"""
        n = int(self.psi_n.value())
        l = int(self.psi_l.value())
        j = float(self.psi_j.value())
        m = float(self.psi_m.value())
        if abs(m) > j + 1e-9:
            return None
        return (n, l, j, m)


# ============================================================
# 光场上层处理：能级对频移求解 + 扫描光强 + 右键对话框
# ============================================================

def level_shift_at_intensity(cache, groups, cfg, B_gauss, level, I_mw):
    """给定光强下，单一能级 (label, qstr) 的 AC 位移 (MHz)。

    扫描目标 = 该能级自己的 AC 位移（与"失谐定频率"的能级对语义区分）。
    统一 direct 后端。能级不在可见组返回 None。
    """
    label, qstr = level
    cfg = dict(cfg)
    cfg['enabled'] = True
    cfg['intensity'] = I_mw
    rows = core.build_level_rows(cache, B_gauss, groups, cfg, groups,
                                 backend='direct')
    for r in rows:
        if r['label'] == label and r['qstr'] == qstr:
            return r['ac']
    return None


def _sweep_level_shifts(keys, shifts, level):
    """从 direct_ac_stark_sweep 结果提取单一能级位移数组 (MHz)；缺失返回 None。"""
    label, qstr = level
    for k in keys:
        n, l, j, q, occ = k
        if core.auto_label(n, l, j) == label and q == qstr:
            return shifts[k]      # 重复 qstr 取第一个
    return None


def scan_intensity_for_level(cache, groups, cfg, B_gauss, level, target_mhz,
                             progress=None):
    """扫描光强使单一能级 (label, qstr) 的 AC 位移达到 target_mhz (MHz)。

    返回 (I_best, actual, status)：
      status = 'ok'          成功（I_best 使位移≈target）
             | 'no_level'    能级缺失
             | 'no_crossing' 扫描未跨越目标（I_best 为最近点，actual 为其位移）
    progress(text)：可选回调，每步输出进度（光强 → 位移）。

    算法（光强与位移正相关、弱场基本线性 → 少量点即可）：
      1) win=1 基组对数粗扫 8 点定位跨越区间；
      2) 区间内按线性关系直接插值求光强（无需细扫 24 点）；
      3) 默认全基组（win=3）验证，超出容差则割线校正一次。
    """
    SCAN_WIN = 1
    tol = max(abs(target_mhz) * 0.01, 0.05)    # MHz
    cfg = dict(cfg)
    cfg['enabled'] = True

    def log(*a):
        if progress is not None:
            progress(' '.join(str(x) for x in a))

    # 1) win=1 对数粗扫 8 点：光强与位移正相关，少量点即可定区间
    grid = np.geomspace(1e-6, 1e9, 8)     # mW/cm²
    log(f'粗扫 {len(grid)} 点（{grid[0]:.1e}~{grid[-1]:.1e} mW/cm²）…')
    keys, s_by_key = core.direct_ac_stark_sweep(
        cache, groups, cfg, B_gauss, grid, sos_n_window=SCAN_WIN)
    shifts = _sweep_level_shifts(keys, s_by_key, level)
    if shifts is None:
        return None, None, 'no_level'
    log(f'粗扫完成：目标能级位移范围 [{shifts.min():+.4g}, {shifts.max():+.4g}] MHz')
    # 找所有跨越目标值的区间（多态共振/非单调时可能有多个）
    crossings = [k for k in range(len(grid) - 1)
                 if (shifts[k] < target_mhz < shifts[k + 1])
                 or (shifts[k + 1] < target_mhz < shifts[k])]
    if not crossings:
        best = int(np.argmin(np.abs(shifts - target_mhz)))
        log(f'未跨越目标：最近点 I={grid[best]:.6g} → 位移 {shifts[best]:+.4f} MHz')
        return float(grid[best]), float(shifts[best]), 'no_crossing'
    # 多个跨越区间取端点位移最接近目标值的一个（最相关段）
    idx = min(crossings,
              key=lambda c: min(abs(shifts[c] - target_mhz),
                                abs(shifts[c + 1] - target_mhz)))
    Ia, Ib = grid[idx], grid[idx + 1]
    sa, sb = shifts[idx], shifts[idx + 1]
    # 2) 光强与位移正相关（弱场线性）：区间内两点线性插值，无需细扫
    t = (target_mhz - sa) / (sb - sa) if sb != sa else 0.5
    t = min(max(t, 1e-9), 1 - 1e-9)
    I_best = Ia + t * (Ib - Ia)
    I_best = min(max(I_best, 1e-6), 1e9)
    log(f'线性插值：I={I_best:.6g} mW/cm²'
        f'（区间 [{Ia:.4g},{Ib:.4g}]，位移 [{sa:+.4f},{sb:+.4f}] MHz）')

    # 3) 默认全基组（win=3）验证；非线性区段割线迭代（最多 3 次）
    actual = level_shift_at_intensity(cache, groups, cfg, B_gauss, level, I_best)
    if actual is None:
        return None, None, 'no_level'
    log(f'验证：I={I_best:.6g} → 位移 {actual:+.4f} MHz')
    for _ in range(3):
        if abs(actual - target_mhz) <= tol:
            break
        I2 = I_best * (1.3 if actual < target_mhz else 0.7)
        S2 = level_shift_at_intensity(cache, groups, cfg, B_gauss, level, I2)
        log(f'割线尝试：I={I2:.6g} → 位移 ' +
            ('无效' if S2 is None else f'{S2:+.4f}') + ' MHz')
        if S2 is None or abs(S2 - actual) <= 1e-12:
            break
        I_best = I_best + (target_mhz - actual) * (I2 - I_best) / (S2 - actual)
        I_best = min(max(I_best, 1e-6), 1e9)
        actual = level_shift_at_intensity(cache, groups, cfg, B_gauss, level, I_best)
        if actual is None:
            actual = 0.0
            break
        log(f'割线校正：I={I_best:.6g} → 位移 {actual:+.4f} MHz')
    return I_best, actual, 'ok'


def _candidate_pairs(cache, groups, B_gauss, cfg):
    """当前光（cfg 偏振）下能实际驱动的跃迁对，按距 cfg 频率失谐升序。

    返回 [(nu_hz, S, la, qa, lb, qb, pol, det_hz)]：det = nu − cfg 频率。
    只保留当前偏振下 direct 后端真正耦合的对（|d|² > 1e-24，与 AC 扫描
    判据一致），否则扫描必然 no_crossing；失谐近者在前。
    """
    off = dict(cfg)
    off['enabled'] = False          # 裸能级（不含 AC 位移）
    off['intensity'] = 0.0
    rows = core.build_level_rows(cache, B_gauss, groups, off, groups)
    coeffs = core.address_pol_coeffs(off)
    lines = []
    for i in range(len(rows)):
        for k in range(i + 1, len(rows)):
            a, b = rows[i], rows[k]
            if (a['n'], a['l'], a['j']) == (b['n'], b['l'], b['j']):
                continue
            S = 0.0
            qs = []
            for q, c in coeffs:
                if abs(c) < 1e-12:
                    continue
                sq = core.transition_strength_q(cache, a, b, q)
                if sq > 1e-24:               # direct 耦合判据 |d|>1e-12
                    S += abs(c) ** 2 * sq
                    qs.append(q)
            if S <= 0.0:
                continue
            nu = abs(b['E0'] - a['E0']) * 1e6
            pol = ','.join('σ+' if q == 1 else ('π' if q == 0 else 'σ-')
                           for q in qs)
            lines.append((nu, S, a['label'], a['qstr'],
                          b['label'], b['qstr'], pol, cfg['frequency_hz'] - nu))
    lines.sort(key=lambda t: abs(t[7]))
    return lines


def _candidate_levels(cache, groups, B_gauss, cfg):
    """全部子能级，按当前光下 |AC 位移| 降序。

    返回 [(label, qstr, ac_mhz)]：受该光影响的能级排前面
    （扫描目标是单一能级，选受影响者才有意义）。
    """
    on = dict(cfg)
    on['enabled'] = True
    rows = core.build_level_rows(cache, B_gauss, groups, on, groups)
    lv = [(r['label'], r['qstr'], r['ac']) for r in rows]
    lv.sort(key=lambda t: -abs(t[2]))
    return lv


class FreqFromTransitionDialog(QDialog):
    """右键频率：选一对跃迁 + 失谐 Δ → 频率 = 跃迁频率 + Δ，确定写回表格。"""

    def __init__(self, page, row, parent=None):
        super().__init__(parent)
        self.setWindowTitle('从跃迁对导出频率')
        cfg = page.field._row_cfg(row)
        if cfg is None:
            self._lines = []
        else:
            self._lines = _candidate_pairs(
                page.cache, page.groups, page.field.b_gauss(), cfg)
        lay = QVBoxLayout(self)
        form = QFormLayout()
        self.pair_combo = QComboBox()
        for (nu, A, la, qa, lb, qb, pol, det) in self._lines:
            self.pair_combo.addItem(
                f'{la} {qa} → {lb} {qb}   ({nu / 1e9:.4f} GHz, {pol}, '
                f'Δ{det / 1e6:+.1f}MHz)',
                (nu, la, qa, lb, qb))
        form.addRow('跃迁对', self.pair_combo)
        self.det_spin = QDoubleSpinBox()
        self.det_spin.setRange(-1e6, 1e6)
        self.det_spin.setDecimals(1)
        self.det_spin.setValue(0.0)
        self.det_spin.setToolTip('失谐 Δ（正 = 蓝失谐，光频率高于跃迁）；\n新频率 = 跃迁频率 + Δ')
        form.addRow('失谐 Δ (MHz)', self.det_spin)
        self.result_label = QLabel('')
        self.result_label.setStyleSheet('font-family: Consolas; font-size: 10pt;')
        form.addRow('导出频率', self.result_label)
        lay.addLayout(form)
        btns = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)
        lay.addWidget(btns)
        self.pair_combo.currentIndexChanged.connect(self._update)
        self.det_spin.valueChanged.connect(self._update)
        if not self._lines:
            self.result_label.setText('当前能级组间没有偶极跃迁（请先添加能级组）')
            btns.button(QDialogButtonBox.Ok).setEnabled(False)
        else:
            self._update()

    def _update(self, *_):
        nu = self.pair_combo.currentData()[0]
        f = nu + self.det_spin.value() * 1e6
        self.result_label.setText(f'{f / 1e9:.6f} GHz')

    def result_freq_hz(self):
        """确认后的目标频率 (Hz)。"""
        nu = self.pair_combo.currentData()[0]
        return nu + self.det_spin.value() * 1e6


class IntensityScanDialog(QDialog):
    """右键强度：选单一能级 + 目标位移 → 扫描光强，确定写回表格。

    扫描目标 = 该能级自身的 AC 位移（与"失谐定频率"的能级对语义区分）。
    扫描过程在日志框实时显示"尝试的光强 → 目标能级位移"。
    """

    def __init__(self, page, row, cfg, parent=None):
        super().__init__(parent)
        self.setWindowTitle('扫描光强导出强度')
        self.page = page
        self.cfg = dict(cfg)
        self._levels = _candidate_levels(
            page.cache, page.groups, page.field.b_gauss(), self.cfg)
        self._I_best = None
        lay = QVBoxLayout(self)
        form = QFormLayout()
        self.level_combo = QComboBox()
        for (label, qstr, ac) in self._levels:
            self.level_combo.addItem(
                f'{label} {qstr}   (当前位移 {ac:+.3f} MHz)',
                (label, qstr))
        form.addRow('目标能级', self.level_combo)
        self.target_spin = QDoubleSpinBox()
        self.target_spin.setRange(-1e6, 1e6)
        self.target_spin.setDecimals(2)
        self.target_spin.setSingleStep(1.0)
        self.target_spin.setValue(20.0)
        self.target_spin.setToolTip(
            '想让该能级自身的 AC 位移量（MHz）；与"失谐定频率"的能级对语义不同')
        form.addRow('目标位移 (MHz)', self.target_spin)
        lay.addLayout(form)
        self.hint = QLabel(
            f'扫描用矩阵对角化（direct），保持 B/偏振/频率不变，\n'
            f'光强 = 该行第 {row + 1} 份光；确定后写入光强单元格。')
        self.hint.setWordWrap(True)
        self.hint.setStyleSheet('color: #666;')
        lay.addWidget(self.hint)
        # 扫描过程日志（每步显示尝试的光强 → 目标能级位移）
        self.log_edit = QPlainTextEdit()
        self.log_edit.setReadOnly(True)
        self.log_edit.setFixedHeight(110)
        self.log_edit.setStyleSheet('font-family: Consolas; font-size: 9pt;')
        lay.addWidget(self.log_edit)
        self.result_label = QLabel('')
        self.result_label.setStyleSheet('font-family: Consolas; font-size: 10pt;')
        lay.addWidget(self.result_label)
        btns = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btns.accepted.connect(self._on_ok)
        btns.rejected.connect(self.reject)
        lay.addWidget(btns)
        if not self._levels:
            self.log_edit.setPlainText('请先添加能级组')
            btns.button(QDialogButtonBox.Ok).setEnabled(False)

    def _log(self, text):
        self.log_edit.appendPlainText(text)
        QApplication.processEvents()      # 阻塞扫描中实时刷新

    def _on_ok(self):
        level = self.level_combo.currentData()
        target = self.target_spin.value()
        if abs(target) < 1e-9:
            QMessageBox.warning(self, '提示', '目标位移不能为 0')
            return
        self.log_edit.clear()
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            I_best, actual, status = scan_intensity_for_level(
                self.page.cache, self.page.groups, self.cfg,
                self.page.field.b_gauss(), level, target, progress=self._log)
        finally:
            QApplication.restoreOverrideCursor()
        if status == 'no_level':
            QMessageBox.warning(
                self, '能级缺失',
                f'扫描时找不到能级 {level[0]} {level[1]}，请重新选择。')
            return
        if status == 'no_crossing':
            QMessageBox.warning(
                self, '未收敛',
                f'光强扫描未跨越目标位移 {target:+.2f} MHz\n'
                f'（可能强驱动多态共振，光强→位移不再单调）。\n'
                f'最接近点：I={I_best:.4g} mW/cm² → 位移 {actual:+.3f} MHz。\n'
                f'请调整目标值，或改用失谐/频率扫描。')
            self._I_best = I_best
            self.result_label.setText(
                f'最近点：I = {I_best:.6g} mW/cm²（位移 {actual:+.3f} MHz）')
            return
        self._I_best = I_best
        self.result_label.setText(
            f'目标位移 {target:+.3f} MHz → 光强 {I_best:.6g} mW/cm²'
            f'（实际 {actual:+.3f} MHz）')
        self.accept()

    def result_intensity(self):
        """确认后的光强 (mW/cm²)。"""
        return self._I_best


class EllipDialog(QDialog):
    """椭圆偏振参数弹窗（α° 传播角 / θ° 主轴方位 / ε° 椭偏角）。

    非模态：打开时改参数（回车/失焦 editingFinished）即实时生效并触发重算；
    点"确定"关闭并固定；点"取消"关闭（保留已调参数）。
    """

    def __init__(self, parent, initial, on_param):
        super().__init__(parent)
        self.setWindowTitle('椭圆偏振参数')
        self._on_param = on_param
        form = QFormLayout(self)
        self.alpha_spin = QDoubleSpinBox()
        self.alpha_spin.setRange(0.0, 90.0)
        self.alpha_spin.setDecimals(1)
        self.alpha_spin.setSingleStep(1.0)
        self.alpha_spin.setToolTip('传播方向与量子化轴 z 的夹角（0°=沿 z，90°=垂直 z）')
        form.addRow('α° 传播角', self.alpha_spin)
        self.theta_spin = QDoubleSpinBox()
        self.theta_spin.setRange(0.0, 180.0)
        self.theta_spin.setDecimals(1)
        self.theta_spin.setSingleStep(1.0)
        self.theta_spin.setToolTip('偏振椭圆主轴在垂直传播方向平面内的方位角（0~180°）')
        form.addRow('θ° 主轴方位', self.theta_spin)
        self.eps_spin = QDoubleSpinBox()
        self.eps_spin.setRange(-45.0, 45.0)
        self.eps_spin.setDecimals(1)
        self.eps_spin.setSingleStep(1.0)
        self.eps_spin.setToolTip('椭偏角（0=线偏，+45=左旋圆偏，−45=右旋圆偏）')
        form.addRow('ε° 椭偏角', self.eps_spin)
        if initial is not None:
            self.alpha_spin.setValue(initial.get('alpha_deg', 0.0))
            self.theta_spin.setValue(initial.get('theta_deg', 0.0))
            self.eps_spin.setValue(initial.get('eps_deg', 0.0))
        for s in (self.alpha_spin, self.theta_spin, self.eps_spin):
            s.editingFinished.connect(self._emit)
        btns = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)
        form.addRow(btns)

    def _emit(self):
        """参数编辑完成（回车/失焦）→ 实时生效触发重算。"""
        self._on_param(self.alpha_spin.value(),
                       self.theta_spin.value(),
                       self.eps_spin.value())
