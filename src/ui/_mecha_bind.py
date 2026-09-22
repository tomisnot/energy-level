# -*- coding: utf-8 -*-
"""GUI widget ↔ mecha 绑定层——取代外部 `param_manager` 框架 + `pm_setup`。

旧 GUI 的标准件（spinbox/combo/…）经 `pm_setup`→外部 `param_manager.ParamRegistry`
绑定；那套 registry 是**独立于 mecha 的第二状态源**（违反 I2）。本层用 **mecha 原生**
绑定取代它：widget 读初值自 `AppBridge`、写经 `bridge.write`（choke point）、订阅 bus
实时刷新（AI 写入即时反映 + 归因高亮）、按 `can_write` 只读门控（两模式）。

**设计成 `pm` 的近 drop-in**（`spinbox/combobox/lineedit/checkbox/ParamSpec` 同名同签名），
面板移植≈机械替换：`from .. import pm_setup as pm` → `from . import _mecha_bind as MB`；
`pm.spinbox(_reg, ...)` → `MB.spinbox(_binder, ...)`；`self.page.param_reg` → `self.page.binder`。

只用 PyQt5 + AppBridge，不依赖 param_manager。
"""
from PyQt5.QtCore import QTimer
from PyQt5.QtWidgets import (QDoubleSpinBox, QSpinBox, QComboBox, QLineEdit,
                             QCheckBox)

__all__ = ['ParamSpec', 'Binder', 'spinbox', 'intbox', 'combobox', 'lineedit',
           'checkbox']

# AI 侧 actor（归因高亮判定；与 mecha.authority.side_of 一致）
_AI_SIDE = ('ai', 'codeact', 'job', 'mcp')
_FLASH_QSS = 'border: 2px solid #a855f7;'      # 归因高亮：AI 刚改的（来源紫）


class ParamSpec:
    """轻量参数元数据（兼容 `pm.ParamSpec(...)` 调用形参；GUI 侧提示/类型用）。"""

    def __init__(self, kind='float', default=None, unit=None, desc='',
                 aliases=(), options=None, group='', lo=None, hi=None,
                 log=False, tier='run', **kw):
        self.kind = kind
        self.default = default
        self.unit = unit
        self.desc = desc
        self.aliases = list(aliases or ())
        self.options = options
        self.group = group
        self.lo = lo
        self.hi = hi
        self.log = log
        self.tier = tier


class Binder:
    """一个 MainWindow 一个 Binder：管 widget↔key 绑定、bus 刷新、只读门控、归因高亮。

    - `make_*`/模块级 `spinbox(binder, ...)`：建绑定控件（读初值自 bridge）。
    - `register_external`：非标参数（光场表/能级组/波形/psi/衰变）——把一个 bridge key
      映射到面板的 read_fn/write_fn（取代旧 `register_panel_params`）。
    - 订阅 bus：外部（AI）改某 key → 刷新对应 widget（blockSignals 防回环）+ 归因高亮。
    - `set_readonly(ro)`：两模式门控——AI 模式下 GUI 控件物理只读（can_write('gui')==False）。
    """

    def __init__(self, bridge, reason_prefix='GUI', notify=None):
        self.bridge = bridge
        self.reason_prefix = reason_prefix
        # notify(msg, timeout)：写权被拒等**必须让人看见**的反馈出口（主窗口状态栏）。
        self._notify = notify
        self._last_reject = None
        self._binds = []          # [(widget, key, setter)]
        self._ext = {}            # key -> (read_fn, write_fn, widget, apply_fn)
        self._updating = False    # 防回环：程序刷新 widget 时不再写回 bridge
        self._flash_timers = {}
        self._unsub = bridge.subscribe(self._on_event)

    # ---------------- 写（经 choke point） ----------------
    def _write(self, key, value):
        if self._updating:
            return
        if not self.bridge.can_write('gui'):
            self._reject(key)
            return
        self.bridge.write(key, value, reason='%s 调 %s' % (self.reason_prefix, key),
                          actor='gui')

    def _reject(self, key):
        """写权被拒 → **不静默**（D5）：状态栏说明"为什么 + 怎么办"（D2 可教学）。

        P2-4：原来是裸 `return` ⇒ 人点了控件"没反应、没提示、没原因"，是静默失效。
        现在把原因与接管路径摆到状态栏；**同一原因只报一次**（连续拖滑块不刷屏），
        原因变化（模式切换）后重新报。

        **超时语义（已确认）**：Qt 的 `showMessage(msg, timeout)` 里 `timeout=0` 表示
        **常驻**（直到 `clearMessage()` 或被下一条消息顶掉）。这里刻意用 **10 s**：
        常驻会造成"看起来永久"的假象（它其实会被下一条日志顶掉），10 s 够人读完，
        也不会让状态栏长期挂着过期信息。
        """
        mode = getattr(self.bridge, 'mode', '?')
        msg = ('写权在 %s 侧（当前 %s 模式）：人对「%s」的修改被拒。'
               '要接管：用人工控制端点切回人工模式（token 在 .human-token）'
               % (mode, mode, key))
        if msg == self._last_reject:
            return
        self._last_reject = msg
        if self._notify is not None:
            try:
                self._notify(msg, 10000)
            except Exception:                                # noqa: BLE001
                pass

    # ---------------- bus 订阅：外部改动 → 刷新 + 归因高亮 ----------------
    def _on_event(self, ev):
        if ev.get('kind') not in ('set', 'undo'):
            return
        actor = str(ev.get('actor') or '')
        if actor.startswith('gui'):
            return                       # 本 GUI 自己的写，不必回刷（避免回环/闪）
        key = ev.get('target')
        ai = actor.split(':')[0] in _AI_SIDE
        self._refresh_key(key, flash=ai)
        ext = self._ext.get(key)
        if ext is not None:
            read_fn, write_fn, widget, apply_fn = ext
            if apply_fn is not None:
                self._updating = True
                try:
                    apply_fn(self.bridge.read(key))   # 用新值（非面板旧值）
                finally:
                    self._updating = False
                if widget is not None and ai:
                    self._flash(widget)

    def _refresh_key(self, key, flash=False):
        val = self.bridge.read(key)
        for (w, k, setter) in self._binds:
            if k != key:
                continue
            self._updating = True
            try:
                setter(w, val)
            finally:
                self._updating = False
            if flash:
                self._flash(w)

    def _flash(self, w):
        """归因高亮：AI 刚改的控件闪一下紫框（2s 后复原，不回流布局）。"""
        try:
            old = w.property('_mb_base_qss')
            if old is None:
                w.setProperty('_mb_base_qss', w.styleSheet())
                old = w.styleSheet()
            w.setStyleSheet((old + ' ' + _FLASH_QSS).strip())
            t = self._flash_timers.get(id(w))
            if t is not None:
                t.stop()
            t = QTimer(w)
            t.setSingleShot(True)
            t.timeout.connect(lambda w=w: w.setStyleSheet(
                w.property('_mb_base_qss') or ''))
            self._flash_timers[id(w)] = t
            t.start(2000)
        except Exception:                                    # noqa: BLE001
            pass

    # ---------------- 只读门控（两模式） ----------------
    def set_readonly(self, ro):
        """AI 模式 ro=True：所有绑定控件物理只读/禁用（不只是看起来只读）。"""
        for (w, _k, _s) in self._binds:
            self._apply_readonly(w, ro)
        for (_k, (_r, _w, widget, _a)) in self._ext.items():
            if widget is not None:
                self._apply_readonly(widget, ro)

    @staticmethod
    def _apply_readonly(w, ro):
        if isinstance(w, (QDoubleSpinBox, QSpinBox, QLineEdit)):
            w.setReadOnly(bool(ro))
        elif isinstance(w, (QComboBox, QCheckBox)):
            w.setEnabled(not ro)
        else:
            try:
                w.setReadOnly(bool(ro))
            except Exception:                                # noqa: BLE001
                w.setEnabled(not ro)

    # ---------------- 标准件工厂 ----------------
    def _init_val(self, key, default):
        v = self.bridge.read(key)
        return default if v is None else v

    def make_spinbox(self, key, default, lo, hi, decimals=2, width=None,
                     parent=None, spec=None, is_int=False):
        kind = (spec.kind if spec is not None else ('int' if is_int else 'float'))
        if kind == 'int':
            w = QSpinBox(parent)
            w.setRange(int(lo), int(hi))
            w.setValue(int(self._init_val(key, default)))
            setter = lambda w, v: w.setValue(int(v if v is not None else default))  # noqa: E731
            w.valueChanged.connect(lambda _x, k=key, ww=w: self._write(k, ww.value()))
        else:
            w = QDoubleSpinBox(parent)
            w.setDecimals(int(decimals))
            w.setRange(float(lo), float(hi))
            w.setValue(float(self._init_val(key, default)))
            setter = lambda w, v: w.setValue(float(v if v is not None else default))  # noqa: E731
            w.valueChanged.connect(lambda _x, k=key, ww=w: self._write(k, ww.value()))
        if width:
            w.setFixedWidth(int(width))
        self._binds.append((w, key, setter))
        return w

    def make_combobox(self, key, options, parent=None, spec=None, index=False):
        w = QComboBox(parent)
        w.addItems([str(o) for o in options])
        cur = self._init_val(key, None)
        if index:                        # 值 = 索引（decay_index 之类）
            if cur is not None and 0 <= int(cur) < len(options):
                w.setCurrentIndex(int(cur))
            w.currentIndexChanged.connect(
                lambda _i, k=key, ww=w: self._write(k, ww.currentIndex()))
            setter = lambda w, v: w.setCurrentIndex(int(v or 0))   # noqa: E731
        else:
            if cur is not None and w.findText(str(cur)) >= 0:
                w.setCurrentText(str(cur))
            w.currentTextChanged.connect(
                lambda _t, k=key, ww=w: self._write(k, ww.currentText()))
            setter = lambda w, v: (w.setCurrentText(str(v))         # noqa: E731
                                   if v is not None else None)
        self._binds.append((w, key, setter))
        return w

    def make_lineedit(self, key, default='', parent=None, spec=None):
        w = QLineEdit(parent)
        w.setText(str(self._init_val(key, default)))
        w.editingFinished.connect(
            lambda k=key, ww=w: self._write(k, ww.text()))
        self._binds.append((w, key, lambda w, v: w.setText('' if v is None else str(v))))
        return w

    def make_checkbox(self, key, default=False, parent=None, spec=None, text=''):
        w = QCheckBox(text, parent)
        w.setChecked(bool(self._init_val(key, default)))
        w.toggled.connect(lambda _b, k=key, ww=w: self._write(k, ww.isChecked()))
        self._binds.append((w, key, lambda w, v: w.setChecked(bool(v))))
        return w

    # ---------------- 非标参数登记（取代 register_panel_params） ----------------
    def register_external(self, key, read_fn, write_fn, widget=None,
                          touch_sig=None, apply_fn=None):
        """把一个 bridge key 映射到面板的 read/write（光场表/能级组/波形/psi/衰变）。

        - read_fn(): 从面板读当前值（→ 存进 bridge 时用）。
        - write_fn(v): 把值写回面板控件。
        - touch_sig: 面板数据变更信号 → 触发时把 read_fn() 写进 bridge。
        - apply_fn: bus 外部改动时刷新面板（缺省用 write_fn）。
        """
        self._ext[key] = (read_fn, write_fn, widget, apply_fn or write_fn)
        if touch_sig is not None:
            touch_sig.connect(lambda *a, k=key: self._write_external(k))
        # 初值：bridge 有则写回面板，否则把面板当前值种进 bridge
        cur = self.bridge.read(key)
        if cur is not None:
            self._updating = True
            try:
                write_fn(cur)
            finally:
                self._updating = False
        else:
            self._seed(key, read_fn)

    def _write_external(self, key):
        if self._updating:
            return
        if not self.bridge.can_write('gui'):
            self._reject(key)
            return
        read_fn = self._ext[key][0]
        self.bridge.write(key, read_fn(), reason='%s 改 %s' % (self.reason_prefix, key),
                          actor='gui')

    def _seed(self, key, read_fn):
        """把面板当前值种进 bridge（不经 gate：用 actor='session' 侧，仅初始化）。"""
        try:
            self.bridge.session.set(key, read_fn(), actor='session',
                                    reason='GUI 初始种子')
        except Exception:                                       # noqa: BLE001
            pass

    def push_all(self):
        """把当前所有 external 面板值推进 bridge（存档/运行前同步用）。"""
        for key, (read_fn, _w, _wid, _a) in self._ext.items():
            self._updating = True
            try:
                self.bridge.session.set(key, read_fn(), actor='gui',
                                        reason='GUI 同步面板值')
            except Exception:                                   # noqa: BLE001
                pass
            finally:
                self._updating = False


# ---------------- 模块级工厂（近 pm.spinbox(reg, ...) 的 drop-in 签名） ----------------
def spinbox(binder, key, default, lo, hi, decimals=2, width=None, parent=None,
            spec=None):
    return binder.make_spinbox(key, default, lo, hi, decimals=decimals,
                               width=width, parent=parent, spec=spec)


def intbox(binder, key, default, lo, hi, width=None, parent=None, spec=None):
    return binder.make_spinbox(key, default, lo, hi, width=width, parent=parent,
                               spec=spec, is_int=True)


def combobox(binder, key, options, parent=None, spec=None, index=False):
    return binder.make_combobox(key, options, parent=parent, spec=spec, index=index)


def lineedit(binder, key, default='', parent=None, spec=None):
    return binder.make_lineedit(key, default=default, parent=parent, spec=spec)


def checkbox(binder, key, default=False, parent=None, spec=None, text=''):
    return binder.make_checkbox(key, default=default, parent=parent, spec=spec,
                                text=text)
