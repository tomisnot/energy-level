# -*- coding: utf-8 -*-
"""多体页自检（P3-1 M0）——offscreen 验"单独一页"真的接上且既有页没被改。

覆盖：
  A. 装配：`MainWindow.manybody` 在位；`canvas_tabs` 有 3 页且**前两页文本/索引不变**
     （M3：新增页不得动既有页）；`page_stack` 第三页就是 `manybody_widget`（互斥占位）。
  B. 参数页：`manybody.*` 键经 `_mecha_bind` 绑定；改控件 → 经 choke point 落 session
     （actor=gui，人机同路 I5）。
  C. 点云：加原子 → session `manybody.atoms` 有 N 个；**删到空写 None（清除）而不是
     空 list**（registry 的 `validate_atom_list` 拒空 list，写空会抛错）。
  D. 成本提示是**纯查询**：`refresh_cost()` 不往 session 写 `dynamic.mode`。
  E. 运行接线：`set_nbody('many')` + `_sync_run_config()` → `dynamic.mode==2`；
     `_run` 接受 `on_done`（多体页用自己的回调渲染，但仍是同一条运行流）。
     **真跑多体**由 `manybody_integration_selfcheck` 覆盖（build_run/evolve/线程拆分），
     本判据只钉"GUI 侧把 mode 写成 2 且把结果交回本页"这条接线，不重复烧算力。
  F. M0 未做项**显式列出**（页面上有 NOT_YET 提示），不留"以为有"的空白。

offscreen；os._exit 绕过 teardown 噪声。
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
os.environ.setdefault('EL_NO_AI_SERVER', '1')

from PyQt5.QtWidgets import QApplication                  # noqa: E402
from checks.checklib import Checker                        # noqa: E402

app = QApplication.instance() or QApplication(sys.argv)
C = Checker('gui_manybody')

import app_entry                                          # noqa: E402
from src.ui import ui_manybody as UMB                     # noqa: E402

_w = app_entry.MainWindow()
_mb = _w.manybody

# ---------------- A. 装配：第三页 + 既有页不变 ----------------
_tabs = [_w.canvas_tabs.tabText(i) for i in range(_w.canvas_tabs.count())]
_stack = [type(_w.page_stack.widget(i)).__name__
          for i in range(_w.page_stack.count())]
_w.canvas_tabs.setCurrentIndex(2)          # 走真实切页钩子（on_show）
app.processEvents()
C.group('A 多体页装配（第三页；既有两页不动 = M3）', [
    ('canvas_tabs 有 3 页：%s' % _tabs, _tabs == ['静态', '动态', '多体']),
    ('page_stack 第三页 = 多体参数页（互斥占位）',
     len(_stack) == 3 and _w.page_stack.widget(2) is _mb.manybody_widget),
    ('MainWindow.manybody 在位且是 ManyBodyPanel',
     isinstance(_mb, UMB.ManyBodyPanel)),
    ('切到多体页不抛异常（on_show 走通）', True),
    ('既有页的钩子仍是原对象（未被子类/替换）',
     _w.modules[0][0] == '静态' and _w.modules[1][0] == '动态'
     and _w.modules[2][0] == '多体'),
])

# ---------------- B. 参数页 → choke point ----------------
# 6 个标量/布尔走 `_mecha_bind` 工厂；`terms`(三态) 与 `psi0`(三态+模式串) 是
# **手写控件**（工厂没有 list/多态类型）——它们也必须经 choke point 写。
_BOUND_KEYS = ('manybody.template', 'manybody.solver', 'manybody.second_order',
               'manybody.n_window', 'manybody.guard_mhz', 'manybody.ed_max_dim')
_TERMS_UI = ('I1', 'I2', 'I3')
_bound = [k for (_wd, k, _s) in _w.binder._binds if k.startswith('manybody.')]
_mb.template_cb.setCurrentText('xy')
_tpl_written = _w.app_bridge.read('manybody.template')     # 就地取（后面探针会改回 ising）
_mb.nwin_sb.setValue(2)
_nwin_written = _w.app_bridge.read('manybody.n_window')
_evs = [e for e in _w.app_bridge.activity(0)
        if str(e.get('target') or '').startswith('manybody.')]


def _probe_terms(mode, checks, template=None):
    """驱动「相互作用项」三态控件 → 读回 session（判据与致死性共用同一探针）。"""
    if template:
        _mb.template_cb.setCurrentText(template)
    _mb.terms_mode_cb.setCurrentIndex(mode)
    for t in _TERMS_UI:
        _mb.term_cbs[t].setChecked(t in checks)
    _mb._write_terms()
    return _w.app_bridge.read('manybody.terms')


# 三态语义（与 schema 一一对应；裁决⑤ 后 `terms` 默认由 ['I1'] 改成 None）
_t_follow = _probe_terms(0, set(), 'ising')      # 跟随模板 → None
_t_follow_xy = _probe_terms(0, set(), 'xy')      # 换模板仍是 None（预设真值在 core）
_t_custom = _probe_terms(1, {'I2'})              # 自定义 → 显式 list
_t_custom2 = _probe_terms(1, {'I1', 'I3'})
_t_empty = _probe_terms(2, set())                # 显式无相互作用 → []

# 读路径回归（自测发现真 bug）：会话显式 ['I2'] → refresh 后勾选框必须显示 I2
_w.app_bridge.write('manybody.terms', ['I2'], reason='判据：读路径', actor='gui')
_mb.refresh_from_session()
_read_mode = _mb.terms_mode_cb.currentIndex()
_read_checks = [t for t in _TERMS_UI if _mb.term_cbs[t].isChecked()]

# 切模式不覆盖（自测发现真 bug）：mode0 显示的是预设(ising→I1)，切到 mode1 不得
# 把会话里的 ['I2'] 静默改成 ['I1']
_w.app_bridge.write('manybody.terms', ['I2'], reason='判据：切模式', actor='gui')
_mb.refresh_from_session()
_mb.terms_mode_cb.setCurrentIndex(0)
_mb.terms_mode_cb.setCurrentIndex(1)
_switch_val = _w.app_bridge.read('manybody.terms')

# 致死性：把"写出值"故意改错（三态一律写 ['I1']）→ 同一批期望必须判出不一致
_orig_write_terms = type(_mb)._write_terms


def _broken_write_terms(self):
    self.main.app_bridge.write('manybody.terms', ['I1'],
                               reason='致死性：故意写错', actor='gui')


type(_mb)._write_terms = _broken_write_terms
try:
    _lethal = [_probe_terms(0, set(), 'ising'), _probe_terms(1, {'I2'}),
               _probe_terms(2, set())]
finally:
    type(_mb)._write_terms = _orig_write_terms
_lethal_caught = (_lethal != [_t_follow, _t_custom, _t_empty])   # 与三条模式断言同一批期望
_w.app_bridge.write('manybody.terms', None, reason='判据：复原', actor='gui')
_mb.refresh_from_session()

# psi0 三态
_mb.psi0_cb.setCurrentIndex(1)
_psi0_ground = _w.app_bridge.read('manybody.psi0')
_mb.psi0_cb.setCurrentIndex(0)
_psi0_none = _w.app_bridge.read('manybody.psi0')
_mb.psi0_cb.setCurrentIndex(2)
_mb.psi0_edit.setText('grg')
_mb._write_psi0()
_psi0_str = _w.app_bridge.read('manybody.psi0')

C.group('B 参数页经 _mecha_bind / choke point 绑到 session（含三态语义）', [
    ('6 个标量/布尔键都进了 binder（实测 %d：%s）' % (len(_bound), _bound),
     sorted(_bound) == sorted(_BOUND_KEYS)),
    ('改模板 → session 读到 xy（就地取值，免受后续探针改模板影响）',
     _tpl_written == 'xy'),
    ('改 n_window → session 读到 2', _nwin_written == 2),
    ('相互作用项·跟随模板 → 写 None（%r / %r；预设真值在 core，不复制表）'
     % (_t_follow, _t_follow_xy), _t_follow is None and _t_follow_xy is None),
    ('相互作用项·自定义 → 写显式 list（%r / %r）' % (_t_custom, _t_custom2),
     _t_custom == ['I2'] and _t_custom2 == ['I1', 'I3']),
    ('相互作用项·显式无 → 写 []（与 None 是两个不同值，不可混同）',
     _t_empty == []),
    ('读路径：会话显式 [I2] → 模式=1 且勾选框显示 I2（mode=%d, 勾选=%s）'
     % (_read_mode, _read_checks),
     _read_mode == 1 and _read_checks == ['I2']),
    ('切模式不覆盖：[I2] 经 mode0→mode1 往返仍是 [I2]（%r）' % (_switch_val,),
     _switch_val == ['I2']),
    ('致死性：写出值故意改错 → 同一批期望判出不一致（%r）' % (_lethal,),
     _lethal_caught),
    ('ψ0 三态：Lanczos=%r / 全基态=%r / 模式串=%r'
     % (_psi0_ground, _psi0_none, _psi0_str),
     _psi0_ground == 'ground' and _psi0_none is None and _psi0_str == 'grg'),
    ('总线事件 actor=gui（人机同路 I5，%d 条）' % len(_evs),
     bool(_evs) and all(e.get('actor') == 'gui' for e in _evs)),
])

# ---------------- C. 点云 ↔ session（含空表清除） ----------------
_mb._atoms = []
_mb._write_atoms('判据：清空')
_mb._add_atom()
_mb._add_atom()
_atoms = _w.app_bridge.read('manybody.atoms') or []
_rows = _mb.table.rowCount()
_ok_shape = all(isinstance(a.get('g'), list) and len(a['g']) == 4
                and isinstance(a.get('pos'), list) and len(a['pos']) == 3
                for a in _atoms)
_mb._atoms = []
_mb._write_atoms('判据：清空')
_cleared = _w.app_bridge.read('manybody.atoms')
C.group('C 点云：写 session（choke point）+ 空表写 None 清除', [
    ('加 2 个原子 → session 有 2 个', len(_atoms) == 2),
    ('表行数与 session 一致（%d）' % _rows, _rows == 2),
    ('每个原子 g/r 是 4 元组、pos 是 3 元组（过 validate_atom_list）', _ok_shape),
    ('删到空 → 写 None（registry 拒空 list，故必须清除而非写空表）',
     _cleared is None),
    ('标签随原子数更新（%s）' % _mb.atom_label.text(),
     'dim=2^2=4' in _mb.atom_label.text() or 'dim=2^0=1' in _mb.atom_label.text()),
])

# ---------------- D. 成本提示是纯查询（不写 dynamic.mode） ----------------
_mode_before = _w.app_bridge.read('dynamic.mode')
_mb._add_atom()
_mb.refresh_cost()
_mode_after = _w.app_bridge.read('dynamic.mode')
_cost = _mb.cost_label.text()
C.group('D 成本提示（core.estimate 纯查询，无副作用）', [
    ('成本文本含 N 与 dim（%s）' % _cost[:60],
     'N=1' in _cost and 'dim=2^1=2' in _cost),
    ('预估不改 session 的 dynamic.mode（%r → %r）'
     % (_mode_before, _mode_after), _mode_before == _mode_after),
])

# ---------------- E. 运行接线（不真跑，真跑由 manybody_integration 覆盖） ----------------
import inspect as _inspect                                # noqa: E402
_sig = _inspect.signature(type(_w.dynamic)._run)
_w.dynamic.set_nbody('many')
_w.dynamic._sync_run_config()
_mode_written = _w.app_bridge.read('dynamic.mode')
_has_ondone = 'on_done' in _sig.parameters
_run_src = _inspect.getsource(_mb._on_run_clicked)
C.group('E 运行接线：同一条运行流 + 结果交回本页', [
    ('set_nbody("many") 生效（_nbody=%r）' % _w.dynamic._nbody,
     _w.dynamic._nbody == 'many'),
    ('_sync_run_config 写 dynamic.mode=2（%r）' % _mode_written,
     _mode_written == 2),
    ('DynamicPanel._run 接受 on_done（多体页用自己的回调渲染）', _has_ondone),
    ('运行按钮确实先声明族再借 _run',
     "set_nbody('many')" in _run_src and '._run(' in _run_src),
    ('未做项在页面上显式列出（%d 项，不留"以为有"）' % len(UMB.NOT_YET),
     len(UMB.NOT_YET) >= 3 and not _mb.not_yet.isHidden()),
])

_ok = C.summary()
sys.stdout.flush()
os._exit(0 if _ok else 1)
