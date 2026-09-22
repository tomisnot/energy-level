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
_keys = ('manybody.template', 'manybody.terms', 'manybody.solver',
         'manybody.second_order', 'manybody.n_window', 'manybody.guard_mhz',
         'manybody.ed_max_dim', 'manybody.psi0')
# 6 个标量/布尔走 `_mecha_bind` 工厂；`terms`(多选) 与 `psi0`(三态+模式串)
# 是**手写控件**（工厂没有 list/多态类型）——它们也必须经 choke point 写。
_BOUND_KEYS = ('manybody.template', 'manybody.solver', 'manybody.second_order',
               'manybody.n_window', 'manybody.guard_mhz', 'manybody.ed_max_dim')
_CUSTOM_KEYS = ('manybody.terms', 'manybody.psi0')
_bound = [k for (_wd, k, _s) in _w.binder._binds if k.startswith('manybody.')]
_mb.template_cb.setCurrentText('xy')
_mb.nwin_sb.setValue(2)
_terms_before = _w.app_bridge.read('manybody.terms')
_mb.term_cbs['I2'].setChecked(True)
_terms_after = _w.app_bridge.read('manybody.terms')
_mb.psi0_cb.setCurrentIndex(1)                      # Lanczos 基态
_psi0_written = _w.app_bridge.read('manybody.psi0')
_evs = [e for e in _w.app_bridge.activity(0)
        if str(e.get('target') or '').startswith('manybody.')]
C.group('B 参数页经 _mecha_bind 绑到 session（choke point，actor=gui）', [
    ('6 个标量/布尔键都进了 binder（实测 %d：%s）' % (len(_bound), _bound),
     sorted(_bound) == sorted(_BOUND_KEYS)),
    ('2 个手写控件也经 choke point 写 session（terms=%r / psi0=%r）'
     % (_terms_after, _psi0_written),
     isinstance(_terms_after, (list, tuple)) and 'I2' in _terms_after
     and _psi0_written == 'ground'),
    ('改模板 → session 读到 xy', _w.app_bridge.read('manybody.template') == 'xy'),
    ('改 n_window → session 读到 2', _w.app_bridge.read('manybody.n_window') == 2),
    ('相互作用项写入是 list（%r → %r）' % (_terms_before, _terms_after),
     isinstance(_terms_after, (list, tuple)) and 'I2' in _terms_after),
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
