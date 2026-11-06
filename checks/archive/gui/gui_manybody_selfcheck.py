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
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
os.environ.setdefault('EL_NO_AI_SERVER', '1')
os.environ.setdefault('RE0_V2_DATA_ROOT', tempfile.mkdtemp(prefix='guimb_'))

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
_BOUND_KEYS = ('manybody.solver', 'manybody.want', 'manybody.eq_backend',
               'manybody.chi_max', 'manybody.second_order',
               'manybody.n_window', 'manybody.guard_mhz', 'manybody.ed_max_dim')
_bound = [k for (_wd, k, _s) in _w.binder._binds if k.startswith('manybody.')]
# 模板/相互作用项已从 UI 撤下（相互作用恒全开）⇒ 不再测其控件绑定。
_mb.nwin_sb.setValue(2)
_nwin_written = _w.app_bridge.read('manybody.n_window')
_evs = [e for e in _w.app_bridge.activity(0)
        if str(e.get('target') or '').startswith('manybody.')]


# psi0 三态（含时演化仍需；模板/相互作用项控件已撤下 ⇒ 不再测其绑定）
_mb.psi0_cb.setCurrentIndex(1)
_psi0_ground = _w.app_bridge.read('manybody.psi0')
_mb.psi0_cb.setCurrentIndex(0)
_psi0_none = _w.app_bridge.read('manybody.psi0')
_mb.psi0_cb.setCurrentIndex(2)
_mb.psi0_edit.setText('grg')
_mb._write_psi0()
_psi0_str = _w.app_bridge.read('manybody.psi0')

C.group('B 参数页经 _mecha_bind / choke point 绑到 session（含三态语义）', [
    ('标量/布尔键都进了 binder（实测 %d：%s）' % (len(_bound), _bound),
     sorted(_bound) == sorted(_BOUND_KEYS)),
    ('改 n_window → session 读到 2', _nwin_written == 2),
    ('ψ0 三态：Lanczos=%r / 全基态=%r / 模式串=%r'
     % (_psi0_ground, _psi0_none, _psi0_str),
     _psi0_ground == 'ground' and _psi0_none is None and _psi0_str == 'grg'),
    ('总线事件 actor=gui（人机同路 I5，%d 条）' % len(_evs),
     bool(_evs) and all(e.get('actor') == 'gui' for e in _evs)),
])

# ---------------- C. 点云 ↔ session（含空表清除） ----------------
# g/r 恒取自「能级组裁剪后的全部能级」（_levels→build_states）。GUI 接线判据隔离 ARC 裁剪：
# 直接打桩 _levels 为「恰好 2 个」，走通加原子路径（能级计数本身由 core 判据覆盖）。
_LV2 = [(5.0, 0.0, 0.5, -0.5), (70.0, 0.0, 0.5, -0.5)]
_mb._levels = lambda: list(_LV2)
_mb._atoms = []
_mb._write_atoms('判据：清空')
_mb._add_atom()
_mb._add_atom()
_atoms = _w.app_bridge.read('manybody.atoms') or []
_rows = _mb.table.rowCount()
_ok_shape = all(isinstance(a.get('g'), list) and len(a['g']) == 4
                and isinstance(a.get('pos'), list) and len(a['pos']) == 3
                for a in _atoms)
_gr_from_groups = all(list(a.get('g') or []) == list(_LV2[0])
                      and list(a.get('r') or []) == list(_LV2[1])
                      for a in _atoms)
_mb._atoms = []
_mb._write_atoms('判据：清空')
_cleared = _w.app_bridge.read('manybody.atoms')
# 守卫：能级数≠2 时 → 不加原子，顶部出现“⚠”一行警告。
_mb._levels = lambda: list(_LV2) + [(70.0, 1.0, 1.5, -0.5)]   # 3 个 → 不合法
_before = len(_mb._atoms)
_mb._add_atom()
_after = len(_mb._atoms)
_warn_txt = _mb.level_status.text()
_mb._levels = lambda: list(_LV2)                                # 恢复恰好 2 个
C.group('C 点云：写 session（choke point）+ 空表写 None 清除', [
    ('加 2 个原子 → session 有 2 个', len(_atoms) == 2),
    ('表行数与 session 一致（%d）' % _rows, _rows == 2),
    ('每个原子 g/r 是 4 元组、pos 是 3 元组（过 validate_atom_list）', _ok_shape),
    ('g/r 来自能级组裁剪后的 2 个能级（非原子表自带）', _gr_from_groups),
    ('删到空 → 写 None（registry 拒空 list，故必须清除而非写空表）',
     _cleared is None),
    ('标签随原子数更新（%s）' % _mb.atom_label.text(),
     'dim=2^2=4' in _mb.atom_label.text() or 'dim=2^0=1' in _mb.atom_label.text()),
    ('能级数≠2 → 不加原子且顶部“⚠”警告（%d→%d）' % (_before, _after),
     _after == _before and '⚠' in _warn_txt),
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

# ---------------- F. M2 结果图：⟨n_i⟩ / C_ij 由**联合 pops 重算**（不依赖 P1-8） ----------------
# 只吃概率向量 ⇒ GUI 侧可自算；且**必须与直接调用 core 观测量逐值一致**（防"自己造一套"）。
import numpy as _np                                        # noqa: E402
from src.core.manybody.observables import (                # noqa: E402
    bit_matrix as _bits, corr_matrix as _corr)

_N = 2
_mb._run_n_atoms = _N
_t = _np.linspace(0.0, 10.0, 5)
_joint = _np.array([[1.0, 0, 0, 0], [0.5, 0.5, 0, 0], [0.25, 0.25, 0.25, 0.25],
                    [0.1, 0.2, 0.3, 0.4], [0.0, 0.0, 0.5, 0.5]])
_mb._draw_result({'t': _t, 'pops': _joint,
                  'states': [{'label': '#%d' % i} for i in range(2 ** _N)]})
_exp_atom = _joint @ _bits(_N, 2 ** _N).T
_exp_corr = _corr(_bits(_N, 2 ** _N), _joint[-1])
_got_atom = _np.array([ln.get_ydata() for ln in _mb.res_ax_atom.lines])
_got_corr = (_mb.res_ax_corr.images[0].get_array()
             if _mb.res_ax_corr.images else None)
# 边缘退化（dim 超全布居上限时引擎只出逐原子边缘）→ C_ij 必须**明说不可得**
_mb._draw_result({'t': _t, 'pops': _np.array([[0., 0.], [0.3, 0.1], [0.5, 0.2],
                                              [0.6, 0.3], [0.4, 0.4]]),
                  'states': [{'label': 'atom0'}, {'label': 'atom1'}]})
_marg_text = ' '.join(x.get_text() for x in _mb.res_ax_corr.texts)
C.group('F M2 结果图：⟨n_i⟩/C_ij 由联合 pops 重算（不依赖 P1-8）', [
    ('联合情形：逐原子 ⟨n_i⟩ 画出 %d 条（=原子数）' % len(_mb.res_ax_atom.lines),
     len(_got_atom) == _N),
    ('⟨n_i⟩ 与直接调用 core.observables 逐值一致',
     _got_atom.shape == _exp_atom.T.shape and _np.allclose(_got_atom, _exp_atom.T)),
    ('联合情形：C_ij 热图已画且与直接调用一致',
     _got_corr is not None and _np.allclose(_got_corr, _exp_corr)),
    ('边缘退化：C_ij **不画**且明说原因（%s）' % _marg_text[:28],
     not _mb.res_ax_corr.images and '不可得' in _marg_text),
    ('边缘退化：⟨n_i⟩ 仍可用（%d 条线）' % len(_mb.res_ax_atom.lines),
     len(_mb.res_ax_atom.lines) == _N),
    ('未做项措辞已更新：C_ij 不再列为"未做"，S 曲线列为"未做（依赖 P1-8）"',
     all('C_ij' not in x for x in UMB.NOT_YET)
     and any('S' in x and 'P1-8' in x for x in UMB.NOT_YET)),
])

# ---------------- G. M1 点云画布交互（点击加原子 / 拖动移动，松开才落账） ----------------
class _Ev:
    """matplotlib 事件的鸭子替身（处理器只读 button/xdata/ydata）。"""

    def __init__(self, x, y, button=1):
        self.xdata, self.ydata, self.button = x, y, button


def _n_atom_events():
    return len([e for e in _w.app_bridge.activity(0)
                if e.get('target') == 'manybody.atoms'])


_mb._atoms = []
_mb._write_atoms('判据：清空')
_mb._rebuild_table()
_n0 = _n_atom_events()
# ① 左键点空白 → 在该位置加一个原子
_mb._on_cloud_press(_Ev(12.5, -3.25))
_a1 = _w.app_bridge.read('manybody.atoms') or []
_added_at = (list(_a1[0]['pos']) if _a1 else None)
_n_after_add = _n_atom_events()
# ② 左键点中它 + 拖动 3 次 + 松开 → 位置更新，且**只落账一次**
_mb._on_cloud_press(_Ev(12.5, -3.25))          # 命中已有原子 → 起拖
for _x in (14.0, 15.0, 16.0):
    _mb._on_cloud_move(_Ev(_x, -5.0))
_n_during_drag = _n_atom_events()
_mb._on_cloud_release(_Ev(16.0, -5.0))
_a2 = _w.app_bridge.read('manybody.atoms') or []
_n_after_drag = _n_atom_events()
# ③ 拖动中不落账（事件数不变），松开后 +1
C.group('G M1 点云画布交互（点击加原子 / 拖动移动；松开才落账）', [
    ('空白点击 → 原子落在点击处（%s）' % (_added_at,),
     len(_a1) == 1 and _added_at is not None
     and abs(_added_at[0] - 12.5) < 1e-9 and abs(_added_at[1] + 3.25) < 1e-9),
    ('点击加原子经 choke point 落 session（+%d 事件）' % (_n_after_add - _n0),
     _n_after_add == _n0 + 1),
    ('拖动 3 次**期间不写 session**（事件数不变：%d）' % _n_during_drag,
     _n_during_drag == _n_after_add),
    ('松开后落账一次（+%d）且位置=最后落点'
     % (_n_after_drag - _n_after_add),
     _n_after_drag == _n_after_add + 1 and len(_a2) == 1
     and abs(_a2[0]['pos'][0] - 16.0) < 1e-9 and abs(_a2[0]['pos'][1] + 5.0) < 1e-9),
    ('表与点云同步（行数=1）', _mb.table.rowCount() == 1),
    ('未做项措辞已更新：点云交互不再列为"未做"',
     all('画布交互' not in x for x in UMB.NOT_YET)),
])

# ---------------- H. 模式分家：演化 / 平衡态 只显示对应组件 ----------------
# 用 isHidden()（只看显式 hide 标志）——窗口在 offscreen 下没 show，isVisible() 恒 False。
_mb._set_mode('evolve')
app.processEvents()
_h0 = dict(ev=_mb.sec_evolve.isHidden(), eq=_mb.sec_eq.isHidden(),
           res=_mb.res_canvas.isHidden(), eqc=_mb.eq_canvas.isHidden(),
           want=_w.app_bridge.read('manybody.want'),
           btn=_mb.run_btn.text())
_mb._set_mode('eq')
app.processEvents()
_h1 = dict(ev=_mb.sec_evolve.isHidden(), eq=_mb.sec_eq.isHidden(),
           res=_mb.res_canvas.isHidden(), eqc=_mb.eq_canvas.isHidden(),
           want=_w.app_bridge.read('manybody.want'),
           btn=_mb.run_btn.text())
_mb.want_cb.setCurrentText('thermal')
_want_thermal = _w.app_bridge.read('manybody.want')
_mb._set_mode('evolve')
app.processEvents()
_h2 = dict(ev=_mb.sec_evolve.isHidden(), eq=_mb.sec_eq.isHidden(),
           want=_w.app_bridge.read('manybody.want'), btn=_mb.run_btn.text())
# 平衡态结果渲染（三种 want 各走一遍，不抛异常且主格有内容）——**当场**抓摘要文本，
# 因为下一次 _draw_eq_result 会 clear 并重写 eq_summary。
_mb._draw_eq_result(dict(want='gs', backend='ed', n_atoms=3, dim=8,
                         E0=-1.5, gap=0.4, n_atom=[0.3, 0.2, 0.35]))
_n_gs_bars = len(_mb.eq_ax_main.patches)
_txt_gs = _mb.eq_summary.text()
_mb._draw_eq_result(dict(want='thermal', backend='kpm', betas=[0.1, 1.0, 10.0],
                         E=[1.0, 0.2, 0.01], C=[0.1, 0.4, 0.02]))
_n_th_lines = len(_mb.eq_ax_main.lines)          # E(β) 在本轴；C(β) 在 twin 轴
_txt_th = _mb.eq_summary.text()
_mb._draw_eq_result(dict(want='dos', backend='kpm', energies=[-1.0, 0.0, 1.0],
                         rho=[0.2, 1.0, 0.3]))
_n_dos_lines = len(_mb.eq_ax_main.lines)
_txt_dos = _mb.eq_summary.text()
_mb._draw_eq_result(None)
_ev_want = [e for e in _w.app_bridge.activity(0)
             if e.get('target') == 'manybody.want']
def _in(sec, w):
    """w 是否挂在分区 sec 下（沿 parent 链找；Qt 会把布局内控件直接挂到布局宿主）。"""
    p = w.parent()
    while p is not None:
        if p is sec:
            return True
        p = p.parent()
    return False


C.group('H 模式分家：演化/平衡态 只显示对应组件 + want 互斥写回', [
    ('演化模式：只显示演化分区与演化结果图（%s）' % _h0,
     not _h0['ev'] and _h0['eq'] and not _h0['res'] and _h0['eqc']
     and (_h0['want'] or '') == '' and '演化' in _h0['btn']),
    ('平衡态模式：只显示平衡态分区与平衡态结果（%s）' % _h1,
     _h1['ev'] and not _h1['eq'] and _h1['res'] and not _h1['eqc']
     and _h1['want'] in ('gs', 'thermal', 'dos') and '平衡态' in _h1['btn']),
    ('两次切换（eq → evolve）后 want 为空串且组件复原（%s）' % _h2,
     (_h2['want'] or '') == '' and not _h2['ev'] and _h2['eq']),
    ('平衡态分区内选 thermal → session want=thermal（%r）' % _want_thermal,
     _want_thermal == 'thermal'),
    ('两分区控件不重叠：含时求解器/ψ0/ED上限只在演化区，后端/betas 只在平衡态区',
     _in(_mb.sec_evolve, _mb.solver_cb) and _in(_mb.sec_evolve, _mb.psi0_cb)
     and _in(_mb.sec_evolve, _mb.edmax_sb)
     and _in(_mb.sec_eq, _mb.backend_cb) and _in(_mb.sec_eq, _mb.betas_edit)
     and not _in(_mb.sec_eq, _mb.solver_cb)
     and not _in(_mb.sec_evolve, _mb.backend_cb)),
    ('gs 结果：⟨n_i⟩ 柱 %d 根且读数带 E0（%s）' % (_n_gs_bars, _txt_gs[:46]),
     _n_gs_bars == 3 and 'E0' in _txt_gs and 'want=gs' in _txt_gs),
    ('thermal 结果：E(β) 曲线 %d 条且读数报 β 点数（%s）'
     % (_n_th_lines, _txt_th[:46]),
     _n_th_lines == 1 and 'β 点数 = 3' in _txt_th and 'want=thermal' in _txt_th),
    ('dos 结果：ρ(E) 曲线 %d 条且读数带分箱/max ρ（%s）'
     % (_n_dos_lines, _txt_dos[:46]),
     _n_dos_lines == 1 and 'want=dos' in _txt_dos and '分箱 = 3' in _txt_dos),
    ('want 写入经 choke point 且 actor=gui（%d 条）' % len(_ev_want),
     bool(_ev_want) and all(e.get('actor') == 'gui' for e in _ev_want)),
])

_ok = C.summary()
sys.stdout.flush()
os._exit(0 if _ok else 1)
