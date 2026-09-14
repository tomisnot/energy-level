# -*- coding: utf-8 -*-
"""UI 自检（离屏运行，无显示依赖）：主窗口/布局/波形面板/动态页/进度。

覆盖（每一条是一个**不变式包**，包内子条件累计进 `SUB=`）：
  1. 主窗口冒烟：两页（静态/动态）切换、能级组添加、QSS 下拉三角
  2. 波形面板：波形增删、参数编辑、序列化/恢复
  3. 动态页：初态/态选择、波形光强预览、荧光读出、自发辐射三档、端到端演化
  4. 进度数组
  5. 存档恢复：登记册 v3 / 旧档兼容 / GUI 两个按钮真实入口（往返一致）

**计数模型**（2026-09-14 重设计，见 checks/checklib.py）：默认摘要只数**包**
（`PASS=` 项数），包内子条件累计进 `SUB=`。同一个不变式的多条碎片断言算 1 项，
但覆盖面不缩水（`SUB` == 改前逐条计数的项数）；`SELFTEST_VERBOSE=1` 展开子条件。

运行：D:\\python310\\python.exe -u checks\\ui_selftest.py
（selftest_all.py 也调用本文件，作为 UI 回归。）
"""
import io
import os
import sys
import threading
import time
import contextlib
import json

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from checks.checklib import Checker                              # noqa: E402

import numpy as np
from PyQt5.QtWidgets import (QApplication, QMessageBox, QComboBox,
                             QDoubleSpinBox, QPushButton, QWidget,
                             QSpinBox)

QMessageBox.information = staticmethod(lambda *a, **k: None)
QMessageBox.warning = staticmethod(lambda *a, **k: None)

import 能级 as main_mod
from src.core import core, waveforms
from src import ui_theme, archive
from src.core.evolve_util import decay, auto_decays
from src.core.field_build import field
from src.core.atomic_levels import make_levels, gaussian, square
from src.core.evolve_rwa import evolve

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ------------------- 计数模型：不变式包（见 checks/checklib.py）-------------------
# 每个 group(...) 开一个包，其后相邻的 check(...) 全部变成该包的**子条件**：
# 整包只计 1 项，子条件逐个进 SUB=（覆盖面不缩水；不过时自动展开子条件）。
# check(...) 的名称与判据表达式一律原样保留——只是不再各自计 1 项。
C = Checker('ui_selftest')
_N_CHECK = 0             # check() 调用计数：与 C.n_sub 对账，防打包漏项
_PKG = None


class _Pkg:
    """一个不变式包（先缓冲子条件，关闭时一次性交给 C.group 登记）。"""

    def __init__(self, name, detail=''):
        self.name, self.detail, self.subs = name, detail, []

    def flush(self):
        C.group(self.name, self.subs, self.detail)


def group(name, detail=''):
    """开新的不变式包：此后 check() 的断言都记入它（直到下一个 group）。"""
    global _PKG
    if _PKG is not None:
        _PKG.flush()
    _PKG = _Pkg(name, detail)


def flush():
    """关闭最后一个包（main 收尾）。"""
    global _PKG
    if _PKG is not None:
        _PKG.flush()
        _PKG = None


def check(name, cond):
    """薄包装：判据表达式不变，只登记为**当前包的子条件**。"""
    global _N_CHECK
    if _PKG is None:
        raise RuntimeError(f'check() 出现在任何 group() 之前：{name}')
    _N_CHECK += 1
    _PKG.subs.append((name, bool(cond)))
    return bool(cond)


# ================= 1. 主窗口冒烟 =================
def test_main(app):
    group('主窗口冒烟（构造/主题/QSS 三角/页面栈/能级组/状态栏）',
          '主窗口能构造且全局主题生效；QSS 含 QComboBox::down-arrow 且资产在位；'
          '两页切换、能级组可加、状态栏日志可见')
    win = main_mod.MainWindow()
    win.show(); app.processEvents()
    check('主窗口构造 + 全局主题', bool(app.styleSheet()))
    # ---- t29：QComboBox down-arrow 三角（全局 QSS + 资产）----
    check('QSS 含 QComboBox::down-arrow',
          'QComboBox::down-arrow' in ui_theme.QSS)
    _arrow = os.path.join(os.path.dirname(ui_theme.__file__),
                          'assets', 'down_arrow.png')
    check('down-arrow 资产存在且非空',
          os.path.isfile(_arrow) and os.path.getsize(_arrow) > 200)
    for idx in (0, 1):
        win.canvas_tabs.setCurrentIndex(idx); app.processEvents()
    check('两页切换（页面栈）', win.page_stack.currentIndex() == 1)
    for n, l, j in ((60, 0, 0.5), (60, 1, 0.5)):
        win.group_panel.n_spin.setValue(n); win.group_panel.l_spin.setValue(l)
        win.group_panel.j_spin.setValue(j)
        win.group_panel.add_group()
    app.processEvents()
    check('添加能级组', len(win.groups) == 2)
    # ---- 用户裁决（2026-09-14）：逐子能级裁剪的**唯一入口 = 左栏右键菜单** ----
    # 这一条此前**没有任何自检**（裁决只写在注释/手册里）——现在补上，防回归：
    # ① 工具条上不得再出现「子能级…」按钮（那一行已删干净）
    btns = [b.text() for b in win.group_panel.findChildren(QPushButton)]
    check('能级组工具条上没有「子能级…」按钮（裁决：入口收敛到右键菜单）',
          not any('子能级' in t for t in btns))
    # ② 左栏能级组列表的右键菜单必须有「子能级…」（右键命中哪行改哪行）
    from PyQt5.QtWidgets import QMenu
    _seen = {}
    _orig_exec = QMenu.exec_
    QMenu.exec_ = lambda self, *a, **k: (_seen.setdefault(
        'acts', [x.text() for x in self.actions()]) and None)   # noqa: E731
    try:
        _it = win.group_panel.group_list.item(0)
        _pos = win.group_panel.group_list.visualItemRect(_it).center()
        win.group_panel.show_context_menu(_pos)
    finally:
        QMenu.exec_ = _orig_exec
    _acts = _seen.get('acts') or []
    check('左栏右键菜单含「子能级…」（逐子能级裁剪的唯一入口）',
          any('子能级' in t for t in _acts))
    win.log('测试日志消息')
    check('状态栏日志', '测试日志消息' in win.statusBar().currentMessage())
    return win


# ================= 2. 波形面板（无通道、无独立窗口） =================
def test_wave_panel(app):
    group('波形面板：增删 / 参数编辑 / 序列化往返',
          '初始为空（仅 stretch）、增删折叠组、默认展开且参数控件齐、频率改动生效 + '
          'changed 信号、✕ 删除、serialize→restore 保型')
    wp = __import__('src.ui.ui_waves', fromlist=['WavePanel']).WavePanel()
    wp.show(); app.processEvents()
    check('波形面板初始为空（仅 stretch）', len(wp.waves) == 0)
    wp._add('gauss'); app.processEvents()
    wp._add('square'); app.processEvents()
    check('添加高斯/方波 → 2 个折叠组', len(wp.waves) == 2
          and wp._box.count() == 3)          # 2 组 + stretch
    w = wp.waves[0]
    check('默认展开 + 参数控件存在',
          getattr(wp._box.itemAt(0).widget(), '_expanded', False)
          and len(wp._box.itemAt(0).widget().findChildren(QDoubleSpinBox)) >= 5)
    # 频率修改 → 波形对象更新 + changed 信号（默认 6.8e6 MHz → THz 档 coef=1e6）
    got = []
    wp.changed.connect(lambda: got.append(1))
    group0 = wp._box.itemAt(0).widget()
    spin0 = group0.findChildren(QDoubleSpinBox)[0]   # 第一个 spin = 频率（THz 显示）
    spin0.setValue(7.5); app.processEvents()
    check('频率修改生效 + changed 信号',
          abs(w.freq_mhz - 7.5e6) < 1e-3 and len(got) >= 1)
    # 删除（叉）：删第 2 个（方波）
    wp._delete(1); app.processEvents()
    check('✕ 删除波形', len(wp.waves) == 1
          and isinstance(wp.waves[0].envelope, waveforms.GaussianEnv))
    # 序列化 roundtrip
    data = waveforms.serialize_waves(wp.waves)
    r = waveforms.restore_waves(data)
    check('波形序列化/恢复', len(r) == 1 and isinstance(r[0].envelope,
          waveforms.GaussianEnv))
    wp.close()


# ================= 3. 动态页：初态/波形预览/自发辐射 =================
def test_dynamic(app, win):
    group('动态页初态 / 场面板 / 波形光强预览',
          '初态索引保留、布居显示态数默认 8、B/E 行随页面切换、波形→多通道 y 轴、'
          '包络采样对齐输出网格含频率光强、内部按平均布居选态')
    dyn = win.dynamic
    for n, l, j in ((60, 1, 1.5), (6, 1, 1.5)):
        win.group_panel.n_spin.setValue(n); win.group_panel.l_spin.setValue(l)
        win.group_panel.j_spin.setValue(j)
        win.group_panel.add_group()
    app.processEvents()
    dyn.psi_combo.setCurrentIndex(2)
    sel = dyn.psi_combo.currentIndex()
    dyn.refresh_groups(); app.processEvents()
    check('初态选择保留', dyn.psi_combo.currentIndex() == sel)
    check('布居显示态数参数默认 8', dyn.topn_sb.value() == 8)
    # 场面板两页共用：动态页显示 B/E 行（光场区隐藏），切静态页恢复
    win.canvas_tabs.setCurrentIndex(1); app.processEvents()
    check('动态页显示场面板(B/E 行)',
          win.field.isVisible() and win.field.b_spin.isVisible()
          and not win.field.lights_box.isVisible())
    win.canvas_tabs.setCurrentIndex(0); app.processEvents()
    check('静态页显示完整场面板', win.field.lights_box.isVisible())
    win.canvas_tabs.setCurrentIndex(1); app.processEvents()

    # 波形光强预览：添加波形 → 顶部图多通道独立 y 轴
    dyn.wave_panel._add('gauss'); app.processEvents()
    dyn.wave_panel._add('square'); app.processEvents()
    check('添加 2 波形 → wave_data 更新', len(dyn.wave_data) == 2)
    check('顶部图多通道光强 y 轴渲染',
          len(dyn.fig_top.axes) >= 2)
    t = np.linspace(0, 1500.0, 1501)
    wd = waveforms.waveforms_to_env(dyn.wave_panel.waves, t)
    check('包络采样=输出网格 + 含频率/光强', len(wd) > 0
          and len(wd[0][1]) == len(t) and wd[0][3] > 0
          and len(wd[0]) >= 5 and wd[0][4] > 0)
    states = dyn._current_states()
    N = len(states)
    res = {'t': t, 'pops': np.zeros((len(t), N))}
    res['pops'][:, 0] = 0.6
    top = dyn._top_pop_rows(states, res)
    check('内部按平均布居选态（态 0 入选）',
          0 in top and len(top) == min(8, N))
    dyn._draw_bot(states, res); app.processEvents()

    # ---- t71：方法选项收敛（默认 mixed、无 ip4/auto、无 near_cutoff 控件）----
    group('方法选项收敛（t71）',
          '默认 mixed、下拉恰为 sambe/hard/mixed、无 near_cutoff 控件')
    check('方法默认 mixed', dyn.method_cb.currentText() == 'mixed')
    check('方法选项 sambe/hard/mixed',
          [dyn.method_cb.itemText(i) for i in range(dyn.method_cb.count())]
          == ['sambe', 'hard', 'mixed'])
    check('无 near_cutoff 控件', not hasattr(dyn, 'near_cutoff_sb'))

    # ---- t25/t27/t30：显示选项 + 荧光读出（单原子绘制 + 双原子约化）----
    group('荧光读出绘制（t25/t27/t30）',
          '读出态下拉含全部能级 + 复杂配置哨兵、不占满宽度、荧光读出时 topn 隐藏、'
          '单原子单线绘制、复杂配置 Σwᵢ·Pᵢ 加权绘制')
    dyn.display_mode_cb.setCurrentIndex(1); app.processEvents()
    check('荧光读出态下拉出现（含全部能级 + 复杂配置）',
          dyn.readout_cb.isVisible() and dyn.readout_cb.count() == N + 1
          and dyn.readout_cb.itemData(dyn.readout_cb.count() - 1) == -1)
    check('readout_cb 不占满（t27 去 stretch）',
          dyn.readout_cb.width() < dyn.width() * 0.6)
    check('荧光读出时 topn 隐藏（t27 互斥）', not dyn.topn_sb.isVisible())
    res2 = {'t': t, 'pops': res['pops']}
    dyn._run_states = list(states)
    dyn._last_res = res2
    dyn._draw_bot(states, res2); app.processEvents()
    check('单原子荧光读出单线绘制', len(dyn.ax_bot.lines) == 1)
    # t30：复杂配置加权读出（权重写回后重绘 → Σw·P 单线）
    dyn._readout_weights = np.ones(N) / N
    dyn.readout_cb.setCurrentIndex(dyn.readout_cb.count() - 1)
    dyn._redraw_last()          # 索引可能已在哨兵项（setCurrentIndex 同值不触发）
    app.processEvents()
    check('复杂配置读出绘制（Σ wᵢ·Pᵢ）',
          len(dyn.ax_bot.lines) == 1
          and '权重全 0' not in dyn.ax_bot.lines[0].get_label())
    # t30：复杂配置弹窗（非模态常驻）——t32 无确定/取消，spinbox 实时写回；
    # 关窗清引用
    group('复杂配置弹窗（t30/t32）',
          '非模态常驻显示、无确定/取消按钮（仅 spinbox）、spinbox 实时写回权重且'
          '窗口保持、关窗清引用')
    dyn._readout_weights = np.zeros(N)
    dyn._open_readout_config()
    app.processEvents()
    check('复杂配置弹窗非模态显示', getattr(dyn, '_readout_dlg', None)
          is not None and dyn._readout_dlg.isVisible())
    sp = dyn._readout_dlg.findChildren(
        __import__('PyQt5.QtWidgets', fromlist=['QDoubleSpinBox']).QDoubleSpinBox)
    check('弹窗无确定/取消按钮（仅 spinbox）',
          len(dyn._readout_dlg.findChildren(
              __import__('PyQt5.QtWidgets',
                         fromlist=['QPushButton']).QPushButton)) == 0)
    sp[0].setValue(0.5)             # 触发 valueChanged → 实时写回 + 重绘
    app.processEvents()
    check('spinbox 实时写回权重（窗口保持）',
          len(dyn._readout_weights) == N
          and np.isclose(dyn._readout_weights[0], 0.5)
          and dyn._readout_dlg.isVisible())
    dyn._readout_dlg.close()
    app.processEvents()
    from PyQt5.QtCore import QEvent
    app.sendPostedEvents(None, QEvent.DeferredDelete)   # WA_DeleteOnClose 延迟删除
    check('关窗清引用', getattr(dyn, '_readout_dlg', None) is None)
    group('读出/显示切换重绘、互斥与基组重建回退',
          '双原子复杂配置 A/B 双线、改读出态/显示态数触发重绘、模式切换后 topn 互斥'
          '恢复、基组重建重置读出权重、复杂配置不作默认（回退真实能级）')
    res3 = {'t': t, 'pops_2d': np.zeros((len(t), N * N))}
    res3['pops_2d'][:, 3] = 0.4
    dyn._is_two_body = True
    dyn._draw_bot_pair(res3); app.processEvents()
    check('双原子复杂配置 A/B 双线', len(dyn.ax_bot.lines) == 2)
    dyn._is_two_body = False
    # t30：改读出态/显示态数触发重绘（解暂存直接转换）
    dyn.readout_cb.setCurrentIndex(0); app.processEvents()
    check('改读出态触发重绘（仍单线）', len(dyn.ax_bot.lines) == 1)
    dyn.display_mode_cb.setCurrentIndex(0); app.processEvents()
    n0 = len(dyn.ax_bot.lines)
    dyn.topn_sb.setValue(3); app.processEvents()
    check('改显示态数触发重绘', len(dyn.ax_bot.lines) >= 1)
    dyn.display_mode_cb.setCurrentIndex(0); app.processEvents()
    check('显示模式恢复态布居', not dyn.readout_cb.isVisible())
    check('态布居时 topn 显示（t27 互斥）', dyn.topn_sb.isVisible())
    # t30：读出权重/读出态=显示参数（v3 不入册不存档）；基组重建重置权重、
    # 哨兵项回退真实能级
    dyn._readout_weights = np.array([0.1, 0.2, 0.3] + [0.0] * (N - 3))
    dyn.refresh_groups(); app.processEvents()
    check('基组重建重置读出率权重', np.allclose(dyn._readout_weights, 0))
    dyn.readout_cb.setCurrentIndex(dyn.readout_cb.count() - 1)   # 复杂配置哨兵
    dyn.refresh_groups(); app.processEvents()
    check('复杂配置不可作默认（重建回退真实能级）',
          dyn.readout_cb.currentData() != -1)

    # ---- 自发辐射三档：自动（ARC 寿命）/ 禁用 / 自定义 ----
    group('自发辐射三档 + 通道增删 + 自定义/自动端到端',
          '默认自动档且配置区隐藏、自定义档显示、禁用档隐藏、衰变通道增删、'
          '自定义纯衰变 loss 曲线 + Σpops+loss=1、ARC 寿命只出 6P 通道、'
          '自动档端到端 6P 按寿命衰减')
    check('自发辐射默认自动档 + 配置区隐藏',
          dyn.decay_mode_cb.currentIndex() == 0
          and not dyn.decay_custom_box.isVisible())
    dyn.decay_mode_cb.setCurrentIndex(2); app.processEvents()
    check('自定义档显示配置区', dyn.decay_custom_box.isVisible())
    dyn.decay_mode_cb.setCurrentIndex(1); app.processEvents()
    check('禁用档隐藏配置区', not dyn.decay_custom_box.isVisible())
    dyn.decay_mode_cb.setCurrentIndex(2); app.processEvents()

    # ---- 自定义通道配置 + 纯衰变端到端 ----
    dyn.decay_from_cb.setCurrentIndex(0)
    dyn.decay_gamma_sb.setValue(5.0)
    dyn.decay_to_cb.setCurrentIndex(dyn.decay_to_cb.count() - 1)   # （丢失）
    dyn._add_decay(); app.processEvents()
    check('衰变通道添加', len(dyn.decay_cfgs) == 1
          and dyn.decay_list.count() == 1)
    dyn._del_decay(); app.processEvents()
    check('衰变通道删除', len(dyn.decay_cfgs) == 0
          and dyn.decay_list.count() == 0)
    # 纯衰变（无驱动波形）：运行 → loss 曲线 + Σpops+loss=1
    dyn.decay_from_cb.setCurrentIndex(0)
    dyn.decay_gamma_sb.setValue(8.0)
    dyn.decay_to_cb.setCurrentIndex(dyn.decay_to_cb.count() - 1)
    dyn._add_decay()
    dyn.psi_combo.setCurrentIndex(0)
    dyn.tmax_sb.setValue(200.0); dyn.npts_sb.setValue(101)
    dyn.method_cb.setCurrentText('hard')   # t71：ip4 档已删——hard 亦 NonHermitian 落点=丢失
    dyn.wave_panel.waves.clear()       # 纯衰变场景：无波形
    dyn._run()
    t0 = time.perf_counter()
    while getattr(dyn, '_worker', None) is not None \
            and time.perf_counter() - t0 < 120:
        time.sleep(0.05)
        app.processEvents()
    res = getattr(dyn, '_last_res', None)
    fresh = getattr(dyn, '_worker', None) is None
    loss_ok = res is not None and 'loss' in res \
        and res['loss'][-1] > 0.3 and res['loss'][-1] < 1.0
    cons = (np.max(np.abs(res['pops'].sum(axis=1) + res['loss'] - 1))
            if loss_ok else float('nan'))
    check(f"自定义纯衰变端到端：loss 曲线 + Σpops+loss=1"
          f"（loss[-1]={res['loss'][-1] if res is not None else None:g}"
          f"，cons={cons:g}，fresh={fresh}）",
          loss_ok and fresh and cons < 1e-4)
    # 注：t71 由 ip4(lindblad 解析) 改 hard(DOP853 数值)——loss 守恒精度从
    # ~1e-9 降为 ~1e-5（DOP853 容差级），阈值相应放宽；物理语义不变

    # ---- 自动档端到端：6P 低激发态按 ARC 寿命自动衰减 ----
    dyn.decay_mode_cb.setCurrentIndex(0); app.processEvents()
    st_now = dyn._current_states()
    i6p = next(i for i, s in enumerate(st_now) if '6P' in s['label'])
    auto_ch = auto_decays(dyn.cache, st_now)
    check('自动档按 ARC 寿命出 6P 通道（排除里德堡）',
          len(auto_ch) > 0
          and all(a['gamma_mhz'] > 1.0 for a in auto_ch)
          and all('6P' in st_now[a['from']]['label'] for a in auto_ch))
    dyn.psi_combo.setCurrentIndex(i6p)
    dyn._run()
    t0 = time.perf_counter()
    while getattr(dyn, '_worker', None) is not None \
            and time.perf_counter() - t0 < 120:
        time.sleep(0.05)
        app.processEvents()
    res = getattr(dyn, '_last_res', None)
    fresh = getattr(dyn, '_worker', None) is None
    loss_auto = res is not None and 'loss' in res and res['loss'][-1] > 0.80
    check(f"自动档端到端：6P 按 ARC 寿命衰变 → loss"
          f"（loss[-1]={res['loss'][-1] if res is not None else None:g}，"
          f"method={res.get('method') if res is not None else None}，"
          f"fresh={fresh}）",
          loss_auto and fresh)


# ================= 4. 波形序列化/恢复 =================
def test_wave_serde(app):
    group('波形序列化往返（包络 + 频率 + 光强 + 偏移）',
          '2 条波形往返后名称/频率/包络类型/起始时刻/光强逐项一致')
    waves = [
        waveforms.Wave('w1', 16826.7, waveforms.GaussianEnv(200.0, 40.0),
                       pol='π', offset_ns=10.0, intensity_mwcm2=1e5),
        waveforms.Wave('w2', 100.0, waveforms.SquareEnv(50.0, 20.0),
                       pol='σ+', offset_ns=0.0, intensity_mwcm2=2e3),
    ]
    data = waveforms.serialize_waves(waves)
    restored = waveforms.restore_waves(data)
    check('波形序列化/恢复（包络+频率+光强+偏移）',
          len(restored) == 2
          and restored[0].name == 'w1'
          and abs(restored[0].freq_mhz - 16826.7) < 1e-9
          and isinstance(restored[0].envelope, waveforms.GaussianEnv)
          and abs(restored[1].envelope.start_ns - 50.0) < 1e-9
          and abs(restored[1].intensity_mwcm2 - 2e3) < 1e-9)
    # t79：此处原有「波形 → real_field 场（近共振驱动对）」一项，依赖已删除的
    # waveforms.build_fields（旧 ip4/Lindblad 管线，现行三引擎都不用）——
    # 随函数一并删除。波形→场的集成由引擎自检覆盖
    # （checks/mixed_nolight_selfcheck.py、evolve_exact_pair_selfcheck.py
    #   均用现行入口 build_fields_exact）。


# ================= 5. 状态保存：登记册即存档（v3），双原子完整，往返一致 =================
def test_state_save(app):
    """archive.snapshot/restore：不存显示参数（topn/readout/psi 索引）；
    双原子科学参数（mode/pair 几何/A·B 波形/A·B 初态身份键）完整；旧档兼容。"""
    group('存档/恢复往返（登记册 v3 + 旧档兼容）',
          '存档不含显示参数（topn/readout/psi 索引）、双原子科学参数完整（v3 三面板'
          '独立）、恢复后模式与科学参数一致且显示参数保持当前值、v1 旧档兼容、'
          'v2 双原子档 A 数据自 waves 回填')
    win = main_mod.MainWindow()
    reg = win.param_reg
    for n, l, j in ((60, 0, 0.5), (60, 1, 0.5)):
        win.group_panel.n_spin.setValue(n)
        win.group_panel.l_spin.setValue(l)
        win.group_panel.j_spin.setValue(j)
        win.group_panel.add_group()
    app.processEvents()
    dyn = win.dynamic
    dyn._on_mode_changed(1)          # 双原子
    app.processEvents()
    dyn.vdd_R_sb.setValue(7.3)
    dyn.vdd_theta_sb.setValue(45.0)
    dyn.vdd_phi_sb.setValue(90.0)
    dyn.wave_panel_A._add('gauss')
    dyn.wave_panel_B._add('square')
    dyn.psi_combo_A.setCurrentIndex(1)
    dyn.psi_combo_B.setCurrentIndex(2)
    dyn.topn_sb.setValue(12)         # 显示参数：不应保存
    app.processEvents()
    snap = archive.snapshot(reg)
    p = snap['params']
    check('存档：不含显示参数（topn/readout/psi 索引）',
          'topn' not in str(p) and 'readout' not in str(p)
          and not any(k.endswith('.psi') for k in p))
    check('存档：双原子科学参数完整（v3 三面板独立：mode/pair/A·B 波形/初态键）',
          p['dynamic.mode'] == 1
          and p['dynamic.pair.R_um'] == 7.3
          and p['dynamic.pair.theta_deg'] == 45.0
          and p['dynamic.pair.phi_deg'] == 90.0
          and p['dynamic.waves_single'] == []     # 单原子面板未加波形
          and len(p['dynamic.waves_A']) == 1      # v3：A 独立存
          and len(p['dynamic.waves_B']) == 1
          and p['dynamic.psi_A'] is not None      # 身份键（非索引）
          and p['dynamic.psi_B'] is not None)
    # 往返
    dyn.vdd_R_sb.setValue(1.0)
    dyn.wave_panel_A.waves.clear()
    archive.restore(reg, snap)
    app.processEvents()
    check('恢复：双原子模式 + 科学参数一致（显示参数保持当前值）',
          dyn._is_two_body
          and abs(dyn.vdd_R_sb.value() - 7.3) < 1e-9
          and len(dyn.wave_panel_A.waves) == 1
          and dyn.psi_combo_A.currentIndex() == 1
          and dyn.topn_sb.value() == 12)
    # 旧档兼容①（_legacy/STIRAP_v1.json：顶层旧格式 v1，无 mode/pair/waves_A/B）
    data = json.load(open(os.path.join(
        _ROOT, 'save_state', '_legacy', 'STIRAP_v1.json'),
        encoding='utf-8'))
    archive.restore(reg, data)
    app.processEvents()
    check('旧档兼容：无 mode/pair → 单原子 + 波形恢复',
          not dyn._is_two_body and len(dyn.wave_panel.waves) >= 2)
    # 旧档兼容②（v2 双原子档：A 数据在 'waves'，waves_A 去重为空）
    v2 = {'meta': {'version': 2}, 'dynamic': {
              'dynamic': {'mode': 1, 'psi_A': 0, 'psi_B': 0,
                          't_max': 100.0, 'npts': 101, 'method': 'hard'},
              'pair': {'R_um': 5.0, 'theta_deg': 0.0, 'phi_deg': 0.0},
              'waves': p['dynamic.waves_A'], 'waves_A': [], 'waves_B': [],
              'decays': []}}
    archive.restore(reg, v2)
    app.processEvents()
    check('旧档兼容：v2 双原子档 A 数据自 waves 回填原子 A',
          dyn._is_two_body and len(dyn.wave_panel_A.waves) == 1)
    win.close(); app.processEvents()


# ================= 5. 进度数组 =================
def test_progress(app):
    """进度协议：正常完成 → prog[0]==1.0 且布居归一。

    t79：原「取消」子用例依赖已删除的 waveforms.build_fields（旧 ip4 管线），
    随函数一并删除。取消路径的覆盖仍在引擎级——见
    checks/evolve_dissipative_selfcheck.py D7（progress_arr 置位 → RuntimeError）。
    """
    group('进度数组协议', '正常完成 → prog[0]==1.0 且布居归一（<1e-6）')
    cache = core.ArcCache(__import__('arc').Rubidium87())

    # ---- 正常完成：进度到 1.0 + 归一 ----
    import mw_bench_firstprinciples as mb
    env = gaussian(200.0, 100.0)
    st2, i02, E2, fld2, _, om2, w2, _, _, _ = mb.build_s1(cache, 22.0, env)
    t2 = np.linspace(0, 400.0, 2001)
    ac2 = E2 - np.array([s['E_mhz'] for s in st2])
    pa2 = np.zeros(2)
    r2 = evolve(st2, [fld2], i02, t2, method='ip4', ac_shift_mhz=ac2,
                       progress_arr=pa2)
    check('正常完成 + 进度到 1.0 + 归一',
          pa2[0] == 1.0
          and np.max(np.abs(r2['pops'].sum(axis=1) - 1)) < 1e-6)


# ================= 6b. 存档/恢复走 GUI 真实方法（t78） =================
def test_archive_roundtrip(app, win):
    """驱动 app_entry 的 `_save_state` / `_load_state`（= 工具栏两个按钮的真实入口）。

    为什么要单独测：参数管理改造把存档实现换成 `archive.snapshot/restore(...)`
    却漏了 `from src import archive`，保存与恢复**两个按钮都 NameError**；
    而下面的 test_stir_states 是自己 `json.load` 旧存档再手工搭场，
    **覆盖不到这条路径**——所以 57/57 全绿时按钮仍不可用。此处 monkeypatch 掉
    文件对话框，直接跑真实方法，并覆盖 v3 自身与 v1 legacy 的迁移路径。
    """
    group('存档两个按钮的真实入口（app_entry._save_state / _load_state）',
          '_save_state 写出 v3（meta.version==3 + params）、_load_state 恢复 v3 自身'
          '与 v1 legacy 均无异常无警告')
    import app_entry as _ae
    import tempfile
    from PyQt5.QtWidgets import QFileDialog

    msgs = []
    _ae.QMessageBox.warning = staticmethod(
        lambda *a, **k: msgs.append(str(a[2]) if len(a) > 2 else ''))
    # 写到系统临时目录（不污染 save_state/）；删除套 try/except——受限沙箱的
    # 安全删除策略会**拒绝**删除操作（fail-closed），不能因此判测试失败
    # （src/ai/server.py 的 .mcp-port 清理也是这么做的）。
    out = os.path.join(tempfile.gettempdir(), '_ui_selftest_roundtrip.json')
    orig_save = QFileDialog.getSaveFileName
    orig_open = QFileDialog.getOpenFileName
    try:
        QFileDialog.getSaveFileName = staticmethod(lambda *a, **k: (out, ''))
        win._save_state()
        snap = (json.load(open(out, encoding='utf-8'))
                if os.path.isfile(out) else {})
        check('存档：_save_state 写出 v3（meta.version==3 + params）',
              snap.get('meta', {}).get('version') == 3
              and isinstance(snap.get('params'), dict))
        for p in (out, os.path.join(_ROOT, 'save_state', '_legacy',
                                    'STIRAP_v1.json')):
            msgs.clear()
            QFileDialog.getOpenFileName = staticmethod(
                lambda *a, _p=p, **k: (_p, ''))
            try:
                win._load_state()
                exc = None
            except Exception as e:                       # noqa: BLE001
                exc = f'{type(e).__name__}: {e}'
            check(f'存档：_load_state 恢复 {os.path.basename(p)} 无异常/无警告'
                  f'（exc={exc} warn={msgs}）', exc is None and not msgs)
    finally:
        QFileDialog.getSaveFileName = orig_save
        QFileDialog.getOpenFileName = orig_open
        _ae.QMessageBox.warning = staticmethod(lambda *a, **k: None)
        try:
            if os.path.isfile(out):
                os.remove(out)
        except OSError:
            pass


# ================= 6. （t79 删除）STIRAP v1 存档恢复建场 =================
# 原 test_stir_states 恢复 save_state/_legacy/STIRAP_v1.json 后手工搭场，
# 依赖已删除的 waveforms.build_fields（旧 ip4 管线）——随函数一并删除。
# 存档/恢复本身仍由 test_archive_roundtrip（走 app_entry._save_state /
# _load_state 真实入口，覆盖 v3 自身与 v1 legacy 迁移）覆盖。


def main():
    app = QApplication([])
    win = test_main(app)
    test_wave_panel(app)
    test_dynamic(app, win)
    test_wave_serde(app)
    test_state_save(app)
    test_archive_roundtrip(app, win)
    test_progress(app)
    win.close(); app.processEvents()
    flush()                          # 关闭最后一个不变式包
    ok = C.summary()
    if C.n_sub != _N_CHECK:          # 打包漏项 → 必须响亮失败（不靠人眼盯数字）
        print(f'[ui_selftest] 内部错误：check() 调用 {_N_CHECK} 次，'
              f'只登记了 {C.n_sub} 条子条件（有断言没进包）', file=sys.stderr)
        ok = False
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
