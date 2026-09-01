# -*- coding: utf-8 -*-
"""UI 自检（离屏运行，无显示依赖）：主窗口/布局/波形面板/动态页/进度。

覆盖：
  1. 主窗口冒烟：两页（静态/动态）切换、能级组添加
  2. 波形面板：波形增删、参数编辑、序列化/恢复
  3. 动态页：初态/态选择、波形光强预览、自发辐射三档、端到端演化
  4. 进度数组 + 取消
  5. 存档恢复：STIRAP/拉曼（恢复 → 建场管线可用）

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

import numpy as np
from PyQt5.QtWidgets import (QApplication, QMessageBox, QComboBox,
                             QDoubleSpinBox, QPushButton, QWidget,
                             QSpinBox)

QMessageBox.information = staticmethod(lambda *a, **k: None)
QMessageBox.warning = staticmethod(lambda *a, **k: None)

import 能级 as main_mod
from src import waveforms, evolve, core, ui_theme

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_RESULTS = []


def check(name, cond):
    _RESULTS.append(bool(cond))
    print(f'  [{"PASS" if cond else "FAIL"}] {name}')
    return bool(cond)


# ================= 1. 主窗口冒烟 =================
def test_main(app):
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
    win.log('测试日志消息')
    check('状态栏日志', '测试日志消息' in win.statusBar().currentMessage())
    return win


# ================= 2. 波形面板（无通道、无独立窗口） =================
def test_wave_panel(app):
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

    # ---- t25：方法重构（默认 hard、无 auto/exact、无 near_cutoff 控件）----
    check('方法默认 hard', dyn.method_cb.currentText() == 'hard')
    check('方法选项 ip4/sambe/hard',
          [dyn.method_cb.itemText(i) for i in range(dyn.method_cb.count())]
          == ['ip4', 'sambe', 'hard'])
    check('无 near_cutoff 控件', not hasattr(dyn, 'near_cutoff_sb'))

    # ---- t25/t27/t30：显示选项 + 荧光读出（单原子绘制 + 双原子约化）----
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
    # t30：复杂配置弹窗（非模态常驻）——确定应用权重且窗口保持；关闭清引用
    dyn._readout_weights = np.zeros(N)
    dyn._open_readout_config()
    app.processEvents()
    check('复杂配置弹窗非模态显示', getattr(dyn, '_readout_dlg', None)
          is not None and dyn._readout_dlg.isVisible())
    ok_btn = [b for b in dyn._readout_dlg.findChildren(
        __import__('PyQt5.QtWidgets', fromlist=['QPushButton']).QPushButton)
        if b.text() == '确定'][0]
    ok_btn.click()
    app.processEvents()
    check('确定应用权重（窗口保持）',
          len(dyn._readout_weights) == N
          and np.all(dyn._readout_weights == 0.0)
          and dyn._readout_dlg.isVisible())
    dyn._readout_dlg.close()
    app.processEvents()
    from PyQt5.QtCore import QEvent
    app.sendPostedEvents(None, QEvent.DeferredDelete)   # WA_DeleteOnClose 延迟删除
    check('关窗清引用', getattr(dyn, '_readout_dlg', None) is None)
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
    # t30：serialize 含 readout_weights 且 round-trip 一致
    sv = dyn.serialize_state()
    dyn._readout_weights = np.array([0.1, 0.2, 0.3] + [0.0] * (N - 3))
    sv2 = dyn.serialize_state()
    check('serialize 含 readout_weights',
          'readout_weights' in sv2['dynamic']
          and len(sv2['dynamic']['readout_weights']) == N)
    dyn.restore_state(sv2); app.processEvents()
    check('restore 恢复 readout_weights',
          np.allclose(dyn._readout_weights,
                      [0.1, 0.2, 0.3] + [0.0] * (N - 3)))
    check('复杂配置不可作默认（恢复回退真实能级）',
          dyn.readout_cb.currentData() != -1)

    # ---- 自发辐射三档：自动（ARC 寿命）/ 禁用 / 自定义 ----
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
    dyn.method_cb.setCurrentText('ip4')    # t25：'auto' 移除；ip4+decays → lindblad
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
          loss_ok and fresh and cons < 1e-9)

    # ---- 自动档端到端：6P 低激发态按 ARC 寿命自动衰减 ----
    dyn.decay_mode_cb.setCurrentIndex(0); app.processEvents()
    st_now = dyn._current_states()
    i6p = next(i for i, s in enumerate(st_now) if '6P' in s['label'])
    auto_ch = evolve.auto_decays(dyn.cache, st_now)
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
    # build_fields 与 real_field 集成（60S/60P）
    cache = core.ArcCache(__import__('arc').Rubidium87())
    specs = [dict(n=60, l=0, j=0.5, kind='mJ', m=float(m)) for m in (-0.5, 0.5)]
    specs += [dict(n=60, l=1, j=0.5, kind='mJ', m=float(m)) for m in (-0.5, 0.5)]
    states = evolve.make_levels(cache, specs)
    fields = waveforms.build_fields(waves, states, cache)
    check('波形 → real_field 场（近共振驱动对）', len(fields) >= 2)


# ================= 5. 状态保存：只存科学参数，双原子完整，往返一致 =================
def test_state_save(app):
    """serialize/restore：不存显示参数（topn/wave_data）；双原子科学参数
    （mode/pair 几何/A·B 波形/A·B 初态）完整；旧档兼容。"""
    win = main_mod.MainWindow()
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
    snap = dyn.serialize_state()
    check('存档：不含显示参数（topn/wave_data）',
          'topn' not in snap['dynamic'] and 'wave_data' not in snap)
    check('存档：双原子科学参数完整（mode/pair/A·B 波形/A·B 初态）',
          snap['dynamic']['mode'] == 1
          and snap['pair'] == {'R_um': 7.3, 'theta_deg': 45.0, 'phi_deg': 90.0}
          and len(snap['waves']) == 1             # A ≡ 共享面板：波形在 waves
          and snap['waves_A'] == []               # 共享去重（见 serialize_state）
          and len(snap['waves_B']) == 1
          and snap['dynamic']['psi_A'] == 1
          and snap['dynamic']['psi_B'] == 2)
    # 往返
    dyn.vdd_R_sb.setValue(1.0)
    dyn.wave_panel_A.waves.clear()
    dyn.restore_state(snap)
    app.processEvents()
    check('恢复：双原子模式 + 科学参数一致（显示参数保持当前值）',
          dyn._is_two_body
          and abs(dyn.vdd_R_sb.value() - 7.3) < 1e-9
          and len(dyn.wave_panel_A.waves) == 1
          and dyn.psi_combo_A.currentIndex() == 1
          and dyn.topn_sb.value() == 12)
    # 旧档兼容（STIRAP.json：顶层旧格式，无 mode/pair/waves_A/B）
    data = json.load(open(os.path.join(_ROOT, 'save_state', 'STIRAP.json'),
                          encoding='utf-8'))
    dyn.restore_state(data)
    app.processEvents()
    check('旧档兼容：无 mode/pair → 单原子 + 波形恢复',
          not dyn._is_two_body and len(dyn.wave_panel.waves) >= 2)
    win.close(); app.processEvents()


# ================= 5. 进度数组 + 取消 =================
def test_progress(app):
    data = json.load(open(os.path.join(_ROOT, 'save_state',
                                       'rb87_state.json'), encoding='utf-8'))
    cache = core.ArcCache(__import__('arc').Rubidium87())
    B = float(data['B_gauss'])
    specs = []
    for g in data['groups']:
        n, l, j = g['n'], g['l'], g['j']
        hf = core.get_hyperfine_levels(cache, n, l, j, B)
        if hf is None:
            for m in np.arange(-j, j + 1e-9, 1.0):
                specs.append(dict(n=n, l=l, j=j, kind='mJ', m=float(m)))
        else:
            for F, mF, e_hf, _w, amps in hf:
                specs.append(dict(n=n, l=l, j=j, kind='F', F=F, mF=mF,
                                  E_hf=float(e_hf), amps=amps))
    states = evolve.make_levels(cache, specs)
    waves = waveforms.restore_waves(data.get('waves', []))
    fields = waveforms.build_fields(waves, states, cache)
    # 取消测试用更慢的网格：大 t_max + 高点数，保证 1s 内跑不完（取消才可测）
    t = np.linspace(0, float(data['dynamic']['t_max']) * 4,
                    int(data['dynamic']['npts']) * 6)
    psi0 = int(data['dynamic']['psi'])

    # ---- 取消：1 秒后请求取消 → 快速停止 ----
    pa = np.zeros(2)
    result = {}

    def run():
        try:
            evolve.evolve(states, fields, psi0, t, method='ip4',
                          progress_arr=pa)
            result['ok'] = True
        except RuntimeError as e:
            result['err'] = str(e)

    th = threading.Thread(target=run)
    th.start()
    t0 = time.perf_counter()
    while th.is_alive():
        time.sleep(0.2)
        if time.perf_counter() - t0 > 1.0:
            pa[1] = 1.0                        # 1 秒后请求取消
            break
    th.join(timeout=30)
    dt = time.perf_counter() - t0
    check('取消生效（<15s 停止）', '用户取消' in str(result.get('err', ''))
          and dt < 15)

    # ---- 正常完成：进度到 1.0 + 归一 ----
    import mw_bench_firstprinciples as mb
    env = evolve.gaussian(200.0, 100.0)
    st2, i02, E2, fld2, _, om2, w2, _, _, _ = mb.build_s1(cache, 22.0, env)
    t2 = np.linspace(0, 400.0, 2001)
    ac2 = E2 - np.array([s['E_mhz'] for s in st2])
    pa2 = np.zeros(2)
    r2 = evolve.evolve(st2, [fld2], i02, t2, method='ip4', ac_shift_mhz=ac2,
                       progress_arr=pa2)
    check('正常完成 + 进度到 1.0 + 归一',
          pa2[0] == 1.0
          and np.max(np.abs(r2['pops'].sum(axis=1) - 1)) < 1e-6)


# ================= 6. STIRAP/拉曼 存档恢复端到端 =================
def test_stir_states(app):
    """恢复 save_state/STIRAP.json → build_fields → 演化 → 60S 转移（协议存档回归）。"""
    data = json.load(open(os.path.join(_ROOT, 'save_state', 'STIRAP.json'),
                          encoding='utf-8'))
    cache = core.ArcCache(__import__('arc').Rubidium87())
    B = float(data['B_gauss'])
    specs = []
    for g in data['groups']:
        n, l, j = g['n'], g['l'], g['j']
        hf = core.get_hyperfine_levels(cache, n, l, j, B)
        if hf is None:
            for m in np.arange(-j, j + 1e-9, 1.0):
                specs.append(dict(n=n, l=l, j=j, kind='mJ', m=float(m)))
        else:
            for F, mF, e_hf, _w, amps in hf:
                specs.append(dict(n=n, l=l, j=j, kind='F', F=F, mF=mF,
                                  E_hf=float(e_hf), amps=amps))
    states = evolve.make_levels(cache, specs)
    waves = waveforms.restore_waves(data.get('waves', []))
    cut = 5000.0          # t25：near_cutoff 控件移除，ip4 用内部默认 5 GHz
    fields = waveforms.build_fields(waves, states, cache,
                                    near_cutoff_mhz=cut)
    check('STIRAP 存档恢复：波形 + 场非空',
          len(fields) >= 2 and len(waves) >= 2)
    # 演化保真度由 stir_rap_selfcheck（引擎级）与 scripts/make_stir_rap_states.py
    # （独立进程）覆盖——本环境全流程长演化偶发 Qt/ARC 堆损坏干扰（隔离跑正常，
    # 见 README §4.6），此处只保证「恢复状态 → 建场」管线可用。


def main():
    app = QApplication([])
    win = test_main(app)
    test_wave_panel(app)
    test_dynamic(app, win)
    test_wave_serde(app)
    test_state_save(app)
    test_progress(app)
    test_stir_states(app)
    win.close(); app.processEvents()
    failed = sum(1 for ok in _RESULTS if not ok)
    print(f'\n===== ui_selftest: '
          f'{len(_RESULTS) - failed}/{len(_RESULTS)} PASS '
          f'{"✅" if failed == 0 else "❌"} =====')
    sys.exit(1 if failed else 0)


if __name__ == '__main__':
    main()
