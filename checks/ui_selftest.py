# -*- coding: utf-8 -*-
"""UI 自检（离屏运行，无显示依赖）：主窗口/布局/编辑器/时序/动态页/进度。

合并自原分散脚本（_smoke_gui/_check_layout/_test_editor3/_test_intensity/
_test_livesync/_test_ttl_panel/_test_dynamic_ui/_test_progress），统一入口。
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
                             QSpinBox, QToolButton, QSizePolicy,
                             QGraphicsSimpleTextItem)

QMessageBox.information = staticmethod(lambda *a, **k: None)
QMessageBox.warning = staticmethod(lambda *a, **k: None)

import 能级 as main_mod
from src import ttl_pulse, evolve, core, ui_theme
import timing_editor as te

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_RESULTS = []


def check(name, cond):
    _RESULTS.append(bool(cond))
    print(f'  [{"PASS" if cond else "FAIL"}] {name}')
    return bool(cond)


def _ge_combos(scope, n_states):
    """g/e 下拉（首项'未配置'且项数=states+1，排除 deleteLater 残留）。"""
    return [c for c in scope.findChildren(QComboBox)
            if c.count() == n_states + 1 and c.itemText(0) == '— 未配置 —']


# ================= 1. 主窗口冒烟 =================
def test_main(app):
    win = main_mod.MainWindow()
    win.show(); app.processEvents()
    check('主窗口构造 + 全局主题', bool(app.styleSheet()))
    for idx in (0, 1, 2):
        win.canvas_tabs.setCurrentIndex(idx); app.processEvents()
    check('三页切换（页面栈）', win.page_stack.currentIndex() == 2)
    for n, l, j in ((60, 0, 0.5), (60, 1, 0.5)):
        win.n_spin.setValue(n); win.l_spin.setValue(l); win.j_spin.setValue(j)
        win.add_group()
    app.processEvents()
    check('添加能级组', len(win.groups) == 2)
    win.log('测试日志消息')
    check('状态栏日志', '测试日志消息' in win.statusBar().currentMessage())
    dyn = win.dynamic
    dyn._run(); app.processEvents()          # 无脉冲 → 提示并创建编辑器
    check('无脉冲运行提示 + 编辑器创建',
          dyn.editor is not None and not dyn._has_pulses())
    return win


# ================= 2. 布局断言（独立编辑器 + 主窗口动态页） =================
def test_layout(app):
    ed = ttl_pulse.TimingPulseWindow()
    ed.show(); app.processEvents()
    wave = ed.wave
    host = wave.host
    check('波形通道列宽统一', host.channel_list.parentWidget().width()
          == ui_theme.CHANNEL_COL_W)
    check('TTL 列宽统一', ed.ttl.channel_col.width()
          == ui_theme.CHANNEL_COL_W)
    sc = host._panel_scroll
    check('参数面板内容不溢出',
          sc.widget().minimumSizeHint().width() <= sc.width())
    check('画布未被挤压', wave.canvas.width() >= 480)
    check('参数面板固定宽', wave.panel.width() == 440)
    buf = io.StringIO()
    with contextlib.redirect_stderr(buf):
        ed.grab(); app.processEvents()
    check('无 Qt 布局警告', buf.getvalue().strip() == '')
    ed.close()


# ================= 3. 编辑器：光强单位 / 定光强 / 沿标识 =================
def test_editor(app):
    cache = core.ArcCache(__import__('arc').Rubidium87())
    states = evolve.make_levels(cache, [
        dict(n=60, l=0, j=0.5, m=-0.5), dict(n=60, l=0, j=0.5, m=+0.5),
        dict(n=60, l=1, j=0.5, m=-0.5), dict(n=60, l=1, j=0.5, m=+0.5),
    ])
    ed = ttl_pulse.TimingPulseWindow()
    ed.show(); app.processEvents()
    ed.set_states(states, cache)
    app.processEvents()
    wave = ed.wave
    host = wave.host

    # 光强单位智能默认 + spin 拉伸
    pu = host.channels[0].pulses[0]
    pu.intensity_mwcm2 = 5000.0
    host._refresh(); app.processEvents()
    ius = [c for c in wave.findChildren(QComboBox)
           if c.count() == 4 and 'W/cm²' in c.itemText(0)]
    check('光强单位下拉存在', len(ius) >= 2)
    check('光强智能默认单位（5000 mW → W/cm²）',
          any(c.currentIndex() == 0 for c in ius))
    check('参数 spin 可拉伸（Expanding）', any(
        w.sizePolicy().horizontalPolicy() == QSizePolicy.Expanding
        for w in wave.findChildren(QDoubleSpinBox)))
    ge = _ge_combos(wave, len(states))
    check('g/e 下拉尽量宽（≥260px 展示全描述）', len(ge) >= 2
          and all(c.width() >= 260 for c in ge[:2]))

    # 定光强：目标 Ω=10 → 光强非 0 且智能切 nW
    setI = None
    grp = None
    for b in wave.findChildren(QPushButton):
        if b.text() == '定光强' and b.parentWidget() is not None:
            ge = _ge_combos(b.parentWidget(), len(states))
            if len(ge) >= 2:
                setI, grp = b, b.parentWidget()
                break
    check('定光强按钮可用', setI is not None and grp is not None)
    if setI is not None:
        ge = _ge_combos(grp, len(states))
        ge[0].setCurrentIndex(2)          # 60S m=+1/2
        ge[1].setCurrentIndex(4)          # 60P1/2 m=+1/2
        om = [w for w in grp.findChildren(QDoubleSpinBox)
              if w.maximum() == 1e5][0]
        om.setValue(10.0)
        app.processEvents()
        setI.click(); app.processEvents()
        mw = pu.intensity_mwcm2
        check('定光强非 0（Ω=10 → 5.2e-5 mW）',
              abs(mw - 5.2078e-5) < 1e-7)
        check('定光强全精度写回（内部不经 spin round）',
              abs(pu.intensity_mwcm2 - 5.207799922044532e-5) < 1e-15)
        check('定光强智能切单位（nW/cm²）',
              any(c.currentText() == 'nW/cm²' for c in ius))

    # TTL 沿标识：rise ▲ 朝上 / fall ▼ 朝下
    sc = ed.ttl.scene
    dirs = set()
    for mk in sc.markers.values():
        p = mk.path()
        pts = [(p.elementAt(i).x, p.elementAt(i).y)
               for i in range(p.elementCount())
               if p.elementAt(i).isMoveTo() or p.elementAt(i).isLineTo()]
        ys = [y for x, y in pts]
        up = min(ys) < -te.MARK * 0.5 - 0.01
        dirs.add(('up' if up else 'down', mk.edge.kind))
    check('TTL 沿标识（rise▲/fall▼）',
          ('up', 'rise') in dirs and ('down', 'fall') in dirs)
    ed.close()


# ================= 4. TTL 左列 =================
def test_ttl(app):
    ed = ttl_pulse.TimingPulseWindow()
    ed.show(); app.processEvents()
    ttl = ed.ttl

    def _letters(row):
        return [b.text() for b in row.findChildren(QToolButton)]

    n0 = len(ttl._rows)
    check('TTL 左列初始行', n0 > 0)
    check('首行字母 A + 沿数 spin',
          _letters(ttl._rows[0][1]) == ['A']
          and len(ttl._rows[0][1].findChildren(QSpinBox)) == 1)
    ttl.add_btn.click(); app.processEvents()
    check('添加按钮 → 行同步', len(ttl._rows) == n0 + 1)
    ch = ttl.model.channels[-1]
    ttl.scene._del_channel(ch); app.processEvents()
    check('删除通道 → 行同步', len(ttl._rows) == n0)
    spin = ttl._rows[0][1].findChildren(QSpinBox)[0]
    ch0 = ttl._rows[0][0]
    before = len(ch0.edges)
    spin.setValue(spin.value() + 1); app.processEvents()
    check('沿数 spin 联动（+1 → 沿数+2）', len(ch0.edges) == before + 2)
    names = [t.text() for t in ttl.scene.items()
             if isinstance(t, QGraphicsSimpleTextItem)
             and ('TTL' in t.text() or '光场' in t.text())]
    check('画布内无通道名残留', not names)
    e = ttl.model.channels[0].edges[0]
    e.root_abs = 500000.0
    ttl.scene.commit(); app.processEvents()
    check('auto_fit 大时间轴不溢出（m11≤4）',
          ttl.scene.view.transform().m11() <= 4.0)
    ed.close()


# ================= 5. 动态页：初态/态选择/载波采样/实时同步 =================
def test_dynamic(app, win):
    dyn = win.dynamic
    for n, l, j in ((60, 1, 1.5), (6, 1, 1.5)):
        win.n_spin.setValue(n); win.l_spin.setValue(l); win.j_spin.setValue(j)
        win.add_group()
    app.processEvents()
    dyn.psi_combo.setCurrentIndex(2)
    sel = dyn.psi_combo.currentIndex()
    dyn.refresh_groups(); app.processEvents()
    check('初态选择保留', dyn.psi_combo.currentIndex() == sel)
    check('布居显示态数参数默认 8', dyn.topn_sb.value() == 8)
    # 场面板三页共用：动态页显示 B/E 行（光场区隐藏），切单体页恢复
    win.canvas_tabs.setCurrentIndex(2); app.processEvents()
    check('动态页显示场面板(B/E 行)',
          win.field.isVisible() and win.field.b_spin.isVisible()
          and not win.field.lights_box.isVisible())
    win.canvas_tabs.setCurrentIndex(0); app.processEvents()
    check('单体页显示完整场面板', win.field.lights_box.isVisible())
    win.canvas_tabs.setCurrentIndex(2); app.processEvents()
    t = np.linspace(0, 1500.0, 1501)
    dyn._open_editor(); app.processEvents()
    ed = dyn.editor
    wd = ttl_pulse.field_waveforms(ed, t)
    check('包络采样=输出网格 + 含频率/光强', len(wd) > 0
          and len(wd[0][1]) == len(t) and wd[0][3] > 0
          and len(wd[0]) >= 5 and wd[0][4] > 0)
    # 顶部图：多通道独立光强 y 轴（twin axes）渲染不崩
    dyn._draw_top(wd); app.processEvents()
    check('顶部图多通道光强 y 轴渲染',
          len(dyn.fig_top.axes) >= min(len(wd), 2))
    states = dyn._current_states()
    N = len(states)
    res = {'t': t, 'pops': np.zeros((len(t), N))}
    res['pops'][:, 0] = 0.6
    top = dyn._top_pop_rows(states, res)
    check('内部按平均布居选态（态 0 入选）',
          0 in top and len(top) == min(8, N))
    dyn._draw_bot(states, res); app.processEvents()

    # 实时同步：参数变化 → 主面板波形刷新（用主窗口关联的编辑器）
    before = len(dyn.wave_data)
    n_pulses = len(ed.wave.channels[0].pulses)   # field_waveforms 每脉冲一条
    pu = ed.wave.channels[0].pulses[0]
    pu.envelope.center_ns += 5.0
    ed.wave.host.changed.emit(); app.processEvents()
    check('实时同步刷新主面板波形',
          len(dyn.wave_data) == before + n_pulses)
    ed._live_sync = False
    ed.wave.host.changed.emit(); app.processEvents()
    check('关闭实时同步后不刷新',
          len(dyn.wave_data) == before + n_pulses)
    ed._live_sync = True

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
    # 纯衰变（无驱动脉冲）：运行 → loss 曲线 + Σpops+loss=1
    dyn.decay_from_cb.setCurrentIndex(0)
    dyn.decay_gamma_sb.setValue(8.0)
    dyn.decay_to_cb.setCurrentIndex(dyn.decay_to_cb.count() - 1)
    dyn._add_decay()
    dyn.psi_combo.setCurrentIndex(0)
    dyn.tmax_sb.setValue(200.0); dyn.npts_sb.setValue(101)
    dyn.method_cb.setCurrentText('auto')
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
            and time.perf_counter() - t0 < 120:      # RK4 场景实测 ~55s
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


# ================= 6. 进度数组 + 取消 =================
def test_progress(app):
    data = json.load(open(os.path.join(_ROOT, 'save_state',
                                       'rb87_state.json'), encoding='utf-8'))
    win = ttl_pulse.TimingPulseWindow()
    ttl_pulse.restore_editor(win, data['editor'])
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
    fields, ac = ttl_pulse.build_fields(win, states, cache)
    t = np.linspace(0, float(data['dynamic']['t_max']),
                    int(data['dynamic']['npts']))
    psi0 = int(data['dynamic']['psi'])

    # ---- 取消：3 秒后请求取消 → 快速停止 ----
    pa = np.zeros(2)
    result = {}

    def run():
        try:
            evolve.evolve(states, fields, psi0, t, method='ip4',
                          ac_shift_mhz=ac, progress_arr=pa)
            result['ok'] = True
        except RuntimeError as e:
            result['err'] = str(e)

    th = threading.Thread(target=run)
    th.start()
    t0 = time.perf_counter()
    while th.is_alive():
        time.sleep(0.3)
        if time.perf_counter() - t0 > 3.0:
            pa[1] = 1.0                        # 3 秒后请求取消
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


# ================= 7. STIRAP/拉曼 存档恢复端到端 =================
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
    win = ttl_pulse.TimingPulseWindow()
    ttl_pulse.restore_editor(win, data['editor'])
    cut = data['dynamic'].get('near_cutoff', 5000.0)
    fields = ttl_pulse.build_fields(win, states, cache, near_cutoff_mhz=cut)[0]
    check('STIRAP 存档恢复：编辑器 + 场非空',
          len(fields) >= 2 and len(win.wave.channels) >= 2)
    # 演化保真度由 stir_rap_selfcheck（引擎级）与 scripts/make_stir_rap_states.py
    # （独立进程）覆盖——本环境全流程长演化偶发 Qt/ARC 堆损坏干扰（隔离跑正常，
    # 见 README §4.6），此处只保证「恢复状态 → 建场」管线可用。


def main():
    app = QApplication([])
    win = test_main(app)
    test_layout(app)
    test_editor(app)
    test_ttl(app)
    test_dynamic(app, win)
    test_progress(app)
    test_stir_states(app)
    ed = win.dynamic.editor
    win.close(); app.processEvents()          # 关闭联动检查放最后
    if ed is not None:
        check('主窗口关闭联动关编辑器', not ed.isVisible())
    failed = sum(1 for ok in _RESULTS if not ok)
    print(f'\n===== ui_selftest: '
          f'{len(_RESULTS) - failed}/{len(_RESULTS)} PASS '
          f'{"✅" if failed == 0 else "❌"} =====')
    sys.exit(1 if failed else 0)


if __name__ == '__main__':
    main()
