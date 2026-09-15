# -*- coding: utf-8 -*-
"""自检：并行扫描**端到端**（真引擎；小基组 + 微波 π 脉冲，mixed 引擎免 JIT）。

这条是坑 19 的落实：**"接口通了"不等于"引擎里通了"**——所有断言都打在
真跑出来的演化结果上，不是打在存储/序列化层。

覆盖（每段 1 个不变式包）：
  A **扫描 ≡ 单点**：同参数下，扫描第 0 点与直接 `build_run+evolve_run`
    的终态布居**逐位一致**（证明网格展开/跨进程传输/落盘都没动物理）
  B **串行 ≡ 并行**：同一批 4 点，1 worker 与 3 workers 的每点终态布居逐位一致
  C **轴真生效**：扫微波频率，远端失谐点的 60P 存活显著高于共振点
  D **取消**：多点半途取消 → 终态 cancelled、已完成点成果保留、
    **runner 及其 worker 子进程全部退出**（无孤儿）
  E **成果落盘**：每点 json + npz 存在、npz 的 final_pop 长度 = 基组态数
  F **记录档位**：record='pops' 有 series_*；record='none' 不产 npz
  G **runspec ⇄ GUI 同源**（需 offscreen 窗 + param_manager；缺则 SKIP）：
    ⚠ P4（2026-09-15）之后 GUI 的 `_current_states`/`_run_*` 已**改调 runspec**，
    所以"基组/场一致"这两条断言变成**接线检查**（证明委托真的接上了、groups/
    B/E 传对了），不再是两套独立实现的对照。真正仍是两条独立实现的只有：
    引擎参数覆写（GUI 的 `_collect_god_kwargs` vs runspec 的
    `god_kwargs_from_params`）、自发辐射通道解析、初态解析——这三条按
    **生效值 / 解析结果**对拍（结构不同但必须等价）。

⚠ 位级一致的前提：两侧都把 BLAS 线程钉成 1（本文件与 runner 都这么做）。
   GUI 交互跑用默认线程数，故"GUI 结果 vs 扫描点"只保证到积分器容差级。

用法：python checks/sweep_e2e_selfcheck.py   （约 1–3 分钟）
"""
import os
import subprocess
import sys
import time

# ⚠ 必须早于 numpy：位级对拍的前提（与 scripts/sweep_runner.py 同一套）
for _k in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
           'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS',
           'NUMBA_NUM_THREADS'):
    os.environ.setdefault(_k, '1')

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import numpy as np                                               # noqa: E402

from checks.checklib import Checker                              # noqa: E402
from src.core import static_atom as core                         # noqa: E402
from src.core import runspec as RS                               # noqa: E402
from src.core.atomic_levels import state_key                     # noqa: E402
from src.sweep import registry as RG                             # noqa: E402

C = Checker('sweep_e2e')
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FREQ0 = 17236.743


def _params(method='mixed', t_max=20.0, npts=101):
    from arc import Rubidium87
    cache = core.ArcCache(Rubidium87())
    groups = [{'n': 60, 'l': 0, 'j': 0.5, 'color': '#88f'},
              {'n': 60, 'l': 1, 'j': 1.5, 'color': '#f88'}]
    states = RS.build_states(cache, groups, 0.0, 0.0)
    want = [s for s in states if s['kind'] == 'mJ' and s['n'] == 60
            and s['l'] == 1 and abs(s['m'] + 0.5) < 1e-9]
    assert want, '基组里没有 60P_{3/2} m=-0.5'
    p = {
        'groups.list': groups,
        'field.B_gauss': 0.0, 'field.E_vpcm': 0.0, 'field.lights': [],
        'dynamic.mode': 0, 'dynamic.method': method,
        'dynamic.t_max': t_max, 'dynamic.npts': npts,
        'dynamic.waves_single': [{
            'name': 'π', 'freq_mhz': FREQ0, 'intensity': 5.5e-5,
            'pol': 'σ-', 'elliptical': None, 'offset_ns': 0.0,
            'env': ['gauss', t_max * 0.5, t_max * 0.2, 1.0], 'drive': None}],
        'dynamic.waves_A': [], 'dynamic.waves_B': [],
        'dynamic.psi_single': list(state_key(want[0])),
        'dynamic.psi_A': None, 'dynamic.psi_B': None, 'dynamic.psi_pair': None,
        'dynamic.decay_index': 1, 'dynamic.decays': [],
        'dynamic.pair.R_um': 12.0, 'dynamic.pair.theta_deg': 30.0,
        'dynamic.pair.phi_deg': 0.0,
    }
    return cache, states, p


def _direct_final(p, cache):
    """直接跑一次（不经 runner）→ 终态布居向量。"""
    run = RS.build_run(cache, p, vdd=False)
    res = RS.evolve_run(run)
    arr = res.get('pops')
    if arr is None:
        arr = res.get('pops_2d')
    return np.asarray(arr)[-1].copy(), run


def _wait(sid, timeout=600, until=None):
    """轮询 status.json 直到终止（或满足 until(status)）。"""
    t0 = time.time()
    last = {}
    while time.time() - t0 < timeout:
        last = RG.status(ROOT, sid)
        if last.get('ok'):
            if until is not None:
                if until(last):
                    return last
            elif not last.get('running'):
                return last
        time.sleep(1.0)
    return last


def _run_sweep(params, **kw):
    r = RG.submit(ROOT, params, **kw)
    assert r.get('ok'), '提交失败：%s' % r.get('msg')
    sid = r['sweep_id']
    st = _wait(sid, timeout=kw.pop('_timeout', 600))
    out = RG.result(ROOT, sid, states=['60P_{3/2} m_J=-0.5'])
    return r, st, out


def _final_of_point(out, k):
    """取某点 60P(-0.5) 的终态布居。

    注意两处都可能带这个键：`registry.result(states=…)` 会把补算结果挂在
    **点顶层** `selected_final_pop`；worker 自己算的那份挂在 `summary` 里。
    """
    for p in (out.get('points') or []):
        if p.get('k') != k or not p.get('ok'):
            continue
        for sel in (p.get('selected_final_pop'),
                    (p.get('summary') or {}).get('selected_final_pop')):
            d = (sel or {}).get('60P_{3/2} m_J=-0.5') or {}
            if d:
                return list(d.values())[0]
    return None


def _npz_final(out, k):
    for p in (out.get('points') or []):
        if p.get('k') == k and p.get('ok') and p.get('npz'):
            d = np.load(p['npz'], allow_pickle=False)
            return d
    return None


print('准备：小基组 + 微波 π 脉冲（mixed，6 态）')
CACHE, STATES, P0 = _params()
print('  基组 %d 态：%s' % (len(STATES), [s['label'] for s in STATES]))

# ---------------------------------------------------------------
print('\nA) 扫描 ≡ 单点（同参数逐位对拍）')
base_final, base_run = _direct_final(P0, CACHE)
print('  直接跑：60P(-0.5)=%.9f' % base_final[3])
_, _, outA = _run_sweep(P0, axes={}, points=[{}], workers=1, tag='e2e-single',
                        basis_n=len(STATES))
vA = _final_of_point(outA, 0)
npzA = _npz_final(outA, 0)
C.group('A 扫描第 0 点 = 直接单点（npz 全精度逐位）', [
    ('扫描跑完 1/1', outA.get('n_done') == 1 and outA.get('n_failed') == 0),
    ('npz 的 final_pop 与直接跑**逐位**一致（全精度锚）',
     npzA is not None and
     float(npzA['final_pop'][3]) == float(base_final[3])),
    ('点摘要 = 该值的 6 位舍入（摘要与全量口径不同是**设计**）',
     vA is not None and vA == round(float(base_final[3]), 6)),
], 'sweep=%s direct=%r npz=%r' % (vA, float(base_final[3]),
                                  None if npzA is None
                                  else float(npzA['final_pop'][3])))

# ---------------------------------------------------------------
print('\nB) 串行 ≡ 并行（4 点 × 1 worker vs 4 点 × 3 workers）')
axis = {'dynamic.waves_single[0].freq_mhz':
        [FREQ0, FREQ0 - 40.0, FREQ0 + 40.0, FREQ0 + 120.0]}
r1, st1, out1 = _run_sweep(P0, axes=axis, workers=1, tag='e2e-serial',
                           basis_n=len(STATES))
r3, st3, out3 = _run_sweep(P0, axes=axis, workers=3, tag='e2e-parallel',
                           basis_n=len(STATES))
v1 = [_final_of_point(out1, k) for k in range(4)]
v3 = [_final_of_point(out3, k) for k in range(4)]
C.group('B 并行不改变物理（每点逐位一致）', [
    ('串行 4/4 完成', out1.get('n_failed') == 0 and out1.get('n_done') == 4),
    ('并行 4/4 完成', out3.get('n_failed') == 0 and out3.get('n_done') == 4),
    ('并行档位确实 > 1', int(st3.get('workers') or 0) == 3),
    ('逐点终态布居逐位一致', v1 == v3 and None not in v1),
], 'serial=%s parallel=%s' % (v1, v3))

# ---------------------------------------------------------------
print('\nC) 轴真生效（频率扫描应改变物理）')
v_axis = [x for x in v3 if x is not None]
C.group('C 频率轴真的在改物理', [
    ('有三个不同的终态值（不是恒定输出）', len(set(v_axis)) >= 3),
    ('大失谐点 60P 存活 > 共振点',
     _final_of_point(out3, 3) > _final_of_point(out3, 0)),
    ('共振点 60P 存活 < 1（确实被驱动）',
     _final_of_point(out3, 0) < 0.9999),
])

# ---------------------------------------------------------------
print('\nD) 取消：半途终止 + 无孤儿进程')
# 故意用**慢**工作点（2000 ns 窗口要解析 17 GHz 载波 ⇒ 每点几十秒），
# 否则点跑得比 1 Hz 轮询还快，"取消时还有活干"这件事根本发生不了
# ——那是测试时序问题，不是产品行为（上一版就因此假红过一次）。
_c_cache, _c_states, PC = _params(t_max=2000.0, npts=201)
many = {'dynamic.waves_single[0].freq_mhz':
        [FREQ0 + 2.0 * i for i in range(8)]}
rD = RG.submit(ROOT, PC, axes=many, workers=2, tag='e2e-cancel',
               basis_n=len(_c_states))
sidD = rD['sweep_id']
_wait(sidD, timeout=420, until=lambda s: (s.get('n_done') or 0) >= 1)
st_mid = RG.status(ROOT, sidD)
cres = RG.cancel(ROOT, sidD, wait_s=90)
stD = _wait(sidD, timeout=180)
pid = rD.get('pid')
time.sleep(1.0)
orphans = None
alive = None
try:
    q = ('Get-CimInstance Win32_Process -Filter "ParentProcessId=%d" | '
         'Measure-Object | Select-Object -ExpandProperty Count' % int(pid))
    o = subprocess.run(['powershell', '-NoProfile', '-Command', q],
                       capture_output=True, text=True, timeout=60)
    orphans = int((o.stdout or '0').strip() or 0)
    t = subprocess.run(['tasklist', '/FI', 'PID eq %d' % int(pid), '/NH'],
                       capture_output=True, text=True, timeout=60)
    alive = str(int(pid)) in (t.stdout or '')
except Exception as e:                             # noqa: BLE001
    print('  （进程检查不可用：%s）' % str(e)[:80])
outD = RG.result(ROOT, sidD)
n_cancelled = sum(1 for p in (outD.get('points') or [])
                  if p.get('cancelled'))
kept = [p for p in (outD.get('points') or []) if p.get('ok')]
C.group('D 取消语义（优雅退出 / 成果保留 / 无孤儿）', [
    ('取消前确实还有点在跑（取消有对象）',
     (st_mid.get('n_done') or 0) < (st_mid.get('n_points') or 0)),
    ('取消调用成功', cres.get('ok') is True),
    ('终态 = cancelled', stD.get('state') == 'cancelled'),
    ('未跑的点被标 cancelled', n_cancelled >= 1),
    ('已完成的点成果保留', len(kept) >= 1),
    ('runner 进程已退出', alive is False),
    ('runner 无残留子进程', orphans == 0),
], 'mid_done=%s cancel=%s state=%s done=%d cancelled=%d orphans=%s alive=%s'
   % (st_mid.get('n_done'), cres.get('state'), stD.get('state'), len(kept),
      n_cancelled, orphans, alive))

# ---------------------------------------------------------------
print('\nE) 成果落盘（json + npz 可读）')
p0 = [p for p in (out1.get('points') or []) if p.get('k') == 0][0]
nz = _npz_final(out1, 0)
C.group('E 每点成果落盘且形状对', [
    ('点摘要 json 有 delta/tag/elapsed',
     bool(p0.get('tag')) and isinstance(p0.get('elapsed_s'), float)),
    ('npz 存在', bool(p0.get('npz')) and os.path.isfile(p0['npz'])),
    ('npz 的 final_pop 长度 = 基组态数',
     nz is not None and len(nz['final_pop']) == len(STATES)),
    ('npz 带标签且与基组一致',
     nz is not None and [str(x) for x in nz['labels']] ==
     [s['label'] for s in STATES]),
])

# ---------------------------------------------------------------
print('\nF) 记录档位（pops / none）')
_, _, outP = _run_sweep(P0, axes={}, points=[{}], workers=1, tag='e2e-pops',
                        basis_n=len(STATES),
                        options={'record': 'pops', 'record_npts': 20})
npzP = _npz_final(outP, 0)
_, _, outN = _run_sweep(P0, axes={}, points=[{}], workers=1, tag='e2e-none',
                        basis_n=len(STATES), options={'record': 'none'})
C.group("F record 档位（pops 给序列 / none 不产 npz）", [
    ("record='pops' 含 series_t/series_pop",
     npzP is not None and 'series_t' in npzP and 'series_pop' in npzP),
    ("record='pops' 序列 ≤ record_npts", npzP is not None and
     len(npzP['series_t']) <= 20),
    ("record='none' 不产 npz",
     not [p for p in (outN.get('points') or []) if p.get('npz')]),
])

# ---------------------------------------------------------------
print('\nG) runspec ⇄ GUI 同源（offscreen；缺 param_manager 则 SKIP）')
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
os.environ.setdefault('EL_NO_AI_SERVER', '1')
try:
    from PyQt5.QtWidgets import QApplication                     # noqa: E402
    import app_entry as M                                        # noqa: E402
    from src.ai.sim_state import SimState                        # noqa: E402
    _app = QApplication.instance() or QApplication(sys.argv)
    win = M.MainWindow()
    st_sim = SimState(win)
    st_sim.set_groups([(60, 0, 0.5), (60, 1, 1.5)])
    st_sim.set_param('dynamic.mode', 0)
    st_sim.set_param('dynamic.method', 'mixed')
    st_sim.set_param('dynamic.t_max', 20.0)
    st_sim.set_param('dynamic.npts', 101)
    st_sim.set_param('dynamic.waves_single', [{
        'name': 'π', 'freq_mhz': FREQ0, 'intensity': 5.5e-5, 'pol': 'σ-',
        'elliptical': None, 'offset_ns': 0.0,
        'env': ['gauss', 10.0, 4.0, 1.0], 'drive': None}])
    st_sim.set_initial_state('single', 60, 1, 1.5, -0.5)
    win.dynamic.refresh_groups()
    gui_states = win.dynamic._current_states()
    params = st_sim.to_dict()['params']
    rs_states = RS.build_states(win.cache, RS.groups_from_params(params),
                                params.get('field.B_gauss'),
                                params.get('field.E_vpcm'))
    run = RS.build_run(win.cache, params, vdd=False)
    same_lab = [a['label'] for a in gui_states] == \
        [b['label'] for b in rs_states]
    same_e = all(abs(a['E_mhz'] - b['E_mhz']) == 0.0
                 for a, b in zip(gui_states, rs_states))
    gui_psi = win.dynamic._psi0_for('single')
    gui_method = win.dynamic.method_cb.currentText()
    # —— 引擎参数覆写：GUI 走 `_collect_god_kwargs`（检视器/会话 store），
    # runner 走 `god_kwargs_from_params`（登记册全量）。两者**结构不同**
    # （前者只带用户改过的，后者带全部生效值），所以不比字典，比**生效值**：
    # 两套覆写喂进 params.override 后，31 个引擎参数必须逐个相等。
    from src.core import params as _P
    _ra = RS.god_kwargs_from_params(params, diag=False)
    _gb = win.dynamic._collect_god_kwargs()
    _ovr_a = _ra.get('param_overrides') or {}
    _ovr_b = _gb.get('param_overrides') or {}
    with _P.override(dict(_ovr_a)):
        _eff_a = {n: _P.get(n) for n in sorted(_P.SPEC_BY_NAME)}
    with _P.override(dict(_ovr_b)):
        _eff_b = {n: _P.get(n) for n in sorted(_P.SPEC_BY_NAME)}
    _solve_a = {k: v for k, v in _ra.items()
                if k != 'param_overrides' and v is not None}
    _solve_b = {k: v for k, v in _gb.items()
                if k != 'param_overrides' and v is not None}

    def _norm(v):
        """list/tuple 视为同一物：登记册快照走 JSON（`json_safe`）会把 tuple
        变成 list —— 实测唯一差异就是 `_NODE_LADDER`（默认 tuple
        `(16,32,64,128)` vs 快照 list `[16,32,64,128]`）。这是**存档格式的
        已知有损点**，不是物理差异，所以归一后再比。"""
        if isinstance(v, (list, tuple)):
            return tuple(_norm(x) for x in v)
        if isinstance(v, dict):
            return tuple(sorted((k, _norm(x)) for k, x in v.items()))
        return v
    _eff_a_n = {n: _norm(v) for n, v in _eff_a.items()}
    _eff_b_n = {n: _norm(v) for n, v in _eff_b.items()}
    _diff_keys = sorted(n for n in _eff_a_n if _eff_a_n[n] != _eff_b_n[n])
    _raw_diff = sorted(n for n in _eff_a if _eff_a[n] != _eff_b[n])
    # —— 自发辐射：GUI 与 runspec 必须解析出同样的通道（自定义档 + 自动档）
    _gui_states = gui_states
    _lab = _gui_states[3]['label']
    win.dynamic.decay_cfgs = [{'from': _lab, 'to': None, 'gamma_mhz': 0.5}]
    _dec_gui = win.dynamic._resolve_decay_cfgs(_gui_states)
    _dec_rs = RS.resolve_decays(win.cache, _gui_states, 2,
                                win.dynamic.decay_cfgs, two_body=False)
    C.group('G 同源：runspec 与 GUI 逐位一致（P4 的验收锚）', [
        ('基组标签序列一致', same_lab),
        ('子能级能量逐位一致（%d 态）' % len(gui_states), same_e),
        ('初态索引一致（GUI=%s / runspec=%s）' % (gui_psi, run['psi0']),
         run['psi0'] == gui_psi),
        ('方法一致（%s）' % gui_method, run['method'] == gui_method),
        ('时间网格端点一致',
         abs(float(run['t'][-1]) - 20.0) == 0.0),
        ('引擎参数覆写：两套写法给出**同一组生效值**（%d 项）'
         % len(_eff_a), not _diff_keys),
        ('未归一时**只允许** list/tuple 归档失型这一个原因（差异键 %s）'
         % (_raw_diff or '无'),
         all(_norm(_eff_a[n]) == _norm(_eff_b[n]) for n in _raw_diff)),
        ('runner 覆盖 ⊇ GUI 显式覆盖（solve 级）%s' % _solve_b,
         all(_solve_a.get(k) == v for k, v in _solve_b.items())),
        ('自定义衰变通道解析一致（%s）' % _dec_rs, _dec_gui == _dec_rs),
    ])
    try:
        win.close()
    except Exception:                              # noqa: BLE001
        pass
except SystemExit:
    raise
except Exception as e:                                         # noqa: BLE001
    C.skip('G runspec ⇄ GUI 同源', '构窗/SimState 不可用：%s' % str(e)[:120])

ok = C.summary()
sys.exit(0 if ok else 1)
