# -*- coding: utf-8 -*-
"""T2(E1) 验收工具：**观测量收敛**（不是配对数收敛）——2026-09-12 使用者改定。

由来（使用者实测，灾难级）：同一个 freeze 脉冲，只换基组，
`C = P_end(60S|初态60S) − P_end(60P|初态60P)` 从 **0.006（7 组）** 变成
**0.9982（13 组）**；`field.n_photon` 2 vs 4 逐位相同 → 差异来自基组本身
（缺 59D_3/2 / 60D_5/2 时那条 Δm=+2 双光子通道不被抵消）。
详见 docs/AI协同-分工与请示.md §六 #8 与 docs/引擎调用与多原子扩展-待办.md E1 段。

本工具做什么
  对每个基组，用**与软件同源**的路径（offscreen 构窗 → SimState.set_groups →
  面板 `_current_states()` → `build_fields_exact` → `evolve_engine('mixed')`）
  跑两次单原子演化（初态 60S / 初态 60P），算 (C, 60S 保留)，然后：
    A. **夹具复核**：7 组 / 13 组的结果要与使用者实测值对上（±5e-3 / ±1e-3）——
       这是"软件确实复现了那条实测"的锚；
    B. **验收判据**：基组间变化 ≤1%（`|ΔC| ≤ 0.01` 且 `|Δ保留| ≤ 0.01`）——
       **T2 未修前预期 FAIL**，这正是它的用途。

用法（**慢**：每基组两次演化，13 组 N=84 单原子已需分钟量级）
    set QT_QPA_PLATFORM=offscreen
    python checks/basis_convergence_selfcheck.py
    python checks/basis_convergence_selfcheck.py --bases 7,13   # 只跑两套（快些）

⚠ 刻意**不注册进 `checks/selftest_all.py`**：它在 T2 修好前是"预期失败"的验收工具，
   进了 4 分钟聚合就会把回归永久染红（工程纪律：聚合必须常绿）。
⚠ 脉冲参数（t_max/npts）与"10 组"的具体组成是我的选择，**待使用者确认**后钉死。
⚠ 需要 ARC 与 offscreen 构窗；软件正在跑实验时**先问**（ARC 争用 + 事故教训 §六 #6）。
"""
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

from checks.checklib import Checker                              # noqa: E402

# ---------------- 基组（使用者给的 7 / 13 组；10 组是我的中间插值，待确认）----------
BASE7 = [(60, 0, 0.5), (60, 1, 1.5), (6, 1, 1.5), (59, 2, 2.5), (58, 3, 2.5),
         (58, 4, 3.5), (58, 4, 4.5)]
EXTRA13 = [(60, 1, 0.5), (58, 3, 3.5), (59, 2, 1.5), (60, 2, 2.5),
           (61, 1, 1.5), (57, 4, 3.5)]
BASE13 = BASE7 + EXTRA13
BASE10 = BASE7 + [(60, 1, 0.5), (59, 2, 1.5), (60, 2, 2.5)]
BASES = {'7组': BASE7, '10组': BASE10, '13组': BASE13}

# ---------------- 脉冲（使用者实测用的那支 freeze）----------------
FREQ_MHZ = 7380.0
INTENSITY = 5.0                     # mW/cm²
FWHM_NS = 40.0
SIGMA_NS = FWHM_NS / (2.0 * np.sqrt(2.0 * np.log(2.0)))    # ≈16.986
CENTER_NS = 50.96
T_MAX_NS = 200.0                    # ⚠ 待确认（使用者原配置未给）
NPTS = 401                          # ⚠ 待确认

# ---------------- 观测量定义（初态与读出态）----------------
LEVEL_S = (60, 0, 0.5, +0.5)        # 60S_1/2 m=+0.5
LEVEL_P = (60, 1, 1.5, -0.5)        # 60P_3/2 m=-0.5

# ---------------- 使用者实测夹具（2026-09-12）----------------
FIXTURE = {'7组': {'C': 0.006, 'ret60S': 0.7144},
           '13组': {'C': 0.9982, 'ret60S': 0.9994}}
TOL_FIX = 5e-3                      # 夹具复核容差（实测只给 4 位小数）
TOL_CONV = 0.01                     # 验收：基组间变化 ≤1%

C = Checker('basis_convergence')


def _idx(states, quad):
    for i, s in enumerate(states):
        if (int(s['n']), int(s['l']), round(float(s['j']), 2),
                round(float(s['m']), 2)) == (quad[0], quad[1],
                                             round(float(quad[2]), 2),
                                             round(float(quad[3]), 2)):
            return i
    return None


def main(which):
    _gui = True
    try:
        os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
        from PyQt5.QtWidgets import QApplication                  # noqa: E402
        import app_entry as M                                     # noqa: E402
        _app = QApplication.instance() or QApplication(sys.argv)
        try:
            win = M.MainWindow()
        except MemoryError as e:
            print('  SKIP 内存不足（软件可能正跑着）：%s' % e)
            _gui = False
    except Exception as e:                                        # noqa: BLE001
        print('  SKIP 构窗不可用：%s' % e)
        _gui = False
    if not _gui:
        return 0

    from src.ai.sim_state import SimState                         # noqa: E402
    from src.core.evolve_engine import evolve_engine              # noqa: E402
    from src.core.field_build import build_fields_exact           # noqa: E402
    from src.core.waveforms import Wave, GaussianEnv              # noqa: E402

    st = SimState(win)
    res = {}
    for name in which:
        groups = BASES[name]
        t0 = time.perf_counter()
        st.set_groups(groups)
        states = win.dynamic._current_states()
        is_, ip = _idx(states, LEVEL_S), _idx(states, LEVEL_P)
        if is_ is None or ip is None:
            print('  [SKIP] %s：目标态不在基组（60S=%s / 60P=%s）'
                  % (name, is_, ip))
            continue
        wave = Wave('freeze', FREQ_MHZ,
                    GaussianEnv(CENTER_NS, SIGMA_NS), pol='椭圆偏',
                    intensity_mwcm2=INTENSITY,
                    elliptical=dict(alpha_deg=0.0, theta_deg=0.0,
                                    eps_deg=0.0))
        fields = build_fields_exact(win.cache, states, [wave])
        t = np.linspace(0.0, T_MAX_NS, NPTS)
        out = {}
        for tag, i0 in (('S', is_), ('P', ip)):
            r = evolve_engine('mixed', states, fields=fields, psi0=i0,
                              t_grid=t, n_body=1)
            out['pS_' + tag] = float(r['pops'][-1, is_])
            out['pP_' + tag] = float(r['pops'][-1, ip])
        out['C'] = out['pS_S'] - out['pP_P']
        out['N'] = len(states)
        res[name] = out
        print('  %s：N=%d  P(60S|S)=%.4f  P(60P|P)=%.4f  C=%.4f  (%.0fs)'
              % (name, out['N'], out['pS_S'], out['pP_P'], out['C'],
                 time.perf_counter() - t0))

    print('\n=== A. 夹具复核（与使用者实测值对拍）===')
    for name, fx in FIXTURE.items():
        if name not in res:
            C.skip('A 夹具 %s' % name, '本次未跑')
            continue
        C.check('A %s：C=%.4f ≈ 实测 %.4f' % (name, res[name]['C'], fx['C']),
                abs(res[name]['C'] - fx['C']) <= TOL_FIX,
                'Δ=%.4f' % (res[name]['C'] - fx['C']))
        C.check('A %s：60S 保留=%.4f ≈ 实测 %.4f'
                % (name, res[name]['pS_S'], fx['ret60S']),
                abs(res[name]['pS_S'] - fx['ret60S']) <= TOL_FIX,
                'Δ=%.4f' % (res[name]['pS_S'] - fx['ret60S']))

    print('\n=== B. 验收判据：基组间观测量变化 ≤1%（T2 未修前预期 FAIL）===')
    if len(res) >= 2:
        names = list(res)
        for a, b in zip(names, names[1:]):
            dC = abs(res[b]['C'] - res[a]['C'])
            dR = abs(res[b]['pS_S'] - res[a]['pS_S'])
            C.check('B %s→%s：|ΔC|=%.4f ≤ %.2f 且 |Δ60S保留|=%.4f ≤ %.2f'
                    % (a, b, dC, TOL_CONV, dR, TOL_CONV),
                    dC <= TOL_CONV and dR <= TOL_CONV,
                    '→ 未收敛：基组裁剪改变了观测量（这就是 T2 要修的）')
    else:
        C.skip('B 收敛判据', '有效基组数 <2')

    ok = C.summary()
    if not ok:
        print('\n提示：B 段失败正是 T2(E1) 的验收目标（不是本工具坏了）。'
              '\n      修好后应全绿；届时可考虑把它并入按需跑的慢自检清单。')
    return 0 if ok else 1


if __name__ == '__main__':
    args = [a for a in sys.argv[1:]]
    which = list(BASES)
    if '--bases' in args:
        which = args[args.index('--bases') + 1].split(',')
        which = [w if w.endswith('组') else w + '组' for w in which]
    sys.exit(main(which))
