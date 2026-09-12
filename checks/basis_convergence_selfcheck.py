# -*- coding: utf-8 -*-
"""T2(E1) 验收工具：**两条可达成、且真能抓 E1** 的判据（2026-09-12 使用者改定第 2 版）。

历史（别重犯）：第 1 版把"`C` 跨 7/10/13 基组变化 ≤1%"当通过条件 —— **物理上不可能**，
因为 10 组基**缺 58F_7/2**（Δm 踢出梯子的必需实中间态），缺了它无论配对选得多对，
60P 都到不了 58G。那不是 E1 的锅，是真·基组截断。

## 判据 1（E1 的正题）：**完备基组 + 无关能级**稳定性
以 13 组为参考基组，再往里加 2–3 个**与该脉冲梯子无关**的远失谐能级
（59P_3/2、62S_1/2、60F_5/2），要求：
    `|ΔC_dual| ≤ 1%`、`|Δ(60S 保留)| ≤ 1%`，**且配对数不得爆炸**
    （E1 的病征正是"加无关能级 → 配对数 150 → 1206"）。
这才是"配对选择是否正确"的判据。

## 判据 2（工具自身的正向断言，防"恒真测试"）：**梯子完备性**
声明该脉冲的实中间态集合 INT = {59D_3/2, 59D_5/2, 58F_5/2, 58F_7/2, 58G_7/2, 58G_9/2}，
并断言"**缺任一 → 结果可以大幅变化，这是预期行为而非缺陷**"：
    · 缺 58F_7/2（10 组）→ 60P 存活 **必须 > 0.3**（实测 10 组 0.5267 vs 13 组 0.0012）；
    · 缺 59D_3/2 + 60D_5/2（7 组）→ 60S 保留 **必须 < 0.8**（实测 0.7144 vs 13 组 0.9994）。
两条都成立 ⇒ 工具真的在测物理。

7/10/13 三点仍作**夹具**（它们钉住"哪一步收敛"），但**不是**通过条件。

## 观测量口径（按使用者 2026-09-12 表格反推，若口径不同请纠正）
每个基组跑 **3 次**单原子演化（mixed）：
    (a) 初态 60S_1/2 m=+0.5 → `ret60S` = P_end(60S)
    (b) 初态 60P_3/2 m=−0.5 → `alive_m05` = P_end(60P, m=−0.5)
    (c) 初态 60P_3/2 m=+1.5 → `alive_p15` = P_end(60P, m=+1.5)
    `C(m) = ret60S − alive(m)`；**`C_dual = min(|C(−0.5)|, |C(+1.5)|)`**（保守取差者）。

## 用法（**慢**，且占引擎通道——软件正在跑作业时**先问**）
    set QT_QPA_PLATFORM=offscreen
    python checks/basis_convergence_selfcheck.py            # 判据 1（13 vs 13u，6 次演化）
    python checks/basis_convergence_selfcheck.py --all      # 再加 7/10 组夹具（+6 次演化）
    python checks/basis_convergence_selfcheck.py --bases 13,13u

⚠ 刻意**不注册进 `checks/selftest_all.py`**：判据 1 在 E1 修好前预期 FAIL，
   进了 4 分钟聚合会把回归永久染红。
⚠ 与软件**同源**构基组（offscreen 构窗 → SimState.set_groups → 面板 `_current_states()`
   → `build_fields_exact` → `evolve_engine('mixed')`），不受"离线通道基组不同源"
   的影响（分工文档 §六 #9）。
"""
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

from checks.checklib import Checker                              # noqa: E402

# ---------------- 基组（使用者钉死）----------------
BASE7 = [(60, 0, 0.5), (60, 1, 1.5), (6, 1, 1.5), (59, 2, 2.5), (58, 3, 2.5),
         (58, 4, 3.5), (58, 4, 4.5)]
BASE10 = BASE7 + [(60, 1, 0.5), (59, 2, 1.5), (60, 2, 2.5)]
BASE13 = BASE10 + [(58, 3, 3.5), (61, 1, 1.5), (57, 4, 3.5)]
# 与脉冲梯子**无关**的远失谐能级（判据 1 的扰动）
IRRELEVANT = [(59, 1, 1.5), (62, 0, 0.5), (60, 3, 2.5)]
BASES = {'7': BASE7, '10': BASE10, '13': BASE13, '13u': BASE13 + IRRELEVANT}
# 梯子实中间态（判据 2 声明的集合）
LADDER = [(59, 2, 1.5), (59, 2, 2.5), (58, 3, 2.5), (58, 3, 3.5),
          (58, 4, 3.5), (58, 4, 4.5)]

# ---------------- 脉冲（使用者钉死的夹具配置）----------------
FREQ_MHZ = 7380.0
INTENSITY = 5.0                     # mW/cm²
SIGMA_NS = 16.9866
CENTER_NS = 50.9597
T_MAX_NS = 150.0
NPTS = 1501
B_GAUSS = 21.7

# ---------------- 初态/读出态 ----------------
S0 = (60, 0, 0.5, +0.5)             # 60S_1/2 m=+0.5
P_M05 = (60, 1, 1.5, -0.5)          # 60P_3/2 m=−0.5
P_P15 = (60, 1, 1.5, +1.5)          # 60P_3/2 m=+1.5

# ---------------- 使用者实测夹具 ----------------
# 来源标记：'sw' = 软件实测值；'off' = 仅离线通道值（两通道互验过 4 位小数）
FIX = {
    '7':  {'ret60S': (0.7144, 'sw'), 'alive_m05': (0.7084, 'sw'),
           'alive_p15': (0.8575, 'off')},
    '10': {'ret60S': (0.9973, 'off'), 'alive_m05': (0.5267, 'off'),
           'alive_p15': (0.9415, 'off')},
    '13': {'ret60S': (0.9994, 'sw'), 'alive_m05': (0.0012, 'sw'),
           'alive_p15': (0.0088, 'off')},
}
TOL_FIX = 5e-3                      # 夹具复核容差
TOL_CONV = 0.01                     # 判据 1：变化 ≤1%
PAIR_RATIO_MAX = 1.5                # 判据 1：配对数不得爆炸

_C = Checker('basis_convergence')


def _idx(states, quad):
    for i, s in enumerate(states):
        if (int(s['n']), int(s['l']), round(float(s['j']), 2),
                round(float(s['m']), 2)) == (quad[0], quad[1],
                                             round(float(quad[2]), 2),
                                             round(float(quad[3]), 2)):
            return i
    return None


def _measure(win, st, name):
    """一个基组：3 次演化 → 观测量 dict（见模块 docstring 的口径）。"""
    from src.core.evolve_engine import evolve_engine
    from src.core.field_build import build_fields_exact
    from src.core.waveforms import Wave, GaussianEnv

    t0 = time.perf_counter()
    st.set_groups(BASES[name])
    st.set_param('field.B_gauss', B_GAUSS)
    states = win.dynamic._current_states()
    i_s, i_m05, i_p15 = (_idx(states, S0), _idx(states, P_M05),
                         _idx(states, P_P15))
    for tag, i0 in (('60S', i_s), ('60P(-0.5)', i_m05), ('60P(+1.5)', i_p15)):
        if i0 is None:
            print('  [SKIP] %s：初态 %s 不在基组' % (name, tag))
            return None
    wave = Wave('freeze', FREQ_MHZ, GaussianEnv(CENTER_NS, SIGMA_NS),
                pol='椭圆偏', intensity_mwcm2=INTENSITY,
                elliptical=dict(alpha_deg=0.0, theta_deg=0.0, eps_deg=0.0))
    fields = build_fields_exact(win.cache, states, [wave])
    t = np.linspace(0.0, T_MAX_NS, NPTS)
    n_pairs = None
    out = {}
    for tag, i0, read in (('ret60S', i_s, i_s), ('alive_m05', i_m05, i_m05),
                          ('alive_p15', i_p15, i_p15)):
        r = evolve_engine('mixed', states, fields=fields, psi0=i0, t_grid=t,
                          n_body=1, diag=(n_pairs is None))
        out[tag] = float(r['pops'][-1, read])
        if n_pairs is None:
            try:
                n_pairs = int(r['diag']['meta']['n_pairs'])
            except Exception:                            # noqa: BLE001
                n_pairs = -1
    out['C_m05'] = out['ret60S'] - out['alive_m05']
    out['C_p15'] = out['ret60S'] - out['alive_p15']
    out['C_dual'] = min(abs(out['C_m05']), abs(out['C_p15']))
    out['N'] = len(states)
    out['n_pairs'] = n_pairs
    print('  %-4s N=%-3d n_pairs=%-6s ret60S=%.4f  alive(-0.5)=%.4f  '
          'alive(+1.5)=%.4f  C(-0.5)=%+.4f  C(+1.5)=%+.4f  C_dual=%.4f  (%.0fs)'
          % (name, out['N'], n_pairs, out['ret60S'], out['alive_m05'],
             out['alive_p15'], out['C_m05'], out['C_p15'], out['C_dual'],
             time.perf_counter() - t0))
    return out


def main(names):
    try:
        os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
        from PyQt5.QtWidgets import QApplication                  # noqa: E402
        import app_entry as M                                     # noqa: E402
        _app = QApplication.instance() or QApplication(sys.argv)
        try:
            win = M.MainWindow()
        except MemoryError as e:
            print('  SKIP 内存不足（软件可能正跑着）：%s' % e)
            return 0
    except Exception as e:                                        # noqa: BLE001
        print('  SKIP 构窗不可用：%s' % e)
        return 0

    from src.ai.sim_state import SimState                         # noqa: E402
    st = SimState(win)
    res = {}
    for nm in names:
        if nm not in BASES:
            print('  [SKIP] 未知基组标签 %r（可用 %s）' % (nm, list(BASES)))
            continue
        m = _measure(win, st, nm)
        if m:
            res[nm] = m

    # ---- A. 夹具复核（钉住"哪一步收敛"，不是通过条件）----
    print('\n=== A. 夹具复核（使用者实测；7/13 的 ret60S/alive(-0.5) 是软件实测值）===')
    for nm, fx in FIX.items():
        if nm not in res:
            _C.skip('A 夹具 %s' % nm, '本次未跑')
            continue
        for key in ('ret60S', 'alive_m05', 'alive_p15'):
            ref, src = fx[key]
            got = res[nm][key]
            _C.check('A %s %s=%.4f ≈ %s %.4f（%s）'
                     % (nm, key, got, '实测' if src == 'sw' else '离线',
                        ref, src),
                     abs(got - ref) <= TOL_FIX, 'Δ=%+.4f' % (got - ref))

    # ---- B. 判据 1：完备基组 + 无关能级（E1 的正题）----
    print('\n=== B. 判据 1：13 组 vs 13+无关能级（E1 验收；修好前预期 FAIL）===')
    if '13' in res and '13u' in res:
        a, b = res['13'], res['13u']
        dC = abs(b['C_dual'] - a['C_dual'])
        dR = abs(b['ret60S'] - a['ret60S'])
        _C.check('B1 |ΔC_dual|=%.4f ≤ %.2f（加无关能级不动观测量）'
                 % (dC, TOL_CONV), dC <= TOL_CONV)
        _C.check('B2 |Δ(60S 保留)|=%.4f ≤ %.2f' % (dR, TOL_CONV),
                 dR <= TOL_CONV)
        pa, pb = a['n_pairs'], b['n_pairs']
        _C.check('B3 配对数不爆炸：%s → %s（≤%.1f×，且为正）'
                 % (pa, pb, PAIR_RATIO_MAX),
                 pa > 0 and pb > 0 and pb <= max(PAIR_RATIO_MAX * pa, pa + 50),
                 'E1 病征：加无关能级曾让配对数 150→1206')
    else:
        _C.skip('B 判据 1', '需要同时跑 13 与 13u')

    # ---- C. 判据 2：梯子完备性（正向断言，防恒真）----
    print('\n=== C. 判据 2：梯子完备性（缺中间态 → 结果可大幅变化 = 预期行为）===')
    print('     声明的实中间态 INT = %s' % (LADDER,))
    if '10' in res:
        _C.check('C1 缺 58F_7/2（10 组）→ 60P 存活必须 > 0.3（实测 0.5267）',
                 res['10']['alive_m05'] > 0.3,
                 'got %.4f' % res['10']['alive_m05'])
    else:
        _C.skip('C1 缺 58F_7/2 的正向断言', '本次未跑 10 组')
    if '7' in res:
        _C.check('C2 缺 59D_3/2+60D_5/2（7 组）→ 60S 保留必须 < 0.8（实测 0.7144）',
                 res['7']['ret60S'] < 0.8, 'got %.4f' % res['7']['ret60S'])
    else:
        _C.skip('C2 Δm=+2 假通道未抵消的正向断言', '本次未跑 7 组')

    # ---- D. 收敛位置（报告，不作通过条件）----
    print('\n=== D. 收敛位置（报告，不作通过条件）===')
    for nm in ('7', '10', '13'):
        if nm in res:
            print('     %-3s 组：ret60S=%.4f  alive(-0.5)=%.4f'
                  % (nm, res[nm]['ret60S'], res[nm]['alive_m05']))
    print('     预期：60S 保留在 **10 组** 就收敛（需 59D_3/2+60D_5/2 抵消 Δm=+2 假通道）；')
    print('           60P 踢出要到 **13 组** 才收敛（需 58F_7/2 作 Δm 梯子实中间态）。')

    ok = _C.summary()
    if not ok:
        print('\n提示：B 段失败正是 T2(E1) 的验收目标（工具没坏）；A/C 段是夹具与正向断言。')
    return 0 if ok else 1


if __name__ == '__main__':
    args = sys.argv[1:]
    if '--bases' in args:
        names = args[args.index('--bases') + 1].split(',')
    elif '--all' in args:
        names = ['7', '10', '13', '13u']
    else:
        names = ['13', '13u']            # 默认：判据 1（E1 验收）
    sys.exit(main(names))
