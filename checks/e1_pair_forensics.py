# -*- coding: utf-8 -*-
"""T2(E1) 取证工具：**固定波形、只改基组**，把"配对数"与"观测量"分开报告。

## 使用者给的原始证据（2026-09-12，本工具的靶子）
| 配置 | 观测 |
|---|---|
| 3 组（60S+60P_3/2+6P_3/2）**+ 波形** | dim=**1452 = 484×3** ⇒ **P=3（有光子块）**、`n_pairs=**1206**` |
| 同上 **不加任何波形** | dim=**484**、`n_pairs=**50**` |
| 3 组 **+ 58G_7/2** | `n_pairs=**150**` |
⇒ 可疑的不是"1206 多"，而是"**加一个组就从 1206 掉到 150**"。

## 本工具做什么（与软件**同源**构基组：offscreen 构窗 → SimState.set_groups →
面板 `_current_states()` → `pair_core.vdd_sparse` → `evolve_engine(n_body=2, diag=True)`）
对每个配置跑一次两体演化，报告：
  · `dim / P / n_pairs`；
  · **配对分类**：`is_mw` 计数、光子扇区分布、两端是否含 6P（n=6）态、
    `|V|` 的唯一值个数与取值、`dD` 的唯一值个数与取值、索引差 `b-a` 的直方图
    （E1 指纹是"|V| 逐位相同 40.430693 MHz、失谐双峰、索引 (i,i+88) 等差"）；
  · **观测量**：交换对 `|P_A(−0.5)S_B(+0.5)⟩ ↔ |S_A(−0.5)P_B(+1.5)⟩` 的末态布居、
    58G 族末态合计、逐原子 60P 约化；
  · 配置间 `max|Δpops|`（"只加一个组，观测量变了多少"）。

## 判据（使用者写死）
**主判据永远是观测量**；配对数只是诊断线索。→ 若加 58G_7/2 后观测量几乎不变，
则"1206→150"属**配对计数不具物理意义**；若观测量大变，才是真配对 bug。

用法（**慢**：每个配置一次两体演化；软件在跑作业时先问）
    set QT_QPA_PLATFORM=offscreen
    python checks/e1_pair_forensics.py                # 3 组 + 波形（复现 1206）
    python checks/e1_pair_forensics.py --all          # 上面 + 无波形 + 加 58G_7/2
"""
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

from checks.checklib import Checker                              # noqa: E402

G3 = [(60, 0, 0.5), (60, 1, 1.5), (6, 1, 1.5)]
G3_58G = G3 + [(58, 4, 3.5)]
R_UM, THETA, PHI = 12.0, 30.0, 0.0
B_GAUSS = 21.7
T_NS, NPTS = 400.0, 101                 # 配对数只由 T 决定（npts 只影响观测量分辨率）
OPT_MHZ, OPT_I = 295724187.476, 8.398e7   # 使用者的"频移光"（3 组下 P=3）
A_P = (60, 1, 1.5, -0.5)                # 原子 A：60P_3/2 m=−0.5
B_S = (60, 0, 0.5, +0.5)                # 原子 B：60S_1/2 m=+0.5
B_P = (60, 1, 1.5, +1.5)                # ΔM=+1 通道的另一端

CASES = {'3+opt': (G3, 'opt'), '3+none': (G3, 'none'), '3+58G+opt': (G3_58G, 'opt')}

_C = Checker('e1_pair_forensics')


def _idx(states, quad):
    for i, s in enumerate(states):
        if (int(s['n']), int(s['l']), round(float(s['j']), 2),
                round(float(s['m']), 2)) == (quad[0], quad[1],
                                             round(float(quad[2]), 2),
                                             round(float(quad[3]), 2)):
            return i
    return None


def run_case(win, st, groups, wave_kind):
    from src.core import pair_core as pc
    from src.core.evolve_engine import evolve_engine
    from src.core.field_build import build_fields_exact
    from src.core.waveforms import Wave, SquareEnv

    st.set_groups(groups)
    st.set_param('field.B_gauss', B_GAUSS)
    states = win.dynamic._current_states()
    N = len(states)
    fields = []
    if wave_kind == 'opt':
        wv = Wave('频移光', OPT_MHZ, SquareEnv(0.0, 0.0, 1.0, T_NS),
                  pol='π', intensity_mwcm2=OPT_I)
        fields = build_fields_exact(win.cache, states, [wv])
    iP, iS, iBp = _idx(states, A_P), _idx(states, B_S), _idx(states, B_P)
    if min(iP, iS, iBp) < 0:
        print('  [SKIP] 目标态不在基组'); return None
    V = pc.vdd_sparse(win.cache, [dict(s) for s in states], R_UM, THETA, PHI)
    t = np.linspace(0.0, T_NS, NPTS)
    t0 = time.perf_counter()
    out = evolve_engine('mixed', states, fields_A=fields, fields_B=[],
                        psi0_A=iP, psi0_B=iS, t_grid=t, n_body=2,
                        vdd_matrix=V, diag=True)
    dt = time.perf_counter() - t0
    d = out.get('diag') or {}
    meta, pr = d.get('meta') or {}, d.get('pairs') or {}
    a = np.asarray(pr.get('a', []), int)
    b = np.asarray(pr.get('b', []), int)
    v = np.asarray(pr.get('v', []), complex)
    dD = np.asarray(pr.get('dD', []), float)
    is_mw = np.asarray(pr.get('is_mw', []), int)
    N2 = N * N
    sec_a = (a // N2) if len(a) else a
    ka, kb = (a % N2), (b % N2)
    ia, ja = (ka // N), (ka % N)
    ib, jb = (kb // N), (kb % N)
    n6 = np.array([(int(states[i]['n']) == 6) or (int(states[j]['n']) == 6)
                   for i, j in zip(ia, ib)]) if len(a) else np.array([], bool)
    pops = np.asarray(out['pops_2d'])
    k_PS, k_SP = iP * N + iS, iS * N + iBp
    g58 = [k for k in range(N2)
           if int(states[k // N]['n']) == 58 or int(states[k % N]['n']) == 58]
    info = dict(N=N, dim=meta.get('dim'), P=meta.get('P'), M=meta.get('M'),
                bw=meta.get('bw'), mw_cap=meta.get('mw_cap'),
                n_acized=sum(int(f.get('n_acized') or 0)
                             for f in (d.get('acize') or {}).get('fields', [])),
                acize=(d.get('acize') or {}).get('fields'),
                n_pairs=int(meta.get('n_pairs', len(a))),
                n_mw=int((is_mw == 1).sum()),
                n_vdd_light=int((is_mw == 0).sum()),
                n_with6P=int(n6.sum()), n_sectors=int(len(set(sec_a.tolist()))),
                n_absV_unique=int(len(set(np.round(np.abs(v), 6).tolist()))),
                absV_top=sorted(set(np.round(np.abs(v), 6).tolist()),
                                reverse=True)[:3],
                n_dD_unique=int(len(set(np.round(dD, 3).tolist()))),
                dD_lo=float(np.min(dD)) if len(dD) else None,
                dD_hi=float(np.max(dD)) if len(dD) else None,
                n_gap_unique=int(len(set((b - a).tolist()))),
                gap_top=sorted(set((b - a).tolist()))[:5],
                end_PS=float(pops[-1, k_PS]), end_SP=float(pops[-1, k_SP]),
                end_58G=float(pops[-1, g58].sum()),
                pA_60P=float(np.asarray(out['pops_A'])[-1, iP]),
                pB_60P=float(np.asarray(out['pops_B'])[-1, iBp]),
                secs=time.perf_counter() - t0, pops=pops)
    del dt
    return info


def main(names):
    os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
    try:
        from PyQt5.QtWidgets import QApplication                  # noqa: E402
        import app_entry as M                                     # noqa: E402
        _app = QApplication.instance() or QApplication(sys.argv)
        try:
            win = M.MainWindow()
        except MemoryError as e:
            print('  SKIP 内存不足：%s' % e); return 0
    except Exception as e:                                        # noqa: BLE001
        print('  SKIP 构窗不可用：%s' % e); return 0
    from src.ai.sim_state import SimState                         # noqa: E402
    st = SimState(win)
    res = {}
    for nm in names:
        groups, wk = CASES[nm]
        print('\n--- %s（%d 组，波形=%s）---' % (nm, len(groups), wk))
        info = run_case(win, st, groups, wk)
        if info:
            res[nm] = info
            print('  N=%(N)d dim=%(dim)s P=%(P)s M=%(M)s **n_pairs=%(n_pairs)d**'
                  '  bw=%(bw)s  AC 化剔除=%(n_acized)d' % info)
            print('  AC 化逐波：%s' % (info['acize'],))
            print('  分类：is_mw=1 → %(n_mw)d；is_mw=0 → %(n_vdd_light)d；'
                  '含 6P 的对 → %(n_with6P)d；扇区数 → %(n_sectors)d'
                  % info)
            print('  |V|：唯一值 %(n_absV_unique)d 个，最大 %(absV_top)s MHz'
                  % info)
            print('  dD ：唯一值 %(n_dD_unique)d 个，范围 [%(dD_lo).3f, '
                  '%(dD_hi).3f] MHz' % info)
            print('  索引差 (b−a)：唯一 %(n_gap_unique)d 个，样例 %(gap_top)s'
                  % info)
            print('  观测量：P_PS=%(end_PS).4f  P_SP=%(end_SP).4f  '
                  '58G 族=%(end_58G).4f  A:60P=%(pA_60P).4f  B:60P=%(pB_60P).4f'
                  % info)
            print('  （%.0f s）' % info['secs'])

    # ---- 只加一个组：配对数 vs 观测量（分开报告）----
    if '3+opt' in res and '3+58G+opt' in res:
        A_, B_ = res['3+opt'], res['3+58G+opt']
        print('\n=== 只加 58G_7/2（波形固定=opt on A）===')
        print('  配对数：%d → %d（×%.2f）'
              % (A_['n_pairs'], B_['n_pairs'],
                 B_['n_pairs'] / max(A_['n_pairs'], 1)))
        for k, lab in (('end_PS', 'P_PS'), ('end_SP', 'P_SP'),
                       ('end_58G', '58G 族'), ('pA_60P', 'A:60P'),
                       ('pB_60P', 'B:60P')):
            print('  观测量 %-7s %.4f → %.4f（Δ%+.4f）'
                  % (lab, A_[k], B_[k], B_[k] - A_[k]))
        # 跨基组维度不同 → **只比观测量**（这就是"主判据"）
        dP = max(abs(A_[k] - B_[k]) for k in
                 ('end_PS', 'end_SP', 'end_58G', 'pA_60P', 'pB_60P'))
        print('  **观测量最大变化 = %.4f**（主判据）' % dP)
        _C.check('配对数取样齐全（两配置都 >0）',
                 A_['n_pairs'] > 0 and B_['n_pairs'] > 0)
        # ⭐ 本工具的**回归锚**（2026-09-12 修 `_DENSE_MAX` 静默丢光场耦合之前，
        #    4 组（dim=2700 > 2200）的 n_pairs 掉到 150 且频移光**完全失效**）：
        _C.check('E1 回归锚：3 组下频移光**有效**（B:60P > 0.5）',
                 A_['pB_60P'] > 0.5, 'B:60P=%.4f' % A_['pB_60P'])
        _C.check('E1 回归锚：＋58G_7/2（dim 超 _DENSE_MAX）后频移光**仍然有效**'
                 '（B:60P > 0.5）——修前这里是 0.0060（光场耦合被静默丢弃）',
                 B_['pB_60P'] > 0.5, 'B:60P=%.4f' % B_['pB_60P'])
        _C.check('E1 回归锚：加一个组后配对数**不塌缩**'
                 '（≥0.5× 3 组；修前 1206→150 = ×0.12）',
                 B_['n_pairs'] >= 0.5 * A_['n_pairs'],
                 '%d → %d' % (A_['n_pairs'], B_['n_pairs']))
    if '3+none' in res and '3+opt' in res:
        A_, C_ = res['3+opt'], res['3+none']
        dO = max(abs(A_[k] - C_[k]) for k in
                 ('end_PS', 'end_SP', 'end_58G', 'pA_60P', 'pB_60P'))
        print('\n=== 波形有没有用（同基组 3 组）===')
        print('  B:60P  opt=%.4f  none=%.4f；观测量最大差 = %.4f'
              % (A_['pB_60P'], C_['pB_60P'], dO))
        _C.check('3 组下频移光**确实在起作用**（观测量最大差 >0.1）',
                 dO > 0.1, 'max|Δ|=%.4f' % dO)
    if '3+none' in res:
        print('\n=== 无波形对照（纯 V_dd）===')
        print('  dim=%s n_pairs=%d（使用者实测 dim=484、n_pairs=50）'
              % (res['3+none']['dim'], res['3+none']['n_pairs']))
        _C.check('无波形：dim 与 n_pairs 复现使用者实测（484 / 50）',
                 res['3+none']['dim'] == 484 and res['3+none']['n_pairs'] == 50,
                 'dim=%s n_pairs=%d' % (res['3+none']['dim'],
                                        res['3+none']['n_pairs']))
    if '3+opt' in res:
        _C.check('带波形：dim=1452（P=3）复现',
                 res['3+opt']['dim'] == 1452,
                 'dim=%s n_pairs=%d' % (res['3+opt']['dim'],
                                        res['3+opt']['n_pairs']))
    return 0 if _C.summary() else 1


if __name__ == '__main__':
    args = sys.argv[1:]
    if '--all' in args:
        names = ['3+opt', '3+none', '3+58G+opt']
    elif '--bases' in args:
        names = args[args.index('--bases') + 1].split(',')
    else:
        names = ['3+opt']
    sys.exit(main(names))
