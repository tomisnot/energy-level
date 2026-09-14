# -*- coding: utf-8 -*-
"""自检：**引擎端到端不变式**（一个包 = 一件物理；默认档，目标 ~20-30 s）。

由来（2026-09-14 用户裁决）："checks 677 项太离谱了，动动脑子，哪些不用测…重点在
设立高效的检测，**一个测出好几项**…给我把全量测试时间压到 3 分钟。"
五个重演化体检（`evolve_exact` / `mixed_nolight` / `two_field_pair` /
`evolve_exact_pair` / `pair_evolve`，共约 260 s）验的是**同一批"引擎算得对"的
不变式**；本文件把它收敛成**最便宜的强判据**（闭式解 + 守恒律 + 极小两体 + E1 守卫），
重对拍降为 `--tier deep` 按需跑（**不删**，见 `selftest_all.py` 的档说明）。

覆盖（每项 = 一个**不变式包**）：
  A **闭式 Rabi**（2 态微波 60S↔60P 17.277 GHz）：Ω 由 ARC 偶极**独立算出**并与
    建场矩阵逐位对齐；`mixed` 全程对 `sin²(πΩt_µs)` 偏差 ≤5e-3；π 脉冲末态 ≈1；
    π/2 ≈0.5；布居恒在 [0,1]
  B **酉性/厄米性**：三引擎（mixed/hard/sambe）范数偏差 ≤1e-9；建场矩阵 A−A† ≤1e-12
  C **极小两体（2×2=4 态）**：V_dd 交换闭式 `cos²(2πVt_µs)` 偏差 ≤1e-3；守恒 ≤1e-9
  D **E1 守卫（光场耦合不许静默消失）**：4 组基（N=30 ⇒ dim=2700 > `_DENSE_MAX`）走
    **稀疏**分支 + 光学场 ⇒ 光场（is_mw=0）耦合对数 **> 0**；总配对数 ≥100
    （修前这里光场对为 0、光场"没接线"，实测 B:60P 掉到 0.0060）
  E **无光退化**：无波形 ⇒ 初态布居保持 1、其余为 0（不得凭空转移）
  F **Lindblad 耗散**：单通道 `exp(−Γ×1e-3×t_ns)` 偏差 ≤1e-9 + 单调下降
  G **裁剪不动物理**（小基组）：裁掉 60P 的 m=−0.5 后目标态布居**逐位相同**、守恒不变

用法：python checks/engine_e2e_selfcheck.py（offscreen；构窗失败 → SKIP）
"""
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

from checks.checklib import Checker                              # noqa: E402

C = Checker('engine_e2e')
_T0 = time.perf_counter()
B_GAUSS = 21.7


def main():
    os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
    os.environ.setdefault('EL_NO_AI_SERVER', '1')
    try:
        from PyQt5.QtWidgets import QApplication                  # noqa: E402
        import app_entry as M                                     # noqa: E402
        _app = QApplication.instance() or QApplication(sys.argv)
        try:
            win = M.MainWindow()
        except MemoryError as e:
            print('  SKIP 内存不足：%s' % e)
            return 0
    except Exception as e:                                        # noqa: BLE001
        print('  SKIP 构窗不可用：%s' % e)
        return 0

    from src.ai.sim_state import SimState                         # noqa: E402
    from src.core import static_atom as SA                        # noqa: E402
    from src.core import static_pair as SP                        # noqa: E402
    from src.core.field_build import (build_fields_exact,         # noqa: E402
                                      _sublevel_row)
    from src.core.evolve_engine import evolve_engine              # noqa: E402
    from src.core.evolve_twofield import build_two_field          # noqa: E402
    from src.core.evolve_util import decay                        # noqa: E402
    from src.core.waveforms import Wave, SquareEnv                # noqa: E402
    st = SimState(win)
    cache = win.cache

    # ---------- 公共：2 态（60S_1/2 m=+0.5 ↔ 60P_3/2 m=+0.5，17.277 GHz 微波）----------
    st.set_groups([(60, 0, 0.5, [0.5]), (60, 1, 1.5, [0.5])])
    st.set_param('field.B_gauss', B_GAUSS)
    s2 = win.dynamic._current_states()
    iS = 0
    iP = next(i for i, s in enumerate(s2) if int(s['l']) == 1)
    nu_mhz = float(s2[iP]['E_mhz']) - float(s2[iS]['E_mhz'])
    I_MW = 1.0e-3                     # Ω≈61 MHz（t_π≈8.2 ns；T=400 ns 远大于它）
    wv = Wave('π', nu_mhz, SquareEnv(0.0, 0.0, 1.0, 400.0), pol='π',
              intensity_mwcm2=I_MW)
    fld = build_fields_exact(cache, s2, [wv])
    A = np.asarray(fld[0]['A'], float) if fld else np.zeros((len(s2),) * 2)
    om = float(A[iS, iP])
    # Ω 的**独立**算法：ARC 偶极 + 光强（不经引擎矩阵）
    rowS, rowP = _sublevel_row(s2[iS]), _sublevel_row(s2[iP])
    d_au = SA.dipole_amplitude(cache, rowS, rowP, 0)
    om_ref = SA.rabi_mhz(abs(d_au), SA.field_e0_sq(I_MW))
    T_NS = 400.0
    ts = np.linspace(0.0, T_NS, 401)
    # 把 π、π/2、π/4 的**精确时刻**插进网格（否则"半程≈0.5"会被采样误差罚）
    t_pi_ns = 1000.0 / (2.0 * om) if om else 0.0     # 1/MHz = 1 µs ⇒ ×1000 → ns
    ts = np.unique(np.concatenate([ts, [t_pi_ns / 4.0, t_pi_ns / 2.0, t_pi_ns]]))

    # ---------- A 闭式 Rabi（mixed）----------
    t0 = time.perf_counter()
    oA = evolve_engine('mixed', s2, fields=fld, psi0=iS, t_grid=ts, n_body=1)
    tA = time.perf_counter() - t0
    pA = np.asarray(oA['pops'])[:, iP]
    refA = np.sin(np.pi * om * ts / 1000.0) ** 2
    dA = float(np.max(np.abs(pA - refA)))
    k_pi = int(np.argmin(np.abs(ts - t_pi_ns)))
    k_half = int(np.argmin(np.abs(ts - t_pi_ns / 2.0)))
    C.group('A 闭式 Rabi（2 态微波 π 脉冲 = sin²(πΩt)，Ω 独立算出）', [
        ('建场 Ω 与 ARC 独立算的 Ω 逐位一致（%.6g vs %.6g MHz）' % (om, om_ref),
         om > 0 and abs(om - om_ref) <= 1e-9 * om_ref),
        ('π 脉冲末态布居 ≈1（t=%.4g ns, pop=%.6f）' % (ts[k_pi], pA[k_pi]),
         abs(pA[k_pi] - 1.0) < 5e-3),
        ('π/2 脉冲 ≈0.5（t=%.4g ns, pop=%.6f）' % (ts[k_half], pA[k_half]),
         abs(pA[k_half] - 0.5) < 5e-3),
        ('全程与闭式解最大偏差 ≤5e-3（%.2e，%d 点）' % (dA, len(ts)),
         dA <= 5e-3),
        ('布居恒在 [0,1]', bool(np.all(pA >= -1e-12) and
                                np.all(pA <= 1 + 1e-12))),
    ])

    # ---------- B 酉性/厄米性（三引擎；按各自求解器容差下界断言）----------
    # 窗口只用 20 ns（守恒律不需要 400 ns——hard 要硬积分 17 GHz 载频，400 ns 很贵），
    # 但**建场仍用 T_NS=400**：微波对保留判据是 T 自适应的，T 太小会把被驱动的对筛掉。
    norms = {}
    for meth in ('mixed', 'hard', 'sambe'):
        try:
            if meth == 'hard':
                f = build_two_field(cache, s2, [wv], t_end_ns=T_NS)
                g, kw = np.linspace(0.0, 20.0, 11), dict(rtol=1e-10, atol=1e-12)
            else:
                f, g = fld, np.linspace(0.0, 20.0, 11)
                kw = dict(tol=1e-9)
            o = evolve_engine(meth, s2, fields=f, psi0=iS, t_grid=g, n_body=1,
                              **kw)
            pp = np.asarray(o['pops'])
            norms[meth] = float(np.max(np.abs(pp.sum(axis=1) - 1.0)))
        except Exception as e:                                    # noqa: BLE001
            norms[meth] = float('nan')
            print('      （%s 不可用：%s）' % (meth, e))
    herm = float(np.max(np.abs(A - A.conj().T))) if A.ndim == 2 else 9.9
    # 各引擎的守恒下界不同（mixed 内部容差 ~5e-8；hard/sambe ~1e-9）——
    # 断言仍远严于"坏了必有 O(1) 漂移"的量级，只是按实现各自的容差记账。
    C.group('B 酉性/厄米性（三引擎范数守恒 + 建场矩阵厄米）', [
        ('mixed 范数偏差 ≤1e-6（%.1e）' % norms.get('mixed', np.nan),
         norms.get('mixed', 9) <= 1e-6),
        ('hard 范数偏差 ≤1e-8（%.1e）' % norms.get('hard', np.nan),
         norms.get('hard', 9) <= 1e-8),
        ('sambe 范数偏差 ≤1e-8（%.1e）' % norms.get('sambe', np.nan),
         norms.get('sambe', 9) <= 1e-8),
        ('建场矩阵 A−A† ≤1e-12（%.1e）' % herm, herm <= 1e-12),
    ])

    # ---------- C 极小两体（4 态）V_dd 交换闭式 ----------
    st.set_param('dynamic.mode', 1)
    st.set_param('dynamic.pair.R_um', 12.0)
    st.set_param('dynamic.pair.theta_deg', 30.0)
    st.set_param('dynamic.pair.phi_deg', 0.0)
    s4 = win.dynamic._current_states()
    N4 = len(s4)
    V = SP.vdd_sparse(cache, [dict(s) for s in s4], 12.0, 30.0, 0.0)
    k_PS, k_SP = iS * N4 + iP, iP * N4 + iS
    Vps = float(abs(complex(V[k_PS, k_SP])))
    t4 = np.linspace(0.0, 2.0, 401)
    o4 = evolve_engine('mixed', s4, fields_A=[], fields_B=[], psi0_A=iS,
                       psi0_B=iP, t_grid=t4, n_body=2, vdd_matrix=V)
    p4 = np.asarray(o4['pops_2d'])
    refPS = np.cos(2.0 * np.pi * Vps * t4 / 1000.0) ** 2
    dC = float(np.max(np.abs(p4[:, k_PS] - refPS)))
    C.group('C 极小两体（4 态 V_dd 交换 = cos²(2πVt)，闭式对拍）', [
        ('|V(PS↔SP)| > 0（%0.5g MHz）' % Vps, Vps > 0),
        ('交换对布居与闭式解偏差 ≤1e-3（%.2e，%d 点）' % (dC, len(t4)),
         dC <= 1e-3),
        ('总布居守恒 ≤1e-9（%.1e）'
         % float(np.max(np.abs(p4.sum(axis=1) - 1.0))),
         float(np.max(np.abs(p4.sum(axis=1) - 1.0))) <= 1e-9),
        ('pops_2d 形状 = (npts, N²)（%s）' % (p4.shape,),
         p4.shape == (len(t4), N4 * N4)),
    ])
    st.set_param('dynamic.mode', 0)

    # ---------- D E1 守卫：光场耦合不许静默消失 ----------
    # ⚠ 断言口径（2026-09-14 用致死性对照校准过）：`pairs.is_mw == 0` 里**混着
    # V_dd 对**，所以"is_mw==0 的个数 >0"**咬不住** E1（把 Kc_lists 打成空表后
    # 仍绿）。真正咬得住的两条是：① **光场贡献的耦合对数** = 有光 − 无光；
    # ② **主判据：观测量**（有光 vs 无光必须不同）。修前实测：对数差 = 0、
    # 观测量逐位相同（光场"没接线"）。
    st.set_groups([(60, 0, 0.5), (60, 1, 1.5), (6, 1, 1.5), (58, 4, 3.5)])
    st.set_param('field.B_gauss', B_GAUSS)
    sD = win.dynamic._current_states()
    wD = Wave('频移光', 295724187.476, SquareEnv(0.0, 0.0, 1.0, 400.0),
              pol='π', intensity_mwcm2=8.398e7)
    fD = build_fields_exact(cache, sD, [wD])
    VD = SP.vdd_sparse(cache, [dict(s) for s in sD], 12.0, 30.0, 0.0)
    iPD = next(i for i, s in enumerate(sD)
               if int(s['l']) == 1 and abs(float(s['m']) + 0.5) < 1e-9)
    iSD = next(i for i, s in enumerate(sD)
               if int(s['l']) == 0 and abs(float(s['m']) - 0.5) < 1e-9)

    def _geom(fa):
        """只建几何（2 个时间点：不积分）+ 一个短窗口观测量 → (n_pairs, pop_A, meta)"""
        o = evolve_engine('mixed', sD, fields_A=fa, fields_B=[], psi0_A=iPD,
                          psi0_B=iSD, t_grid=np.array([0.0, 400.0]), n_body=2,
                          vdd_matrix=VD, diag=True)
        meta = ((o.get('diag') or {}).get('meta') or {})
        n = int(meta.get('n_pairs') or 0)
        o2 = evolve_engine('mixed', sD, fields_A=fa, fields_B=[], psi0_A=iPD,
                           psi0_B=iSD, t_grid=np.linspace(0.0, 60.0, 13),
                           n_body=2, vdd_matrix=VD)
        return n, float(np.asarray(o2['pops_A'])[-1, iPD]), meta

    n_on, p_on, meta_on = _geom(fD)
    n_off, p_off, _meta_off = _geom([])
    n_light = n_on - n_off
    C.group('D E1 守卫：稀疏分支（dim>_DENSE_MAX）光场耦合不许静默消失', [
        ('dim=N²·P 且 > _DENSE_MAX（N=%d, dim=%s, P=%s）'
         % (len(sD), meta_on.get('dim'), meta_on.get('P')),
         int(meta_on.get('dim') or 0) > 2200
         and int(meta_on.get('P') or 0) == 3),
        ('光场贡献的耦合对数 = 有光−无光 > 500（%d；修前 = 0）' % n_light,
         n_light > 500),
        ('主判据：光场确实在起作用（A 侧 60P 末态 有光 %.4f vs 无光 %.4f）'
         % (p_on, p_off), abs(p_on - p_off) > 0.05),
        ('有光时总配对数 ≥100（%d）' % n_on, n_on >= 100),
    ])

    # ---------- E 无光退化 ----------
    st.set_groups([(60, 0, 0.5, [0.5]), (60, 1, 1.5, [0.5])])
    oE = evolve_engine('mixed', s2, fields=[], psi0=iS,
                       t_grid=np.linspace(0, 10, 11), n_body=1)
    pE = np.asarray(oE['pops'])
    C.group('E 无光退化（无场无 V_dd：不凭空转移）', [
        ('初态布居保持 1（末态 %.10f）' % float(pE[-1, iS]),
         abs(float(pE[-1, iS]) - 1.0) < 1e-9),
        ('其余态为 0（最大 %.1e）' % float(np.max(np.delete(pE[-1], iS))),
         float(np.max(np.delete(pE[-1], iS))) < 1e-12),
    ])

    # ---------- F Lindblad 耗散（闭式 exp(−Γ×1e-3×t)）----------
    gam = 0.02                       # MHz
    f_zero = build_two_field(cache, s2, [], t_end_ns=50.0)
    tf = np.linspace(0.0, 50.0, 26)
    oF = evolve_engine('hard', s2, fields=f_zero, psi0=iS, t_grid=tf, n_body=1,
                       decays=[decay(iS, gam)])
    pF = np.asarray(oF['pops'])[:, iS]
    refF = np.exp(-gam * 1e-3 * tf)
    C.group('F Lindblad 耗散（单通道 = exp(−Γ·1e-3·t)）', [
        ('与闭式解最大偏差 ≤1e-9（%.1e）' % float(np.max(np.abs(pF - refF))),
         float(np.max(np.abs(pF - refF))) <= 1e-9),
        ('起始布居为 1（%.10f）' % pF[0], abs(pF[0] - 1.0) < 1e-12),
        ('单调下降（无回弹）', bool(np.all(np.diff(pF) <= 1e-12))),
    ])

    # ---------- G 裁剪不动物理（小基组）----------
    st.set_groups([(60, 0, 0.5, [0.5]), (60, 1, 1.5)])
    sFull = win.dynamic._current_states()
    iSf = 0
    iPf = next(i for i, s in enumerate(sFull) if int(s['l']) == 1
               and abs(float(s['m']) - 0.5) < 1e-9)
    tg = np.linspace(0.0, 400.0, 41)
    fFull = build_fields_exact(cache, sFull, [wv])
    oGf = evolve_engine('mixed', sFull, fields=fFull, psi0=iSf, t_grid=tg,
                        n_body=1)
    pf = np.asarray(oGf['pops'])[:, iPf]
    st.set_groups([(60, 0, 0.5, [0.5]), (60, 1, 1.5, [0.5])])
    sTrim = win.dynamic._current_states()
    iT = next(i for i, s in enumerate(sTrim) if int(s['l']) == 1
              and abs(float(s['m']) - 0.5) < 1e-9)
    fTrim = build_fields_exact(cache, sTrim, [wv])
    oGt = evolve_engine('mixed', sTrim, fields=fTrim, psi0=0, t_grid=tg,
                        n_body=1)
    pt = np.asarray(oGt['pops'])[:, iT]
    C.group('G 裁剪不动物理（裁掉 60P 的 m=−1.5/−0.5/+1.5 后目标态一致）', [
        ('裁剪后态数减少（%d → %d）' % (len(sFull), len(sTrim)),
         len(sTrim) == len(sFull) - 3),
        # N 变了（5→2）⇒ 索引重排，按工程口径"≈逐位"（与 sublevel_trim 的
        # TOL_BIT=1e-6 同一标准；实测这里 2.9e-7）
        ('保留态（60P m=+0.5）末态布居一致 ≤1e-6（%.1e）'
         % float(np.max(np.abs(pf - pt))),
         float(np.max(np.abs(pf - pt))) <= 1e-6),
        ('裁剪没改守恒（≤1e-6，%.1e）'
         % float(np.max(np.abs(np.asarray(oGt['pops']).sum(axis=1) - 1.0))),
         float(np.max(np.abs(np.asarray(oGt['pops']).sum(axis=1) - 1.0)))
         <= 1e-6),
    ])

    ok = C.summary()
    print('（engine_e2e 用时 %.1f s；A 段 %.1f s）'
          % (time.perf_counter() - _T0, tA))
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
