# -*- coding: utf-8 -*-
"""mixed 引擎契约自检（2026-09-22）：`psi_amp` / `_BW_STATS` / diag 写入端。

覆盖三条**已修但原先无判据**的缺陷（手册 §8.2/§8.3/§8.4/§8.8）——没有判据的修复
下次改动就会悄悄退化，故每条都带**致死性**（把修复改回去必须变红）。

  A. `psi_amp` 契约（P1-2）
     A1 单体 mixed：长度必须是 **N**（物理维），不是 N·P（扩展维）。
        原实现 `Xf = exp(−iC·D·T)·y_out[-1]` 给的是扩展维的 Sambe X
        ⇒ 与 hard/sambe/reference/twofield 的 N 维物理振幅不一致。
        致死：改回 `Xf` ⇒ A1 变红（N=2、P=3 时长度 6≠2）。
     A2 单体 mixed：`|psi_amp|²` 必须等于末态 pops（与引擎自己的重构同源）。
     A3 两体 mixed：`record_amp=True` 必须**不崩**且长度 = N²。
        原实现 `sector_sum_amp(X, …)` 的 `X` 只在 scipy 分支存在
        ⇒ 生产路径（driver='numba'）**必崩** `UnboundLocalError`。
        致死：改回 ⇒ A3 直接异常（红）。
     A4 两体 mixed：scipy 对照路径也必须能 `record_amp`（另一半的同类坑）。
     A5 两体 mixed：`|psi_amp|²` == 末态 pops。

  B. `_BW_STATS` 跨运行独立（P1-3）
     原实现唯一清零口在 `if diag:` 块内 ⇒ `diag=False` 的运行仍然累积，
     随后任一 `diag=True` 的运行报出"本进程至今所有运行之和"（实测 16/32/48）。
     致死：去掉 `_reset_bw_stats()` ⇒ B2 变红（第 4 次报 4×）。

  C. diag 写入端（P1-4）
     `diag.summarize/warnings` 读的 `meta.keep/n_drop_cap/n_drop_phase`、
     `pairs.n/max_abs_v_mhz`、`acize.n_shifted` 原先**全仓无写入端**
     ⇒ 那些字段恒 None、"成本窗剔除=改了模型"的告警**永不触发**（哑告警）。
     致死：删掉写入端 ⇒ C1/C2/C3 变红。

用法：python checks\\mixed_contract_selfcheck.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

from checks.checklib import Checker                            # noqa: E402
from arc import Rubidium87                                     # noqa: E402
from src.core import static_atom as core                       # noqa: E402
from src.core.atomic_levels import make_levels                 # noqa: E402
from src.core import diag as _diag                             # noqa: E402
import src.core.evolve_mixed as EM                             # noqa: E402
from src.core.waveforms import Wave, GaussianEnv, SquareEnv     # noqa: E402

C = Checker('mixed_contract')
cache = core.ArcCache(Rubidium87())


def _levels(gap_mhz):
    return make_levels(cache, [
        dict(n=60, l=0, j=0.5, kind='mJ', m=0.5, E_mhz=0.0, label='g'),
        dict(n=60, l=1, j=0.5, kind='mJ', m=0.5, E_mhz=float(gap_mhz),
             label='e')])


A2 = np.zeros((2, 2)); A2[0, 1] = A2[1, 0] = 1.0


def _opt(gap_mhz, freq_mhz, sig=30.0, T=100.0):
    """光学场（> MW_MAX_MHZ ⇒ 进光子块 ⇒ P>1）。"""
    st = _levels(gap_mhz)
    w = Wave('光', freq_mhz, GaussianEnv(0.5 * T, sig), pol='π',
             intensity_mwcm2=1e3)
    return st, EM_group(st, w)


def EM_group(states, wave):
    from src.core.field_build import build_fields_exact
    return build_fields_exact(cache, states, [wave])


# ======================= A. psi_amp 契约 =======================
# A1/A2：单体 mixed —— 光学场使 P=3（扩展维 = N·P = 6 ≠ N = 2）
stA, fA = _opt(1000.0, 295.7237e6, T=100.0)
tA = np.linspace(0.0, 100.0, 5)
rA = EM.evolve_mixed(stA, fA, 0, tA, record_amp=True)
ampA = np.asarray(rA['psi_amp'])
PA = int(rA['error_report']['P'])
C.group('A 单体 mixed psi_amp：N 维物理振幅、与 pops 同源', [
    (f'P>1（确有扩展维，长度检查才有意义；P={PA}）', PA > 1),
    (f'长度 = N = {len(stA)}（不是 N·P = {len(stA) * PA}）',
     ampA.shape == (len(stA),)),
    ('|psi_amp|² == 末态 pops',
     float(np.max(np.abs(np.abs(ampA) ** 2 - rA['pops'][-1]))) < 1e-12),
])

# A3/A5：两体 mixed —— 生产路径（numba）record_amp 必须不崩
stB, fB = _opt(1000.0, 295.7237e6, T=20.0)
tB = np.linspace(0.0, 20.0, 3)
_pair_err = None
try:
    rB = EM.evolve_mixed_pair(stB, fB, fB, 0, 0, tB, record_amp=True)
    ampB = np.asarray(rB['psi_amp'])
except Exception as e:                                        # noqa: BLE001
    _pair_err = '%s: %s' % (type(e).__name__, e)
    rB, ampB = None, None
C.group('A 两体 mixed psi_amp：numba 生产路径不崩、长度 N²、与 pops 同源', [
    (f'record_amp=True 不抛异常（{_pair_err or "OK"}）', _pair_err is None),
    (f'长度 = N² = {len(stB) ** 2}',
     ampB is not None and ampB.shape == (len(stB) ** 2,)),
    ('|psi_amp|² == 末态 pops',
     ampB is not None
     and float(np.max(np.abs(np.abs(ampB) ** 2 - rB['pops'][-1]))) < 1e-12),
])

# A4：scipy 对照路径也必须能 record_amp（原来只修一半会把这条弄崩）
_sci_err = None
try:
    rS = EM.evolve_mixed_pair(stB, fB, fB, 0, 0, tB, record_amp=True,
                              driver='scipy')
    ampS = np.asarray(rS['psi_amp'])
except Exception as e:                                        # noqa: BLE001
    _sci_err = '%s: %s' % (type(e).__name__, e)
    ampS = None
C.check(f'A scipy 对照路径 record_amp 亦不崩（{_sci_err or "OK"}）',
        _sci_err is None,
        f'两体 scipy 路径 record_amp 失败：{_sci_err}')
C.check('A scipy 路径 psi_amp 亦为 N²',
        ampS is not None and ampS.shape == (len(stB) ** 2,),
        f'shape={None if ampS is None else ampS.shape}')

# ======================= B. _BW_STATS 跨运行独立 =======================
# 场景：光学场 + 40 GHz 能级差 ⇒ 光场对 |ΔD| ≥ BW_OPT(20 GHz) 被丢 ⇒ n_drop_bw>0
stC = _levels(40000.0)
fC = EM_group(stC, Wave('光', 295.7237e6, GaussianEnv(0.25, 30.0), pol='π',
                        intensity_mwcm2=1e3))
tC = np.linspace(0.0, 0.5, 3)

# 前 3 次 **diag=False**（原实现的累积来源），第 4 次 diag=True
for _ in range(3):
    EM.evolve_mixed_pair(stC, fC, fC, 0, 0, tC, diag=False)
rD = EM.evolve_mixed_pair(stC, fC, fC, 0, 0, tC, diag=True)
n_bw = rD['diag']['meta'].get('n_drop_bw')
# 单次运行的真值（干净进程状态下的独立测量）
EM._reset_bw_stats()
rD1 = EM.evolve_mixed_pair(stC, fC, fC, 0, 0, tC, diag=True)
n_bw1 = rD1['diag']['meta'].get('n_drop_bw')
C.group('B _BW_STATS 每次运行独立（不跨运行累积）', [
    (f'确有 BW 丢弃（n_drop_bw={n_bw1} > 0，场景有效）',
     n_bw1 is not None and int(n_bw1) > 0),
    (f'3 次 diag=False 后 diag=True 仍报单次值（{n_bw} == {n_bw1}，'
     f'累积会报 {3 * int(n_bw1 or 0) + int(n_bw1 or 0)}）',
     n_bw == n_bw1),
])

# ======================= C. diag 写入端（哑读反例） =======================
# 场景：微波对**仅因成本窗**被剔除（Θ 判据本会保留）⇒ n_drop_cap>0 且告警可触发
stE = _levels(1000.0)
w_mw = Wave('微波', 900.0, SquareEnv(0.0, 1.0, 1.0, 100.0), pol='π',
            intensity_mwcm2=1.0)
fE = EM_group(stE, w_mw)
tE = np.linspace(0.0, 100.0, 3)
rE = EM.evolve_mixed(stE, fE, 0, tE, diag=True, phase_budget=1.0)
mE = rE['diag']['meta']
pE = rE['diag']['pairs']
aE = rE['diag']['acize']
sE = _diag.summarize(rE['diag'])
warnsE = _diag.warnings(rE['diag'])
_cost_warn = [w for w in warnsE if '成本窗' in w]
C.group('C diag 哑读键现在有写入端（meta/pairs/acize 三处）', [
    (f'meta.keep 是 int（得 {mE.get("keep")!r}）', isinstance(mE.get('keep'), int)),
    (f'meta.n_drop_cap 是 int（得 {mE.get("n_drop_cap")!r}）',
     isinstance(mE.get('n_drop_cap'), int)),
    (f'meta.n_drop_phase 是 int（得 {mE.get("n_drop_phase")!r}）',
     isinstance(mE.get('n_drop_phase'), int)),
    (f'pairs.n == 配对数（得 {pE.get("n")!r}）',
     pE.get('n') == int(np.asarray(pE.get('v')).size)),
    (f'pairs.max_abs_v_mhz 是 float（得 {pE.get("max_abs_v_mhz")!r}）',
     isinstance(pE.get('max_abs_v_mhz'), float)),
    (f'acize.n_shifted 是 int（得 {aE.get("n_shifted")!r}）',
     isinstance(aE.get('n_shifted'), int)),
    ('summarize 的 drop_cap/drop_phase 不再恒 None',
     sE.get('drop_cap') is not None and sE.get('drop_phase') is not None),
])

C.group('C 成本窗剔除的告警**真的能触发**（原为哑告警，永不触发）', [
    (f'本场景确实发生了成本窗剔除（n_drop_cap={mE.get("n_drop_cap")} > 0）',
     int(mE.get('n_drop_cap') or 0) > 0),
    (f'"成本窗"告警出现（得 {len(_cost_warn)} 条）', bool(_cost_warn)),
    (f'告警点名条数（{_cost_warn[0] if _cost_warn else "—"}）',
     bool(_cost_warn) and str(int(mE.get('n_drop_cap'))) in _cost_warn[0]),
])

# C 的对偶：两体 diag 也要有同样的写入端
rF = EM.evolve_mixed_pair(stE, fE, fE, 0, 0, tE, diag=True,
                          phase_budget=1.0)
mF = rF['diag']['meta']
C.group('C 两体 diag 同样补齐（此前 D1 只补了 n_drop_bw）', [
    (f'meta.keep 是 int（得 {mF.get("keep")!r}）', isinstance(mF.get('keep'), int)),
    (f'meta.n_drop_cap 是 int（得 {mF.get("n_drop_cap")!r}）',
     isinstance(mF.get('n_drop_cap'), int)),
    (f'meta.n_drop_phase 是 int（得 {mF.get("n_drop_phase")!r}）',
     isinstance(mF.get('n_drop_phase'), int)),
    (f'pairs.n 存在（得 {rF["diag"]["pairs"].get("n")!r}）',
     rF['diag']['pairs'].get('n') is not None),
    (f'acize.n_shifted 存在（得 {rF["diag"]["acize"].get("n_shifted")!r}）',
     rF['diag']['acize'].get('n_shifted') is not None),
])

sys.exit(0 if C.summary() else 1)
