# -*- coding: utf-8 -*-
"""hard 的 V_dd 打包自检（t77/t78）：显式零矩阵不再崩 + 判据语义 + 缓存不受污染。

背景①（t77）：`hard._pack_pair_hard` 原先用 `V.nonzero()`（会滤掉显式存储的
零）配 `V.data`（含显式零）取三元组——两者长度失配，掩码即 IndexError。生产
路径 `vdd_sparse` 目前不产显式零，但掩码筛选、`load_npz` 读回、上帝模式手动
keep 都会产出，属必踩。修复 = 一律从同一个 `tocoo()` 取。

背景②（t78）：V_dd 丢弃判据统一为累积相位 Θ = C·T·|V|²/max(|ΔE|,|V|)
≥ PHASE_TOL（与 mixed 共用），取代原幅度阈值 `_VDD_MIN_FRAC`。

覆盖：
  1. 判据语义：共振强对留 / 共振极弱剔 / 34.6 GHz 双翻转留 / 591 THz 远失谐剔；
  2. 正常矩阵：选取集 == 独立复算集（按 Θ 公式，不复用实现）；
  3. 显式零矩阵：不抛异常，且选取集 == 独立复算集，无零值混入；
  4. 全零矩阵：返回空三元组；
  5. 不污染 vdd_sparse 的缓存矩阵（nnz 不变）。

用法：python checks\\hard_vdd_pack_selfcheck.py
"""
import os
import sys

import numpy as np
import scipy.sparse as sp

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

from checks.checklib import Checker                        # noqa: E402
from arc import Rubidium87                                 # noqa: E402
from src.core import static_atom as core                   # noqa: E402
from src.core.atomic_levels import make_levels             # noqa: E402
from src.core.static_pair import vdd_sparse                # noqa: E402
from src.core.waveforms import Wave, GaussianEnv           # noqa: E402
from src.core.hard import build_two_field, _pack_pair_hard  # noqa: E402
from src.core.coupling import PHASE_TOL                  # noqa: E402
from src.core.evolve_util import CONV                     # noqa: E402
from src.core.coupling import static_phase_keep            # noqa: E402

C = Checker('hard_vdd_pack')
Cv = CONV
_T = 200.0                        # 与 packed() 的演化窗口一致
cache = core.ArcCache(Rubidium87())
states = make_levels(cache, [dict(n=60, l=0, j=0.5, kind='mJ', m=+0.5),
                             dict(n=60, l=0, j=0.5, kind='mJ', m=-0.5),
                             dict(n=60, l=1, j=1.5, kind='mJ', m=+1.5),
                             dict(n=60, l=1, j=1.5, kind='mJ', m=+0.5)])
mw = [Wave('微波', 17287.41, GaussianEnv(70, 20), pol='σ-',
           intensity_mwcm2=1.2e-3)]
meta = build_two_field(cache, states, mw)
Elab = np.array([s['E_mhz'] for s in states], float)
_Ep = np.array([Elab[i] + Elab[j] for i in range(len(Elab))
                for j in range(len(Elab))])


def packed(Vin, T=_T):
    """→ 打包出的 V_dd (row, col, val) 三数组。"""
    pk = _pack_pair_hard(meta, meta, T, Vin, 20001, E_lab=Elab)
    return pk[10], pk[11], pk[12]


def ref_keep(Vin, T=_T):
    """独立复算（不复用实现里的函数）：Θ = C·T·|V|²/max(|ΔE|,|V|) ≥ θ_tol。"""
    co = Vin.tocsr().astype(complex).tocoo()
    d = np.asarray(co.data)
    v = np.abs(d)
    dE = np.abs(_Ep[co.row] - _Ep[co.col])
    den = np.maximum(dE, v)
    th = np.zeros(v.shape, float)
    np.divide((Cv * T) * v * v, den, out=th, where=(den > 0.0))
    m = (d != 0) & (th >= PHASE_TOL)
    return {(int(r), int(c)) for r, c in zip(co.row[m], co.col[m])}


V = vdd_sparse(cache, [dict(s) for s in states], R_um=12.0,
               theta_deg=30.0, phi_deg=0.0)

# ---- 1. 判据语义（四条极限：Θ 的两个分支各两例）----
_v = np.array([2.0, 1e-6, 2.13, 2.13])
_dE = np.array([0.0, 0.0, 34570.0, 5.91e8])
_k, _i = static_phase_keep(_v, _dE, _T)
C.check('Θ 判据：共振强留 / 共振极弱剔 / 34.6GHz 双翻转留 / 591THz 远失谐剔',
        bool(_k[0]) and (not _k[1]) and bool(_k[2]) and (not _k[3]),
        f'keep={_k.tolist()}（θ_tol={PHASE_TOL:.0e}）')
C.check('info 字段完整', {'n_keep', 'n_drop', 'max_v_drop',
                          'max_theta_drop', 'max_rate_keep'} <= set(_i))

# ---- 2. 正常矩阵 ----
r, c, v = packed(V)
got = {(int(a), int(b)) for a, b in zip(r, c)}
C.check('正常矩阵：选取集 == 独立复算集',
        got == ref_keep(V), f'{len(got)} vs {len(ref_keep(V))}')

# ---- 3. 显式零矩阵（掩码未 eliminate_zeros）----
co = V.tocoo()
mask = np.abs(co.data) > np.median(np.abs(co.data))
Vz = sp.csr_matrix((co.data * mask, (co.row, co.col)), shape=V.shape)
C.check('构造确实含显式零', int((Vz.data == 0).sum()) > 0,
        f'nnz={Vz.nnz} 真非零={int((Vz.data != 0).sum())}')
exp = ref_keep(Vz)
try:
    rz, cz, vz = packed(Vz)
    gotz = {(int(a), int(b)) for a, b in zip(rz, cz)}
    C.check('显式零矩阵不再抛异常', True)
    C.check('显式零矩阵：选取集 == 独立复算集',
            gotz == exp, f'{len(gotz)} vs {len(exp)}')
    C.check('无零值混入 vdd_vals', bool(np.all(vz != 0)))
except Exception as e:                                     # noqa: BLE001
    C.check('显式零矩阵不再抛异常', False, f'{type(e).__name__}: {e}')

# ---- 4. 全零矩阵 ----
V0 = sp.csr_matrix(V.shape, dtype=complex)
r0, c0, v0 = packed(V0)
C.check('全零矩阵返回空三元组', r0.size == 0 and v0.size == 0)

# ---- 5. 缓存不被污染 ----
before = vdd_sparse(cache, [dict(s) for s in states], R_um=12.0,
                    theta_deg=30.0, phi_deg=0.0).nnz
packed(V)
after = vdd_sparse(cache, [dict(s) for s in states], R_um=12.0,
                   theta_deg=30.0, phi_deg=0.0).nnz
C.check('vdd_sparse 缓存矩阵 nnz 不变', before == after,
        f'{before} → {after}')

sys.exit(0 if C.summary() else 1)
