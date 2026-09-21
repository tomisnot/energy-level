# -*- coding: utf-8 -*-
"""M1 ED 引擎自检（锚 A1/A2/A4 + 装配/衰减约定）。**不碰 ARC**（假 calib，快且确定）。

A1a N=1 拉比：⟨n(t)⟩=sin²(CONV·Ω·t/2) vs 解析 <1e-10
A1b N=2 blockade：对称 3 能级块 {|gg⟩,|S⟩,|rr⟩} **独立构造** vs 引擎投影 <1e-10
A2  对称扇区：守恒时 sector 分解 ≡ 全空间 <1e-12
A4  张量积回归：关相互作用关耗散 ⇒ 双原子 = 两个独立单原子拉比之积 <1e-10
H   稀疏装配厄米（gamma=0）；build_H_ed ≡ build_H_sparse.toarray()
D   非厄米衰减：P_r(t)=exp(-Γ×1e-3·t)（约定同 hard.py 的 Gamma_ns/CONV）<1e-9
R   Result 契约键齐全 + 布居归一 + 附加观测量
（A3 耗散 vs Lindblad 交叉验证按裁决③留 M4；此处只验非厄米衰减数学约定。）
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import numpy as np

from checks.checklib import Checker                            # noqa: E402
from src.core.manybody import assemble as asm                  # noqa: E402
from src.core.manybody import engine_ed as ed                  # noqa: E402
from src.core.evolve_util import CONV                          # noqa: E402

C = Checker('manybody_m1_ed')


def _calib(n_atoms, pairs, terms=('I1', 'I2', 'I3')):
    """假 calib（不碰 ARC）：atoms 只有 id/pos，pairs 给 V/J/A。"""
    return dict(terms=list(terms),
                atoms=[dict(id=i, pos=(float(i), 0.0, 0.0))
                       for i in range(n_atoms)],
                pairs=pairs, single_basis=[])


def _pair(i, j, V=0.0, J=0.0j, A=0.0j):
    return dict(i=i, j=j, V=V, J=J, A=A, raw_V=V, raw_J=J, raw_A=A,
                delta={}, info={})


t = np.linspace(0.0, 200.0, 201)
Om = 2 * np.pi * 5.0                       # 5 MHz 拉比

# ---------------- A1a N=1 拉比 ----------------
c1 = _calib(1, [], terms=('I1',))
r1 = ed.evolve_manybody(c1, None, t, Om=[Om])
pred = np.sin(CONV * Om * t / 2.0) ** 2
C.group('A1a N=1 拉比 ⟨n(t)⟩=sin²(CONV·Ω·t/2)', [
    ('逐时刻 vs 解析 max|Δ| < 1e-10',
     float(np.abs(r1['pops_atom'][:, 0] - pred).max()) < 1e-10),
    ('布居归一（|Σpops−1| < 1e-12）',
     float(np.abs(r1['pops'].sum(1) - 1).max()) < 1e-12),
])

# ---------------- A1b N=2 blockade（对称 3 能级块独立构造） ----------------
V_blk = 500.0                              # ≫ Ω ⇒ blockade
c2 = _calib(2, [_pair(0, 1, V=V_blk)], terms=('I1',))
r2 = ed.evolve_manybody(c2, 'gg', t, Om=[Om, Om])
H3 = np.array([[0.0, Om / np.sqrt(2), 0.0],
               [Om / np.sqrt(2), 0.0, Om / np.sqrt(2)],
               [0.0, Om / np.sqrt(2), V_blk]], complex)
w3, U3 = np.linalg.eigh(H3)
c0 = U3.conj().T @ np.array([1.0, 0.0, 0.0], complex)
psi3 = U3 @ (np.exp(-1j * CONV * np.outer(w3, t)) * c0[:, None])
P_gg, P_S, P_rr = (np.abs(psi3[i]) ** 2 for i in range(3))
eng = r2['pops']                           # |gg⟩=0,|gr⟩=1,|rg⟩=2,|rr⟩=3
C.group('A1b N=2 blockade ≡ 对称 3 能级块独立解', [
    ('P_gg vs 解析 < 1e-10', float(np.abs(eng[:, 0] - P_gg).max()) < 1e-10),
    ('P_rr vs 解析 < 1e-10', float(np.abs(eng[:, 3] - P_rr).max()) < 1e-10),
    ('P_gr+P_rg ≡ P_S（对称驱动）< 1e-10',
     float(np.abs(eng[:, 1] + eng[:, 2] - P_S).max()) < 1e-10),
    ('blockade：max P_rr < 0.05（V≫Ω 抑制双激发）',
     float(P_rr.max()) < 0.05),
])

# ---------------- A2 对称扇区 ≡ 全空间 ----------------
c2x = _calib(2, [_pair(0, 1, V=3.0, J=-0.9 + 0.4j)], terms=('I1', 'I2'))
rsec = ed.evolve_manybody(c2x, 'gr', t, use_sectors=True)
rfull = ed.evolve_manybody(c2x, 'gr', t, use_sectors=False)
C.group('A2 激发数守恒扇区 ≡ 全空间（无驱动 + 无 A）', [
    ('sector 后端被启用（backend 含 -sectors）', 'sectors' in rsec['backend']),
    ('逐时刻布居 sector vs full < 1e-12',
     float(np.abs(rsec['pops'] - rfull['pops']).max()) < 1e-12),
])

# ---------------- A4 张量积回归（关相互作用） ----------------
c2free = _calib(2, [], terms=())
rfree = ed.evolve_manybody(c2free, 'gg', t, Om=[Om, Om])
Pg, Pr = 1.0 - pred, pred
kron = np.stack([Pg * Pg, Pg * Pr, Pr * Pg, Pr * Pr], axis=1)
C.group('A4 关相互作用 ⇒ 双原子 = 两个独立拉比之积', [
    ('4 态布居 vs 张量积 < 1e-10',
     float(np.abs(rfree['pops'] - kron).max()) < 1e-10),
])

# ---------------- H 装配：厄米 + 稀疏≡稠密委托 ----------------
call_h = _calib(2, [_pair(0, 1, V=1.3, J=-0.9 + 0.0j, A=1.46 - 3.5j)])
Hs = asm.build_H_sparse(call_h, Om=[1.0, 2.0], delta=[0.5, -0.3])
Hd = Hs.toarray()
C.group('H 装配：稀疏厄米 + build_H_ed 委托一致', [
    ('gamma=0 ⇒ 厄米（max|H−H†| < 1e-12）',
     float(np.abs(Hd - Hd.conj().T).max()) < 1e-12),
    ('build_H_ed ≡ build_H_sparse.toarray()',
     float(np.abs(asm.build_H_ed(call_h, Om=[1.0, 2.0],
                                 delta=[0.5, -0.3]) - Hd).max()) == 0.0),
    ('有 gamma ⇒ 非厄米（对角带负虚部）',
     bool(np.any(asm.build_H_sparse(call_h, gamma=[1.0, 0.0]).diagonal().imag
                 < 0))),
])

# ---------------- D 非厄米衰减约定 ----------------
Gam = 2 * np.pi * 2.0                      # Γ MHz
tt = np.linspace(0.0, 200.0, 201)
c1d = _calib(1, [], terms=('I1',))
rd = ed.evolve_manybody(c1d, 'r', tt, gamma=[Gam])
pred_decay = np.exp(-Gam * 1e-3 * tt)      # P_r(t)=exp(-Γ×1e-3·t)（同 hard/evolve_util）
C.group('D 非厄米衰减 P_r(t)=exp(-Γ×1e-3·t)（约定同 hard.py）', [
    ('⟨n(t)⟩ vs 解析衰减 < 1e-9',
     float(np.abs(rd['pops_atom'][:, 0] - pred_decay).max()) < 1e-9),
    ('loss = 1 − 存活（末态）< 1e-9',
     float(abs(rd['loss'][-1] - (1.0 - pred_decay[-1]))) < 1e-9),
    ('backend 标 nonhermitian', 'nonhermitian' in rd['backend']),
])

# ---------------- R Result 契约 ----------------
need = {'t', 'pops', 'states', 'method', 'backend', 'error_report', 'loss'}
C.group('R Result 契约键齐全 + 附加观测量', [
    ('含全部契约键（t/pops/states/method/backend/error_report/loss）',
     need <= set(r2.keys())),
    ('states 与 pops 各列一一对应',
     len(r2['states']) == r2['pops'].shape[1]),
    ('method=manybody', r2['method'] == 'manybody'),
    ('附加观测量 pops_atom/S/corr 在',
     all(k in r2 for k in ('pops_atom', 'S', 'corr'))),
    ('pops_atom 形状 (n_t, N)', r2['pops_atom'].shape == (len(t), 2)),
])

sys.exit(0 if C.summary() else 1)
