# -*- coding: utf-8 -*-
"""场致加热与损失 · 引擎接线自检（Re0 重构 · 第 14 刀 Phase H3）。

守两件事：
  ① `split_field_loss` 的分项记账数学（spont/scatter/photoion/escape）与
     env²×逐态口径一致（合成 pops，纯单元）；
  ② mixed 单体端到端**消费** `field_loss`：开启时散射把布居抽出、分项守恒
     `Σpops + spont+scatter+photoion ≈ 1`；关闭（None）时不产出分项、与不接
     逐位一致（"None→不变"的硬闸由全基线里所有 `mixed_*` 判据共同把守——它们
     都走同一 `rhs_phi_pairs` + 驱动，loss_base=None 时新块被跳过）。

注：hard 用独立内核 `_rhs_hard_pair`（非 rhs_phi_pairs），其时变损耗接线列为
后续（capability 已在共用内核 + 装配器就绪）。见 内部记录（未随仓发布，存档在仓外）。
运行：D:\python310\python.exe checks\field_loss_engine_selfcheck.py
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from src.core.field_loss import FieldLossConfig
from src.core.field_build import build_field_loss
from src.core.evolve_util import split_field_loss
from src.core.atomic_levels import make_levels, square
from src.core.evolve_mixed import evolve_mixed
from checklib import Checker

ck = Checker('field_loss_engine_selfcheck')

GAM_6P = 8.464
GAM_60S = 4.35e-3


# ---------------- 包 1：split_field_loss 记账数学（合成 pops） ----------------
subs = []
t = np.linspace(0.0, 100.0, 101)              # ns
# 散射通道：src=0，env≡1，out=2 MHz → /ns 速率 = 2e-3；pops[:,0]≡1
env1 = lambda ts: np.ones_like(np.atleast_1d(np.asarray(ts, float)))
cfg = FieldLossConfig(2)
cfg.add_scatter(0, env1, out_mhz=2.0)
pops = np.zeros((len(t), 2))
pops[:, 0] = 1.0
G0 = np.zeros(2)
comp = split_field_loss(t, pops, G0, cfg)
sc_end = 2.0 * 1e-3 * t[-1]                    # 常数速率梯形积分精确
subs.append((f'散射分项 ∫env²·Γdt = {comp["scatter"][-1]:.5f} ≈ {sc_end:.5f}',
             abs(comp['scatter'][-1] - sc_end) < 1e-12
             and comp['spont'][-1] == 0.0 and comp['photoion'][-1] == 0.0))
# 自发辐射分项：Γ_ns 常数
Gn = np.array([0.0, 3e-3])
pops2 = np.zeros((len(t), 2))
pops2[:, 1] = 1.0
comp2 = split_field_loss(t, pops2, Gn, None)
subs.append((f'自发辐射分项 ∫Γdt={comp2["spont"][-1]:.5f}≈{3e-3*100:.5f}',
             abs(comp2['spont'][-1] - 3e-3 * 100.0) < 1e-12))
# budget 逃逸：tot 驱动 N_sc，loss_esc=1-exp(-N_sc/n_crit)
from src.core.field_loss import TrapModel
cfg3 = FieldLossConfig(2)
cfg3.add_scatter(0, env1, out_mhz=0.0, tot_mhz=10.0)   # 不抽布居，只计事件
trap = TrapModel(1013.8e-9, depth_K=1e-3, kappa_geo=2.0)   # n_crit≈4664
comp3 = split_field_loss(t, pops, np.zeros(2), cfg3, trap=trap)
n_sc_end = 10.0 * 1e-3 * t[-1]
esc_exp = 1.0 - np.exp(-n_sc_end / trap.n_crit)
subs.append((f'budget 逃逸 loss_esc={comp3["escape"][-1]:.3e}≈{esc_exp:.3e}',
             abs(comp3['escape'][-1] - esc_exp) < 1e-12 and comp3['scatter'][-1] == 0.0))
ck.group('H3 split_field_loss：spont/scatter/photoion/escape 分项数学', subs)


# ---------------- 包 2：mixed 单体端到端消费 field_loss ----------------
class _StubAtom:
    def __init__(self, tau_map):
        self.tau_map = tau_map

    def getStateLifetime(self, n, l, j, temperature=0.0):
        return self.tau_map[(int(n), int(l), round(float(j), 3))]


class _StubCache:
    def __init__(self, tau_map):
        self.atom = _StubAtom(tau_map)

    def E_mhz(self, n, l, j):
        return 0.0


stub = _StubCache({(60, 0, 0.5): 1e-6 / GAM_60S, (6, 1, 1.5): 1e-6 / GAM_6P})
states = make_levels(stub, [
    dict(n=60, l=0, j=0.5, m=-0.5, E_mhz=0.0),        # 60S（里德堡源态）
    dict(n=6, l=1, j=1.5, m=0.5, E_mhz=8000.0)])       # 短寿命伙伴
A = np.zeros((2, 2))
OM = 2000.0
A[0, 1] = A[1, 0] = OM
fields = [{'kind': 'exact', 'A': A, 'omega_L_mhz': 0.0,
           'waveform': square(0.0, 1e12), 'label': '远失谐光',
           'intensity_mwcm2': 0.0}]
cfg_engine = build_field_loss(stub, states, fields, [],
                              {'near_cutoff_mhz': 5000.0})
t2 = np.linspace(0.0, 200.0, 101)
subs = []
subs.append(('装配器产出非空 config（散射 on 60S>0）',
             cfg_engine is not None
             and cfg_engine.peak_loss_rates_mhz()[0] > 0.0))
on = evolve_mixed(states, fields, 0, t2, field_loss=cfg_engine)
off = evolve_mixed(states, fields, 0, t2, field_loss=None)
comp_on = on.get('loss_components')
subs.append(('开启→产出 loss_components；关闭→无该键',
             isinstance(comp_on, dict) and 'loss_components' not in off))
if isinstance(comp_on, dict):
    tot = (on['pops'].sum(axis=1) + comp_on['spont'] + comp_on['scatter']
           + comp_on['photoion'])
    dev = float(np.max(np.abs(tot - 1.0)))
    subs.append((f'分项守恒 Σpops+Σloss−1 最大={dev:.2e} (<1e-3)', dev < 1e-3))
    subs.append((f'散射致损>0 且 60S 被抽出 scatter_end={comp_on["scatter"][-1]:.3e}',
                 comp_on['scatter'][-1] > 0.0))
    subs.append((f'开启总损失 > 关闭（off loss_end={off["loss"][-1]:.1e}）',
                 on['loss'][-1] > off['loss'][-1] + 1e-9))
else:
    subs.append(('分项守恒（无 comp）', False))
    subs.append(('散射致损（无 comp）', False))
    subs.append(('开/关差异（无 comp）', False))
ck.group('H3 mixed 消费 field_loss：散射抽布居 + 分项守恒 + 开/关差异', subs)

ok = ck.summary()
sys.exit(0 if ok else 1)
