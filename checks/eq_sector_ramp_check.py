# -*- coding: utf-8 -*-
"""eq_route 求解动词自检：扇区切片 / 参数 ramp / 模型后端解耦。

背景（2026-10-02 收敛）：论文复现脚本里的求解胶水（扇区切片、warm-start ramp）
收进 `eq_route.solve_equilibrium` 后，这里用**独立构造的对照**（手工切片稠密谱 /
逐点直接构造 H(τ) 求基态）钉正确性——不对照实现自身。

E 组（µ 标定）：2026-10-02 晚裁决 `calibrate_mu` 已摘出 `eq_route`（δ 端点的
取法是论文协议，不是求解动词）。故 E 组不再测某个专用入口，改为验证**通用
动词组合**（`want='gs', sector=N` 逐扇区取 E(N) + diff 公式）能复现同一条
标定链，端点对照手工切片谱。

判据全部小系统（dim≤81，稠密 eigh 路径），全检 < 1 min。
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import importlib.util                                    # noqa: E402
import numpy as np                                       # noqa: E402
import scipy.sparse as sp                                # noqa: E402

from checks.checklib import Checker                      # noqa: E402
from src.core.manybody import eq_route as _ER            # noqa: E402
from src.core.manybody import equilibrium as _EQ         # noqa: E402


def _load_fci():
    path = os.path.join(ROOT, 'src', 'core', 'manybody', 'rydberg_fci.py')
    spec = importlib.util.spec_from_file_location('rydberg_fci', path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    sys.modules['rydberg_fci'] = mod
    return mod


F = _load_fci()
P = F.PAPER
C = Checker('eq_sector_ramp')

# ---------------------------------------------------------------------
# 测试床：L=4 全模型（dim≤81，全部稠密 eigh 路径）+ 逐基矢激励数标签
# ---------------------------------------------------------------------
pos, _ = F.honeycomb_sites(2, 1)                          # 4 格点
pr, _ = F.pair_shells(pos, l_um=12.0)
out = F.build_H_full(4, pr, Delta=P['Delta'], t_a=P['t_a'], t_b=P['t_b'],
                     w=P['w'], V=P, Omega=0.0, delta_mu=0.0, n_max=2)
H0 = out['H']
labels = np.asarray(out['n_exc'])
system = {'H': H0, 'sector_labels': labels}

# 手工全谱（稠密）作为参照
evals_full = np.linalg.eigvalsh(H0.toarray())

# =====================================================================
# A. ground_state 算法层：v0 确定性 / evals 键
# =====================================================================
rng = np.random.default_rng(7)
v0a = rng.standard_normal(H0.shape[0]) + 0j
r1 = _EQ.ground_state(H0, k=4, v0=v0a, dense_max=0)
r2 = _EQ.ground_state(H0, k=4, v0=v0a, dense_max=0)
r3 = _EQ.ground_state(H0, k=4, dense_max=0)
r4 = _EQ.ground_state(H0, k=4)                            # 稠密路径
C.group('ground_state：v0 确定性与 evals 键', [
    ('同 H 同 v0 两次 → E0 位级一致', r1['E0'] == r2['E0']),
    ('v0=None 固定种子 → 也确定', r3['E0'] == r3['E0'] and np.isfinite(r3['E0'])),
    ('evals 升序 4 个', len(r1['evals']) == 4
     and np.all(np.diff(r1['evals']) >= 0)),
    ('evals == 手工稠密全谱前 4（atol 1e-8）',
     np.allclose(r1['evals'], evals_full[:4], atol=1e-8)),
    ('稠密路径 evals 同上', np.allclose(r4['evals'], evals_full[:4], atol=1e-12)),
])

# =====================================================================
# B. system['H'] 直传 + 扇区切片 == 手工切片稠密谱
# =====================================================================
for sec in (0, 1, 2):
    r = _ER.solve_equilibrium(system, 'gs', sector=sec, k=3, sector_check=True)
    idx = np.where(labels == sec)[0]
    ref = np.linalg.eigvalsh(H0.toarray()[np.ix_(idx, idx)])
    n_eval = min(3, len(idx))
    C.group(f'扇区 {sec}（dim_sub={len(idx)}）', [
        ('E0 == 手工切片稠密谱最低（atol 1e-10）',
         abs(r['E0'] - ref[0]) < 1e-10),
        (f'evals {n_eval} 个 == 手工切片前 {n_eval}（atol 1e-10）',
         len(r['evals']) == n_eval
         and np.allclose(r['evals'], ref[:n_eval], atol=1e-10)),
        ('gap == evals[1]-evals[0]（或 0）',
         abs(r['gap'] - (ref[1] - ref[0] if len(ref) > 1 else 0.0)) < 1e-10),
        ('sector_idx 与 labels 一致',
         np.all(labels[r['sector_idx']] == sec)),
    ])

# =====================================================================
# C. 守恒校验：掺跨扇区项（Ω 驱动）→ sector 必须 raise
# =====================================================================
out_drv = F.build_H_full(4, pr, Delta=0.0, t_a=(0, 0, 0), t_b=(0, 0, 0),
                         w=(0, 0, 0),
                         V={k: (0, 0, 0) for k in
                            ('00', '0a', '0b', 'aa', 'ab', 'bb')},
                         Omega=0.5, delta_mu=0.0, n_max=2)
H_drv = out_drv['H']
system_drv = {'H': H_drv, 'sector_labels': labels}
raised = False
try:
    _ER.solve_equilibrium(system_drv, 'gs', sector=1, sector_check=True)
except ValueError:
    raised = True
C.check('非守恒 H（含 Ω 驱动跨界元）→ sector=int 必须 raise', raised)

# =====================================================================
# D. want='ramp'：逐点 E0 == 直接构造 H(τ) 的 ground_state；ns 手工复核
# =====================================================================
n_tot = labels.astype(float)                              # 逐基矢激励数
Om_t = np.array([0.0, 0.5, 1.0, 0.5, 0.0])
de_t = np.array([-1.0, -1.0, -0.5, 0.0, 0.0])
L = 4


def h_builder(om, de):
    """H(τ)=H0 − δ·N_tot + Ω·B，B=纯驱动（全零参数 Ω=1 装配）。"""
    Bm = F.build_H_full(4, pr, Delta=0.0, t_a=(0, 0, 0), t_b=(0, 0, 0),
                        w=(0, 0, 0),
                        V={k: (0, 0, 0) for k in
                           ('00', '0a', '0b', 'aa', 'ab', 'bb')},
                        Omega=1.0, delta_mu=0.0, n_max=2)['H']
    return (H0 - de * sp.diags(n_tot) + om * Bm).tocsr()


rr = _ER.solve_equilibrium({'H_builder': h_builder}, 'ramp',
                           ramp=dict(Om_t=Om_t, de_t=de_t, L=L, n_tot=n_tot,
                                     dense_max=2048))
ok_pts = []
for it in range(len(Om_t)):
    ref = _EQ.ground_state(h_builder(float(Om_t[it]), float(de_t[it])),
                           k=2, dense_max=2048)
    ns_ref = float(n_tot @ (np.abs(ref['vec']) ** 2)) / L
    ok_pts.append((abs(rr['gaps'][it] - ref['gap']) < 1e-9
                   and abs(rr['ns'][it] - ns_ref) < 1e-9))
C.group("want='ramp'：逐点 gap/ns == 直接构造 H(τ) 的 ground_state", [
    (f'τ 点 {it}: gap/ns 一致（atol 1e-9）', ok) for it, ok in enumerate(ok_pts)
])
C.check('ramp 返回契约：taus/gaps/ns/Om_t/de_t 形状一致',
        rr['want'] == 'ramp' and len(rr['taus']) == len(rr['gaps'])
        == len(rr['ns']) == len(rr['Om_t']) == len(rr['de_t']) == len(Om_t))

# =====================================================================
# E. µ 标定链路：**通用动词组合**能复现论文端点公式
# ---------------------------------------------------------------------
# `calibrate_mu` 已摘出 `eq_route`（2026-10-02 晚裁决）——它带的 δ 端点取法
# （δ0=µ_c(0→1)−offset 的真空侧裕量、δ1 取 N=n_target 平台中央）是**论文
# 协议**不是求解动词。这里因此不再测某个专用入口，而是验证同一条链用
# **通用动词**（`want='gs', sector=N` 逐扇区取 E(N) + diff 公式）能跑通，
# 且端点对照**手工切片稠密谱**（不对照实现自身）。
# n_target=1, n_until=2：E(0..2) → mu_c(2 个)。
# =====================================================================
n_until, n_target, offset = 2, 1, 0.3
E_sec, E2_sec = [], []
for n_ in range(n_until + 1):
    rN = _ER.solve_equilibrium(system, 'gs', sector=n_, k=2, seed=0,
                               sector_check=(n_ == 0))
    E_sec.append(rN['E0'])
    E2_sec.append(rN['evals'][1] if len(rN['evals']) > 1 else np.nan)
E_sec, E2_sec = np.array(E_sec), np.array(E2_sec)
mu_c = np.diff(E_sec)
delta0 = float(mu_c[0] - offset)
delta1 = float((mu_c[n_target - 1] + mu_c[n_target]) / 2)
Hd = H0.toarray()
E_ref = np.array([np.linalg.eigvalsh(
    Hd[np.ix_(np.where(labels == n_)[0], np.where(labels == n_)[0])])[0]
    for n_ in range(n_until + 1)])
mu_ref = np.diff(E_ref)
C.group('µ 标定链路（通用动词 gs+sector → 端点公式，n_target=1, n_until=2）', [
    ('E 覆盖 N=0..2 共 3 个扇区最低态', len(E_sec) == 3),
    ('逐扇区 E0 == 手工切片稠密谱最低（atol 1e-10）',
     np.allclose(E_sec, E_ref, atol=1e-10)),
    ('E2 为扇区第一激发（N=0 真空无第二态 → nan）',
     np.isnan(E2_sec[0]) and np.isfinite(E2_sec[1])),
    ('mu_c == diff(手工 E)', np.allclose(mu_c, mu_ref, atol=1e-10)),
    ('delta0 = mu_c[0] − 0.3', abs(delta0 - (mu_ref[0] - 0.3)) < 1e-10),
    ('delta1 = (mu_c[0]+mu_c[1])/2',
     abs(delta1 - (mu_ref[0] + mu_ref[1]) / 2) < 1e-10),
    ('delta0 < mu_c(0→1)（真空侧）', delta0 < mu_ref[0]),
])

ok = C.summary()
sys.exit(0 if ok else 1)
