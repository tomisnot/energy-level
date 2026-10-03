# -*- coding: utf-8 -*-
"""Rydberg FCI 引擎（src/core/manybody/rydberg_fci.py）自检——arXiv:2202.00699 复现。

纯数值判据（无 Qt、无 GUI）：几何/基矢/装配/守恒律/单粒子拓扑/有效模型，
全部对照**解析结果或论文数值**，不对照自身。要点：

- h.c. 完备性：pairs 单向遍历，跳迁矩阵元若漏共轭镜像 → 厄米性 FAIL（2026-10-02
  实测抓到过这个 bug：初版只装 j→i 单侧）。
- N_exc 守恒：Ω=0 时 H 不跨激发数扇区（逐非零元检查行/列扇区号相等）。
- 截断 vs 全谱：n_max 截断在 Ω=0 时必须与全空间按扇区切块的谱**逐本征值相等**。
- 三角环路通量：NNN 弯折 (i,k,j) 上三段自旋翻转矩阵元乘积的相位必须等于
  逐段解析相位之积（坐标直接算）——自旋轨道相位 e^{∓2iφ} 实现的独立定钉。
- 单粒子拓扑：最低带 |C|=1、C0+C1=0（Fukui），平坦度 f≈2.7、σ_B≈0.4（论文 §II）。
- 整体旋转不变性：谱不随格点整体旋转变化（自旋轨道相位的规范自洽）。
- 有效模型：NNN 相关跳迁 n̄_k=0/1 两值上的解析矩阵元、NNNN=−ζt_b、单体极限
  （N=1 扇区 == 手工单体矩阵）、p_ij 手性的叉积一致性与镜像反号。

耗时纪律：全检 < 3 min（最大稠密对角 3^6=729 / 2^12=4096；eigsh 冒烟 2^16）。
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import importlib.util                                    # noqa: E402
from collections import Counter, defaultdict             # noqa: E402
import numpy as np                                       # noqa: E402
import scipy.sparse as sp                                # noqa: E402

from checks.checklib import Checker                      # noqa: E402


def _load_fci():
    """按文件路径直接加载 rydberg_fci（单文件自包含，只依赖 numpy/scipy）。

    为什么不走 `from src.core.manybody import ...`：`src/core/__init__.py` 连带
    导入 evolve_twofield→coupling→numba，本 venv 不含 numba；引擎模块本身与
    numba 无关，按路径加载即可（sim 脚本同款 loader）。
    """
    path = os.path.join(ROOT, 'src', 'core', 'manybody', 'rydberg_fci.py')
    spec_ = importlib.util.spec_from_file_location('rydberg_fci', path)
    mod = importlib.util.module_from_spec(spec_)
    spec_.loader.exec_module(mod)
    sys.modules['rydberg_fci'] = mod
    return mod


F = _load_fci()

C = Checker('fci_rydberg')
V_NONE = dict([('00', (0, 0, 0)), ('0a', (0, 0, 0)), ('0b', (0, 0, 0)),
               ('aa', (0, 0, 0)), ('ab', (0, 0, 0)), ('bb', (0, 0, 0))])


def herm_max(H):
    """稀疏厄米偏差 ‖H−H†‖_max。"""
    D = H - H.conj().T
    return float(abs(D).max()) if D.nnz else 0.0


def _wrap_min(d, wraps):
    """PBC 最小像位移（自检独立实现：暴力扫 9 个卷绕组合取最近）。"""
    best, bd = float(np.hypot(*d)), np.asarray(d, float)
    for m in (-1, 0, 1):
        for n in (-1, 0, 1):
            cd = np.asarray(d, float) - m * wraps[0] - n * wraps[1]
            if float(np.hypot(*cd)) < best:
                best, bd = float(np.hypot(*cd)), cd
    return bd


# =====================================================================
# A. 几何与壳层配对
# =====================================================================
pos16, sub16 = F.honeycomb_sites(4, 2)
wr16 = [4 * np.array([18.0, 6.0 * np.sqrt(3.0)]),
        2 * np.array([18.0, -6.0 * np.sqrt(3.0)])]
pr16, drop16 = F.pair_shells(pos16, wraps=wr16, l_um=12.0)
pos20, _ = F.honeycomb_sites(5, 2)
wr20 = [5 * np.array([18.0, 6.0 * np.sqrt(3.0)]),
        2 * np.array([18.0, -6.0 * np.sqrt(3.0)])]
pr20, drop20 = F.pair_shells(pos20, wraps=wr20, l_um=12.0)
pos18, _ = F.honeycomb_sites(3, 3)
pr18, _ = F.pair_shells(pos18, l_um=12.0)
deg18 = [np.bincount([p['i'] for p in pr18 if p['shell'] == s]
                     + [p['j'] for p in pr18 if p['shell'] == s],
                     minlength=18) for s in range(3)]
# 6×6 开放片：内部格点（原胞指数 2..3 × 2..3）必须复现无限蜂巢配位数 3/6/3
pos72, sub72 = F.honeycomb_sites(6, 6)
pr72, _ = F.pair_shells(pos72, l_um=12.0)
deg72 = [defaultdict(int) for _ in range(3)]
for _p in pr72:
    deg72[_p['shell']][_p['i']] += 1
    deg72[_p['shell']][_p['j']] += 1
_a1 = 12 * np.array([1.5, np.sqrt(3) / 2])
_a2 = 12 * np.array([1.5, -np.sqrt(3) / 2])
inner = []
for _i in range(72):
    for _m in range(6):
        for _n in range(6):
            for _e in (np.zeros(2), 12 * np.array([1.0, 0.0])):
                if np.allclose(pos72[_i], _m * _a1 + _n * _a2 + _e):
                    inner.append((_i, _m, _n))
sel_inner = [i for i, m, n in inner if 2 <= m <= 3 and 2 <= n <= 3]
cnt16 = Counter(p['shell'] for p in pr16)
cnt20 = Counter(p['shell'] for p in pr20)

C.group('蜂巢几何：格点数 / 子晶格 / 壳层配对', [
    ('16 格点 = 4×2 原胞，A/B 各半',
     len(pos16) == 16 and (sub16 == 0).sum() == 8),
    ('环面 4×2：所有对有归属（pairs+dropped = C(16,2)=120）',
     len(pr16) + drop16 == 120),
    ('环面 4×2 壳层计数 {NN:24, NNN:40, NNNN:16}（T2=2a₂ 半卷绕 → 竖直'
     ' NNN/NNNN 方向坍缩，有限尺寸效应，非丢边）',
     (cnt16[0], cnt16[1], cnt16[2]) == (24, 40, 16)),
    ('环面 4×2：NN 对 = 3L/2 = 24（NN 位移无半卷绕歧义）', cnt16[0] == 24),
    ('环面 5×2：所有对有归属（120 收录 + 90 远对丢弃 = C(20,2)=190）',
     len(pr20) + drop20 == 190 and cnt20[0] == 30),
    ('开放片 6×6 内部格点：配位数恰 = 无限蜂巢 3/6/3',
     len(sel_inner) == 8
     and {deg72[0][i] for i in sel_inner} == {3}
     and {deg72[1][i] for i in sel_inner} == {6}
     and {deg72[2][i] for i in sel_inner} == {3}),
    ('phi 跟随 PBC 最小像方向（环面 96 对全量复算）',
     all(abs(p['phi'] - np.arctan2(d[1], d[0])) < 1e-9 for p in pr16
         for d in [_wrap_min(pos16[p['j']] - pos16[p['i']], wr16)])),
    ('shell 与距离 r/l 一致（96 对全量）',
     all(abs(p['R_um'] / 12.0 - F._SHELL_R[p['shell']]) < F._SHELL_TOL
         for p in pr16)),
])

# 上面 phi 复算那条写得绕，直接重算一遍覆盖（开放片，最小像=原图像）
pos8, sub8 = F.honeycomb_sites(2, 2)
pr8, _ = F.pair_shells(pos8, l_um=12.0)
pos6, sub6 = F.honeycomb_sites(3, 1)
pr6, _ = F.pair_shells(pos6, l_um=12.0)
C.check('phi == arctan2(r_jy−r_iy, r_jx−r_ix)（开放片逐对）',
        all(abs(p['phi'] - np.arctan2(*(pos6[p['j']] - pos6[p['i']])[::-1])) < 1e-9
            for p in pr6))

# =====================================================================
# B. 基矢
# =====================================================================
cs4, na4, nb4 = F.build_states_full(4)
cs4t, _, _ = F.build_states_full(4, n_max=2)
C.group('基矢编码：3^L 全空间 / N_exc 截断 / na-nb 解码', [
    ('build_states_full(4)：81 个、升序、唯一',
     len(cs4) == 81 and np.all(np.diff(cs4) > 0)),
    ('n_max=2 → 1+8+24=33 个', len(cs4t) == 33),
    ('n_max=L 退化为全空间', np.array_equal(F.build_states_full(4, n_max=4)[0], cs4)),
    ('na/nb 与 code 一致（81 态全查：逐格点解码求和 == 记录总量）',
     all(na4[t] == sum(((int(cs4[t]) >> (2 * k)) & 3) == 1 for k in range(4))
         and nb4[t] == sum(((int(cs4[t]) >> (2 * k)) & 3) == 2 for k in range(4))
         for t in range(81))),
])

# =====================================================================
# C. 全模型装配：解析小系统 + 厄米性 + 守恒律
# =====================================================================
# C1. L=1 解析：基序 |0⟩(code 0),|a⟩(1),|b⟩(2)；H=[[0,0,Ω],[0,Δ−δ,0],[Ω,0,−δ]]
out1 = F.build_H_full(1, [], Delta=10.0, t_a=(0, 0, 0), t_b=(0, 0, 0),
                      w=(0, 0, 0), V=V_NONE, Omega=0.7, delta_mu=1.3)
ev1 = np.sort(np.linalg.eigvalsh(out1['H'].toarray()))
want1 = np.sort([10.0 - 1.3,
                 (-1.3 + np.hypot(1.3, 2 * 0.7)) / 2.0,
                 (-1.3 - np.hypot(1.3, 2 * 0.7)) / 2.0])
C.check('L=1 解析谱（a 态 Δ−δ；{0,b} 块 (−δ±√(δ²+4Ω²))/2）',
        np.allclose(ev1, want1, atol=1e-12), f'{ev1} vs {want1}')

# C2. V^{ab} 定向：只有反向物种占据（a_i b_j / b_i a_j）带 +0.25
pairs2 = [dict(i=0, j=1, R_um=12.0, phi=0.0, shell=0)]
out2 = F.build_H_full(2, pairs2, Delta=0.0, t_a=(0, 0, 0), t_b=(0, 0, 0),
                      w=(0, 0, 0), V=dict(V_NONE, ab=(0.25, 0, 0)))
diag2 = np.asarray(out2['H'].diagonal())
want2 = 0.25 * (out2['na'][0] * out2['nb'][1] + out2['nb'][0] * out2['na'][1])
C.check('V^{ab} 对角 = 0.25·(a₀b₁+b₀a₁)，a·a / b·b / 真空组合为 0',
        np.allclose(diag2, want2, atol=1e-14))

# C3. 厄米性（论文全参数 + Ω 驱动；h.c. bug 回归测试）
out8 = F.build_H_full(8, pr8, Delta=F.PAPER['Delta'], t_a=F.PAPER['t_a'],
                      t_b=F.PAPER['t_b'], w=F.PAPER['w'], V=F.PAPER,
                      Omega=0.5, delta_mu=1.0)
hm8 = herm_max(out8['H'])
C.check('全模型厄米性 ‖H−H†‖_max=0（L=8 全参数 + Ω）', hm8 < 1e-12,
        f'max|H−H†|={hm8:.2e}')

# C4. Ω=0 → N_exc 守恒
out8s = F.build_H_full(8, pr8, Delta=F.PAPER['Delta'], t_a=F.PAPER['t_a'],
                       t_b=F.PAPER['t_b'], w=F.PAPER['w'], V=F.PAPER,
                       Omega=0.0, delta_mu=0.5)
Hc8 = out8s['H'].tocoo()
C.check('Ω=0：所有非零元 n_exc(行)=n_exc(列)（跳迁守恒激发数）',
        bool(np.all(out8s['n_exc'][Hc8.row] == out8s['n_exc'][Hc8.col])))

# C5. Ω 驱动选择定则：只 0↔b（a 不被驱动）
out8d = F.build_H_full(8, pr8, Delta=F.PAPER['Delta'], t_a=F.PAPER['t_a'],
                       t_b=F.PAPER['t_b'], w=F.PAPER['w'], V=F.PAPER,
                       Omega=0.5, delta_mu=0.0)
Hd8 = out8d['H'].tocoo()
dr = np.abs(out8d['n_exc'][Hd8.row] - out8d['n_exc'][Hd8.col]) == 1
codes8 = out8d['codes']
xors = codes8[Hd8.row[dr]] ^ codes8[Hd8.col[dr]]
ii = (np.log2(xors.astype(float)).astype(int) - 1) // 2
occ_c = (codes8[Hd8.col[dr]] >> (2 * ii)) & 3
occ_r = (codes8[Hd8.row[dr]] >> (2 * ii)) & 3
ok_sel = dr.sum() > 0 and bool(np.all(
    ((occ_c == 0) & (occ_r == 2)) | ((occ_c == 2) & (occ_r == 0))))
C.check(f'Ω 项只耦合 0↔b（{int(dr.sum())} 个驱动元，a 全程不参与）', ok_sel)

# C6. 截断 vs 全谱（Ω=0，L=4 蝶形片：全 81，N_exc≤2 扇区 33）
pos4c, _ = F.honeycomb_sites(1, 1)
pr4c, _ = F.pair_shells(pos4c, l_um=12.0)
out4f = F.build_H_full(4, pr4c, Delta=F.PAPER['Delta'], t_a=F.PAPER['t_a'],
                       t_b=F.PAPER['t_b'], w=F.PAPER['w'], V=F.PAPER,
                       Omega=0.0, delta_mu=0.8)
H4f = out4f['H'].toarray()
mask = out4f['n_exc'] <= 2
ev_sect = np.sort(np.linalg.eigvalsh(H4f[np.ix_(mask, mask)]))
out4t = F.build_H_full(4, pr4c, Delta=F.PAPER['Delta'], t_a=F.PAPER['t_a'],
                       t_b=F.PAPER['t_b'], w=F.PAPER['w'], V=F.PAPER,
                       Omega=0.0, delta_mu=0.8, n_max=2)
ev_trunc = np.sort(np.linalg.eigvalsh(out4t['H'].toarray()))
C.check('Ω=0 截断(n_max=2) 谱 == 全空间 N_exc≤2 扇区谱（L=4 逐本征值）',
        np.allclose(ev_sect, ev_trunc, atol=1e-10),
        f'max diff={np.abs(ev_sect - ev_trunc).max():.2e}, dim={int(mask.sum())}')

# C7. dropped_drive 计数：L=4, n_max=2, Ω≠0 → 每格点 C(3,2)·2²=12 ×4 = 48
out4d = F.build_H_full(4, [], Delta=10.0, t_a=(0, 0, 0), t_b=(0, 0, 0),
                       w=(0, 0, 0), V=V_NONE, Omega=1.0, n_max=2)
C.check('截断驱动丢元计数 dropped_drive=48（产生方向超出 n_max 被显式上报）',
        out4d['dropped_drive'] == 48, f"got {out4d['dropped_drive']}")

# =====================================================================
# D. 三角环路通量（自旋轨道相位的独立定钉）
# =====================================================================
nn6 = {(min(p['i'], p['j']), max(p['i'], p['j'])) for p in pr6 if p['shell'] == 0}
found = None
for p in pr6:
    if p['shell'] != 1:
        continue
    i, j = p['i'], p['j']
    ks = [k for k in range(6) if k not in (i, j)
          and (min(k, i), max(k, i)) in nn6 and (min(k, j), max(k, j)) in nn6]
    if len(ks) == 1:
        found = (i, j, ks[0])
        break
ok_flux, flux_detail = False, ''
if found:
    i, j, k = found
    out6 = F.build_H_full(6, pr6, Delta=100.0, t_a=(0, 0, 0), t_b=(0, 0, 0),
                          w=(1.0, 1.0, 0.0), V=V_NONE)
    H6 = out6['H'].toarray()
    idx = {int(c): t for t, c in enumerate(out6['codes'])}
    s0 = idx[1 << (2 * i)]                      # a_i
    s1 = idx[2 << (2 * k)]                      # b_k   ⟨b_k|H|a_i⟩ (NN)
    s2 = idx[1 << (2 * j)]                      # a_j   ⟨a_j|H|b_k⟩ (NN)
    s3 = idx[2 << (2 * i)]                      # b_i   ⟨b_i|H|a_j⟩ (NNN)
    prod = H6[s1, s0] * H6[s2, s1] * H6[s3, s2]
    ang = lambda a, b: np.arctan2(*(pos6[b] - pos6[a])[::-1])
    phi_ik, phi_kj, phi_ij = ang(i, k), ang(k, j), ang(i, j)
    flux_want = np.exp(2j * phi_ik) * np.exp(-2j * phi_kj) * np.exp(2j * phi_ij)
    ok_flux = (abs(abs(prod) - 1.0) < 1e-9
               and abs(np.angle(prod * np.conj(flux_want))) < 1e-9)
    flux_detail = (f'环 {i}->{k}->{j}->{i}: |Π|={abs(prod):.6f}, '
                   f'arg={np.angle(prod):+.4f} vs 解析 {np.angle(flux_want):+.4f}')
C.check('NNN 三角环路通量 = 逐段自旋翻转解析相位之积（|Π|=ω³ 且 arg 对上）',
        ok_flux, flux_detail or '没找到唯一公共近邻的 NNN 对（几何异常）')

# =====================================================================
# E. 单粒子带结构：连续性 / Chern / 平坦度 / 旋转不变
# =====================================================================
path, _ = F.kpath_KGM(40)
bands = F.band_structure(path)
met = F.band_metrics(Nk=24)
c0 = F.chern_fukui(24, band=0)
c1 = F.chern_fukui(24, band=1)
C.group('Bloch 4 带与拓扑（论文 Fig.1(c)：C=±1、f≈2.7、σ_B≈0.4）', [
    ('能带形状 (n_k, 4)', bands.shape[1] == 4),
    ('bloch_H 厄米（conj(ph) bug 回归：错→eigvalsh 下三角静默吞掉）',
     (lambda Hk: float(np.abs(Hk - Hk.conj().T).max()) < 1e-12)(
         F.bloch_H(np.array([0.37, -0.91]), Delta=F.PAPER['Delta'],
                   t_a=F.PAPER['t_a'], t_b=F.PAPER['t_b'], w=F.PAPER['w']))),
    ('Δ 加在 a 物种（基 [A_a,A_b,B_a,B_b] 偶数位；Γ 对角 = Δ−6t^a_NNN）',
     abs(F.bloch_H(np.zeros(2), Delta=F.PAPER['Delta'], t_a=F.PAPER['t_a'],
                   t_b=F.PAPER['t_b'], w=F.PAPER['w'])[0, 0]
         - (F.PAPER['Delta'] - 6 * F.PAPER['t_a'][1])) < 1e-9),
    ('沿 KΓMK 无带交叉跳变（相邻 k 点逐带连续 < 0.4 MHz）',
     np.abs(np.diff(bands, axis=0)).max() < 0.4),
    (f'最低带 Chern = {c0}（Fukui, Nk=24）→ |C|=1', abs(c0) == 1),
    (f'前两带陈数反号（C0={c0}, C1={c1}）', c0 + c1 == 0),
    (f"最低带平坦度 f={met['flatness']:.2f}（论文 2.7，容差 ±0.4）",
     abs(met['flatness'] - 2.7) < 0.4),
    (f"Berry 曲率涨落 σ_B={met['sigma_B']:.2f}（论文 0.4，容差 ±0.15）",
     abs(met['sigma_B'] - 0.4) < 0.15),
])

# 3×3 环面单粒子谱：实空间引擎（V=0 口径）== bloch 9 个允许 k 的 4 带
pos18b, _ = F.honeycomb_sites(3, 3)
wr18b = [3 * np.array([18.0, 6 * np.sqrt(3.0)]),
         3 * np.array([18.0, -6 * np.sqrt(3.0)])]
pr18b, _ = F.pair_shells(pos18b, wraps=wr18b, l_um=12.0)
out_sp = F.build_H_full(18, pr18b, Delta=F.PAPER['Delta'], t_a=F.PAPER['t_a'],
                        t_b=F.PAPER['t_b'], w=F.PAPER['w'], V=V_NONE,
                        Omega=0.0, n_max=1)
ev_real = np.sort(np.linalg.eigvalsh(
    out_sp['H'].toarray()[np.ix_(out_sp['n_exc'] == 1, out_sp['n_exc'] == 1)]))
_a1 = 12 * np.array([1.5, np.sqrt(3) / 2])
_a2 = 12 * np.array([1.5, -np.sqrt(3) / 2])
_det = _a1[0] * _a2[1] - _a1[1] * _a2[0]
_b1 = 2 * np.pi / _det * np.array([_a2[1], -_a2[0]])
_b2 = 2 * np.pi / _det * np.array([-_a1[1], _a1[0]])
ev_blo = np.sort(np.concatenate([
    np.linalg.eigvalsh(F.bloch_H((m / 3) * _b1 + (n / 3) * _b2,
                                 Delta=F.PAPER['Delta'], t_a=F.PAPER['t_a'],
                                 t_b=F.PAPER['t_b'], w=F.PAPER['w']))
    for m in range(3) for n in range(3)]))
C.check('3×3 环面单粒子谱：实空间引擎 == bloch 9k×4 带（V=0 口径，差 < 1e-10）',
        np.allclose(ev_real, ev_blo, atol=1e-10),
        f'max diff={np.abs(ev_real - ev_blo).max():.2e}')

# 整体旋转不变性（L=6 开放片，全 3^6=729 稠密）
th = 0.7
Rm = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
kw = dict(Delta=5.0, t_a=(1, .2, .1), t_b=(.5, .1, .05), w=(2, .4, .3), V=F.PAPER)
ev_a6 = np.linalg.eigvalsh(F.build_H_full(6, pr6, **kw)['H'].toarray())
ev_b6 = np.linalg.eigvalsh(
    F.build_H_full(6, F.pair_shells(pos6 @ Rm.T, l_um=12.0)[0], **kw)['H'].toarray())
C.check('整体旋转 θ=0.7 rad：全模型谱不变（L=6 全空间）',
        np.allclose(ev_a6, ev_b6, atol=1e-9),
        f'max diff={np.abs(ev_a6 - ev_b6).max():.2e}')

# =====================================================================
# F. 有效模型（Eq.5）
# =====================================================================
pos12, _ = F.honeycomb_sites(3, 2)
wr12 = [3 * np.array([18.0, 6.0 * np.sqrt(3.0)]),
        2 * np.array([18.0, -6.0 * np.sqrt(3.0)])]
pr12, _ = F.pair_shells(pos12, wraps=wr12, l_um=12.0)
oute = F.build_H_eff(12, pr12, pos12, Omega=0.3, delta_mu=0.4, wraps=wr12)
oute0 = F.build_H_eff(12, pr12, pos12, Omega=0.0, wraps=wr12)
Hc_e = oute0['H'].tocoo()
offd_mask = Hc_e.row != Hc_e.col              # 只看非对角（对角含 λt_b·n_pairs 不算跳迁）
offd = Hc_e.data[offd_mask]
reals = offd[np.abs(offd.imag) < 1e-12].real
tb, zt, lam = F.EFFECTIVE['t_b'], abs(F.EFFECTIVE['zeta'] * F.EFFECTIVE['t_b']), F.EFFECTIVE['lam']
ok_amps = bool(np.all((np.abs(reals - (-tb)) < 1e-12)
                      | (np.abs(reals - (-F.EFFECTIVE['zeta'] * tb)) < 1e-12)
                      | (np.abs(reals - (-tb / 3**1.5)) < 1e-12))  # n̄_k=1 的 NNN
               and (np.abs(reals + tb) < 1e-12).sum() > 0
               and (np.abs(reals - (-F.EFFECTIVE['zeta'] * tb)) < 1e-12).sum() > 0)
C.group('有效模型装配（Eq.5，L=12 环面）', [
    ('厄米性 ‖H−H†‖=0（含 Ω 驱动 + NNN 相关项）', herm_max(oute['H']) < 1e-12),
    ('Ω=0 时 N_tot 守恒（逐非零元）',
     bool(np.all(oute0['n_tot'][Hc_e.row] == oute0['n_tot'][Hc_e.col]))),
    ('实数跳迁幅值只有 −t_b（NN）与 −ζt_b（NNNN），且两者都出现', ok_amps),
    ('NNN 相关项模长 ∈ [|t_b(3^{-3/2}−λ)|, t_b(3^{-3/2}+λ)]',
     bool(np.all((np.abs(offd) > tb / 3**1.5 - lam * tb - 1e-9)
                 & (np.abs(offd) < tb / 3**1.5 + lam * tb + 1e-9)
                 | (np.abs(np.abs(offd) - tb) < 1e-9)
                 | (np.abs(np.abs(offd) - zt) < 1e-9)))),
])

# NNN 相关跳迁解析值（4 格点蝶形，取第一条 NNN 对）
pos4m, _ = F.honeycomb_sites(2, 1)
pr4m, _ = F.pair_shells(pos4m, l_um=12.0)
nnn4 = [p for p in pr4m if p['shell'] == 1]
ok_nnn, detail_nnn = False, ''
if nnn4:
    p = nnn4[0]
    i, j = p['i'], p['j']
    k, pp = F._middle_site(pos4m, i, j, l_um=12.0)
    Ht = F.build_H_eff(4, pr4m, pos4m)['H'].toarray()
    amp_nk0 = Ht[1 << i, 1 << j]                       # j→i，k 空
    amp_nk1 = Ht[(1 << i) | (1 << k), (1 << j) | (1 << k)]   # k 被占
    w0 = -tb / 3**1.5 - lam * tb * np.exp(2j * np.pi * pp / 3.0)
    w1 = -tb / 3**1.5
    ok_nnn = (abs(amp_nk0 - w0) < 1e-12 and abs(amp_nk1 - w1) < 1e-12)
    detail_nnn = (f'(i,j,k)=({i},{j},{k}) p={pp}; '
                  f'nk0={amp_nk0:.5f} want {w0:.5f}; nk1={amp_nk1:.5f} want {w1:.5f}')
C.check('NNN 相关跳迁：n̄_k=0 → −t_b(3^{-3/2}+λe^{i2πp/3})；n̄_k=1 → −t_b·3^{-3/2}',
        ok_nnn, detail_nnn or '4 格点蝶形无 NNN 对（几何异常）')

# p_ij 手性：与叉积转向逐对一致（顺时针 p=+1）
posC, _ = F.honeycomb_sites(3, 3)
prC, _ = F.pair_shells(posC, l_um=12.0)
nnnC = [q for q in prC if q['shell'] == 1]
ok_p, n_p = True, 0
for q in nnnC:
    i, j = q['i'], q['j']
    k, pp = F._middle_site(posC, i, j, l_um=12.0)
    dik = posC[k] - posC[i]
    dkj = posC[j] - posC[k]
    cross = dik[0] * dkj[1] - dik[1] * dkj[0]
    ok_p &= (pp == (1 if cross < 0 else -1))
    n_p += 1
C.check(f'p_ij 与叉积转向逐对一致（{n_p} 个 NNN 对）', ok_p and n_p > 0)

# 镜像（y→−y）反号
posCm = posC * np.array([1.0, -1.0])
sign_flip = all(
    F._middle_site(posC, q['i'], q['j'], l_um=12.0)[1]
    == -F._middle_site(posCm, q['i'], q['j'], l_um=12.0)[1] for q in nnnC)
C.check(f'镜像翻转 y→−y：{len(nnnC)} 个 NNN 对的 p_ij 全体反号', sign_flip)

# 单体极限：N=1 扇区谱 == 手工单体跳迁矩阵谱
# （单粒子时中间格点必空 → NNN 幅 = −t_b(3^{-3/2} + λ e^{i2πp/3})，n̄_k=0 口径）
H1 = oute0['H'].toarray()
sel1 = np.where(oute0['n_tot'] == 1)[0]
Hm = np.zeros((12, 12), complex)
for q in pr12:
    i, j, s = q['i'], q['j'], q['shell']
    if s == 0:
        amp = -tb
    elif s == 1:
        _k, _p = F._middle_site(pos12, i, j, wraps=wr12, l_um=12.0)
        amp = -tb / 3**1.5 - lam * tb * np.exp(2j * np.pi * _p / 3.0)
    else:
        amp = -F.EFFECTIVE['zeta'] * tb
    Hm[i, j] = amp
    Hm[j, i] = np.conj(amp)
C.check('有效模型 N=1 扇区 == 手工单体跳迁矩阵（谱逐点相等）',
        np.allclose(np.sort(np.linalg.eigvalsh(H1[np.ix_(sel1, sel1)])),
                    np.sort(np.linalg.eigvalsh(Hm)), atol=1e-10))

# =====================================================================
# G. ramp 剖面 + 大维度冒烟（求解动词在 equilibrium/eq_route；
#    ramp 波形属论文协议，归 rydberg_fci）
# =====================================================================
from src.core.manybody import equilibrium as _EQ        # noqa: E402

Om, de = F.sweep_profile(np.array([0.0, 1 / 3, 0.5, 2 / 3, 1.0]),
                         Omega_max=1.0, delta0=-1.47, delta1=-0.73)
C.group('三段式 sweep 剖面（Fig.4(a)，rydberg_fci.sweep_profile）', [
    ('Ω(0)=0、Ω(1/3)=Ω(1/2)=Ω(2/3)=Ω_max、Ω(1)=0',
     np.allclose(Om, [0, 1, 1, 1, 0])),
    ('δ 端点精确、中段线性（δ(1/2)=(δ0+δ1)/2）',
     np.allclose(de, [-1.47, -1.47, -1.10, -0.73, -0.73])),
])

# L=16 有效模型（2^16=65536）eigsh 真冒烟——sim 脚本的 ramp 就是这个规模
pos16b, _ = F.honeycomb_sites(4, 2)
pr16b, _ = F.pair_shells(pos16b, l_um=12.0)
oute16 = F.build_H_eff(16, pr16b, pos16b, Omega=0.3, delta_mu=-1.0)
r16 = _EQ.ground_state(oute16['H'], k=6)
w16 = r16['evals']
C.check('L=16 有效模型（dim=65536）eigsh 收敛：6 个最低本征值实数、升序',
        np.all(np.isfinite(w16.real)) and np.all(np.diff(w16.real) >= -1e-9),
        f'{np.round(w16.real, 4)}')

out10 = F.build_H_full(10, F.pair_shells(F.honeycomb_sites(5, 1)[0], l_um=12.0)[0],
                       Delta=F.PAPER['Delta'], t_a=F.PAPER['t_a'], t_b=F.PAPER['t_b'],
                       w=F.PAPER['w'], V=F.PAPER, Omega=0.0, delta_mu=0.0, n_max=3)
r10 = _EQ.ground_state(out10['H'], k=4)
w10 = r10['evals']
C.check(f"L=10 全模型 n_max=3（dim={out10['dim']}）低 4 本征值冒烟",
        len(w10) == 4 and np.all(np.isfinite(w10.real)))

# =====================================================================
# H. 卷绕簇 honeycomb_cluster（论文 Fig.2(a) 的菱形/斜平行四边形环面簇）
# =====================================================================
posR, subR, wrR = F.honeycomb_cluster([[3, 1], [1, 3]])
prR, dropR = F.pair_shells(posR, wraps=wrR, l_um=12.0)
_cntR = Counter(q['shell'] for q in prR)
C.group('16 菱形簇（T1=3a1+a2、T2=a1+3a2，论文 Fig.2(a) 首簇）', [
    ('L=16 且格点无重复', len(posR) == 16
     and len(set(map(tuple, np.round(posR, 6)))) == 16),
    ('卷绕矢 = l(6, ±√3) = (72, ±12√3) µm',
     np.allclose(wrR, [[72.0, 12 * np.sqrt(3)], [72.0, -12 * np.sqrt(3)]])),
    ('菱形等边（|T1|=|T2|）',
     np.isclose(np.hypot(*wrR[0]), np.hypot(*wrR[1]))),
    ('壳层计数 = {24,40,16}：半卷绕 (T1−T2)/2 = a1−a2 恰为 NNN 矢量，'
     'NNN 轴坍缩（与 4×2 矩形环面同构）',
     (_cntR[0], _cntR[1], _cntR[2]) == (24, 40, 16)),
    ('有归属 + dropped = C(16,2) = 120', len(prR) + dropR == 120),
])

posP, subP, wrP = F.honeycomb_cluster([[1, -4], [2, 2]])
prP, dropP = F.pair_shells(posP, wraps=wrP, l_um=12.0)
_cntP = Counter(q['shell'] for q in prP)
C.group('20 斜平行四边形簇（T1=a1+2a2、T2=−4a1+2a2）', [
    ('L=20 且格点无重复', len(posP) == 20
     and len(set(map(tuple, np.round(posP, 6)))) == 20),
    ('壳层计数 = 无限晶格值 {30,60,30}（无卷绕坍缩）',
     (_cntP[0], _cntP[1], _cntP[2]) == (30, 60, 30)),
    ('有归属 + dropped = C(20,2) = 190', len(prP) + dropP == 190),
])

pos4c, _, wr4c = F.honeycomb_cluster([[4, 0], [0, 2]])
C.check('honeycomb_cluster 对角 M=[[4,0],[0,2]] 与 honeycomb_sites(4,2) 同构'
        '（L=16、wraps=[4a1,2a2]）',
        len(pos4c) == 16
        and np.allclose(wr4c, [4 * np.array([18.0, 6 * np.sqrt(3)]),
                               2 * np.array([18.0, -6 * np.sqrt(3)])]))

ok = C.summary()
sys.stdout.flush()
os._exit(0 if ok else 1)
