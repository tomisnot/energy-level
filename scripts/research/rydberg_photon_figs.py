# -*- coding: utf-8 -*-
"""涌现光子谱学 · 科研绘图（引擎实算，真实 ARC 偶极矩阵元）。

对应 00_方向_涌现光子谱学.md §6 清单与 §7 判据。产出五张图（figs_photon/）：
  figA  双 m 通道 J(θ) vs 文档 (1−3cos²θ)：Δm=0 逐位符合；Δm=+1 反号
        （半幅）——球基 Condon-Shortley 相位，m 通道是设计旋钮
  figB  键符号图（2 通道 × 2 场方向）：谁给干净 Néel、谁阻挫
        （约定 H=ΣJ(S⁺S⁻+h.c.)：J>0=AFM——文档 A.0 标签与其 A.2 自洽约定
         相反，2D 方格的干净 Néel 在"场⊥面"，非文档推荐的"场∥面"）
  figC  J̃(k) 布里渊区（无限格点 FFT）：序波矢 = argmin J̃；
        Δm=0 tilt=0 → (π,π) Néel ✓；tilt=90 → 混合符号 → 有限-Q/阻挫
  figD  软化指纹 log-log（直接格点求和，C₃ 来自引擎标定）：
        左 FM@Γ（Δm=+1, tilt=0）：α≈1 —— 2D 长程 1/R³ 把 FM 磁振子从
          k² 抬成 |q| 线性（文档表"FM=z2"只对短程成立）；
          且 ω/(v|q|) 对 |q| 线性下降 = 文档 §4.2 指纹，但它属于 Γ
        右 AFM@Q*（Δm=0, tilt=0）：α≈2 —— 交错因子使求和解析，
          |q| 非解析不出现在 Q*（文档 §4.2"反铁磁在 Q 处同样展开"需修正）
  figE  有限 L=12 团簇 vs 无限格点（同通道 Δm=+1 tilt=90）：
        上一轮 α≈1.85 = 有限尺寸伪影（最小 k·a≈0.52 够不到 |q| 窗口）
运行：python scripts/research/rydberg_photon_figs.py
"""
import os
import sys
import time

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager as _fm

sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))

from arc import Rubidium87
from src.core import core
from src.manybody.model import ManyBodyModel
from src.manybody import assemble as asm

# ⚠ arc 包 import 时会把 font.family 改成 serif（无中文字形）——必须在其后设置
for _fp in (r'C:\Windows\Fonts\msyh.ttc', r'C:\Windows\Fonts\simhei.ttf',
            r'C:\Windows\Fonts\simsun.ttc'):
    if os.path.exists(_fp):
        _fm.fontManager.addfont(_fp)
_CJK = next((f.name for f in _fm.fontManager.ttflist
             if f.name in ('Microsoft YaHei', 'SimHei', 'SimSun')),
            'DejaVu Sans')
plt.rcParams.update({'font.size': 10, 'figure.dpi': 150,
                     'font.family': 'sans-serif',
                     'font.sans-serif': [_CJK, 'DejaVu Sans'],
                     'axes.unicode_minus': False,
                     'axes.spines.top': False, 'axes.spines.right': False})

G = (60, 0, 0.5, 0.5)           # 60S_1/2 m=+1/2 (g)
R1 = (60, 1, 1.5, 1.5)          # 60P_3/2 m=+3/2 (r)，Δm=+1
R0 = (60, 1, 1.5, 0.5)          # 60P_3/2 m=+1/2 (r)，Δm=0（文档公式通道）
A_UM = 10.0
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'figs_photon')
os.makedirs(OUT, exist_ok=True)

cache = core.ArcCache(Rubidium87())
_t0 = time.time()


def say(*a):
    print(f'[{time.time() - _t0:6.1f}s]', *a, flush=True)


def P2(c):
    return 0.5 * (3.0 * c * c - 1.0)


def calib_J2(rket, posA, posB):
    m = ManyBodyModel(terms=('I2',), n_window=1)
    m.add_atom(G, rket, posA)
    m.add_atom(G, rket, posB)
    return asm.calibrate(cache, m, second_order=False)['pairs'][0][
        'raw_J'].real


def calib_JR(rket, R, theta_deg):
    """标定单对 (R,θ,φ=0) 的 raw_J。"""
    t = np.deg2rad(theta_deg)
    return calib_J2(rket, (0, 0, 0),
                    (R * np.sin(t), 0, R * np.cos(t)))


# =====================================================================
# A. 双通道角结构
# =====================================================================
say('figA: 双通道 J(θ)…')
th = np.linspace(0, 90, 19)
ct = np.cos(np.deg2rad(th))
J_p1 = np.array([calib_J2(R1, (0, 0, 0), (A_UM * s, 0, A_UM * c))
                 for s, c in zip(np.sin(np.deg2rad(th)), ct)])
J_0 = np.array([calib_J2(R0, (0, 0, 0), (A_UM * s, 0, A_UM * c))
                for s, c in zip(np.sin(np.deg2rad(th)), ct)])
sp1 = np.sum(J_p1 * P2(ct)) / np.sum(P2(ct) ** 2)
s0 = np.sum(J_0 * (1 - 3 * ct ** 2)) / np.sum((1 - 3 * ct ** 2) ** 2)
r_p1 = np.max(np.abs(J_p1 - sp1 * P2(ct))) / max(abs(sp1), 1e-30)
r_0 = np.max(np.abs(J_0 - s0 * (1 - 3 * ct ** 2))) / max(abs(s0), 1e-30)
say(f'  Δm=+1 ∝P₂ 残差 {r_p1:.1e} (sc={sp1:.4g}) | '
    f'Δm=0 ∝(1−3c²) 残差 {r_0:.1e} (sc={s0:.4g})')

fig, ax = plt.subplots(figsize=(7.0, 4.8))
ax.axhline(0, color='gray', lw=0.6)
ax.plot(th, J_p1, 'o', ms=5, color='tab:red',
        label='引擎 Δm=+1（60S m=1/2 → 60P$_{3/2}$ m=3/2）')
ax.plot(th, sp1 * P2(ct), '-', lw=1.5, color='tab:red', alpha=0.55,
        label=rf'拟合 $\propto +P_2(\cos\theta)$（残差 {r_p1:.0e}）')
ax.plot(th, J_0, 's', ms=5, color='tab:blue',
        label='引擎 Δm=0（60S m=1/2 → 60P$_{3/2}$ m=1/2）')
ax.plot(th, s0 * (1 - 3 * ct ** 2), '-', lw=1.5, color='tab:blue',
        alpha=0.55,
        label=rf'文档公式 $(1-3\cos^2\theta)$（残差 {r_0:.0e}）')
ma = np.degrees(np.arccos(1 / np.sqrt(3)))
ax.axvline(ma, color='green', lw=0.8, ls=':')
ax.text(ma + 1, 0.55 * np.abs(J_0).max(), f'magic角 {ma:.1f}°',
        color='green', fontsize=8, rotation=90)
ax.set_xlabel('键相对量子化轴 z（=场方向）θ [deg]')
ax.set_ylabel(r'$J_{ij}=\langle rg|V_{\rm dd}|gr\rangle$ [MHz]')
ax.set_title('图 A：偶极交换角结构（引擎实算）——Δm=0 逐位符合文档公式；\n'
             'Δm=+1 与文档**反号**（球基 Condon-Shortley 相位）\n'
             r'$\Rightarrow$ 换 m 通道 = 翻转 FM/AFM 归属，是免费的实验设计旋钮',
             fontsize=10.5)
ax.legend(fontsize=7.5, loc='lower left')

axin = ax.inset_axes([0.60, 0.50, 0.36, 0.38])
Rs = np.array([2, 3, 4, 6, 8, 10, 14, 20], float) * A_UM
Js0 = np.array([calib_J2(R0, (0, 0, 0), (r, 0, 0)) for r in Rs])
C3_0 = float(np.mean(Js0 * Rs ** 3))            # Δm=0: J(θ=90)=+C₃/R³
axin.plot(Rs / A_UM, Js0 * Rs ** 3 / A_UM ** 3, 's', ms=4,
          color='tab:purple')
axin.axhline(C3_0 / A_UM ** 3, color='tab:purple', ls=':', lw=1)
axin.set_xscale('log')
axin.set_xlabel('R/a', fontsize=7)
axin.set_ylabel(r'$J\cdot R^3$ [MHz·µm³]', fontsize=7)
axin.tick_params(labelsize=7)
axin.set_title(r'$1/R^3$ 律 OK', fontsize=8)
fig.tight_layout()
fig.savefig(os.path.join(OUT, 'figA_angular_structure.png'))
plt.close(fig)
say(f'figA 完成；C₃(Δm=0) = {C3_0:.4g} MHz·µm³')

# =====================================================================
# B. 键符号图：2 通道 × 2 场方向
# =====================================================================
say('figB: 键符号图…')


def lattice(L, tilt_deg):
    t = np.deg2rad(tilt_deg)
    u = np.array([1.0, 0, 0])
    v = np.array([0, np.cos(t), np.sin(t)])
    pos = np.array([(ix - (L - 1) / 2) * A_UM * u
                    + (iy - (L - 1) / 2) * A_UM * v
                    for ix in range(L) for iy in range(L)])
    return pos, u, v


fig, axes = plt.subplots(2, 2, figsize=(9.8, 9.0))
for ci, (rk, cname) in enumerate([(R0, 'Δm=0（文档公式通道）'),
                                  (R1, 'Δm=+1')]):
    for ti, tilt in enumerate((0.0, 90.0)):
        ax = axes[ci, ti]
        pos, u, v = lattice(5, tilt)
        m = ManyBodyModel(terms=('I2',), n_window=1)
        for p in pos:
            m.add_atom(G, rk, p)
        cal = asm.calibrate(cache, m, second_order=False)
        P2d = np.array([[np.dot(p, u), np.dot(p, v)] for p in pos])
        jmax = max(abs(p['raw_J'].real) for p in cal['pairs'])
        for p in cal['pairs']:
            R = np.linalg.norm(pos[p['i']] - pos[p['j']])
            if R > 1.6 * A_UM:
                continue
            c = 'tab:red' if p['raw_J'].real > 0 else 'tab:blue'
            lw = 0.8 + 3.5 * abs(p['raw_J'].real) / jmax
            ax.plot([P2d[p['i'], 0], P2d[p['j'], 0]],
                    [P2d[p['i'], 1], P2d[p['j'], 1]], '-', color=c, lw=lw,
                    alpha=0.85, zorder=1)
        ax.plot(P2d[:, 0], P2d[:, 1], 'o', ms=6, color='k', zorder=2)
        nn = [p for p in cal['pairs']
              if abs(np.linalg.norm(pos[p['i']] - pos[p['j']]) - A_UM) < 1e-6]
        npos = sum(p['raw_J'].real > 0 for p in nn)
        state = ('均匀 AFM → Néel 成立（线性模）' if npos == len(nn) else
                 '均匀 FM（J<0）' if npos == 0 else '正负混合 → 阻挫!')
        ax.set_title(f'{cname} · 场与面法线夹角 tilt={tilt:.0f}°\n'
                     f'NN：{npos} 正(J>0,AFM) / {len(nn)-npos} 负(J<0,FM)'
                     f' → {state}', fontsize=9)
        ax.set_aspect('equal')
        ax.axis('off')
fig.suptitle('图 B：键符号（红 J>0=AFM，蓝 J<0=FM，线粗 ∝|J|；约定 '
             r'H=$\sum$J(S$^+$S$^-$+h.c.)）\n干净 Néel = Δm=0 + 场⊥面（tilt=0）；'
             'tilt=90 两通道都阻挫 —— 文档 A.0 的 FM/AFM 标签与其 A.2 '
             '路径积分自洽约定相反，2D 推荐方向应互换', fontsize=10)
fig.tight_layout()
fig.savefig(os.path.join(OUT, 'figB_bond_sign_maps.png'))
plt.close(fig)
say('figB 完成')

# =====================================================================
# C. J̃(k) 色图（无限格点 FFT 求和；序波矢 = argmin）
# =====================================================================
say('figC: J̃(k) 色图…')
Ng = 1024
gi = np.fft.fftfreq(Ng, d=1.0 / Ng)
GX, GZ = np.meshgrid(gi, gi, indexing='xy')
RR = np.hypot(GX, GZ) * A_UM
MZ_ = (RR > 1e-9) & (RR <= (Ng // 2 - 2) * A_UM)
kx = np.fft.fftshift(np.fft.fftfreq(Ng, d=A_UM)) * 2 * np.pi


def jk_of(Jg):
    return np.fft.fftshift(np.fft.fft2(Jg)).real


# Δm=0, tilt=0：面=x–y，所有键 ⊥z → J = +C₃/R³
Jg1 = np.zeros_like(RR)
Jg1[MZ_] = C3_0 / RR[MZ_] ** 3
# Δm=0, tilt=90：面=x–z，cosθ = n_z → J = C₃(1−3n_z²)/R³
Jg2 = np.zeros_like(RR)
Jg2[MZ_] = C3_0 * (1 - 3 * (GZ[MZ_] * A_UM / RR[MZ_]) ** 2) / RR[MZ_] ** 3

fig, axes = plt.subplots(1, 2, figsize=(11.6, 5.1))
for ax, (Jk, ttl, ylab) in zip(
        axes,
        [(jk_of(Jg1), 'Δm=0 · tilt=0（场⊥面）\nargmin 在 (±π,±π) → Néel 序成立',
          r'$k_y a$'),
         (jk_of(Jg2), 'Δm=0 · tilt=90（场∥面）\n键符号混合 → 序波矢移向有限 Q（阻挫候选）',
          r'$k_z a$')]):
    diff = Jk - Jk.min()
    imin = np.unravel_index(np.argmin(Jk), Jk.shape)
    im = ax.pcolormesh(kx * A_UM, kx * A_UM, diff, cmap='magma',
                       shading='auto')
    ax.set_xlabel(r'$k_x a$'); ax.set_ylabel(ylab)
    ax.set_title(ttl, fontsize=9.5)
    for s in (1, -1):
        ax.plot(s * np.pi, kx[imin[0]] * A_UM, 'c+', ms=13, mew=2)
        ax.plot(kx[imin[1]] * A_UM, s * np.pi, 'c+', ms=13, mew=2)
    ax.plot(kx[imin[1]] * A_UM, kx[imin[0]] * A_UM, 'c+', ms=16, mew=2.5)
    say(f'  {ttl.splitlines()[0]}: argmin=({kx[imin[1]]*A_UM/np.pi:+.2f},'
        f'{kx[imin[0]]*A_UM/np.pi:+.2f})·π')
    fig.colorbar(im, ax=ax, shrink=0.85,
                 label=r'$\tilde J(k)-\min$ [MHz]')
fig.suptitle(r'图 C：$\tilde J(k)$（MHz，无限格点求和，$C_3$ 来自引擎标定）——'
             '硬核玻色序波矢 = argmin $\tilde J$（青色 +）；\n'
             '沿不同轴等值线形状不同 = 刚度张量各向异性 = 各向异性"光速"'
             '（判据③）', fontsize=10.5)
fig.tight_layout()
fig.savefig(os.path.join(OUT, 'figC_Jk_map.png'))
plt.close(fig)
say('figC 完成')

# =====================================================================
# D. 软化指纹：FM@Γ vs AFM@Q*（直接格点求和）
# =====================================================================
say('figD: 小-q 标度…')
# 系数 = 引擎直接标定值（tilt=0 面内所有键皆 ⊥z ⇒ 均匀 K/R³）
K_p1 = calib_JR(R1, A_UM, 90.0) * A_UM ** 3      # Δm=+1 @θ=90：<0 ⇒ tilt=0 FM
K_0 = C3_0                                        # Δm=0  @θ=90：>0 ⇒ tilt=0 AFM
say(f'  K(Δm=+1,θ=90)={K_p1:.4g}（FM 侧）  K(Δm=0,θ=90)={K_0:.4g}（AFM 侧）')
M = 1200
mm = np.arange(-M, M + 1)
MX, MZ = np.meshgrid(mm, mm, indexing='xy')
Rm = np.hypot(MX, MZ) * A_UM
okm = Rm > 1e-9
inv3 = np.zeros_like(Rm)
inv3[okm] = 1.0 / Rm[okm] ** 3
par = (-1.0) ** (MX + MZ)


def soft_fm(q):
    """FM@Γ（Δm=+1, tilt=0，J=K_p1/R³<0）：ω=J̃(q)−J̃(0)
    = −K_p1·Σ(1−cos q·R)/R³ ∝ |q|（2D 偶极 1/R³ 非解析 ⇒ 磁振子抬成线性，
    文档表"FM→k²"只对短程成立）。"""
    return -K_p1 * np.sum((1.0 - np.cos(q[0] * MX * A_UM
                                        + q[1] * MZ * A_UM))
                          * inv3 * okm)


def soft_afm(q):
    """AFM@Q*=(π,π)（Δm=0, tilt=0，J=K_0/R³>0）：J̃(Q*+q)−J̃(Q*)
    = −K_0·Σ(−1)^{m+n}(1−cos q·R)/R³ —— 交错 ⇒ 绝对收敛 ⇒ 解析 ∝q²。"""
    return -K_0 * np.sum(par * (1.0 - np.cos(q[0] * MX * A_UM
                                             + q[1] * MZ * A_UM))
                         * inv3 * okm)


ks = np.logspace(-2.0, np.log10(0.9), 16)
fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.9))
for ax, fn, ttl in [
        (axes[0], soft_fm,
         'FM @ Γ（Δm=+1, tilt=0）\nα≈1：长程 1/R³ 把 FM 磁振子抬成 |q| 线性\n'
         '（文档表"FM→k²"只对短程成立）'),
        (axes[1], soft_afm,
         'AFM @ Q*=(π,π)（Δm=0, tilt=0）\nα≈2：交错因子使求和解析\n'
         '故 |q| 软化不在 Q*（文档 §4.2 需修正）')]:
    for (dxy, lab) in [((1, 0), r'$q\parallel x$'),
                       ((1, 1), r'$q\parallel[11]$')]:
        d = np.array(dxy, float) / np.linalg.norm(dxy)
        vals = np.maximum(np.abs([fn(q * d / A_UM) for q in ks]), 1e-30)
        ax.loglog(ks, vals, 'o-', ms=4, lw=1.2, label=lab)
        win = (ks > 0.015) & (ks < 0.5)
        a = np.polyfit(np.log(ks[win]), np.log(vals[win]), 1)[0]
        ax.text(0.016, vals[4] * (1.7 if dxy[1] == 0 else 0.30),
                f'α≈{a:.2f}', fontsize=10)
    mid = ks[len(ks) // 2]
    vmid = np.abs(fn(mid * np.array([1.0, 0]) / A_UM))
    ax.loglog(ks, vmid / mid * ks, ':', lw=0.7, alpha=0.5, color='k',
              label=r'$|q|$ 参考')
    ax.loglog(ks, vmid / mid ** 2 * ks ** 2, ':', lw=0.7, alpha=0.5,
              color='gray', label=r'$q^2$ 参考')
    ax.set_xlabel(r'$|q|\cdot a$')
    ax.set_ylabel(r'$|\tilde J(Q)-\tilde J(Q+q)|$ [MHz]')
    ax.set_title(ttl, fontsize=9)
    ax.legend(fontsize=8); ax.grid(alpha=0.25, which='both')
fig.suptitle('图 D：判据④正面检验 —— |q| 非解析软化是 **Γ（FM/超流侧）** '
             '现象；AFM 的 Q* 处修正是 q² 型', fontsize=11)
fig.tight_layout()
fig.savefig(os.path.join(OUT, 'figD_softening.png'))
plt.close(fig)
say('figD 完成')

# =====================================================================
# E. 有限尺寸 vs 无限（同一 Δm=+1 模型：截断到 12×12 vs 求到 M）
# =====================================================================
say('figE: 有限尺寸对照（同模型截断 vs 无限）…')


def soft_t90(qx, Mcut=None):
    """tilt=90（面=x–z）沿 k_x：J̃(0)−J̃(q) = −2K_p1·Σ P₂(n_z)(1−cos)/R³。
    Mcut≠None 时把求和限制在 |m|,|n|≤Mcut（模拟 L=2·Mcut 有限团簇）。"""
    with np.errstate(divide='ignore', invalid='ignore'):
        pz = np.where(Rm > 0, MZ * A_UM / Rm, 0.0)
    sel = okm if Mcut is None else (okm & (np.abs(MX) <= Mcut)
                                    & (np.abs(MZ) <= Mcut))
    return -2.0 * K_p1 * np.sum(P2(pz) * (1.0 - np.cos(qx * MX * A_UM))
                                * inv3 * sel)


ks = np.logspace(-2.0, np.log10(0.9), 16)
inf90 = np.maximum(np.abs([soft_t90(k / A_UM) for k in ks]), 1e-30)
# L=12 团簇：半宽 6 格；可分辨最小 k·a=2π/12≈0.52
fin90 = np.maximum(np.abs([soft_t90(k / A_UM, Mcut=6) for k in ks]), 1e-30)
win_f = (ks > 0.4) & (ks < 0.95)
a_f = np.polyfit(np.log(ks[win_f]), np.log(fin90[win_f]), 1)[0]
a_i = np.polyfit(np.log(ks[(ks > 0.015) & (ks < 0.5)]),
                 np.log(inf90[(ks > 0.015) & (ks < 0.5)]), 1)[0]

fig, ax = plt.subplots(figsize=(6.6, 4.8))
ax.loglog(ks, inf90, color='tab:blue', lw=1.6,
          label=rf'无限格点（Δm=+1, tilt=90）→ α≈{a_i:.2f}')
ax.loglog(ks, fin90, 'D-', ms=5, lw=1.2, color='tab:orange',
          label=rf'12×12 截断求和 → 窗口内 α≈{a_f:.2f}')
mid = ks[len(ks) // 2]
ax.loglog(ks, inf90[len(ks) // 2] / mid * ks, ':', lw=0.7, alpha=0.5,
          color='k', label=r'$|q|$ 参考')
ax.loglog(ks, inf90[len(ks) // 2] / mid ** 2 * ks ** 2, ':', lw=0.8,
          alpha=0.6, color='gray', label=r'$q^2$ 参考')
ax.axvspan(0.5, 0.9, color='orange', alpha=0.13)
ax.text(0.055, 3e-3, '有限团簇可分辨窗口\n(最小 k·a=2π/12≈0.52)\n'
        '截断和未收敛：整体抬高、斜率畸变', fontsize=8, color='darkorange')
ax.set_xlabel(r'$|q|\cdot a$')
ax.set_ylabel(r'$|\tilde J(0)-\tilde J(q)|$ [MHz]')
ax.set_title('图 E：有限尺寸为何测不到真 |q| —— 12×12 截断和在小-k 抬高且畸变\n'
             '（真 |q| 只在无限求和显现；上一轮引擎标定 12×12 求和 α≈1.85 '
             '即此类未收敛伪影）', fontsize=10.5)
ax.legend(fontsize=8); ax.grid(alpha=0.25, which='both')
fig.tight_layout()
fig.savefig(os.path.join(OUT, 'figE_finite_size.png'))
plt.close(fig)

np.savez(os.path.join(OUT, 'photon_data.npz'),
         th=th, J_p1=J_p1, J_0=J_0, C3_0=C3_0, Rs=Rs, Js0=Js0, ks=ks,
         inf90=inf90, fin90=fin90)
say(f'全部完成 → {OUT}')
