# -*- coding: utf-8 -*-
"""project.py（t63 第 1 层统一）：跨扇区 nphase 投影公共层——单/两体共用。

从 evolve_sambe._reconstruct / evolve_sambe_pair._physical_pops 抽出同一数学：
物理振幅 ψ_i(t) = Σ_n e^{−iC·n·ω·t}·X[n·N+i]（X 为扩展态/φ 已转回实验室系）——
参数化 N（单原子）/ N²（两体）、P、扇区光子数 ns、ω。

- project_pops(X_t, P, N_phys, ns, omega, t)：跨扇区投影 → 物理布居
  （len(N_phys)）。X_t 为 φ→X 转换后的扩展态（调用侧先乘 e^{−iC·D·t}）。
"""
import numpy as np

_C = 2 * np.pi * 1e-3


def project_psi(X_t, P, N_phys, ns, omega, t):
    """跨扇区相干投影：ψ = Σ_n e^{−iC·n·ω·t}·X[n 扇区]。

    X_t: 扩展态（(P·N_phys,)——φ 已转回实验室系 e^{−iC·D·t}·φ）；
    ns: 扇区光子数数组 (P,)；omega: 载频 MHz。
    返回物理振幅 ψ (N_phys,)。
    """
    psi = np.zeros(N_phys, complex)
    for pn in range(P):
        ph = np.exp(-1j * _C * float(ns[pn]) * float(omega) * float(t))
        psi = psi + ph * X_t[pn * N_phys: (pn + 1) * N_phys]
    return psi


def project_pops(X_t, P, N_phys, ns, omega, t):
    """跨扇区投影布居：|ψ|²。"""
    psi = project_psi(X_t, P, N_phys, ns, omega, t)
    return np.abs(psi) ** 2
