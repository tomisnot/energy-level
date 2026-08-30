# -*- coding: utf-8 -*-
"""Sambe 光子块核心层（与态空间维度 N 无关）——重构第 2b 步。

从 _SambeSetup（evolve_exact.py）/ _PairSambeSetup（evolve_exact_pair.py）
抽取共享的"光子块几何构造"：混合进制 stride / 量子数 / 块对角 / 重构相位 /
中央光子块。两类 __init__ 的这段逻辑逐行相同（仅 E 维度 N 与 N² 不同），
两体版额外加 V_dd/Γ 对角（在调度层处理，不在此核心）。

photon_geometry(E, groups, Ms)：
  E      : 任意长度 N 的能量数组（单体 E_mhz / 两体 E_pair）
  groups : [dict(omega_mhz, As|K2_w, envs|envs_w)] 载频组
  Ms     : int（统一 M）或 list（逐载频 M）
  返回   : dict(P, dim, strides, nvecs, diag, nphase, p_center)

⚠️ 教训（2026-08-30）：核心层只做"逐行复制已验证代码"，禁止凭记忆重写
（手写 _D 系数错 54 处的教训）。任何数值改动都必须对照 EP/selfcheck 基线。
"""
import numpy as np

_CONV = 2 * np.pi * 1e-3  # rad/ns per MHz


def photon_geometry(E, groups, Ms):
    """Sambe 光子块几何（任意 N，N 无关）。逐行对应 _SambeSetup.__init__
    的 172-206 行与 _PairSambeSetup.__init__ 的 111-126/152-156 行。"""
    E = np.asarray(E, float)
    N = E.size
    G = len(groups)
    if isinstance(Ms, (int, np.integer)):
        Ms = [int(Ms)] * G
    else:
        Ms = [int(m) for m in Ms]
        if len(Ms) != G:
            raise ValueError(f'M 长度 {len(Ms)} != 载频组数 {G}')
    Ds = [2 * m + 1 for m in Ms]                    # 每组块数
    P = 1
    for d in Ds:
        P *= d
    idx = np.arange(P)
    strides = []
    nvecs = []
    base = np.zeros(P)
    stride = 1
    for gi, g in enumerate(groups):
        Dg = Ds[gi]
        Mg = Ms[gi]
        nvec = (idx // stride) % Dg
        strides.append(stride)
        nvecs.append(nvec)
        # Sambe 块对角 = E_i − (n_g−M_g)·ω_g（与耦合方向 n→n+1 配套）
        base = base - (nvec - Mg) * g['omega_mhz']
        stride *= Dg
    diag = np.tile(E, P) + np.repeat(base, N)
    # 重构相位系数：θ_p(t) = nphase[p]·t（rad，t 单位 ns）
    nphase = np.zeros(P)
    for nvec, g in zip(nvecs, groups):
        nphase = nphase + nvec * (_CONV * g['omega_mhz'])
    # 中央光子块（各组量子数 n=0，即中心化索引 M_g）：物理态嵌入位置
    p_center = sum(Mg * st for Mg, st in zip(Ms, strides))
    return dict(P=P, dim=N * P, strides=strides, nvecs=nvecs,
                diag=diag, nphase=nphase, p_center=p_center, Ms=Ms,
                Ds=Ds)
