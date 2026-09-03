# -*- coding: utf-8 -*-
"""GPU 传播模块（t57）：全频段两体 Sambe 的 CUDA 路径——torch 全驻留 + const 段合并。

移植自研究原型 tempsimu/_t50_full.py（GPU≡CPU 8.57e-11 验证、620ns ~145s/46 eigh、
峰值显存 2.14GB<6.4）。引擎入口 evolve_sambe_pair(device='cuda') 调用本模块。

算法：
1. K0/Kcw 预传 GPU 驻留（float64；无耗散实对称 → 实 eigh；有耗散 → 复厄米）
2. 块划分：活动波模式切换点（包络 >1e-9）分块
3. 块内 const 检测（包络变化 < 1e-9 相对）→ 单 eigh + 输出点向量化传播
4. 变区 → h 子步逐段 eigh（frozen 语义）
"""
import numpy as np
import torch

_CONV = 2 * np.pi * 1e-3


def gpu_propagate_pair(setup, groups, psi0_2d, t_grid, frozen_h_ns=8.0,
                       log=None):
    """GPU 全驻留传播。setup = _PairSambeSetup（稠密）；返回 pops_2d (len(t), N²)。"""
    log = log or (lambda *a: None)
    T = float(t_grid[-1])
    N = setup.N
    N2 = setup.N2
    dev = 'cuda'
    try:
        torch.backends.cuda.preferred_linalg_library('cusolver')
    except Exception:
        pass
    # 稠密驻留（无耗散 → 实 float64；有耗散 → 复）
    has_imag = np.iscomplexobj(setup.K_static) and \
        np.any(np.imag(setup.K_static) != 0)
    if not has_imag:
        K0_g = torch.from_numpy(
            np.ascontiguousarray(np.real(setup.K_static))).cuda()
    else:
        K0_g = torch.from_numpy(
            np.ascontiguousarray(setup.K_static)).cuda()
    envs_all = []
    for gi, g in enumerate(groups):
        for wf, Kcw in zip(g['envs_w'], setup.Kc_dense[gi]):
            Kcw_np = np.asarray(Kcw)
            if np.iscomplexobj(Kcw_np) and np.any(np.imag(Kcw_np) != 0):
                envs_all.append((wf, torch.from_numpy(
                    np.ascontiguousarray(Kcw_np)).cuda()))
            else:
                envs_all.append((wf, torch.from_numpy(
                    np.ascontiguousarray(np.real(Kcw_np))).cuda()))
    n_w = len(envs_all)
    log(f'[sambe_pair-gpu] 驻留 K0+{n_w} Kcw 完成（dim={setup.dim}）')


    def env_vec(tv):
        return np.array([float(np.asarray(wf(np.atleast_1d(tv))).flat[0])
                         for wf, _ in envs_all])

    # ---- 块划分：活动波模式切换 ----
    ts_c = np.arange(0.0, T + 0.01, 0.2)
    Et = np.stack([env_vec(tv) for tv in ts_c])
    active = np.abs(Et) > 1e-9
    switch = [0.0]
    for i in range(1, len(ts_c)):
        if not np.array_equal(active[i], active[i - 1]):
            switch.append(float(ts_c[i]))
    switch.append(T)
    blks = [(switch[k], switch[k + 1]) for k in range(len(switch) - 1)]
    # 合并相邻窄块（<0.4ns）
    blks2 = []
    for b in blks:
        if blks2 and b[0] - blks2[-1][1] < 0.4:
            blks2[-1] = (blks2[-1][0], b[1])
        else:
            blks2.append(b)
    blks = blks2

    # 嵌入扩展空间（psi0_2d 是物理态 N²）
    from .evolve_sambe_pair import _embed_pair
    X0_ext = _embed_pair(psi0_2d, setup)
    X = torch.from_numpy(np.ascontiguousarray(X0_ext)) \
        .to(torch.complex128).cuda()
    pops = np.zeros((len(t_grid), N2))
    Xm0 = X0_ext.reshape(setup.P, N2)
    pops[0] = np.abs(Xm0.conj().T @ Xm0).diagonal().real
    n_eigh = 0
    t0_all = None
    for (b0, b1) in blks:
        e0 = env_vec(b0)
        e1 = env_vec(b1)
        const = np.allclose(e0, e1, atol=1e-9, rtol=1e-9)
        idx = np.where((t_grid >= b0 - 1e-9) & (t_grid <= b1 + 1e-9))[0]
        idx = idx[idx > 0]
        if const:
            mid = 0.5 * (b0 + b1)
            K = K0_g.clone()
            for (wf, Kg), c in zip(envs_all, env_vec(mid)):
                if c != 0.0:
                    K = K + c * Kg
            w, v = torch.linalg.eigh(K)
            n_eigh += 1
            vc = v.to(torch.complex128)
            a0 = vc.t() @ X
            if len(idx):
                ts_b = torch.from_numpy(t_grid[idx] - b0).to(torch.float64) \
                    .cuda()
                Aph = torch.exp(-1j * _CONV * w[:, None] * ts_b[None, :])
                Xall = vc @ (a0[:, None] * Aph.to(torch.complex128))
                for ki, ii in enumerate(idx):
                    Xm = Xall[:, ki].reshape(setup.P, N2)
                    pops[ii] = torch.abs(Xm.conj().T @ Xm).diagonal() \
                        .real.cpu().numpy()
            hh = b1 - b0
            X = vc @ (a0 * torch.exp(-1j * _CONV * w * hh)
                      .to(torch.complex128))
        else:
            tb = b0
            while tb < b1 - 1e-9:
                hh = min(frozen_h_ns, b1 - tb)
                mid = tb + 0.5 * hh
                K = K0_g.clone()
                for (wf, Kg), c in zip(envs_all, env_vec(mid)):
                    if c != 0.0:
                        K = K + c * Kg
                w, v = torch.linalg.eigh(K)
                n_eigh += 1
                vc = v.to(torch.complex128)
                a0 = vc.t() @ X
                # 子步内输出点（船长 t57 修复）：内部时刻用 (w,v) 从子步起点
                # 精确传播——eigh 一次可任意时刻求值，零额外成本
                for ii in idx:
                    if tb - 1e-9 <= t_grid[ii] <= tb + hh + 1e-9:
                        dt_i = t_grid[ii] - tb
                        if dt_i < 1e-12:
                            Xm = X.reshape(setup.P, N2)
                        else:
                            X_i = vc @ (a0 * torch.exp(
                                -1j * _CONV * w * dt_i)
                                .to(torch.complex128))
                            Xm = X_i.reshape(setup.P, N2)
                        pops[ii] = torch.abs(Xm.conj().T @ Xm).diagonal() \
                            .real.cpu().numpy()
                X = vc @ (a0 * torch.exp(-1j * _CONV * w * hh)
                          .to(torch.complex128))
                tb = tb + hh
        log(f'[sambe_pair-gpu] 块 [{b0:.0f},{b1:.0f}] 完成'
            f'（eigh {n_eigh}）')
    torch.cuda.synchronize()
    log(f'[sambe_pair-gpu] 完成：{n_eigh} eigh，'
        f'峰值显存 {torch.cuda.max_memory_allocated()/1e9:.2f}GB')
    return pops, n_eigh
