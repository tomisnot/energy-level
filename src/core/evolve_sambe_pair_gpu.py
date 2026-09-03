# -*- coding: utf-8 -*-
"""GPU 传播模块（t57 第 2 版）：与 CPU frozen 逐位对齐的 CUDA 路径。

CPU frozen（引擎 _pair_propagate）按**输出网格区间**分段 + eigh 复用缓存
（包络值键相同共享分解）+ _physical_pops 相干投影（nphase）。本版 GPU 主循环
镜像该语义：
- 逐输出区间 [t_i, t_i+1]（nsub 子步，与 CPU 相同）
- 包络值缓存键 → const 相邻区间共享 eigh（引擎级 const 合并同款）
- 输出用 _physical_pops 同款 nphase 相干投影
→ 与 CPU 同一冻结点/同一 eigh 复用/同一投影 → 逐位一致（≤1e-10 目标）
"""
import numpy as np
import torch

_CONV = 2 * np.pi * 1e-3


def gpu_propagate_pair(setup, groups, psi0_2d, t_grid, frozen_h_ns=8.0,
                       log=None):
    """GPU 全驻留传播（镜像 CPU frozen 语义）。返回 pops_2d (len(t), N²)。"""
    log = log or (lambda *a: None)
    N2 = setup.N2
    try:
        torch.backends.cuda.preferred_linalg_library('cusolver')
    except Exception:
        pass
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

    def env_vec(tv):
        return np.array([float(np.asarray(wf(np.atleast_1d(tv))).flat[0])
                         for wf, _ in envs_all])

    from .evolve_sambe_pair import _embed_pair
    X0_ext = _embed_pair(psi0_2d, setup)
    X = torch.from_numpy(np.ascontiguousarray(X0_ext)) \
        .to(torch.complex128).cuda()
    pops = np.zeros((len(t_grid), N2))
    Xm0 = X0_ext.reshape(setup.P, N2)
    pops[0] = np.abs(Xm0.conj().T @ Xm0).diagonal().real
    n_eigh = 0
    n_prop = 0
    _key = [None]
    _val = [None]        # (w, v)

    def seg_prop(ta, tb, Xc):
        nonlocal n_eigh, n_prop
        nsub = max(1, int(np.ceil((tb - ta) / float(frozen_h_ns))))
        hh = (tb - ta) / nsub
        for _s in range(nsub):
            t1 = ta + _s * hh
            mid = t1 + 0.5 * hh
            envs_key = tuple(
                round(float(np.asarray(wf(np.atleast_1d(mid))).flat[0]), 12)
                for wf, _ in envs_all)
            if _key[0] == envs_key and _val[0] is not None:
                w, v = _val[0]
            else:
                K = K0_g.clone()
                for (wf, Kg), c in zip(envs_all, env_vec(mid)):
                    if c != 0.0:
                        K = K + c * Kg
                if K.is_complex() and torch.any(torch.imag(K) != 0):
                    w, v = torch.linalg.eigh(K)       # 复厄米
                else:
                    w, v = torch.linalg.eigh(K.real)  # 实对称
                n_eigh += 1
                _key[0] = envs_key
                _val[0] = (w, v)
            vc = v.to(torch.complex128)
            a0 = vc.t() @ Xc
            Xc = vc @ (a0 * torch.exp(-1j * _CONV * w * hh)
                       .to(torch.complex128))
            n_prop += 1
        return Xc

    # 主循环：逐输出区间（镜像 CPU frozen）
    Xc = X
    tprev = 0.0
    for i in range(1, len(t_grid)):
        tb = float(t_grid[i])
        Xc = seg_prop(tprev, tb, Xc)
        # 输出：_physical_pops 同款 nphase 相干投影
        Xm = Xc.reshape(setup.P, N2)
        ph = np.exp(-1j * setup.nphase * float(tb))
        psi = ph @ Xm.cpu().numpy()
        pops[i] = np.abs(psi) ** 2
        tprev = tb
    torch.cuda.synchronize()
    log(f'[sambe_pair-gpu] 完成：{n_eigh} eigh / {n_prop} 传播步，'
        f'峰值显存 {torch.cuda.max_memory_allocated()/1e9:.2f}GB')
    return pops, n_eigh
