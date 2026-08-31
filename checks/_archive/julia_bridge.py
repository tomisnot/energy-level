# 已退役，仅参考，2026-08-31（t26：Julia 后端全面退役归档；正式项目不再引用本文件）。
# 原位置：checks/ 根目录 julia_bridge.py

# -*- coding: utf-8 -*-
"""Julia 内核桥接设施（_julia_bridge.py）。

Python(ARC/GUI/参考) ⊕ Julia(计算内核) 的子进程桥：
  - 定位 Julia 可执行文件（JULIA_EXE > PATH > D:\\julia-1.10.10\\bin）
  - 统一子进程环境（JULIA_DEPOT_PATH 落 D 盘、清华 Pkg 镜像）
  - 请求/响应 NPZ 读写（schema v1，见 julia/src/io.jl 头注释）
  - 运行 CLI 并解析 PROG 进度行

schema 约定（与 julia/src/io.jl 严格同步，改动必须双侧同步）：
  请求 req.npz：
    jl_version=1, N, E_pair(N²), vdd_row/col(int64, 0-based COO),
    vdd_data(complex128 MHz), n_groups, group_omega(K),
    k2r_{g}/k2c_{g}(int64)/k2v_{g}(float64 MHz), n_fields, f_group,
    f_kind(0=gauss 1=square 2=sampled), f_p1..f_p4, f_samp_t_{i}/f_samp_v_{i},
    psi0(2, 0-based), t_grid(ns), tol, gamma_pair(N², /ns, 空则无耗散),
    m_max, dense_max, node_ladder, dt_min_frac
"""
import os
import subprocess
import sys

import numpy as np

JL_VERSION = 1
_HERE = os.path.dirname(os.path.abspath(__file__))
_JULIA_DIR = os.path.normpath(os.path.join(_HERE, '..', 'julia'))


def find_julia():
    """定位 julia.exe：JULIA_EXE 环境变量 > PATH > D:\\julia-1.10.10。"""
    exe = os.environ.get('JULIA_EXE')
    if exe and os.path.isfile(exe):
        return exe
    from shutil import which
    hit = which('julia')
    if hit:
        return hit
    for d in ('D:\\julia-1.10.10\\bin', 'D:\\julia-1.10\\bin'):
        cand = os.path.join(d, 'julia.exe')
        if os.path.isfile(cand):
            return cand
    return None


def julia_env():
    """子进程环境：depot 落 D 盘 + 清华镜像（未显式设置时注入）。"""
    env = dict(os.environ)
    env.setdefault('JULIA_DEPOT_PATH', 'D:\\julia\\depot')
    env.setdefault('JULIA_PKG_SERVER',
                   'https://mirrors.tuna.tsinghua.edu.cn/julia')
    env.setdefault('JULIA_NUM_THREADS', 'auto')
    return env


def run_cli(req_path, resp_path, echo=False, cancel_file=None, timeout=600):
    """运行 CLI；返回 (returncode, stdout_text)。"""
    exe = find_julia()
    if exe is None:
        raise RuntimeError('未找到 julia.exe（安装于 D:\\julia-1.10.10）')
    cmd = [exe, '--project=' + _JULIA_DIR,
           os.path.join(_JULIA_DIR, 'bin', 'evolve_pair_cli.jl'),
           req_path, resp_path]
    if echo:
        cmd.append('--echo')
    if cancel_file:
        cmd += ['--cancel-file', cancel_file]
    p = subprocess.run(cmd, capture_output=True, text=True,
                       encoding='utf-8', errors='replace', env=julia_env(),
                       timeout=timeout, cwd=_JULIA_DIR)
    return p.returncode, (p.stdout or '') + (p.stderr or '')


# ---------------- 请求构建（schema v1 写出） ----------------

def _coo(m):
    """scipy 稀疏 → (row, col, data) 0-based COO。"""
    import scipy.sparse as sp
    if m is None:
        return (np.zeros(0, np.int64), np.zeros(0, np.int64),
                np.zeros(0, np.float64))
    coo = sp.coo_matrix(m)
    return (coo.row.astype(np.int64), coo.col.astype(np.int64),
            coo.data)


def env_arrays(envelope, offset_ns=0.0):
    """包络对象 → (kind, p1..p4, samp_t, samp_v)，参数做绝对时间平移。"""
    from src.core.waveforms import GaussianEnv, SquareEnv
    if isinstance(envelope, GaussianEnv):
        return (0, (envelope.center_ns + offset_ns, envelope.sigma_ns,
                    envelope.peak, 0.0), None, None)
    if isinstance(envelope, SquareEnv):
        fall = (envelope.fall_start_ns
                if envelope.fall_start_ns is not None
                else envelope.start_ns + 2 * envelope.edge_ns)
        return (1, (envelope.start_ns + offset_ns, envelope.edge_ns,
                    envelope.height, fall + offset_ns), None, None)
    raise NotImplementedError(
        '采样包络（f_kind=2）导出在 S5.5 接入；当前支持高斯/方波')


def write_request(path, N, E_pair, vdd=None, groups=None, fields=None,
                  psi0=(0, 0), t_grid=None, tol=1e-6, gamma_A=None,
                  gamma_B=None, m_max=8, dense_max=700,
                  node_ladder=(16, 32, 64, 128), dt_min_frac=2e5,
                  record_amp=False, sparse_backend=0,
                  mF_states=None, charges=None, block_mf=False,
                  skip_probe=False):
    """写出请求 NPZ。

    groups: [(omega_mhz, K2_coo_or_sparse), ...] 相异载频组
    fields: [(group_idx, kind, p1..p4, samp_t|None, samp_v|None), ...]
    gamma_A/gamma_B: N² 寿命速率 /ns（None 或空 = 无耗散；须同时给）
    """
    d = {
        'jl_version': np.array(JL_VERSION),
        'N': np.array(int(N), np.int64),
        'E_pair': np.asarray(E_pair, np.float64),
        't_grid': np.asarray(t_grid, np.float64),
        'tol': np.array(float(tol)),
        'm_max': np.array(int(m_max), np.int64),
        'dense_max': np.array(int(dense_max), np.int64),
        'node_ladder': np.asarray(node_ladder, np.int64),
        'dt_min_frac': np.array(float(dt_min_frac)),
        'record_amp': np.array(int(bool(record_amp)), np.int64),
        'sparse_backend': np.array(int(sparse_backend), np.int64),
        'block_mF': np.array(int(bool(block_mf)), np.int64),
        'skip_probe': np.array(int(bool(skip_probe)), np.int64),
    }
    vr, vc, vv = _coo(vdd)
    if len(vr):                      # 空数组不写键（NPZ.jl/ZipFile 对 0 字节成员 EOF）
        d['vdd_row'] = vr
        d['vdd_col'] = vc
        d['vdd_data'] = vv.astype(np.complex128)
    groups = groups or []
    if groups:
        d['n_groups'] = np.array(len(groups), np.int64)
        d['group_omega'] = np.array([float(g[0]) for g in groups],
                                    np.float64)
    for gi, (_, k2) in enumerate(groups):
        r, c, v = _coo(k2)
        if len(r):
            d[f'k2r_{gi}'] = r
            d[f'k2c_{gi}'] = c
            d[f'k2v_{gi}'] = v.astype(np.float64)
    fields = fields or []
    if fields:
        d['n_fields'] = np.array(len(fields), np.int64)
        d['f_group'] = np.array([f[0] for f in fields], np.int64)
        d['f_kind'] = np.array([f[1] for f in fields], np.int64)
        for i in range(1, 5):
            d[f'f_p{i}'] = np.array(
                [float(f[2][i - 1]) for f in fields], np.float64)
        for i, f in enumerate(fields):
            st, sv = f[3], f[4]
            if st is not None:
                d[f'f_samp_t_{i}'] = np.asarray(st, np.float64)
                d[f'f_samp_v_{i}'] = np.asarray(sv, np.float64)
    if gamma_A is not None:
        d['gamma_A'] = np.asarray(gamma_A, np.float64)
        d['gamma_B'] = np.asarray(gamma_B, np.float64)
    if mF_states is not None:
        d['mF_states'] = np.asarray(mF_states, np.float64)
        d['charges'] = np.asarray(charges, np.float64)
    # 初态：((i,j) 索引对) 或 ((vecA, vecB) 复矢量直积)
    if len(psi0) == 2 and np.isscalar(psi0[0]):
        d['psi0'] = np.asarray(psi0, np.int64)
    else:
        d['psi0_A'] = np.asarray(psi0[0], np.complex128)
        d['psi0_B'] = np.asarray(psi0[1], np.complex128)
    np.savez(path, **d)


# ============================================================
# S5.5：任意波形 → 主音分解（FFT 谱分解 → multi-tone exact）
# ============================================================

def decompose_tones(t_ns, amp, max_tones=6, peak_frac=1e-3,
                    env_points=2001, min_tone_frac=0.05):
    """任意波形采样 → 稀疏多音分解。

    amp(t) ≈ Σ_k env_k(t)·cos(2π·f_k·t)（env 带符号；载频相位限 0/π）。
    步骤：rfft（t 以 ns 计 → 频率单位 GHz）→ 峰检测（局部极大 >
    peak_frac·max）→ 质心细化载频 → 带通（共轭对称）→ 解析信号解调
    得慢变复包络 → 实部为带符号包络。
    残差（max|amp−recon|/max|amp|）> 2e-2 或音相位非 0/π → raise（指引 ip4）。

    返回 [(freq_mhz, env_t, env_v), ...]（采样包络，供 f_kind=2）, resid。
    """
    t = np.asarray(t_ns, float)
    a = np.asarray(amp, float)
    n = len(t)
    if n < 16:
        raise ValueError('任意波形采样点过少（≥16）')
    dt = float(t[1] - t[0])
    if np.max(np.abs(np.diff(t)) - dt) > 1e-9 * max(abs(dt), 1.0):
        raise ValueError('任意波形采样须等间隔（非均匀请先重采样）')
    if np.max(np.abs(a)) <= 0:
        raise ValueError('任意波形幅值全零')

    spec_r = np.fft.rfft(a)                    # (n//2+1,) 复数
    mag = np.abs(spec_r)
    freqs_ghz = np.fft.rfftfreq(n, dt)         # t 单位 ns → GHz
    thresh = peak_frac * mag.max()

    # 迭代 deflation 峰检测：最强峰 → 沿谱展开窗口至连续低于阈值 →
    # 整窗吸收（含包络旁瓣）→ 置零 → 下一峰。避免把旁瓣误当独立音。
    remaining = mag.copy()
    min_tone = min_tone_frac * mag.max()
    peaks = []                      # (k0, lo, hi)
    for _ in range(max_tones):
        k0 = int(np.argmax(remaining))
        if remaining[k0] <= max(thresh, min_tone):
            break
        # 窗口扩展：连续 gap_bins 低于阈值才停（桥接旁瓣谷）
        gap = 0
        w = 0
        while k0 - w - 1 >= 0 and gap < 8:
            if remaining[k0 - w - 1] > thresh:
                w += 1
                gap = 0
            else:
                gap += 1
                w += 1
        lo = max(0, k0 - w)
        gap = 0
        w = 0
        while k0 + w + 1 < len(remaining) and gap < 8:
            if remaining[k0 + w + 1] > thresh:
                w += 1
                gap = 0
            else:
                gap += 1
                w += 1
        hi = min(len(remaining), k0 + w + 1)
        peaks.append((k0, lo, hi))
        remaining[lo:hi] = 0.0
    if not peaks:
        raise ValueError('主音分解失败：谱中无显著峰（密集谱/噪声类波形）'
                         '——请改用 method=auto（ip4 实验室系引擎）')

    tones = []
    recon = np.zeros(n)
    for k, lo, hi in peaks:
        seg = mag[lo:hi]
        f_ghz = float(np.sum(freqs_ghz[lo:hi] * seg) / np.sum(seg))  # 质心
        # 带通（保持共轭对称 → 实信号）
        mask_r = np.zeros_like(spec_r)
        mask_r[lo:hi] = spec_r[lo:hi]
        b = np.fft.irfft(mask_r, n)            # 孤立实音
        # 解析信号 → 慢变复包络
        Z = np.fft.fft(b)
        Z[n // 2 + 1:] = 0
        z_ana = np.fft.ifft(Z) * 2.0           # A(t)e^{iφ}e^{2πi f_c t}
        base = z_ana * np.exp(-2j * np.pi * f_ghz * t)
        env = np.real(base)                 # A(t)cosφ（φ=0/π 时为带符号包络）
        imag_ratio = float(np.max(np.abs(np.imag(base)))) /             (float(np.max(np.abs(env))) + 1e-30)
        if imag_ratio > 0.05:
            raise ValueError(
                f'主音 f={f_ghz*1e3:.6g} MHz 相位非 0/π'
                f'（|Im/Re|={imag_ratio:.2f}）——相位调制类波形请改用 '
                'method=auto（ip4）')
        recon += env * np.cos(2 * np.pi * f_ghz * t)
        tones.append((f_ghz * 1e3, k, env))    # GHz → MHz

    resid = float(np.max(np.abs(a - recon)) / np.max(np.abs(a)))
    if resid > 0.02:
        raise ValueError(
            f'主音分解残差 {resid:.2e} > 2e-2（谱不稀疏：啁啾/密集多频）'
            '——请改用 method=auto（ip4 实验室系引擎）')

    idx = np.unique(np.linspace(0, n - 1, min(env_points, n)).astype(int))
    out = [(f_mhz, t[idx], env[idx]) for f_mhz, _, env in tones]
    return out, resid
