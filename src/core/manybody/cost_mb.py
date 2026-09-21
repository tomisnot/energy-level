# -*- coding: utf-8 -*-
"""多体 ED 成本/内存粗估（**无 Qt**，仿 core/cost.py 口径）。

⚠ 这些是"粗估不是承诺"：只用来决定"要不要走异步 / ED 是否可行 / 要不要降 N"，
与 cost.py 同一纪律。标定常数 MB_*_CAL 目前是**量级估计**（未经 cost.py 那样的
逐点实测标定），M1 先给保守外推；A8 锚（χ×N 耗时/内存实测）在 M3 细化。
维度 dim=2^N 是唯一主约束（前置文档 §4.2 断言）。
"""
__all__ = ['MB_EIGH_CAL', 'MB_KRYLOV_CAL', 'MB_DENSE_MAX_DEFAULT',
           'ed_feasible', 'estimate_mb_cost', 'estimate_mb_peak_gb']

# 稠密 eigh 档标定（量级）：eigh 成本 ∝ dim³。以 dim=1024 复 eigh ≈ 0.3 s 为锚。
MB_EIGH_CAL = {'dim': 1024, 'sec': 0.3}
# 稀疏 expm_multiply 逐区间步进档标定（量级）：∝ npts · N · dim（每步 Krylov×matvec）。
MB_KRYLOV_CAL = {'dim': 65536, 'n_atoms': 16, 'npts': 201, 'sec': 60.0}
MB_DENSE_MAX_DEFAULT = 4096          # 与 engine_ed._MB_DENSE_MAX 默认一致
_MB_FULLPOP_MAX_DEFAULT = 65536


def ed_feasible(n_atoms, ed_max_dim):
    """dim=2^N 是否在 ED 上限内（solver=auto 据此判可行）。"""
    return (1 << int(n_atoms)) <= int(ed_max_dim)


def estimate_mb_cost(n_atoms, t_max_ns, npts=201,
                     dense_max=MB_DENSE_MAX_DEFAULT):
    """多体 ED 演化耗时**粗估**（秒）。分稠密 eigh / 稀疏 Krylov 两档外推。

    与 t_max 近似无关（步数由 npts 定，非 t_max）；稠密档一次 eigh 后全时刻向量化，
    稀疏档逐区间步进 ∝ npts。
    """
    n = int(n_atoms)
    dim = 1 << n
    npts = max(int(npts), 2)
    if dim <= int(dense_max):
        s = MB_EIGH_CAL['sec'] * (dim / MB_EIGH_CAL['dim']) ** 3
        s += 1e-9 * dim * dim * npts          # 传播 dim²·npts
        return float(s)
    ref = MB_KRYLOV_CAL
    s = ref['sec'] * (npts / ref['npts'])
    s *= (dim / ref['dim']) * (n / ref['n_atoms'])
    return float(s)


def estimate_mb_peak_gb(n_atoms, npts=201, dense_max=MB_DENSE_MAX_DEFAULT,
                        fullpop_max=_MB_FULLPOP_MAX_DEFAULT):
    """多体 ED 峰值内存**粗估**（GB）：H（稠密 dim² / 稀疏 N·dim）+ psi/pops 主导。"""
    n = int(n_atoms)
    dim = 1 << n
    npts = max(int(npts), 2)
    gb = 0.0
    if dim <= int(dense_max):
        gb += 2 * dim * dim * 16 / 1e9        # 稠密 H + evecs（复双精度）
    else:
        gb += n * dim * 16 / 1e9              # 稀疏 H（~N·dim 非零）
    if dim <= int(fullpop_max):
        gb += npts * dim * 8 / 1e9            # 全联合 pops (n_t, dim)
        if dim <= int(dense_max):
            gb += dim * npts * 16 / 1e9       # psi_all (dim, n_t)（稠密路径）
    else:
        gb += npts * n * 8 / 1e9              # 只逐原子边缘
    gb += dim * 16 / 1e9                      # 若干态矢
    return float(gb)
