# -*- coding: utf-8 -*-
"""acize.py（t63 第 1 层统一）：AC 化公共层——单/两体共用。

从 evolve_mixed.ac_ize_physical_E + evolve_mixed_pair._ac_ize_pair 抽出泛化：
- classify_pairs(A, omega, r_crit)：r=Ω/ω 判据分类（t58）——远强对 vs 近/弱对
- far_sos_shift(A, E, i, k, omega)：A² 等价 far_sos 双向位移（复用
  static_atom.far_sos_shift_mhz 语义——A²·ν/(2(ν²−ω²)) 形式）
- acize_physical(E, fields_opt, r_crit)：返回 (A_new_list, shifts_list)——
  **shifts_list 为 per-field N 维位移（唯一实现：env² 全量时变语义——K0 纯 E，
  位移全由 RHS 时变对角 Σ shifts_f·env_f(t)² 承担——t62e/t62f 修复定案）**
"""
import numpy as np

R_CRIT = 0.0135               # t58: M_max=2 的 r 判据


def classify_pairs(A, omega, r_crit=R_CRIT):
    """按 r=Ω/ω 分类：True = 保留（近/弱对——光子块动态）；False = AC 化。"""
    A = np.asarray(A, float)
    r = np.abs(A) / omega
    keep = (r <= r_crit) & (np.abs(A) > 1e-9)
    return keep


def far_sos_pair_shift(A_ij, E_k, E_i, omega):
    """单对 (i,k) 对源态 i 的 A² 等价 far_sos 位移（MHz）。

    = −A_ij²·(E_k−E_i)/(2·((E_k−E_i)²−ω²))——与
    static_atom.far_sos_shift_mhz（d²·w/(w²−ω²) 结构）同语义；A² ∝ d²·E0²
    的 Rabi 形式。t60 仲裁：±18.23 与 M=4 锚一致（C1≈A 6.87e-3）。
    """
    nu = float(E_k) - float(E_i)
    return -(float(A_ij) ** 2) * nu / (nu ** 2 - omega ** 2) / 2.0


def acize_physical(E, fields_opt, r_crit=R_CRIT):
    """远强对 AC 化（唯一实现）：剔除 + per-field 位移。

    fields_opt: kind='exact' 光场列表（A 全偶极矩阵 + omega_L_mhz）。
    返回 (A_new_list, shifts_list)：
      A_new_list[f]：剔除远强对后的光场 A（近对保留）
      shifts_list[f]：第 f 个光场的 N 维位移向量（env² 全量时变语义——
        K0 用纯 E；RHS 时变对角 Σ_f shifts_f·env_f(t)² 由调用侧组装）
    """
    N = len(E)
    shifts_list = []
    A_new = []
    for f in fields_opt:
        A = np.asarray(f['A'], float).copy()
        w = float(f['omega_L_mhz'])
        sh_f = np.zeros(N)
        keep = classify_pairs(A, w, r_crit)
        for i in range(N):
            for k in range(i + 1, N):
                if not keep[i, k]:
                    if abs(A[i, k]) < 1e-9:
                        continue
                    dE = far_sos_pair_shift(A[i, k], E[k], E[i], w)
                    sh_f[i] += dE
                    sh_f[k] -= dE
                    A[i, k] = A[k, i] = 0.0
        A_new.append(A)
        shifts_list.append(sh_f)
    return A_new, shifts_list
