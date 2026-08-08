# -*- coding: utf-8 -*-
"""
AC-Stark 位移：直接哈密顿量对角化交叉验证（独立于 core.py 的解析/微扰路径）
============================================================================

背景
----
core.py 的 AC-Stark（ac_stark_shift_mhz）采用"逐伙伴分流"的解析/微扰路径：
  * 近共振（|Δ| < max(5 GHz, Rabi)）：两能级 dressed 精确解（RWA）
  * 远失谐：非 RWA 二阶项（rotating + counter-rotating，即 SOS）
  * 每个偶极伙伴独立求和（多态同时近共振时两两独立近似失效）
  * 超精细态按"核自旋平均"处理（(2I+1) 归一化 / mJ 分量非相干加权）

该方法在下列情形是近似的：
  1) 多个态同时近共振（README 4.7 已有 UI 警告，core 仍按独立对处理）
  2) 近共振强驱动下 RWA 反旋项（Bloch-Siegert 位移）
  3) 超精细态间耦合的相干性（核自旋平均模型 vs 相干 HFS 矩阵元）

本文件构造含光子数（Floquet）的完整哈密顿量并【直接数值对角化】：
    H_F = Σ_n (H0 + n·ħω)⊗|n⟩⟨n| + (V/2)⊗(|n+1⟩⟨n| + |n⟩⟨n+1|),  V = −d·E0
其中偶极矩阵元与 core.transition_strength_q 完全同约定（|d|² 逐一相等，见 S0）。
一次对角化自洽包含：近共振 dressed + 远失谐 SOS（含反旋项）+ 多态耦合
+ 超精细相干性，可作为 core 的独立"地面真值"参考。

用法
----
    & "D:\python310\python.exe" core_ac_directcheck.py          # 全量（快速）
    & "D:\python310\python.exe" core_ac_directcheck.py --quick  # 只跑快速场景
    & "D:\python310\python.exe" core_ac_directcheck.py --full-sos  # 含完整 SOS 背景精确对拍
    & "D:\python310\python.exe" core_ac_directcheck.py --no-torch

依赖
----
    numpy / scipy（必须）；torch（可选）：基组×光子块 >= 1000 且 CUDA 可用时
    自动用 GPU 并行对角化（complex128 保证 kHz 级精度）。

性能提示
--------
    * S1/S1c 用近邻截断基组 n∈[55,65]（~130 子能级），快速且包含主要背景。
    * --full-sos 的 S1b 含完整 SOS 背景（~970 子能级），偶极耦合构建需要
      ARC 计算 ~2.6 万个里德堡态径向矩阵元（Numerov）。ARC 用持久化
      SQLite 数据库缓存，因此【只有首次运行 --full-sos 慢】（约 5–15 分钟），
      之后为毫秒级数据库查询；该数据库同时加速 UI 对里德堡组的首次计算。
    * --quick 跳过 S1c，约 10 秒内跑完 S0/S1/S3/S5。

结果解读
--------
    * S0   : 自检——本文件矩阵元与 core 完全一致（保证比较有意义）
    * S1   : Rydberg–Rydberg（无超精细）近邻全基组扫描。注意 60P3/2 距
             60P1/2 仅 ~461 MHz，在 Ω~14 MHz 时背景贡献数百 kHz，必须纳入
             （SOS_N_RANGE=(55,65) 截断即可，更远态 <~kHz）。
    * S1b  : --full-sos 完整背景（n∈[20,100]）精确对拍，含邻态(61P)共振。
    S1c : 强驱动下 RWA 反旋项（Bloch-Siegert）探测——core 的 RWA 误差随
          Ω/ω 增大而增大，超出实验容差时需改用直接法。
    S3   : 6P3/2↔60S 超精细↔里德堡（T9 场景）。6P 侧（F→mJ，单通道无干涉）
           应与 core 一致；60S 侧会暴露 core 的"上态漏耦合"（pair_contribution
           用 m2=m+q 只覆盖源为低态的吸收方向），见 S3 警告。
    S5   : 5S↔6P3/2 超精细↔超精细（F–F）。core 对超精细源做非相干加权，
           直接法用相干 HFS 矩阵元（与 ARC getDipoleMatrixElementHFS 一致），
           差异即该近似的量级（INFO，需结合实验场景判断）。
"""

import os
import sys
import time
import importlib.util
import numpy as np
from scipy.constants import c, epsilon_0, h

# Windows 控制台默认 GBK，无法打印 ²/σ± 等字符；强制 UTF-8 输出避免崩溃。
if sys.platform == 'win32' and hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)     # 项目根（core.py 所在）

# ---- 加载 core（不触发 GUI / 不执行 UI 代码） ----
spec = importlib.util.spec_from_file_location('core', os.path.join(_ROOT, 'core.py'))
core = importlib.util.module_from_spec(spec)
spec.loader.exec_module(core)

from arc.alkali_atom_data import Rubidium87

try:
    import torch
    _HAS_TORCH = True
except Exception:
    torch = None
    _HAS_TORCH = False

_CUDA_OK = _HAS_TORCH and torch.cuda.is_available()
_TORCH_MIN_SIZE = 1000        # 基组×光子块 >= 此值且 GPU 可用时自动用 torch

_PASS = 0
_FAIL = 0


def check(name, cond, detail=''):
    global _PASS, _FAIL
    if cond:
        _PASS += 1
        print(f'  [PASS] {name}')
    else:
        _FAIL += 1
        print(f'  [FAIL] {name}   {detail}')


# ============================================================
# 1) 复偶极矩阵元（与 core.transition_strength_q 同约定，|·|² 应完全相等）
# ============================================================

def dipole_me(cache, a, b, q):
    """⟨b|d_q|a⟩（a.u.，复）。分支逻辑与 core.transition_strength_q 完全一致，
    但保留相位（amps 为复振幅），用于构造哈密顿量；|返回值|² 应等于
    core.transition_strength_q(cache, a, b, q)。"""
    if a['kind'] == 'mJ' and b['kind'] == 'mJ':
        m2 = a['mj'] + q
        if abs(m2 - b['mj']) > 1e-9:
            return 0.0j
        return cache.dme(a['n'], a['l'], a['j'], a['mj'],
                         b['n'], b['l'], b['j'], m2, q)
    if a['kind'] == 'F' and b['kind'] == 'mJ':
        amp = 0.0j
        for mj, am in a['amps'].items():
            m2 = mj + q
            if abs(m2 - b['mj']) > 1e-9:
                continue
            amp += am.conjugate() * cache.dme(
                a['n'], a['l'], a['j'], mj, b['n'], b['l'], b['j'], m2, q)
        return amp
    if a['kind'] == 'mJ' and b['kind'] == 'F':
        amp = 0.0j
        for mj, am in b['amps'].items():
            m1 = mj + q
            if abs(m1 - a['mj']) > 1e-9:
                continue
            amp += am.conjugate() * cache.dme(
                b['n'], b['l'], b['j'], mj, a['n'], a['l'], a['j'], m1, q)
        return amp
    # F -- F（mF 守恒）
    if abs((a['mF'] + q) - b['mF']) > 1e-9:
        return 0.0j
    amp = 0.0j
    for mj1, am1 in a['amps'].items():
        mj2 = mj1 + q
        if abs(mj2) <= b['j'] + 1e-9:
            am2 = b['amps'].get(mj2)
            if am2 is not None:
                amp += am1.conjugate() * am2 * cache.dme(
                    a['n'], a['l'], a['j'], mj1, b['n'], b['l'], b['j'], mj2, q)
    return amp


# ============================================================
# 2) 直接对角化基组
# ============================================================

def _sublevel_states(cache, n, l, j, B):
    """单个 (n,l,j) 的全部子能级状态（与 core._group_sublevel_rows 一致，
    额外带 qstr 便于匹配 core 的行）。"""
    out = []
    hf = core.get_hyperfine_levels(cache, n, l, j, B)
    if hf is None:
        gJ = cache.gJ(l, j)
        for m in np.arange(-j, j + 1, 1.0):
            e = cache.E_mhz(n, l, j) + gJ * core.MU_B_MHZ_PER_G * m * B
            out.append(dict(n=n, l=l, j=j, kind='mJ', mj=float(m), mF=None,
                            amps=None, E_mhz=float(e), qstr=f'm_J={m:+.1f}'))
    else:
        for F, mF, e_hf, w, amps in hf:
            e = cache.E_mhz(n, l, j) + e_hf
            out.append(dict(n=n, l=l, j=j, kind='F', mj=None, mF=mF, amps=amps,
                            E_mhz=float(e), qstr=f'F={F:.1f}, m_F={mF:+.1f}'))
    return out


def _partners_in_range(n, l, j, n_range):
    """在指定 n 范围内生成 (n,l,j) 的偶极伙伴（与 core.dipole_partners 逻辑一致）。"""
    n_min, n_max = n_range
    out = []
    for l1 in (l - 1, l + 1):
        if l1 < 0:
            continue
        for j1 in (l1 - 0.5, l1 + 0.5):
            if j1 < 0.5 or abs(j - j1) > 1.0 + 1e-9:
                continue
            for n1 in range(max(n_min, l1 + 1), n_max + 1):
                out.append((n1, l1, j1))
    return out


# 覆盖 SOS 基组 n 范围；None = 与 core 完全一致（完整 [20,100]）。
# 例：(55, 65) 只取源附近 55..65，包含主要近邻背景（如 60P3/2 距 60P1/2 仅
# ~461 MHz，在 Ω~14 MHz 时贡献数百 kHz，必须纳入）；更远态贡献 < ~kHz。
SOS_N_RANGE = None


def build_direct_basis(cache, groups, B, include_sos=True):
    """构造直接对角化用的电子子能级基组。

    含全部已添加组的子能级 +（include_sos=True）每个组偶极伙伴的 SOS 基组
    子能级。默认用 core.dipole_partners（完整 [20,100]，与 core 完全对应）；
    若设置模块级 SOS_N_RANGE=(nmin,nmax)，则截断到该 n 范围（快速模式）。
    返回 states 列表（每个含 n/l/j/kind/mj|mF/amps/E_mhz/qstr）。
    """
    states = []
    seen = set()

    def add(n, l, j):
        if (n, l, j) in seen:
            return
        seen.add((n, l, j))
        states.extend(_sublevel_states(cache, n, l, j, B))

    for g in groups:
        add(g['n'], g['l'], g['j'])
    if include_sos:
        for g in groups:
            if SOS_N_RANGE is not None:
                partners = _partners_in_range(g['n'], g['l'], g['j'], SOS_N_RANGE)
            else:
                partners = core.dipole_partners(g['n'], g['l'], g['j'])
            for (n1, l1, j1) in partners:
                add(n1, l1, j1)
    return states


# ============================================================
# 3) Floquet 哈密顿量
# ============================================================

def build_coupling(cache, states, address_cfg):
    """与频率无关的耦合结构。返回 (E_mhz 数组, 耦合矩阵 C, t_mhz)。

    C_ij = −d_ij·E0/2（MHz），d_ij 为寻址偏振 q 的复偶极矩阵元（a.u.），
    E0² = 2I/(ε0·c)。t_mhz = (e·a0·E0)/(2·h·1e6) 是 |d|=1 a.u. 时的耦合。
    偶极选择定则预过滤（Δl=±1, |Δj|≤1）避免对禁戒对做无谓 ARC 调用。
    """
    N = len(states)
    E = np.array([s['E_mhz'] for s in states], dtype=np.float64)
    q = core.POLARIZATION_Q[address_cfg['polarization']]
    I_wm2 = address_cfg['intensity'] * core.MW_PER_CM2_TO_W_PER_M2
    E0 = (2.0 * I_wm2 / (epsilon_0 * c)) ** 0.5
    t_mhz = (core.E_CHARGE * core.A0 * E0) / (2.0 * h * 1e6)
    C = np.zeros((N, N), dtype=np.complex128)
    verbose = N >= 500
    t0 = time.time()
    for i in range(N):
        if verbose and i % 200 == 0:
            print(f'      [耦合构建] {i}/{N} 子能级，{time.time()-t0:.0f}s'
                  '（ARC 径向矩阵元，最耗时步骤）', flush=True)
        si = states[i]
        for j in range(i + 1, N):
            sj = states[j]
            if abs(si['l'] - sj['l']) != 1 or abs(si['j'] - sj['j']) > 1.0 + 1e-9:
                continue
            d = dipole_me(cache, si, sj, q)
            if abs(d) > 1e-16:
                v = -t_mhz * d
                C[i, j] = v
                C[j, i] = np.conj(v)
    return E, C, t_mhz


def build_floquet_hamiltonian(states, address_cfg, n_photon=1, E=None, C=None):
    """构造 Floquet 哈密顿量（光子块 −n_photon..+n_photon）。

        i==j, p==p' : E_i + p·ν（MHz，ν 为寻址光频率）
        |p−p'|==1   : C_ij（偶极耦合，仅寻址偏振 q）
    返回 (N·M, N·M) 复数 Hermitian 矩阵。
    """
    N = len(states)
    M = 2 * n_photon + 1
    freq_mhz = address_cfg['frequency_hz'] / 1e6
    H = np.zeros((N * M, N * M), dtype=np.complex128)
    for p in range(M):
        sl = slice(p * N, (p + 1) * N)
        H[sl, sl] = np.diag(E) + (p - n_photon) * freq_mhz * np.eye(N)
    for p in range(M - 1):
        H[p * N:(p + 1) * N, (p + 1) * N:(p + 2) * N] = C
        H[(p + 1) * N:(p + 2) * N, p * N:(p + 1) * N] = C.conj().T
    return H


# ============================================================
# 4) 对角化与位移提取
# ============================================================

def _eigh(H, use_torch=None):
    """对角化。H 可为单矩阵 (N,N) 或批次 (B,N,N)。use_torch=None 自动选择。
    torch 路径用 complex128 保证 kHz 级精度，返回 torch 张量（仍在 GPU）。"""
    size = H.shape[-1]
    if use_torch is None:
        use_torch = _CUDA_OK and size >= _TORCH_MIN_SIZE
    if use_torch:
        if not (_HAS_TORCH and torch.cuda.is_available()):
            raise RuntimeError('请求 torch-GPU，但 torch/CUDA 不可用（加 --no-torch 用 numpy）')
        t = torch.as_tensor(H, dtype=torch.complex128, device='cuda')
        return torch.linalg.eigh(t)          # (evals, evecs)，均在 GPU
    return np.linalg.eigh(H)


def _occ_keys(items):
    """为（可能含重复 qstr 的）子能级列表生成唯一键 (n,l,j,qstr,occ)。

    B≠0 时 get_hyperfine_levels 的 F 标注启发式可能给两个本征态同一个
    qstr（如 6P3/2 在 21.7 G 下出现两个 'F=0.0, m_F=+0.0'）。core 行与
    本文件基组都按 get_hyperfine_levels 的返回顺序遍历，故出现序号 occ
    一致，可唯一对齐。
    """
    seen = {}
    keys = []
    for it in items:
        base = (it['n'], it['l'], it['j'], it['qstr'])
        occ = seen.get(base, 0)
        seen[base] = occ + 1
        keys.append(base + (occ,))
    return keys


def _extract_shifts(states, evals, evecs, n_photon):
    """从（可批量）对角化结果提取各子能级 AC 位移。

    shift = E_quasi − E_bare（MHz）。E_quasi 取与 |state, 光子块0⟩ 重叠最大的
    本征值；重叠可比（如共振 50/50 劈裂）时再按"离裸能量最近"定支。
    返回 dict 列表（每批次一个）：[(n,l,j,qstr,occ)] -> (shift_mhz, overlap)。
    evals/evecs 可为 numpy 或 torch(GPU)。
    """
    is_torch = _HAS_TORCH and isinstance(evals, torch.Tensor)
    batch = evals.ndim == 2
    B = evals.shape[0] if batch else 1
    N = len(states)
    M = evals.shape[-1] // N
    p0 = n_photon                                  # 光子块 0 的块下标
    if is_torch:
        dev = evals.device
        E0mat = torch.zeros((N * M, N), dtype=torch.complex128, device=dev)
        E0mat[p0 * N:(p0 + 1) * N, :] = torch.eye(N, dtype=torch.complex128, device=dev)
        ov = (evecs.conj().transpose(-2, -1) @ E0mat).abs() ** 2   # (B,NM,N) 或 (NM,N)
    else:
        E0mat = np.zeros((N * M, N), dtype=np.complex128)
        E0mat[p0 * N:(p0 + 1) * N, :] = np.eye(N)
        ov = np.abs(np.conj(evecs).T @ E0mat) ** 2
    out = []
    for b in range(B):
        ev = evals[b] if batch else evals
        ovb = ov[b] if batch else ov
        if is_torch:
            ev = ev.cpu().numpy()
            ovb = ovb.cpu().numpy()
        keys = _occ_keys(states)
        d = {}
        for s in range(N):
            ov_s = ovb[:, s]
            best = int(np.argmax(ov_s))
            thr = 0.3 * ov_s[best]
            cand = np.where(ov_s >= thr)[0]
            k = int(cand[np.argmin(np.abs(ev[cand] - states[s]['E_mhz']))])
            d[keys[s]] = (float(ev[k] - states[s]['E_mhz']), float(ov_s[k]))
        out.append(d)
    return out[0] if not batch else out


def _diagonalize_scan(states, cfgs, E, C, t_mhz, n_photon, use_torch=None):
    """对多个 address 配置对角化。GPU 且显存足够时批量并行，否则逐点。"""
    N = len(states)
    M = 2 * n_photon + 1
    S = N * M
    if use_torch is None:
        use_torch = _CUDA_OK and S >= _TORCH_MIN_SIZE
    if use_torch and _HAS_TORCH and torch.cuda.is_available():
        free = torch.cuda.mem_get_info()[0]
        need = len(cfgs) * S * S * 16            # complex128
        if need < 0.6 * free:
            Hs = np.stack([build_floquet_hamiltonian(states, c, n_photon, E, C)
                           for c in cfgs])
            t0 = time.time()
            evals, evecs = _eigh(Hs, use_torch=True)
            print(f'   [torch-GPU] 批量对角化 {len(cfgs)}×{S}² 矩阵，{time.time()-t0:.1f}s')
            return _extract_shifts(states, evals, evecs, n_photon)
        print(f'   [torch-GPU] 批量需 {need/1e9:.1f} GB 超显存，改为逐点（仍用 GPU）')
    out = []
    for c in cfgs:
        H = build_floquet_hamiltonian(states, c, n_photon, E, C)
        t0 = time.time()
        evals, evecs = _eigh(H, use_torch)
        tag = 'torch-GPU' if (_HAS_TORCH and isinstance(evals, torch.Tensor)) else 'numpy'
        if S >= _TORCH_MIN_SIZE:
            print(f'   [{tag}] 对角化 {S}²，{time.time()-t0:.1f}s')
        out.append(_extract_shifts(states, evals, evecs, n_photon))
    return out


# ============================================================
# 5) 公共 API：单个配置的直接对角化 AC 位移
# ============================================================

def direct_ac_shift_mhz(cache, groups, B, address_cfg, include_sos=True,
                        n_photon=1, use_torch=None):
    """直接对角化 Floquet 哈密顿量，返回所有已添加组子能级的 AC 位移。

    参数：
        cache       : core.ArcCache 实例
        groups      : 能级组列表（含 n/l/j/label/color）
        B           : 磁场 (G)
        address_cfg : dict(enabled, intensity, frequency_hz, polarization)
        include_sos : 是否含完整偶极伙伴 SOS 背景基组（与 core 对应）
        n_photon    : Floquet 光子块截断（≥1；默认 1 已含反旋项）
        use_torch   : None=自动（大基组且 CUDA 可用时用 GPU），True/False 强制

    返回：dict[(n,l,j,qstr,occ)] -> (shift_mhz, overlap)。occ 为重复 qstr
    的出现序号（B≠0 时 get_hyperfine_levels 可能给出重复 F 标注）。
    """
    states = build_direct_basis(cache, groups, B, include_sos)
    E, C, t_mhz = build_coupling(cache, states, address_cfg)
    H = build_floquet_hamiltonian(states, address_cfg, n_photon, E, C)
    evals, evecs = _eigh(H, use_torch)
    return _extract_shifts(states, evals, evecs, n_photon)


# ============================================================
# 6) 与 core 的逐点对比
# ============================================================

def scan_compare(name, cache, groups, B, base_freq_hz, dets_mhz, intensity, pol,
                 include_sos=True, n_photon=1, use_torch=None, rtol=0.02, atol=0.05,
                 check_label=None, info_label=None, info_only=False):
    """对一组失谐扫描点：core.build_level_rows 的行位移 vs 直接对角化。

    check_label : 只对该标签的行做 PASS/FAIL（如 '6P3/2'）
    info_label  : 额外打印该标签行的 core/direct 数值（不判对错）
    info_only   : 全部只报告不判对错
    """
    print(f'== {name} ==')
    states = build_direct_basis(cache, groups, B, include_sos)
    S = len(states) * (2 * n_photon + 1)
    print(f'   B={B} G   I={intensity} mW/cm²   pol={pol}')
    print(f'   Δ(MHz) = {dets_mhz}')
    print(f'   基组 {len(states)} 子能级 × {2*n_photon+1} 光子块 = {S}² '
          f'（include_sos={include_sos}）')
    cfgs = [dict(enabled=True, intensity=intensity,
                 frequency_hz=base_freq_hz + d * 1e6, polarization=pol)
            for d in dets_mhz]
    E, C, t_mhz = build_coupling(cache, states, cfgs[0])
    direct = _diagonalize_scan(states, cfgs, E, C, t_mhz, n_photon, use_torch)

    labels = {}
    cores = []
    for cfg in cfgs:
        rows = core.build_level_rows(cache, B, groups, cfg, groups, backend='legacy')
        rkeys = _occ_keys(rows)
        for k, r in zip(rkeys, rows):
            labels[k] = r['label']
        cores.append(dict(zip(rkeys, (r['ac'] for r in rows))))

    for i, dd in enumerate(dets_mhz):
        core_d = cores[i]
        dir_d = direct[i]
        bad = []
        checked = 0
        worst = (None, -1.0)
        for k, c in sorted(core_d.items()):
            lab = labels.get(k)
            if check_label is not None and lab != check_label:
                continue
            if k not in dir_d:
                continue
            dv, ov = dir_d[k]
            if not np.isfinite(dv):
                bad.append((k, c, dv, ov, float('inf')))
                continue
            checked += 1
            # 共振 50/50 劈裂时重叠 < 0.6，符号有歧义 → 比较模
            dev = abs(abs(dv) - abs(c)) if ov < 0.6 else abs(dv - c)
            if dev > worst[1]:
                worst = (k, dev)
            if dev > atol + rtol * abs(c):
                bad.append((k, c, dv, ov, dev))
        if info_only:
            print(f'   [INFO] Δ={dd:+.0f} MHz：{checked} 行，'
                  f'最大 |direct−core| = {worst[1]*1000 if worst[1] >= 0 else float("nan"):.1f} kHz'
                  + (f'（{len(bad)} 行超差）' if bad else ''))
        else:
            check(f'{name} @Δ={dd:+.0f} MHz（{checked} 行）', not bad and checked > 0,
                  f'{len(bad)} 行超差')
        for k, c, dv, ov, dev in bad[:4]:
            print(f'      {labels.get(k, "?"):<8} {k[3]:<18} core={c:+.4f} '
                  f'direct={dv:+.4f} MHz  dev={dev*1000:.1f} kHz  ov={ov:.2f}')
        if info_label is not None:
            for k, c in sorted(core_d.items()):
                if labels.get(k) != info_label or k not in dir_d:
                    continue
                dv, ov = dir_d[k]
                print(f'      [INFO行] {info_label:<8} {k[3]:<18} core={c:+.4f} '
                      f'direct={dv:+.4f} MHz  ov={ov:.2f}')
    print()


# ============================================================
# 7) 场景
# ============================================================

def s0():
    """自检：本文件矩阵元与 core 完全一致。"""
    print('S0  偶极矩阵元自洽：dipole_me 与 core.transition_strength_q 逐项对拍')
    cache = core.ArcCache(Rubidium87())
    groups = [dict(n=5, l=0, j=0.5, label='5S', color='r'),
              dict(n=6, l=1, j=1.5, label='6P3/2', color='b'),
              dict(n=60, l=0, j=0.5, label='60S', color='g'),
              dict(n=60, l=1, j=0.5, label='60P1/2', color='m')]
    rows = core.build_level_rows(cache, 21.7, [], {}, groups)
    r5 = [r for r in rows if r['label'] == '5S']
    r6 = [r for r in rows if r['label'] == '6P3/2']
    r60s = [r for r in rows if r['label'] == '60S']
    r60p = [r for r in rows if r['label'] == '60P1/2']
    pairs = ([(a, b) for a in r5 for b in r6] +
             [(a, b) for a in r60s for b in r6] +
             [(a, b) for a in r6 for b in r60s] +
             [(a, b) for a in r60s for b in r60p])
    maxdev = 0.0
    n = 0
    for a, b in pairs:
        for q in (-1, 0, 1):
            d = dipole_me(cache, a, b, q)
            s_core = core.transition_strength_q(cache, a, b, q)
            maxdev = max(maxdev, abs(abs(d) ** 2 - s_core))
            n += 1
    check(f'S0  |dipole_me|² == transition_strength_q（{n} 项）', maxdev < 1e-9,
          f'maxdev={maxdev:.2e}')
    print()


def s1():
    """Rydberg–Rydberg 近邻全基组：含主要 SOS 背景（60P3/2 距 60P1/2 仅 ~461 MHz，
    在 Ω~14 MHz 时贡献数百 kHz，必须纳入）。用 SOS_N_RANGE=(55,65) 截断，
    快速且与 core 精确一致到 ~kHz（更远态贡献可忽略）。"""
    print('S1  Rydberg 近邻全基组扫描（60S↔60P1/2，含主要 SOS 背景）')
    print('    背景截断 n∈[55,65]（含 60P3/2、59P/61P 等主要贡献者；')
    print('    更远态贡献 <~kHz 可忽略）。精确含完整背景请 --full-sos。')
    cache = core.ArcCache(Rubidium87())
    groups = [dict(n=60, l=0, j=0.5, label='60S', color='r'),
              dict(n=60, l=1, j=0.5, label='60P1/2', color='b')]
    base = core.transition_frequency_hz(cache.atom, 60, 0, 0.5, 60, 1, 0.5)
    cfg0 = dict(enabled=True, intensity=0.0001, frequency_hz=base, polarization='π')
    pairs = core.collect_transitions(cache, groups, cfg0, 0.0)
    if pairs:
        p = min(pairs, key=lambda x: abs(x[4]))
        print(f'   [参考] 60S↔60P 近共振 Rabi Ω/2π ≈ {p[5]/(2*np.pi)/1e6:.3f} MHz'
              f' @ I=0.0001 mW/cm²')
    global SOS_N_RANGE
    SOS_N_RANGE = (55, 65)
    try:
        print('   （共振点 Δ=0 的 Autler-Townes 分支歧义/多态耦合由 S3 与 T9 验证，'
              '此处用 ±5 MHz）')
        scan_compare('S1', cache, groups, 0.0, base,
                     [-200, -5, 5, 1000],
                     0.0001, 'π', include_sos=True, rtol=0.02, atol=0.02)
    finally:
        SOS_N_RANGE = None


def s1b():
    """Rydberg 完整 SOS 背景精确对拍（--full-sos 触发）。
    含全部偶极伙伴（n∈[20,100]），近/远失谐 + 邻态(61P)共振一次全验。
    首次运行需构建 ARC 径向矩阵元数据库（约 5–15 分钟），之后为秒级。"""
    print('S1b Rydberg 全基组（含完整 SOS 背景）精确对拍（--full-sos）')
    cache = core.ArcCache(Rubidium87())
    groups = [dict(n=60, l=0, j=0.5, label='60S', color='r'),
              dict(n=60, l=1, j=0.5, label='60P1/2', color='b')]
    base = core.transition_frequency_hz(cache.atom, 60, 0, 0.5, 60, 1, 0.5)
    f61p = core.transition_frequency_hz(cache.atom, 60, 0, 0.5, 61, 1, 0.5)
    det_61p = int(round((f61p - base) / 1e6))
    # SOS_N_RANGE 保持 None（完整基组）
    scan_compare('S1b', cache, groups, 0.0, base,
                 [-200, 0, 200, 5000, det_61p],
                 0.0001, 'π', include_sos=True, rtol=0.02, atol=0.02)


def s1c():
    """强驱动下 RWA 反旋项（Bloch-Siegert）探测（INFO）。"""
    print('S1c RWA 反旋项（Bloch-Siegert）探测：强驱动下 direct 相对 core 的偏差（INFO）')
    print('    core 近共振用 RWA dressed（无反旋项）；直接法含反旋项。偏差 ~ Ω²/(4·ω_eg)。')
    cache = core.ArcCache(Rubidium87())
    groups = [dict(n=60, l=0, j=0.5, label='60S', color='r'),
              dict(n=60, l=1, j=0.5, label='60P1/2', color='b')]
    base = core.transition_frequency_hz(cache.atom, 60, 0, 0.5, 60, 1, 0.5)
    for I in (0.01, 0.04):
        cfg0 = dict(enabled=True, intensity=I, frequency_hz=base, polarization='π')
        pairs = core.collect_transitions(cache, groups, cfg0, 0.0)
        om = (min(pairs, key=lambda x: abs(x[4]))[5] / (2 * np.pi) / 1e6
              if pairs else float('nan'))
        print(f'   [参考] Ω/2π ≈ {om:.0f} MHz @ I={I} mW/cm²（ω/2π=16.83 GHz，'
              f'Ω/ω = {om/16830:.2%}，RWA 反旋误差 ~Ω²/(4ω) 量级）')
        global SOS_N_RANGE
        SOS_N_RANGE = (55, 65)
        try:
            scan_compare(f'S1c(I={I})', cache, groups, 0.0, base, [-200, 0, 200], I, 'π',
                         include_sos=True, rtol=0.02, atol=0.05, info_only=True)
        finally:
            SOS_N_RANGE = None


def s3():
    """超精细↔里德堡（T9 场景）。6P 侧（低态吸收方向）应一致；
    60S 侧会暴露 core 的"上态漏耦合"问题（pair_contribution 用 m2=m+q
    只覆盖源为低态的吸收方向），见下方警告。"""
    print('S3  超精细↔里德堡（6P3/2 ↔ 60S，B=21.7 G，σ⁻，T9 场景）')
    print('    6P 侧（低态 F→mJ 吸收方向）：core 与直接法应一致（dressed ±Ω/2）。')
    print('    60S 侧（高态）core 用 m2=m+q 只覆盖"源为低态"的吸收方向，')
    print('    会漏掉 σ⁻ 与 6P F=3mF=3 的共振耦合 → 见下方警告与 INFO 行。')
    cache = core.ArcCache(Rubidium87())
    B = 21.7
    groups = [dict(n=6, l=1, j=1.5, label='6P3/2', color='r'),
              dict(n=60, l=0, j=0.5, label='60S', color='b')]
    rows_6p = core._group_sublevel_rows(cache, 6, 1, 1.5, B)
    rows_60s = core._group_sublevel_rows(cache, 60, 0, 0.5, B)
    b63 = next(r for r in rows_6p if r['qstr'] == 'F=3.0, m_F=+3.0')
    a60 = next(r for r in rows_60s if r['qstr'] == 'm_J=+0.5')
    base = (a60['E_mhz'] - b63['E_mhz']) * 1e6
    cfg0 = dict(enabled=True, intensity=10000.0, frequency_hz=base, polarization='σ-')
    pairs = core.collect_transitions(cache, groups, cfg0, B)
    if pairs:
        p = min(pairs, key=lambda x: abs(x[4]))
        print(f'   [参考] F=3,mF=3 ↔ mJ=+0.5 共振 Rabi Ω/2π ≈ {p[5]/(2*np.pi)/1e6:.3f} MHz'
              f' @ I=10000 mW/cm²')
    scan_compare('S3', cache, groups, B, base, [0, 20], 10000.0, 'σ-',
                 include_sos=False, rtol=0.03, atol=0.01,
                 check_label='6P3/2', info_label='60S')
    # —— 上态共振漏耦合检查（60S 侧）：直接法应给出 −Ω/2，core 只给 ~kHz ——
    direct_res = direct_ac_shift_mhz(cache, groups, B, cfg0, include_sos=False)
    rows_res = core.build_level_rows(cache, B, groups, cfg0, groups, backend='legacy')
    rk = _occ_keys(rows_res)
    for k, r in zip(rk, rows_res):
        if r['label'] == '60S' and r['qstr'] == 'm_J=+0.5' and k in direct_res:
            dv, ov = direct_res[k]
            if abs(dv) > 0.05 and abs(r['ac']) < 0.05 * abs(dv):
                print('  [警告] core 的 60S m_J=+0.5 在 σ⁻ 共振处位移 ≈ '
                      f'{r["ac"]*1000:.0f} kHz，直接法（仅 6P+60S 基组）≈ {dv:.3f} MHz：')
                print('         core 漏掉上态(60S)↔6P F=3mF=3 的共振耦合：')
                print('         pair_contribution_mhz 用 m2=m+q，只覆盖"源为低态"'
                      '的吸收方向；')
                print('         dressed 物理上该对两侧都位移 ±Ω/2（Autler-Townes），'
                      'core 只画了 6P 侧。')
                print('         即使按 mI 核自旋平均（÷(2I+1)=4）也不应只有 ~kHz。')
                print('         建议：核对/修复 core 的 pair_contribution，或对该'
                      '场景改用本文件 direct_ac_shift_mhz。')
    print()


def s5():
    """超精细↔超精细（F–F）：相干 vs 核自旋平均模型差异（INFO）。
    实际偏差大小取决于具体通道（本场景测得很小）；若某 F–F 通道相干项
    显著，两者会分叉。"""
    print('S5  超精细↔超精细（5S↔6P3/2）相干 vs 核自旋平均模型（INFO）')
    print('    core 对超精细源态做"mJ 分量非相干加权 +(2I+1) 归一"（核自旋平均）；')
    print('    直接法用相干 HFS 矩阵元（与 ARC getDipoleMatrixElementHFS 一致）。')
    cache = core.ArcCache(Rubidium87())
    groups = [dict(n=5, l=0, j=0.5, label='5S', color='r'),
              dict(n=6, l=1, j=1.5, label='6P3/2', color='b')]
    base = core.transition_frequency_hz(cache.atom, 5, 0, 0.5, 6, 1, 1.5)
    scan_compare('S5', cache, groups, 0.0, base, [-300, 0, 300], 100.0, 'π',
                 include_sos=False, rtol=0.05, atol=0.02, info_only=True)


# ============================================================

def main():
    global _CUDA_OK
    print('=' * 72)
    print('AC-Stark 直接哈密顿量对角化交叉验证（vs core.py）')
    print('=' * 72)
    if _HAS_TORCH and torch.cuda.is_available():
        print(f'torch {torch.__version__}，GPU: {torch.cuda.get_device_name(0)}')
    elif _HAS_TORCH:
        print('torch 已安装但无 CUDA → 用 numpy/LAPACK（大基组会慢）')
    else:
        print('torch 未安装 → 用 numpy/LAPACK（大基组会慢）')
    args = sys.argv[1:]
    if '--no-torch' in args:
        _CUDA_OK = False
        print('--no-torch：禁用 GPU')
    quick = '--quick' in args
    full_sos = '--full-sos' in args
    s0()
    print('-' * 72)
    s1()
    print('-' * 72)
    if full_sos:
        s1b()
        print('-' * 72)
    if not quick:
        s1c()
        print('-' * 72)
    s3()
    print('-' * 72)
    s5()
    print('-' * 72)
    print(f'汇总: PASS={_PASS}  FAIL={_FAIL}')
    sys.exit(1 if _FAIL else 0)


if __name__ == '__main__':
    main()
