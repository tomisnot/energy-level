# -*- coding: utf-8 -*-
"""Rb-87 原子物理核心计算库（无 UI 依赖，可单独审查）。

本文件只包含物理模型与数值计算。UI 层（能级.py）直接渲染本文件输出的
数据结构，不做任何二次换算，以保证"结果即所见、所见即结果"。

============================================================
物理模型总览
============================================================
1) 能级
   - 里德堡态（n>=20）：只画 fine-structure 的 m_J 子能级（忽略核自旋），
     能级 E = E0 + g_J·mu_B·B·m_J，AC 位移按 m_J 逐项计算。
   - 低激发态：在 |m_J, m_I> 基下按 m_F 分块数值对角化
        H = A·I·J + B·Q + mu_B·B·(g_J·J_z + g_I·I_z)
     得到超精细子能级 (F, m_F, E) 及本征向量（m_J 分量复振幅，
     用于超精细态与里德堡态之间的跃迁偶极矩阵元）。

2) AC-Stark（寻址光）
   对每个偶极耦合伙伴按失谐量分流：
   - 近共振（|Δ| < max(5 GHz, Rabi)）: dressed 两能级精确解
       源态(低能)位移 = (1/2)(sign·sqrt(Δ²+Ω²) − Δ)，目标态与之反对称。
       该式在所有失谐区成立：远失谐回微扰 Ω²/4Δ，共振给 Autler-Townes。
   - 远失谐: 非 RWA 二阶项（rotating + counter-rotating 都保留）
       ΔE = −(E0²/4ħ)·Σ |d|²·2ω_ki/(ω_ki²−ω²)
     即 ARC DynamicPolarizability 的物理。RWA 在远失谐区失效，
     只留 rotating 项会把里德堡-里德堡对位移高估 ~10⁴ 倍。
   - 失谐用 |E_k−E_g|/h 计算，保证正旋与反旋（双光子）共振都归入
     近共振分支（dressed）而不发散。

3) 跃迁频谱
   子能级间全部偶极允许跃迁，强度用爱因斯坦 A 系数：
       A = ω³/(3πε₀ħc³)·|⟨a|d|b⟩|²
   矩阵元按 m_J 分辨；低激发态用其 |m_J,m_I> 本征振幅叠加
   （m_I 由 m_F−m_J 唯一确定）。偏振 q 逐项计算，给出该峰的偏振组成。

单位约定:
   - 能级: MHz
   - 频率: Hz
   - 偶极矩阵元: 原子单位（a.u.，ARC 返回值）
   - 光强: mW/cm²（换算 W/m² 因子 10）
============================================================
"""

import numpy as np
from scipy.constants import physical_constants as pc, h, hbar, c, epsilon_0

# ---------------- 物理常数与单位换算 ----------------
# ARC 的 getEnergy 返回的是能量相对电离阈的绑定能，单位是 eV（见 ARC 源码 docstring）。
# eV -> Hz 的换算因子是 e/h。
EV_TO_HZ = pc['elementary charge'][0] / h   # Hz/eV
MU_B = pc['Bohr magneton'][0]                # J/T
MU_B_MHZ_PER_G = MU_B / h / 1e4 / 1e6        # MHz/G（1 T = 1e4 G）
E_CHARGE = pc['elementary charge'][0]        # C
A0 = pc['Bohr radius'][0]                    # m
MW_PER_CM2_TO_W_PER_M2 = 10.0                # 1 mW/cm^2 = 10 W/m^2
AC_EFFECT_THRESHOLD_MHZ = 0.05               # |AC| > 50 kHz 视为受寻址光影响
DEFAULT_NUCLEAR_SPIN_I = 1.5                  # 兜底默认：仅当 ARC 原子类无 I 属性时使用
                                              # （正常情况下从 atom.I 读取，换原子无需改）
RYDBERG_N_MIN = 20                            # n >= 20 视为里德堡态（超精细可忽略）
AC_FAR_DETUNE_CUTOFF_HZ = 5e9                # |Δ| >= max(此值, Rabi) 视为远失谐 → SOS 背景
SOS_BASIS_N_MIN = 5                           # 远失谐 SOS 基组的 n 下限
SOS_BASIS_N_MAX = 100                         # 远失谐 SOS 基组的 n 上限

POLARIZATION_LABELS = ['σ+', 'π', 'σ-']
POLARIZATION_Q = {'σ+': 1, 'π': 0, 'σ-': -1}

_SPECTROSCOPIC_LETTERS = 'SPDFGHIKLMNOQRT'


def state_energy_mhz(atom, n, l, j):
    """ARC 态能量（单位 eV，相对电离阈为负）-> MHz。"""
    return atom.getEnergy(n, l, j) * EV_TO_HZ / 1e6


def transition_frequency_hz(atom, n1, l1, j1, n2, l2, j2):
    """跃迁频率 (Hz)。

    ARC 的 getEnergy 返回单位是 eV，因此用 (E2 - E1) * e / h 即可得到正确的
    频率。这与 ARC 自带的 getTransitionFrequency 结果一致（已验证：
    60S->60P 两者都给出 16.826667 GHz）。此处保留独立实现便于核查。
    """
    return (atom.getEnergy(n2, l2, j2) - atom.getEnergy(n1, l1, j1)) * EV_TO_HZ


def validate_state(n, l, j, s=0.5):
    """校验 (n, l, j) 是否为合法原子态，返回 (ok, 错误信息)。"""
    if n < 1:
        return False, 'n 必须 >= 1'
    if l < 0 or l >= n:
        return False, f'l 必须满足 0 <= l < n={n}'
    if abs(2 * j - round(2 * j)) > 1e-9:
        return False, 'j 必须是半整数（如 0.5, 1.5, ...）'
    if not (abs(l - s) - 1e-9 <= j <= l + s + 1e-9):
        return False, f'j 必须在 [|l-s|, l+s] = [{abs(l - s):.1f}, {l + s:.1f}] 内'
    return True, ''


def _j_frac(j):
    """把 j 转成字符串：0.5->'1/2'，1.5->'3/2'，整数 j->'1' 等。"""
    if abs(j - round(j)) < 1e-9:
        return f'{int(round(j))}'
    return f'{int(round(2 * j))}/2'


def auto_label(n, l, j):
    """自动生成能级标签，如 60S_{1/2}、6P_{3/2}、5D_{5/2}。"""
    if l < len(_SPECTROSCOPIC_LETTERS):
        letter = _SPECTROSCOPIC_LETTERS[l]
    else:
        letter = str(l)
    return f'{n}{letter}_{{{_j_frac(j)}}}'


def _sos_basis_range(n):
    """远失谐 SOS 基组的 n 范围（按当前态类型分段）。

    - 里德堡态（n>=20）：只取里德堡伙伴 [20, 100]。低激发伙伴（如 6P）
      对里德堡态的矩阵元极小，寻址对由"已添加组"分支显式处理。
    - 低激发态：只取近邻低态 [5, 19]，远里德堡伙伴贡献可忽略。
    这样避免大量慢速 Numerov 波函数计算，又不丢失重要物理贡献。
    """
    if n >= RYDBERG_N_MIN:
        return RYDBERG_N_MIN, SOS_BASIS_N_MAX
    return SOS_BASIS_N_MIN, RYDBERG_N_MIN - 1


_dipole_partners_cache = {}


def dipole_partners(n, l, j):
    """(n, l, j) 态的全部偶极耦合伙伴 (n', l', j')，用于远失谐 SOS 背景。"""
    key = (n, l, j)
    out = _dipole_partners_cache.get(key)
    if out is None:
        n_min, n_max = _sos_basis_range(n)
        out = []
        for l1 in (l - 1, l + 1):
            if l1 < 0:
                continue
            for j1 in (l1 - 0.5, l1 + 0.5):
                if j1 < 0.5 or abs(j - j1) > 1.0 + 1e-9:
                    continue
                for n1 in range(max(n_min, l1 + 1), n_max + 1):
                    out.append((n1, l1, j1))
        _dipole_partners_cache[key] = out
    return out


class ArcCache:
    """缓存不随磁场/频率/视野变化的 ARC 数据，加速滑块实时刷新。

    - 态能量、Landé g 因子、超精细常数：首次访问时向 ARC 查询后缓存
    - 偶极矩阵元：不依赖磁场/频率/视野，按 (态1, 态2, q) 缓存
    只有添加/删除能级组才会引入新的键；磁场、频率、视野变化不会触发 ARC 重算。
    """

    def __init__(self, atom):
        self.atom = atom
        # 核自旋 I 与核 g 因子直接取自 ARC 原子实例（atom.I / atom.gI）。
        # 换原子（如 Rb85/Cs/Li/Na/K）时只需换 atom 对象，无需改本库代码。
        self.I = float(getattr(atom, 'I', DEFAULT_NUCLEAR_SPIN_I))
        self.gI = float(getattr(atom, 'gI', 0.0))
        self._state = {}   # (n,l,j) -> dict(E_mhz, gJ, A_hz, B_hz)
        self._gj = {}      # (l,j) -> gJ
        self._dme = {}     # (n1,l1,j1,m1,n2,l2,j2,m2,q) -> d_au

    def gJ(self, l, j):
        key = (l, j)
        g = self._gj.get(key)
        if g is None:
            g = self.atom.getLandegjExact(l, j, 0.5)
            self._gj[key] = g
        return g

    def _info(self, n, l, j):
        key = (n, l, j)
        info = self._state.get(key)
        if info is None:
            A_hz = B_hz = None
            if n < RYDBERG_N_MIN:
                try:
                    A_hz, B_hz = self.atom.getHFSCoefficients(n, l, j)
                except ValueError:
                    pass
            info = {
                'E_mhz': self.atom.getEnergy(n, l, j) * EV_TO_HZ / 1e6,
                'gJ': self.gJ(l, j),
                'A_hz': A_hz,
                'B_hz': B_hz,
            }
            self._state[key] = info
        return info

    def E_mhz(self, n, l, j):
        return self._info(n, l, j)['E_mhz']

    def hfs(self, n, l, j):
        """返回 (A_hz, B_hz)；若为里德堡态或无超精细数据返回 None。"""
        info = self._info(n, l, j)
        if info['A_hz'] is None:
            return None
        return info['A_hz'], info['B_hz']

    def dme(self, n1, l1, j1, m1, n2, l2, j2, m2, q):
        key = (n1, l1, j1, m1, n2, l2, j2, m2, q)
        d = self._dme.get(key)
        if d is None:
            d = self.atom.getDipoleMatrixElement(n1, l1, j1, m1, n2, l2, j2, m2, q)
            self._dme[key] = d
        return d

    def transition_frequency_hz(self, n1, l1, j1, n2, l2, j2):
        # E_mhz 差 × 1e6 = Hz
        return (self.E_mhz(n2, l2, j2) - self.E_mhz(n1, l1, j1)) * 1e6


def _candidate_F(j, I, mF):
    """给定 J、I、mF，返回所有可能的 F（|J-I| <= F <= J+I 且 F >= |mF|）。"""
    return [F for F in np.arange(abs(j - I), j + I + 1, 1.0) if F >= abs(mF) - 1e-9]


def get_hyperfine_levels(cache, n, l, j, B_gauss):
    """低激发态的超精细子能级（含磁场下的 Zeeman 效应）。

    在 |m_J, m_I> 基下按 m_F 分块数值对角化超精细 + Zeeman 哈密顿量：
        H = A·I·J + B·Q + mu_B·B·(g_J J_z + g_I I_z)
    核自旋 I 与核 g 因子 g_I 取自 ARC 原子实例（atom.I / atom.gI），
    换原子时无需改动本函数。
    返回 [(F, m_F, E_mhz, mJ_weights, mJ_amps)]：
    - mJ_weights: {m_J: 权重}，各电子 m_J 分量占比（对 m_I 求和）
    - mJ_amps   : {m_J: 复振幅}（m_I = m_F − m_J 由分块唯一确定），
                  供跃迁偶极矩阵元计算用
    若是里德堡态（n 很大或 ARC 无超精细数据）返回 None，表示忽略核自旋。
    """
    hfs = cache.hfs(n, l, j)
    if hfs is None:
        return None  # 里德堡态或无超精细数据：忽略核自旋

    A_hz, B_hz = hfs
    I = cache.I          # 核自旋，来自 ARC 原子实例 atom.I
    gJ = cache.gJ(l, j)
    gI = cache.gI        # 核 g 因子，来自 ARC 原子实例 atom.gI
    mJ_list = np.arange(-j, j + 1, 1.0)
    mI_list = np.arange(-I, I + 1, 1.0)
    states = [(float(mj), float(mi)) for mj in mJ_list for mi in mI_list]
    idx = {s: k for k, s in enumerate(states)}
    N = len(states)

    # I·J 矩阵（含升降算符项）。基为 (m_J, m_I)：
    #   I+ J-: (m_J, m_I) -> (m_J-1, m_I+1)
    #   I- J+: (m_J, m_I) -> (m_J+1, m_I-1)
    IJ = np.zeros((N, N))
    for k, (mj, mi) in enumerate(states):
        IJ[k, k] = mj * mi
        ci = I * (I + 1) - mi * (mi + 1)      # I+ 从 mi 升到 mi+1
        cj = j * (j + 1) - mj * (mj - 1)      # J- 从 mj 降到 mj-1
        t = (mj - 1, mi + 1)
        if ci > 0 and cj > 0 and t in idx:
            IJ[k, idx[t]] += 0.5 * np.sqrt(ci * cj)
        ci = I * (I + 1) - mi * (mi - 1)      # I- 从 mi 降到 mi-1
        cj = j * (j + 1) - mj * (mj + 1)      # J+ 从 mj 升到 mj+1
        t = (mj + 1, mi - 1)
        if ci > 0 and cj > 0 and t in idx:
            IJ[k, idx[t]] += 0.5 * np.sqrt(ci * cj)

    H_hf_hz = A_hz * IJ
    # 电四极项（B 常数），仅对 J > 1/2 且 I > 1/2 存在
    if abs(B_hz) > 1e-30 and I > 0.5 and j > 0.5:
        denom = 2.0 * I * (2 * I - 1) * j * (2 * j - 1)
        if abs(denom) > 1e-30:
            IJ2 = IJ @ IJ
            Q = 3.0 * IJ2 + 1.5 * IJ - I * (I + 1) * j * (j + 1) * np.eye(N)
            H_hf_hz = H_hf_hz + B_hz * Q / denom

    H_hf_mhz = H_hf_hz / 1e6
    H_z_mhz = np.zeros((N, N))
    for k, (mj, mi) in enumerate(states):
        H_z_mhz[k, k] = MU_B_MHZ_PER_G * B_gauss * (gJ * mj + gI * mi)

    mF_list = sorted(set(mj + mi for mj, mi in states))
    results = []
    for mF in mF_list:
        sub = [k for k, (mj, mi) in enumerate(states) if abs((mj + mi) - mF) < 1e-9]
        # 零场本征态（用于标注每个本征态的 F）
        evals0, evecs0 = np.linalg.eigh(H_hf_mhz[np.ix_(sub, sub)])
        F_cand = _candidate_F(j, I, mF)
        E_F = [A_hz / 2.0 * (F * (F + 1) - I * (I + 1) - j * (j + 1)) / 1e6 for F in F_cand]
        order0 = np.argsort(evals0)
        orderF = np.argsort(E_F)
        F_by_idx = {int(k): F_cand[int(orderF[pos])] for pos, k in enumerate(order0)}
        # 加磁场后对角化
        H_tot = H_hf_mhz[np.ix_(sub, sub)] + H_z_mhz[np.ix_(sub, sub)]
        evals, evecs = np.linalg.eigh(H_tot)
        for k in range(len(evals)):
            ov = np.abs(evecs[:, k] @ evecs0)
            best = int(np.argmax(ov))
            F = F_by_idx.get(best, F_cand[0])
            # 本征态在 |m_J, m_I> 基下的 m_J 分量权重（对 m_I 求和）与复振幅
            # （m_I = m_F − m_J 由分块唯一确定，供跃迁偶极矩阵元计算用）
            mJ_weights = {}
            mJ_amps = {}
            for pos, idx_s in enumerate(sub):
                mj_s, mi_s = states[idx_s]
                c = evecs[pos, k]
                mJ_weights[mj_s] = mJ_weights.get(mj_s, 0.0) + abs(c) ** 2
                mJ_amps[mj_s] = complex(c)
            results.append((float(F), float(mF), float(evals[k]), mJ_weights, mJ_amps))
    return results


# ============================================================
# AC-Stark
# ============================================================

def _sublevel_energy(cache, n, l, j, m, B_gauss):
    """里德堡 mJ 子能级的裸能量（MHz，fine + Zeeman）。"""
    return cache.E_mhz(n, l, j) + cache.gJ(l, j) * MU_B_MHZ_PER_G * m * B_gauss


def _dressed_shift_mhz(delta_rad, omega_rabi, is_source):
    """dressed 两能级位移 (MHz)。

    delta_rad = 2π·失谐 (rad/s)，omega_rabi = Rabi (rad/s)。
    源态(低能)位移 = ½(sign·√(Δ²+Ω²) − Δ)，目标态与之严格反对称。
    远失谐回微扰 Ω²/4Δ，共振给 Autler-Townes（±Ω/2）。
    """
    sign_delta = 1.0 if delta_rad >= 0 else -1.0
    sq = np.sqrt(delta_rad ** 2 + omega_rabi ** 2)
    if is_source:
        shift_rad = 0.5 * (sign_delta * sq - delta_rad)
    else:
        shift_rad = 0.5 * (delta_rad - sign_delta * sq)
    return shift_rad / (2.0 * np.pi) / 1e6


def _partner_sublevels(cache, n1, l1, j1, m2, B_gauss):
    """伙伴态 (n1,l1,j1) 中 mJ=m2 分量的子能级列表 [(E_mhz, weight)]。

    - 里德堡态：单一子能级，能量 = fine + Zeeman(m2)，权重 1。
    - 低激发态：遍历超精细子能级 (F,mF)，取 mJ=m2 分量振幅非零者，
      能量 = fine + e_hf，权重 = |振幅|²/(2I+1)。

    归一化说明：当前态若为 mJ 分量（核自旋未定），m2 分量在伙伴超精细
    基里分布在 (2I+1) 个 mI 通道（Σ|amp|² = 2I+1），dressed/SOS 应取
    核自旋平均，故权重除以 (2I+1)。否则（如 6S↔6P 的远失谐对拍）会
    高估 (2I+1) 倍。
    """
    hf = get_hyperfine_levels(cache, n1, l1, j1, B_gauss)
    if hf is None:
        return [(_sublevel_energy(cache, n1, l1, j1, m2, B_gauss), 1.0)]
    norm = 2.0 * cache.I + 1.0
    out = []
    for F, mF, e_hf, w, amps in hf:
        am = amps.get(m2)
        if am is not None and abs(am) > 1e-12:
            out.append((cache.E_mhz(n1, l1, j1) + e_hf, abs(am) ** 2 / norm))
    return out


def pair_contribution_mhz(cache, n, l, j, m, n1, l1, j1, q,
                          E0_sq, omega, freq, E_i_sub, B_gauss, address_cfg):
    """单个偶极伙伴 (n1,l1,j1) 对当前子能级 (n,l,j,m) 的 AC 位移 (MHz)。

    失谐用**子能级能量**（含超精细/Zeeman）算到最精细：
    - 当前态能量 E_i_sub（MHz，fine + 超精细 + Zeeman）
    - 伙伴按 mJ=m+q 分量分解到子能级（里德堡单值；低激发超精细各 F 态
      按 |振幅|² 加权），每个子能级有各自失谐与 Rabi。
    近共振（|Δ| < max(AC_FAR_DETUNE_CUTOFF_HZ, Rabi)）→ dressed 精确解；
    远失谐 → 非 RWA 二阶项（rotating + counter-rotating）。
    """
    m2 = m + q
    if abs(m2) > j1 + 1e-9:
        return 0.0
    d_au = cache.dme(n, l, j, m, n1, l1, j1, m2, q)
    if abs(d_au) < 1e-12:
        return 0.0
    d_cm = abs(d_au) * E_CHARGE * A0
    omega_rabi = np.sqrt(E0_sq) * d_cm / hbar            # fine 层面 Rabi (rad/s)
    total = 0.0
    for E_k, weight in _partner_sublevels(cache, n1, l1, j1, m2, B_gauss):
        nu_gk = (E_k - E_i_sub) * 1e6                    # Hz，带符号（子能级能量差）
        delta = freq - abs(nu_gk)
        if abs(delta) < max(AC_FAR_DETUNE_CUTOFF_HZ, omega_rabi / (2.0 * np.pi)):
            # 近共振：dressed 精确解。耦合 Rabi = Ω_fine·|振幅|（=√weight）
            om_f = omega_rabi * (weight ** 0.5)
            total += _dressed_shift_mhz(
                2.0 * np.pi * delta, om_f, is_source=(E_i_sub < E_k))
        else:
            # 远失谐：非 RWA 二阶项（|d|² 按权重，ω_ki 用子能级能量差）。
            # dE_j 量纲是能量 (J)，换算 MHz 除以 h = 2π·ħ
            # （曾缺 ħ，导致 SOS 背景被低估 ~1e34 倍）。
            w_ki = 2.0 * np.pi * nu_gk
            dE_j = -(E0_sq / (4.0 * hbar)) * (weight * d_cm ** 2) * (2.0 * w_ki) / (w_ki ** 2 - omega ** 2)
            total += dE_j / (2.0 * np.pi * hbar) / 1e6
    return total


def ac_stark_shift_mhz(cache, group, m, groups, address_cfg, B_gauss=0.0, E_sub=None):
    """子能级总 AC-Stark 位移 (MHz) = 近共振 dressed 精确解 + 远失谐 SOS 背景。

    当前子能级：能级组的 mJ 分量 m。E_sub 为其子能级能量（MHz，
    含超精细/Zeeman）；None 时按里德堡 mJ 计算（fine + g_J·μ_B·B·m）。
    dressed/SOS 失谐一律用子能级能量差（最精细），与 collect_transitions
    显示一致。

    遍历两类偶极伙伴：
    1) 已添加的能级组（可跨越基组 n 范围，如 6P↔60S 寻址对）；
    2) 该态偶极耦合的完整基组（未添加的里德堡/低激发态）。
    """
    if not address_cfg.get('enabled', False):
        return 0.0

    q = POLARIZATION_Q[address_cfg['polarization']]
    n, l, j = group['n'], group['l'], group['j']
    if E_sub is None:
        E_sub = _sublevel_energy(cache, n, l, j, m, B_gauss)
    freq = address_cfg['frequency_hz']
    I_wm2 = address_cfg['intensity'] * MW_PER_CM2_TO_W_PER_M2
    E0_sq = 2.0 * I_wm2 / (epsilon_0 * c)                # V²/m²
    omega = 2.0 * np.pi * freq                           # rad/s
    total = 0.0

    # 1) 已添加组之间的全部耦合（近 → dressed，远 → SOS 项）
    for k in groups:
        if (k['n'], k['l'], k['j']) == (n, l, j):
            continue
        total += pair_contribution_mhz(
            cache, n, l, j, m, k['n'], k['l'], k['j'], q,
            E0_sq, omega, freq, E_sub, B_gauss, address_cfg)

    # 2) 完整偶极基组中未添加的伙伴（远失谐背景 + 未添加的近共振态）
    added = {(k['n'], k['l'], k['j']) for k in groups}
    for (n1, l1, j1) in dipole_partners(n, l, j):
        if (n1, l1, j1) == (n, l, j) or (n1, l1, j1) in added:
            continue
        total += pair_contribution_mhz(
            cache, n, l, j, m, n1, l1, j1, q,
            E0_sq, omega, freq, E_sub, B_gauss, address_cfg)

    return total


# ============================================================
# 子能级行（UI 直接渲染）
# ============================================================

def build_level_rows(cache, field_B, groups, address_cfg, visible):
    """计算可见能级组的全部子能级，返回行 dict 列表（UI 直接绘制）。

    每行字段：
      x        : m_J 或 m_F（横轴坐标）
      E0       : 裸能量 (MHz)，含 Zeeman，不含 AC
      E        : 移后能量 (MHz) = E0 + ac
      ac       : AC-Stark 位移 (MHz)
      affected : |ac| > AC_EFFECT_THRESHOLD_MHZ
      label, qstr, color : 显示用
      n, l, j, kind('mJ'/'F'), mj 或 mF/amps : 供跃迁频谱计算
    里德堡态忽略核自旋；低激发态含超精细，AC 位移 = 各 m_J 分量
    （超精细本征振幅）的 dressed/SOS 加权和。
    """
    rows = []
    for group in visible:
        n, l, j = group['n'], group['l'], group['j']
        color = group.get('color', '#1f77b4')
        E0_mhz = cache.E_mhz(n, l, j)
        hf = get_hyperfine_levels(cache, n, l, j, field_B)
        if hf is None:
            # 里德堡态：只画 fine-structure 的 m_J 子能级（忽略核自旋）
            gJ = cache.gJ(l, j)
            for m in np.arange(-j, j + 1, 1.0):
                zeeman = gJ * MU_B_MHZ_PER_G * m * field_B
                ac = ac_stark_shift_mhz(cache, group, m, groups, address_cfg,
                                        B_gauss=field_B)
                affected = abs(ac) > AC_EFFECT_THRESHOLD_MHZ
                E_level = E0_mhz + zeeman
                rows.append(dict(
                    x=float(m), E0=E_level, E=E_level + ac, ac=ac, color=color,
                    label=group['label'], affected=affected,
                    qstr=f'm_J={m:+.1f}',
                    n=n, l=l, j=j, kind='mJ', mj=m, mF=None, amps=None))
        else:
            # 低激发态：画出所有超精细子能级 (F, m_F)
            for F, mF, e_hf, mJ_weights, mJ_amps in hf:
                E_sub_level = E0_mhz + e_hf   # 超精细子能级能量（含 Zeeman）
                ac = sum(
                    w * ac_stark_shift_mhz(cache, group, mj, groups, address_cfg,
                                           B_gauss=field_B, E_sub=E_sub_level)
                    for mj, w in mJ_weights.items())
                affected = abs(ac) > AC_EFFECT_THRESHOLD_MHZ
                E_level = E0_mhz + e_hf
                rows.append(dict(
                    x=float(mF), E0=E_level, E=E_level + ac, ac=ac, color=color,
                    label=group['label'], affected=affected,
                    qstr=f'F={F:.1f}, m_F={mF:+.1f}',
                    n=n, l=l, j=j, kind='F', mj=None, mF=mF, amps=mJ_amps))
    return rows


# ============================================================
# 跃迁与频谱
# ============================================================

def _group_sublevel_rows(cache, n, l, j, B_gauss=0.0):
    """构造单个能级组的子能级行（含超精细/Zeeman 能量），供最近跃迁对查找。"""
    rows = []
    hf = get_hyperfine_levels(cache, n, l, j, B_gauss)
    if hf is None:
        gJ = cache.gJ(l, j)
        for m in np.arange(-j, j + 1, 1.0):
            e = cache.E_mhz(n, l, j) + gJ * MU_B_MHZ_PER_G * m * B_gauss
            rows.append(dict(n=n, l=l, j=j, kind='mJ', mj=float(m), mF=None,
                             amps=None, E_mhz=float(e), qstr=f'm_J={m:+.1f}'))
    else:
        for F, mF, e_hf, w, amps in hf:
            e = cache.E_mhz(n, l, j) + e_hf
            rows.append(dict(n=n, l=l, j=j, kind='F', mj=None, mF=mF, amps=amps,
                             E_mhz=float(e), qstr=f'F={F:.1f}, m_F={mF:+.1f}'))
    return rows


def _closest_sublevel_pair(cache, gi, gk, q, E0, B_gauss, freq_hz):
    """返回 gi/gk 间与当前频率失谐最小的耦合子能级对。

    返回 (qstr_i, qstr_k, det_hz, omega_hz)：
    - det 为该子能级对失谐（ν_L − ν_pair，ν_pair 含超精细/Zeeman 能量）
    - omega 为该子能级对的 Rabi (rad/s)
    无耦合返回 None。超精细态 qstr 含 F/m_F，里德堡态为 m_J。
    """
    rows_i = _group_sublevel_rows(cache, gi['n'], gi['l'], gi['j'], B_gauss)
    rows_k = _group_sublevel_rows(cache, gk['n'], gk['l'], gk['j'], B_gauss)
    best, best_abs = None, float('inf')
    for a in rows_i:
        for b in rows_k:
            s = transition_strength_q(cache, a, b, q)
            if s < 1e-16:
                continue
            nu_pair = (b['E_mhz'] - a['E_mhz']) * 1e6       # Hz（gi 低能 → gk 高能）
            det = freq_hz - nu_pair
            if abs(det) < best_abs:
                best_abs = abs(det)
                d_cm = (s ** 0.5) * E_CHARGE * A0           # 该子能级对有效 d
                omega = E0 * d_cm / hbar
                best = (a['qstr'], b['qstr'], det, omega)
    return best


def collect_transitions(cache, groups, address_cfg, field_B=0.0):
    """遍历所有能级组对，返回耦合对的物理信息列表。

    每项 (label_src, qstr_src, label_tgt, qstr_tgt, det_hz, omega_hz)：
    - qstr_src/qstr_tgt：该组对中**失谐最小**的耦合子能级对
      （超精细态含 F、m_F；里德堡态为 m_J），即与当前寻址光频率
      最匹配的跃迁对（而非矩阵元最强）。
    - det_hz：该子能级对的失谐（含超精细/Zeeman 能量）。
    - omega_hz：该子能级对的 Rabi (rad/s)。
    仅统计至少存在一个非零偶极耦合的跃迁对。
    """
    pairs = []
    if not address_cfg.get('enabled', False) or len(groups) < 2:
        return pairs

    q = POLARIZATION_Q[address_cfg['polarization']]
    I_wm2 = address_cfg['intensity'] * MW_PER_CM2_TO_W_PER_M2
    E0 = (2.0 * I_wm2 / (epsilon_0 * c)) ** 0.5
    freq = address_cfg['frequency_hz']

    for i in range(len(groups)):
        for k in range(i + 1, len(groups)):
            gi, gk = groups[i], groups[k]
            # 恒以能量低的态为源（gi）、能量高的态为目标（gk）
            if cache.E_mhz(gi['n'], gi['l'], gi['j']) > cache.E_mhz(gk['n'], gk['l'], gk['j']):
                gi, gk = gk, gi
            sub = _closest_sublevel_pair(cache, gi, gk, q, E0, field_B, freq)
            if sub is None:
                continue
            qstr_i, qstr_k, det, omega = sub
            pairs.append((gi['label'], qstr_i, gk['label'], qstr_k, det, omega))
    return pairs


def transition_strength_q(cache, a, b, q):
    """子能级对 (a,b) 在偏振 q 下的偶极强度 |<a|d_q|b>|²。

    偶极算符只作用于电子（Δm_J = q），因此耦合矩阵元必须命中对方
    子能级的 mJ：mJ-mJ 需 m2 == b['mj']；F-mJ 需 mj+q == b['mj']；
    mJ-F 需 mj+q == a['mj']。缺此匹配会把"与其他 mJ 态的耦合"错误
    归到本对（实测 bug：6P3/2 F=3 mF=-1 ↔ 60S mJ=+0.5 在 σ- 下被
    误报非零，实际是到 60S mJ=-0.5 的耦合）。
    """
    if a['kind'] == 'mJ' and b['kind'] == 'mJ':
        m2 = a['mj'] + q
        if abs(m2 - b['mj']) > 1e-9:
            return 0.0
        d = cache.dme(a['n'], a['l'], a['j'], a['mj'],
                      b['n'], b['l'], b['j'], m2, q)
        return abs(d) ** 2
    elif a['kind'] == 'F' and b['kind'] == 'mJ':
        amp = 0.0j
        for mj, am in a['amps'].items():
            m2 = mj + q
            if abs(m2 - b['mj']) > 1e-9:
                continue
            amp += am.conjugate() * cache.dme(
                a['n'], a['l'], a['j'], mj, b['n'], b['l'], b['j'], m2, q)
        return abs(amp) ** 2
    elif a['kind'] == 'mJ' and b['kind'] == 'F':
        amp = 0.0j
        for mj, am in b['amps'].items():
            m1 = mj + q
            if abs(m1 - a['mj']) > 1e-9:
                continue
            amp += am.conjugate() * cache.dme(
                b['n'], b['l'], b['j'], mj, a['n'], a['l'], a['j'], m1, q)
        return abs(amp) ** 2
    else:  # F -- F（两个超精细子能级）
        # 偶极算符只作用于电子（J），核自旋 m_I 不变 ⟺ m_F 守恒：
        # mF_b = mF_a + q。缺此检查会给出违反选择定则的非零强度
        # （交叉验证发现：ARC getDipoleMatrixElementHFS 对不满足者返回 0）。
        if abs((a['mF'] + q) - b['mF']) > 1e-9:
            return 0.0
        amp = 0.0j
        for mj1, am1 in a['amps'].items():
            mj2 = mj1 + q
            if abs(mj2) <= b['j'] + 1e-9:
                am2 = b['amps'].get(mj2)
                if am2 is not None:
                    amp += am1.conjugate() * am2 * cache.dme(
                        a['n'], a['l'], a['j'], mj1,
                        b['n'], b['l'], b['j'], mj2, q)
        return abs(amp) ** 2
    return 0.0


def spectrum_lines(cache, rows, ac_on):
    """计算子能级之间全部偶极允许跃迁的频谱线。

    返回 [(freq_hz, A, label_a, label_b, qstr_a, qstr_b, color_a, pol), ...]：
    - freq: 跃迁频率 (Hz)，ac_on=True 时用 AC 移后能量
    - A   : 爱因斯坦 A 系数 (s^-1)，作为谱线强度
    - qstr_a/qstr_b: 两子能级描述（供峰表显示）
    - pol : 产生该峰的偏振（σ+/π/σ-，可能多个）
    矩阵元按 m_J 分辨计算：里德堡态为纯 m_J；低激发态超精细子能级
    用其 |m_J,m_I> 本征振幅叠加（m_I 由 m_F−m_J 确定）。
    """
    lines = []
    for i in range(len(rows)):
        for k in range(i + 1, len(rows)):
            a, b = rows[i], rows[k]
            if (a['n'], a['l'], a['j']) == (b['n'], b['l'], b['j']):
                continue
            S = 0.0
            qs = []
            for q in (-1, 0, 1):
                sq = transition_strength_q(cache, a, b, q)
                if sq > 1e-16:
                    S += sq
                    qs.append(q)
            if S < 1e-16:
                continue
            Ea = a['E'] if ac_on else a['E0']
            Eb = b['E'] if ac_on else b['E0']
            nu = abs(Eb - Ea) * 1e6                          # Hz
            A = ((2 * np.pi * nu) ** 3 / (3 * np.pi * epsilon_0 * hbar * c ** 3)
                 * S * (E_CHARGE * A0) ** 2)
            pol = ','.join('σ+' if q == 1 else ('π' if q == 0 else 'σ-') for q in qs)
            lines.append((nu, A, a['label'], b['label'], a['qstr'], b['qstr'],
                          a['color'], pol))
    return lines
