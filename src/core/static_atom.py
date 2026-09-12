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

4) 频移计算后端（build_level_rows，唯一后端 = direct）
   - 'direct'：Floquet 直接矩阵对角化（见 direct_ac_stark_shifts）——一次
     对角化得全部子能级位移，自洽包含近共振 dressed + 远失谐反旋项
     (rotating+counter-rotating) + 多态同时耦合 + 超精细相干性；
     仅用 numpy（np.linalg.eigh），无需 GPU。基组自动纳入每个已添加组
     的偶极伙伴（|Δn|≤DIRECT_SOS_N_WINDOW），保证亚 MHz 精度。
     （legacy「逐伙伴解析/微扰」后端已于 2025-08 删除——全部静态任务
       交给第一性原理。）

单位约定:
   - 能级: MHz
   - 频率: Hz
   - 偶极矩阵元: 原子单位（a.u.，ARC 返回值）
   - 光强: mW/cm²（换算 W/m² 因子 10）
============================================================
"""

import numpy as np
from scipy.constants import physical_constants as pc, h, hbar, c, epsilon_0
from . import params as _P
# t80：删两个本模块**自己不用**的 AC 分流常量导入：
#   · `AC_FAR_DETUNE_CUTOFF_HZ` —— 只是 `AC_NEAR_CUTOFF_MHZ` 的 Hz 版定义式，
#     仓库内无第二消费者；
#   · `AC_NEAR_CUTOFF_MHZ` —— 本模块不引用，但 `field_build.real_field` 曾
#     经 `static_atom.AC_NEAR_CUTOFF_MHZ` **绕道**取它（把本模块当转发站）。
#     已改为 field_build 直连 params 单源，故这里可以清掉。

# ---------------- 物理常数与单位换算 ----------------
# ARC 的 getEnergy 返回的是能量相对电离阈的绑定能，单位是 eV（见 ARC 源码 docstring）。
# eV -> Hz 的换算因子是 e/h。
EV_TO_HZ = pc['elementary charge'][0] / h   # Hz/eV
MU_B = pc['Bohr magneton'][0]                # J/T
MU_B_MHZ_PER_G = MU_B / h / 1e4 / 1e6        # MHz/G（1 T = 1e4 G）
E_CHARGE = pc['elementary charge'][0]        # C
A0 = pc['Bohr radius'][0]                    # m
MW_PER_CM2_TO_W_PER_M2 = 10.0                # 1 mW/cm^2 = 10 W/m^2
DEFAULT_NUCLEAR_SPIN_I = 1.5                  # 兜底默认：仅当 ARC 原子类无 I 属性时使用
                                              # （正常情况下从 atom.I 读取，换原子无需改）
SOS_BASIS_N_MIN = 5                           # 远失谐 SOS 基组的 n 下限
SOS_BASIS_N_MAX = 100                         # 远失谐 SOS 基组的 n 上限

POLARIZATION_Q = {'σ+': 1, 'π': 0, 'σ-': -1}


def elliptical_coeffs(alpha_deg, theta_deg, eps_deg):
    """任意椭偏光 → 球基复系数 (c_{+1}, c_0, c_{-1})，满足 Σ|c|²=1。

    参数（度）：
      alpha_deg : 传播方向与量子化轴 z 的夹角（0~90°）。α=0 → 传播沿 z，
                  电场在 xy 平面（只有 σ± 分量、无 π）；α=90° → 传播垂直 z。
      theta_deg : 偏振椭圆主轴在垂直于传播方向平面内的方位角（0~180°）。
      eps_deg   : 椭偏角（−45°~+45°）。0=线偏，±45°=圆偏（正/负旋）。

    约定：传播 k=(sinα, 0, cosα)，垂直基 e1=(cosα,0,−sinα)、e2=(0,1,0)；
    球基 d_{+1}=−(d_x+id_y)/√2、d_0=d_z、d_{−1}=(d_x−id_y)/√2。
    特例：α=0,θ=0,ε=0 → x 线偏（σ± 等权）；α=0,ε=45° → σ⁺ 圆偏；
          α=90°,θ=0,ε=0 → 纯 π（电场沿 z）。
    """
    a = np.deg2rad(alpha_deg)
    t = np.deg2rad(theta_deg)
    e = np.deg2rad(eps_deg)
    ca, sa = np.cos(a), np.sin(a)
    # Jones 矢量（e1/e2 基上的复振幅）
    A = np.cos(e) * np.cos(t) + 1j * np.sin(e) * np.sin(t)
    B = np.cos(e) * np.sin(t) - 1j * np.sin(e) * np.cos(t)
    Ex, Ey, Ez = ca * A, B, -sa * A
    cp = -(Ex + 1j * Ey) / np.sqrt(2)
    c0 = Ez
    cm = (Ex - 1j * Ey) / np.sqrt(2)
    norm = (abs(cp) ** 2 + abs(c0) ** 2 + abs(cm) ** 2) ** 0.5
    if norm < 1e-12:
        return 0.0j, 0.0j, 0.0j
    return cp / norm, c0 / norm, cm / norm


def address_pol_coeffs(address_cfg):
    """返回寻址光在球基上的复系数 [(q, c_q), ...]（仅非零分量）。

    - address_cfg 含 'elliptical'（dict: alpha_deg/theta_deg/eps_deg）→ 任意椭偏；
    - 否则按 'polarization'（σ+/π/σ-）单 q 系数。
    满足 Σ|c_q|² = 1。所有物理量（AC 位移、Rabi、谱线强度）据此叠加。
    """
    ell = address_cfg.get('elliptical')
    if ell:
        cp, c0, cm = elliptical_coeffs(
            float(ell.get('alpha_deg', 0.0)),
            float(ell.get('theta_deg', 0.0)),
            float(ell.get('eps_deg', 0.0)))
        return [(1, cp), (0, c0), (-1, cm)]
    q = POLARIZATION_Q.get(address_cfg.get('polarization', 'π'), 0)
    return [(q, 1.0)]

# ---------------- 频移计算后端 ----------------
# AC-Stark 唯一后端：direct（Floquet 直接矩阵对角化，第一性原理）。
# 一次对角化得到全部子能级 AC 位移，自洽包含近共振 dressed + 远失谐反旋项
# + 多态耦合 + 超精细相干性；用 numpy 的 np.linalg.eigh，无需 GPU。
# （legacy「逐伙伴解析/微扰」后端已于 2025-08 删除——全部静态任务交给
#  第一性原理 direct。）
# 直接后端的自动基组窗口：每个已添加组自动纳入其偶极伙伴中 |Δn|≤此值的态。
# （|Δn|=0 的同 n 细结构伙伴，如 60P3/2 距 60P1/2 仅 ~461 MHz，也会被纳入；
#  保证亚 MHz 精度。典型寻址强度 Ω~10~100 MHz 下默认 3 足够；
#  极强驱动/宽扫描可调大。）

_SPECTROSCOPIC_LETTERS = 'SPDFGHIKLMNOQRT'


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
    if n >= _P.get('RYDBERG_N_MIN'):
        return _P.get('RYDBERG_N_MIN'), SOS_BASIS_N_MAX
    return SOS_BASIS_N_MIN, _P.get('RYDBERG_N_MIN') - 1


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
            if n < _P.get('RYDBERG_N_MIN'):
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
    F 标号采用**绝热约定**（Breit–Rabi 图惯例）：每个 m_F 块内对零场 F 基
    做双射指派（总 overlap 最大），每个 (F, m_F) 恰好一个态、标签随 B 连续；
    深度混合处（Zeeman≈HFS，如 6P3/2 m_F=0 在 ~21.7 G）F 只是"零场延拓"
    的近似标记，能量与 m_J 振幅仍是精确本征值/本征矢。
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
        # F 标号 = 绝热约定（Breit–Rabi 图）：块内**双射指派**，每个 B 本征态
        # 配一个不重复的零场 F（使总 overlap 最大），标签沿磁场连续。
        # 逐态 argmax 在深度混合处会碰撞（如 6P3/2 m_F=0 在 ~21.7 G 时
        # F=0/F=1 各约 50%，两个本征态都标 F=0，F=1 缺失）；双射后每个
        # (F, m_F) 恰好一个态。能量与 m_J 振幅不受影响（来自本征值/本征矢）。
        O = np.abs(evecs.conj().T @ evecs0) ** 2
        if len(evals) > 1:
            from scipy.optimize import linear_sum_assignment
            row, col = linear_sum_assignment(-O)
            assign = {int(k): int(j) for k, j in zip(row, col)}
        else:
            assign = {0: 0}
        for k in range(len(evals)):
            best = assign.get(k, int(np.argmax(O[k])))
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
# AC-Stark（唯一后端 = direct Floquet 对角化，见 direct_ac_stark_shifts）
# ============================================================

def _sublevel_energy(cache, n, l, j, m, B_gauss):
    """里德堡 mJ 子能级的裸能量（MHz，fine + Zeeman）。"""
    return cache.E_mhz(n, l, j) + cache.gJ(l, j) * MU_B_MHZ_PER_G * m * B_gauss


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


# ---------------- 真实光场换算（动态引擎复用：光强 → Rabi / 远失谐 AC） ----------------
def field_e0_sq(intensity_mwcm2):
    """光强 (mW/cm²) → 电场幅度平方 E0² (V²/m²)。"""
    return 2.0 * intensity_mwcm2 * MW_PER_CM2_TO_W_PER_M2 / (epsilon_0 * c)


def rabi_mhz(d_au, E0_sq, pol_frac=1.0):
    """偶极矩阵元 |d|（a.u.）在光强下的 Rabi 频率 (MHz，Ω/2π)。

    E0_sq: E0² (V²/m²)；pol_frac: 偏振分量系数（该 q 分量的振幅占比）。
    """
    d_cm = abs(d_au) * E_CHARGE * A0
    omega_rads = pol_frac * np.sqrt(max(E0_sq, 0.0)) * d_cm / hbar
    return omega_rads / (2.0 * np.pi) / 1e6        # MHz


def intensity_mwcm2_from_rabi(d_au, om_mhz, pol_frac=1.0):
    """给定偶极矩阵元 |d|（a.u.）要达到的 Rabi (MHz, Ω/2π) → 光强 (mW/cm²)。"""
    d_cm = abs(d_au) * E_CHARGE * A0
    if d_cm <= 0:
        return 0.0
    omega_rads = om_mhz * 1e6 * 2.0 * np.pi
    E0_sq = (omega_rads / (pol_frac * d_cm) * hbar) ** 2
    return E0_sq * epsilon_0 * c / (2.0 * 10.0)   # mW/cm²


def far_sos_shift_mhz(d_au, E0_sq, omega_rad, nu_rad):
    """单对远失谐非 RWA 二阶 AC 位移 (MHz)，对「源态」的贡献。

    nu_rad: 能级差角频率（带符号，rad/s，= 伙伴−源）；
    omega_rad: 光角频率。源低（nu>0）→ 负位移（下移），源高 → 正。
    """
    d_cm = abs(d_au) * E_CHARGE * A0
    w_ki = nu_rad
    denom = w_ki ** 2 - omega_rad ** 2
    if abs(denom) < 1e-6 * max(w_ki ** 2, omega_rad ** 2, 1.0):
        # 近共振（SOS 二阶公式发散）：该对应由 real_field 走 dressed 分支；
        # 兜底返回 0 防除零（位移应来自 dressed 精确解）。
        return 0.0
    dE = -(E0_sq / (4.0 * hbar)) * (d_cm ** 2) * (2.0 * w_ki) / denom
    return dE / (2.0 * np.pi * hbar) / 1e6          # MHz


def dc_stark_shift_mhz(cache, n, l, j, m, groups, E_vpcm, B_gauss=0.0):
    """子能级 DC-Stark 位移 (MHz)：ΔE_i = Σ_k |⟨k|d|i⟩|²E²/(E_i−E_k)。

    二阶 SOS（第一性原理）：静电场无旋转波，所有偶极中间态都贡献。
    遍历全部偶极伙伴（已添加组 + dipole_partners 未添加的），全部 mJ 通道
    （吸收 m+q / 发射 m−q 两方向，dme 参数相应交换）与核自旋平均权重。
    E_vpcm: 静电场强度 (V/cm)。

    量纲推导：ΔE(J) = Σ d_cm²·E²/(E_i−E_k)；E_i−E_k 为 J；换算 MHz 除以
    h²·1e12（h=2πħ），(E_i−E_k) 以 MHz 代入。
    """
    E = E_vpcm * 100.0                     # V/cm → V/m
    E2 = E * E
    total = 0.0
    E_i = _sublevel_energy(cache, n, l, j, m, B_gauss)
    added = {(g['n'], g['l'], g['j']) for g in groups}
    targets = [(g['n'], g['l'], g['j']) for g in groups]
    for p in dipole_partners(n, l, j):
        if p not in added:
            targets.append(p)
    h2 = (2.0 * np.pi * hbar) ** 2 * 1e12   # h²·1e12（MHz 分母换算因子）

    def add_contrib(m2, d_au):
        nonlocal total
        if abs(d_au) < 1e-12:
            return
        d_cm2 = (abs(d_au) * E_CHARGE * A0) ** 2
        for E_k, weight in _partner_sublevels(cache, n1, l1, j1, m2, B_gauss):
            if abs(E_k - E_i) < 1e-9:
                continue
            total += E2 * weight * d_cm2 / (h2 * (E_i - E_k))

    for (n1, l1, j1) in targets:
        if (n1, l1, j1) == (n, l, j):
            continue
        for q in (-1, 0, 1):
            m2 = m + q                       # 吸收方向
            if abs(m2) <= j1 + 1e-9:
                add_contrib(m2, cache.dme(n, l, j, m, n1, l1, j1, m2, q))
            m2 = m - q                       # 发射方向（dme 参数交换）
            if abs(m2) <= j1 + 1e-9:
                add_contrib(m2, cache.dme(n1, l1, j1, m2, n, l, j, m, q))
    return total


# ============================================================
# 直接矩阵对角化频移后端（Floquet，numpy，AC-Stark 唯一后端）
# ============================================================

_DIRECT_COUPLING_KEY = None
_DIRECT_COUPLING = None    # (states, E_mhz, C_unit)：单位耦合，只依赖(基组,B,偏振)，不依赖频率/强度
_DIRECT_SHIFT_CACHE = {}   # 结果缓存：键=(耦合键, n_photon, freq, I) → 位移 dict（同参免重复对角化）
_DIRECT_SHIFT_CACHE_MAX = 64
# 面板级 AC-Stark 配置（fields.py 光子块/窗口控件写入；后端缺省读取，静态+动态共用）
_AC_UI_CFG = dict(n_photon=None, sos_n_window=None)


def set_ac_stark_config(n_photon=None, sos_n_window=None):
    """设置全局 AC-Stark 精度配置（光子块阶数 / 偶极伙伴窗口）。

    供静态页面板控件调用；direct_ac_stark_shifts / direct_ac_stark_sweep
    在参数缺省（None）时读取此配置 → 静态页与动态页（real_field 经
    _floquet_ac_static）统一受控。None 表示保持当前值。
    """
    if n_photon is not None:
        _AC_UI_CFG['n_photon'] = int(n_photon)
    if sos_n_window is not None:
        _AC_UI_CFG['sos_n_window'] = int(sos_n_window)


def _occ_keys_direct(items):
    """生成唯一键 (n,l,j,qstr,occ)。

    防御性处理（2025-08 起 get_hyperfine_levels 用双射指派，同一 (F,mF)
    不再重复；occ 保留以防其它路径意外重复）。"""
    seen = {}
    keys = []
    for it in items:
        base = (it['n'], it['l'], it['j'], it['qstr'])
        occ = seen.get(base, 0)
        seen[base] = occ + 1
        keys.append(base + (occ,))
    return keys


def dipole_amplitude(cache, a, b, q):
    """⟨b|d_q|a⟩（a.u.，复）。transition_strength_q 取其模平方。

    偶极算符只作用于电子（Δm_J = q），耦合须命中对方子能级的 mJ：
    mJ-mJ 需 m2==b['mj']；F-mJ 需 mj+q==b['mj']；mJ-F 需 mj+q==a['mj']；
    F--F 需 m_F 守恒（mF_b = mF_a + q，核自旋不参与）。缺匹配会把耦合
    错归到其他 mJ 态 / 给出违反选择定则的非零强度。
    """
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


def _direct_basis(cache, groups, B_gauss, window):
    """直接法基组：全部已添加组子能级 + 每组偶极伙伴（|Δn|≤window）。

    里德堡组（n≥RYDBERG_N_MIN）也加入其偶极伙伴（如 60S 的 60P/59P/61P），
    但其中**远失谐**的里德堡↔里德堡对会在 direct_ac_stark_shifts 中按频率
    屏蔽（见该函数 docstring），既保留近共振对的真实贡献，又避免有限光子块
    截断在边界块产生虚假大位移。
    """
    states = []
    seen = set()

    def add(n, l, j):
        if (n, l, j) in seen:
            return
        seen.add((n, l, j))
        states.extend(_group_sublevel_rows(cache, n, l, j, B_gauss))

    for g in groups:
        add(g['n'], g['l'], g['j'])
    for g in groups:
        n, l, j = g['n'], g['l'], g['j']
        n_min = max(1, n - window)
        n_max = n + window
        for l1 in (l - 1, l + 1):
            if l1 < 0:
                continue
            for j1 in (l1 - 0.5, l1 + 0.5):
                if j1 < 0.5 or abs(j - j1) > 1.0 + 1e-9:
                    continue
                for n1 in range(max(n_min, l1 + 1), n_max + 1):
                    add(n1, l1, j1)
    return states


def _direct_coupling(cache, states, address_cfg):
    """构造与频率/强度无关的耦合结构。返回 (E_mhz 数组, 单位 E0 耦合 C_unit)。

    C_unit_ij = −d_ij·(e·a0)/(2h·1e6)（MHz），实际强度下 C = E0·C_unit，
    E0² = 2I/(ε0·c)。偶极选择定则预过滤（Δl=±1, |Δj|≤1）。返回单位 E0
    耦合使光强扫描可一次构建、按 sqrt(I) 重标定（见 direct_ac_stark_sweep）。
    """
    N = len(states)
    E = np.array([s['E_mhz'] for s in states], dtype=np.float64)
    coeffs = address_pol_coeffs(address_cfg)
    t_unit = (E_CHARGE * A0) / (2.0 * h * 1e6)
    C_unit = np.zeros((N, N), dtype=np.complex128)
    for q, c_q in coeffs:
        if abs(c_q) < 1e-12:
            continue
        for i in range(N):
            si = states[i]
            for j in range(i + 1, N):
                sj = states[j]
                if abs(si['l'] - sj['l']) != 1 or abs(si['j'] - sj['j']) > 1.0 + 1e-9:
                    continue
                d = dipole_amplitude(cache, si, sj, q)
                if abs(d) > 1e-16:
                    # 同一对 (i,j) 只能经唯一 q 耦合（选择定则），直接叠加
                    v = -t_unit * c_q * d
                    C_unit[i, j] += v
                    C_unit[j, i] += np.conj(v)
    return E, C_unit


def direct_ac_stark_shifts(cache, groups, address_cfg, B_gauss=0.0,
                           n_photon=None, sos_n_window=None):
    """直接矩阵对角化计算全部已添加组子能级的 AC 位移（MHz）。

    基组 = 已添加组子能级 + 各组偶极伙伴（|Δn|≤sos_n_window，默认
    DIRECT_SOS_N_WINDOW）。构造含光子数的 Floquet 哈密顿量并一次对角化：
        H_F = Σ_n (H0 + n·ħω)⊗|n⟩⟨n| + (V/2)⊗(|n+1⟩⟨n| + |n⟩⟨n+1|)
    自洽包含近共振 dressed + 远失谐反旋项 + 多态耦合 + 超精细相干性。

    **n_photon**：光子块阶数（块数 = 2n_photon+1）。默认 4（9 块）：
    近共振下 n=1 与 n→∞ 已收敛，但 4 阶显著压低强光/多态场景的
    截断偏差（实测 60S↔6P 场景 n=1 与 n=4 差 ~6%，n≥4 收敛）。

    **缓存**：耦合结构 C_unit（单位 E0，与频率/强度无关）模块级缓存，
    只在 (基组, B, 偏振/椭圆, window) 变化时重算偶极矩阵元；完整结果
    （位移 dict）再按键 (以上 + n_photon + 频率 + 强度) 缓存——同参
    直接返回，不重复对角化；改频/改强只重标定耦合并重对角化。
    """
    if not address_cfg.get('enabled', False) or not groups:
        return {}
    if n_photon is None:
        n_photon = _AC_UI_CFG['n_photon'] or 4          # 面板配置优先，缺省 4
    if sos_n_window is None:
        sos_n_window = (_AC_UI_CFG['sos_n_window']
                        or _P.get('DIRECT_SOS_N_WINDOW'))
    global _DIRECT_COUPLING_KEY, _DIRECT_COUPLING
    ckey = (tuple(sorted((g['n'], g['l'], g['j']) for g in groups)),
            round(float(B_gauss), 6),
            str(address_cfg.get('elliptical')) + '|'
            + str(address_cfg['polarization']),
            sos_n_window)
    if _DIRECT_COUPLING_KEY != ckey or _DIRECT_COUPLING is None:
        states = _direct_basis(cache, groups, B_gauss, sos_n_window)
        E, C_unit = _direct_coupling(cache, states, address_cfg)
        _DIRECT_COUPLING_KEY = ckey
        _DIRECT_COUPLING = (states, E, C_unit)
    states, E, C_unit = _DIRECT_COUPLING
    N = len(states)
    freq_mhz = address_cfg['frequency_hz'] / 1e6
    Ival = float(address_cfg['intensity'])
    # 结果缓存：同参数直接返回（不重对角化）
    rkey = (ckey, int(n_photon), round(freq_mhz, 6), round(Ival, 6))
    hit = _DIRECT_SHIFT_CACHE.get(rkey)
    if hit is not None:
        return hit
    M = 2 * n_photon + 1
    # 按强度重标定单位耦合：C = E0·C_unit（E0² = 2I/(ε0·c)）
    C = (2.0 * Ival * MW_PER_CM2_TO_W_PER_M2
         / (epsilon_0 * c)) ** 0.5 * C_unit
    # 注：旧版曾屏蔽 |ω0−ω|>ω0 的远失谐里德堡↔里德堡耦合，理由为
    # "有限光子块截断的虚假 ~Ω²/ω 位移"。实测（2025-09）该伪影在
    # 二能级与全伙伴基组中均不出现（n=1 即收敛至 0.1%），屏蔽反而把
    # 真实远失谐位移硬置 0（~6.6 MHz 偏差）——已删除，统一按 n_photon
    # 管理精度（默认 4；极端强光/近简并流形可调高并加收敛自检）。
    # t23：实数化——Floquet H 实测完全实对称（C_unit 虚部恒 0、E 实对角、
    # 无耗散），实数 eigh（syevd）比复数（heevd）快 ~7×（2142 维
    # 16.8s→2.4s）。若 C 出现非零虚部（偶极复相位场景）回退复数路径，
    # 物理不变，仅数值路径。
    if not np.any(C.imag):
        H = np.zeros((N * M, N * M), dtype=np.float64)
        for p in range(M):
            sl = slice(p * N, (p + 1) * N)
            H[sl, sl] = np.diag(E) + (p - n_photon) * freq_mhz * np.eye(N)
        for p in range(M - 1):
            a, b = p * N, (p + 1) * N
            H[a:b, b:b + N] = C.real
            H[b:b + N, a:b] = C.real.T
    else:
        H = np.zeros((N * M, N * M), dtype=np.complex128)
        for p in range(M):
            sl = slice(p * N, (p + 1) * N)
            H[sl, sl] = np.diag(E) + (p - n_photon) * freq_mhz * np.eye(N)
        for p in range(M - 1):
            a, b = p * N, (p + 1) * N
            H[a:b, b:b + N] = C
            H[b:b + N, a:b] = C.conj().T
    evals, evecs = np.linalg.eigh(H)
    keys = _occ_keys_direct(states)
    out = _floquet_shifts(evals, evecs, E, keys, n_photon)
    if len(_DIRECT_SHIFT_CACHE) >= _DIRECT_SHIFT_CACHE_MAX:
        _DIRECT_SHIFT_CACHE.pop(next(iter(_DIRECT_SHIFT_CACHE)))
    _DIRECT_SHIFT_CACHE[rkey] = out
    return out


def _floquet_shifts(evals, evecs, E, keys, n_photon, E0mat=None):
    """从 Floquet 本征值/向量提取各子能级位移（MHz）。

    通过本征态对光子块 0 的投影确定归属：取重叠最大者；共振 50/50 劈裂
    （重叠 <30%·max）时取离裸能量最近的分支。E0mat 可复用（扫描批量调用）。
    """
    N = len(keys)
    M = 2 * n_photon + 1
    p0 = n_photon                          # 光子块 0
    if E0mat is None:
        # t23：实数路径（Floquet H 实对称 → evecs 实）——E0mat 用 float64；
        # 若调用方传复数 evecs（复数回退路径），实数 E0mat 仍兼容
        # （np.conj(evecs).T @ E0mat 对复/实混合正确）。
        E0mat = np.zeros((N * M, N))
        E0mat[p0 * N:(p0 + 1) * N, :] = np.eye(N)
    ov = np.abs(np.conj(evecs).T @ E0mat) ** 2     # (NM, N)：本征态对源的重叠
    out = {}
    for s in range(N):
        ov_s = ov[:, s]
        best = int(np.argmax(ov_s))
        thr = 0.3 * ov_s[best]
        cand = np.where(ov_s >= thr)[0]
        # 共振 50/50 劈裂时取离裸能量最近的分支
        k = int(cand[np.argmin(np.abs(evals[cand] - E[s]))])
        out[keys[s]] = float(evals[k] - E[s])
    return out


def direct_ac_stark_sweep(cache, groups, address_cfg, B_gauss, intensities,
                          n_photon=None, sos_n_window=None):
    """一次构建 Floquet 结构，对一系列光强批量计算全部子能级位移（MHz）。

    耦合矩阵 C ∝ E0 ∝ sqrt(I)，基组与对角块不随光强变化；本函数只构建
    一次（基组, E, C_unit），对每个 I 重标定耦合并重对角化，比反复调用
    direct_ac_stark_shifts 快一个量级（省去每次重建基组与偶极矩阵元），
    用于光强扫描。sos_n_window 可用精简基组（如 1）加速定位，
    再以更全基组（如 2，误差 <0.3%）局部迭代收敛。

    返回 (keys, shifts)：
      keys  : [(n,l,j,qstr,occ), ...]（与 direct_ac_stark_shifts 键一致）
      shifts: dict[occ_key] -> np.array（长度 = len(intensities)，MHz）
    """
    if (not address_cfg.get('enabled', False) or not groups
            or len(intensities) == 0):
        return [], {}
    if n_photon is None:
        n_photon = _AC_UI_CFG['n_photon'] or 4
    if sos_n_window is None:
        sos_n_window = (_AC_UI_CFG['sos_n_window']
                        or _P.get('DIRECT_SOS_N_WINDOW'))
    states = _direct_basis(cache, groups, B_gauss, sos_n_window)
    N = len(states)
    M = 2 * n_photon + 1
    E = np.array([s['E_mhz'] for s in states], dtype=np.float64)
    keys = _occ_keys_direct(states)
    freq_mhz = address_cfg['frequency_hz'] / 1e6
    _, C_unit = _direct_coupling(cache, states, address_cfg)
    # 与光强无关的光子块结构：H(I) = H_diag + E0(I)·H_coup（无里德堡屏蔽，
    # 见 direct_ac_stark_shifts —— 实测截断伪影不存在，屏蔽反致真实位移清零）
    NM = N * M
    # t23：实数化（同 direct_ac_stark_shifts）——C_unit 虚部恒 0 → 实对称路径
    cplx = np.any(C_unit.imag)
    H_diag = np.zeros((NM, NM), dtype=np.complex128 if cplx else np.float64)
    for p in range(M):
        sl = slice(p * N, (p + 1) * N)
        H_diag[sl, sl] = np.diag(E) + (p - n_photon) * freq_mhz * np.eye(N)
    H_coup = np.zeros((NM, NM), dtype=np.complex128 if cplx else np.float64)
    for p in range(M - 1):
        a, b = p * N, (p + 1) * N
        if cplx:
            H_coup[a:b, b:b + N] = C_unit
            H_coup[b:b + N, a:b] = C_unit.conj().T
        else:
            H_coup[a:b, b:b + N] = C_unit.real
            H_coup[b:b + N, a:b] = C_unit.real.T
    p0 = n_photon
    E0mat = np.zeros((NM, N), dtype=np.complex128 if cplx else np.float64)
    E0mat[p0 * N:(p0 + 1) * N, :] = np.eye(N)
    shifts = {k: np.zeros(len(intensities)) for k in keys}
    for t, I in enumerate(intensities):
        E0 = (2.0 * I * MW_PER_CM2_TO_W_PER_M2 / (epsilon_0 * c)) ** 0.5
        H = H_diag + E0 * H_coup
        evals, evecs = np.linalg.eigh(H)
        sh = _floquet_shifts(evals, evecs, E, keys, n_photon, E0mat)
        for k in keys:
            shifts[k][t] = sh[k]
    return keys, shifts

# ============================================================
# 子能级行（UI 直接渲染）
# ============================================================

def build_level_rows(cache, field_B, groups, address_cfg, visible):
    """计算可见能级组的全部子能级，返回行 dict 列表（UI 直接绘制）。

    AC-Stark 唯一后端 = direct（Floquet 矩阵对角化，第一性原理）；
    legacy（逐伙伴解析/微扰）已于 2025-08 删除。
    每行字段：x/E0/E/ac/affected/label/qstr/color/n/l/j/kind('mJ'/'F')/mj 或 mF/amps。
    """
    return build_level_rows_direct(cache, field_B, groups, address_cfg, visible)


def _base_level_rows(cache, field_B, visible):
    """构造可见能级组的全部子能级行（ac=0，E=E0，affected=False）。

    里德堡态忽略核自旋（只画 fine-structure 的 m_J 子能级）；低激发态
    画出所有超精细子能级 (F, m_F)。AC 位移由 direct 一次对角化填入
    （build_level_rows_direct）。
    行字段：x/E0/E/ac/affected/label/qstr/color/n/l/j/kind/mj 或 mF/amps。
    """
    rows = []
    for group in visible:
        n, l, j = group['n'], group['l'], group['j']
        color = group.get('color', '#1f77b4')
        E0_mhz = cache.E_mhz(n, l, j)
        hf = get_hyperfine_levels(cache, n, l, j, field_B)
        if hf is None:
            gJ = cache.gJ(l, j)
            for m in np.arange(-j, j + 1, 1.0):
                E_level = E0_mhz + gJ * MU_B_MHZ_PER_G * m * field_B
                rows.append(dict(
                    x=float(m), E0=E_level, E=E_level, ac=0.0, color=color,
                    label=group['label'], affected=False,
                    qstr=f'm_J={m:+.1f}',
                    n=n, l=l, j=j, kind='mJ', mj=m, mF=None, amps=None))
        else:
            for F, mF, e_hf, _w, mJ_amps in hf:
                E_level = E0_mhz + e_hf
                rows.append(dict(
                    x=float(mF), E0=E_level, E=E_level, ac=0.0, color=color,
                    label=group['label'], affected=False,
                    qstr=f'F={F:.1f}, m_F={mF:+.1f}',
                    n=n, l=l, j=j, kind='F', mj=None, mF=mF, amps=mJ_amps))
    return rows


def _fill_ac(rows, acs):
    """把逐行 AC 位移（列表/生成器）写回行结构：E = E0 + ac，affected 标记。"""
    for r, ac in zip(rows, acs):
        r['ac'] = ac
        r['E'] = r['E0'] + ac
        r['affected'] = abs(ac) > _P.get('AC_EFFECT_THRESHOLD_MHZ')
    return rows


def build_level_rows_direct(cache, field_B, groups, address_cfg, visible):
    """AC-Stark 唯一后端：ac 来自一次 Floquet 对角化（direct_ac_stark_shifts），
    自洽含近共振 dressed + 远失谐反旋项 + 多态耦合。"""
    rows = _base_level_rows(cache, field_B, visible)
    if address_cfg.get('enabled', False):
        shifts = direct_ac_stark_shifts(cache, groups, address_cfg, field_B)
        _fill_ac(rows, (shifts.get(key, 0.0) for key in _occ_keys_direct(rows)))
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

    coeffs = address_pol_coeffs(address_cfg)
    I_wm2 = address_cfg['intensity'] * MW_PER_CM2_TO_W_PER_M2
    E0 = (2.0 * I_wm2 / (epsilon_0 * c)) ** 0.5
    freq = address_cfg['frequency_hz']

    for i in range(len(groups)):
        for k in range(i + 1, len(groups)):
            gi, gk = groups[i], groups[k]
            # 恒以能量低的态为源（gi）、能量高的态为目标（gk）
            if cache.E_mhz(gi['n'], gi['l'], gi['j']) > cache.E_mhz(gk['n'], gk['l'], gk['j']):
                gi, gk = gk, gi
            # 任意偏振：遍历非零 q 分量，取全局失谐最小的耦合子能级对
            best = None
            for q, c_q in coeffs:
                if abs(c_q) < 1e-12:
                    continue
                sub = _closest_sublevel_pair(cache, gi, gk, q, E0, field_B, freq)
                if sub is None:
                    continue
                qstr_i, qstr_k, det, omega = sub
                if best is None or abs(det) < abs(best[2]):
                    best = (qstr_i, qstr_k, det, abs(c_q) * omega)
            if best is None:
                continue
            qstr_i, qstr_k, det, omega = best
            pairs.append((gi['label'], qstr_i, gk['label'], qstr_k, det, omega))
    return pairs


def transition_strength_q(cache, a, b, q):
    """子能级对 (a,b) 在偏振 q 下的偶极强度 |⟨a|d_q|b⟩|²。

    复用 dipole_amplitude（同一矩阵元与选择定则逻辑），只取模平方。
    矩阵元按 m_J 分辨：里德堡态为纯 m_J；低激发态超精细子能级用其
    |m_J,m_I> 本征振幅叠加（m_I 由 m_F−m_J 确定）。
    """
    return abs(dipole_amplitude(cache, a, b, q)) ** 2


def spectrum_lines(cache, rows, ac_on, coeffs=None):
    """计算子能级之间全部偶极允许跃迁的频谱线。

    coeffs 为球基偏振复系数（address_pol_coeffs 返回）；None 表示全偏振等权
    （无寻址光视图）。有系数时谱线强度按各 q 分量 |c_q|² 加权，pol 只标注
    非零系数对应的 q。

    返回 [(freq_hz, A, label_a, label_b, qstr_a, qstr_b, color_a, pol), ...]：
    - freq: 跃迁频率 (Hz)，ac_on=True 时用 AC 移后能量
    - A   : 爱因斯坦 A 系数 (s^-1)，作为谱线强度
    - qstr_a/qstr_b: 两子能级描述（供峰表显示）
    - pol : 产生该峰的偏振（σ+/π/σ-，可能多个）
    矩阵元按 m_J 分辨计算：里德堡态为纯 m_J；低激发态超精细子能级
    用其 |m_J,m_I> 本征振幅叠加（m_I 由 m_F−m_J 确定）。
    """
    if coeffs is None:
        qw = [(-1, 1.0), (0, 1.0), (1, 1.0)]      # 全偏振等权（原行为）
    else:
        qw = [(q, abs(c) ** 2) for q, c in coeffs if abs(c) > 1e-12]
    lines = []
    for i in range(len(rows)):
        for k in range(i + 1, len(rows)):
            a, b = rows[i], rows[k]
            if (a['n'], a['l'], a['j']) == (b['n'], b['l'], b['j']):
                continue
            S = 0.0
            qs = []
            for q, w in qw:
                sq = transition_strength_q(cache, a, b, q)
                if sq > 1e-16:
                    S += w * sq
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


def __getattr__(name):
    """PEP 562 回落：把历史 re-export 的引擎参数转发到注册表。

    这些模块过去用 `from .params import X` 顺带充当转发站（外部代码
    `core.RYDBERG_N_MIN` / `from .evolve_rwa import RWA_TOL`）。参数读取点
    已改为 `_P.get(...)`，转发站随之消失——这里补一层回落，
    使旧式属性访问继续可用，并且同样响应 `params.override()`。
    """
    from . import params as _p
    if name in _p.DEFAULTS or name in _p.ALIASES:
        return _p.get(name)
    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')
