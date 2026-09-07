# -*- coding: utf-8 -*-
"""两体张量化调度层（与引擎无关）——重构第 3 步。

从 pair_evolve.py 原样搬移的两体工具（零逻辑改动，仅换模块）：
- tensor_states      ：单原子 states → 两体 product 基（k=i·N+j）
- _as_product_psi    ：单体初态 → 两体积态 |ψA⟩⊗|ψB⟩
- tensor_fields      ：A/B 单体场 → 两体场（A⊗I + I⊗B）
- _reduce_A/_reduce_B：两体布居 → 单原子约化布居
- _pair_loss_vectors ：寿命丢失向量（依赖 evolve.decay）
- states_to_single   ：两体 states → 单原子去重

使用方：pair_evolve（原定义处，改为 import）、hard_pair、evolve_exact_pair。
"""
import numpy as np

from .evolve_util import _resolve_state_idx  # t74 工具归位
from .field_build import _field_geometry


def tensor_states(states):
    """单原子 states → 两体 product 基组（N² 维）。

    states = [s0, s1, ..., s_{N-1}]（evolve.make_levels 输出）
    product basis: |s_i⟩⊗|s_j⟩，k = i*N + j

    返回：
    states_2d : list of dict，每个含 ket_i, ket_j, label, E_mhz, index
      N : 单体维度
    """
    N = len(states)
    states_2d = []
    for i in range(N):
        for j in range(N):
            si, sj = states[i], states[j]
            E = si['E_mhz'] + sj['E_mhz']
            # 使用普通 ASCII 连接符，避免部分 Matplotlib 字体无法显示 ⊗。
            # 两体图例仍保留两个单体的完整能级标签。
            label = f"({si['label']}) x ({sj['label']})"
            states_2d.append(dict(
                ket_i=i, ket_j=j, label=label, E_mhz=E,
                index=i * N + j))
    return states_2d, N


def _as_product_psi(psi0_A, psi0_B, N):
    """单体初态 → 两体积态 |ψ_A⟩⊗|ψ_B⟩（N² 元向量）。"""
    a = np.asarray(psi0_A, complex).ravel()
    b = np.asarray(psi0_B, complex).ravel()
    assert len(a) == N and len(b) == N, \
        f'初态维度不匹配：A={len(a)}, B={len(b)}, N={N}'
    return np.kron(a, b)


def _tensor_field_atom(field_raw, states, N, atom='A'):
    """单体场 → 两体场（M_A⊗I_B 或 I_A⊗M_B）。

    field_raw: evolve.field() / real_field() / group_field() 输出
    atom='A'：场作用在原子 A 上
    atom='B'：场作用在原子 B 上
    """
    # 先归一化：pair/group 场 → multi 格式（含 M, A, ac_shift, g_idx, e_idx）
    g = _field_geometry(states, field_raw)
    M = g['M']
    A = g['A']
    ac = g['ac_shift']
    N1 = M.shape[0]
    assert N1 == N, f'场维度 {N1} ≠ 基组维度 {N}'

    I_other = np.eye(N, dtype=complex)
    M_2d = np.kron(M, I_other) if atom == 'A' else np.kron(I_other, M)
    A_2d = np.kron(A, I_other) if atom == 'A' else np.kron(I_other, A)
    # 对角 AC 位移：ac[i] 加到两体 |i,j⟩ 或 |j,i⟩
    ac_2d = np.zeros(N * N)
    if atom == 'A':
        for i in range(N):
            for j in range(N):
                ac_2d[i * N + j] = ac[i]
    else:
        for i in range(N):
            for j in range(N):
                ac_2d[i * N + j] = ac[j]

    g_idx_2d, e_idx_2d = [], []
    for gi, ei in zip(g.get('g_idx', []), g.get('e_idx', [])):
        if atom == 'A':
            for j in range(N):
                g_idx_2d.append(gi * N + j)
                e_idx_2d.append(ei * N + j)
        else:
            for i in range(N):
                g_idx_2d.append(i * N + gi)
                e_idx_2d.append(i * N + ei)

    return {
        'kind': 'multi',
        'M': M_2d,
        'A': A_2d,
        'ac_shift': ac_2d,
        'g_idx': g_idx_2d,
        'e_idx': e_idx_2d,
        'omega0_mhz': g['omega0_mhz'],
        'real_omega0_mhz': g.get('real_omega0_mhz', g['omega0_mhz']),
        'omega_L_mhz': g['omega_L_mhz'],
        'waveform': g['waveform'],
    }


def tensor_fields(fields_A, fields_B, states, N):
    """两组单体场 → 两体场列表（A⊗I + I⊗B）。

    fields_A/B: 各自的 evolve field 列表
    states: 单原子能级列表（供 _field_geometry 归一化）
    N: 单体维度

    返回两体场列表，直接喂给 pair_evolve。
    """
    fields_2d = []
    for f in fields_A:
        fields_2d.append(_tensor_field_atom(f, states, N, atom='A'))
    for f in fields_B:
        fields_2d.append(_tensor_field_atom(f, states, N, atom='B'))
    return fields_2d


def _reduce_A(pops_2d, N):
    """两体布居 → 原子 A 约化布居（trace out B）。"""
    n_t = pops_2d.shape[0]
    pops_A = np.zeros((n_t, N))
    for i in range(N):
        for j in range(N):
            pops_A[:, i] += pops_2d[:, i * N + j]
    return pops_A


def _reduce_B(pops_2d, N):
    """两体布居 → 原子 B 约化布居（trace out A）。"""
    n_t = pops_2d.shape[0]
    pops_B = np.zeros((n_t, N))
    for i in range(N):
        for j in range(N):
            pops_B[:, j] += pops_2d[:, i * N + j]
    return pops_B


def _pair_loss_vectors(states, decays_A, decays_B, N, K):
    """两体衰变通道 → 每两体基元的耗散率向量 (GammaA, GammaB)。

    - 通道为单原子 decay() 列表（auto_decays 输出即可）：from 解析到单原子
      索引，Γ（MHz）×1e-3 = 速率 /ns（与单原子 Lindblad 同约定，L 不乘 2π）；
    - A 通道 from=i → 两体基 |i⊗j⟩（∀j）按 Γ 耗散；B 通道 from=j → |i⊗j⟩；
    - 仅支持落点=丢失（to=None，寿命/自发辐射场景）：布居离开建模空间，
      确定性 NonHermitian 演化（H 对应对角 -iΓ/2，ψ 幅度 e^{-Γt/2}），
      漏出累计为 lossA/lossB——严格 L=√Γ|loss⟩⟨from| 主方程在建模空间的解；
    - 显式落点（to 在空间）/纯退相（from==to）的双原子版本暂不支持
      （需密度矩阵回写/相干衰减，见 TODO）——显式拒绝并提示用单原子引擎。
    """
    GammaA = np.zeros(K, float)
    GammaB = np.zeros(K, float)
    for decays, atom, target in ((decays_A or [], 'A', GammaA),
                                 (decays_B or [], 'B', GammaB)):
        for ch in decays:
            g_idx = _resolve_state_idx(states, ch['from'])
            if g_idx is None:
                raise ValueError(
                    f'两体耗散：{atom} 通道 from 无法解析到基组 '
                    f'({ch["from"]!r})')
            to = ch.get('to')
            if to is not None and _resolve_state_idx(states, to) \
                    is not None:
                raise NotImplementedError(
                    '两体 Lindblad v1 仅支持落点=丢失（寿命通道/auto_decays）；'
                    '显式落点 / 纯退相请先用单原子引擎（动态页单原子模式）。')
            gam_ns = float(ch['gamma_mhz']) * 1e-3     # MHz → /ns
            if gam_ns <= 0:
                continue
            if atom == 'A':
                for j in range(N):
                    target[g_idx * N + j] += gam_ns
            else:
                for i in range(N):
                    target[i * N + g_idx] += gam_ns
    has = bool(np.any(GammaA) or np.any(GammaB))
    return GammaA, GammaB, has


def states_to_single(states):
    """evolve states → pair_core 兼容的 groups 列表。"""
    seen = set()
    groups = []
    for s in states:
        key = (s['n'], s['l'], s['j'])
        if key not in seen:
            seen.add(key)
            groups.append(dict(n=s['n'], l=s['l'], j=s['j']))
    return groups
