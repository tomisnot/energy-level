# block.jl — mF-光子荷不变量分块（S6，零误差对称性重组）。
#
# 不变量：I = mF_sum(k) + Σ_g n_g(p)·s_g，s_g = −q_g（每吸收一个 g 载频
# 光子原子 mF 变化 q_g：σ⁻ q=-1 → s=+1；π q=0 → s=0；σ⁺ q=+1 → s=-1）。
#   吸收一个 σ⁻ 光子：n+1 且 mF−1 → mF_sum + n 守恒 ✓。
# 全部算符（Kc 梯子、V_dd 转动标量、Zeeman、超精细、衰变）严格保 I →
# 全空间按 I 分块独立演化 = 纯重组（零物理误差）。
# 初始态唯一确定 I0；其余块不可达、精确解耦、直接丢弃（Exp1 实测 15.7×）。
#
# 实现：全 SambeSetup（固定 M）上按 I0 提取可达类状态索引 → SubSetup
# （矩阵切片 + k/p 映射，接口字段与 SambeSetup 对齐）→ 现有传播子原样
# 工作在子空间；物理布居经 k 映射散回全对态基。
#
# 极化限制：仅纯 σ±/π 载频可分块；椭圆偏振（q 不定）→ 分块禁用。

using LinearAlgebra
using SparseArrays

abstract type AbstractSambeSetup end

"每载频组的 mF 荷（每吸收光子的 mF 变化）：σ⁺=+1, π=0, σ⁻=-1"
charge_of_pol(pol::AbstractString) =
    pol == "σ+" ? 1.0 : pol == "σ-" ? -1.0 : pol == "π" ? 0.0 : NaN

"对态 mF_sum（全基组序；k1 = (i1-1)N + j1，镜像 kron 顺序）"
function pair_mF(mF_states::Vector{Float64}, N::Int)
    return Float64[mF_states[i] + mF_states[j] for i in 1:N for j in 1:N]
end

"""可达类状态索引（全空间 1-based）。

判据：mF_sum(k) + Σ_g s_g·n_g(p) = I0（s_g = −q_g 为光子荷）。
返回 (class_idx, k_of, p_of)。"""
function class_index(s::SambeSetup, mF_states::Vector{Float64},
                     charges::Vector{Float64}, I0::Real)
    N = s.N
    mF_pair = pair_mF(mF_states, N)
    N2 = s.N2
    P = s.P
    s_fac = zeros(Float64, P)                 # Σ_g s_g·n_g(p)
    for (g, stride) in enumerate(s.strides)
        s_fac .+= (-charges[g]) .* s.nvecs[g] # s_g = −q_g
    end
    class_idx = Int[]
    k_of = Int[]
    p_of = Int[]
    for p in 1:P
        base = (p - 1) * N2
        for k in 1:N2
            if abs(mF_pair[k] + s_fac[p] - I0) < 1e-9
                push!(class_idx, base + k)
                push!(k_of, k)
                push!(p_of, p)
            end
        end
    end
    return class_idx, k_of, p_of
end

"分块后的子问题（接口字段与 SambeSetup 对齐 + 映射）"
struct SubSetup <: AbstractSambeSetup
    dim::Int
    N2::Int
    N::Int
    diag::Vector{ComplexF64}
    K_static::Any
    Kc_ext::Vector{Any}
    nphase::Vector{Float64}
    k_of::Vector{Int}            # 类内态 → 对态 k（1-based）
    p_of::Vector{Int}            # 类内态 → 全空间光子索引 p（1-based）
    p_center::Int                # 中心块全空间 p（1-based）
    mF::Vector{Float64}          # 对态 mF_sum（全基组序）
    dense::Bool
    has_gamma::Bool
    envs::EnvGroups
    P::Int                       # 名义每载频块数 D（报告用）
    M::Int
end

_realify_matrix(A::Matrix{ComplexF64}) = all(iszero, imag.(A)) ? real(A) : A
_realify_matrix(A::Matrix{Float64}) = A
_realify_matrix(S::SparseMatrixCSC{Float64,Int}) = S
function _realify_matrix(S::SparseMatrixCSC{ComplexF64,Int})
    nz = nonzeros(S)
    if isempty(nz) || all(iszero, imag.(nz))
        return SparseMatrixCSC(S.m, S.n, S.colptr, S.rowval, real.(nz))
    end
    return S
end

"从全 setup 提取类子问题（矩阵切片 + 映射）"
function SubSetup(s::SambeSetup, class_idx::Vector{Int}, k_of::Vector{Int},
                  p_of::Vector{Int}, mF_states::Vector{Float64};
                  dense_max::Int = 700)
    N = s.N
    N2 = N * N
    mF_pair = pair_mF(mF_states, N)
    if s.dense
        Ks = _realify_matrix(Matrix(s.K_static)[class_idx, class_idx])
        Kcs = [_realify_matrix(Matrix(Kc)[class_idx, class_idx])
               for Kc in s.Kc_ext]
    else
        Ks = _realify_matrix(s.K_static[class_idx, class_idx])
        Kcs = [_realify_matrix(Kc[class_idx, class_idx]) for Kc in s.Kc_ext]
    end
    return SubSetup(length(class_idx), N2, N, s.diag[class_idx],
                    Ks, Kcs, s.nphase[p_of], k_of, p_of, s.p_center,
                    mF_pair, length(class_idx) <= dense_max, s.has_gamma,
                    s.envs, s.D, s.M)
end

"分块物理布居：类内态按 k 映射聚幅后取模方（相位求和语义不变）"
function physical_pops(s::SubSetup, X::AbstractVector{ComplexF64}, tq::Real)
    amp = zeros(ComplexF64, s.N2)
    ph = exp.(-1im .* s.nphase .* Float64(tq))
    for m in eachindex(s.k_of)
        amp[s.k_of[m]] += ph[m] * X[m]
    end
    return abs2.(amp)
end

"分块物理幅值（record_amp）"
function physical_amp(s::SubSetup, X::AbstractVector{ComplexF64}, tq::Real)
    out = zeros(ComplexF64, s.N2)
    ph = exp.(-1im .* s.nphase .* Float64(tq))
    for m in eachindex(s.k_of)
        out[s.k_of[m]] += ph[m] * X[m]
    end
    return out
end

"分块嵌入：初态置于 (k0, 中心块 p_center) 的类内位置"
function embed_pair(psi_2d::AbstractVector{ComplexF64}, s::SubSetup)
    X = zeros(ComplexF64, s.dim)
    off = s.N2 * (s.p_center - 1)            # 全空间中心块起点（0-based）
    for m in eachindex(s.k_of)
        if s.p_of[m] == s.p_center
            k = s.k_of[m]
            X[m] = psi_2d[k]
        end
    end
    return X
end

"分块 M 探针：类内峰值包络传播，M vs M−1 物理布居差"
function m_probe_blocked(req::Request, vdd, gamma_ns, psi0_2d, T::Real;
                         m_max::Int = 8, m_start::Int = 2,
                         sp_path = nothing, log = nothing)
    plog = log === nothing ? (_ -> nothing) : log
    mF_states = req.mF_states
    charges = req.charges
    mF_init = mF_states[req.psi0_A[1]] + mF_states[req.psi0_B[1]]
    I0 = mF_init + sum(-charges[g] * m_start
                       for g in 1:length(charges); init = 0.0)
    run_at(Mc) = begin
        stp_full = SambeSetup(req, Mc, vdd, gamma_ns)
        I0c = mF_init + sum(-charges[g] * Mc
                            for g in 1:length(charges); init = 0.0)
        cidx, kof, pof = class_index(stp_full, mF_states, charges, I0c)
        stp = SubSetup(stp_full, cidx, kof, pof, mF_states;
                       dense_max = req.dense_max)
        c_list = group_peaks(stp, T)
        K = K_seg(stp, c_list)
        X = embed_pair(psi0_2d, stp)
        if stp.dense
            de = dense_decomp(K)
            X = prop_step(stp, K, 0.0, T, X, de)
            return physical_pops(stp, X, T), 1.0
        end
        sp_path === nothing && error("稀疏探针需段传播子")
        T_pr = min(T, max(10.0, 0.25 * T))
        ext_scale = (T / T_pr)^2
        n_seg = max(1, ceil(Int, T_pr / 1.0))
        hs = T_pr / n_seg
        ta = 0.0
        for _ in 1:n_seg
            X, _ = sp_path(stp, ta, ta + hs, X, req.tol)
            ta += hs
        end
        return physical_pops(stp, X, T_pr), ext_scale
    end
    Mc = max(2, m_start, m_analytic(req, req.tol))
    prev = nothing
    ext_prev = 1.0
    while true
        cur, ext = run_at(Mc)
        if prev !== nothing
            d = maximum(abs.(cur .- prev)) * max(ext, ext_prev)
            plog("[pair_exact] 探针 M=$Mc vs M-1 差=$d")
            d <= req.tol && return Mc - 1, d
        end
        prev = cur
        ext_prev = ext
        Mc >= m_max && return Mc, NaN
        Mc += 1
    end
end
