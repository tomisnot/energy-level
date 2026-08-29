# sambe.jl — 两体 Sambe (t,t') 扩展空间，逐行镜像 _PairSambeSetup。
#
# 索引 idx = k + N²·p（k=i·N+j 两体基最快；p 为混合进制光子索引，
# 组 g 步长 stride_g = D^(g-1)，D = 2M+1）。
# K_static = diag(E_pair + 光子偏移 − iΓ/2) + kron(I_P, V_dd)
# Kc_g     = kron(T_g + T_gᵀ, 0.5·K2_g)（光子邻接 ⊗ 单位包络耦合）
# 物理投影 ψ_phys(t) = Σ_p e^{−i·nphase_p·t}·X_p（nphase 单位 rad/ns）

using LinearAlgebra
using SparseArrays

"逐场包络求和器（组内 c_g(t)），由 Request.fields 构造"
struct EnvGroups
    envs::Vector{Vector{Envelope}}   # 每组的场包络列表
end
EnvGroups(req::Request) = EnvGroups([
    [f.env for f in req.fields if f.group == g] for g in 1:length(req.groups)])

"c_g(t)：标量"
(cg::EnvGroups)(g::Int, t::Real) =
    sum(e(t) for e in cg.envs[g]; init = 0.0)

"c_g(t)：数组"
function (cg::EnvGroups)(g::Int, ts::AbstractVector)
    out = zeros(Float64, length(ts))
    for e in cg.envs[g], (i, t) in enumerate(ts)
        out[i] += e(t)
    end
    out
end

"c_at_nodes：全部组 → (G, n)"
function (cg::EnvGroups)(ts::AbstractVector)
    G = length(cg.envs)
    out = Matrix{Float64}(undef, G, length(ts))
    for g in 1:G
        out[g, :] .= cg(g, ts)
    end
    return out
end

"实化：虚部全零 → 实数（BLAS/eigh 提速；耗散存在时保持复数，镜像 _realify）"
_realify(A::Matrix{ComplexF64}) = all(iszero, imag.(A)) ? real(A) : A
function _realify(S::SparseMatrixCSC{ComplexF64,Int})
    nz = nonzeros(S)
    if isempty(nz) || all(iszero, imag.(nz))
        return SparseMatrixCSC(S.m, S.n, S.colptr, S.rowval, real.(nz))
    end
    return S
end

"两体 Sambe 扩展空间结构（K_static/Kc_ext 类型随实化/稠密而定）"
abstract type AbstractSambeSetup end

struct SambeSetup{KS,KC} <: AbstractSambeSetup
    N::Int
    N2::Int
    M::Int
    D::Int
    P::Int
    dim::Int
    strides::Vector{Int}
    nvecs::Vector{Vector{Int}}       # 0-based 光子量子数（0..D-1）
    diag::Vector{ComplexF64}         # MHz（虚部 = −Γ/2 或 0）
    K_static::KS
    Kc_ext::Vector{KC}
    nphase::Vector{Float64}          # rad/ns
    p_center::Int                    # 0-based 中心光子块
    dense::Bool
    has_gamma::Bool
    envs::EnvGroups
end

"""构造（镜像 _PairSambeSetup.__init__）。

req: 请求对象；M: 光子块阶数；vdd: V_dd（MHz，N²×N²）或 nothing；
gamma_ns: 寿命速率 /ns（N²）或 nothing（内部转 Γ_MHz = gam/_CONV）。"""
function SambeSetup(req::Request, M::Int, vdd, gamma_ns)
    N = req.N
    N2 = N * N
    E_pair = req.E_pair
    groups = req.groups
    G = length(groups)
    D = 2 * M + 1
    P = D^G
    dim = N2 * P
    # 混合进制光子索引（0-based p；base[p] = −Σ(n_g−M)·ω_g）
    strides = [D^(g - 1) for g in 1:G]
    nvecs = Vector{Vector{Int}}(undef, G)
    base = zeros(Float64, P)
    for g in 1:G
        nvec = [(p ÷ strides[g]) % D for p in 0:P-1]
        nvecs[g] = nvec
        base .-= (nvec .- M) .* groups[g].omega
    end
    # 对角：tile(E_pair, P) + repeat(base, N2)（k 最快；1-based k1 = (i1-1)N+j1
    # 与 Python 0-based k=i·N+j 的扁平顺序一致）
    diag = ComplexF64[E_pair[k] + base[p] for p in 1:P for k in 1:N2]
    has_gamma = gamma_ns !== nothing && any(!iszero, gamma_ns)
    if has_gamma
        diag .-= 0.5im .* repeat(gamma_ns ./ _CONV, P)
    end
    # V_dd 块对角嵌入 kron(I_P, V_dd)
    vdd_ext = if vdd === nothing || nnz(vdd) == 0
        nothing
    else
        kron(spdiagm(0 => fill(one(ComplexF64), P)),
             SparseMatrixCSC{ComplexF64,Int}(vdd))
    end
    K_static = spdiagm(0 => diag)
    if vdd_ext !== nothing
        K_static = K_static + vdd_ext
    end
    # 重构相位系数
    nphase = zeros(Float64, P)
    for g in 1:G
        nphase .+= nvecs[g] .* (_CONV * groups[g].omega)
    end
    p_center = sum(M * s for s in strides; init = 0)
    # 逐组光子邻接耦合 kron(T+Tᵀ, 0.5·K2)（K2 实对称 → Kc 实数）
    idx0 = 0:P-1
    Kc_ext = Vector{Any}(undef, G)
    for g in 1:G
        s0 = strides[g]
        rs = Int[]
        cs = Int[]
        for s in 0:D-2
            src = [p for p in idx0 if (p ÷ s0) % D == s]   # 0-based
            append!(rs, src .+ s0 .+ 1)                    # p_dst（1-based）
            append!(cs, src .+ 1)
        end
        T = sparse(rs, cs, ones(Float64, length(rs)), P, P)
        Kh = 0.5 .* SparseMatrixCSC{ComplexF64,Int}(groups[g].K2)
        Kc = kron(T, Kh) + kron(transpose(T), adjoint(Kh))
        Kc_ext[g] = _realify(SparseMatrixCSC{ComplexF64,Int}(Kc))
    end
    dense = dim <= req.dense_max
    envs = EnvGroups(req)
    if dense
        Ks = _realify(Matrix(SparseMatrixCSC{ComplexF64,Int}(K_static)))
        Kcs = Matrix.(Kc_ext)
        return SambeSetup{typeof(Ks),eltype(Kcs)}(
            N, N2, M, D, P, dim, strides, nvecs, diag, Ks, Kcs,
            nphase, p_center, dense, has_gamma, envs)
    else
        Ks = _realify(SparseMatrixCSC{ComplexF64,Int}(K_static))
        Kcs = SparseMatrixCSC{eltype(Kc_ext[1]),Int}.(Kc_ext)
        return SambeSetup{typeof(Ks),eltype(Kcs)}(
            N, N2, M, D, P, dim, strides, nvecs, diag, Ks, Kcs,
            nphase, p_center, dense, has_gamma, envs)
    end
end

"段算符 K_static + Σ c_g·Kc_g（镜像 K_seg；c_list 长度 = 组数）"
function K_seg(s::AbstractSambeSetup, c_list::Vector{Float64})
    K = s.K_static
    for (c, Kc) in zip(c_list, s.Kc_ext)
        if c != 0.0
            K = K .+ c .* Kc
        end
    end
    return K
end

"两体初态直积 |ψ_A⟩⊗|ψ_B⟩（镜像 _as_product_psi / np.kron 顺序）"
function product_psi(psiA::AbstractVector{ComplexF64},
                     psiB::AbstractVector{ComplexF64}, N::Int)
    return vec(kron(psiA, psiB))
end

"物理初态嵌入中心光子块（镜像 _embed_pair）"
function embed_pair(psi_2d::AbstractVector{ComplexF64}, s::AbstractSambeSetup)
    X = zeros(ComplexF64, s.dim)
    off = s.N2 * s.p_center
    X[off+1:off+s.N2] .= psi_2d
    return X
end

"扩展态 → 两体物理布居（镜像 _physical_pops）"
function physical_pops(s::AbstractSambeSetup, X::AbstractVector{ComplexF64}, tq::Real)
    return abs2.(physical_amp(s, X, tq))
end

"物理幅值 ψ_phys(t) = Σ_p e^{−i·nphase_p·t}·X_p（镜像 _physical_pops 内核）"
function physical_amp(s::AbstractSambeSetup, X::AbstractVector{ComplexF64}, tq::Real)
    # 注意：必须用 permutedims（纯转置），adjoint(') 会共轭整个态 → 相位翻转
    Xm = permutedims(reshape(X, s.N2, s.P), (2, 1))   # (P, N2)
    ph = exp.(-1im .* s.nphase .* Float64(tq))        # (P,)
    return vec(sum(ph .* Xm; dims = 1))
end
