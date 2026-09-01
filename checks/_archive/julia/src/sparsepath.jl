# sparsepath.jl — 稀疏路径（gen-3 移植）：对角相互作用绘景 + Volterra–GL
# 节点配置（全权重 Wpart）+ GMRES 直解 + 因果 Picard 预条件。
# 逐行镜像 _volterra_rhs / _volterra_rule / _subspace_segment。
#
# 核心性质（交接文档 §2.3）：IP 被积函数只含非对角 R（V_dd+驱动，
# 范数 ~百 MHz），收敛速度与巨大对角能量标度（~THz）无关。
#
# 稳健性约定（gen-3 教训）：
#   1. 节点方程必须用全部 q' 的部分积分权重（Lagrange 基尾部仍有贡献），
#      截断为因果 cumsum 会引入一阶误差（范数漂移 2.5e-2/200 段）；
#   2. GMRES 未收敛（info≠0）→ est=Inf 交主循环细分，绝不返回垃圾态；
#   3. 误差估计 = max(节点阶梯差, GMRES 相对残差)。

using LinearAlgebra
using SparseArrays

# ---- 规则缓存 ----
const _GL_CACHE = Dict{Int,Tuple{Vector{Float64},Vector{Float64}}}()
const _VOLT_RULE_CACHE = Dict{Int,Tuple{Vector{Float64},Vector{Float64},
                                       Matrix{Float64}}}()

function _gl_cached(n::Int)
    get!(_GL_CACHE, n) do
        gl_nodes(n)
    end
end

"""n 点 GL 节点/权重 + 部分积分矩阵 Wpart[q,s] = ∫_{-1}^{x_q} L_s(x)dx
（镜像 _volterra_rule；全权重——Lagrange 尾部贡献不可截断）。"""
function volterra_rule(n::Int)
    get!(_VOLT_RULE_CACHE, n) do
        x, w = _gl_cached(n)
        V = [x[s]^p for p in 0:n-1, s in 1:n]        # V[p+1, s] = x_s^p
        B = inv(V)                                    # B[s, p+1]
        Ip = [(x[q]^(p + 1) - (-1.0)^(p + 1)) / (p + 1)
              for q in 1:n, p in 0:n-1]
        Wpart = Ip * transpose(B)                     # (n, n) [q, s]
        (x, w, Wpart)
    end
end

# ---- 左预条件重启 GMRES（语义对齐 scipy：rtol=0/atol、restart、maxiter=外循环数）----
"""重启 GMRES（左预条件）。

matvec: A·v；Mapp: M·v（M ≈ A⁻¹）。返回 (x, info)：info=0 收敛，
info=1 达 maxiter 未收敛。停机：预条件残差 ≤ atol。"""
function gmres_lp(matvec::F, b::Vector{ComplexF64}, Mapp::G, dim::Int;
                  restart::Int = 40, maxiter::Int = 60,
                  atol::Float64 = 0.0) where {F,G}
    x = zeros(ComplexF64, dim)
    bn = norm(b)
    bn == 0.0 && return x, 0
    m = min(restart, dim)
    V = Matrix{ComplexF64}(undef, dim, m + 1)
    H = Matrix{ComplexF64}(undef, m + 1, m)
    for _ in 1:maxiter
        # 停机判据（scipy 语义）：真残差 ‖b − A·x‖ ≤ atol
        # （预条件残差只驱动 Krylov 迭代；病态系统下二者可差数量级）
        rn = norm(b .- matvec(x))
        rn <= atol && return x, 0
        r = Mapp(b .- matvec(x))
        β = norm(r)
        fill!(H, zero(ComplexF64))
        V[:, 1] .= r ./ β
        jbreak = m
        for j in 1:m
            w = Mapp(matvec(view(V, :, j)))
            for i in 1:j
                H[i, j] = dot(view(V, :, i), w)
                w .-= H[i, j] .* view(V, :, i)
            end
            H[j+1, j] = norm(w)
            if abs(H[j+1, j]) <= 1e-14 * max(β, 1.0)  # happy breakdown
                jbreak = j
                break
            end
            V[:, j+1] .= w ./ H[j+1, j]
        end
        # 最小二乘：min ‖H(1:j+1,1:j)·y − βe1‖（小矩阵，直接 \）
        Hm = H[1:jbreak+1, 1:jbreak]
        g = zeros(ComplexF64, jbreak + 1)
        g[1] = β
        y = Hm \ g
        x .+= V[:, 1:jbreak] * y
    end
    rn = norm(b .- matvec(x))
    return x, (rn <= atol) ? 0 : 1
end

# ---- IP Volterra–GMRES 段传播子 ----

const _NODE_LADDER_SPARSE = [16, 32, 64, 128]
const _VOLT_MAX_ITER = 60
const _VOLT_PICARD_FRAC = 0.05
const _VOLT_PRE_SWEEPS = 8

"""R(s_q)·φ_q = (K_static − D)φ + Σ_g c_g(s_q)·Kc_g·φ（逐节点列；
镜像 _volterra_rhs。y 复用缓冲）。"""
function volterra_rhs!(y::Matrix{ComplexF64}, s::AbstractSambeSetup,
                       phi::AbstractMatrix{ComplexF64},
                       c_nodes::Matrix{Float64})
    mul!(y, s.K_static, phi)
    y .-= s.diag .* phi
    for (gi, Kc) in enumerate(s.Kc_ext)
        # Julia 广播按前导维对齐：包络行须显式成 (1, n_nodes)
        y .+= reshape(c_nodes[gi, :], 1, :) .* (Kc * phi)
    end
    return y
end

"""段传播（稀疏路径；镜像 _subspace_segment）。

φ = e^{+iD(t−ta)}ψ：φ(t) = X + (−iC)∫ e^{+iD(t−s)}R(s)φ(s)ds，
GL 节点配置为线性系统 (I−T)φ = b，GMRES 直解（预条件 = 因果 cumsum
Picard 扫掠）。节点倍增差 + GMRES 残差联合估计；未达 tol 或 GMRES
未收敛 → est=Inf 交主循环细分（不返回垃圾态）。
返回 (psi_out, est)。"""
function segment_sparse(s::AbstractSambeSetup, ta::Real, tb::Real,
                        X::Vector{ComplexF64}, tol::Float64;
                        ladder::Vector{Int} = _NODE_LADDER_SPARSE,
                        const_c::Union{Nothing,Vector{Float64}} = nothing)
    C = _CONV
    h = tb - ta
    dim = s.dim
    diag = s.diag
    prev_phi_tb = nothing
    est_eff = Inf
    phi_tb_ip = nothing
    for n_nodes in ladder
        glx, glw, Wpart = volterra_rule(n_nodes)
        nodes = ta .+ 0.5 .* h .* (glx .+ 1.0)
        ws = 0.5 .* h .* glw
        Wp = Wpart .* (0.5 * h)                       # 映射回 [ta, t_q]
        c_nodes = if const_c !== nothing
            repeat(reshape(const_c, :, 1), 1, n_nodes)
        else
            s.envs(nodes)                             # (G, n)
        end
        Pi = exp.(1im .* C .* (diag .* (nodes' .- ta)))    # (dim, n)
        Pinv = exp.(-1im .* C .* (diag .* (nodes' .- ta)))
        dim_n = dim * n_nodes
        ybuf = Matrix{ComplexF64}(undef, dim, n_nodes)

        _t_full = let s = s, c_nodes = c_nodes, Wp = Wp, Pi = Pi,
                        Pinv = Pinv, dim = dim, n_nodes = n_nodes,
                        ybuf = ybuf
            v -> begin
                phi = reshape(v, dim, n_nodes)
                volterra_rhs!(ybuf, s, phi, c_nodes)
                Sy = (Pinv .* ybuf) * transpose(Wp)   # S[d,q'] = Σ_q Wp[q',q]·Pinv·y
                return vec((-1im * C) .* (Pi .* Sy))
            end
        end

        bvec = repeat(X, n_nodes)                     # (dim,n) 逐列复制后展平
        Aop = v -> v .- _t_full(v)
        atol = max(tol * _VOLT_PICARD_FRAC * max(norm(bvec), 1.0), 1e-15)

        _pre = let s = s, c_nodes = c_nodes, Pi = Pi, Pinv = Pinv,
                    ws = ws, dim = dim, n_nodes = n_nodes, ybuf = ybuf
            v -> begin
                r = reshape(v, dim, n_nodes)
                phi = zeros(ComplexF64, dim, n_nodes)
                for _ in 1:_VOLT_PRE_SWEEPS
                    volterra_rhs!(ybuf, s, phi, c_nodes)
                    S = cumsum(Pinv .* ybuf .* reshape(ws, 1, :); dims = 2)
                    phi = r .+ (-1im * C) .* (Pi .* S)
                end
                return vec(phi)
            end
        end

        sol, info = gmres_lp(Aop, bvec, _pre, dim_n;
                             restart = min(40, dim_n),
                             maxiter = _VOLT_MAX_ITER, atol = atol)
        if info != 0                                  # 未收敛 → 细分信号
            est_eff = Inf
            break
        end
        phi = reshape(sol, dim, n_nodes)
        res = max(norm(Aop(sol) .- bvec) / max(norm(bvec), 1.0), 1e-16)
        # 端点：全区间 GL 求积（同一节点集，与阶梯自洽）
        volterra_rhs!(ybuf, s, phi, c_nodes)
        S_tb = vec(sum(Pinv .* ybuf .* reshape(ws, 1, :); dims = 2))
        phi_tb_ip = X .+ (-1im * C) .* exp.(1im .* C .* diag .* h) .* S_tb
        if prev_phi_tb !== nothing
            est_eff = max(norm(phi_tb_ip .- prev_phi_tb), res)
            est_eff <= tol && break
        end
        prev_phi_tb = phi_tb_ip
    end
    if !isfinite(est_eff)
        return copy(X), Inf                           # 细分信号（不返回垃圾态）
    end
    X_tb = exp.(-1im .* C .* diag .* h) .* phi_tb_ip
    return X_tb, est_eff
end
