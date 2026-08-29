# wosc.jl — Wosc 振荡解析求积权重（Filon 型；交接文档 §5.3 路线图 1）。
#
# 思想：IP Volterra 积分的被积函数 = 快振荡相位因子 e^{−iκx} × 慢变部分
# （Lagrange 基 / 包络×基）。gen-3 用 GL 节点数值积整个被积函数 →
# 节点数 n ~ C·ΔD_max·h（振荡求积瓶颈，段长被钉死）；Wosc 把振荡因子
# 解析积分（(e^{iΔD·h}−1)/(iΔD) 类闭式），节点只留给慢变部分。
#
# 权重（移植 tempsimu/_pair_wosc_test2.py 的 _colloc_Wosc 契约）：
#   Zpart[r, s] = ∫_{−1}^{x_r} L_s(x)·e^{λx} dx     （部分积分，Volterra 上限）
#   Zfull[s]    = ∫_{−1}^{+1} L_s(x)·e^{λx} dx      （全区间，段端点）
# 反导数 J_p 递推：∫x^p e^{λx}dx = x^p e^{λx}/λ − (p/λ)·∫x^{p−1}e^{λx}dx
#   J_0(z) = e^{λz}/λ；λ=0（及 |λ| 极小）退化为多项式积分公式。

using LinearAlgebra

# Wosc 求积节点阶梯（Vandermonde 病态上界 → 封顶 32）
const _NODE_LADDER_WOSC = [16, 32]

"反导数 J_p(z)，p = 0..deg（∫ x^p e^{λx}dx 的原函数在 z 处的值）"
function _jpoly(z::Real, lam::ComplexF64, deg::Int)
    J = Vector{ComplexF64}(undef, deg + 1)
    ez = exp(lam * z)
    J[1] = ez / lam
    for p in 1:deg
        J[p+1] = z^p * ez / lam - (p / lam) * J[p]
    end
    return J
end

"λ→0 极限：∫_{−1}^{z} x^p dx 的原函数差（调用方做定积分）"
function _jpoly_zero!(J::Vector{ComplexF64}, z::Real, deg::Int)
    zp = 1.0
    for p in 0:deg
        J[p+1] = zp / (p + 1)
        zp *= z
    end
    return J
end

"""单个 λ 的 Wosc 权重。

返回 (Zpart, Zfull)：Zpart[r, s] = ∫_{−1}^{x_r} L_s(x)e^{λx}dx，
Zfull[s] = ∫_{−1}^{1} L_s(x)e^{λx}dx。λ=0 时退化为多项式权重
（Zpart ≡ volterra_rule 的 Wpart，Zfull ≡ GL 权重 w）。"""
function wosc_weights(n::Integer, lam::ComplexF64;
                      glx::Vector{Float64}, Bmat::Matrix{Float64})
    if abs(lam) < 1e-8
        # λ→0：多项式公式（消除小 |λ| 递推相消）
        Zpart = Matrix{ComplexF64}(undef, n, n)
        Zfull = Vector{ComplexF64}(undef, n)
        for r in 1:n
            for s in 1:n
                acc = 0.0
                for p in 0:n-1
                    acc += Bmat[s, p+1] * (glx[r]^(p + 1) - _m1(p + 1)) / (p + 1)
                end
                Zpart[r, s] = acc
            end
        end
        for s in 1:n
            acc = 0.0
            for p in 0:n-1
                acc += Bmat[s, p+1] * (1.0 - _m1(p + 1)) / (p + 1)
            end
            Zfull[s] = acc
        end
        return Zpart, Zfull
    end
    Jm1 = _jpoly(-1.0, lam, n - 1)
    Jp1 = _jpoly(1.0, lam, n - 1)
    Zpart = Matrix{ComplexF64}(undef, n, n)
    Zfull = Vector{ComplexF64}(undef, n)
    for r in 1:n
        Jr = _jpoly(glx[r], lam, n - 1)
        for s in 1:n
            acc = zero(ComplexF64)
            for p in 0:n-1
                acc += Bmat[s, p+1] * (Jr[p+1] - Jm1[p+1])
            end
            Zpart[r, s] = acc
        end
    end
    for s in 1:n
        acc = zero(ComplexF64)
        for p in 0:n-1
            acc += Bmat[s, p+1] * (Jp1[p+1] - Jm1[p+1])
        end
        Zfull[s] = acc
    end
    return Zpart, Zfull
end

# 辅助：x_r^p 与 (−1)^p（避免上面的解析式可读性差）
@inline x_g(r::Int, glx::Vector{Float64}, p::Int) = glx[r]^(p)
@inline _m1(p::Int) = (isodd(p) ? -1.0 : 1.0)

"""Lagrange 系数表 Bmat[s, p+1]（x^p 展开系数；镜像 _colloc_rule 的 B）。
⚠ n ≳ 20 时 Float64 Vandermonde 求逆病态（n=32 误差 ~8e-9，n=64 全废）——
生产路径用 stable 版（BigFloat 求逆后转 Float64，每 n 缓存一次）。"""
function lagrange_coeffs(n::Integer, glx::Vector{Float64})
    V = [glx[s]^p for p in 0:n-1, s in 1:n]
    return inv(V)
end

const _LAGRANGE_BIG_CACHE = Dict{Int,Matrix{Float64}}()

"""稳定版 Lagrange 系数表（BigFloat 求逆 → Float64；按 n 缓存）。"""
function lagrange_coeffs_stable(n::Integer, glx::Vector{Float64})
    get!(_LAGRANGE_BIG_CACHE, n) do
        setprecision(BigFloat, 256) do
            gb = BigFloat.(glx)
            V = [gb[s]^p for p in 0:n-1, s in 1:n]
            Bbig = inv(V)
            Float64.(Bbig)
        end
    end
end

# ============================================================
# Wosc 段传播子（S5）：对角相互作用绘景 + 逐能组解析振荡权重
# ============================================================
# IP Volterra：φ(t) = X + (−iC)∫ e^{iC·d_a(t_r−s)}·(Rφ)(s) ds。
# 相位因子按行能量 d_a 分组解析积分（Filon：慢因子 = Rφ 在节点插值）：
#   I_a(r) = (h/2)·e^{iκ·x_r}·Σ_s Z̃(λ)[r,s]·(Rφ)(x_s)，κ = C·Re(d_a)·h/2
# 耗散（Im(d)≠0）：e^{−C·Im(d)·Δ} 拆成行因子×节点因子，均不进振荡权重。
# gen-3 的 Wp·Pinv 数值积分在 κ 大时需要 n ~ κ 个节点；Wosc 与 κ 无关，
# 段长只受包络/Filon 插值误差限制。
#
# ⚠ 精度上界（实测，λ=−1500im vs BigFloat 参考）：
#   n=16 → 9e-15；n=32 → 8.4e-9；n=64 → 7e-2（Vandermonde 求逆病态）。
#   故 Wosc 阶梯封顶 32；est 仍超 tol 时走真中点细分（插值误差主导，
#   细分比加节点更有效）。gen-3 的 64/128 级权重同样病态（被 early-accept
#   掩盖），迁移注意事项见 docs/Julia引擎交接文档.md。

"相异 Re(diag) 能量分组（每 setup 一次）"
struct WoscGroups
    vals::Vector{Float64}              # 相异能量（MHz）
    rows::Vector{Vector{Int}}          # 各组的行索引
end
function WoscGroups(s::AbstractSambeSetup)
    dict = Dict{Float64,Vector{Int}}()
    for a in 1:s.dim
        k = round(real(s.diag[a]); digits = 14)   # 吸收浮点尾差
        push!(get!(dict, k, Int[]), a)
    end
    vals = collect(keys(dict))
    rows = [dict[v] for v in vals]
    return WoscGroups(vals, rows)
end

"""Wosc 段传播子（镜像 segment_sparse 的接口与稳健性约定）。

返回 (psi_out, est)。GMRES/节点阶梯/est 语义与 gen-3 完全一致；
预条件 = 因果截断的解析权重 Picard 扫掠。"""
function segment_sparse_wosc(s::AbstractSambeSetup, ta::Real, tb::Real,
                             X::Vector{ComplexF64}, tol::Float64;
                             ladder::Vector{Int} = _NODE_LADDER_WOSC,
                             const_c::Union{Nothing,Vector{Float64}} = nothing,
                             wg::WoscGroups = WoscGroups(s))
    C = _CONV
    h = tb - ta
    dim = s.dim
    diag = s.diag
    prev_phi_tb = nothing
    est_eff = Inf
    phi_tb_ip = nothing
    for n_nodes in ladder
        glx, glw = _gl_cached(n_nodes)
        nodes = ta .+ 0.5 .* h .* (glx .+ 1.0)
        ws = 0.5 .* h .* glw
        c_nodes = if const_c !== nothing
            repeat(reshape(const_c, :, 1), 1, n_nodes)
        else
            s.envs(nodes)
        end
        dim_n = dim * n_nodes
        ybuf = Matrix{ComplexF64}(undef, dim, n_nodes)

        # 逐能组解析权重（λ = −iκ；κ = C·Re(d)·h/2）
        Bmat = lagrange_coeffs_stable(n_nodes, glx)
        Zpart_v = Vector{Matrix{ComplexF64}}(undef, length(wg.vals))
        phase_v = Vector{Matrix{ComplexF64}}(undef, length(wg.vals))
        for (gi, d) in enumerate(wg.vals)
            kap = C * d * h / 2
            lam = -1im * kap
            Zt, _ = wosc_weights(n_nodes, lam; glx = glx, Bmat = Bmat)
            # Zp = (h/2)·Z̃（不含相位）；行相位 e^{iκx_r} 在 Sd 上乘一次
            phase_v[gi] = reshape(exp.(1im .* kap .* glx), 1, :)
            Zpart_v[gi] = (h / 2) .* Zt
        end

        _t_full = let s = s, c_nodes = c_nodes, dim = dim, n_nodes = n_nodes,
                        ybuf = ybuf, wg = wg, Zpart_v = Zpart_v,
                        phase_v = phase_v, ta = ta, h = h, C = C
            v -> begin
                phi = reshape(v, dim, n_nodes)
                volterra_rhs!(ybuf, s, phi, c_nodes)
                S = zeros(ComplexF64, dim, n_nodes)
                for (gi, rows) in enumerate(wg.rows)
                    Y = ybuf[rows, :]
                    if s.has_gamma
                        # 节点因子 e^{−C·Im(d)·(s−ta)}（慢，节点采样）
                        dI = imag(s.diag[rows[1]])
                        Y .*= reshape(exp.(-C .* dI .* (nodes .- ta)), 1, :)
                    end
                    Sd = (Y * transpose(Zpart_v[gi])) .* phase_v[gi]
                    S[rows, :] .= Sd
                end
                # 注意：X 只出现在 RHS（bvec），不进算符（镜像 gen-3 结构）
                return vec((-1im * C) .* S)
            end
        end

        bvec = repeat(X, n_nodes)
        Aop = v -> v .- _t_full(v)
        atol = max(tol * _VOLT_PICARD_FRAC * max(norm(bvec), 1.0), 1e-15)

        _pre = let s = s, c_nodes = c_nodes, dim = dim, n_nodes = n_nodes,
                    ybuf = ybuf, ta = ta, h = h, C = C, glx = glx, glw = glw
            # 预条件 = gen-3 式因果 cumsum（GL 权重；仅求近似求逆，
            # tril 解析截断在 n=32 会发散——实测）
            Pi = exp.(1im .* C .* (s.diag .* permutedims(nodes .- ta)))
            Pinv = exp.(-1im .* C .* (s.diag .* permutedims(nodes .- ta)))
            v -> begin
                r = reshape(v, dim, n_nodes)
                phi = zeros(ComplexF64, dim, n_nodes)
                for _ in 1:_VOLT_PRE_SWEEPS
                    volterra_rhs!(ybuf, s, phi, c_nodes)
                    S = cumsum(Pinv .* ybuf .* reshape(0.5 * h .* glw, 1, :);
                               dims = 2)
                    phi = r .+ (-1im * C) .* (Pi .* S)
                end
                return vec(phi)
            end
        end

        sol, info = gmres_lp(Aop, bvec, _pre, dim_n;
                             restart = min(40, dim_n),
                             maxiter = _VOLT_MAX_ITER, atol = atol)
        res = max(norm(Aop(sol) .- bvec) / max(norm(bvec), 1.0), 1e-16)
        if get(ENV, "ELP_DEBUG", "") == "1"
            println(stderr, "[wosc dbg] n=", n_nodes, " info=", info,
                    " res=", res)
        end
        if info != 0
            est_eff = Inf
            break
        end
        phi = reshape(sol, dim, n_nodes)
        # 端点：全区间解析权重
        volterra_rhs!(ybuf, s, phi, c_nodes)
        S_tb = zeros(ComplexF64, dim)
        for (gi, rows) in enumerate(wg.rows)
            Y = ybuf[rows, :]
            if s.has_gamma
                dI = imag(s.diag[rows[1]])
                Y .*= reshape(exp.(-C .* dI .* (nodes .- ta)), 1, :)
            end
            _, Zf = wosc_weights(n_nodes, -1im * C * wg.vals[gi] * h / 2;
                                 glx = glx, Bmat = Bmat)
            S_tb[rows] .= (h / 2) .* exp.(1im .* (C * wg.vals[gi] * h / 2)) .*
                          (Y * Zf)
        end
        phi_tb_ip = X .+ (-1im * C) .* S_tb
        if prev_phi_tb !== nothing
            est_eff = max(norm(phi_tb_ip .- prev_phi_tb), res)
            est_eff <= tol && break
        end
        prev_phi_tb = phi_tb_ip
    end
    if !isfinite(est_eff)
        return copy(X), Inf
    end
    # 端点回 IP→薛定谔绘景：X_tb = e^{−iC·d·h}·φ_tb（逐行）
    X_tb = similar(phi_tb_ip)
    for (gi, rows) in enumerate(wg.rows)
        dI = s.has_gamma ? imag(s.diag[rows[1]]) : 0.0
        X_tb[rows] .= phi_tb_ip[rows] .* exp.(-1im .* C .* wg.vals[gi] .* h .+
                                              C .* dI .* h)
    end
    return X_tb, est_eff
end
