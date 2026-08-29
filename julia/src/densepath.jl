# densepath.jl — 稠密路径：冻结中点本征分解 + 本征基 Duhamel 缺陷修正。
# 逐行镜像 _dense_decomp / _prop_step / _defect_pass。

using LinearAlgebra

"段算符本征分解结果：evals + evecs + left（left = evecs† 厄米 / inv(evecs) 非厄米）"
struct DenseEig{E,V,L}
    evals::E
    evecs::V
    left::L
end

"""稠密段算符本征分解（自动判厄米/非厄米；镜像 _dense_decomp）。

厄米判定用绝对容差 1e-10：能量标度可达 1e6 MHz，默认 rtol 会吞掉
NonHermitian 虚部（~Γ 量级 ≪ 1e-5·|E|），误判后 eigh 丢弃衰减项。"""
function dense_decomp(K::AbstractMatrix)
    herm = all(abs.(K - adjoint(K)) .<= 1e-10)
    if herm
        ev = eigen(Hermitian(K))
        return DenseEig(ev.values, ev.vectors, adjoint(ev.vectors))
    end
    ev = eigen(K)
    return DenseEig(ev.values, ev.vectors, inv(ev.vectors))
end

"冻结段精确指数传播（镜像 _prop_step）：X(tb) = V·(a0·e^{−iC·dt·ε})"
function prop_step(s::AbstractSambeSetup, K::AbstractMatrix, ta::Real, tb::Real,
                   X::AbstractVector{ComplexF64}, de::DenseEig)
    dt = tb - ta
    a0 = de.left * X
    return de.evecs * (a0 .* exp.(-1im * _CONV * dt .* de.evals))
end

"""本征基 Duhamel 缺陷扫描（镜像 _defect_pass，单级冻结被积态）。

δ(tb) = −iC·Σ_q w_q·U_f(tb,t_q)·ΔK(t_q)·ψ_f(t_q)；
ΔK(t_q) = Σ_g (c_g(t_q) − c_g(mid))·Kc_g。全程本征基向量运算。
返回 (delta_tb, max_dc)。"""
function defect_pass(s::AbstractSambeSetup, ta::Real, tb::Real,
                     X::AbstractVector{ComplexF64},
                     nodes::Vector{Float64}, ws::Vector{Float64},
                     c_nodes::Matrix{Float64}, c_mid::Vector{Float64},
                     de::DenseEig)
    G, n = size(c_nodes)
    dc = c_nodes .- reshape(c_mid, G, 1)          # (G, n)
    max_dc = isempty(dc) ? 0.0 : float(maximum(abs.(dc)))
    if max_dc < 1e-12
        return zeros(ComplexF64, s.dim), max_dc
    end
    a0 = de.left * X
    # 冻结态在节点的物理演化（本征基系数 × 相位）→ 物理基 Psif (dim, n)
    phi_tq = exp.(-1im * _CONV .* (de.evals .* (nodes' .- ta)))  # (dim,n)
    Psif = de.evecs * (a0 .* phi_tq)
    delta = zeros(ComplexF64, s.dim)
    for q in 1:n
        v = zeros(ComplexF64, s.dim)
        for gi in 1:G
            if dc[gi, q] != 0.0
                v .+= dc[gi, q] .* (s.Kc_ext[gi] * Psif[:, q])
            end
        end
        if any(!iszero, v)
            b0 = de.left * v
            up = de.evecs * (b0 .* exp.(-1im * _CONV .* de.evals .* (tb - nodes[q])))
            delta .+= ws[q] .* up
        end
    end
    return (-1im * _CONV) .* delta, max_dc
end

"组中点包络列表 c_mids（镜像 setup.c_mids）"
c_mids(s::AbstractSambeSetup, t::Real) =
    [s.envs(g, t) for g in 1:length(s.envs.envs)]

"""稠密段传播（镜像 rec2 稠密分支）：冻结 eigh + 节点阶梯缺陷，
est = 阶梯差；est ≤ tol 或包络平坦即接受。返回 (psi_out, est)。"""
function segment_dense(s::AbstractSambeSetup, ta::Real, tb::Real,
                       X::AbstractVector{ComplexF64}, tol::Float64,
                       ladder::Vector{Int})
    h = tb - ta
    K = K_seg(s, c_mids(s, 0.5 * (ta + tb)))
    de = dense_decomp(K)
    a0 = de.left * X
    psi_frozen = de.evecs * (a0 .* exp.(-1im * _CONV * (tb - ta) .* de.evals))
    est = 0.0
    psi_out = psi_frozen
    prev = nothing
    for n in ladder
        glx, glw = gl_nodes(n)
        nodes = ta .+ 0.5 * h .* (glx .+ 1.0)
        ws = 0.5 * h .* glw
        c_nodes = s.envs(nodes)
        c_mid = c_mids(s, 0.5 * (ta + tb))
        delta, max_dc = defect_pass(s, ta, tb, X, nodes, ws, c_nodes,
                                    c_mid, de)
        if max_dc < 1e-12                       # 包络平坦：冻结即精确
            return psi_frozen, 0.0
        end
        cur = psi_frozen .+ delta
        psi_out = cur
        if prev !== nothing
            est = norm(cur - prev)
            est <= tol && break
        end
        prev = cur
    end
    return psi_out, est
end
