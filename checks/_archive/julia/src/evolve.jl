# evolve.jl — 主控：M 探针 → Sambe 结构 → 常数包络快速路径 / 自适应分段
# 主循环（稠密：本征基缺陷；稀疏：IP Volterra / Wosc）→ loss → 约化布居。
# 逐行镜像 evolve_exact_pair(...) 主流程。

using LinearAlgebra
using Statistics
using Printf

"用户取消（CLI 转退出码 2）"
struct Cancelled <: Exception end

"取消文件探测（CLI 传路径，库调用可传 nothing）"
is_cancelled(path::Union{Nothing,AbstractString}) =
    path !== nothing && isfile(path)

"""演化结果（与 pair_evolve/evolve_exact_pair 返回契约同键）"""
Base.@kwdef struct PairResult
    t::Vector{Float64}
    pops_2d::Matrix{Float64}        # (n_t, N²)
    pops_A::Matrix{Float64}         # (n_t, N)
    pops_B::Matrix{Float64}         # (n_t, N)
    loss_A::Vector{Float64}
    loss_B::Vector{Float64}
    n_segments::Int
    max_diff::Float64
    dt_min_hit::Bool
    max_norm_dev::Float64
    photon_blocks::Vector{Int}      # 每组 2M+1
    M::Int
    probe_diff::Float64
    timing::Vector{Float64}         # [总, 探针, 段传播, 其他]
    psi_amp::Union{Nothing,Vector{ComplexF64}} = nothing
end

"约化布居（镜像 _reduce_A：pops_A[t,i]=Σ_j pops_2d[t,i·N+j]；B 按 j）"
function reduce_A(pops_2d::AbstractMatrix, N::Int)
    n_t = size(pops_2d, 1)
    out = Matrix{Float64}(undef, n_t, N)
    for ti in 1:n_t
        m = reshape(@view(pops_2d[ti, :]), N, N)     # M[j1, i1]（列主序转置）
        out[ti, :] .= vec(sum(m; dims = 1))
    end
    return out
end
function reduce_B(pops_2d::AbstractMatrix, N::Int)
    n_t = size(pops_2d, 1)
    out = Matrix{Float64}(undef, n_t, N)
    for ti in 1:n_t
        m = reshape(@view(pops_2d[ti, :]), N, N)     # M[j1, i1]（列主序转置）
        out[ti, :] .= vec(sum(m; dims = 2))
    end
    return out
end

"""两体第一性原理含时演化主入口（镜像 evolve_exact_pair）。

进度/取消：progress(frac)；cancelled()→Bool（真则抛 Cancelled）。
record_amp=true 时结果附 psi_amp（末时刻物理幅值）。"""
function evolve_pair(req::Request; record_amp::Bool = false,
                     progress = nothing, cancelled = nothing, log = nothing)
    t0 = time()
    plog = log === nothing ? (_ -> nothing) : log
    pprog = progress === nothing ? (_ -> nothing) : progress
    iscancelled = cancelled === nothing ? () -> false : cancelled

    t = req.t_grid
    n_t = length(t)
    T = t[end]
    N = req.N
    N2 = N * N
    h_floor = max(T / req.dt_min_frac, 1e-9)
    ladder = req.node_ladder

    psiA = req.psi0_A
    psiB = req.psi0_B
    psi0_2d = product_psi(psiA, psiB, N)

    has_gamma = req.gamma_A !== nothing
    gamma_ns = has_gamma ? req.gamma_A .+ req.gamma_B : nothing

    # ---- M 确定（skip_probe / 分块探针 / 全空间探针）----
    blocked = req.block_mF && req.mF_states !== nothing &&
              req.charges !== nothing && !any(isnan, req.charges)
    # 分块 I0 从初态主分量的 mF_sum 定（积型初态 → 唯一类）
    mF_pair_all = blocked ? Float64[req.mF_states[i] + req.mF_states[j]
                           for i in 1:N for j in 1:N] : Float64[]
    k0_dom = blocked ? argmax(abs.(psi0_2d)) : 1
    mF_init = blocked ? mF_pair_all[k0_dom] : 0.0 && req.mF_states !== nothing &&
              req.charges !== nothing && !any(isnan, req.charges)
    sp_dispatch = (stp, ta, tb, Xc, tl) ->
                  (req.sparse_backend == 1 ?
                   segment_sparse_wosc(stp, ta, tb, Xc, tl) :
                   segment_sparse(stp, ta, tb, Xc, tl))
    if req.skip_probe
        M = m_analytic(req, req.tol)
        mdiff = NaN
        tp = 0.0
        plog(@printf("[pair_exact] skip_probe: m_analytic M=%d", M))
    else
        tp = @elapsed M, mdiff = if blocked
            m_probe_blocked(req, req.vdd, gamma_ns, psi0_2d, T;
                            m_max = req.m_max, log = plog,
                            sp_path = sp_dispatch)
        else
            m_probe(req, req.vdd, gamma_ns, psi0_2d, T;
                    m_max = req.m_max, log = plog, sp_path = sp_dispatch)
        end
        plog(@printf("[pair_exact] 光子块收敛：M=%d%s", M,
                     isnan(mdiff) ? "（达上限）" : @sprintf("（探针差 %.2e）", mdiff)))
    end

    setup_full = SambeSetup(req, M, req.vdd, gamma_ns)
    setup = if blocked
        I0 = mF_init + sum(-req.charges[g] * M
                           for g in 1:length(req.charges); init = 0.0)
        cidx, kof, pof = class_index(setup_full, req.mF_states,
                                     req.charges, I0)
        plog(@printf("[pair_exact] mF 分块：可达类 %d / %d 态",
                     length(cidx), setup_full.dim))
        SubSetup(setup_full, cidx, kof, pof, req.mF_states;
                 dense_max = req.dense_max)
    else
        setup_full
    end

    # ---- 全常数包络快速路径（稠密；镜像：逐场 64 点 mean 求和）----
    n_chk = 64
    ts_chk = range(0.0, T; length = n_chk)
    c_chk = setup.envs(ts_chk)                    # (G, n_chk) 组内包络和
    all_const = isempty(setup.envs.envs) ||
                maximum(maximum(c) - minimum(c) for c in eachrow(c_chk);
                        init = 0.0) <= 1e-9
    c_peaks = isempty(setup.envs.envs) ? Float64[] :
              vec(mean(c_chk; dims = 2))          # 和的均值 = 均值的和
    report = (n_segments = Ref(0), max_diff = Ref(0.0),
              dt_min_hit = Ref(false))

    pops_2d = Matrix{Float64}(undef, n_t, N2)
    X_final = Vector{ComplexF64}(undef, setup.dim)
    t_seg = 0.0

    if (all_const || isempty(setup.envs.envs)) && setup.dense
        # 一次本征分解 + 全输出时刻向量化重构（支持非厄米）
        K = K_seg(setup, c_peaks)
        de = dense_decomp(K)
        X0 = embed_pair(psi0_2d, setup)
        a0 = de.left * X0
        Aexp = exp.(-1im * _CONV .* (de.evals .* permutedims(t)))  # (dim,n_t)
        Xall = de.evecs * (a0 .* Aexp)                              # (dim,n_t)
        if setup isa SubSetup
            # 分块：类态按 k 映射散回（类内布局非 (N2,P) 积）
            PH = exp.(-1im .* (setup.nphase .* permutedims(t)))     # (dim_c,n_t)
            for ti in 1:n_t
                amp = zeros(ComplexF64, N2)
                for m in 1:setup.dim
                    amp[setup.k_of[m]] += PH[m, ti] * Xall[m, ti]
                end
                pops_2d[ti, :] .= abs2.(amp)
            end
        else
            Xm = reshape(Xall, N2, setup.P, :)                      # (N2,P,n_t)
            PH = exp.(-1im .* (setup.nphase .* permutedims(t)))     # (P,n_t)
            for ti in 1:n_t
                psi = vec(sum(reshape(PH[:, ti], 1, :) .* Xm[:, :, ti];
                              dims = 2))
                pops_2d[ti, :] .= abs2.(psi)
            end
        end
        X_final .= Xall[:, end]
        report.n_segments[] = 1
    else
        # ---- 自适应分段主循环 ----
        X = embed_pair(psi0_2d, setup)

        function rec2(ta::Float64, tb::Float64, Xcur::Vector{ComplexF64})
            iscancelled() && throw(Cancelled())
            if setup.dense
                psi_out, est = segment_dense(setup, ta, tb, Xcur, req.tol,
                                             ladder)
            elseif req.sparse_backend == 1
                psi_out, est = segment_sparse_wosc(setup, ta, tb, Xcur,
                                                   req.tol)
            else
                psi_out, est = segment_sparse(setup, ta, tb, Xcur, req.tol)
            end
            if (!isfinite(est) || est > req.tol) && (tb - ta) > h_floor
                tm = 0.5 * (ta + tb)
                Xm_ = rec2(ta, tm, Xcur)
                return rec2(tm, tb, Xm_)
            end
            report.max_diff[] = max(report.max_diff[],
                                    isfinite(est) ? est : 1e3)
            report.n_segments[] += 1
            if !isfinite(est) || est > req.tol
                report.dt_min_hit[] = true
            end
            return psi_out
        end

        tprev = 0.0
        tseg_acc = 0.0
        for i in 2:n_t
            pprog(min(tprev / T, 0.999))
            tsi = @elapsed X = rec2(tprev, Float64(t[i]), X)
            tseg_acc += tsi
            pops_2d[i, :] .= physical_pops(setup, X, t[i])
            tprev = Float64(t[i])
        end
        pops_2d[1, :] .= abs2.(psi0_2d)
        X_final .= X
        t_seg = tseg_acc
    end

    # ---- loss 曲线（A/B 各自速率 × 布居梯形积分，镜像 Python）----
    loss_A = zeros(n_t)
    loss_B = zeros(n_t)
    norm_dev = 0.0
    if has_gamma
        dt_arr = diff(t)
        rate_A = pops_2d * req.gamma_A              # /ns
        rate_B = pops_2d * req.gamma_B
        loss_A[2:end] .= cumsum(0.5 .* (rate_A[2:end] .+ rate_A[1:end-1]) .*
                                dt_arr)
        loss_B[2:end] .= cumsum(0.5 .* (rate_B[2:end] .+ rate_B[1:end-1]) .*
                                dt_arr)
        norm_dev = float(maximum(abs.(vec(sum(pops_2d; dims = 2)) .+
                                      loss_A .+ loss_B .- 1.0)))
    else
        norm_dev = float(maximum(abs.(vec(sum(pops_2d; dims = 2)) .- 1.0)))
    end

    pops_A = reduce_A(pops_2d, N)
    pops_B = reduce_B(pops_2d, N)

    plog(@printf("[pair_exact] 完成：%d 段，最大残差 %.2e（tol=%.1e），范数偏差 %.2e",
                 report.n_segments[], report.max_diff[], req.tol, norm_dev))

    psi_amp = nothing
    if record_amp
        psi_amp = physical_amp(setup, X_final, T)
    end

    return PairResult(t = t, pops_2d = pops_2d, pops_A = pops_A,
                      pops_B = pops_B, loss_A = loss_A, loss_B = loss_B,
                      n_segments = report.n_segments[],
                      max_diff = report.max_diff[],
                      dt_min_hit = report.dt_min_hit[],
                      max_norm_dev = max(norm_dev, 0.0),
                      photon_blocks = fill(2 * M + 1, length(req.groups)),
                      M = M, probe_diff = mdiff,
                      timing = [time() - t0, tp, t_seg, 0.0],
                      psi_amp = psi_amp)
end
