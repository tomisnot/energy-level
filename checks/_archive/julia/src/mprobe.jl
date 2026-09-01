# mprobe.jl — 光子块阶数探针，镜像 _pair_m_probe。
# 常数峰值包络传播，M vs M−1 布居差 ≤ tol 收敛（接受 M−1）。
# 稀疏路径：探针窗口 T_pr = min(T, max(10ns, T/4))，1ns 分段链，
# 截断泄漏布居差 ~T² → 按 (T/T_pr)² 外推（保守上界）。

const _ENV_PROBE_N = 2001

"组峰值包络（每组：该组全部场包络之和在 [0,T] 的最大值）"
function group_peaks(s::AbstractSambeSetup, T::Real)
    ts = range(0.0, T; length = _ENV_PROBE_N)
    G = length(s.envs.envs)
    return Float64[maximum(s.envs(g, ts)) for g in 1:G]
end

"""解析下界（镜像单原子 _m_analytic）：弱驱动截断 (r/4)^(2(M−1)) ≤ tol。
r_g = max|K2_g| / ω_g（组峰值耦合/载频）。防止探针在中等 r 下低估 M。"""
function m_analytic(req::Request, tol::Float64)
    isempty(req.groups) && return 2
    r = 0.0
    for g in req.groups
        mx = nnz(g.K2) == 0 ? 0.0 : maximum(abs.(nonzeros(g.K2)))
        r = max(r, mx / g.omega)
    end
    r <= 0 && return 2
    lr = log(r / 4)
    lr >= 0 && return req.m_max          # 非弱驱动：下界无约束，走探针
    M = ceil(Int, 1 + log(tol) / (2 * lr))
    return clamp(M, 2, req.m_max)
end

"""M 递增收敛探针。返回 (M_accepted, probe_diff)。

M 从 max(2, m_start) 起步；达 m_max 上限返回 (m_max, NaN)。
sp_path: 稀疏段传播子函数 (setup, ta, tb, X, tol) → (X_out, est)，
在 S3 接入（稀疏探针用）；此前稀疏探针抛错。"""
function m_probe(req::Request, vdd, gamma_ns, psi0_2d, T::Real;
                 m_max::Int = 8, m_start::Int = 2,
                 sp_path = nothing, log = nothing)
    plog = log === nothing ? (_ -> nothing) : log
    run_at(Mc) = begin
        stp = SambeSetup(req, Mc, vdd, gamma_ns)
        c_list = group_peaks(stp, T)
        K = K_seg(stp, c_list)
        X = embed_pair(psi0_2d, stp)
        if stp.dense
            de = dense_decomp(K)
            X = prop_step(stp, K, 0.0, T, X, de)
            return physical_pops(stp, X, T), 1.0
        end
        sp_path === nothing &&
            error("稀疏探针需 S3 段传播子（segment_sparse）")
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
            plog("[pair_exact] 探针 M=$Mc vs M-1 差=$(d)（外推 ×$(max(ext, ext_prev))）")
            d <= req.tol && return Mc - 1, d
        end
        prev = cur
        ext_prev = ext
        Mc >= m_max && return Mc, NaN
        Mc += 1
    end
end
