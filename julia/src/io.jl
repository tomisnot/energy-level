# io.jl — NPZ 请求/响应 schema（schema v1）。
#
# 请求 req.npz（Python 侧导出，镜像 evolve_exact_pair 的输入）：
#   jl_version   int            schema 版本（当前 1）
#   N            int            单原子态数
#   E_pair       (N2,) float64  两体能量对角 MHz（k=i·N+j）
#   vdd_row/col  (nnz,) int64   V_dd 的 COO 行列（空数组 = 无 V_dd）
#   vdd_data     (nnz,) complex128   V_dd 非零元 MHz
#   n_groups     int            相异载频组数 K
#   group_omega  (K,) float64   各组载频 MHz
#   k2r_{g}/k2c_{g}/k2v_{g}     组 g 的 K2 COO 三元组（float64，N²×N²）
#   n_fields     int            场数（组内同频场逐场携带包络，求和成 c_g(t)）
#   f_group      (n_fields,) int64   场 → 组（0-based）
#   f_kind       (n_fields,) int64   0=gauss 1=square 2=sampled
#   f_p1..f_p4   (n_fields,) float64 高斯: center,sigma,peak,0
#                                    方波: start,edge,height,fall_start
#                                    采样: 空（用 f_samp_t_{i}/f_samp_v_{i}）
#   f_samp_t_{i}/f_samp_v_{i}   采样包络数组（仅 kind=2 的场）
#   psi0         (2,) int64     初态 (i_A, i_B)
#   t_grid       (n_t,) float64 输出时刻 ns
#   tol          float64        单旋钮容差
#   gamma_pair   (N2,) float64  寿命速率 /ns（空数组 = 无耗散）
#   m_max        int            光子块阶数上限（默认 8）
#   dense_max    int            稠密/稀疏分界（默认 700）
#   node_ladder  (n,) int64     缺陷求积节点阶梯（默认 16/32/64/128）
#   dt_min_frac  float64        细分下限（默认 2e5）
#
# 响应 resp.npz：ok(1 成功/0 出错/2 用户取消)、err_msg(UInt8 UTF-8)、
#   t, pops_2d (n_t,N2), pops_A/B (n_t,N), loss_A/B (n_t,),
#   n_segments, max_diff, dt_min_hit, max_norm_dev, photon_blocks (K,),
#   timing (4,) [总耗时, 探针, 段传播, 权重构建], threads。
# --echo 模式回显解析结果（G0 往返校验，e_ 前缀）。

using NPZ
using SparseArrays

"逐场包络定义（组内求和成 c_g(t)）"
struct FieldDef
    group::Int          # 0-based 组索引
    env::Envelope
end

"载频组：相异载频 + 单位包络两体耦合"
struct GroupDef
    omega::Float64
    K2::SparseMatrixCSC{Float64,Int}
end

"请求对象（镜像 evolve_exact_pair 的输入参数）"
Base.@kwdef struct Request
    N::Int
    E_pair::Vector{Float64}
    vdd::Union{Nothing,SparseMatrixCSC{ComplexF64,Int}}
    groups::Vector{GroupDef}
    fields::Vector{FieldDef}
    psi0_A::Vector{ComplexF64}                  # 初态（任意矢量直积）
    psi0_B::Vector{ComplexF64}
    t_grid::Vector{Float64}
    tol::Float64
    gamma_A::Union{Nothing,Vector{Float64}}     # /ns（loss 曲线 A 原子）
    gamma_B::Union{Nothing,Vector{Float64}}     # /ns（loss 曲线 B 原子）
    m_max::Int = 8
    dense_max::Int = 700
    node_ladder::Vector{Int} = [16, 32, 64, 128]
    dt_min_frac::Float64 = 2e5
    record_amp::Bool = false
    sparse_backend::Int = 0                      # 0=gen-3, 1=Wosc
    mF_states::Union{Nothing,Vector{Float64}} = nothing   # 各单原子态 mF
    charges::Union{Nothing,Vector{Float64}} = nothing     # 各载频组光子荷
    block_mF::Bool = false                       # mF-光子荷不变量分块开关
    skip_probe::Bool = false                     # 跳过 M 探针（m_analytic 直定 M）
end

_as_int(x) = Int((isa(x, AbstractArray) ? x[1] : x))
_as_float(x) = Float64(isa(x, AbstractArray) ? x[1] : x)

function _coo_matrix(d, rkey, ckey, vkey, ::Type{T}, n::Int) where {T}
    haskey(d, rkey) || return nothing
    r = d[rkey]; c = d[ckey]; v = d[vkey]
    nnz = length(r)
    nnz == 0 && return nothing
    # 请求为 0-based COO（Python 约定）→ Julia 1-based CSC
    return sparse(Int64.(r) .+ 1, Int64.(c) .+ 1, T.(v), n, n)
end

"""读取并校验请求（io_schema v1）"""
function read_request(path::AbstractString)
    d = npzread(path)
    ver = _as_int(d["jl_version"])
    ver == 1 || throw(ArgumentError("schema 版本不支持: $ver（期望 1）"))
    N = _as_int(d["N"])
    N >= 1 || throw(ArgumentError("N ≥ 1"))
    N2 = N * N
    E_pair = Float64.(d["E_pair"])
    length(E_pair) == N2 || throw(ArgumentError("E_pair 长度 ≠ N²"))

    vdd = _coo_matrix(d, "vdd_row", "vdd_col", "vdd_data", ComplexF64, N2)

    K = haskey(d, "n_groups") ? _as_int(d["n_groups"]) : 0
    omega = haskey(d, "group_omega") ? Float64.(d["group_omega"]) :
            Float64[]
    length(omega) == K || throw(ArgumentError("group_omega 长度 ≠ n_groups"))
    groups = Vector{GroupDef}(undef, K)
    for g in 1:K
        k2 = _coo_matrix(d, "k2r_$(g-1)", "k2c_$(g-1)", "k2v_$(g-1)",
                         Float64, N2)
        if k2 === nothing
            k2 = spzeros(Float64, N2, N2)
        end
        groups[g] = GroupDef(omega[g], k2)
    end

    nf = haskey(d, "n_fields") ? _as_int(d["n_fields"]) : 0
    f_group = nf > 0 ? Int.(d["f_group"]) : Int[]
    f_kind = nf > 0 ? Int.(d["f_kind"]) : Int[]
    P = [nf > 0 ? Float64.(d["f_p$i"]) : Float64[] for i in 1:4]
    fields = Vector{FieldDef}(undef, nf)
    for i in 1:nf
        kind = f_kind[i]      # schema 约定：0=gauss 1=square 2=sampled
        st = Float64[]
        sv = Float64[]
        if kind == 2
            st = Float64.(d["f_samp_t_$(i-1)"])
            sv = Float64.(d["f_samp_v_$(i-1)"])
        end
        env = make_env(kind, P[1][i], P[2][i], P[3][i], P[4][i], st, sv)
        fields[i] = FieldDef(f_group[i] + 1, env)   # 转 1-based 组
    end

    # 初态：优先 psi0_A/psi0_B 任意矢量直积；否则 psi0 索引对（0-based）
    if haskey(d, "psi0_A") && length(d["psi0_A"]) > 0
        psiA = ComplexF64.(d["psi0_A"])
        psiB = ComplexF64.(d["psi0_B"])
        length(psiA) == N && length(psiB) == N ||
            throw(ArgumentError("psi0_A/psi0_B 长度 ≠ N"))
    else
        iA = Int(d["psi0"][1]) + 1
        iB = Int(d["psi0"][2]) + 1
        psiA = zeros(ComplexF64, N); psiA[iA] = 1.0
        psiB = zeros(ComplexF64, N); psiB[iB] = 1.0
    end
    t_grid = Float64.(d["t_grid"])
    length(t_grid) >= 2 || throw(ArgumentError("t_grid 至少 2 点"))
    tol = _as_float(d["tol"])
    tol > 0 || throw(ArgumentError("tol > 0"))
    gamA = if haskey(d, "gamma_A") && length(d["gamma_A"]) > 0
        Float64.(d["gamma_A"])
    else
        nothing
    end
    gamB = if haskey(d, "gamma_B") && length(d["gamma_B"]) > 0
        Float64.(d["gamma_B"])
    else
        nothing
    end
    (gamA === nothing && gamB === nothing) ||
        (gamA !== nothing && gamB !== nothing) ||
        throw(ArgumentError("gamma_A/gamma_B 必须同时提供或同时缺省"))
    if gamA !== nothing
        length(gamA) == N2 && length(gamB) == N2 ||
            throw(ArgumentError("gamma_A/gamma_B 长度 ≠ N²"))
    end

    m_max = haskey(d, "m_max") ? _as_int(d["m_max"]) : 8
    dense_max = haskey(d, "dense_max") ? _as_int(d["dense_max"]) : 700
    ladder = haskey(d, "node_ladder") ? Int.(d["node_ladder"]) :
             [16, 32, 64, 128]
    frac = haskey(d, "dt_min_frac") ? _as_float(d["dt_min_frac"]) : 2e5

    # 组归属校验
    all(1 .<= f_group .+ 1 .<= K) || throw(ArgumentError("f_group 越界"))
    return Request(N=N, E_pair=E_pair, vdd=vdd, groups=groups,
                   fields=fields, psi0_A=psiA, psi0_B=psiB,
                   t_grid=t_grid, tol=tol,
                   gamma_A=gamA, gamma_B=gamB, m_max=m_max,
                   dense_max=dense_max, node_ladder=ladder,
                   dt_min_frac=frac,
                   record_amp=haskey(d, "record_amp") &&
                              _as_int(d["record_amp"]) != 0,
                   sparse_backend=haskey(d, "sparse_backend") ?
                                  _as_int(d["sparse_backend"]) : 0,
                   mF_states=haskey(d, "mF_states") &&
                             length(d["mF_states"]) == N ?
                             Float64.(d["mF_states"]) : nothing,
                   charges=haskey(d, "charges") &&
                           length(d["charges"]) == K ?
                           Float64.(d["charges"]) : nothing,
                   block_mF=haskey(d, "block_mF") &&
                             _as_int(d["block_mF"]) != 0 &&
                             haskey(d, "mF_states") &&
                             length(d["mF_states"]) == N &&
                             haskey(d, "charges") &&
                             length(d["charges"]) == K,
                   skip_probe=haskey(d, "skip_probe") &&
                              _as_int(d["skip_probe"]) != 0)
end

"组 c_g(t)：组内各场包络求和（镜像 setup.c_mids/c_at_nodes）"
function group_c(req::Request, ts::AbstractVector)
    G = length(req.groups)
    out = zeros(Float64, G, length(ts))
    for f in req.fields
        out[f.group, :] .+= f.env.(ts)
    end
    return out
end
group_c(req::Request, t::Real) =
    [sum(f.env(t) for f in req.fields if f.group == g;
         init = 0.0) for g in 1:length(req.groups)]

# ---------- 响应写出 ----------

function _write_err(path, ok::Int, msg::AbstractString)
    npzwrite(path, Dict("ok" => [ok],
                        "err_msg" => collect(codeunits(msg))))
end

"""写出结果响应（由 evolve_pair 的 PairResult 组装）"""
function write_result(path, req::Request, res; msg::AbstractString = "")
    d = Dict{String,Any}(
        "ok" => [1],
        "err_msg" => collect(codeunits(msg)),
        "t" => res.t,
        "pops_2d" => res.pops_2d,
        "pops_A" => res.pops_A,
        "pops_B" => res.pops_B,
        "loss_A" => res.loss_A,
        "loss_B" => res.loss_B,
        "n_segments" => [res.n_segments],
        "max_diff" => [res.max_diff],
        "dt_min_hit" => [res.dt_min_hit ? 1 : 0],
        "max_norm_dev" => [res.max_norm_dev],
        "photon_blocks" => res.photon_blocks,
        "timing" => res.timing,
        "threads" => [Threads.nthreads()],
        "M" => [res.M],
    )
    if res.psi_amp !== nothing
        d["psi_amp"] = res.psi_amp
    end
    npzwrite(path, d)
end

"""--echo 模式：回显解析后的请求（G0 往返校验）"""
function write_echo(path::AbstractString, req::Request)
    d = Dict{String,Any}(
        "ok" => [1],
        "e_N" => [req.N],
        "e_E_pair" => req.E_pair,
        "e_psi0_A" => req.psi0_A,
        "e_psi0_B" => req.psi0_B,
        "e_t_grid" => req.t_grid,
        "e_tol" => [req.tol],
        "e_group_omega" => [g.omega for g in req.groups],
        "e_n_groups" => [length(req.groups)],
        "e_n_fields" => [length(req.fields)],
        "e_f_group" => [f.group - 1 for f in req.fields],
        "e_m_max" => [req.m_max],
        "e_dense_max" => [req.dense_max],
        "e_node_ladder" => req.node_ladder,
        "e_dt_min_frac" => [req.dt_min_frac],
    )
    if req.vdd === nothing
        d["e_vdd_row"] = Int[]
        d["e_vdd_col"] = Int[]
        d["e_vdd_data"] = ComplexF64[]
    else
        coo = findnz(req.vdd)
        d["e_vdd_row"] = coo[1] .- 1
        d["e_vdd_col"] = coo[2] .- 1
        d["e_vdd_data"] = coo[3]
    end
    if req.gamma_A === nothing
        d["e_gamma_A"] = Float64[]
        d["e_gamma_B"] = Float64[]
    else
        d["e_gamma_A"] = req.gamma_A
        d["e_gamma_B"] = req.gamma_B
    end
    for (g, gr) in enumerate(req.groups)
        coo = nnz(gr.K2) == 0 ? (Int[], Int[], Float64[]) : findnz(gr.K2)
        d["e_k2r_$(g-1)"] = coo[1] .- 1
        d["e_k2c_$(g-1)"] = coo[2] .- 1
        d["e_k2v_$(g-1)"] = coo[3]
    end
    for (i, f) in enumerate(req.fields)
        d["e_f_kind_$(i-1)"] = [f.env isa GaussEnv ? 0 :
                                 f.env isa SquareEnv ? 1 : 2]
        if f.env isa GaussEnv
            d["e_f_p_$(i-1)"] = [f.env.center, f.env.sigma, f.env.peak, 0.0]
        elseif f.env isa SquareEnv
            d["e_f_p_$(i-1)"] = [f.env.start, f.env.edge, f.env.height,
                                 f.env.fallstart]
        else
            d["e_f_p_$(i-1)"] = zeros(4)
            d["e_f_samp_t_$(i-1)"] = f.env.t
            d["e_f_samp_v_$(i-1)"] = f.env.v
        end
    end
    npzwrite(path, d)
end
