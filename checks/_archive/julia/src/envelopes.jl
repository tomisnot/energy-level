# envelopes.jl — 包络解析式，逐行镜像 src/core/waveforms.py。
# 约定：包络参数已由 Python 导出方按 offset_ns 做绝对时间平移
# （Wave.waveform(t) = envelope(t − offset)），Julia 侧直接用绝对 ns。

"""包络基类型。全部在标量上定义，数组求值用广播。"""
abstract type Envelope end

# 高斯振幅包络：peak·exp(−(τ−center)²/2σ²)（全域定义，无遮罩）
struct GaussEnv <: Envelope
    center::Float64   # ns（绝对时间）
    sigma::Float64    # ns，> 0
    peak::Float64     # (0,1]
end
(e::GaussEnv)(t::Real) = e.peak * exp(-(t - e.center)^2 / (2 * e.sigma^2))

# 余弦缓变（raised cosine），x 截断到 [0,1]（镜像 _smooth）
_smooth(x::Real) = 0.5 * (1.0 - cos(pi * clamp(x, 0.0, 1.0)))

# 缓变方波：start→start+edge 余弦上升、平台至 fall_start、
# fall_start→fall_start+edge 余弦下降；区间外归零；edge≤0 退化理想方波。
struct SquareEnv <: Envelope
    start::Float64      # ns
    edge::Float64       # ns，≥0
    height::Float64     # (0,1]
    fallstart::Float64  # ns，≥ start
end
function (e::SquareEnv)(t::Real)
    if e.edge <= 0
        return (e.start <= t <= e.fallstart) ? e.height : 0.0
    end
    rise = _smooth((t - e.start) / e.edge)
    fall = 1.0 - _smooth((t - e.fallstart) / e.edge)
    env = e.height * rise * fall
    return (t < e.start || t > e.fallstart + e.edge) ? 0.0 : env
end

# 采样包络（schema kind=2）：单调三次 Hermite（PCHIP/Fritsch–Carlson）。
# 包络慢变 → 插值误差远小于 tol；段内细分兜底。端点常数外推。
struct SampledEnv <: Envelope
    t::Vector{Float64}   # 升序 ns
    v::Vector{Float64}   # [0,1]
    m::Vector{Float64}   # PCHIP 切线
    lo::Float64
    hi::Float64
end
function SampledEnv(ts::AbstractVector, vs::AbstractVector)
    n = length(ts)
    n >= 2 || throw(ArgumentError("采样包络至少 2 点"))
    issorted(ts) || throw(ArgumentError("采样包络时间须升序"))
    length(vs) == n || throw(ArgumentError("t/v 长度不一致"))
    h = [Float64(ts[i+1]) - Float64(ts[i]) for i in 1:n-1]
    δ = [(Float64(vs[i+1]) - Float64(vs[i])) / h[i] for i in 1:n-1]
    m = zeros(n)
    m[1] = δ[1]
    m[n] = δ[n-1]
    for i in 2:n-1
        if δ[i-1] * δ[i] <= 0
            m[i] = 0.0
        else
            w1 = 2.0 * h[i] + h[i-1]
            w2 = h[i] + 2.0 * h[i-1]
            m[i] = (w1 + w2) / (w1 / δ[i-1] + w2 / δ[i])
        end
    end
    SampledEnv(collect(Float64, ts), collect(Float64, vs), m,
               Float64(vs[1]), Float64(vs[n]))
end
function (e::SampledEnv)(t::Real)
    t < e.t[1] && return e.lo
    t > e.t[end] && return e.hi
    i = clamp(searchsortedfirst(e.t, t) - 1, 1, length(e.t) - 1)
    h = e.t[i+1] - e.t[i]
    s = (t - e.t[i]) / h
    h00 = (1 + 2s) * (1 - s)^2
    h10 = s * (1 - s)^2
    h01 = s^2 * (3 - 2s)
    h11 = s^2 * (s - 1)
    return h00 * e.v[i] + h10 * h * e.m[i] + h01 * e.v[i+1] + h11 * h * e.m[i+1]
end

# 由 schema 参数构造包络（kind: 0=gauss 1=square 2=sampled）
function make_env(kind::Integer, p1::Real, p2::Real, p3::Real, p4::Real,
                  st::AbstractVector, sv::AbstractVector)
    kind == 0 && return GaussEnv(p1, p2, p3)
    kind == 1 && return SquareEnv(p1, p2, p3, p4)
    kind == 2 && return SampledEnv(st, sv)
    throw(ArgumentError("未知包络类型 kind=$kind"))
end

"组内各场包络求和（镜像 setup.c_mids / c_at_nodes 的 envs 求和语义）"
envsum(envs::Vector{<:Envelope}, t::Real) = sum(e(t) for e in envs)
function envsum(envs::Vector{<:Envelope}, ts::AbstractVector)
    out = zeros(Float64, length(ts))
    for e in envs, (i, t) in enumerate(ts)
        out[i] += e(t)
    end
    out
end
