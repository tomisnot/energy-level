# EnergyLevelPair.jl — 两体第一性原理含时引擎（Julia 内核）
#
# 与 src/core/evolve_exact_pair.py 逐条镜像：
#   物理模型（无 secular/RWA/旋转系/近远分流，完整 V_dd）：
#     H_2d(t) = [H_A0 + Σ_f Ω_f·env_f(t)·cos(ω_f t)·d̂_A]⊗I
#             + I⊗[H_B0 + Σ_f Ω_f·env_f(t)·cos(ω_f t)·d̂_B] + V_dd
#   两体 Sambe (t,t') 扩展空间：idx = k + N²·p（k=i·N+j 最快）
#   稠密路径（dim ≤ dense_max）：冻结中点本征分解 + 本征基 Duhamel 缺陷
#   稀疏路径：对角相互作用绘景 + Volterra–GL 配置 + GMRES（gen-3 移植）
#   新增 Wosc：振荡解析求积（(e^{iΔD·h}−1)/(iΔD) 逐分量权重）
#
# 单位约定（全镜像）：能量/频率 MHz，时间 ns，_CONV = 2π×1e-3（MHz·ns→rad）；
# Rabi 周频约定（A 存 Ω/2π），厄米判定绝对容差 1e-10。
module EnergyLevelPair

const _CONV = 2π * 1e-3
const VERSION = v"0.1.0"

include("envelopes.jl")   # 包络解析式（镜像 waveforms.py，含采样包络 PCHIP）
include("gl.jl")          # Gauss–Legendre 节点/权重
include("io.jl")          # NPZ 请求/响应 schema（镜像 evolve_exact_pair 输入输出）
include("sambe.jl")       # SambeSetup（镜像 _PairSambeSetup）
include("densepath.jl")   # 稠密路径（镜像 _dense_decomp/_prop_step/_defect_pass）
include("sparsepath.jl")  # gen-3 稀疏路径（S3 接入；当前占位）
include("mprobe.jl")      # 光子块探针（镜像 _pair_m_probe）
include("block.jl")       # mF-光子荷不变量分块（S6）
include("evolve.jl")      # 主控（镜像 evolve_exact_pair）

include("wosc.jl")        # Wosc 振荡解析求积权重（S4/S5）

end # module
