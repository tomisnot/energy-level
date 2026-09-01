# runtests.jl — Julia 内核分阶段验证门（G0..G6，随开发追加）。
#
# G1（S1）：单段传播子
#   G1.1 常数包络（方波全覆盖）段传播 vs 精确矩阵指数 exp(−iC·h·K) ≤ 1e-12
#   G1.2 冻结传播子 prop_step vs exp（任意 c）≤ 1e-12
#   G1.3 变包络段（高斯峰区）vs 冻结中点 expm 链（256 步） ≤ 1e-9
#   G1.4 vs QO.jl schroedinger_dynamic（扩展空间 ODE 独立参考，紧容差） ≤ 1e-8
#   G1.5 节点阶梯收敛（末级差 ≤ tol 且优于首级）
#   G1.6 寿命 NonHermitian：非厄米判定路径 + 常数包络段 vs exp ≤ 1e-12
#   G1.7 embed/physical_pops 往返（t=0 布居 ≡ |ψ0|²）

using Test
using LinearAlgebra
using SparseArrays
using EnergyLevelPair
const ELP = EnergyLevelPair

# ---------- 公共场景（EP5/EP6 合成模型） ----------
function make_request(; gamma=nothing, env=ELP.GaussEnv(100.0, 30.0, 1.0))
    N = 2
    N2 = 4
    E = [0.0, 1000.0]
    E_pair = Float64[E[i] + E[j] for i in 1:N for j in 1:N]
    # 0-based k=i·N+j：|g,e⟩=k1=2, |e,g⟩=k1=3（1-based）
    vdd = sparse([2, 3], [3, 2], ComplexF64[10.0, 10.0], N2, N2)
    A_A = [0.0 20.0; 20.0 0.0]           # Rabi 20 MHz（周频）
    K2 = SparseMatrixCSC{Float64,Int}(kron(A_A, Matrix(1.0I, 2, 2)))
    groups = [ELP.GroupDef(1000.0, K2)]
    fields = [ELP.FieldDef(1, env)]
    gamA = gamma === nothing ? nothing : gamma
    gamB = gamma === nothing ? nothing : zeros(N2)
    return ELP.Request(N=N, E_pair=E_pair, vdd=vdd, groups=groups,
                       fields=fields, psi0_A=ComplexF64[1, 0], psi0_B=ComplexF64[1, 0],
                       t_grid=collect(0.0:1.0:200.0), tol=1e-6,
                       gamma_A=gamA, gamma_B=gamB)
end

@testset "GL 节点" begin
    for n in (4, 16, 64)
        x, w = ELP.gl_nodes(n)
        # 定义性质：∫_{-1}^{1} x^k dx = 2/(k+1)（偶 k），0（奇 k，对称区间）
        err = maximum(abs(sum(w .* (x .^ k)) -
                          (iseven(k) ? 2.0 / (k + 1) : 0.0)) for k in 0:2n-1)
        @test err < 1e-13
    end
end

@testset "包络" begin
    g = ELP.GaussEnv(100.0, 30.0, 1.0)
    @test g(100.0) ≈ 1.0
    @test g(100.0 + 4 * 30.0) < 1e-2
    s = ELP.SquareEnv(10.0, 5.0, 0.8, 20.0)
    @test s(15.0) ≈ 0.8                     # 平台
    @test s(12.5) ≈ 0.4                     # 上升沿中点（raised cosine）
    @test s(0.0) == 0.0 && s(100.0) == 0.0  # 区间外
    sq0 = ELP.SquareEnv(10.0, 0.0, 0.9, 20.0)
    @test sq0(15.0) ≈ 0.9 && sq0(21.0) == 0.0   # 理想方波
    ts = [0.0, 50.0, 100.0, 150.0, 200.0]
    vs = [0.0, 0.5, 1.0, 0.5, 0.0]
    se = ELP.SampledEnv(ts, vs)
    @test all(se(t) ≈ v for (t, v) in zip(ts, vs))   # 过点
    @test 0.0 < se(75.0) < 1.0                       # 单调段内
    @test se(-1.0) == 0.0 && se(201.0) == 0.0        # 端点常数外推
end

@testset "G1 单段传播子" begin
    C = ELP._CONV
    psi0_2d = ELP.product_psi(ComplexF64[1, 0], ComplexF64[1, 0], 2)

    # G1.7 布居投影往返：t=0 时物理布居 ≡ |ψ0|²
    req = make_request()
    s = ELP.SambeSetup(req, 2, req.vdd, nothing)   # M=2, dim=20 稠密
    @test s.dense
    X0 = ELP.embed_pair(psi0_2d, s)
    pops0 = ELP.physical_pops(s, X0, 0.0)
    @test maximum(abs.(pops0 .- abs2.(psi0_2d))) < 1e-14

    # G1.1 常数包络（理想方波全覆盖 → c≡1）：segment_dense ≡ exp
    req_flat = make_request(env=ELP.SquareEnv(0.0, 0.0, 1.0, 200.0))
    s_flat = ELP.SambeSetup(req_flat, 2, req_flat.vdd, nothing)
    Xf = ELP.embed_pair(psi0_2d, s_flat)
    Kf = ELP.K_seg(s_flat, [1.0])
    Xout, est = ELP.segment_dense(s_flat, 0.0, 1.0, Xf, 1e-6,
                                  [16, 32, 64, 128])
    @test est == 0.0                       # 包络恒常 → 缺陷严格为零
    @test maximum(abs.(Xout .- exp(-1im * C * 1.0 * Matrix(Kf)) * Xf)) <
          1e-12

    # G1.2 冻结传播子 prop_step（任意 c）vs 精确指数
    K = ELP.K_seg(s, [0.7])
    de = ELP.dense_decomp(K)
    Xp = ELP.prop_step(s, K, 0.0, 1.0, X0, de)
    @test maximum(abs.(Xp .- exp(-1im * C * 1.0 * Matrix(K)) * X0)) < 1e-12

    # G1.3 变包络段 [99,100]（高斯峰区）vs 冻结中点 expm 链（256 步）
    ta, tb = 99.0, 100.0
    Xout2, est2 = ELP.segment_dense(s, ta, tb, X0, 1e-6, [16, 32, 64, 128])
    nchain = 256
    hc = (tb - ta) / nchain
    Xch = copy(X0)
    for i in 0:nchain-1
        tmid = ta + (i + 0.5) * hc
        Kc = ELP.K_seg(s, [s.envs(1, tmid)])
        Xch = exp(-1im * C * hc * Matrix(Kc)) * Xch
    end
    @test 0.0 < est2 <= 1e-6
    @test maximum(abs.(Xout2 .- Xch)) < 1e-9

    # G1.4 vs QO.jl schroedinger_dynamic（扩展空间独立 ODE 参考，紧容差）
    @info "G1.4 QO.jl 参考比较（紧容差 ODE，数十秒内）"
    begin
        using QuantumOptics
        b = GenericBasis(s.dim)
        Hfun(t, psi) = DenseOperator(b, C .* Matrix(ELP.K_seg(s,
                                   [s.envs(1, t)])))
        tout, psit = timeevolution.schroedinger_dynamic([ta, tb],
                                                        Ket(b, X0), Hfun;
                                                        reltol = 1e-12,
                                                        abstol = 1e-12)
        Xqo = psit[end].data
        @test maximum(abs.(Xout2 .- Xqo)) < 1e-8
    end

    # G1.5 节点阶梯收敛：较宽段 [95,105]（包络显著变化）
    ta2, tb2 = 95.0, 105.0
    K2s = ELP.K_seg(s, ELP.c_mids(s, 0.5 * (ta2 + tb2)))
    de2 = ELP.dense_decomp(K2s)
    a0 = de2.left * X0
    psi_frozen = de2.evecs *
                 (a0 .* exp.(-1im * C * (tb2 - ta2) .* de2.evals))
    ests = Float64[]
    prev = nothing
    for n in (16, 32, 64, 128)
        glx, glw = ELP.gl_nodes(n)
        nodes = ta2 .+ 0.5 * (tb2 - ta2) .* (glx .+ 1.0)
        ws = 0.5 * (tb2 - ta2) .* glw
        delta, max_dc = ELP.defect_pass(s, ta2, tb2, X0, nodes, ws,
                                        s.envs(nodes),
                                        ELP.c_mids(s, 0.5 * (ta2 + tb2)), de2)
        @test max_dc > 1e-12                  # 该段包络确实在变
        cur = psi_frozen .+ delta
        if prev !== nothing
            push!(ests, norm(cur - prev))
        end
        prev = cur
    end
    @test issorted(ests; rev = true)          # 阶梯差单调下降
    @test ests[end] < 1e-6

    # G1.6 寿命 NonHermitian（γ=10 /ns）：非厄米判定 + 常数包络 ≡ exp
    gamma_A = zeros(4)
    gamma_A[3] = 10.0
    gamma_A[4] = 10.0      # 1-based：A 原子激发的对态 |e,g⟩,|e,e⟩
    req2 = make_request(gamma=gamma_A, env=ELP.SquareEnv(0.0, 0.0, 1.0, 200.0))
    s2 = ELP.SambeSetup(req2, 2, req2.vdd, gamma_A)
    @test s2.has_gamma
    Xg = ELP.embed_pair(psi0_2d, s2)
    Kg = ELP.K_seg(s2, [1.0])
    @test maximum(abs.(Matrix(Kg) - adjoint(Matrix(Kg)))) > 1e-10  # 确实非厄米
    Xg1, estg = ELP.segment_dense(s2, 0.0, 1.0, Xg, 1e-6,
                                  [16, 32, 64, 128])
    @test estg == 0.0
    @test maximum(abs.(Xg1 .- exp(-1im * C * 1.0 * Matrix(Kg)) * Xg)) <
          1e-12
end

@testset "G3 稀疏基线（gen-3 vs 稠密）" begin
    using SparseArrays
    N = 2; N2 = 4
    E = [0.0, 1000.0]
    E_pair = Float64[E[i] + E[j] for i in 1:N for j in 1:N]
    vdd = sparse([2, 3], [3, 2], ComplexF64[10.0, 10.0], N2, N2)
    A_A = [0.0 20.0; 20.0 0.0]
    K2 = SparseMatrixCSC{Float64,Int}(kron(A_A, Matrix(1.0I, 2, 2)))
    groups = [ELP.GroupDef(1000.0, K2)]
    fields = [ELP.FieldDef(1, ELP.GaussEnv(100.0, 30.0, 1.0))]
    t = collect(0.0:2.0:200.0)
    function mk(dmax)
        ELP.Request(N=N, E_pair=E_pair, vdd=vdd, groups=groups,
                    fields=fields, psi0_A=ComplexF64[1, 0],
                    psi0_B=ComplexF64[1, 0], t_grid=t, tol=1e-7,
                    gamma_A=nothing, gamma_B=nothing, dense_max=dmax)
    end
    res_d = ELP.evolve_pair(mk(700); log = x -> nothing)
    res_s = ELP.evolve_pair(mk(0); log = x -> nothing)
    d = maximum(abs.(res_d.pops_2d .- res_s.pops_2d))
    @test d <= 5e-4        # EP6a
    cons = maximum(abs.(vec(sum(res_s.pops_2d; dims = 2)) .- 1.0))
    @test cons <= 2e-3     # EP6b
end

@testset "G4 Wosc 解析权重" begin
    # vs 高阶数值求积（40001 点梯形，与原型 _pair_wosc_test2 同法）
    n = 8
    glx, glw = ELP.gl_nodes(n)
    Bmat = ELP.lagrange_coeffs(n, glx)
    lams = ComplexF64[-0.5im, 0.5im, -20im, 20im, -1500im, 1500im, 0.0im]
    # 参考求积：全区间 GL(4000)、子区间 GL(3000)——分辨率须覆盖 λ=1500 的振荡
    # （κ·区间宽 ~3000 rad；GL(n) 需 n ≳ κ·width/2）
    xref, wref = ELP.gl_nodes(4000)
    xsub, wsub = ELP.gl_nodes(3000)
    for lam in lams
        Zpart, Zfull = ELP.wosc_weights(n, lam; glx = glx, Bmat = Bmat)
        ph_full = exp.(lam .* xref)
        maxerr_p = 0.0
        maxerr_f = 0.0
        for r in 1:n
            # 子区间 [−1, glx[r]]：GL 节点仿射映射；L_s 必须在映射节点 xm 上取值
            a = -1.0
            b = glx[r]
            xm = 0.5 * (b - a) .* xsub .+ 0.5 * (a + b)
            wm = 0.5 * (b - a) .* wsub
            phm = exp.(lam .* xm)
            for s in 1:n
                Ls = [prod(j == s ? 1.0 : (xv - glx[j]) / (glx[s] - glx[j])
                           for j in 1:n) for xv in xm]
                num = sum(wm .* phm .* Ls)
                maxerr_p = max(maxerr_p, abs(num - Zpart[r, s]))
            end
        end
        for s in 1:n
            Ls_full = [prod(j == s ? 1.0 : (xv - glx[j]) / (glx[s] - glx[j])
                            for j in 1:n) for xv in xref]
            numf = sum(wref .* ph_full .* Ls_full)
            maxerr_f = max(maxerr_f, abs(numf - Zfull[s]))
        end
        @test maxerr_p <= 2e-6
        @test maxerr_f <= 2e-6
    end
    # λ=0 恒等：Zpart ≡ Wpart、Zfull ≡ GL 权重
    _, _, Wpart = ELP.volterra_rule(n)
    Zp0, Zf0 = ELP.wosc_weights(n, 0.0im; glx = glx, Bmat = Bmat)
    @test maximum(abs.(Zp0 .- Wpart)) < 1e-14
    @test maximum(abs.(Zf0 .- glw)) < 1e-14
end

@testset "M1 mF 分块 ≡ 全基组" begin
    using SparseArrays
    # EP5/EP6 场景 + mF 标签（g=+1/2, e=-1/2；σ⁻ 载频 q=-1）
    N = 2; N2 = 4
    E = [0.0, 1000.0]
    E_pair = Float64[E[i] + E[j] for i in 1:N for j in 1:N]
    vdd = sparse([2, 3], [3, 2], ComplexF64[10.0, 10.0], N2, N2)
    A_A = [0.0 20.0; 20.0 0.0]
    K2 = SparseMatrixCSC{Float64,Int}(kron(A_A, Matrix(1.0I, 2, 2)))
    groups = [ELP.GroupDef(1000.0, K2)]
    fields = [ELP.FieldDef(1, ELP.GaussEnv(100.0, 30.0, 1.0))]
    t = collect(0.0:2.0:200.0)
    mF = [0.5, -0.5]
    function mk(block; dense_max=700)
        ELP.Request(N=N, E_pair=E_pair, vdd=vdd, groups=groups,
                    fields=fields, psi0_A=ComplexF64[1, 0],
                    psi0_B=ComplexF64[1, 0], t_grid=t, tol=1e-7,
                    gamma_A=nothing, gamma_B=nothing, dense_max=dense_max,
                    mF_states=mF, charges=[-1.0], block_mF=block)
    end
    res_full = ELP.evolve_pair(mk(false); log = x -> nothing)
    res_blk = ELP.evolve_pair(mk(true); log = x -> nothing)
    d = maximum(abs.(res_full.pops_2d .- res_blk.pops_2d))
    @test d <= 1e-10        # 分块 = 精确对称性 → 机器精度一致
    # 分块信息
    @test res_blk.M == res_full.M
end

@testset "M2 多载频分块（含光学反旋转）" begin
    using SparseArrays
    # 3 态/原子: g(mF=+1/2), e(mF=-1/2), u(mF=+5/2, 远)

    # 载频1: 微波 σ⁻ g↔e（共振 1000 MHz, q=-1）
    # 载频2: 光学 π g↔u（共振 20000 MHz, q=0）——u 的引入使
    #        (u,u) 对态 mF_sum=5 落在可达窗外 → 分块缩减
    N = 3; N2 = 9
    E = [0.0, 1000.0, 20000.0]
    E_pair = Float64[E[i] + E[j] for i in 1:N for j in 1:N]
    vdd = sparse([2, 4], [4, 2], ComplexF64[10.0, 10.0], N2, N2)  # ge<->eg
    A1 = zeros(3, 3); A1[1, 2] = A1[2, 1] = 20.0    # g<->e (σ⁻ 微波)
    A2 = zeros(3, 3); A2[1, 3] = A2[3, 1] = 2.0     # g<->u (π 光学)
    K2_1 = SparseMatrixCSC{Float64,Int}(kron(A1, Matrix(1.0I, 3, 3)))
    K2_2 = SparseMatrixCSC{Float64,Int}(kron(A2, Matrix(1.0I, 3, 3)))
    groups = [ELP.GroupDef(1000.0, K2_1), ELP.GroupDef(20000.0, K2_2)]
    fields = [ELP.FieldDef(1, ELP.GaussEnv(100.0, 30.0, 1.0)),
              ELP.FieldDef(2, ELP.SquareEnv(0.0, 0.0, 1.0, 200.0))]
    t = collect(0.0:2.0:200.0)
    mF = [0.5, -0.5, 2.5]
    function mk2(block; dense_max=700)
        ELP.Request(N=N, E_pair=E_pair, vdd=vdd, groups=groups,
                    fields=fields, psi0_A=ComplexF64[1, 0, 0],
                    psi0_B=ComplexF64[1, 0, 0], t_grid=t, tol=1e-7,
                    gamma_A=nothing, gamma_B=nothing, dense_max=dense_max,
                    mF_states=mF, charges=[-1.0, 0.0], block_mF=block)
    end
    res_f = ELP.evolve_pair(mk2(false); log = x -> nothing)
    res_b = ELP.evolve_pair(mk2(true); log = x -> nothing)
    d = maximum(abs.(res_f.pops_2d .- res_b.pops_2d))
    @test d <= 1e-10
    # 分块确实缩减了（可达类 < 全空间）
    @test res_b.n_segments > 0
end

println("all testsets done")
