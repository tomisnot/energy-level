# gl.jl — Gauss–Legendre 节点/权重（镜像 evolve_exact._gl_nodes = numpy.leggauss）。
# Newton 迭代 + Legendre 递推，对称初值，机器精度（~1e-16）。
# GL 节点是与 numpy 各自独立算到机器精度的标准数学量，无须逐位对齐。

"n 点 Gauss–Legendre 节点与权重（[−1,1]）"
function gl_nodes(n::Integer)
    n >= 1 || throw(ArgumentError("n ≥ 1"))
    x = zeros(Float64, n)
    w = zeros(Float64, n)
    m = (n + 1) ÷ 2
    for i in 1:m
        # 经典初值（i−0.25)/(n+0.5)π 的余弦
        z = cos(pi * (i - 0.25) / (n + 0.5))
        dp = 0.0
        for _ in 1:100
            p0 = 1.0
            p1 = 0.0
            for j in 1:n
                p2 = p1
                p1 = p0
                p0 = ((2 * j - 1) * z * p1 - (j - 1) * p2) / j
            end
            # z*P_n − P_{n−1} 的导数形式
            dp = n * (z * p0 - p1) / (z * z - 1.0)
            dz = -p0 / dp
            z += dz
            abs(dz) < 1e-15 && break
        end
        xi = -z
        x[i] = xi
        x[n+1-i] = z
        p0 = 1.0
        p1 = 0.0
        for j in 1:n
            p2 = p1
            p1 = p0
            p0 = ((2 * j - 1) * z * p1 - (j - 1) * p2) / j
        end
        dp = n * (z * p0 - p1) / (z * z - 1.0)
        wi = 2.0 / ((1.0 - z * z) * dp * dp)
        w[i] = wi
        w[n+1-i] = wi
    end
    return x, w
end
