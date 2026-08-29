#!/usr/bin/env julia
# evolve_pair_cli.jl — Python↔Julia 子进程 CLI（S7 集成入口，S0 起可用）。
#
# 用法：
#   julia --project=<julia目录> bin/evolve_pair_cli.jl req.npz resp.npz [--echo] [--cancel-file 路径]
#
# stdout 进度协议：按行输出 "PROG <fraction>"（0..1）；
# 取消：cancel 文件存在 → 写 ok=2 响应并以退出码 2 结束。
# 退出码：0 成功 / 1 引擎错误 / 2 用户取消 / 3 用法错误。

import Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using EnergyLevelPair
const ELP = EnergyLevelPair

function usage()
    println(stderr, "usage: evolve_pair_cli.jl req.npz resp.npz [--echo] [--cancel-file path]")
    exit(3)
end

function main()
    args = parse_args()
    req_path, resp_path = args[:req], args[:resp]
    cancel = args[:cancel]
    try
        req = ELP.read_request(req_path)
        if args[:echo]
            ELP.write_echo(resp_path, req)
            println("ECHO_OK")
            flush(stdout)
            return
        end
        # ---- 引擎主入口 ----
        res = ELP.evolve_pair(req;
                              record_amp = req.record_amp,
                              progress = frac -> begin
                                  println("PROG ", frac)
                                  flush(stdout)
                              end,
                              cancelled = () -> ELP.is_cancelled(cancel),
                              log = msg -> println("LOG ", msg))
        ELP.write_result(resp_path, req, res)
        println("DONE")
        flush(stdout)
    catch err
        err isa ELP.Cancelled && return finish_cancel(resp_path)
        msg = sprint(showerror, err)
        @info "engine error" msg
        ELP._write_err(resp_path, 0, msg)
        exit(1)
    end
end

finish_cancel(resp_path) = begin
    ELP._write_err(resp_path, 2, "用户取消")
    exit(2)
end

function parse_args()
    a = ARGS
    length(a) >= 2 || usage()
    out = Dict{Symbol,Any}(:req => a[1], :resp => a[2], :echo => false,
                           :cancel => nothing)
    i = 3
    while i <= length(a)
        if a[i] == "--echo"
            out[:echo] = true
        elseif a[i] == "--cancel-file" && i < length(a)
            out[:cancel] = a[i+1]
            i += 1
        else
            usage()
        end
        i += 1
    end
    return out
end

main()
