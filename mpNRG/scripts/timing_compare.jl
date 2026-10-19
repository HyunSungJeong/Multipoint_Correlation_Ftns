# Timing comparison of the TNOps refactor (docs/TNOPS_REFACTOR.md, Sec. 9.4):
# the P10 performance table (forward sweep and one 2p, 3p, 4p PSF versus
# N_keep = 25, 50, 100; SIAM U = 0.2, Δ = 0.04, Λ = 4, T = 1e-4, grid
# n_dec = 8) measured on this machine, with the same number of threads, for
# the code at the tag `pre-tensor-refactor` (a git worktree of the tag) and
# for the current code. Writes notebooks/figures/mpnrg/timing_compare.jld2,
# which the notebook section "Refactor regression (TNOps)" reads.
#
#   julia -t 4 --project=mpNRG mpNRG/scripts/timing_compare.jl [worktree dir]
#
# Each code version is measured in a fresh process (this script with
# `--measure out.jld2`, run in the project of that version), after a small
# warm-up run that compiles every code path, so that compilation is not timed.
# The worktree is created if it does not exist (default: a directory next to
# the repository) and left in place.

using JLD2, Printf

const NKEEP = (25, 50, 100)
const MEASURE = length(ARGS) ≥ 2 && ARGS[1] == "--measure"
MEASURE && @eval using MultipointNRG          # the version of the project this process runs in

function measure(out::AbstractString)
    d_(σ) = MultipointNRG.cop(MultipointNRG.Mode(-1, σ)); dd_(σ) = MultipointNRG.cdag(MultipointNRG.Mode(-1, σ))
    n_(σ) = MultipointNRG.num(MultipointNRG.Mode(-1, σ))
    M = MultipointNRG
    m = M.siam(U = 0.2, Δ = 0.04); T = 1e-4; N = M.chain_length_for(T; Λ = 4.0, margin = 4)
    g = M.LogGrid(emin = 1e-6, emax = 10.0, ndec = 8)
    opsets = ((d_(:up), dd_(:up)), (d_(:up), dd_(:up), n_(:dn)), (d_(:up), dd_(:up), d_(:dn), dd_(:dn)))
    # warm-up: every code path once, on a short chain
    let run = M.run_nrg(m, M.NRGParams(N = 4, Nkeep = 16)), f = M.fdm(run; T = 0.01)
        foreach(ops -> M.psf(run, f, ops; grid = g), opsets)
    end
    sweep = Float64[]; tp = zeros(3, length(NKEEP)); costs = Any[]
    for (j, Nk) in enumerate(NKEEP)
        t0 = time(); run = M.run_nrg(m, M.NRGParams(N = N, Nkeep = Nk)); push!(sweep, time() - t0)
        f = M.fdm(run; T = T)
        for (i, ops) in enumerate(opsets)
            r = M.psf(run, f, ops; grid = g)
            tp[i, j] = r.meta[:time]
            i == 3 && push!(costs, Dict(String(k) => v for (k, v) in r.meta[:cost]))
        end
        @printf("  Nkeep %4d: sweep %.2f s, 2p %.2f s, 3p %.2f s, 4p %.1f s\n", Nk, sweep[end], tp[:, j]...)
    end
    jldsave(out; sweep, psf = tp, costs, N, nkeep = collect(NKEEP), threads = Threads.nthreads())
end

function main(args)
    here = @__DIR__
    repo = normpath(joinpath(here, "..", ".."))
    wt = isempty(args) ? normpath(joinpath(repo, "..", "mpnrg_pre_tensor_refactor")) : abspath(args[1])
    if !isdir(wt)
        run(`git -C $repo worktree add --detach $wt pre-tensor-refactor`)
    end
    commit(dir) = strip(read(`git -C $dir rev-parse HEAD`, String))
    nt = Threads.nthreads()
    jl = Base.julia_cmd()
    res = Dict{String,Any}()
    for (name, proj) in (("pre-tensor-refactor", joinpath(wt, "mpNRG")), ("refactored", joinpath(repo, "mpNRG")))
        println(name, " (", commit(proj), ")")
        run(`$jl --project=$proj -e 'using Pkg; Pkg.instantiate()'`)
        tmp = tempname() * ".jld2"
        run(`$jl -t $nt --project=$proj $(@__FILE__) --measure $tmp`)
        d = load(tmp)
        d["commit"] = commit(proj)
        res[name] = d
        rm(tmp)
    end
    cpu = Sys.cpu_info()[1].model
    out = joinpath(repo, "mpNRG", "notebooks", "figures", "mpnrg", "timing_compare.jld2")
    jldsave(out; results = res, cpu, ncpu = Sys.CPU_THREADS, threads = nt, julia = string(VERSION),
            date = string(round(Int, time())))
    println("written ", out)
    old = res["pre-tensor-refactor"]; new = res["refactored"]
    @printf("%6s  %-22s %-22s %-22s %-22s\n", "Nkeep", "sweep old/new [s]", "2p old/new [s]", "3p old/new [s]", "4p old/new [s]")
    for (j, Nk) in enumerate(NKEEP)
        cols = [(old["sweep"][j], new["sweep"][j]); [(old["psf"][i, j], new["psf"][i, j]) for i in 1:3]]
        @printf("%6d  %s\n", Nk, join([@sprintf("%7.2f/%7.2f (%.2f×)", a, b, b / a) for (a, b) in cols], " "))
    end
end

if MEASURE
    measure(ARGS[2])
else
    main(ARGS)
end
