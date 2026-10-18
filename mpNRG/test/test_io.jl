# I/O and z-averaging (Sec. 5.8).

using Test, MultipointNRG, LinearAlgebra

@testset "I/O and z-averaging" begin
    m = siam(U = 0.5, εd = -0.1, Δ = 0.1)
    g = LogGrid(emin = 1e-6, emax = 10.0, ndec = 16)
    p = NRGParams(N = 6, Nkeep = 32)
    zs = zaveraged_psfs(m, p, (d_(:up), dd_(:up)); T = 1e-2, nz = 2, grid = g)
    @test length(zs) == 2
    for r in values(zs)
        @test r.meta[:zaveraged] == 2
        @test r.total ≈ r.meta[:sum_rule] atol = 1e-12
    end
    @test sum(r.total for r in values(zs)) ≈ 1 atol = 1e-12
    run = run_nrg(m, p); f = fdm(run; T = 1e-2)
    r = psf(run, f, (d_(:up), dd_(:up)); grid = g)
    g2 = LogGrid(emin = 1e-5, emax = 10.0, ndec = 16)
    r2 = psf(run, f, (d_(:up), dd_(:up)); grid = g2)
    @test_throws ArgumentError zaverage([r, r2])                # different grids refused
    r3 = psf(run, f, (dd_(:up), d_(:up)); grid = g)
    @test_throws ArgumentError zaverage([r, r3])                # different operators refused
    path = joinpath(mktempdir(), "run.jld2")
    r4 = psf(run, f, (d_(:up), dd_(:up), d_(:dn), dd_(:dn)); grid = g, abs = true)
    save_results(path, run, f, Dict("2p" => r, "4p" => r4); thermal = Dict("n_up" => thermal_average(run, f, n_(:up))))
    d = load_results(path)
    @test d["metadata"]["N"] == 6 && d["metadata"]["Lambda"] == 4.0
    @test haskey(d["metadata"], "git_commit") && haskey(d["metadata"], "package_version")
    @test d["psfs"]["2p"]["histogram"] ≈ histogram(r)
    @test size(d["psfs"]["4p"]["bins"], 2) == 3 && sum(d["psfs"]["4p"]["values"]) ≈ r4.total
    @test all(d["psfs"]["4p"]["abs_values"] .≥ abs.(d["psfs"]["4p"]["values"]) .- 1e-15)
end

@testset "files written by the pre-refactor code still load (TNOps refactor, Sec. 5)" begin
    # written by test/baseline/make_baseline.jl at the tag pre-tensor-refactor (bb8b197)
    d = load_results(joinpath(@__DIR__, "baseline", "old_output.jld2"))
    @test Set(keys(d)) == Set(["metadata", "psfs", "thermal_averages"])
    @test startswith(d["metadata"]["git_commit"], "bb8b197")
    @test d["metadata"]["N"] == 2 && d["metadata"]["symmetry"] == "U1(:N) × U1(:twoSz) × Z2(:Pf)"
    # the same calculation with the current code reproduces the stored PSFs
    m = siam(U = 0.5, εd = -0.1, Δ = 0.1)
    run = run_nrg(m, NRGParams(N = 2, Nkeep = typemax(Int))); f = fdm(run; T = 0.05)
    g = LogGrid(emin = 1e-6, emax = 10.0, ndec = 16)
    old = d["psfs"]["[1, 2]"]
    @test old["mode"] == "binned" && old["grid"]["ndec"] == 16
    @test maximum(abs.(old["histogram"] .- histogram(psf(run, f, (d_(:up), dd_(:up)); grid = g)))) ≤ 1e-12
    ex = d["psfs"]["[0]"]
    @test size(ex["positions"], 1) == 1 && length(ex["weights"]) == size(ex["positions"], 2)
    @test sum(ex["weights"]) ≈ ex["total"] atol = 1e-12
    @test d["thermal_averages"]["n_up"] ≈ thermal_average(run, f, n_(:up)) atol = 1e-12
end
