# Sec. 5.2.10: selection rules and symmetry independence (part of V1, needs the
# PSF engine).

using Test, MultipointNRG, LinearAlgebra

@testset "selection rules: pruned exactly where the shift arithmetic says" begin
    m = siam(U = 0.5, εd = -0.1, Δ = 0.1)
    run = run_nrg(m, NRGParams(N = 2, Nkeep = 30))
    f = fdm(run; T = 0.05)
    g = LogGrid(emin = 1e-6, emax = 10.0, ndec = 8)
    # 3p call with total shift ≠ 0: exact zero, recorded in the metadata
    r = psf(run, f, (d_(:up), dd_(:up), d_(:dn)); grid = g)
    @test r.meta[:pruned] && iszero(r.total) && all(iszero, r.data)
    @test !haskey(r.meta, :cost)                               # no NRG work was done
    # (d↑, d†↑, d↓, d†↓): total shift neutral in every permutation, none pruned
    rs = psfs(run, f, (d_(:up), dd_(:up), d_(:dn), dd_(:dn)); grid = g)
    @test length(rs) == 24
    @test all(!r.meta[:pruned] for r in values(rs))
    G = run.model.G
    for (p, r) in rs
        shifts = r.meta[:shifts]
        @test isneutral(foldl(fuse, shifts))
    end
    # (d↑, d†↓, d↑, d†↓): ΔN = 0 but Δ(2S_z) = -4 in every permutation: all pruned
    rs2 = psfs(run, f, (d_(:up), dd_(:dn), d_(:up), dd_(:dn)); grid = g)
    @test all(r.meta[:pruned] && iszero(r.total) for r in values(rs2))
    # with only the parity declared, the same call is not pruned but its weight vanishes
    mp = siam(U = 0.5, εd = -0.1, Δ = 0.1, symmetry = :parity)
    runp = run_nrg(mp, NRGParams(N = 2, Nkeep = 30))
    fp = fdm(runp; T = 0.05)
    rp = psf(runp, fp, (d_(:up), dd_(:dn), d_(:up), dd_(:dn)); grid = g)
    @test !rp.meta[:pruned] && abs(rp.total) ≤ 1e-13
    # inhomogeneous operators: psf_multilinear skips non-neutral component tuples
    a = d_(:up) + d_(:dn)
    rm = psf_multilinear(run, f, (a, a'); grid = g)
    @test length(rm.meta[:skipped]) == 2 && rm.meta[:ncomponent_tuples] == 2
    @test rm.total ≈ psf(run, f, (d_(:up), dd_(:up)); grid = g).total + psf(run, f, (d_(:dn), dd_(:dn)); grid = g).total
    @test_throws ArgumentError psf(run, f, (a, a'); grid = g)
end

@testset "symmetry independence: Z2 vs U1×Z2 vs U1×U1×Z2" begin
    # identical energies, thermal averages and PSFs: the symmetry layer changes
    # efficiency, not results (truncation falls in the same gap in all runs).
    T = 0.04
    runs = []
    for sym in (:parity, :charge, :charge_spin, :flavor)
        m = siam(U = 0.5, εd = -0.12, Δ = 0.1, B = 0.03, symmetry = sym)
        run = run_nrg(m, NRGParams(N = 3, Nkeep = 48))
        push!(runs, (sym, run, fdm(run; T = T)))
    end
    g = LogGrid(emin = 1e-6, emax = 10.0, ndec = 16)
    _, run0, f0 = runs[3]
    Eall(run, n) = sort(vcat([shell(run, n).E[q] .+ shell(run, n).shift for q in sectors(shell(run, n).space)]...))
    ops2 = (d_(:up), dd_(:up)); ops3 = (d_(:up), dd_(:up), n_(:dn)); ops4 = (d_(:up), dd_(:dn), d_(:dn), dd_(:up))
    kw = (exact = true, wtol = 1e-15, max_terms = 20_000_000)
    ref2 = psf(run0, f0, ops2; kw...); ref3 = psf(run0, f0, ops3; kw...)
    ref4 = psf(run0, f0, ops4; grid = g)
    for (sym, run, f) in runs
        @testset "$sym" begin
            @test [totalkept(sh) for sh in run.shells] == [totalkept(sh) for sh in run0.shells]
            for n in -1:run.params.N
                @test maximum(abs.(Eall(run, n) .- Eall(run0, n))) ≤ 1e-10
            end
            @test f.logZ ≈ f0.logZ atol = 1e-10
            for o in (n_(:up), n_(:up) * n_(:dn), dd_(:dn) * f0_(:dn) + f0_(:dn)' * d_(:dn))
                @test thermal_average(run, f, o) ≈ thermal_average(run0, f0, o) atol = 1e-10
            end
            r2 = psf(run, f, ops2; kw...); r3 = psf(run, f, ops3; kw...)
            @info "dropped |weight| below wtol" sym r2.meta[:dropped_abs_weight] r3.meta[:dropped_abs_weight] length(r3.weights)
            @test r2.meta[:dropped_abs_weight] ≤ 1e-8 && r3.meta[:dropped_abs_weight] ≤ 1e-8
            @test test_functions_error(r2.positions, r2.weights, ref2.positions, ref2.weights) ≤ 1e-10
            @test test_functions_error(r3.positions, r3.weights, ref3.positions, ref3.weights) ≤ 1e-10
            r4 = psf(run, f, ops4; grid = g)
            @test abs(r4.total - ref4.total) ≤ 1e-10
            @test dict_maxdiff(hist_dict(r4), hist_dict(ref4)) ≤ 1e-10
        end
    end
end
