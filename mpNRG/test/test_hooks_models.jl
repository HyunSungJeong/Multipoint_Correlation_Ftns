# Hooks (Sec. 5.10) and the built-in models (Sec. 5.11).

using Test, MultipointNRG, LinearAlgebra

@testset "hooks: per-shell context and the singular-value recorder" begin
    m = siam(U = 0.5, εd = -0.1, Δ = 0.1)
    run = run_nrg(m, NRGParams(N = 5, Nkeep = 24))
    f = fdm(run; T = 0.02)
    g = LogGrid(emin = 1e-6, emax = 10.0, ndec = 16)
    rec = SingularValueRecorder()
    seen = Dict{Int,Any}()
    probe = (run, n, ctx) -> begin
        seen[n] = (length(carried(ctx)), length(call_operators(ctx)),
                   isometries(ctx) === shell(run, n).A, rhoKK(ctx) === MultipointNRG.rhoK(f, n))
        # the refinement map reproduces the stored projections
        if n > run.n0
            O = project(run, d_(:up), n - 1)
            R = refinement_map(ctx)(O)
            @assert all(norm(getblock(R, q) - getblock(project(run, d_(:up), n), q)) < 1e-12 for q in sectors(shell(run, n).space))
        end
    end
    ops = (d_(:up), dd_(:up), n_(:dn))
    r1 = psf(run, f, ops; grid = g, schedule = :breadth, hooks = [rec, probe])
    r0 = psf(run, f, ops; grid = g)
    @test r1.total ≈ r0.total atol = 1e-14                     # hooks do not change results
    @test dict_maxdiff(hist_dict(r1), hist_dict(r0)) ≤ 1e-15
    @test any(v[1] > 0 for v in values(seen))                  # composites are carried
    @test all(v[3] && v[4] for v in values(seen))
    @test !isempty(rec.data) && any(!isempty(v) for v in values(rec.data))
    for (n, v) in rec.data, (born, depth, label, pos, k, sv) in v
        @test born < n && depth ≥ 1 && issorted(sv; rev = true)
    end
end

@testset "built-in models run" begin
    g = LogGrid(emin = 1e-6, emax = 10.0, ndec = 8)
    for m in (kondo(J = 0.4), two_orbital(ε = -0.1, Hint = kanamori(U = 0.2, J = 0.04), Δ = 0.05),
              siam(U = 0.3, hyb = Dict(:up => BoxHybridization(0.05), :dn => BoxHybridization(0.08))),
              siam(U = 0.3, hyb = FunctionHybridization(ω -> 0.05 * sqrt(1 - ω^2))))
        run = run_nrg(m, NRGParams(N = 6, Nkeep = 64))
        f = fdm(run; T = 1e-2)
        @test f.info[:checks].trace ≈ 1 atol = 1e-12
        op = m.name == "Kondo" ? cdag(Mode(0, :up)) * cop(Mode(0, :up)) : num(impurity_modes(m)[1])
        a = psf(run, f, (op, op); grid = g)
        @test a.total ≈ thermal_average(run, f, op * op) atol = 1e-12
    end
end
