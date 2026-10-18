# V6: internal consistency (task sheet Sec. 6, V6).

using Test, MultipointNRG, LinearAlgebra

# per-decade coarse graining of a binned histogram (comparisons at NRG resolution)
function decades(h, g)
    z = zerobin(g)
    dec(i) = i == z ? 0 : sign(i - z) * cld(abs(i - z), g.ndec)
    out = Dict{Any,Float64}()
    for (k, v) in h
        kk = map(dec, k)
        out[kk] = get(out, kk, 0.0) + v
    end
    return out
end

@testset "V6a equivalent trace forms and schedules agree" begin
    m = siam(U = 0.5, εd = -0.1, Δ = 0.1)
    N = 4; T = 0.02
    run = run_nrg(m, NRGParams(N = N, Nkeep = 24))
    f = fdm(run; T = T)
    g = LogGrid(emin = 1e-6, emax = 10.0, ndec = 16)
    # The forms differ in which energies define a sliced frequency: Lee's form
    # slices ε_i = E_k^n - E_1^n directly; the left-referenced form assembles
    # (E_k^n - E_j^n) + (E_j^n̄ - E_1^n̄) with the second part refined at a later
    # shell n̄. They coincide where the NRG approximation is exact (V4d) and
    # otherwise agree to NRG accuracy; total weights agree exactly.
    for ops in [(d_(:up), dd_(:up), n_(:dn)), (n_(:dn), d_(:up), dd_(:up)), (d_(:up), n_(:dn), dd_(:up))]
        a = psf(run, f, ops; exact = true, slicing = :lee)
        b = psf(run, f, ops; exact = true, slicing = :left)
        @test abs(a.total - b.total) ≤ 1e-13
        ab = psf(run, f, ops; grid = g, slicing = :lee); bb = psf(run, f, ops; grid = g, slicing = :left)
        @test abs(ab.total - bb.total) ≤ 1e-13
        # Lee's form is bin(exact) exactly for ℓ = 3 (no left-referenced slices)
        @test dict_maxdiff(hist_dict(ab), bin_exact(a, g)) ≤ 1e-14
        dd = dict_maxdiff(decades(hist_dict(ab), g), decades(hist_dict(bb), g))
        @info "V6a: 3p forms, max per-decade difference" ops dd total = ab.total
        @test dd ≤ 0.05 * abs(ab.total)
    end
    ops4 = (d_(:up), dd_(:dn), d_(:dn), dd_(:up))
    a = psf(run, f, ops4; grid = g, slicing = :lee); b = psf(run, f, ops4; grid = g, slicing = :left)
    c = psf(run, f, ops4; grid = g, slicing = :lee, schedule = :breadth)
    @test abs(a.total - b.total) ≤ 1e-13
    @test dict_maxdiff(hist_dict(a), hist_dict(c)) ≤ 1e-14      # schedules: identical
    dd = dict_maxdiff(decades(hist_dict(a), g), decades(hist_dict(b), g))
    @info "V6a: 4p forms, max per-decade difference" dd total = a.total
    @test dd ≤ 0.05 * abs(a.total)
end

@testset "V6b total weights are independent of z" begin
    m = siam(U = 0.5, εd = -0.1, Δ = 0.1)
    g = LogGrid(emin = 1e-6, emax = 10.0, ndec = 16)
    for z in (0.25, 0.5, 0.75, 1.0)
        run = run_nrg(m, NRGParams(N = 8, Nkeep = 64, z = z))
        f = fdm(run; T = 1e-3)
        a = psf(run, f, (d_(:up), dd_(:up)); grid = g); b = psf(run, f, (dd_(:up), d_(:up)); grid = g)
        @test a.total + b.total ≈ 1 atol = 1e-12                  # ⟨{d, d†}⟩ = 1
        x = psf(run, f, (d_(:up), dd_(:up), n_(:dn)); grid = g); y = psf(run, f, (dd_(:up), d_(:up), n_(:dn)); grid = g)
        @test x.total + y.total ≈ thermal_average(run, f, n_(:dn)) atol = 1e-12
    end
end

@testset "V6c convergence with N_keep towards the exact result" begin
    m = siam(U = 0.5, εd = -0.1, Δ = 0.1)
    N = 3; T = 0.03
    chs = wilson_chains(m, N)
    ed = model_ed(m, chs, N, T)
    errs = Dict{Int,Vector{Float64}}()
    for Nk in (8, 16, 32, 64, 128, 1024)
        run = run_nrg(m, NRGParams(N = N, Nkeep = Nk); chains = chs)
        f = fdm(run; T = T)
        e = Float64[]
        for ops in [(d_(:up), dd_(:up)), (d_(:up), dd_(:up), n_(:dn))]
            r = psf(run, f, ops; exact = true, max_terms = 20_000_000)
            P, w = ed_psf(ed, ops)
            push!(e, test_functions_error(r.positions, r.weights, P, w; ks = (0.5, 2.0, -3.0)))
        end
        errs[Nk] = e
    end
    @info "V6c: test-function error vs ED as N_keep grows" errs
    for i in 1:2
        @test errs[1024][i] ≤ 1e-11                       # no truncation left
        @test errs[128][i] ≤ errs[16][i]
        @test errs[64][i] ≤ errs[8][i]
    end
end
