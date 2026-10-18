# V4a (PSF weight sum rules with truncation) and V4d (P6-P8).

using Test, MultipointNRG, LinearAlgebra
include(joinpath(@__DIR__, "as_helpers.jl"))

@testset "V4 PSFs with truncation" begin
    m = siam(U = 0.5, εd = -0.1, Δ = 0.1)
    N = 3; T = 0.03
    run = run_nrg(m, NRGParams(N = N, Nkeep = 20))
    @test run.n0 == 1
    @test all(totaldisc(shell(run, n)) > 0 for n in 1:N)
    @test all(totalkept(shell(run, n)) > 0 for n in 1:(N - 1))
    f = fdm(run; T = T)
    U = as_basis(run)
    ms = chain_modes(n -> site_modes(m, n), N)
    B = FockBasis(ms)
    @test U[N].basis.modes == ms[(end - length(U[N].basis.modes) + 1):end]
    dense(o) = densematrix(o, B)
    Id = Matrix(1.0I, 4^(N + 2), 4^(N + 2))

    ops = Dict("d↑" => d_(:up), "d†↑" => dd_(:up), "n↓" => n_(:dn), "d↓" => d_(:dn), "d†↓" => dd_(:dn),
               "q↑" => comm(d_(:up), 0.5 * n_(:up) * n_(:dn)), "x" => dd_(:dn) * f0_(:dn))

    @testset "V4a total weights = Tr[ϱ O¹⋯O^ℓ] (FDM and brute force in the AS basis)" begin
        ρ = as_rho(run, U, f)
        g = LogGrid(emin = 1e-6, emax = 10.0, ndec = 16)
        tuples = [("d↑", "d†↑"), ("d†↑", "d↑"), ("x", "x"), ("q↑", "d†↑"),
                  ("d↑", "d†↑", "n↓"), ("n↓", "d↑", "d†↑"), ("d↑", "n↓", "d†↑"), ("x", "d↑", "d†↑"),
                  ("d↑", "d†↑", "d↓", "d†↓"), ("d↑", "d†↓", "d↓", "d†↑"), ("d†↓", "d↑", "d†↑", "d↓"),
                  ("q↑", "d†↓", "d↓", "d†↑"), ("x", "d↑", "d†↑", "x")]
        for tup in tuples
            os = Tuple(name == "x" && i > 1 && tup[1] == "x" ? ops[name]' : ops[name] for (i, name) in enumerate(tup))
            prod = reduce(*, os)
            brute = tr(ρ * dense(prod))
            fdm_avg = thermal_average(run, f, prod)
            @test fdm_avg ≈ brute atol = 1e-13
            r = psf(run, f, os; grid = g)
            @test abs(r.total - fdm_avg) ≤ 1e-12
            @test abs(r.total - brute) ≤ 1e-12
            if length(os) ≤ 3
                re = psf(run, f, os; exact = true, max_terms = 5_000_000)
                @test abs(re.total - brute) ≤ 1e-12
                # 2p and 3p: no left-referenced slices, so binned = bin(exact)
                @test dict_maxdiff(hist_dict(r), bin_exact(re, g)) ≤ 1e-13
            end
            if length(os) ≥ 3
                @test r.meta[:cost][:composites] > 0       # refinement terms occur
            end
        end
    end
end

@testset "V4d decoupled tail: Lee's formulas are exact with truncation" begin
    # With t_n = ε_n = 0 beyond site n_c the AS states are exact eigenstates
    # and the K sectors refine into exactly degenerate states, so the NRG
    # approximation, the shell-diagonal expansion of nonlocal operators and the
    # slicing are all exact: NRG PSFs (with truncation, all sector strings and
    # nested refinements) must equal ED of the same chain, positions included.
    m = siam(U = 0.5, εd = -0.1, Δ = 0.1)
    N = 3; nc = 1; T = 0.05
    chs = decoupled_tail(wilson_chains(m, N), nc)
    run = run_nrg(m, NRGParams(N = N, Nkeep = 40); chains = chs)
    @test all(totaldisc(shell(run, n)) > 0 for n in 2:N) && totalkept(shell(run, 2)) > 0
    f = fdm(run; T = T)
    ed = model_ed(m, chs, N, T)
    x = dd_(:dn) * f0_(:dn)
    for (ops, slicing) in [((d_(:up), dd_(:up)), :lee), ((d_(:up), dd_(:up), n_(:dn)), :lee),
                           ((n_(:dn), d_(:up), dd_(:up)), :lee), ((d_(:up), n_(:dn), dd_(:up)), :left),
                           ((d_(:up), dd_(:up), d_(:dn), dd_(:dn)), :lee), ((d_(:up), dd_(:dn), d_(:dn), dd_(:up)), :lee),
                           ((d_(:up), dd_(:dn), d_(:dn), dd_(:up)), :left), ((x, d_(:up), dd_(:up), x'), :lee)]
        r = psf(run, f, ops; exact = true, max_terms = 50_000_000, slicing = slicing)
        P, w = ed_psf(ed, ops)
        @test test_functions_error(r.positions, r.weights, P, w) ≤ 1e-12
        # refinement terms carry a finite share of the weight
        deep = sum(abs(r.weights[j]) for j in eachindex(r.weights) if r.labels[j][2] > 0; init = 0.0)
        length(ops) ≥ 3 && @test deep > 1e-3 * sum(abs, r.weights)
    end
end
