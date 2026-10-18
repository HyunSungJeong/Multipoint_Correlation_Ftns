# V4b and the AS-basis projector identities (P4).

using Test, MultipointNRG, LinearAlgebra
include(joinpath(@__DIR__, "as_helpers.jl"))

@testset "V4 operators with truncation" begin
    m = siam(U = 0.5, εd = -0.1, Δ = 0.1)
    N = 3; T = 0.03
    run = run_nrg(m, NRGParams(N = N, Nkeep = 20))
    @test run.n0 == 1
    @test all(totaldisc(shell(run, n)) > 0 for n in 1:N)
    @test all(totalkept(shell(run, n)) > 0 for n in 1:(N - 1))
    U = as_basis(run)
    ms = chain_modes(n -> site_modes(m, n), N)
    B = FockBasis(ms)
    @test U[N].basis.modes == ms[(end - length(U[N].basis.modes) + 1):end]
    dense(o) = densematrix(o, B)
    Id = Matrix(1.0I, 4^(N + 2), 4^(N + 2))

    @testset "AS basis: completeness and projector products (Lee Eqs. (24), (27))" begin
        P = Dict((n, X) => as_projector(run, U, n, X) for n in run.n0:N for X in (:K, :D))
        @test norm(sum(P[(n, :D)] for n in run.n0:N) - Id) ≤ 1e-12          # Eq. (24)
        for n in run.n0:N, nb in run.n0:N, X in (:K, :D), Xb in (:K, :D)
            prod = P[(n, X)] * P[(nb, Xb)]
            expect = n < nb ? (X === :K ? P[(nb, Xb)] : 0 * Id) :
                     n == nb ? (X === Xb ? P[(n, X)] : 0 * Id) :
                     (Xb === :K ? P[(n, X)] : 0 * Id)
            @test norm(prod - expect) ≤ 1e-12                               # Eq. (27)
        end
    end

    ops = Dict("d↑" => d_(:up), "d†↑" => dd_(:up), "n↓" => n_(:dn), "d↓" => d_(:dn), "d†↓" => dd_(:dn),
               "q↑" => comm(d_(:up), 0.5 * n_(:up) * n_(:dn)), "x" => dd_(:dn) * f0_(:dn))

    @testset "V4b shell-diagonal expansion reassembles the exact operator (Lee p. 9)" begin
        for (name, o) in ops
            S = zeros(4^(N + 2), 4^(N + 2))
            nop = max(maxsite(o), -1)
            for n in max(run.n0, nop):N
                On = project(run, o, n)
                for X in (:K, :D), Xb in (:K, :D)
                    (X === :K && Xb === :K) && continue
                    S .+= as_embed(run, U, On, n, X, Xb)
                end
            end
            # shells below n0 have no D sector; the expansion starts at n0
            @test norm(S - dense(o)) ≤ 1e-12 * max(1, norm(dense(o)))
        end
    end

end
