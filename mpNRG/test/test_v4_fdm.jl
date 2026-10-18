# V4c: the density matrix with truncation (P5).

using Test, MultipointNRG, LinearAlgebra
include(joinpath(@__DIR__, "as_helpers.jl"))

@testset "V4 density matrix with truncation" begin
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

    @testset "V4c density matrix: Tr ϱ = 1, ρ̄ ≥ 0, ρ̄ = partial trace of ϱ" begin
        ρ = as_rho(run, U, f)
        @test tr(ρ) ≈ 1 atol = 1e-13
        @test f.info[:checks].trace ≈ 1 atol = 1e-14
        @test f.info[:checks].min_eigenvalue ≥ -1e-15
        @test f.info[:checks].kk_trace_deviation ≤ 1e-14
        # ρ̄^n (both sectors) equals the partial trace of the dense ϱ over sites > n
        for n in run.n0:N
            sh = shell(run, n)
            dn = size(U[n].cols[first(sectors(sh.space))], 1); de = 4^(N - n)
            R = reshape(ρ, dn, de, dn, de)
            red = zeros(dn, dn)
            for e in 1:de
                red .+= R[:, e, :, e]
            end
            for q in sectors(sh.space)
                @test norm(U[n].cols[q]' * red * U[n].cols[q] - MultipointNRG.rho_full(f, run, n, q)) ≤ 1e-13
            end
        end
    end

end
