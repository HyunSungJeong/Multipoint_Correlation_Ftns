# V2: discretization and Wilson chain (task sheet Sec. 6, V2; Sec. 5.3).

using Test, MultipointNRG, LinearAlgebra

@testset "V2a sum rule Σ|V_k|² = (1/π)∫Δ for every z and scheme" begin
    for h in (BoxHybridization(0.1), BoxHybridization(0.3; D = 2.0),
              FunctionHybridization(ω -> 0.1 * sqrt(1 - ω^2); D = 1.0),
              TabulatedHybridization(collect(range(-1, 1; length = 41)), [0.05 + 0.05 * x^2 for x in range(-1, 1; length = 41)]))
        ref = total_coupling(h)
        h isa BoxHybridization && @test ref ≈ 2 * h.D * h.Δ / π rtol = 1e-15
        for z in (0.25, 0.5, 0.75, 1.0), scheme in (:zitko, :wilson, :campo)
            star = discretize(h, Discretization(Λ = 4.0, z = z, scheme = scheme), 30)
            @test Float64(sum(star.V .^ 2)) ≈ ref rtol = 1e-14
        end
    end
end

@testset "V2b Žitko-Pruschke energies: closed form vs App. B ODE" begin
    h = BoxHybridization(0.1)
    for z in (0.3, 0.7, 1.0)
        d = Discretization(Λ = 4.0, z = z)
        E1 = MultipointNRG._zitko_box(d, 1.0, 20, BigFloat)
        E2 = zitko_energies(MultipointNRG._OneSide{typeof(h),Float64}(h, 1), d, 1.0, 20)
        # RK4 with the default step 1/512: truncation error ~1e-13 (measured
        # 4th-order convergence: 5e-10, 3e-11, 2e-12, 1.3e-13 for h = 1/64 … 1/512)
        @test maximum(abs.(Float64.(E1) .- E2) ./ Float64.(E1)) ≤ 1e-12
        E3 = zitko_energies(MultipointNRG._OneSide{typeof(h),Float64}(h, 1), d, 1.0, 20; h = 1 // 256)
        e2 = maximum(abs.(Float64.(E1) .- E2) ./ Float64.(E1)); e3 = maximum(abs.(Float64.(E1) .- E3) ./ Float64.(E1))
        z == 1.0 || @test 10 < e3 / e2 < 25                      # 4th order
        # E_1 for z = 1 is Campo-Oliveira's (1 - Λ^{-1})/ln Λ
        z == 1.0 && @test Float64(E1[1]) ≈ (1 - 1 / 4) / log(4)
    end
end

@testset "V2c chain moments reproduce the star (extended precision)" begin
    h = BoxHybridization(0.1)
    for z in (0.5, 1.0)
        d = Discretization(Λ = 4.0, z = z)
        star = setprecision(() -> discretize(h, d, 24), BigFloat, 512)
        N = 12
        ε, t = lanczos_chain(star, N + 1; prec = 512)
        setprecision(BigFloat, 512) do
            for k in 0:(2N + 1)
                μs = star_moments(star, k); μc = chain_moments(ε, t, k)
                # |E_k| ≤ D = 1, so |μ_k| ≤ μ_0 sets the scale (odd moments vanish by symmetry)
                @test Float64(abs(μs - μc) / star_moments(star, 0)) ≤ 1e-12
            end
        end
    end
end

@testset "V2d t_n Λ^{n/2} approaches a constant; two-precision check" begin
    h = BoxHybridization(0.1)
    c = wilson_chain(h, 40; Λ = 4.0, z = 1.0)
    @test c.precision_check.sites_accepted == 41
    @test c.precision_check.max_relative_deviation ≤ 1e-12
    r = [c.t[n + 1] * 4.0^(n / 2) for n in 1:40]
    @test abs(r[end] - r[end - 1]) / r[end] ≤ 1e-10
    @test abs(r[30] - r[40]) / r[40] ≤ 1e-8
    # Wilson's asymptotic value for the box: (1 + 1/Λ)/2 · Λ^{1/2}... up to the
    # discretization prescription; here only the convergence is asserted
    @test maximum(abs.(c.eps)) ≤ 1e-12                       # particle-hole symmetric band
    # the impurity coupling is t_0 = sqrt((1/π)∫Δ)
    @test c.t[1] ≈ sqrt(2 * 0.1 / π) rtol = 1e-14
    # star of the finite chain: unitary equivalent, Σ V_k² = t_0²
    E, V = chain_star(c, 6)
    @test sum(V .^ 2) ≈ c.t[1]^2 rtol = 1e-13
    @test length(E) == 7
    # an asymmetric hybridization still passes the two-precision check
    ha = FunctionHybridization(ω -> 0.1 * (1 + 0.5ω); D = 1.0)
    ca = wilson_chain(ha, 30)
    @test ca.precision_check.max_relative_deviation ≤ 1e-12
    @test maximum(abs.(ca.eps[20:end] ./ ca.t[20:end])) < 1   # ε_n decays at least as fast as t_n
end
