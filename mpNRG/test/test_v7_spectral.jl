# V7 (spectral, P6): ⟨n_d⟩ = 1 and the Friedel sum rule of the SIAM.

using Test, MultipointNRG, LinearAlgebra

@testset "V7 SIAM physics" begin
    m = siam(U = 0.5, Δ = 0.02)                       # U/(πΔ) ≈ 8, εd = -U/2
    T = 1e-9
    N = chain_length_for(T; Λ = 4.0, margin = 6)
    p = NRGParams(N = N, Nkeep = 400)
    run = run_nrg(m, p)
    @testset "⟨n_d⟩ = 1 at particle-hole symmetry; Friedel sum rule" begin
        f = fdm(run; T = T)
        nd = thermal_average(run, f, n_(:up) + n_(:dn))
        @test abs(nd - 1) ≤ 1e-10
        g = LogGrid(emin = 1e-12, emax = 10.0, ndec = 32)
        # Discrete FDM data at |ε| ≲ T are inaccurate (Lee Sec. VII B); the Fermi
        # kernel smears but does not remove this, so A(ω) dips by a few percent
        # at ω → 0 (also at U = 0). With Λ = 4 the broadened A(ω) also carries a
        # log-periodic ripple of a few percent. The Friedel value is therefore
        # checked as the logarithmic average of π Δ A(ω) over the plateau
        # 10T ≤ ω ≤ T_K/10 (z-averaged, n_z = 4); A(0) itself is reported.
        zs = zaveraged_psfs(m, p, (d_(:up), dd_(:up)); T = T, nz = 4, grid = g)
        Aat(ω) = broaden(zs[[1, 2]], ω; b = 0.6) .+ broaden(zs[[2, 1]], -ω; b = 0.6)
        ωs = 10 .^ range(-8, -6.5; length = 31)                  # T_K ≈ 4e-6 for these parameters
        plateau = sum(π * m.params[:Δ] .* Aat(ωs)) / length(ωs)
        fr0 = π * m.params[:Δ] * Aat([0.0])[1]
        @info "V7: Friedel sum rule π Δ A(ω)" plateau fr0
        @test abs(plateau - 1) ≤ 0.02
        @test abs(fr0 - 1) ≤ 0.06
    end
end
