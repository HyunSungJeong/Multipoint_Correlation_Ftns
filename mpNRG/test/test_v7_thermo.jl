# V7 (thermodynamics, P5): entropy flow and susceptibility of the SIAM.

using Test, MultipointNRG, LinearAlgebra

@testset "V7 SIAM physics" begin
    m = siam(U = 0.5, Δ = 0.02)                       # U/(πΔ) ≈ 8, εd = -U/2
    T = 1e-9
    N = chain_length_for(T; Λ = 4.0, margin = 6)
    p = NRGParams(N = N, Nkeep = 400)
    run = run_nrg(m, p)
    runb = run_nrg(bath_model(m), p; chains = run.chains)
    @testset "entropy flow ln4 → ln2 → 0 and T χ" begin
        th = evenodd_average(impurity_thermodynamics_shell(run, runb))
        S = th.entropy; X = th.Tchi
        @test abs(S[1] - log(4)) ≤ 0.02                          # free orbital
        i_lm = findfirst(t -> t < 1e-3, th.T)
        @test abs(S[i_lm] - log(2)) ≤ 0.03                       # local moment
        @test 0.18 ≤ X[i_lm] ≤ 0.26                              # T χ → 1/4 (moment)
        @test abs(X[1] - 1 / 8) ≤ 0.01                           # T χ = 1/8 (free orbital)
        @test abs(S[end]) ≤ 1e-3 && abs(X[end]) ≤ 1e-3           # Kondo singlet
        # FDM and conventional agree (same run, single z)
        Ts = [1e-2, 1e-4, 1e-8]
        tf = impurity_thermodynamics_fdm(run, runb, Ts)
        @test abs(tf.entropy[2] - log(2)) ≤ 0.06
        @test abs(tf.entropy[3]) ≤ 0.08
    end
end
