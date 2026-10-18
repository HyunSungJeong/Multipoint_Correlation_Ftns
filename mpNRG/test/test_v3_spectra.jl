# V3 (spectra, P3): without truncation the shell energies of the last shell are
# the exact eigenenergies of the finite chain.

using Test, MultipointNRG, LinearAlgebra

@testset "V3a spectra without truncation vs reference ED" begin
    models = [("SIAM asym", siam(U = 0.5, εd = -0.1, Δ = 0.1), 2),
              ("SIAM field", siam(U = 0.4, εd = -0.15, B = 0.05, Δ = 0.08), 2),
              ("RLM", resonant_level(εd = 0.05, Δ = 0.1), 4),
              ("Kondo", kondo(J = 0.3, B = 0.02), 2),
              ("two-orbital", two_orbital(ε = (-0.2, -0.1), Hint = kanamori(U = 0.3, J = 0.05), Δ = 0.05), 0)]
    for (name, m, N) in models
        @testset "$name" begin
            run = run_nrg(m, NRGParams(N = N, Nkeep = typemax(Int)))
            sh = shell(run, N)
            @test totalkept(sh) == 0 && totaldisc(sh) == dim(sh.space)
            ed = model_ed(m, run.chains, N, 0.1)
            Enrg = sort(vcat([sh.E[q] .+ sh.shift for q in sectors(sh.space)]...))
            Eed = sort(vcat(values(ed.E)...))
            @test length(Enrg) == length(Eed)
            @test maximum(abs.(Enrg .- Eed)) ≤ 1e-12
            # sector by sector (block structure consistent with the declared symmetries)
            for q in sectors(sh.space)
                @test sort(sh.E[q] .+ sh.shift) ≈ sort(ed.E[q]) atol = 1e-12
            end
            # flow data recorded per shell
            @test all(haskey(s.info, :truncation) for s in run.shells)
        end
    end
end
