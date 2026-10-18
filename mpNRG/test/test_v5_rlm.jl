# V5: resonant level model (U = 0). The single-particle problem of the finite
# chain is solved exactly; NRG 2p PSFs with truncation are compared bin by bin,
# and after broadening with the continuum A(ω) of the box. Deviations are
# reported; the hard requirement is the exact total weight (V4a).

using Test, MultipointNRG, LinearAlgebra

# exact discrete PSF S[d, d†](ε) = Σ_j |u_{dj}|² (1 - f(λ_j)) δ(ε - λ_j) of the
# finite chain (spinless level coupled to the chain)
function rlm_exact(m, chain, N, T)
    εd = m.params[:εd]
    n = N + 2
    H = zeros(n, n)
    H[1, 1] = εd; H[1, 2] = H[2, 1] = chain.t[1]
    for k in 0:N
        H[k + 2, k + 2] = chain.eps[k + 1]
        k ≥ 1 && (H[k + 1, k + 2] = H[k + 2, k + 1] = chain.t[k + 1])
    end
    F = eigen(Symmetric(H))
    w = F.vectors[1, :] .^ 2
    fermi = 1 ./ (exp.(F.values ./ T) .+ 1)
    return F.values, w .* (1 .- fermi), w .* fermi
end

@testset "V5 resonant level model" begin
    m = resonant_level(εd = 0.02, Δ = 0.05)
    T = 1e-4
    N = chain_length_for(T; Λ = 4.0, margin = 4)
    run = run_nrg(m, NRGParams(N = N, Nkeep = 64))
    f = fdm(run; T = T)
    g = LogGrid(emin = 1e-7, emax = 10.0, ndec = 32)
    d = cop(Mode(-1, :d))
    r = psf(run, f, (d, d'); grid = g)
    rr = psf(run, f, (d', d); grid = g)
    # exact total weights (V4a) and the anticommutator sum rule
    @test abs(r.total - r.meta[:sum_rule]) ≤ 1e-12
    @test r.total + rr.total ≈ 1 atol = 1e-12
    λ, wp, wm = rlm_exact(m, run.chains[1], N, T)
    hx = bin_terms(reshape(λ, 1, :), wp, g)
    # occupation from NRG (FDM) vs exact single-particle value
    nexact = sum(wm)
    @test abs(rr.total - nexact) ≤ 1e-3
    # bin-by-bin comparison after coarse graining into decades (NRG positions are
    # approximate: report the deviation)
    dec(i) = i == zerobin(g) ? 0 : sign(i - zerobin(g)) * cld(abs(i - zerobin(g)), g.ndec)
    coarse(h) = (acc = Dict{Int,Float64}(); for (k, v) in h; acc[dec(k[1])] = get(acc, dec(k[1]), 0.0) + v; end; acc)
    cN = coarse(hist_dict(r)); cE = coarse(hx)
    dev = maximum(abs(get(cN, k, 0.0) - get(cE, k, 0.0)) for k in union(keys(cN), keys(cE)))
    @info "V5: max per-decade weight deviation NRG vs exact discrete" dev occupation_error = abs(rr.total - nexact)
    @test dev ≤ 0.05
    # broadened vs continuum A(ω) of the box: A = -Im G/π, G = [ω - ε_d - Δ(ω)]⁻¹
    Δ = m.params[:Δ]; D = 1.0; εd = m.params[:εd]
    hyb(ω) = Δ / π * log(abs((D + ω) / (D - ω))) - im * Δ
    Acont(ω) = -imag(1 / (ω - εd - hyb(ω))) / π
    ωs = [-0.2, -0.1, -0.05, -0.02, 0.0, 0.01, 0.02, 0.04, 0.1, 0.2]
    # single z with Λ = 4: log-periodic discretization artefacts need b ≲ Λ - 1
    # (Lee Sec. VI B); with z-averaging (n_z = 4) a narrow b = 0.3 is possible
    A1 = broaden(r, ωs; b = 0.6) .+ broaden(rr, -ωs; b = 0.6)
    rel1 = maximum(abs.(A1 .- Acont.(ωs)) ./ Acont.(ωs))
    zs = zaveraged_psfs(m, NRGParams(N = N, Nkeep = 64), (d, d'); T = T, nz = 4, grid = g)
    A4 = broaden(zs[[1, 2]], ωs; b = 0.3) .+ broaden(zs[[2, 1]], -ωs; b = 0.3)
    rel = maximum(abs.(A4 .- Acont.(ωs)) ./ Acont.(ωs))
    @info "V5: broadened NRG A(ω) vs continuum box model (max relative deviation)" single_z_b06 = rel1 nz4_b03 = rel
    @test rel ≤ 0.12
    @test sum(r.total for r in values(zs)) ≈ 1 atol = 1e-12
end
