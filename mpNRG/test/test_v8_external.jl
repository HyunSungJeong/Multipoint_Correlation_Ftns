# V8: external benchmark against the published mpNRG PSFs of Frankenbach et al.
# (LMU Open Data 613). Needs the extracted dataset
#   refs/lmu613/lmu613_siam05_T0.05.h5   (scripts/extract_lmu613.py; refs/ is gitignored)
# and is skipped without it. The full comparison (all 24 permutations of both
# spin flavors, marginals) is done by scripts/v8_benchmark.jl and shown in the
# notebook; this test checks a light subset. The dataset's NRG uses SU(2) spin
# symmetry with 300 kept multiplets and its own binning ("bary": weights split
# between neighbouring grid points linearly in ln|ε|), so agreement is not exact.

using Test, MultipointNRG, LinearAlgebra, JLD2

const V8DATA = joinpath(@__DIR__, "..", "refs", "lmu613", "lmu613_siam05_T0.05.h5")

function logbary!(h, x, w, od)
    z = findfirst(==(0.0), od)
    abs(x) < 1e-14 && (h[z] += w; return)
    side = x > 0 ? ((z + 1):length(od)) : reverse(1:(z - 1))
    a = abs.(od[side]); ax = abs(x)
    ax ≤ a[1] && (h[side[1]] += w; return)
    ax ≥ a[end] && (h[side[end]] += w; return)
    i = searchsortedlast(a, ax)
    t = (log(ax) - log(a[i])) / (log(a[i + 1]) - log(a[i]))
    h[side[i]] += (1 - t) * w; h[side[i + 1]] += t * w
end

@testset "V8 external benchmark (Frankenbach et al., LMU Open Data 613)" begin
    if !isfile(V8DATA)
        @info "V8 skipped: $V8DATA not found (see scripts/extract_lmu613.py)"
        @test_skip false
    else
        ref = jldopen(V8DATA)
        od = ref["odisc"]
        U = ref["params/U"]; T = ref["params/T"]; Δ = ref["params/Rhoin_max"] * π
        @test ref["params/Lambda"] == 4.0 && ref["params/N"] == 12 && ref["params/nz"] == 2
        m = siam(U = U, εd = -U / 2, Δ = Δ)
        g = LogGrid(emin = 10^(-21 / 8), emax = 10^(1.5) * (1 + 1e-12), ndec = 8)
        @test maximum(abs.(centres(g) .- od) ./ max.(abs.(od), 1e-300)) ≤ 1e-12   # the same grid
        h = zeros(length(od))
        runs = [let run = run_nrg(m, NRGParams(N = 12, Nkeep = 300, z = z)); (run, fdm(run; T = T)); end for z in (0.5, 1.0)]
        for (run, f) in runs
            r = psf(run, f, (d_(:up), dd_(:up)); exact = true, max_terms = 50_000_000)
            for j in eachindex(r.weights)
                logbary!(h, r.positions[1, j], r.weights[j] / 2, od)
            end
        end
        A = ref["psf2/F1_F1dag"]
        @test sum(h) ≈ sum(A) atol = 1e-10                          # ⟨d d†⟩ = 1/2 both
        @test abs(sum(h .* od) - sum(A .* od)) ≤ 0.01 * abs(sum(A .* od))   # first moment
        dev = maximum(abs.(h .- A)) / maximum(A)
        @info "V8: 2p S[d↑, d†↑] max bin deviation / max bin" dev
        @test dev ≤ 0.15
        # connected 4p totals (σσ′ and σσ flavors) from FDM thermal averages
        d1, d3 = d_(:up), d_(:dn)
        for (fl, O) in ((2, [d1, d1', d3, d3']), (1, [d1, d1', d1, d1']))
            for p in ([1, 2, 3, 4], [1, 3, 2, 4], [2, 1, 4, 3], [1, 4, 3, 2])
                Op = O[p]
                con = 0.0
                for (run, f) in runs
                    t(a, b) = thermal_average(run, f, a * b)
                    full = thermal_average(run, f, Op[1] * Op[2] * Op[3] * Op[4])
                    con += (full - (t(Op[1], Op[2]) * t(Op[3], Op[4]) - t(Op[1], Op[3]) * t(Op[2], Op[4]) +
                                    t(Op[1], Op[4]) * t(Op[2], Op[3]))) / 2
                end
                theirs = sum(ref["psf4/$(join(p))/flavor$fl"])
                @test abs(con - theirs) ≤ max(0.01 * abs(theirs), 2e-4)
            end
        end
    end
end
