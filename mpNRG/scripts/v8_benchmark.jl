# V8: compare with the published mpNRG PSFs of Frankenbach et al. (LMU Open
# Data 613, dataset siam05_U0.05_T0.05_Delta0.0318). Writes
# notebooks/figures/mpnrg/v8_results.jld2, read by the validation notebook.
#
# Run (about 3 hours on 4 threads in total; every part is saved separately in
# notebooks/figures/mpnrg/v8_parts/ and skipped if present, so the script can be
# resumed):
#   python3 mpNRG/scripts/extract_lmu613.py <unzipped dataset> mpNRG/refs/lmu613/lmu613_siam05_T0.05.h5
#   julia -t 4 --project=mpNRG/test mpNRG/scripts/v8_benchmark.jl [Nkeep4p] [part]
# with part in {2p, totals, 4p, collect, all} (default all).

using MultipointNRG, LinearAlgebra, JLD2, Printf

const DATA = joinpath(@__DIR__, "..", "refs", "lmu613", "lmu613_siam05_T0.05.h5")
const OUT = joinpath(@__DIR__, "..", "notebooks", "figures", "mpnrg", "v8_results.jld2")
Nk4 = length(ARGS) ≥ 1 ? parse(Int, ARGS[1]) : 100
PART = length(ARGS) ≥ 2 ? ARGS[2] : "all"
const PARTS = joinpath(@__DIR__, "..", "notebooks", "figures", "mpnrg", "v8_parts")
mkpath(PARTS)
partfile(name) = joinpath(PARTS, name * ".jld2")
done(name) = isfile(partfile(name))
save_part(name, x) = jldsave(partfile(name); x = x)
load_part(name) = load(partfile(name), "x")

ref = jldopen(DATA)
od = ref["odisc"]
U = ref["params/U"]; T = ref["params/T"]; Δ = ref["params/Rhoin_max"] * π   # true Δ (README of the dataset)
Λ = ref["params/Lambda"]; N = Int(ref["params/N"])
m = siam(U = U, εd = -U / 2, Δ = Δ)
# the dataset's grid: 10^{k/8} for |ε| from 10^{-21/8} to 10^{12/8}, plus 0
g = LogGrid(emin = 10^(-21 / 8), emax = 10^(1.5) * (1 + 1e-12), ndec = 8)
@assert maximum(abs.(centres(g) .- od) ./ max.(abs.(od), 1e-300)) < 1e-12
zs = (0.5, 1.0)

d(σ) = cop(Mode(-1, σ))
Hint = U * num(Mode(-1, :up)) * num(Mode(-1, :dn))
q(σ) = comm(d(σ), Hint)
F = Dict("F1" => d(:up), "F1dag" => d(:up)', "F3" => d(:dn), "F3dag" => d(:dn)', "Q1" => q(:up), "Q1dag" => q(:up)')

# barycentric splitting in ln|ε| between neighbouring grid points (the
# dataset's binning, "bary"; identified in STATUS.md)
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

out = Dict{String,Any}("odisc" => od, "U" => U, "T" => T, "Delta" => Δ, "Lambda" => Λ, "N" => N, "Nkeep4p" => Nk4)

# --- 2p: exact positions, then the dataset's binning; z-averaged
names2 = [("F1", "F1dag"), ("F1dag", "F1"), ("Q1", "Q1dag"), ("Q1dag", "Q1"), ("F1", "Q1dag"), ("Q1dag", "F1"),
          ("Q1", "F1dag"), ("F1dag", "Q1")]
for Nk in (400, 800)
    (PART in ("2p", "all") && !done("psf2_Nk$Nk")) || continue
    h2 = Dict(join(n, "_") => zeros(length(od)) for n in names2)
    hc = Dict(join(n, "_") => zeros(length(od)) for n in names2)
    for z in zs
        run = run_nrg(m, NRGParams(N = N, Nkeep = Nk, Λ = Λ, z = z)); f = fdm(run; T = T)
        for nm in names2
            ops = (F[nm[1]], F[nm[2]])
            r = psf(run, f, ops; exact = true, max_terms = 100_000_000)
            for j in eachindex(r.weights)
                logbary!(h2[join(nm, "_")], r.positions[1, j], r.weights[j] / length(zs), od)
            end
            rb = psf(run, f, ops; grid = g)
            hc[join(nm, "_")] .+= histogram(rb) ./ length(zs)
        end
    end
    save_part("psf2_Nk$Nk", Dict("bary" => h2, "centre" => hc))
    println("2p done for Nkeep = $Nk"); flush(stdout)
end

# --- connected 4p totals for every permutation, from FDM thermal averages
perms = all_perms(4)
opsnames = ["F1", "F1dag", "F3", "F3dag"]
spinops(fl) = fl == 1 ? [d(:up), d(:up)', d(:up), d(:up)'] : [d(:up), d(:up)', d(:dn), d(:dn)']
ζ = -1
function con_total(run, f, O)
    t(a, b) = thermal_average(run, f, a * b)
    full = thermal_average(run, f, O[1] * O[2] * O[3] * O[4])
    dis = t(O[1], O[2]) * t(O[3], O[4]) + ζ * t(O[1], O[3]) * t(O[2], O[4]) + t(O[1], O[4]) * t(O[2], O[3])
    return full - dis, full, dis
end
for Nk in (400, 800)
    (PART in ("totals", "all") && !done("totals_Nk$Nk")) || continue
    tot = Dict{String,Any}()
    for fl in (1, 2), p in perms
        tot["Nk$(Nk)_fl$(fl)_" * join(p)] = 0.0
    end
    for z in zs
        run = run_nrg(m, NRGParams(N = N, Nkeep = Nk, Λ = Λ, z = z)); f = fdm(run; T = T)
        for fl in (1, 2), p in perms
            O = spinops(fl)[p]
            tot["Nk$(Nk)_fl$(fl)_" * join(p)] += con_total(run, f, O)[1] / length(zs)
        end
    end
    save_part("totals_Nk$Nk", tot)
    println("connected totals done for Nkeep = $Nk"); flush(stdout)
end

# --- 4p PSFs (binned, Lee's centre binning) for all 24 permutations of both
# flavors; the ε₁ and ε₃ marginals of their connected parts need only 2p
# histograms (the disconnected terms factorize there, Lee's Eq. after (79))
function marginals_con(run, f, O, g)
    r = psf(run, f, Tuple(O); grid = g, sparse = false)
    H = histogram(r)
    m1 = vec(sum(H; dims = (2, 3))); m3 = vec(sum(H; dims = (1, 2)))
    s2(a, b) = histogram(psf(run, f, (a, b); grid = g))
    t(a, b) = thermal_average(run, f, a * b)
    m1 .-= s2(O[1], O[2]) .* t(O[3], O[4]) .+ ζ .* s2(O[1], O[3]) .* t(O[2], O[4]) .+ s2(O[1], O[4]) .* t(O[2], O[3])
    m3 .-= t(O[1], O[2]) .* s2(O[3], O[4]) .+ ζ .* t(O[1], O[3]) .* s2(O[2], O[4]) .+ t(O[2], O[3]) .* s2(O[1], O[4])
    return m1, m3, r.total, r.meta[:time]
end
if PART in ("4p", "all")
    runs = Dict{Float64,Any}()
    # flavor 2 (σσ′) first; for each permutation both z values, so a partial run gives complete z-averages
    for fl in (2, 1), p in perms, z in zs
        name = "m4_Nk$(Nk4)_z$(z)_fl$(fl)_" * join(p)
        done(name) && continue
        haskey(runs, z) || (r = run_nrg(m, NRGParams(N = N, Nkeep = Nk4, Λ = Λ, z = z)); runs[z] = (r, fdm(r; T = T)))
        m1, m3, tt, tm = marginals_con(runs[z]..., spinops(fl)[p], g)
        save_part(name, Dict("m1" => m1, "m3" => m3, "total" => tt, "time" => tm))
        @printf("z = %.2f flavor %d perm %s: %.1f s\n", z, fl, join(p), tm); flush(stdout)
    end
end
PART in ("collect", "all") || exit()
for Nk in (400, 800)
    done("psf2_Nk$Nk") && (x = load_part("psf2_Nk$Nk"); out["psf2_bary_Nk$Nk"] = x["bary"]; out["psf2_centre_Nk$Nk"] = x["centre"])
    done("totals_Nk$Nk") && (out["con_totals_Nk$Nk"] = load_part("totals_Nk$Nk"))
end
marg = Dict{String,Any}()
for fl in (1, 2), p in perms
    key = "fl$(fl)_" * join(p)
    parts = [load_part("m4_Nk$(Nk4)_z$(z)_" * key) for z in zs if done("m4_Nk$(Nk4)_z$(z)_" * key)]
    length(parts) == length(zs) || continue
    marg[key] = Dict("m1" => sum(x["m1"] for x in parts) ./ length(zs), "m3" => sum(x["m3"] for x in parts) ./ length(zs),
                     "time" => sum(x["time"] for x in parts))
end
out["marginals"] = marg

# reference marginals and totals from the dataset
refm = Dict{String,Any}()
for fl in (1, 2), p in perms
    A = ref["psf4/$(join(p))/flavor$fl"]
    refm["fl$(fl)_" * join(p)] = Dict("m1" => vec(sum(A; dims = (2, 3))), "m3" => vec(sum(A; dims = (1, 2))), "total" => sum(A))
end
out["ref_marginals"] = refm
out["ref_psf2"] = Dict(k => ref["psf2/$k"] for k in keys(ref["psf2"]))
jldsave(OUT; results = out)
println("written ", OUT)
