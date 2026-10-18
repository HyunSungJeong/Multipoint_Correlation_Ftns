# Comparison of a recomputed baseline with the frozen one (TNOPS_REFACTOR.md,
# Sec. 8). Used by test_baseline.jl and by the notebook section "Refactor
# regression (TNOps)".
#
# Tolerances (fixed before the refactor, not adjusted afterwards):
#   energies, ρ̄ eigenvalues, shell weights: 1e-12 relative to the largest value
#     of the same shell (energies) / of the same vector (weights, eigenvalues);
#   ln Z, Eref, shifts, thermal averages, thermodynamics: 1e-12 relative,
#     with an absolute floor of 1e-14 for values near zero;
#   PSF histograms bin by bin, totals: 1e-12 of the largest |weight| of the PSF;
#   exact-mode positions: 1e-12; their weights summed per position: 1e-12 of
#     the largest |weight|;
#   singular values: 1e-12 of the largest singular value of the same slice;
#   kept counts and the integer parts of truncation records: identical;
#   the sets of nonzero bins and of exact positions: identical.

const TOL = 1e-12

struct Row
    quantity::String
    maxdev::Float64
    tol::Float64
    pass::Bool
    note::String
end

mutable struct Cmp
    rows::Dict{String,Row}
    flips::Vector{String}
    kept_identical::Bool
end
Cmp() = Cmp(Dict{String,Row}(), String[], true)

function record!(c::Cmp, q, dev, tol; note = "", pass = dev ≤ tol)
    old = get(c.rows, q, nothing)
    if old === nothing
        c.rows[q] = Row(q, dev, tol, pass, note)
    else
        c.rows[q] = Row(q, max(old.maxdev, dev), tol, old.pass && pass, isempty(note) ? old.note : note)
    end
end

reldev(a, b; floor = 1e-14) = abs(a - b) / max(abs(b), floor / TOL)

function cmp_dictvec!(c, q, new::AbstractDict, old::AbstractDict, scale)
    if Set(keys(new)) != Set(keys(old))
        record!(c, q, Inf, TOL; note = "different sector sets"); return
    end
    for k in keys(old)
        a = new[k]; b = old[k]
        if length(a) != length(b)
            record!(c, q, Inf, TOL; note = "different lengths in sector $k"); continue
        end
        isempty(b) && continue
        record!(c, q, maximum(abs.(a .- b)) / max(scale, 1e-300), TOL)
    end
end

maxabs(d::AbstractDict) = maximum((maximum(abs, v; init = 0.0) for v in values(d)); init = 0.0)

function cmp_shells!(c, case, new, old)
    for k in keys(old)
        if startswith(k, "E/")
            cmp_dictvec!(c, "energies", new[k], old[k], maxabs(old[k]))
        elseif startswith(k, "nkept/")
            same = new[k] == old[k]
            same || (c.kept_identical = false)
            record!(c, "kept counts (identical)", same ? 0.0 : Inf, 0.0; note = same ? "" : "$case shell $k")
        elseif startswith(k, "trunc/")
            a = new[k]; b = old[k]
            ints = all(a[f] == b[f] for f in ("requested", "actual", "cut_moved", "total"))
            record!(c, "truncation records (integers identical)", ints ? 0.0 : Inf, 0.0)
            ga, gb = a["gap"], b["gap"]
            record!(c, "truncation gaps", (isnan(ga) && isnan(gb)) ? 0.0 : reldev(ga, gb), TOL)
        elseif startswith(k, "shift/") || startswith(k, "omega/")
            record!(c, "ground-state shifts", reldev(new[k], old[k]), TOL)
        elseif k == "n0"
            record!(c, "n₀ (identical)", new[k] == old[k] ? 0.0 : Inf, 0.0)
        end
    end
end

function cmp_fdm!(c, new, old)
    record!(c, "shell weights w_n", maximum(abs.(new["w"] .- old["w"])) / maximum(abs, old["w"]), TOL)
    record!(c, "ln Z", reldev(new["logZ"], old["logZ"]), TOL)
    record!(c, "E_ref", reldev(new["Eref"], old["Eref"]), TOL)
    for k in keys(old)
        startswith(k, "rhoK_eig/") || continue
        cmp_dictvec!(c, "eigenvalues of ρ̄_KK", new[k], old[k], max(maxabs(old[k]), 1e-300))
    end
end

function cmp_thermal!(c, new, old)
    for k in keys(old)
        q = startswith(k, "thermo/") ? "thermodynamics" : "thermal averages"
        a = new[k]; b = old[k]
        (a isa Number && isnan(a) && isnan(b)) && continue
        record!(c, q, reldev(a, b), TOL)
    end
end

function cmp_psf!(c, case, name, new, old)
    if old["mode"] == "binned"
        bo = unpackbins(old["bins"]); bn = unpackbins(new["bins"])
        scale = max(maximum(abs, values(bo); init = 0.0), 1e-300)
        ko = Set(keys(bo)); kn = Set(keys(bn))
        record!(c, "sets of nonzero bins (identical)", ko == kn ? 0.0 : Inf, 0.0;
                note = ko == kn ? "" : "$case $name: $(length(setdiff(kn, ko))) new, $(length(setdiff(ko, kn))) missing")
        dev = 0.0
        for k in union(ko, kn)
            d = abs(get(bn, k, 0.0) - get(bo, k, 0.0)) / scale
            if d > TOL
                push!(c.flips, "$case $name bin $k: |Δ| = $(d) of max")
            end
            dev = max(dev, d)
        end
        record!(c, "PSF histograms (bin by bin)", dev, TOL)
        record!(c, "PSF totals", abs(new["total"] - old["total"]) / scale, TOL)
        if haskey(old, "absbins")
            ao = unpackbins(old["absbins"]); an = unpackbins(new["absbins"])
            sa = max(maximum(abs, values(ao); init = 0.0), 1e-300)
            da = maximum((abs(get(an, k, 0.0) - get(ao, k, 0.0)) / sa for k in union(keys(ao), keys(an))); init = 0.0)
            record!(c, "|weight| histogram (abs = true)", da, TOL)
        end
    else
        Po, wo = old["clusters"]; Pn, wn = new["clusters"]
        if size(Po) != size(Pn)
            record!(c, "exact-mode positions", Inf, TOL; note = "$case $name: $(size(Pn, 2)) vs $(size(Po, 2)) distinct positions")
            return
        end
        scale = max(maximum(abs, wo; init = 0.0), 1e-300)
        record!(c, "exact-mode positions", maximum(abs.(Pn .- Po); init = 0.0), TOL)
        record!(c, "exact-mode weights per position", maximum(abs.(wn .- wo); init = 0.0) / scale, TOL)
        record!(c, "PSF totals", abs(new["total"] - old["total"]) / scale, TOL)
    end
    haskey(old, "sum_rule") && !isnan(old["sum_rule"]) &&
        record!(c, "FDM sum rules of the PSFs", reldev(new["sum_rule"], old["sum_rule"]), TOL)
end

function cmp_svals!(c, new, old)
    if Set(keys(new)) != Set(keys(old))
        record!(c, "singular values (SingularValueRecorder)", Inf, TOL; note = "different slice sets"); return
    end
    dev = 0.0
    for (k, s) in old
        t = new[k]
        length(t) == length(s) || (dev = Inf; break)
        isempty(s) && continue
        dev = max(dev, maximum(abs.(t .- s)) / max(maximum(s), 1e-300))
    end
    record!(c, "singular values (SingularValueRecorder)", dev, TOL)
end

"""
    compare_baseline(new, old) -> Cmp

All rows of the comparison, the list of bins that moved (bin flips) and
whether all kept counts are identical.
"""
function compare_baseline(new::AbstractDict, old::AbstractDict)
    c = Cmp()
    for (case, ob) in old
        nb = new[case]
        cmp_shells!(c, case, nb["shells"], ob["shells"])
        cmp_fdm!(c, nb["fdm"], ob["fdm"])
        cmp_thermal!(c, nb["thermal"], ob["thermal"])
        for (name, r) in ob["psf"]
            if name == "svals_3p"
                cmp_svals!(c, nb["psf"][name], r)
            else
                cmp_psf!(c, case, name, nb["psf"][name], r)
            end
        end
    end
    return c
end

passed(c::Cmp) = all(r.pass for r in values(c.rows)) && c.kept_identical && isempty(c.flips)

function print_comparison(io::IO, c::Cmp)
    rows = sort(collect(values(c.rows)); by = r -> r.quantity)
    w = maximum(length(r.quantity) for r in rows)
    println(io, rpad("quantity", w), "   max deviation   tolerance   result")
    for r in rows
        dv = isinf(r.maxdev) ? "Inf" : (r.maxdev == 0 ? "0" : string(round(r.maxdev; sigdigits = 2)))
        tl = r.tol == 0 ? "identical" : string(r.tol)
        println(io, rpad(r.quantity, w), "   ", rpad(dv, 13), "   ", rpad(tl, 9), "   ", r.pass ? "pass" : "FAIL",
                isempty(r.note) ? "" : "  ($(r.note))")
    end
    println(io, "kept counts identical in every shell of every case: ", c.kept_identical)
    println(io, "bin flips: ", isempty(c.flips) ? "none" : "")
    for s in c.flips
        println(io, "  ", s)
    end
end
print_comparison(c::Cmp) = print_comparison(stdout, c)
