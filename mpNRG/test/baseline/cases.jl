# Regression baseline of the TNOps refactor (TNOPS_REFACTOR.md, Sec. 8): the
# calculations, shared by `make_baseline.jl` (run once with the code at the
# tag `pre-tensor-refactor`) and `test_baseline.jl` (run after every refactor
# phase). Only basis-independent quantities are recorded: energies, kept
# counts, truncation records, shifts, shell weights, ln Z, eigenvalues of
# ρ̄^n_KK, thermal averages, thermodynamics, PSF histograms, exact-mode
# positions with the weights summed per position, singular values.
#
# Everything here goes through the public API, so the same code runs before
# and after the refactor; `bl_rhoK_blocks` is the one accessor whose return
# type the refactor changes.

using MultipointNRG, LinearAlgebra

const BL_GRID = LogGrid(emin = 1e-6, emax = 10.0, ndec = 16)
const BL_GRID4 = LogGrid(emin = 1e-4, emax = 10.0, ndec = 4)    # 4p: coarser, keeps the file small

d_(σ) = cop(Mode(-1, σ)); dd_(σ) = cdag(Mode(-1, σ)); n_(σ) = num(Mode(-1, σ))

"""The baseline runs: name => (model, params, T)."""
function bl_cases()
    m1 = siam(U = 0.5, εd = -0.1, Δ = 0.1)                                # U1×U1×Z2
    m2 = siam(U = 0.5, εd = -0.1, Δ = 0.1, symmetry = :parity)            # Z2(:Pf) only
    m3 = siam(U = 0.4, εd = -0.13, B = 0.02, Δ = 0.08)                    # no spin degeneracy (|w| case)
    return [
        "untrunc" => (m1, NRGParams(N = 2, Nkeep = typemax(Int)), 0.05),
        "trunc" => (m1, NRGParams(N = 5, Nkeep = 24), 1e-3),
        "parity" => (m2, NRGParams(N = 4, Nkeep = 32), 1e-3),
        "field" => (m3, NRGParams(N = 5, Nkeep = 28), 2e-3),
    ]
end

skey(q) = q.q                                  # sector label as a plain tuple

# ρ̄^n_KK as sector => dense matrix (pre-refactor: a Dict of matrices)
bl_rhoK_blocks(f, n) = rhoK(f, n)

function bl_shells(run)
    out = Dict{String,Any}()
    for n in -1:run.params.N
        sh = shell(run, n)
        out["E/$n"] = Dict(skey(q) => copy(sh.E[q]) for q in sectors(sh.space))
        out["nkept/$n"] = Dict(skey(q) => nkept(sh, q) for q in sectors(sh.space))
        rec = sh.info[:truncation]
        out["trunc/$n"] = Dict(string(k) => v for (k, v) in pairs(rec))
        out["shift/$n"] = sh.shift
        out["omega/$n"] = sh.info[:ω]
    end
    out["n0"] = run.n0
    return out
end

function bl_fdm(run, f)
    out = Dict{String,Any}("w" => copy(f.w), "logZ" => f.logZ, "Eref" => f.Eref)
    for n in -1:run.params.N
        out["rhoK_eig/$n"] = Dict(skey(q) => sort(real(eigvals(Hermitian((M + M') / 2))))
                                  for (q, M) in bl_rhoK_blocks(f, n) if !isempty(M))
    end
    return out
end

const BL_THERMAL = [
    "n_up" => n_(:up), "n_up n_dn" => n_(:up) * n_(:dn),
    "hop0" => dd_(:up) * cop(Mode(0, :up)) + cdag(Mode(0, :up)) * d_(:up),
    "n_f0" => num(Mode(0, :dn)),
]

function bl_thermal(run, f)
    out = Dict{String,Any}()
    for (k, e) in BL_THERMAL
        out[k] = thermal_average(run, f, e)
    end
    out["product d d† n"] = thermal_average_product(run, f, [d_(:up), dd_(:up), n_(:dn)])
    th = fdm_thermodynamics(run, f)
    for k in keys(th)
        out["thermo/$k"] = th[k]
    end
    return out
end

# a histogram as a sorted bin-index matrix (nnz × ℓ-1, Int16) and its values
function packbins(pairs)
    ks = sort!([Tuple(k) for (k, v) in pairs if !iszero(v)])
    d = Dict(Tuple(k) => v for (k, v) in pairs)
    L = isempty(ks) ? 0 : length(ks[1])
    K = zeros(Int16, length(ks), L)
    for (i, k) in enumerate(ks), j in 1:L
        K[i, j] = k[j]
    end
    return Dict("keys" => K, "vals" => [d[k] for k in ks])
end
unpackbins(p) = Dict(Tuple(Int.(p["keys"][i, :])) => p["vals"][i] for i in eachindex(p["vals"]))

function bl_psf_record(r)
    d = Dict{String,Any}("total" => r.total, "sum_rule" => get(r.meta, :sum_rule, NaN), "mode" => string(r.mode))
    if r.mode === :binned
        d["bins"] = packbins(nonzero_bins(r))
        if r.absdata !== nothing
            a = r.absdata
            d["absbins"] = packbins(a isa AbstractDict ? collect(a) : [Tuple(I) => a[I] for I in CartesianIndices(a)])
        end
    else
        d["clusters"] = cluster_terms(r.positions, r.weights)
    end
    return d
end

"""
    cluster_terms(P, w; tol = 1e-10) -> (positions, weights)

Exact-mode terms summed per position (positions equal within `tol` in every
component): the per-term weights depend on the eigenbasis inside degenerate
subspaces, their sums per position do not.
"""
function cluster_terms(P::AbstractMatrix, w::AbstractVector; tol = 1e-10)
    L = size(P, 1)
    isempty(w) && return (zeros(L, 0), eltype(w)[])
    cols = sortperm(collect(eachcol(P)); by = Tuple)
    pos = Vector{Vector{Float64}}(); ws = eltype(w)[]
    for j in cols
        p = P[:, j]
        if !isempty(pos) && maximum(abs.(p .- pos[end])) ≤ tol
            ws[end] += w[j]
        else
            push!(pos, p); push!(ws, w[j])
        end
    end
    return (reduce(hcat, pos), ws)
end

function bl_psfs(name, run, f)
    out = Dict{String,Any}()
    q = comm(d_(:up), 0.5 * n_(:up) * n_(:dn))          # composite q↑ = [d↑, H_int]
    sets = Any[("2p_d", (d_(:up), dd_(:up))), ("2p_q", (q, q'))]
    name in ("trunc", "parity", "untrunc") && push!(sets, ("3p", (q, dd_(:up), n_(:dn))))
    name in ("trunc", "untrunc") && push!(sets, ("4p", (d_(:up), dd_(:up), d_(:dn), dd_(:dn))))
    for (lab, ops) in sets
        for (p, r) in psfs(run, f, ops; grid = lab == "4p" ? BL_GRID4 : BL_GRID)
            out["$lab/$(join(p))"] = bl_psf_record(r)
        end
    end
    if name == "trunc"
        # complex operator (no complex model among the built-ins)
        c = (1.0 + 0.5im) * d_(:up)
        out["2p_complex"] = bl_psf_record(psf(run, f, (c, c'); grid = BL_GRID))
        out["3p_left"] = bl_psf_record(psf(run, f, (d_(:up), n_(:dn), dd_(:up)); grid = BL_GRID, slicing = :left))
        rec = SingularValueRecorder()
        psf(run, f, (d_(:up), dd_(:up), n_(:dn)); grid = BL_GRID, schedule = :breadth, hooks = [rec])
        sv = Dict{Any,Any}()
        for (n, v) in rec.data, (born, depth, label, pos, k, s) in v
            sv[(n, born, depth, label, pos, k)] = s
        end
        out["svals_3p"] = sv
    end
    if name == "field"
        out["3p_abs"] = bl_psf_record(psf(run, f, (d_(:up), dd_(:up), n_(:dn)); grid = BL_GRID, abs = true))
    end
    if name == "untrunc"
        for (lab, ops) in (("exact2p", (d_(:up), dd_(:up))), ("exact3p", (d_(:up), dd_(:up), n_(:dn))))
            out[lab] = bl_psf_record(psf(run, f, ops; exact = true, max_terms = 20_000_000))
        end
    elseif name == "trunc"
        out["exact2p"] = bl_psf_record(psf(run, f, (d_(:up), dd_(:up)); exact = true))
    end
    return out
end

"""    compute_baseline() -> Dict — every recorded quantity of every case."""
function compute_baseline(; verbose = true)
    out = Dict{String,Any}()
    for (name, (m, p, T)) in bl_cases()
        t0 = time()
        run = run_nrg(m, p); f = fdm(run; T = T)
        out[name] = Dict{String,Any}("shells" => bl_shells(run), "fdm" => bl_fdm(run, f),
                                     "thermal" => bl_thermal(run, f), "psf" => bl_psfs(name, run, f))
        verbose && (println("baseline case $name: $(round(time() - t0; digits = 1)) s"); flush(stdout))
    end
    return out
end
