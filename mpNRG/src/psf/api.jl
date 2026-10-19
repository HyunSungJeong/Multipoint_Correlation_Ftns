# Public PSF interface (task sheet Sec. 5.7, 4.4, 4.6).

"""
    PSFResult

One partial spectral function `S[O¹, …, O^ℓ](ε₁, …, ε_{ℓ-1})` for operators given
**in permuted order** (Kugler Eq. (28) / Lee Eq. (5): the cycle of eigenstates
`(1, …, ℓ)` carries weight `ρ₁ ∏ᵢ (Oⁱ)_{i,i+1}` at `εᵢ = E_{i+1} - E₁`).
Fermionic permutation signs ζ_p belong to the kernels and are **not** applied.

Fields:
  - `ops`: operator names; `shifts`: their symmetry shifts; `ℓ`;
  - `mode`: `:binned` or `:exact`; `grid`: the [`LogGrid`](@ref) (binned);
  - `data`: the histogram (binned): a dense `Array{T,ℓ-1}` indexed by bin
    indices, or a `Dict{NTuple{ℓ-1,Int},T}` of nonzero bins (sparse); `absdata`
    the histogram of `|weight|` if requested;
  - `positions` (`(ℓ-1) × nterms`), `weights`, `labels` (exact mode: unbinned
    Lehmann terms with `(shell, depth, sector-string code)`);
  - `total`: the total weight; `meta`: the sum-rule reference `⟨O¹⋯O^ℓ⟩` from
    the FDM, pruning information, cost counters, timing, clamping counts.
"""
struct PSFResult{T,L}
    ops::Vector{String}
    shifts::Vector{Any}
    ℓ::Int
    mode::Symbol
    grid::Union{Nothing,LogGrid}
    data::Any
    absdata::Any
    positions::Matrix{Float64}
    weights::Vector{T}
    labels::Vector{NTuple{3,Int}}
    total::T
    meta::Dict{Symbol,Any}
end

npoint(r::PSFResult) = r.ℓ
total_weight(r::PSFResult) = r.total
issparse_hist(r::PSFResult) = r.data isa AbstractDict

"""
    histogram(r; dense = true) -> Array

The binned histogram as a dense array (built from the sparse form if needed;
for 4p on a fine grid this can be large — check `nbins(r.grid)^3` first).
"""
function histogram(r::PSFResult{T,L}; abs::Bool = false) where {T,L}
    r.mode === :binned || throw(ArgumentError("histogram needs a binned result"))
    d = abs ? r.absdata : r.data
    d === nothing && throw(ArgumentError("no |weight| histogram was requested (use abs = true)"))
    d isa AbstractArray && return d
    nb = nbins(r.grid)
    A = zeros(abs ? Float64 : T, ntuple(_ -> nb, L))
    for (k, v) in d
        A[k...] += v
    end
    return A
end

"""
    nonzero_bins(r) -> Vector{Pair{NTuple,T}}

The nonzero bins of a binned result as `bin-index tuple => weight`.
"""
function nonzero_bins(r::PSFResult{T,L}) where {T,L}
    r.mode === :binned || throw(ArgumentError("needs a binned result"))
    d = r.data
    d isa AbstractDict && return [k => v for (k, v) in d if !iszero(v)]
    return [Tuple(I) => d[I] for I in CartesianIndices(d) if !iszero(d[I])]
end

"""
    bin_exact(r, grid; sparse = r.ℓ ≥ 4) -> Dict or Array

Bin the unbinned Lehmann terms of an exact-mode result on `grid` (for
comparisons with binned runs and with binned reference data).
"""
function bin_exact(r::PSFResult{T,L}, g::LogGrid) where {T,L}
    r.mode === :exact || throw(ArgumentError("needs an exact-mode result"))
    d = Dict{NTuple{L,Int},T}()
    for j in eachindex(r.weights)
        k = ntuple(i -> binindex(g, r.positions[i, j])[1], Val(L))
        d[k] = get(d, k, zero(T)) + r.weights[j]
    end
    return d
end

"""
    bin_terms(positions, weights, grid) -> Dict

Bin arbitrary Lehmann terms (e.g. from [`ed_psf`](@ref)) on `grid`.
"""
function bin_terms(P::AbstractMatrix, w::AbstractVector{T}, g::LogGrid) where {T}
    L = size(P, 1)
    d = Dict{NTuple{L,Int},T}()
    for j in eachindex(w)
        k = ntuple(i -> binindex(g, P[i, j])[1], L)
        d[k] = get(d, k, zero(T)) + w[j]
    end
    return d
end

_opname(o::OpExpr) = string(o)
_opname(o::LocalOperator) = o.name

"""
    psf(run, f, ops; grid = nothing, exact = false, abs = false, sparse = ℓ ≥ 4,
        slicing = :lee, schedule = :depth, hooks = [], max_terms = 2_000_000) -> PSFResult

The ℓp PSF (ℓ = 2, 3, 4) of homogeneous local operators `ops` (a tuple of
`OpExpr` or [`LocalOperator`](@ref)) **in permuted order**, from run `run` and
density matrix `f = fdm(run; T)`.

  - Binned (default): all frequencies on the shared `grid`. Approximations:
    the NRG approximation (Lee Eq. (19)) in the energies, binning (Lee
    Sec. V B), and slicing (the sliced frequencies keep the resolution of the
    shell where they are created, Lee p. 14; for the 4p KKDK term ε₂ is
    assembled as `centre(ε₂-ε₁) + ε₁`, Lee Eq. (71b)).
  - `exact = true`: unbinned Lehmann terms with exact NRG energy differences
    (small runs; hard guard `max_terms`; `wtol` drops terms with
    `|weight| ≤ wtol`, e.g. the roundoff remnants of symmetry-forbidden
    matrix elements when fewer symmetries are declared; the dropped total
    `|weight|` is recorded in `meta[:dropped_abs_weight]`).
  - `abs = true`: also the histogram of `|weight|`, where a weight is one
    `(shell, sector string, s₁, s_ℓ, slice)` contribution, the finest
    decomposition the sliced evaluation provides.
  - `slicing = :left`: an alternative trace form for diagnostics (V6): every
    composite is sliced relative to its left endpoint.
  - `schedule`: `:depth` or `:breadth` (see [`run_tasks!`](@ref)); `hooks`:
    per-shell callbacks (`hooks/`).

A call whose total shift `Δ₁ ⊕ ⋯ ⊕ Δ_ℓ` is not neutral returns an exact zero,
recorded as `meta[:pruned] = true` (the trace forces a neutral total shift).
"""
function psf(run::NRGRun{S,Tr}, f::FDM, ops::Tuple; grid::Union{Nothing,LogGrid} = nothing,
             exact::Bool = false, abs::Bool = false, sparse::Union{Nothing,Bool} = nothing,
             slicing::Symbol = :lee, schedule::Symbol = :depth, hooks = Any[], max_terms::Integer = 2_000_000,
             wtol::Real = 0.0) where {S,Tr}
    ℓ = length(ops)
    2 ≤ ℓ ≤ 4 || throw(ArgumentError("ℓ = $ℓ: only 2, 3 and 4 operators are supported"))
    L = ℓ - 1
    exact || grid !== nothing || throw(ArgumentError("binned mode needs a grid (or exact = true)"))
    slicing in (:lee, :left) || throw(ArgumentError("slicing must be :lee or :left"))
    t0 = time()
    lops = [local_operator(run, o) for o in ops]
    for o in lops
        ishomogeneous(o) || throw(ArgumentError(
            "operator $(o.name) is inhomogeneous (shifts $(collect(keys(o.comps)))); use psf_multilinear"))
    end
    shifts = Any[opshift(o) for o in lops]
    names = [o.name for o in lops]
    exprs = [isempty(o.comps) ? OpExpr{Float64}() : first(values(o.comps)) for o in lops]
    T = promote_type(Tr, (eltype(e) <: Real ? Float64 : ComplexF64 for e in exprs)...)
    meta = Dict{Symbol,Any}(:slicing => slicing, :schedule => schedule, :T => f.temp,
                            :shifts => shifts, :n0 => run.n0)
    tot = foldl(fuse, shifts; init = neutral(S))
    usesparse = sparse === nothing ? L ≥ 3 : sparse
    if !isneutral(tot) || any(e -> isempty(e.terms), exprs)
        meta[:pruned] = true
        meta[:pruned_reason] = isneutral(tot) ? "zero operator" : "total shift $tot ≠ neutral"
        data = exact ? nothing : (usesparse ? Dict{NTuple{L,Int},T}() : zeros(T, ntuple(_ -> nbins(grid), L)))
        return PSFResult{T,L}(names, shifts, ℓ, exact ? :exact : :binned, grid, data, nothing,
                              zeros(L, 0), T[], NTuple{3,Int}[], zero(T), meta)
    end
    meta[:pruned] = false
    meta[:sum_rule] = thermal_average(run, f, reduce(*, exprs))
    nstart = maximum(o.nop for o in lops)
    V = exact ? Float64 : Int
    F0 = ntuple(_ -> zero(V), L)
    tags0 = ntuple(_ -> TAG_NONE, L)
    root_ops = [EOp{T,S,L,V}(sh, tags0, SlicedTensor(Dict(F0 => convert_eltype(T, project(run, e, nstart)))), e, -2)
                for (sh, e) in zip(shifts, exprs)]
    root = PSFTask{T,S,L,V}(root_ops, collect(1:ℓ), nstart, false, 0, -2, "root")
    mode = exact ? ExactMode() : BinnedMode(grid)
    acc = if exact
        ExactAcc{T,L}(NTuple{L,Float64}[], T[], NTuple{3,Int}[], max_terms, zero(T), Float64(wtol), 0.0)
    else
        nb = nbins(grid)
        # accumulate densely whenever the histogram fits (much faster than a
        # dictionary); a sparse result is extracted at the end
        densacc = !usesparse || nb^L ≤ 2^25
        BinAcc{T,L}(densacc ? zeros(T, ntuple(_ -> nb, L)) : nothing, Dict{NTuple{L,Int},T}(),
                    (abs && densacc) ? zeros(Float64, ntuple(_ -> nb, L)) : nothing,
                    Dict{NTuple{L,Int},Float64}(), abs, zero(T))
    end
    ctx = EngineCtx(run, f, mode, acc, slicing, collect(Any, hooks), Dict{Symbol,Any}())
    run_tasks!(ctx, root; schedule = schedule)
    meta[:cost] = copy(ctx.stats)
    meta[:time] = time() - t0
    if exact
        meta[:dropped_abs_weight] = acc.dropped
        P = zeros(L, length(acc.w))
        for (j, p) in enumerate(acc.pos)
            P[:, j] .= p
        end
        return PSFResult{T,L}(names, shifts, ℓ, :exact, grid, nothing, nothing, P, acc.w, acc.label, acc.total, meta)
    end
    meta[:nclamped] = mode.nclamped
    if usesparse && acc.dense !== nothing
        data = Dict{NTuple{L,Int},T}(Tuple(I) => acc.dense[I] for I in CartesianIndices(acc.dense) if !iszero(acc.dense[I]))
        absd = abs ? Dict{NTuple{L,Int},Float64}(Tuple(I) => acc.absdense[I] for I in CartesianIndices(acc.absdense)
                                                  if !iszero(acc.absdense[I])) : nothing
    else
        data = usesparse ? acc.sparse : acc.dense
        absd = abs ? (usesparse ? acc.abssparse : acc.absdense) : nothing
    end
    return PSFResult{T,L}(names, shifts, ℓ, :binned, grid, data, absd, zeros(L, 0), T[], NTuple{3,Int}[], acc.total, meta)
end

"""
    all_perms(ℓ) -> Vector{Vector{Int}}

All `ℓ!` permutations of `1:ℓ` in lexicographic order.
"""
function all_perms(ℓ::Integer)
    ℓ == 1 && return [[1]]
    out = Vector{Int}[]
    for p in all_perms(ℓ - 1), pos in 1:ℓ
        q = copy(p); insert!(q, pos, ℓ)
        push!(out, q)
    end
    return sort!(out)
end

"""
    psfs(run, f, ops; grid, perms = :all, kwargs...) -> Dict{Vector{Int},PSFResult}

PSFs of all requested permutations `p` (`ops[p]` in permuted order), on one
shared grid. Permutations whose total shift is not neutral are pruned before
any NRG work (`meta[:pruned]`); the list is in `result[:pruned]`.
"""
function psfs(run::NRGRun, f::FDM, ops::Tuple; perms = :all, kwargs...)
    ℓ = length(ops)
    ps = perms === :all ? all_perms(ℓ) : collect(perms)
    out = Dict{Vector{Int},PSFResult}()
    for p in ps
        out[p] = psf(run, f, Tuple(ops[i] for i in p); kwargs...)
    end
    return out
end

"""
    psf_multilinear(run, f, ops; kwargs...) -> PSFResult

PSF of possibly inhomogeneous operators (Sec. 5.2.3): every operator is split
into homogeneous components and the multilinear sum over component tuples is
evaluated; tuples with a non-neutral total shift vanish exactly and are
skipped (listed in `meta[:skipped]`).
"""
function psf_multilinear(run::NRGRun{S}, f::FDM, ops::Tuple; kwargs...) where {S}
    lops = [local_operator(run, o) for o in ops]
    comps = [collect(o.comps) for o in lops]
    results = PSFResult[]; skipped = Any[]
    for tup in Iterators.product(comps...)
        shifts = [c[1] for c in tup]
        if !isneutral(foldl(fuse, shifts; init = neutral(S)))
            push!(skipped, shifts); continue
        end
        push!(results, psf(run, f, Tuple(c[2] for c in tup); kwargs...))
    end
    isempty(results) && return psf(run, f, Tuple(first(c)[2] for c in comps); kwargs...)
    r = reduce(add_psf, results)
    r.meta[:skipped] = skipped
    r.meta[:ncomponent_tuples] = length(results)
    return r
end

"""
    add_psf(a, b) -> PSFResult

Sum of two PSFs on the same grid (refuses different grids or modes).
"""
function add_psf(a::PSFResult{TA,L}, b::PSFResult{TB,L}) where {TA,TB,L}
    a.mode == b.mode || throw(ArgumentError("cannot add binned and exact PSFs"))
    T = promote_type(TA, TB)
    if a.mode === :binned
        a.grid == b.grid || throw(ArgumentError("cannot add PSFs on different grids"))
        data = if a.data isa AbstractDict && b.data isa AbstractDict
            d = Dict{NTuple{L,Int},T}(a.data)
            for (k, v) in b.data
                d[k] = get(d, k, zero(T)) + v
            end
            d
        else
            histogram(a) .+ histogram(b)
        end
        return PSFResult{T,L}(a.ops, a.shifts, a.ℓ, :binned, a.grid, data, nothing, zeros(L, 0), T[],
                              NTuple{3,Int}[], a.total + b.total, merge(copy(a.meta), Dict(:summed => true)))
    end
    return PSFResult{T,L}(a.ops, a.shifts, a.ℓ, :exact, nothing, nothing, nothing, hcat(a.positions, b.positions),
                          vcat(T.(a.weights), T.(b.weights)), vcat(a.labels, b.labels), a.total + b.total,
                          merge(copy(a.meta), Dict(:summed => true)))
end
