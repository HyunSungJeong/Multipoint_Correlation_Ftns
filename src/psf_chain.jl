# I1, I2 of the AIM task: PSFs that scale.
#
# The dense `psf` of psf.jl visits all n^ℓ eigenstate tuples. Eq. (28) only
# needs cycles 1 → 2 → ⋯ → ℓ → 1 along nonzero matrix elements, and for a
# charge- and spin-conserving operator those are confined to one (N, Sz) sector
# per row. Here the rotated, otol-filtered operators are stored as row-wise
# adjacency lists and the cycles are walked directly: cost ~ n d^{ℓ-1} with d
# the nonzeros per row, instead of n^ℓ. `psf` itself is left untouched as the
# reference implementation; the tests compare the two peak by peak.
#
# Merged peaks are accumulated in a Dict keyed on the position rounded to
# TOL_DEG cells (I2), with a neighbour pass that re-joins the rare pair split
# across a cell boundary, instead of the pairwise search of merge_peaks.

"""
    default_psf_method(sp) -> Symbol

`:dense` (the reference [`psf`](@ref)) for Fock spaces of up to 16 states — the
atom, the dimer and the one-site AIM, whose results are thereby unchanged —
and the chain walk `:chain` above that.
"""
default_psf_method(sp::Spectrum) = length(sp.E) <= 16 ? :dense : :chain

# ---------------------------------------------------------------------------
# I2: peak accumulator
# ---------------------------------------------------------------------------

"""
    PeakAccumulator{N}()

Merges PSF peaks with `N` cumulative positions on the fly. `add!(acc, pos, w)`
adds weight `w` at `pos`; peaks whose positions agree to `TOL_DEG` in every
coordinate end up in one entry, which keeps the position of its first peak.
Lookup is a `Dict` on `round.(Int, pos ./ TOL_DEG)`; see [`merged_peaks`](@ref)
for how peaks split across a cell boundary are re-joined.
"""
struct PeakAccumulator{N}
    index::Dict{NTuple{N,Int},Int}
    pos::Vector{NTuple{N,Float64}}
    w::Vector{ComplexF64}
end
PeakAccumulator{N}() where {N} = PeakAccumulator{N}(Dict{NTuple{N,Int},Int}(), NTuple{N,Float64}[], ComplexF64[])

@inline _cell(pos::NTuple{N,Float64}) where {N} = ntuple(i -> round(Int, pos[i] / TOL_DEG), Val(N))

@inline function add!(acc::PeakAccumulator{N}, pos::NTuple{N,Float64}, w::Number) where {N}
    k = _cell(pos)
    i = get(acc.index, k, 0)
    if i == 0
        push!(acc.pos, pos); push!(acc.w, w)
        acc.index[k] = length(acc.w)
    else
        acc.w[i] += w
    end
    return acc
end

"""
    merged_peaks(acc; wtol = TOL_WEIGHT) -> (positions, weights)

The merged peaks, in first-seen order, with `|weight| ≤ wtol` dropped. Before
that, a neighbour pass re-joins entries whose cells differ by at most one in
every coordinate and whose positions agree to `TOL_DEG`: genuine degeneracies
are exact to ~1e-14, so two such entries can only be one peak that straddled a
cell boundary. (The tests assert afterwards that no two surviving positions are
closer than `10 TOL_DEG` in every coordinate.)
"""
function merged_peaks(acc::PeakAccumulator{N}; wtol::Real = TOL_WEIGHT) where {N}
    alive = trues(length(acc.w))
    w = copy(acc.w)
    offsets = [δ for δ in Iterators.product(ntuple(_ -> -1:1, Val(N))...) if any(!=(0), δ)]
    for (k, i) in acc.index
        alive[i] || continue
        for δ in offsets
            j = get(acc.index, ntuple(t -> k[t] + δ[t], Val(N)), 0)
            (j == 0 || j == i || !alive[j]) && continue
            all(abs(acc.pos[i][t] - acc.pos[j][t]) ≤ TOL_DEG for t in 1:N) || continue
            a, b = minmax(i, j)                     # keep the first-seen position
            w[a] += w[b]; alive[b] = false
        end
    end
    keep = [i for i in eachindex(w) if alive[i] && abs(w[i]) > wtol]
    return acc.pos[keep], w[keep]
end

"""
    merge_peaks(terms; tol = TOL_DEG, wtol = TOL_WEIGHT) -> (positions, weights)

Merge PSF peaks through a [`PeakAccumulator`](@ref) (I2). Same contract and
output order as the pairwise [`merge_peaks_pairwise`](@ref), which it replaces:
peaks whose positions agree to `tol` in every coordinate are summed, merged
weights `≤ wtol` in modulus are dropped.
"""
function merge_peaks(terms::Vector{PSFTerm}; tol::Real = TOL_DEG, wtol::Real = TOL_WEIGHT)
    tol == TOL_DEG || return merge_peaks_pairwise(terms; tol = tol, wtol = wtol)
    isempty(terms) && return Vector{Float64}[], ComplexF64[]
    N = length(terms[1].position)
    acc = PeakAccumulator{N}()
    for t in terms
        add!(acc, ntuple(i -> t.position[i], N), t.weight)
    end
    pos, w = merged_peaks(acc; wtol = wtol)
    return [collect(p) for p in pos], w
end

# ---------------------------------------------------------------------------
# I1: sparse operators and chain walks
# ---------------------------------------------------------------------------

"""
    SparseOp

A rotated, `otol`-filtered operator: the dense matrix (for closing a cycle,
`(O_ℓ̄)_{s_ℓ s_1}`) and its nonzeros as row-wise adjacency lists.
"""
struct SparseOp
    dense::Matrix{ComplexF64}
    rows::Vector{Vector{Tuple{Int,ComplexF64}}}
end

"""    sparse_op(sp, O; otol) — rotate `O` to the eigenbasis, zero `|O_ab| ≤ otol`, index its rows."""
function sparse_op(sp::Spectrum, O::AbstractMatrix; otol::Real = 0.0)
    R = ComplexF64.(to_eigenbasis(sp, O))
    otol > 0 && (R[abs.(R) .<= otol] .= 0)
    rows = [[(j, R[i, j]) for j in axes(R, 2) if R[i, j] != 0] for i in axes(R, 1)]
    return SparseOp(R, rows)
end

"""
    chain_walk4!(emit, sp, A, B, C, D)

Every nonzero cycle of Eq. (28) for ℓ = 4 and the operator order `(A, B, C, D)`
([`SparseOp`](@ref)s): for each `s₁` with `ρ > 0`, each nonzero `s₂` of row `s₁`
of `A`, … , closing with `D[s₄, s₁]`. Calls `emit(pos, w)` with the cumulative
positions `(E₂-E₁, E₃-E₁, E₄-E₁)` and the weight `ρ₁ A B C D`, multiplied in the
same order as the dense [`psf`](@ref), and returns the number of cycles.
"""
function chain_walk4!(emit::F, sp::Spectrum, A::SparseOp, B::SparseOp, C::SparseOp,
                      D::SparseOp) where {F}
    E, ρ = sp.E, sp.ρ
    n = 0
    @inbounds for s1 in eachindex(ρ)
        r = ρ[s1]
        r > 0 || continue
        w1 = ComplexF64(r)
        for (s2, a) in A.rows[s1]
            w2 = w1 * a
            iszero(w2) && continue
            for (s3, b) in B.rows[s2]
                w3 = w2 * b
                iszero(w3) && continue
                for (s4, c) in C.rows[s3]
                    w4 = w3 * c
                    iszero(w4) && continue
                    w = w4 * D.dense[s4, s1]
                    iszero(w) && continue
                    emit((E[s2] - E[s1], E[s3] - E[s1], E[s4] - E[s1]), w)
                    n += 1
                end
            end
        end
    end
    return n
end

"""    chain_walk2!(emit, sp, A, B) — as [`chain_walk4!`](@ref) for ℓ = 2: position `E₂ - E₁`, weight `ρ₁ A B`."""
function chain_walk2!(emit::F, sp::Spectrum, A::SparseOp, B::SparseOp) where {F}
    E, ρ = sp.E, sp.ρ
    n = 0
    @inbounds for s1 in eachindex(ρ)
        r = ρ[s1]
        r > 0 || continue
        for (s2, a) in A.rows[s1]
            w = ComplexF64(r) * a * B.dense[s2, s1]
            iszero(w) && continue
            emit((E[s2] - E[s1],), w)
            n += 1
        end
    end
    return n
end

"""
    psf_chain(sp, Os; otol = 0.0) -> Vector{PSFTerm}

Chain-walk version of [`psf`](@ref) for any ℓ, returning the same unmerged
terms (in a different order). Used by `permuted_psfs(...; method = :chain)` and
by the tests; the tables use the allocation-free [`chain_walk4!`](@ref).
"""
function psf_chain(sp::Spectrum, Os::Tuple; otol::Real = 0.0)
    ℓ = length(Os)
    ℓ ≥ 2 || throw(ArgumentError("Eq. (28) needs at least ℓ = 2 operators, got $ℓ"))
    ops = [sparse_op(sp, O; otol = otol) for O in Os]
    E, ρ = sp.E, sp.ρ
    terms = PSFTerm[]
    st = zeros(Int, ℓ)
    function rec(i, w)
        if i == ℓ
            wf = w * ops[ℓ].dense[st[ℓ], st[1]]
            iszero(wf) && return
            push!(terms, PSFTerm([E[st[k + 1]] - E[st[1]] for k in 1:(ℓ - 1)], wf, copy(st)))
            return
        end
        for (s, a) in ops[i].rows[st[i]]
            wn = w * a
            iszero(wn) && continue
            st[i + 1] = s
            rec(i + 1, wn)
        end
    end
    for s1 in eachindex(ρ)
        ρ[s1] > 0 || continue
        st[1] = s1
        rec(1, ComplexF64(ρ[s1]))
    end
    return terms
end

"""
    ChainStats

Per-permutation bookkeeping of [`psf4_tables_chain`](@ref): surviving cycles of
`S` and merged peaks after the part is taken.
"""
struct ChainStats
    cycles::Vector{Int}
    peaks::Vector{Int}
end

"""
    psf4_tables_chain(sp, Os; part = :connected, is_fermionic, otol = 0.0, wtol = TOL_WEIGHT)
        -> (Vector{PSF4}, ChainStats)

The 24 permuted, merged 4p PSFs of `Os` written directly into [`PSF4`](@ref)
storage: the four operators are rotated and indexed once, each permutation is
walked with [`chain_walk4!`](@ref) straight into a [`PeakAccumulator`](@ref),
and for `part = :connected` (`:disconnected`) the Eq. (31) pairings built from
chain-walked 2p PSFs are added with weight −1 (+1) before merging.
"""
function psf4_tables_chain(sp::Spectrum, Os::Tuple; part::Symbol = :connected,
                           is_fermionic::AbstractVector{Bool} = fill(true, 4),
                           otol::Real = 0.0, wtol::Real = TOL_WEIGHT)
    length(Os) == 4 || throw(ArgumentError("PSF4 is for ℓ = 4"))
    part in (:full, :connected, :disconnected) ||
        throw(ArgumentError("part must be :full, :connected or :disconnected, got $part"))
    ops = [sparse_op(sp, O; otol = otol) for O in Os]
    # merged 2p PSFs for every ordered pair, for the Eq. (31) pairings
    p2 = Dict{Tuple{Int,Int},Tuple{Vector{Float64},Vector{ComplexF64}}}()
    if part !== :full
        for i in 1:4, j in 1:4
            i == j && continue
            acc = PeakAccumulator{1}()
            chain_walk2!((pos, w) -> add!(acc, pos, w), sp, ops[i], ops[j])
            pos, w = merged_peaks(acc; wtol = 0.0)
            p2[(i, j)] = ([x[1] for x in pos], w)
        end
    end
    out = PSF4[]
    stats = ChainStats(Int[], Int[])
    for p in all_permutations(4)
        ζ = permutation_sign(p, is_fermionic)
        acc = PeakAccumulator{3}()
        ncyc = part === :disconnected ? 0 :
               chain_walk4!((pos, w) -> add!(acc, pos, w), sp, ops[p[1]], ops[p[2]], ops[p[3]], ops[p[4]])
        if part !== :full
            s = part === :connected ? -1 : 1
            ζ23 = (is_fermionic[p[2]] && is_fermionic[p[3]]) ? -1 : 1
            a1, b1 = p2[(p[1], p[2])]; a2, b2 = p2[(p[3], p[4])]       # (1̄2̄)(3̄4̄): (a, 0, c)
            for x in eachindex(a1), y in eachindex(a2)
                add!(acc, (a1[x], 0.0, a2[y]), s * b1[x] * b2[y])
            end
            a1, b1 = p2[(p[1], p[3])]; a2, b2 = p2[(p[2], p[4])]       # (1̄3̄)(2̄4̄): (a, a+b, b)
            for x in eachindex(a1), y in eachindex(a2)
                add!(acc, (a1[x], a1[x] + a2[y], a2[y]), s * ζ23 * b1[x] * b2[y])
            end
            a1, b1 = p2[(p[1], p[4])]; a2, b2 = p2[(p[2], p[3])]       # (1̄4̄)(2̄3̄): (a, a+b, a)
            for x in eachindex(a1), y in eachindex(a2)
                add!(acc, (a1[x], a1[x] + a2[y], a1[x]), s * b1[x] * b2[y])
            end
        end
        pos, w = merged_peaks(acc; wtol = wtol)
        push!(out, PSF4(Tuple(p), ζ, pos, w))
        push!(stats.cycles, ncyc); push!(stats.peaks, length(w))
    end
    return out, stats
end

"""    psf2_tables_chain(sp, A, B; is_fermionic, otol) -> Vector{PSF2} — chain-walk [`psf2_tables`](@ref)."""
function psf2_tables_chain(sp::Spectrum, A::AbstractMatrix, B::AbstractMatrix;
                           is_fermionic::AbstractVector{Bool} = [true, true], otol::Real = 0.0,
                           wtol::Real = TOL_WEIGHT)
    ops = (sparse_op(sp, A; otol = otol), sparse_op(sp, B; otol = otol))
    out = PSF2[]
    for p in all_permutations(2)
        acc = PeakAccumulator{1}()
        chain_walk2!((pos, w) -> add!(acc, pos, w), sp, ops[p[1]], ops[p[2]])
        pos, w = merged_peaks(acc; wtol = wtol)
        push!(out, PSF2(Tuple(p), permutation_sign(p, is_fermionic), [x[1] for x in pos], w))
    end
    return out
end
