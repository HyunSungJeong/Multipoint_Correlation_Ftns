# The PSF engine for ℓ = 2, 3, 4 (task sheet Sec. 5.7; Lee Sec. V).
#
# One routine evaluates the ℓp PSF of *any* tuple of carried operators,
# irrespective of how they were generated and whether they are sliced
# (Lee p. 18). For each shell n it expands the operator product
# shell-diagonally, Lee Eqs. (49)-(51), and sorts the sector strings
# X = (X_1, …, X_m) ≠ (K, …, K) of the m operators into two classes:
#
#   * adequately refined (X_1 = D, or X_1 = K and no other K): evaluated as a
#     shell-n trace with ρ̄^n_{X_1X_1}, all frequencies binned
#     [Lee Eqs. (57)-(59) for ℓ = 2; the 5 D X̄ X̂, KDD terms of Eq. (62);
#      the 9 D X̄ X̂ X̃, KDDD terms of Eq. (68)];
#   * needing refinement (X_1 = K and at least one more K): the operators
#     between consecutive K states are multiplied into composite KK operators
#     whose internal (D) frequencies are *sliced* (Lee Eq. (60)), and the PSF of
#     these composites — with fewer operators — is computed by the same
#     routine over the later shells n̄ > n
#     [3p: KKD, KDK via 2p PSFs of composites, Eqs. (64b), (65b), (66);
#      4p: KKDD, KDKD, KDDK via 2p PSFs, Eqs. (70a)-(70c);
#      KKKD, KKDK, KDKK via 3p PSFs, Eqs. (71a)-(71c), which recurse once more].
#
# The refinements therefore arise from nested calls, not from hand-written
# routines per diagram. Slicing references (which state a sliced frequency is
# measured from) follow Lee: a composite that starts at state 1 (the state
# carrying ρ) is sliced from the left, one that ends at state 1 from the right,
# both giving the original ε_i directly (Lee p. 15: associate ε_i rather than
# -ε_i with operators where possible); a composite between two other K states
# (only KKDK for ℓ = 4) is sliced relative to its left endpoint, giving
# ε_2 - ε_1 as in Lee Eq. (71b), and ε_2 is restored by adding the refined ε_1.
#
# Operators are TNOps tensors (`SlicedTensor`s of rank-2 `BlockTensor`s with
# K/D-tagged shell legs, nrg/tensors.jl). The row-by-row spectral bookkeeping
# (binning each row's energy differences, grouping states by bin key) is dense
# code on single symmetry blocks; these are the
#
#   declared dense kernels (TNOPS_REFACTOR.md Sec. 4.4):
#     shell_trace!, _trace_step!, _composite_left, _comp_left_step!, _addrow!,
#     _composite_right, _comp_right_step!, _addcol!
#
# They see operators only through `block_view(O, q, X, X̄)`: the dense block
# with columns in the X̄ states of label q and rows in the X states of its
# target label (ρ̄_KK likewise). Nothing else in mpNRG/src touches blocks
# (a test in test_nrg_tensors.jl checks this).

# ---------------------------------------------------------------- modes

"""
    BinnedMode(grid) / ExactMode()

How frequencies are recorded: `BinnedMode` moves every frequency to its bin of
the [`LogGrid`](@ref) (binning and slicing, Lee Sec. V B); `ExactMode` keeps the
exact NRG energy differences (unbinned Lehmann terms, small runs only).
"""
abstract type FreqMode end
mutable struct BinnedMode <: FreqMode
    grid::LogGrid
    nclamped::Int
end
BinnedMode(g::LogGrid) = BinnedMode(g, 0)
struct ExactMode <: FreqMode end

keyval(::BinnedMode) = Int
keyval(::ExactMode) = Float64
@inline function freqkey(b::BinnedMode, x::Float64)
    i, c = binindex(b.grid, x)
    c && (b.nclamped += 1)
    return i
end
@inline freqkey(::ExactMode, x::Float64) = x
@inline centreval(b::BinnedMode, k::Int) = b.grid.centres[k]
@inline centreval(::ExactMode, k::Float64) = k

# ---------------------------------------------------------------- carried operators

const TAG_NONE = Int8(0)
const TAG_DIRECT = Int8(1)    # value is E_k - E_1 (an original frequency)
const TAG_LEFT = Int8(2)      # value is E_k - E_left (relative to the operator's left endpoint)

"""
    EOp

An operator carried by the engine at shell `n`: its symmetry shift, and a
`SlicedTensor` of shell operators (TNOps rank-2 tensors on the shell leg, or
on the kept leg for composites). Slice keys are `NTuple{L,V}` over the `L = ℓ-1`
original frequencies: entry `g` holds the sliced value (bin index or exact
energy) of frequency `g` if `tags[g] ≠ TAG_NONE`. `orig_left` is the original
index of its left endpoint state. Top-level local operators carry their
expression, so that their shell projections come from the cache.
"""
mutable struct EOp{T,S,L,V}
    shift::S
    tags::NTuple{L,Int8}
    slices::SlicedTensor{NTuple{L,V},T,2,S}
    expr::Union{Nothing,OpExpr}
    born::Int
end

"""
    PSFTask

A PSF of `m` carried operators whose reduced state `i` is original state
`orig[i]` (`orig[1] = 1`), with matrices at shell `n`. `pending` means the
operators are KK composites of shell `n` that must be refined before
evaluation (Lee Eq. (66)).
"""
mutable struct PSFTask{T,S,L,V}
    ops::Vector{EOp{T,S,L,V}}
    orig::Vector{Int}
    n::Int
    pending::Bool
    depth::Int
    born::Int
    label::String
end

# ---------------------------------------------------------------- accumulators

"""
    BinAcc / ExactAcc

Accumulators of PSF weight: binned histograms (dense for `L ≤ 2` or when
requested, sparse dictionary of bin tuples otherwise) with an optional
histogram of `|weight|`; or a list of exact Lehmann terms (positions, weights,
and `(shell, depth, string)` labels) with a hard size guard.
"""
mutable struct BinAcc{T,L}
    dense::Union{Nothing,Array{T,L}}
    sparse::Dict{NTuple{L,Int},T}
    absdense::Union{Nothing,Array{Float64,L}}
    abssparse::Dict{NTuple{L,Int},Float64}
    doabs::Bool
    total::T
end
mutable struct ExactAcc{T,L}
    pos::Vector{NTuple{L,Float64}}
    w::Vector{T}
    label::Vector{NTuple{3,Int}}
    maxterms::Int
    total::T
    wtol::Float64
    dropped::Float64
end

@inline function accumulate!(a::BinAcc{T,L}, F::NTuple{L,Int}, c, lab, absc = abs(c)) where {T,L}
    a.total += c
    if a.dense !== nothing
        @inbounds a.dense[F...] += c
        a.doabs && (@inbounds a.absdense[F...] += absc)
    else
        a.sparse[F] = get(a.sparse, F, zero(T)) + c
        a.doabs && (a.abssparse[F] = get(a.abssparse, F, 0.0) + absc)
    end
    return a
end
@inline function accumulate!(a::ExactAcc{T,L}, F::NTuple{L,Float64}, c, lab, absc = abs(c)) where {T,L}
    if abs(c) ≤ a.wtol
        a.dropped += abs(c)
        return a
    end
    length(a.w) < a.maxterms || throw(ErrorException(
        "exact mode exceeded max_terms = $(a.maxterms) Lehmann terms; use binned mode or raise max_terms"))
    a.total += c
    push!(a.pos, F); push!(a.w, c); push!(a.label, lab)
    return a
end

# ---------------------------------------------------------------- context

mutable struct EngineCtx{R,F,M,A}
    run::R
    fdm::F
    mode::M
    acc::A
    slicing::Symbol            # :lee or :left
    hooks::Vector{Any}
    stats::Dict{Symbol,Any}
    buf::Vector{ComplexF64}    # last-step accumulation buffer over the bins of one frequency
    absbuf::Vector{Float64}
    touched::Vector{Int}
end
EngineCtx(run, f, mode, acc, slicing, hooks, stats) =
    EngineCtx(run, f, mode, acc, slicing, hooks, stats,
              zeros(ComplexF64, mode isa BinnedMode ? nbins(mode.grid) : 0),
              zeros(mode isa BinnedMode ? nbins(mode.grid) : 0), Int[])

_count!(ctx, key, d = 1) = (ctx.stats[key] = get(ctx.stats, key, 0) + d)

_strings(m::Int) = [X for X in Iterators.product(ntuple(_ -> (:K, :D), m)...) if !all(==(:K), X)]
_strcode(X) = join(x === :K ? "K" : "D" for x in X)

@inline function _fill(F::NTuple{L,V}, g::Int, v::V) where {L,V}
    return ntuple(i -> i == g ? v : F[i], Val(L))
end

# merge the TAG_DIRECT entries of key k into F (positions with tags == TAG_DIRECT)
@inline function _mergedirect(F::NTuple{L,V}, k::NTuple{L,V}, tags::NTuple{L,Int8}) where {L,V}
    return ntuple(i -> tags[i] == TAG_DIRECT ? k[i] : F[i], Val(L))
end

# converted values of the TAG_LEFT entries of key k, shifted by the exact
# energy δ = E_left - E_ref: new value = key(centre(k) + δ)
@inline function _mergeleft(mode, F::NTuple{L,V}, k::NTuple{L,V}, tags::NTuple{L,Int8}, δ::Float64) where {L,V}
    return ntuple(i -> tags[i] == TAG_LEFT ? freqkey(mode, centreval(mode, k[i]) + δ) : F[i], Val(L))
end

_hasleft(tags) = any(==(TAG_LEFT), tags)

# ---------------------------------------------------------------- operator access

# current tensors of a carried operator at shell n
function _advance!(ctx::EngineCtx, op::EOp{T}, n::Int) where {T}
    if op.expr !== nothing
        k = first(keys(op.slices))
        op.slices = SlicedTensor(Dict(k => convert_eltype(T, project(ctx.run, op.expr, n))))
    else
        op.slices = mapslices(op.slices) do M
            _count!(ctx, :refinements)
            convert_eltype(T, refine(ctx.run, M, n))
        end
    end
    return op
end

function advance!(ctx::EngineCtx, t::PSFTask)
    n = t.n + 1
    for op in t.ops
        _advance!(ctx, op, n)
    end
    t.n = n
    t.pending = false
    return t
end

# ---------------------------------------------------------------- sector chains

# sectors q_1..q_{m+1} (q_{m+1} = q_1) of the states between the operators,
# starting from q1: q_{i+1} = q_i ⊖ Δ_i.
function _sector_chain(ops, q1)
    qs = [q1]
    for op in ops
        push!(qs, fuse(qs[end], dual(op.shift)))
    end
    return qs
end

# ---------------------------------------------------------------- traces

"""
    shell_trace!(ctx, task, X)

An adequately refined term: the shell-`n` trace
`Σ ρ̄^n_{X₁X₁} O¹_{X₁X₂} ⋯ O^m_{X_mX₁}` with every visible frequency
`E_{s_i} - E_{s_1}` and every sliced frequency recorded (binned or exact).
Lee Eqs. (57)-(59) (ℓ = 2), (62) (ℓ = 3), (68) (ℓ = 4), generalized to sliced
inputs.
"""
function shell_trace!(ctx::EngineCtx, t::PSFTask{T,S,L,V}, X) where {T,S,L,V}
    run = ctx.run; f = ctx.fdm; mode = ctx.mode
    n = t.n; sh = shell(run, n); m = length(t.ops)
    lab = (n, t.depth, _strhash(X))
    _hasleft(t.ops[1].tags) && error("internal: the first operator cannot carry left-referenced slices")
    for q1 in sectors(sh.space)
        qs = _sector_chain(t.ops, q1)
        qs[end] == q1 || continue
        rr = Vector{UnitRange{Int}}(undef, m)
        ok = true
        for i in 1:m
            haskey(sh.space, qs[i]) || (ok = false; break)
            rr[i] = xrange(sh, qs[i], X[i])
            isempty(rr[i]) && (ok = false; break)
        end
        ok || continue
        Es = [sh.E[qs[i]][rr[i]] for i in 1:m]
        ρ = X[1] === :K ? block_view(rhoK(f, n), q1, :K, :K) : nothing
        X[1] === :K && ρ === nothing && continue       # ρ̄_KK of q1 vanishes
        ρd = X[1] === :D ? rhoD(f, n)[q1] : nothing
        F0 = ntuple(_ -> zero(V), Val(L))
        for (k1, M1) in t.ops[1].slices
            B1 = block_view(M1, qs[2], X[1], m == 1 ? X[1] : X[2])
            B1 === nothing && continue
            R = ρ === nothing ? (ρd .* B1) : ρ * B1
            Fk = _mergedirect(F0, k1, t.ops[1].tags)
            for s1 in axes(R, 1)
                v = R[s1, :]
                all(iszero, v) && continue
                E1 = Es[1][s1]
                # keys of the visible frequencies E_{s_i} - E_{s_1}, once per row
                vis = [i == 1 ? V[] : V[freqkey(mode, e - E1) for e in Es[i]] for i in 1:m]
                grp = [i == 1 || i == m ? Tuple{V,Vector{Int}}[] : _group_by_key(vis[i]) for i in 1:m]
                _trace_step!(ctx, t, X, qs, rr, Es, 2, v, Fk, E1, s1, lab, vis, grp)
            end
        end
        _count!(ctx, :traces)
    end
    return nothing
end

_strhash(X) = foldl((a, x) -> 2a + (x === :D ? 1 : 0), X; init = 1)

# states grouped by their key: [(key, indices), …] in key order
function _group_by_key(keys::Vector{V}) where {V}
    out = Tuple{V,Vector{Int}}[]
    isempty(keys) && return out
    p = sortperm(keys)
    start = 1
    for j in 2:(length(p) + 1)
        if j > length(p) || keys[p[j]] != keys[p[start]]
            push!(out, (keys[p[start]], p[start:(j - 1)]))
            start = j
        end
    end
    return out
end

function _trace_step!(ctx, t::PSFTask{T,S,L,V}, X, qs, rr, Es, i, v, F, E1, s1, lab, vis, grp) where {T,S,L,V}
    m = length(t.ops); mode = ctx.mode
    op = t.ops[i]
    g = t.orig[i] - 1                 # visible frequency of state i
    Ei = Es[i]; Ki = vis[i]
    nxt = i == m ? 1 : i + 1
    hl = _hasleft(op.tags)
    for (k, M) in op.slices
        B = block_view(M, qs[i + 1], X[i], X[nxt])
        B === nothing && continue
        Fk = _mergedirect(F, k, op.tags)
        if i == m && !hl && mode isa BinnedMode
            # only the last visible frequency varies: accumulate over its bins
            # in a dense buffer, flush once per group
            buf = ctx.buf; ab = ctx.absbuf; touched = ctx.touched
            for s in eachindex(v)
                vs = v[s]
                iszero(vs) && continue
                c = vs * B[s, s1]
                iszero(c) && continue
                b = Ki[s]
                if iszero(ab[b])
                    push!(touched, b)
                end
                buf[b] += c; ab[b] += abs(c)
            end
            for b in touched
                accumulate!(ctx.acc, _fill(Fk, g, b), T <: Real ? real(buf[b]) : buf[b], lab, ab[b])
                buf[b] = 0; ab[b] = 0.0
            end
            empty!(touched)
        elseif i == m
            for s in eachindex(v)
                vs = v[s]
                iszero(vs) && continue
                c = vs * B[s, s1]
                iszero(c) && continue
                F2 = _fill(Fk, g, Ki[s])
                hl && (F2 = _mergeleft(mode, F2, k, op.tags, Ei[s] - E1))
                accumulate!(ctx.acc, F2, c, lab)
            end
        elseif !hl
            # the group of a state depends only on its visible key: precomputed
            for (key, idx) in grp[i]
                vi = view(v, idx)
                all(iszero, vi) && continue
                w = transpose(view(B, idx, :)) * vi
                all(iszero, w) && continue
                _trace_step!(ctx, t, X, qs, rr, Es, i + 1, w, _fill(Fk, g, key), E1, s1, lab, vis, grp)
            end
        else
            groups = Dict{NTuple{L,V},Vector{Int}}()
            for s in eachindex(v)
                iszero(v[s]) && continue
                F2 = _fill(Fk, g, Ki[s])
                F2 = _mergeleft(mode, F2, k, op.tags, Ei[s] - E1)
                push!(get!(() -> Int[], groups, F2), s)
            end
            for (F2, idx) in groups
                w = transpose(B[idx, :]) * v[idx]
                all(iszero, w) && continue
                _trace_step!(ctx, t, X, qs, rr, Es, i + 1, w, F2, E1, s1, lab, vis, grp)
            end
        end
    end
end

# ---------------------------------------------------------------- composites

"""
    composites(ctx, task, X) -> PSFTask

For a sector string with `X₁ = K` and more K's, multiply the operators between
consecutive K states into composite KK operators of shell `n` with their
internal (D) frequencies sliced (Lee Eq. (60); slicing is exact bookkeeping,
the only approximation is the bin resolution of the sliced frequency, Lee
p. 14). Returns the reduced PSF task over the later shells `n̄ > n`.
"""
function composites(ctx::EngineCtx, t::PSFTask{T,S,L,V}, X) where {T,S,L,V}
    m = length(t.ops)
    J = findall(==(:K), collect(X))
    @assert J[1] == 1 && length(J) ≥ 2
    newops = EOp{T,S,L,V}[]
    nseg = length(J)
    for s in 1:nseg
        a = J[s]; b = s < nseg ? J[s + 1] : m + 1
        kind = s == 1 ? :first : (s == nseg ? :last : :middle)
        op = if kind === :last && ctx.slicing === :lee
            _composite_right(ctx, t, X, a)
        else
            _composite_left(ctx, t, X, a, b, kind === :first)
        end
        push!(newops, op)
        _count!(ctx, :composites)
    end
    any(op -> isempty(op.slices), newops) && return nothing
    return PSFTask{T,S,L,V}(newops, t.orig[J], t.n, true, t.depth + 1, t.n, _strcode(X))
end

function _newcomposite(t::PSFTask{T,S,L,V}, ops, tags, acc::Dict, sh) where {T,S,L,V}
    Δ = foldl(fuse, (op.shift for op in ops); init = neutral(S))
    kept = kept_leg(sh.leg)
    slices = SlicedTensor{NTuple{L,V},T}(TNOps.legs(kk_tensor(kept, Δ, Dict{S,Matrix{T}}(), T)), Δ)
    for (k, blocks) in acc
        B = kk_tensor(kept, Δ, blocks, T)          # all-zero blocks are not stored
        nblocks(B) == 0 || setslice!(slices, k, B)
    end
    return EOp{T,S,L,V}(Δ, tags, slices, nothing, sh.n)
end

# Tags of the composite: union of the constituents' tags after conversion, plus
# the internal states' own frequencies.
function _composite_tags(t::PSFTask{T,S,L,V}, a, b, newtag::Int8, firstseg::Bool, rightref::Bool) where {T,S,L,V}
    tags = fill(TAG_NONE, L)
    for i in a:(b - 1)
        op = t.ops[i]
        for g in 1:L
            op.tags[g] == TAG_NONE && continue
            tg = op.tags[g]
            if tg == TAG_LEFT
                if i == a
                    tg = firstseg ? TAG_DIRECT : TAG_LEFT       # left endpoint kept
                else
                    tg = newtag                                  # converted at internal state i
                end
            end
            tags[g] = tg
        end
        i > a && (tags[t.orig[i] - 1] = newtag)                  # internal state i
    end
    return Tuple(tags)
end

# Left processing of segment a → b (states a and b are K, a+1..b-1 are D):
# keys relative to the left endpoint (direct if the segment starts at state 1).
function _composite_left(ctx::EngineCtx, t::PSFTask{T,S,L,V}, X, a, b, firstseg::Bool) where {T,S,L,V}
    run = ctx.run; mode = ctx.mode; sh = shell(run, t.n); m = length(t.ops)
    newtag = firstseg ? TAG_DIRECT : TAG_LEFT
    tags = _composite_tags(t, a, b, newtag, firstseg, false)
    acc = Dict{NTuple{L,V},Dict{S,Matrix{T}}}()
    ops = t.ops[a:(b - 1)]
    Xc(i) = i == m + 1 ? X[1] : X[i]
    for qa in sectors(sh.space)
        nkept(sh, qa) > 0 || continue
        qs = Vector{S}(undef, b - a + 1); qs[1] = qa
        ok = true
        for (j, i) in enumerate(a:(b - 1))
            qs[j + 1] = fuse(qs[j], dual(t.ops[i].shift))
            haskey(sh.space, qs[j + 1]) || (ok = false; break)
        end
        ok || continue
        xs = [Xc(a + j - 1) for j in 1:(b - a + 1)]
        rr = [xrange(sh, qs[j], xs[j]) for j in 1:(b - a + 1)]
        any(isempty, rr) && continue
        Es = [sh.E[qs[j]][rr[j]] for j in 1:(b - a + 1)]
        qb = qs[end]
        F0 = ntuple(_ -> zero(V), Val(L))
        op1 = t.ops[a]
        for (k, M) in op1.slices
            B = block_view(M, qs[2], xs[1], xs[2])
            B === nothing && continue
            # tags of op a: direct entries copied; left entries kept (or made
            # direct for the first segment) with their value unchanged
            Fk = ntuple(g -> op1.tags[g] == TAG_NONE ? F0[g] : k[g], Val(L))
            for sa in axes(B, 1)
                v = B[sa, :]
                all(iszero, v) && continue
                if b == a + 1
                    _addrow!(acc, Fk, qb, sa, v, nkept(sh, qa), nkept(sh, qb), T)
                else
                    _comp_left_step!(ctx, t, acc, a, b, 2, qs, rr, xs, Es, v, Fk, Es[1][sa], sa, qa, newtag, sh)
                end
            end
        end
    end
    return _newcomposite(t, ops, tags, acc, sh)
end

function _addrow!(acc, F, qb, sa, v, nra, ncb, ::Type{T}) where {T}
    d = get!(valtype(acc), acc, F)
    M = get!(() -> zeros(T, nra, ncb), d, qb)
    @views M[sa, :] .+= v
    return nothing
end

# j: position within the segment (state a + j - 1 is internal, D)
function _comp_left_step!(ctx, t::PSFTask{T,S,L,V}, acc, a, b, j, qs, rr, xs, Es, v, F, Eref, sa, qa, newtag, sh) where {T,S,L,V}
    mode = ctx.mode
    i = a + j - 1                     # current state index (reduced), internal
    g = t.orig[i] - 1
    op = t.ops[i]
    hl = _hasleft(op.tags)
    Ei = Es[j]
    for (k, M) in op.slices
        B = block_view(M, qs[j + 1], xs[j], xs[j + 1])
        B === nothing && continue
        Fk = _mergedirect(F, k, op.tags)
        groups = Dict{NTuple{L,V},Vector{Int}}()
        for s in eachindex(v)
            iszero(v[s]) && continue
            δ = Ei[s] - Eref
            F2 = _fill(Fk, g, freqkey(mode, δ))
            hl && (F2 = _mergeleft(mode, F2, k, op.tags, δ))
            push!(get!(() -> Int[], groups, F2), s)
        end
        for (F2, idx) in groups
            w = transpose(B[idx, :]) * v[idx]
            all(iszero, w) && continue
            if i + 1 == b
                _addrow!(acc, F2, qs[end], sa, w, nkept(sh, qa), nkept(sh, qs[end]), T)
            else
                _comp_left_step!(ctx, t, acc, a, b, j + 1, qs, rr, xs, Es, w, F2, Eref, sa, qa, newtag, sh)
            end
        end
    end
end

# Right processing of the last segment a → 1 (state m+1 ≡ 1): keys relative to
# state 1, i.e. direct.
function _composite_right(ctx::EngineCtx, t::PSFTask{T,S,L,V}, X, a) where {T,S,L,V}
    run = ctx.run; mode = ctx.mode; sh = shell(run, t.n); m = length(t.ops)
    tags = _composite_tags(t, a, m + 1, TAG_DIRECT, false, true)
    acc = Dict{NTuple{L,V},Dict{S,Matrix{T}}}()
    ops = t.ops[a:m]
    nseg = m - a + 1                  # number of operators
    for q1 in sectors(sh.space)
        nkept(sh, q1) > 0 || continue
        # backward chain: state m+1 = 1 has sector q1; state i has q_i = q_{i+1} ⊕ Δ_i
        qs = Vector{S}(undef, nseg + 1); qs[end] = q1
        ok = true
        for (j, i) in enumerate(m:-1:a)
            qs[end - j] = fuse(qs[end - j + 1], t.ops[i].shift)
            haskey(sh.space, qs[end - j]) || (ok = false; break)
        end
        ok || continue
        # qs[j] is the sector of state a + j - 1
        Xs(j) = j == nseg + 1 ? :K : X[a + j - 1]
        xs = [Xs(j) for j in 1:(nseg + 1)]
        rr = [xrange(sh, qs[j], xs[j]) for j in 1:(nseg + 1)]
        any(isempty, rr) && continue
        Es = [sh.E[qs[j]][rr[j]] for j in 1:(nseg + 1)]
        F0 = ntuple(_ -> zero(V), Val(L))
        opm = t.ops[m]
        for (k, M) in opm.slices
            B = block_view(M, q1, xs[nseg], xs[nseg + 1])
            B === nothing && continue
            Fk = _mergedirect(F0, k, opm.tags)
            for s1 in axes(B, 2)
                w = B[:, s1]
                all(iszero, w) && continue
                if nseg == 1
                    # single operator: its left entries stay relative to its left endpoint
                    Fk1 = ntuple(g -> opm.tags[g] == TAG_LEFT ? k[g] : Fk[g], Val(L))
                    _addcol!(acc, Fk1, q1, s1, w, nkept(sh, qs[1]), nkept(sh, q1), T)
                else
                    _comp_right_step!(ctx, t, acc, a, nseg, qs, rr, xs, Es, w, Fk, k, Es[end][s1], s1, q1, sh)
                end
            end
        end
    end
    return _newcomposite(t, ops, tags, acc, sh)
end

function _addcol!(acc, F, q1, s1, w, nra, nc1, ::Type{T}) where {T}
    d = get!(valtype(acc), acc, F)
    M = get!(() -> zeros(T, nra, nc1), d, q1)
    @views M[:, s1] .+= w
    return nothing
end

# w lives on state j_state = a + j - 1 (internal D), reached by applying op
# i = a + j - 1 with key kprev (whose left entries refer to state i).
function _comp_right_step!(ctx, t::PSFTask{T,S,L,V}, acc, a, j, qs, rr, xs, Es, w, F, kprev, Eref, s1, q1, sh) where {T,S,L,V}
    mode = ctx.mode
    i = a + j - 1
    g = t.orig[i] - 1
    opi = t.ops[i]
    hl = _hasleft(opi.tags)
    Ei = Es[j]
    groups = Dict{NTuple{L,V},Vector{Int}}()
    for s in eachindex(w)
        iszero(w[s]) && continue
        δ = Ei[s] - Eref
        F2 = _fill(F, g, freqkey(mode, δ))
        hl && (F2 = _mergeleft(mode, F2, kprev, opi.tags, δ))
        push!(get!(() -> Int[], groups, F2), s)
    end
    op = t.ops[i - 1]
    for (F2, idx) in groups, (k, M) in op.slices
        B = block_view(M, qs[j], xs[j - 1], xs[j])
        B === nothing && continue
        Fk = _mergedirect(F2, k, op.tags)
        w2 = B[:, idx] * w[idx]
        all(iszero, w2) && continue
        if i - 1 == a
            Fk1 = ntuple(gg -> op.tags[gg] == TAG_LEFT ? k[gg] : Fk[gg], Val(L))
            _addcol!(acc, Fk1, q1, s1, w2, nkept(sh, qs[1]), nkept(sh, q1), T)
        else
            _comp_right_step!(ctx, t, acc, a, j - 1, qs, rr, xs, Es, w2, Fk, k, Eref, s1, q1, sh)
        end
    end
end

# ---------------------------------------------------------------- shell processing

"""
    process_shell!(ctx, task, spawn)

Evaluate every sector string of `task` at its current shell: adequately refined
strings as shell traces, the others by spawning reduced PSF tasks (`spawn` is
called with each child).
"""
function process_shell!(ctx::EngineCtx, t::PSFTask, spawn)
    m = length(t.ops)
    for X in _strings(m)
        if X[1] === :D || count(==(:K), X) == 1
            shell_trace!(ctx, t, X)
        else
            t.n < ctx.run.params.N || continue
            child = composites(ctx, t, X)
            child === nothing || spawn(child)
        end
    end
end

"""
    run_tasks!(ctx, root; schedule = :depth)

Drive the PSF tasks over the shells. `:depth` finishes each spawned task over
all later shells before continuing (least memory); `:breadth` runs one forward
sweep over the shells and advances all carried composites together, which is
the order in which the hooks of `hooks/` see every composite being carried at
shell `n` (Sec. 5.10). Both give identical results.
"""
function run_tasks!(ctx::EngineCtx, root::PSFTask; schedule::Symbol = :depth)
    N = ctx.run.params.N
    if schedule === :depth
        if Threads.nthreads() > 1 && isempty(ctx.hooks)
            _run_depth_parallel!(ctx, root)
        else
            _run_depth!(ctx, root)
        end
    elseif schedule === :breadth
        active = PSFTask[root]
        n = root.n
        while !isempty(active)
            next = PSFTask[]
            spawned = PSFTask[]
            for t in active
                (t.pending || t.n < n) && advance!(ctx, t)
                process_shell!(ctx, t, c -> push!(spawned, c))
            end
            isempty(ctx.hooks) || _call_hooks(ctx, n, active)
            n == N && break
            for t in active
                push!(next, t)
            end
            append!(next, spawned)
            active = next
            n += 1
        end
    else
        throw(ArgumentError("schedule must be :depth or :breadth"))
    end
    return ctx
end

function _run_depth!(ctx::EngineCtx, t::PSFTask)
    N = ctx.run.params.N
    t.pending && advance!(ctx, t)
    while true
        process_shell!(ctx, t, c -> _run_depth!(ctx, c))
        isempty(ctx.hooks) || _call_hooks(ctx, t.n, PSFTask[t])
        t.n == N && return nothing
        advance!(ctx, t)
    end
end

# A context for a task running on another thread: own accumulator, buffers,
# counters and clamping counter; shares the (read-only) run, FDM and grid.
function _subctx(ctx::EngineCtx)
    mode = ctx.mode isa BinnedMode ? BinnedMode(ctx.mode.grid, 0) : ctx.mode
    return EngineCtx(ctx.run, ctx.fdm, mode, _emptyacc(ctx.acc), ctx.slicing, ctx.hooks, Dict{Symbol,Any}())
end
_emptyacc(a::BinAcc{T,L}) where {T,L} =
    BinAcc{T,L}(a.dense === nothing ? nothing : zero(a.dense), Dict{NTuple{L,Int},T}(),
                a.absdense === nothing ? nothing : zero(a.absdense), Dict{NTuple{L,Int},Float64}(), a.doabs, zero(T))
_emptyacc(a::ExactAcc{T,L}) where {T,L} =
    ExactAcc{T,L}(NTuple{L,Float64}[], T[], NTuple{3,Int}[], a.maxterms, zero(T), a.wtol, 0.0)

function _merge!(ctx::EngineCtx, sub::EngineCtx)
    a = ctx.acc; b = sub.acc
    a.total += b.total
    if a isa BinAcc
        a.dense === nothing || (a.dense .+= b.dense)
        a.absdense === nothing || (a.absdense .+= b.absdense)
        for (k, v) in b.sparse; a.sparse[k] = get(a.sparse, k, zero(v)) + v; end
        for (k, v) in b.abssparse; a.abssparse[k] = get(a.abssparse, k, 0.0) + v; end
    else
        length(a.w) + length(b.w) ≤ a.maxterms || throw(ErrorException(
            "exact mode exceeded max_terms = $(a.maxterms) Lehmann terms; use binned mode or raise max_terms"))
        append!(a.pos, b.pos); append!(a.w, b.w); append!(a.label, b.label); a.dropped += b.dropped
    end
    ctx.mode isa BinnedMode && (ctx.mode.nclamped += sub.mode.nclamped)
    for (k, v) in sub.stats
        ctx.stats[k] = get(ctx.stats, k, 0) + v
    end
    return ctx
end

"""
    _run_depth_parallel!(ctx, root)

Depth-first schedule with the reduced PSF tasks spawned by the root task run
concurrently on the available threads (each with its own accumulator, merged
at the end; at most `Threads.nthreads()` at a time). Identical results to the
serial schedule up to the summation order of floating-point additions.
"""
function _run_depth_parallel!(ctx::EngineCtx, root::PSFTask)
    N = ctx.run.params.N
    sem = Base.Semaphore(Threads.nthreads())
    lk = ReentrantLock()
    jobs = Task[]
    spawn = c -> begin
        Base.acquire(sem)
        push!(jobs, Threads.@spawn begin
            try
                sub = _subctx(ctx)
                _run_depth!(sub, c)
                lock(() -> _merge!(ctx, sub), lk)
            finally
                Base.release(sem)
            end
        end)
    end
    root.pending && advance!(ctx, root)
    while true
        # traces of the root run on this thread, under the lock that merges sub-results
        lock(lk) do
            for X in _strings(length(root.ops))
                if X[1] === :D || count(==(:K), X) == 1
                    shell_trace!(ctx, root, X)
                end
            end
        end
        if root.n < N
            for X in _strings(length(root.ops))
                (X[1] === :D || count(==(:K), X) == 1) && continue
                child = composites(ctx, root, X)
                child === nothing || spawn(child)
            end
        end
        root.n == N && break
        advance!(ctx, root)
    end
    foreach(wait, jobs)
    return nothing
end
