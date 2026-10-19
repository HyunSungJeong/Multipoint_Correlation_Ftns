# Block-sparse tensors with an abelian symmetry.
#
# A `BlockTensor{T,N,S}` has `N` legs, a flux `Δ::S` and a dictionary from
# sector tuples `NTuple{N,LegSector}` (one sector per leg) to dense
# `Array{T,N}` blocks. Block rule: a block may exist only if the charges of its
# sectors fuse to the flux, i.e. the fused labels of the `Out` legs equal the
# fused labels of the `In` legs fused with Δ. Absent blocks are zero and not
# stored. `T` changes only through explicit arithmetic (promotion of the two
# operands).
#
# Rank 2 with legs (`Out` row, `In` column) is the block operator of the
# pre-refactor layer: the flux is its shift, `target(A, q) = q ⊕ Δ`, and
# `getblock(A, q)` is keyed by the column label.

"""
    BlockTensor{T,N,S}

A block-sparse tensor: legs `legs::NTuple{N,Leg{S}}`, flux `flux::S`, and blocks
`Dict{NTuple{N,LegSector{S}},Array{T,N}}`, block `k` of size
`(dim(legs[1], k[1]), …, dim(legs[N], k[N]))`. See the README for the block
rule and the index conventions.
"""
struct BlockTensor{T<:Number,N,S<:Sector}
    legs::NTuple{N,Leg{S}}
    flux::S
    blocks::Dict{NTuple{N,LegSector{S}},Array{T,N}}
end

"""    BlockTensor{T}(legs, flux) — the zero tensor."""
BlockTensor{T}(legs::NTuple{N,Leg{S}}, flux::S) where {T,N,S} =
    BlockTensor{T,N,S}(legs, flux, Dict{NTuple{N,LegSector{S}},Array{T,N}}())

const BlockKey{N,S} = NTuple{N,LegSector{S}}

Base.eltype(::Type{<:BlockTensor{T}}) where {T} = T
Base.eltype(::BlockTensor{T}) where {T} = T
Base.ndims(::BlockTensor{T,N}) where {T,N} = N
sectortype(::BlockTensor{T,N,S}) where {T,N,S} = S
legs(A::BlockTensor) = getfield(A, :legs)
leg(A::BlockTensor, i::Integer) = getfield(A, :legs)[i]
flux(A::BlockTensor) = getfield(A, :flux)
"""    nblocks(A) — number of stored blocks."""
nblocks(A::BlockTensor) = length(getfield(A, :blocks))
"""    blockkeys(A) — the sector tuples of the stored blocks."""
blockkeys(A::BlockTensor) = keys(getfield(A, :blocks))
"""
    eachblock(A)

The stored blocks as `key => array` pairs. Read only: mutate a tensor through
[`setblock!`](@ref) only.
"""
eachblock(A::BlockTensor) = pairs(getfield(A, :blocks))
"""    blockdims(A, key) — the size of block `key` (0 along legs lacking the sector)."""
blockdims(A::BlockTensor{T,N}, k::BlockKey) where {T,N} = ntuple(i -> dim(leg(A, i), k[i]), Val(N))

# compatibility accessors of the rank-2 operator (pre-refactor field names)
function Base.getproperty(A::BlockTensor{T,N}, s::Symbol) where {T,N}
    if s === :shift
        return getfield(A, :flux)
    elseif s === :row
        N == 2 || throw(ArgumentError(":row is defined for rank-2 tensors only"))
        return getfield(A, :legs)[1]
    elseif s === :col
        N == 2 || throw(ArgumentError(":col is defined for rank-2 tensors only"))
        return getfield(A, :legs)[2]
    end
    return getfield(A, s)
end
Base.propertynames(::BlockTensor) = (:legs, :flux, :blocks, :shift, :row, :col)

"""
    keyflux(legs, key) -> S

The fused charges of the sectors `key` on `legs` ([`charge`](@ref)); a block is
allowed iff this equals the flux.
"""
function keyflux(ls::NTuple{N,Leg{S}}, k::BlockKey{N,S}) where {N,S}
    q = neutral(S)
    for i in 1:N
        q = fuse(q, charge(ls[i], k[i]))
    end
    return q
end
keyflux(::Tuple{}, ::Tuple{}) = throw(ArgumentError("keyflux of a rank-0 key needs the tensor: use keyflux(A, ())"))
"""    keyflux(A, key) — `keyflux(legs(A), key)`, also for rank 0 (the neutral label)."""
keyflux(A::BlockTensor{T,0,S}, ::Tuple{}) where {T,S} = neutral(S)
keyflux(A::BlockTensor{T,N,S}, k::BlockKey{N,S}) where {T,N,S} = keyflux(legs(A), k)

"""    isallowed(A, key) — every sector exists on its leg and the block rule holds."""
isallowed(A::BlockTensor{T,N,S}, k::BlockKey{N,S}) where {T,N,S} =
    all(i -> haskey(leg(A, i), k[i]), 1:N) && keyflux(A, k) == flux(A)

"""
    allowed_keys(legs, flux) -> Vector{NTuple{N,LegSector}}

Every sector tuple the block rule allows, in lexicographic order of the
sector indices (first leg fastest). The last leg's label is fixed by the
others, so no forbidden tuple is enumerated.
"""
function allowed_keys(ls::NTuple{N,Leg{S}}, Δ::S) where {N,S}
    out = BlockKey{N,S}[]
    N == 0 && return out
    last = ls[N]
    byLabel = Dict{S,Vector{LegSector{S}}}()
    for s in legsectors(last)
        push!(get!(() -> LegSector{S}[], byLabel, s.label), s)
    end
    heads = N == 1 ? [()] : vec(collect(Iterators.product((legsectors(l) for l in ls[1:(N - 1)])...)))
    for h in heads
        q = neutral(S)
        for i in 1:(N - 1)
            q = fuse(q, charge(ls[i], h[i]))
        end
        c = fuse(Δ, dual(q))                         # charge required on the last leg
        lab = arrow(last) == Out ? c : dual(c)
        for s in get(byLabel, lab, LegSector{S}[])
            push!(out, (h..., s))
        end
    end
    return out
end
allowed_keys(A::BlockTensor) = allowed_keys(legs(A), flux(A))

"""
    setblock!(A, key, M)

Store block `M` for the sector tuple `key`, checking that the sectors exist,
that the block rule allows the block and that the size matches. Empty blocks
are not stored.
"""
function setblock!(A::BlockTensor{T,N,S}, k::BlockKey{N,S}, M::AbstractArray) where {T,N,S}
    for i in 1:N
        haskey(leg(A, i), k[i]) || throw(ArgumentError("leg $i ($(legstring(leg(A, i)))) has no sector $(k[i])"))
    end
    keyflux(A, k) == flux(A) ||
        throw(ArgumentError("block $k is forbidden by the flux $(flux(A)) (its charges fuse to $(keyflux(A, k)))"))
    sz = blockdims(A, k)
    size(M) == sz || throw(DimensionMismatch("block $k must have size $sz, got $(size(M))"))
    isempty(M) && return A
    getfield(A, :blocks)[k] = convert(Array{T,N}, M)
    return A
end

"""
    getblock(A, key) -> Array

The block of the sector tuple `key`; a zero array of the right size if it is
not stored.
"""
function getblock(A::BlockTensor{T,N,S}, k::BlockKey{N,S}) where {T,N,S}
    b = get(getfield(A, :blocks), k, nothing)
    b === nothing || return b
    return zeros(T, blockdims(A, k))
end
"""    hasblock(A, key) — whether block `key` is stored."""
hasblock(A::BlockTensor{T,N,S}, k::BlockKey{N,S}) where {T,N,S} = haskey(getfield(A, :blocks), k)
"""    blockornothing(A, key) — the stored block or `nothing`."""
blockornothing(A::BlockTensor{T,N,S}, k::BlockKey{N,S}) where {T,N,S} = get(getfield(A, :blocks), k, nothing)

Base.copy(A::BlockTensor{T,N,S}) where {T,N,S} =
    BlockTensor{T,N,S}(legs(A), flux(A), Dict(k => copy(M) for (k, M) in eachblock(A)))

"""    convert_eltype(T, A) — `A` with element type `T` (no copy if it already has it)."""
convert_eltype(::Type{T}, A::BlockTensor{T}) where {T} = A
convert_eltype(::Type{T}, A::BlockTensor{T2,N,S}) where {T,T2,N,S} =
    BlockTensor{T,N,S}(legs(A), flux(A), Dict(k => convert(Array{T,N}, M) for (k, M) in eachblock(A)))

"""
    randtensor(rng, T, legs, flux; density = 1)

A tensor with standard-normal blocks on a random fraction `density` of the
allowed sector tuples (for tests and examples).
"""
function randtensor(rng, ::Type{T}, ls::NTuple{N,Leg{S}}, Δ::S; density::Real = 1.0) where {T,N,S}
    A = BlockTensor{T}(ls, Δ)
    for k in allowed_keys(ls, Δ)
        rand(rng) ≤ density || continue
        setblock!(A, k, randn(rng, T, blockdims(A, k)))
    end
    return A
end

# ---------------------------------------------------------------- rank 2

"""
    target(A, q) -> label

For a rank-2 tensor: the row label reached from column label `q`, `q ⊕ Δ` for
legs (`Out`, `In`).
"""
function target(A::BlockTensor{T,2,S}, q::S) where {T,S}
    r, c = legs(A)
    qc = arrow(c) == Out ? q : dual(q)
    need = fuse(flux(A), dual(qc))
    return arrow(r) == Out ? need : dual(need)
end

"""
    getblock(A, q) -> Matrix

For a rank-2 tensor: the block of all sectors with column label `q` and row
label `target(A, q)`, the tags of each label stacked in ascending order (for
untagged legs this is the stored block itself); a zero matrix of the right
size if nothing is stored.
"""
function getblock(A::BlockTensor{T,2,S}, q::S) where {T,S}
    r = target(A, q)
    rl, cl = legs(A)
    rs = [s for s in legsectors(rl) if s.label == r]; cs = [s for s in legsectors(cl) if s.label == q]
    if length(rs) ≤ 1 && length(cs) ≤ 1
        (isempty(rs) || isempty(cs)) && return zeros(T, dim(rl, r), dim(cl, q))
        return getblock(A, (rs[1], cs[1]))
    end
    M = zeros(T, dim(rl, r), dim(cl, q))
    ro = 0
    for a in rs
        co = 0
        for b in cs
            B = blockornothing(A, (a, b))
            B === nothing || (M[ro .+ (1:size(B, 1)), co .+ (1:size(B, 2))] .= B)
            co += dim(cl, b)
        end
        ro += dim(rl, a)
    end
    return M
end

"""
    setblock!(A, q, M)

For a rank-2 tensor: store the label block of column label `q` (layout of
[`getblock`](@ref)`(A, q)`), split into its tagged blocks.
"""
function setblock!(A::BlockTensor{T,2,S}, q::S, M::AbstractMatrix) where {T,S}
    r = target(A, q)
    rl, cl = legs(A)
    size(M) == (dim(rl, r), dim(cl, q)) ||
        throw(DimensionMismatch("block for $q must be $(dim(rl, r))×$(dim(cl, q)), got $(size(M))"))
    isempty(M) && return A
    ro = 0
    for a in legsectors(rl)
        a.label == r || continue
        co = 0
        for b in legsectors(cl)
            b.label == q || continue
            setblock!(A, (a, b), M[ro .+ (1:dim(rl, a)), co .+ (1:dim(cl, b))])
            co += dim(cl, b)
        end
        ro += dim(rl, a)
    end
    return A
end

"""
    BlockTensor{T}(row::Leg, col::Leg, Δ) — the zero rank-2 operator with legs
    (`row` as `Out`, `col` as `In`), the layout of the pre-refactor `BlockOp`.
"""
BlockTensor{T}(row::Leg{S}, col::Leg{S}, Δ::S) where {T,S} = BlockTensor{T}((witharrow(row, Out), witharrow(col, In)), Δ)

"""
    identity_tensor(l, T = Float64)

The identity on leg `l` as a neutral rank-2 tensor with legs (`l` as `Out`, `l`
as `In`).
"""
function identity_tensor(l::Leg{S}, ::Type{T} = Float64) where {S,T}
    A = BlockTensor{T}(l, l, neutral(S))
    for (s, d) in zip(legsectors(l), l.dims)
        setblock!(A, (s, s), Matrix{T}(I, d, d))
    end
    return A
end

"""
    subtensor(A, keep) -> BlockTensor

`A` restricted to the sectors `s` of leg `i` with `keep[i](s)` (`nothing` keeps
the whole leg); the legs shrink accordingly. Example: the kept-kept part of an
NRG shell operator, `subtensor(O, (s -> s.tag == 1, s -> s.tag == 1))`.
"""
function subtensor(A::BlockTensor{T,N,S}, keep::NTuple{N,Any}) where {T,N,S}
    ls = ntuple(i -> keep[i] === nothing ? leg(A, i) : subleg(leg(A, i), keep[i]), Val(N))
    B = BlockTensor{T}(ls, flux(A))
    for (k, M) in eachblock(A)
        all(i -> keep[i] === nothing || keep[i](k[i]), 1:N) || continue
        getfield(B, :blocks)[k] = M
    end
    return B
end

"""
    restrict(A, ranges) -> BlockTensor

`A` restricted to index ranges inside sectors: `ranges[i]` is `nothing` (keep
leg `i`) or a `Dict(sector => range)`; sectors absent from a dictionary are
dropped. The new legs have the range lengths as dimensions.
"""
function restrict(A::BlockTensor{T,N,S}, ranges::NTuple{N,Any}) where {T,N,S}
    ls = ntuple(Val(N)) do i
        ranges[i] === nothing && return leg(A, i)
        l = leg(A, i)
        Leg{S}(collect(keys(ranges[i])), [length(r) for r in values(ranges[i])], arrow(l), legname(l))
    end
    B = BlockTensor{T}(ls, flux(A))
    for (k, M) in eachblock(A)
        idx = ntuple(Val(N)) do i
            ranges[i] === nothing ? (1:size(M, i)) : get(ranges[i], k[i], nothing)
        end
        any(isnothing, idx) && continue
        any(isempty, idx) && continue
        getfield(B, :blocks)[k] = M[idx...]
    end
    return B
end

"""
    restrict(A::BlockTensor{T,2}, rows::Dict{label,range}, cols::Dict{label,range})

Rank-2 form keyed by labels (untagged legs), as in the pre-refactor layer.
"""
function restrict(A::BlockTensor{T,2,S}, rows::AbstractDict{S,UnitRange{Int}}, cols::AbstractDict{S,UnitRange{Int}}) where {T,S}
    return restrict(A, (Dict(LegSector(q) => r for (q, r) in rows), Dict(LegSector(q) => r for (q, r) in cols)))
end
