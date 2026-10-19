# Fusion of two legs into one (and its inverse), and Kronecker products of
# operators on fused legs.
#
# Layout (unchanged from the pre-refactor NRG fusion map): fusing an `outer`
# leg (slow index σ) with an `inner` leg (fast index s), the fused sector
# `q′ = σ ⊕ q` collects the pairs `(σ, q)` in the order "outer sectors
# sorted, then inner sectors sorted"; pair `(σ, q)` occupies the fused indices
# `offset .+ (1:dσ*dq)`, and the product state `(iσ, s)` has index
# `offset + (iσ-1)*dq + s`, i.e. `kron(e_iσ, e_s)`. Read as a tensor index this
# is column-major order over (inner, outer): fusing legs `(a, b)` of a tensor
# with `fuse_legs(A, (a, b))` makes `a` the inner (fast) and `b` the outer
# (slow) leg.

"""
    FusionPair(σ, q, offset, dσ, dq)

One contribution `(σ, q)` (outer sector σ, inner sector q) to a fused sector,
occupying the fused indices `offset .+ (1:dσ*dq)`; the product state `(iσ, s)`
has index `offset + (iσ-1)*dq + s`.
"""
struct FusionPair{S<:Sector}
    σ::LegSector{S}
    q::LegSector{S}
    offset::Int
    dσ::Int
    dq::Int
end

"""
    FusionMap

The fusion of an `outer` and an `inner` leg into the leg `out`: for every
fused sector the ordered pairs with their offsets, and
`lookup[(σ, q)] = (fused sector, pair index)`. The fused sectors are
untagged. (`F.loc` and `F.prev` are accepted for `outer` and `inner`, the
names of the NRG use: local site and previous shell.)
"""
struct FusionMap{S<:Sector}
    outer::Leg{S}
    inner::Leg{S}
    out::Leg{S}
    pairs::Dict{LegSector{S},Vector{FusionPair{S}}}
    lookup::Dict{Tuple{LegSector{S},LegSector{S}},Tuple{LegSector{S},Int}}
end

function Base.getproperty(F::FusionMap, s::Symbol)
    s === :loc && return getfield(F, :outer)
    s === :prev && return getfield(F, :inner)
    return getfield(F, s)
end

"""
    fusionmap(outer::Leg, inner::Leg) -> FusionMap

Fuse two legs with the same arrow; the fused leg has that arrow and the fused
labels `σ ⊕ q`. Pairs are enumerated outer sectors first (sorted), then inner
sectors (sorted), which reproduces the pre-refactor NRG product basis.
"""
function fusionmap(Vo::Leg{S}, Vi::Leg{S}) where {S}
    arrow(Vo) == arrow(Vi) || throw(ArgumentError("fusing legs with different arrows ($(arrow(Vo)), $(arrow(Vi)))"))
    pairs = Dict{LegSector{S},Vector{FusionPair{S}}}()
    dims = Dict{LegSector{S},Int}()
    lookup = Dict{Tuple{LegSector{S},LegSector{S}},Tuple{LegSector{S},Int}}()
    for (σ, dσ) in zip(legsectors(Vo), Vo.dims), (q, dq) in zip(legsectors(Vi), Vi.dims)
        qq = LegSector(fuse(σ.label, q.label), 0)
        off = get(dims, qq, 0)
        v = get!(() -> FusionPair{S}[], pairs, qq)
        push!(v, FusionPair{S}(σ, q, off, dσ, dq))
        lookup[(σ, q)] = (qq, length(v))
        dims[qq] = off + dσ * dq
    end
    return FusionMap{S}(Vo, Vi, Leg(dims; arrow = arrow(Vo)), pairs, lookup)
end

"""
    kron_fused(X, Y, Frow, Fcol) -> BlockTensor

The rank-2 operator `X ⊗ Y` on fused legs: `X` acts on the outer leg, `Y` on
the inner leg; `Fcol` is the fusion of the column (input) legs and `Frow` of
the row (output) legs. The flux is `ΔX ⊕ ΔY`. Each block is a Kronecker product
placed at the pair offsets (the pre-refactor `kron_op`, block for block).
"""
function kron_fused(X::BlockTensor{TX,2,S}, Y::BlockTensor{TY,2,S}, Frow::FusionMap{S}, Fcol::FusionMap{S}) where {TX,TY,S}
    samespace(leg(X, 2), Fcol.outer) && samespace(leg(Y, 2), Fcol.inner) ||
        throw(DimensionMismatch("input legs do not match the column fusion map"))
    samespace(leg(X, 1), Frow.outer) && samespace(leg(Y, 1), Frow.inner) ||
        throw(DimensionMismatch("output legs do not match the row fusion map"))
    T = promote_type(TX, TY)
    Δ = fuse(flux(X), flux(Y))
    R = BlockTensor{T}(Frow.out, Fcol.out, Δ)
    # X and Y blocks by column sector (rank-2 operators: at most one block per
    # column sector and row label; tagged rows are handled per row sector)
    Xc = Dict{LegSector{S},Vector{Tuple{LegSector{S},Matrix{TX}}}}()
    for ((a, b), M) in eachblock(X)
        push!(get!(() -> Tuple{LegSector{S},Matrix{TX}}[], Xc, b), (a, M))
    end
    Yc = Dict{LegSector{S},Vector{Tuple{LegSector{S},Matrix{TY}}}}()
    for ((a, b), M) in eachblock(Y)
        push!(get!(() -> Tuple{LegSector{S},Matrix{TY}}[], Yc, b), (a, M))
    end
    rb = getfield(R, :blocks)
    for (qin, prs) in Fcol.pairs
        qout = LegSector(target(R, qin.label), 0)
        haskey(Frow.pairs, qout) || continue
        blk = nothing
        for p in prs
            xs = get(Xc, p.σ, nothing); xs === nothing && continue
            ys = get(Yc, p.q, nothing); ys === nothing && continue
            for (σ2, Xb) in xs, (q2, Yb) in ys
                hit = get(Frow.lookup, (σ2, q2), nothing)
                hit === nothing && continue
                qq, k = hit
                @assert qq == qout
                p2 = Frow.pairs[qout][k]
                if blk === nothing
                    blk = zeros(T, dim(Frow.out, qout), dim(Fcol.out, qin))
                end
                blk[p2.offset .+ (1:(p2.dσ * p2.dq)), p.offset .+ (1:(p.dσ * p.dq))] .+= kron(Xb, Yb)
            end
        end
        blk === nothing || (rb[(qout, qin)] = blk)
    end
    return R
end

"""
    fuse_legs(A, (a, b)) -> (A′, F::FusionMap)

Fuse legs `a` (inner, fast) and `b` (outer, slow) of `A` into one leg placed
at position `min(a, b)`; the other legs keep their order. `F` is the fusion map
(`fusionmap(leg(A, b), leg(A, a))`). Both legs must have the same arrow.
"""
function fuse_legs(A::BlockTensor{T,N,S}, ab::Tuple{Int,Int}) where {T,N,S}
    a, b = ab
    a != b || throw(ArgumentError("fuse_legs needs two different legs"))
    F = fusionmap(leg(A, b), leg(A, a))
    p = min(a, b)
    rest = Tuple(i for i in 1:N if i != a && i != b)
    before = Tuple(i for i in rest if i < p); after = Tuple(i for i in rest if i > p)
    perm = (before..., a, b, after...)
    ls = (map(i -> leg(A, i), before)..., F.out, map(i -> leg(A, i), after)...)
    C = BlockTensor{T}(ls, flux(A))
    cb = getfield(C, :blocks)
    nb = length(before)
    for (k, M) in eachblock(A)
        qq, ip = F.lookup[(k[b], k[a])]
        pr = F.pairs[qq][ip]
        kout = (map(i -> k[i], before)..., qq, map(i -> k[i], after)...)
        blk = get!(cb, kout) do
            zeros(T, ntuple(i -> dim(ls[i], kout[i]), N - 1))
        end
        Mp = permutedims(M, perm)
        sz = size(Mp)
        Mr = reshape(Mp, (sz[1:nb]..., sz[nb + 1] * sz[nb + 2], sz[(nb + 3):end]...))
        idx = ntuple(i -> i == nb + 1 ? (pr.offset .+ (1:(pr.dσ * pr.dq))) : Colon(), N - 1)
        view(blk, idx...) .= Mr
    end
    return C, F
end

"""
    split_legs(A′, i, F) -> A

Inverse of [`fuse_legs`](@ref): split leg `i` of `A′` (the fused leg of `F`)
into the inner and the outer leg of `F`, at positions `i` and `i+1`. All-zero
sub-blocks are not stored.
"""
function split_legs(A::BlockTensor{T,N,S}, i::Int, F::FusionMap{S}) where {T,N,S}
    samespace(leg(A, i), F.out) || throw(DimensionMismatch("leg $i is not the fused leg of the fusion map"))
    # the split legs take the arrow of the leg being split
    flipped = arrow(leg(A, i)) != arrow(F.out)
    li = flipped ? dual(F.inner) : F.inner; lo = flipped ? dual(F.outer) : F.outer
    ls = (ntuple(j -> leg(A, j), i - 1)..., li, lo, ntuple(j -> leg(A, i + j), N - i)...)
    C = BlockTensor{T}(ls, flux(A))
    cb = getfield(C, :blocks)
    for (k, M) in eachblock(A)
        for pr in F.pairs[k[i]]
            idx = ntuple(j -> j == i ? (pr.offset .+ (1:(pr.dσ * pr.dq))) : Colon(), N)
            sub = M[idx...]
            all(iszero, sub) && continue
            sz = size(sub)
            B = reshape(sub, (sz[1:(i - 1)]..., pr.dq, pr.dσ, sz[(i + 1):end]...))
            cb[(k[1:(i - 1)]..., pr.q, pr.σ, k[(i + 1):end]...)] = B
        end
    end
    return C
end
