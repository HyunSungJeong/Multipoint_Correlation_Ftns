# Sector spaces and block-sparse operators (task sheet Sec. 5.2.4).
#
# Everything above `symmetry/` talks to block data only through the functions
# of this file and of `fusion.jl` (Sec. 5.2.9).

"""
    SectorSpace{S}

An ordered list of `(label, dimension)` pairs with labels sorted, so that runs
are reproducible. Dimensions count block rows; the number of states a row
stands for is [`degeneracy`](@ref)`(label)` (1 for abelian labels).
"""
struct SectorSpace{S<:Sector}
    labels::Vector{S}
    dims::Vector{Int}
    index::Dict{S,Int}
    function SectorSpace{S}(labels::Vector{S}, dims::Vector{Int}) where {S}
        length(labels) == length(dims) || throw(DimensionMismatch("one dimension per label"))
        p = sortperm(labels)
        labels = labels[p]; dims = dims[p]
        allunique(labels) || throw(ArgumentError("duplicate sector labels"))
        keep = dims .> 0
        labels = labels[keep]; dims = dims[keep]
        return new{S}(labels, dims, Dict(q => i for (i, q) in enumerate(labels)))
    end
end

SectorSpace(labels::Vector{S}, dims::AbstractVector{<:Integer}) where {S<:Sector} =
    SectorSpace{S}(labels, collect(Int, dims))
SectorSpace(d::AbstractDict{S,<:Integer}) where {S<:Sector} =
    SectorSpace{S}(collect(keys(d)), collect(Int, values(d)))

sectors(V::SectorSpace) = V.labels
"""    dim(V, q) — block dimension of sector `q` in `V` (0 if absent)."""
dim(V::SectorSpace{S}, q::S) where {S} = (i = get(V.index, q, 0); i == 0 ? 0 : V.dims[i])
"""    dim(V) — total number of states, `Σ_q degeneracy(q) dim(q)`."""
dim(V::SectorSpace) = sum((degeneracy(q) * d for (q, d) in zip(V.labels, V.dims)); init = 0)
nblockrows(V::SectorSpace) = sum(V.dims; init = 0)
Base.haskey(V::SectorSpace{S}, q::S) where {S} = haskey(V.index, q)
Base.:(==)(V::SectorSpace, W::SectorSpace) = V.labels == W.labels && V.dims == W.dims
Base.hash(V::SectorSpace, h::UInt) = hash(V.dims, hash(V.labels, h))
Base.show(io::IO, V::SectorSpace) =
    print(io, "SectorSpace(", join(["$(q.q)=>$d" for (q, d) in zip(V.labels, V.dims)], ", "), ")")

"""
    offsets(V) -> Dict{S,Int}

Row offset of each sector in the dense embedding used by [`todense`](@ref): the
sectors are stacked in the sorted label order of `V`.
"""
function offsets(V::SectorSpace{S}) where {S}
    off = Dict{S,Int}(); o = 0
    for (q, d) in zip(V.labels, V.dims)
        off[q] = o; o += d
    end
    return off
end

"""
    BlockOp{T,S}

A block-sparse operator with a definite symmetry shift `Δ`: it maps sector `q`
of the column space to sector `q ⊕ Δ` of the row space. `blocks[q]` is the
dense `dim(q ⊕ Δ) × dim(q)` matrix; absent blocks are zero and not stored. `T`
is `Float64` unless the model or an operator is genuinely complex (no silent
promotion: mixing types in `+` or `*` promotes explicitly through `promote_type`
of the two operands only).
"""
struct BlockOp{T<:Number,S<:Sector}
    row::SectorSpace{S}
    col::SectorSpace{S}
    shift::S
    blocks::Dict{S,Matrix{T}}
end

"""    BlockOp{T}(row, col, shift) — the zero operator."""
BlockOp{T}(row::SectorSpace{S}, col::SectorSpace{S}, shift::S) where {T,S} =
    BlockOp{T,S}(row, col, shift, Dict{S,Matrix{T}}())

Base.eltype(::BlockOp{T}) where {T} = T
shift(A::BlockOp) = A.shift
rowspace(A::BlockOp) = A.row
colspace(A::BlockOp) = A.col
Base.size(A::BlockOp) = (nblockrows(A.row), nblockrows(A.col))

"""    target(A, q) — the row sector `q ⊕ Δ` reached from column sector `q`."""
target(A::BlockOp{T,S}, q::S) where {T,S} = fuse(q, A.shift)

"""
    getblock(A, q) -> Matrix

Block of `A` acting on column sector `q`; a zero matrix of the right size if
absent (or a 0×0 matrix if either sector is missing from its space).
"""
function getblock(A::BlockOp{T,S}, q::S) where {T,S}
    b = get(A.blocks, q, nothing)
    b === nothing || return b
    return zeros(T, dim(A.row, target(A, q)), dim(A.col, q))
end

"""
    setblock!(A, q, M)

Store block `M` for column sector `q`, checking its size against the spaces.
Blocks of a sector missing from either space must be empty.
"""
function setblock!(A::BlockOp{T,S}, q::S, M::AbstractMatrix) where {T,S}
    r = target(A, q)
    size(M) == (dim(A.row, r), dim(A.col, q)) ||
        throw(DimensionMismatch("block for $q must be $(dim(A.row, r))×$(dim(A.col, q)), got $(size(M))"))
    isempty(M) && return A
    A.blocks[q] = convert(Matrix{T}, M)
    return A
end

function Base.:*(A::BlockOp{TA,S}, B::BlockOp{TB,S}) where {TA,TB,S}
    A.col == B.row || throw(DimensionMismatch("inner sector spaces differ"))
    T = promote_type(TA, TB)
    C = BlockOp{T}(A.row, B.col, fuse(A.shift, B.shift))
    for (q, Bq) in B.blocks
        r = target(B, q)
        Ar = get(A.blocks, r, nothing)
        Ar === nothing && continue
        C.blocks[q] = Ar * Bq
    end
    return C
end

function Base.:+(A::BlockOp{TA,S}, B::BlockOp{TB,S}) where {TA,TB,S}
    A.row == B.row && A.col == B.col || throw(DimensionMismatch("sector spaces differ"))
    A.shift == B.shift || throw(ArgumentError("cannot add operators with shifts $(A.shift) and $(B.shift)"))
    T = promote_type(TA, TB)
    C = BlockOp{T}(A.row, A.col, A.shift)
    for (q, M) in A.blocks
        C.blocks[q] = copy(convert(Matrix{T}, M))
    end
    for (q, M) in B.blocks
        C.blocks[q] = haskey(C.blocks, q) ? C.blocks[q] + M : copy(convert(Matrix{T}, M))
    end
    return C
end

Base.:-(A::BlockOp) = (-1) * A
Base.:-(A::BlockOp, B::BlockOp) = A + (-1) * B

function Base.:*(a::Number, A::BlockOp{TA,S}) where {TA,S}
    T = promote_type(typeof(a), TA)
    return BlockOp{T,S}(A.row, A.col, A.shift, Dict(q => a * M for (q, M) in A.blocks))
end
Base.:*(A::BlockOp, a::Number) = a * A

"""
    adjoint(A)

Hermitian conjugate: row and column spaces swap, the shift is dualized and every
block is conjugate-transposed (block `q → q⊕Δ` becomes block `q⊕Δ → q`).
"""
function Base.adjoint(A::BlockOp{T,S}) where {T,S}
    C = BlockOp{T}(A.col, A.row, dual(A.shift))
    for (q, M) in A.blocks
        C.blocks[target(A, q)] = Matrix(M')
    end
    return C
end

"""
    tr(A)

Trace `Σ_q degeneracy(q) tr A_q`; zero unless `A` is neutral (a block of a
non-neutral operator never sits on the diagonal).
"""
function LinearAlgebra.tr(A::BlockOp{T}) where {T}
    isneutral(A.shift) || return zero(T)
    A.row == A.col || throw(DimensionMismatch("trace needs equal row and column spaces"))
    return sum((degeneracy(q) * tr(M) for (q, M) in A.blocks); init = zero(T))
end

"""
    blockeigen(H) -> Dict{S, (values, vectors)}

Block-wise eigendecomposition of a hermitian neutral operator. Eigenvalues are
tagged with their sector through the dictionary key; within a sector they are
ascending. Sectors without a stored block have all eigenvalues zero.
"""
function blockeigen(H::BlockOp{T,S}) where {T,S}
    isneutral(H.shift) || throw(ArgumentError("eigen needs a neutral operator"))
    H.row == H.col || throw(DimensionMismatch("eigen needs equal row and column spaces"))
    out = Dict{S,Tuple{Vector{Float64},Matrix{T}}}()
    for (q, d) in zip(H.col.labels, H.col.dims)
        M = getblock(H, q)
        F = eigen(Hermitian((M + M') / 2))
        out[q] = (F.values, F.vectors)
    end
    return out
end

"""
    blocksvd(A) -> Dict{S, SVD}

Block-wise singular value decomposition, keyed by column sector.
"""
blocksvd(A::BlockOp{T,S}) where {T,S} = Dict{S,SVD}(q => svd(M) for (q, M) in A.blocks)

"""
    todense(A) -> Matrix

Embed `A` into an explicit matrix. Rows and columns are ordered by stacking the
sectors of the row and column spaces in their sorted label order
([`offsets`](@ref)). For tests only.
"""
function todense(A::BlockOp{T}) where {T}
    ro = offsets(A.row); co = offsets(A.col)
    D = zeros(T, nblockrows(A.row), nblockrows(A.col))
    for (q, M) in A.blocks
        r = target(A, q)
        D[ro[r] .+ (1:size(M, 1)), co[q] .+ (1:size(M, 2))] .= M
    end
    return D
end

"""
    identity_op(V, T = Float64)

The identity on `V` as a neutral `BlockOp`.
"""
function identity_op(V::SectorSpace{S}, ::Type{T} = Float64) where {S,T}
    A = BlockOp{T}(V, V, neutral(S))
    for (q, d) in zip(V.labels, V.dims)
        A.blocks[q] = Matrix{T}(I, d, d)
    end
    return A
end

"""    norm(A) — Frobenius norm over all blocks."""
LinearAlgebra.norm(A::BlockOp) = sqrt(sum((sum(abs2, M) for M in values(A.blocks)); init = 0.0))

"""
    restrict(A, rows, cols) -> BlockOp

Restrict `A` to subspaces: `rows[r]` and `cols[q]` are index ranges into the
blocks of sector `r` (row) and `q` (column). Sectors absent from the
dictionaries are dropped. The result lives on the subspaces with dimensions
`length(rows[r])`, `length(cols[q])`.
"""
function restrict(A::BlockOp{T,S}, rows::Dict{S,UnitRange{Int}}, cols::Dict{S,UnitRange{Int}}) where {T,S}
    rs = SectorSpace(Dict(q => length(r) for (q, r) in rows))
    cs = SectorSpace(Dict(q => length(r) for (q, r) in cols))
    C = BlockOp{T}(rs, cs, A.shift)
    for (q, M) in A.blocks
        r = target(A, q)
        (haskey(rows, r) && haskey(cols, q)) || continue
        rr = rows[r]; cc = cols[q]
        (isempty(rr) || isempty(cc)) && continue
        C.blocks[q] = M[rr, cc]
    end
    return C
end

"""
    SlicedBlockOp{T,S,Kt}

A block-sparse operator with one or more open frequency legs (Lee Eq. (60) and
Sec. V D): for every slice key `k::Kt` (a bin index, a tuple of bin indices for
two open legs, or an exact energy in Lehmann mode) it stores a [`BlockOp`](@ref)
on common spaces with a common shift. The slices are kept as a sparse map from
key to operator: only bins that receive weight are stored, which is the
measured fill of every sliced object built by the PSF engine (it is reported in
the PSF metadata). Contractions over the symmetry legs act slice by slice; the
bin legs pass through untouched ([`mapslices_op`](@ref)).
"""
struct SlicedBlockOp{T<:Number,S<:Sector,Kt}
    row::SectorSpace{S}
    col::SectorSpace{S}
    shift::S
    slices::Dict{Kt,BlockOp{T,S}}
end

SlicedBlockOp{T,S,Kt}(row::SectorSpace{S}, col::SectorSpace{S}, shift::S) where {T,S,Kt} =
    SlicedBlockOp{T,S,Kt}(row, col, shift, Dict{Kt,BlockOp{T,S}}())

nslices(A::SlicedBlockOp) = length(A.slices)

"""
    mapslices_op(f, A::SlicedBlockOp) -> SlicedBlockOp

Apply a contraction `f` over the symmetry legs to every slice, leaving the bin
legs untouched. All slices must map to operators on common spaces.
"""
function mapslices_op(f, A::SlicedBlockOp{T,S,Kt}) where {T,S,Kt}
    out = Dict{Kt,BlockOp{T,S}}()
    for (k, B) in A.slices
        out[k] = f(B)
    end
    isempty(out) && return A
    B1 = first(values(out))
    return SlicedBlockOp{T,S,Kt}(B1.row, B1.col, B1.shift, out)
end
