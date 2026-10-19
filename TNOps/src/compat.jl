# The names of the pre-refactor rank-2 layer of MultipointNRG (at bb8b197),
# kept for compatibility: aliases of the generic types and thin wrappers.
# Semantics as before, except that the blocks of a tensor are keyed by
# sector tuples (`eachblock`), not by column label: label-keyed access goes
# through `getblock(A, q)` / `setblock!(A, q, M)`.

"""
    SectorSpace{S} = Leg{S}
    SectorSpace(dims::Dict{S,Int}) -> Leg{S}

The pre-refactor sector space: an untagged `Out` leg with the given label
dimensions (`sectors`, `dim(V, q)`, `dim(V)`, `offsets`, `haskey` as before).
"""
const SectorSpace = Leg

"""
    BlockOp{T,S} = BlockTensor{T,2,S}
    BlockOp{T}(row, col, Δ)

The pre-refactor block operator: a rank-2 block tensor with legs (row `Out`,
column `In`) and flux `Δ` (`A.shift`, `A.row`, `A.col`; `target`, the
column-label keyed `getblock`/`setblock!`, `todense` in sorted sector order).
"""
const BlockOp{T<:Number,S<:Sector} = BlockTensor{T,2,S}
BlockTensor{T,2}(row::Leg{S}, col::Leg{S}, Δ::S) where {T,S} = BlockTensor{T}(row, col, Δ)
BlockTensor{T,2,S}(row::Leg{S}, col::Leg{S}, Δ::S) where {T,S} = BlockTensor{T}(row, col, Δ)
# the pre-refactor 4-argument form: blocks keyed by column label
function BlockTensor{T,2,S}(row::Leg{S}, col::Leg{S}, Δ::S, blocks::AbstractDict{S,<:AbstractMatrix}) where {T,S}
    A = BlockTensor{T}(row, col, Δ)
    for (q, M) in blocks
        setblock!(A, q, M)
    end
    return A
end

# the column labels that carry a stored block
_collabels(A::BlockTensor{T,2}) where {T} = sort!(unique(k[2].label for k in blockkeys(A)))

"""
    blockeigen(H) -> Dict{S, (values, vectors)}

Eigendecomposition of a hermitian neutral rank-2 tensor per column label (all
tags of a label together), ascending within a label; labels without a
stored block have all eigenvalues zero. See also [`eigen`](@ref).
"""
function blockeigen(H::BlockTensor{T,2,S}) where {T,S}
    isneutral(flux(H)) || throw(ArgumentError("eigen needs a neutral operator"))
    samespace(leg(H, 1), leg(H, 2)) || throw(DimensionMismatch("eigen needs equal row and column spaces"))
    out = Dict{S,Tuple{Vector{Float64},Matrix{T}}}()
    for q in sectors(leg(H, 2))
        M = getblock(H, q)
        F = _heigen(Hermitian((M + M') / 2))
        out[q] = (F.values, F.vectors)
    end
    return out
end

"""
    blocksvd(A) -> Dict{S, SVD}

Singular value decomposition of every stored label block (keyed by column
label). See also [`svd`](@ref).
"""
blocksvd(A::BlockTensor{T,2,S}) where {T,S} = Dict{S,SVD}(q => svd(getblock(A, q)) for q in _collabels(A))

"""    identity_op(V, T = Float64) — the identity on `V` ([`identity_tensor`](@ref))."""
identity_op(V::Leg, ::Type{T} = Float64) where {T} = identity_tensor(V, T)

"""
    SlicedBlockOp{T,S,Kt} = SlicedTensor{Kt,T,2,S}

The pre-refactor sliced operator: a [`SlicedTensor`](@ref) of rank-2 tensors.
"""
const SlicedBlockOp{T<:Number,S<:Sector,Kt} = SlicedTensor{Kt,T,2,S}

"""    mapslices_op(f, A) — `mapslices(f, A)` of a sliced tensor."""
mapslices_op(f, A::SlicedTensor) = mapslices(f, A)

"""
    kron_op(X, Y, Frow, Fcol)

The pre-refactor name of [`kron_fused`](@ref): `X ⊗ Y` on the fused legs.
"""
kron_op(X::BlockTensor{TX,2,S}, Y::BlockTensor{TY,2,S}, Frow::FusionMap{S}, Fcol::FusionMap{S}) where {TX,TY,S} =
    kron_fused(X, Y, Frow, Fcol)
