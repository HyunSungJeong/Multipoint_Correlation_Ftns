# Dense embeddings (tests, references, small problems).
#
# Index map: on leg `l`, sector `s` occupies the dense indices
# `offsets(l)[s] .+ (1:dim(l, s))`, the sectors stacked in their sorted
# `(label, tag)` order; inside a block the indices keep their order. A block
# with key `(s₁, …, s_N)` is therefore the dense sub-array
# `D[offsets(l₁)[s₁] .+ (1:d₁), …, offsets(l_N)[s_N] .+ (1:d_N)]`.

"""
    todense(A) -> Array{T,N}

Embed `A` into a dense array with the index map above (for rank 2: sectors of
the row and column legs stacked in sorted order, as for the pre-refactor
`BlockOp`).
"""
function todense(A::BlockTensor{T,N}) where {T,N}
    offs = map(offsets, legs(A))
    D = zeros(T, map(nblockrows, legs(A)))
    for (k, M) in eachblock(A)
        idx = ntuple(i -> offs[i][k[i]] .+ (1:size(M, i)), Val(N))
        copyto!(view(D, idx...), M)
    end
    return D
end

"""
    fromdense(D, legs, flux; atol = 0) -> BlockTensor

The block tensor with dense embedding `D` (index map above). Only blocks the
block rule allows are stored (and only if nonzero); any entry with
`|D[i]| > atol` outside the allowed blocks is an error.
"""
function fromdense(D::AbstractArray{T,N}, ls::NTuple{N,Leg{S}}, Δ::S; atol::Real = 0) where {T,N,S}
    size(D) == map(nblockrows, ls) || throw(DimensionMismatch("dense array of size $(size(D)) for legs of size $(map(nblockrows, ls))"))
    offs = map(offsets, ls)
    A = BlockTensor{T}(ls, Δ)
    covered = falses(size(D))
    for k in allowed_keys(ls, Δ)
        idx = ntuple(i -> offs[i][k[i]] .+ (1:dim(ls[i], k[i])), Val(N))
        view(covered, idx...) .= true
        M = copy(view(D, idx...))
        any(x -> abs(x) > atol, M) && setblock!(A, k, M)
    end
    for I in CartesianIndices(D)
        covered[I] && continue
        abs(D[I]) > atol && throw(ArgumentError("fromdense: weight $(D[I]) at $(Tuple(I)) lies outside the blocks allowed by the flux $Δ"))
    end
    return A
end
