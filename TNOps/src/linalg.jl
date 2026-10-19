# Block-wise linear algebra on BlockTensors: arithmetic, leg permutations,
# contractions, traces and outer products. No dense intermediates.

# ---------------------------------------------------------------- arithmetic

function _checksame(A::BlockTensor, B::BlockTensor, what)
    ndims(A) == ndims(B) || throw(DimensionMismatch("$what: ranks $(ndims(A)) and $(ndims(B)) differ"))
    for i in 1:ndims(A)
        leg(A, i) == leg(B, i) || throw(DimensionMismatch("$what: leg $i differs: $(legstring(leg(A, i))) vs $(legstring(leg(B, i)))"))
    end
    flux(A) == flux(B) || throw(ArgumentError("cannot add tensors with fluxes $(flux(A)) and $(flux(B))"))
end

function Base.:+(A::BlockTensor{TA,N,S}, B::BlockTensor{TB,N,S}) where {TA,TB,N,S}
    _checksame(A, B, "+")
    T = promote_type(TA, TB)
    C = BlockTensor{T}(legs(A), flux(A))
    cb = getfield(C, :blocks)
    for (k, M) in eachblock(A)
        cb[k] = copy(convert(Array{T,N}, M))
    end
    for (k, M) in eachblock(B)
        cb[k] = haskey(cb, k) ? cb[k] + M : copy(convert(Array{T,N}, M))
    end
    return C
end

Base.:-(A::BlockTensor) = (-1) * A
Base.:-(A::BlockTensor, B::BlockTensor) = A + (-1) * B

function Base.:*(a::Number, A::BlockTensor{TA,N,S}) where {TA,N,S}
    T = promote_type(typeof(a), TA)
    return BlockTensor{T,N,S}(legs(A), flux(A), Dict(k => a * M for (k, M) in eachblock(A)))
end
Base.:*(A::BlockTensor, a::Number) = a * A
Base.:/(A::BlockTensor, a::Number) = inv(a) * A

Base.conj(A::BlockTensor{T,N,S}) where {T,N,S} =
    BlockTensor{T,N,S}(legs(A), flux(A), Dict(k => conj(M) for (k, M) in eachblock(A)))

"""
    adjoint(A)            (rank 2)
    adjoint(A, rows, cols)

Hermitian conjugate of `A` read as a map from the legs `cols` to the legs
`rows`: the result has legs `(dual.(legs[cols])..., dual.(legs[rows])...)`,
flux `dual(Δ)` and conjugated, permuted blocks. For rank 2 this is the
operator adjoint: legs swap (row ↔ column), block `q → q⊕Δ` becomes
`q⊕Δ → q`.
"""
function Base.adjoint(A::BlockTensor{T,2,S}) where {T,S}
    r, c = legs(A)
    C = BlockTensor{T}((dual(c), dual(r)), dual(flux(A)))
    cb = getfield(C, :blocks)
    for ((a, b), M) in eachblock(A)
        cb[(b, a)] = Matrix(M')
    end
    return C
end
function Base.adjoint(A::BlockTensor{T,N,S}, rows::NTuple{NR,Int}, cols::NTuple{NC,Int}) where {T,N,S,NR,NC}
    NR + NC == N && sort!([rows..., cols...]) == collect(1:N) ||
        throw(ArgumentError("adjoint: $rows and $cols must partition the legs 1:$N"))
    p = (cols..., rows...)
    ls = ntuple(i -> dual(leg(A, p[i])), Val(N))
    C = BlockTensor{T}(ls, dual(flux(A)))
    cb = getfield(C, :blocks)
    for (k, M) in eachblock(A)
        cb[ntuple(i -> k[p[i]], Val(N))] = conj(permutedims(M, p))
    end
    return C
end

"""    norm(A) — Frobenius norm over all blocks."""
LinearAlgebra.norm(A::BlockTensor) = sqrt(sum((sum(abs2, M) for M in values(getfield(A, :blocks))); init = 0.0))

"""
    dot(A, B) — `Σ conj(A) .* B` over all blocks (equal legs and flux).
"""
function LinearAlgebra.dot(A::BlockTensor, B::BlockTensor)
    _checksame(A, B, "dot")
    s = zero(promote_type(eltype(A), eltype(B)))
    for (k, M) in eachblock(A)
        N2 = blockornothing(B, k)
        N2 === nothing || (s += dot(M, N2))
    end
    return s
end

function Base.isapprox(A::BlockTensor, B::BlockTensor; atol::Real = 0, rtol::Real = atol > 0 ? 0 : sqrt(eps()))
    _checksame(A, B, "isapprox")
    return norm(A - B) ≤ max(atol, rtol * max(norm(A), norm(B)))
end

# ---------------------------------------------------------------- permutation

"""
    permute(A, perm) -> BlockTensor

Leg permutation: leg `i` of the result is leg `perm[i]` of `A`. Purely
bosonic: no fermionic sign is ever inserted (signs enter only through
[`apply_parity`](@ref)).
"""
function permute(A::BlockTensor{T,N,S}, p::NTuple{N,Int}) where {T,N,S}
    isperm(p) || throw(ArgumentError("$p is not a permutation of 1:$N"))
    ls = ntuple(i -> leg(A, p[i]), Val(N))
    C = BlockTensor{T}(ls, flux(A))
    cb = getfield(C, :blocks)
    for (k, M) in eachblock(A)
        cb[ntuple(i -> k[p[i]], Val(N))] = permutedims(M, p)
    end
    return C
end
permute(A::BlockTensor, p::AbstractVector{<:Integer}) = permute(A, Tuple(Int.(p)))

# ---------------------------------------------------------------- contraction

function _legname(A, i)
    n = legname(leg(A, i))
    return n == Symbol("") ? "leg $i" : "leg $i (:$n)"
end

"""
    contract(A, legsA, B, legsB) -> BlockTensor

Contract legs `legsA` of `A` with legs `legsB` of `B` (pairwise; each pair must
be dual, [`isdual`](@ref)). The result has the uncontracted legs of `A`, then
those of `B`, and the fused flux. Per matching block pair the blocks are
permuted (only if needed), reshaped to matrices and multiplied with `mul!`,
accumulating into the output block; output blocks are computed in parallel
over threads. Only block pairs whose contracted sectors agree are touched, so
no block the flux forbids is ever computed.
"""
# the legs of 1:N not in `ia`, in order (length known at compile time)
@inline function _complement(::Val{N}, ia::NTuple{NC,Int}) where {N,NC}
    return ntuple(Val(N - NC)) do k
        c = 0
        for i in 1:N
            i in ia && continue
            c += 1
            c == k && return i
        end
        return 0
    end
end

function contract(A::BlockTensor{TA,NA,S}, ia::NTuple{NC,Int}, B::BlockTensor{TB,NB,S}, ib::NTuple{NC,Int}) where {TA,NA,TB,NB,S,NC}
    for (x, y) in zip(ia, ib)
        isdual(leg(A, x), leg(B, y)) || throw(ArgumentError(
            "cannot contract $(_legname(A, x)) $(legstring(leg(A, x))) with $(_legname(B, y)) $(legstring(leg(B, y))): not dual"))
    end
    allunique(ia) && allunique(ib) || throw(ArgumentError("contracted legs repeat"))
    all(i -> 1 ≤ i ≤ NA, ia) && all(i -> 1 ≤ i ≤ NB, ib) || throw(ArgumentError("contracted leg out of range"))
    oa = _complement(Val(NA), ia); ob = _complement(Val(NB), ib)
    T = promote_type(TA, TB)
    ls = (map(i -> leg(A, i), oa)..., map(i -> leg(B, i), ob)...)
    C = BlockTensor{T}(ls, fuse(flux(A), flux(B)))
    _contract_blocks!(C, A, ia, oa, B, ib, ob)
    return C
end

function _contract_blocks!(C::BlockTensor{T,NO,S}, A::BlockTensor{TA,NA,S}, ia::NTuple{NC,Int}, oa::NTuple{MA,Int},
                           B::BlockTensor{TB,NB,S}, ib::NTuple{NC,Int}, ob::NTuple{MB,Int}) where {T,NO,S,TA,NA,TB,NB,NC,MA,MB}
    pa = (oa..., ia...); pb = (ib..., ob...)
    ida = pa == ntuple(identity, Val(NA)); idb = pb == ntuple(identity, Val(NB))
    # A blocks by contracted key, as matrices (open × contracted)
    Aby = Dict{NTuple{NC,LegSector{S}},Vector{Tuple{NTuple{MA,LegSector{S}},Matrix{TA}}}}()
    for (k, M) in eachblock(A)
        kc = map(i -> k[i], ia); ko = map(i -> k[i], oa)
        Mp = ida ? M : permutedims(M, pa)
        m = 1
        for i in oa
            m *= size(M, i)
        end
        push!(get!(() -> Tuple{NTuple{MA,LegSector{S}},Matrix{TA}}[], Aby, kc), (ko, reshape(Mp, m, :)))
    end
    # output keys with their contributing block pairs
    work = Dict{NTuple{NO,LegSector{S}},Vector{Tuple{Matrix{TA},Matrix{TB}}}}()
    for (k, M) in eachblock(B)
        kc = map(i -> k[i], ib)
        as = get(Aby, kc, nothing)
        as === nothing && continue
        ko = map(i -> k[i], ob)
        Mp = idb ? M : permutedims(M, pb)
        n = 1
        for i in ob
            n *= size(M, i)
        end
        Bm = reshape(Mp, :, n)
        for (koa, Am) in as
            push!(get!(() -> Tuple{Matrix{TA},Matrix{TB}}[], work, (koa..., ko...)), (Am, Bm))
        end
    end
    keysout = collect(keys(work))
    results = Vector{Array{T,NO}}(undef, length(keysout))
    ls = legs(C)
    flops = 0
    for v in values(work), (a, b) in v
        flops += size(a, 1) * size(a, 2) * size(b, 2)
    end
    if Threads.nthreads() > 1 && length(keysout) > 1 && flops > 2^18
        Threads.@threads for j in eachindex(keysout)
            results[j] = _blockjob(work[keysout[j]], keysout[j], ls, T)
        end
    else
        for j in eachindex(keysout)
            results[j] = _blockjob(work[keysout[j]], keysout[j], ls, T)
        end
    end
    cb = getfield(C, :blocks)
    for (j, kout) in enumerate(keysout)
        cb[kout] = results[j]
    end
    return C
end

function _blockjob(prs::Vector{Tuple{Matrix{TA},Matrix{TB}}}, kout::NTuple{NO,LegSector{S}}, ls::NTuple{NO,Leg{S}},
                   ::Type{T}) where {TA,TB,NO,S,T}
    m = size(prs[1][1], 1); n = size(prs[1][2], 2)
    Cm = zeros(T, m, n)
    for (Am, Bm) in prs
        mul!(Cm, Am, Bm, one(T), one(T))
    end
    return reshape(Cm, ntuple(i -> dim(ls[i], kout[i]), Val(NO)))
end
contract(A::BlockTensor, ia::AbstractVector{<:Integer}, B::BlockTensor, ib::AbstractVector{<:Integer}) =
    contract(A, Tuple(Int.(ia)), B, Tuple(Int.(ib)))
contract(A::BlockTensor, ia::Integer, B::BlockTensor, ib::Integer) = contract(A, (Int(ia),), B, (Int(ib),))

"""    A * B — rank-2 matrix product, `contract(A, (2,), B, (1,))`."""
Base.:*(A::BlockTensor{TA,2,S}, B::BlockTensor{TB,2,S}) where {TA,TB,S} = contract(A, (2,), B, (1,))

"""    outer(A, B) — the tensor product (legs of `A`, then of `B`)."""
outer(A::BlockTensor, B::BlockTensor) = contract(A, (), B, ())

# ---------------------------------------------------------------- traces

"""
    tr(A)

Trace of a rank-2 tensor whose column leg is the dual of its row leg:
`Σ degeneracy(label) tr(block)` over the diagonal blocks. A tensor with a
non-neutral flux has no diagonal block: its trace is an exact zero, returned
without touching any block.
"""
function LinearAlgebra.tr(A::BlockTensor{T,2}) where {T}
    isneutral(flux(A)) || return zero(T)
    isdual(leg(A, 1), leg(A, 2)) || throw(DimensionMismatch("trace needs dual row and column legs"))
    s = zero(T)
    for ((a, b), M) in eachblock(A)
        a == b && (s += degeneracy(a.label) * tr(M))
    end
    return s
end

"""
    tr(A, i, j) -> BlockTensor (or a number for rank 2)

Partial trace over the dual leg pair `(i, j)`; every traced state counts
`degeneracy(label)` times.
"""
function LinearAlgebra.tr(A::BlockTensor{T,N,S}, i::Integer, j::Integer) where {T,N,S}
    N == 2 && return tr(i < j ? A : permute(A, (2, 1)))
    i != j || throw(ArgumentError("trace over one leg"))
    isdual(leg(A, i), leg(A, j)) || throw(ArgumentError("cannot trace $(_legname(A, i)) with $(_legname(A, j)): not dual"))
    keep = Tuple(k for k in 1:N if k != i && k != j)
    ls = ntuple(k -> leg(A, keep[k]), N - 2)
    C = BlockTensor{T}(ls, flux(A))
    cb = getfield(C, :blocks)
    for (k, M) in eachblock(A)
        k[i] == k[j] || continue
        Mp = permutedims(M, (keep..., i, j))
        d = size(M, i)
        R = reshape(Mp, :, d * d)
        acc = zeros(T, size(R, 1))
        for a in 1:d
            acc .+= view(R, :, a + (a - 1) * d)
        end
        acc .*= degeneracy(k[i].label)
        ko = ntuple(m -> k[keep[m]], N - 2)
        B = reshape(acc, ntuple(m -> size(M, keep[m]), N - 2))
        cb[ko] = haskey(cb, ko) ? cb[ko] + B : B
    end
    return C
end
