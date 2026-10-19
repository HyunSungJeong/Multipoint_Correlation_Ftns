# Factorizations on a leg partition, and the global degeneracy-safe cut.
#
# A tensor is read as a matrix from its column legs to its row legs; multiple
# legs on either side are fused first (fuse_legs, outermost leg slowest) and
# split again afterwards. For rank 2 nothing is fused. The tensor must be in
# operator form: row legs `Out`, column legs `In` (use `dual` / `permute`
# otherwise); the new bond legs are then `In` on the left factor.

function _check_operator_form(A::BlockTensor{T,2}, what) where {T}
    arrow(leg(A, 1)) == Out && arrow(leg(A, 2)) == In ||
        throw(ArgumentError("$what needs operator form: row legs Out, column legs In (got $(arrow(leg(A, 1))), $(arrow(leg(A, 2))))"))
end

# reduce A to a rank-2 tensor (rows, cols) by permuting and fusing; returns the
# matrix and a closure that splits a leg of the result back
function _asmatrix(A::BlockTensor{T,N,S}, rows::NTuple{NR,Int}, cols::NTuple{NC,Int}) where {T,N,S,NR,NC}
    NR + NC == N && sort!([rows..., cols...]) == collect(1:N) ||
        throw(ArgumentError("rows $rows and cols $cols must partition the legs 1:$N"))
    B = permute(A, (rows..., cols...))
    rmaps = FusionMap{S}[]; cmaps = FusionMap{S}[]
    # fuse the column legs (positions NR+1 … N) right to left, then the rows
    nc = NC
    while nc > 1
        B, F = fuse_legs(B, (NR + nc - 1, NR + nc))
        push!(cmaps, F); nc -= 1
    end
    nr = NR
    while nr > 1
        B, F = fuse_legs(B, (nr - 1, nr))
        push!(rmaps, F); nr -= 1
    end
    return B, rmaps, cmaps
end

function _splitback(X::BlockTensor, i::Int, maps)
    for F in reverse(maps)
        X = split_legs(X, i, F)
    end
    return X
end

"""
    eigen(H) / eigen(H, rows, cols) -> (values, U)

Eigendecomposition of a hermitian tensor with neutral flux, read as a map from
the legs `cols` to the legs `rows` (rank 2: rows `(1,)`, cols `(2,)`).
`values[b]` are the ascending eigenvalues of bond sector `b` (sector-tagged
spectra); `U` has the row legs and a new bond leg (`In`, one untagged sector
per row label, with the number of eigenvalues as dimension), so that
`H = U Λ U†`. All tags of a label are diagonalized together (a neutral tensor
couples them); each label block is symmetrized as `(M + M')/2` before the
dense `eigen`, and labels are diagonalized in parallel over threads. Labels
without a stored block have all eigenvalues zero.
"""
function LinearAlgebra.eigen(H::BlockTensor{T,2,S}) where {T,S}
    _check_operator_form(H, "eigen")
    isneutral(flux(H)) || throw(ArgumentError("eigen needs a neutral tensor"))
    isdual(leg(H, 1), leg(H, 2)) || throw(DimensionMismatch("eigen needs dual row and column legs"))
    row = leg(H, 1)
    qs = sectors(row)                    # labels: the tags of a label are diagonalized together
    vals = Vector{Vector{Float64}}(undef, length(qs))
    vecs = Vector{Matrix{T}}(undef, length(qs))
    Threads.@threads for i in eachindex(qs)
        M = getblock(H, qs[i])
        F = eigen(Hermitian((M + M') / 2))
        vals[i] = F.values; vecs[i] = F.vectors
    end
    bs = [LegSector(q) for q in qs]
    bond = Leg(bs, [length(v) for v in vals]; arrow = In, name = :bond)
    U = BlockTensor{T}((row, bond), neutral(S))
    for (i, q) in enumerate(qs)
        _setlabelrows!(U, q, bs[i], vecs[i])
    end
    return Dict(zip(bs, vals)), U
end
function LinearAlgebra.eigen(H::BlockTensor{T,N,S}, rows::NTuple{NR,Int}, cols::NTuple{NC,Int}) where {T,N,S,NR,NC}
    B, rmaps, cmaps = _asmatrix(H, rows, cols)
    vals, U = eigen(B)
    return vals, _splitback(U, 1, rmaps)
end

"""
    svd(A) / svd(A, rows, cols) -> (U, s, V)

Singular value decomposition per label block: `A = U Σ V†` with `U` (row legs,
bond `In`), `s[b]` the singular values of bond sector `b` (descending) and `V`
(column legs as `Out`, bond `In`), so that `V†` maps the columns to the bond.
The bond sectors carry the row labels (untagged); for a rank-2 `A` all tags of
a label pair are decomposed together.
"""
function LinearAlgebra.svd(A::BlockTensor{T,2,S}) where {T,S}
    _check_operator_form(A, "svd")
    rl, cl = legs(A)
    us = Dict{S,Matrix{T}}(); ss = Dict{S,Vector{Float64}}(); vs = Dict{S,Matrix{T}}()
    for q in sectors(cl)
        r = target(A, q)
        haskey(rl, r) || continue
        M = getblock(A, q)
        F = svd(M)
        us[r] = F.U; ss[r] = F.S; vs[r] = F.V
    end
    bond = Leg(Dict(LegSector(r) => length(s) for (r, s) in ss); arrow = In, name = :bond)
    U = BlockTensor{T}((rl, bond), neutral(S))
    V = BlockTensor{T}((dual(cl), bond), dual(flux(A)))
    for (r, Ur) in us
        b = LegSector(r)
        _setlabelrows!(U, r, b, Ur)
        q = target(V, r)                                   # column label of the block
        _setlabelrows!(V, q, b, vs[r])
    end
    return U, Dict(LegSector(r) => s for (r, s) in ss), V
end

# store the rows of M (all tags of label r on leg 1 stacked) as the blocks (s, b)
function _setlabelrows!(X::BlockTensor{T,2,S}, r::S, b::LegSector{S}, M::AbstractMatrix) where {T,S}
    size(M, 2) == 0 && return X
    o = 0
    for s in legsectors(leg(X, 1))
        s.label == r || continue
        d = dim(leg(X, 1), s)
        setblock!(X, (s, b), M[o .+ (1:d), :])
        o += d
    end
    return X
end

function LinearAlgebra.svd(A::BlockTensor{T,N,S}, rows::NTuple{NR,Int}, cols::NTuple{NC,Int}) where {T,N,S,NR,NC}
    B, rmaps, cmaps = _asmatrix(A, rows, cols)
    U, s, V = svd(B)
    return _splitback(U, 1, rmaps), s, _splitback(V, 1, cmaps)
end

"""
    qr(A) / qr(A, rows, cols) -> (Q, R)

Thin QR decomposition per label block, `A = Q R`: `Q` has the row legs and a
bond leg (`In`, untagged, labelled by the row labels), `R` the bond (`Out`)
and the column legs.
"""
function LinearAlgebra.qr(A::BlockTensor{T,2,S}) where {T,S}
    _check_operator_form(A, "qr")
    rl, cl = legs(A)
    qs = Dict{S,Matrix{T}}(); rs = Dict{S,Matrix{T}}(); qcol = Dict{S,S}()
    for q in sectors(cl)
        r = target(A, q)
        haskey(rl, r) || continue
        F = qr(getblock(A, q))
        qs[r] = Matrix(F.Q); rs[r] = Matrix(F.R); qcol[r] = q
    end
    bond = Leg(Dict(LegSector(r) => size(Q, 2) for (r, Q) in qs); arrow = In, name = :bond)
    Q = BlockTensor{T}((rl, bond), neutral(S))
    R = BlockTensor{T}((dual(bond), cl), flux(A))
    for (r, Qm) in qs
        b = LegSector(r)
        _setlabelrows!(Q, r, b, Qm)
        Rm = rs[r]; o = 0
        for s in legsectors(cl)
            s.label == qcol[r] || continue
            d = dim(cl, s)
            setblock!(R, (b, s), Rm[:, o .+ (1:d)])
            o += d
        end
    end
    return Q, R
end
function LinearAlgebra.qr(A::BlockTensor{T,N,S}, rows::NTuple{NR,Int}, cols::NTuple{NC,Int}) where {T,N,S,NR,NC}
    B, rmaps, cmaps = _asmatrix(A, rows, cols)
    Q, R = qr(B)
    return _splitback(Q, 1, rmaps), _splitback(R, 2, cmaps)
end

# ---------------------------------------------------------------- global cut

degeneracy(s::LegSector) = degeneracy(s.label)

"""
    global_cut(spectra; nkeep, threshold = nothing, gaptol) -> (kept, record)

The global cut over sector-tagged spectra (`Dict(sector => ascending values)`):

  1. all values are sorted together, each counted `degeneracy(sector)` times;
  2. the cut is placed at the `nkeep`-th value, or at the `threshold` (the
     number of values `≤ threshold`, at least one) if that keeps fewer;
  3. it is moved **up** into the nearest gap larger than `gaptol`, so that no
     degenerate multiplet is split, also when it is spread over sectors.

Returns the number of kept values per sector (the lowest ones of each sector)
and the record `(requested, actual, gap, cut_moved, total)` with the absolute
gap at the cut (`NaN` if everything is kept).
"""
function global_cut(E::AbstractDict{K,<:AbstractVector{<:Real}}; nkeep::Integer, threshold = nothing,
                    gaptol::Real) where {K}
    ks = sort!(collect(keys(E)))
    all_e = Float64[]
    for q in ks
        for e in E[q]
            for _ in 1:degeneracy(q)
                push!(all_e, e)
            end
        end
    end
    sort!(all_e)
    ntot = length(all_e)
    want = min(nkeep, ntot)
    if threshold !== nothing
        want = min(want, count(≤(threshold), all_e))
        want = max(want, 1)
    end
    want ≥ ntot && return Dict(q => length(E[q]) for q in ks),
                          (requested = want, actual = ntot, gap = NaN, cut_moved = false, total = ntot)
    cut = want
    while cut < ntot && all_e[cut + 1] - all_e[cut] ≤ gaptol
        cut += 1
    end
    if cut == ntot
        return Dict(q => length(E[q]) for q in ks),
               (requested = want, actual = ntot, gap = NaN, cut_moved = true, total = ntot)
    end
    thr = (all_e[cut] + all_e[cut + 1]) / 2
    kept = Dict(q => count(≤(thr), E[q]) for q in ks)
    return kept, (requested = want, actual = cut, gap = all_e[cut + 1] - all_e[cut], cut_moved = cut != want, total = ntot)
end
