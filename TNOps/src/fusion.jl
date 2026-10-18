# Fusion of a new Wilson site with the kept space of the previous shell
# (task sheet Sec. 5.2.5).
#
# Convention (Lee p. 5, Weichselbaum & von Delft 2007): the product basis is
# built "as if filling the chain in the order c†_N ⋯ c†_{-1}|0⟩", so that
#
#     |σ_n⟩ ⊗ |s⟩_{n-1} := c†(σ_n) |s⟩_{n-1} ,
#
# with the creators of the new site n to the LEFT of those of all earlier
# sites. Two representation rules follow (derived in STATUS.md, verified in V1):
#
#   (R1) an operator O on earlier sites with parity p_O acts on the product
#        basis as (Z_n)^{p_O} ⊗ O, Z_n = (-1)^{N_n} the parity of site n;
#   (R2) an operator O_n on site n alone acts as O_n ⊗ 1.
#
# Both factors are obtained from `kron_op` with the parity read off the Pf
# labels; no sign is entered by hand.

"""
    FusionPair(σ, q, offset, dσ, dq)

One contribution `(σ, q)` (local sector σ of the new site, kept sector q of the
previous shell) to a product sector `q′ = σ ⊕ q`, occupying rows
`offset .+ (1:dσ*dq)` of the product block. Within the pair, the product state
`(iσ, s)` has index `offset + (iσ-1)*dq + s`, i.e. `kron(e_iσ, e_s)`.
"""
struct FusionPair{S<:Sector}
    σ::S
    q::S
    offset::Int
    dσ::Int
    dq::Int
end

"""
    FusionMap

For every product sector `q′`, the ordered list of contributing pairs
`(σ, q)` with their row offsets (local sectors in sorted order, then kept
sectors in sorted order). It defines the product basis of every block of `H_n`
and is stored with the shell's isometries, since the operator expansions need
it. `lookup[(σ, q)] = (q′, pair index)`.
"""
struct FusionMap{S<:Sector}
    loc::SectorSpace{S}
    prev::SectorSpace{S}
    out::SectorSpace{S}
    pairs::Dict{S,Vector{FusionPair{S}}}
    lookup::Dict{Tuple{S,S},Tuple{S,Int}}
end

"""
    fusionmap(Vloc, Vprev) -> FusionMap

Fuse the local sector space of a new site with the (kept) sector space of the
previous shell. Precomputed once per shell.
"""
function fusionmap(Vloc::SectorSpace{S}, Vprev::SectorSpace{S}) where {S}
    pairs = Dict{S,Vector{FusionPair{S}}}()
    dims = Dict{S,Int}()
    lookup = Dict{Tuple{S,S},Tuple{S,Int}}()
    for (σ, dσ) in zip(Vloc.labels, Vloc.dims), (q, dq) in zip(Vprev.labels, Vprev.dims)
        qq = fuse(σ, q)
        off = get(dims, qq, 0)
        v = get!(() -> FusionPair{S}[], pairs, qq)
        push!(v, FusionPair{S}(σ, q, off, dσ, dq))
        lookup[(σ, q)] = (qq, length(v))
        dims[qq] = off + dσ * dq
    end
    return FusionMap{S}(Vloc, Vprev, SectorSpace(dims), pairs, lookup)
end

"""
    kron_op(X, Y, Frow, Fcol) -> BlockOp

The operator `X ⊗ Y` on product spaces: `X` acts on the local sectors of the new
site, `Y` on the sectors of the previous shell; `Fcol` is the fusion map of the
column (input) space and `Frow` of the row (output) space. The shift is
`ΔX ⊕ ΔY`. Each block is a Kronecker product placed at the pair offsets of the
fusion maps.
"""
function kron_op(X::BlockOp{TX,S}, Y::BlockOp{TY,S}, Frow::FusionMap{S}, Fcol::FusionMap{S}) where {TX,TY,S}
    X.col == Fcol.loc && Y.col == Fcol.prev || throw(DimensionMismatch("input spaces do not match the column fusion map"))
    X.row == Frow.loc && Y.row == Frow.prev || throw(DimensionMismatch("output spaces do not match the row fusion map"))
    T = promote_type(TX, TY)
    Δ = fuse(X.shift, Y.shift)
    R = BlockOp{T}(Frow.out, Fcol.out, Δ)
    for (qin, prs) in Fcol.pairs
        qout = fuse(qin, Δ)
        haskey(Frow.pairs, qout) || continue
        blk = nothing
        for p in prs
            Xb = get(X.blocks, p.σ, nothing); Xb === nothing && continue
            Yb = get(Y.blocks, p.q, nothing); Yb === nothing && continue
            σ2 = target(X, p.σ); q2 = target(Y, p.q)
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
        blk === nothing || (R.blocks[qin] = blk)
    end
    return R
end
