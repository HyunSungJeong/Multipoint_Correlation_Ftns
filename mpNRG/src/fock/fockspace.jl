# Dense Fock spaces with an explicit Jordan-Wigner order (task sheet Sec. 4.3,
# 5.1). Used to build local site matrices, local operators at shell n_op, and
# the dense reference ED of `reference/`.

"""
    FockBasis(modes; allowed = nothing)

The Fock space of an ordered list of modes. Basis state `b` (a bit pattern,
mode `k` stored in bit `M-k`, so mode 1 is the most significant bit) is

    |b⟩ = (c†_{m_1})^{b_1} (c†_{m_2})^{b_2} ⋯ (c†_{m_M})^{b_M} |0⟩ ,

i.e. the creators stand in list order from left to right. With Lee's
convention (p. 5) of filling the chain "in the order c†_N ⋯ c†_{-1}|0⟩", the
list for sites `-1..n` is `[modes of site n; …; modes of site -1]`, and the
basis index then factorises as `kron(site n, …, site -1)`.

`allowed` optionally restricts the basis to a subset of bit patterns (e.g. the
singly occupied impurity of the Kondo model); operators must then preserve the
subset. Basis states are listed in increasing bit-pattern order.
"""
struct FockBasis
    modes::Vector{Mode}
    states::Vector{UInt64}
    index::Dict{UInt64,Int}
end

function FockBasis(modes::AbstractVector{Mode}; allowed = nothing)
    M = length(modes)
    M ≤ 30 || throw(ArgumentError("dense Fock space of $M modes is too large"))
    allunique(modes) || throw(ArgumentError("duplicate modes"))
    states = allowed === nothing ? collect(UInt64(0):(UInt64(1) << M - 1)) : sort!(UInt64.(collect(allowed)))
    return FockBasis(collect(modes), states, Dict(s => i for (i, s) in enumerate(states)))
end

Base.length(B::FockBasis) = length(B.states)

"""    occupation(B, i, m) — the occupation (0/1) of mode `m` in basis state `i`."""
function occupation(B::FockBasis, i::Integer, m::Mode)
    k = findfirst(==(m), B.modes)
    k === nothing && throw(ArgumentError("mode $m not in basis"))
    return Int((B.states[i] >> (length(B.modes) - k)) & 1)
end

"""
    apply_ladder(B, l, s) -> (sign, s′) or (0, s)

Action of ladder operator `l` on bit pattern `s`: the Jordan-Wigner sign is
`(-1)^(number of occupied modes to the left of the mode in the creator order)`.
"""
@inline function apply_ladder(M::Int, k::Int, dag::Bool, s::UInt64)
    b = M - k
    occ = (s >> b) & 1
    (occ == 1) == dag && return (0, s)
    left = count_ones(s >> (b + 1))
    sgn = isodd(left) ? -1 : 1
    return (sgn, s ⊻ (UInt64(1) << b))
end

"""
    densematrix(a, B; T = eltype(a)) -> Matrix

Dense matrix of the operator expression `a` on Fock basis `B`, built word by
word from the Jordan-Wigner action of each ladder operator: no sign is entered
by hand. Every mode of `a` must belong to `B`; with a restricted basis the
operator must not leave it.
"""
function densematrix(a::OpExpr, B::FockBasis; T::Type = eltype(a))
    M = length(B.modes)
    pos = Dict(m => k for (k, m) in enumerate(B.modes))
    D = zeros(T, length(B), length(B))
    for (w, c) in a.terms
        ks = Int[]
        for l in w
            k = get(pos, l.mode, 0)
            k == 0 && throw(ArgumentError("mode $(l.mode) is not in the Fock basis"))
            push!(ks, k)
        end
        for (j, s0) in enumerate(B.states)
            s = s0; sg = 1
            for t in length(w):-1:1
                sgn, s = apply_ladder(M, ks[t], w[t].dag, s)
                sgn == 0 && (sg = 0; break)
                sg *= sgn
            end
            sg == 0 && continue
            i = get(B.index, s, 0)
            i == 0 && throw(ArgumentError("operator leaves the restricted Fock basis"))
            D[i, j] += sg * c
        end
    end
    return D
end

"""
    statecharge(B, i, charges) -> Sector

Label of Fock basis state `i`: the fused charges of its occupied modes.
"""
function statecharge(B::FockBasis, i::Integer, charges::AbstractDict{Mode,S}) where {S}
    q = neutral(S); M = length(B.modes)
    for (k, m) in enumerate(B.modes)
        if (B.states[i] >> (M - k)) & 1 == 1
            q = fuse(q, charges[m])
        end
    end
    return q
end

"""
    LocalSpace(site, flavors; charges, allowed = nothing)

The local Fock space of one site: modes `Mode(site, f)` for the `flavors` in
local Jordan-Wigner order, with their charges `Q_m` (the quantum numbers of
`c†_m|0⟩`, Sec. 5.2.3). The local order is used consistently for the impurity
and for every bath site (Sec. 5.2.5).
"""
struct LocalSpace{S<:Sector}
    site::Int
    basis::FockBasis
    charges::Dict{Mode,S}
    labels::Vector{S}                  # label of each local basis state
    sectors::Dict{S,Vector{Int}}       # local basis states per sector, ascending
end

function LocalSpace(site::Integer, flavors::AbstractVector{Symbol}, charges::AbstractDict{Mode,S};
                    allowed = nothing) where {S<:Sector}
    ms = [Mode(site, f) for f in flavors]
    B = FockBasis(ms; allowed = allowed)
    ch = Dict{Mode,S}(m => charges[m] for m in ms)
    for m in ms
        parity(ch[m]) == 1 || throw(ArgumentError("mode $m must be fermion-parity odd (Pf = 1)"))
    end
    labels = [statecharge(B, i, ch) for i in 1:length(B)]
    for i in 1:length(B)
        Nf = count_ones(B.states[i])
        parity(labels[i]) == mod(Nf, 2) ||
            throw(ArgumentError("local state $i has parity label inconsistent with (-1)^N"))
    end
    secs = Dict{S,Vector{Int}}()
    for (i, q) in enumerate(labels)
        push!(get!(() -> Int[], secs, q), i)
    end
    return LocalSpace{S}(Int(site), B, ch, labels, secs)
end

Base.length(L::LocalSpace) = length(L.basis)
localmodes(L::LocalSpace) = L.basis.modes
"""    localspace_sectors(L) — `SectorSpace` of the local states."""
localspace_sectors(L::LocalSpace) = SectorSpace(Dict(q => length(v) for (q, v) in L.sectors))

"""
    localop(L, a) -> BlockOp

The operator expression `a` (acting on the modes of `L` only) as a block
operator on the local sector space. Within a sector the local basis states keep
their ascending Fock order. `a` must be homogeneous.
"""
function localop(L::LocalSpace{S}, a::OpExpr; T::Type = eltype(a)) where {S}
    Δ = opshift(a, L.charges)
    D = densematrix(a, L.basis; T = T)
    V = localspace_sectors(L)
    A = BlockOp{T}(V, V, Δ)
    for (q, cols) in L.sectors
        r = fuse(q, Δ)
        haskey(L.sectors, r) || continue
        blk = D[L.sectors[r], cols]
        any(!iszero, blk) && (A.blocks[q] = blk)
    end
    # every nonzero dense element must be accounted for by the blocks
    nb = sum((count(!iszero, M) for M in values(A.blocks)); init = 0)
    nb == count(!iszero, D) || throw(ArgumentError("operator is not homogeneous on the local space"))
    return A
end

"""
    parity_op(L) -> BlockOp

The local parity `Z = (-1)^N` of the site, read off the `Pf` labels.
"""
function parity_op(L::LocalSpace{S}) where {S}
    V = localspace_sectors(L)
    Z = BlockOp{Float64}(V, V, neutral(S))
    for (q, v) in L.sectors
        Z.blocks[q] = (isodd(parity(q)) ? -1.0 : 1.0) * Matrix{Float64}(I, length(v), length(v))
    end
    return Z
end
