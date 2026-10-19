# Abelian symmetry factors and sector labels (task sheet Sec. 5.2.1, 5.2.2).
#
# A symmetry is a product G = G_1 × ⋯ × G_K of U(1) and Z_n factors. The
# fermion parity Z2(:Pf) is always present as the last factor, also when the
# user declares no symmetry at all; the fermionic signs of the Wilson-chain
# construction (Sec. 5.2.5) are read off it.

"""
    SymFactor

One abelian factor of a symmetry group: [`U1`](@ref) or [`Zn`](@ref).
"""
abstract type SymFactor end

"""
    U1(name)

A U(1) factor with integer labels fused by addition, e.g. `U1(:N)` for the
total charge or `U1(:twoSz)` for twice the spin projection (so that half
integers never appear).
"""
struct U1 <: SymFactor
    name::Symbol
end

"""
    Zn(name, n)

A cyclic factor with labels `0, …, n-1` fused by addition mod `n`. `Z2(name)` is
the case `n = 2`; the fermion parity is `Z2(:Pf)`.
"""
struct Zn <: SymFactor
    name::Symbol
    n::Int
    function Zn(name::Symbol, n::Integer)
        n ≥ 2 || throw(ArgumentError("Z_n needs n ≥ 2, got $n"))
        return new(name, Int(n))
    end
end

"""    Z2(name) — the cyclic factor `Zn(name, 2)`."""
Z2(name::Symbol) = Zn(name, 2)

modulus(::U1) = 0
modulus(f::Zn) = f.n
factorname(f::SymFactor) = f.name

"""
    Sector{M,K}

An abelian sector label: an `isbits` wrapper around `NTuple{K,Int}` in canonical
form (Z_n components reduced mod n). The type parameter `M::NTuple{K,Int}`
holds the moduli (`0` for U(1) factors), so that [`fuse`](@ref) and
[`dual`](@ref) need no group object. Labels are hashable and cheap to compare;
they order lexicographically, which fixes the deterministic sector order of a
[`SectorSpace`](@ref).
"""
struct Sector{M,K}
    q::NTuple{K,Int}
    function Sector{M,K}(q::NTuple{K,Int}) where {M,K}
        return new{M,K}(map((x, m) -> m == 0 ? x : mod(x, m), q, M))
    end
end

Base.isless(a::Sector{M,K}, b::Sector{M,K}) where {M,K} = isless(a.q, b.q)
# content hash: the default hash of a struct value mixes in the identity of
# its type, which differs between builds (code loaded); dictionaries keyed by
# labels would then iterate, and accumulate, in a build-dependent order
Base.hash(a::Sector, h::UInt) = hash(a.q, h ⊻ 0x5ec7a8e3c1d2b4f6)
Base.getindex(a::Sector, i::Integer) = a.q[i]
Base.length(::Sector{M,K}) where {M,K} = K
Base.show(io::IO, a::Sector) = print(io, "Sector", a.q)

"""
    fuse(a, b) -> Sector

Componentwise sum of two labels, mod `n` for Z_n factors: the label of a
product state, or the shift of a product of operators.
"""
@inline fuse(a::Sector{M,K}, b::Sector{M,K}) where {M,K} = Sector{M,K}(map(+, a.q, b.q))
fuse(a::Sector, b::Sector, c::Sector, rest::Sector...) = fuse(fuse(a, b), c, rest...)

"""    dual(a) — componentwise negation (mod `n` for Z_n factors)."""
@inline dual(a::Sector{M,K}) where {M,K} = Sector{M,K}(map(-, a.q))

"""    neutral(a) / neutral(G) — the identity label."""
neutral(::Type{Sector{M,K}}) where {M,K} = Sector{M,K}(ntuple(_ -> 0, K))
neutral(a::Sector) = neutral(typeof(a))

"""    isneutral(a) — whether `a` is the identity label."""
isneutral(a::Sector) = all(iszero, a.q)

"""
    parity(a) -> Int

The fermion-parity component of `a` (0 even, 1 odd). The parity factor is
always the last one (Sec. 5.2.2).
"""
parity(a::Sector{M,K}) where {M,K} = a.q[K]

"""
    degeneracy(a) -> Int

Number of states represented by one block row of sector `a`. It is `1` for every
abelian label; a non-abelian layer would return the multiplet dimension. Every
state count above `symmetry/` (partition functions, traces, FDM shell weights)
goes through this function (Sec. 5.2.9).
"""
degeneracy(::Sector) = 1

"""
    SymmetryGroup(factors...)

The abelian group `G = G_1 × ⋯ × G_K`. The fermion parity `Z2(:Pf)` is appended
as the last factor unless the user already declared it. Example: the SIAM with
`U1(:N) × U1(:twoSz)`:

```julia
G = SymmetryGroup(U1(:N), U1(:twoSz))      # factors (N, twoSz, Pf)
sector(G; N = 1, twoSz = 1)                 # Sector(1, 1, 1): Pf is derived from N
```
"""
struct SymmetryGroup{K,M}
    factors::NTuple{K,SymFactor}
    function SymmetryGroup(factors::SymFactor...)
        fs = collect(SymFactor, factors)
        names = map(factorname, fs)
        length(unique(names)) == length(names) ||
            throw(ArgumentError("symmetry factor names must be distinct: $names"))
        ip = findfirst(==(:Pf), names)
        if ip === nothing
            push!(fs, Z2(:Pf))
        else
            (fs[ip] isa Zn && fs[ip].n == 2) ||
                throw(ArgumentError(":Pf is reserved for the fermion parity Z2(:Pf)"))
            # keep the parity last
            f = fs[ip]; deleteat!(fs, ip); push!(fs, f)
        end
        K = length(fs)
        M = Tuple(modulus.(fs))
        return new{K,M}(Tuple(fs))
    end
end

Base.show(io::IO, G::SymmetryGroup) =
    print(io, join([f isa U1 ? "U1(:$(f.name))" : "Z$(f.n)(:$(f.name))" for f in G.factors], " × "))

"""    labeltype(G) — the concrete `Sector` type of group `G`."""
labeltype(::SymmetryGroup{K,M}) where {K,M} = Sector{M,K}
nfactors(::SymmetryGroup{K}) where {K} = K
factornames(G::SymmetryGroup) = map(factorname, G.factors)
neutral(G::SymmetryGroup) = neutral(labeltype(G))

"""
    sector(G, q::NTuple) / sector(G; name = value, ...)

Build a label of `G`. With keywords, unspecified factors are zero; the parity
must then be given explicitly or is set from `N` when a factor `:N` exists.
"""
sector(G::SymmetryGroup, q::NTuple{K,Int}) where {K} = labeltype(G)(q)
sector(G::SymmetryGroup, q::AbstractVector{<:Integer}) = labeltype(G)(Tuple(Int.(q)))
function sector(G::SymmetryGroup; kwargs...)
    names = factornames(G)
    q = zeros(Int, length(names))
    for (k, v) in kwargs
        i = findfirst(==(k), names)
        i === nothing && throw(ArgumentError("no symmetry factor named $k in $G"))
        q[i] = v
    end
    if !haskey(kwargs, :Pf) && :N in names
        q[end] = q[findfirst(==(:N), names)]
    end
    return sector(G, q)
end

"""
    factor_index(G, name) -> Int

Position of the factor `name` in the labels of `G`.
"""
function factor_index(G::SymmetryGroup, name::Symbol)
    i = findfirst(==(name), factornames(G))
    i === nothing && throw(ArgumentError("no symmetry factor named $name in $G"))
    return i
end
