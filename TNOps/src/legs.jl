# Legs: tagged sector spaces with an arrow.
#
# A leg is a *physical* sector space — an ordered list of sectors, each a
# charge label with an integer tag and a dimension — plus an arrow (`In` or
# `Out`) and an optional name. The charge a sector carries out of a tensor is
# its label on an `Out` leg and the dual label on an `In` leg; `dual(leg)`
# flips the arrow, which dualizes the charges of all its sectors while the
# physical space (labels, tags, dimensions, order) stays the same. A leg can
# be contracted only with its dual.

"""
    Arrow

`Out` (ket-like, e.g. the row leg of an operator) or `In` (bra-like, e.g. the
column leg).
"""
@enum Arrow::Int8 In = 0 Out = 1
Base.hash(a::Arrow, h::UInt) = hash(Int8(a), hash(:Arrow, h))       # content hash (see Sector)

flip(a::Arrow) = a == In ? Out : In

"""
    LegSector(label, tag = 0)

One sector of a leg: a charge label and an integer tag. The tag distinguishes
blocks that share a label (e.g. the kept and discarded states of an NRG shell);
`0` means untagged. Sectors order by `(label, tag)`.
"""
struct LegSector{S<:Sector}
    label::S
    tag::Int
end
LegSector(q::Sector) = LegSector(q, 0)

label(s::LegSector) = s.label
tag(s::LegSector) = s.tag
Base.isless(a::LegSector{S}, b::LegSector{S}) where {S} = isless((a.label, a.tag), (b.label, b.tag))
Base.hash(s::LegSector, h::UInt) = hash(s.tag, hash(s.label, hash(:LegSector, h)))   # content hash (see Sector)
Base.show(io::IO, s::LegSector) = s.tag == 0 ? print(io, s.label) : print(io, s.label, "#", s.tag)

"""
    Leg(sectors, dims; arrow = Out, name = Symbol(""))
    Leg(Dict(sector => dim); arrow, name)
    Leg(Dict(label => dim); tag = 0, arrow, name)

A leg: sectors sorted by `(label, tag)`, sectors of dimension 0 dropped,
duplicates refused.
"""
struct Leg{S<:Sector}
    sectors::Vector{LegSector{S}}
    dims::Vector{Int}
    index::Dict{LegSector{S},Int}
    labels::Vector{S}                 # distinct labels, sorted
    arrow::Arrow
    name::Symbol
    function Leg{S}(secs::Vector{LegSector{S}}, dims::Vector{Int}, arrow::Arrow, name::Symbol) where {S}
        length(secs) == length(dims) || throw(DimensionMismatch("one dimension per sector"))
        any(<(0), dims) && throw(ArgumentError("negative sector dimension"))
        p = sortperm(secs)
        secs = secs[p]; dims = dims[p]
        allunique(secs) || throw(ArgumentError("duplicate leg sectors"))
        keep = dims .> 0
        secs = secs[keep]; dims = dims[keep]
        labs = unique!([s.label for s in secs])
        return new{S}(secs, dims, Dict(s => i for (i, s) in enumerate(secs)), labs, arrow, name)
    end
end

Leg(secs::AbstractVector{LegSector{S}}, dims::AbstractVector{<:Integer}; arrow::Arrow = Out,
    name::Symbol = Symbol("")) where {S} = Leg{S}(collect(secs), collect(Int, dims), arrow, name)
Leg(d::AbstractDict{LegSector{S},<:Integer}; kwargs...) where {S} = Leg(collect(keys(d)), collect(values(d)); kwargs...)
Leg(labels::AbstractVector{S}, dims::AbstractVector{<:Integer}; tag::Integer = 0, kwargs...) where {S<:Sector} =
    Leg([LegSector(q, Int(tag)) for q in labels], dims; kwargs...)
Leg(d::AbstractDict{S,<:Integer}; tag::Integer = 0, kwargs...) where {S<:Sector} =
    Leg([LegSector(q, Int(tag)) for q in keys(d)], collect(values(d)); kwargs...)

"""    legsectors(l) — the sectors `LegSector(label, tag)` of `l`, in order."""
legsectors(l::Leg) = l.sectors
"""    sectors(l) — the distinct charge labels of `l`, sorted."""
sectors(l::Leg) = l.labels
arrow(l::Leg) = l.arrow
legname(l::Leg) = l.name
withname(l::Leg{S}, n::Symbol) where {S} = Leg{S}(copy(l.sectors), copy(l.dims), l.arrow, n)
witharrow(l::Leg{S}, a::Arrow) where {S} = a == l.arrow ? l : Leg{S}(copy(l.sectors), copy(l.dims), a, l.name)
nsectors(l::Leg) = length(l.sectors)
Base.eltype(::Type{Leg{S}}) where {S} = LegSector{S}

"""    dim(l, s::LegSector) / dim(l, q) — dimension of a sector, or of all sectors with label `q`."""
dim(l::Leg{S}, s::LegSector{S}) where {S} = (i = get(l.index, s, 0); i == 0 ? 0 : l.dims[i])
function dim(l::Leg{S}, q::S) where {S}
    # fast path: (q, 0) is the first sector of its label (sorted by (label, tag))
    i = get(l.index, LegSector(q, 0), 0)
    i > 0 && (i == length(l.sectors) || l.sectors[i + 1].label != q) && return l.dims[i]
    d = 0
    for (s, n) in zip(l.sectors, l.dims)
        s.label == q && (d += n)
    end
    return d
end
"""    dim(l) — number of states, `Σ degeneracy(label) · dim`."""
dim(l::Leg) = sum((degeneracy(s.label) * d for (s, d) in zip(l.sectors, l.dims)); init = 0)
"""    nblockrows(l) — number of block rows (`Σ dim` without degeneracies)."""
nblockrows(l::Leg) = sum(l.dims; init = 0)
Base.haskey(l::Leg{S}, s::LegSector{S}) where {S} = haskey(l.index, s)
Base.haskey(l::Leg{S}, q::S) where {S} = insorted(q, l.labels)
"""    tags(l, q) — the tags of the sectors with label `q`, ascending."""
tags(l::Leg{S}, q::S) where {S} = [s.tag for s in l.sectors if s.label == q]
"""    tagged(l, q, t) — `LegSector(q, t)`; refuses a sector absent from `l`."""
function tagged(l::Leg{S}, q::S, t::Integer) where {S}
    s = LegSector(q, Int(t))
    haskey(l, s) || throw(ArgumentError("leg $(legstring(l)) has no sector $s"))
    return s
end

"""
    charge(l, s) -> label

The charge that sector `s` of leg `l` carries out of the tensor: its label on
an `Out` leg, the dual label on an `In` leg.
"""
charge(l::Leg{S}, s::LegSector{S}) where {S} = l.arrow == Out ? s.label : dual(s.label)

"""
    dual(l)

The dual leg: the same sectors and dimensions with the arrow flipped, so every
sector carries the dual charge. A leg contracts only with its dual.
"""
dual(l::Leg{S}) where {S} = Leg{S}(copy(l.sectors), copy(l.dims), flip(l.arrow), l.name)

"""    samespace(a, b) — equal sectors and dimensions (arrows and names ignored)."""
samespace(a::Leg, b::Leg) = a.sectors == b.sectors && a.dims == b.dims
"""    isdual(a, b) — `b` is the dual of `a`: the same space with opposite arrows."""
isdual(a::Leg, b::Leg) = samespace(a, b) && a.arrow != b.arrow

Base.:(==)(a::Leg, b::Leg) = samespace(a, b) && a.arrow == b.arrow
Base.hash(l::Leg, h::UInt) = hash(l.arrow, hash(l.dims, hash(l.sectors, h)))

function legstring(l::Leg)
    nm = l.name == Symbol("") ? "" : ":$(l.name) "
    return "$(nm)[$(l.arrow); " * join(["$s=>$d" for (s, d) in zip(l.sectors, l.dims)], ", ") * "]"
end
Base.show(io::IO, l::Leg) = print(io, "Leg(", legstring(l), ")")

"""
    offsets(l) -> Dict{LegSector,Int}

Offset of every sector in the dense index of the leg: sectors are stacked in
their sorted `(label, tag)` order ([`todense`](@ref)).
"""
function offsets(l::Leg{S}) where {S}
    off = Dict{LegSector{S},Int}(); o = 0
    for (s, d) in zip(l.sectors, l.dims)
        off[s] = o; o += d
    end
    return off
end

"""
    retag(l, split) -> Leg

Split the sectors of an untagged (or tagged) leg into tagged sub-sectors:
`split(s, d)` returns a vector of `(tag, dim)` pairs whose dimensions add up to
`d`. Example: the kept/discarded split of an NRG shell,
`retag(l, (s, d) -> [(1, nk[s]), (2, d - nk[s])])`.
"""
function retag(l::Leg{S}, split) where {S}
    secs = LegSector{S}[]; dims = Int[]
    for (s, d) in zip(l.sectors, l.dims)
        parts = split(s, d)
        sum((p[2] for p in parts); init = 0) == d || throw(ArgumentError("retag: parts of $s do not add up to $d"))
        for (t, n) in parts
            push!(secs, LegSector(s.label, Int(t))); push!(dims, n)
        end
    end
    return Leg{S}(secs, dims, l.arrow, l.name)
end

"""    subleg(l, keep) — the leg restricted to the sectors `s` with `keep(s)`."""
subleg(l::Leg{S}, keep) where {S} =
    Leg{S}([s for s in l.sectors if keep(s)], [d for (s, d) in zip(l.sectors, l.dims) if keep(s)], l.arrow, l.name)
