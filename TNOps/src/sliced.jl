# Sliced tensors: a sparse map from slice key to BlockTensor.
#
# The key type is arbitrary (tuples of bin indices, or exact Float64 energies
# in Lehmann mode), so slices are not turned into tensor legs; only keys that
# carry weight are stored. All slices share the legs and the flux.

"""
    SlicedTensor{Kt,T,N,S}

A sparse map `slices::Dict{Kt,BlockTensor{T,N,S}}` from slice keys to block
tensors on common legs `legs` with a common flux `flux`. Contractions over the
tensor legs act slice by slice ([`mapslices`](@ref)); the slice keys pass
through untouched.
"""
struct SlicedTensor{Kt,T<:Number,N,S<:Sector}
    legs::NTuple{N,Leg{S}}
    flux::S
    slices::Dict{Kt,BlockTensor{T,N,S}}
end

"""    SlicedTensor{Kt,T}(legs, flux) — no slices."""
SlicedTensor{Kt,T}(ls::NTuple{N,Leg{S}}, Δ::S) where {Kt,T,N,S} =
    SlicedTensor{Kt,T,N,S}(ls, Δ, Dict{Kt,BlockTensor{T,N,S}}())

"""
    SlicedTensor(k => A, …) / SlicedTensor(d::Dict{Kt,BlockTensor})

Build from slices with common legs and flux (at least one slice).
"""
function SlicedTensor(d::AbstractDict{Kt,BlockTensor{T,N,S}}) where {Kt,T,N,S}
    isempty(d) && throw(ArgumentError("SlicedTensor(dict) needs at least one slice; use SlicedTensor{Kt,T}(legs, flux)"))
    A1 = first(values(d))
    st = SlicedTensor{Kt,T}(legs(A1), flux(A1))
    for (k, A) in d
        setslice!(st, k, A)
    end
    return st
end

legs(st::SlicedTensor) = st.legs
leg(st::SlicedTensor, i::Integer) = st.legs[i]
flux(st::SlicedTensor) = st.flux
Base.eltype(::SlicedTensor{Kt,T}) where {Kt,T} = T
keytype_slices(::SlicedTensor{Kt}) where {Kt} = Kt
"""    nslices(st) — number of stored slices."""
nslices(st::SlicedTensor) = length(st.slices)
Base.length(st::SlicedTensor) = length(st.slices)
Base.isempty(st::SlicedTensor) = isempty(st.slices)
Base.keys(st::SlicedTensor) = keys(st.slices)
Base.values(st::SlicedTensor) = values(st.slices)
Base.pairs(st::SlicedTensor) = pairs(st.slices)
Base.iterate(st::SlicedTensor, s...) = iterate(st.slices, s...)
Base.haskey(st::SlicedTensor, k) = haskey(st.slices, k)
Base.getindex(st::SlicedTensor, k) = st.slices[k]
getslice(st::SlicedTensor, k, default) = get(st.slices, k, default)

"""    setslice!(st, k, A) — store slice `k` (legs and flux must match)."""
function setslice!(st::SlicedTensor{Kt,T,N,S}, k, A::BlockTensor{T,N,S}) where {Kt,T,N,S}
    all(i -> leg(A, i) == leg(st, i), 1:N) || throw(DimensionMismatch("slice $k has different legs"))
    flux(A) == flux(st) || throw(ArgumentError("slice $k has flux $(flux(A)) ≠ $(flux(st))"))
    st.slices[k] = A
    return st
end

"""
    mapslices(f, st::SlicedTensor) -> SlicedTensor

Apply `f` (a contraction over the tensor legs) to every slice; slices for which
`f` returns a tensor without blocks are dropped. All results must share legs
and flux. With no slices left, the result has the legs and flux of
`f` applied to nothing and is returned as `nothing`-free: an empty
`SlicedTensor` with the input's key type and the first result's layout, or
`st` itself if it was empty.
"""
function Base.mapslices(f, st::SlicedTensor{Kt,T,N,S}) where {Kt,T,N,S}
    isempty(st) && return st
    out = nothing
    for (k, A) in st.slices
        B = f(A)
        if out === nothing
            out = SlicedTensor{Kt,eltype(B)}(legs(B), flux(B))
        end
        nblocks(B) == 0 && continue
        setslice!(out, k, B)
    end
    return out
end

Base.copy(st::SlicedTensor{Kt,T,N,S}) where {Kt,T,N,S} =
    SlicedTensor{Kt,T,N,S}(st.legs, st.flux, Dict(k => copy(A) for (k, A) in st.slices))
