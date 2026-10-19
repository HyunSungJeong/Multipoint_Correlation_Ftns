# Fermionic signs. They enter only through these two functions: every
# structural operation of TNOps (permute, contract, fuse, …) is bosonic.

"""
    parity_tensor(l, T = Float64)

The parity operator `Z = (-1)^p` on leg `l` as a neutral rank-2 tensor (legs
`l` as `Out` and `In`), `p` the fermion-parity component of each sector's label.
"""
function parity_tensor(l::Leg{S}, ::Type{T} = Float64) where {S,T}
    Z = BlockTensor{T}(l, l, neutral(S))
    for (s, d) in zip(legsectors(l), l.dims)
        setblock!(Z, (s, s), (isodd(parity(s.label)) ? -one(T) : one(T)) * Matrix{T}(I, d, d))
    end
    return Z
end

"""
    apply_parity(A, i; power = 1) -> BlockTensor

Multiply every block of `A` by `(-1)^(p · power)`, `p` the parity of its sector
on leg `i`: the action of `Z^power` on that leg without forming `Z`. Used for
the parity string of fermionic operators on earlier sites (`Z^p ⊗ O`).
"""
function apply_parity(A::BlockTensor{T,N,S}, i::Integer; power::Integer = 1) where {T,N,S}
    iseven(power) && return A
    C = BlockTensor{T}(legs(A), flux(A))
    cb = getfield(C, :blocks)
    for (k, M) in eachblock(A)
        cb[k] = isodd(parity(k[i].label)) ? -M : M
    end
    return C
end
