# Step 3 of Block 4: the Matsubara convolution kernel, Kugler Eq. (46).
#
# Deliberately Eq. (46) and not Eq. (45): the two are equivalent, but (46) is
# "more convenient for numerical computations, as it is manifestly free from
# divergences" -- it never forms 1/0 and then cancels it, it branches instead.

"""
    MatsubaraFreq(m)

A Matsubara frequency `ω = m π / β`, stored by its **integer** index `m`.
Odd `m` is fermionic, `ω = (2n+1)π/β`; even `m` is bosonic, `ω = 2nπ/β`.

Carrying the integer rather than the float matters for Eq. (46). The composite
`Ω_{1̄⋯ī} = iω_{1̄⋯ī} - ω'_{1̄⋯ī}` vanishes only when *both* parts vanish, and
the Matsubara part vanishes exactly or not at all. Testing the integer sum
`Σ m = 0` is therefore exact, whereas float-comparing `Ω` against zero would
confuse a genuinely small frequency — `π/β` is tiny at large `β` — with a
composite that is identically zero and needs the anomalous branch.
"""
struct MatsubaraFreq
    m::Int
end

"""    value(f::MatsubaraFreq, β) -> Float64 — the frequency `ω = m π / β`."""
value(f::MatsubaraFreq, β::Real) = f.m * π / β

"""    fermionic(f::MatsubaraFreq) -> Bool — true for odd `m`."""
fermionic(f::MatsubaraFreq) = isodd(f.m)

"""    bosonic(f::MatsubaraFreq) -> Bool — true for even `m`."""
bosonic(f::MatsubaraFreq) = iseven(f.m)

"""
    fermionic_freq(n) -> MatsubaraFreq

The fermionic frequency `ω_n = (2n+1)π/β`.
"""
fermionic_freq(n::Integer) = MatsubaraFreq(2n + 1)

"""
    bosonic_freq(n) -> MatsubaraFreq

The bosonic frequency `ω_n = 2nπ/β`.
"""
bosonic_freq(n::Integer) = MatsubaraFreq(2n)

"""
    composites(ms, ωprime, β; atol = 1e-10) -> (Ω, vanishing)

Build the composite frequencies `Ω_{1̄⋯ī} = iω_{1̄⋯ī} - ω'_{1̄⋯ī}` of Eq. (35c)
for `i = 1 … ℓ-1`, together with a flag per entry saying whether it vanishes.

`ms` holds the `ℓ` external [`MatsubaraFreq`](@ref) in the **permuted** order
`ω_p`, and `ωprime` the `ℓ-1` partial sums `ω'_{1̄⋯ī}` of a PSF term, i.e. the
`position` field of a [`PSFTerm`](@ref) for the same permutation.

An entry vanishes iff the integer Matsubara sum is exactly zero *and* the
spectral partial sum is zero to `atol`. The latter happens through a degeneracy
`E_{i+1} = E_1` in the spectrum, as Eq. (45) notes.
"""
function composites(ms::AbstractVector{MatsubaraFreq}, ωprime::AbstractVector{<:Real},
                    β::Real; atol::Real = 1e-10)
    ℓ = length(ms)
    length(ωprime) == ℓ - 1 ||
        throw(DimensionMismatch("expected $(ℓ-1) spectral partial sums, got $(length(ωprime))"))

    Ω = Vector{ComplexF64}(undef, ℓ - 1)
    vanishing = Vector{Bool}(undef, ℓ - 1)
    msum = 0
    for i in 1:(ℓ - 1)
        msum += ms[i].m
        Ω[i] = im * (msum * π / β) - ωprime[i]
        vanishing[i] = (msum == 0) && (abs(ωprime[i]) <= atol)
    end
    return Ω, vanishing
end

"""
    matsubara_kernel(Ω, vanishing, β) -> ComplexF64

The Matsubara kernel of Kugler Eq. (46), in its two-branch form:

    K(Ω_p) = ∏_{i=1}^{ℓ-1} Ω⁻¹_{1̄⋯ī}                              if ∏_i Ω_{1̄⋯ī} ≠ 0,
           = -½ [ β + Σ_{i≠j} Ω⁻¹_{1̄⋯ī} ] ∏_{i≠j} Ω⁻¹_{1̄⋯ī}       if Ω_{1̄⋯j̄} = 0.

`vanishing` marks which composites are identically zero; it is not inferred
from `Ω` itself, because `Ω` is a float and the distinction is exact — see
[`MatsubaraFreq`](@ref) and [`composites`](@ref).

The second branch is what makes this kernel finite by construction: the
diverging factor `1/Ω_{1̄⋯j̄}` is *excluded* from the product rather than formed
and cancelled later. The regular branch needs no exclusion, since as
`Ω_{1̄⋯j̄} → 0` the divergence is cancelled by one from a cyclically related
permutation.

For `ℓ = 2` the anomalous branch has an empty product and an empty sum, giving
`K = -β/2`.

Eq. (46) covers at most one vanishing composite, which is all that can occur
for any 2p function, a 3p function with one bosonic operator, or a fermionic 4p
function. More than one is rejected rather than silently mishandled.
"""
function matsubara_kernel(Ω::AbstractVector{<:Number}, vanishing::AbstractVector{Bool},
                          β::Real)
    length(Ω) == length(vanishing) ||
        throw(DimensionMismatch("Ω and vanishing must have equal length"))
    isempty(Ω) && throw(ArgumentError("need at least one composite frequency (ℓ ≥ 2)"))

    js = findall(vanishing)

    if isempty(js)
        # regular branch: ∏ Ω⁻¹
        K = one(ComplexF64)
        for Ωi in Ω
            K /= Ωi
        end
        return K
    elseif length(js) == 1
        # anomalous branch: the j-th factor is dropped from both the sum and
        # the product, so nothing divergent is ever formed.
        j = js[1]
        acc = zero(ComplexF64)
        K = one(ComplexF64)
        for (i, Ωi) in enumerate(Ω)
            i == j && continue
            acc += inv(Ωi)
            K /= Ωi
        end
        return -0.5 * (β + acc) * K
    else
        throw(ArgumentError(
            "Eq. (46) admits at most one vanishing composite Ω, found $(length(js)) " *
            "at positions $js. That is outside the cases the paper covers " *
            "(2p, 3p with one bosonic operator, fermionic 4p)."))
    end
end

"""
    matsubara_kernel(ms, ωprime, β; atol = 1e-10) -> ComplexF64

Convenience form: build the composites with [`composites`](@ref) and evaluate
Eq. (46). `ms` are the `ℓ` external frequencies in permuted order, `ωprime` the
`ℓ-1` spectral partial sums of the matching PSF term.
"""
function matsubara_kernel(ms::AbstractVector{MatsubaraFreq}, ωprime::AbstractVector{<:Real},
                          β::Real; atol::Real = 1e-10)
    Ω, vanishing = composites(ms, ωprime, β; atol = atol)
    return matsubara_kernel(Ω, vanishing, β)
end
