# The Keldysh formalism: Kugler Sec. II D, Eqs. (49), (52), (63), (67).
#
# The PSFs of Eq. (28) are formalism independent, so nothing here touches the
# spectrum or the PSF code. Only the kernel changes, and the prefactor 2^{1-ℓ/2}.
#
# Conventions, all read from the printed page (the text layer garbles several):
#
#   * a permutation p acts on slot contents, ī = p(i), so O_p = (O_p(1), …);
#   * a Keldysh component k = k_1⋯k_ℓ, k_i ∈ {1,2}, is stored as the ordered
#     list of the slots of its 2's, k = [η_1 … η_α] with η_j < η_{j+1}
#     (p. 13: 1111 = [], 2121 = [13]);
#   * moving an entry to a new slot needs the INVERSE permutation:
#     μ_j = p⁻¹(η_j) (p. 13; the text layer prints the exponent as "p-1").

# ---------------------------------------------------------------------------
# K1: complex frequency tuples, Eq. (49)
# ---------------------------------------------------------------------------

"""
    omega_eta(ω, η, γ0) -> Vector{ComplexF64}

The complex frequency tuple `ω^[η]` of Kugler Eq. (49), with the choice stated
below it on p. 10:

    ω^[η]_i = ω_i + i(ℓ-1)γ₀    (i = η),
    ω^[η]_i = ω_i - iγ₀         (i ≠ η).

Slot `η` alone gets a positive imaginary part. The `(ℓ-1)` makes the imaginary
parts sum to zero, `γ^[η]_{1⋯ℓ} = 0`, so `ω^[η]_{1⋯ℓ} = ω_{1⋯ℓ} = 0` survives
the shift — which the caller's `ω` must satisfy.

`γ₀ > 0` is a required argument rather than a module constant: the paper keeps
it "small but finite" for numerics and infinitesimal for analysis, and which
components converge as `γ₀ → 0⁺` is itself one of the things to measure.
"""
function omega_eta(ω::AbstractVector{<:Real}, η::Integer, γ0::Real)
    ℓ = length(ω)
    1 ≤ η ≤ ℓ || throw(ArgumentError("η must be a slot in 1:$ℓ, got $η"))
    γ0 > 0 || throw(ArgumentError("γ₀ must be positive, got $γ0"))
    return ComplexF64[i == η ? ω[i] + im * (ℓ - 1) * γ0 : ω[i] - im * γ0 for i in 1:ℓ]
end

# ---------------------------------------------------------------------------
# K2: Keldysh index bookkeeping, p. 13
# ---------------------------------------------------------------------------

"""
    keldysh_slots(k) -> Vector{Int}

The ordered slot list `[η_1 … η_α]` of a Keldysh component given as digits,
e.g. `keldysh_slots([2,1,2,1]) == [1,3]` and `keldysh_slots([1,1,1,1]) == Int[]`.
"""
function keldysh_slots(k::AbstractVector{<:Integer})
    all(x -> x == 1 || x == 2, k) || throw(ArgumentError("Keldysh indices are 1 or 2, got $k"))
    return [i for i in eachindex(k) if k[i] == 2]
end

"""
    keldysh_digits(slots, ℓ) -> Vector{Int}

Inverse of [`keldysh_slots`](@ref): `keldysh_digits([1,3], 4) == [2,1,2,1]`.
"""
function keldysh_digits(slots::AbstractVector{<:Integer}, ℓ::Integer)
    all(s -> 1 ≤ s ≤ ℓ, slots) || throw(ArgumentError("slots must lie in 1:$ℓ"))
    allunique(slots) || throw(ArgumentError("slots must be distinct"))
    return [i in slots ? 2 : 1 for i in 1:ℓ]
end

"""
    keldysh_label(slots, ℓ) -> String

The component as the paper writes it, e.g. `"2121"`.
"""
keldysh_label(slots::AbstractVector{<:Integer}, ℓ::Integer) = join(keldysh_digits(slots, ℓ))

"""
    all_keldysh_components(ℓ) -> Vector{Vector{Int}}

All `2^ℓ` components as slot lists, in increasing order of `α` then
lexicographically.
"""
function all_keldysh_components(ℓ::Integer)
    comps = [keldysh_slots([(n >> (ℓ - i)) & 1 == 1 ? 2 : 1 for i in 1:ℓ]) for n in 0:(2^ℓ - 1)]
    return sort(comps; by = c -> (length(c), c))
end

"""
    permuted_keldysh(k, p) -> (k_p, η̄̂, μ)

The permuted Keldysh component of p. 13, for `k = [η_1 … η_α]` and permutation
`p`:

    μ_j = p⁻¹(η_j) ,     [η̂_1 … η̂_α] = sort{μ_1, …, μ_α} ,     η̄̂_j = p(η̂_j) .

A 2 in slot `η_j` of `k` is carried by `p` to slot `μ_j` of `k_p` — this needs
the **inverse** permutation, because `p` acts on slot contents. The returned
`k_p = [η̂_1 … η̂_α]` is the permuted component; `η̄̂` lists, in the order of the
permuted list, which *external* label each of its 2's came from. Each `η̄̂_j` is
some `η_{j'}`, with `j` and `j'` related in a `p`-dependent way — that is how the
imaginary parts in Eq. (67b) come to be fixed by the external indices while the
sum runs over the permuted list.

`μ` is returned too, unsorted, since p. 13's worked example quotes it:
`k = 1212 = [24]` with `p = (4123)` gives `[μ₁μ₂] = [31]` and `k_p = 2121 = [13]`.
"""
function permuted_keldysh(k::AbstractVector{<:Integer}, p::AbstractVector{<:Integer})
    pinv = invperm(p)
    μ = [pinv[η] for η in k]
    khat = sort(μ)
    ηbarhat = [p[h] for h in khat]
    return khat, ηbarhat, μ
end

# ---------------------------------------------------------------------------
# K3: kernels
# ---------------------------------------------------------------------------

"""
    _partial_product(z, p, ωprime) -> ComplexF64

`∏_{i=1}^{ℓ-1} 1 / (z_{1̄⋯ī} - ω'_{1̄⋯ī})`, the product shared by every kernel
below: `z` is a complex tuple over the *external* slots, and its partial sums
are taken in the permuted order `p`.
"""
function _partial_product(z::AbstractVector{<:Number}, p::AbstractVector{<:Integer},
                          ωprime::AbstractVector{<:Real})
    acc = zero(ComplexF64)
    K = one(ComplexF64)
    for i in 1:(length(z) - 1)
        acc += z[p[i]]
        K /= (acc - ωprime[i])
    end
    return K
end

"""
    retarded_kernel(ω, ωprime, η, p, γ0) -> ComplexF64

The fully retarded kernel of Kugler Eq. (52), convolved with a PSF term:

    K^[η](ω_p - ω'_p) = ∏_{i=1}^{ℓ-1} 1 / (ω^[η̄]_{1̄⋯ī} - ω'_{1̄⋯ī}) ,    η̄ = p(η) .

Here `η` labels a slot of the **permuted** tuple, as in Eq. (52), so the complex
tuple used is `ω^[p(η)]` over the external slots; the positive imaginary part
then lands in permuted slot `η`, the Fourier partner of the largest time.
`ω` is the external real tuple and `ωprime` the PSF term's partial sums.
"""
function retarded_kernel(ω::AbstractVector{<:Real}, ωprime::AbstractVector{<:Real},
                         η::Integer, p::AbstractVector{<:Integer}, γ0::Real)
    return _partial_product(omega_eta(ω, p[η], γ0), p, ωprime)
end

"""
    keldysh_kernel(ω, ωprime, k, p, γ0) -> ComplexF64

The Keldysh-basis kernel of Kugler Eq. (67b), for the external component
`k = [η_1 … η_α]` and permutation `p`:

    K^[η̂_1…η̂_α](ω_p - ω'_p) = Σ_{j=1}^{α} (-1)^{j-1} ∏_{i=1}^{ℓ-1} 1 / (ω^[η̄̂_j]_{1̄⋯ī} - ω'_{1̄⋯ī}) .

The `α` summands differ **only** in which complex tuple `ω^[η̄̂_j]` enters; the
product structure is identical. The superscript is the external label `η̄̂_j`,
while the sign `(-1)^{j-1}` counts position in the *permuted* list — using
`η̂_j` in the superscript instead is right for `α = 1` and wrong for `α ≥ 2`.

`α = 0` returns 0: `K^[] = K^{1⋯1} = 0` follows from Eq. (63) (p. 13).
"""
function keldysh_kernel(ω::AbstractVector{<:Real}, ωprime::AbstractVector{<:Real},
                        k::AbstractVector{<:Integer}, p::AbstractVector{<:Integer}, γ0::Real)
    isempty(k) && return zero(ComplexF64)
    _, ηbarhat, _ = permuted_keldysh(k, p)
    K = zero(ComplexF64)
    for (j, η) in enumerate(ηbarhat)
        K += (-1)^(j - 1) * _partial_product(omega_eta(ω, η, γ0), p, ωprime)
    end
    return K
end

"""
    keldysh_kernel_eq63(ω, ωprime, k, p, γ0) -> ComplexF64

The same kernel built independently from Kugler Eq. (63) and Eq. (52):

    K^{k_p} = Σ_{λ=1}^{ℓ} (-1)^{λ-1} (-1)^{k_{1̄⋯(λ-1)̄}} [1 + (-1)^{k_λ̄}]/2 · K^[λ](ω_p) ,

summing over permuted slots `λ` and reading the permuted digits
`k_ī = k_{p(i)}` directly, with no ordered lists at all. For `λ = η̂_j` the two
signs collapse to `(-1)^{j-1}` and `K^[λ]` uses `ω^[p(λ)] = ω^[η̄̂_j]`, so this
must agree with [`keldysh_kernel`](@ref) term by term. Kept as a cross-check of
the K2 bookkeeping, which it bypasses.
"""
function keldysh_kernel_eq63(ω::AbstractVector{<:Real}, ωprime::AbstractVector{<:Real},
                             k::AbstractVector{<:Integer}, p::AbstractVector{<:Integer}, γ0::Real)
    ℓ = length(ω)
    digits = keldysh_digits(k, ℓ)
    kp = [digits[p[i]] for i in 1:ℓ]                  # k_ī = k_{p(i)}
    K = zero(ComplexF64)
    ksum = 0                                          # k_{1̄⋯(λ-1)̄}
    for λ in 1:ℓ
        if kp[λ] == 2                                 # [1 + (-1)^{k_λ̄}]/2 = 1
            K += (-1)^(λ - 1) * (-1)^ksum * retarded_kernel(ω, ωprime, λ, p, γ0)
        end
        ksum += kp[λ]
    end
    return K
end

# ---------------------------------------------------------------------------
# K4: assembly, Eq. (67a)
# ---------------------------------------------------------------------------

"""
    keldysh_correlator(m, Os, ω, k; γ0, kwargs...) -> ComplexF64

One Keldysh component of the `ℓ`p correlator, Kugler Eq. (67a):

    G^[η_1…η_α](ω) = (2/2^{ℓ/2}) Σ_p ζ^p ∫ d^{ℓ-1}ω'_p K^[η̂_1…η̂_α](ω_p - ω'_p) S[O_p](ω'_p) .

`Os` in its unpermuted order, `ω` a real tuple with `Σω = 0`, and
`k = [η_1 … η_α]` the component. The permutation loop and `ζ^p` are the
Matsubara code's, via [`permuted_psfs`](@ref); the only changes are the kernel
and the prefactor `2^{1-ℓ/2}`. The `ω'` integral stays a finite sum over the
PSF's delta peaks — no grid, no broadening.

Keywords: `γ0` (required, > 0), `is_fermionic`, `sp`, `part` (`:full`,
`:connected` or `:disconnected`, via Eq. (31)), `cache` (a precomputed
[`permuted_psfs`](@ref), which must match `part`), and `kernel_impl`
(`:eq67b`, default, or `:eq63` for the independent route).
"""
function keldysh_correlator(m::HubbardAtomModel, Os::Tuple, ω::AbstractVector{<:Real},
                            k::AbstractVector{<:Integer}; γ0::Real,
                            is_fermionic::AbstractVector{Bool} = fill(true, length(Os)),
                            sp::Spectrum = spectrum(m), part::Symbol = :full,
                            cache::Union{Nothing,Vector{PermutedPSF}} = nothing,
                            kernel_impl::Symbol = :eq67b)
    ℓ = length(Os)
    length(ω) == ℓ || throw(DimensionMismatch("got $(length(ω)) frequencies for $ℓ operators"))
    abs(sum(ω)) ≤ 1e-12 * max(1.0, maximum(abs, ω)) ||
        throw(ArgumentError("external frequencies must conserve energy, Σω = 0; got $(sum(ω))"))
    kernel_impl in (:eq67b, :eq63) ||
        throw(ArgumentError("kernel_impl must be :eq67b or :eq63, got $kernel_impl"))
    isempty(k) && return zero(ComplexF64)           # K^[] = 0: the 1⋯1 component
    kfun = kernel_impl === :eq67b ? keldysh_kernel : keldysh_kernel_eq63

    data = cache === nothing ?
           permuted_psfs(sp, Os; is_fermionic = is_fermionic, part = part) : cache
    total = zero(ComplexF64)
    for d in data, t in d.terms
        total += d.ζ * kfun(ω, t.position, k, d.p, γ0) * t.weight
    end
    return 2.0^(1 - ℓ / 2) * total
end

"""
    keldysh_components(m, Os, ω; γ0, kwargs...) -> Dict{String,ComplexF64}

All `2^ℓ` Keldysh components at one frequency, keyed by the paper's digit
strings (`"1111"`, `"2111"`, …). The PSFs are computed once and shared.
"""
function keldysh_components(m::HubbardAtomModel, Os::Tuple, ω::AbstractVector{<:Real};
                            γ0::Real, is_fermionic::AbstractVector{Bool} = fill(true, length(Os)),
                            sp::Spectrum = spectrum(m), part::Symbol = :full,
                            cache::Union{Nothing,Vector{PermutedPSF}} = nothing)
    ℓ = length(Os)
    data = cache === nothing ?
           permuted_psfs(sp, Os; is_fermionic = is_fermionic, part = part) : cache
    return Dict(keldysh_label(k, ℓ) =>
                keldysh_correlator(m, Os, ω, k; γ0 = γ0, is_fermionic = is_fermionic,
                                   sp = sp, cache = data)
                for k in all_keldysh_components(ℓ))
end
