# Step 2 of Block 4: partial spectral functions, Kugler Eq. (28).
#
# Kept general in the number of operators ℓ, so the one implementation serves
# ℓ = 2 (Step 4), ℓ = 3, and ℓ = 4 (Step 5).

"""
    PSFTerm

One delta contribution to a partial spectral function.

  - `position::Vector{Float64}` — the `ℓ-1` partial sums
    `ω'_{1̄⋯ī} = E_{i+1} - E_1` at which the deltas of Eq. (28) sit, in the
    *permuted* operator order handed to [`psf`](@ref). These are exactly the
    arguments the kernel of Eq. (46) consumes, via
    `Ω_{1̄⋯ī} = iω_{1̄⋯ī} - ω'_{1̄⋯ī}`.
  - `weight::ComplexF64` — the accompanying weight
    `ρ_1 ∏_{i=1}^{ℓ} (O_ī)_{i,i+1}`, the product running cyclically so that the
    last factor is `(O_ℓ̄)_{ℓ,1}`.
  - `states::Vector{Int}` — the eigenstate cycle `(1, …, ℓ)` that produced the
    term, kept for debugging. Empty for terms returned by
    [`aggregate_psf`](@ref), which merges several cycles.

Because the PSFs are sums of deltas, the `ω'` integral of Eq. (39) is a finite
sum over these terms: no grids and no broadening.
"""
struct PSFTerm
    position::Vector{Float64}
    weight::ComplexF64
    states::Vector{Int}
end

"""
    npoint(t::PSFTerm) -> Int

The number of operators `ℓ` the term belongs to, i.e. `length(position) + 1`.
"""
npoint(t::PSFTerm) = length(t.position) + 1

"""
    psf(sp::Spectrum, Os; wtol = 0.0) -> Vector{PSFTerm}
    psf(sp::Spectrum, O1, O2, ...; wtol = 0.0)

Partial spectral function of Kugler Eq. (28),

    S[O_p](ω'_p) = Σ_{1⋯ℓ} ρ_1 [ ∏_{i=1}^{ℓ-1} (O_ī)_{i,i+1} δ(ω'_{1̄⋯ī} - E_{i+1,1}) ] (O_ℓ̄)_{ℓ,1}

with `E_{i+1,1} = E_{i+1} - E_1`, returned as a list of `(position, weight)`
pairs — one [`PSFTerm`](@ref) per eigenstate cycle with nonvanishing weight.

`Os` is the operator tuple **already in the permuted order** `O_p = (O_1̄, …, O_ℓ̄)`;
this function implements Eq. (28) for one given `p`. Steps 4 and 5 loop over
permutations and call it once per permutation, pairing each returned `position`
with the matching `iω_{1̄⋯ī}`. Operators are passed in the Fock basis and
rotated internally with [`to_eigenbasis`](@ref).

Combining Eqs. (26)–(28) gives Eq. (29), whose numerator
`∏_{i=1}^{ℓ} (O_ī)_{i,i+1}` is the cyclic product stored in `weight`.

`wtol` drops terms whose weight has modulus `≤ wtol`; the default `0.0` keeps
everything except exact zeros, so no contribution is silently discarded — at
large `βU` legitimate weights can be as small as `e^{-βU}`.
"""
function psf(sp::Spectrum, Os::Tuple; wtol::Real = 0.0)
    ℓ = length(Os)
    ℓ ≥ 2 || throw(ArgumentError("Eq. (28) needs at least ℓ = 2 operators, got $ℓ"))
    n = length(sp.E)
    all(O -> size(O) == (n, n), Os) ||
        throw(DimensionMismatch("every operator must be $n×$n to match the spectrum"))

    Orot = map(O -> ComplexF64.(to_eigenbasis(sp, O)), Os)
    E = sp.E
    terms = PSFTerm[]

    for states in Iterators.product(ntuple(_ -> 1:n, ℓ)...)
        ρ1 = sp.ρ[states[1]]
        iszero(ρ1) && continue

        # weight = ρ_1 ∏_{i=1}^{ℓ} (O_ī)_{i,i+1}, cyclically so that i = ℓ
        # contributes (O_ℓ̄)_{ℓ,1}
        w = ComplexF64(ρ1)
        for i in 1:ℓ
            j = i == ℓ ? 1 : i + 1
            w *= Orot[i][states[i], states[j]]
            iszero(w) && break
        end
        abs(w) > wtol || continue

        # delta positions: the partial sums ω'_{1̄⋯ī} = E_{i+1} - E_1
        pos = [E[states[i + 1]] - E[states[1]] for i in 1:(ℓ - 1)]
        push!(terms, PSFTerm(pos, w, collect(states)))
    end

    return terms
end

psf(sp::Spectrum, Os::AbstractMatrix...; kwargs...) = psf(sp, Os; kwargs...)

"""
    aggregate_psf(terms; postol = 1e-10, wtol = 0.0) -> Vector{PSFTerm}

Merge terms sitting at the same `position` by summing their weights, and return
them sorted by position. Several eigenstate cycles can produce coincident
deltas — the Hubbard atom's spectrum is degenerate, so this is the rule rather
than the exception — and the merged list is the PSF as an actual measure.

`postol` sets the resolution at which two positions count as equal; `wtol`
drops merged weights of modulus `≤ wtol`, the default `0.0` dropping only
those that cancel exactly.

Aggregating is optional: Eq. (39) is linear in the PSF, so Steps 4-6 give the
same answer from the raw or the merged list. It matters for reading a PSF off
the screen, and for checking pole positions and residues against closed forms.
"""
function aggregate_psf(terms::Vector{PSFTerm}; postol::Real = 1e-10, wtol::Real = 0.0)
    isempty(terms) && return PSFTerm[]
    digits = max(0, round(Int, -log10(postol)))

    acc = Dict{Vector{Float64}, ComplexF64}()
    order = Vector{Float64}[]
    for t in terms
        key = round.(t.position; digits = digits)
        haskey(acc, key) || push!(order, key)
        acc[key] = get(acc, key, zero(ComplexF64)) + t.weight
    end

    out = PSFTerm[]
    for key in order
        abs(acc[key]) > wtol || continue
        push!(out, PSFTerm(key, acc[key], Int[]))
    end
    sort!(out; by = t -> t.position)
    return out
end

"""
    psf_frequencies(t::PSFTerm) -> Vector{Float64}

The individual frequencies `(ω'_1̄, …, ω'_ℓ̄)` behind a term's partial sums,

    ω'_1̄ = position[1],   ω'_ī = position[i] - position[i-1],   ω'_ℓ̄ = -position[ℓ-1],

which is to say `ω'_ī = E_{i+1} - E_i` across the cycle. They satisfy
`ω'_{1̄⋯ℓ̄} = 0`, the energy conservation understood throughout Sec. II.
"""
function psf_frequencies(t::PSFTerm)
    P = t.position
    ℓ = length(P) + 1
    ω = Vector{Float64}(undef, ℓ)
    ω[1] = P[1]
    for i in 2:(ℓ - 1)
        ω[i] = P[i] - P[i - 1]
    end
    ω[ℓ] = -P[ℓ - 1]
    return ω
end

"""
    psf_total_weight(terms) -> ComplexF64

Sum of all weights in a PSF. Summing Eq. (28) over every eigenstate cycle
collapses the cyclic product to a trace,

    Σ_terms weight = Σ_1 ρ_1 (O_1̄ O_2̄ ⋯ O_ℓ̄)_{1,1} = tr(ρ O_1̄ O_2̄ ⋯ O_ℓ̄) = ⟨O_1̄ O_2̄ ⋯ O_ℓ̄⟩ ,

so the total weight of a PSF is the equal-time expectation value of the
operator product, in the same order. This is the sum rule Step 2 is checked
against, and it is independent of `ℓ`.
"""
psf_total_weight(terms::Vector{PSFTerm}) = sum(t -> t.weight, terms; init = zero(ComplexF64))
