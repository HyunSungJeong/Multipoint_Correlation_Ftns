# Steps 4-6 of Block 4: assembling Matsubara correlators from the spectral
# representation, Kugler Eq. (39),
#
#     G(iω) = Σ_p ζ_p ∫ d^{ℓ-1}ω'_p K(iω_p - ω'_p) S[O_p](ω'_p) ,
#
# and extracting the 4p vertex from it. Because the PSFs of Eq. (28) are sums
# of deltas, the ω' integral is a finite sum over PSF terms: no grids, no
# broadening, kernels evaluated exactly at the poles.

"""
    all_permutations(n) -> Vector{Vector{Int}}

Every permutation of `1:n`, as index vectors. `ℓ = 4` gives the 24 summands of
Eq. (39). Written out rather than pulled from a package, to keep the
dependencies at stdlib.
"""
function all_permutations(n::Integer)
    n == 0 && return [Int[]]
    out = Vector{Vector{Int}}()
    cur = Int[]
    function rec(rest)
        if isempty(rest)
            push!(out, copy(cur))
            return
        end
        for k in eachindex(rest)
            push!(cur, rest[k])
            rec(deleteat!(copy(rest), k))
            pop!(cur)
        end
    end
    rec(collect(1:n))
    return out
end

"""
    permutation_sign(p, is_fermionic) -> Int

The sign `ζ_p` of Eq. (17a): `+1` (`-1`) if `O_p` differs from `O` by an even
(odd) number of transpositions of **fermionic** operators.

So it is not simply the parity of `p`: bosonic operators — a density `n_σ`, say
— commute through and must not be counted. The sign is the parity of the
permutation induced on the fermionic entries alone, which is the parity of the
number of inversions among them.
"""
function permutation_sign(p::AbstractVector{<:Integer}, is_fermionic::AbstractVector{Bool})
    f = [pi for pi in p if is_fermionic[pi]]
    inversions = 0
    for a in 1:length(f), b in (a + 1):length(f)
        f[a] > f[b] && (inversions += 1)
    end
    return iseven(inversions) ? 1 : -1
end

"""
    correlator(m::HubbardAtomModel, Os, ms; kwargs...) -> ComplexF64

The Matsubara `ℓ`p correlator `G(iω)` of Kugler Eq. (39), for the operator
tuple `Os` in its **unpermuted** order and external frequencies `ms`, a vector
of `ℓ` [`MatsubaraFreq`](@ref) obeying `Σ ω = 0`.

The sum over permutations of Eq. (39) is taken here: for each `p`, the
operators are permuted to `O_p`, [`psf`](@ref) supplies the PSF terms of
Eq. (28) in that order, and each term's `position` pairs with the matching
partial sums `iω_{1̄⋯ī}` to form the composites that [`matsubara_kernel`](@ref)
consumes.

Keywords:

  - `is_fermionic` — which operators are fermionic, for `ζ_p`. Defaults to all.
  - `sp` — a precomputed [`Spectrum`](@ref); pass one to avoid rediagonalising
    when sweeping frequencies.
  - `atol` — tolerance on the spectral part of a vanishing composite.
"""
function correlator(m::HubbardAtomModel, Os::Tuple, ms::AbstractVector{MatsubaraFreq};
                    is_fermionic::AbstractVector{Bool} = fill(true, length(Os)),
                    sp::Spectrum = spectrum(m), atol::Real = 1e-10)
    ℓ = length(Os)
    ℓ ≥ 2 || throw(ArgumentError("need at least ℓ = 2 operators, got $ℓ"))
    length(ms) == ℓ ||
        throw(DimensionMismatch("got $(length(ms)) frequencies for $ℓ operators"))
    length(is_fermionic) == ℓ ||
        throw(DimensionMismatch("is_fermionic must have one entry per operator"))
    sum(f -> f.m, ms) == 0 ||
        throw(ArgumentError("external frequencies must conserve energy, Σω = 0; " *
                            "got Σm = $(sum(f -> f.m, ms))"))

    total = zero(ComplexF64)
    for p in all_permutations(ℓ)
        ζ = permutation_sign(p, is_fermionic)
        Op = ntuple(i -> Os[p[i]], ℓ)
        msp = [ms[p[i]] for i in 1:ℓ]
        for t in psf(sp, Op)
            Ω, vanishing = composites(msp, t.position, m.β; atol = atol)
            total += ζ * matsubara_kernel(Ω, vanishing, m.β) * t.weight
        end
    end
    return total
end

# ---------------------------------------------------------------------------
# Step 4: ℓ = 2
# ---------------------------------------------------------------------------

"""
    propagator(m::HubbardAtomModel, ν::MatsubaraFreq; kwargs...) -> ComplexF64

The single-particle propagator `G(iν) = -⟨T d_σ(τ) d†_σ⟩` in Matsubara
frequency, assembled from Eq. (39) with `O = (d_σ, d†_σ)`.

This is the Step 4 milestone: it must reproduce Appendix E's exact result
`G(iν) = ½ Σ± (iν ± U/2)⁻¹` to machine precision, which
[`propagator_exact`](@ref) supplies.
"""
function propagator(m::HubbardAtomModel, ν::MatsubaraFreq;
                    sp::Spectrum = spectrum(m), ops::Operators = operators(m),
                    atol::Real = 1e-10)
    fermionic(ν) ||
        throw(ArgumentError("the propagator takes a fermionic frequency, got m = $(ν.m)"))
    return correlator(m, (ops.d_up, ops.dag_up), [ν, MatsubaraFreq(-ν.m)];
                      sp = sp, atol = atol)
end

"""
    propagator_exact(m::HubbardAtomModel, ν::MatsubaraFreq) -> ComplexF64

Appendix E's closed form, `G(iν) = ½ Σ± (iν ± U/2)⁻¹`, whose self-energy
`Σ(iν) = U²/(4iν)` is exact for the Hubbard atom. Note it is odd,
`G(-iν) = -G(iν)`.
"""
function propagator_exact(m::HubbardAtomModel, ν::MatsubaraFreq)
    u = half_interaction(m)
    iν = im * value(ν, m.β)
    return 0.5 * (inv(iν + u) + inv(iν - u))
end

# ---------------------------------------------------------------------------
# Step 5: ℓ = 4, the vertex
# ---------------------------------------------------------------------------

"""
    vertex_frequencies(n1, n2, n3) -> Vector{MatsubaraFreq}

The four fermionic frequencies of a 4p function from three integer indices,
with `ω_4 = -ω_{123}` fixed by energy conservation. Since `m_i = 2n_i + 1` are
all odd, `m_4 = -(m_1+m_2+m_3)` is odd too, so the fourth leg is automatically
fermionic for any choice of `n1, n2, n3`.
"""
function vertex_frequencies(n1::Integer, n2::Integer, n3::Integer)
    m1, m2, m3 = 2n1 + 1, 2n2 + 1, 2n3 + 1
    return [MatsubaraFreq(m1), MatsubaraFreq(m2), MatsubaraFreq(m3),
            MatsubaraFreq(-(m1 + m2 + m3))]
end

"""
    spin_operators(ops::Operators, σ::Symbol) -> (d, d†)

The annihilation and creation operator for spin `σ ∈ (:up, :dn)`.
"""
function spin_operators(ops::Operators, σ::Symbol)
    σ === :up && return (ops.d_up, ops.dag_up)
    σ === :dn && return (ops.d_dn, ops.dag_dn)
    throw(ArgumentError("spin must be :up or :dn, got $σ"))
end

"""
    correlator_4p(m, σ, σ′, ms; kwargs...) -> ComplexF64

The full 4p correlator `G_{σσ′}` of Eq. (71), from Eq. (39) with
`O = (d_σ, d†_σ, d_σ′, d†_σ′)` — all 24 permutations.
"""
function correlator_4p(m::HubbardAtomModel, σ::Symbol, σ′::Symbol,
                       ms::AbstractVector{MatsubaraFreq};
                       sp::Spectrum = spectrum(m), ops::Operators = operators(m),
                       atol::Real = 1e-10)
    dσ, dagσ = spin_operators(ops, σ)
    dσ′, dagσ′ = spin_operators(ops, σ′)
    return correlator(m, (dσ, dagσ, dσ′, dagσ′), ms; sp = sp, atol = atol)
end

"""
    disconnected_4p(m, σ, σ′, ms; kwargs...) -> ComplexF64

The disconnected part of Eq. (73),

    G^dis_{σσ′}(iω₁,iω₂,iω₃) = β G(iω₁) G(iω₃) (δ_{σσ′} δ_{ω₂₃,0} - δ_{ω₁₂,0}) .

The two Kronecker deltas are exact integer tests on the Matsubara indices. Note
`δ_{ω₂₃,0} = δ_{ω₁₄,0}` by energy conservation, so this is the same object the
third Wick pairing of Eq. (31) produces; the pairing `⟨d_σ d_σ′⟩⟨d†_σ d†_σ′⟩`
vanishes outright.

By default the legs use the *computed* propagator, so the subtraction is
internally consistent with Eq. (39); pass `exact_legs = true` to use
Appendix E's closed form instead.
"""
function disconnected_4p(m::HubbardAtomModel, σ::Symbol, σ′::Symbol,
                         ms::AbstractVector{MatsubaraFreq};
                         sp::Spectrum = spectrum(m), ops::Operators = operators(m),
                         atol::Real = 1e-10, exact_legs::Bool = false)
    G = exact_legs ? (ν -> propagator_exact(m, ν)) :
                     (ν -> propagator(m, ν; sp = sp, ops = ops, atol = atol))
    δ23 = (ms[2].m + ms[3].m == 0) ? 1 : 0
    δ12 = (ms[1].m + ms[2].m == 0) ? 1 : 0
    spin_factor = (σ === σ′ ? δ23 : 0) - δ12
    spin_factor == 0 && return zero(ComplexF64)
    return m.β * G(ms[1]) * G(ms[3]) * spin_factor
end

"""
    connected_4p(m, σ, σ′, ms; kwargs...) -> ComplexF64

`G^con = G - G^dis`, Eq. (73).
"""
function connected_4p(m::HubbardAtomModel, σ::Symbol, σ′::Symbol,
                      ms::AbstractVector{MatsubaraFreq}; kwargs...)
    return correlator_4p(m, σ, σ′, ms; kwargs...) -
           disconnected_4p(m, σ, σ′, ms; kwargs...)
end

"""
    vertex(m, σ, σ′, ms; kwargs...) -> ComplexF64

The full 4p vertex `F_{σσ′}` of Eq. (74), obtained by amputating the external
legs of the connected correlator:

    F_{σσ′}(iω₁,iω₂,iω₃) = G^con_{σσ′}(iω₁,iω₂,iω₃) / [G(iω₁) G(-iω₂) G(iω₃) G(-iω₄)] .

Note the sign flips on legs 2 and 4, which carry the creation operators.

This is a pure Matsubara calculation, so Appendix D does not enter: its
imaginary frequency shifts are needed only for the Keldysh amputation of
Eq. (76), where the legs are evaluated at real frequencies. On the Matsubara
axis the legs sit exactly on discrete frequencies and there is nothing to
shift.
"""
function vertex(m::HubbardAtomModel, σ::Symbol, σ′::Symbol,
                ms::AbstractVector{MatsubaraFreq};
                sp::Spectrum = spectrum(m), ops::Operators = operators(m),
                atol::Real = 1e-10, exact_legs::Bool = false)
    kw = (sp = sp, ops = ops, atol = atol)
    Gcon = correlator_4p(m, σ, σ′, ms; kw...) -
           disconnected_4p(m, σ, σ′, ms; kw..., exact_legs = exact_legs)
    G = exact_legs ? (ν -> propagator_exact(m, ν)) : (ν -> propagator(m, ν; kw...))
    legs = G(ms[1]) * G(MatsubaraFreq(-ms[2].m)) * G(ms[3]) * G(MatsubaraFreq(-ms[4].m))
    return Gcon / legs
end

"""
    vertex_exact(m, σσ′, ms) -> ComplexF64

The closed-form Hubbard-atom vertex of Eqs. (85a) and (85b), with `u = U/2`,
`th = tanh(βu/2)` and `i ∈ {1,2,3,4}`:

    F↑↓ = 2u + u³ Σᵢ(iωᵢ)² / ∏ᵢ(iωᵢ) - 6u⁵ / ∏ᵢ(iωᵢ)
          + β u² [δ_{ω₁₂} th + δ_{ω₁₃}(th-1) + δ_{ω₁₄}(th+1)] / [∏ᵢ(iωᵢ+u) ∏ᵢ(iωᵢ)] ,

    F↑↑ = β u² (δ_{ω₁₄} - δ_{ω₁₂}) / [∏ᵢ(iωᵢ+u) ∏ᵢ(iωᵢ)] .

`σσ′` is `:updn` or `:upup`. The deltas are `δ_{ω,0}` on the composite
frequencies and are evaluated as exact integer tests. Both denominators are
safe: `ωᵢ` is fermionic and so never zero, and `iωᵢ + u` cannot vanish for real
`u` and real nonzero `ωᵢ`.
"""
function vertex_exact(m::HubbardAtomModel, σσ′::Symbol, ms::AbstractVector{MatsubaraFreq})
    u, β = half_interaction(m), m.β
    iω = [im * value(f, β) for f in ms]
    prod_iω = prod(iω)
    prod_iω_u = prod(iω .+ u)
    th = tanh(β * u / 2)
    δ(k) = (ms[1].m + ms[k].m == 0) ? 1.0 : 0.0
    δ12, δ13, δ14 = δ(2), δ(3), δ(4)

    if σσ′ === :updn
        return 2u + u^3 * sum(x -> x^2, iω) / prod_iω - 6u^5 / prod_iω +
               β * u^2 * (δ12 * th + δ13 * (th - 1) + δ14 * (th + 1)) *
               prod_iω_u / prod_iω
    elseif σσ′ === :upup
        return β * u^2 * (δ14 - δ12) * prod_iω_u / prod_iω
    end
    throw(ArgumentError("σσ′ must be :updn or :upup, got $σσ′"))
end
