# The Hubbard dimer: two sites, four fermionic modes, 16 states.
#
# Nothing in Eqs. (28), (39), (46), (67) is specific to the atom: Eq. (28) is a
# Lehmann sum over arbitrary operators O^i, and locality enters the paper only
# from Sec. III on. So the dimer reuses the whole spectral machinery and only
# supplies a new Hamiltonian, new operators, and the momentum bookkeeping of
# the 4p vertex. No paper in the project treats the dimer; the closed forms
# below are the task sheet's, and they are checked, not assumed.

"""
    HubbardDimerModel(t, U, β)

Two-site Hubbard model at half filling,

    H = -t Σ_σ (c†_{1σ} c_{2σ} + c†_{2σ} c_{1σ}) + U Σ_i n_{i↑} n_{i↓} - (U/2) Σ_{iσ} n_{iσ} ,

particle-hole symmetric like the atom at `εd = -U/2`. With hopping `-t`, the
momentum `k = 0` orbital is bonding (single-particle energy `-t`) and `k = π`
antibonding (`+t`). `t = 0` is allowed: two decoupled Hubbard atoms.
"""
struct HubbardDimerModel <: AbstractModel
    t::Float64
    U::Float64
    β::Float64

    function HubbardDimerModel(t::Real, U::Real, β::Real)
        β > 0 || throw(ArgumentError("β must be positive, got $β"))
        isfinite(t) && isfinite(U) || throw(ArgumentError("t and U must be finite"))
        return new(float(t), float(U), float(β))
    end
end

# ---------------------------------------------------------------------------
# Operators
# ---------------------------------------------------------------------------

"""
    jordan_wigner(nmodes; strings = true) -> Vector{Matrix{Float64}}

Annihilation operators of `nmodes` fermionic modes on the `2^nmodes`-dimensional
Fock space with basis index `1 + Σ_j n_j 2^(j-1)`, so mode 1 is the fastest
factor. Mode `j` carries the parity string `(-1)^{n_1 + ⋯ + n_{j-1}}` over every
mode before it:

    c_j = 1 ⊗ ⋯ ⊗ 1 ⊗ f ⊗ z ⊗ ⋯ ⊗ z .

`strings = false` drops the parity factors. That gives hard-core bosons, which
commute between different modes instead of anticommuting; it exists only so the
tests can show the anticommutator assertion catching it.
"""
function jordan_wigner(nmodes::Integer; strings::Bool = true)
    string_op = strings ? _Z : _I
    return [foldl(kron, [mode > j ? _I : mode == j ? _F : string_op for mode in nmodes:-1:1])
            for j in 1:nmodes]
end

_spin_index(σ::Symbol) = σ === :up ? 1 : σ === :dn ? 2 :
    throw(ArgumentError("spin must be :up or :dn, got $σ"))

"""
    DimerOperators

The 16×16 operators of the dimer in the Fock basis.

  - `c` — the four mode annihilators in Jordan-Wigner order `1↑, 1↓, 2↑, 2↓`;
    use [`site_op`](@ref).
  - `ck` — momentum annihilators `c_{kσ} = (c_{1σ} + e^{ik} c_{2σ})/√2`, stored
    in the order `(k=0,↑), (k=0,↓), (k=π,↑), (k=π,↓)`; use [`momentum_op`](@ref).
  - `N`, `Sz` — total charge and spin.
  - `P` — site exchange `1 ↔ 2`, see [`site_exchange`](@ref).
  - `H` — the Hamiltonian, built as products of the matrices above and never by
    entering matrix elements by hand.
"""
struct DimerOperators
    c::Vector{Matrix{Float64}}
    ck::Vector{Matrix{Float64}}
    N::Matrix{Float64}
    Sz::Matrix{Float64}
    P::Matrix{Float64}
    H::Matrix{Float64}
end

"""    site_op(ops, i, σ) — the annihilator `c_{iσ}`, site `i ∈ (1, 2)`."""
function site_op(ops::DimerOperators, i::Integer, σ::Symbol)
    i in (1, 2) || throw(ArgumentError("site must be 1 or 2, got $i"))
    return ops.c[2(i - 1) + _spin_index(σ)]
end

"""
    momentum_op(ops, q, σ) — the annihilator `c_{kσ}` with `k = qπ`, `q ∈ (0, 1)`.

Momenta are carried as the integer `q`, so that the selection rule
`k₁ - k₂ + k₃ - k₄ ≡ 0 (mod 2π)` is the exact test `iseven(q₁ + q₂ + q₃ + q₄)`.
"""
function momentum_op(ops::DimerOperators, q::Integer, σ::Symbol)
    q in (0, 1) || throw(ArgumentError("q must be 0 (k = 0) or 1 (k = π), got $q"))
    return ops.ck[2q + _spin_index(σ)]
end

"""
    site_exchange(ops) -> Matrix

`P = ∏_σ (1 - 2 n_{πσ}) = (-1)^{N_π}`: the unitary of the site exchange `1 ↔ 2`.
It maps `c_{0σ} → c_{0σ}` and `c_{πσ} → -c_{πσ}`, hence `c_{1σ} ↔ c_{2σ}`, and
commutes with `H`. Every correlator with an odd number of `k = π` operators is
odd under `P` and so vanishes: the D5 selection rule.
"""
site_exchange(ops::DimerOperators) = ops.P

"""
    operators(m::HubbardDimerModel; strings = true) -> DimerOperators

Build all operators from [`jordan_wigner`](@ref) in mode order
`1↑, 1↓, 2↑, 2↓`. With this ordering `c†_{1↑} c_{2↑}` crosses mode `1↓` and
picks up its string sign automatically, because it is formed as a matrix product
of the two mode operators.
"""
function operators(m::HubbardDimerModel; strings::Bool = true)
    c = jordan_wigner(4; strings = strings)
    cd = permutedims.(c)
    n = [cd[j] * c[j] for j in 1:4]
    c1 = (c[1], c[2])                      # site 1, (↑, ↓)
    c2 = (c[3], c[4])                      # site 2, (↑, ↓)
    ck = [(c1[s] + (-1)^q * c2[s]) / sqrt(2) for q in 0:1 for s in 1:2]

    Nop = sum(n)
    Sz = (n[1] - n[2] + n[3] - n[4]) / 2
    nk = [permutedims(A) * A for A in ck]
    P = (I - 2nk[3]) * (I - 2nk[4])        # (-1)^{N_π}

    hop = sum(permutedims(c1[s]) * c2[s] + permutedims(c2[s]) * c1[s] for s in 1:2)
    H = -m.t * hop + m.U * (n[1] * n[2] + n[3] * n[4]) - (m.U / 2) * Nop
    return DimerOperators(c, ck, Nop, Sz, Matrix(P), H)
end

"""
    spectrum(m::HubbardDimerModel; atol = 1e-10) -> Spectrum

Diagonalise the 16×16 `H`. Degenerate blocks are gauge-fixed by
`Q = N + √2 Sz + √3 P`, distinct on every `(N, Sz, P)` sector. The Boltzmann
weights are shifted by `E₀ = min E` — at large `β` the factors `e^{βU}` and
`cosh βc` in `Z` overflow otherwise.
"""
function spectrum(m::HubbardDimerModel; atol::Float64 = 1e-10)
    ops = operators(m)
    return _spectrum(ops.H, ops.N + sqrt(2.0) * ops.Sz + sqrt(3.0) * ops.P, m.β; atol = atol)
end

# ---------------------------------------------------------------------------
# Closed forms (task sheet, Sec. 2)
# ---------------------------------------------------------------------------

"""    dimer_c(m) — `c = √(U²/4 + 4t²)`."""
dimer_c(m::HubbardDimerModel) = sqrt(m.U^2 / 4 + 4m.t^2)

"""    dimer_alpha(m) -> (α₊, α₋) — `α± = ½(1 ± 2t/c)`."""
function dimer_alpha(m::HubbardDimerModel)
    c = dimer_c(m)
    return (0.5 * (1 + 2m.t / c), 0.5 * (1 - 2m.t / c))
end

"""
    dimer_levels_analytic(m) -> Vector{Tuple{Float64,Int}}

The spectrum as `(energy, degeneracy)` pairs:

| energy       | deg. | states                                          |
|:-------------|:-----|:------------------------------------------------|
| `-U/2 - c`   | 1    | N = 2 singlet (ground state for U > 0, t ≠ 0)   |
| `-U`         | 3    | N = 2 triplet                                   |
| `-U/2 - t`   | 4    | N = 1 bonding (×2 spin), N = 3 (×2)             |
| `-U/2 + t`   | 4    | N = 1 antibonding (×2), N = 3 (×2)              |
| `0`          | 3    | N = 0, N = 4, N = 2 antisymmetric-ionic singlet |
| `-U/2 + c`   | 1    | N = 2 excited singlet                           |
"""
function dimer_levels_analytic(m::HubbardDimerModel)
    t, U, c = m.t, m.U, dimer_c(m)
    return [(-U / 2 - c, 1), (-U, 3), (-U / 2 - t, 4), (-U / 2 + t, 4), (0.0, 3), (-U / 2 + c, 1)]
end

"""
    dimer_partition_function_analytic(m; shifted = false) -> Float64

    Z = 3 + 3e^{βU} + 8e^{βU/2} cosh βt + 2e^{βU/2} cosh βc .

`shifted = true` returns `Z e^{βE₀}` with `E₀ = -U/2 - c`, which never
overflows: every exponent is then `≤ 0` because `c ≥ U/2` and `c ≥ 2|t|`.
"""
function dimer_partition_function_analytic(m::HubbardDimerModel; shifted::Bool = false)
    t, U, β, c = m.t, m.U, m.β, dimer_c(m)
    shifted || return 3 + 3exp(β * U) + 8exp(β * U / 2) * cosh(β * t) +
                      2exp(β * U / 2) * cosh(β * c)
    return 3exp(-β * (U / 2 + c)) + 3exp(β * (U / 2 - c)) +
           4 * (exp(β * (t - c)) + exp(-β * (t + c))) + 1 + exp(-2β * c)
end

"""
    dimer_propagator_exact(m, q, z) -> ComplexF64

The task sheet's closed form of `G_k`, at an arbitrary complex frequency `z`
(`z = iν` on the Matsubara axis, `z = ω ± iγ₀` for the retarded/advanced
continuation):

    G_{k=0}(z) = A [α₊/(z - (t-c)) + α₋/(z - (t+c))] + B [1/(z - (U/2-t)) + 1/(z + (U/2+t))] ,
    A = 2e^{βU/2}(cosh βt + cosh βc)/Z ,   B = (3/2)(1 + e^{βU} + 2e^{βU/2} cosh βt)/Z ,

and `G_{k=π}(z) = -G_{k=0}(-z)` by particle-hole symmetry. `A` and `B` are
evaluated with every exponent shifted by `-β(U/2 + c)`, as the task's numerical
note asks.
"""
function dimer_propagator_exact(m::HubbardDimerModel, q::Integer, z::Number)
    q == 1 && return -dimer_propagator_exact(m, 0, -z)
    q == 0 || throw(ArgumentError("q must be 0 or 1, got $q"))
    t, U, β, c = m.t, m.U, m.β, dimer_c(m)
    αp, αm = dimer_alpha(m)
    Zs = dimer_partition_function_analytic(m; shifted = true)
    A = (exp(β * (t - c)) + exp(-β * (t + c)) + 1 + exp(-2β * c)) / Zs
    B = 1.5 * (exp(-β * (U / 2 + c)) + exp(β * (U / 2 - c)) +
               exp(β * (t - c)) + exp(-β * (t + c))) / Zs
    return A * (αp / (z - (t - c)) + αm / (z - (t + c))) +
           B * (1 / (z - (U / 2 - t)) + 1 / (z + (U / 2 + t)))
end

dimer_propagator_exact(m::HubbardDimerModel, q::Integer, ν::MatsubaraFreq) =
    dimer_propagator_exact(m, q, im * value(ν, m.β))

# ---------------------------------------------------------------------------
# Computed objects, with the PSFs cached per operator tuple
# ---------------------------------------------------------------------------

"""
    DimerContext(m; otol = 1e-12)

Bundles the model, its [`Spectrum`](@ref) and [`DimerOperators`](@ref) with
lazily filled caches of [`permuted_psfs`](@ref): one per 2p or 4p operator
tuple (and PSF part). The 4p PSFs are the expensive step — `16⁴ × 24 ≈ 1.6×10⁶`
eigenstate cycles — so each is computed once and then reused across all
frequencies, Keldysh components and `γ₀`.

`otol` is the matrix-element tolerance of [`psf`](@ref). With the default
`1e-12` a 4p PSF keeps a few hundred genuine terms per operator tuple instead of
~4×10⁵ roundoff ones; `otol = 0` reproduces the unfiltered sum (used by the
tests to show that nothing physical is dropped).
"""
struct DimerContext
    m::HubbardDimerModel
    sp::Spectrum
    ops::DimerOperators
    otol::Float64
    cache::Dict{Any,Vector{PermutedPSF}}
end

DimerContext(m::HubbardDimerModel; otol::Real = 1e-12, atol::Float64 = 1e-10) =
    DimerContext(m, spectrum(m; atol = atol), operators(m), float(otol),
                 Dict{Any,Vector{PermutedPSF}}())

_dag(A) = permutedims(A)

"""
    momentum_allowed(qs) -> Bool

Momentum selection rule of D5: `k₁ - k₂ + k₃ - k₄ ≡ 0 (mod 2π)`, i.e. an even
number of `k = π` entries.
"""
momentum_allowed(qs) = iseven(sum(qs))

"""    all_momentum_tuples() — the 16 tuples `(q₁, q₂, q₃, q₄)`, `qᵢ ∈ {0, 1}`."""
all_momentum_tuples() = [(q1, q2, q3, q4) for q1 in 0:1 for q2 in 0:1 for q3 in 0:1 for q4 in 0:1]

_spins(σσ′::Symbol) = σσ′ === :updn ? (:up, :dn) : σσ′ === :upup ? (:up, :up) :
    throw(ArgumentError("σσ′ must be :updn or :upup, got $σσ′"))

"""
    dimer_operators_4p(ops, qs, σσ′) -> NTuple{4,Matrix}

`O = (c_{k₁σ}, c†_{k₂σ}, c_{k₃σ′}, c†_{k₄σ′})`, with `σ = ↑` and `σ′ = ↓` or `↑`
for `σσ′ = :updn` or `:upup`.
"""
function dimer_operators_4p(ops::DimerOperators, qs, σσ′::Symbol)
    σ, σ′ = _spins(σσ′)
    return (momentum_op(ops, qs[1], σ), _dag(momentum_op(ops, qs[2], σ)),
            momentum_op(ops, qs[3], σ′), _dag(momentum_op(ops, qs[4], σ′)))
end

"""
    dimer_psf_cache(ctx, key) -> Vector{PermutedPSF}

The permuted PSFs of one operator tuple, computed on first use. `key` is
`(:k, q, σ)` for `(c_{kσ}, c†_{kσ})`, `(:site, i, j, σ)` for `(c_{iσ}, c†_{jσ})`,
or `(:four, qs, σσ′, part)` for [`dimer_operators_4p`](@ref).
"""
function dimer_psf_cache(ctx::DimerContext, key::Tuple)
    return get!(ctx.cache, key) do
        ops = ctx.ops
        if key[1] === :k
            _, q, σ = key
            A = momentum_op(ops, q, σ)
            permuted_psfs(ctx.sp, (A, _dag(A)); otol = ctx.otol)
        elseif key[1] === :site
            _, i, j, σ = key
            permuted_psfs(ctx.sp, (site_op(ops, i, σ), _dag(site_op(ops, j, σ))); otol = ctx.otol)
        elseif key[1] === :four
            _, qs, σσ′, part = key
            permuted_psfs(ctx.sp, dimer_operators_4p(ops, qs, σσ′); part = part, otol = ctx.otol)
        else
            throw(ArgumentError("unknown cache key $key"))
        end
    end
end

"""
    dimer_propagator(ctx, q, ν; σ = :up) -> ComplexF64

`G_k(iν)` from Eq. (39) with `O = (c_{kσ}, c†_{kσ})`, `k = qπ`.
"""
function dimer_propagator(ctx::DimerContext, q::Integer, ν::MatsubaraFreq; σ::Symbol = :up)
    A = momentum_op(ctx.ops, q, σ)
    return correlator(ctx.m, (A, _dag(A)), [ν, MatsubaraFreq(-ν.m)];
                      sp = ctx.sp, cache = dimer_psf_cache(ctx, (:k, q, σ)))
end

"""
    dimer_site_propagator(ctx, i, j, ν; σ = :up) -> ComplexF64

`G_{ij}(iν)` from Eq. (39) with the site operators `O = (c_{iσ}, c†_{jσ})`.
"""
function dimer_site_propagator(ctx::DimerContext, i::Integer, j::Integer, ν::MatsubaraFreq;
                               σ::Symbol = :up)
    Os = (site_op(ctx.ops, i, σ), _dag(site_op(ctx.ops, j, σ)))
    return correlator(ctx.m, Os, [ν, MatsubaraFreq(-ν.m)];
                      sp = ctx.sp, cache = dimer_psf_cache(ctx, (:site, i, j, σ)))
end

"""
    dimer_correlator_4p(ctx, qs, σσ′, ms; part = :full) -> ComplexF64

The 4p correlator of [`dimer_operators_4p`](@ref), from Eq. (39); `part` splits
the PSF by Eq. (31) first.
"""
function dimer_correlator_4p(ctx::DimerContext, qs, σσ′::Symbol,
                             ms::AbstractVector{MatsubaraFreq}; part::Symbol = :full)
    Os = dimer_operators_4p(ctx.ops, qs, σσ′)
    return correlator(ctx.m, Os, ms; sp = ctx.sp,
                      cache = dimer_psf_cache(ctx, (:four, Tuple(qs), σσ′, part)))
end

_leg(ctx::DimerContext, q, ν, exact::Bool) =
    exact ? dimer_propagator_exact(ctx.m, q, ν) : dimer_propagator(ctx, q, ν)

"""
    dimer_disconnected_4p(ctx, qs, σσ′, ms; exact_legs = false) -> ComplexF64

Eq. (73) with a momentum Kronecker delta attached to each Wick pairing, since
every contraction `⟨c_{k_a} c†_{k_b}⟩` is diagonal in `k`:

    G^dis = β G_{k₁}(iω₁) G_{k₃}(iω₃) (δ_{σσ′} δ_{k₁k₄} δ_{k₂k₃} δ_{ω₂₃,0} - δ_{k₁k₂} δ_{k₃k₄} δ_{ω₁₂,0}) .
"""
function dimer_disconnected_4p(ctx::DimerContext, qs, σσ′::Symbol,
                               ms::AbstractVector{MatsubaraFreq}; exact_legs::Bool = false)
    q1, q2, q3, q4 = qs
    δ23 = (ms[2].m + ms[3].m == 0) && q1 == q4 && q2 == q3 && σσ′ === :upup
    δ12 = (ms[1].m + ms[2].m == 0) && q1 == q2 && q3 == q4
    f = Int(δ23) - Int(δ12)
    f == 0 && return zero(ComplexF64)
    return ctx.m.β * _leg(ctx, q1, ms[1], exact_legs) * _leg(ctx, q3, ms[3], exact_legs) * f
end

"""    dimer_connected_4p(ctx, qs, σσ′, ms) — `G^con = G - G^dis`."""
dimer_connected_4p(ctx::DimerContext, qs, σσ′::Symbol, ms::AbstractVector{MatsubaraFreq};
                   exact_legs::Bool = false) =
    dimer_correlator_4p(ctx, qs, σσ′, ms) - dimer_disconnected_4p(ctx, qs, σσ′, ms; exact_legs)

_legs(ctx, qs, ms, exact) =
    _leg(ctx, qs[1], ms[1], exact) * _leg(ctx, qs[2], MatsubaraFreq(-ms[2].m), exact) *
    _leg(ctx, qs[3], ms[3], exact) * _leg(ctx, qs[4], MatsubaraFreq(-ms[4].m), exact)

"""
    dimer_vertex(ctx, qs, σσ′, ms; exact_legs = false) -> ComplexF64

The vertex in the momentum basis, Eq. (74) applied leg by leg:

    F_{k₁k₂k₃k₄}(iω) = G^con_{k₁k₂k₃k₄}(iω) / [G_{k₁}(iω₁) G_{k₂}(-iω₂) G_{k₃}(iω₃) G_{k₄}(-iω₄)] .

The amputation is done here because the propagator is diagonal in `k`. In the
site basis the legs are 2×2 matrices `G_{ij}` and a scalar division is wrong for
`t ≠ 0`; site components follow by transforming the four legs of `F` afterwards.
"""
function dimer_vertex(ctx::DimerContext, qs, σσ′::Symbol, ms::AbstractVector{MatsubaraFreq};
                      exact_legs::Bool = false)
    return dimer_connected_4p(ctx, qs, σσ′, ms; exact_legs) / _legs(ctx, qs, ms, exact_legs)
end

"""
    dimer_delta_part(ctx, qs, σσ′, ms; channel = [1, 2]) -> ComplexF64

The δ-function contribution of one frequency channel to the vertex: the
anomalous term of Eq. (45) (see [`anomalous_part`](@ref)) applied to the
**connected** PSF of Eq. (31), then amputated like the vertex. For the Hubbard
atom and `channel = [1,2]` this is the `βu² δ_{ω₁₂} th ∏(iωᵢ+u)/∏(iωᵢ)` term of
Eq. (85a).
"""
function dimer_delta_part(ctx::DimerContext, qs, σσ′::Symbol, ms::AbstractVector{MatsubaraFreq};
                          channel::AbstractVector{<:Integer} = [1, 2])
    Os = dimer_operators_4p(ctx.ops, qs, σσ′)
    a = anomalous_part(ctx.m, Os, ms; channel = channel, sp = ctx.sp,
                       cache = dimer_psf_cache(ctx, (:four, Tuple(qs), σσ′, :connected)))
    return a / _legs(ctx, qs, ms, false)
end

"""
    dimer_keldysh_4p(ctx, qs, σσ′, ω, k; γ0, part = :full) -> ComplexF64

One Keldysh component `k = [η₁ … η_α]` of the 4p correlator, Eq. (67a), at a
real tuple `ω` with `Σω = 0`. Same cached PSFs as the Matsubara objects.
"""
function dimer_keldysh_4p(ctx::DimerContext, qs, σσ′::Symbol, ω::AbstractVector{<:Real},
                          k::AbstractVector{<:Integer}; γ0::Real, part::Symbol = :full)
    Os = dimer_operators_4p(ctx.ops, qs, σσ′)
    return keldysh_correlator(ctx.m, Os, ω, k; γ0 = γ0, sp = ctx.sp,
                              cache = dimer_psf_cache(ctx, (:four, Tuple(qs), σσ′, part)))
end
