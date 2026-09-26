# The Anderson impurity model with a discrete bath of N_b sites.
#
# Eq. (79) of Kugler, Lee & von Delft, PRX 11, 041006 (2021) at the particle-
# hole symmetric point εd = -U/2, with the band replaced by N_b bath sites:
#
#     H = -U/2 Σ_σ n_dσ + U n_d↑ n_d↓ + Σ_{n,σ} [ε_n b†_nσ b_nσ + V_n (d†_σ b_nσ + b†_nσ d_σ)] .
#
# The bath enters impurity correlators only through the hybridisation function,
# Δ(ν) = Σ_ε π|V_ε|² δ(ν - ε), box-shaped Δ θ(D - |ν|) in the paper (p. 16); on
# the Matsubara axis the discrete version is Δ(iν) = Σ_n |V_n|² / (iν - ε_n).
# Everything downstream of `spectrum(m)` is the model-independent machinery of
# the earlier tasks; this file supplies the Hamiltonian, the operators and the
# closed forms used as checks.

"""
    discretize_box(Δ, D, edges) -> (ε, V)

Discretise the box hybridisation `Δ(ν) = Δ θ(D - |ν|)` with one bath site per
interval `[x_{n-1}, x_n]` of `edges`, matching the zeroth and first moment of
the box on each interval:

    |V_n|² = (Δ/π)(x_n - x_{n-1}) ,     ε_n = (x_{n-1} + x_n) / 2 .

Summing gives the sum rule `Σ_n |V_n|² = 2DΔ/π` for any partition of `[-D, D]`;
symmetric edges keep the model particle-hole symmetric. `V_n ≥ 0` is returned.
"""
function discretize_box(Δ::Real, D::Real, edges::AbstractVector{<:Real})
    length(edges) ≥ 2 || throw(ArgumentError("need at least two edges"))
    issorted(edges; lt = <=) || throw(ArgumentError("edges must be strictly increasing"))
    Δ ≥ 0 || throw(ArgumentError("Δ must be non-negative"))
    x = float.(edges)
    ε = [(x[n] + x[n + 1]) / 2 for n in 1:(length(x) - 1)]
    V = [sqrt(Δ / π * (x[n + 1] - x[n])) for n in 1:(length(x) - 1)]
    return ε, V
end

"""
    box_edges(Nb, D = 1; Λ = nothing) -> Vector{Float64}

The partitions of `[-D, D]` used for the benchmarks: `{-D, D}` for `Nb = 1`,
`{-D, 0, D}` for 2, `{-D, -D/3, D/3, D}` for 3. With `Λ` given and `Nb = 3`, the
logarithmic partition `{-D, -D/Λ, D/Λ, D}`; for `Nb ≤ 2` the two coincide.
"""
function box_edges(Nb::Integer, D::Real = 1.0; Λ::Union{Nothing,Real} = nothing)
    Nb == 1 && return [-float(D), float(D)]
    Nb == 2 && return [-float(D), 0.0, float(D)]
    Nb == 3 && return Λ === nothing ? [-D, -D / 3, D / 3, D] .* 1.0 : [-D, -D / Λ, D / Λ, D] .* 1.0
    throw(ArgumentError("presets exist for Nb = 1, 2, 3 (the task stops at Nb = 3); pass edges directly"))
end

"""
    AIMModel(U, β, ε, V)

Anderson impurity model at `εd = -U/2` with bath levels `ε` and real hoppings
`V` (one per bath site). `Nb = length(ε)`; the Fock space has `4^(Nb+1)` states.
Build the benchmark models with [`aim_model`](@ref).
"""
struct AIMModel <: ImpurityModel
    U::Float64
    β::Float64
    ε::Vector{Float64}
    V::Vector{Float64}

    function AIMModel(U::Real, β::Real, ε::AbstractVector{<:Real}, V::AbstractVector{<:Real})
        β > 0 || throw(ArgumentError("β must be positive, got $β"))
        length(ε) == length(V) || throw(DimensionMismatch("one hopping per bath level"))
        isempty(ε) && throw(ArgumentError("need at least one bath site"))
        all(isfinite, ε) && all(isfinite, V) && isfinite(U) ||
            throw(ArgumentError("parameters must be finite"))
        return new(float(U), float(β), float.(collect(ε)), float.(collect(V)))
    end
end

"""
    aim_model(Nb; U, β, Δ = 0.1, D = 1, edges = box_edges(Nb, D)) -> AIMModel

The benchmark model: the box of half-width `D` and height `Δ` discretised by
[`discretize_box`](@ref) on `edges`.
"""
function aim_model(Nb::Integer; U::Real, β::Real, Δ::Real = 0.1, D::Real = 1.0,
                   edges::AbstractVector{<:Real} = box_edges(Nb, D))
    length(edges) == Nb + 1 || throw(ArgumentError("need Nb + 1 = $(Nb + 1) edges"))
    ε, V = discretize_box(Δ, D, edges)
    return AIMModel(U, β, ε, V)
end

"""    nbath(m::AIMModel) — the number of bath sites `N_b`."""
nbath(m::AIMModel) = length(m.ε)

default_otol(::AIMModel) = 1e-12

"""
    AIMOperators

Operators of the AIM in the Fock basis: `c` holds the `2(Nb+1)` mode
annihilators in Jordan-Wigner order `d↑, d↓, b₁↑, b₁↓, …`; `N`, `Sz` total
charge and spin; `H` the Hamiltonian, built from products of the mode matrices.
"""
struct AIMOperators
    c::Vector{Matrix{Float64}}
    N::Matrix{Float64}
    Sz::Matrix{Float64}
    H::Matrix{Float64}
end

"""    bath_op(ops, n, σ) — the annihilator `b_{nσ}` of bath site `n`."""
bath_op(ops::AIMOperators, n::Integer, σ::Symbol) = ops.c[2n + _spin_index(σ)]

impurity_operators(::AIMModel, ops::AIMOperators, σ::Symbol) =
    (ops.c[_spin_index(σ)], permutedims(ops.c[_spin_index(σ)]))

"""
    operators(m::AIMModel) -> AIMOperators

All operators from [`jordan_wigner`](@ref)`(2(Nb+1))`. The hybridisation
`d†_σ b_nσ` crosses every mode between `d_σ` and `b_nσ` and picks up those
string signs automatically, because it is a product of the mode matrices.
"""
function operators(m::AIMModel)
    Nb = nbath(m)
    c = jordan_wigner(2(Nb + 1))
    n = [permutedims(A) * A for A in c]
    Nop = sum(n)
    Sz = sum(n[2k + 1] - n[2k + 2] for k in 0:Nb) / 2
    H = -(m.U / 2) * (n[1] + n[2]) + m.U * (n[1] * n[2])
    for k in 1:Nb, s in 1:2
        d = c[s]; b = c[2k + s]
        H += m.ε[k] * n[2k + s] + m.V[k] * (permutedims(d) * b + permutedims(b) * d)
    end
    return AIMOperators(c, Nop, Sz, H)
end

"""
    spectrum(m::AIMModel; atol = 1e-10) -> Spectrum

Diagonalise `H` (`4^(Nb+1)` states); degenerate blocks gauge-fixed by
`N + √2 Sz`; Boltzmann weights shifted by the ground-state energy.
"""
function spectrum(m::AIMModel; atol::Float64 = 1e-10)
    ops = operators(m)
    return _spectrum(ops.H, ops.N + sqrt(2.0) * ops.Sz, m.β; atol = atol)
end

"""    hybridization(m, z) — `Δ(z) = Σ_n |V_n|² / (z - ε_n)` at complex `z`."""
hybridization(m::AIMModel, z::Number) = sum(abs2(m.V[n]) / (z - m.ε[n]) for n in eachindex(m.ε))

"""
    propagator_free(m, z) -> ComplexF64

The `U = 0` impurity propagator `G_d(z) = [z - Δ(z)]⁻¹` with the level at 0
(Hartree-shifted, particle-hole symmetric), at any complex `z`: `iν` on the
Matsubara axis, `ω + iγ₀` for the retarded continuation (A3).
"""
propagator_free(m::AIMModel, z::Number) = inv(z - hybridization(m, z))

"""
    free_poles(m) -> (e, r)

Poles `e_a` and residues `r_a` of [`propagator_free`](@ref),
`G₀(z) = Σ_a r_a / (z - e_a)`, from the single-particle matrix
`[0 Vᵀ; V diag(ε)]`: `r_a = |⟨d|a⟩|²`, `Σ_a r_a = 1`.
"""
function free_poles(m::AIMModel)
    Nb = nbath(m)
    h = zeros(Nb + 1, Nb + 1)
    h[1, 2:end] = m.V; h[2:end, 1] = m.V
    for n in 1:Nb
        h[n + 1, n + 1] = m.ε[n]
    end
    F = eigen(Symmetric(h))
    return F.values, abs2.(F.vectors[1, :])
end

"""
    chi0_bubble(m, mω) -> Float64

The bare bubble of B3 [derived on the task sheet], for bosonic `ω_m = mω π/β`
(`mω` even), from the poles of the particle-hole symmetric `G₀`:

    χ₀(iω_m) = -(1/β) Σ_ν G₀(iν) G₀(iν + iω_m)
             = -Σ_{a,b} r_a r_b × { [n_F(e_a) - n_F(e_b)] / (iω_m + e_a - e_b)   unless m = 0, e_a = e_b
                                  { -β n_F(e_a)[1 - n_F(e_a)]                    for m = 0, e_a = e_b .

With this convention the atom (`G₀ = 1/iν`) gives `χ₀ = β/4 δ_{ω,0}` and Eq. (84)
reproduces Appendix E's second-order vertex.
"""
function chi0_bubble(m::AIMModel, mω::Integer)
    iseven(mω) || throw(ArgumentError("χ₀ takes a bosonic frequency, got m = $mω"))
    e, r = free_poles(m)
    β = m.β
    nF(x) = 0.5 * (1 - tanh(β * x / 2))
    χ = zero(ComplexF64)
    for a in eachindex(e), b in eachindex(e)
        if mω == 0 && abs(e[a] - e[b]) ≤ TOL_DEG
            χ += r[a] * r[b] * β * nF(e[a]) * (1 - nF(e[a]))
        else
            χ -= r[a] * r[b] * (nF(e[a]) - nF(e[b])) / (im * mω * π / β + e[a] - e[b])
        end
    end
    return χ
end

"""
    vertex_second_order(m, σσ′, ms) -> ComplexF64

Eq. (84) (p. 16) with the discrete-bath bubble [`chi0_bubble`](@ref):

    F_{σσ′} = U δ_{σ,σ̄′} - U² [δ_{σ,σ′} χ₀(iω₁₂) + δ_{σ,σ̄′} χ₀(iω₁₃) - χ₀(iω₁₄)] .

`σσ′ ∈ (:updn, :upup)`, `ms` integer Matsubara indices of the four legs.
"""
function vertex_second_order(m::AIMModel, σσ′::Symbol, ms)
    mi = [x isa MatsubaraFreq ? x.m : Int(x) for x in ms]
    χ(k) = chi0_bubble(m, mi[1] + mi[k])
    U = m.U
    σσ′ === :updn && return U - U^2 * (χ(3) - χ(4))
    σσ′ === :upup && return -U^2 * (χ(2) - χ(4))
    throw(ArgumentError("σσ′ must be :updn or :upup"))
end

# ---------------------------------------------------------------------------
# N_b = 1 closed forms (A1) [derived on the task sheet]
# ---------------------------------------------------------------------------

function _nb1_ab(m::AIMModel)
    nbath(m) == 1 && abs(m.ε[1]) ≤ TOL_DEG ||
        throw(ArgumentError("the closed forms hold for Nb = 1 with the bath level at ε = 0"))
    V, U = m.V[1], m.U
    return sqrt(U^2 + 64V^2) / 4, sqrt(U^2 + 16V^2) / 4
end

"""
    aim_levels_nb1(m) -> Vector{Tuple{Float64,Int}}

`(energy, degeneracy)` of the `N_b = 1` AIM with the bath site at `ε = 0`,
`a = ¼√(U² + 64V²)`, `b = ¼√(U² + 16V²)`:
`-U/4-a` (1, singlet ground state), `-U/4-b` (4), `-U/2` (3, triplet), `0` (3),
`-U/4+b` (4), `-U/4+a` (1).
"""
function aim_levels_nb1(m::AIMModel)
    a, b = _nb1_ab(m)
    U = m.U
    return [(-U / 4 - a, 1), (-U / 4 - b, 4), (-U / 2, 3), (0.0, 3), (-U / 4 + b, 4), (-U / 4 + a, 1)]
end

"""
    aim_partition_function_nb1(m) -> Float64

`Z = 3 + 3e^{βU/2} + 8e^{βU/4} cosh βb + 2e^{βU/4} cosh βa`.
"""
function aim_partition_function_nb1(m::AIMModel)
    a, b = _nb1_ab(m)
    U, β = m.U, m.β
    return 3 + 3exp(β * U / 2) + 8exp(β * U / 4) * cosh(β * b) + 2exp(β * U / 4) * cosh(β * a)
end

"""    aim_poles_nb1(m) — the eight pole positions of `G_d`: ±(a+b), ±(a-b), ±(U/4+b), ±(U/4-b)."""
function aim_poles_nb1(m::AIMModel)
    a, b = _nb1_ab(m)
    u = m.U / 4
    return sort([s * x for x in (a + b, a - b, u + b, u - b) for s in (1, -1)])
end

"""    singlet_triplet_gap_nb1(m) — `Δ_ST = a - U/4` (≈ 8V²/U for U ≫ V)."""
singlet_triplet_gap_nb1(m::AIMModel) = _nb1_ab(m)[1] - m.U / 4

# ---------------------------------------------------------------------------
# Spectral helpers shared by the benchmarks
# ---------------------------------------------------------------------------

"""
    propagator_poles(m::ImpurityModel; σ = :up, sp, ops, otol, tol = 1e-9) -> Vector{Tuple{Float64,Float64}}

Pole positions and residues of `G_d(z) = Σ_E res_E / (z - E)`, read off the
ℓ = 2 PSFs: a peak of `S[d, d†]` at `ω'` is a pole at `E = ω'` with residue
`w`; a peak of `S[d†, d]` at `ω'` is a pole at `E = -ω'` with residue `-ζw = w`.
Poles closer than `tol` are merged; the residues sum to one.
"""
function propagator_poles(m::ImpurityModel; σ::Symbol = :up, sp::Spectrum = spectrum(m),
                          ops = operators(m), otol::Real = default_otol(m), tol::Real = 1e-9)
    d, ddag = impurity_operators(m, ops, σ)
    poles = Tuple{Float64,Float64}[]
    for P in permuted_psfs(sp, (d, ddag); otol = otol), t in P.terms
        E = P.p[1] == 1 ? t.position[1] : -t.position[1]
        push!(poles, (E, P.p[1] == 1 ? real(t.weight) : -P.ζ * real(t.weight)))
    end
    sort!(poles)
    merged = Tuple{Float64,Float64}[]
    for (E, r) in poles
        if !isempty(merged) && abs(E - merged[end][1]) ≤ tol
            merged[end] = (merged[end][1], merged[end][2] + r)
        else
            push!(merged, (E, r))
        end
    end
    return merged
end

"""
    local_susceptibility(m::ImpurityModel; sp, ops, otol) -> Float64

`χ_loc = ∫₀^β dτ ⟨S^z_d(τ) S^z_d⟩` with `S^z_d = (n_d↑ - n_d↓)/2`: minus the
bosonic ℓ = 2 correlator of Eq. (39) at `ω = 0`, where the anomalous branch of
Eq. (46) supplies the Curie term `β⟨(S^z_d)²⟩` of degenerate states.
"""
function local_susceptibility(m::ImpurityModel; sp::Spectrum = spectrum(m), ops = operators(m),
                              otol::Real = default_otol(m))
    du, duu = impurity_operators(m, ops, :up)
    dd, ddd = impurity_operators(m, ops, :dn)
    Sz = (duu * du - ddd * dd) / 2
    G = correlator(m, (Sz, Sz), [MatsubaraFreq(0), MatsubaraFreq(0)];
                   is_fermionic = [false, false], sp = sp, otol = otol)
    return -real(G)
end

"""
    delta_part(m::ImpurityModel, σσ′, ms; channel = [1, 2], sp, ops, otol) -> ComplexF64

The δ-function contribution of one frequency channel to the Matsubara vertex:
the anomalous term of Eq. (45) ([`anomalous_part`](@ref)) on the connected PSF
of Eq. (31), amputated with the computed propagators like Eq. (74). For the
Hubbard atom and `channel = [1, 2]` it is the `βu² δ_{ω₁₂} th ∏(iωᵢ+u)/∏(iωᵢ)`
term of Eq. (85a); the dimer analogue is [`dimer_delta_part`](@ref).
"""
function delta_part(m::ImpurityModel, σσ′::Symbol, ms::AbstractVector{MatsubaraFreq};
                    channel::AbstractVector{<:Integer} = [1, 2], sp::Spectrum = spectrum(m),
                    ops = operators(m), otol::Real = default_otol(m))
    σ′ = σσ′ === :updn ? :dn : σσ′ === :upup ? :up : throw(ArgumentError("σσ′ must be :updn or :upup"))
    Os = (impurity_operators(m, ops, :up)..., impurity_operators(m, ops, σ′)...)
    a = anomalous_part(m, Os, ms; channel = channel, sp = sp, part = :connected, otol = otol)
    G(ν) = propagator(m, ν; sp = sp, ops = ops, otol = otol)
    return a / (G(ms[1]) * G(MatsubaraFreq(-ms[2].m)) * G(ms[3]) * G(MatsubaraFreq(-ms[4].m)))
end

"""
    singlet_triplet_gap(m::AIMModel; sp, ops) -> Float64

`Δ_ST`: the lowest energy in the `S^z = 1` sector (whose lowest state is the
triplet) minus the energy of the singlet ground state. Equals
[`singlet_triplet_gap_nb1`](@ref) for `N_b = 1`. Not necessarily the lowest
excitation: for `N_b = 1` the odd-charge doublets at `a - b` lie below it.

Only defined for odd `N_b`. At half filling the system holds `N_b + 1`
electrons, so for even `N_b` the ground state is a Kramers doublet — the local
moment is never quenched — and this throws.
"""
function singlet_triplet_gap(m::AIMModel; sp::Spectrum = spectrum(m), ops = operators(m))
    count(e -> e - minimum(sp.E) < 1e-9, sp.E) == 1 || throw(ArgumentError(
        "the ground state is degenerate (Nb = $(nbath(m)): $(nbath(m) + 1) electrons at half filling); " *
        "there is no singlet-triplet gap"))
    Sz = diag(to_eigenbasis(sp, ops.Sz))
    return minimum(sp.E[i] for i in eachindex(Sz) if abs(Sz[i] - 1) < 1e-8) - minimum(sp.E)
end
