"""
    HubbardAtom

Exact many-body spectrum of the half-filled Hubbard atom, i.e. the ``U/Δ → ∞``
limit of the symmetric Anderson impurity model of

    Kugler, Lee & von Delft, *Multipoint Correlation Functions: Spectral
    Representation and Numerical Evaluation*, Phys. Rev. X **11**, 041006 (2021)

[Eq. (79) with the bath removed]. The Hamiltonian is

    H = εd (n↑ + n↓) + U n↑ n↓ ,    εd = -U/2 ,

acting on the four-dimensional Fock space spanned by `|0⟩, |↑⟩, |↓⟩, |↑↓⟩`.

This module is **Step 1 of Block 4**: build the operators with correct
fermionic signs, diagonalise `H`, and form the Boltzmann weights
`ρᵢ = e^(-βEᵢ)/Z`. Steps 2-6 (partial spectral functions via Eq. (28), the
kernel of Eq. (46), and the ℓ = 2 / ℓ = 4 correlators of Eq. (39)) build on the
eigenbasis exposed here via `to_eigenbasis`.

# Basis convention

The Fock basis is indexed by `i = 1 + n↑ + 2 n↓`, so

| index | state  | `(N, Sz)`  |
|:------|:-------|:-----------|
| 1     | `|0⟩`   | `(0, 0)`   |
| 2     | `|↑⟩`   | `(1, +1/2)`|
| 3     | `|↓⟩`   | `(1, -1/2)`|
| 4     | `|↑↓⟩`  | `(2, 0)`   |

with `|↑↓⟩ = d†↑ d†↓ |0⟩`. Operators are built by a Jordan-Wigner transform
with fermionic mode order `(↑, ↓)`, so that the parity string over the `↑`
mode supplies the sign in `d↓|↑↓⟩ = -|↑⟩` rather than it being patched in by
hand.
"""
module HubbardAtom

using LinearAlgebra

export HubbardAtomModel, Operators, Spectrum,
       operators, spectrum, to_eigenbasis, thermal_average, quantum_numbers,
       eigenenergies_analytic, partition_function_analytic,
       level_position, half_interaction,
       PSFTerm, psf, aggregate_psf, psf_frequencies, psf_total_weight, npoint,
       BASIS_LABELS, DIM

"""Dimension of the Hubbard-atom Fock space."""
const DIM = 4

"""Labels of the Fock basis states, in the index order used throughout."""
const BASIS_LABELS = ["|0⟩", "|↑⟩", "|↓⟩", "|↑↓⟩"]

# Single-mode building blocks for the Jordan-Wigner transform.
const _F = [0.0 1.0; 0.0 0.0]   # annihilation:  f|1⟩ = |0⟩
const _Z = [1.0 0.0; 0.0 -1.0]  # parity:        (-1)^n
const _I = [1.0 0.0; 0.0 1.0]

"""
    HubbardAtomModel(U, β)

Half-filled Hubbard atom at interaction `U` and inverse temperature `β`.
The level position is pinned to `εd = -U/2`, which is the particle-hole
symmetric point.
"""
struct HubbardAtomModel
    U::Float64
    β::Float64

    function HubbardAtomModel(U::Real, β::Real)
        β > 0 || throw(ArgumentError("β must be positive, got $β"))
        isfinite(U) || throw(ArgumentError("U must be finite, got $U"))
        return new(float(U), float(β))
    end
end

"""Level position `εd = -U/2` (particle-hole symmetric point)."""
level_position(m::HubbardAtomModel) = -m.U / 2

"""Half interaction `u = U/2`, the combination appearing in Kugler Eq. (85)."""
half_interaction(m::HubbardAtomModel) = m.U / 2

"""
    Operators

The 4x4 second-quantised operators of the Hubbard atom, all in the Fock basis
of `BASIS_LABELS`. Fields: `d_up`, `d_dn`, `dag_up`, `dag_dn` (annihilation
and creation), `n_up`, `n_dn` (occupations), `N` (total charge), `Sz` (spin),
and `H` (the Hamiltonian).
"""
struct Operators
    d_up::Matrix{Float64}
    d_dn::Matrix{Float64}
    dag_up::Matrix{Float64}
    dag_dn::Matrix{Float64}
    n_up::Matrix{Float64}
    n_dn::Matrix{Float64}
    N::Matrix{Float64}
    Sz::Matrix{Float64}
    H::Matrix{Float64}
end

"""
    operators(m::HubbardAtomModel) -> Operators

Build `H`, `dσ`, `d†σ` and `nσ` as 4x4 matrices. The annihilation operators
follow from a Jordan-Wigner transform in mode order `(↑, ↓)`:

    d↑ = I ⊗ f ,    d↓ = f ⊗ z ,

where the Kronecker products are ordered `kron(A_↓, B_↑)`, matching the basis
index `1 + n↑ + 2 n↓`. The parity factor `z` on the `↑` mode is what makes
`d↓|↑↓⟩ = -|↑⟩`; dropping it would silently break every fermionic
anticommutator and hence every permutation sign `ζp` in Kugler Eq. (39).
"""
function operators(m::HubbardAtomModel)
    d_up = kron(_I, _F)   # no Jordan-Wigner string: ↑ is the first mode
    d_dn = kron(_F, _Z)   # parity string over the ↑ mode

    dag_up = permutedims(d_up)
    dag_dn = permutedims(d_dn)

    n_up = dag_up * d_up
    n_dn = dag_dn * d_dn

    N = n_up + n_dn
    Sz = (n_up - n_dn) / 2

    H = level_position(m) * N + m.U * (n_up * n_dn)

    return Operators(d_up, d_dn, dag_up, dag_dn, n_up, n_dn, N, Sz, H)
end

"""
    Spectrum

Result of diagonalising `H`. Fields:

  - `E::Vector{Float64}` — eigenenergies, ascending.
  - `V::Matrix{Float64}` — eigenvectors as *columns*, so `H * V ≈ V * Diagonal(E)`.
  - `ρ::Vector{Float64}` — Boltzmann weights `e^(-βEᵢ)/Z`, summing to 1.
  - `Z::Float64` — partition function `Σᵢ e^(-βEᵢ)`.
"""
struct Spectrum
    E::Vector{Float64}
    V::Matrix{Float64}
    ρ::Vector{Float64}
    Z::Float64
end

"""
    _canonicalise!(E, V, Q; atol) -> (E, V)

Fix the gauge inside degenerate eigenspaces. `eigen` is free to return any
orthonormal basis of a degenerate block, which would make the eigenvectors
non-reproducible and mix charge and spin sectors. Rotating each block to
diagonalise the operator `Q` — chosen below to take a distinct value on every
`(N, Sz)` sector — pins a basis with good quantum numbers instead.

Physically the choice is immaterial for the correlators of Kugler Eq. (39),
which sum over a complete eigenbasis, but a reproducible basis with resolved
quantum numbers makes Steps 2-6 debuggable. The overall sign of each
eigenvector is pinned afterwards for the same reason.
"""
function _canonicalise!(E::Vector{Float64}, V::Matrix{Float64},
                        Q::Matrix{Float64}; atol::Float64=1e-10)
    start = 1
    for k in 2:(length(E) + 1)
        if k == length(E) + 1 || abs(E[k] - E[start]) > atol
            if k - start > 1
                block = V[:, start:(k - 1)]
                M = permutedims(block) * Q * block
                F = eigen(Hermitian((M + permutedims(M)) / 2))
                rotated = block * F.vectors
                V[:, start:(k - 1)] = rotated[:, sortperm(F.values)]
            end
            start = k
        end
    end

    # Fix the remaining freedom, the overall sign of each eigenvector, by
    # demanding that its first significant component be positive. LAPACK makes
    # no such promise, so without this the rotated operators of
    # `to_eigenbasis` can flip sign between platforms or library versions.
    # The signs cancel in Eq. (28), which closes a cycle over the eigenbasis,
    # but pinning them keeps intermediate results comparable run to run.
    for j in axes(V, 2)
        idx = findfirst(x -> abs(x) > 1e-12, view(V, :, j))
        if idx !== nothing && V[idx, j] < 0
            @views V[:, j] .*= -1
        end
    end

    return E, V
end

"""
    spectrum(m::HubbardAtomModel; atol=1e-10) -> Spectrum

Diagonalise `H` and form the Boltzmann weights `ρᵢ = e^(-βEᵢ)/Z`.

The weights are evaluated in the shifted form `e^(-β(Eᵢ-E₀))/Σⱼ e^(-β(Eⱼ-E₀))`
with `E₀ = min E`, which is algebraically identical but does not overflow at
large `βU`; `Z` itself is then reported as `e^(-βE₀) Σⱼ e^(-β(Eⱼ-E₀))`.
"""
function spectrum(m::HubbardAtomModel; atol::Float64=1e-10)
    ops = operators(m)
    H = ops.H

    # Q separates all four (N, Sz) sectors: the irrational coefficient keeps
    # N + √2 Sz from accidentally coinciding on distinct sectors.
    Q = ops.N + sqrt(2.0) * ops.Sz

    F = eigen(Hermitian((H + permutedims(H)) / 2))
    E = collect(F.values)
    V = Matrix(F.vectors)
    _canonicalise!(E, V, Q; atol=atol)

    E0 = minimum(E)
    w = exp.(-m.β .* (E .- E0))
    Zshift = sum(w)
    ρ = w ./ Zshift
    Z = exp(-m.β * E0) * Zshift

    return Spectrum(E, V, ρ, Z)
end

"""
    to_eigenbasis(sp::Spectrum, O::AbstractMatrix) -> Matrix

Rotate an operator from the Fock basis into the energy eigenbasis,
`Õ = V' O V`. Kugler Eq. (28) needs exactly these matrix elements
`(O)_{i,i+1}`, which is why Step 1 exposes the eigenvectors and not just the
eigenvalues.
"""
to_eigenbasis(sp::Spectrum, O::AbstractMatrix) = permutedims(sp.V) * O * sp.V

"""
    thermal_average(sp::Spectrum, O::AbstractMatrix) -> Float64

Thermal expectation value `⟨O⟩ = Σᵢ ρᵢ Õᵢᵢ = tr(ρ O)`.
"""
function thermal_average(sp::Spectrum, O::AbstractMatrix)
    return sum(sp.ρ .* diag(to_eigenbasis(sp, O)))
end

"""
    quantum_numbers(sp::Spectrum, ops::Operators) -> (N, Sz)

Charge and spin expectation value of each eigenstate. After
`spectrum` has canonicalised the degenerate blocks these are sharp
quantum numbers, so the returned vectors should be (near-)integers and
half-integers respectively.
"""
function quantum_numbers(sp::Spectrum, ops::Operators)
    Nrot = to_eigenbasis(sp, ops.N)
    Szrot = to_eigenbasis(sp, ops.Sz)
    return diag(Nrot), diag(Szrot)
end

"""
    eigenenergies_analytic(m::HubbardAtomModel) -> Vector{Float64}

Closed-form spectrum, ascending. With `u = U/2` and `εd = -U/2`,

    E(|0⟩) = 0 ,   E(|↑⟩) = E(|↓⟩) = -u ,   E(|↑↓⟩) = 2εd + U = 0 ,

so the two singly occupied states lie `u` below the degenerate empty and
doubly occupied states.
"""
function eigenenergies_analytic(m::HubbardAtomModel)
    u = half_interaction(m)
    return sort([0.0, -u, -u, 0.0])
end

"""
    partition_function_analytic(m::HubbardAtomModel) -> Float64

Partition function checked by hand:

    Z = Σᵢ e^(-βEᵢ) = 1 + e^(βu) + e^(βu) + 1 = 2(1 + e^(βu)) ,   u = U/2 .

The `tanh(βu/2)` of Kugler Eq. (85) is the corresponding polarisation
`(e^(βu) - 1)/(e^(βu) + 1)` of this spectrum, which is why getting `Z` right
here is a prerequisite for the Step 5 benchmark.
"""
function partition_function_analytic(m::HubbardAtomModel)
    return 2 + 2 * exp(m.β * half_interaction(m))
end

include("psf.jl")   # Step 2: partial spectral functions, Eq. (28)

end # module
