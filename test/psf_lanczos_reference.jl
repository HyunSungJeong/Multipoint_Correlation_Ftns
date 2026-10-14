# Independent references and fixtures for the PSF Lanczos task.
#
# Shared by test/test_psf_lanczos.jl and notebooks/psf_lanczos.ipynb. Nothing in
# this file calls src/psf_lanczos.jl: the references use the operators directly.
#
#   (R1) operator-space block Lanczos with the plain thermal product and the full
#        Liouvillian, in the energy eigenbasis;
#   (R2) the resolvent function F(z₁, z₂) as a Lehmann sum over raw psf terms and,
#        independently, as Tr[ρ O¹ R(z₁){O² R(z₂)(O³)}].
#
# Requires `using .HubbardAtom` and `LinearAlgebra` in the including scope.

# --------------------------------------------------------------------------
# Deterministic random numbers (splitmix64), independent of the Random stdlib
# --------------------------------------------------------------------------

mutable struct SplitMix64
    s::UInt64
end

function _next!(r::SplitMix64)
    r.s += 0x9e3779b97f4a7c15
    z = r.s
    z = (z ⊻ (z >> 30)) * 0xbf58476d1ce4e5b9
    z = (z ⊻ (z >> 27)) * 0x94d049bb133111eb
    return z ⊻ (z >> 31)
end

"""uniform on [-1, 1)"""
runif!(r::SplitMix64) = 2 * ((_next!(r) >> 11) * 2.0^-53) - 1

rand_real(r::SplitMix64, n) = [runif!(r) for _ in 1:n, _ in 1:n]
rand_complex(r::SplitMix64, n) = [complex(runif!(r), runif!(r)) for _ in 1:n, _ in 1:n]

# --------------------------------------------------------------------------
# Fixtures, §5
# --------------------------------------------------------------------------

"""
    spectrum_from_H(H, β) -> Spectrum

`Spectrum` of a real symmetric `H` straight from `eigen`, with the shifted
Boltzmann weights of `_spectrum`. Real eigenvectors keep `to_eigenbasis`
(which transposes) correct for complex operators (pitfall P1).
"""
function spectrum_from_H(H::AbstractMatrix{<:Real}, β::Real)
    F = eigen(Symmetric(Matrix{Float64}(H)))
    E = collect(F.values)
    w = exp.(-β .* (E .- minimum(E)))
    return Spectrum(E, Matrix(F.vectors), w ./ sum(w), exp(-β * minimum(E)) * sum(w))
end

const F1_SEED = 0x5eed_0001
const F1_DIM = 6
const F1_BETA = 0.7

function _f1_hamiltonian(r)
    A = rand_real(r, F1_DIM)
    return (A + A') / 2
end

"""
    fixture(id; U, β, Nb) -> NamedTuple (label, sp, O1s, O2, O3s, otol)

The inputs of §5. `O1s`, `O3s` are row vectors of Fock-basis matrices, `O2` a
single matrix; `otol` is used for all three PSFs (pitfall P3).

  - `:F1`  random real symmetric `H`, `D = 6`, `β = 0.7`; random complex `O¹, O², O³`.
  - `:F1b` same `H`; `M = M′ = 2` random complex operators.
  - `:F2`  Hubbard atom `(U, β)`: `(d↑, n↓, d†↑)`.
  - `:F3`  Hubbard dimer `(t, U, β) = (1, 4, 2)`: `O¹ = c_{α′↑}`, `O² = n_{1↓}`, `O³ = c†_{α↑}`.
  - `:F4`  AIM `aim_model(Nb; U = 1, β = 10)`: `(d↑, n_{d↓}, d†↑)`.
"""
function fixture(id::Symbol; U::Real = 1.0, β::Real = 1.0, Nb::Integer = 1)
    if id === :F1 || id === :F1b
        r = SplitMix64(F1_SEED)
        sp = spectrum_from_H(_f1_hamiltonian(r), F1_BETA)
        if id === :F1
            O1, O2, O3 = rand_complex(r, F1_DIM), rand_complex(r, F1_DIM), rand_complex(r, F1_DIM)
            return (label = "F1", sp = sp, O1s = [O1], O2 = O2, O3s = [O3], otol = 0.0)
        end
        for _ in 1:3; rand_complex(r, F1_DIM); end         # skip F1's operators
        O1s = [rand_complex(r, F1_DIM) for _ in 1:2]
        O2 = rand_complex(r, F1_DIM)
        O3s = [rand_complex(r, F1_DIM) for _ in 1:2]
        return (label = "F1b", sp = sp, O1s = O1s, O2 = O2, O3s = O3s, otol = 0.0)
    elseif id === :F2
        m = HubbardAtomModel(U, β)
        ops = operators(m)
        return (label = "F2 U=$U β=$β", sp = spectrum(m), O1s = [ops.d_up], O2 = ops.n_dn,
                O3s = [ops.dag_up], otol = default_otol(m))
    elseif id === :F3
        m = HubbardDimerModel(1.0, 4.0, 2.0)
        ops = operators(m)
        c = [site_op(ops, i, :up) for i in 1:2]
        cdn = site_op(ops, 1, :dn)
        return (label = "F3", sp = spectrum(m), O1s = c, O2 = adjoint(cdn) * cdn,
                O3s = [Matrix(adjoint(x)) for x in c], otol = 1e-12)
    elseif id === :F4
        m = aim_model(Nb; U = 1.0, β = 10.0)
        ops = operators(m)
        d, dd = impurity_operators(m, ops, :up)
        ddn, dddn = impurity_operators(m, ops, :dn)
        return (label = "F4 Nb=$Nb", sp = spectrum(m), O1s = [d], O2 = dddn * ddn, O3s = [dd],
                otol = default_otol(m))
    end
    throw(ArgumentError("unknown fixture $id"))
end

"""Every fixture of §5 (F2 over the test `PARAMS` without `U = 0`)."""
all_fixtures(params) = vcat([fixture(:F1), fixture(:F1b)],
                            [fixture(:F2; U = U, β = β) for (U, β) in params if U != 0],
                            [fixture(:F3), fixture(:F4; Nb = 1), fixture(:F4; Nb = 2)])

"""Natural energy scale: the largest `|E_a - E_b|`."""
energy_scale(sp::Spectrum) = maximum(sp.E) - minimum(sp.E)

# --------------------------------------------------------------------------
# (R1) Operator-space block Lanczos
# --------------------------------------------------------------------------

# Operators are stored as columns y = vec(√ρ_b X̃_ab), X̃ in the energy eigenbasis,
# so that the plain thermal product becomes the Euclidean one:
#   (A|B)  = Σ_b ρ_b (Ã†B̃)_bb = Σ_ab ρ_b conj(Ã_ab) B̃_ab = y_A† y_B ,
#   (𝓛X̃)_ab = (E_a - E_b) X̃_ab                    (elementwise in either form).
# Rotated matrix elements |X̃_ab| ≤ otol are zeroed exactly as `psf` does (P3), so
# that the reference sees the same operators as the PSFs.
struct OperatorSpace
    D::Int
    L::Vector{Float64}       # vec of E_a - E_b
    sqrtρ::Vector{Float64}   # vec of √ρ_b
    sp::Spectrum
    otol::Float64
end

function OperatorSpace(sp::Spectrum; otol::Real = 0.0)
    D = length(sp.E)
    return OperatorSpace(D, vec([sp.E[a] - sp.E[b] for a in 1:D, b in 1:D]),
                         vec([sqrt(sp.ρ[b]) for a in 1:D, b in 1:D]), sp, float(otol))
end

"""rotated, otol-filtered matrix elements X̃ of a Fock-basis operator"""
function rotated(os::OperatorSpace, X::AbstractMatrix)
    R = ComplexF64.(to_eigenbasis(os.sp, X))
    os.otol > 0 && (R[abs.(R) .<= os.otol] .= 0)
    return R
end

opvec(os::OperatorSpace, X::AbstractMatrix) = os.sqrtρ .* vec(rotated(os, X))
ip(os::OperatorSpace, X::AbstractVecOrMat, Y::AbstractVecOrMat) = X' * Y

"""Gram matrix `[(A_i|B_j)]` of operator vectors, from the definition."""
ref_gram(sp::Spectrum, As::AbstractVector, Bs::AbstractVector) =
    [thermal_average(sp, adjoint(A) * B) for A in As, B in Bs]

"""
    ref_lanczos(sp, Os; tol = 1e-12) -> NamedTuple

(R1): block Lanczos for `𝓛` from `|𝐪₁) = |𝐎)𝐍⁻¹`, `𝐍` the principal root of the
Gram matrix `(𝐎|𝐎)`, with the plain product, full reorthogonalisation (twice)
and principal-root `𝐭_n`. A block narrows to the singular values of its new
residual above `tol·max|E_ab|` (the Gram matrix `𝐭²` below `tol²` relative); the
chain stops when none is left. Returns `N`, `eps`, `t` (`t[1]` unused), `Q` (the
Lanczos operators as columns `vec(√ρ_b q̃_ab)`, eigenbasis), `widths`. `otol`
zeroes rotated matrix elements as in `psf` (pitfall P3).
"""
function ref_lanczos(sp::Spectrum, Os::AbstractVector; tol::Real = 1e-12, otol::Real = 0.0,
                     os = OperatorSpace(sp; otol = otol))
    M = length(Os)
    X = reduce(hcat, [opvec(os, O) for O in Os])
    G = Hermitian(ip(os, X, X))
    Fg = eigen(G)
    N = Fg.vectors * Diagonal(sqrt.(Fg.values)) * Fg.vectors'
    Qn = X * (Fg.vectors * Diagonal(1 ./ sqrt.(Fg.values)) * Fg.vectors')
    scale = energy_scale(sp)
    Qbuf = zeros(ComplexF64, length(os.L), 4M)     # Lanczos operators, grown by doubling
    Qbuf[:, 1:M] = Qn
    cols = M
    eps = Matrix{ComplexF64}[]
    t = Matrix{ComplexF64}[zeros(ComplexF64, M, M)]
    widths = [M]
    Qprev = zeros(ComplexF64, length(os.L), 0)
    while true
        W = os.L .* Qn
        ϵ = ip(os, Qn, W)
        push!(eps, (ϵ + ϵ') / 2)
        P = W - Qn * eps[end]
        length(eps) > 1 && (P -= Qprev * t[end]')
        for _ in 1:2
            Q = view(Qbuf, :, 1:cols)
            P -= Q * ip(os, Q, P)
        end
        F = eigen(Hermitian(ip(os, P, P)))
        λ = max.(F.values, 0.0)
        keep = sortperm(λ; rev = true)
        keep = keep[sqrt.(λ[keep]) .> tol * scale]
        r, w = length(keep), size(Qn, 2)
        r == 0 && break
        Xk, s = F.vectors[:, keep], sqrt.(λ[keep])
        if r == w
            T = Xk * Diagonal(s) * Xk'
            Qnext = P * (Xk * Diagonal(1 ./ s) * Xk')
        else
            T = Diagonal(s) * Xk'
            Qnext = P * Xk * Diagonal(1 ./ s)
        end
        push!(t, Matrix(T))
        push!(widths, r)
        cols + r > size(Qbuf, 2) && (Qbuf = hcat(Qbuf, zeros(ComplexF64, size(Qbuf, 1), size(Qbuf, 2))))
        Qbuf[:, (cols + 1):(cols + r)] = Qnext
        cols += r
        Qprev, Qn = Qn, Qnext
    end
    return (N = Matrix(N), eps = eps, t = t, Q = Qbuf[:, 1:cols], widths = widths)
end

"""
    ref_coupling(os, Qbra, O2, Qket) -> Matrix

`V_mn = (q′_m | M_{O²} | q_n) = ⟨q′_m† O² q_n⟩` from definition (6), for every
pair of Lanczos operators (columns `vec(√ρ_b q̃_ab)`).
"""
function ref_coupling(os::OperatorSpace, Qbra::AbstractMatrix, O2::AbstractMatrix, Qket::AbstractMatrix)
    D = os.D
    O2t = rotated(os, O2)
    # columns are √ρ_b q̃_ab; left multiplication by Õ² acts on the row index a
    O2Q = reshape(O2t * reshape(Qket, D, :), D * D, :)
    return ip(os, Qbra, O2Q)
end

"""
    ref_krylov(fx; tol = 1e-12) -> NamedTuple

(R1) for a fixture: ket chain from `O3s`, bra chain from `O1s†`, and
`Vref` from definition (6). Fields `N1`, `N3`, `eps1`, `t1`, `eps3`, `t3`, `V`,
`widths1`, `widths3`, `Q1`, `Q3`, `os`.
"""
function ref_krylov(fx; tol::Real = 1e-12)
    os = OperatorSpace(fx.sp; otol = fx.otol)
    ket = ref_lanczos(fx.sp, fx.O3s; tol = tol, os = os)
    bra = ref_lanczos(fx.sp, [Matrix(adjoint(O)) for O in fx.O1s]; tol = tol, os = os)
    V = ref_coupling(os, bra.Q, fx.O2, ket.Q)
    return (N1 = bra.N, N3 = ket.N, eps1 = bra.eps, t1 = bra.t, eps3 = ket.eps, t3 = ket.t, V = V,
            widths1 = bra.widths, widths3 = ket.widths, Q1 = bra.Q, Q3 = ket.Q, os = os)
end

"""Full block-tridiagonal matrix of a chain (independent of `jacobi_matrix`)."""
function ref_jacobi(eps, t)
    w = [size(e, 1) for e in eps]
    off = cumsum([0; w])
    J = zeros(ComplexF64, off[end], off[end])
    for n in eachindex(eps)
        J[(off[n] + 1):off[n + 1], (off[n] + 1):off[n + 1]] = eps[n]
        if n < length(eps)
            J[(off[n + 1] + 1):off[n + 2], (off[n] + 1):off[n + 1]] = t[n + 1]
            J[(off[n] + 1):off[n + 1], (off[n + 1] + 1):off[n + 2]] = t[n + 1]'
        end
    end
    return J
end

# --------------------------------------------------------------------------
# (R2) Lehmann resolvent
# --------------------------------------------------------------------------

"""
    ref_resolvent_lehmann(fx, z1, z2) -> Matrix

`[𝐅]_{α′α} = Σ_terms w / ((z₁ - p₁)(z₂ - p₂))` over the raw `psf(sp, O1s[α′], O2, O3s[α])`.
"""
function ref_resolvent_lehmann(fx, z1::Number, z2::Number; terms = ref_raw_psf(fx))
    return [sum(t -> t.weight / ((z1 - t.position[1]) * (z2 - t.position[2])), T; init = zero(ComplexF64))
            for T in terms]
end

"""Raw 3p term lists `psf(sp, O1s[α′], O2, O3s[α]; otol)`, `M′×M`."""
ref_raw_psf(fx) = [psf(fx.sp, O1, fx.O2, O3; otol = fx.otol) for O1 in fx.O1s, O3 in fx.O3s]

"""
    ref_resolvent_trace(fx, z1, z2) -> Matrix

`Tr[ρ O¹_{α′} 𝓡(z₁){O² 𝓡(z₂)(O³_α)}]` with `[𝓡(z)X]_ab = X_ab/(z - E_ab)` in the
eigenbasis: Eq. (4) evaluated with operators, no PSF involved.
"""
function ref_resolvent_trace(fx, z1::Number, z2::Number)
    sp = fx.sp
    D = length(sp.E)
    Eab = [sp.E[a] - sp.E[b] for a in 1:D, b in 1:D]
    R(z, X) = X ./ (z .- Eab)
    rot(X) = ComplexF64.(to_eigenbasis(sp, X))
    O2 = rot(fx.O2)
    return [sum(sp.ρ .* diag(rot(O1) * R(z1, O2 * R(z2, rot(O3))))) for O1 in fx.O1s, O3 in fx.O3s]
end
