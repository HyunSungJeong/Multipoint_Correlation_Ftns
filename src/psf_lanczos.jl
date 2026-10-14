# PSF Lanczos: Liouvillian matrix elements of 3p PSFs from PSF data alone.
#
# The 3p PSF of Kugler Eq. (28) (identity permutation, cumulative frequencies
# ε₁ = E₂₁, ε₂ = E₃₁) is the nested Liouvillian form
#
#     S(ε₁, ε₂) = (O¹† | δ(ε₁ - L) M_{O²} δ(ε₂ - L) | O³) ,
#
# with L = [H, ·], M_A X = A X and the plain thermal product (A|B) = ⟨A†B⟩. Two
# block Lanczos chains, from |O³) (ket) and |O¹†) (bra), give block-tridiagonal
# matrices 𝕁, 𝕁′ of Liouvillian matrix elements and a coupling matrix
# V_mn = (q′_m | M_{O²} | q_n), and the resolvent function follows as
#
#     F(z₁, z₂) = N₁ Σ_mn [(z₁ - 𝕁′)⁻¹]_{1m} V_mn [(z₂ - 𝕁)⁻¹]_{n1} N₃ .
#
# Nothing here touches an operator: 𝕁 and 𝕁′ are the recurrence coefficients of
# the orthonormal (matrix) polynomials of the measures W₃ ∝ S[O³†, O³] and
# W₁ ∝ S[O¹, O¹†], and V is the projection of S onto those polynomials. For
# discrete PSFs everything is a finite sum, done in the pole basis (Golub-Welsch;
# Kovalska et al., Suppl. Sec. S-2; Pelz et al., App. E 2, Algorithm 1, here with
# the plain product and the full L). The tests compare with an operator-space
# Lanczos that uses the operators directly (test/psf_lanczos_reference.jl).

# --------------------------------------------------------------------------
# PSF data: discrete matrix measures and the 3p weights on their supports
# --------------------------------------------------------------------------

_postol_digits(postol::Real) = max(0, round(Int, -log10(postol)))

# the rounding of `aggregate_psf`; `+ 0.0` maps -0.0 to 0.0, which `Dict`
# would otherwise treat as a different key
_pkey(x::Real, digits::Integer) = round(x; digits = digits) + 0.0

"""
    PSFKrylovInput

The PSF data the algorithm consumes (§2 step 1 of the task sheet), and nothing
else:

  - `points1`, `weights1` — distinct support points `a_α` of
    `[𝐀₁]_{α′β′} = S[O¹_{α′}, O¹†_{β′}]` and their `M′×M′` weights `𝐚′_α`;
  - `points3`, `weights3` — the same for `[𝐀₃]_{αβ} = S[O³†_α, O³_β]`, `M×M`;
  - `s` — the aggregated 3p weights: one `(α, β, 𝐬_{αβ})` per support point
    `(a_α, b_β)` of `[𝐒]_{α′α} = S[O¹_{α′}, O², O³_α]`, `𝐬` of size `M′×M`;
  - `S` — the aggregated 3p PSFs themselves, `M′×M` term lists;
  - `digits` — the rounding (from `postol`) used to identify support points.

A support point's position is the mean of the raw positions that round to its
key, so the rounding only identifies points and never moves them.
"""
struct PSFKrylovInput
    points1::Vector{Float64}
    weights1::Vector{Matrix{ComplexF64}}
    points3::Vector{Float64}
    weights3::Vector{Matrix{ComplexF64}}
    s::Vector{Tuple{Int,Int,Matrix{ComplexF64}}}
    S::Matrix{Vector{PSFTerm}}
    digits::Int
end

# Collect the distinct support points of a matrix of 2p term lists (entry
# (α, β) is the PSF of the pair) and their matrix weights.
function _matrix_measure(T::Matrix{Vector{PSFTerm}}, digits::Integer)
    M = size(T, 1)
    index = Dict{Float64,Int}()
    possum = Float64[]
    count = Int[]
    weights = Matrix{ComplexF64}[]
    for β in 1:M, α in 1:M, t in T[α, β]
        k = _pkey(t.position[1], digits)
        i = get(index, k, 0)
        if i == 0
            push!(possum, 0.0); push!(count, 0); push!(weights, zeros(ComplexF64, M, M))
            i = index[k] = length(weights)
        end
        possum[i] += t.position[1]
        count[i] += 1
        weights[i][α, β] += t.weight
    end
    points = possum ./ count
    order = sortperm(points)
    rank = invperm(order)
    return points[order], weights[order], Dict(k => rank[i] for (k, i) in index)
end

# the support point a 3p coordinate belongs to: by rounded key, falling back to
# the nearest point within the rounding resolution
function _match_point(x::Real, index::Dict{Float64,Int}, points::Vector{Float64}, digits::Integer)
    i = get(index, _pkey(x, digits), 0)
    i != 0 && return i
    j = argmin(abs.(points .- x))
    abs(points[j] - x) <= 10.0^(-digits) && return j
    throw(ArgumentError("3p coordinate $x lies outside the support of the 2p measure; " *
                        "the factorization lemma (kernels of B_β) fails for these inputs"))
end

"""
    psf_krylov_input(sp, O1s, O2, O3s; postol = 1e-10, otol = 0.0) -> PSFKrylovInput

Build the PSF data of §2 step 1 with [`psf`](@ref) and [`aggregate_psf`](@ref):

    [𝐒]_{α′α}  = aggregate_psf(psf(sp, O1s[α′], O2, O3s[α]))
    [𝐀₃]_{αβ}  = psf(sp, O3s[α]', O3s[β])
    [𝐀₁]_{α′β′} = psf(sp, O1s[α′], O1s[β′]')

Operators are row vectors `O1s` (length `M′`) and `O3s` (length `M`) of Fock-basis
matrices; adjoints are formed with `adjoint`. Support points are identified by
the rounded keys `aggregate_psf` uses (`postol`), and the `(ε₁, ε₂)` coordinates
of `𝐒` are matched to the points of `𝐀₁` and `𝐀₃` the same way.
"""
function psf_krylov_input(sp::Spectrum, O1s::AbstractVector, O2::AbstractMatrix, O3s::AbstractVector;
                          postol::Real = 1e-10, otol::Real = 0.0)
    digits = _postol_digits(postol)
    M1, M3 = length(O1s), length(O3s)
    A3 = [psf(sp, adjoint(O3s[α]), O3s[β]; otol = otol) for α in 1:M3, β in 1:M3]
    A1 = [psf(sp, O1s[α], adjoint(O1s[β]); otol = otol) for α in 1:M1, β in 1:M1]
    points3, weights3, index3 = _matrix_measure(A3, digits)
    points1, weights1, index1 = _matrix_measure(A1, digits)

    S = [aggregate_psf(psf(sp, O1s[α′], O2, O3s[α]; otol = otol); postol = postol) for α′ in 1:M1, α in 1:M3]
    acc = Dict{Tuple{Int,Int},Matrix{ComplexF64}}()
    for α in 1:M3, α′ in 1:M1, t in S[α′, α]
        a = _match_point(t.position[1], index1, points1, digits)
        b = _match_point(t.position[2], index3, points3, digits)
        w = get!(() -> zeros(ComplexF64, M1, M3), acc, (a, b))
        w[α′, α] += t.weight
    end
    s = sort!([(k[1], k[2], w) for (k, w) in acc]; by = x -> (x[1], x[2]))
    return PSFKrylovInput(points1, weights1, points3, weights3, s, S, digits)
end

# --------------------------------------------------------------------------
# Block Jacobi matrices from a discrete matrix measure (pole basis)
# --------------------------------------------------------------------------

# principal square root and its inverse of a Hermitian positive-definite matrix
function _psd_sqrt(A::AbstractMatrix)
    F = eigen(Hermitian((A + A') / 2))
    any(<=(0), F.values) && throw(ArgumentError("weight matrix is not positive definite: " *
                                                "the operators of a block are linearly dependent"))
    r = sqrt.(F.values)
    return Matrix(F.vectors * Diagonal(r) * F.vectors'), Matrix(F.vectors * Diagonal(1 ./ r) * F.vectors')
end

"""
    jacobi_from_measure(points, weights; ranktol = 1e-12, defltol = 1e-12) -> NamedTuple

§2 steps 2–4: the block Jacobi matrix of a discrete matrix measure
`Σ_β 𝐚_β δ(ω - b_β)` (`points` = `b_β`, `weights` = `M×M` matrices `𝐚_β`).

 2. `𝐍 = (Σ_β 𝐚_β)^{1/2}` (principal root), `𝐰_β = 𝐍⁻¹𝐚_β𝐍⁻¹`.
 3. `𝐰_β = 𝐁_β†𝐁_β` from an eigendecomposition, `𝐁_β` of size `r_β×M`; eigenvalues
    below `ranktol` times the largest eigenvalue *of the same* `𝐰_β` are rank
    deficiency, not weight, and are dropped (a small but genuine scalar weight
    is never dropped). `𝐁 = (𝐁₁; 𝐁₂; …)` has `R = Σ r_β` rows and `𝐁†𝐁 = 𝟏`.
 4. Block Lanczos on `Ω = diag(b_β 𝟏_{r_β})` from `𝐔₁ = 𝐁`, with full
    reorthogonalisation (twice) and principal-root `𝐭_{n+1} = (𝐏†𝐏)^{1/2}`,
    run until exactly `R` columns exist. There is no small-`t` stop: a block
    narrows only when `𝐏` is rank deficient (singular values below
    `defltol·max|b_β|`, block measures only) or when fewer than its width
    columns remain. A narrowed block (deflation) has a rectangular
    `𝐭_{n+1}` of size `r_{n+1}×r_n` and no principal-root gauge.

The recursion is `Ω𝐔_n = 𝐔_n𝛜_n + 𝐔_{n+1}𝐭_{n+1} + 𝐔_{n-1}𝐭_n†`, so `𝕁` has
`𝛜_n` on the diagonal, `𝐭_{n+1}` below it and `𝐭_{n+1}†` above it. The columns of
`𝐔` are the blocks `𝐔_n = (𝐁_β 𝐩_n(b_β))_β` of the orthonormal polynomials
evaluated on the support; `𝐔` is unitary.

Returns `(N, Ninv, eps, t, U, B, rows, widths)`: `t[1]` is an unused `M×M`
zero placeholder, `rows[β]` the rows of `𝐔` belonging to `b_β`, `widths[n]`
the width of block `n`.
"""
function jacobi_from_measure(points::AbstractVector{<:Real}, weights::AbstractVector{<:AbstractMatrix};
                             ranktol::Real = 1e-12, defltol::Real = 1e-12)
    length(points) == length(weights) || throw(DimensionMismatch("one weight matrix per point"))
    M = size(first(weights), 1)
    N, Ninv = _psd_sqrt(sum(weights))

    B = Matrix{ComplexF64}[]
    rows = UnitRange{Int}[]
    R = 0
    for a in weights
        F = eigen(Hermitian(Ninv * a * Ninv))
        λmax = maximum(F.values)
        keep = λmax > 0 ? findall(>(ranktol * λmax), F.values) : Int[]
        push!(B, Matrix(Diagonal(sqrt.(F.values[keep])) * F.vectors[:, keep]'))
        push!(rows, (R + 1):(R + length(keep)))
        R += length(keep)
    end
    Ωd = reduce(vcat, [fill(float(points[β]), length(rows[β])) for β in eachindex(points)])
    eps, t, U, widths = _pole_lanczos(Ωd, reduce(vcat, B), defltol)
    return (N = N, Ninv = Ninv, eps = eps, t = t, U = U, B = B, rows = rows, widths = widths)
end

# §2 step 4: block Lanczos on a diagonal matrix, until all R columns exist
function _pole_lanczos(Ωd::Vector{Float64}, U1::Matrix{ComplexF64}, defltol::Real)
    R, M = size(U1)
    scale = max(maximum(abs, Ωd; init = 0.0), floatmin())
    U = zeros(ComplexF64, R, R)
    U[:, 1:M] = U1
    eps = Matrix{ComplexF64}[]
    t = Matrix{ComplexF64}[zeros(ComplexF64, M, M)]
    widths = [M]
    cols = M
    Un, Uprev = U1, zeros(ComplexF64, R, 0)
    while true
        W = Ωd .* Un
        ϵ = Un' * W
        push!(eps, (ϵ + ϵ') / 2)
        cols >= R && break
        P = W - Un * eps[end]
        length(eps) > 1 && (P -= Uprev * t[end]')
        for _ in 1:2                                    # full reorthogonalisation, twice
            Q = view(U, :, 1:cols)
            P -= Q * (Q' * P)
        end
        F = eigen(Hermitian(P' * P))
        w = size(Un, 2)
        λ = max.(F.values, 0.0)
        order = sortperm(λ; rev = true)
        r = min(w, R - cols)
        if w > 1                                        # deflation: block measures only
            r = min(r, count(>(defltol * scale), sqrt.(λ)))
        end
        if r == 0
            @warn "block Lanczos lost rank after $cols of $R columns"
            break
        end
        keep = order[1:r]
        X, s = F.vectors[:, keep], sqrt.(λ[keep])
        if r == w                                       # principal root: 𝐭 = (𝐏†𝐏)^{1/2}
            T = X * Diagonal(s) * X'
            Unext = P * (X * Diagonal(1 ./ s) * X')
        else                                            # narrowed block, rectangular 𝐭
            T = Diagonal(s) * X'
            Unext = P * X * Diagonal(1 ./ s)
        end
        push!(t, Matrix(T))
        U[:, (cols + 1):(cols + r)] = Unext
        cols += r
        push!(widths, r)
        Uprev, Un = Un, Unext
    end
    return eps, t, U[:, 1:cols], widths
end

"""
    jacobi_matrix(eps, t; N = length(eps)) -> Matrix{ComplexF64}

The block-tridiagonal `𝕁` of the first `N` blocks: `𝛜_n` on the diagonal,
`𝐭_{n+1}` below and `𝐭_{n+1}†` above it (rectangular for narrowed blocks).
"""
function jacobi_matrix(eps::AbstractVector, t::AbstractVector; N::Integer = length(eps))
    w = [size(e, 1) for e in eps[1:N]]
    off = cumsum([0; w])
    J = zeros(ComplexF64, off[end], off[end])
    for n in 1:N
        J[(off[n] + 1):off[n + 1], (off[n] + 1):off[n + 1]] = eps[n]
        if n < N
            J[(off[n + 1] + 1):off[n + 2], (off[n] + 1):off[n + 1]] = t[n + 1]
            J[(off[n] + 1):off[n + 1], (off[n + 1] + 1):off[n + 2]] = t[n + 1]'
        end
    end
    return J
end

# --------------------------------------------------------------------------
# The full pipeline
# --------------------------------------------------------------------------

"""
    PSFKrylov

Output of [`psf_krylov`](@ref). The fields named on the task sheet:

  - `N1`, `N3` — `𝐍₁ = (∫𝐀₁)^{1/2}`, `𝐍₃ = (∫𝐀₃)^{1/2}` (principal roots);
  - `eps1`, `t1` — bra chain `𝛜′_m`, `𝐭′_m` (`t1[1]` unused); `eps3`, `t3` — ket chain;
  - `V` — the `R′×R` coupling matrix, blocks `𝐕_mn` of size `width1[m]×width3[n]`;
  - `M1`, `M3` — block sizes `M′`, `M`;

and the pole-basis data they came from: support points `points1`, `points3`,
factors `B1`, `B3` (`𝐰_β = 𝐁_β†𝐁_β`), their row ranges `rows1`, `rows3` in the
unitaries `U1` (`𝐔′`) and `U3` (`𝐔`), block widths `widths1`, `widths3`, the
rescaled kernel `kernel` (`𝐊`, `R′×R`), and the raw 3p weights `s` of the input.
"""
struct PSFKrylov
    N1::Matrix{ComplexF64}
    N3::Matrix{ComplexF64}
    eps1::Vector{Matrix{ComplexF64}}
    t1::Vector{Matrix{ComplexF64}}
    eps3::Vector{Matrix{ComplexF64}}
    t3::Vector{Matrix{ComplexF64}}
    V::Matrix{ComplexF64}
    M1::Int
    M3::Int
    points1::Vector{Float64}
    points3::Vector{Float64}
    B1::Vector{Matrix{ComplexF64}}
    B3::Vector{Matrix{ComplexF64}}
    rows1::Vector{UnitRange{Int}}
    rows3::Vector{UnitRange{Int}}
    U1::Matrix{ComplexF64}
    U3::Matrix{ComplexF64}
    widths1::Vector{Int}
    widths3::Vector{Int}
    kernel::Matrix{ComplexF64}
    s::Vector{Tuple{Int,Int,Matrix{ComplexF64}}}
end

"""
    psf_krylov(sp, O1s, O2, O3s; postol = 1e-10, otol = 0.0, ranktol = 1e-12) -> PSFKrylov
    psf_krylov(input::PSFKrylovInput; ranktol = 1e-12, defltol = 1e-12) -> PSFKrylov

§2 of the task sheet, from PSF data only. Steps 1 ([`psf_krylov_input`](@ref)),
2–5 ([`jacobi_from_measure`](@ref) for the ket measure `𝐀₃` and the bra measure
`𝐀₁`), then

 6. the rescaled kernel `𝐊_{αβ} = (𝐁′_α†)⁺ 𝐬̃_{αβ} 𝐁_β⁺`, `𝐬̃ = 𝐍₁⁻¹𝐬𝐍₃⁻¹`
    (scalar: `s̃/√(w′_α w_β)`), assembled into an `R′×R` matrix;
 7. `𝐕 = 𝐔′†𝐊𝐔`, i.e. the projection formula
    `𝐕_mn = ∬ 𝐩′_m(ε₁)† 𝐍₁⁻¹𝐒(ε₁,ε₂)𝐍₃⁻¹ 𝐩_n(ε₂)`.

Step 6 is exact for exact Lehmann data at `T > 0`: a 3p peak needs the
matrix elements that put weight into `𝐖₁`, `𝐖₃` at the same energies, so
`ker 𝐁_β ⊆ ker 𝐬̃_{αβ}` and `𝐬̃_{αβ} = 𝐁′_α†𝐊_{αβ}𝐁_β`.
"""
function psf_krylov(inp::PSFKrylovInput; ranktol::Real = 1e-12, defltol::Real = 1e-12)
    bra = jacobi_from_measure(inp.points1, inp.weights1; ranktol = ranktol, defltol = defltol)
    ket = jacobi_from_measure(inp.points3, inp.weights3; ranktol = ranktol, defltol = defltol)
    R1, R3 = size(bra.U, 1), size(ket.U, 1)
    Bp1 = [pinv(Matrix(B')) for B in bra.B]        # (𝐁′_α†)⁺, r′_α × M′
    Bp3 = [pinv(B) for B in ket.B]                 # 𝐁_β⁺,     M × r_β
    Kmat = zeros(ComplexF64, R1, R3)
    for (a, b, w) in inp.s
        Kmat[bra.rows[a], ket.rows[b]] = Bp1[a] * (bra.Ninv * w * ket.Ninv) * Bp3[b]
    end
    V = bra.U' * Kmat * ket.U
    M1, M3 = size(bra.N, 1), size(ket.N, 1)
    return PSFKrylov(bra.N, ket.N, bra.eps, bra.t, ket.eps, ket.t, V, M1, M3,
                     inp.points1, inp.points3, bra.B, ket.B, bra.rows, ket.rows, bra.U, ket.U,
                     bra.widths, ket.widths, Kmat, inp.s)
end

function psf_krylov(sp::Spectrum, O1s::AbstractVector, O2::AbstractMatrix, O3s::AbstractVector;
                    postol::Real = 1e-10, otol::Real = 0.0, ranktol::Real = 1e-12, defltol::Real = 1e-12)
    inp = psf_krylov_input(sp, O1s, O2, O3s; postol = postol, otol = otol)
    return psf_krylov(inp; ranktol = ranktol, defltol = defltol)
end

# --------------------------------------------------------------------------
# Resolvent, Eqs. (7), (8), (11)
# --------------------------------------------------------------------------

"""
    resolvent_column(eps, t, z; N = length(eps)) -> Vector{Matrix{ComplexF64}}

The first block column `[(z - 𝕁_N)⁻¹]_{n1}`, `n = 1, …, N`, of the chain truncated
after `N` blocks, as products of tail continued fractions, Eq. (11):

    𝐠_{N+1} = 0,   𝐠_n = [z - 𝛜_n - 𝐭_{n+1}†𝐠_{n+1}𝐭_{n+1}]⁻¹,
    [(z - 𝕁)⁻¹]_{n1} = 𝐠_n 𝐭_n ⋯ 𝐠₂ 𝐭₂ 𝐠₁ .

(`𝐭†` reduces to `𝐭` for the Hermitian principal-root blocks of Eq. (11).) The
first block row of a Hermitian `𝕁′` follows as `[(z - 𝕁′)⁻¹]_{1m} = ([(z̄ - 𝕁′)⁻¹]_{m1})†`.
"""
function resolvent_column(eps::AbstractVector, t::AbstractVector, z::Number; N::Integer = length(eps))
    g = Vector{Matrix{ComplexF64}}(undef, N)
    for n in N:-1:1
        A = z * I - eps[n]
        n < N && (A -= t[n + 1]' * g[n + 1] * t[n + 1])
        g[n] = inv(A)
    end
    x = Vector{Matrix{ComplexF64}}(undef, N)
    x[1] = g[1]
    for n in 2:N
        x[n] = g[n] * (t[n] * x[n - 1])
    end
    return x
end

_nblocks(N, chain) = N === :all ? length(chain) : Int(N)

"""
    resolvent_krylov(K, z1, z2; Nbra = :all, Nket = :all) -> Matrix{ComplexF64}

The resolvent function of Eq. (4), `M′×M`, from the Krylov data, Eq. (7) with
Eq. (11):

    𝐅(z₁, z₂) = 𝐍₁ Σ_mn [(z₁ - 𝕁′)⁻¹]_{1m} 𝐕_mn [(z₂ - 𝕁)⁻¹]_{n1} 𝐍₃ ,

exact for the full chains. `Nbra`, `Nket` truncate the chains after that many
blocks (the continued fractions are cut with `𝐠_{N+1} = 0`).
"""
function resolvent_krylov(K::PSFKrylov, z1::Number, z2::Number; Nbra = :all, Nket = :all)
    Nb, Nk = _nblocks(Nbra, K.eps1), _nblocks(Nket, K.eps3)
    row = reduce(hcat, [x' for x in resolvent_column(K.eps1, K.t1, conj(z1); N = Nb)])   # M′ × R′_N
    col = reduce(vcat, resolvent_column(K.eps3, K.t3, z2; N = Nk))                         # R_N × M
    return K.N1 * row * K.V[1:size(row, 2), 1:size(col, 1)] * col * K.N3
end

# --------------------------------------------------------------------------
# Reconstruction, Eq. (12)
# --------------------------------------------------------------------------

"""
    reconstruct_psf(K::PSFKrylov; V = K.V, wtol = 0.0) -> Matrix{Vector{PSFTerm}}

Eq. (12) in the pole basis,

    𝐒(a_α, b_β) = 𝐍₁ 𝐁′_α† [𝐔′ 𝐕 𝐔†]_{αβ} 𝐁_β 𝐍₃ ,

which is `𝐍₁𝐖₁(ε₁)[Σ_mn 𝐩′_m(ε₁)𝐕_mn𝐩_n(ε₂)†]𝐖₃(ε₂)𝐍₃` on the support. Returns one
term list per `(α′, α)`, with a term at every `(a_α, b_β)` whose weight exceeds
`wtol` in modulus. With the algorithm's own `V` this is a tautology (`𝐔` is
unitary); pass a `V` from elsewhere (an operator-space Lanczos in the same gauge)
to make it a test.
"""
function reconstruct_psf(K::PSFKrylov; V::AbstractMatrix = K.V, wtol::Real = 0.0)
    C = K.U1 * V * K.U3'
    out = [PSFTerm[] for _ in 1:K.M1, _ in 1:K.M3]
    for b in eachindex(K.points3), a in eachindex(K.points1)
        (isempty(K.rows1[a]) || isempty(K.rows3[b])) && continue
        w = K.N1 * K.B1[a]' * C[K.rows1[a], K.rows3[b]] * K.B3[b] * K.N3
        for α in 1:K.M3, α′ in 1:K.M1
            abs(w[α′, α]) > wtol && push!(out[α′, α], PSFTerm([K.points1[a], K.points3[b]], w[α′, α], Int[]))
        end
    end
    foreach(x -> sort!(x; by = t -> t.position), out)
    return out
end

"""
    reconstruct_psf(J1, J3, V, N1, N3; postol = 1e-10, wtol = 0.0) -> Matrix{Vector{PSFTerm}}

Eq. (12) from the chains alone, in any gauge: diagonalise `𝕁′ = X′Λ′X′†` and
`𝕁 = XΛX†` (Golub-Welsch). Eigenvalues are the support points, grouped by the
rounded keys of `postol`; with `X_{1,β}` the first-block rows of the eigenvectors of
group `β`,

    𝐒(a_α, b_β) = 𝐍₁ X′_{1,α} (X′_α† 𝐕 X_β) X_{1,β}† 𝐍₃ ,

invariant under rotations inside a degenerate group. `J1`, `J3` are full
block-tridiagonal matrices ([`jacobi_matrix`](@ref)), `V` must be in their gauge.
"""
function reconstruct_psf(J1::AbstractMatrix, J3::AbstractMatrix, V::AbstractMatrix,
                         N1::AbstractMatrix, N3::AbstractMatrix; postol::Real = 1e-10, wtol::Real = 0.0)
    digits = _postol_digits(postol)
    M1, M3 = size(N1, 1), size(N3, 1)
    groups(F) = begin
        d = Dict{Float64,Vector{Int}}()
        for (i, λ) in enumerate(F.values)
            push!(get!(d, _pkey(λ, digits), Int[]), i)
        end
        collect(values(d))
    end
    F1, F3 = eigen(Hermitian((J1 + J1') / 2)), eigen(Hermitian((J3 + J3') / 2))
    G1, G3 = groups(F1), groups(F3)
    Y = F1.vectors' * V * F3.vectors
    out = [PSFTerm[] for _ in 1:M1, _ in 1:M3]
    for g3 in G3, g1 in G1
        w = N1 * F1.vectors[1:M1, g1] * Y[g1, g3] * F3.vectors[1:M3, g3]' * N3
        pos = [sum(F1.values[g1]) / length(g1), sum(F3.values[g3]) / length(g3)]
        for α in 1:M3, α′ in 1:M1
            abs(w[α′, α]) > wtol && push!(out[α′, α], PSFTerm(pos, w[α′, α], Int[]))
        end
    end
    foreach(x -> sort!(x; by = t -> t.position), out)
    return out
end
