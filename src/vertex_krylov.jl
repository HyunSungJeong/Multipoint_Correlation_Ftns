# Discrete Krylov representation of the Matsubara regular part of the 3p vertex.
#
# Implements Algorithm 2 of the project note
#   [VD] "Jacobi matrices and couplings of the 3p vertex from discrete spectral data"
# on top of
#   [3p] "Liouvillian representation of the 3p vertex".
# Equation numbers (VDn), lemma / proposition numbers and the check labels
# (C1)-(C8) refer to [VD]. Input are discrete PSF weights on one symmetric grid
# (VD18)-(VD24); output are the Jacobi matrices, normalizations, couplings
# (VD27) and the self-energy coefficients (VD4), which assemble the regular part
# K̃^(12) of Lihm's 3p estimator through (VD6)-(VD12).
#
# Conventions (task sheet Sec. 2): one fixed pair (i1, i2) per data set, the
# bosonic composite B = q_{i1 i2} is called `Bq` in code (`siam(B = …)` is the
# magnetic field); X = (d†_1..d†_N, q†_1..q†_N), X_d = θX = (d_1..d_N, q_1..q_N);
# PSFs follow Kugler Eq. (28) and carry no permutation sign.
#
# Depends on LinearAlgebra only (and on the PSF / kernel code of this package).

# ============================================================================
# Small generic linear-algebra helpers (work for Float64 and BigFloat)
# ============================================================================

# Hermitian eigendecomposition: closed form for 1×1 and 2×2 (any real type),
# LAPACK otherwise. Returns (ascending eigenvalues, orthonormal eigenvectors).
function _vk_heigen(A::AbstractMatrix{T}) where {T}
    n = size(A, 1)
    R = real(T)
    if n == 1
        return R[real(A[1, 1])], ones(T, 1, 1)
    elseif n == 2 && !(R <: Union{Float32,Float64})
        a = real(A[1, 1]); d = real(A[2, 2]); c = (A[1, 2] + conj(A[2, 1])) / 2
        m = (a + d) / 2
        r = sqrt(((a - d) / 2)^2 + abs2(c))
        λ = R[m - r, m + r]
        V = zeros(T, 2, 2)
        if iszero(r)
            V[1, 1] = one(T); V[2, 2] = one(T)
            return λ, V
        end
        for j in 1:2
            x = T[c, λ[j] - a]
            y = T[λ[j] - d, conj(c)]
            v = norm(x) ≥ norm(y) ? x : y
            V[:, j] = v / norm(v)
        end
        return λ, V
    else
        F = eigen(Hermitian((A + A') / 2))
        return collect(F.values), Matrix(F.vectors)
    end
end

# principal square root of a Hermitian positive semidefinite matrix
function _vk_hsqrt(A::AbstractMatrix)
    size(A, 1) == 1 && return fill(convert(eltype(A), sqrt(max(real(A[1, 1]), zero(real(eltype(A)))))), 1, 1)
    λ, V = _vk_heigen(A)
    return V * Diagonal(sqrt.(max.(λ, zero(eltype(λ))))) * V'
end

_vk_opnorm(A::AbstractMatrix) = size(A) == (1, 1) ? abs(A[1, 1]) : opnorm(Matrix(A))
_vk_opnorm(x::Number) = abs(x)
_vk_maxabs(A) = isempty(A) ? 0.0 : Float64(maximum(abs, A))

# ============================================================================
# Grids (VD13), task sheet Sec. 2.3
# ============================================================================

"""
    ExactGrid(positions; degtol = 1e-10)

The bin grid of exact data (VD13): every bin holds one distinct transition
energy. The values `|E|` of all positions are clustered with absolute tolerance
`degtol` (single linkage); the cluster means are the positive centres, their
mirrors the negative ones, and `0` is a centre if any `|E| ≤ degtol`. A position
`x` is mapped to the centre of its cluster with the sign of `x` ([`bin_of`](@ref)),
so the bin map is odd by construction, `c(-E) = -c(E)` (VD13).

Fields: `b` (centres, ascending), `bar` (mirror map `b[bar[β]] = -b[β]`), `degtol`,
`spread` (largest within-cluster spread of `|E|`, reported), `zspread` (largest
`|E|` assigned to the zero bin).
"""
struct ExactGrid
    b::Vector{Float64}
    bar::Vector{Int}
    degtol::Float64
    spread::Float64
    zspread::Float64
    lo::Vector{Float64}
    hi::Vector{Float64}
    K::Int
    haszero::Bool
end

function ExactGrid(pos::AbstractVector{<:Real}; degtol::Real = 1e-10)
    a = sort!(abs.(Float64.(collect(pos))))
    zs = a[a .<= degtol]
    haszero = !isempty(zs)
    a = a[a .> degtol]
    lo = Float64[]; hi = Float64[]; c = Float64[]
    spread = 0.0
    i = 1
    while i ≤ length(a)
        j = i
        while j < length(a) && a[j + 1] - a[j] ≤ degtol
            j += 1
        end
        push!(lo, a[i]); push!(hi, a[j]); push!(c, sum(view(a, i:j)) / (j - i + 1))
        spread = max(spread, a[j] - a[i])
        i = j + 1
    end
    K = length(c)
    b = vcat(-reverse(c), haszero ? [0.0] : Float64[], c)
    nb = length(b)
    return ExactGrid(b, [nb + 1 - i for i in 1:nb], Float64(degtol), spread,
                     haszero ? maximum(zs) : 0.0, lo, hi, K, haszero)
end

"""
    bin_of(g, x) -> Int

Index of the bin of position `x` on grid `g` (an [`ExactGrid`](@ref) or a
[`BinnedGrid`](@ref)). For an `ExactGrid` it throws if `x` is not within one of
the clusters the grid was built from.
"""
function bin_of(g::ExactGrid, x::Real)
    ax = abs(x)
    z = g.haszero ? 1 : 0
    if ax ≤ g.degtol
        g.haszero || throw(ArgumentError("position $x lies in the zero bin, which this ExactGrid does not have"))
        return g.K + 1
    end
    k = searchsortedlast(g.lo, ax)
    if !(k ≥ 1 && ax ≤ g.hi[k] + g.degtol)
        k2 = k + 1
        (k2 ≤ g.K && g.lo[k2] - ax ≤ g.degtol) ||
            throw(ArgumentError("position $x is not on the ExactGrid (not within degtol of a cluster)"))
        k = k2
    end
    return x > 0 ? g.K + z + k : g.K + 1 - k
end

"""
    BinnedGrid(b, bar, binfun; label = "")

A binned grid (VD13) given by its centres `b` (ascending, symmetric), the mirror
map `bar`, and a bin function `binfun(x) -> index` that must be odd
(`binfun(-x) == bar[binfun(x)]`). Used for `MultipointNRG.LogGrid` by the mpNRG
adapter; `clamped` counts positions that the bin function reported as clamped.
"""
struct BinnedGrid{F}
    b::Vector{Float64}
    bar::Vector{Int}
    binfun::F
    label::String
end
BinnedGrid(b, bar, f; label::AbstractString = "") = BinnedGrid(Vector{Float64}(b), Vector{Int}(bar), f, String(label))
bin_of(g::BinnedGrid, x::Real) = g.binfun(x)

"""
    check_grid(b, bar) -> NamedTuple

Grid sanity (task sheet Sec. 3.3): strictly ascending centres, `b[bar[β]] = -b[β]`
exactly, and `bar` an involution.
"""
function check_grid(b::AbstractVector, bar::AbstractVector{<:Integer})
    asc = all(i -> b[i] < b[i + 1], 1:(length(b) - 1))
    sym = all(i -> b[bar[i]] == -b[i], eachindex(b))
    inv = all(i -> bar[bar[i]] == i, eachindex(bar))
    return (ascending = asc, symmetric = sym, involution = inv, ok = asc && sym && inv)
end

# ============================================================================
# Discrete data (VD18)-(VD24), task sheet Sec. 2.4
# ============================================================================

"""Sparse 3p weights: `(α, β) => block`, missing keys are zero blocks."""
const VKBlocks = Dict{NTuple{2,Int},Matrix{ComplexF64}}

"""
    DiscreteData

Discrete spectral input of Algorithm 2 for one channel `(i1, i2)` (VD18)-(VD24):

  - `N` orbitals; grid centres `b` and mirror map `bar` (VD13);
  - `W[β]`: the `2N×2N` Pelz weights `W̃_β` (VD19), assembled from both plain
    orderings, `[W̃_β]_ab = S_β[X_a†, X_b] + S_β̄[X_b, X_a†]`;
  - `Wplain[β]`: the first part `S_β[X_a†, X_b]` alone (Boltzmann diagnostic);
  - `Whole[γ]`: the hole Pelz weight `(X_d|Π_γ X_d)_+`, assembled from
    `S_γ[X_a, X_b†] + S_γ̄[X_b†, X_a]` (check C5);
  - `wB[β] = S_β[B†, B]`, `wBd[β] = S_β[B, B†]` (VD21);
  - the six 3p families (VD23), sparse `(α, β) => block`, with the barred indices
    of (VD23) already applied (task sheet (I3)):
    `s1` (2N×2N), `s2` (2N×2N), `s3[x]` (N×2N, row `i3`), `s4[x]` (2N×N, column
    `i3`), `s5[y]` (N×2N, row `i4`), `s6[y]` (2N×N, column `i4`), with
    `x ∈ (:q, :d)` and `y ∈ (:qd, :dd)` (`q†`, `d†`);
  - `T`; `meta`: provenance, references for the sum rules, clamping counts.
"""
struct DiscreteData
    N::Int
    b::Vector{Float64}
    bar::Vector{Int}
    W::Vector{Matrix{ComplexF64}}
    Wplain::Vector{Matrix{ComplexF64}}
    Whole::Vector{Matrix{ComplexF64}}
    wB::Vector{ComplexF64}
    wBd::Vector{ComplexF64}
    s1::VKBlocks
    s2::VKBlocks
    s3::Dict{Symbol,VKBlocks}
    s4::Dict{Symbol,VKBlocks}
    s5::Dict{Symbol,VKBlocks}
    s6::Dict{Symbol,VKBlocks}
    T::Float64
    meta::Dict{Symbol,Any}
end

Base.show(io::IO, d::DiscreteData) =
    print(io, "DiscreteData(N = $(d.N), $(length(d.b)) bins, T = $(d.T), ",
          "nnz(s1) = $(length(d.s1)) blocks, model = $(get(d.meta, :model, "?")))")

# Operator keys: (:X, a), (:Xd, a), (:B, 0), (:Bd, 0); X = (d†_1..d†_N, q†_1..q†_N).
const _KB = (:B, 0)
const _KBd = (:Bd, 0)

"""
    vk_psf_tuples(N) -> (two, three)

The operator-key tuples of all PSFs Algorithm 2 needs for `N` orbitals:
`2·(2N)² + 2` two-point PSFs ((I1), (I2) and the hole measure) and `6·(2N)²`
three-point PSFs ((I3); 24 for `N = 1`).
"""
function vk_psf_tuples(N::Integer)
    n2 = 2N
    two = Tuple{Tuple{Symbol,Int},Tuple{Symbol,Int}}[]
    for a in 1:n2, b in 1:n2
        push!(two, ((:Xd, a), (:X, b)))
        push!(two, ((:X, b), (:Xd, a)))
    end
    push!(two, (_KBd, _KB)); push!(two, (_KB, _KBd))
    two = unique(two)
    three = NTuple{3,Tuple{Symbol,Int}}[]
    for a in 1:n2, b in 1:n2
        push!(three, ((:Xd, a), _KB, (:X, b)))       # s1
        push!(three, ((:X, a), _KB, (:Xd, b)))       # s2
        push!(three, (_KB, (:Xd, a), (:X, b)))       # s3  (x = Xd[a])
        push!(three, ((:X, a), (:Xd, b), _KB))       # s4  (x = Xd[b])
        push!(three, (_KB, (:X, a), (:Xd, b)))       # s5  (y = X[a])
        push!(three, ((:Xd, a), (:X, b), _KB))       # s6  (y = X[b])
    end
    return two, unique(three)
end

# index of x_{i} (x = :q or :d) in X_d and of y_{i} (y = :qd or :dd) in X
_xidx(x::Symbol, i::Int, N::Int) = x === :d ? i : (x === :q ? N + i : throw(ArgumentError("x must be :q or :d")))
_yidx(y::Symbol, i::Int, N::Int) = y === :dd ? i : (y === :qd ? N + i : throw(ArgumentError("y must be :qd or :dd")))

"""
    build_discrete_data(N, b, bar, S2, S3; T, tr2 = nothing, tr3 = nothing, meta) -> DiscreteData

Assemble [`DiscreteData`](@ref) from histogram providers on one symmetric grid:
`S2(k1, k2)` returns the 2p weights `S_β[O1, O2]` (vector over bins) and
`S3(k1, k2, k3)` the 3p weights `S_αβ[O1, O2, O3]` as a `Dict((α, β) => w)`, for
operator keys `(:X, a)`, `(:Xd, a)`, `(:B, 0)`, `(:Bd, 0)`. Implements (VD19),
(VD21) and (VD23) with the barred indices of the task sheet's (I3). Optional
`tr2(k1, k2) = ⟨O1 O2⟩` and `tr3(k1, k2, k3) = ⟨O1 O2 O3⟩` give the independent
references of the sum rules (stored in `meta[:sumrule_ref]`, `meta[:trace3_ref]`).
"""
function build_discrete_data(N::Integer, b::AbstractVector, bar::AbstractVector{<:Integer}, S2, S3;
                             T::Real, tr2 = nothing, tr3 = nothing, meta = Dict{Symbol,Any}())
    nb = length(b); n2 = 2N
    g = check_grid(b, bar)
    g.ok || throw(ArgumentError("grid is not symmetric: $g"))
    cache2 = Dict{Any,Vector{ComplexF64}}()
    get2(k1, k2) = get!(() -> ComplexF64.(S2(k1, k2)), cache2, (k1, k2))
    W = [zeros(ComplexF64, n2, n2) for _ in 1:nb]
    Wp = [zeros(ComplexF64, n2, n2) for _ in 1:nb]
    Wh = [zeros(ComplexF64, n2, n2) for _ in 1:nb]
    for a in 1:n2, c in 1:n2
        A = get2((:Xd, a), (:X, c))         # S_β[X_a†, X_c]
        Bv = get2((:X, c), (:Xd, a))        # S_β[X_c, X_a†]
        Ah = get2((:X, a), (:Xd, c))        # S_γ[X_a, X_c†]
        Bh = get2((:Xd, c), (:X, a))        # S_γ[X_c†, X_a]
        for β in 1:nb
            W[β][a, c] = A[β] + Bv[bar[β]]
            Wp[β][a, c] = A[β]
            Wh[β][a, c] = Ah[β] + Bh[bar[β]]
        end
    end
    wB = get2(_KBd, _KB)
    wBd = get2(_KB, _KBd)

    totals3 = Dict{Any,ComplexF64}()
    function addfam!(fam::VKBlocks, keys, rows, cols, r, c, mapα, mapβ)
        d = S3(keys...)
        tot = zero(ComplexF64)
        for ((γ, δ), w) in d
            iszero(w) && continue
            tot += w
            k = (mapα(γ), mapβ(δ))
            blk = get!(() -> zeros(ComplexF64, rows, cols), fam, k)
            blk[r, c] += w
        end
        totals3[keys] = tot
        return fam
    end
    id = identity; br = i -> bar[i]
    s1 = VKBlocks(); s2 = VKBlocks()
    s3 = Dict(x => VKBlocks() for x in (:q, :d)); s4 = Dict(x => VKBlocks() for x in (:q, :d))
    s5 = Dict(y => VKBlocks() for y in (:qd, :dd)); s6 = Dict(y => VKBlocks() for y in (:qd, :dd))
    for a in 1:n2, c in 1:n2
        addfam!(s1, ((:Xd, a), _KB, (:X, c)), n2, n2, a, c, id, id)       # S(α,β)[X_a†, B, X_c]
        addfam!(s2, ((:X, a), _KB, (:Xd, c)), n2, n2, a, c, br, br)       # S(ᾱ,β̄)[X_a, B, X_c†]
    end
    for x in (:q, :d), i in 1:N, c in 1:n2
        j = _xidx(x, i, N)
        addfam!(s3[x], (_KB, (:Xd, j), (:X, c)), N, n2, i, c, id, id)     # S(α,β)[B, x_i, X_c]
        addfam!(s4[x], ((:X, c), (:Xd, j), _KB), n2, N, c, i, br, id)     # S(ᾱ,β)[X_c, x_i, B]
    end
    for y in (:qd, :dd), i in 1:N, c in 1:n2
        j = _yidx(y, i, N)
        addfam!(s5[y], (_KB, (:X, j), (:Xd, c)), N, n2, i, c, id, br)     # S(α,β̄)[B, y_i, X_c†]
        addfam!(s6[y], ((:Xd, c), (:X, j), _KB), n2, N, c, i, id, id)     # S(α,β)[X_c†, y_i, B]
    end

    meta = Dict{Symbol,Any}(meta)
    meta[:totals3] = totals3
    meta[:totals2] = Dict{Any,ComplexF64}(k => sum(v) for (k, v) in cache2)
    if tr2 !== nothing
        meta[:sumrule_ref] = ComplexF64[tr2((:Xd, a), (:X, c)) + tr2((:X, c), (:Xd, a)) for a in 1:n2, c in 1:n2]
        meta[:trace2_ref] = Dict{Any,ComplexF64}(k => tr2(k...) for k in keys(cache2))
    end
    if tr3 !== nothing
        meta[:trace3_ref] = Dict{Any,ComplexF64}(k => tr3(k...) for k in keys(totals3))
    end
    return DiscreteData(Int(N), Vector{Float64}(b), Vector{Int}(bar), W, Wp, Wh, wB, wBd,
                        s1, s2, s3, s4, s5, s6, Float64(T), meta)
end

"""
    discrete_data_terms(terms, N; grid = :exact, degtol = 1e-10, T, tr2, tr3, meta) -> DiscreteData

Build [`DiscreteData`](@ref) from exact Lehmann terms. `terms(keys...)` returns
`(P, w)` for the PSF of the keyed operators (`P` the `(ℓ-1)×n` positions, `w` the
weights), e.g. from `HubbardAtom.psf` or `MultipointNRG.ed_psf`. All PSFs of
[`vk_psf_tuples`](@ref) are computed once; with `grid = :exact` an
[`ExactGrid`](@ref) is built from all their positions (2p and both 3p
coordinates), otherwise `grid` is a [`BinnedGrid`](@ref) or `ExactGrid` used as
given. Weights are summed per bin.
"""
function discrete_data_terms(terms, N::Integer; grid = :exact, degtol::Real = 1e-10, T::Real,
                             tr2 = nothing, tr3 = nothing, meta = Dict{Symbol,Any}())
    two, three = vk_psf_tuples(N)
    raw = Dict{Any,Tuple{Matrix{Float64},Vector{ComplexF64}}}()
    for k in vcat(Any[t for t in two], Any[t for t in three])
        P, w = terms(k...)
        raw[k] = (Matrix{Float64}(P), ComplexF64.(w))
    end
    g = if grid === :exact
        allpos = Float64[]
        for (P, _) in values(raw)
            append!(allpos, vec(P))
        end
        ExactGrid(allpos; degtol = degtol)
    else
        grid
    end
    nb = length(g.b)
    nclamped = Ref(0)
    function binx(x)
        i = bin_of(g, x)
        return i
    end
    function S2(k1, k2)
        P, w = raw[(k1, k2)]
        h = zeros(ComplexF64, nb)
        for j in eachindex(w)
            h[binx(P[1, j])] += w[j]
        end
        return h
    end
    function S3(k1, k2, k3)
        P, w = raw[(k1, k2, k3)]
        d = Dict{NTuple{2,Int},ComplexF64}()
        for j in eachindex(w)
            key = (binx(P[1, j]), binx(P[2, j]))
            d[key] = get(d, key, zero(ComplexF64)) + w[j]
        end
        return d
    end
    meta = Dict{Symbol,Any}(meta)
    meta[:grid] = g
    meta[:nterms] = sum(length(v[2]) for v in values(raw))
    if g isa ExactGrid
        meta[:grid_kind] = :exact
        meta[:degtol] = g.degtol
        meta[:grid_spread] = g.spread
        meta[:grid_zspread] = g.zspread
    else
        meta[:grid_kind] = :binned
    end
    return build_discrete_data(N, g.b, g.bar, S2, S3; T = T, tr2 = tr2, tr3 = tr3, meta = meta)
end

# ============================================================================
# Hubbard atom data (task sheet Sec. 3.1)
# ============================================================================

"""
    atom_hamiltonian(m::HubbardAtomModel; εd = level_position(m), h = 0) -> Matrix

`H = εd (n↑ + n↓) + U n↑ n↓ - (h/2)(n↑ - n↓)` on the atom's Fock space (the field
convention of `MultipointNRG.siam`). `HubbardAtomModel` pins `εd = -U/2`; this
function frees it for the vertex-Krylov tests (task sheet: `εd = -0.3`, `U = 1`).
"""
function atom_hamiltonian(m::HubbardAtomModel; εd::Real = level_position(m), h::Real = 0.0)
    o = operators(m)
    return εd * o.N + m.U * (o.n_up * o.n_dn) - (h / 2) * (o.n_up - o.n_dn)
end

"""
    atom_spectrum(m::HubbardAtomModel; εd = level_position(m), h = 0) -> Spectrum

Diagonalization of [`atom_hamiltonian`](@ref) at `β = m.β`, with the same gauge
fixing as [`spectrum`](@ref).
"""
function atom_spectrum(m::HubbardAtomModel; εd::Real = level_position(m), h::Real = 0.0)
    o = operators(m)
    return _spectrum(atom_hamiltonian(m; εd = εd, h = h), o.N + sqrt(2.0) * o.Sz, m.β)
end

_spin_other(σ::Symbol) = σ === :up ? :dn : (σ === :dn ? :up : throw(ArgumentError("spin must be :up or :dn")))
_spin_sign(σ::Symbol) = σ === :up ? 1 : -1

"""
    channel_pair(σ, channel) -> (i1, i2)

The fixed pair of the bosonic composite `B = q_{i1 i2}` for the particle spin
`σ` (task sheet Sec. 2.1): `:updn` gives `(σ̄, σ̄)`, `:upup` gives `(σ, σ)`.
"""
channel_pair(σ::Symbol, channel::Symbol) =
    channel === :updn ? (_spin_other(σ), _spin_other(σ)) :
    channel === :upup ? (σ, σ) : throw(ArgumentError("channel must be :updn or :upup"))

"""
    vk_atom_operators(m, sp; spins, i12, εd, h, rot = nothing) -> NamedTuple

Dense Fock-basis operators of the atom, built from their definitions (task sheet
Sec. 2.1): `Hint = U n↑ n↓`, `q_i = [d_i, Hint]`, `q_i† = [Hint, d_i†]`,
`Bq = {q_{i1}, d_{i2}†}`; `X = (d†_i…, q†_i…)`, `Xd = (d_i…, q_i…)`. With `rot`
(an `N×N` unitary, `N = length(spins)`) the orbitals are `d'_i = Σ_σ rot[i,σ] d_σ`
(a rotated orbital basis for the `N = 2` block test); then `i12` are indices.
"""
function vk_atom_operators(m::HubbardAtomModel; spins = [:up], i12 = (:dn, :dn), rot = nothing)
    o = operators(m)
    dσ(σ) = σ === :up ? o.d_up : o.d_dn
    Hint = m.U * (o.n_up * o.n_dn)
    dbase = [ComplexF64.(dσ(σ)) for σ in spins]
    N = length(spins)
    ds = rot === nothing ? dbase : [sum(rot[i, j] * dbase[j] for j in 1:N) for i in 1:N]
    dds = [Matrix(d') for d in ds]
    q = [d * Hint - Hint * d for d in ds]                  # q_i = [d_i, Hint]
    qd = [Hint * dd - dd * Hint for dd in dds]             # q_i† = [Hint, d_i†]
    all(i -> norm(qd[i] - q[i]') ≤ 1e-14 * max(1, norm(q[i])), 1:N) ||
        throw(ErrorException("[Hint, d†] is not the adjoint of [d, Hint]"))
    # B = {q_{i1}, d†_{i2}}: spin labels refer to d_↑, d_↓ (which need not be in X,
    # e.g. B = q_{↓↓} for the (↑↓) channel); integers refer to the orbitals of X
    function qd_of(s)
        if s isa Symbol
            dd = ComplexF64.(dσ(s))
            return dd * Hint - Hint * dd, Matrix(dd')
        end
        return q[s], dds[s]
    end
    q1, _ = qd_of(i12[1])
    _, dd2 = qd_of(i12[2])
    Bq = q1 * dd2 + dd2 * q1                               # {q_{i1}, d†_{i2}}
    X = vcat(dds, qd)
    Xd = vcat(ds, q)
    return (X = X, Xd = Xd, Bq = Bq, Bqd = Matrix(Bq'), Hint = Hint, d = ds, q = q, i12 = i12)
end

"""
    discrete_data_atom(m::HubbardAtomModel, sp; σ = :up, channel = :updn, i12 = nothing,
                       grid = :exact, degtol = 1e-10, εd = level_position(m), h = 0, rot = nothing)
        -> DiscreteData

Discrete data (VD18)-(VD24) of the Hubbard atom from HubbardAtom ED (task sheet
Sec. 3.1). `sp` must be the spectrum of [`atom_hamiltonian`](@ref) with the same
`εd`, `h` (checked). All PSFs are computed with `psf(sp, Os)` and mapped to the
grid. `σ` may be a vector of spins (block code, `N = length(σ)`), in which case
`i12` must be given. `meta[:eps_sigma]` holds the level energies for the Dyson
check (C3).
"""
function discrete_data_atom(m::HubbardAtomModel, sp::Spectrum; σ = :up, channel::Symbol = :updn,
                            i12 = nothing, grid = :exact, degtol::Real = 1e-10,
                            εd::Real = level_position(m), h::Real = 0.0, rot = nothing)
    H = atom_hamiltonian(m; εd = εd, h = h)
    norm(sp.V' * H * sp.V - Diagonal(sp.E)) ≤ 1e-10 * max(1, norm(H)) ||
        throw(ArgumentError("sp is not the spectrum of atom_hamiltonian(m; εd = $εd, h = $h)"))
    spins = σ isa Symbol ? [σ] : collect(σ)
    pair = i12 === nothing ? (length(spins) == 1 ? channel_pair(spins[1], channel) :
                              throw(ArgumentError("give i12 for N > 1"))) : i12
    ops = vk_atom_operators(m; spins = spins, i12 = pair, rot = rot)
    N = length(spins)
    opof(k) = k[1] === :X ? ops.X[k[2]] : k[1] === :Xd ? ops.Xd[k[2]] : k[1] === :B ? ops.Bq : ops.Bqd
    function terms(ks...)
        ts = psf(sp, Tuple(opof(k) for k in ks))
        P = isempty(ts) ? zeros(length(ks) - 1, 0) : reduce(hcat, [t.position for t in ts])
        return P, [t.weight for t in ts]
    end
    tr2(k1, k2) = thermal_average(sp, opof(k1) * opof(k2))
    tr3(k1, k2, k3) = thermal_average(sp, opof(k1) * opof(k2) * opof(k3))
    epsσ = [εd - _spin_sign(s) * h / 2 for s in spins]
    o = operators(m)
    nσ(s) = s === :up ? o.n_up : o.n_dn
    meta = Dict{Symbol,Any}(:model => "Hubbard atom", :U => m.U, :εd => εd, :h => h, :β => m.β,
                            :spins => spins, :i12 => pair, :channel => channel, :rot => rot,
                            :eps_sigma => epsσ, :source => "HubbardAtom ED",
                            :occupations => Dict(s => real(thermal_average(sp, nσ(s))) for s in (:up, :dn)))
    # U = 0 propagator of the X-orbitals (C3); rotated orbitals d' = rot·d give rot·G0·rot†
    G0d(z) = Matrix(Diagonal(ComplexF64[1 / (z - e) for e in epsσ]))
    meta[:G0] = rot === nothing ? G0d : (z -> rot * G0d(z) * rot')
    return discrete_data_terms(terms, N; grid = grid, degtol = degtol, T = 1 / m.β,
                               tr2 = tr2, tr3 = tr3, meta = meta)
end

# ============================================================================
# Sanity checks on DiscreteData (task sheet Sec. 3.3)
# ============================================================================

"""
    data_sanity(data) -> NamedTuple

Checks of task sheet Sec. 3.3 on [`DiscreteData`](@ref): grid symmetry and `bar`
involution; hermiticity of `W̃_β` and its smallest eigenvalue relative to
`max‖W̃‖` (PSD); `w^B`, `w^{B†}` real and nonnegative; the sum rule
`Σ_β W̃_β = (X|X)_+` against the independent reference; the 3p totals against
`Tr[ρ O¹O²O³]`; the hole-measure identity (C5); and the Boltzmann-shortcut
diagnostic `max‖(1 + e^{-b_β/T}) S_β[X†,X] - W̃_β‖ / max‖W̃‖` (exact data: zero).
"""
function data_sanity(d::DiscreteData)
    g = check_grid(d.b, d.bar)
    Wmax = maximum(W -> _vk_opnorm(W), d.W; init = 0.0)
    herm = maximum(W -> _vk_maxabs(W - W'), d.W; init = 0.0) / max(Wmax, floatmin())
    minλ = minimum(W -> iszero(W) ? 0.0 : minimum(eigvals(Hermitian((W + W') / 2))), d.W; init = 0.0) /
           max(Wmax, floatmin())
    wBmin = minimum(real, d.wB; init = 0.0)
    wBdmin = minimum(real, d.wBd; init = 0.0)
    wBim = max(maximum(x -> abs(imag(x)), d.wB; init = 0.0), maximum(x -> abs(imag(x)), d.wBd; init = 0.0))
    S = sum(d.W)
    sumrule = haskey(d.meta, :sumrule_ref) ? _vk_maxabs(S - d.meta[:sumrule_ref]) : NaN
    tot3 = NaN
    if haskey(d.meta, :trace3_ref)
        ref = d.meta[:trace3_ref]
        tot3 = maximum(k -> abs(d.meta[:totals3][k] - ref[k]), collect(keys(ref)); init = 0.0)
    end
    tot2 = NaN
    if haskey(d.meta, :trace2_ref)
        ref = d.meta[:trace2_ref]
        tot2 = maximum(k -> abs(d.meta[:totals2][k] - ref[k]), collect(keys(ref)); init = 0.0)
    end
    c5 = check_C5(d)
    T = d.T
    boltz = 0.0
    for β in eachindex(d.b)
        iszero(d.Wplain[β]) && iszero(d.W[β]) && continue
        f = 1 + exp(-d.b[β] / T)
        isfinite(f) || continue
        boltz = max(boltz, _vk_maxabs(f * d.Wplain[β] - d.W[β]))
    end
    return (grid_ok = g.ok, bar_involution = g.involution, herm_err = herm, min_eig_rel = minλ,
            wB_min = wBmin, wBd_min = wBdmin, wB_imag = wBim, sumrule_err = sumrule,
            sumrule = S, totals2_err = tot2, totals3_err = tot3, C5 = c5,
            boltzmann_diff = boltz / max(Wmax, floatmin()))
end

"""
    check_C5(data) -> Float64

Hole symmetry (C5, Lemma 3(ii)): the hole Pelz weight at `-b_β`, assembled
independently in `data.Whole`, must equal `W̃_β^T`. Returns
`max_β ‖Whole[β̄] - W̃_β^T‖ / max‖W̃‖`.
"""
function check_C5(d::DiscreteData)
    Wmax = maximum(W -> _vk_opnorm(W), d.W; init = 0.0)
    err = maximum(β -> _vk_maxabs(d.Whole[d.bar[β]] - transpose(d.W[β])), eachindex(d.b); init = 0.0)
    return err / max(Wmax, floatmin())
end

# ============================================================================
# Stage 1: factors and coordinates (VD25)-(VD26), Lemma 3
# ============================================================================

"""
    coords(data; ranktol = 1e-13, T = Float64, normtol = 1e-12)
        -> (Bf, v, w, D, u, σ, Nbar, c, cp, rows, diag)

Stage 1 of Algorithm 2 (VD25)-(VD26), Lemma 3. Each `W̃_β = B_β† B_β` is factored
by an eigendecomposition, keeping eigenvalues `> ranktol·λ_max(W̃_β)` of the same
`β` (the dropped mass is reported); no normalization (Lemma 3: the coordinates
are those of `W̃_β` itself). `c_β = √w^B_β`, `c′_β = √w^{B†}_β` (negative
rounding clamped to 0 and reported). `v = (B_β E_d)_β`, `w = (B_β E_q)_β`,
`σ = v†w`, `u = w - vσ`, `N̄ = (u†u)^{1/2}` (principal root); `‖v†v - 1‖ ≤ normtol`
is asserted. `D[k]` is the energy of coordinate row `k`, `rows[β]` the rows of bin `β`.

Generic in the real type `T` (e.g. `BigFloat`); for `N = 1` all eigenproblems
are 2×2 and solved in closed form. Real data are processed in real arithmetic.
"""
function coords(d::DiscreteData; ranktol::Real = 1e-13, T::Type = Float64, normtol::Real = 1e-12)
    N = d.N; n2 = 2N; nb = length(d.b)
    isreal_data = all(W -> all(x -> iszero(imag(x)), W), d.W)
    E = isreal_data ? T : Complex{T}
    Bf = Vector{Matrix{E}}(undef, nb)
    rows = Vector{UnitRange{Int}}(undef, nb)
    R = 0
    dropped = zero(T); dropped_neg = zero(T); ndropped = 0; maxrelneg = zero(T)
    for β in 1:nb
        Wβ = isreal_data ? Matrix{E}(real.(d.W[β])) : Matrix{E}(d.W[β])
        if all(iszero, Wβ)
            Bf[β] = zeros(E, 0, n2); rows[β] = (R + 1):R
            continue
        end
        λ, V = _vk_heigen(Wβ)
        λmax = maximum(λ)
        keep = λmax > 0 ? findall(>(ranktol * λmax), λ) : Int[]
        for k in setdiff(eachindex(λ), keep)
            dropped += abs(λ[k]); ndropped += 1
            if λ[k] < 0
                dropped_neg += -λ[k]
                maxrelneg = max(maxrelneg, -λ[k] / max(λmax, floatmin(T)))
            end
        end
        Bf[β] = Matrix{E}(Diagonal(sqrt.(λ[keep])) * V[:, keep]')
        rows[β] = (R + 1):(R + length(keep))
        R += length(keep)
    end
    D = T[T(d.b[β]) for β in 1:nb for _ in rows[β]]
    v = reduce(vcat, [Bf[β][:, 1:N] for β in 1:nb]; init = zeros(E, 0, N))
    w = reduce(vcat, [Bf[β][:, (N + 1):n2] for β in 1:nb]; init = zeros(E, 0, N))
    vv = v' * v
    normerr = Float64(_vk_maxabs(vv - I))
    normerr ≤ normtol || throw(ErrorException("Stage 1: ‖v†v - 1‖ = $normerr > $normtol " *
                                              "(sum rule of the d block violated)"))
    σ = v' * w
    u = w - v * σ
    Nbar = _vk_hsqrt(u' * u)
    # bosonic factors
    function bfac(wv)
        im_max = maximum(x -> abs(imag(x)), wv; init = 0.0)
        re = real.(wv)
        neg = findall(<(0), re)
        c = T[sqrt(T(max(x, 0.0))) for x in re]
        return c, (imag_max = im_max, nclamped = length(neg),
                   clamped_min = isempty(neg) ? 0.0 : minimum(re[neg]))
    end
    c, cdiag = bfac(d.wB)
    cp, cpdiag = bfac(d.wBd)
    diag = Dict{Symbol,Any}(:R => R, :ranktol => ranktol, :rank_dropped_mass => Float64(dropped),
                            :rank_dropped_negative => Float64(dropped_neg), :rank_ndropped => ndropped,
                            :rank_max_rel_negative => Float64(maxrelneg), :norm_err => normerr,
                            :ranks => [length(r) for r in rows], :c_clamp => cdiag, :cp_clamp => cpdiag,
                            :real_arithmetic => isreal_data, :eltype => E)
    return (Bf = Bf, v = v, w = w, D = D, u = u, σ = σ, Nbar = Nbar, c = c, cp = cp, rows = rows, diag = diag)
end

# ============================================================================
# Algorithm 1: projected block Lanczos in coordinates (VD Sec. 5)
# ============================================================================

"""
    projected_lanczos(Dd, U1, Π; tol = 1e-12, maxsteps = typemax(Int))
        -> (eps, t, U, widths, resid, diag)

Algorithm 1 of [VD]: block Lanczos of `ΠDΠ` (`D = Diagonal(Dd)`) from the
orthonormal block `U1`, in coordinates. `Π === nothing` is the identity;
otherwise `Π` is a matrix `V` with orthonormal columns and the projector is
`1 - VV†`. Each step: `Y = ΠDU_n`, `ϵ_n = U_n†Y` (symmetrized); full
reorthogonalization against all previous blocks, twice, each followed by
`Y ← ΠY`; `t_{n+1} = (Y†Y)^{1/2}` (principal root). The chain stops when
`‖t_{n+1}‖ ≤ tol·max|D|` (terminated), and `resid = ‖Y‖` is reported; a
singular but nonzero `t_{n+1}` (deflation, `N > 1`) throws.

`eps[n]` = `ϵ_n`, `t[n]` = `t_{n+1}` (coupling of blocks `n` and `n+1`,
`length(t) = length(eps) - 1`), `U` the coordinates `[U_1 U_2 …]`.
Generic in the element type (Float64, BigFloat, complex).
"""
function projected_lanczos(Dd::AbstractVector{<:Real}, U1::AbstractMatrix, Π; tol::Real = 1e-12,
                           maxsteps::Integer = typemax(Int))
    R, k = size(U1)
    TT = promote_type(eltype(Dd), eltype(U1))
    RT = real(TT)
    scale = maximum(abs, Dd; init = zero(RT))
    thr = tol * scale
    Dv = Vector{RT}(Dd)
    applyΠ!(Y) = Π === nothing ? Y : (Y .-= Π * (Π' * Y); Y)
    cap = min(R, k * 16)
    Ubuf = zeros(TT, R, max(cap, k))
    Ubuf[:, 1:k] = U1
    ncols = k
    eps = Matrix{TT}[]
    ts = Matrix{TT}[]
    resid = zero(RT)
    terminated = false
    nsteps = 0
    while true
        nsteps += 1
        Un = view(Ubuf, :, (ncols - k + 1):ncols)
        Y = Dv .* Un
        applyΠ!(Y)
        ϵ = Un' * Y
        push!(eps, (ϵ + ϵ') / 2)
        for _ in 1:2
            Uc = view(Ubuf, :, 1:ncols)
            Y .-= Uc * (Uc' * Y)
            applyΠ!(Y)
        end
        t = _vk_hsqrt(Y' * Y)
        tn = _vk_opnorm(t)
        resid = RT(_vk_opnorm(Y))
        if tn ≤ thr
            terminated = true
            break
        end
        if k > 1
            sv = svdvals(Matrix{ComplexF64}(Y))
            minimum(sv) ≤ thr && throw(ErrorException(
                "projected_lanczos: deflation at step $nsteps (singular values $(Float64.(sv)) of the new " *
                "block, ‖t‖ = $(Float64(tn)) > tol·max|D| = $(Float64(thr))); block deflation is out of scope"))
        end
        if nsteps ≥ maxsteps || ncols + k > R
            break
        end
        push!(ts, t)
        if ncols + k > size(Ubuf, 2)
            newcap = min(R, 2 * size(Ubuf, 2))
            nb = zeros(TT, R, newcap)
            nb[:, 1:ncols] = view(Ubuf, :, 1:ncols)
            Ubuf = nb
        end
        Ubuf[:, (ncols + 1):(ncols + k)] = Y / t
        ncols += k
    end
    U = Ubuf[:, 1:ncols]
    G = U' * U
    orth = Float64(_vk_maxabs(G - I))
    projleak = Π === nothing ? 0.0 : Float64(_vk_maxabs(Π' * U))
    diag = Dict{Symbol,Any}(:steps => length(eps), :terminated => terminated, :resid => Float64(resid),
                            :threshold => Float64(thr), :orth_loss => orth, :proj_leak => projleak,
                            :maxsteps_hit => !terminated)
    return (eps = eps, t = ts, U = U, widths = fill(k, length(eps)), resid = resid, diag = diag)
end

"""
    BlockJacobi{T}

A block-tridiagonal (Jacobi) matrix with `k×k` blocks: `eps[n]` on the diagonal,
`lower[n]` at block `(n+1, n)`, `upper[n]` at block `(n, n+1)`.
"""
struct BlockJacobi{T}
    k::Int
    eps::Vector{Matrix{T}}
    lower::Vector{Matrix{T}}
    upper::Vector{Matrix{T}}
end

function BlockJacobi(k::Integer, eps::AbstractVector{<:AbstractMatrix}, lower::AbstractVector{<:AbstractMatrix},
                     upper::AbstractVector{<:AbstractMatrix})
    S = promote_type(map(eltype, eps)..., map(eltype, lower)..., map(eltype, upper)...)
    return BlockJacobi{S}(Int(k), [Matrix{S}(e) for e in eps], [Matrix{S}(l) for l in lower],
                          [Matrix{S}(u) for u in upper])
end

nblocks(J::BlockJacobi) = length(J.eps)
Base.size(J::BlockJacobi) = (J.k * nblocks(J), J.k * nblocks(J))

"""    jacobi_dense(J) -> Matrix — the full matrix of a [`BlockJacobi`](@ref)."""
function jacobi_dense(J::BlockJacobi{T}) where {T}
    k = J.k; L = nblocks(J)
    M = zeros(T, k * L, k * L)
    for n in 1:L
        r = ((n - 1) * k + 1):(n * k)
        M[r, r] = J.eps[n]
        if n < L
            s = (n * k + 1):((n + 1) * k)
            M[s, r] = J.lower[n]
            M[r, s] = J.upper[n]
        end
    end
    return M
end

_jacobi_transpose(J::BlockJacobi) = BlockJacobi(J.k, [Matrix(transpose(e)) for e in J.eps],
                                                [Matrix(transpose(u)) for u in J.upper],
                                                [Matrix(transpose(l)) for l in J.lower])

_vk_convert(::Type{S}, J::BlockJacobi) where {S} =
    BlockJacobi(J.k, [S.(e) for e in J.eps], [S.(l) for l in J.lower], [S.(u) for u in J.upper])

"""
    resolvent_col(J, z) -> Matrix   (kL × k)

The first block column `[(z - J)^{-1}]_{n1}` of a [`BlockJacobi`](@ref), by one
block-tridiagonal solve (block Thomas algorithm; no inversion of `J`).
"""
function resolvent_col(J::BlockJacobi, z::Number)
    k = J.k; L = nblocks(J)
    S = promote_type(eltype(J.eps[1]), typeof(z), ComplexF64)
    if k == 1
        a = S[z - J.eps[n][1, 1] for n in 1:L]
        lo = S[-J.lower[n][1, 1] for n in 1:(L - 1)]
        up = S[-J.upper[n][1, 1] for n in 1:(L - 1)]
        cp = zeros(S, L); yp = zeros(S, L)
        m = a[1]
        cp[1] = L > 1 ? up[1] / m : zero(S)
        yp[1] = one(S) / m
        for n in 2:L
            m = a[n] - lo[n - 1] * cp[n - 1]
            cp[n] = n < L ? up[n] / m : zero(S)
            yp[n] = (-lo[n - 1] * yp[n - 1]) / m
        end
        x = zeros(S, L)
        x[L] = yp[L]
        for n in (L - 1):-1:1
            x[n] = yp[n] - cp[n] * x[n + 1]
        end
        return reshape(x, L, 1)
    end
    Id = Matrix{S}(I, k, k)
    A = [z * Id - S.(J.eps[n]) for n in 1:L]
    Lo = [-S.(J.lower[n]) for n in 1:(L - 1)]
    Up = [-S.(J.upper[n]) for n in 1:(L - 1)]
    Cp = Vector{Matrix{S}}(undef, L); Yp = Vector{Matrix{S}}(undef, L)
    F = lu(A[1])
    Cp[1] = L > 1 ? F \ Up[1] : zeros(S, k, k)
    Yp[1] = F \ Id
    for n in 2:L
        F = lu(A[n] - Lo[n - 1] * Cp[n - 1])
        Cp[n] = n < L ? F \ Up[n] : zeros(S, k, k)
        Yp[n] = F \ (-Lo[n - 1] * Yp[n - 1])
    end
    X = Vector{Matrix{S}}(undef, L)
    X[L] = Yp[L]
    for n in (L - 1):-1:1
        X[n] = Yp[n] - Cp[n] * X[n + 1]
    end
    return reduce(vcat, X)
end

"""
    resolvent_row(J, z) -> Matrix   (k × kL)

The first block row `[(z - J)^{-1}]_{1m}`, from the first block column of `J^T`.
"""
resolvent_row(J::BlockJacobi, z::Number) = Matrix(transpose(resolvent_col(_jacobi_transpose(J), z)))

# ============================================================================
# Stage 2: chains (VD Prop. 4)
# ============================================================================

"""
    single_particle_basis(cs; tol = 1e-12) -> (Ω, dims, diag)

[VD] Prop. 13 (exact Wilson-chain data): an orthonormal basis `Ω` of the block
Krylov space of `T = D - w v†` from `v`, by block Arnoldi with full
reorthogonalization (twice); stops when the new block's norm is
`≤ tol·max|D|`. `span Ω = Φ(𝕂₀)`, the coordinates of the single-particle
Krylov space of `d†` under `L₀`.
"""
function single_particle_basis(cs; tol::Real = 1e-12)
    D = cs.D; v = cs.v; w = cs.w
    N = size(v, 2)
    scale = maximum(abs, D; init = zero(eltype(D)))
    thr = tol * scale
    Tm(X) = D .* X .- w * (v' * X)
    Ω = Matrix(v)
    blk = Matrix(v)
    norms = Float64[]
    while size(Ω, 2) < length(D)
        Y = Tm(blk)
        for _ in 1:2
            Y .-= Ω * (Ω' * Y)
        end
        nrm = _vk_opnorm(Y)
        push!(norms, Float64(nrm))
        nrm ≤ thr && break
        F = svd(Matrix(Y))
        keep = findall(>(thr), F.S)
        isempty(keep) && break
        blk = F.U[:, keep]
        Ω = hcat(Ω, blk)
    end
    return (Ω = Ω, dim = size(Ω, 2), diag = Dict(:block_norms => norms, :threshold => Float64(thr)))
end

"""
    VKChains

Output of [`chains`](@ref): `Jbar` (`J̄`), `Ubar` (`Ū`, coordinates of the
particle chain), `Nbar`, `σ`; `Jd = -J̄^T`; bosonic `JB`, `UB`, `NB` and `JBd`,
`UBd`, `NBd`; `r[n] = Ū_n† D v` (C4); the Stage-1 output `cs`; `diag`.
"""
struct VKChains
    Jbar::BlockJacobi
    Ubar::Matrix
    Nbar::Matrix
    σ::Matrix
    Jd::BlockJacobi
    JB::BlockJacobi{Float64}
    UB::Matrix{Float64}
    NB::Float64
    JBd::BlockJacobi{Float64}
    UBd::Matrix{Float64}
    NBd::Float64
    r::Vector{Matrix}
    cs::Any
    diag::Dict{Symbol,Any}
end

function _jacobi_from_lanczos(pl)
    return BlockJacobi(size(pl.eps[1], 1), pl.eps, pl.t, [Matrix(t') for t in pl.t])
end

"""
    hole_jacobi(Jbar) -> BlockJacobi

The hole chain `J̄_d = -J̄^T` ([VD] Prop. 4(ii), [3p] Lemma 2); no iteration.
"""
hole_jacobi(J::BlockJacobi) = BlockJacobi(J.k, [-Matrix(transpose(e)) for e in J.eps],
                                          [-Matrix(transpose(u)) for u in J.upper],
                                          [-Matrix(transpose(l)) for l in J.lower])

"""
    bosonic_chain(b, wts; tol = 1e-12, imtol = 1e-14) -> (J, U, N, diag)

Bosonic chain of [VD] Prop. 4(iii): Algorithm 1 with `D_B = diag(b_β)`,
`U_1 = (c_β/N_B)_β`, `Π = 1`, `N_B² = Σ_β w_β`. The weights must be real to
`imtol·max|w|` (asserted); negative rounding is clamped to 0 (reported in Stage 1).
"""
function bosonic_chain(b::AbstractVector, wts::AbstractVector; tol::Real = 1e-12, imtol::Real = 1e-14)
    wmax = maximum(abs, wts; init = 0.0)
    im_max = maximum(x -> abs(imag(x)), wts; init = 0.0)
    im_max ≤ imtol * max(wmax, 1.0) || throw(ErrorException("bosonic weights not real: max |Im| = $im_max"))
    c = sqrt.(max.(real.(wts), 0.0))
    NB = sqrt(sum(abs2, c))
    NB > 0 || throw(ErrorException("bosonic chain: N_B = 0"))
    pl = projected_lanczos(Vector{Float64}(b), reshape(c / NB, :, 1), nothing; tol = tol)
    J = _jacobi_from_lanczos(pl)
    maxim = maximum(e -> maximum(abs ∘ imag, e), vcat(J.eps, J.lower); init = 0.0)
    maxim ≤ imtol * max(1.0, maximum(abs, b)) || throw(ErrorException("bosonic chain coefficients not real: $maxim"))
    Jr = _vk_convert(Float64, BlockJacobi(1, [real.(e) for e in J.eps], [real.(l) for l in J.lower],
                                          [real.(u) for u in J.upper]))
    return (J = Jr, U = Matrix{Float64}(real.(pl.U)), N = NB, diag = pl.diag)
end

"""
    chains(data; ranktol = 1e-13, tol = 1e-12, T = Float64, stabilize = false, cs = nothing) -> VKChains

Stages 1-2 of Algorithm 2 ([VD] Prop. 4):

  - particle chain: Algorithm 1 with `D`, `U_1 = u N̄^{-1}`, `Π = 1 - vv†`
    (or `Π = 1 - ΩΩ†` with `stabilize = true`, [VD] Prop. 13), giving `J̄` and
    `Ū`; `r_n = Ū_n† D v` (C4);
  - hole chain `J̄_d = -J̄^T`, `Ū^d = -Ū^*` (no iteration);
  - bosonic chains of `B` and `B†` ([`bosonic_chain`](@ref)), real, Float64.

`T` is the real type of Stage 1 and the particle chain (e.g. `BigFloat`).
"""
function chains(d::DiscreteData; ranktol::Real = 1e-13, tol::Real = 1e-12, T::Type = Float64,
                stabilize::Bool = false, cs = nothing, maxsteps::Integer = typemax(Int))
    cs === nothing && (cs = coords(d; ranktol = ranktol, T = T))
    N = d.N
    U1 = cs.u * inv(cs.Nbar)
    spb = nothing
    Π = cs.v
    if stabilize
        spb = single_particle_basis(cs; tol = tol)
        Π = spb.Ω
    end
    pl = projected_lanczos(cs.D, U1, Π; tol = tol, maxsteps = maxsteps)
    Jbar = _jacobi_from_lanczos(pl)
    L = length(pl.eps)
    Dv = cs.D .* cs.v
    r = [Matrix(pl.U[:, ((n - 1) * N + 1):(n * N)]' * Dv) for n in 1:L]
    Jd = hole_jacobi(Jbar)
    bB = bosonic_chain(d.b, d.wB; tol = tol)
    bBd = bosonic_chain(d.b, d.wBd; tol = tol)
    diag = Dict{Symbol,Any}(:particle => pl.diag, :B => bB.diag, :Bd => bBd.diag, :L => L,
                            :LB => nblocks(bB.J), :LBd => nblocks(bBd.J), :stabilized => stabilize,
                            :coords => cs.diag)
    if spb !== nothing
        diag[:Omega_dim] = spb.dim
        diag[:Omega_diag] = spb.diag
        diag[:Omega_overlap] = Float64(_vk_maxabs(spb.Ω' * pl.U))
    end
    return VKChains(Jbar, pl.U, Matrix(cs.Nbar), Matrix(cs.σ), Jd, bB.J, bB.U, bB.N, bBd.J, bBd.U, bBd.N,
                    r, cs, diag)
end

# ============================================================================
# Stage 3: kernels (VD Lemma 6, Table 2)
# ============================================================================

"""
    BlockKernel

A block-sparse kernel matrix `C` on the coordinate rows of a bra leg and a ket
leg: `blocks[(α, β)] = C_αβ` (`r′_α × r_β`), `rows_bra[α]`, `rows_ket[β]`.
"""
struct BlockKernel
    blocks::Dict{NTuple{2,Int},Matrix{ComplexF64}}
    rows_bra::Vector{UnitRange{Int}}
    rows_ket::Vector{UnitRange{Int}}
    Rbra::Int
    Rket::Int
end

# pseudo-inverses of the factors of one leg; scalar legs (vector of c's) use 1/c
# for c > rtol·max c and 0 otherwise
function _leg_pinvs(F::AbstractVector{<:AbstractMatrix}, rtol, side::Symbol)
    P = Matrix{ComplexF64}[]
    rows = UnitRange{Int}[]
    R = 0
    for M in F
        Mc = Matrix{ComplexF64}(M)
        r = side === :bra ? size(Mc, 2) : size(Mc, 1)
        push!(P, r == 0 ? zeros(ComplexF64, side === :bra ? 0 : size(Mc, 2), side === :bra ? size(Mc, 1) : 0) :
                 pinv(Mc; rtol = rtol))
        push!(rows, (R + 1):(R + r))
        R += r
    end
    return P, rows, R, 0
end
function _leg_pinvs(c::AbstractVector{<:Real}, rtol, side::Symbol)
    cmax = maximum(c; init = 0.0)
    P = Matrix{ComplexF64}[]
    rows = UnitRange{Int}[]
    ndrop = 0
    for (β, x) in enumerate(c)
        if x > rtol * cmax
            push!(P, fill(ComplexF64(1 / x), 1, 1))
        else
            push!(P, zeros(ComplexF64, 1, 1)); x > 0 && (ndrop += 1)
        end
        push!(rows, β:β)
    end
    return P, rows, length(c), ndrop
end
_leg_factor(F::AbstractVector{<:AbstractMatrix}, i) = Matrix{ComplexF64}(F[i])
_leg_factor(c::AbstractVector{<:Real}, i) = fill(ComplexF64(c[i]), 1, 1)

"""
    kernel(s, Fbra, Fket; rtol = 1e-12) -> (C::BlockKernel, diag)

Stage 3 (VD Lemma 6): `C_αβ = (F′†_α)^+ s_αβ F_β^+` with Moore-Penrose
pseudo-inverses (`pinv(·; rtol)`); a scalar leg is passed as a vector of `c`'s
and uses `1/c` for `c > rtol·max c`, else `0`. `Fbra[α]` is the bra factor
`F′†_α` (a matrix with as many rows as `s_αβ`), `Fket[β]` the ket factor `F_β`.
`diag[:resid]` is the factorization residual (C7),
`max_{αβ}‖s_αβ - F′†_α C_αβ F_β‖ / max‖s‖` over **all** stored blocks.
"""
function kernel(s::VKBlocks, Fbra, Fket; rtol::Real = 1e-12)
    Pb, rb, Rb, nd1 = _leg_pinvs(Fbra, rtol, :bra)
    Pk, rk, Rk, nd2 = _leg_pinvs(Fket, rtol, :ket)
    blocks = Dict{NTuple{2,Int},Matrix{ComplexF64}}()
    resid = 0.0; smax = 0.0
    for ((α, β), sab) in s
        smax = max(smax, _vk_maxabs(sab))
        C = Pb[α] * sab * Pk[β]
        rec = isempty(C) ? zeros(ComplexF64, size(sab)) : _leg_factor(Fbra, α) * C * _leg_factor(Fket, β)
        resid = max(resid, _vk_maxabs(sab - rec))
        isempty(C) || (blocks[(α, β)] = C)
    end
    diag = Dict{Symbol,Any}(:resid => resid / max(smax, floatmin()), :resid_abs => resid, :smax => smax,
                            :nblocks => length(blocks), :scalar_dropped => nd1 + nd2)
    return BlockKernel(blocks, rb, rk, Rb, Rk), diag
end

"""
    sandwich(Abra, C, Aket) -> Matrix

`Abra† C Aket` for a [`BlockKernel`](@ref) (Stage 4), accumulated block by block.
"""
function sandwich(Abra::AbstractMatrix, C::BlockKernel, Aket::AbstractMatrix)
    realC = all(b -> all(x -> iszero(imag(x)), b), values(C.blocks))
    S = (realC && eltype(Abra) <: Real && eltype(Aket) <: Real) ? Float64 : ComplexF64
    # CA = C * Aket, accumulated block by block (C is block-sparse), then one BLAS product
    CA = zeros(S, C.Rbra, size(Aket, 2))
    for ((α, β), blk) in C.blocks
        ra = C.rows_bra[α]; rb = C.rows_ket[β]
        if length(ra) == 1 && length(rb) == 1
            c = S <: Real ? real(blk[1, 1]) : blk[1, 1]
            @views CA[ra[1], :] .+= c .* Aket[rb[1], :]
        else
            @views CA[ra, :] .+= (S <: Real ? real.(blk) : blk) * Aket[rb, :]
        end
    end
    return Matrix{S}(Abra)' * CA
end

"""
    kernels(data, ch; rtol = 1e-12) -> (C, diag)

All kernels of Stage 3 with the factors of [VD] Table 2:

| family | bra factor `F′†_α` | ket factor `F_β` |
|:--|:--|:--|
| (1) | `B_α†` | `B_β` |
| (2) | `B_α^T` | `B_β^*` |
| (3) | `c′_α` | `B_β` |
| (4) | `B_α^T` | `c_β` |
| (5) | `c′_α` | `B_β^*` |
| (6) | `B_α†` | `c_β` |

For `N > 1` the families 3-6 are computed per orbital index `i3` / `i4`.
`diag[:C7]` is the largest factorization residual over the six families.
"""
function kernels(d::DiscreteData, ch::VKChains; rtol::Real = 1e-12)
    cs = ch.cs
    N = d.N
    Bf = [Matrix{ComplexF64}(B) for B in cs.Bf]
    Bdag = [Matrix(B') for B in Bf]
    BT = [Matrix(transpose(B)) for B in Bf]
    Bconj = [conj.(B) for B in Bf]
    c = Float64.(cs.c); cp = Float64.(cs.cp)
    rowsel(s::VKBlocks, i::Int) = VKBlocks(k => v[i:i, :] for (k, v) in s)
    colsel(s::VKBlocks, i::Int) = VKBlocks(k => v[:, i:i] for (k, v) in s)
    C = Dict{Any,BlockKernel}()
    D = Dict{Any,Any}()
    C[1], D[1] = kernel(d.s1, Bdag, Bf; rtol = rtol)
    C[2], D[2] = kernel(d.s2, BT, Bconj; rtol = rtol)
    for x in (:q, :d), i in 1:N
        C[(3, x, i)], D[(3, x, i)] = kernel(rowsel(d.s3[x], i), cp, Bf; rtol = rtol)
        C[(4, x, i)], D[(4, x, i)] = kernel(colsel(d.s4[x], i), BT, c; rtol = rtol)
    end
    for y in (:qd, :dd), i in 1:N
        C[(5, y, i)], D[(5, y, i)] = kernel(rowsel(d.s5[y], i), cp, Bconj; rtol = rtol)
        C[(6, y, i)], D[(6, y, i)] = kernel(colsel(d.s6[y], i), Bdag, c; rtol = rtol)
    end
    fam(k) = k isa Integer ? k : k[1]
    C7fam = Dict(f => maximum(D[k][:resid] for k in keys(D) if fam(k) == f) for f in 1:6)
    return C, Dict{Symbol,Any}(:per_kernel => D, :C7_family => C7fam, :C7 => maximum(values(C7fam)))
end

# ============================================================================
# Stage 4: couplings (VD27), Prop. 7
# ============================================================================

"""
    VKCouplings

Couplings of (VD27): `V1 = Ū†C¹Ū`, `V2 = Ū^T C² Ū^*` (`NL×NL`); `V3[x][i3] =
U_{B†}† C^{(3,x)} Ū` (`L_{B†}×NL`), `V4[x][i3] = -Ū^T C^{(4,x)} U_B` (`NL×L_B`),
`V5[y][i4] = -U_{B†}† C^{(5,y)} Ū^*` (`L_{B†}×NL`), `V6[y][i4] = Ū† C^{(6,y)} U_B`
(`NL×L_B`), with `x ∈ (:q, :d)`, `y ∈ (:qd, :dd)`.
"""
struct VKCouplings{T<:Number}
    V1::Matrix{T}
    V2::Matrix{T}
    V3::Dict{Symbol,Vector{Matrix{T}}}
    V4::Dict{Symbol,Vector{Matrix{T}}}
    V5::Dict{Symbol,Vector{Matrix{T}}}
    V6::Dict{Symbol,Vector{Matrix{T}}}
end

"""
    couplings(data, ch, C) -> VKCouplings

Stage 4 of Algorithm 2, (VD27) ([VD] Prop. 7). The hole coordinates `-Ū^*` give
the signs of `V^{(4)}` and `V^{(5)}`; they cancel in `V̄^{(2)}`.
"""
function couplings(d::DiscreteData, ch::VKChains, C)
    N = d.N
    U = eltype(ch.Ubar) <: Real ? Matrix{Float64}(ch.Ubar) : Matrix{ComplexF64}(ch.Ubar)   # real data stay real
    Uc = conj.(U)
    UB = ch.UB; UBd = ch.UBd
    V1 = sandwich(U, C[1], U)
    V2 = sandwich(Uc, C[2], Uc)                         # Ū^T C Ū^*:  (Ū^*)† = Ū^T
    V3 = Dict(x => [sandwich(UBd, C[(3, x, i)], U) for i in 1:N] for x in (:q, :d))
    V4 = Dict(x => [-sandwich(Uc, C[(4, x, i)], UB) for i in 1:N] for x in (:q, :d))
    V5 = Dict(y => [-sandwich(UBd, C[(5, y, i)], Uc) for i in 1:N] for y in (:qd, :dd))
    V6 = Dict(y => [sandwich(U, C[(6, y, i)], UB) for i in 1:N] for y in (:qd, :dd))
    T = promote_type(eltype(V1), eltype(V2), (eltype(v) for D in (V3, V4, V5, V6) for vs in values(D) for v in vs)...)
    cv(D) = Dict(k => [Matrix{T}(v) for v in vs] for (k, vs) in D)
    return VKCouplings{T}(Matrix{T}(V1), Matrix{T}(V2), cv(V3), cv(V4), cv(V5), cv(V6))
end

# ============================================================================
# Stage 5: self-energy coefficients (VD4), Lemma 8
# ============================================================================

"""
    Gtilde(data, z) -> Matrix

The augmented propagator `G̃(z) = Σ_β W̃_β / (z - b_β)` (Lemma 8), `2N×2N`.
"""
function Gtilde(d::DiscreteData, z::Number)
    n2 = 2d.N
    G = zeros(ComplexF64, n2, n2)
    for β in eachindex(d.b)
        W = d.W[β]
        iszero(W) && continue
        G .+= W ./ (z - d.b[β])
    end
    return G
end

"""
    sigmas(data, ch, z) -> (ΣR, ΣL, ΣS)

Stage 5 (VD4): `Σ^R = G̃_dd^{-1} G̃_dq`, `Σ^L = G̃_qd G̃_dd^{-1}`, and the symmetric
estimator from the chain, `Σ^S = σ + N̄[(z - J̄)^{-1}]_{11}N̄`.
"""
function sigmas(d::DiscreteData, ch::VKChains, z::Number)
    N = d.N
    G = Gtilde(d, z)
    Gdd = G[1:N, 1:N]; Gdq = G[1:N, (N + 1):(2N)]; Gqd = G[(N + 1):(2N), 1:N]
    ΣR = Gdd \ Gdq
    ΣL = Gqd / Gdd
    ΣS = sigma_chain(ch, z)
    return ΣR, ΣL, ΣS
end

"""    sigma_chain(ch, z) — `Σ^S(z) = σ + N̄[(z - J̄)^{-1}]_{11}N̄` (VD4)."""
function sigma_chain(ch::VKChains, z::Number)
    N = size(ch.Nbar, 1)
    Nb = ComplexF64.(ch.Nbar)
    col = resolvent_col(ch.Jbar, z)
    return ComplexF64.(ch.σ) + Nb * ComplexF64.(col[1:N, :]) * Nb
end

# ============================================================================
# Assembly (VD6)-(VD12)
# ============================================================================

"""Permutation signs `ζ^p` of `(p1, …, p6)` for `(q12, T3, T4)` (task sheet Sec. 2.5)."""
const VK_PERMS = ([2, 1, 3], [3, 1, 2], [1, 2, 3], [3, 2, 1], [1, 3, 2], [2, 3, 1])
vk_zeta() = [permutation_sign(p, [false, true, true]) for p in VK_PERMS]

"""Fermionic Matsubara frequency `(2n+1)πT`."""
vk_fermionic(n::Integer, T::Real) = (2n + 1) * π * T

"""
    VKResolventCache

Caches the first block rows / columns of the four resolvents that (VD7)-(VD12)
need, keyed by frequency.
"""
struct VKResolventCache
    row::Dict{Tuple{Symbol,ComplexF64},Matrix{ComplexF64}}
    col::Dict{Tuple{Symbol,ComplexF64},Matrix{ComplexF64}}
    sig::Dict{ComplexF64,Any}
end
VKResolventCache() = VKResolventCache(Dict{Tuple{Symbol,ComplexF64},Matrix{ComplexF64}}(),
                                      Dict{Tuple{Symbol,ComplexF64},Matrix{ComplexF64}}(), Dict{ComplexF64,Any}())

function _jac(ch::VKChains, s::Symbol)
    s === :J && return ch.Jbar
    s === :Jd && return ch.Jd
    s === :B && return ch.JB
    s === :Bd && return ch.JBd
    throw(ArgumentError("unknown chain $s"))
end
_row(ch, cache, s, z) = get!(() -> ComplexF64.(resolvent_row(_jac(ch, s), z)), cache.row, (s, ComplexF64(z)))
_col(ch, cache, s, z) = get!(() -> ComplexF64.(resolvent_col(_jac(ch, s), z)), cache.col, (s, ComplexF64(z)))
_sig(d, ch, cache, z) = get!(() -> sigmas(d, ch, z), cache.sig, ComplexF64(z))

"""
    Ktilde(ch, cp, data, ω12, ω3, ω4; cache = VKResolventCache(), perms = false)
        -> ComplexF64 (N = 1) / Matrix (N > 1)

The regular part `K̃^(12)(iω12, iω3, iω4) = Σ_p ζ^p K_p` of (VD6)-(VD12), with
`Σ^L(iω3)` in slot 3 and `Σ^R(-iω4)` in slot 4 (VD5). Frequencies are real
numbers (`ω12 + ω3 + ω4 = 0`, `ω12 ≠ 0`). The matrix of `p2` is transposed
(VD8); the minus signs of (VD10), (VD11) are part of the formulas. Resolvent
elements are obtained by block-tridiagonal solves, cached in `cache`. With
`perms = true` returns `(total, [K_p1, …, K_p6])` (without `ζ`).
"""
function Ktilde(ch::VKChains, cp::VKCouplings, d::DiscreteData, ω12::Real, ω3::Real, ω4::Real;
                cache::VKResolventCache = VKResolventCache(), perms::Bool = false)
    abs(ω12 + ω3 + ω4) ≤ 1e-12 * max(1, abs(ω3), abs(ω4)) || throw(ArgumentError("need ω12 + ω3 + ω4 = 0"))
    ω12 == 0 && throw(ArgumentError("the regular part needs ω12 ≠ 0"))
    N = d.N
    Nb = ComplexF64.(ch.Nbar); NbT = Matrix(transpose(Nb))
    _, ΣL, _ = _sig(d, ch, cache, im * ω3)
    ΣR, _, _ = _sig(d, ch, cache, -im * ω4)
    r3 = Nb * _row(ch, cache, :J, im * ω3)            # N̄[(iω3 - J̄)^{-1}]_{1m}
    c4 = _col(ch, cache, :J, -im * ω4) * Nb           # [(-iω4 - J̄)^{-1}]_{n1}N̄
    rd4 = NbT * _row(ch, cache, :Jd, im * ω4)         # N̄^T[(iω4 - J̄_d)^{-1}]_{1m}
    cd3 = _col(ch, cache, :Jd, -im * ω3) * NbT        # [(-iω3 - J̄_d)^{-1}]_{n1}N̄^T
    rBd = ch.NBd * _row(ch, cache, :Bd, im * ω12)     # N_{B†}[(iω12 - J_{B†})^{-1}]_{1m}
    cB = _col(ch, cache, :B, -im * ω12) * ch.NB       # [(-iω12 - J_B)^{-1}]_{n1}N_B
    K1 = r3 * (cp.V1 * c4)                            # (VD7)
    K2 = Matrix(transpose(rd4 * (cp.V2 * cd3)))       # (VD8): the product is [K_p2]_{i4 i3}
    K3 = zeros(ComplexF64, N, N); K4 = zeros(ComplexF64, N, N)
    K5 = zeros(ComplexF64, N, N); K6 = zeros(ComplexF64, N, N)
    # (VD5) is linear in the couplings: contract each V with the resolvent vectors
    # first, then dress with Σ^L / Σ^R (avoids forming V^{(a,T)} per frequency)
    k3 = Dict(x => [rBd * (cp.V3[x][j] * c4) for j in 1:N] for x in (:q, :d))      # 1×N each
    k4 = Dict(x => [rd4 * (cp.V4[x][j] * cB) for j in 1:N] for x in (:q, :d))      # N×1 each
    k5 = Dict(y => [rBd * (cp.V5[y][j] * cd3) for j in 1:N] for y in (:qd, :dd))   # 1×N each
    k6 = Dict(y => [r3 * (cp.V6[y][j] * cB) for j in 1:N] for y in (:qd, :dd))     # N×1 each
    for i3 in 1:N
        K3[i3, :] = vec(k3[:q][i3] - sum(ΣL[i3, j] * k3[:d][j] for j in 1:N))      # (VD9)
        K4[i3, :] = -vec(k4[:q][i3] - sum(ΣL[i3, j] * k4[:d][j] for j in 1:N))     # (VD10)
    end
    for i4 in 1:N
        K5[:, i4] = -vec(k5[:qd][i4] - sum(k5[:dd][j] * ΣR[j, i4] for j in 1:N))   # (VD11)
        K6[:, i4] = vec(k6[:qd][i4] - sum(k6[:dd][j] * ΣR[j, i4] for j in 1:N))    # (VD12)
    end
    Ks = [K1, K2, K3, K4, K5, K6]
    ζ = vk_zeta()
    tot = sum(ζ[p] * Ks[p] for p in 1:6)
    out = N == 1 ? tot[1, 1] : tot
    perms || return out
    return out, (N == 1 ? [K[1, 1] for K in Ks] : Ks)
end

# ============================================================================
# Reference: Lihm's estimator directly from the same data (task sheet Sec. 5.1)
# ============================================================================

const _VK_DUMMY_SP = Ref{Any}(nothing)
function _vk_dummy_spectrum()
    _VK_DUMMY_SP[] === nothing && (_VK_DUMMY_SP[] = Spectrum(Float64[], zeros(0, 0), Float64[], 1.0))
    return _VK_DUMMY_SP[]
end
const _VK_DUMMY_OPS = (zeros(0, 0), zeros(0, 0), zeros(0, 0))

"""
    direct_caches(data) -> Dict

`PermutedPSF` caches for the correlators `G[q_{i1i2}, x_{i3}, y_{i4}]`, `x ∈ (:q, :d)`,
`y ∈ (:qd, :dd)`, built from the six families of `data` (no recomputed PSFs).
Every permutation PSF of `(q12, x, y)` is an entry of one family; bin weights
become `PSFTerm`s at the bin centres (`b[α]`, or `-b[α]` for barred indices):

| perm | operator order | family | positions |
|:--|:--|:--|:--|
| (2,1,3) | `(x, B, y)` | `s1[α,β][a_x, b_y]` | `(b_α, b_β)` |
| (3,1,2) | `(y, B, x)` | `s2[α,β][a_y, b_x]` | `(-b_α, -b_β)` |
| (1,2,3) | `(B, x, y)` | `s3[x][α,β][i3, b_y]` | `(b_α, b_β)` |
| (3,2,1) | `(y, x, B)` | `s4[x][α,β][a_y, i3]` | `(-b_α, b_β)` |
| (1,3,2) | `(B, y, x)` | `s5[y][α,β][i4, b_x]` | `(b_α, -b_β)` |
| (2,3,1) | `(x, y, B)` | `s6[y][α,β][a_x, i4]` | `(b_α, b_β)` |

Keys: `(x, i3, y, i4)`; values: `Vector{PermutedPSF}` (all six permutations, with ζ).
With `zeta = false` all ζ are set to 1 (raw permutation terms).
"""
function direct_caches(d::DiscreteData; zeta::Bool = true)
    N = d.N; b = d.b
    out = Dict{Any,Vector{PermutedPSF}}()
    isf = [false, true, true]
    function terms(fam::VKBlocks, r::Int, c::Int, sα::Int, sβ::Int)
        ts = PSFTerm[]
        for ((α, β), blk) in fam
            w = blk[r, c]
            iszero(w) && continue
            push!(ts, PSFTerm([sα * b[α], sβ * b[β]], w, Int[]))
        end
        return ts
    end
    for x in (:q, :d), i3 in 1:N, y in (:qd, :dd), i4 in 1:N
        ax = _xidx(x, i3, N); by = _yidx(y, i4, N)
        perms = PermutedPSF[]
        for p in all_permutations(3)
            ζ = zeta ? permutation_sign(p, isf) : 1
            ts = if p == [2, 1, 3]
                terms(d.s1, ax, by, 1, 1)
            elseif p == [3, 1, 2]
                terms(d.s2, by, ax, -1, -1)
            elseif p == [1, 2, 3]
                terms(d.s3[x], i3, by, 1, 1)
            elseif p == [3, 2, 1]
                terms(d.s4[x], by, i3, -1, 1)
            elseif p == [1, 3, 2]
                terms(d.s5[y], i4, ax, 1, -1)
            else # [2, 3, 1]
                terms(d.s6[y], ax, i4, 1, 1)
            end
            push!(perms, PermutedPSF(p, ζ, ts))
        end
        out[(x, i3, y, i4)] = perms
    end
    return out
end

"""
    direct_correlators(caches, N, ω12, ω3, ω4; perm = nothing) -> Dict

`G[q12, x_{i3}, y_{i4}](iω12, iω3, iω4)` as `N×N` matrices for `(x, y)`, evaluated
with `regular_sum` (Kugler's regular kernel) from the caches; the frequencies are
passed as `z = [iω12, iω3, iω4]`. With `perm = k` only permutation `p_k`
(numbering of [`VK_PERMS`](@ref)) contributes.
"""
function direct_correlators(caches, N::Integer, ω12::Real, ω3::Real, ω4::Real; perm = nothing)
    z = ComplexF64[im * ω12, im * ω3, im * ω4]
    sp = _vk_dummy_spectrum()
    G = Dict{Tuple{Symbol,Symbol},Matrix{ComplexF64}}()
    for x in (:q, :d), y in (:qd, :dd)
        M = zeros(ComplexF64, N, N)
        for i3 in 1:N, i4 in 1:N
            c = caches[(x, i3, y, i4)]
            c = perm === nothing ? c : [pp for pp in c if pp.p == VK_PERMS[perm]]
            M[i3, i4] = regular_sum(sp, _VK_DUMMY_OPS, z; is_fermionic = [false, true, true], cache = c)
        end
        G[(x, y)] = M
    end
    return G
end

"""
    Kdirect(data, caches, ω12, ω3, ω4; perm = nothing, cache = VKResolventCache(), ch = nothing)

Lihm's estimator [Lihm Eq. (118)] directly from the PSFs of `data` (reference for C8):
`K = G_{qq†} - Σ^L(iω3) G_{dq†} - G_{qd†} Σ^R(-iω4) + Σ^L(iω3) G_{dd†} Σ^R(-iω4)`,
with `Σ^L`, `Σ^R` of (VD4) from `G̃`. Returns a number for `N = 1`.
"""
function Kdirect(d::DiscreteData, caches, ω12::Real, ω3::Real, ω4::Real; perm = nothing)
    N = d.N
    G = direct_correlators(caches, N, ω12, ω3, ω4; perm = perm)
    G3 = Gtilde(d, im * ω3); G4 = Gtilde(d, -im * ω4)
    ΣL = G3[(N + 1):(2N), 1:N] / G3[1:N, 1:N]
    ΣR = G4[1:N, 1:N] \ G4[1:N, (N + 1):(2N)]
    K = G[(:q, :qd)] - ΣL * G[(:d, :qd)] - G[(:q, :dd)] * ΣR + ΣL * G[(:d, :dd)] * ΣR
    return N == 1 ? K[1, 1] : K
end

# ============================================================================
# Pipeline and checks (C1)-(C8)
# ============================================================================

"""
    VKResult

Everything Algorithm 2 produces for one data set: `data`, `ch` ([`VKChains`](@ref)),
`C` (kernels), `kdiag` (Stage-3 diagnostics, incl. C7), `cp` ([`VKCouplings`](@ref)).
"""
struct VKResult
    data::DiscreteData
    ch::VKChains
    C::Dict{Any,BlockKernel}
    kdiag::Dict{Symbol,Any}
    cp::VKCouplings
end

"""
    vertex_krylov(data; ranktol = 1e-13, tol = 1e-12, rtol = 1e-12, stabilize = false) -> VKResult

Algorithm 2 of [VD], Stages 1-4 (Stage 5 is evaluated per frequency by
[`sigmas`](@ref) and [`Ktilde`](@ref)).
"""
function vertex_krylov(d::DiscreteData; ranktol::Real = 1e-13, tol::Real = 1e-12, rtol::Real = 1e-12,
                       stabilize::Bool = false)
    ch = chains(d; ranktol = ranktol, tol = tol, stabilize = stabilize)
    C, kd = kernels(d, ch; rtol = rtol)
    cp = couplings(d, ch, C)
    return VKResult(d, ch, C, kd, cp)
end

"""
    vk_testpoints(data; nmats = 50, nreal = 50, η = 0.1) -> Vector{ComplexF64}

The test points of C1, C3, C6: 50 fermionic Matsubara points `i(2n+1)πT`,
`n = -25..24`, and 50 points `x + iη` with `x` uniform in `±1.2·max|b_β|` over the
bins that carry weight.
"""
function vk_testpoints(d::DiscreteData; nmats::Integer = 50, nreal::Integer = 50, η::Real = 0.1)
    ns = (-(nmats ÷ 2)):(nmats - nmats ÷ 2 - 1)
    zs = ComplexF64[im * vk_fermionic(n, d.T) for n in ns]
    supp = [abs(d.b[β]) for β in eachindex(d.b) if !iszero(d.W[β]) || !iszero(d.wB[β]) || !iszero(d.wBd[β])]
    xmax = 1.2 * maximum(supp; init = 1.0)
    append!(zs, ComplexF64[x + im * η for x in range(-xmax, xmax; length = nreal)])
    return zs
end

"""
    check_C1(data, ch; zs) -> Float64

(C1) self-energy, algebraic (Lemma 8(ii)):
`max_z ‖N̄[(z - J̄)^{-1}]_{11}N̄ - (G̃_qq - G̃_qd G̃_dd^{-1} G̃_dq)‖ / max_z ‖G̃_qq - …‖`.
"""
function check_C1(d::DiscreteData, ch::VKChains; zs = vk_testpoints(d))
    N = d.N
    err = 0.0; ref = 0.0
    for z in zs
        G = Gtilde(d, z)
        Gdd = G[1:N, 1:N]; Gdq = G[1:N, (N + 1):(2N)]; Gqd = G[(N + 1):(2N), 1:N]; Gqq = G[(N + 1):(2N), (N + 1):(2N)]
        schur = Gqq - Gqd * (Gdd \ Gdq)
        chain = sigma_chain(ch, z) - ComplexF64.(ch.σ)
        err = max(err, _vk_maxabs(chain - schur)); ref = max(ref, _vk_maxabs(schur))
    end
    return err / max(ref, floatmin())
end

"""
    check_C3(data, ch; zs, G0 = data.meta[:G0]) -> Float64

(C3) Dyson: `Σ^S(z)` against `G_0(z)^{-1} - G̃_dd(z)^{-1}`; for the atom
`G_0(z) = (z - ε_σ)^{-1}`, for a Wilson chain `G_0` is the impurity Green's
function of the same chain at `U = 0` (adapter). Relative to `max‖Σ^S - σ‖`.
"""
function check_C3(d::DiscreteData, ch::VKChains; zs = vk_testpoints(d), G0 = get(d.meta, :G0, nothing))
    G0 === nothing && return NaN
    N = d.N
    err = 0.0; ref = 0.0
    for z in zs
        Gdd = Gtilde(d, z)[1:N, 1:N]
        dy = inv(G0(z)) - inv(Gdd)
        ΣS = sigma_chain(ch, z)
        err = max(err, _vk_maxabs(ΣS - dy)); ref = max(ref, _vk_maxabs(ΣS - ComplexF64.(ch.σ)))
    end
    return err / max(ref, floatmin())
end

"""    check_C4(ch) -> Float64 — (C4) bath leakage `max_n ‖r_n - δ_{n1}N̄‖ / ‖N̄‖`, `r_n = Ū_n† D v`."""
function check_C4(ch::VKChains)
    Nb = ch.Nbar
    nrm = _vk_opnorm(Nb)
    return Float64(maximum(n -> _vk_maxabs(ch.r[n] - (n == 1 ? Nb : zero(Nb))), eachindex(ch.r)) / nrm)
end

"""    leakage_profile(ch) -> Vector{Float64} — `‖r_n - δ_{n1}N̄‖ / ‖N̄‖` for every `n` (C4 plots)."""
leakage_profile(ch::VKChains) =
    [Float64(_vk_maxabs(ch.r[n] - (n == 1 ? ch.Nbar : zero(ch.Nbar))) / _vk_opnorm(ch.Nbar)) for n in eachindex(ch.r)]

"""
    check_C6(data, ch; zs) -> (B, Bd)

(C6) Gauss quadrature: `N_B²[(z - J_B)^{-1}]_{11}` against `Σ_β w^B_β/(z - b_β)`,
likewise for `B†`; relative to the max of the reference.
"""
function check_C6(d::DiscreteData, ch::VKChains; zs = vk_testpoints(d))
    function one(J, NB, w)
        err = 0.0; ref = 0.0
        for z in zs
            g = NB^2 * resolvent_col(J, z)[1, 1]
            r = sum(real(w[β]) / (z - d.b[β]) for β in eachindex(d.b) if !iszero(w[β]); init = 0.0im)
            err = max(err, abs(g - r)); ref = max(ref, abs(r))
        end
        return err / max(ref, floatmin())
    end
    return (B = one(ch.JB, ch.NB, d.wB), Bd = one(ch.JBd, ch.NBd, d.wBd))
end

"""
    matsubara_box(data; n3s = -10:9, n4s = -10:9) -> Vector{NTuple{5}}

The C8 frequency box: `(n3, n4, ω12, ω3, ω4)` with fermionic `ω3 = (2n3+1)πT`,
`ω4 = (2n4+1)πT`, `ω12 = -ω3 - ω4` and `ω12 ≠ 0` (`n3 + n4 + 1 ≠ 0`).
"""
function matsubara_box(d::DiscreteData; n3s = -10:9, n4s = -10:9)
    out = Tuple{Int,Int,Float64,Float64,Float64}[]
    for n3 in n3s, n4 in n4s
        n3 + n4 + 1 == 0 && continue
        ω3 = vk_fermionic(n3, d.T); ω4 = vk_fermionic(n4, d.T)
        push!(out, (n3, n4, -(2 * (n3 + n4 + 1)) * π * d.T, ω3, ω4))
    end
    return out
end

"""
    K_box(res; n3s = -10:9, n4s = -10:9) -> NamedTuple

`K̃^(12)` and `K^direct` on the Matsubara box (NaN at `ω12 = 0`), as matrices
indexed `[n3, n4]` (for `N = 1`; for `N > 1` arrays `[i3, i4, n3, n4]`), together
with the six permutation terms of both (without ζ), `Kt_p[p]`, `Kd_p[p]`.
"""
function K_box(res::VKResult; n3s = -10:9, n4s = -10:9)
    d = res.data; N = d.N
    caches = direct_caches(d)
    rawc = direct_caches(d; zeta = false)
    cache = VKResolventCache()
    sz = N == 1 ? (length(n3s), length(n4s)) : (N, N, length(n3s), length(n4s))
    Kt = fill(complex(NaN), sz); Kd = fill(complex(NaN), sz)
    Ktp = [fill(complex(NaN), sz) for _ in 1:6]; Kdp = [fill(complex(NaN), sz) for _ in 1:6]
    for (n3, n4, ω12, ω3, ω4) in matsubara_box(d; n3s = n3s, n4s = n4s)
        i = findfirst(==(n3), n3s); j = findfirst(==(n4), n4s)
        idx = N == 1 ? (i, j) : (:, :, i, j)
        kt, ktp = Ktilde(res.ch, res.cp, d, ω12, ω3, ω4; cache = cache, perms = true)
        for p in 1:6
            Ktp[p][idx...] = ktp[p]
            Kdp[p][idx...] = Kdirect(d, rawc, ω12, ω3, ω4; perm = p)
        end
        Kt[idx...] = kt
        Kd[idx...] = Kdirect(d, caches, ω12, ω3, ω4)
    end
    return (Kt = Kt, Kd = Kd, Kt_p = Ktp, Kd_p = Kdp, n3s = collect(n3s), n4s = collect(n4s))
end

"""
    check_C8(box; degenerate_ratio = 1e-10) -> NamedTuple

(C8) assembly over the box of [`K_box`](@ref):

  - `rel = max‖K̃ - K^direct‖ / max‖K^direct‖` (the definition of the task sheet);
  - `rel_perm = max‖K̃ - K^direct‖ / max_p max‖K_p^direct‖` (scale of the six
    permutation terms);
  - `perm = max_p max‖K̃_p - K_p^direct‖ / max_p max‖K_p^direct‖` (each
    permutation term separately, Prop. 10 holds term by term);
  - `abs`, `maxKd`, `maxKp`; `degenerate = maxKd < degenerate_ratio·maxKp`.

`value` is `rel`, except when `K^direct` cancels to zero (`degenerate`), where
`rel` is 0/0 and `value = rel_perm` (deviation documented in
`vertex_krylov_STATUS.md`: the atom's (↑↑) channel, where `B = U n_↓` commutes
with `H`, `d_↑` and `q_↑`, so `K^(12) ≡ 0` at `ω12 ≠ 0`).
"""
function check_C8(box; degenerate_ratio::Real = 1e-10)
    m = .!isnan.(real.(box.Kd))
    err = maximum(abs, box.Kt[m] - box.Kd[m])
    maxKd = maximum(abs, box.Kd[m])
    maxKp = maximum(p -> maximum(abs, box.Kd_p[p][m]), 1:6)
    perr = maximum(p -> maximum(abs, box.Kt_p[p][m] - box.Kd_p[p][m]), 1:6)
    rel = err / max(maxKd, floatmin())
    relp = err / max(maxKp, floatmin())
    deg = maxKd < degenerate_ratio * maxKp
    return (value = deg ? relp : rel, rel = rel, rel_perm = relp, perm = perr / max(maxKp, floatmin()),
            abs = err, maxKd = maxKd, maxKp = maxKp, degenerate = deg)
end

"""
    zacinskis_check(res; zs) -> NamedTuple

(C2) implementation cross-check and [VD] Prop. 11 (task sheet Sec. 5.4):
Zacinskis' `S̃ = Ṽ†Ṽ`, `S = ([S̃^{-1}]_{22})^{-1}` (checked against `N̄²`), their
start block (checked against `Ū_1`), and `Ω′ = Y†DY`, `Y = [Ū_1, R̃_ā Ū_ā]` with
`Ū_ā` the eigenvectors of `R̃_ā†DR̃_ā` (assembled from their blocks), compared
with `PDP` on `ran P` (sorted spectra, first-block resolvent). The discrete
self-energy of the diagonalized `Ω′` is compared with `Σ^S`, and block Lanczos
of the diagonalized `Ω′` from the image of `Ū_1` with the Stage-2 blocks.
"""
function zacinskis_check(res::VKResult; zs = vk_testpoints(res.data), tol::Real = 1e-12)
    ch = res.ch; cs = ch.cs; d = res.data
    N = d.N
    v = ComplexF64.(cs.v); w = ComplexF64.(cs.w); D = Float64.(cs.D)
    R = length(D)
    Vt = hcat(v, w)
    St = Vt' * Vt
    Sinv = inv(St)
    S = inv(Sinv[(N + 1):(2N), (N + 1):(2N)])
    Nb = ComplexF64.(ch.Nbar)
    S_err = _vk_maxabs(S - Nb^2) / max(_vk_maxabs(Nb^2), floatmin())
    Shalf = _vk_hsqrt(S)
    U1z = (Vt * Sinv)[:, (N + 1):(2N)] * Shalf
    U1 = ComplexF64.(ch.Ubar[:, 1:N])
    U1_err = _vk_maxabs(U1z - U1)
    # orthonormal complement of ran Ṽ
    F = qr(Vt)
    Q = F.Q * Matrix{ComplexF64}(I, R, R)
    Ra_bar = Q[:, (2N + 1):R]
    Ωbb = Hermitian(Ra_bar' * (D .* Ra_bar))
    Eb = _vk_eigen_or_empty(Ωbb)
    Ub = Eb.vectors
    Ωaa = U1' * (D .* U1)
    Vbar = Ub' * (Ra_bar' * (D .* U1))
    Ωp = [Ωaa Vbar'; Vbar Diagonal(Eb.values)]
    Ωp = Hermitian((Ωp + Ωp') / 2)
    # PDP on ran P with an independent orthonormal basis of ran P
    Fv = qr(v)
    Qv = Fv.Q * Matrix{ComplexF64}(I, R, R)
    Yp = Qv[:, (N + 1):R]
    PDP = Hermitian(Yp' * (D .* Yp))
    spec_err = maximum(abs, sort(_vk_eigen_or_empty(Ωp).values) - sort(_vk_eigen_or_empty(PDP).values); init = 0.0) /
               max(maximum(abs, D), floatmin())
    # first-block weights via resolvents
    Ez = eigen(Ωp)
    φ = Matrix(Ez.vectors[1:N, :]')                # image of Ū_1 in the eigenbasis of Ω′
    ψ = Yp' * U1                                  # Ū_1 in the basis Yp of ran P
    EP = eigen(PDP)
    ψe = EP.vectors' * ψ
    wt_err = 0.0; sig_err = 0.0; sref = 0.0
    for z in zs
        gz = φ' * (φ ./ (z .- Ez.values))
        gp = ψe' * (ψe ./ (z .- EP.values))
        wt_err = max(wt_err, _vk_maxabs(gz - gp))
        Σz = ComplexF64.(ch.σ) + Nb * gz * Nb
        Σs = sigma_chain(ch, z)
        sig_err = max(sig_err, _vk_maxabs(Σz - Σs)); sref = max(sref, _vk_maxabs(Σs - ComplexF64.(ch.σ)))
    end
    # Lanczos of the diagonalized Ω′ from the image of Ū_1 (Zacinskis Eqs. (A16)-(A25))
    plz = projected_lanczos(Ez.values, φ, nothing; tol = tol)
    Lz = length(plz.eps); L = nblocks(ch.Jbar)
    Lc = min(Lz, L)
    scale = maximum(abs, D)
    # per-block deviation profile of the Jacobi blocks (relative to max|D|)
    dev = [max(_vk_maxabs(plz.eps[n] - ComplexF64.(ch.Jbar.eps[n])),
               n < Lc ? _vk_maxabs(plz.t[n] - ComplexF64.(ch.Jbar.lower[n])) : 0.0) / scale for n in 1:Lc]
    eps_err = maximum(n -> _vk_maxabs(plz.eps[n] - ComplexF64.(ch.Jbar.eps[n])), 1:Lc; init = 0.0) / scale
    t_err = Lc > 1 ? maximum(n -> _vk_maxabs(plz.t[n] - ComplexF64.(ch.Jbar.lower[n])), 1:(Lc - 1); init = 0.0) / scale : 0.0
    bad = findfirst(>(1e-8), dev)
    return (S_err = S_err, U1_err = U1_err, spec_err = spec_err, weight_err = wt_err,
            sigma_err = sig_err / max(sref, floatmin()), eps_err = eps_err, t_err = t_err,
            block_dev = dev, n_agree = bad === nothing ? Lc : bad - 1,
            L_zac = Lz, L_chain = L, dim_ranP = R - N)
end

_vk_eigen_or_empty(A::Hermitian) = size(A, 1) == 0 ?
    (values = Float64[], vectors = zeros(ComplexF64, 0, 0)) : eigen(A)

"""
    check_C2(res; zs) -> (Float64, NamedTuple)

(C2) implementation cross-check: the largest of the relative deviations of
[`zacinskis_check`](@ref) between the two routes for the same matrix `PDP` on
`ran P` — `S - N̄²`, start block, spectra and first-block weights of `Ω′`, and the
self-energy `Σ^S`. The Jacobi blocks of the two Lanczos runs are reported
separately (`block_dev`, `n_agree`, `L_zac`, `L_chain`): Jacobi coefficients of a
discrete measure with weights spanning many decades are ill-conditioned, so two
Float64 runs from different bases agree only on the leading blocks.
"""
function check_C2(res::VKResult; zs = vk_testpoints(res.data))
    z = zacinskis_check(res; zs = zs)
    return max(z.S_err, z.U1_err, z.spec_err, z.weight_err, z.sigma_err), z
end

"""
    atom_closed_form(res) -> NamedTuple

[VD] App. F for the atom (`N = 1`): `J̄ = ε_σ + U(1-n)`, `N̄² = U²n(1-n)`, `σ = Un`,
`n = ⟨n_σ̄⟩`, chain length 1 with residual `< 1e-12`. Returns the deviations.
"""
function atom_closed_form(res::VKResult)
    d = res.data; ch = res.ch
    d.N == 1 || throw(ArgumentError("closed form for N = 1"))
    s = d.meta[:spins][1]
    n = d.meta[:occupations][_spin_other(s)]
    U = d.meta[:U]; ε = d.meta[:eps_sigma][1]
    J = ε + U * (1 - n); N2 = U^2 * n * (1 - n); σ = U * n
    pd = ch.diag[:particle]
    return (L = nblocks(ch.Jbar), resid = pd[:resid], J_err = abs(ch.Jbar.eps[1][1, 1] - J),
            N2_err = abs(ch.Nbar[1, 1]^2 - N2), σ_err = abs(ch.σ[1, 1] - σ), J = J, N2 = N2, σ = σ, n = n)
end

"""
    vk_checks(res; box = true, zac = true, G0 = nothing) -> Dict{Symbol,Any}

All checks of task sheet Sec. 5.2 for one result: `:C1`, `:C2` (with the full
[`zacinskis_check`](@ref) in `:zac`), `:C3`, `:C4`, `:C5`, `:C6` (max of B and
B†), `:C7`, `:C8` (with the box in `:box`), the sanity checks (`:sanity`) and the
chain lengths.
"""
function vk_checks(res::VKResult; box::Bool = true, zac::Bool = true, G0 = nothing)
    d = res.data; ch = res.ch
    out = Dict{Symbol,Any}()
    out[:sanity] = data_sanity(d)
    out[:C1] = check_C1(d, ch)
    if zac
        c2, z = check_C2(res)
        out[:C2] = c2; out[:zac] = z
    end
    out[:C3] = check_C3(d, ch; G0 = G0 === nothing ? get(d.meta, :G0, nothing) : G0)
    out[:C4] = check_C4(ch)
    out[:C5] = check_C5(d)
    c6 = check_C6(d, ch)
    out[:C6] = max(c6.B, c6.Bd); out[:C6_parts] = c6
    out[:C7] = res.kdiag[:C7]; out[:C7_family] = res.kdiag[:C7_family]
    if box
        bx = K_box(res)
        out[:box] = bx
        c8 = check_C8(bx)
        out[:C8] = c8.value; out[:C8_parts] = c8
    end
    out[:L] = nblocks(ch.Jbar); out[:LB] = nblocks(ch.JB); out[:LBd] = nblocks(ch.JBd)
    out[:R] = ch.cs.diag[:R]; out[:nbins] = length(d.b)
    out[:particle_resid] = ch.diag[:particle][:resid]
    out[:particle_terminated] = ch.diag[:particle][:terminated]
    return out
end

# ============================================================================
# Convention checks (task sheet Secs. 2.2, 2.5, 5.1)
# ============================================================================

"""
    psf_convention_check(sp, O1, O2) -> Float64

Task sheet Sec. 2.2: the 2p PSF of `psf(sp, (O1, O2))` against an explicit
Lehmann sum `Σ ρ_1 (O1)_{12} (O2)_{21} δ(ε - (E_2 - E_1))`, compared position by
position (aggregated at `1e-10`). Returns the largest weight difference.
"""
function psf_convention_check(sp::Spectrum, O1::AbstractMatrix, O2::AbstractMatrix)
    A = to_eigenbasis(sp, O1); Bm = to_eigenbasis(sp, O2)
    n = length(sp.E)
    ref = Dict{Float64,ComplexF64}()
    for i in 1:n, j in 1:n
        w = sp.ρ[i] * A[i, j] * Bm[j, i]
        iszero(w) && continue
        k = round(sp.E[j] - sp.E[i]; digits = 10) + 0.0
        ref[k] = get(ref, k, 0.0im) + w
    end
    got = Dict{Float64,ComplexF64}()
    for t in psf(sp, (O1, O2))
        k = round(t.position[1]; digits = 10) + 0.0
        got[k] = get(got, k, 0.0im) + t.weight
    end
    return maximum(k -> abs(get(ref, k, 0.0im) - get(got, k, 0.0im)), union(keys(ref), keys(got)); init = 0.0)
end

"""
    liouvillian_3p(sp, Os, z1, z2) -> ComplexF64

The [3p] Def. 3 / Thm. 1 resolvent function
`F(z1, z2) = (R(z̄1)O¹†| M_{O²} R(z2) O³)` (plain product), evaluated with the
Liouvillian superoperator in the eigenbasis, `(R(w)A)_{ab} = A_{ab}/(w - E_a + E_b)`,
`(A|B) = Tr[ρA†B]`. Independent of the PSF code (used to fix the frequency
convention of `regular_sum`, task sheet Sec. 5.1).
"""
function liouvillian_3p(sp::Spectrum, Os::Tuple, z1::Number, z2::Number)
    E = sp.E; n = length(E)
    O1 = ComplexF64.(to_eigenbasis(sp, Os[1])); O2 = ComplexF64.(to_eigenbasis(sp, Os[2]))
    O3 = ComplexF64.(to_eigenbasis(sp, Os[3]))
    A = Matrix(O1')                                       # O¹†
    RA = [A[a, b] / (conj(z1) - (E[a] - E[b])) for a in 1:n, b in 1:n]   # R(z̄1) O¹†
    RC = [O3[a, b] / (z2 - (E[a] - E[b])) for a in 1:n, b in 1:n]        # R(z2) O³
    return sum(sp.ρ[a] * (RA' * O2 * RC)[a, a] for a in 1:n)             # Tr[ρ (R A)† O² (R C)]
end

"""
    liouvillian_regular(sp, Os, ω) -> ComplexF64

[3p] Sec. III.3: `Σ_p ζ^p F_p[O_p](iα_{p̄1}, -iα_{p̄3})` for `Os = (q12, T3, T4)`-type
operators `(O1, O2, O3)` (first bosonic, others fermionic) at real frequencies
`ω = (ω1, ω2, ω3)`, `Σω = 0`, with the permutation table of [3p].
"""
function liouvillian_regular(sp::Spectrum, Os::Tuple, ω::AbstractVector{<:Real})
    tot = 0.0im
    for p in VK_PERMS
        ζ = permutation_sign(p, [false, true, true])
        Op = (Os[p[1]], Os[p[2]], Os[p[3]])
        tot += ζ * liouvillian_3p(sp, Op, im * ω[p[1]], -im * ω[p[3]])
    end
    return tot
end
