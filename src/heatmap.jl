# Vertex heat maps of the Hubbard atom in the MF and KF: Figs. 10 and 11.
#
# Fast, allocation-free point evaluation of the 4p vertex on 2D frequency grids.
# Nothing here is new physics: the kernels are Eqs. (46), (52)/(63) and the
# assembly Eq. (67a), exactly as in kernel.jl / keldysh.jl, which remain the
# reference implementation the tests compare against. What is new:
#
#   H1  a point function `vertex_point!` and grid drivers, with the PSFs stored
#       as concrete vectors (`PSF4`, `PSF2`) and every per-point quantity kept
#       in tuples / SMatrix, so that one point allocates nothing;
#   H2  the disconnected part removed at the PSF level, Eq. (30)/(31), with
#       coincident peaks merged before subtracting;
#   H3  Keldysh amputation, Eq. (76), with the external legs built from
#       Eq. (D2) — the imaginary shifts 3γ₀ / γ₀ of Appendix D.
#
# Equations are read from the printed page of Kugler, Lee & von Delft,
# PRX 11, 041006 (2021): Eq. (76) p. 15, the bare vertex p. 17, Eqs. (D1),
# (D2) p. 28.

"""Tolerance for degeneracies, coincident peaks and Kronecker deltas."""
const TOL_DEG = 1e-10

"""Merged PSF peaks with `|weight| ≤ TOL_WEIGHT` are dropped after subtraction."""
const TOL_WEIGHT = 1e-14

# ---------------------------------------------------------------------------
# Concrete PSF storage
# ---------------------------------------------------------------------------

"""
    PSF4

The PSF of one permutation of a 4p operator tuple, in concrete storage: the
permutation `p`, its sign `ζ`, the cumulative peak positions
`(ω'_1̄, ω'_1̄2̄, ω'_1̄2̄3̄)` and the weights.
"""
struct PSF4
    p::NTuple{4,Int}
    ζ::Int
    pos::Vector{NTuple{3,Float64}}
    w::Vector{ComplexF64}
end

"""    PSF2 — as [`PSF4`](@ref) for a 2p tuple: one cumulative position per peak."""
struct PSF2
    p::NTuple{2,Int}
    ζ::Int
    pos::Vector{Float64}
    w::Vector{ComplexF64}
end

"""
    merge_peaks(terms; tol = TOL_DEG, wtol = TOL_WEIGHT) -> (positions, weights)

Merge PSF peaks whose positions agree to `tol` in every coordinate, summing
their weights, then drop merged peaks with `|weight| ≤ wtol`. Applied to the
list `S` followed by `-S^dis`, this performs the subtraction `S^con = S - S^dis`
peak by peak, so cancelling peaks disappear before any kernel sees them. A
plain pairwise search: the atom has at most a few hundred peaks per permutation.
"""
function merge_peaks(terms::Vector{PSFTerm}; tol::Real = TOL_DEG, wtol::Real = TOL_WEIGHT)
    pos = Vector{Vector{Float64}}()
    wts = ComplexF64[]
    for t in terms
        k = findfirst(q -> all(abs.(q .- t.position) .<= tol), pos)
        if k === nothing
            push!(pos, copy(t.position))
            push!(wts, t.weight)
        else
            wts[k] += t.weight
        end
    end
    keep = findall(w -> abs(w) > wtol, wts)
    return pos[keep], wts[keep]
end

"""
    psf4_tables(sp, Os; part = :connected, is_fermionic, merge = true) -> Vector{PSF4}

The 24 permuted PSFs of a 4p tuple `Os`, restricted to `part` (Eq. 31) and, by
default, merged with [`merge_peaks`](@ref).
"""
function psf4_tables(sp::Spectrum, Os::Tuple; part::Symbol = :connected,
                     is_fermionic::AbstractVector{Bool} = fill(true, 4), merge::Bool = true)
    length(Os) == 4 || throw(ArgumentError("PSF4 is for ℓ = 4"))
    out = PSF4[]
    for d in permuted_psfs(sp, Os; is_fermionic = is_fermionic, part = part)
        pos, w = merge ? merge_peaks(d.terms) :
                 ([t.position for t in d.terms], [t.weight for t in d.terms])
        push!(out, PSF4(Tuple(d.p), d.ζ, [Tuple(x) for x in pos], w))
    end
    return out
end

"""    psf2_tables(sp, A, B; is_fermionic) -> Vector{PSF2} — the two permuted PSFs of `(A, B)`."""
function psf2_tables(sp::Spectrum, A::AbstractMatrix, B::AbstractMatrix;
                     is_fermionic::AbstractVector{Bool} = [true, true])
    out = PSF2[]
    for d in permuted_psfs(sp, (A, B); is_fermionic = is_fermionic)
        pos, w = merge_peaks(d.terms)
        push!(out, PSF2(Tuple(d.p), d.ζ, [x[1] for x in pos], w))
    end
    return out
end

# ---------------------------------------------------------------------------
# Keldysh component bookkeeping
# ---------------------------------------------------------------------------

"""
Keldysh components are stored as a 2×2×2×2 array `G[k₁,k₂,k₃,k₄]`, `kᵢ ∈ {1,2}`;
`_KIDX[c]` is the digit tuple of the column-major linear index `c ∈ 1:16`.
"""
const _KIDX = Tuple(Tuple(I) for I in CartesianIndices((2, 2, 2, 2)))

"""
    keldysh_coefficients(p) -> SMatrix{16,4,Float64}

The coefficients of Eq. (63) for permutation `p`: component `c` of the
correlator kernel is `Σ_λ coef[c, λ] K^[λ](ω_p)`, with `K^[λ]` the fully
retarded kernel of Eq. (52) and `λ` a slot of the permuted tuple. With the
permuted digits `k_ī = k_{p(i)}`,

    coef[c, λ] = (-1)^{λ-1} (-1)^{k_1̄ + ⋯ + k_(λ-1)̄}   if k_λ̄ = 2,   else 0 .

This is `keldysh_kernel_eq63`, tabulated; the tests check it against
[`keldysh_kernel`](@ref) (Eq. 67b).
"""
function keldysh_coefficients(p::NTuple{4,Int})
    C = zeros(16, 4)
    for c in 1:16
        k = _KIDX[c]
        ksum = 0
        for λ in 1:4
            kλ = k[p[λ]]
            if kλ == 2
                C[c, λ] = (-1)^(λ - 1) * (-1)^ksum
            end
            ksum += kλ
        end
    end
    return SMatrix{16,4,Float64}(C)
end

"""    KFPerm — a [`PSF4`](@ref) together with its Eq. (63) coefficient table."""
struct KFPerm
    psf::PSF4
    coef::SMatrix{16,4,Float64,64}
end

# ---------------------------------------------------------------------------
# Parameters and tables
# ---------------------------------------------------------------------------

"""
    HeatmapParams(U, β, γ0)

Physical parameters of the heat maps: `U`, `β = 1/T`, and the Keldysh
broadening `γ₀` (unused in the MF).
"""
struct HeatmapParams
    U::Float64
    β::Float64
    γ0::Float64
end

"""
    MFTables, KFTables

Everything the point functions need for one spin configuration `σσ′`: the
connected PSFs of `(d_σ, d†_σ, d_σ′, d†_σ′)` (per permutation) and the 2p PSFs
of `(d_σ, d†_σ)` for the external legs. Build with [`heatmap_tables`](@ref).
"""
struct MFTables
    con::Vector{PSF4}
    leg::Vector{PSF2}
    updn::Bool
end

struct KFTables
    con::Vector{KFPerm}
    leg::Vector{PSF2}
    updn::Bool
end

"""
    heatmap_tables(m::HubbardAtomModel, σσ′, formalism; part = :connected)

Tables for `σσ′ ∈ (:updn, :upup)` and `formalism ∈ (:MF, :KF)`. `part` selects
the PSF part fed to the 4p kernel (`:connected` for the vertex; `:full` and
`:disconnected` for the tests).
"""
function heatmap_tables(m::HubbardAtomModel, σσ′::Symbol, formalism::Symbol;
                        part::Symbol = :connected, sp::Spectrum = spectrum(m))
    ops = operators(m)
    σ, σ′ = σσ′ === :updn ? (:up, :dn) : σσ′ === :upup ? (:up, :up) :
        throw(ArgumentError("σσ′ must be :updn or :upup"))
    dσ, dagσ = spin_operators(ops, σ)
    dσ′, dagσ′ = spin_operators(ops, σ′)
    con = psf4_tables(sp, (dσ, dagσ, dσ′, dagσ′); part = part)
    leg = psf2_tables(sp, dσ, dagσ)
    formalism === :MF && return MFTables(con, leg, σσ′ === :updn)
    formalism === :KF && return KFTables([KFPerm(P, keldysh_coefficients(P.p)) for P in con],
                                         leg, σσ′ === :updn)
    throw(ArgumentError("formalism must be :MF or :KF"))
end

# ---------------------------------------------------------------------------
# MF point evaluation
# ---------------------------------------------------------------------------

"""
    mf_correlator_point(ms, β, perms) -> ComplexF64

`G(iω)` of Eq. (39) with the kernel of Eq. (46), for integer Matsubara indices
`ms` (`ω = mπ/β`), summed over the given [`PSF4`](@ref) tables. The same
two-branch logic as [`matsubara_kernel`](@ref), without allocating: the
Matsubara part of a composite vanishes iff its integer partial sum is 0.
"""
function mf_correlator_point(ms::NTuple{4,Int}, β::Float64, perms::Vector{PSF4})
    G = zero(ComplexF64)
    s = π / β
    @inbounds for P in perms
        p = P.p
        m1 = ms[p[1]]; m2 = m1 + ms[p[2]]; m3 = m2 + ms[p[3]]
        acc = zero(ComplexF64)
        for t in eachindex(P.w)
            x = P.pos[t]
            Ω1 = complex(-x[1], m1 * s); Ω2 = complex(-x[2], m2 * s); Ω3 = complex(-x[3], m3 * s)
            v1 = m1 == 0 && abs(x[1]) <= TOL_DEG
            v2 = m2 == 0 && abs(x[2]) <= TOL_DEG
            v3 = m3 == 0 && abs(x[3]) <= TOL_DEG
            nv = v1 + v2 + v3
            K = if nv == 0
                1 / (Ω1 * Ω2 * Ω3)
            elseif nv == 1
                # anomalous branch of Eq. (46): drop the vanishing factor
                a, b = v1 ? (Ω2, Ω3) : v2 ? (Ω1, Ω3) : (Ω1, Ω2)
                -0.5 * (β + 1 / a + 1 / b) / (a * b)
            else
                throw(ArgumentError("more than one vanishing composite: outside Eq. (46)"))
            end
            acc += K * P.w[t]
        end
        G += P.ζ * acc
    end
    return G
end

"""    mf_leg(m, β, legs) — `G(iω)` for integer `m`, from the 2p PSF tables."""
function mf_leg(m::Int, β::Float64, legs::Vector{PSF2})
    G = zero(ComplexF64)
    @inbounds for P in legs
        z = P.p[1] == 1 ? m : -m                # partial sum ω_1̄ in permuted order
        acc = zero(ComplexF64)
        for t in eachindex(P.w)
            acc += P.w[t] / complex(-P.pos[t], z * π / β)
        end
        G += P.ζ * acc
    end
    return G
end

# ---------------------------------------------------------------------------
# KF point evaluation
# ---------------------------------------------------------------------------

@inline _zeta(ω::NTuple{4,Float64}, η::Int, γ0::Float64) =
    ntuple(i -> i == η ? complex(ω[i], 3γ0) : complex(ω[i], -γ0), 4)   # Eq. (49), ℓ = 4

@inline function _partials(z::NTuple{4,ComplexF64}, p::NTuple{4,Int})
    a = z[p[1]]; b = a + z[p[2]]
    return (a, b, b + z[p[3]])
end

"""
    kf_correlator_point!(G, ω, γ0, perms)

All 16 Keldysh components of `G(ω)`, Eq. (67a), written into `G` (length 16,
column-major over `k₁k₂k₃k₄`). For each permutation and PSF peak, the four
fully retarded kernels `K^[λ]` of Eq. (52) are formed once and distributed over
the components with the Eq. (63) coefficient table.
"""
function kf_correlator_point!(G::AbstractVector{ComplexF64}, ω::NTuple{4,Float64},
                              γ0::Float64, perms::Vector{KFPerm})
    Z = (_zeta(ω, 1, γ0), _zeta(ω, 2, γ0), _zeta(ω, 3, γ0), _zeta(ω, 4, γ0))
    @inbounds for c in 1:16
        G[c] = 0
    end
    @inbounds for P in perms
        p = P.psf.p
        # partial sums of ω^[η] in permuted order, for the external label η = p(λ)
        S1 = _partials(Z[p[1]], p); S2 = _partials(Z[p[2]], p)
        S3 = _partials(Z[p[3]], p); S4 = _partials(Z[p[4]], p)
        a1 = a2 = a3 = a4 = zero(ComplexF64)
        pos = P.psf.pos; w = P.psf.w
        for t in eachindex(w)
            x = pos[t]; wt = w[t]
            a1 += wt / ((S1[1] - x[1]) * (S1[2] - x[2]) * (S1[3] - x[3]))
            a2 += wt / ((S2[1] - x[1]) * (S2[2] - x[2]) * (S2[3] - x[3]))
            a3 += wt / ((S3[1] - x[1]) * (S3[2] - x[2]) * (S3[3] - x[3]))
            a4 += wt / ((S4[1] - x[1]) * (S4[2] - x[2]) * (S4[3] - x[3]))
        end
        C = P.coef
        ζ = P.psf.ζ
        for c in 1:16
            G[c] += ζ * (C[c, 1] * a1 + C[c, 2] * a2 + C[c, 3] * a3 + C[c, 4] * a4)
        end
    end
    @inbounds for c in 1:16
        G[c] *= 0.5                                   # 2^{1-ℓ/2}, ℓ = 4
    end
    return G
end

"""
    leg(ν, a, b, legs, γ0) -> SMatrix{2,2,ComplexF64}

The external-leg propagator of Eq. (75), `[0 G^A; G^R G^K]`, at real frequency
`ν`, with the imaginary shifts of Eq. (D1):

    ω^[1] = (ν + aiγ₀, -ν - aiγ₀) ,    ω^[2] = (ν - biγ₀, -ν + biγ₀) ,

so `G^R` carries `ν + aiγ₀`, `G^A` carries `ν - biγ₀`, and `G^K` is the
difference of the two Lehmann terms with the same pair of shifts, weighted by
`ρ₁ + ζρ₂` — Eq. (D2). For the 4p legs of Eq. (76): `G(ωᵢ)`, `i = 1, 3`, takes
`(a, b) = (3, 1)`; `G(-ωᵢ)`, `i = 2, 4`, takes `(1, 3)`. With `a = b = 1` it is
the ordinary ℓ = 2 Keldysh output.
"""
function leg(ν::Float64, a::Real, b::Real, legs::Vector{PSF2}, γ0::Float64)
    z1 = (complex(ν, a * γ0), complex(-ν, -a * γ0))     # ω^[1]
    z2 = (complex(ν, -b * γ0), complex(-ν, b * γ0))     # ω^[2]
    GR = GA = GK = zero(ComplexF64)
    @inbounds for P in legs
        s = P.p[1]                                       # slot of the permuted first operator
        r1 = r2 = zero(ComplexF64)
        for t in eachindex(P.w)
            r1 += P.w[t] / (z1[s] - P.pos[t])
            r2 += P.w[t] / (z2[s] - P.pos[t])
        end
        GR += P.ζ * r1
        GA += P.ζ * r2
        # G^[12], Eq. (67b): ηbarhat = (p(1), p(2)), signs (+, -)
        GK += P.ζ * (s == 1 ? r1 - r2 : r2 - r1)
    end
    return SMatrix{2,2,ComplexF64}(0, GR, GA, GK)       # column-major: [0 GA; GR GK]
end

"""
    amputate!(F, G, A1, B2, A3, B4)

Eq. (2) of the task sheet — Eq. (76) inverted leg by leg:

    F^{k₁′k₂′k₃′k₄′} = Σ_k (A₁)_{k₁′k₁} (B₂)_{k₂′k₂} (A₃)_{k₃′k₃} (B₄)_{k₄′k₄} G^{con; k₁k₂k₃k₄} ,

with `Aᵢ = G(ωᵢ)⁻¹` and `Bᵢ = (G(-ωᵢ)ᵀ)⁻¹`. Done as four successive single-leg
contractions over 2×2×2×2 static arrays; nothing allocates. `G` is copied into a
static array before `F` is written, so `F` and `G` may be the same vector.
"""
function amputate!(F::AbstractVector{ComplexF64}, G::AbstractVector{ComplexF64},
                   A1::SMatrix{2,2,ComplexF64}, B2::SMatrix{2,2,ComplexF64},
                   A3::SMatrix{2,2,ComplexF64}, B4::SMatrix{2,2,ComplexF64})
    T0 = SArray{Tuple{2,2,2,2},ComplexF64,4,16}(ntuple(c -> G[c], Val(16)))
    T1 = SArray{Tuple{2,2,2,2}}(ntuple(Val(16)) do c
        i, j, k, l = _KIDX[c]
        A1[i, 1] * T0[1, j, k, l] + A1[i, 2] * T0[2, j, k, l]
    end)
    T2 = SArray{Tuple{2,2,2,2}}(ntuple(Val(16)) do c
        i, j, k, l = _KIDX[c]
        B2[j, 1] * T1[i, 1, k, l] + B2[j, 2] * T1[i, 2, k, l]
    end)
    T3 = SArray{Tuple{2,2,2,2}}(ntuple(Val(16)) do c
        i, j, k, l = _KIDX[c]
        A3[k, 1] * T2[i, j, 1, l] + A3[k, 2] * T2[i, j, 2, l]
    end)
    @inbounds for c in 1:16
        i, j, k, l = _KIDX[c]
        F[c] = B4[l, 1] * T3[i, j, k, 1] + B4[l, 2] * T3[i, j, k, 2]
    end
    return F
end

"""
    ph_frequencies(ν, ν′, ω) -> NTuple{4}

The particle-hole parametrisation of Eq. (72): `(ν, -ν-ω, ν′+ω, -ν′)`.
"""
ph_frequencies(ν, νp, ω) = (ν, -ν - ω, νp + ω, -νp)

"""
    vertex_point!(out, ν, ν′, ω, params, tables)

The vertex at one point of the particle-hole plane, Eq. (72).

  - `tables::KFTables`: `ν, ν′, ω` real; writes all 16 Keldysh components of
    `F^{k}` (vertex indices) into `out[1:16]`, column-major over `k₁k₂k₃k₄`.
    `G^con` from the connected PSFs (H2), legs from Eq. (D2) (H3), amputation
    by Eq. (76).
  - `tables::MFTables`: `ν, ν′, ω` are integer Matsubara indices (`ω = mπ/β`,
    odd for `ν, ν′`, even for `ω`); writes `F` into `out[1]`, amputated by
    Eq. (74).

Allocation-free after compilation.
"""
function vertex_point!(out::AbstractVector{ComplexF64}, ν::Float64, νp::Float64, ω::Float64,
                       prm::HeatmapParams, tab::KFTables)
    w = ph_frequencies(ν, νp, ω)
    kf_correlator_point!(out, w, prm.γ0, tab.con)            # G^con into out ...
    A1 = inv(leg(w[1], 3, 1, tab.leg, prm.γ0))
    A3 = inv(leg(w[3], 3, 1, tab.leg, prm.γ0))
    B2 = inv(transpose(leg(-w[2], 1, 3, tab.leg, prm.γ0)))
    B4 = inv(transpose(leg(-w[4], 1, 3, tab.leg, prm.γ0)))
    amputate!(out, out, A1, B2, A3, B4)                      # ... then F over it
    return out
end

function vertex_point!(out::AbstractVector{ComplexF64}, ν::Int, νp::Int, ω::Int,
                       prm::HeatmapParams, tab::MFTables)
    ms = ph_frequencies(ν, νp, ω)
    Gc = mf_correlator_point(ms, prm.β, tab.con)
    legs = mf_leg(ms[1], prm.β, tab.leg) * mf_leg(-ms[2], prm.β, tab.leg) *
           mf_leg(ms[3], prm.β, tab.leg) * mf_leg(-ms[4], prm.β, tab.leg)
    out[1] = Gc / legs
    return out
end

"""
    kf_correlator_grid(νs, ν′s, ω, params, tables) -> Array{ComplexF64,6}

`G^con` (or whichever PSF part the tables hold) on the grid, stored as
`G[k₁,k₂,k₃,k₄, i_ν, j_ν′]`.
"""
function kf_correlator_grid(νs::AbstractVector{Float64}, νps::AbstractVector{Float64}, ω::Float64,
                            prm::HeatmapParams, tab::KFTables)
    out = Array{ComplexF64}(undef, 2, 2, 2, 2, length(νs), length(νps))
    buf = zeros(ComplexF64, 16)
    for j in eachindex(νps), i in eachindex(νs)
        kf_correlator_point!(buf, ph_frequencies(νs[i], νps[j], ω), prm.γ0, tab.con)
        off = 16 * ((i - 1) + length(νs) * (j - 1))        # the 16 components are contiguous
        @inbounds for c in 1:16
            out[off + c] = buf[c]
        end
    end
    return out
end

"""
    vertex_grid(νs, ν′s, ω, params, tables)

The vertex on a grid with the first frequency index innermost.

  - KF: `F[k₁,k₂,k₃,k₄, i_ν, j_ν′]`, an `Array{ComplexF64,6}` of size
    `(2,2,2,2,Nν,Nν′)`, so the 16 components at one point are contiguous and
    `F[1,2,2,2,:,:]` is `F^[1]`.
  - MF: `F[i_ν, j_ν′]`, integer Matsubara indices.

`heatmap(νs, ν′s, F[..., :, :])` then puts `ν` on the horizontal axis.
Uses `Threads.@threads` over `ν′` when Julia runs with more than one thread.
"""
function vertex_grid(νs::AbstractVector{Float64}, νps::AbstractVector{Float64}, ω::Float64,
                     prm::HeatmapParams, tab::KFTables)
    F = Array{ComplexF64}(undef, 2, 2, 2, 2, length(νs), length(νps))
    Threads.@threads for j in eachindex(νps)
        buf = zeros(ComplexF64, 16)
        for i in eachindex(νs)
            vertex_point!(buf, νs[i], νps[j], ω, prm, tab)
            off = 16 * ((i - 1) + length(νs) * (j - 1))    # F[:,:,:,:,i,j] is contiguous
            @inbounds for c in 1:16
                F[off + c] = buf[c]
            end
        end
    end
    return F
end

function vertex_grid(νs::AbstractVector{Int}, νps::AbstractVector{Int}, ω::Int,
                     prm::HeatmapParams, tab::MFTables)
    F = Array{ComplexF64}(undef, length(νs), length(νps))
    Threads.@threads for j in eachindex(νps)
        buf = zeros(ComplexF64, 1)
        for i in eachindex(νs)
            vertex_point!(buf, νs[i], νps[j], ω, prm, tab)
            F[i, j] = buf[1]
        end
    end
    return F
end

# ---------------------------------------------------------------------------
# Bare vertex and analytic references
# ---------------------------------------------------------------------------

"""
    bare_vertex_kf(k, U, updn) -> Float64

`F₀^{k₁k₂k₃k₄} = ½U δ_{σ,σ̄′}` if `k₁₂₃₄ = k₁+k₂+k₃+k₄` is odd, else 0 (p. 17:
"a fourfold Keldysh rotation then generates the prefactor [1-(-1)^{k₁…₄}]/4").
"""
bare_vertex_kf(k::NTuple{4,Int}, U::Real, updn::Bool) = (updn && isodd(sum(k))) ? U / 2 : 0.0

"""    bare_vertex_mf(U, updn) — `F₀ = U δ_{σ,σ̄′}` (p. 16, below Eq. 84)."""
bare_vertex_mf(U::Real, updn::Bool) = updn ? float(U) : 0.0

"""
    fret_updn(ω, η, U, γ0) -> ComplexF64

The fully retarded vertex continued from the first line of Eq. (85a), Eq. (3)
of the task sheet (from "2F^[η](ω) = F(iω)|_{iω→ω^[η]}", p. 16):

    F^[η]_{↑↓}(ω) = ½ [2u + u³ Σᵢ zᵢ² / ∏ᵢ zᵢ - 6u⁵ / ∏ᵢ zᵢ] ,   u = U/2 ,

with `z_η = ω_η + 3iγ₀` and `zᵢ = ωᵢ - iγ₀` otherwise. `F^[η]_{↑↑} = 0`.
The anomalous lines of Eq. (85) are not continued (p. 17).
"""
function fret_updn(ω::NTuple{4,Float64}, η::Int, U::Real, γ0::Real)
    u = U / 2
    z = _zeta(ω, η, float(γ0))
    P = prod(z)
    return 0.5 * (2u + u^3 * sum(x -> x^2, z) / P - 6u^5 / P)
end

"""
    symlog(x, linthresh) -> Float64

Symmetric-log map of `x` onto `[-1, 1]` for colour scales spanning `±10⁰` down
to `±linthresh`: logarithmic in `|x|` for `|x| ≥ linthresh`, linear through
zero below it, with each decade and the linear band taking equal width. `|x| ≥ 1`
maps to `±1`. Colourbar ticks go at `symlog.(ticks, linthresh)`.

(The task sheet refers to a `symlog` in `ref_plots.jl`, which was not supplied;
this is the standard definition that reproduces its reference colour bars.)
"""
function symlog(x::Real, lt::Real)
    n = log10(1 / lt)                          # decades between lt and 1
    ax = abs(x)
    y = ax <= lt ? ax / lt : 1 + log10(ax / lt)
    return sign(x) * min(y, n + 1) / (n + 1)
end
