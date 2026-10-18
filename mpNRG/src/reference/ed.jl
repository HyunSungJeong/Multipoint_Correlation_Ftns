# Dense brute-force exact diagonalization of finite Wilson chains (task sheet
# Sec. 5.12). Small and slow by design, independent of `nrg/`: it uses only the
# Fock-space construction of `fock/` (with the block-sparse containers of
# `symmetry/` as plain data structures) and the chains of `bath/`.

"""
    EDResult

Exact eigenstates of a finite Fock space: per symmetry sector `q`, the
eigenvalues `E[q]` (ascending, absolute) and eigenvectors `U[q]` (columns, on
the Fock states `fockidx[q]`), the inverse temperature `β`, the Boltzmann
weights `ρ[q]` and the partition function `Z` (with the ground-state energy
`E0` subtracted in the exponent).
"""
struct EDResult{S,T}
    basis::FockBasis
    charges::Dict{Mode,S}
    space::SectorSpace{S}
    fockidx::Dict{S,Vector{Int}}
    E::Dict{S,Vector{Float64}}
    U::Dict{S,Matrix{T}}
    β::Float64
    E0::Float64
    Z::Float64
    ρ::Dict{S,Vector{Float64}}
end

const ED_MAXDIM = 16384

"""
    ed_reference(H, modes, charges; β, allowed = nothing, maxdim = 16384) -> EDResult

Diagonalize the operator expression `H` on the Fock space of `modes` (in the
Jordan-Wigner order of the list; for a chain use `[site N; …; site 0; impurity]`
to match the NRG convention), block by block in the sectors of the mode
`charges`. `allowed` restricts the Fock basis (bit patterns). Refuses
dimensions beyond `maxdim` (a few thousand states is the intended range).
"""
function ed_reference(H::OpExpr, ms::AbstractVector{Mode}, ch::AbstractDict{Mode,S}; β::Real,
                      allowed = nothing, maxdim::Integer = ED_MAXDIM) where {S}
    nst = allowed === nothing ? (1 << length(ms)) : length(allowed)
    nst ≤ maxdim || throw(ArgumentError("ED of $nst states exceeds maxdim = $maxdim"))
    B = FockBasis(ms; allowed = allowed)
    T = eltype(H) <: Real ? Float64 : ComplexF64
    Hd = densematrix(H, B; T = T)
    labels = [statecharge(B, i, ch) for i in 1:length(B)]
    idx = Dict{S,Vector{Int}}()
    for (i, q) in enumerate(labels)
        push!(get!(() -> Int[], idx, q), i)
    end
    # H must not couple different sectors
    for j in axes(Hd, 2), i in axes(Hd, 1)
        abs(Hd[i, j]) > 1e-12 && labels[i] != labels[j] && throw(ArgumentError("H breaks the declared charges"))
    end
    E = Dict{S,Vector{Float64}}(); U = Dict{S,Matrix{T}}()
    for (q, ii) in idx
        M = Hd[ii, ii]
        F = eigen(Hermitian((M + M') / 2))
        E[q] = F.values; U[q] = F.vectors
    end
    E0 = minimum(minimum(e) for e in values(E))
    Z = sum(sum(exp.(-β .* (e .- E0))) for e in values(E))
    ρ = Dict(q => exp.(-β .* (e .- E0)) ./ Z for (q, e) in E)
    sp = SectorSpace(Dict(q => length(ii) for (q, ii) in idx))
    return EDResult{S,T}(B, Dict{Mode,S}(ch), sp, idx, E, U, Float64(β), E0, Z, ρ)
end

"""
    ed_operator(ed, O) -> BlockOp

The homogeneous operator expression `O` in the ED eigenbasis, as a block
operator on the sectors of `ed`.
"""
function ed_operator(ed::EDResult{S,T}, O::OpExpr) where {S,T}
    Δ = opshift(O, ed.charges)
    Tt = promote_type(T, eltype(O) <: Real ? Float64 : ComplexF64)
    Od = densematrix(O, ed.basis; T = Tt)
    A = BlockOp{Tt}(ed.space, ed.space, Δ)
    for (q, ii) in ed.fockidx
        r = fuse(q, Δ)
        haskey(ed.fockidx, r) || continue
        M = ed.U[r]' * Od[ed.fockidx[r], ii] * ed.U[q]
        A.blocks[q] = M
    end
    return A
end

"""    ed_thermal_average(ed, O) — `Tr[ρ O]` for a neutral operator expression."""
function ed_thermal_average(ed::EDResult, O::OpExpr)
    A = ed_operator(ed, O)
    isneutral(A.shift) || return zero(eltype(A))
    return sum(sum(ed.ρ[q] .* diag(M)) for (q, M) in A.blocks)
end

"""
    ed_psf(ed, ops; wtol = 0) -> (positions::Matrix{Float64}, weights::Vector)

Exact Lehmann PSF, Kugler Eq. (28) / Lee Eq. (5), for operators given **in
permuted order** `(O¹, …, O^ℓ)`: one term per eigenstate cycle `(1, …, ℓ)`
with weight `ρ_1 ∏_i (O^i)_{i,i+1}` (cyclic) at positions
`ε_i = E_{i+1} - E_1`, `i = 1..ℓ-1`. Brute-force enumeration over nonzero
blocks; terms with `|w| ≤ wtol` are dropped.
"""
function ed_psf(ed::EDResult{S,T}, ops::Tuple; wtol::Real = 0.0) where {S,T}
    ℓ = length(ops)
    Bs = [ed_operator(ed, O) for O in ops]
    tot = foldl(fuse, (B.shift for B in Bs); init = neutral(S))
    Tw = promote_type(map(eltype, Bs)...)
    pos = Float64[]; ws = Tw[]
    isneutral(tot) || return reshape(pos, ℓ - 1, 0), ws
    for q1 in sectors(ed.space)
        # follow the shifts: state i+1 lies in sector q_{i+1} = q_i ⊖ Δ_i
        qs = [q1]
        ok = true
        for i in 1:ℓ
            qn = fuse(qs[end], dual(Bs[i].shift))
            haskey(ed.E, qn) || (ok = false; break)
            push!(qs, qn)
        end
        ok || continue
        Ms = [getblock(Bs[i], qs[i + 1]) for i in 1:ℓ]   # rows: q_i, cols: q_{i+1}
        E1 = ed.E[q1]
        for s1 in eachindex(E1)
            ρ1 = ed.ρ[q1][s1]
            iszero(ρ1) && continue
            _ed_rec!(pos, ws, Ms, ed, qs, 1, s1, s1, Tw(ρ1), Float64[], E1[s1], ℓ, wtol)
        end
    end
    return reshape(pos, ℓ - 1, :), ws
end

function _ed_rec!(pos, ws, Ms, ed, qs, i, s1, si, w, p, E1, ℓ, wtol)
    if i == ℓ
        wt = w * Ms[ℓ][si, s1]
        if abs(wt) > wtol && !iszero(wt)
            append!(pos, p); push!(ws, wt)
        end
        return
    end
    Mi = Ms[i]; En = ed.E[qs[i + 1]]
    for sn in axes(Mi, 2)
        m = Mi[si, sn]
        iszero(m) && continue
        _ed_rec!(pos, ws, Ms, ed, qs, i + 1, s1, sn, w * m, vcat(p, En[sn] - E1), E1, ℓ, wtol)
    end
end

"""
    chain_modes(site_modes, N) -> Vector{Mode}

The Jordan-Wigner mode list `[site N; …; site 0; impurity]` for a chain, from a
function `site_modes(n)` returning the modes of site `n` (n = -1 the impurity).
"""
chain_modes(site_modes, N::Integer) = reduce(vcat, [site_modes(n) for n in N:-1:-1])

"""
    chain_allowed(imp_states, nimp, M) -> Vector{UInt64}

Allowed bit patterns of a chain of `M` modes whose last `nimp` modes are the
impurity, restricted to the impurity patterns `imp_states` (e.g. the singly
occupied Kondo impurity); `nothing` if the impurity is unrestricted.
"""
function chain_allowed(imp_states, nimp::Integer, M::Integer)
    length(imp_states) == (1 << nimp) && return nothing
    return sort!(UInt64[(UInt64(e) << nimp) | UInt64(a) for e in 0:((1 << (M - nimp)) - 1) for a in imp_states])
end
