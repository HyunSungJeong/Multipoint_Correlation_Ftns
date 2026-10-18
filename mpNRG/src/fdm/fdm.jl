# Full density matrix (task sheet Sec. 5.6; Weichselbaum & von Delft, PRL 99,
# 076402 (2007); Lee Sec. IV D-E).

"""
    FDM

The full density matrix of a run at temperature `temp` (Lee Eqs. (40)-(45)):

  - `logZ`: `ln Z` with `Z = Σ_n d^{N-n} Z_n^D` (energies measured from `Eref`,
    the lowest absolute energy of all discarded states);
  - `w[n+2] = w_n = d^{N-n} Z_n^D / Z`, the shell weights of Weichselbaum &
    von Delft (2007) Eq. (8) / Lee Eq. (44); `Σ_n w_n = 1`;
  - `rhoD[n+2][q]`: the diagonal of `ρ̄^n_{DD}`, Lee Eq. (44),
    `w_n e^{-βE^n_s}/Z_n^D`;
  - `rhoK[n+2][q]`: `ρ̄^n_{KK}`, from the backward sweep Lee Eq. (45).

`d^{N-n}` is the dimension of the environment (sites `> n`), counted through
[`degeneracy`](@ref). Approximation: the NRG approximation Lee Eq. (19)
(discarded states of shell n carry the shell-n energies, Eq. (41)).
"""
struct FDM{S<:Sector,T<:Number}
    temp::Float64
    β::Float64
    Eref::Float64
    logZ::Float64
    w::Vector{Float64}
    logdenv::Vector{Float64}
    rhoD::Vector{Dict{S,Vector{Float64}}}
    rhoK::Vector{Dict{S,Matrix{T}}}
    info::Dict{Symbol,Any}
end

"""    weight(f, n) — the shell weight `w_n`."""
weight(f::FDM, n::Integer) = f.w[n + 2]
rhoD(f::FDM, n::Integer) = f.rhoD[n + 2]
rhoK(f::FDM, n::Integer) = f.rhoK[n + 2]

_localdim(L::LocalSpace) = sum(degeneracy(q) * length(v) for (q, v) in L.sectors)

"""
    fdm(run; T) -> FDM

Shell weights, reduced density matrices `ρ̄^n_{DD}` (Lee Eq. (44)) and, by a
backward sweep from `N` down to `-1`, `ρ̄^n_{KK}` (Lee Eq. (45)):

    ρ̄^n_{KK} = Σ_X M^{σ_{n+1}}_{KX} ρ̄^{n+1}_{XX} M^{σ_{n+1} †}_{XK} ,

the partial trace over site `n+1` of the shell-(n+1) reduced density matrix,
done per pair of the fusion map.
"""
function fdm(run::NRGRun{S,Tm}; T::Real) where {S,Tm}
    Tt = Float64(T)
    β = 1 / Tt
    N = run.params.N
    shells = run.shells
    # environment dimensions d^{N-n}, in logs
    logdenv = zeros(N + 2)
    for n in (N - 1):-1:-1
        logdenv[n + 2] = logdenv[n + 3] + log(_localdim(shell(run, n + 1).loc))
    end
    Eref = minimum(minimum(sh.E[q][drange(sh, q)] .+ sh.shift; init = Inf)
                   for sh in shells for q in sectors(sh.space) if ndisc(sh, q) > 0)
    # log Z_n^D
    logZD = fill(-Inf, N + 2)
    for sh in shells
        terms = Float64[]
        for q in sectors(sh.space)
            ndisc(sh, q) > 0 || continue
            for e in sh.E[q][drange(sh, q)]
                push!(terms, -β * (e + sh.shift - Eref) + log(degeneracy(q)))
            end
        end
        isempty(terms) || (logZD[sh.n + 2] = _logsumexp(terms))
    end
    logZ = _logsumexp(logdenv .+ logZD)
    w = exp.(logdenv .+ logZD .- logZ)
    rD = [Dict{S,Vector{Float64}}() for _ in shells]
    for sh in shells, q in sectors(sh.space)
        ndisc(sh, q) > 0 || continue
        rD[sh.n + 2][q] = exp.(logdenv[sh.n + 2] .- β .* (sh.E[q][drange(sh, q)] .+ sh.shift .- Eref) .- logZ)
    end
    # backward sweep for ρ̄_KK
    rK = [Dict{S,Matrix{Tm}}() for _ in shells]
    for n in (N - 1):-1:-1
        sh1 = shell(run, n + 1); sh = shell(run, n)
        Fu = sh1.fusion
        acc = Dict{S,Matrix{Tm}}(q => zeros(Tm, nkept(sh, q), nkept(sh, q)) for q in sectors(sh.space) if nkept(sh, q) > 0)
        for (qq, prs) in Fu.pairs
            A = sh1.A[qq]
            kq = nkept(sh1, qq)
            ρfull = zeros(Tm, size(A, 2), size(A, 2))
            kq > 0 && (ρfull[1:kq, 1:kq] .= rK[n + 3][qq])
            if ndisc(sh1, qq) > 0
                d = rD[n + 3][qq]
                for (i, j) in enumerate(drange(sh1, qq))
                    ρfull[j, j] = d[i]
                end
            end
            for pr in prs, iσ in 1:pr.dσ
                rows = pr.offset + (iσ - 1) * pr.dq .+ (1:pr.dq)
                Ar = view(A, rows, :)
                acc[pr.q] .+= Ar * ρfull * Ar'
            end
        end
        rK[n + 2] = acc
    end
    f = FDM{S,Tm}(Tt, β, Eref, logZ, w, logdenv, rD, rK, Dict{Symbol,Any}())
    f.info[:checks] = fdm_checks(run, f)
    return f
end

function _logsumexp(x::AbstractVector{<:Real})
    m = maximum(x)
    isfinite(m) || return m
    return m + log(sum(exp.(x .- m)))
end

"""
    fdm_checks(run, f) -> NamedTuple

The checks of Sec. 5.6: `Σ_n w_n = 1` (equivalently `Tr ϱ = 1`), the smallest
eigenvalue of every `ρ̄^n` (must be ≥ 0 up to rounding), and the consistency
`tr ρ̄^n_{KK} = Σ_{n̄>n} w_n̄` of the backward sweep.
"""
function fdm_checks(run::NRGRun, f::FDM)
    N = run.params.N
    trρ = sum(f.w)
    λmin = Inf; kkdev = 0.0
    for n in -1:N
        sh = shell(run, n)
        for (q, M) in rhoK(f, n)
            isempty(M) && continue
            λmin = min(λmin, eigmin(Hermitian((M + M') / 2)))
        end
        for (q, d) in rhoD(f, n)
            λmin = min(λmin, minimum(d))
        end
        trK = sum((degeneracy(q) * real(tr(M)) for (q, M) in rhoK(f, n)); init = 0.0)
        later = sum(f.w[(n + 3):end]; init = 0.0)
        kkdev = max(kkdev, abs(trK - later))
    end
    return (trace = trρ, min_eigenvalue = λmin, kk_trace_deviation = kkdev)
end

"""
    rho_full(f, run, n, q) -> Matrix

The reduced density matrix `ρ̄^n` of sector `q` on all states of the shell:
`blockdiag(ρ̄^n_{KK}, ρ̄^n_{DD})` (it is sector diagonal, Lee Eq. (42)).
"""
function rho_full(f::FDM{S,T}, run::NRGRun, n::Integer, q) where {S,T}
    sh = shell(run, n)
    d = dim(sh.space, q)
    M = zeros(T, d, d)
    k = nkept(sh, q)
    k > 0 && haskey(rhoK(f, n), q) && (M[1:k, 1:k] .= rhoK(f, n)[q])
    if ndisc(sh, q) > 0
        for (i, j) in enumerate(drange(sh, q))
            M[j, j] = rhoD(f, n)[q][i]
        end
    end
    return M
end

"""
    thermal_average(run, f, expr) -> Number

`⟨O⟩ = Tr[ϱ O]` of a local operator, Lee Eqs. (46)-(47): the shell trace
`Σ_X tr[ρ̄^{n}_{XX} O^{n}_{XX}]` at `n = n_op`, the shell where `O` is first
represented (exact for local operators given the FDM `ϱ`). Inhomogeneous
operators are reduced to their neutral component (the others have zero
trace).
"""
function thermal_average(run::NRGRun{S}, f::FDM, expr::OpExpr) where {S}
    ch = charges(run.model, max(maxsite(expr), 0))
    comps = components(expr, ch)
    c = get(comps, neutral(S), nothing)
    c === nothing && return 0.0
    n = max(maxsite(c), -1)
    O = shell_operator(run, c, n)
    return shell_trace(run, f, O, n)
end

"""
    shell_trace(run, f, O, n) -> Number

`Σ_q degeneracy(q) tr[ρ̄^n_q O^n_{q}]` for a neutral shell-`n` operator, Lee
Eq. (48) summed over sectors.
"""
function shell_trace(run::NRGRun, f::FDM, O::BlockOp, n::Integer)
    isneutral(O.shift) || return zero(eltype(O))
    s = zero(promote_type(eltype(O), Float64))
    for (q, M) in O.blocks
        s += degeneracy(q) * tr(rho_full(f, run, n, q) * M)
    end
    return s
end

"""
    thermal_average_product(run, f, exprs) -> Number

`⟨O¹ O² ⋯ O^ℓ⟩` of local operators through the shell-diagonal product
expansion of Lee Eq. (50), `Σ_n Σ_{X…≠K…K} tr[ρ̄^n_{X₁X₁} O¹_{X₁X₂} ⋯ O^ℓ_{X_ℓX₁}]`,
evaluated from the projections of the individual operators. Independent of
[`thermal_average`](@ref) of the product operator; the two must agree
(exact identity for local operators).
"""
function thermal_average_product(run::NRGRun{S}, f::FDM, exprs::AbstractVector{<:OpExpr}) where {S}
    ℓ = length(exprs)
    ops = [local_operator(run, e) for e in exprs]
    shifts = map(opshift, ops)
    isneutral(foldl(fuse, shifts; init = neutral(S))) || return 0.0
    nstart = maximum(o.nop for o in ops)
    acc = 0.0 + 0.0im
    for n in nstart:run.params.N
        sh = shell(run, n)
        Os = [project(run, e, n) for e in exprs]
        for q1 in sectors(sh.space)
            qs = [q1]
            for i in 1:ℓ
                push!(qs, fuse(qs[end], dual(shifts[i])))
            end
            all(q -> haskey(sh.space, q), qs) || continue
            Ms = [getblock(Os[i], qs[i + 1]) for i in 1:ℓ]
            ρ = rho_full(f, run, n, q1)
            for X in Iterators.product(ntuple(_ -> (:K, :D), ℓ)...)
                all(==(:K), X) && continue
                r = [xrange(sh, qs[i], X[i]) for i in 1:ℓ]
                any(isempty, r) && continue
                P = ρ[r[1], r[1]]
                for i in 1:ℓ
                    nxt = i == ℓ ? r[1] : r[i + 1]
                    P = P * Ms[i][r[i], nxt]
                end
                acc += degeneracy(q1) * tr(P)
            end
        end
    end
    return isreal(acc) || abs(imag(acc)) < 1e-14 * max(1, abs(acc)) ? real(acc) : acc
end
