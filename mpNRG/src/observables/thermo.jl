# Thermodynamics (task sheet Sec. 5.9): impurity entropy and susceptibility
# from the full density matrix and from the conventional shell-wise formulas
# (Krishna-murthy, Wilkins & Wilson 1980; Bulla, Costi & Pruschke RMP 80, 395,
# Sec. III A), plus occupations.

"""
    bath_model(m) -> ImpurityModel

The same Wilson chain without the impurity (an empty impurity space), for the
subtraction `X_imp = X_total - X_bath` of impurity thermodynamics.
"""
function bath_model(m::ImpurityModel{S}) where {S}
    imp = LocalSpace(-1, Symbol[], Dict{Mode,S}())
    chans = [BathChannel(c.name, c.charge, c.hyb) for c in m.channels]
    return ImpurityModel(m.G, imp, OpExpr{Float64}(), chans; name = m.name * " (bath only)", check = false)
end

# 2S_z of a sector label, or `nothing` if the group has no :twoSz factor
function _twoSz(G::SymmetryGroup)
    names = factornames(G)
    i = findfirst(==(:twoSz), names)
    return i
end

"""
    fdm_thermodynamics(run, f) -> NamedTuple

Total entropy `S = β⟨H - E_ref⟩ + ln Z`, internal energy, and `T χ =
⟨S_z²⟩ - ⟨S_z⟩²` (needs a `U1(:twoSz)` factor) from the FDM in the
Anders-Schiller basis (NRG approximation, Lee Eq. (19)). The environment of a
discarded state of shell `n` (sites `> n`, free in the NRG approximation)
contributes its degeneracy to `Z` and `Σ_{n′>n} ⟨S_{z,n′}²⟩_free` to `⟨S_z²⟩`.
"""
function fdm_thermodynamics(run::NRGRun, f::FDM)
    N = run.params.N
    iS = _twoSz(run.model.G)
    E = 0.0; Sz2 = 0.0; Sz1 = 0.0
    # free-site ⟨S_z²⟩ of every site
    sitesz2 = zeros(N + 2)
    if iS !== nothing
        for n in 0:N
            L = shell(run, n).loc
            sitesz2[n + 2] = sum((q.q[iS] / 2)^2 * length(v) for (q, v) in L.sectors) / _localdim(L)
        end
    end
    envsz2 = [sum(sitesz2[(n + 3):end]; init = 0.0) for n in -1:N]
    for n in -1:N
        sh = shell(run, n)
        for (q, d) in rhoD(f, n)
            e = sh.E[q][drange(sh, q)] .+ sh.shift .- f.Eref
            E += degeneracy(q) * sum(d .* e)
            if iS !== nothing
                sz = q.q[iS] / 2
                Sz2 += degeneracy(q) * sum(d) * (sz^2 + envsz2[n + 2])
                Sz1 += degeneracy(q) * sum(d) * sz
            end
        end
    end
    S = f.β * E + f.logZ
    return (T = f.temp, energy = E, entropy = S, Tchi = iS === nothing ? NaN : Sz2 - Sz1^2, Sz = Sz1)
end

"""
    impurity_thermodynamics_fdm(run, runbath, Ts) -> NamedTuple of vectors

`S_imp(T)` and `T χ_imp(T)` on the temperatures `Ts` from FDM averages of the
model run and of the bath-only run (same chain), one backward sweep per `T`.
"""
function impurity_thermodynamics_fdm(run::NRGRun, runbath::NRGRun, Ts::AbstractVector{<:Real})
    S = Float64[]; X = Float64[]
    for T in Ts
        a = fdm_thermodynamics(run, fdm(run; T = T))
        b = fdm_thermodynamics(runbath, fdm(runbath; T = T))
        push!(S, a.entropy - b.entropy); push!(X, a.Tchi - b.Tchi)
    end
    return (T = collect(Float64, Ts), entropy = S, Tchi = X)
end

"""
    shell_thermodynamics(run; βbar = 0.75) -> NamedTuple of vectors

Conventional shell-wise thermodynamics (Krishna-murthy, Wilkins & Wilson
1980): at shell `n` the temperature is `T_n = ω_n / β̄` and all states of shell
`n` (kept and discarded) enter `Z_n = Σ e^{-E/T_n}`; `S_n = ln Z_n + ⟨E⟩/T_n`,
`T χ_n = ⟨S_z²⟩ - ⟨S_z⟩²`. Approximation: the chain is cut at site n (states of
later sites are ignored), valid for `T_n` well above the scale of later sites.
"""
function shell_thermodynamics(run::NRGRun; βbar::Real = 0.75)
    iS = _twoSz(run.model.G)
    Ts = Float64[]; S = Float64[]; X = Float64[]; ns = Int[]
    for n in 0:run.params.N
        sh = shell(run, n)
        Tn = energy_scale(run, n) / βbar
        lw = Float64[]; es = Float64[]; szs = Float64[]
        for q in sectors(sh.space), e in sh.E[q]
            for _ in 1:degeneracy(q)
                push!(lw, -e / Tn); push!(es, e); push!(szs, iS === nothing ? 0.0 : q.q[iS] / 2)
            end
        end
        lZ = _logsumexp(lw)
        p = exp.(lw .- lZ)
        push!(ns, n); push!(Ts, Tn)
        push!(S, lZ + sum(p .* es) / Tn)
        push!(X, iS === nothing ? NaN : sum(p .* szs .^ 2) - sum(p .* szs)^2)
    end
    return (n = ns, T = Ts, entropy = S, Tchi = X)
end

"""
    impurity_thermodynamics_shell(run, runbath; βbar = 0.75)

`S_imp`, `T χ_imp` shell by shell: model minus bath-only run at equal `T_n`.
"""
function impurity_thermodynamics_shell(run::NRGRun, runbath::NRGRun; βbar::Real = 0.75)
    a = shell_thermodynamics(run; βbar = βbar); b = shell_thermodynamics(runbath; βbar = βbar)
    return (n = a.n, T = a.T, entropy = a.entropy .- b.entropy, Tchi = a.Tchi .- b.Tchi)
end

"""
    evenodd_average(th) -> NamedTuple

Average consecutive shells of a shell-wise result (`(X_n + X_{n+1})/2` at
`T = √(T_n T_{n+1})`), removing the even/odd alternation that comes from the
different chain parities of the model and the bath-only run at the Kondo
(strong-coupling) fixed point (Krishna-murthy, Wilkins & Wilson 1980).
"""
function evenodd_average(th)
    k = length(th.n) - 1
    return (n = th.n[1:k], T = sqrt.(th.T[1:k] .* th.T[2:end]),
            entropy = (th.entropy[1:k] .+ th.entropy[2:end]) ./ 2, Tchi = (th.Tchi[1:k] .+ th.Tchi[2:end]) ./ 2)
end

"""
    impurity_thermodynamics_zavg(model, params, Ts; nz = 4) -> NamedTuple

FDM impurity thermodynamics averaged over `z = 1/n_z, …, 1` (Oliveira &
Oliveira 1994), which removes most of the log-periodic discretization
oscillations at large Λ.
"""
function impurity_thermodynamics_zavg(model::ImpurityModel, p::NRGParams, Ts::AbstractVector{<:Real}; nz::Integer = 4)
    S = zeros(length(Ts)); X = zeros(length(Ts))
    bm = bath_model(model)
    for k in 1:nz
        pk = NRGParams(N = p.N, Nkeep = p.Nkeep, Etrunc = p.Etrunc, gaptol = p.gaptol, Λ = p.Λ, z = k / nz,
                       scheme = p.scheme, dense_max = p.dense_max)
        run = run_nrg(model, pk)
        rb = run_nrg(bm, pk; chains = run.chains)
        th = impurity_thermodynamics_fdm(run, rb, Ts)
        S .+= th.entropy ./ nz; X .+= th.Tchi ./ nz
    end
    return (T = collect(Float64, Ts), entropy = S, Tchi = X)
end
