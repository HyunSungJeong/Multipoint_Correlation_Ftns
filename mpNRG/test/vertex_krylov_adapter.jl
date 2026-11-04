# vertex_krylov adapter (task sheet Sec. 3.2): DiscreteData of the SIAM Wilson
# chain from mpNRG, either from ED of the chain (`model_ed` + `ed_psf`) or from
# NRG (`psf(run, f, ops; grid)`). The algorithm itself lives in HubbardAtom
# (src/vertex_krylov.jl) and does not depend on MultipointNRG.
#
# Requires `helpers.jl` (model_ed, d_, dd_, n_) to be included first.

import HubbardAtom as HA
using MultipointNRG, LinearAlgebra

"""
    vk_operators(m; spins = [:up], i12 = (:dn, :dn)) -> NamedTuple

The operators of task sheet Sec. 2.1 as mpNRG operator expressions, built from
their definitions as in `test_v3_psf.jl`: `Hint = U n↑ n↓`, `q(σ) = comm(d_(σ), Hint)`,
`q†(σ) = comm(Hint, dd_(σ))`, `Bq = acomm(q(i1), dd_(i2))`;
`X = (d†_σ…, q†_σ…)`, `Xd = (d_σ…, q_σ…)`.
"""
function vk_operators(m; spins = [:up], i12 = (:dn, :dn))
    U = m.params[:U]
    Hint = U * n_(:up) * n_(:dn)
    q(σ) = comm(d_(σ), Hint)
    qd(σ) = comm(Hint, dd_(σ))
    Bq = acomm(q(i12[1]), dd_(i12[2]))
    X = vcat(Any[dd_(σ) for σ in spins], Any[qd(σ) for σ in spins])
    Xd = vcat(Any[d_(σ) for σ in spins], Any[q(σ) for σ in spins])
    return (X = X, Xd = Xd, Bq = Bq, Bqd = Bq', Hint = Hint)
end

_vk_opof(ops, k) = k[1] === :X ? ops.X[k[2]] : k[1] === :Xd ? ops.Xd[k[2]] : k[1] === :B ? ops.Bq : ops.Bqd

"""
    vk_loggrid(g::LogGrid) -> (HA.BinnedGrid, clamped::Ref{Int})

A `MultipointNRG.LogGrid` as a `HubbardAtom.BinnedGrid`: centres `centres(g)`,
mirror map `i ↦ nbins + 1 - i`, bin function `binindex(g, x)[1]`; positions that
`binindex` clamps (below the first or above the last bin) are counted in `clamped`.
"""
function vk_loggrid(g::LogGrid)
    b = centres(g)
    nb = length(b)
    clamped = Ref(0)
    f = function (x)
        i, c = binindex(g, x)
        c && (clamped[] += 1)
        return i
    end
    return HA.BinnedGrid(b, [nb + 1 - i for i in 1:nb], f; label = string(g)), clamped
end

_vk_spins(σ) = σ isa Symbol ? [σ] : collect(σ)
function _vk_pair(spins, channel, i12)
    i12 !== nothing && return i12
    length(spins) == 1 || throw(ArgumentError("give i12 for N > 1"))
    return HA.channel_pair(spins[1], channel)
end

"""
    vk_G0(m, chains, N, T; spins) -> z -> Matrix

The impurity Green's function `G_0(z)` (Pelz product, `{d, d†}`) of the **same
chain at `U = 0`** (same `εd`, field and chain), from ED: for the Dyson check
(C3), `Σ^S(z)` vs `G_0(z)^{-1} - G̃_dd(z)^{-1}` (task sheet Sec. 5.2).
"""
function vk_G0(m, chs, N, T; spins = [:up])
    p = m.params
    m0 = siam(U = 0.0, εd = p[:εd], B = p[:B], Δ = p[:Δ], D = p[:D])
    ed0 = model_ed(m0, chs, N, T)
    poles = map(spins) do σ
        P1, w1 = ed_psf(ed0, (d_(σ), dd_(σ)))      # S[d, d†](ε): weight at +ε
        P2, w2 = ed_psf(ed0, (dd_(σ), d_(σ)))      # S[d†, d](ε): weight at -ε in {d, d†}
        (vcat(vec(P1), -vec(P2)), ComplexF64.(vcat(w1, w2)))
    end
    return z -> Matrix(Diagonal(ComplexF64[sum(w ./ (z .- P)) for (P, w) in poles]))
end

"""
    discrete_data_ed(m, chains, N, T, grid; σ = :up, channel = :updn, i12 = nothing) -> DiscreteData

Discrete data (VD18)-(VD24) from ED of the Wilson chain with sites `-1..N`
(`model_ed` + `ed_psf`, exact Lehmann positions). `grid = :exact` builds an
`ExactGrid`; a `LogGrid` bins the exact positions with `binindex` (clampings
counted in `meta[:nclamped]`). `σ` may be a vector of spins (block code).
"""
function discrete_data_ed(m, chs, N::Integer, T::Real, grid; σ = :up, channel::Symbol = :updn,
                          i12 = nothing, degtol::Real = 1e-10, G0::Bool = true)
    spins = _vk_spins(σ)
    pair = _vk_pair(spins, channel, i12)
    ops = vk_operators(m; spins = spins, i12 = pair)
    t0 = time()
    ed = model_ed(m, chs, N, T)
    terms(ks...) = ed_psf(ed, Tuple(_vk_opof(ops, k) for k in ks))
    tr2(k1, k2) = ed_thermal_average(ed, _vk_opof(ops, k1) * _vk_opof(ops, k2))
    tr3(k1, k2, k3) = ed_thermal_average(ed, _vk_opof(ops, k1) * _vk_opof(ops, k2) * _vk_opof(ops, k3))
    p = m.params
    meta = Dict{Symbol,Any}(:model => "SIAM Wilson chain", :source => "mpNRG ED (model_ed + ed_psf)",
                            :U => p[:U], :εd => p[:εd], :field => p[:B], :Δ => p[:Δ], :D => p[:D],
                            :N => N, :nstates => 4^(N + 2), :T => T, :spins => spins, :i12 => pair,
                            :channel => channel, :chain_eps => [c.eps for c in chs], :chain_t => [c.t for c in chs])
    g = grid
    clamped = Ref(0)
    if grid isa LogGrid
        g, clamped = vk_loggrid(grid)
        meta[:loggrid] = grid
    end
    d = HA.discrete_data_terms(terms, length(spins); grid = g, degtol = degtol, T = T, tr2 = tr2, tr3 = tr3,
                               meta = meta)
    d.meta[:nclamped] = clamped[]
    G0 && (d.meta[:G0] = vk_G0(m, chs, N, T; spins = spins))
    d.meta[:build_time] = time() - t0
    return d
end

"""
    discrete_data_nrg(run, f, grid; σ = :up, channel = :updn, i12 = nothing, m) -> DiscreteData

Discrete data from NRG: one `psf(run, f, ops; grid)` call per entry of task
sheet Sec. 2.4 (dense histograms indexed by `centres(grid)`), on one shared
`LogGrid`. The sum-rule references use `thermal_average_product`. Clamping
counts of the engine are summed in `meta[:nclamped]`.
"""
function discrete_data_nrg(run, f, grid::LogGrid; m, σ = :up, channel::Symbol = :updn, i12 = nothing)
    spins = _vk_spins(σ)
    pair = _vk_pair(spins, channel, i12)
    ops = vk_operators(m; spins = spins, i12 = pair)
    nb = nbins(grid)
    nclamped = Ref(0)
    t0 = time()
    function S2(k1, k2)
        r = psf(run, f, (_vk_opof(ops, k1), _vk_opof(ops, k2)); grid = grid)
        nclamped[] += get(r.meta, :nclamped, 0)
        h = histogram(r)
        length(h) == nb || throw(DimensionMismatch("2p histogram size"))
        return ComplexF64.(vec(h))
    end
    function S3(k1, k2, k3)
        r = psf(run, f, (_vk_opof(ops, k1), _vk_opof(ops, k2), _vk_opof(ops, k3)); grid = grid)
        nclamped[] += get(r.meta, :nclamped, 0)
        return Dict{NTuple{2,Int},ComplexF64}(k => ComplexF64(v) for (k, v) in nonzero_bins(r))
    end
    tr2(k1, k2) = thermal_average_product(run, f, [_vk_opof(ops, k1), _vk_opof(ops, k2)])
    tr3(k1, k2, k3) = thermal_average_product(run, f, [_vk_opof(ops, k1), _vk_opof(ops, k2), _vk_opof(ops, k3)])
    b = centres(grid)
    pm = run.params
    meta = Dict{Symbol,Any}(:model => "SIAM Wilson chain", :source => "mpNRG NRG (psf on LogGrid)",
                            :U => m.params[:U], :εd => m.params[:εd], :field => m.params[:B], :Δ => m.params[:Δ],
                            :D => m.params[:D], :N => pm.N, :Nkeep => pm.Nkeep, :Λ => pm.Λ, :z => pm.z, :T => f.temp,
                            :spins => spins, :i12 => pair, :channel => channel, :grid_kind => :binned,
                            :loggrid => grid, :chain_eps => [c.eps for c in run.chains],
                            :chain_t => [c.t for c in run.chains])
    d = HA.build_discrete_data(length(spins), b, [nb + 1 - i for i in 1:nb], S2, S3; T = f.temp, tr2 = tr2,
                               tr3 = tr3, meta = meta)
    d.meta[:nclamped] = nclamped[]
    d.meta[:build_time] = time() - t0
    return d
end

"""
    data_diff(a, b) -> NamedTuple

Direct comparison of two `DiscreteData` on the same grid (V3a: NRG vs binned ED):
`max|ΔW̃|`, `max|Δw^B|`, `max|Δs|` over all six families.
"""
function data_diff(a::HA.DiscreteData, b::HA.DiscreteData)
    a.b == b.b || throw(ArgumentError("different grids"))
    dW = maximum(β -> maximum(abs, a.W[β] - b.W[β]), eachindex(a.b))
    dB = max(maximum(abs, a.wB - b.wB), maximum(abs, a.wBd - b.wBd))
    function fd(x, y)
        m = 0.0
        for k in union(keys(x), keys(y))
            u = get(x, k, nothing); v = get(y, k, nothing)
            m = max(m, maximum(abs, (u === nothing ? zero(v) : u) - (v === nothing ? zero(u) : v)))
        end
        return m
    end
    ds = max(fd(a.s1, b.s1), fd(a.s2, b.s2),
             maximum(fd(a.s3[x], b.s3[x]) for x in (:q, :d)), maximum(fd(a.s4[x], b.s4[x]) for x in (:q, :d)),
             maximum(fd(a.s5[y], b.s5[y]) for y in (:qd, :dd)), maximum(fd(a.s6[y], b.s6[y]) for y in (:qd, :dd)))
    return (W = dW, wB = dB, s = ds)
end

# ----------------------------------------------------------------------------
# Case runners shared by test_vertex_krylov_nrg.jl and notebooks/vertex_krylov.ipynb
# ----------------------------------------------------------------------------

"""The model of the V2/V3 rows (task sheet Sec. 6): `siam(U = 0.5, εd = -0.1, Δ = 0.1, B = field)`."""
vk_model(; field = 0.0) = siam(U = 0.5, εd = -0.1, Δ = 0.1, B = field)

"""The shared grid of the V2b/V3 rows: `LogGrid(emin = 1e-5, emax = 10.0, ndec = 16)` (as mpNRG V3c)."""
vk_grid() = LogGrid(emin = 1e-5, emax = 10.0, ndec = 16)

"""
    vk_run(d; G0 = d.meta[:G0], diagnostic = exact, bigfloat = false) -> Dict

Run Algorithm 2 on `d` (`HA.vertex_krylov`; for exact data the Stage-2 default is
the stabilized particle chain of [VD] Prop. 13, amendment A2) and all checks
(`HA.vk_checks`). For exact data also the diagnostic `out[:unstab]`: the
unstabilized chain (`Π = 1 - vv†`) with its C4 (report, amendment A1), C1,
length, leakage profile, and the change of `Σ^S` and `K̃` against the default
chain. With `bigfloat`, the particle chain in `BigFloat` (precision 128):
`out[:big]` for the default chain (C4, length, residual, deviation of the Float64
Jacobi blocks, `n_agree`), and `out[:big_unstab]` (C4, length) for the
unstabilized one.
"""
function vk_run(d::HA.DiscreteData; G0 = get(d.meta, :G0, nothing),
                diagnostic::Bool = get(d.meta, :grid_kind, :binned) === :exact,
                bigfloat::Bool = false, timings = Dict{Symbol,Float64}(), gram::Symbol = :assert)
    exact = get(d.meta, :grid_kind, :binned) === :exact
    t0 = time(); res = HA.vertex_krylov(d; gram = gram); timings[:algorithm] = time() - t0
    t0 = time(); c = HA.vk_checks(res; G0 = G0); timings[:checks] = time() - t0
    out = Dict{Symbol,Any}(:res => res, :checks => c, :timings => timings)
    if diagnostic
        t0 = time()
        chu = HA.chains(d; stabilize = false, cs = res.ch.cs)
        Cu, kdu = HA.kernels(d, chu)
        cpu = HA.couplings(d, chu, Cu)
        bxu = HA.K_box(HA.VKResult(d, chu, Cu, kdu, cpu))
        bx = c[:box]
        m = .!isnan.(real.(bx.Kd))
        zs = HA.vk_testpoints(d)
        dΣ = maximum(z -> maximum(abs, HA.sigma_chain(chu, z) - HA.sigma_chain(res.ch, z)), zs)
        out[:unstab] = (C4 = HA.check_C4(chu), C1 = HA.check_C1(d, chu), L = HA.nblocks(chu.Jbar),
                        C8 = HA.check_C8(bxu).value, dK = maximum(abs, bxu.Kt[m] - bx.Kt[m]) / maximum(abs, bx.Kd[m]),
                        dSigma = dΣ, profile = HA.leakage_profile(chu))
        timings[:unstabilized] = time() - t0
    end
    if bigfloat
        t0 = time()
        setprecision(BigFloat, 128) do
            chb = HA.chains(d; T = BigFloat, stabilize = exact, gram = gram)
            L = min(HA.nblocks(chb.Jbar), HA.nblocks(res.ch.Jbar))
            scale = maximum(abs, d.b[[β for β in eachindex(d.b) if !iszero(d.W[β])]])
            dev = [Float64(max(maximum(abs, chb.Jbar.eps[n] - res.ch.Jbar.eps[n]),
                               n < L ? maximum(abs, chb.Jbar.lower[n] - res.ch.Jbar.lower[n]) : 0.0)) / scale for n in 1:L]
            bad = findfirst(>(1e-8), dev)
            out[:big] = (C4 = Float64(HA.check_C4(chb)), L = HA.nblocks(chb.Jbar), resid = chb.diag[:particle][:resid],
                         orth = chb.diag[:particle][:orth_loss], dev = dev, n_agree = bad === nothing ? L : bad - 1,
                         profile = HA.leakage_profile(chb), C1 = HA.check_C1(d, chb))
            if exact
                chbu = HA.chains(d; T = BigFloat, stabilize = false)
                out[:big_unstab] = (C4 = Float64(HA.check_C4(chbu)), L = HA.nblocks(chbu.Jbar),
                                    profile = HA.leakage_profile(chbu))
            end
        end
        timings[:bigfloat] = time() - t0
    end
    return out
end

"""
    vk_lighten!(out) -> out

Drop the heavy objects of a [`vk_run`](@ref) result (kernels, couplings,
coordinates), keeping the checks, the box, and the chain coefficients
(`out[:chain]`: `ε̄_n`, `t̄_{n+1}`, `a_n`, `b_n`, the leakage profile). Used by the
notebook to keep memory bounded.
"""
function vk_lighten!(out)
    res = out[:res]
    res === nothing && return out
    ch = res.ch
    out[:chain] = (eps = [e[1, 1] for e in ch.Jbar.eps], t = [t[1, 1] for t in ch.Jbar.lower],
                   a = [e[1, 1] for e in ch.JB.eps], b = [t[1, 1] for t in ch.JB.lower],
                   profile = HA.leakage_profile(ch))
    out[:res] = nothing
    return out
end

"""
    vk_case_ed(Nch, gridkind, chan; field = 0.0, T = 0.05, bigfloat = false) -> Dict

One V2 case: ED of the Wilson chain with sites `-1..Nch`, data on an `ExactGrid`
(`gridkind = :exact`, V2a) or binned on [`vk_grid`](@ref) (`:log`, V2b), channel
`chan`, then [`vk_run`](@ref).
"""
function vk_case_ed(Nch::Integer, gridkind::Symbol, chan::Symbol; field = 0.0, T = 0.05, bigfloat::Bool = false, keep::Bool = true,
                    σ = :up, i12 = nothing)
    m = vk_model(; field = field)
    chs = wilson_chains(m, Nch)
    timings = Dict{Symbol,Float64}()
    g = gridkind === :exact ? :exact : vk_grid()
    t0 = time(); d = discrete_data_ed(m, chs, Nch, T, g; σ = σ, channel = chan, i12 = i12); timings[:data] = time() - t0
    out = vk_run(d; bigfloat = bigfloat, timings = timings)
    keep || vk_lighten!(out)
    out[:data] = d
    return out
end

"""
    vk_case_nrg(Nch, chan; field = 0.0, T = 0.05, Nkeep = typemax(Int), chains = nothing, compare_ed = true) -> Dict

One V3 case: NRG run (`NRGParams(N = Nch, Nkeep)`, optionally on given chains),
data on [`vk_grid`](@ref) from `psf(run, f, ops; grid)`, then [`vk_run`](@ref);
with `compare_ed`, also the binned ED data of the same chain on the same grid,
its run, and [`data_diff`](@ref).
"""
function vk_case_nrg(Nch::Integer, chan::Symbol; field = 0.0, T = 0.05, Nkeep::Integer = typemax(Int), keep::Bool = true,
                     chains = nothing, compare_ed::Bool = true)
    m = vk_model(; field = field)
    g = vk_grid()
    timings = Dict{Symbol,Float64}()
    t0 = time()
    nrun = chains === nothing ? run_nrg(m, NRGParams(N = Nch, Nkeep = Nkeep)) :
           run_nrg(m, NRGParams(N = Nch, Nkeep = Nkeep); chains = chains)
    f = fdm(nrun; T = T)
    timings[:nrg] = time() - t0
    t0 = time(); d = discrete_data_nrg(nrun, f, g; m = m, σ = :up, channel = chan); timings[:data] = time() - t0
    d.meta[:truncation] = (kept = [totalkept(shell(nrun, n)) for n in 0:Nch], disc = [totaldisc(shell(nrun, n)) for n in 0:Nch])
    out = Dict{Symbol,Any}()
    if compare_ed
        t0 = time(); de = discrete_data_ed(m, nrun.chains, Nch, T, g; σ = :up, channel = chan); timings[:ed_data] = time() - t0
        out[:ed_data] = de
        out[:diff] = data_diff(d, de)
        d.meta[:G0] = de.meta[:G0]
        out[:ed] = vk_lighten!(vk_run(de))
    end
    merge!(out, vk_run(d; timings = timings))
    keep || vk_lighten!(out)
    out[:data] = d
    return out
end

# ----------------------------------------------------------------------------
# V3c: physical chain, truncated NRG (task sheet Sec. 6, Step 2)
# ----------------------------------------------------------------------------

"""
    vk_v3c_grid()

The V3c grid, `LogGrid(emin = 1e-9, emax = 10.0, ndec = 16)`. Deviation from
`vk_grid()` (emin = 1e-5), recorded in vertex_krylov_STATUS.md: the chains of V3c
resolve energies down to `~T Λ^{-4}`; with emin = 1e-5 a single 3p PSF at
`T = 1e-4` has ~5e6 (N_keep = 64) to ~3e8 (N_keep = 512) clamped contributions.
"""
vk_v3c_grid() = LogGrid(emin = 1e-9, emax = 10.0, ndec = 16)

"""Chain length of V3c: `chain_length_for(T; Λ = 4)` (as the mpNRG validation runs, Λ = 4, margin 8)."""
vk_v3c_length(T) = chain_length_for(T; Λ = 4.0)

"""
    vk_G0_chain(meta; spins = meta[:spins]) -> z -> Matrix

The impurity Green's function `G_0(z) = [(z - h_σ)^{-1}]_{-1,-1}` of the **same chain
at `U = 0`**, from the single-particle Hamiltonian of the Wilson chain stored in
`meta` (on-site `εd - s_σ B/2, ε_0, …, ε_N`; hoppings `t_0, …, t_N` between sites
`n-1` and `n`, the convention of `hamiltonian_expr`). Exact for the discretized
chain; used for the Dyson check (C3) of the long V3c chains, where ED is impossible.
"""
function vk_G0_chain(meta; spins = meta[:spins])
    εd = meta[:εd]; Bf = meta[:field]
    hs = map(spins) do σ
        c = σ === :up ? 1 : 2
        e = meta[:chain_eps][c]; t = meta[:chain_t][c]
        L = length(e)
        h = zeros(L + 1, L + 1)
        h[1, 1] = εd - (σ === :up ? 1 : -1) * Bf / 2
        for n in 1:L
            h[n + 1, n + 1] = e[n]
            h[n, n + 1] = h[n + 1, n] = t[n]
        end
        eigen(Symmetric(h))
    end
    return z -> Matrix(Diagonal(ComplexF64[sum(abs2.(F.vectors[1, :]) ./ (z .- F.values)) for F in hs]))
end

"""
    vk_v3c_data(T, Nkeep, chan; grid = vk_v3c_grid()) -> DiscreteData

V3c data: `run_nrg(vk_model(), NRGParams(N = vk_v3c_length(T), Nkeep))` (Λ = 4, z = 1),
`fdm(run; T)`, and [`discrete_data_nrg`](@ref) on `grid`. Records the truncation
(kept / discarded per shell), the clamping count and the timings in `meta`.
"""
function vk_v3c_data(T::Real, Nkeep::Integer, chan::Symbol; grid = vk_v3c_grid())
    m = vk_model()
    N = vk_v3c_length(T)
    t0 = time()
    nrun = run_nrg(m, NRGParams(N = N, Nkeep = Nkeep))
    f = fdm(nrun; T = T)
    tnrg = time() - t0
    t0 = time()
    d = discrete_data_nrg(nrun, f, grid; m = m, σ = :up, channel = chan)
    d.meta[:time_nrg] = tnrg
    d.meta[:time_data] = time() - t0
    d.meta[:truncation] = (kept = [totalkept(shell(nrun, n)) for n in 0:N], disc = [totaldisc(shell(nrun, n)) for n in 0:N])
    return d
end

"""
    vk_case_v3c(d; bigfloat = true) -> Dict

Run Algorithm 2 and all checks on V3c data `d` (report only), with the U = 0 chain
propagator of [`vk_G0_chain`](@ref) for C3 and the particle chain also in BigFloat.
Truncated NRG weights `W̃_β` need not be Hermitian bin by bin (Stage 1 uses the
Hermitian part; the anti-Hermitian part is reported) nor positive semidefinite; Stage 1 drops their
negative eigenvalues (reported: `coords` diag `:rank_dropped_negative`), so
`v†v ≠ 1` and Stage 1 runs with `gram = :general` where the specified assertion
`‖v†v - 1‖ ≤ 1e-12` would fail (reported: `:gram_general`, `:norm_err`; deviation in
vertex_krylov_STATUS.md). `out[:psd]` summarizes the non-PSD part of the data.
"""
function vk_case_v3c(d::HA.DiscreteData; bigfloat::Bool = true)
    o = vk_run(d; G0 = vk_G0_chain(d.meta), bigfloat = bigfloat, diagnostic = false, gram = :general)
    neg = 0.0; worst = 0.0; nneg = 0
    for W in d.W
        iszero(W) && continue
        λ = eigvals(Hermitian((W + W') / 2))
        m = minimum(λ)
        if m < -1e-13 * maximum(abs, λ)
            nneg += 1; neg += -sum(min.(λ, 0.0)); worst = max(worst, -m / maximum(abs, λ))
        end
    end
    cd = o[:res].ch.cs.diag
    o[:psd] = (neg_mass = neg, worst_rel = worst, nbins_neg = nneg, norm_err = cd[:norm_err],
               gram_general = cd[:gram_general], dropped_negative = cd[:rank_dropped_negative],
               antiherm_rel = cd[:antiherm_rel], antiherm_sum = cd[:antiherm_sum],
               wB_neg = (cd[:c_clamp].nclamped, cd[:c_clamp].clamped_min), wBd_neg = (cd[:cp_clamp].nclamped, cd[:cp_clamp].clamped_min))
    return o
end

"""    vk_strip(d) — `d` without the closure in `meta[:G0]` (for JLD2 storage)."""
function vk_strip(d::HA.DiscreteData)
    meta = copy(d.meta); delete!(meta, :G0)
    return HA.DiscreteData(d.N, d.b, d.bar, d.W, d.Wplain, d.Whole, d.wB, d.wBd, d.s1, d.s2, d.s3, d.s4, d.s5, d.s6, d.T, meta)
end
