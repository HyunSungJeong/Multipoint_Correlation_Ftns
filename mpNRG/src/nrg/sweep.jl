# Iterative diagonalization along the Wilson chain (task sheet Sec. 5.4).
#
# The tensor contractions are the named functions of nrg/tensors.jl.
#
# Lee Sec. III A: shell n is obtained by adding site n to the kept (K) sector
# of shell n-1 and diagonalizing H^n on the product space
# {|σ_n⟩ ⊗ |s⟩^{n-1}_K}; the eigenstates are |s′⟩^n_X = |σ_n⟩ ⊗ |s⟩^{n-1}_K
# [M^{σ_n}_{KX}]_{s s′}, Lee Eq. (15a). They are then split into K and D
# sectors (all D at n = N).

"""
    NRGParams(; N, Nkeep = 1000, Etrunc = nothing, gaptol = 1e-8, Λ = 4, z = 1,
              scheme = :zitko, dense_max = 4096)

Parameters of the forward sweep.

  - `N`: last Wilson site (sites `0..N`). Choose it so that `Λ^{-N/2} ≪ T`
    (Lee p. 5); [`chain_length_for`](@ref) does this with a margin.
  - `Nkeep`: maximal number of kept states per shell (`typemax(Int)`: no
    truncation). `Etrunc`: energy cutoff in rescaled units `E/ω_n`,
    `ω_n = D Λ^{-(n-1)/2}`. With both, states below `Etrunc` are kept, at most
    `Nkeep` of them.
  - `gaptol`: the cut is moved up into the nearest gap larger than
    `gaptol·ω_n`, so no degenerate multiplet is split (Sec. 5.2.7).
  - `dense_max`: largest dense Fock dimension for the dense embedding of
    shells (used to build local operators and for tests).
"""
Base.@kwdef struct NRGParams
    N::Int
    Nkeep::Int = 1000
    Etrunc::Union{Nothing,Float64} = nothing
    gaptol::Float64 = 1e-8
    Λ::Float64 = 4.0
    z::Float64 = 1.0
    scheme::Symbol = :zitko
    dense_max::Int = 4096
end

"""
    chain_length_for(T; Λ = 4, margin = 8) -> N

The chain length with `Λ^{-N/2} = T Λ^{-margin/2}`, i.e. `margin` extra sites
beyond the temperature scale (task sheet Sec. 4.2).
"""
chain_length_for(T::Real; Λ::Real = 4.0, margin::Integer = 8) = ceil(Int, -2 * log(T) / log(Λ)) + margin

"""
    Shell{S,T}

Data of Wilson shell `n` (Lee Sec. III A), per symmetry block:

  - `space`: all states of the shell (label → dimension);
  - `leg`: the same states as a TNOps leg with the K/D split in its tags
    (`(q, KTAG)` with `nkept(q)`, `(q, DTAG)` with `ndisc(q)` states);
  - `nkeep[q]`: number of kept states of label `q` (they are the lowest
    `nkeep[q]` of the label; the rest are discarded);
  - `E[q]`: energies relative to `shift`, ascending (`E_abs = E + shift`);
  - `shift`: the accumulated ground-state energy, so that energies of different
    shells share one absolute scale (needed by the FDM weights);
  - `fusion`: the product basis `{|σ_n⟩ ⊗ |s⟩^{n-1}_K}`, a TNOps `FusionMap`
    (site leg outer, kept leg of shell n-1 inner; `nothing` at n = -1);
  - `A`: the isometry of Lee Eq. (15) as a TNOps tensor with legs (old, site,
    new) — rank 2 (site, new) at n = -1 —, `Adag` its adjoint (`nrg/tensors.jl`),
    `plan` its [`RefinePlan`](@ref) for the refinement map (`nothing` at n = -1);
  - `loc`: the local space of site `n`;
  - `F`: the annihilators `f_{nν}` of site `n` in the shell-`n` basis (all
    states, rank-2 tensors on `leg`), needed for the hopping to site `n+1`.
"""
struct Shell{S<:Sector,T<:Number}
    n::Int
    space::SectorSpace{S}
    leg::Leg{S}
    nkeep::Dict{S,Int}
    E::Dict{S,Vector{Float64}}
    shift::Float64
    fusion::Union{Nothing,FusionMap{S}}
    A::BlockTensor
    Adag::BlockTensor
    plan::Union{Nothing,RefinePlan{S,T}}
    loc::LocalSpace{S}
    F::Vector{BlockTensor{T,2,S}}
    info::Dict{Symbol,Any}
end

nkept(sh::Shell, q) = get(sh.nkeep, q, 0)
ndisc(sh::Shell, q) = dim(sh.space, q) - nkept(sh, q)
"""    keptspace(sh) / discspace(sh) — sector spaces of the K and D sectors."""
keptspace(sh::Shell) = SectorSpace(Dict(q => nkept(sh, q) for q in sectors(sh.space)))
discspace(sh::Shell) = SectorSpace(Dict(q => ndisc(sh, q) for q in sectors(sh.space)))
"""    krange(sh, q) / drange(sh, q) — index ranges of the K and D states of sector q."""
krange(sh::Shell, q) = 1:nkept(sh, q)
drange(sh::Shell, q) = (nkept(sh, q) + 1):dim(sh.space, q)
xrange(sh::Shell, q, X::Symbol) = X === :K ? krange(sh, q) : drange(sh, q)
totalkept(sh::Shell) = sum((degeneracy(q) * nkept(sh, q) for q in sectors(sh.space)); init = 0)
totaldisc(sh::Shell) = dim(sh.space) - totalkept(sh)

"""
    NRGRun

Result of [`run_nrg`](@ref): the model, the Wilson chains, the parameters and
every shell `n = -1..N` (`shell(run, n)`), plus `n0`, the first shell with
discarded states (Lee p. 5: the last shell diagonalized without truncation),
and flow-diagram and timing data.
"""
struct NRGRun{S<:Sector,T<:Number}
    model::ImpurityModel{S}
    chains::Vector{WilsonChain}
    params::NRGParams
    shells::Vector{Shell{S,T}}
    n0::Int
    info::Dict{Symbol,Any}
    cache::Dict{Any,Any}
end

shell(run::NRGRun, n::Integer) = run.shells[n + 2]
chainlength(run::NRGRun) = run.params.N
"""    energy_scale(run, n) — `ω_n = D Λ^{-(n-1)/2}`, the rescaling of shell n."""
energy_scale(run::NRGRun, n::Integer) = _bandD(run.model) * run.params.Λ^(-(n - 1) / 2)
_bandD(m::ImpurityModel) = isempty(m.channels) ? 1.0 : maximum(bandwidth(c.hyb) for c in m.channels)

_eltype(e::OpExpr) = eltype(e) <: Real ? Float64 : ComplexF64

"""
    truncate_shell(E, space, ω; Nkeep, Etrunc, gaptol, last) -> (nkeep, record)

The global NRG cut (Sec. 5.2.7), a thin wrapper of `TNOps.global_cut` that
adds the NRG scale and the last-shell rule: the energies of all blocks are
sorted together (each counted `degeneracy(q)` times), the cut is placed at the
`Nkeep`-th state or the energy cutoff `Etrunc·ω` (whichever keeps fewer), and
then moved **up** into the nearest gap larger than `gaptol·ω`, so that no
degenerate multiplet is split even when it is spread over several blocks; at
the last shell (`last = true`) every state is discarded. The record holds the
gap in units of ω. Approximation: truncation (Lee Sec. III A, justified by
energy-scale separation, Wilson 1975).
"""
function truncate_shell(E::Dict{S,Vector{Float64}}, space, ω::Real;
                        Nkeep::Integer, Etrunc, gaptol::Real, last::Bool) where {S}
    if last
        ntot = sum((degeneracy(q) * length(E[q]) for q in sectors(space)); init = 0)
        return Dict(q => 0 for q in sectors(space)),
               (requested = 0, actual = 0, gap = NaN, cut_moved = false, total = ntot)
    end
    nk, rec = global_cut(E; nkeep = Nkeep, threshold = Etrunc === nothing ? nothing : Etrunc * ω, gaptol = gaptol * ω)
    return nk, (requested = rec.requested, actual = rec.actual, gap = rec.gap / ω, cut_moved = rec.cut_moved, total = rec.total)
end

# the cut of a shell: energies relative to their minimum, global truncation,
# statistics; `res` receives the relative energies, the shift and the record
function _cutter(p::NRGParams, ω, last::Bool, res::Dict{Symbol,Any})
    return E -> begin
        Eg = minimum(minimum(e) for e in values(E))
        Erel = Dict(q => e .- Eg for (q, e) in E)
        space = SectorSpace(Dict(q => length(e) for (q, e) in E))
        nk, rec = truncate_shell(Erel, space, ω; Nkeep = p.Nkeep, Etrunc = p.Etrunc, gaptol = p.gaptol, last = last)
        res[:Erel] = Erel; res[:Eg] = Eg; res[:space] = space; res[:rec] = rec; res[:nk] = nk
        nk
    end
end

function _shell_info(res, ω)
    sizes = sort!([length(e) for e in values(res[:Erel])]; rev = true)
    return Dict{Symbol,Any}(:truncation => res[:rec], :nkeep_per_block => copy(res[:nk]),
                            :largest_block => isempty(sizes) ? 0 : sizes[1], :block_sizes => sizes, :ω => ω)
end

"""
    run_nrg(model, params; chains = nothing) -> NRGRun

The forward sweep (Lee Sec. III A; Wilson 1975; Krishna-murthy, Wilkins &
Wilson 1980): diagonalize the impurity (shell -1), add site 0 with the
impurity-bath coupling, then add sites `1..N` one at a time, building `H^n`
per symmetry block on the product space of the new site and the kept states of
shell `n-1` (`product_hamiltonian`), diagonalizing each block with dense
`eigen` (threaded over blocks) and truncating globally
(`diagonalize_shell`). The hopping `t_n Σ_ν (f†_{nν} f_{n-1,ν} + h.c.)` is
assembled as `kron_fused(f†_{nν} Z_n, f_{n-1,ν})` by rule (R1) of
`nrg/tensors.jl`; no sign is entered by hand.

Approximation: truncation (see [`truncate_shell`](@ref)); the NRG
approximation Lee Eq. (19) is made later, when shell energies are used as
eigenenergies of the full chain.
"""
function run_nrg(model::ImpurityModel{S}, p::NRGParams; chains = nothing) where {S}
    t0 = time()
    chs = chains === nothing ? wilson_chains(model, p.N; Λ = p.Λ, z = p.z, scheme = p.scheme) : chains
    all(c -> chainlength(c) ≥ p.N, chs) || throw(ArgumentError("chains shorter than N = $(p.N)"))
    T = promote_type(_eltype(model.Himp), _eltype(model.Hsite0),
                     (c.coupling === nothing ? Float64 : _eltype(c.coupling) for c in model.channels)...)
    shells = Shell{S,T}[]
    runinfo = Dict{Symbol,Any}(:timing => Float64[], :flow => Dict{Int,Any}())

    # ---- shell -1: the impurity
    ts = time()
    imp = model.impurity
    Hloc = localop(imp, model.Himp; T = T)
    ω = energy_scale_param(model, p, -1)
    res = Dict{Symbol,Any}()
    _, nk, A = diagonalize_shell(Hloc, nothing; cut = _cutter(p, ω, false, res))
    push!(shells, _make_shell(-1, res, nk, res[:Eg], nothing, A, imp, BlockTensor{T,2,S}[], _shell_info(res, ω), nothing))
    push!(runinfo[:timing], time() - ts)

    # ---- shells 0..N
    for n in 0:p.N
        ts = time()
        prev = shells[end]
        loc = site_space(model, n)
        Fu = fuse_site(site_leg(loc), kept_leg(prev.leg))
        Hprod = if n == 0
            _site0_product_hamiltonian(model, chs, prev, loc, Fu, T)
        else
            _chain_product_hamiltonian(model, chs, n, prev, loc, Fu, T)
        end
        last = n == p.N
        ω = energy_scale_param(model, p, n)
        res = Dict{Symbol,Any}()
        _, nk, A = diagonalize_shell(Hprod, Fu; cut = _cutter(p, ω, last, res))
        # H^0 is built from the absolute site-0 Hamiltonian; H^n (n ≥ 1) from
        # kept energies relative to the previous shift
        shift = (n == 0 ? 0.0 : prev.shift) + res[:Eg]
        Adag = isometry_adjoint(A)
        # annihilators of site n in the shell basis (rule R2: f_n ⊗ 1)
        Fs = BlockTensor{T,2,S}[lift_site(A, Adag, localop(loc, cop(Mode(n, c.name)); T = T)) for c in model.channels]
        push!(shells, _make_shell(n, res, nk, shift, Fu, A, loc, Fs, _shell_info(res, ω), prev, Adag))
        push!(runinfo[:timing], time() - ts)
    end
    n0 = findfirst(sh -> totaldisc(sh) > 0, shells)
    n0 = shells[n0].n
    runinfo[:total_time] = time() - t0
    run = NRGRun{S,T}(model, chs, p, shells, n0, runinfo, Dict{Any,Any}())
    return run
end

function _make_shell(n, res, nk::Dict{S}, shift, Fu, A, loc::LocalSpace{S}, Fs::Vector{BlockTensor{T,2,S}}, info, prev,
                     Adag = isometry_adjoint(A)) where {S,T}
    leg = TNOps.witharrow(TNOps.leg(A, ndims(A)), Out)
    E = res[:Erel]; space = res[:space]
    plan = Fu === nothing ? nothing : RefinePlan(A, Fu)
    return Shell{S,T}(n, space, leg, Dict{S,Int}(nk), E, shift, Fu, A, Adag, plan, loc, Fs, info)
end

energy_scale_param(m::ImpurityModel, p::NRGParams, n::Integer) = _bandD(m) * p.Λ^(-(n - 1) / 2)

# H^0 on the product space {|σ_0⟩ ⊗ |s⟩^{-1}_K}: the site-0 Hamiltonian
# (impurity, coupling, ε_0) embedded densely on the Fock space of sites (0, -1)
# and projected onto the product basis.
function _site0_product_hamiltonian(model, chs, prev::Shell{S,T}, loc, Fu, ::Type{T}) where {S,T}
    H0 = site0_hamiltonian(model, chs)
    B = _product_embedding(prev, loc, Fu, nothing)
    Hd = densematrix(H0, B.basis; T = T)
    return product_hamiltonian_dense(Fu, B.cols, Hd)
end

# H^n (n ≥ 1): E_{n-1} ⊗ 1 + h_loc ⊗ 1 + Σ_ν t_n [(f†_{nν} Z_n) ⊗ f_{n-1,ν} + h.c.].
function _chain_product_hamiltonian(model, chs, n, prev::Shell{S,T}, loc, Fu, ::Type{T}) where {S,T}
    L = site_leg(loc)
    hloc = BlockTensor{T}(L, L, neutral(S))
    for (c, ch) in zip(model.channels, chs)
        hloc = hloc + ch.eps[n + 1] * localop(loc, num(Mode(n, c.name)); T = T)
    end
    Z = parity_op(loc)
    kk = (s -> s.tag == KTAG, s -> s.tag == KTAG)
    hops = [(ch.t[n + 1], localop(loc, cdag(Mode(n, c.name)); T = T) * Z, subtensor(prev.F[ν], kk))   # (t, f†_{nν} Z_n, f_{n-1,ν}|_KK)
            for (ν, (c, ch)) in enumerate(zip(model.channels, chs))]
    Ekept = Dict(q => prev.E[q][1:nkept(prev, q)] for q in sectors(prev.space) if nkept(prev, q) > 0)
    return product_hamiltonian(Fu, Ekept, hloc, hops, T)
end

"""
    ProductEmbedding

Dense embedding of the product basis of shell `n` into the Fock space of
sites `n, n-1, …, -1` (mode order `[site n; …; site -1]`): `cols[q′]` holds the
dense vectors of the product states of sector `q′`. Built from the dense
embedding of shell `n-1`.
"""
struct ProductEmbedding{S,T}
    basis::FockBasis
    cols::Dict{S,Matrix{T}}
end

# Dense vectors of the shell-(n-1) K states (columns, per sector) on the
# Fock space of sites n-1..-1, and the product embedding for shell n.
function _product_embedding(prev::Shell{S,T}, loc::LocalSpace{S}, Fu::FusionMap{S}, Uprev) where {S,T}
    if prev.n == -1
        # shell -1: columns are impurity eigenvectors on the impurity Fock basis
        Bp = prev.loc.basis
        Uprev = Dict{S,Matrix{T}}()
        for (q, states) in prev.loc.sectors
            U = zeros(T, length(Bp), length(states))
            U[states, :] .= isometry_block(prev.A, nothing, q)
            Uprev[q] = U[:, krange(prev, q)]
        end
        prevmodes = Bp.modes; prevstates = Bp.states
    else
        prevmodes = Uprev.basis.modes; prevstates = Uprev.basis.states
        Uprev = Dict(q => Uprev.cols[q][:, krange(prev, q)] for q in keys(Uprev.cols) if nkept(prev, q) > 0)
    end
    lb = loc.basis
    ms = vcat(lb.modes, prevmodes)
    np = length(prevmodes)
    # product of local and previous bit patterns (site n most significant)
    states = UInt64[(sl << np) | sp for sl in lb.states for sp in prevstates]
    B = FockBasis(ms, sort(states), Dict(s => i for (i, s) in enumerate(sort(states))))
    cols = Dict{S,Matrix{T}}()
    for (qs, prs) in Fu.pairs
        qq = qs.label
        U = zeros(T, length(B), dim(Fu.out, qs))
        for p in prs
            lst = loc.sectors[p.σ.label]; Up = Uprev[p.q.label]
            for (iσ, li) in enumerate(lst), s in 1:p.dq
                col = p.offset + (iσ - 1) * p.dq + s
                for (ip, sp) in enumerate(prevstates)
                    v = Up[ip, s]
                    iszero(v) && continue
                    U[B.index[(lb.states[li] << np) | sp], col] = v
                end
            end
        end
        cols[qq] = U
    end
    return ProductEmbedding{S,T}(B, cols)
end

"""
    dense_embedding(run, n) -> ProductEmbedding

Dense vectors of **all** eigenstates of shell `n` (columns per sector, K
states first) on the Fock space of sites `n..-1`. Cached; refused if the Fock
dimension exceeds `params.dense_max`. Used to build local operators at their
shell and by the tests (Anders-Schiller basis, V1, V4).
"""
function dense_embedding(run::NRGRun{S,T}, n::Integer) where {S,T}
    key = (:dense, n)
    haskey(run.cache, key) && return run.cache[key]
    sh = shell(run, n)
    if n == -1
        Bp = sh.loc.basis
        cols = Dict{S,Matrix{T}}()
        for (q, states) in sh.loc.sectors
            U = zeros(T, length(Bp), length(states))
            U[states, :] .= isometry_block(sh.A, nothing, q)
            cols[q] = U
        end
        emb = ProductEmbedding{S,T}(Bp, cols)
    else
        nf = length(sh.loc.basis) * length(dense_embedding(run, n - 1).basis)
        nf ≤ run.params.dense_max || throw(ArgumentError(
            "dense embedding of shell $n needs a Fock space of dimension $nf > dense_max = $(run.params.dense_max)"))
        prev = shell(run, n - 1)
        Pe = _product_embedding(prev, sh.loc, sh.fusion, n - 1 == -1 ? nothing : dense_embedding(run, n - 1))
        U = isometry_matrix(sh.A, sh.fusion)
        cols = Dict(q => Pe.cols[q] * getblock(U, q) for q in sectors(sh.space))
        emb = ProductEmbedding{S,T}(Pe.basis, cols)
    end
    run.cache[key] = emb
    return emb
end
