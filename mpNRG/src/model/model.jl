# Impurity models (task sheet Sec. 5.11) and the symmetry checks of Sec. 5.2.6.

"""
    BathChannel(name, charge, hyb; coupling = nothing)

One bath flavor ν: Wilson-site modes `Mode(n, name)` with charge `charge`
(the label of the impurity flavor it couples to, Sec. 5.2.3), hybridization
function `hyb`, and the impurity operator `a_ν` it hybridizes with. The coupling
term is `t_0 (a†_ν f_{0ν} + f†_{0ν} a_ν)` with `t_0 = sqrt((1/π)∫Δ_ν)`.
`coupling = nothing` means no hybridization term (the impurity then couples
through the model's explicit site-0 term, as in the Kondo model).
"""
struct BathChannel{S<:Sector}
    name::Symbol
    charge::S
    hyb::Hybridization
    coupling::Union{Nothing,OpExpr}
end
BathChannel(name::Symbol, charge::Sector, hyb::Hybridization; coupling = nothing) =
    BathChannel{typeof(charge)}(name, charge, hyb, coupling)

"""
    ImpurityModel(G, impurity, Himp, channels; Hsite0 = 0, name = "custom", params = Dict())

A quantum impurity model with a finite local Fock space:

  - `G::SymmetryGroup` and the impurity `LocalSpace` (site -1) with mode charges;
  - `Himp`: the local Hamiltonian (an `OpExpr` on the impurity modes);
  - `channels`: the bath flavors, diagonal in flavor (Sec. 5.3);
  - `Hsite0`: an optional extra term on the impurity and Wilson site 0
    (e.g. the Kondo exchange `J S·s_0`).

At construction every term is embedded into the dense Fock space of its modes
and checked against every factor of `G` (Sec. 5.2.6); a violation raises an
error naming the term and the broken factor. There is no silent fallback to a
smaller symmetry.
"""
struct ImpurityModel{S<:Sector}
    G::SymmetryGroup
    impurity::LocalSpace{S}
    Himp::OpExpr
    channels::Vector{BathChannel{S}}
    Hsite0::OpExpr
    name::String
    params::Dict{Symbol,Any}
end

function ImpurityModel(G::SymmetryGroup, impurity::LocalSpace{S}, Himp::OpExpr, channels::Vector{BathChannel{S}};
                       Hsite0::OpExpr = OpExpr{Float64}(), name::AbstractString = "custom",
                       params::AbstractDict = Dict{Symbol,Any}(), check::Bool = true) where {S}
    labeltype(G) == S || throw(ArgumentError("impurity labels do not belong to $G"))
    allunique(c.name for c in channels) || throw(ArgumentError("channel names must be distinct"))
    m = ImpurityModel{S}(G, impurity, Himp, channels, Hsite0, String(name), Dict{Symbol,Any}(params))
    check && check_model_symmetry(m)
    return m
end

impurity_modes(m::ImpurityModel) = localmodes(m.impurity)
nchannels(m::ImpurityModel) = length(m.channels)
"""    site_modes(m, n) — the modes of Wilson site `n` in local Jordan-Wigner order."""
site_modes(m::ImpurityModel, n::Integer) = n < 0 ? impurity_modes(m) : [Mode(n, c.name) for c in m.channels]

"""    site_space(m, n) — the `LocalSpace` of Wilson site `n ≥ 0`."""
function site_space(m::ImpurityModel{S}, n::Integer) where {S}
    n < 0 && return m.impurity
    ch = Dict{Mode,S}(Mode(n, c.name) => c.charge for c in m.channels)
    return LocalSpace(n, [c.name for c in m.channels], ch)
end

"""
    charges(m, N) -> Dict{Mode,Sector}

Charges of all modes on sites `-1..N`.
"""
function charges(m::ImpurityModel{S}, N::Integer) where {S}
    ch = Dict{Mode,S}(m.impurity.charges)
    for n in 0:N, c in m.channels
        ch[Mode(n, c.name)] = c.charge
    end
    return ch
end

"""
    hybridization_term(m, c::BathChannel, t0) -> OpExpr

`t_0 (a†_ν f_{0ν} + f†_{0ν} a_ν)` for a channel with a coupling operator.
"""
function hybridization_term(m::ImpurityModel, c::BathChannel, t0::Real)
    c.coupling === nothing && return OpExpr{Float64}()
    f0 = cop(Mode(0, c.name))
    a = c.coupling
    return t0 * (a' * f0 + f0' * a)
end

"""
    site0_hamiltonian(m, chains) -> OpExpr

Everything acting on the impurity and Wilson site 0: `Himp + Hsite0 +
Σ_ν [t_0 (a†_ν f_{0ν} + h.c.) + ε_0 n_{0ν}]`.
"""
function site0_hamiltonian(m::ImpurityModel, chains::AbstractVector{WilsonChain})
    H = m.Himp + m.Hsite0
    for (c, ch) in zip(m.channels, chains)
        H = H + hybridization_term(m, c, ch.t[1]) + ch.eps[1] * num(Mode(0, c.name))
    end
    return H
end

"""
    hamiltonian_expr(m, chains, N) -> OpExpr

The full Hamiltonian of the chain with sites `-1..N` as an operator expression
(for the reference ED and for checks).
"""
function hamiltonian_expr(m::ImpurityModel, chains::AbstractVector{WilsonChain}, N::Integer)
    H = site0_hamiltonian(m, chains)
    for (c, ch) in zip(m.channels, chains), n in 1:N
        fn = cop(Mode(n, c.name)); fp = cop(Mode(n - 1, c.name))
        H = H + ch.eps[n + 1] * num(Mode(n, c.name)) + ch.t[n + 1] * (fn' * fp + fp' * fn)
    end
    return H
end

"""
    check_symmetry(expr, charges, G; term = "term", allowed = nothing)

Embed `expr` into the dense Fock space of its modes and verify that every
nonzero matrix element connects Fock states with equal labels, i.e.
`[H, Q_k] = 0` for every U(1) generator and invariance under every Z_n factor.
On failure the error names the term and the broken factor(s).
"""
function check_symmetry(expr::OpExpr, ch::AbstractDict{Mode,S}, G::SymmetryGroup;
                        term::AbstractString = "term", allowed = nothing) where {S}
    ms = modes(expr)
    isempty(ms) && return true
    B = FockBasis(ms; allowed = allowed)
    D = densematrix(expr, B; T = ComplexF64)
    labels = [statecharge(B, i, ch) for i in 1:length(B)]
    names = factornames(G)
    broken = Set{Symbol}()
    for j in axes(D, 2), i in axes(D, 1)
        abs(D[i, j]) > 1e-13 || continue
        labels[i] == labels[j] && continue
        for k in eachindex(names)
            labels[i].q[k] != labels[j].q[k] && push!(broken, names[k])
        end
    end
    isempty(broken) || throw(ArgumentError(
        "$term breaks the declared symmetry factor(s) $(collect(broken)) of $G; " *
        "drop $(join(collect(broken), ", ")) from the symmetry group (e.g. a transverse field breaks U1(:twoSz), " *
        "pairing breaks U1(:N), spin-orbit coupling breaks U1(:twoSz))"))
    return true
end

"""
    check_model_symmetry(m)

The checks of Sec. 5.2.6: the local Hamiltonian, every impurity-bath coupling,
the site-0 term and the on-site bath terms must respect `G`, and each channel
must couple only to impurity operators with its own label.
"""
function check_model_symmetry(m::ImpurityModel)
    ch = charges(m, 1)
    check_symmetry(m.Himp, ch, m.G; term = "the local Hamiltonian")
    check_symmetry(m.Hsite0, ch, m.G; term = "the site-0 term")
    for c in m.channels
        if c.coupling !== nothing
            Δ = opshift(c.coupling, ch)
            Δ == dual(c.charge) || throw(ArgumentError(
                "channel $(c.name) (charge $(c.charge)) couples to an impurity operator with shift $Δ; " *
                "per-flavor charges require the hybridization to be diagonal in flavor"))
            check_symmetry(hybridization_term(m, c, 1.0), ch, m.G; term = "the hybridization of channel $(c.name)")
        end
        check_symmetry(num(Mode(1, c.name)) + cop(Mode(1, c.name))' * cop(Mode(0, c.name)) +
                       cop(Mode(0, c.name))' * cop(Mode(1, c.name)), ch, m.G; term = "the bath of channel $(c.name)")
    end
    return true
end

"""
    wilson_chains(m, N; Λ = 4, z = 1, scheme = :zitko, kwargs...) -> Vector{WilsonChain}

One Wilson chain with sites `0..N` per channel (Sec. 5.3).
"""
wilson_chains(m::ImpurityModel, N::Integer; kwargs...) =
    [wilson_chain(c.hyb, N; kwargs...) for c in m.channels]
