# Built-in models (task sheet Sec. 5.11).

"""
    spin_symmetry(symmetry) -> (G, charge(σ))

Symmetry groups for spinful models with flavors `:up`, `:dn`:

  - `:charge_spin`: `U1(:N) × U1(:twoSz) × Z2(:Pf)`, `Q_σ = (1, ±1, 1)`;
  - `:charge`:      `U1(:N) × Z2(:Pf)`;
  - `:spin`:        `U1(:twoSz) × Z2(:Pf)`;
  - `:flavor`:      `U1(:Nup) × U1(:Ndn) × Z2(:Pf)` (per-flavor charges);
  - `:parity`:      `Z2(:Pf)` only.
"""
function spin_symmetry(symmetry::Symbol)
    if symmetry == :charge_spin
        G = SymmetryGroup(U1(:N), U1(:twoSz))
        return G, σ -> sector(G; N = 1, twoSz = σ == :up ? 1 : -1)
    elseif symmetry == :charge
        G = SymmetryGroup(U1(:N))
        return G, σ -> sector(G; N = 1)
    elseif symmetry == :spin
        G = SymmetryGroup(U1(:twoSz))
        return G, σ -> sector(G; twoSz = σ == :up ? 1 : -1, Pf = 1)
    elseif symmetry == :flavor
        G = SymmetryGroup(U1(:Nup), U1(:Ndn))
        return G, σ -> σ == :up ? sector(G; Nup = 1, Pf = 1) : sector(G; Ndn = 1, Pf = 1)
    elseif symmetry == :parity
        G = SymmetryGroup()
        return G, σ -> sector(G; Pf = 1)
    end
    throw(ArgumentError("unknown symmetry $symmetry"))
end

_hyb(x::Hybridization) = x
_hyb(x::Real) = BoxHybridization(x)

"""
    siam(; U, εd = -U/2, B = 0, Δ = 0.1, D = 1, hyb = nothing, symmetry = :charge_spin)

Single-impurity Anderson model, Lee Eq. (88):

    H = Σ_σ ε_d n_dσ + U n_d↑ n_d↓ - (B/2)(n_d↑ - n_d↓) + Σ_σ t_0 (d†_σ f_{0σ} + h.c.) + bath ,

with a box hybridization `Δ θ(D - |ω|)` per spin unless `hyb` is given (a
`Hybridization`, or a `Dict(:up => h↑, :dn => h↓)` for per-spin
hybridizations). The field `B` couples to the impurity only.
"""
function siam(; U::Real, εd::Real = -U / 2, B::Real = 0.0, Δ::Real = 0.1, D::Real = 1.0,
              hyb = nothing, symmetry::Symbol = :charge_spin)
    G, Q = spin_symmetry(symmetry)
    S = labeltype(G)
    imp = LocalSpace(-1, [:up, :dn], Dict{Mode,S}(Mode(-1, :up) => Q(:up), Mode(-1, :dn) => Q(:dn)))
    du = Mode(-1, :up); dd = Mode(-1, :dn)
    H = εd * (num(du) + num(dd)) + U * num(du) * num(dd) - (B / 2) * (num(du) - num(dd))
    hy = hyb === nothing ? Dict(:up => BoxHybridization(Δ; D = D), :dn => BoxHybridization(Δ; D = D)) :
         hyb isa AbstractDict ? Dict(k => _hyb(v) for (k, v) in hyb) : Dict(:up => _hyb(hyb), :dn => _hyb(hyb))
    chans = [BathChannel(σ, Q(σ), hy[σ]; coupling = cop(Mode(-1, σ))) for σ in (:up, :dn)]
    return ImpurityModel(G, imp, H, chans; name = "SIAM",
                         params = Dict(:U => U, :εd => εd, :B => B, :Δ => Δ, :D => D, :symmetry => symmetry))
end

"""
    resonant_level(; εd = 0, Δ = 0.1, D = 1, spinful = false)

The resonant level model (`U = 0`): one spinless level (`spinful = false`,
symmetry `U1(:N) × Z2(:Pf)`) or the `U = 0` SIAM.
"""
function resonant_level(; εd::Real = 0.0, Δ::Real = 0.1, D::Real = 1.0, spinful::Bool = false)
    spinful && return siam(U = 0.0, εd = εd, Δ = Δ, D = D)
    G = SymmetryGroup(U1(:N))
    S = labeltype(G)
    q = sector(G; N = 1)
    d = Mode(-1, :d)
    imp = LocalSpace(-1, [:d], Dict{Mode,S}(d => q))
    chans = [BathChannel(:d, q, BoxHybridization(Δ; D = D); coupling = cop(d))]
    return ImpurityModel(G, imp, εd * num(d), chans; name = "RLM", params = Dict(:εd => εd, :Δ => Δ, :D => D))
end

"""
    spin_ops(modes_up, modes_dn) -> (Sx, Sy, Sz, Sp, Sm)

Spin operators `S = ½ c† σ c` of a spinful orbital.
"""
function spin_ops(up::Mode, dn::Mode)
    Sp = cdag(up) * cop(dn); Sm = cdag(dn) * cop(up)
    Sz = 0.5 * (num(up) - num(dn))
    return (Sx = 0.5 * (Sp + Sm), Sp = Sp, Sm = Sm, Sz = Sz)
end

"""
    kondo(; J, B = 0, D = 1, symmetry = :charge_spin)

The Kondo (s-d) model `H = J S_imp · s_0 - B S^z_imp` with a flat band of
half-bandwidth `D`. The impurity is a spin-1/2 represented by a fermionic level
restricted to single occupancy (impurity local space spanned by `d†_↑|0⟩`,
`d†_↓|0⟩`; the restriction is exact since `H` conserves `n_d`). The Wilson chain
is that of the normalized band (site 0 is `f_0 = Σ_k c_k / √N`); `J` is in
units of the band (ρ J = J/(2D) for the box).
"""
function kondo(; J::Real, B::Real = 0.0, D::Real = 1.0, symmetry::Symbol = :charge_spin)
    G, Q = spin_symmetry(symmetry)
    S = labeltype(G)
    du = Mode(-1, :up); dd = Mode(-1, :dn)
    # single occupancy: bit patterns 01 (dn) and 10 (up) of the (up, dn) basis
    imp = LocalSpace(-1, [:up, :dn], Dict{Mode,S}(du => Q(:up), dd => Q(:dn)); allowed = UInt64[1, 2])
    Si = spin_ops(du, dd); s0 = spin_ops(Mode(0, :up), Mode(0, :dn))
    Hs0 = J * (0.5 * (Si.Sp * s0.Sm + Si.Sm * s0.Sp) + Si.Sz * s0.Sz)
    chans = [BathChannel(σ, Q(σ), BoxHybridization(1 / (2D) * π; D = D)) for σ in (:up, :dn)]
    return ImpurityModel(G, imp, -B * Si.Sz, chans; Hsite0 = Hs0, name = "Kondo",
                         params = Dict(:J => J, :B => B, :D => D, :symmetry => symmetry))
end

"""
    two_orbital(; ε, Hint, Δ = 0.1, D = 1, symmetry = :charge_spin)

Two-orbital Anderson model with orbitals `a`, `b` (impurity flavors
`:aup, :adn, :bup, :bdn`, one bath channel per flavor) and a generic local
interaction `Hint` (an `OpExpr`, e.g. Kanamori; see `examples/`). `ε` is a
level energy or a pair `(εa, εb)`.
"""
function two_orbital(; ε = 0.0, Hint::OpExpr, Δ::Real = 0.1, D::Real = 1.0, symmetry::Symbol = :charge_spin)
    G, Q = spin_symmetry(symmetry)
    S = labeltype(G)
    fl = (:aup, :adn, :bup, :bdn)
    spin(f) = endswith(String(f), "up") ? :up : :dn
    ch = Dict{Mode,S}(Mode(-1, f) => Q(spin(f)) for f in fl)
    imp = LocalSpace(-1, collect(fl), ch)
    εa, εb = ε isa Tuple ? ε : (ε, ε)
    H = εa * (num(Mode(-1, :aup)) + num(Mode(-1, :adn))) + εb * (num(Mode(-1, :bup)) + num(Mode(-1, :bdn))) + Hint
    chans = [BathChannel(f, Q(spin(f)), BoxHybridization(Δ; D = D); coupling = cop(Mode(-1, f))) for f in fl]
    return ImpurityModel(G, imp, H, chans; name = "two-orbital AIM", params = Dict(:ε => ε, :Δ => Δ, :D => D))
end

"""
    kanamori(; U, Up = U - 2J, J) -> OpExpr

The Kanamori interaction for orbitals `a`, `b` (an example of a generic local
interaction for [`two_orbital`](@ref)): intra-orbital `U`, inter-orbital `U′`
(opposite spins) and `U′ - J` (equal spins), spin flip and pair hopping `J`.
"""
function kanamori(; U::Real, J::Real, Up::Real = U - 2J)
    n(f) = num(Mode(-1, f)); c(f) = cop(Mode(-1, f))
    H = U * (n(:aup) * n(:adn) + n(:bup) * n(:bdn))
    H = H + Up * (n(:aup) * n(:bdn) + n(:adn) * n(:bup)) + (Up - J) * (n(:aup) * n(:bup) + n(:adn) * n(:bdn))
    H = H - J * (c(:aup)' * c(:adn) * c(:bdn)' * c(:bup) + c(:bup)' * c(:bdn) * c(:adn)' * c(:aup))
    H = H + J * (c(:aup)' * c(:adn)' * c(:bdn) * c(:bup) + c(:bup)' * c(:bdn)' * c(:adn) * c(:aup))
    return H
end
