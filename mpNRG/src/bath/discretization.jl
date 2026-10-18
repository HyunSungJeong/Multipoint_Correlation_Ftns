# Logarithmic discretization of a hybridization function (task sheet Sec. 5.3).
#
# Intervals are bounded by ±D Λ^{-k-z} (Lee p. 23): on the positive side
#
#     ε_1 = D,   ε_j = D Λ^{2-j-z}  (j ≥ 2),   I_j = [ε_{j+1}, ε_j],
#
# with z ∈ (0, 1] (Yoshida, Whitaker & Oliveira 1990; Oliveira & Oliveira 1994;
# Žitko & Pruschke 2009, notation of their Eq. (21) and App. B). The negative
# side is the mirror image with Δ(-ω). Each interval is represented by one
# star level E_j with |V_j|² = (1/π) ∫_{I_j} Δ(ω) dω.

"""
    Discretization(; Λ = 4, z = 1, scheme = :zitko, nint = nothing)

Parameters of the logarithmic discretization.

  - `scheme = :zitko`: representative energies from Žitko & Pruschke,
    PRB 79, 085106 (2009) Eq. (22) and App. B, Eq. (B1)–(B4) (the
    prescription Lee uses, his Ref. [23]); closed form for the box, a stable
    backward integration of the App. B differential equation otherwise.
  - `scheme = :wilson`: Wilson's prescription, the Δ-weighted mean energy of
    each interval, `E_j = ∫ω Δ / ∫Δ` (Bulla, Costi & Pruschke RMP 80, 395,
    Eq. (23)); the midpoint for a box.
  - `scheme = :campo`: Campo & Oliveira, PRB 72, 104432 (2005):
    `E_j = ∫_{I_j} Δ / ∫_{I_j} Δ(ω)/ω`.

`nint` intervals per side (default: chosen by `wilson_chain` from the chain
length). The weight below the last interval, `∫_0^{ε_{nint+1}} Δ`, is added to
the last interval so that the sum rule `Σ|V|² = (1/π)∫Δ` holds exactly
(approximation: it shifts the lowest star level's weight by `O(Λ^{-nint})`).
"""
Base.@kwdef struct Discretization
    Λ::Float64 = 4.0
    z::Float64 = 1.0
    scheme::Symbol = :zitko
    nint::Union{Nothing,Int} = nothing
    function Discretization(Λ, z, scheme, nint)
        Λ > 1 || throw(ArgumentError("Λ must exceed 1"))
        0 < z ≤ 1 || throw(ArgumentError("z must lie in (0, 1]"))
        scheme in (:zitko, :wilson, :campo) || throw(ArgumentError("unknown scheme $scheme"))
        return new(Float64(Λ), Float64(z), scheme, nint)
    end
end

"""
    interval_edges(d, D, J; T = BigFloat) -> Vector{T}

Positive-side edges `ε_1 > ε_2 > ⋯ > ε_{J+1}`: `ε_1 = D`, `ε_j = D Λ^{2-j-z}`.
"""
function interval_edges(d::Discretization, D::Real, J::Integer; T::Type = BigFloat)
    Λ = T(d.Λ); z = T(d.z)
    return [j == 1 ? T(D) : T(D) * Λ^(2 - j - z) for j in 1:(J + 1)]
end

"""
    StarBath

The discretized star of one flavor: levels `E` and couplings `V` (`|V_k|² =
(1/π)∫_{I_k}Δ`), stored in extended precision. Ordered positive side first
(from high to low energy), then the negative side.
"""
struct StarBath{T<:AbstractFloat}
    E::Vector{T}
    V::Vector{T}
end

# Žitko-Pruschke representative energies on one side (s = ±1) for a box:
# E_1 = D[(1-Λ^{-z})/lnΛ + 1 - z], E_j = D Λ^{2-j-z}(1-Λ^{-1})/lnΛ (j ≥ 2),
# Žitko & Pruschke Eq. (22) and the Campo-Oliveira j ≥ 2 values it keeps.
function _zitko_box(d::Discretization, D::Real, J::Integer, ::Type{T}) where {T}
    Λ = T(d.Λ); z = T(d.z); lΛ = log(Λ)
    E = Vector{T}(undef, J)
    E[1] = T(D) * ((1 - Λ^(-z)) / lΛ + 1 - z)
    for j in 2:J
        E[j] = T(D) * Λ^(2 - j - z) * (1 - 1 / Λ) / lΛ
    end
    return E
end

"""
    zitko_energies(Δside, d, D, J; T = BigFloat, h = 1/256) -> Vector

Žitko-Pruschke representative energies for an arbitrary one-sided density
`Δside(ω)` (ω > 0), from their App. B: with `x = j + z` and the Ansatz
`E(x) = D f(x) Λ^{2-x}`,

    f′(x) = ln Λ f(x) - Λ^{x-2} ∫_{ε(x+1)}^{ε(x)} Δ(ω) dω / (D Δ(E(x))) ,

Žitko & Pruschke Eq. (B4). Forward integration is unstable (the homogeneous
solution grows as Λ^x, the "stiffness" noted there), so we integrate
*backward* from `x_max = J + 1 + 8` with the asymptotic value
`f(∞) = (1 - Λ^{-1})/ln Λ`, which requires `Δ(0⁺) > 0`. The value `f(1)` is
returned as a diagnostic (it must equal `1/Λ`). RK4 in type `T` (Float64 suffices: the backward direction is stable), step `h`.
"""
function zitko_energies(Δside, d::Discretization, D::Real, J::Integer; T::Type = Float64, h = 1 // 512)
    Λ = T(d.Λ); lΛ = log(Λ); Dt = T(D)
    εx(x) = x < 2 ? Dt : Dt * Λ^(2 - x)
    intx(x) = Δside(εx(x + 1), εx(x))     # ∫_{ε(x+1)}^{ε(x)} Δ
    function rhs(x, f)
        E = Dt * f * Λ^(2 - x)
        ρ = T(Δside(E))
        ρ > 0 || throw(ArgumentError("Δ vanishes at ω = $(Float64(E)); use scheme = :wilson or :campo"))
        return lΛ * f - Λ^(x - 2) * intx(x) / (Dt * ρ)
    end
    xmax = T(J + 1 + 8)
    f = (1 - 1 / Λ) / lΛ
    hh = T(h); x = xmax
    function rk4(x, f, δ)        # one backward step of size δ from x
        k1 = rhs(x, f); k2 = rhs(x - δ / 2, f - δ / 2 * k1)
        k3 = rhs(x - δ / 2, f - δ / 2 * k2); k4 = rhs(x - δ, f - δ * k3)
        return f - δ / 6 * (k1 + 2k2 + 2k3 + k4)
    end
    # The main integration stays on the grid xmax - k h, which contains the
    # integers where ε(x) has kinks; each target x = j + z is reached by a
    # separate partial step from the grid point just above it.
    targets = sort([T(j) + T(d.z) for j in 1:J]; rev = true)
    out = Dict{Int,T}()
    for (ti, xt) in enumerate(targets)
        while x - hh ≥ xt - hh / 1000
            f = rk4(x, f, hh); x -= hh
        end
        δ = x - xt
        out[J + 1 - ti] = δ > 0 ? rk4(x, f, δ) : f
    end
    E = [Dt * out[j] * Λ^(2 - (T(j) + T(d.z))) for j in 1:J]
    return E
end

"""
    discretize(h, d, J; T = BigFloat) -> StarBath

Discretize hybridization `h` on `J` intervals per side with scheme `d.scheme`.
"""
function discretize(h::Hybridization, d::Discretization, J::Integer; T::Type = BigFloat)
    D = bandwidth(h)
    ed = interval_edges(d, D, J; T = T)
    Es = T[]; Vs = T[]
    for s in (1, -1)
        # one-sided density and its integral on [a, b] ⊂ (0, D]
        side = _OneSide{typeof(h),T}(h, s)
        w = [side(ed[j + 1], ed[j]) for j in 1:J]
        w[J] += side(zero(T), ed[J + 1])           # remainder below the last edge
        E = if d.scheme == :zitko
            h isa BoxHybridization ? _zitko_box(d, D, J, T) :
                T.(zitko_energies(_OneSide{typeof(h),Float64}(h, s), d, D, J; T = Float64))
        elseif d.scheme == :wilson
            [_moment(h, s, ed[j + 1], ed[j], 1, T) / max(w[j], floatmin(T)) for j in 1:J]
        else
            [w[j] / max(_moment(h, s, ed[j + 1], ed[j], -1, T), floatmin(T)) for j in 1:J]
        end
        for j in 1:J
            w[j] > 0 || continue
            push!(Es, s * E[j]); push!(Vs, sqrt(w[j] / T(π)))
        end
    end
    return StarBath{T}(Es, Vs)
end

# Fix the closure for Žitko's integral: zitko_energies needs Δside(a, b) = ∫_a^b
# and Δside(ω) = Δ(ω); we pass a callable object providing both.
struct _OneSide{H,T}
    h::H
    s::Int
end
(o::_OneSide{H,T})(ω) where {H,T} = T(o.h(Float64(o.s * ω)))
(o::_OneSide{H,T})(a, b) where {H,T} = o.s == 1 ? integrate(o.h, a, b; T = T) : integrate(o.h, -b, -a; T = T)

# ∫_a^b ω^p Δ(sω) dω for p = ±1 on (0, D], by Gauss-Legendre (exact for the box).
function _moment(h::Hybridization, s::Int, a, b, p::Int, ::Type{T}) where {T}
    if h isa BoxHybridization
        return p == 1 ? T(h.Δ) * (T(b)^2 - T(a)^2) / 2 : T(h.Δ) * log(T(b) / T(a))
    end
    x, w = _gausslegendre(64, T)
    # integrate in log ω for accuracy on log intervals
    la = log(T(a)); lb = log(T(b)); m = (la + lb) / 2; r = (lb - la) / 2
    return r * sum(w[i] * exp(m + r * x[i])^(p + 1) * T(h(Float64(s * exp(m + r * x[i])))) for i in eachindex(x))
end
