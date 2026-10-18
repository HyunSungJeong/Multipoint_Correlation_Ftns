# Hybridization functions (task sheet Sec. 5.3).
#
# Δ(ω) is defined as in Lee Sec. VII A: Δ(ν) = Σ_ε π|V_ε|² δ(ν - ε), so that the
# total coupling of a channel is Σ_k |V_k|² = (1/π) ∫ dω Δ(ω).

"""
    Hybridization

A real-frequency hybridization function `Δ(ω) ≥ 0` of one flavor, with support
in `[-D, D]`. Subtypes: [`BoxHybridization`](@ref),
[`FunctionHybridization`](@ref), [`TabulatedHybridization`](@ref).
"""
abstract type Hybridization end

"""
    BoxHybridization(Δ; D = 1)

The box `Δ(ω) = Δ θ(D - |ω|)` of Kugler Eq. (79) and Lee Sec. VII A.
"""
struct BoxHybridization <: Hybridization
    Δ::Float64
    D::Float64
    BoxHybridization(Δ::Real; D::Real = 1.0) = (Δ ≥ 0 && D > 0) ? new(Float64(Δ), Float64(D)) :
        throw(ArgumentError("need Δ ≥ 0 and D > 0"))
end

"""
    FunctionHybridization(f; D = 1, nquad = 64)

An analytic hybridization `Δ(ω) = f(ω)` on `[-D, D]`. Interval integrals use
tanh-sinh quadrature (accurate also with integrable endpoint singularities;
`f` is evaluated in `Float64`, so integrals carry Float64 accuracy).
"""
struct FunctionHybridization{F} <: Hybridization
    f::F
    D::Float64
    nquad::Int
end
FunctionHybridization(f; D::Real = 1.0, nquad::Integer = 64) = FunctionHybridization(f, Float64(D), Int(nquad))

"""
    TabulatedHybridization(ω, Δ)

Hybridization tabulated on an increasing real-frequency grid (DMFT use),
interpolated linearly and zero outside `[ω[1], ω[end]]`. `D = max(|ω₁|, |ω_end|)`.
"""
struct TabulatedHybridization <: Hybridization
    ω::Vector{Float64}
    Δ::Vector{Float64}
    D::Float64
    function TabulatedHybridization(ω::AbstractVector{<:Real}, Δ::AbstractVector{<:Real})
        length(ω) == length(Δ) ≥ 2 || throw(ArgumentError("need matching grids of length ≥ 2"))
        issorted(ω; lt = <=) || throw(ArgumentError("ω must be strictly increasing"))
        all(≥(0), Δ) || throw(ArgumentError("Δ(ω) must be non-negative"))
        return new(Float64.(ω), Float64.(Δ), max(abs(ω[1]), abs(ω[end])))
    end
end

bandwidth(h::Hybridization) = h.D

"""    (h::Hybridization)(ω) — evaluate `Δ(ω)`."""
(h::BoxHybridization)(ω::Real) = abs(ω) ≤ h.D ? h.Δ : zero(h.Δ)
(h::FunctionHybridization)(ω::Real) = abs(ω) ≤ h.D ? h.f(ω) : zero(float(ω))
function (h::TabulatedHybridization)(ω::Real)
    (ω < h.ω[1] || ω > h.ω[end]) && return 0.0
    i = clamp(searchsortedlast(h.ω, ω), 1, length(h.ω) - 1)
    t = (ω - h.ω[i]) / (h.ω[i + 1] - h.ω[i])
    return (1 - t) * h.Δ[i] + t * h.Δ[i + 1]
end

# Gauss-Legendre nodes and weights on [-1, 1] by Newton iteration, in type T.
function _gausslegendre(n::Int, ::Type{T}) where {T}
    x = zeros(T, n); w = zeros(T, n)
    for i in 1:n
        z = cos(T(π) * (i - T(1) / 4) / (n + T(1) / 2))
        for _ in 1:100
            p1 = one(T); p2 = zero(T)
            for j in 1:n
                p3 = p2; p2 = p1
                p1 = ((2j - 1) * z * p2 - (j - 1) * p3) / j
            end
            pp = n * (z * p1 - p2) / (z^2 - 1)
            z1 = z; z = z1 - p1 / pp
            abs(z - z1) < eps(T) * 4 && break
        end
        p1 = one(T); p2 = zero(T)
        for j in 1:n
            p3 = p2; p2 = p1
            p1 = ((2j - 1) * z * p2 - (j - 1) * p3) / j
        end
        pp = n * (z * p1 - p2) / (z^2 - 1)
        x[i] = z; w[i] = 2 / ((1 - z^2) * pp^2)
    end
    return x, w
end

"""
    integrate(h, a, b; T = BigFloat) -> T

`∫_a^b dω Δ(ω)` (for `a < b`), exactly for the box and the table, by Gauss-Legendre
quadrature for an analytic `Δ`.
"""
function integrate(h::BoxHybridization, a::Real, b::Real; T::Type = BigFloat)
    lo = max(T(a), -T(h.D)); hi = min(T(b), T(h.D))
    return hi > lo ? T(h.Δ) * (hi - lo) : zero(T)
end
function integrate(h::FunctionHybridization, a::Real, b::Real; T::Type = BigFloat)
    lo = max(T(a), -T(h.D)); hi = min(T(b), T(h.D))
    hi > lo || return zero(T)
    # tanh-sinh (double-exponential) quadrature: robust against endpoint
    # singularities such as the √ band edges of a semicircle
    m = (hi + lo) / 2; r = (hi - lo) / 2
    s = zero(T)
    step = T(1) / 32
    for k in -160:160
        t = k * step
        u = T(π) / 2 * sinh(t)
        x = tanh(u)
        w = T(π) / 2 * cosh(t) / cosh(u)^2
        (abs(x) < 1 && w > 0) || continue
        s += w * T(h.f(Float64(m + r * x)))
    end
    return r * step * s
end
function integrate(h::TabulatedHybridization, a::Real, b::Real; T::Type = BigFloat)
    lo = max(T(a), T(h.ω[1])); hi = min(T(b), T(h.ω[end]))
    hi > lo || return zero(T)
    pts = vcat(lo, T.(h.ω[(h.ω .> lo) .& (h.ω .< hi)]), hi)
    s = zero(T)
    for i in 1:(length(pts) - 1)   # trapezoid is exact for the linear interpolant
        s += (pts[i + 1] - pts[i]) * (T(h(Float64(pts[i]))) + T(h(Float64(pts[i + 1])))) / 2
    end
    return s
end

"""
    total_coupling(h) -> Float64

`(1/π) ∫ dω Δ(ω)`, which equals `Σ_k |V_k|²` for every discretization
(V2 sum rule); `2DΔ/π` for the box.
"""
total_coupling(h::Hybridization) = Float64(integrate(h, -h.D, h.D) / big(π))
