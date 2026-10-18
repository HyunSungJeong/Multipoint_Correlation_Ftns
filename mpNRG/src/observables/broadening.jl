# Broadening of 2p PSFs (task sheet Sec. 5.9; Lee App. C, Eq. (C1); Lee &
# Weichselbaum PRB 94, 235127 (2016); Weichselbaum & von Delft 2007).
# 3p and 4p PSFs are not broadened (Lee Sec. VI B; out of scope).

"""
    delta_SL(ε′, ε̄, b) — symmetric log-Gaussian kernel, Lee Eq. (C1):

    δ_SL(ε′, ε̄) = θ(ε′ε̄) / (√π b |ε̄|) exp[-(ln|ε̄/ε′|/b - b/4)²] ,

normalized in `ε′` for every `ε̄ ≠ 0`.
"""
@inline function delta_SL(εp::Real, εb::Real, b::Real)
    εp * εb > 0 || return 0.0
    return exp(-(log(abs(εb / εp)) / b - b / 4)^2) / (sqrt(π) * b * abs(εb))
end

"""    delta_F(ε, ε′, bF) — derivative of the Fermi function, Lee Eq. (C1): `(2b_F)^{-1}(1 + cosh((ε-ε′)/b_F))^{-1}`."""
@inline delta_F(ε::Real, εp::Real, bF::Real) = 1 / (2bF * (1 + cosh((ε - εp) / bF)))

"""
    broaden(r::PSFResult, ω; b = 0.6, bF = nothing, nx = 401) -> Vector

Broadened 2p spectral function `A(ω) = Σ_ε̄ S(ε̄) ∫dε′ δ_F(ω, ε′) δ_SL(ε′, ε̄)`
(Lee App. C, Eq. (C1)): a symmetric log-Gaussian of width `b` in `ln|ε|`
convolved with the derivative of the Fermi function of width `bF` (default
`T/2` of the run's temperature; `bF = 0` disables it), which produces the
linear crossover below `T`. The zero bin is broadened by `δ_F` alone. The
convolution is done for each `ω` by a trapezoid rule in `x = (ε′-ω)/b_F` on
`[-40, 40]` with `nx` points (δ_F decays as `e^{-|x|}`), evaluating the
log-Gaussian sum exactly at every `ε′`. Approximation: broadening (smooths
discretization artefacts; it does not preserve the discrete structure).
"""
function broaden(r::PSFResult, ω::AbstractVector{<:Real}; b::Real = 0.6, bF = nothing, nx::Integer = 401)
    r.ℓ == 2 || throw(ArgumentError("only 2p PSFs are broadened (Lee Sec. VI B)"))
    r.mode === :binned || throw(ArgumentError("broaden needs a binned PSF"))
    g = r.grid
    h = histogram(r)
    bFv = bF === nothing ? r.meta[:T] / 2 : Float64(bF)
    cs = centres(g)
    nzb = [i for i in eachindex(h) if !iszero(h[i]) && i != zerobin(g)]
    T = eltype(h)
    Asl(x) = sum((h[i] * delta_SL(x, cs[i], b) for i in nzb); init = zero(T))
    w0 = h[zerobin(g)]
    out = zeros(T, length(ω))
    if bFv > 0
        xs = range(-40, 40; length = nx); dx = step(xs)
        kern = [dx / (2 * (1 + cosh(x))) for x in xs]       # δ_F in units of x, normalized
        for (k, x0) in enumerate(ω)
            out[k] = sum(kern[j] * Asl(x0 + bFv * xs[j]) for j in eachindex(xs)) + w0 * delta_F(x0, 0.0, bFv)
        end
    else
        for (k, x0) in enumerate(ω)
            out[k] = Asl(x0)
        end
    end
    return out
end
