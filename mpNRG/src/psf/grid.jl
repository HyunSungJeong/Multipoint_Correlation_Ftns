# The logarithmic frequency grid for binning and slicing (task sheet Sec. 4.5;
# Lee Sec. V B, p. 13).

"""
    LogGrid(; emin, emax = 10, ndec = 32, ztol = min(1e-14, emin / 10))

Lee's grid (p. 13): `ε[±m] = ±10^{(|m|-1)/n_dec} ε_min` for `m ≠ 0`, `ε[0] = 0`,
truncated to `|ε[m]| ≤ ε_max`. Each grid point is the centre of its bin on the
logarithmic axis (Lee Fig. 1(a)): bin `m ≥ 1` collects `|ε| ∈
[10^{(m-1.5)/n_dec}, 10^{(m-0.5)/n_dec}) ε_min`. The zero bin collects
`|ε| ≤ ztol` (weight at zero frequency up to numerical precision, needed for
anomalous Matsubara terms and disconnected parts); the default `ztol = 1e-14`
lies above the rounding noise of energy differences between degenerate levels
(energies are O(1) at the first shells) and below `emin`. Approximation (binning,
Lee Sec. V B): every weight is moved to its bin centre; weight below the first
bin (`ztol < |ε| < ε_min 10^{-1/(2n_dec)}`) goes to bin ±1 and weight above the
last bin to the outermost bin; such clampings are counted.

One grid is shared by all permutations and all `ℓ` in a run (Lee p. 18): every
histogram stores its grid, and combining histograms on different grids is an
error.
"""
struct LogGrid
    emin::Float64
    emax::Float64
    ndec::Int
    ztol::Float64
    M::Int
    centres::Vector{Float64}
end

function LogGrid(; emin::Real, emax::Real = 10.0, ndec::Integer = 32, ztol::Real = min(1e-14, emin / 10))
    0 < emin < emax || throw(ArgumentError("need 0 < emin < emax"))
    0 ≤ ztol < emin || throw(ArgumentError("need 0 ≤ ztol < emin"))
    M = floor(Int, ndec * log10(emax / emin) + 1e-9) + 1
    pos = [10.0^((m - 1) / ndec) * emin for m in 1:M]
    centres = vcat(-reverse(pos), 0.0, pos)
    return LogGrid(Float64(emin), Float64(emax), Int(ndec), Float64(ztol), M, centres)
end

Base.:(==)(a::LogGrid, b::LogGrid) = a.emin == b.emin && a.emax == b.emax && a.ndec == b.ndec && a.ztol == b.ztol
Base.hash(g::LogGrid, h::UInt) = hash((g.emin, g.emax, g.ndec, g.ztol), h)
Base.show(io::IO, g::LogGrid) =
    print(io, "LogGrid(emin=$(g.emin), emax=$(g.emax), ndec=$(g.ndec), ztol=$(g.ztol), $(nbins(g)) bins)")

"""    nbins(g) — number of bins `2M + 1`."""
nbins(g::LogGrid) = 2g.M + 1
"""    zerobin(g) — index of the zero bin."""
zerobin(g::LogGrid) = g.M + 1
"""    centres(g) — bin centres, ascending (index `i` ↔ `m = i - M - 1`)."""
centres(g::LogGrid) = g.centres

"""
    binindex(g, x) -> (index, clamped::Bool)

Bin of frequency `x`: the zero bin for `|x| ≤ ztol`, else the bin whose centre
is nearest on the logarithmic axis, clamped to `±1 … ±M`.
"""
@inline function binindex(g::LogGrid, x::Real)
    ax = abs(x)
    ax ≤ g.ztol && return (g.M + 1, false)
    m = round(Int, g.ndec * log10(ax / g.emin)) + 1
    clamped = m < 1 || m > g.M
    m = clamp(m, 1, g.M)
    return (g.M + 1 + (x > 0 ? m : -m), clamped)
end
