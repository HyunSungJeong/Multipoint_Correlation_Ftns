# Resonant level model (U = 0): NRG vs the exact single-particle solution.
using MultipointNRG, LinearAlgebra
m = resonant_level(εd = 0.02, Δ = 0.05)
T = 1e-4
N = chain_length_for(T; Λ = 4.0)
g = LogGrid(emin = 1e-7, emax = 10.0, ndec = 32)
d = cop(Mode(-1, :d))
zs = zaveraged_psfs(m, NRGParams(N = N, Nkeep = 64), (d, d'); T = T, nz = 4, grid = g)
ω = [-0.1, -0.02, 0.0, 0.02, 0.1]
A = broaden(zs[[1, 2]], ω; b = 0.3) .+ broaden(zs[[2, 1]], -ω; b = 0.3)
Δ = 0.05; εd = 0.02
hyb(x) = Δ / π * log(abs((1 + x) / (1 - x))) - im * Δ
for (x, a) in zip(ω, A)
    println("ω = $x  A_NRG = ", round(a, digits = 4), "  A_exact = ", round(-imag(1 / (x - εd - hyb(x))) / π, digits = 4))
end
