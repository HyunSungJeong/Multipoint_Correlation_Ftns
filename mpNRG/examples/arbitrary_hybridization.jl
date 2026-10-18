# An arbitrary (tabulated, e.g. DMFT) hybridization: semicircular Δ(ω) on a grid.
using MultipointNRG
ωs = collect(range(-1, 1; length = 401))
Δtab = TabulatedHybridization(ωs, 0.05 .* sqrt.(max.(0, 1 .- ωs .^ 2)))
m = siam(U = 0.2, hyb = Δtab)
c = wilson_chain(Δtab, 30)
println("t_n Λ^{n/2}: ", [round(c.t[n + 1] * 4.0^(n / 2), digits = 4) for n in 20:25])
println("two-precision check: ", c.precision_check)
run = run_nrg(m, NRGParams(N = 30, Nkeep = 300))
f = fdm(run; T = 1e-6)
d = cop(Mode(-1, :up))
r = psf(run, f, (d, d'); grid = LogGrid(emin = 1e-8, emax = 10.0, ndec = 16))
println("total weight ", r.total, " = ⟨d d†⟩ = ", r.meta[:sum_rule])
