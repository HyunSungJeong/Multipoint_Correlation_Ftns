# Kondo (s-d) model: entropy flow ln2 → 0 and the spin susceptibility.
using MultipointNRG
m = kondo(J = 0.3)                               # ρJ = J/(2D)
p = NRGParams(N = 50, Nkeep = 300)
run = run_nrg(m, p)
runb = run_nrg(bath_model(m), p; chains = run.chains)
th = evenodd_average(impurity_thermodynamics_shell(run, runb))
for i in 1:6:length(th.n)
    println("T = ", th.T[i], "  S_imp = ", round(th.entropy[i], digits = 4), "  Tχ_imp = ", round(th.Tchi[i], digits = 4))
end
# a spin-spin PSF: S[S⁺, S⁻] of the impurity spin
Si = spin_ops(Mode(-1, :up), Mode(-1, :dn))
f = fdm(run; T = 1e-8)
r = psf(run, f, (Si.Sp, Si.Sm); grid = LogGrid(emin = 1e-12, emax = 10.0, ndec = 16))
println("⟨S⁺S⁻⟩ = ", r.total)
