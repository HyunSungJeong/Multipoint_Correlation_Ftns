# SIAM: 2p, 3p and 4p PSFs, thermodynamics and the spectral function.
# Run with: julia -t 4 --project=mpNRG mpNRG/examples/siam.jl  (about 20 min on 2 threads, mostly the 4p PSF)
using MultipointNRG

m = siam(U = 0.2, Δ = 0.04)                     # εd = -U/2, box hybridization, D = 1
T = 1e-3
# The 4p PSF dominates the run time; it grows quickly with N_keep and as ~N^3 with
# the chain length, i.e. with ln(1/T) (timing table in STATUS.md).
p = NRGParams(N = chain_length_for(T; Λ = 4.0), Nkeep = 100)
run = run_nrg(m, p)
f = fdm(run; T = T)
println("n₀ = ", run.n0, ", kept states per shell: ", [totalkept(sh) for sh in run.shells])
println("⟨n_d⟩ = ", thermal_average(run, f, num(Mode(-1, :up)) + num(Mode(-1, :dn))))

d(σ) = cop(Mode(-1, σ))
g = LogGrid(emin = 1e-5, emax = 10.0, ndec = 16)
S2 = psf(run, f, (d(:up), d(:up)'); grid = g)
S3 = psf(run, f, (d(:up), d(:up)', num(Mode(-1, :dn))); grid = g)
S4 = psf(run, f, (d(:up), d(:up)', d(:dn), d(:dn)'); grid = g)
for S in (S2, S3, S4)
    println(S.ℓ, "p: total weight ", S.total, "  (FDM sum rule ", S.meta[:sum_rule], ")  cost ", S.meta[:cost])
end

ω = collect(range(-0.3, 0.3; length = 61))
A = broaden(S2, ω) .+ broaden(psf(run, f, (d(:up)', d(:up)); grid = g), -ω)
println("A(0) π Δ = ", A[31] * π * m.params[:Δ])

runb = run_nrg(bath_model(m), p; chains = run.chains)
th = evenodd_average(impurity_thermodynamics_shell(run, runb))
println("S_imp at the highest and lowest T: ", th.entropy[1], " ", th.entropy[end])
