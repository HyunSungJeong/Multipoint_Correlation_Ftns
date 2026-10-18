# Two-orbital Anderson model with a Kanamori interaction (a generic local
# interaction given as an operator expression).
using MultipointNRG
U, J = 0.3, 0.05
Hint = kanamori(U = U, J = J)
m = two_orbital(ε = -(3U - 5J) / 2, Hint = Hint, Δ = 0.05)    # half filling
p = NRGParams(N = 24, Nkeep = 600)
run = run_nrg(m, p)
f = fdm(run; T = 1e-4)
N = sum(num(Mode(-1, s)) for s in (:aup, :adn, :bup, :bdn))
println("⟨N_d⟩ = ", thermal_average(run, f, N))
c = cop(Mode(-1, :aup))
r = psf(run, f, (c, c'); grid = LogGrid(emin = 1e-6, emax = 10.0, ndec = 16))
println("S[a↑, a†↑] total = ", r.total)
