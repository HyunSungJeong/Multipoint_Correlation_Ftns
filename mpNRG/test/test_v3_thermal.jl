# V3 (thermal averages, P5): without truncation the FDM is the exact thermal
# state of the finite chain.

using Test, MultipointNRG, LinearAlgebra

@testset "V3b partition function and thermal averages vs ED" begin
    m = siam(U = 0.5, εd = -0.1, Δ = 0.1)
    N = 3; T = 0.04
    run = run_nrg(m, NRGParams(N = N, Nkeep = typemax(Int)))
    f = fdm(run; T = T)
    ed = model_ed(m, run.chains, N, T)
    # Z: both with their ground-state energy subtracted
    @test f.Eref ≈ ed.E0 atol = 1e-12
    @test f.logZ ≈ log(ed.Z) rtol = 1e-12
    @test f.info[:checks].trace ≈ 1 atol = 1e-14
    Hint = m.params[:U] * n_(:up) * n_(:dn)
    q_up = comm(d_(:up), Hint)
    ops = [n_(:up), n_(:up) * n_(:dn), dd_(:dn) * f0_(:dn) + f0_(:dn)' * d_(:dn), d_(:up) * dd_(:up),
           q_up * q_up', acomm(q_up, dd_(:up)), dd_(:up) * cop(Mode(1, :up)) + cop(Mode(1, :up))' * d_(:up)]
    for o in ops
        @test thermal_average(run, f, o) ≈ ed_thermal_average(ed, o) atol = 1e-12
        @test thermal_average_product(run, f, [o]) ≈ thermal_average(run, f, o) atol = 1e-13
    end
    # products evaluated through the shell expansion of Lee Eq. (50)
    @test thermal_average_product(run, f, [d_(:up), dd_(:up), n_(:dn)]) ≈ ed_thermal_average(ed, d_(:up) * dd_(:up) * n_(:dn)) atol = 1e-12
    @test thermal_average_product(run, f, [d_(:up), dd_(:dn), d_(:dn), dd_(:up)]) ≈
          ed_thermal_average(ed, d_(:up) * dd_(:dn) * d_(:dn) * dd_(:up)) atol = 1e-12
end
