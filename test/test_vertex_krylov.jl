# vertex_krylov task, phases V0 and V1: discrete Krylov representation of the
# 3p vertex ([VD] Algorithm 2) on the Hubbard atom (task sheet Secs. 3.3, 5, 6).
#
# Tolerances are those of the V1 row of the task sheet's Sec. 6 (guaranteed by
# [VD] Theorem 2: exact Lehmann data at T > 0).

using Test
using LinearAlgebra

@testset "vertex_krylov V0/V1 (Hubbard atom)" begin
    U = 1.0; εd = -0.3
    Ts = (0.1, 0.5)

    @testset "V0 conventions" begin
        # ζ signs of (p1..p6) for (q12, T3, T4), task sheet Sec. 2.5
        @test vk_zeta() == [1, -1, 1, -1, -1, 1]
        # PSF convention, Kugler Eq. (28), by explicit Lehmann summation (Sec. 2.2)
        m = HubbardAtomModel(U, 1 / 0.1)
        sp = atom_spectrum(m; εd = εd)
        ops = vk_atom_operators(m; spins = [:up], i12 = (:dn, :dn))
        @test psf_convention_check(sp, ops.Xd[1], ops.X[1]) ≤ 1e-15
        @test psf_convention_check(sp, ops.Xd[2], ops.X[1]) ≤ 1e-15
        @test psf_convention_check(sp, ops.Bqd, ops.Bq) ≤ 1e-15
        # composites from their definitions: q = [d, Hint] = U n↓ d↑ (atom)
        o = operators(m)
        @test ops.q[1] ≈ U * o.n_dn * o.d_up
        @test ops.Bq ≈ U * o.n_up            # q↓↓ = {q↓, d†↓} = U n↑
        # regular_sum frequency convention = [3p] Sec. III.3 (Liouvillian form)
        for (n3, n4) in ((0, 0), (2, -5), (-3, 1))
            ω3 = vk_fermionic(n3, 0.1); ω4 = vk_fermionic(n4, 0.1); ω12 = -ω3 - ω4
            z = ComplexF64[im * ω12, im * ω3, im * ω4]
            for (x, y) in ((ops.q[1], ops.X[2]), (ops.d[1], ops.X[1]))
                rs = regular_sum(sp, (ops.Bq, x, y), z; is_fermionic = [false, true, true])
                @test abs(rs - liouvillian_regular(sp, (ops.Bq, x, y), [ω12, ω3, ω4])) ≤ 1e-13
            end
        end
    end

    @testset "V0 grids: symmetric, odd bin map" begin
        pos = [0.3, -0.3 + 1e-13, 1.2, 0.0, 1e-12, -0.7, 0.7000000000004, 2.5]
        g = ExactGrid(pos; degtol = 1e-10)
        @test check_grid(g.b, g.bar).ok
        @test g.spread ≤ 1e-12
        for x in pos
            @test bin_of(g, -x) == g.bar[bin_of(g, x)]
            @test abs(g.b[bin_of(g, x)] - x) ≤ 1e-10
        end
        @test_throws ArgumentError bin_of(g, 5.0)
    end

    for T in Ts, chan in (:updn, :upup)
        m = HubbardAtomModel(U, 1 / T)
        sp = atom_spectrum(m; εd = εd)
        d = discrete_data_atom(m, sp; σ = :up, channel = chan, εd = εd)

        @testset "V0 sanity T=$T channel=$chan" begin
            s = data_sanity(d)
            @test s.grid_ok && s.bar_involution
            @test s.herm_err ≤ 1e-14
            @test s.min_eig_rel ≥ -1e-13
            @test s.wB_min ≥ -1e-14 && s.wBd_min ≥ -1e-14 && s.wB_imag ≤ 1e-14
            @test s.sumrule_err ≤ 1e-13
            @test s.sumrule[1, 1] ≈ 1
            @test s.totals2_err ≤ 1e-13
            @test s.totals3_err ≤ 1e-13
            @test s.C5 ≤ 1e-12
            @test s.boltzmann_diff ≤ 1e-13          # exact data: the shortcut agrees
            # (I3) entries reproduce operator-level correlators: direct caches vs regular_sum
            ops = vk_atom_operators(m; spins = [:up], i12 = d.meta[:i12])
            caches = direct_caches(d)
            for (n3, n4) in ((0, 0), (1, -4))
                ω3 = vk_fermionic(n3, T); ω4 = vk_fermionic(n4, T); ω12 = -ω3 - ω4
                G = direct_correlators(caches, 1, ω12, ω3, ω4)
                z = ComplexF64[im * ω12, im * ω3, im * ω4]
                for (x, xo) in ((:q, ops.q[1]), (:d, ops.d[1])), (y, yo) in ((:qd, ops.X[2]), (:dd, ops.X[1]))
                    ref = regular_sum(sp, (ops.Bq, xo, yo), z; is_fermionic = [false, true, true])
                    @test abs(G[(x, y)][1, 1] - ref) ≤ 1e-13
                end
            end
        end

        @testset "V1 T=$T channel=$chan" begin
            res = vertex_krylov(d)
            cf = atom_closed_form(res)
            @test cf.L == 1
            @test cf.resid < 1e-12
            @test cf.J_err ≤ 1e-12 && cf.N2_err ≤ 1e-12 && cf.σ_err ≤ 1e-12
            c = vk_checks(res)
            @test c[:C1] ≤ 1e-11
            @test c[:C2] ≤ 1e-11
            @test c[:C3] ≤ 1e-11
            @test c[:C4] ≤ 1e-11
            @test c[:C5] ≤ 1e-12
            @test c[:C6] ≤ 1e-12
            @test c[:C7] ≤ 1e-11
            @test c[:C8] ≤ 1e-10
            @test c[:C8_parts].perm ≤ 1e-10       # term by term (Prop. 10)
            # the (↑↑) channel of the atom has K ≡ 0 (B = U n↓ is conserved and commutes
            # with d↑, q↑); C8 is then measured on the permutation-term scale
            @test c[:C8_parts].degenerate == (chan === :upup)
            # bosonic legs of the atom are trivial: J_B is 1×1 with entry 0
            @test nblocks(res.ch.JB) == 1 && abs(res.ch.JB.eps[1][1]) ≤ 1e-15
            # stabilization: Ω = span{v}, nothing changes
            chs = chains(d; stabilize = true)
            @test chs.diag[:Omega_dim] == 1
            @test chs.diag[:Omega_overlap] ≤ 1e-10
            @test abs(chs.Jbar.eps[1][1] - res.ch.Jbar.eps[1][1]) ≤ 1e-12
            # BigFloat particle chain (precision 128)
            setprecision(BigFloat, 128) do
                chb = chains(d; T = BigFloat)
                @test nblocks(chb.Jbar) == 1
                @test abs(Float64(chb.Jbar.eps[1][1, 1]) - res.ch.Jbar.eps[1][1, 1]) ≤ 1e-14
                @test abs(Float64(chb.Nbar[1, 1]) - res.ch.Nbar[1, 1]) ≤ 1e-14
            end
        end
    end

    @testset "V1 helpers: resolvents and Lanczos" begin
        # resolvent_col / resolvent_row against a dense inverse, k = 1 and k = 2
        J1 = BlockJacobi(1, [fill(0.3, 1, 1), fill(-0.2, 1, 1), fill(1.1, 1, 1)],
                         [fill(0.5, 1, 1), fill(0.7, 1, 1)], [fill(0.5, 1, 1), fill(0.7, 1, 1)])
        z = 0.4 + 0.3im
        Rinv = inv(z * I - jacobi_dense(J1))
        @test maximum(abs, resolvent_col(J1, z) - Rinv[:, 1:1]) ≤ 1e-14
        @test maximum(abs, resolvent_row(hole_jacobi(J1), z) - inv(z * I - jacobi_dense(hole_jacobi(J1)))[1:1, :]) ≤ 1e-14
        A = [0.2 0.1im; -0.1im 0.5]; Bm = [0.4 0.0; 0.1 0.3]
        J2 = BlockJacobi(2, [A, -A], [Bm], [Matrix(Bm')])
        R2 = inv(z * I - jacobi_dense(J2))
        @test maximum(abs, resolvent_col(J2, z) - R2[:, 1:2]) ≤ 1e-14
        @test maximum(abs, resolvent_row(J2, z) - R2[1:2, :]) ≤ 1e-14
        @test jacobi_dense(hole_jacobi(J2)) ≈ -transpose(jacobi_dense(J2))
        # projected Lanczos: terminates on the cyclic space of ΠDΠ and reproduces the measure
        D = [-1.0, -0.5, 0.0, 0.5, 1.0, 1.0]
        v = normalize(ones(6)); u = normalize([1.0, -2, 0.5, 1, 0.3, 0.3]); u -= v * (v' * u); u = normalize(u)
        pl = projected_lanczos(D, reshape(u, :, 1), reshape(v, :, 1))
        @test pl.diag[:terminated]
        @test pl.diag[:orth_loss] ≤ 1e-14
        @test pl.diag[:proj_leak] ≤ 1e-14
        Jp = BlockJacobi(1, pl.eps, pl.t, [Matrix(t') for t in pl.t])
        P = I - v * v'
        ref = u' * inv(z * I - P * Diagonal(D) * P + v * v') * u   # resolvent of PDP on ran P (v-direction shifted away)
        @test abs(resolvent_col(Jp, z)[1] - ref) ≤ 1e-13
        # deflation is detected for N > 1
        # (block {(e1+e3)/√2, e2} of diag(0,1,2): the next block has rank 1 but is nonzero)
        @test_throws ErrorException projected_lanczos([0.0, 1.0, 2.0], [1/sqrt(2) 0.0; 0.0 1.0; 1/sqrt(2) 0.0], nothing)
    end

    @testset "V1 N=2 block test (spin-unresolved atom)" begin
        T = 0.1
        m = HubbardAtomModel(U, 1 / T)
        for h in (0.0, 0.07)
            sp = atom_spectrum(m; εd = εd, h = h)
            d2 = discrete_data_atom(m, sp; σ = [:up, :dn], i12 = (:dn, :dn), εd = εd, h = h)
            r2 = vertex_krylov(d2)
            c2 = vk_checks(r2)
            for k in (:C1, :C3, :C4, :C7)
                @test c2[k] ≤ 1e-11
            end
            @test c2[:C5] ≤ 1e-12 && c2[:C6] ≤ 1e-12 && c2[:C8] ≤ 1e-10
            # block-diagonal: reproduces the N = 1 results (B = q↓↓ in both)
            ru = vertex_krylov(discrete_data_atom(m, sp; σ = :up, channel = :updn, εd = εd, h = h))
            rd = vertex_krylov(discrete_data_atom(m, sp; σ = :dn, channel = :upup, εd = εd, h = h))
            bu = K_box(ru); bd = K_box(rd); b2 = c2[:box]
            msk = .!isnan.(real.(bu.Kd))
            @test maximum(abs, b2.Kt[1, 1, :, :][msk] - bu.Kt[msk]) ≤ 1e-12
            @test maximum(abs, b2.Kt[2, 2, :, :][msk] - bd.Kt[msk]) ≤ 1e-12
            @test maximum(abs, b2.Kt[1, 2, :, :][msk]) ≤ 1e-14
            @test maximum(abs, b2.Kt[2, 1, :, :][msk]) ≤ 1e-14
            for z in vk_testpoints(r2.data)
                Sz = sigma_chain(r2.ch, z)
                @test maximum(abs, Sz - Diagonal([sigma_chain(ru.ch, z)[1], sigma_chain(rd.ch, z)[1]])) ≤ 1e-12
            end
        end
        # rotated, complex orbital basis: non-diagonal and asymmetric K exercise the
        # p2 transposition (VD8) and the index contractions of (VD5)
        θ, φ = 0.4, 0.9
        rot = [cos(θ) exp(im * φ) * sin(θ); -exp(-im * φ) * sin(θ) cos(θ)]
        sp = atom_spectrum(m; εd = εd, h = 0.07)
        dr = discrete_data_atom(m, sp; σ = [:up, :dn], i12 = (1, 1), εd = εd, h = 0.07, rot = rot)
        cr = vk_checks(vertex_krylov(dr))
        @test cr[:C1] ≤ 1e-11 && cr[:C3] ≤ 1e-11 && cr[:C4] ≤ 1e-11 && cr[:C7] ≤ 1e-11 && cr[:C6] ≤ 1e-12
        @test cr[:C8] ≤ 1e-10
        bx = cr[:box]; msk = .!isnan.(real.(bx.Kd[1, 1, :, :]))
        @test maximum(abs, bx.Kd[1, 2, :, :][msk] - bx.Kd[2, 1, :, :][msk]) > 1e-2   # genuinely asymmetric
    end
end
