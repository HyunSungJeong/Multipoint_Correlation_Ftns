# The Hubbard dimer: D1-D11 of the dimer task sheet.
#
# Paper equations are Kugler, Lee & von Delft, PRX 11, 041006 (2021), read from
# the printed page. The dimer closed forms are the task sheet's own; they are
# checked here against the 16-state exact diagonalisation, never assumed.

const DIMER_PARAMS = [(0.7, 2.3, 3.0), (0.3, 4.0, 10.0), (1.0, 0.5, 1.0)]

_M(v...) = MatsubaraFreq.(collect(v))
# generic, then one point per δ channel, then two channels at once
const DIMER_FREQS = [_M(1, 3, 5, -9), _M(1, -1, 3, -3), _M(1, 3, -1, -3),
                     _M(1, 3, -3, -1), _M(1, -1, 1, -1), _M(3, -1, -3, 1)]
const SPINS = (:updn, :upup)

@testset "Hubbard dimer, D1-D11" begin

    @testset "operators: Jordan-Wigner, anticommutators, symmetries" begin
        m = HubbardDimerModel(0.7, 2.3, 3.0)
        ops = operators(m)
        I16 = Matrix(1.0I, 16, 16)
        for a in 1:4, b in 1:4
            @test ops.c[a] * ops.c[b]' + ops.c[b]' * ops.c[a] ≈ (a == b) * I16 atol = 1e-14
            @test norm(ops.c[a] * ops.c[b] + ops.c[b] * ops.c[a]) < 1e-14
        end
        # momentum operators are canonical too
        for a in 1:4, b in 1:4
            @test ops.ck[a] * ops.ck[b]' + ops.ck[b]' * ops.ck[a] ≈ (a == b) * I16 atol = 1e-14
        end
        # without strings the modes commute instead: the assertion catches it
        bad = jordan_wigner(4; strings = false)
        @test norm(bad[1] * bad[3] + bad[3] * bad[1]) > 1
        # symmetries of H
        @test norm(ops.H * ops.N - ops.N * ops.H) < 1e-13
        @test norm(ops.H * ops.Sz - ops.Sz * ops.H) < 1e-13
        P = site_exchange(ops)
        @test P * P ≈ I16
        @test norm(ops.H * P - P * ops.H) < 1e-13
        for σ in (:up, :dn)
            @test P * site_op(ops, 1, σ) * P ≈ site_op(ops, 2, σ)
            @test P * momentum_op(ops, 0, σ) * P ≈ momentum_op(ops, 0, σ)
            @test P * momentum_op(ops, 1, σ) * P ≈ -momentum_op(ops, 1, σ)
        end
        @test_throws ArgumentError site_op(ops, 3, :up)
        @test_throws ArgumentError momentum_op(ops, 2, :up)
        @test_throws ArgumentError HubbardDimerModel(1, 1, 0)
    end

    @testset "D1: spectrum and Z" begin
        for (t, U, β) in DIMER_PARAMS
            m = HubbardDimerModel(t, U, β)
            sp = spectrum(m)
            lev = sort(vcat([fill(e, d) for (e, d) in dimer_levels_analytic(m)]...))
            @test length(lev) == 16
            @test maximum(abs, sp.E - lev) < 1e-13
            @test sp.Z ≈ dimer_partition_function_analytic(m) rtol = 1e-13
            E0 = -U / 2 - dimer_c(m)
            @test minimum(sp.E) ≈ E0 atol = 1e-13
            @test dimer_partition_function_analytic(m; shifted = true) ≈
                  sp.Z * exp(β * E0) rtol = 1e-13
            # ground state: nondegenerate N = 2 singlet
            ops = operators(m)
            @test sp.E[2] - sp.E[1] > 1e-3
            @test to_eigenbasis(sp, ops.N)[1, 1] ≈ 2 atol = 1e-12
        end
        # the shifted Z does not overflow where the plain one does
        m = HubbardDimerModel(0.3, 4.0, 1000.0)
        @test isinf(dimer_partition_function_analytic(m))
        @test isfinite(dimer_partition_function_analytic(m; shifted = true))
        @test all(isfinite, spectrum(m).ρ)
    end

    @testset "D2: G_k against the closed form" begin
        for (t, U, β) in DIMER_PARAMS
            m = HubbardDimerModel(t, U, β)
            ctx = DimerContext(m)
            for q in 0:1, n in -6:5, σ in (:up, :dn)
                ν = fermionic_freq(n)
                @test abs(dimer_propagator(ctx, q, ν; σ = σ) -
                          dimer_propagator_exact(m, q, ν)) < 1e-14
            end
            # G_π(iν) = -G_0(-iν) holds in the computed objects too
            for n in -3:3
                ν = fermionic_freq(n)
                @test dimer_propagator(ctx, 1, ν) ≈
                      -dimer_propagator(ctx, 0, MatsubaraFreq(-ν.m)) atol = 1e-14
            end
            # sum rule: residues add to 1, i.e. G(z) → 1/z
            z = 1e7im
            @test z * dimer_propagator_exact(m, 0, z) ≈ 1 atol = 1e-6
        end
        # limits of the closed form
        m = HubbardDimerModel(0.8, 0.0, 2.0)                       # U = 0
        @test all(dimer_propagator_exact(m, 0, 0.3im + x) ≈ 1 / (0.3im + x + 0.8) for x in -1:1)
        for β in (0.5, 4.0)                                        # t = 0: the atom
            m = HubbardDimerModel(0.0, 2.6, β)
            @test dimer_propagator_exact(m, 0, 0.7im) ≈
                  0.5 * (1 / (0.7im + 1.3) + 1 / (0.7im - 1.3))
        end
        m = HubbardDimerModel(0.4, 2.0, 200.0)                     # T → 0
        αp, αm = dimer_alpha(m)
        c = dimer_c(m)
        @test dimer_propagator_exact(m, 0, 0.5im) ≈
              αp / (0.5im - (0.4 - c)) + αm / (0.5im - (0.4 + c)) rtol = 1e-12
        # trap 1: a flipped hopping sign has the same spectrum and Z ...
        m = HubbardDimerModel(0.7, 2.3, 3.0)
        mw = HubbardDimerModel(-0.7, 2.3, 3.0)
        @test spectrum(mw).E ≈ spectrum(m).E
        # ... but exchanges G_0 and G_π, which D2 catches
        cw = DimerContext(mw)
        ν = fermionic_freq(0)
        @test abs(dimer_propagator(cw, 0, ν) - dimer_propagator_exact(m, 0, ν)) > 0.1
        @test dimer_propagator(cw, 0, ν) ≈ dimer_propagator_exact(m, 1, ν) atol = 1e-14
    end

    @testset "D3: local and nonlocal propagators from site operators" begin
        for (t, U, β) in DIMER_PARAMS
            ctx = DimerContext(HubbardDimerModel(t, U, β))
            for n in -4:3
                ν = fermionic_freq(n)
                G0, Gπ = dimer_propagator(ctx, 0, ν), dimer_propagator(ctx, 1, ν)
                G11 = dimer_site_propagator(ctx, 1, 1, ν)
                G12 = dimer_site_propagator(ctx, 1, 2, ν)
                @test abs(G11 - (G0 + Gπ) / 2) < 1e-14
                @test abs(G12 - (G0 - Gπ) / 2) < 1e-14
                @test dimer_site_propagator(ctx, 2, 2, ν) ≈ G11 atol = 1e-14
                @test dimer_site_propagator(ctx, 2, 1, ν) ≈ G12 atol = 1e-14
                @test abs(real(G11)) < 1e-14                     # particle-hole symmetry
                @test abs(G12) > 1e-3                            # nonlocal part is real, finite
                @test abs(imag(G12)) < 1e-14
            end
        end
    end

    @testset "D4: Keldysh ℓ = 2, retarded/advanced and FDT (C4)" begin
        for (t, U, β) in DIMER_PARAMS
            m = HubbardDimerModel(t, U, β)
            ctx = DimerContext(m)
            c = dimer_c(m)
            for q in 0:1
                A = momentum_op(ctx.ops, q, :up)
                Os = (A, permutedims(A))
                cache = dimer_psf_cache(ctx, (:k, q, :up))
                G(ω, k, γ0) = keldysh_correlator(m, Os, [ω, -ω], k; γ0 = γ0,
                                                 sp = ctx.sp, cache = cache)
                γ0 = 0.05
                for ω in range(-3, 3; length = 13)
                    GR, GA = G(ω, [1], γ0), G(ω, [2], γ0)
                    @test abs(GR - dimer_propagator_exact(m, q, ω + im * γ0)) < 1e-13 * abs(GR)
                    @test abs(GA - dimer_propagator_exact(m, q, ω - im * γ0)) < 1e-13 * abs(GA)
                    @test GA ≈ conj(GR) atol = 1e-13
                    @test G(ω, Int[], γ0) == 0                    # V1: G^{11} = 0
                end
                # FDT at every pole: G^K/(G^R - G^A) → tanh(βE/2) as O(γ₀²)
                poles = [t - c, t + c, U / 2 - t, -(U / 2 + t)] .* (q == 0 ? 1 : -1)
                for E in poles
                    errs = [abs(G(E, [1, 2], g) / (G(E, [1], g) - G(E, [2], g)) - tanh(β * E / 2))
                            for g in (1e-2, 1e-3, 1e-4)]
                    @test errs[3] < 1e-6
                    @test 50 < errs[1] / errs[2] < 200                  # factor ≈ 100 per decade
                    @test 50 < errs[2] / errs[3] < 200
                end
            end
        end
    end

    ctx = DimerContext(HubbardDimerModel(0.7, 2.3, 3.0))

    @testset "D5: momentum selection rule" begin
        tuples = all_momentum_tuples()
        @test count(!momentum_allowed, tuples) == 8
        # with the matrix-element filter: forbidden tuples are exactly zero ...
        for qs in tuples, σσ′ in SPINS, ms in DIMER_FREQS
            G = dimer_correlator_4p(ctx, qs, σσ′, ms)
            if momentum_allowed(qs)
                continue
            end
            @test G == 0
            @test dimer_vertex(ctx, qs, σσ′, ms) == 0
        end
        # ... and the allowed ones are not
        @test all(maximum(abs(dimer_correlator_4p(ctx, qs, :updn, ms)) for ms in DIMER_FREQS) > 1e-2
                  for qs in tuples if momentum_allowed(qs))
        # without it, every one of the 1.6×10⁶ cycles is kept: forbidden tuples
        # are then zero to roundoff, and the allowed ones are unchanged
        ctx0 = DimerContext(ctx.m; otol = 0.0)
        for qs in [(1, 0, 0, 0), (0, 1, 1, 1), (1, 1, 1, 0)], ms in DIMER_FREQS[1:2]
            @test abs(dimer_correlator_4p(ctx0, qs, :updn, ms)) < 1e-14
        end
        for qs in [(0, 0, 0, 0), (0, 1, 0, 1)], ms in DIMER_FREQS[1:2]
            @test dimer_correlator_4p(ctx0, qs, :updn, ms) ≈
                  dimer_correlator_4p(ctx, qs, :updn, ms) atol = 1e-14
        end
    end

    @testset "Eq. (73) with momentum deltas vs the Eq. (31) PSF split" begin
        for qs in all_momentum_tuples(), σσ′ in SPINS, ms in DIMER_FREQS
            @test abs(dimer_correlator_4p(ctx, qs, σσ′, ms; part = :connected) -
                      dimer_connected_4p(ctx, qs, σσ′, ms)) < 1e-14
            @test abs(dimer_correlator_4p(ctx, qs, σσ′, ms; part = :disconnected) -
                      dimer_disconnected_4p(ctx, qs, σσ′, ms)) < 1e-14
        end
        # the leg choice does not matter: computed and closed-form legs agree
        for ms in DIMER_FREQS
            @test dimer_vertex(ctx, (0, 0, 1, 1), :updn, ms) ≈
                  dimer_vertex(ctx, (0, 0, 1, 1), :updn, ms; exact_legs = true) rtol = 1e-12
        end
    end

    @testset "D6: U = 0 null test" begin
        for β in (3.0, 10.0)
            c0 = DimerContext(HubbardDimerModel(0.7, 0.0, β))
            for qs in all_momentum_tuples(), σσ′ in SPINS, ms in DIMER_FREQS
                @test abs(dimer_connected_4p(c0, qs, σσ′, ms)) < 1e-13
                @test abs(dimer_correlator_4p(c0, qs, σσ′, ms; part = :connected)) < 1e-13
                @test abs(dimer_vertex(c0, qs, σσ′, ms)) < 1e-12
            end
        end
    end

    @testset "D7: first order in U, F/U → ½ δ" begin
        Us = (1e-2, 5e-3)
        cs = [DimerContext(HubbardDimerModel(0.7, U, 3.0)) for U in Us]
        for qs in all_momentum_tuples(), ms in DIMER_FREQS
            f(σσ′) = [dimer_vertex(c, qs, σσ′, ms) / U for (c, U) in zip(cs, Us)]
            a, b = f(:updn)
            lim = 2b - a                                          # Richardson, linear in U
            @test abs(lim - (momentum_allowed(qs) ? 0.5 : 0.0)) < 2e-5
            a, b = f(:upup)                                        # no same-spin bare vertex
            @test abs(2b - a) < 2e-5
        end
    end

    @testset "D8: t → 0, F → ½ F^atom δ, both branches of Eq. (46)" begin
        U, β = 2.3, 3.0
        ma = HubbardAtomModel(U, β)
        errs = Float64[]
        for t in (0.0, 1e-1, 1e-2, 1e-3)
            c = DimerContext(HubbardDimerModel(t, U, β))
            e = 0.0
            for qs in all_momentum_tuples(), σσ′ in SPINS, ms in DIMER_FREQS
                ref = momentum_allowed(qs) ? 0.5 * vertex_exact(ma, σσ′, ms) : 0.0
                e = max(e, abs(dimer_vertex(c, qs, σσ′, ms) - ref))
            end
            push!(errs, e)
        end
        @test errs[1] < 1e-12                          # t = 0 exactly: the anomalous branch
        @test 50 < errs[2] / errs[3] < 200             # t ≠ 0: regular branch, error ∝ t²
        @test 50 < errs[3] / errs[4] < 200
        @test errs[4] < 1e-3
    end

    @testset "D9: crossing and SU(2)" begin
        for qs in all_momentum_tuples(), ms in DIMER_FREQS
            Fuu = dimer_vertex(ctx, qs, :upup, ms)
            Fud = dimer_vertex(ctx, qs, :updn, ms)
            Fx = dimer_vertex(ctx, (qs[3], qs[2], qs[1], qs[4]), :updn,
                              [ms[3], ms[2], ms[1], ms[4]])
            @test abs(Fuu - (Fud - Fx)) < 1e-12
            # the other crossing named on p. 16, ω₂ ↔ ω₄
            Fy = dimer_vertex(ctx, (qs[1], qs[4], qs[3], qs[2]), :updn,
                              [ms[1], ms[4], ms[3], ms[2]])
            @test abs(Fuu - (Fud - Fy)) < 1e-12
        end
    end

    @testset "D10: Keldysh — selection rule, U = 0, Eq. (69)" begin
        ω = [0.37, -1.1, 0.52, 0.21]
        ω[4] = -sum(ω[1:3])
        γ0 = 0.1
        c0 = DimerContext(HubbardDimerModel(0.7, 0.0, 3.0))
        for qs in all_momentum_tuples(), σσ′ in SPINS, k in all_keldysh_components(4)
            G = dimer_keldysh_4p(ctx, qs, σσ′, ω, k; γ0 = γ0)
            momentum_allowed(qs) || @test G == 0
            @test abs(dimer_keldysh_4p(c0, qs, σσ′, ω, k; γ0 = γ0, part = :connected)) < 1e-12
        end
        @test abs(dimer_keldysh_4p(ctx, (0, 0, 0, 0), :updn, ω, [1, 2]; γ0 = γ0)) > 0.1
        # unfiltered PSFs: forbidden components zero to roundoff
        ctx0 = DimerContext(ctx.m; otol = 0.0)
        for k in ([1], [1, 3], [1, 2, 4])
            @test abs(dimer_keldysh_4p(ctx0, (1, 0, 0, 0), :updn, ω, k; γ0 = γ0)) < 1e-13
        end
        # Eq. (69) for every fully retarded component
        for qs in all_momentum_tuples(), σσ′ in SPINS, η in 1:4, γ0 in (0.1, 0.01)
            Os = dimer_operators_4p(ctx.ops, qs, σσ′)
            cache = dimer_psf_cache(ctx, (:four, qs, σσ′, :full))
            lhs = 2 * dimer_keldysh_4p(ctx, qs, σσ′, ω, [η]; γ0 = γ0)
            rhs = regular_sum(ctx.sp, Os, omega_eta(ω, η, γ0); cache = cache)
            @test abs(lhs - rhs) ≤ 1e-12 * max(1.0, abs(rhs))
        end
    end

    @testset "Eq. (45) anomalous term = the δ terms of Eq. (85), atom" begin
        U, β = 2.3, 3.0
        ma = HubbardAtomModel(U, β)
        ops = operators(ma)
        sp = spectrum(ma)
        u, th = U / 2, tanh(β * U / 4)
        Os = (ops.d_up, ops.dag_up, ops.d_dn, ops.dag_dn)
        cache = permuted_psfs(sp, Os; part = :connected)
        for ms in DIMER_FREQS
            G(f) = propagator_exact(ma, f)
            legs = G(ms[1]) * G(MatsubaraFreq(-ms[2].m)) * G(ms[3]) * G(MatsubaraFreq(-ms[4].m))
            iω = [im * value(f, β) for f in ms]
            fac = β * u^2 * prod(iω .+ u) / prod(iω)
            for (ch, coef) in (([1, 2], th), ([1, 3], th - 1), ([1, 4], th + 1))
                fires = ms[1].m + ms[ch[2]].m == 0
                a = anomalous_part(ma, Os, ms; channel = ch, sp = sp, cache = cache) / legs
                @test abs(a - fires * coef * fac) < 1e-12 * max(1.0, abs(fac))
            end
        end
    end

    @testset "D11 support: δ(ω₁₂) part of the dimer vertex" begin
        ms = _M(1, -1, 3, -3)
        # t = 0: half of the atom's, for the momentum-conserving tuples
        U, β = 4.0, 2.0
        c = DimerContext(HubbardDimerModel(0.0, U, β))
        u = U / 2
        iω = [im * value(f, β) for f in ms]
        atom = β * u^2 * tanh(β * u / 2) * prod(iω .+ u) / prod(iω)
        @test dimer_delta_part(c, (0, 0, 0, 0), :updn, ms) ≈ 0.5atom rtol = 1e-12
        @test dimer_delta_part(c, (1, 1, 0, 0), :updn, ms) ≈ 0.5atom rtol = 1e-12
        @test dimer_delta_part(c, (0, 0, 0, 0), :updn, ms; channel = [1, 3]) == 0
        # large β(c - U/2): Boltzmann suppressed with rate exactly the singlet-triplet gap
        m1 = HubbardDimerModel(0.8, U, 1.0)
        gap = dimer_c(m1) - U / 2
        lnD(β) = log(abs(dimer_delta_part(DimerContext(HubbardDimerModel(0.8, U, β)),
                                          (0, 0, 0, 0), :updn, ms)))
        rate = -(lnD(80.0) - lnD(60.0)) / 20
        @test rate ≈ gap rtol = 0.05
    end

    @testset "traps 2 and 3: site-basis amputation, accidental degeneracy" begin
        # trap 2: the site vertex is the k vertex with its four legs transformed;
        # dividing the site G^con by local G₁₁ legs is wrong for t ≠ 0
        u = [1 1; 1 -1] / sqrt(2)                       # c_k = Σ_i u[k,i] c_i, u = u⁻¹
        ms = DIMER_FREQS[1]
        Fk = Dict(qs => dimer_vertex(ctx, qs, :updn, ms) for qs in all_momentum_tuples())
        site(is) = sum(prod(u[qs[a] + 1, is[a]] for a in 1:4) * Fk[qs] for qs in all_momentum_tuples())
        Gc = Dict(qs => dimer_connected_4p(ctx, qs, :updn, ms) for qs in all_momentum_tuples())
        Gcsite(is) = sum(prod(u[qs[a] + 1, is[a]] for a in 1:4) * Gc[qs] for qs in all_momentum_tuples())
        G11(f) = dimer_site_propagator(ctx, 1, 1, f)
        naive = Gcsite((1, 1, 1, 1)) /
                (G11(ms[1]) * G11(MatsubaraFreq(-ms[2].m)) * G11(ms[3]) * G11(MatsubaraFreq(-ms[4].m)))
        @test abs(naive - site((1, 1, 1, 1))) > 0.02 * abs(site((1, 1, 1, 1)))
        # and at t = 0 the two coincide
        c = DimerContext(HubbardDimerModel(0.0, 2.3, 3.0))
        Fk0 = Dict(qs => dimer_vertex(c, qs, :updn, ms) for qs in all_momentum_tuples())
        F1111 = sum(prod(u[qs[a] + 1, 1] for a in 1:4) * Fk0[qs] for qs in all_momentum_tuples())
        @test F1111 ≈ vertex_exact(HubbardAtomModel(2.3, 3.0), :updn, ms) rtol = 1e-12
        # trap 3: at t = U/2 the level -U meets -U/2 - t and the anomalous
        # branch fires unexpectedly; with the explicit tolerance the vertex is
        # continuous through that point
        for ms in DIMER_FREQS[1:2]
            F(t) = dimer_vertex(DimerContext(HubbardDimerModel(t, 2.0, 3.0)), (0, 0, 0, 0), :updn, ms)
            @test abs(F(1.0) - (F(1.0 + 1e-6) + F(1.0 - 1e-6)) / 2) < 1e-8
        end
    end
end
