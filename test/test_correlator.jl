# Steps 4-6 of Block 4: Eq. (39), the propagator, the 4p vertex, the null test.

@testset "Hubbard atom, Steps 4-6: correlators and the vertex" begin

    @testset "permutations and ζ_p" begin
        @test length(all_permutations(1)) == 1
        @test length(all_permutations(2)) == 2
        @test length(all_permutations(3)) == 6
        @test length(all_permutations(4)) == 24          # the 24 summands of Eq. (39)
        @test allunique(all_permutations(4))
        @test all(p -> sort(p) == collect(1:4), all_permutations(4))

        allf(n) = fill(true, n)
        @test permutation_sign([1, 2, 3, 4], allf(4)) == 1
        @test permutation_sign([2, 1, 3, 4], allf(4)) == -1      # one transposition
        @test permutation_sign([2, 1, 4, 3], allf(4)) == 1       # two
        @test permutation_sign([4, 3, 2, 1], allf(4)) == 1       # six inversions
        @test permutation_sign([2, 3, 4, 1], allf(4)) == -1      # three inversions

        # ζ_p counts transpositions of FERMIONIC operators only: a bosonic
        # operator commutes through and must not contribute a sign.
        mixed = [true, false, true]        # operator 2 is bosonic
        @test permutation_sign([1, 3, 2], mixed) == 1    # fermions stay in order 1,3
        @test permutation_sign([3, 2, 1], mixed) == -1   # fermions swap to 3,1
        @test permutation_sign([2, 1, 3], mixed) == 1    # only the boson moved
    end

    @testset "correlator guards" begin
        m = HubbardAtomModel(2.0, 1.0)
        ops = operators(m)
        ν = fermionic_freq(0)
        @test_throws ArgumentError correlator(m, (ops.d_up,), [ν])
        @test_throws DimensionMismatch correlator(m, (ops.d_up, ops.dag_up), [ν])
        # Σω must vanish
        @test_throws ArgumentError correlator(m, (ops.d_up, ops.dag_up), [ν, ν])
        @test_throws ArgumentError propagator(m, bosonic_freq(1))
    end

    # ----------------------------------------------------------------------
    # Step 4: the ℓ = 2 milestone
    # ----------------------------------------------------------------------
    @testset "Step 4, propagator vs Appendix E: U=$U, β=$β" for (U, β) in PARAMS
        m = HubbardAtomModel(U, β)
        sp = spectrum(m)
        ops = operators(m)
        for n in -8:8
            ν = fermionic_freq(n)
            got = propagator(m, ν; sp = sp, ops = ops)
            want = propagator_exact(m, ν)
            @test got ≈ want rtol = 1e-12
        end
        # G(iν) is purely imaginary at half filling, and odd
        for n in -4:4
            ν = fermionic_freq(n)
            @test real(propagator(m, ν; sp = sp, ops = ops)) ≈ 0.0 atol = 1e-12
            @test propagator(m, MatsubaraFreq(-ν.m); sp = sp, ops = ops) ≈
                  -propagator(m, ν; sp = sp, ops = ops)
        end
    end

    @testset "Step 4, the closed form itself" begin
        # G(iν) = ½ Σ± (iν ± u)⁻¹ = -iν/(ν²+u²), the two poles of the atom.
        for (U, β) in PARAMS
            m = HubbardAtomModel(U, β)
            u = half_interaction(m)
            for n in -4:4
                ν = fermionic_freq(n)
                x = value(ν, β)
                @test propagator_exact(m, ν) ≈ -im * x / (x^2 + u^2)
            end
        end
        # at U = 0 it is the free propagator 1/(iν)
        m0 = HubbardAtomModel(0.0, 2.0)
        for n in -4:4
            ν = fermionic_freq(n)
            @test propagator(m0, ν) ≈ inv(im * value(ν, 2.0)) rtol = 1e-12
        end
    end

    # ----------------------------------------------------------------------
    # Step 5: ℓ = 4, the vertex
    # ----------------------------------------------------------------------
    @testset "vertex_frequencies" begin
        for n1 in -3:3, n2 in -3:3, n3 in -3:3
            ms = vertex_frequencies(n1, n2, n3)
            @test length(ms) == 4
            @test sum(f -> f.m, ms) == 0            # energy conservation
            @test all(fermionic, ms)                # all four legs fermionic
        end
    end

    @testset "spin_operators" begin
        ops = operators(HubbardAtomModel(1.0, 1.0))
        @test spin_operators(ops, :up) == (ops.d_up, ops.dag_up)
        @test spin_operators(ops, :dn) == (ops.d_dn, ops.dag_dn)
        @test_throws ArgumentError spin_operators(ops, :sideways)
    end

    @testset "disconnected part, Eq. (73)" begin
        m = HubbardAtomModel(2.0, 5.0)
        sp = spectrum(m)
        ops = operators(m)
        kw = (sp = sp, ops = ops)
        for n1 in -2:2, n2 in -2:2, n3 in -2:2
            ms = vertex_frequencies(n1, n2, n3)
            δ12 = (ms[1].m + ms[2].m == 0)
            δ23 = (ms[2].m + ms[3].m == 0)
            # δ_{ω23,0} = δ_{ω14,0} by energy conservation
            @test δ23 == (ms[1].m + ms[4].m == 0)

            dud = disconnected_4p(m, :up, :dn, ms; kw...)
            duu = disconnected_4p(m, :up, :up, ms; kw...)
            # opposite spins keep only the δ_{ω12} pairing
            @test (dud == 0) == !δ12
            # Equal spins: the two Wick pairings of Eq. (73) enter with
            # opposite signs, so G^dis vanishes both when neither delta fires
            # AND when both do -- they cancel exactly.
            @test (duu == 0) == ((δ23 ? 1 : 0) - (δ12 ? 1 : 0) == 0)
            if δ12 && δ23
                @test duu == 0
            end
            # and the closed form itself
            G = ν -> propagator(m, ν; kw...)
            @test dud ≈ m.β * G(ms[1]) * G(ms[3]) * (0 - (δ12 ? 1 : 0))
            @test duu ≈ m.β * G(ms[1]) * G(ms[3]) * ((δ23 ? 1 : 0) - (δ12 ? 1 : 0))
        end
    end

    @testset "Step 5, vertex vs Eqs. (85): U=$U, β=$β" for (U, β) in PARAMS
        m = HubbardAtomModel(U, β)
        sp = spectrum(m)
        ops = operators(m)
        kw = (sp = sp, ops = ops)
        # scale for a fair absolute comparison at points where F vanishes
        scale = maximum(abs(vertex_exact(m, :updn, vertex_frequencies(a, b, c)))
                        for a in -2:2, b in -2:2, c in -2:2)
        for n1 in -2:2, n2 in -2:2, n3 in -2:2
            ms = vertex_frequencies(n1, n2, n3)
            for (σ, σ′, tag) in [(:up, :dn, :updn), (:up, :up, :upup)]
                got = vertex(m, σ, σ′, ms; kw...)
                want = vertex_exact(m, tag, ms)
                if abs(want) > 1e-6 * max(scale, 1.0)
                    @test got ≈ want rtol = 1e-9
                else
                    # F is identically zero here; require it numerically so
                    @test abs(got) <= 1e-8 * max(scale, 1.0)
                end
            end
        end
    end

    @testset "Step 5, F↑↑ from crossing symmetry: U=$U, β=$β" for (U, β) in PARAMS
        # Eq. (85): F↑↑(iω) = F↑↓(iω) - F↑↓(iω′), with ω′ obtained from ω by
        # exchanging ω₁ ↔ ω₃. A check on the closed forms independent of the
        # numerics.
        m = HubbardAtomModel(U, β)
        for n1 in -3:3, n2 in -3:3, n3 in -3:3
            ms = vertex_frequencies(n1, n2, n3)
            swapped = [ms[3], ms[2], ms[1], ms[4]]
            @test vertex_exact(m, :upup, ms) ≈
                  vertex_exact(m, :updn, ms) - vertex_exact(m, :updn, swapped) atol = 1e-8
        end
    end

    @testset "Step 5, closed forms vs Appendix E at order U²" begin
        # Appendix E states, independently of Eq. (85), that expanding the
        # vertex to second order gives
        #     F↑↓ = U + ¼βU²(δ_{ω14} - δ_{ω13}),   F↑↑ = ¼βU²(δ_{ω14} - δ_{ω12}).
        β = 3.0
        for U in (1e-3, 1e-4, 1e-5)
            m = HubbardAtomModel(U, β)
            for n1 in -2:2, n2 in -2:2, n3 in -2:2
                ms = vertex_frequencies(n1, n2, n3)
                δ12 = (ms[1].m + ms[2].m == 0) ? 1 : 0
                δ13 = (ms[1].m + ms[3].m == 0) ? 1 : 0
                δ14 = (ms[1].m + ms[4].m == 0) ? 1 : 0
                # the O(U³) remainder must shrink faster than the O(U²) term
                tol = 50 * U^3 * β
                @test real(vertex_exact(m, :updn, ms)) ≈
                      U + 0.25 * β * U^2 * (δ14 - δ13) atol = tol
                @test real(vertex_exact(m, :upup, ms)) ≈
                      0.25 * β * U^2 * (δ14 - δ12) atol = tol
            end
        end
    end

    # ----------------------------------------------------------------------
    # Step 6: the null test
    # ----------------------------------------------------------------------
    @testset "Step 6, null test at U = 0" begin
        # Every connected object must vanish identically. This exercises all 24
        # permutations and every ζ_p sign against a target of exactly zero, and
        # it runs through the anomalous branch of Eq. (46): at U = 0 the whole
        # spectrum is degenerate, so ω' = 0 for every PSF term and every
        # vanishing bosonic composite triggers it.
        for β in (0.5, 2.0, 7.0)
            m = HubbardAtomModel(0.0, β)
            sp = spectrum(m)
            ops = operators(m)
            kw = (sp = sp, ops = ops)
            for n1 in -2:2, n2 in -2:2, n3 in -2:2
                ms = vertex_frequencies(n1, n2, n3)
                for (σ, σ′) in [(:up, :dn), (:up, :up)]
                    @test abs(connected_4p(m, σ, σ′, ms; kw...)) <= 1e-10
                    @test abs(vertex(m, σ, σ′, ms; kw...)) <= 1e-8
                end
            end
            # the closed forms agree that there is nothing there
            for n1 in -2:2, n2 in -2:2, n3 in -2:2
                ms = vertex_frequencies(n1, n2, n3)
                @test vertex_exact(m, :updn, ms) == 0
                @test vertex_exact(m, :upup, ms) == 0
            end
        end
    end

    @testset "exact and computed legs agree" begin
        # Amputating with the closed-form propagator or with our own Eq. (39)
        # one must give the same vertex, since Step 4 matched to machine
        # precision.
        m = HubbardAtomModel(2.0, 5.0)
        sp = spectrum(m)
        ops = operators(m)
        for n1 in -2:2, n2 in -2:2, n3 in -2:2
            ms = vertex_frequencies(n1, n2, n3)
            a = vertex(m, :up, :dn, ms; sp = sp, ops = ops, exact_legs = false)
            b = vertex(m, :up, :dn, ms; sp = sp, ops = ops, exact_legs = true)
            @test a ≈ b rtol = 1e-9
        end
    end
end
