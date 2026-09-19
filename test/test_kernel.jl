# Step 3 of Block 4: the Matsubara kernel, Kugler Eq. (46).

@testset "Hubbard atom, Step 3: the Eq. (46) kernel" begin

    @testset "MatsubaraFreq" begin
        β = 4.0
        @test value(MatsubaraFreq(3), β) ≈ 3π / β
        @test fermionic(fermionic_freq(0)) && fermionic_freq(0).m == 1
        @test fermionic(fermionic_freq(-1)) && fermionic_freq(-1).m == -1
        @test bosonic(bosonic_freq(0)) && bosonic_freq(0).m == 0
        @test bosonic(bosonic_freq(2)) && bosonic_freq(2).m == 4
        @test !fermionic(bosonic_freq(3))
        @test !bosonic(fermionic_freq(3))
        # a fermionic frequency is never zero, which is why ∏(iωᵢ) is safe
        for n in -5:5
            @test value(fermionic_freq(n), β) != 0
        end
    end

    @testset "composites: partial sums and vanishing detection" begin
        β = 2.0
        ms = [fermionic_freq(0), fermionic_freq(-1), fermionic_freq(0), fermionic_freq(-1)]
        @test sum(f -> f.m, ms) == 0                      # m = (1,-1,1,-1)

        # ω'  = 0 everywhere: Ω is purely the Matsubara part
        Ω, van = composites(ms, [0.0, 0.0, 0.0], β)
        @test Ω[1] ≈ im * π / β                           # m₁ = 1
        @test Ω[2] ≈ 0.0                                  # m₁₂ = 0
        @test Ω[3] ≈ im * π / β                           # m₁₂₃ = 1
        @test van == [false, true, false]                 # only the bosonic zero

        # a nonzero spectral part unblocks the middle composite
        Ω2, van2 = composites(ms, [0.0, 0.7, 0.0], β)
        @test Ω2[2] ≈ -0.7
        @test van2 == [false, false, false]

        @test_throws DimensionMismatch composites(ms, [0.0, 0.0], β)
    end

    @testset "integer exactness beats a float tolerance" begin
        # The point of carrying the integer m: at huge β a genuinely nonzero
        # bosonic frequency is numerically smaller than any sane tolerance.
        # ω = 2π/β = 6.3e-12 here, well under atol = 1e-10, yet Ω ≠ 0.
        β = 1e12
        ms = [MatsubaraFreq(1), MatsubaraFreq(1), MatsubaraFreq(-1), MatsubaraFreq(-1)]
        Ω, van = composites(ms, [0.0, 0.0, 0.0], β; atol = 1e-10)
        @test abs(Ω[2]) < 1e-10          # numerically tiny ...
        @test van[2] == false            # ... but correctly NOT flagged as zero
        # and the kernel therefore takes the regular branch
        @test matsubara_kernel(Ω, van, β) ≈ prod(inv, Ω)
    end

    @testset "regular branch" begin
        β = 3.0
        Ω2 = ComplexF64[2.0 + 1.0im]
        @test matsubara_kernel(Ω2, [false], β) ≈ inv(Ω2[1])

        Ω4 = ComplexF64[1.0 + 0.5im, -2.0 + 1.0im, 0.25 - 3.0im]
        @test matsubara_kernel(Ω4, [false, false, false], β) ≈
              inv(Ω4[1]) * inv(Ω4[2]) * inv(Ω4[3])
    end

    @testset "anomalous branch" begin
        β = 3.0
        # ℓ = 2: empty product and empty sum leave K = -β/2
        @test matsubara_kernel(ComplexF64[0.0], [true], β) ≈ -β / 2

        # ℓ = 4 with j = 2. This is literally the second part of Eq. (B26):
        #   -½[β + 1/Ω_1̄ + 1/Ω_1̄2̄3̄] / (Ω_1̄ Ω_1̄2̄3̄)
        Ω = ComplexF64[1.5 + 0.5im, 0.0, -0.75 + 2.0im]
        want = -0.5 * (β + inv(Ω[1]) + inv(Ω[3])) * inv(Ω[1]) * inv(Ω[3])
        @test matsubara_kernel(Ω, [false, true, false], β) ≈ want

        # the vanishing factor is excluded, never formed: no Inf or NaN anywhere
        K = matsubara_kernel(Ω, [false, true, false], β)
        @test isfinite(real(K)) && isfinite(imag(K))

        # j = 1 and j = 3 pick out the complementary products
        Ωb = ComplexF64[0.0, 2.0 + 0.0im, 4.0 + 0.0im]
        @test matsubara_kernel(Ωb, [true, false, false], β) ≈
              -0.5 * (β + 1 / 2 + 1 / 4) * (1 / 2) * (1 / 4)
        Ωc = ComplexF64[2.0 + 0.0im, 4.0 + 0.0im, 0.0]
        @test matsubara_kernel(Ωc, [false, false, true], β) ≈
              -0.5 * (β + 1 / 2 + 1 / 4) * (1 / 2) * (1 / 4)
    end

    @testset "guards" begin
        β = 1.0
        @test_throws ArgumentError matsubara_kernel(ComplexF64[], Bool[], β)
        @test_throws DimensionMismatch matsubara_kernel(ComplexF64[1.0], [false, false], β)
        # Eq. (46) covers at most one vanishing composite
        @test_throws ArgumentError matsubara_kernel(ComplexF64[0.0, 1.0, 0.0],
                                                    [true, false, true], β)
    end

    @testset "kernel is finite wherever Eq. (46) applies" begin
        # Sweep the physical composites of the Hubbard atom and check that the
        # kernel never produces Inf or NaN -- the whole point of Eq. (46) over
        # Eq. (45) with an ε-regulator.
        for (U, β) in PARAMS
            m = HubbardAtomModel(U, β)
            sp = spectrum(m)
            ops = operators(m)
            Os = (ops.d_up, ops.dag_up, ops.d_dn, ops.dag_dn)
            for n1 in -2:2, n2 in -2:2, n3 in -2:2
                ms = vertex_frequencies(n1, n2, n3)
                for p in all_permutations(4)
                    msp = [ms[p[i]] for i in 1:4]
                    Op = ntuple(i -> Os[p[i]], 4)
                    for t in psf(sp, Op)
                        Ω, van = composites(msp, t.position, β)
                        @test count(van) <= 1     # at most one, as the paper says
                        K = matsubara_kernel(Ω, van, β)
                        @test isfinite(real(K)) && isfinite(imag(K))
                    end
                end
            end
        end
    end
end
