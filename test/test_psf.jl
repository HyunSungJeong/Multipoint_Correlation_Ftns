# Step 2 of Block 4: partial spectral functions, Kugler Eq. (28).

@testset "Hubbard atom, Step 2: partial spectral functions" begin

    @testset "argument guards" begin
        sp = spectrum(HubbardAtomModel(2.0, 1.0))
        ops = operators(HubbardAtomModel(2.0, 1.0))
        @test_throws ArgumentError psf(sp, (ops.d_up,))
        @test_throws ArgumentError psf(sp, ())
        @test_throws DimensionMismatch psf(sp, ops.d_up, zeros(3, 3))
    end

    # Operator tuples spanning ℓ = 2, 3 and 4. The ℓ = 4 entries include the
    # (dσ, d†σ, dσ', d†σ') orderings the Step 5 vertex needs.
    tuples(ops) = [
        (ops.d_up, ops.dag_up),
        (ops.dag_up, ops.d_up),
        (ops.d_dn, ops.dag_dn),
        (ops.n_up, ops.n_dn),
        (ops.n_up, ops.d_up, ops.dag_up),
        (ops.d_up, ops.dag_up, ops.n_dn),
        (ops.d_up, ops.dag_up, ops.d_up, ops.dag_up),
        (ops.d_up, ops.dag_up, ops.d_dn, ops.dag_dn),
        (ops.d_up, ops.dag_dn, ops.d_dn, ops.dag_up),
        (ops.d_up, ops.d_dn, ops.dag_dn, ops.dag_up),
    ]

    @testset "sum rule ⟨O₁⋯O_ℓ⟩: U=$U, β=$β" for (U, β) in PARAMS
        m = HubbardAtomModel(U, β)
        ops = operators(m)
        sp = spectrum(m)

        for Os in tuples(ops)
            terms = psf(sp, Os)
            # Σ weights = tr(ρ O₁O₂⋯O_ℓ) = ⟨O₁⋯O_ℓ⟩, in the same order
            product = reduce(*, Os)
            @test psf_total_weight(terms) ≈ thermal_average(sp, product) atol = 1e-12
            # aggregation is weight preserving
            @test psf_total_weight(aggregate_psf(terms)) ≈
                  psf_total_weight(terms) atol = 1e-12
        end
    end

    @testset "term structure: U=$U, β=$β" for (U, β) in PARAMS
        m = HubbardAtomModel(U, β)
        ops = operators(m)
        sp = spectrum(m)
        E = sp.E

        for Os in tuples(ops)
            ℓ = length(Os)
            terms = psf(sp, Os)
            for t in terms
                @test npoint(t) == ℓ
                @test length(t.position) == ℓ - 1
                @test length(t.states) == ℓ

                s = t.states
                # positions are the partial sums E_{i+1} - E_1 of Eq. (28)
                for i in 1:(ℓ - 1)
                    @test t.position[i] ≈ E[s[i + 1]] - E[s[1]]
                end

                # individual frequencies: ω'_ī = E_{i+1} - E_i around the cycle,
                # and they conserve energy
                ω = psf_frequencies(t)
                @test length(ω) == ℓ
                @test sum(ω) ≈ 0.0 atol = 1e-12
                for i in 1:ℓ
                    j = i == ℓ ? 1 : i + 1
                    @test ω[i] ≈ E[s[j]] - E[s[i]]
                end
                # frequencies rebuild the partial sums
                @test cumsum(ω)[1:(ℓ - 1)] ≈ t.position

                @test !iszero(t.weight)
            end
        end
    end

    @testset "ℓ = 2 closed form: U=$U, β=$β" for (U, β) in PARAMS
        U == 0 && continue          # both poles merge at ω' = 0; handled below
        m = HubbardAtomModel(U, β)
        ops = operators(m)
        sp = spectrum(m)
        u = half_interaction(m)

        # S[d,d†] sits at ω' = ∓u with weights ρ|0⟩ = 1/Z and ρ|↓⟩ = e^{βu}/Z
        for Os in ((ops.d_up, ops.dag_up), (ops.dag_up, ops.d_up))
            agg = aggregate_psf(psf(sp, Os))
            @test length(agg) == 2
            byω = Dict(round(t.position[1]; digits = 9) => t.weight for t in agg)
            @test haskey(byω, round(-u; digits = 9))
            @test haskey(byω, round(u; digits = 9))
            @test byω[round(-u; digits = 9)] ≈ 1 / sp.Z
            @test byω[round(u; digits = 9)] ≈ exp(β * u) / sp.Z
        end

        # The local spectral function A(ω) = S[d,d†](ω) + S[d†,d](-ω) of the
        # Hubbard atom is ½δ(ω-u) + ½δ(ω+u), the two poles of Appendix E's
        # G(iν) = ½ Σ± (iν ± U/2)⁻¹. Temperature drops out entirely.
        fwd = aggregate_psf(psf(sp, ops.d_up, ops.dag_up))
        bwd = aggregate_psf(psf(sp, ops.dag_up, ops.d_up))
        A = Dict{Float64, ComplexF64}()
        for t in fwd
            k = round(t.position[1]; digits = 9)
            A[k] = get(A, k, zero(ComplexF64)) + t.weight
        end
        for t in bwd
            k = round(-t.position[1]; digits = 9)   # note the reflection
            A[k] = get(A, k, zero(ComplexF64)) + t.weight
        end
        @test length(A) == 2
        for (pos, wt) in A
            @test abs(pos) ≈ abs(u)
            @test wt ≈ 0.5
        end
        @test sum(values(A)) ≈ 1.0      # ⟨{d, d†}⟩ = 1
    end

    @testset "U = 0 collapses every delta to ω' = 0" begin
        m = HubbardAtomModel(0.0, 2.0)
        ops = operators(m)
        sp = spectrum(m)
        for Os in tuples(ops)
            raw = psf(sp, Os)
            agg = aggregate_psf(raw)
            total = thermal_average(sp, reduce(*, Os))
            if isempty(raw)
                # An identically vanishing operator product has no PSF terms at
                # all -- e.g. n↑ d↑ d†↑ = n↑(1 - n↑) = 0, since n↑ is a projector.
                # Every eigenstate cycle vanishes, not merely their sum.
                @test total ≈ 0.0 atol = 1e-14
                @test isempty(agg)
            else
                @test length(agg) == 1                   # one position only
                @test all(≈(0.0), first(agg).position)   # and it is the origin
                @test first(agg).weight ≈ total atol = 1e-12
            end
        end
        # the spectral function becomes a single pole of unit weight at ω = 0
        fwd = psf_total_weight(psf(sp, ops.d_up, ops.dag_up))
        bwd = psf_total_weight(psf(sp, ops.dag_up, ops.d_up))
        @test fwd + bwd ≈ 1.0
    end

    @testset "identically zero products have empty PSFs: U=$U, β=$β" for (U, β) in PARAMS
        # d↑ d†↑ = 1 - n↑, so n↑ d↑ d†↑ = n↑(1 - n↑) = 0. The vanishing is
        # term by term, not a cancellation: a cycle would need a state with
        # n↑ = 1 to also be d↑ applied to a state with n↑ = 2.
        m = HubbardAtomModel(U, β)
        ops = operators(m)
        sp = spectrum(m)

        @test ops.n_up * ops.d_up * ops.dag_up ≈ zeros(DIM, DIM)
        @test isempty(psf(sp, ops.n_up, ops.d_up, ops.dag_up))
        @test psf_total_weight(psf(sp, ops.n_up, ops.d_up, ops.dag_up)) == 0

        # Every cyclic arrangement of these three vanishes, since the cyclic
        # product is the same up to rotation: d↑d†↑n↑ = (1 - n↑)n↑ = 0 as well.
        @test isempty(psf(sp, ops.d_up, ops.dag_up, ops.n_up))
        @test isempty(psf(sp, ops.dag_up, ops.n_up, ops.d_up))

        # Interleaving differently does not vanish: d↑ n↑ d†↑ = (1 - n↑)² = 1 - n↑.
        @test ops.d_up * ops.n_up * ops.dag_up ≈ I(DIM) - ops.n_up
        @test !isempty(psf(sp, ops.d_up, ops.n_up, ops.dag_up))
        @test psf_total_weight(psf(sp, ops.d_up, ops.n_up, ops.dag_up)) ≈
              thermal_average(sp, ops.d_up * ops.n_up * ops.dag_up) atol = 1e-12
    end

    @testset "invariant under eigenvector sign gauge" begin
        # Eq. (28) closes a cycle over the eigenbasis, so flipping the sign of
        # any eigenvector must leave the PSF untouched. Step 1 pins these signs
        # for reproducibility; this check shows nothing observable rides on it.
        m = HubbardAtomModel(2.0, 3.0)
        ops = operators(m)
        sp = spectrum(m)

        for col in 1:DIM
            Vflip = copy(sp.V)
            Vflip[:, col] .*= -1
            flipped = Spectrum(sp.E, Vflip, sp.ρ, sp.Z)
            for Os in tuples(ops)
                a = aggregate_psf(psf(sp, Os))
                b = aggregate_psf(psf(flipped, Os))
                @test length(a) == length(b)
                for (ta, tb) in zip(a, b)
                    @test ta.position ≈ tb.position
                    @test ta.weight ≈ tb.weight atol = 1e-14
                end
            end
        end
    end

    @testset "aggregation and wtol" begin
        m = HubbardAtomModel(2.0, 5.0)
        ops = operators(m)
        sp = spectrum(m)
        Os = (ops.d_up, ops.dag_up, ops.d_up, ops.dag_up)

        raw = psf(sp, Os)
        agg = aggregate_psf(raw)
        @test length(agg) <= length(raw)
        @test issorted([t.position for t in agg])
        @test all(t -> isempty(t.states), agg)          # merged terms drop states
        @test psf_total_weight(agg) ≈ psf_total_weight(raw) atol = 1e-12

        # wtol prunes small weights, and pruning only ever removes terms
        big = psf(sp, Os; wtol = 0.1)
        @test length(big) <= length(raw)
        @test all(t -> abs(t.weight) > 0.1, big)

        # an empty PSF aggregates to an empty PSF
        @test isempty(aggregate_psf(PSFTerm[]))
        @test psf_total_weight(PSFTerm[]) == 0
    end

    @testset "varargs and tuple forms agree" begin
        m = HubbardAtomModel(2.0, 5.0)
        ops = operators(m)
        sp = spectrum(m)
        a = psf(sp, (ops.d_up, ops.dag_up))
        b = psf(sp, ops.d_up, ops.dag_up)
        @test length(a) == length(b)
        @test psf_total_weight(a) ≈ psf_total_weight(b)
    end
end
