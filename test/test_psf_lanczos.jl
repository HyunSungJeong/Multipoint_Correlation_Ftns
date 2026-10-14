# The PSF Lanczos task: Liouvillian matrix elements of 3p PSFs, L1-L11.
#
# The algorithm (src/psf_lanczos.jl) sees only PSF data; every check compares it
# with something that does not: the operator-space block Lanczos (R1), the
# Lehmann resolvent (R2), closed forms, or exact moments. The check functions
# live in psf_lanczos_checks.jl, shared with notebooks/psf_lanczos.ipynb. L12 is
# an observation and lives only in the notebook.

include(joinpath(@__DIR__, "psf_lanczos_checks.jl"))

@testset "PSF Lanczos: Liouvillian matrix elements of 3p PSFs" begin

    runs = [run_fixture(fx) for fx in all_fixtures(PARAMS)]
    byname(l) = only(filter(r -> r.fx.label == l, runs))

    @testset "L1 input contract: $(r.fx.label)" for r in runs
        c = check_L1(r)
        @test c.positions ≤ 1e-14
        @test c.gram ≤ 1e-14
        @test c.total_weight ≤ 1e-14
    end

    @testset "L2 analytic anchor, atom: U=$U, β=$β" for (U, β) in PARAMS
        U == 0 && continue                       # the measure collapses to one point
        c = check_L2(U, β)
        @test c.nlevels == 2
        @test c.peaks ≤ 1e-14
        @test c.N3sq ≤ 1e-14
        @test c.chain ≤ 1e-14
        @test c.bra_eq_ket ≤ 1e-14
    end

    @testset "L3 Jacobi blocks vs operator space: $(r.fx.label)" for r in runs
        c = check_L3(r)
        @test c.eps ≤ 1e-12
        @test c.t ≤ 1e-12
        @test c.levels == min(20, length(r.K.eps3), length(r.K.eps1))
        @test c.R == c.R_ref_eff                 # support points counted with rank
        @test !any(c.deflation) && !any(c.deflation_ref)
        r.fx.label == "F1" && @test c.R == (31, 31) == c.R_ref    # D(D-1) + 1
        # the operator-space chain resolves the ~1e-15 splitting of degenerate
        # Liouvillian eigenvalues (P5): one direction per nonzero matrix element
        r.fx.label == "F4 Nb=2" && @test c.R_ref == (244, 244)
    end

    @testset "L4 Gauss quadrature: $(r.fx.label)" for r in runs
        c = check_L4(r)
        @test c.eig ≤ 1e-12
        @test c.w ≤ 1e-12
        @test c.eig_ref ≤ 1e-12                  # 𝕁_ref from (R1): non-tautological
        @test c.w_ref ≤ 1e-12
    end

    @testset "L5 orthonormality, polynomials: $(r.fx.label)" for r in runs
        startswith(r.fx.label, "F4") && continue
        c = check_L5(r)
        @test c.orth ≤ 1e-11
        if r.fx.label == "F2 U=8.0 β=10.0"
            # βU = 80: 𝐩₂(b) = (b - ϵ₁)/t₂ with t₂ = 1.6e-8 and b - ϵ₁ = 2u e^{-βu}
            # below the roundoff of ϵ₁; the recursion loses 5e-8 (README, L5)
            @test_broken c.poly ≤ 1e-11
            @test c.poly ≤ 1e-7
        else
            @test c.poly ≤ 1e-11
        end
    end

    @testset "L6 coupling matrix: $(r.fx.label)" for r in runs
        c = check_L6(r)
        @test c.V ≤ 1e-12
        @test c.sumrule ≤ 1e-12
        @test c.blocks == (min(20, length(r.K.eps1)), min(20, length(r.K.eps3)))
    end

    @testset "L7 reconstruction, Eq. (12): $(r.fx.label)" for r in runs
        c = check_L7(r)
        # (𝕁, 𝕁′, 𝐕_ref) where the two chains share a gauge, (𝕁_ref, 𝕁′_ref, 𝐕_ref) otherwise
        primary = c.gauge_shared ? c.alg_chains_Vref : c.ref_chains
        @test primary ≤ 1e-12
        @test c.gauge_shared == (r.fx.label != "F4 Nb=2")
        # extra, not on the sheet: the reference's own chains, by Golub-Welsch;
        # eigenvectors cost ε‖𝕁‖/gap, 1.4e-12 for F1
        @test c.ref_chains ≤ 1e-11
        @test c.ref_positions ≤ 1e-10            # aggregate_psf rounds positions to postol
        @test c.own_V ≤ 1e-12                    # sanity: tautological
    end

    @testset "L8 resolvent: $(r.fx.label)" for r in runs
        c = check_L8(r)
        @test c.krylov ≤ 1e-11
        @test c.trace ≤ 1e-11                    # the two forms of (R2) agree
        @test c.cf ≤ 1e-11                       # Eq. (11) vs (z - 𝕁)⁻¹
    end

    @testset "L9 truncation, moments: $(r.fx.label)" for r in filter(r -> r.fx.label in ("F1", "F4 Nb=1", "F4 Nb=2"), runs)
        c = check_L9(r)
        @test c.inside ≤ 1e-10
        @test c.boundary > 1e-7                  # negative control: no moment doubling
    end

    @testset "L10 factorization lemma: $(r.fx.label)" for r in runs
        c = check_L10(r)
        @test c.outside == 0
        @test c.factorization ≤ 1e-13
    end

    @testset "L11 non-uniqueness: $(r.fx.label)" for r in filter(r -> r.fx.label == "F1" || startswith(r.fx.label, "F2"), runs)
        c = check_L11(r)
        @test c.reconstruction ≤ 1e-12           # the PSF is still reproduced exactly ...
        @test c.chain > 1e-3                     # ... but 𝕁 is not Liouvillian
    end
end
