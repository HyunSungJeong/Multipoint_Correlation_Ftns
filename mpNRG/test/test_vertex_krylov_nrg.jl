# vertex_krylov task, phases V2 and V3a/V3b (task sheet Sec. 6): the discrete
# Krylov representation of the 3p vertex ([VD] Algorithm 2, in HubbardAtom) on
# SIAM Wilson chains, from ED of the chain (V2a exact, V2b binned) and from NRG
# (V3a untruncated, V3b decoupled tail with truncation).
#
# Not part of the default runtests.jl: it takes about an hour and up to ~8 GB of
# memory (Nch = 2 on the ExactGrid; timings in vertex_krylov_STATUS.md). Run with
#   julia -t 4 --project=mpNRG/test mpNRG/test/test_vertex_krylov_nrg.jl
#
# Guaranteed tolerances ([VD] Theorem 2) are asserted; "report" columns are
# measured only (the notebook tabulates them). C4 for exact data is asserted for
# the stabilized chain ([VD] Prop. 13); without stabilization it fails (finding
# F1 of the notebook / vertex_krylov_STATUS.md) and is recorded with @test_broken.

using Test, MultipointNRG, LinearAlgebra
isdefined(@__MODULE__, :model_ed) || include(joinpath(@__DIR__, "helpers.jl"))
include(joinpath(@__DIR__, "vertex_krylov_adapter.jl"))

const VK_FAST = get(ENV, "VK_FAST", "0") == "1"      # only the N = 1 chain (quick check)

function vk_assert_common(c; C1, C56, C7, C8)
    s = c[:sanity]
    @test s.grid_ok && s.bar_involution
    @test s.herm_err ≤ 1e-13
    @test s.min_eig_rel ≥ -1e-13
    @test s.wB_min ≥ -1e-14 && s.wBd_min ≥ -1e-14 && s.wB_imag ≤ 1e-14
    @test s.sumrule_err ≤ 1e-12
    @test s.totals3_err ≤ 1e-12
    @test c[:C1] ≤ C1
    @test c[:C5] ≤ C56
    @test c[:C6] ≤ C56
    @test c[:C7] ≤ C7
    @test c[:C8] ≤ C8
    @test c[:C8_parts].perm ≤ C8
    @test c[:particle_terminated]
end

@testset "vertex_krylov V2/V3 (Wilson chain)" begin
    Nchs = VK_FAST ? (1,) : (1, 2)

    @testset "V2 odd bin map of the LogGrid" begin
        g = vk_grid()
        bg, _ = vk_loggrid(g)
        @test HA.check_grid(bg.b, bg.bar).ok
        for x in vcat(10.0 .^ range(-7, 1.5; length = 400), [0.0, 1e-15, 1e-14, 3e-14, 9.99])
            @test HA.bin_of(bg, -x) == bg.bar[HA.bin_of(bg, x)]
        end
    end

    @testset "V2a ED, ExactGrid: Nch=$Nch field=$field $chan" for Nch in Nchs, field in (0.0, 0.05),
                                                                   chan in (:updn, :upup)
        GC.gc()
        out = vk_case_ed(Nch, :exact, chan; field = field, bigfloat = Nch == 1)
        c = out[:checks]
        vk_assert_common(c; C1 = 1e-10, C56 = 1e-12, C7 = 1e-10, C8 = 1e-9)
        @test c[:C3] ≤ 1e-9
        @test c[:C2] ≤ 1e-10                       # Σ^S, spectra and weights of Zacinskis' route
        # C4: guaranteed by Lemma 12 for exact data in exact arithmetic
        st = out[:stab]
        @test st.C4 ≤ 1e-9                          # stabilized chain (Prop. 13)
        @test st.Omega_overlap ≤ 1e-10
        @test st.Omega_dim == Nch + 2                # single-particle levels of the chain (sites -1..Nch)
        @test st.C1 ≤ 1e-10 && st.C8 ≤ 1e-9
        @test st.dSigma ≤ 1e-10 && st.dK ≤ 1e-9      # outputs unchanged by the stabilization
        @test_broken c[:C4] ≤ 1e-9                   # finding F1: unstabilized chain leaks (also in BigFloat)
        if Nch == 1
            @test out[:big].C4 > 1e-9                # the leak is seeded by the data, not by Float64 rounding
        end
    end

    @testset "V2b ED, LogGrid: Nch=$Nch field=$field $chan" for Nch in Nchs, field in (0.0, 0.05),
                                                                  chan in (:updn, :upup)
        GC.gc()
        out = vk_case_ed(Nch, :log, chan; field = field)
        c = out[:checks]
        @test out[:data].meta[:nclamped] == 0
        vk_assert_common(c; C1 = 1e-10, C56 = 1e-12, C7 = 1e-10, C8 = 1e-9)
    end

    @testset "V2 N=2 block test on the chain with field (Nch=1)" begin
        field = 0.05
        o2 = vk_case_ed(1, :exact, :updn; field = field, σ = [:up, :dn], i12 = (:dn, :dn))
        c2 = o2[:checks]
        vk_assert_common(c2; C1 = 1e-10, C56 = 1e-12, C7 = 1e-10, C8 = 1e-9)
        ou = vk_case_ed(1, :exact, :updn; field = field)                  # σ = ↑, B = q↓↓
        od = vk_case_ed(1, :exact, :upup; field = field, σ = :dn)         # σ = ↓, B = q↓↓
        b2 = c2[:box]; bu = ou[:checks][:box]; bd = od[:checks][:box]
        m = .!isnan.(real.(bu.Kd))
        sc = maximum(abs, bu.Kd[m])
        @test maximum(abs, b2.Kt[1, 1, :, :][m] - bu.Kt[m]) ≤ 1e-9 * sc
        @test maximum(abs, b2.Kt[2, 2, :, :][m] - bd.Kt[m]) ≤ 1e-9 * sc
        @test maximum(abs, b2.Kt[1, 2, :, :][m]) ≤ 1e-12 * sc && maximum(abs, b2.Kt[2, 1, :, :][m]) ≤ 1e-12 * sc
        zs = HA.vk_testpoints(o2[:data])
        dΣ = maximum(z -> maximum(abs, HA.sigma_chain(o2[:res].ch, z) -
                                       Diagonal([HA.sigma_chain(ou[:res].ch, z)[1], HA.sigma_chain(od[:res].ch, z)[1]])), zs)
        @test dΣ ≤ 1e-10
    end

    @testset "V3a NRG (Nkeep = ∞) vs binned ED: Nch=$Nch field=$field $chan" for Nch in Nchs, field in (0.0, 0.05),
                                                                                chan in (:updn, :upup)
        GC.gc()
        out = vk_case_nrg(Nch, chan; field = field)
        @test out[:diff].W ≤ 1e-13 && out[:diff].wB ≤ 1e-13 && out[:diff].s ≤ 1e-13
        vk_assert_common(out[:checks]; C1 = 1e-10, C56 = 1e-11, C7 = 1e-9, C8 = 1e-8)
    end

    @testset "V3b decoupled tail, truncated NRG: $chan" for chan in (:updn, :upup)
        m = vk_model()
        chs = decoupled_tail(wilson_chains(m, 3), 1)
        out = vk_case_nrg(3, chan; Nkeep = 40, chains = chs)
        tr = out[:data].meta[:truncation]
        @test any(>(0), tr.disc[2:end - 1])           # truncation is active
        @test out[:diff].W ≤ 1e-12 && out[:diff].wB ≤ 1e-12 && out[:diff].s ≤ 1e-12
        vk_assert_common(out[:checks]; C1 = 1e-10, C56 = 1e-11, C7 = 1e-9, C8 = 1e-8)
    end
end
