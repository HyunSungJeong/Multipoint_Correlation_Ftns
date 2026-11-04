# vertex_krylov task, phases V2 and V3 (task sheet Sec. 6): the discrete
# Krylov representation of the 3p vertex ([VD] Algorithm 2, in HubbardAtom) on
# SIAM Wilson chains, from ED of the chain (V2a exact, V2b binned) and from NRG
# (V3a untruncated, V3b decoupled tail with truncation, V3c physical chain with
# truncation: report only, so only the code path is tested, on the smallest V3c case;
# the V3c results are produced by mpNRG/scripts/vertex_krylov_v3c.jl and the notebook).
#
# Not part of the default runtests.jl: it takes about 90 min and up to ~8 GB of
# memory (Nch = 2 on the ExactGrid; timings in vertex_krylov_STATUS.md). Run with
#   julia -t 4 --project=mpNRG/test mpNRG/test/test_vertex_krylov_nrg.jl
#
# Guaranteed tolerances ([VD] Theorem 2, task sheet amended as recorded in
# vertex_krylov_STATUS.md) are asserted; "report" columns are measured only (the
# notebook tabulates them). Amendment A1/A2: for exact data the Stage-2 default is
# the stabilized particle chain ([VD] Prop. 13) and its C4 is guaranteed; C4 of the
# unstabilized chain (Π = 1 - vv†) is report only. Amendment A3: the C2 invariants
# (Σ^S, spectra, first-block weights, S = N̄²) are guaranteed at 1e-10, the
# Jacobi-block agreement is report only.

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
    @test c[:C2] ≤ 1e-10                           # A3: Zacinskis invariants
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
        # C4 of the default (stabilized, Prop. 13) chain: guaranteed (amendment A1)
        @test c[:stabilized]
        @test c[:C4] ≤ 1e-9
        @test c[:Omega_overlap] ≤ 1e-10
        @test c[:Omega_dim] == Nch + 2               # single-particle levels of the chain (sites -1..Nch)
        # diagnostic: the unstabilized chain gives the same outputs (task sheet Sec. 5.3);
        # its C4 is report only (finding F1: it leaks, also in BigFloat)
        u = out[:unstab]
        @test u.C1 ≤ 1e-10 && u.C8 ≤ 1e-9
        @test u.dSigma ≤ 1e-10 && u.dK ≤ 1e-9
        @info "V2a Nch=$Nch field=$field $chan: C4 unstabilized (report) = $(u.C4)"
        if Nch == 1
            @test out[:big].C4 ≤ 1e-9                # BigFloat, default chain
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

    @testset "V3c physical chain, truncated NRG: code path only (report)" begin
        GC.gc()
        d = vk_v3c_data(1e-2, 64, :updn)
        o = vk_case_v3c(d)
        c = o[:checks]
        @test d.meta[:N] == vk_v3c_length(1e-2) && any(>(0), d.meta[:truncation].disc)
        @test c[:sanity].grid_ok
        @test all(isfinite, (c[:C1], c[:C2], c[:C3], c[:C4], c[:C5], c[:C6], c[:C7], c[:C8]))
        @test haskey(o, :big)
        @info "V3c T=1e-2 Nkeep=64 (↑↓), report: C1 $(c[:C1]) C7 $(c[:C7]) C8 $(c[:C8])"
    end
end
