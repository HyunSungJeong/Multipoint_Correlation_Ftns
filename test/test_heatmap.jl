# Vertex heat maps of the Hubbard atom: H1-H4 of the heat-map task sheet.
#
# Setup of Figs. 10 and 11: U = 1, T = U/50, γ₀ = T, ω = 0. Equations are
# Kugler, Lee & von Delft, PRX 11, 041006 (2021), read from the printed page.

const HM_U = 1.0
const HM_T = HM_U / 50
const HM_PRM = HeatmapParams(HM_U, 1 / HM_T, HM_T)

# Independent reference for the external legs: the Lehmann sums of Eq. (D2)
# written directly over eigenstates, with no PSF code involved.
function _lehmann_leg(m, ν, a, b, γ0)
    sp = spectrum(m)
    d = to_eigenbasis(sp, operators(m).d_up)
    GR = GA = GK = zero(ComplexF64)
    for i in 1:DIM, j in 1:DIM
        w = abs2(d[i, j])                       # A_{12} B_{21} = |⟨1|d|2⟩|²
        w == 0 && continue
        E = sp.E[j] - sp.E[i]                   # E_{2̲1̲}
        r = sp.ρ[i] + sp.ρ[j]                   # ρ₁ - ζρ₂ with ζ = -1
        GR += w * r / (ν + im * a * γ0 - E)
        GA += w * r / (ν - im * b * γ0 - E)
        # G^K carries ρ₁ + ζρ₂ = ρ₁ - ρ₂
        GK += w * (sp.ρ[i] - sp.ρ[j]) * (1 / (ν + im * a * γ0 - E) - 1 / (ν - im * b * γ0 - E))
    end
    return [0 GA; GR GK]
end

# Independent reference for the vertex: G^con from the reference Keldysh code
# (keldysh_correlator, Eq. 67b), legs from _lehmann_leg, generic matrix inverses.
function _slow_vertex(m, σσ′, ν, νp, γ0)
    ops = operators(m)
    Os = (ops.d_up, ops.dag_up, spin_operators(ops, σσ′ === :updn ? :dn : :up)...)
    w = [ν, -ν, νp, -νp]
    G = zeros(ComplexF64, 2, 2, 2, 2)
    for I in CartesianIndices(G)
        G[I] = keldysh_correlator(m, Os, w, keldysh_slots(collect(Tuple(I))); γ0 = γ0,
                                  part = :connected)
    end
    A1 = inv(_lehmann_leg(m, w[1], 3, 1, γ0)); A3 = inv(_lehmann_leg(m, w[3], 3, 1, γ0))
    B2 = inv(transpose(_lehmann_leg(m, -w[2], 1, 3, γ0)))
    B4 = inv(transpose(_lehmann_leg(m, -w[4], 1, 3, γ0)))
    F = zeros(ComplexF64, 2, 2, 2, 2)
    for Ip in CartesianIndices(F), I in CartesianIndices(G)
        a, b, c, e = Tuple(Ip); i, j, k, l = Tuple(I)
        F[Ip] += A1[a, i] * B2[b, j] * A3[c, k] * B4[e, l] * G[I]
    end
    return F
end

@testset "Vertex heat maps, H1-H4" begin
    m = HubbardAtomModel(HM_U, 1 / HM_T)
    γ0 = HM_PRM.γ0
    tabs = Dict((s, f) => heatmap_tables(m, s, f) for s in (:updn, :upup), f in (:MF, :KF))
    rng_pts = [(-1 + 2 * ((17i) % 23) / 23, -1 + 2 * ((29i + 5) % 31) / 31) for i in 1:20]

    @testset "H1: point function and grid driver" begin
        buf = zeros(ComplexF64, 16)
        for s in (:updn, :upup)
            tab = tabs[(s, :KF)]
            for (ν, νp) in rng_pts
                vertex_point!(buf, ν, νp, 0.0, HM_PRM, tab)
                ref = _slow_vertex(m, s, ν, νp, γ0)
                scale = maximum(abs, ref)
                @test maximum(abs, buf .- vec(ref)) ≤ 1e-12 * max(scale, 1.0)
            end
            # grid driver = point function, first index ν
            νs = [-0.3, 0.0, 0.25]; νps = [-0.5, 0.1]
            F = vertex_grid(νs, νps, 0.0, HM_PRM, tab)
            @test size(F) == (2, 2, 2, 2, 3, 2)
            for i in 1:3, j in 1:2
                vertex_point!(buf, νs[i], νps[j], 0.0, HM_PRM, tab)
                @test vec(F[:, :, :, :, i, j]) == buf
            end
            # zero allocations, type stability
            vertex_point!(buf, 0.3, 0.1, 0.0, HM_PRM, tab)
            @test (@allocated vertex_point!(buf, 0.3, 0.1, 0.0, HM_PRM, tab)) == 0
            @inferred vertex_point!(buf, 0.3, 0.1, 0.0, HM_PRM, tab)
            # MF
            tmf = tabs[(s, :MF)]
            b1 = zeros(ComplexF64, 1)
            σ′ = s === :updn ? :dn : :up
            for (i, (x, y)) in enumerate(rng_pts)
                mν, mνp = 2 * round(Int, 24x) + 1, 2 * round(Int, 24y) - 1
                vertex_point!(b1, mν, mνp, 0, HM_PRM, tmf)
                ms = MatsubaraFreq.([mν, -mν, mνp, -mνp])
                ref = vertex(m, :up, σ′, ms)            # correlator-level subtraction
                @test abs(b1[1] - ref) ≤ 1e-12 * max(abs(ref), 1.0)
                @test abs(b1[1] - vertex_exact(m, s, ms)) ≤ 1e-12 * max(abs(ref), 1.0)
            end
            vertex_point!(b1, 3, 5, 0, HM_PRM, tmf)
            @test (@allocated vertex_point!(b1, 3, 5, 0, HM_PRM, tmf)) == 0
            Fm = vertex_grid([1, 3], [-1, 5], 0, HM_PRM, tmf)
            vertex_point!(b1, 3, -1, 0, HM_PRM, tmf)
            @test Fm[2, 1] == b1[1]
        end
        # the Eq. (63) table reproduces the Eq. (67b) kernel for every p and k
        ω = [0.3, -0.1, 0.45, -0.65]
        wp = [0.2, -0.4, 0.5]
        for p in all_permutations(4)
            C = keldysh_coefficients(Tuple(p))
            R = [retarded_kernel(ω, wp, λ, p, 0.02) for λ in 1:4]
            for c in 1:16
                k = keldysh_slots(collect(Tuple(CartesianIndices((2, 2, 2, 2))[c])))
                @test sum(C[c, λ] * R[λ] for λ in 1:4) ≈ keldysh_kernel(ω, wp, k, p, 0.02) atol = 1e-12
            end
        end
    end

    @testset "H2: disconnected part at the PSF level" begin
        # MF: kernel (46) on S^dis = first line of Eq. (73), on the ω = 0 plane
        b1 = zeros(ComplexF64, 1)
        for s in (:updn, :upup)
            tdis = heatmap_tables(m, s, :MF; part = :disconnected)
            σ′ = s === :updn ? :dn : :up
            for n in (-3, 0, 2), np in (-2, 0, 3)
                mν, mνp = 2n + 1, 2np + 1
                ms = (mν, -mν, mνp, -mνp)
                got = mf_correlator_point(ms, HM_PRM.β, tdis.con)
                ref = disconnected_4p(m, :up, σ′, MatsubaraFreq.(collect(ms)))
                @test abs(got - ref) ≤ 1e-12 * max(abs(ref), 1.0)
            end
        end
        # merging really removes peaks: S^con has fewer peaks than S + S^dis
        ops = operators(m)
        Os = (ops.d_up, ops.dag_up, ops.d_dn, ops.dag_dn)
        raw = sum(length(d.terms) for d in permuted_psfs(spectrum(m), Os; part = :connected))
        merged = sum(length(P.w) for P in tabs[(:updn, :MF)].con)
        @test merged < raw
        # KF: Σ_p ζ K^[η] * S^dis = 0 for η = 1…4, relative to the individual terms
        for s in (:updn, :upup)
            tdis = heatmap_tables(m, s, :KF; part = :disconnected)
            for (ν, νp) in rng_pts[1:8], η in 1:4
                w = [ν, -ν, νp, -νp]
                tot = zero(ComplexF64); mag = 0.0
                for P in tdis.con, t in eachindex(P.psf.w)
                    x = P.psf.w[t] * P.psf.ζ *
                        keldysh_kernel(w, collect(P.psf.pos[t]), [η], collect(P.psf.p), γ0)
                    tot += x; mag += abs(x)
                end
                @test mag > 1                       # individual terms are O(1/γ₀) or larger
                @test abs(tot) ≤ 1e-13 * mag
            end
        end
    end

    @testset "H3: Keldysh legs and exact identities" begin
        tab = tabs[(:updn, :KF)]
        ops = operators(m)
        Os2 = (ops.d_up, ops.dag_up)
        for ν in (-0.7, -0.01, 0.0, 0.3)
            L = leg(ν, 1, 1, tab.leg, γ0)
            @test L[1, 1] == 0
            @test L[2, 1] ≈ keldysh_correlator(m, Os2, [ν, -ν], [1]; γ0 = γ0) atol = 1e-13
            @test L[1, 2] ≈ keldysh_correlator(m, Os2, [ν, -ν], [2]; γ0 = γ0) atol = 1e-13
            @test L[2, 2] ≈ keldysh_correlator(m, Os2, [ν, -ν], [1, 2]; γ0 = γ0) atol = 1e-13
            for (a, b) in ((3, 1), (1, 3))
                @test leg(ν, a, b, tab.leg, γ0) ≈ _lehmann_leg(m, ν, a, b, γ0) atol = 1e-12
            end
        end
        # F^2222 ≡ 0; F^[η]_↑↑ ≡ 0
        buf = zeros(ComplexF64, 16)
        for (ν, νp) in rng_pts
            for s in (:updn, :upup)
                vertex_point!(buf, ν, νp, 0.0, HM_PRM, tabs[(s, :KF)])
                @test abs(buf[16]) ≤ 1e-12 * maximum(abs, buf)
                if s === :upup
                    for η in 1:4
                        c = LinearIndices((2, 2, 2, 2))[ntuple(i -> i == η ? 1 : 2, 4)...]
                        @test abs(buf[c]) ≤ 1e-12 * maximum(abs, buf)
                    end
                end
            end
        end
        # bare vertex
        @test bare_vertex_kf((1, 2, 2, 2), 1.0, true) == 0.5
        @test bare_vertex_kf((1, 1, 2, 2), 1.0, true) == 0
        @test bare_vertex_kf((1, 2, 2, 2), 1.0, false) == 0
        @test bare_vertex_mf(1.0, true) == 1 && bare_vertex_mf(1.0, false) == 0
    end

    @testset "H4: fully retarded vertex against Eq. (3)" begin
        buf = zeros(ComplexF64, 16)
        grid = range(-1, 1; length = 41)                   # contains ν = 0
        for ν in grid, νp in grid[1:4:end]
            vertex_point!(buf, ν, νp, 0.0, HM_PRM, tabs[(:updn, :KF)])
            w = ph_frequencies(ν, νp, 0.0)
            for η in 1:4
                c = LinearIndices((2, 2, 2, 2))[ntuple(i -> i == η ? 1 : 2, 4)...]
                ref = fret_updn(w, η, HM_U, γ0)
                @test abs(buf[c] - ref) ≤ 1e-11 * abs(ref)
            end
        end
        # 1122 and 1111 are purely imaginary (Fig. 11 caption)
        F = vertex_grid(collect(range(-1, 1; length = 21)), collect(range(-1, 1; length = 21)), 0.0,
                        HM_PRM, tabs[(:updn, :KF)])
        for k in ((1, 1, 2, 2), (1, 1, 1, 1))
            X = F[k..., :, :] .- bare_vertex_kf(k, HM_U, true)
            @test maximum(abs, real(X)) ≤ 1e-12 * maximum(abs, imag(X))
        end
    end

    @testset "P1 targets from Eq. (85)" begin
        ns = -24:23
        ms = [2n + 1 for n in ns]
        norm = 3π / 2 * (HM_U / (2π * HM_T))^5
        Fud = vertex_grid(ms, ms, 0, HM_PRM, tabs[(:updn, :MF)])
        Fuu = vertex_grid(ms, ms, 0, HM_PRM, tabs[(:upup, :MF)])
        Xud = real.(Fud .- HM_U) ./ HM_U ./ norm
        Xuu = real.(Fuu) ./ HM_U ./ norm
        @test maximum(abs, imag.(Fud)) ≤ 1e-10 * maximum(abs, Fud)
        @test all(Xud .> 0)
        @test maximum(Xud) ≈ 0.951 atol = 5e-4
        i = findfirst(==(1), ms)                          # ν = πT
        @test Xud[i, i] == maximum(Xud)
        @test minimum(Xud) ≈ 8.8e-5 rtol = 0.01
        @test all(Xuu .≤ 1e-12)
        @test minimum(Xuu) ≈ -0.344 atol = 5e-4
        @test all(abs(Xuu[j, j]) ≤ 1e-12 for j in eachindex(ms))
        # every point equals Eq. (85)
        @test maximum(abs(Fud[i, j] - vertex_exact(m, :updn, MatsubaraFreq.([ms[i], -ms[i], ms[j], -ms[j]])))
                      for i in eachindex(ms), j in eachindex(ms)) ≤ 1e-9
    end

    @testset "symlog" begin
        lt = 1e-3
        @test symlog(1.0, lt) == 1 && symlog(-1.0, lt) == -1 && symlog(0.0, lt) == 0
        @test symlog(5.0, lt) == 1                        # clamped
        @test symlog(lt, lt) ≈ 1 / 4
        @test symlog(1e-2, lt) ≈ 2 / 4
        @test symlog(-0.5lt, lt) ≈ -1 / 8
        xs = sort(randn(200) .* 10.0 .^ rand(-5:0, 200))
        @test issorted(symlog.(xs, lt))
    end
end
