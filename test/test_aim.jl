# The Anderson impurity model with a discrete bath: the AIM task sheet.
#
# Paper equations are Kugler, Lee & von Delft, PRX 11, 041006 (2021), read from
# the printed page. Reference values marked "sheet" come from the task sheet's
# independent Python exact diagonalisation, not from this code.

# the vertex from the MF tables at arbitrary integer Matsubara indices
function _aim_Fmf(m, tabs, ms)
    Gc = mf_correlator_point(ms, m.β, tabs.con)
    L = mf_leg(ms[1], m.β, tabs.leg) * mf_leg(-ms[2], m.β, tabs.leg) *
        mf_leg(ms[3], m.β, tabs.leg) * mf_leg(-ms[4], m.β, tabs.leg)
    return Gc / L
end

# generic, one point per δ channel, two channels at once, then scattered triples
const AIM_FREQS = [(1, 3, 5, -9), (1, -1, 3, -3), (1, 3, -1, -3), (1, 3, -3, -1), (1, -1, 1, -1),
                   (3, -1, -3, 1), (5, -3, 1, -3)]
const AIM_FREQS20 = vcat(AIM_FREQS,
    [(2a + 1, 2b + 1, 2c + 1, -(2a + 2b + 2c + 3)) for (a, b, c) in
     ((0, 1, -2), (2, -3, 1), (-1, -1, 4), (3, 0, -2), (1, 2, 2), (-2, 1, 0), (0, -4, 3),
      (4, -1, -1), (-3, 2, 2), (1, -2, -1), (2, 2, -5), (-1, 3, -3), (5, -2, 1))])

_impurity4(m, ops, σσ′) = (impurity_operators(m, ops, :up)...,
                           impurity_operators(m, ops, σσ′ === :updn ? :dn : :up)...)

# same merged peaks, matched by position rounded to 1e-9 (raw floats that agree to
# ~1e-16 can sort in a different order)
function _same_tables(A, B; wtol = 1e-14)
    key(x) = round.(x; digits = 9)
    for (a, b) in zip(A, B)
        a.p == b.p || return false
        length(a.w) == length(b.w) || return false
        ia, ib = sortperm(key.(a.pos)), sortperm(key.(b.pos))
        all(maximum(abs.(a.pos[x] .- b.pos[y])) ≤ 1e-12 for (x, y) in zip(ia, ib)) || return false
        all(abs(a.w[x] - b.w[y]) ≤ wtol for (x, y) in zip(ia, ib)) || return false
    end
    return length(A) == length(B)
end

@testset "Anderson impurity model, discrete bath" begin

    @testset "Sec. 1: discretisation, operators, symmetries" begin
        for Nb in 1:3
            ε, V = discretize_box(0.1, 1.0, box_edges(Nb))
            @test sum(abs2, V) ≈ 2 * 1.0 * 0.1 / π rtol = 1e-14           # sum rule
            @test ε ≈ -reverse(ε) atol = 1e-15                               # ε_n → -ε_{Nb+1-n}
        end
        @test discretize_box(0.1, 1.0, box_edges(1))[1] == [0.0]
        @test discretize_box(0.1, 1.0, box_edges(2))[1] ≈ [-0.5, 0.5]
        @test discretize_box(0.1, 1.0, box_edges(3))[1] ≈ [-2 / 3, 0, 2 / 3]
        edges = [-1.0, -0.37, 0.2, 0.9, 1.0]                                  # any partition
        @test sum(abs2, discretize_box(0.3, 1.0, edges)[2]) ≈ 2 * 0.3 / π rtol = 1e-14
        @test box_edges(3; Λ = 2.0) ≈ [-1, -0.5, 0.5, 1]
        @test_throws ArgumentError AIMModel(1.0, 0.0, [0.0], [0.1])
        @test_throws DimensionMismatch AIMModel(1.0, 1.0, [0.0, 0.1], [0.1])
        for Nb in 1:3
            m = aim_model(Nb; U = 1.3, β = 2.0)
            ops = operators(m)
            n = 4^(Nb + 1)
            @test size(ops.H) == (n, n)
            Id = Matrix(1.0I, n, n)
            ok = true
            for a in eachindex(ops.c), b in eachindex(ops.c)
                ok &= norm(ops.c[a] * ops.c[b]' + ops.c[b]' * ops.c[a] - (a == b) * Id) < 1e-13
                ok &= norm(ops.c[a] * ops.c[b] + ops.c[b] * ops.c[a]) < 1e-13
            end
            @test ok                                                         # all 2(Nb+1) modes
            @test norm(ops.H * ops.N - ops.N * ops.H) < 1e-12
            @test norm(ops.H * ops.Sz - ops.Sz * ops.H) < 1e-12
            for β in (0.5, 10.0, 100.0)
                sp = spectrum(aim_model(Nb; U = 1.3, β = β))
                for σ in (:up, :dn)
                    d, dd = impurity_operators(m, ops, σ)
                    @test thermal_average(sp, dd * d) ≈ 0.5 atol = 1e-12       # half filling
                end
            end
        end
    end

    @testset "A1: closed-form spectrum, Nb = 1" begin
        for (U, β, Δ) in ((2.0, 10.0, 0.1), (0.5, 3.0, 0.1), (3.0, 1.0, 0.4))
            m = aim_model(1; U = U, β = β, Δ = Δ)
            sp = spectrum(m)
            lev = sort(vcat([fill(e, d) for (e, d) in aim_levels_nb1(m)]...))
            @test maximum(abs, sp.E - lev) < 1e-13
            @test sp.Z ≈ aim_partition_function_nb1(m) rtol = 1e-13
            P = propagator_poles(m; sp = sp)
            @test length(P) == 8
            @test maximum(abs, [p[1] for p in P] - aim_poles_nb1(m)) < 1e-13
            @test sum(p[2] for p in P) ≈ 1 atol = 1e-13
        end
        # T → 0 residues at (U, Δ, D) = (2, 0.1, 1), sheet
        P = propagator_poles(aim_model(1; U = 2.0, β = 1000.0))
        res = Dict(round(p[1]; digits = 6) => p[2] for p in P)
        m = aim_model(1; U = 2.0, β = 1000.0)
        a, b = sqrt(4 + 64m.V[1]^2) / 4, sqrt(4 + 16m.V[1]^2) / 4
        @test res[round(a - b; digits = 6)] ≈ 0.172914 atol = 5e-7
        @test res[round(-(a - b); digits = 6)] ≈ 0.172914 atol = 5e-7
        @test res[round(a + b; digits = 6)] ≈ 0.327086 atol = 5e-7
        @test res[round(-(a + b); digits = 6)] ≈ 0.327086 atol = 5e-7
        @test sum(p[2] for p in P if abs(abs(p[1]) - (a - b)) > 1e-6 && abs(abs(p[1]) - (a + b)) > 1e-6) < 1e-12
    end

    @testset "A2: G_d(iπT) against the sheet's reference ED" begin
        ref = Dict((1, 0.5, 10) => -1.768830080892028, (1, 0.5, 100) => -0.485140003453097,
                   (1, 2.0, 10) => -0.737281826717041, (1, 2.0, 100) => -0.473355579919950,
                   (2, 0.5, 10) => -2.030691733046821, (2, 0.5, 100) => -0.949216078082626,
                   (2, 2.0, 10) => -0.429277187551225, (2, 2.0, 100) => -0.059820038830754,
                   (3, 0.5, 10) => -2.000856350026648, (3, 0.5, 100) => -1.391281201028393,
                   (3, 2.0, 10) => -0.513738264064291, (3, 2.0, 100) => -1.162698345747195)
        for ((Nb, U, β), r) in ref
            m = aim_model(Nb; U = U, β = β)
            sp = spectrum(m)
            @test default_psf_method(sp) === (Nb == 1 ? :dense : :chain)
            G = propagator(m, fermionic_freq(0); sp = sp)
            @test abs(imag(G) - r) ≤ 1e-12 * abs(r)
            @test abs(real(G)) ≤ 1e-12                                        # rounding level
            # the chain walk and the dense PSF agree
            Gd = correlator(m, impurity_operators(m, operators(m), :up), [fermionic_freq(0), MatsubaraFreq(-1)];
                            sp = sp, otol = 1e-12, method = :dense)
            @test abs(G - Gd) ≤ 1e-13
        end
    end

    @testset "A3: U = 0, G_d = [z - Δ(z)]⁻¹, MF and KF" begin
        for Nb in 1:3
            m = aim_model(Nb; U = 0.0, β = 10.0)
            sp = spectrum(m); ops = operators(m)
            for k in -4:3
                ν = fermionic_freq(k)
                @test abs(propagator(m, ν; sp = sp) - propagator_free(m, im * value(ν, 10.0))) < 1e-13
            end
            Os = impurity_operators(m, ops, :up)
            cache = permuted_psfs(sp, Os; otol = 1e-12)
            for ω in (-1.3, -0.4, 0.0, 0.25, 0.9), γ0 in (0.1, 0.01)
                GR = keldysh_correlator(m, Os, [ω, -ω], [1]; γ0 = γ0, sp = sp, cache = cache)
                @test abs(GR - propagator_free(m, ω + im * γ0)) ≤ 1e-12 * max(1, abs(GR))
            end
        end
        # the bare bubble of B3 against the sheet's brute Matsubara sum, Nb = 2, β = 10
        m = aim_model(2; U = 0.0, β = 10.0)
        @test real(chi0_bubble(m, 0)) ≈ 1.893894 atol = 5e-7
        @test real(chi0_bubble(m, 2)) ≈ 0.140828 atol = 5e-7
        @test real(chi0_bubble(m, 6)) ≈ 0.028022 atol = 5e-7
        # and the atom limit: G₀ = 1/iν gives χ₀ = β/4 δ_{ω,0}
        m0 = AIMModel(0.0, 7.0, [0.0], [0.0])
        @test real(chi0_bubble(m0, 0)) ≈ 7.0 / 4
        @test abs(chi0_bubble(m0, 4)) < 1e-14
    end

    @testset "I1: chain walk = dense PSF, peak by peak" begin
        at = HubbardAtomModel(2.0, 5.0); ao = operators(at)
        dm = HubbardDimerModel(0.7, 2.3, 3.0); dop = operators(dm)
        am = aim_model(1; U = 2.0, β = 10.0); aop = operators(am)
        cases = ((spectrum(at), (ao.d_up, ao.dag_up, ao.d_dn, ao.dag_dn), 0.0),
                 (spectrum(dm), dimer_operators_4p(dop, (0, 1, 1, 0), :updn), 1e-12),
                 (spectrum(am), _impurity4(am, aop, :updn), 1e-12),
                 (spectrum(am), _impurity4(am, aop, :upup), 1e-12))
        for (sp, Os, otol) in cases, part in (:full, :connected)
            D = psf4_tables(sp, Os; part = part, otol = otol, method = :dense)
            C, st = psf4_tables_chain(sp, Os; part = part, otol = otol)
            @test length(C) == 24
            @test _same_tables(D, C)
            if part === :full
                @test st.cycles == [length(psf(sp, ntuple(i -> Os[P.p[i]], 4); otol = otol)) for P in D]
            end
        end
        # unmerged terms of any ℓ: the same multiset of (states, weight)
        sp, Os, otol = cases[3]
        for ℓ in (2, 3, 4)
            O = Os[1:ℓ]
            t1 = sort([(t.states, t.weight) for t in psf(sp, O; otol = otol)]; by = x -> x[1])
            t2 = sort([(t.states, t.weight) for t in psf_chain(sp, O; otol = otol)]; by = x -> x[1])
            @test length(t1) == length(t2)
            @test all(a[1] == b[1] && a[2] == b[2] for (a, b) in zip(t1, t2))
        end
        # the 2p tables
        d, dd = impurity_operators(am, aop, :up)
        L1 = psf2_tables(spectrum(am), d, dd; otol = 1e-12, method = :dense)
        L2 = psf2_tables(spectrum(am), d, dd; otol = 1e-12, method = :chain)
        @test all(sort(a.pos) ≈ sort(b.pos) && sort(real(a.w)) ≈ sort(real(b.w)) for (a, b) in zip(L1, L2))
    end

    @testset "I2: Dict merge = pairwise merge; straddling guard" begin
        for m in (HubbardAtomModel(2.0, 5.0), HubbardDimerModel(0.7, 2.3, 3.0))
            sp = spectrum(m); ops = operators(m)
            Os = m isa HubbardAtomModel ? (ops.d_up, ops.dag_up, ops.d_dn, ops.dag_dn) :
                 dimer_operators_4p(ops, (0, 0, 0, 0), :updn)
            otol = m isa HubbardAtomModel ? 0.0 : 1e-12
            for part in (:full, :connected), d in permuted_psfs(sp, Os; part = part, otol = otol)
                p1, w1 = merge_peaks_pairwise(d.terms)
                p2, w2 = merge_peaks(d.terms)
                @test p1 == p2
                @test w1 == w2
            end
        end
        # after merging, no two positions closer than 10 TOL_DEG in every coordinate
        m = aim_model(2; U = 2.0, β = 10.0)
        T, _ = psf4_tables_chain(spectrum(m), _impurity4(m, operators(m), :updn); otol = 1e-12)
        for P in T[1:4]
            cells = Dict{NTuple{3,Int},Vector{Int}}()
            for (i, x) in enumerate(P.pos)
                push!(get!(cells, ntuple(k -> floor(Int, x[k] / (10TOL_DEG)), 3), Int[]), i)
            end
            close = 0
            for (k, idx) in cells, δ in Iterators.product(-1:1, -1:1, -1:1)
                for j in get(cells, ntuple(t -> k[t] + δ[t], 3), Int[]), i in idx
                    i < j && all(abs(P.pos[i][t] - P.pos[j][t]) < 10TOL_DEG for t in 1:3) && (close += 1)
                end
            end
            @test close == 0
        end
        # a peak split across a cell boundary is re-joined
        x = 0.5 * TOL_DEG
        terms = [PSFTerm([x - 1e-15, 1.0, 2.0], 1.0 + 0im, Int[]), PSFTerm([x + 1e-15, 1.0, 2.0], 2.0 + 0im, Int[]),
                 PSFTerm([3.0, 1.0, 2.0], 5.0 + 0im, Int[])]
        pos, w = merge_peaks(terms)
        @test length(w) == 2 && w[1] == 3.0 && pos[1] == terms[1].position
    end

    @testset "B1: U = 0 null test, MF and KF" begin
        for Nb in (1, 2)
            m = aim_model(Nb; U = 0.0, β = 10.0)
            sp = spectrum(m); ops = operators(m)
            for s in (:updn, :upup)
                T, _ = psf4_tables_chain(sp, _impurity4(m, ops, s); otol = 1e-12, wtol = 0.0)
                @test maximum(maximum(abs, P.w; init = 0.0) for P in T) < 1e-14    # S^con ≡ 0
                tm = heatmap_tables(m, s, :MF; sp = sp); tk = heatmap_tables(m, s, :KF; sp = sp)
                @test all(abs(_aim_Fmf(m, tm, ms)) < 1e-12 for ms in AIM_FREQS)
                buf = zeros(ComplexF64, 16)
                for (ν, νp, ω) in ((0.3, -0.2, 0.0), (0.0, 0.1, 0.0), (-0.7, 0.5, 0.15))
                    vertex_point!(buf, ν, νp, ω, HeatmapParams(0.0, 10.0, 0.1), tk)
                    @test maximum(abs, buf) < 1e-12
                end
            end
        end
    end

    @testset "B2: V → 0, F → F^atom, both branches of Eq. (46)" begin
        U, β = 2.0, 5.0
        at = HubbardAtomModel(U, β)
        for Nb in (1, 2)
            dev = Float64[]
            for V in (0.0, 1e-1, 1e-2, 1e-3)
                m = AIMModel(U, β, aim_model(Nb; U = U, β = β).ε, fill(V, Nb))
                e = 0.0
                for s in (:updn, :upup)
                    tm = heatmap_tables(m, s, :MF)
                    for ms in AIM_FREQS
                        e = max(e, abs(_aim_Fmf(m, tm, ms) - vertex_exact(at, s, MatsubaraFreq.(collect(ms)))))
                    end
                end
                push!(dev, e)
            end
            @test dev[1] < 1e-12                        # V = 0: the atom exactly
            @test 80 < dev[2] / dev[3] < 120            # ∝ V²
            @test 95 < dev[3] / dev[4] < 105
        end
    end

    @testset "B3: weak coupling, Eq. (84) with the discrete-bath χ₀" begin
        β = 10.0
        Us = (1e-2, 5e-3, 2.5e-3)
        for Nb in (1, 2), s in (:updn, :upup)
            tabs = [heatmap_tables(aim_model(Nb; U = U, β = β), s, :MF) for U in Us]
            m1 = aim_model(Nb; U = 1.0, β = β)
            for ms in AIM_FREQS20
                f = [(_aim_Fmf(aim_model(Nb; U = U, β = β), t, ms) - (s === :updn ? U : 0.0)) / U^2
                     for (t, U) in zip(tabs, Us)]
                lim = (8f[3] - 6f[2] + f[1]) / 3                   # Richardson, O(U) and O(U²) removed
                bracket = vertex_second_order(m1, s, ms) - (s === :updn ? 1.0 : 0.0)   # -[χ₀ ...] at U = 1
                @test abs(lim - bracket) < 3e-5 * max(1, abs(bracket))
            end
        end
    end

    @testset "B4: crossing" begin
        for Nb in (1, 2)
            m = aim_model(Nb; U = 2.0, β = 5.0)
            sp = spectrum(m)
            tud = heatmap_tables(m, :updn, :MF; sp = sp); tuu = heatmap_tables(m, :upup, :MF; sp = sp)
            for ms in AIM_FREQS
                Fuu, Fud = _aim_Fmf(m, tuu, ms), _aim_Fmf(m, tud, ms)
                @test abs(Fuu - (Fud - _aim_Fmf(m, tud, (ms[3], ms[2], ms[1], ms[4])))) < 1e-11
                @test abs(Fuu - (Fud - _aim_Fmf(m, tud, (ms[1], ms[4], ms[3], ms[2])))) < 1e-11
            end
        end
    end

    @testset "B5: Keldysh identities and the fully retarded vertex" begin
        γ0 = 0.1
        for Nb in (1, 2)
            m = aim_model(Nb; U = 2.0, β = 5.0)
            sp = spectrum(m); ops = operators(m)
            prm = HeatmapParams(2.0, 5.0, γ0)
            du, duu = impurity_operators(m, ops, :up)
            leg2 = permuted_psfs(sp, (du, duu); otol = 1e-12)
            Gt(x) = regular_sum(sp, (du, duu), [x, -x]; cache = leg2)
            buf = zeros(ComplexF64, 16); g = zeros(ComplexF64, 16)
            for s in (:updn, :upup)
                Os = _impurity4(m, ops, s)
                full = permuted_psfs(sp, Os; otol = 1e-12)
                tfull = heatmap_tables(m, s, :KF; sp = sp, part = :full)
                tcon = heatmap_tables(m, s, :KF; sp = sp)
                nonzero_uu = 0.0
                for (ν, νp, ω) in ((0.3, -0.2, 0.0), (0.0, 0.1, 0.0), (-0.7, 0.5, 0.15), (0.45, 0.45, -0.3))
                    kf_correlator_point!(g, ph_frequencies(ν, νp, ω), γ0, tfull.con)
                    @test g[1] == 0                                   # G^1111 = 0
                    vertex_point!(buf, ν, νp, ω, prm, tcon)
                    @test abs(buf[16]) ≤ 1e-12 * maximum(abs, buf)   # F^2222 ≡ 0
                    w = collect(ph_frequencies(ν, νp, ω))
                    for η in 1:4
                        z = omega_eta(w, η, γ0)
                        rhs = regular_sum(sp, Os, z; cache = full) / (Gt(z[1]) * Gt(-z[2]) * Gt(z[3]) * Gt(-z[4]))
                        c = LinearIndices((2, 2, 2, 2))[ntuple(i -> i == η ? 1 : 2, 4)...]
                        @test abs(2buf[c] - rhs) ≤ 1e-11 * abs(rhs)
                        s === :upup && (nonzero_uu = max(nonzero_uu, abs(buf[c])))
                    end
                end
                # trap 1: F^[η]_↑↑ ≡ 0 is atom-specific
                s === :upup && @test nonzero_uu > 1
            end
        end
    end

    @testset "O1: local spin susceptibility (sheet table)" begin
        ref = ((1.0, 0.18048, 0.18276), (10.0, 1.95387, 2.49989), (30.0, 2.23440, 7.5),
               (100.0, 2.02469, 25.0), (1000.0, 2.02467, 250.0))
        for (β, aim, atom) in ref
            @test local_susceptibility(aim_model(1; U = 2.0, β = β)) ≈ aim atol = 5e-6
            @test local_susceptibility(HubbardAtomModel(2.0, β)) ≈ atom atol = 5e-6
        end
        @test singlet_triplet_gap_nb1(aim_model(1; U = 2.0, β = 1.0)) ≈ 0.2104 atol = 5e-5
        for Δ in (0.1, 0.03, 4e-4)
            m = aim_model(1; U = 2.0, β = 1.0, Δ = Δ)
            @test singlet_triplet_gap(m) ≈ singlet_triplet_gap_nb1(m) atol = 1e-12
        end
        # even Nb: odd electron number, Kramers-doublet ground state, Curie-like χ_loc
        @test_throws ArgumentError singlet_triplet_gap(aim_model(2; U = 1.0, β = 1.0))
        χ50, χ500 = (local_susceptibility(aim_model(2; U = 1.0, β = β)) for β in (50.0, 500.0))
        @test 9 < χ500 / χ50 < 11                                   # ∝ β
        χ50, χ500 = (local_susceptibility(aim_model(3; U = 1.0, β = β)) for β in (50.0, 500.0))
        @test 0.8 < χ500 / χ50 < 1.0                                # saturates: singlet
    end

    @testset "O2 support: δ(ω₁₂) part reproduces Eq. (85) at V = 0" begin
        U, β = 2.0, 4.0
        ms = MatsubaraFreq.([1, -1, 3, -3])
        u, th = U / 2, tanh(β * U / 4)
        iω = [im * value(f, β) for f in ms]
        atom = β * u^2 * th * prod(iω .+ u) / prod(iω)
        @test delta_part(HubbardAtomModel(U, β), :updn, ms) ≈ atom rtol = 1e-12
        @test delta_part(AIMModel(U, β, [0.0], [0.0]), :updn, ms) ≈ atom rtol = 1e-12
    end
end
