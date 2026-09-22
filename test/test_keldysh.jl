# The Keldysh formalism: K1-K4 and the validation ladder V1-V6.
#
# Equations are Kugler, Lee & von Delft, PRX 11, 041006 (2021), read from the
# printed page. Test order follows the task: combinatorics in isolation first,
# then kernels, then physics.

# Negative controls: kernels with a known bookkeeping error, used to show which
# checks are sensitive to which mistake.
_P(z, p, wp) = (acc = zero(ComplexF64); K = one(ComplexF64);
                for i in 1:(length(z) - 1); acc += z[p[i]]; K /= (acc - wp[i]); end; K)
# (-1)^{j-1} paired with the EXTERNAL order of the 2's instead of the permuted
# order. Right for α ≤ 1, wrong for α ≥ 2: the trap the task describes.
_kernel_extorder(ω, wp, k, p, g) = isempty(k) ? zero(ComplexF64) :
    sum((-1)^(j - 1) * _P(omega_eta(ω, e, g), p, wp) for (j, e) in enumerate(k))
_G(cache, ω, k, g, K) = 2.0^(1 - length(ω) / 2) *
    sum(d.ζ * K(ω, t.position, k, d.p, g) * t.weight for d in cache for t in d.terms;
        init = zero(ComplexF64))

@testset "Keldysh formalism, Eqs. (49)-(69)" begin

    # ------------------------------------------------------------------
    # K1
    # ------------------------------------------------------------------
    @testset "K1: complex tuples ω^[η], Eq. (49)" begin
        ω = [0.3, -1.1, 0.7, 0.1]
        for η in 1:4, γ0 in (1e-1, 1e-6)
            z = omega_eta(ω, η, γ0)
            @test real(z) ≈ ω
            @test imag(z[η]) ≈ 3γ0                         # (ℓ-1)γ₀ in slot η
            @test all(imag(z[i]) ≈ -γ0 for i in 1:4 if i != η)
            @test abs(sum(z)) < 1e-14                      # ω^[η]_{1⋯ℓ} = 0
        end
        # p. 10 above Eq. (52): the composite ω^[η]_{1̄⋯ī} has negative
        # imaginary part for 1 ≤ i < μ and positive for μ ≤ i < ℓ, μ = p⁻¹(η)
        for ℓ in 2:4, η in 1:ℓ, p in all_permutations(ℓ)
            w = [-1.0 + 0.6i for i in 1:ℓ]; w[end] = -sum(w[1:end-1])
            z = omega_eta(w, η, 1e-3)
            μ = invperm(p)[η]
            acc = zero(ComplexF64)
            for i in 1:(ℓ - 1)
                acc += z[p[i]]
                @test i < μ ? imag(acc) < 0 : imag(acc) > 0
            end
        end
        @test_throws ArgumentError omega_eta(ω, 5, 0.1)
        @test_throws ArgumentError omega_eta(ω, 1, 0.0)
    end

    # ------------------------------------------------------------------
    # K2
    # ------------------------------------------------------------------
    @testset "K2: Keldysh index bookkeeping, p. 13" begin
        @test keldysh_slots([1, 1, 1, 1]) == Int[]              # 1111 = []
        @test keldysh_slots([2, 1, 2, 1]) == [1, 3]             # 2121 = [13]
        @test keldysh_digits([1, 3], 4) == [2, 1, 2, 1]
        @test keldysh_label([2, 4], 4) == "1212"
        @test length(all_keldysh_components(4)) == 16
        @test_throws ArgumentError keldysh_slots([1, 3])

        # the worked example, including the intermediate [μ₁μ₂]
        khat, ηbh, μ = permuted_keldysh([2, 4], [4, 1, 2, 3])
        @test μ == [3, 1]                                       # [μ₁μ₂] = [31]
        @test khat == [1, 3]                                    # k_p = 2121 = [13]
        @test keldysh_label(khat, 4) == "2121"
        # NB: p(η_j) instead of p⁻¹(η_j) ALSO gives k_p = [13] here -- p and
        # p⁻¹ both map {2,4} to {1,3}. Only [μ₁μ₂] separates them, which is
        # why the intermediate must be reproduced.
        @test sort([4, 1, 2, 3][[2, 4]]) == [1, 3]
        @test [4, 1, 2, 3][[2, 4]] != μ

        # Fig. 3: ℓ = 4, η = 4, p = (3142) → p⁻¹ = (2413), μ = 3, η̄ = 2
        @test invperm([3, 1, 4, 2]) == [2, 4, 1, 3]
        @test invperm([3, 1, 4, 2])[4] == 3
        @test [3, 1, 4, 2][4] == 2

        # Definitive check against a convention-free reference: permuting the
        # digits, k_p = (k_{p(1)}, …, k_{p(ℓ)}). And {η̄̂_j} must be a
        # permutation of {η_j}, for every component and every p.
        for ℓ in 2:4
            wrong = 0
            for k in all_keldysh_components(ℓ), p in all_permutations(ℓ)
                ref = keldysh_slots([keldysh_digits(k, ℓ)[p[i]] for i in 1:ℓ])
                khat, ηbh, _ = permuted_keldysh(k, p)
                @test khat == ref
                @test issorted(khat)
                @test sort(ηbh) == sort(k)
                @test ηbh == [p[h] for h in khat]
                wrong += sort([p[e] for e in k]) != ref
            end
            # every ℓ = 2 permutation is an involution, so ℓ = 2 cannot tell
            # p from p⁻¹; ℓ ≥ 3 can, and often
            @test ℓ == 2 ? wrong == 0 : wrong > 0
        end
    end

    # ------------------------------------------------------------------
    # K3
    # ------------------------------------------------------------------
    @testset "K3: kernels, Eqs. (52), (63), (67b)" begin
        ω = [0.4, -0.9, 1.3, -0.8]
        wp = [0.1, -0.2, 0.3]
        # α = 0: K^[] = K^{1⋯1} = 0
        for p in all_permutations(4)
            @test keldysh_kernel(ω, wp, Int[], p, 0.1) == 0
            @test keldysh_kernel_eq63(ω, wp, Int[], p, 0.1) == 0
        end
        # α = 1 recovers Eq. (52). Eq. (52)'s label is a PERMUTED slot, so the
        # retarded kernel is called with η_perm = p⁻¹(η_ext), whence η̄ = η_ext.
        for ℓ in 2:4, η in 1:ℓ, p in all_permutations(ℓ)
            w = [-1.3 + 0.7i for i in 1:ℓ]; w[end] = -sum(w[1:end-1])
            v = [-0.4 + 0.3i for i in 1:(ℓ - 1)]
            @test keldysh_kernel(w, v, [η], p, 0.02) ≈
                  retarded_kernel(w, v, invperm(p)[η], p, 0.02) rtol = 1e-14
        end
        # Eq. (67b) against the independent route Eq. (63) + Eq. (52), which
        # reads permuted digits directly and uses no ordered lists
        for ℓ in 2:4, k in all_keldysh_components(ℓ), p in all_permutations(ℓ)
            w = [-0.9 + 0.8i for i in 1:ℓ]; w[end] = -sum(w[1:end-1])
            v = [0.3 - 0.35i for i in 1:(ℓ - 1)]
            @test keldysh_kernel(w, v, k, p, 0.05) ≈ keldysh_kernel_eq63(w, v, k, p, 0.05) atol = 1e-12
        end
    end

    m = HubbardAtomModel(2.0, 5.0)
    sp = spectrum(m)
    ops = operators(m)
    Oud = (ops.d_up, ops.dag_up, ops.d_dn, ops.dag_dn)
    Ouu = (ops.d_up, ops.dag_up, ops.d_up, ops.dag_up)

    # ------------------------------------------------------------------
    # V1
    # ------------------------------------------------------------------
    @testset "V1: G^{1⋯1} = 0 for ℓ = 2, 3, 4" begin
        for (Os, fer) in [((ops.d_up, ops.dag_up), [true, true]),
                          ((ops.d_up, ops.n_up, ops.dag_up), [true, false, true]),
                          (Oud, fill(true, 4))]
            ℓ = length(Os)
            for γ0 in (0.3, 1e-4), s in 1:3
                w = [sin(1.7s + i) for i in 1:ℓ]; w[end] = -sum(w[1:end-1])
                @test keldysh_correlator(m, Os, w, Int[]; γ0 = γ0, sp = sp,
                                         is_fermionic = fer) == 0
            end
        end
    end

    # ------------------------------------------------------------------
    # V2
    # ------------------------------------------------------------------
    @testset "V2: ℓ = 2 against Appendix C" begin
        # (C2a)/(C2b), written out independently as eigenstate sums. They hold
        # exactly at finite γ₀, fermionic and bosonic.
        function C2(sp, A, B, ω, γ0, ζ)
            Ar = to_eigenbasis(sp, A); Br = to_eigenbasis(sp, B)
            E, ρ = sp.E, sp.ρ
            R = Ad = K = zero(ComplexF64)
            for a in eachindex(E), b in eachindex(E)
                c = Ar[a, b] * Br[b, a]; c == 0 && continue
                E21 = E[b] - E[a]
                R += c * (ρ[a] - ζ * ρ[b]) / (ω + im * γ0 - E21)
                Ad += c * (ρ[a] - ζ * ρ[b]) / (ω - im * γ0 - E21)
                K += c * (ρ[a] + ζ * ρ[b]) * (1 / (ω + im * γ0 - E21) - 1 / (ω - im * γ0 - E21))
            end
            return R, Ad, K
        end
        for (U, β) in PARAMS
            mm = HubbardAtomModel(U, β); spp = spectrum(mm); o = operators(mm)
            for (A, B, ζ, fer) in [(o.d_up, o.dag_up, -1, [true, true]),
                                   (o.n_up, o.n_dn, 1, [false, false]),
                                   (o.n_up, o.n_up, 1, [false, false])],
                γ0 in (0.3, 1e-3), ω in (-1.7, 0.4, 1.3)
                R, Ad, K = C2(spp, A, B, ω, γ0, ζ)
                gc = k -> keldysh_correlator(mm, (A, B), [ω, -ω], k; γ0 = γ0, sp = spp,
                                             is_fermionic = fer)
                @test gc([1]) ≈ R atol = 1e-12         # G²¹ = G^[1]  = G^R
                @test gc([2]) ≈ Ad atol = 1e-12        # G¹² = G^[2]  = G^A
                @test gc([1, 2]) ≈ K atol = 1e-12      # G²² = G^[12] = G^K
                @test gc(Int[]) == 0                   # G¹¹ = 0
            end
        end

        # the Hubbard atom in closed form, exact at finite γ₀
        u = half_interaction(m); th = tanh(m.β * u / 2)
        for γ0 in (0.5, 1e-4), ω in (-2.3, -0.4, 0.0, 0.7)
            ga = k -> keldysh_correlator(m, (ops.d_up, ops.dag_up), [ω, -ω], k; γ0 = γ0, sp = sp)
            @test ga([1]) ≈ 0.5 * (1 / (ω + im * γ0 + u) + 1 / (ω + im * γ0 - u)) atol = 1e-12
            @test ga([2]) ≈ 0.5 * (1 / (ω - im * γ0 + u) + 1 / (ω - im * γ0 - u)) atol = 1e-12
            @test ga([1, 2]) ≈ 0.5 * (-th * (1 / (ω + im * γ0 + u) - 1 / (ω - im * γ0 + u)) +
                                      th * (1 / (ω + im * γ0 - u) - 1 / (ω - im * γ0 - u))) atol = 1e-12
        end

        # FDT, Eq. (C4): holds only as γ₀ → 0 -- the paper says so explicitly.
        # At the pole ω = E the ratio converges to tanh(βE/2) as O(γ₀²).
        for E in (1.0, -1.0)
            errs = map((1e-2, 1e-3, 1e-4)) do γ0
                gf = k -> keldysh_correlator(m, (ops.d_up, ops.dag_up), [E, -E], k; γ0 = γ0, sp = sp)
                abs(gf([1, 2]) / (gf([1]) - gf([2])) - tanh(m.β * E / 2))
            end
            @test errs[1] > errs[2] > errs[3]
            @test errs[3] < 1e-7
            @test errs[1] / errs[2] ≈ 100 rtol = 0.05          # quadratic in γ₀
        end
        # ...and at finite γ₀ off the pole it genuinely fails
        go = k -> keldysh_correlator(m, (ops.d_up, ops.dag_up), [0.6, -0.6], k; γ0 = 0.1, sp = sp)
        @test abs(go([1, 2]) - tanh(m.β * 0.6 / 2) * (go([1]) - go([2]))) > 1e-3
    end

    # ------------------------------------------------------------------
    # V3
    # ------------------------------------------------------------------
    @testset "V3: fully retarded vs the continued regular MF sum, Eq. (69)" begin
        cases = [((ops.d_up, ops.dag_up), [true, true]),
                 ((ops.d_up, ops.n_up, ops.dag_up), [true, false, true]),
                 ((ops.n_dn, ops.d_up, ops.dag_up), [false, true, true]),
                 (Oud, fill(true, 4)), (Ouu, fill(true, 4))]
        for (Os, fer) in cases
            ℓ = length(Os)
            cache = permuted_psfs(sp, Os; is_fermionic = fer)
            for s in 1:6, γ0 in (0.2, 1e-3), η in 1:ℓ
                w = [2 * sin(0.9s + 1.3i) for i in 1:ℓ]; w[end] = -sum(w[1:end-1])
                kf = keldysh_correlator(m, Os, w, [η]; γ0 = γ0, sp = sp, cache = cache,
                                        is_fermionic = fer)
                mf = regular_sum(sp, Os, omega_eta(w, η, γ0); cache = cache, is_fermionic = fer)
                @test 2.0^(ℓ / 2 - 1) * kf ≈ mf atol = 1e-10       # Eq. (69)
            end
        end

        # The ↑↑ fully retarded components vanish identically: F↑↑ in
        # Eq. (85b) is purely anomalous, so there is no regular part to continue.
        cud = permuted_psfs(sp, Oud); cuu = permuted_psfs(sp, Ouu)
        w = [0.9, -0.3, 1.4, -2.0]
        for η in 1:4
            big = abs(keldysh_correlator(m, Oud, w, [η]; γ0 = 1e-3, sp = sp, cache = cud))
            @test big > 1
            @test abs(keldysh_correlator(m, Ouu, w, [η]; γ0 = 1e-3, sp = sp, cache = cuu)) < 1e-10 * big
        end

        # The flag the task asks for: kernel = :regular gives G̃ of Eq. (68b).
        # It agrees with regular_sum on the axis, and refuses where a composite
        # vanishes -- exactly where the full Eq. (46) kernel adds anomalous terms
        # that Eq. (69) must NOT continue.
        nsing = 0
        for n1 in -2:2, n2 in -2:2, n3 in -2:2
            ms = vertex_frequencies(n1, n2, n3)
            try
                a = correlator(m, Oud, ms; sp = sp, kernel = :regular)
                @test a ≈ regular_sum(sp, Oud, [im * value(f, m.β) for f in ms]) atol = 1e-12
                @test a ≈ correlator(m, Oud, ms; sp = sp) atol = 1e-12  # same off the singular set
            catch e
                e isa DomainError || rethrow()
                nsing += 1
            end
        end
        @test nsing > 0
        @test_throws ArgumentError correlator(m, Oud, vertex_frequencies(0, 0, 0); sp = sp,
                                              kernel = :bogus)
    end

    # ------------------------------------------------------------------
    # Eq. (31): the formalism-independent split, validated in the MF
    # ------------------------------------------------------------------
    @testset "Eq. (31) disconnected PSF reproduces Eq. (73)" begin
        for (U, β) in PARAMS
            mm = HubbardAtomModel(U, β); spp = spectrum(mm); o = operators(mm)
            for (σ, σp) in [(:up, :dn), (:up, :up)]
                dσ, dgσ = spin_operators(o, σ); dσp, dgσp = spin_operators(o, σp)
                Os = (dσ, dgσ, dσp, dgσp)
                cd = permuted_psfs(spp, Os; part = :disconnected)
                cc = permuted_psfs(spp, Os; part = :connected)
                for n1 in -2:2, n2 in -2:2, n3 in -1:1
                    ms = vertex_frequencies(n1, n2, n3)
                    @test correlator(mm, Os, ms; sp = spp, cache = cd) ≈
                          disconnected_4p(mm, σ, σp, ms; sp = spp, ops = o) atol = 1e-12
                    @test correlator(mm, Os, ms; sp = spp, cache = cc) ≈
                          connected_4p(mm, σ, σp, ms; sp = spp, ops = o) atol = 1e-12
                end
            end
        end
        @test_throws ArgumentError psf_disconnected(sp, (ops.d_up, ops.dag_up))
        @test_throws ArgumentError permuted_psfs(sp, (ops.d_up, ops.dag_up); part = :connected)
    end

    # ------------------------------------------------------------------
    # V4 and V6
    # ------------------------------------------------------------------
    pts = Dict("generic" => [0.37, -1.13, 0.71, 0.05],
               "w12"     => [0.37, -0.37, 0.71, -0.71],
               "w13"     => [0.37, 0.71, -0.37, -0.71],
               "w14"     => [0.37, 0.71, -0.71, -0.37])
    # the slots paired by each vanishing bosonic frequency
    pairs = Dict("w12" => (1, 2), "w13" => (1, 3), "w14" => (1, 4))
    # |G| scaling between γ₀ = 1e-4 and 1e-6, as a slope in log γ₀
    function scaling(cache, w, k, K = keldysh_kernel)
        a = abs(_G(cache, w, k, 1e-4, K)); b = abs(_G(cache, w, k, 1e-6, K))
        max(a, b) < 1e-6 && return :zero
        s = (log10(b) - log10(a)) / (-2)
        s < -0.5 ? :delta : s > 0.5 ? :vanish : :finite
    end
    # does component k (a slot list) have exactly one 2 inside the pair?
    split(k, pr) = count(in(pr), k) == 1

    @testset "V4: all 16 components under a γ₀ scan (U=2, β=5)" begin
        cud = permuted_psfs(sp, Oud; part = :connected)
        cuu = permuted_psfs(sp, Ouu; part = :connected)
        for k in all_keldysh_components(4)
            α = length(k)
            for (name, w) in pts
                sud = scaling(cud, w, k)
                if α == 0
                    @test sud == :zero
                elseif α == 1 || α == 3
                    @test sud == :finite                  # never a δ, at any point
                elseif α == 4
                    @test sud == :vanish
                elseif name == "generic"
                    @test sud == :vanish                  # α = 2: → 0 linearly
                else
                    # the rule: at ω_ab = 0, the α = 2 components with exactly
                    # one 2 in {a,b} carry a δ; the others vanish
                    @test sud == (split(k, pairs[name]) ? :delta : :vanish)
                end
            end
            # fully retarded ↑↑ components are identically zero (V3)
            α == 1 && @test all(abs(_G(cuu, w, k, 1e-6, keldysh_kernel)) < 1e-6 for w in values(pts))
        end
    end

    @testset "V4 catches the external-order trap that V3 cannot" begin
        cuu = permuted_psfs(sp, Ouu; part = :connected)
        w = pts["generic"]
        for k in all_keldysh_components(4)
            a = _G(cuu, w, k, 1e-3, keldysh_kernel)
            b = _G(cuu, w, k, 1e-3, _kernel_extorder)
            if length(k) ≤ 1
                @test a ≈ b atol = 1e-12                 # invisible to V1-V3
            end
        end
        ndiff = sum(scaling(cuu, w, k) != scaling(cuu, w, k, _kernel_extorder)
                    for k in all_keldysh_components(4) for w in values(pts))
        @test ndiff > 10
    end

    @testset "V6: which components carry the anomalous weight" begin
        cuu = permuted_psfs(sp, Ouu; part = :connected)
        # F↑↑ in Eq. (85b) has δ_{ω14} and δ_{ω12} but NO δ_{ω13}: at ω13 = 0
        # nothing in the ↑↑ channel may carry a δ, while ↑↓ does
        @test all(scaling(cuu, pts["w13"], k) != :delta for k in all_keldysh_components(4))
        cud = permuted_psfs(sp, Oud; part = :connected)
        @test any(scaling(cud, pts["w13"], k) == :delta for k in all_keldysh_components(4))
        # ↑↑ at ω12 and ω14 follows the same split rule as ↑↓
        for name in ("w12", "w14"), k in all_keldysh_components(4)
            length(k) == 2 || continue
            @test scaling(cuu, pts[name], k) == (split(k, pairs[name]) ? :delta : :vanish)
        end

        # Quantitatively: the δ weight c = lim γ₀·G^con,k scales with U as the
        # anomalous coefficients of Eq. (85). th = tanh(βu/2) ≈ βu/2 at small U,
        # so the ↑↓ δ_{ω12} term, ∝ βu²·th, carries one power of U more.
        function exponent(Os, w, k)
            cs = map((0.08, 0.02)) do U
                mm = HubbardAtomModel(U, 5.0)
                c = permuted_psfs(spectrum(mm), Os; part = :connected)
                abs(1e-7 * _G(c, w, k, 1e-7, keldysh_kernel))
            end
            (log10(cs[2]) - log10(cs[1])) / (log10(0.02) - log10(0.08))
        end
        @test exponent(Oud, pts["w12"], [1, 3]) ≈ 3 atol = 0.1      # βu²·th
        @test exponent(Oud, pts["w13"], [1, 2]) ≈ 2 atol = 0.1      # βu²(th-1)
        @test exponent(Oud, pts["w14"], [1, 2]) ≈ 2 atol = 0.1      # βu²(th+1)
        @test exponent(Ouu, pts["w12"], [1, 3]) ≈ 2 atol = 0.1
        @test exponent(Ouu, pts["w14"], [1, 2]) ≈ 2 atol = 0.1

        # Appendix E's model case, at ℓ = 2: a conserved density has G^R ≡ 0
        # ([H, n] = 0), while G^K carries all the weight as δ(ω). It is the KF
        # face of the MF anomalous term G(iω) ∝ β δ_{ω,0}.
        gn = (k, ω, γ0) -> keldysh_correlator(m, (ops.n_up, ops.n_up), [ω, -ω], k; γ0 = γ0, sp = sp,
                                              is_fermionic = [false, false])
        for ω in (0.0, 0.8), γ0 in (1e-2, 1e-5)
            @test abs(gn([1], ω, γ0)) < 1e-12
        end
        @test abs(gn([1, 2], 0.0, 1e-5)) / abs(gn([1, 2], 0.0, 1e-3)) ≈ 100 rtol = 1e-6   # ∝ 1/γ₀
        @test abs(gn([1, 2], 0.8, 1e-5)) < 1e-4                                          # δ(ω) only
    end

    # ------------------------------------------------------------------
    # V5
    # ------------------------------------------------------------------
    @testset "V5: U = 0 null test" begin
        m0 = HubbardAtomModel(0.0, 2.0); sp0 = spectrum(m0); o0 = operators(m0)
        for Os in [(o0.d_up, o0.dag_up, o0.d_dn, o0.dag_dn), (o0.d_up, o0.dag_up, o0.d_up, o0.dag_up)]
            con = permuted_psfs(sp0, Os; part = :connected)
            # Wick's theorem: S - S^dis vanishes as a measure, for every p
            for d in con
                @test all(abs(t.weight) < 1e-14 for t in aggregate_psf(d.terms))
            end
            # hence every connected Keldysh component vanishes, at generic and
            # δ points alike, at any γ₀
            for w in values(pts), k in all_keldysh_components(4), γ0 in (1e-1, 1e-4)
                @test abs(keldysh_correlator(m0, Os, w, k; γ0 = γ0, sp = sp0, cache = con)) < 1e-10
            end
        end
        # Caveat, asserted so it stays documented: because the split happens on
        # the PSFs, the cancellation above is per permutation and does not test
        # ζ_p or K2. Nor does the tempting alternative, "the full G vanishes at
        # generic points": any alternating-sign kernel reduces there to the
        # regular sum, which is identically zero at U = 0 -- the trap kernel
        # even gives exactly zero. V3, the Eq. (63) identity and V4 are the
        # checks with teeth for ζ_p and K2.
        full = permuted_psfs(sp0, (o0.d_up, o0.dag_up, o0.d_dn, o0.dag_dn))
        w = pts["generic"]
        trap = maximum(abs(_G(full, w, k, 1e-4, _kernel_extorder)) for k in all_keldysh_components(4))
        @test trap < 1e-12
    end
end
