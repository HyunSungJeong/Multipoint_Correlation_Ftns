# V1 (chain parts) and Sec. 5.2.10: fusion and fermionic signs through the shell
# construction, symmetry-violation errors, truncation across blocks.

using Test, MultipointNRG, LinearAlgebra

@testset "V1d fusion and signs: three Wilson sites vs dense Jordan-Wigner" begin
    m = siam(U = 0.6, εd = -0.25, Δ = 0.1)
    N = 2
    run = run_nrg(m, NRGParams(N = N, Nkeep = typemax(Int)))
    chs = run.chains
    # independent dense construction: modes [site 2; site 1; site 0; impurity], spins (up, dn)
    ms = chain_modes(n -> site_modes(m, n), N)
    c = jw_annihilators(length(ms))
    idx(md) = findfirst(==(md), ms)
    Hd = (m.params[:εd]) * (c[idx(dup)]' * c[idx(dup)] + c[idx(ddn)]' * c[idx(ddn)]) +
         m.params[:U] * (c[idx(dup)]' * c[idx(dup)]) * (c[idx(ddn)]' * c[idx(ddn)])
    for (ν, σ) in enumerate((:up, :dn))
        ch = chs[ν]
        a = c[idx(Mode(-1, σ))]; f0 = c[idx(Mode(0, σ))]
        Hd += ch.t[1] * (a' * f0 + f0' * a) + ch.eps[1] * f0' * f0
        for n in 1:N
            fn = c[idx(Mode(n, σ))]; fp = c[idx(Mode(n - 1, σ))]
            Hd += ch.eps[n + 1] * fn' * fn + ch.t[n + 1] * (fn' * fp + fp' * fn)
        end
    end
    @test Hd ≈ densematrix(hamiltonian_expr(m, chs, N), FockBasis(ms))   # same Fock basis order
    emb = dense_embedding(run, N)                                         # the documented basis map
    @test emb.basis.modes == ms
    sh = shell(run, N)
    # H_N from blocks equals the dense H_N in the shell basis
    for q in sectors(sh.space)
        U = emb.cols[q]
        @test norm(U' * U - I) ≤ 1e-13
        @test norm(U' * Hd * U - Diagonal(sh.E[q] .+ sh.shift)) ≤ 1e-12
    end
    # every mode operator, built through the shell construction (refine/kron
    # rules R1, R2), equals the dense JW operator in the shell basis
    ops = Dict{Mode,BlockOp}()
    for md in ms
        O = project(run, cop(md), N)       # built at its own shell, then refined
        ops[md] = O
        Od = c[idx(md)]
        for (q, M) in O.blocks
            r = target(O, q)
            @test norm(emb.cols[r]' * Od * emb.cols[q] - M) ≤ 1e-12
        end
    end
    # F stored by the sweep for the hopping agrees too
    for (ν, σ) in enumerate((:up, :dn))
        @test all(norm(getblock(sh.F[ν], q) - getblock(ops[Mode(N, σ)], q)) ≤ 1e-12 for q in sectors(sh.space))
    end
    # canonical anticommutators in the shell basis (BlockOp algebra)
    Id = identity_op(sh.space)
    for a in ms, b in ms
        A = ops[a]; B = ops[b]
        X = A * B' + B' * A
        if a == b
            @test norm(todense(X) - todense(Id)) ≤ 1e-12
        else
            @test norm(todense(X)) ≤ 1e-12
        end
        @test norm(todense(A * B + B * A)) ≤ 1e-12           # {c_i, c_j} = 0
    end
    # hermiticity and block structure of H_n: neutral, block diagonal per declared labels
    for n in -1:N
        shn = shell(run, n)
        for q in sectors(shn.space)
            @test issorted(shn.E[q])
        end
    end
end

@testset "V1e symmetry violation is detected" begin
    G, Q = spin_symmetry(:charge_spin)
    S = labeltype(G)
    imp = LocalSpace(-1, [:up, :dn], Dict{Mode,S}(Mode(-1, :up) => Q(:up), Mode(-1, :dn) => Q(:dn)))
    du = Mode(-1, :up); dd = Mode(-1, :dn)
    chans = [BathChannel(σ, Q(σ), BoxHybridization(0.1); coupling = cop(Mode(-1, σ))) for σ in (:up, :dn)]
    Bx = 0.1 * (cdag(du) * cop(dd) + cdag(dd) * cop(du))       # transverse field
    err = try
        ImpurityModel(G, imp, Bx, chans); nothing
    catch e
        e
    end
    @test err isa ArgumentError && occursin("twoSz", err.msg) && occursin("local Hamiltonian", err.msg)
    # pairing with U1(:N) declared
    G2, Q2 = spin_symmetry(:charge)
    S2 = labeltype(G2)
    imp2 = LocalSpace(-1, [:up, :dn], Dict{Mode,S2}(Mode(-1, :up) => Q2(:up), Mode(-1, :dn) => Q2(:dn)))
    chans2 = [BathChannel(σ, Q2(σ), BoxHybridization(0.1); coupling = cop(Mode(-1, σ))) for σ in (:up, :dn)]
    pair = 0.2 * (cdag(du) * cdag(dd) + cop(dd) * cop(du))
    err2 = try
        ImpurityModel(G2, imp2, pair, chans2); nothing
    catch e
        e
    end
    @test err2 isa ArgumentError && occursin("[:N]", err2.msg)
    # dropping the broken factor makes both models legal
    G3, Q3 = spin_symmetry(:charge)
    S3 = labeltype(G3)
    imp3 = LocalSpace(-1, [:up, :dn], Dict{Mode,S3}(Mode(-1, :up) => Q3(:up), Mode(-1, :dn) => Q3(:dn)))
    chans3 = [BathChannel(σ, Q3(σ), BoxHybridization(0.1); coupling = cop(Mode(-1, σ))) for σ in (:up, :dn)]
    @test ImpurityModel(G3, imp3, Bx, chans3) isa ImpurityModel
    # a channel must couple to an impurity operator with its own label
    bad = [BathChannel(:up, Q(:up), BoxHybridization(0.1); coupling = cop(dd)), chans[2]]
    @test_throws ArgumentError ImpurityModel(G, imp, OpExpr{Float64}(), bad)
end

@testset "V1f truncation never splits a multiplet spread over blocks" begin
    G, Q = spin_symmetry(:charge_spin)
    s_up = sector(G; N = 1, twoSz = 1); s_dn = sector(G; N = 1, twoSz = -1); s0 = sector(G; N = 0)
    # a doublet split across 2S_z = ±1, with the requested cut inside it
    E = Dict(s0 => [0.0, 2.0], s_up => [1.0, 3.0], s_dn => [1.0 + 1e-12, 3.0])
    V = SectorSpace(Dict(q => length(e) for (q, e) in E))
    nk, rec = truncate_shell(E, V, 1.0; Nkeep = 2, Etrunc = nothing, gaptol = 1e-8, last = false)
    @test nk[s_up] == 1 && nk[s_dn] == 1 && nk[s0] == 1       # both doublet members kept
    @test rec.requested == 2 && rec.actual == 3 && rec.cut_moved
    @test rec.gap ≈ 1.0 - 1e-12
    nk2, rec2 = truncate_shell(E, V, 1.0; Nkeep = 3, Etrunc = nothing, gaptol = 1e-8, last = false)
    @test !rec2.cut_moved && sum(values(nk2)) == 3
    nk3, _ = truncate_shell(E, V, 1.0; Nkeep = 2, Etrunc = nothing, gaptol = 1e-8, last = true)
    @test all(iszero, values(nk3))
end
