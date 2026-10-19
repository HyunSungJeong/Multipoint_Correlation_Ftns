# Moved from mpNRG/test/test_v1_symmetry.jl (V1a, V1b of the mpNRG
# verification record): the abelian labels and the rank-2 block operators.

using Test, TNOps, LinearAlgebra, Random

@testset "V1a labels: group axioms and canonical forms" begin
    G = SymmetryGroup(U1(:N), U1(:twoSz), Zn(:C3, 3))
    @test factornames(G) == (:N, :twoSz, :C3, :Pf)          # parity appended last
    S = labeltype(G)
    rng = MersenneTwister(1)
    rand_label() = sector(G, [rand(rng, -5:5), rand(rng, -5:5), rand(rng, -7:7), rand(rng, -3:3)])
    for _ in 1:200
        a, b, c = rand_label(), rand_label(), rand_label()
        @test fuse(fuse(a, b), c) == fuse(a, fuse(b, c))
        @test fuse(a, b) == fuse(b, a)
        @test fuse(a, neutral(G)) == a
        @test isneutral(fuse(a, dual(a)))
        @test 0 ≤ a.q[3] < 3 && 0 ≤ a.q[4] < 2              # Z_n components canonical
    end
    @test sector(G, (0, 0, 4, 3)) == sector(G, (0, 0, 1, 1))
    @test isbitstype(S)
    @test degeneracy(rand_label()) == 1
    # the parity factor is present even with no declared symmetry
    G0 = SymmetryGroup()
    @test factornames(G0) == (:Pf,)
    @test_throws ArgumentError SymmetryGroup(U1(:N), U1(:N))
end

function random_space(rng, G, labels)
    SectorSpace(Dict(sector(G, l) => rand(rng, 1:4) for l in labels))
end

function random_blockop(rng, row, col, Δ)
    A = BlockOp{Float64}(row, col, Δ)
    for q in sectors(col)
        r = fuse(q, Δ)
        haskey(row, r) || continue
        rand(rng) < 0.85 && setblock!(A, q, randn(rng, dim(row, r), dim(col, q)))
    end
    return A
end

@testset "V1b block algebra against todense" begin
    rng = MersenneTwister(2)
    G = SymmetryGroup(U1(:N), U1(:twoSz))
    labs = [(n, s, mod(n, 2)) for n in -1:3 for s in -3:3 if mod(n + s, 2) == 0 || true]
    labs = unique([(n, s, mod(n, 2)) for (n, s, _) in labs])
    V = random_space(rng, G, labs)
    shifts = [sector(G; N = 1, twoSz = 1), sector(G; N = -1, twoSz = 1), neutral(G), sector(G; N = 0, twoSz = 2, Pf = 0)]
    for Δa in shifts, Δb in shifts
        A = random_blockop(rng, V, V, Δa); B = random_blockop(rng, V, V, Δb)
        @test norm(todense(A * B) - todense(A) * todense(B)) ≤ 1e-14 * max(1, norm(todense(A)) * norm(todense(B)))
        @test norm(todense(A') - todense(A)') == 0
        if Δa == Δb
            @test norm(todense(A + B) - (todense(A) + todense(B))) ≤ 1e-14 * (1 + norm(todense(A)))
        else
            @test_throws ArgumentError A + B
        end
        C = A * B
        if isneutral(C.shift)
            @test abs(tr(C) - tr(todense(C))) ≤ 1e-13 * (1 + abs(tr(todense(C))))
        else
            @test tr(C) == 0                                  # exactly zero: no diagonal block
            @test abs(tr(todense(C))) ≤ 1e-14 * (1 + norm(todense(C)))
        end
    end
    @test norm(todense(2.5 * random_blockop(rng, V, V, shifts[1])) ) > 0
    # block-wise eigen of a hermitian neutral operator vs dense eigen
    H = random_blockop(rng, V, V, neutral(G)); H = H + H'
    ev = blockeigen(H)
    allvals = sort(vcat([v for (v, _) in values(ev)]...))
    @test norm(allvals - eigvals(Hermitian(todense(H)))) ≤ 1e-12
    for (q, (vals, vecs)) in ev
        @test norm(getblock(H, q) * vecs - vecs * Diagonal(vals)) ≤ 1e-12
    end
    # block-wise svd
    A = random_blockop(rng, V, V, shifts[1])
    sv = sort(vcat([s.S for s in values(blocksvd(A))]...); rev = true)
    @test norm(sv - svdvals(todense(A))[1:length(sv)]) ≤ 1e-12
end
