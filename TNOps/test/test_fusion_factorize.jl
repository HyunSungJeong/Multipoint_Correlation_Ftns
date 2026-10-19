# Fusion, factorizations, the global cut, parity and ncon (refactor sheet
# Sec. 3.7), including the comparison with the frozen output of the
# pre-refactor fusionmap / kron_op / truncate_shell.

using Test, TNOps, LinearAlgebra, Random
import TNOps: legsectors, blockkeys, isallowed, legs, leg, flux

include(joinpath(@__DIR__, "fixtures", "old_layer_fixtures.jl"))

const TOL = 1e-13
relerr(a, b) = norm(a - b) / max(norm(b), 1e-300)

# ---------------------------------------------------------------- frozen fixtures

lab(M, t) = Sector{M,length(M)}(t)
legof(M, sp) = Leg(Dict(LegSector(lab(M, t)) => d for (t, d) in sp))
function opof(M, d)
    row = legof(M, d["row"]); col = legof(M, d["col"])
    A = BlockTensor{Float64}(row, col, lab(M, d["shift"]))
    for (t, B) in d["blocks"]
        setblock!(A, lab(M, t), B)
    end
    return A
end

@testset "fusionmap and kron_fused reproduce the pre-refactor output exactly" begin
    M = FX_MODULI
    for (n, fx) in FX_FUSION
        F = fusionmap(legof(M, fx["loc"]), legof(M, fx["prev"]))
        @test Set((s.label.q, d) for (s, d) in zip(legsectors(F.out), F.out.dims)) == Set(fx["out"])
        @test length(F.pairs) == length(fx["pairs"])
        for (t, prs) in fx["pairs"]
            mine = F.pairs[LegSector(lab(M, t))]
            @test [(p.σ.label.q, p.q.label.q, p.offset, p.dσ, p.dq) for p in mine] == prs   # order and offsets
        end
        @test all(F.lookup[(LegSector(lab(M, a)), LegSector(lab(M, b)))] == (LegSector(lab(M, c)), k)
                  for ((a, b), (c, k)) in fx["lookup"])
        @test F.loc === F.outer && F.prev === F.inner
    end
    for (name, fx) in FX_KRON
        X = opof(M, fx["X"]); Y = opof(M, fx["Y"])
        F = fusionmap(legof(M, fx["fusion"]["loc"]), legof(M, fx["fusion"]["prev"]))
        R = kron_fused(X, Y, F, F)
        old = opof(M, fx["out"])
        @test Set(blockkeys(R)) == Set(blockkeys(old))
        @test all(getblock(R, k) == getblock(old, k) for k in blockkeys(old))        # bit for bit
    end
end

@testset "global_cut gives the kept counts of the old truncate_shell" begin
    nchecked = 0
    for fx in FX_TRUNCATE
        fx["last"] && continue                         # the last-shell rule stays in mpNRG
        M = fx["moduli"]
        E = Dict(LegSector(lab(M, t)) => Float64.(e) for (t, e) in fx["E"])
        ω = fx["omega"]
        kept, rec = global_cut(E; nkeep = fx["Nkeep"], gaptol = fx["gaptol"] * ω)
        @test all(kept[LegSector(lab(M, t))] == v for (t, v) in fx["nk"])
        r = fx["rec"]
        @test (rec.requested, rec.actual, rec.cut_moved, rec.total) == (r["requested"], r["actual"], r["cut_moved"], r["total"])
        @test (isnan(rec.gap) && isnan(r["gap"])) || rec.gap / ω == r["gap"]
        nchecked += 1
    end
    @test nchecked ≥ 20
    # degeneracy-safe: a multiplet spread over two sectors is never split
    G = SymmetryGroup(U1(:N)); S = labeltype(G)
    E = Dict(LegSector(sector(G, (0, 0))) => [0.0, 1.0, 2.0], LegSector(sector(G, (1, 1))) => [1.0 + 1e-12, 3.0])
    kept, rec = global_cut(E; nkeep = 2, gaptol = 1e-8)
    @test kept[LegSector(sector(G, (0, 0)))] == 2 && kept[LegSector(sector(G, (1, 1)))] == 1 && rec.cut_moved
    kept, rec = global_cut(E; nkeep = 5, threshold = 1.5, gaptol = 1e-8)
    @test sum(values(kept)) == 3 && rec.requested == 3
end

# ---------------------------------------------------------------- random tensors

const GROUPS = [
    ("U1xU1xZ2", SymmetryGroup(U1(:N), U1(:twoSz)), rng -> (n = rand(rng, -2:2); (n, rand(rng, -2:2), mod(n, 2)))),
    ("Z3", SymmetryGroup(Zn(:C, 3)), rng -> (rand(rng, 0:2), rand(rng, 0:1))),
    ("parity", SymmetryGroup(), rng -> (rand(rng, 0:1),)),
]
function randleg(rng, G, gen; nsec = rand(rng, 2:4), dmax = 3, arrow = Out)
    secs = unique!([LegSector(sector(G, gen(rng)), rand(rng) < 0.3 ? rand(rng, 1:2) : 0) for _ in 1:(3nsec)])[1:min(nsec, end)]
    return Leg(secs, [rand(rng, 1:dmax) for _ in secs]; arrow = arrow)
end
randflux(rng, ls) = TNOps.keyflux(ls, ntuple(i -> rand(rng, legsectors(ls[i])), length(ls)))

@testset "fuse_legs ∘ split_legs = identity, and against dense: $gname" for (gname, G, gen) in GROUPS
    rng = MersenneTwister(sum(codeunits(gname)) + 1)
    for trial in 1:8, T in (Float64, ComplexF64)
        N = rand(rng, 2:4)
        ar = rand(rng, (In, Out))
        ls = ntuple(i -> randleg(rng, G, gen; arrow = i ≤ 2 ? ar : rand(rng, (In, Out))), N)
        A = randtensor(rng, T, ls, randflux(rng, ls))
        B, F = fuse_legs(A, (1, 2))
        @test all(k -> isallowed(B, k), blockkeys(B))
        @test split_legs(B, 1, F) ≈ A atol = 0
        @test fuse_legs(split_legs(B, 1, F), (1, 2))[1] ≈ B atol = 0
        # dense: fused index = i_inner + (i_outer - 1) d_inner within each pair
        D = todense(A); DB = todense(B)
        offB = TNOps.offsets(leg(B, 1)); offA1 = TNOps.offsets(ls[1]); offA2 = TNOps.offsets(ls[2])
        ok = true
        for (qq, prs) in F.pairs, p in prs, iσ in 1:p.dσ, s in 1:p.dq
            iB = offB[qq] + p.offset + (iσ - 1) * p.dq + s
            iA1 = offA1[p.q] + s; iA2 = offA2[p.σ] + iσ
            ok &= DB[iB, ntuple(_ -> Colon(), N - 2)...] == D[iA1, iA2, ntuple(_ -> Colon(), N - 2)...]
        end
        @test ok
    end
end

@testset "factorizations reconstruct the tensor: $gname" for (gname, G, gen) in GROUPS
    rng = MersenneTwister(sum(codeunits(gname)) + 2)
    S = labeltype(G)
    for T in (Float64, ComplexF64)
        # eigen, rank 2
        l = randleg(rng, G, gen; dmax = 4)
        X = randtensor(rng, T, (l, dual(l)), neutral(S))
        H = X + X'
        vals, U = eigen(H)
        Λ = BlockTensor{Float64}((dual(leg(U, 2)), leg(U, 2)), neutral(S))
        for (s, v) in vals
            isempty(v) || setblock!(Λ, (s, s), Matrix(Diagonal(v)))
        end
        @test relerr(todense(U * Λ * U'), todense(H)) ≤ TOL
        @test sort(reduce(vcat, values(vals))) ≈ eigvals(Hermitian(todense(H))) atol = 1e-12
        # eigen on a leg partition (rank 4 hermitian map (a,b) → (a,b))
        a = randleg(rng, G, gen; dmax = 2); b = randleg(rng, G, gen; dmax = 2)
        Y = randtensor(rng, T, (a, b, dual(a), dual(b)), neutral(S))
        H4 = Y + adjoint(Y, (1, 2), (3, 4))
        v4, U4 = eigen(H4, (1, 2), (3, 4))
        M4 = reshape(todense(H4), TNOps.nblockrows(a) * TNOps.nblockrows(b), :)
        @test sort(reduce(vcat, values(v4))) ≈ eigvals(Hermitian(M4)) atol = 1e-11
        @test ndims(U4) == 3 && leg(U4, 1) == a && leg(U4, 2) == b
        # svd and qr, rank 2 with a flux, and on a partition
        r = randleg(rng, G, gen; dmax = 3); c = randleg(rng, G, gen; dmax = 3, arrow = In)
        A = randtensor(rng, T, (r, c), randflux(rng, (r, c)))
        Uu, s, V = svd(A)
        Σ = BlockTensor{Float64}((dual(leg(Uu, 2)), leg(Uu, 2)), neutral(S))
        for (k, v) in s
            isempty(v) || setblock!(Σ, (k, k), Matrix(Diagonal(v)))
        end
        @test relerr(todense(Uu * Σ * V'), todense(A)) ≤ TOL
        Q, R = qr(A)
        @test relerr(todense(Q * R), todense(A)) ≤ TOL
        @test norm(todense(Q' * Q) - I) ≤ 1e-12
        B3 = randtensor(rng, T, (r, a, c), randflux(rng, (r, a, c)))
        U3, s3, V3 = svd(B3, (1, 2), (3,))
        Σ3 = BlockTensor{Float64}((dual(leg(U3, 3)), leg(U3, 3)), neutral(S))
        for (k, v) in s3
            isempty(v) || setblock!(Σ3, (k, k), Matrix(Diagonal(v)))
        end
        @test relerr(todense(contract(contract(U3, (3,), Σ3, (1,)), (3,), V3', (1,))), todense(B3)) ≤ TOL
        Q3, R3 = qr(B3, (1, 2), (3,))
        @test relerr(todense(contract(Q3, (3,), R3, (1,))), todense(B3)) ≤ TOL
    end
end

@testset "apply_parity and parity_tensor against the dense parity matrix: $gname" for (gname, G, gen) in GROUPS
    rng = MersenneTwister(sum(codeunits(gname)) + 3)
    l = randleg(rng, G, gen; dmax = 3); m = randleg(rng, G, gen; dmax = 2)
    A = randtensor(rng, ComplexF64, (l, m), randflux(rng, (l, m)))
    Z = todense(parity_tensor(l))
    @test Z == Diagonal([isodd(parity(s.label)) ? -1.0 : 1.0 for (s, d) in zip(legsectors(l), l.dims) for _ in 1:d])
    @test todense(apply_parity(A, 1)) == Z * todense(A)
    @test todense(apply_parity(A, 1; power = 2)) == todense(A)
    @test todense(apply_parity(A, 2)) == todense(A) * todense(parity_tensor(m))
end

@testset "ncon against explicit dense loops: $gname" for (gname, G, gen) in GROUPS
    rng = MersenneTwister(sum(codeunits(gname)) + 4)
    for T in (Float64, ComplexF64)
        a = randleg(rng, G, gen; dmax = 2); b = randleg(rng, G, gen; dmax = 2)
        c = randleg(rng, G, gen; dmax = 2); d = randleg(rng, G, gen; dmax = 2)
        X = randtensor(rng, T, (a, b, c), randflux(rng, (a, b, c)))
        Y = randtensor(rng, T, (dual(b), d), randflux(rng, (dual(b), d)))
        W = randtensor(rng, T, (dual(c), dual(d)), randflux(rng, (dual(c), dual(d))))
        R = ncon([X, Y, W], [[-1, 1, 2], [1, 3], [2, 3]]; order = [3, 1, 2])
        Dx, Dy, Dw = todense(X), todense(Y), todense(W)
        ref = zeros(T, size(Dx, 1))
        for i in axes(Dx, 1), j in axes(Dx, 2), k in axes(Dx, 3), m in axes(Dy, 2)
            ref[i] += Dx[i, j, k] * Dy[j, m] * Dw[k, m]
        end
        @test norm(todense(R) - ref) ≤ TOL * max(1, norm(ref))
        # open legs permuted: X Y with result legs (d, a, c)
        R2 = ncon([X, Y], [[-2, 1, -3], [1, -1]])
        ref2 = zeros(T, size(Dy, 2), size(Dx, 1), size(Dx, 3))
        for i in axes(Dx, 1), j in axes(Dx, 2), k in axes(Dx, 3), m in axes(Dy, 2)
            ref2[m, i, k] += Dx[i, j, k] * Dy[j, m]
        end
        @test norm(todense(R2) - ref2) ≤ TOL * max(1, norm(ref2))
        # a trace inside the network
        Z = randtensor(rng, T, (a, b, dual(a)), randflux(rng, (a, b, dual(a))))
        R3 = ncon([Z], [[1, -1, 1]])
        Dz = todense(Z)
        @test norm(todense(R3) - [sum(Dz[i, j, i] for i in axes(Dz, 1)) for j in axes(Dz, 2)]) ≤ TOL * max(1, norm(Dz))
    end
end
