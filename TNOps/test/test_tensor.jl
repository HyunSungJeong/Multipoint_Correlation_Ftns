# BlockTensor, SlicedTensor and their linear algebra against dense arrays
# (refactor sheet Sec. 3.7): ranks 1-5, 2-5 sectors per leg, random fluxes,
# symmetries U1×U1×Z2, Z3 (×Pf) and parity only, real and complex elements.

using Test, TNOps, LinearAlgebra, Random
import TNOps: legsectors, allowed_keys, keyflux, blockkeys, eachblock, nblocks, isdual, charge

const TOL = 1e-13

# the three symmetry choices with label generators
const GROUPS = [
    ("U1xU1xZ2", SymmetryGroup(U1(:N), U1(:twoSz)), rng -> (n = rand(rng, -2:2); (n, rand(rng, -2:2), mod(n, 2)))),
    ("Z3", SymmetryGroup(Zn(:C, 3)), rng -> (rand(rng, 0:2), rand(rng, 0:1))),
    ("parity", SymmetryGroup(), rng -> (rand(rng, 0:1),)),
]

# a random leg with 2-5 sectors (distinct (label, tag)), dims 1..dmax
function randleg(rng, G, gen; nsec = rand(rng, 2:5), dmax = 3, arrow = rand(rng, (In, Out)), tagged = rand(rng) < 0.3)
    secs = Set{Any}()
    tries = 0
    while length(secs) < nsec && tries < 200
        tries += 1
        push!(secs, LegSector(sector(G, gen(rng)), tagged ? rand(rng, 0:2) : 0))
    end
    secs = sort!(collect(secs))         # independent of hash order
    return Leg(LegSector{labeltype(G)}[s for s in secs], [rand(rng, 1:dmax) for _ in secs]; arrow = arrow)
end

# a random flux that admits at least one block
function randflux(rng, ls)
    S = typeof(legsectors(ls[1])[1].label)
    k = ntuple(i -> rand(rng, legsectors(ls[i])), length(ls))
    return keyflux(ls, k)
end

relerr(a, b) = norm(a - b) / max(norm(b), 1e-300)

@testset "legs: sorting, duality, charges" begin
    rng = MersenneTwister(11)
    G = SymmetryGroup(U1(:N), U1(:twoSz)); S = labeltype(G)
    l = Leg(Dict(LegSector(sector(G, (1, 1, 1)), 2) => 3, LegSector(sector(G, (1, 1, 1)), 1) => 2,
                 LegSector(sector(G, (0, 0, 0))) => 1, LegSector(sector(G, (2, 0, 0))) => 0))
    @test legsectors(l) == [LegSector(sector(G, (0, 0, 0))), LegSector(sector(G, (1, 1, 1)), 1), LegSector(sector(G, (1, 1, 1)), 2)]
    @test dim(l, sector(G, (1, 1, 1))) == 5 && dim(l) == 6 && !haskey(l, sector(G, (2, 0, 0)))
    @test TNOps.tags(l, sector(G, (1, 1, 1))) == [1, 2]
    dl = dual(l)
    @test arrow(dl) == In && dual(dl) == l && isdual(l, dl) && !isdual(l, l)
    s = legsectors(l)[2]
    @test charge(dl, s) == dual(charge(l, s))
    @test_throws ArgumentError Leg([s, s], [1, 2])
end

@testset "BlockTensor vs dense: $gname" for (gname, G, gen) in GROUPS
    rng = MersenneTwister(sum(codeunits(gname)))
    for N in 1:5, T in (Float64, ComplexF64)
        dmax = N ≤ 3 ? 3 : 2
        ls = ntuple(_ -> randleg(rng, G, gen; dmax = dmax), N)
        Δ = randflux(rng, ls)
        A = randtensor(rng, T, ls, Δ; density = 0.8)
        # forbidden blocks are never allocated
        @test all(k -> keyflux(ls, k) == Δ, blockkeys(A))
        @test Set(blockkeys(A)) ⊆ Set(allowed_keys(ls, Δ))
        D = todense(A)
        @test fromdense(D, ls, Δ) ≈ A atol = 0
        @test norm(todense(fromdense(D, ls, Δ)) - D) == 0
        # arithmetic
        B = randtensor(rng, T, ls, Δ; density = 0.8)
        @test relerr(todense(A + B), D + todense(B)) ≤ TOL
        @test relerr(todense(A - 2.5 * B), D - 2.5 * todense(B)) ≤ TOL
        @test relerr(todense(conj(A)), conj(D)) ≤ TOL
        @test abs(norm(A) - norm(D)) ≤ TOL * norm(D)
        @test abs(dot(A, B) - dot(D, todense(B))) ≤ TOL * max(1, norm(D) * norm(todense(B)))
        @test isapprox(A, A + 1e-15 * B; atol = 1e-13)
        # permute
        p = Tuple(randperm(rng, N))
        @test norm(todense(permute(A, p)) - permutedims(D, p)) == 0
        @test permute(permute(A, p), Tuple(invperm(collect(p)))) ≈ A atol = 0
        # weight outside the allowed blocks is refused
        if length(allowed_keys(ls, Δ)) < prod(length.(legsectors.(ls)))
            Dbad = copy(D)
            bad = first(k for k in Iterators.product(legsectors.(ls)...) if keyflux(ls, k) != Δ)
            offs = map(TNOps.offsets, ls)
            Dbad[ntuple(i -> offs[i][bad[i]] + 1, N)...] = 1.0
            @test_throws ArgumentError fromdense(Dbad, ls, Δ)
        end
    end
end

# explicit dense contraction by loops
function dense_contract(DA, ia, DB, ib)
    NA = ndims(DA); NB = ndims(DB)
    oa = [i for i in 1:NA if !(i in ia)]; ob = [i for i in 1:NB if !(i in ib)]
    T = promote_type(eltype(DA), eltype(DB))
    C = zeros(T, (size(DA)[oa]..., size(DB)[ob]...))
    for Ia in CartesianIndices(DA), Jb in CartesianIndices(size(DB)[ob])
        # B index: contracted part from Ia, open part from Jb
        bi = zeros(Int, NB)
        for (x, y) in zip(ia, ib)
            bi[y] = Ia[x]
        end
        for (m, y) in enumerate(ob)
            bi[y] = Jb[m]
        end
        C[Tuple(Ia)[oa]..., Tuple(Jb)...] += DA[Ia] * DB[bi...]
    end
    return C
end

@testset "contract, outer, tr vs explicit loops: $gname" for (gname, G, gen) in GROUPS
    rng = MersenneTwister(7 + sum(codeunits(gname)))
    for trial in 1:12, T in (Float64, ComplexF64)
        NA = rand(rng, 1:4); NB = rand(rng, 1:4)
        nc = rand(rng, 0:min(NA, NB, 2))
        lsA = ntuple(_ -> randleg(rng, G, gen; dmax = 2), NA)
        ia = Tuple(randperm(rng, NA)[1:nc]); ib = Tuple(randperm(rng, NB)[1:nc])
        lsB = Vector{Any}(undef, NB)
        for (x, y) in zip(ia, ib)
            lsB[y] = dual(lsA[x])
        end
        for y in 1:NB
            isassigned(lsB, y) || (lsB[y] = randleg(rng, G, gen; dmax = 2))
        end
        lsB = Tuple(lsB)
        A = randtensor(rng, T, lsA, randflux(rng, lsA); density = 0.9)
        B = randtensor(rng, Float64, lsB, randflux(rng, lsB); density = 0.9)
        C = contract(A, ia, B, ib)
        Cd = dense_contract(todense(A), ia, todense(B), ib)
        @test size(todense(C)) == size(Cd)
        @test norm(todense(C) - Cd) ≤ TOL * max(1, norm(Cd))
        @test all(k -> TNOps.isallowed(C, k), blockkeys(C))                          # no forbidden block
        @test TNOps.flux(C) == fuse(TNOps.flux(A), TNOps.flux(B))
        nc == 0 && @test norm(todense(outer(A, B)) - Cd) ≤ TOL * max(1, norm(Cd))
    end
    # mismatched legs: duality errors name the legs
    l1 = randleg(rng, G, gen); l2 = randleg(rng, G, gen)
    A = randtensor(rng, Float64, (l1, l2), randflux(rng, (l1, l2)))
    @test_throws ArgumentError contract(A, (1,), A, (1,))            # same arrow: not dual
    err = try contract(A, (2,), A, (2,)) catch e; e end
    @test err isa ArgumentError && occursin("leg 2", err.msg) && occursin("not dual", err.msg)
    # traces: rank 2 and partial
    l = randleg(rng, G, gen; dmax = 3, arrow = Out)
    H = randtensor(rng, ComplexF64, (l, dual(l)), neutral(labeltype(G)))
    @test abs(tr(H) - tr(todense(H))) ≤ TOL * max(1, norm(todense(H)))
    q = first(x for x in (sector(G, gen(rng)) for _ in 1:50) if !isneutral(x))
    if any(k -> keyflux((l, dual(l)), k) == q, Iterators.product(legsectors(l), legsectors(l)))
        X = randtensor(rng, Float64, (l, dual(l)), q)
        @test tr(X) === 0.0                                          # exact zero, no block touched
    end
    m = randleg(rng, G, gen; dmax = 2)
    R = randtensor(rng, Float64, (l, m, dual(l)), randflux(rng, (l, m, dual(l))))
    Rt = tr(R, 1, 3)
    Dr = todense(R)
    @test norm(todense(Rt) - [sum(Dr[a, j, a] for a in axes(Dr, 1)) for j in axes(Dr, 2)]) ≤ TOL * max(1, norm(Dr))
end

@testset "rank-2 operator layout (BlockOp semantics)" begin
    rng = MersenneTwister(3)
    G = SymmetryGroup(U1(:N), U1(:twoSz)); S = labeltype(G)
    V = Leg(Dict(sector(G, (n, s, mod(n, 2))) => rand(rng, 1:3) for n in -1:2 for s in -2:2 if iseven(n + s)))
    Δ = sector(G; N = 1, twoSz = 1)
    A = BlockTensor{Float64}(V, V, Δ)
    for q in sectors(V)
        r = fuse(q, Δ)
        haskey(V, r) || continue
        @test target(A, q) == r
        setblock!(A, q, randn(rng, dim(V, r), dim(V, q)))
    end
    D = todense(A)
    for q in sectors(V)
        haskey(V, target(A, q)) || continue
        o = TNOps.offsets(V)
        @test getblock(A, q) == D[o[LegSector(target(A, q))] .+ (1:dim(V, target(A, q))), o[LegSector(q)] .+ (1:dim(V, q))]
    end
    @test A.row == TNOps.leg(A, 1) && A.shift == Δ
    B = BlockTensor{Float64}(V, V, dual(Δ))
    for q in sectors(V)
        haskey(V, target(B, q)) && setblock!(B, q, randn(rng, dim(V, target(B, q)), dim(V, q)))
    end
    @test norm(todense(A * B) - D * todense(B)) ≤ TOL * norm(D) * norm(todense(B))
    @test norm(todense(A') - D') == 0
    Id = identity_tensor(V)
    @test todense(Id) == Matrix(1.0I, dim(V), dim(V))
    # tagged legs: getblock(A, q) assembles the tags, K-like tag 1 first
    Vt = TNOps.retag(V, (s, d) -> d ≥ 2 ? [(1, 1), (2, d - 1)] : [(1, d)])
    At = BlockTensor{Float64}(Vt, Vt, Δ)
    for q in sectors(V)
        haskey(V, target(A, q)) && setblock!(At, q, getblock(A, q))
    end
    @test norm(todense(At) - D) == 0
    @test all(getblock(At, q) == getblock(A, q) for q in sectors(V))
    sub = TNOps.subtensor(At, (s -> s.tag == 1, s -> s.tag == 1))
    @test all(k -> k[1].tag == 1 && k[2].tag == 1, blockkeys(sub))
    # retag a tensor leg: blocks split along it, dense embedding unchanged
    Ar = retag(A, 2, (s, d) -> d ≥ 2 ? [(1, 1), (2, d - 1)] : [(1, d)])
    @test todense(Ar) == D && TNOps.leg(Ar, 2) == dual(Vt)
    @test all(k -> TNOps.isallowed(Ar, k), blockkeys(Ar))
end

@testset "SlicedTensor" begin
    rng = MersenneTwister(5)
    G = SymmetryGroup(U1(:N)); S = labeltype(G)
    V = Leg(Dict(sector(G, (n, mod(n, 2))) => 2 for n in 0:3))
    Δ = sector(G, (1, 1))
    st = SlicedTensor{Tuple{Int,Int},Float64}((V, dual(V)), Δ)
    for k in ((1, 2), (3, 4), (5, 5))
        setslice!(st, k, randtensor(rng, Float64, (V, dual(V)), Δ))
    end
    @test nslices(st) == 3 && haskey(st, (3, 4))
    W = randtensor(rng, Float64, (V, dual(V)), neutral(S))
    m = mapslices(A -> W * A, st)
    @test all(norm(todense(m[k]) - todense(W) * todense(st[k])) ≤ TOL * 10 for k in keys(st))
    ex = SlicedTensor{Float64,Float64}((V, dual(V)), Δ)          # exact-mode keys: energies
    setslice!(ex, 0.125, st[(1, 2)])
    @test nslices(ex) == 1 && ex[0.125] === st[(1, 2)]
    @test_throws ArgumentError setslice!(st, (9, 9), W)          # wrong flux
end
