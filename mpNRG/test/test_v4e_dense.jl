# V4e: an independent dense brute-force evaluation of Lee's 2p/3p/4p formulas
# with truncation. Everything is done with dense matrices on the full chain
# and explicit Anders-Schiller vectors |se⟩ = |e⟩ ⊗ |s⟩^n: no BlockOp, no
# fusion map, no refine. Operators are carried from shell to shell as full-chain
# matrices and re-expressed in the AS basis of each shell, composites are built
# in that basis (internal D states sliced, Lee Eq. (60)) and mapped back to the
# full chain. The recursion follows the same reading of Lee Secs. V A-D as the
# engine, so this test checks the engine's implementation (block bookkeeping,
# fusion signs, refinement, sector chains, slice keys), not the reading itself;
# positions of refined terms are compared bin by bin.

using Test, MultipointNRG, LinearAlgebra

# AS basis of shell n on the full chain: columns kron(e_env, U_n[:, s]) for all
# states s of shell n and environment basis states e, with their shell energies
# and sectors.
function dense_as(run, n, Df)
    N = run.params.N
    sh = shell(run, n)
    emb = dense_embedding(run, n)
    dn = length(emb.basis)
    de = Df ÷ dn
    cols = Vector{Float64}[]; E = Float64[]; X = Symbol[]
    for q in sectors(sh.space)
        U = emb.cols[q]
        for s in 1:dim(sh.space, q), e in 1:de
            v = zeros(Df)
            v[(e - 1) * dn .+ (1:dn)] .= U[:, s]
            push!(cols, v); push!(E, sh.E[q][s] + sh.shift); push!(X, s ≤ nkept(sh, q) ? :K : :D)
        end
    end
    return reduce(hcat, cols), E, X
end

struct DOp
    tags::Vector{Int8}                    # per original frequency: 0 none, 1 direct, 2 left
    slices::Dict{Vector{Int},Matrix{Float64}}   # full-chain matrices
end

mutable struct DCtx
    run; ρ; g; L::Int; Df::Int
    hist::Dict{Vector{Int},Float64}
    AS::Dict{Int,Any}
end

key(ctx, x) = binindex(ctx.g, x)[1]
cen(ctx, k) = centres(ctx.g)[k]

function dense_psf!(ctx, ops::Vector{DOp}, orig, nfrom)
    N = ctx.run.params.N
    for n in nfrom:N
        W, E, X = ctx.AS[n]
        Ms = [Dict(k => W' * M * W for (k, M) in op.slices) for op in ops]
        R = W' * ctx.ρ * W
        m = length(ops)
        for Xs in Iterators.product(ntuple(_ -> (:K, :D), m)...)
            all(==(:K), Xs) && continue
            if Xs[1] === :D || count(==(:K), Xs) == 1
                dense_trace!(ctx, ops, Ms, R, E, X, Xs, orig)
            elseif n < N
                child = dense_composites(ctx, ops, Ms, W, E, X, Xs, orig)
                child === nothing || dense_psf!(ctx, child..., n + 1)
            end
        end
    end
end

idx(X, x) = findall(==(x), X)

function dense_trace!(ctx, ops, Ms, R, E, X, Xs, orig)
    m = length(ops); L = ctx.L
    I = [idx(X, Xs[i]) for i in 1:m]
    for (k1, M1) in Ms[1]
        B = R[I[1], I[1]] * M1[I[1], I[2]]
        F = fill(0, L)
        for g in 1:L
            ops[1].tags[g] != 0 && (F[g] = k1[g])
        end
        for (a, c1) in enumerate(I[1])
            dtrace_step!(ctx, ops, Ms, E, I, orig, 2, B[a, :], F, E[c1], a)
        end
    end
end

function dtrace_step!(ctx, ops, Ms, E, I, orig, i, v, F, E1, a1)
    m = length(ops)
    nxt = i == m ? 1 : i + 1
    for (k, M) in Ms[i]
        B = M[I[i], I[nxt]]
        Fk = copy(F)
        for g in 1:ctx.L
            ops[i].tags[g] == 1 && (Fk[g] = k[g])
        end
        groups = Dict{Vector{Int},Vector{Int}}()
        for (s, c) in enumerate(I[i])
            iszero(v[s]) && continue
            δ = E[c] - E1
            F2 = copy(Fk); F2[orig[i] - 1] = key(ctx, δ)
            for g in 1:ctx.L
                ops[i].tags[g] == 2 && (F2[g] = key(ctx, cen(ctx, k[g]) + δ))
            end
            push!(get!(() -> Int[], groups, F2), s)
        end
        for (F2, ss) in groups
            if i == m
                w = sum(v[s] * B[s, a1] for s in ss)
                ctx.hist[F2] = get(ctx.hist, F2, 0.0) + w
            else
                w = vec(v[ss]' * B[ss, :])
                dtrace_step!(ctx, ops, Ms, E, I, orig, i + 1, w, F2, E1, a1)
            end
        end
    end
end

# composites of the K-separated segments, as full-chain matrices
function dense_composites(ctx, ops, Ms, W, E, X, Xs, orig)
    m = length(ops); L = ctx.L
    J = findall(==(:K), collect(Xs))
    newops = DOp[]
    for t in eachindex(J)
        a = J[t]; b = t < length(J) ? J[t + 1] : m + 1
        first_seg = t == 1; last_seg = t == length(J)
        sl = Dict{Vector{Int},Matrix{Float64}}()          # AS-basis composite per key
        Iseg = [idx(X, (j == m + 1 ? :K : Xs[j])) for j in a:b]
        tags = zeros(Int8, L)
        for i in a:(b - 1), g in 1:L
            ops[i].tags[g] == 0 && continue
            tg = ops[i].tags[g]
            if tg == 2
                tg = i == a ? (first_seg ? Int8(1) : Int8(2)) : (last_seg || first_seg ? Int8(1) : Int8(2))
            end
            tags[g] = tg
        end
        for i in (a + 1):(b - 1)
            tags[orig[i] - 1] = (first_seg || last_seg) ? 1 : 2
        end
        add!(k, r, c, val) = (Mk = get!(() -> zeros(size(W, 2), size(W, 2)), sl, k); Mk[r, c] += val)
        if last_seg && !first_seg
            # right processing: keys relative to the right endpoint (state 1)
            for (kL, ML) in Ms[m], (c1i, c1) in enumerate(Iseg[end])
                w = ML[Iseg[end - 1], c1]
                F = fill(0, L)
                for g in 1:L
                    ops[m].tags[g] == 1 && (F[g] = kL[g])
                end
                if a == m
                    for g in 1:L
                        ops[m].tags[g] == 2 && (F[g] = kL[g])
                    end
                    for (s, c) in enumerate(Iseg[1]); add!(F, c, c1, w[s]); end
                else
                    dright!(ctx, ops, Ms, E, Iseg, orig, a, m, w, F, kL, E[c1], c1, add!)
                end
            end
        else
            # left processing: keys relative to the left endpoint
            for (ka, Ma) in Ms[a], (cai, ca) in enumerate(Iseg[1])
                v = Ma[ca, Iseg[2]]
                F = fill(0, L)
                for g in 1:L
                    ops[a].tags[g] != 0 && (F[g] = ka[g])
                end
                if b == a + 1
                    for (s, c) in enumerate(Iseg[2]); add!(F, ca, c, v[s]); end
                else
                    dleft!(ctx, ops, Ms, E, Iseg, orig, a, b, 2, v, F, E[ca], ca, add!)
                end
            end
        end
        isempty(sl) && return nothing
        push!(newops, DOp(tags, Dict(k => W * M * W' for (k, M) in sl)))
    end
    return newops, orig[J]
end

function dleft!(ctx, ops, Ms, E, Iseg, orig, a, b, j, v, F, Eref, ca, add!)
    i = a + j - 1
    for (k, M) in Ms[i]
        B = M[Iseg[j], Iseg[j + 1]]
        Fk = copy(F)
        for g in 1:ctx.L
            ops[i].tags[g] == 1 && (Fk[g] = k[g])
        end
        groups = Dict{Vector{Int},Vector{Int}}()
        for (s, c) in enumerate(Iseg[j])
            iszero(v[s]) && continue
            δ = E[c] - Eref
            F2 = copy(Fk); F2[orig[i] - 1] = key(ctx, δ)
            for g in 1:ctx.L
                ops[i].tags[g] == 2 && (F2[g] = key(ctx, cen(ctx, k[g]) + δ))
            end
            push!(get!(() -> Int[], groups, F2), s)
        end
        for (F2, ss) in groups
            w = vec(v[ss]' * B[ss, :])
            if i + 1 == b
                for (s, c) in enumerate(Iseg[j + 1]); add!(F2, ca, c, w[s]); end
            else
                dleft!(ctx, ops, Ms, E, Iseg, orig, a, b, j + 1, w, F2, Eref, ca, add!)
            end
        end
    end
end

function dright!(ctx, ops, Ms, E, Iseg, orig, a, i, w, F, kprev, Eref, c1, add!)
    j = i - a + 1
    groups = Dict{Vector{Int},Vector{Int}}()
    for (s, c) in enumerate(Iseg[j])
        iszero(w[s]) && continue
        δ = E[c] - Eref
        F2 = copy(F); F2[orig[i] - 1] = key(ctx, δ)
        for g in 1:ctx.L
            ops[i].tags[g] == 2 && (F2[g] = key(ctx, cen(ctx, kprev[g]) + δ))
        end
        push!(get!(() -> Int[], groups, F2), s)
    end
    for (F2, ss) in groups, (k, M) in Ms[i - 1]
        B = M[Iseg[j - 1], Iseg[j]]
        Fk = copy(F2)
        for g in 1:ctx.L
            ops[i - 1].tags[g] == 1 && (Fk[g] = k[g])
        end
        w2 = B[:, ss] * w[ss]
        if i - 1 == a
            for g in 1:ctx.L
                ops[a].tags[g] == 2 && (Fk[g] = k[g])
            end
            for (s, c) in enumerate(Iseg[1]); add!(Fk, c, c1, w2[s]); end
        else
            dright!(ctx, ops, Ms, E, Iseg, orig, a, i - 1, w2, Fk, k, Eref, c1, add!)
        end
    end
end

function dense_reference_psf(run, f, exprs, g)
    N = run.params.N
    m = run.model
    ms = chain_modes(n -> site_modes(m, n), N)
    B = FockBasis(ms)
    Df = length(B)
    # dense FDM density matrix in the AS basis
    ρ = zeros(Df, Df)
    AS = Dict{Int,Any}()
    for n in run.n0:N
        AS[n] = dense_as(run, n, Df)
        W, E, X = AS[n]
        for (c, x) in enumerate(X)
            x === :D || continue
            ρ .+= exp(-f.β * (E[c] - f.Eref) - f.logZ) .* (W[:, c] * W[:, c]')
        end
    end
    L = length(exprs) - 1
    ctx = DCtx(run, ρ, g, L, Df, Dict{Vector{Int},Float64}(), AS)
    ops = [DOp(zeros(Int8, L), Dict(fill(0, L) => densematrix(e, B))) for e in exprs]
    nstart = max(run.n0, maximum(max(maxsite(e), -1) for e in exprs))
    dense_psf!(ctx, ops, collect(1:length(exprs)), nstart)
    return ctx.hist, tr(ρ)
end

@testset "V4e dense brute force of Lee's recursion with truncation" begin
    # spinless resonant level: 2 states per site, so long chains stay small
    m = resonant_level(εd = 0.03, Δ = 0.08)
    d = cop(Mode(-1, :d))
    g = LogGrid(emin = 1e-6, emax = 10.0, ndec = 32)
    for (N, Nk, T, tuples) in [(6, 8, 0.02, [(d, d'), (d', d), (d, d', d' * d), (d' * d, d, d'), (d, d' * d, d')]),
                               (4, 6, 0.05, [(d, d', d, d'), (d, d, d', d'), (d' * d, d, d', d' * d)])]
        run = run_nrg(m, NRGParams(N = N, Nkeep = Nk))
        @test run.n0 < N - 1                                   # several truncated shells
        f = fdm(run; T = T)
        for ops in tuples
            href, trρ = dense_reference_psf(run, f, ops, g)
            @test trρ ≈ 1 atol = 1e-13
            for slicing in (:lee, :left)
                slicing === :left && length(ops) < 3 && continue
                r = psf(run, f, ops; grid = g, slicing = slicing, sparse = true)
                if slicing === :lee
                    hd = Dict(collect(k) => v for (k, v) in r.data)
                    dmax = maximum(abs(get(hd, k, 0.0) - get(href, k, 0.0)) for k in union(keys(hd), keys(href)))
                    @test dmax ≤ 1e-13
                    @test r.meta[:cost][:traces] > 0
                    length(ops) ≥ 3 && @test r.meta[:cost][:composites] > 0
                end
            end
        end
    end
end
