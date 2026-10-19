# The NRG tensor layer nrg/tensors.jl (TNOps refactor, Sec. 4.3): every named
# function against the pre-refactor implementation through the fixtures
# frozen at the tag pre-tensor-refactor (test/baseline/baseline.jld2), and a
# shell construction built from these functions alone against a dense
# Kronecker / Jordan-Wigner reference (helpers.jl: jw_annihilators).

using Test, MultipointNRG, TNOps, LinearAlgebra, JLD2
import MultipointNRG as M
import MultipointNRG.TNOps as TN

const FX = load(joinpath(@__DIR__, "baseline", "baseline.jld2"), "fixtures")
const MOD = FX["trunc/moduli"]
const SL = Sector{MOD,length(MOD)}
lab(t) = SL(Tuple(t))

shellleg(n) = M.shell_leg(Dict(lab(t) => d for (t, d) in FX["space/$n"]), Dict(lab(t) => k for (t, k) in FX["nkept/$n"]))
fxfusion(n) = FX["fusion/$n"]
siteleg(n) = TN.withname(Leg(Dict(LegSector(lab(t)) => d for (t, d) in fxfusion(n)["loc"])), :site)
# the operator of a fixture on given tagged legs (legacy blocks: full label blocks, kept first)
fxop(d, l) = M.shell_tensor(l, lab(d["shift"]), Dict(lab(t) => B for (t, B) in d["blocks"]), Float64)
relerr(a, b) = norm(a - b) / max(norm(b), 1e-300)

# A^n as a rank-3 tensor from the fixture of the old Shell.A
function fxisometry(n)
    Fu = M.fuse_site(siteleg(n), M.kept_leg(shellleg(n - 1)))
    U = BlockTensor{Float64}(Fu.out, shellleg(n), neutral(SL))
    for (t, B) in FX["A/$n"]
        setblock!(U, lab(t), B)
    end
    return TN.split_legs(U, 1, Fu), Fu
end

@testset "nrg/tensors.jl against the frozen pre-refactor implementation" begin
    n0 = FX["trunc/n0"]
    for n in (n0 + 1, n0 + 3)
        A, Fu = fxisometry(n)
        Adag = M.isometry_adjoint(A)
        # fuse_site = the old fusionmap (labels, order, offsets)
        for (t, prs) in fxfusion(n)["pairs"]
            @test [(p.σ.label.q, p.q.label.q, p.offset, p.dσ, p.dq) for p in Fu.pairs[LegSector(lab(t))]] == prs
        end
        @test all(p.q.tag == M.KTAG for v in values(Fu.pairs) for p in v)       # inner leg = kept states
        # the isometry reproduces the old matrices
        @test all(M.isometry_block(A, Fu, lab(t)) == B for (t, B) in FX["A/$n"])
        # refine (odd and even operators)
        for labop in ("d_up", "n_dn", "ddag_dn")
            fx = FX["refine/$n/$labop"]
            O = fxop(fx["in"], shellleg(n - 1))
            R = M.refine_KK(A, Adag, O)
            Rold = fxop(fx["out"], shellleg(n))
            @test relerr(todense(R), todense(Rold)) ≤ 1e-13
            # the RefinePlan path of the PSF engine: same tensor (legs, blocks) to rounding
            Rp = M.refine_KK(M.RefinePlan(A, Fu), O)
            @test TN.legs(Rp) == TN.legs(R) && TN.flux(Rp) == TN.flux(R)
            @test relerr(todense(Rp), todense(R)) ≤ 1e-14 && relerr(todense(Rp), todense(Rold)) ≤ 1e-13
            Ok = M.kk_tensor(M.kept_leg(shellleg(n - 1)), TN.flux(O), Dict(q => getblock(O, q)[1:M.dim(shellleg(n - 1), LegSector(target(O, q), M.KTAG)), 1:M.dim(shellleg(n - 1), LegSector(q, M.KTAG))]
                                                                   for q in sectors(shellleg(n - 1)) if haskey(shellleg(n - 1), target(O, q))), Float64)
            @test relerr(todense(M.refine_KK(M.RefinePlan(A, Fu), Ok)), todense(R)) ≤ 1e-14     # O on the kept leg
            # sliced: slice by slice
            st = SlicedTensor(Dict((1, 2) => O, (3, 4) => 2.0 * O))
            sr = M.refine_KK(A, Adag, st)
            @test relerr(todense(sr[(3, 4)]), 2 * todense(Rold)) ≤ 1e-13
        end
        # _rotate of f_n ⊗ 1 and lift_site of f_n
        fx = FX["rotate/$n"]
        P = BlockTensor{Float64}(Fu.out, Fu.out, lab(fx["P"]["shift"]))
        for (t, B) in fx["P"]["blocks"]
            setblock!(P, lab(t), B)
        end
        old = fxop(fx["out"], shellleg(n))
        @test relerr(todense(M.rotate(A, P, Fu)), todense(old)) ≤ 1e-13
        L = siteleg(n)
        fl = TN.fromdense(jw_annihilators(2)[1], (L, dual(L)), dual(lab((1, 1, 1))))   # f_{n↑}
        @test relerr(todense(M.lift_site(A, Adag, fl)), todense(old)) ≤ 1e-13
        # FDM backward step
        fx = FX["fdm/$n"]
        lK = M.kept_leg(shellleg(n))
        ρK = M.kk_tensor(lK, neutral(SL), Dict(lab(t) => B for (t, B) in fx["rhoK_n"]), Float64)
        ρ = M.rho_full(shellleg(n), ρK, Dict(lab(t) => v for (t, v) in fx["rhoD_n"]), Float64)
        ρprev = M.backward_step(A, Adag, ρ)
        lK1 = M.kept_leg(shellleg(n - 1))
        old = M.kk_tensor(lK1, neutral(SL), Dict(lab(t) => B for (t, B) in fx["rhoK_prev"]), Float64)
        @test TN.samespace(TN.leg(ρprev, 1), lK1)
        @test relerr(todense(ρprev), todense(old)) ≤ 1e-13
        # block_view: the tagged blocks of the full label block
        O = fxop(FX["refine/$n/d_up"]["out"], shellleg(n))
        for q in sectors(shellleg(n))
            r = target(O, q); haskey(shellleg(n), r) || continue
            full = getblock(O, q); nkr = M.dim(shellleg(n), LegSector(r, M.KTAG)); nkq = M.dim(shellleg(n), LegSector(q, M.KTAG))
            B = M.block_view(O, q, :K, :D)
            B === nothing || @test B == full[1:nkr, (nkq + 1):end]
            B = M.block_view(O, q, :D, :K)
            B === nothing || @test B == full[(nkr + 1):end, 1:nkq]
        end
    end
end

@testset "shell construction from nrg/tensors.jl vs dense Jordan-Wigner (3 sites, spinful, interacting)" begin
    # SIAM-like chain: impurity -1 with U, sites 0 and 1; mode order of the
    # dense reference [1↑, 1↓, 0↑, 0↓, d↑, d↓] (later sites' creators to the left)
    G = SymmetryGroup(U1(:N), U1(:twoSz)); S = labeltype(G)
    L = TN.withname(Leg(Dict(LegSector(sector(G; N = 0, twoSz = 0)) => 1, LegSector(sector(G; N = 1, twoSz = 1)) => 1,
                             LegSector(sector(G; N = 1, twoSz = -1)) => 1, LegSector(sector(G; N = 2, twoSz = 0)) => 1)), :site)
    c2 = jw_annihilators(2)        # local basis |00⟩, |0↓⟩, |↑0⟩, |↑↓⟩ = the sorted sectors of L
    f = Dict(:up => TN.fromdense(c2[1], (L, dual(L)), dual(sector(G; N = 1, twoSz = 1))),
             :dn => TN.fromdense(c2[2], (L, dual(L)), dual(sector(G; N = 1, twoSz = -1))))
    nσ(σ) = f[σ]' * f[σ]
    Z = TN.parity_tensor(L)
    @test todense(Z) == Diagonal([1.0, -1.0, -1.0, 1.0])
    εd, U, ε, t = -0.21, 0.53, [0.07, -0.04], [0.31, 0.17]
    keepall(E) = Dict(q => length(e) for (q, e) in E)
    # shell -1
    Himp = εd * (nσ(:up) + nσ(:dn)) + U * (nσ(:up) * nσ(:dn))
    E, nk, A = M.diagonalize_shell(Himp, nothing; cut = keepall)
    fsh = Dict(σ => M.rotate(A, f[σ], nothing) for σ in (:up, :dn))     # f_{-1σ} in the shell basis
    l = TN.leg(A, 2)
    for n in 0:1
        Fu = M.fuse_site(L, M.kept_leg(l))
        hops = [(t[n + 1], f[σ]' * Z, subtensor(fsh[σ], (s -> s.tag == M.KTAG, s -> s.tag == M.KTAG))) for σ in (:up, :dn)]
        H = M.product_hamiltonian(Fu, Dict(q => e[1:nk[q]] for (q, e) in E), ε[n + 1] * (nσ(:up) + nσ(:dn)), hops, Float64)
        E, nk, A = M.diagonalize_shell(H, Fu; cut = keepall)
        Adag = M.isometry_adjoint(A)
        prev = fsh
        fsh = Dict(σ => M.lift_site(A, Adag, f[σ]) for σ in (:up, :dn))           # f_{nσ}, rule (R2)
        ref = Dict(σ => M.refine_KK(A, Adag, prev[σ]) for σ in (:up, :dn))       # f_{n-1,σ}, rule (R1)
        Id = Matrix(1.0I, TN.nblockrows(TN.leg(A, 3)), TN.nblockrows(TN.leg(A, 3)))
        for σ in (:up, :dn), τ in (:up, :dn)
            # canonical anticommutators of the refined and the new-site modes: fail without the parity string
            @test norm(todense(fsh[σ] * ref[τ] + ref[τ] * fsh[σ])) ≤ 1e-12
            @test norm(todense(fsh[σ] * ref[τ]' + ref[τ]' * fsh[σ])) ≤ 1e-12
            @test norm(todense(ref[σ] * ref[τ]' + ref[τ]' * ref[σ]) - (σ == τ) * Id) ≤ 1e-12
        end
        l = TN.leg(A, 3)
        # backward step of the identity: the partial trace over the site gives d_site · 1
        dK(q) = M.dim(l, LegSector(q, M.KTAG))
        ρK = M.kk_tensor(M.kept_leg(l), neutral(S), Dict(q => Matrix(1.0I, dK(q), dK(q)) for q in sectors(l)), Float64)
        ρ = M.rho_full(l, ρK, Dict{S,Vector{Float64}}(), Float64)            # no truncation: no D states
        ρp = M.backward_step(A, Adag, ρ)
        @test norm(todense(ρp) - 4I) ≤ 1e-12
        @test abs(M.shell_trace(ρ, fsh[:up]' * fsh[:up]) - TN.nblockrows(l) / 2) ≤ 1e-12      # ⟨n⟩ = 1/2 at infinite T
    end
    # energies of the 3-site chain vs dense exact diagonalization
    c = jw_annihilators(6)
    nn(k) = c[k]' * c[k]
    Hd = εd * (nn(5) + nn(6)) + U * nn(5) * nn(6) + ε[1] * (nn(3) + nn(4)) + ε[2] * (nn(1) + nn(2))
    for s in 0:1
        Hd += t[1] * (c[3 + s]' * c[5 + s] + c[5 + s]' * c[3 + s]) + t[2] * (c[1 + s]' * c[3 + s] + c[3 + s]' * c[1 + s])
    end
    @test norm(sort(reduce(vcat, values(E))) - eigvals(Hermitian(Hd))) ≤ 1e-12
end

@testset "no block dictionaries outside nrg/tensors.jl and the declared kernels (Sec. 4.1, 4.4)" begin
    src = normpath(joinpath(@__DIR__, "..", "src"))
    # the declared dense kernels of psf/engine.jl (listed at the top of that file)
    kernels = ("shell_trace!", "_trace_step!", "_composite_left", "_comp_left_step!", "_addrow!",
               "_composite_right", "_comp_right_step!", "_addcol!")
    hits = String[]
    nfiles = 0
    for (dir, _, files) in walkdir(src), f in files
        endswith(f, ".jl") || continue
        path = joinpath(dir, f); rel = relpath(path, src)
        nfiles += 1
        rel == joinpath("nrg", "tensors.jl") && continue
        fn = ""                                   # the top-level function the line belongs to
        for (i, line) in enumerate(eachline(path))
            m = match(r"^function\s+([A-Za-z_][A-Za-z_0-9!]*)", line)
            m === nothing || (fn = m.captures[1])
            occursin(".blocks", line) && !(rel == joinpath("psf", "engine.jl") && fn in kernels) && push!(hits, "$rel:$i")
            line == "end" && (fn = "")
        end
    end
    @test nfiles ≥ 15                         # the walk did see the package
    @test isempty(hits)
    isempty(hits) || @info "block dictionaries accessed at" hits
end
