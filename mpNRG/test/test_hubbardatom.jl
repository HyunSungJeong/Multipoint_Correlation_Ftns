# V3 references (b) and (c) from the repository's `HubbardAtom` package, and the
# export of PSFs to its kernels (task sheet Sec. 5.8). The converter lives here
# so that `src/` stays independent of HubbardAtom.

using Test, MultipointNRG, LinearAlgebra
import HubbardAtom as HA

"""
    to_psfterms(r) -> Vector{HA.PSFTerm}

Export an mpNRG PSF to HubbardAtom: exact mode term by term; binned mode one
term per nonzero bin at the bin centres.
"""
function to_psfterms(r)
    if r.mode === :exact
        return [HA.PSFTerm(r.positions[:, j], ComplexF64(r.weights[j]), Int[]) for j in eachindex(r.weights)]
    end
    c = centres(r.grid)
    return [HA.PSFTerm([c[i] for i in k], ComplexF64(v), Int[]) for (k, v) in nonzero_bins(r)]
end

"""    to_permuted(rs, is_fermionic) — a `psfs` dictionary as HubbardAtom's PermutedPSF cache."""
to_permuted(rs, isf) = [HA.PermutedPSF(p, HA.permutation_sign(p, isf), to_psfterms(r)) for (p, r) in rs]

ha_terms_matrix(ts) = (reduce(hcat, [t.position for t in ts]), [t.weight for t in ts])

@testset "V3b chain vs HubbardAtom.AIMModel on the star of the finite chain" begin
    U = 0.4; T = 0.05
    m = siam(U = U, Δ = 0.1)                     # εd = -U/2, as AIMModel requires
    N = 2
    run = run_nrg(m, NRGParams(N = N, Nkeep = typemax(Int)))
    f = fdm(run; T = T)
    E, V = chain_star(run.chains[1], N)          # star of the finite chain (3 bath sites)
    ha = HA.AIMModel(U, 1 / T, E, V)
    sp = HA.spectrum(ha)
    ops = HA.operators(ha)
    du, dn = ops.c[1], ops.c[2]
    for (mine, theirs) in [((d_(:up), dd_(:up)), (du, du')),
                           ((d_(:up), dd_(:up), n_(:dn)), (du, du', dn' * dn)),
                           ((d_(:up), dd_(:dn), d_(:dn), dd_(:up)), (du, dn', dn, du'))]
        r = psf(run, f, mine; exact = true, max_terms = 20_000_000)
        P, w = ha_terms_matrix(HA.psf_chain(sp, theirs; otol = 1e-13))
        @test test_functions_error(r.positions, r.weights, P, w) ≤ 1e-10
    end
end

@testset "V3c atom limit V → 0 vs HubbardAtom exact atom PSFs" begin
    # t_0 = 0 decouples the impurity. Without truncation the impurity PSFs are
    # those of the atom. (With truncation they are not exactly: the global cut
    # keeps different bath states for different impurity levels, E ≤ E_cut - E_imp,
    # so refined composites pick up bath-energy positions; see STATUS.md.)
    U = 0.6; T = 0.1
    m = siam(U = U, Δ = 0.1)
    N = 2
    chs = wilson_chains(m, N)
    chs0 = [MultipointNRG.WilsonChain(c.eps, vcat(0.0, c.t[2:end]), c.hyb, c.disc, c.nint, c.precision_check) for c in chs]
    run = run_nrg(m, NRGParams(N = N, Nkeep = typemax(Int)); chains = chs0)    # impurity decoupled
    f = fdm(run; T = T)
    atom = HA.HubbardAtomModel(U, 1 / T)
    sp = HA.spectrum(atom)
    o = HA.operators(atom)
    for (mine, theirs) in [((d_(:up), dd_(:up)), (o.d_up, o.dag_up)),
                           ((d_(:up), dd_(:up), n_(:dn)), (o.d_up, o.dag_up, o.n_dn)),
                           ((d_(:up), dd_(:dn), d_(:dn), dd_(:up)), (o.d_up, o.dag_dn, o.d_dn, o.dag_up)),
                           ((d_(:up), dd_(:up), d_(:up), dd_(:up)), (o.d_up, o.dag_up, o.d_up, o.dag_up))]
        r = psf(run, f, mine; exact = true, max_terms = 20_000_000)
        P, w = ha_terms_matrix(HA.psf(sp, theirs))
        @test test_functions_error(r.positions, r.weights, P, w) ≤ 1e-11
    end
end

@testset "export to HubbardAtom kernels: Matsubara correlators" begin
    U = 0.4; T = 0.05
    m = siam(U = U, Δ = 0.1)
    N = 1
    run = run_nrg(m, NRGParams(N = N, Nkeep = typemax(Int)))
    f = fdm(run; T = T)
    E, V = chain_star(run.chains[1], N)
    ha = HA.AIMModel(U, 1 / T, E, V)
    ops = HA.operators(ha)
    du = ops.c[1]
    isf = [true, true]
    rs = psfs(run, f, (d_(:up), dd_(:up)); exact = true)
    cache = to_permuted(rs, isf)
    for n in (0, 1, 5)
        ν = HA.fermionic_freq(n)
        G_mine = HA.correlator(ha, (du, du'), [ν, HA.MatsubaraFreq(-ν.m)]; cache = cache)
        G_ref = HA.correlator(ha, (du, du'), [ν, HA.MatsubaraFreq(-ν.m)])
        @test abs(G_mine - G_ref) ≤ 1e-11 * max(1, abs(G_ref))
    end
    # binned export: the kernel sees bin centres; agreement to the binning resolution
    g = LogGrid(emin = 1e-6, emax = 10.0, ndec = 32)
    rsb = psfs(run, f, (d_(:up), dd_(:up)); grid = g)
    cacheb = to_permuted(rsb, isf)
    ν = HA.fermionic_freq(0)
    Gb = HA.correlator(ha, (du, du'), [ν, HA.MatsubaraFreq(-ν.m)]; cache = cacheb)
    Gr = HA.correlator(ha, (du, du'), [ν, HA.MatsubaraFreq(-ν.m)])
    @test abs(Gb - Gr) ≤ 0.05 * abs(Gr)
end
