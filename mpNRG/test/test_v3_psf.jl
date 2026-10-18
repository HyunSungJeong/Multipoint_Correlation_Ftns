# V3 (PSFs, P6-P8): without truncation every state is discarded only at n = N,
# the Anders-Schiller basis is the exact eigenbasis, and the 2p, 3p and 4p PSFs
# must equal the exact Lehmann sums of the finite chain, for all permutations,
# both unbinned (exact = true) and binned (against identically binned ED data).

using Test, MultipointNRG, LinearAlgebra

function v3_compare(run, f, ed, ops, g; tol = 1e-10)
    r = psf(run, f, ops; exact = true, max_terms = 20_000_000)
    P, w = ed_psf(ed, ops)
    if r.meta[:pruned]                               # symmetry zero: ED must agree
        @test isempty(w) && iszero(r.total)
        return r
    end
    scale = max(maximum(abs, w; init = 0.0), 1e-300)
    @test test_functions_error(r.positions, r.weights, P, w) ≤ tol * max(1, sum(abs, w))
    @test abs(r.total - sum(w)) ≤ tol * max(1, sum(abs, w))
    rb = psf(run, f, ops; grid = g)
    hb = bin_terms(P, w, g)
    @test dict_maxdiff(hist_dict(rb), hb) ≤ tol * scale + 1e-14
    @test abs(rb.total - r.meta[:sum_rule]) ≤ 1e-12
    return r
end

@testset "V3c PSFs without truncation vs reference ED" begin
    m = siam(U = 0.5, εd = -0.1, Δ = 0.1)
    T = 0.05
    g = LogGrid(emin = 1e-5, emax = 10.0, ndec = 16)
    Hint = m.params[:U] * n_(:up) * n_(:dn)
    q(σ) = comm(d_(σ), Hint)
    bos = acomm(q(:up), dd_(:dn))                    # {q↑, d†↓}, bosonic, 2S_z = -2
    x = dd_(:dn) * f0_(:dn)                          # impurity-site-0 operator d†↓ f_{0↓}
    @testset "N = 1 (64 states), all permutations" begin
        N = 1
        run = run_nrg(m, NRGParams(N = N, Nkeep = typemax(Int)))
        f = fdm(run; T = T)
        ed = model_ed(m, run.chains, N, T)
        for ops in [(d_(:up), dd_(:up)), (n_(:dn), n_(:dn)), (q(:up), dd_(:up)), (x, x'), (bos, bos')]
            for p in all_perms(2)
                v3_compare(run, f, ed, Tuple(ops[i] for i in p), g)
            end
        end
        for ops in [(d_(:up), dd_(:up), n_(:dn)), (q(:up), dd_(:dn), bos'), (d_(:dn), x, dd_(:dn))]
            for p in all_perms(3)
                v3_compare(run, f, ed, Tuple(ops[i] for i in p), g)
            end
        end
        for ops in [(d_(:up), dd_(:up), d_(:dn), dd_(:dn)), (q(:up), dd_(:up), d_(:dn), dd_(:dn))]
            for p in all_perms(4)
                v3_compare(run, f, ed, Tuple(ops[i] for i in p), g)
            end
        end
        v3_compare(run, f, ed, (bos, x', x, bos'), g)
        r0 = v3_compare(run, f, ed, (d_(:dn), x, dd_(:up)), g)     # total 2S_z = 2: pruned
        @test r0.meta[:pruned]
    end
    @testset "N = 2 (256 states, 3 bath sites)" begin
        N = 2
        run = run_nrg(m, NRGParams(N = N, Nkeep = typemax(Int)))
        f = fdm(run; T = T)
        ed = model_ed(m, run.chains, N, T)
        v3_compare(run, f, ed, (d_(:up), dd_(:up)), g)
        v3_compare(run, f, ed, (d_(:up), dd_(:up), n_(:dn)), g)
        v3_compare(run, f, ed, (d_(:up), dd_(:dn), d_(:dn), dd_(:up)), g)
        v3_compare(run, f, ed, (q(:up), x', x, dd_(:up)), g)
    end
end
