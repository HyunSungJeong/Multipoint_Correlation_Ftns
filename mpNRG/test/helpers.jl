# Shared test helpers. The dense Jordan-Wigner construction here is written
# independently of `fock/` (Kronecker products of Pauli-type matrices), so that
# V1 compares the package against code it does not share.

using LinearAlgebra

"""
    jw_annihilators(M) -> Vector{Matrix}

Annihilators of `M` fermionic modes on the `2^M`-dimensional Fock space, mode 1
the most significant tensor factor: `c_k = Z ⊗ ⋯ ⊗ Z ⊗ a ⊗ 1 ⊗ ⋯ ⊗ 1`, with
`a = [0 1; 0 0]` acting on the occupation basis (|0⟩, |1⟩) and the parity
string `Z = diag(1, -1)` on the modes to its left.
"""
function jw_annihilators(M::Integer)
    a = [0.0 1.0; 0.0 0.0]; Z = [1.0 0.0; 0.0 -1.0]; I2 = Matrix(1.0I, 2, 2)
    out = Matrix{Float64}[]
    for k in 1:M
        mats = [j < k ? Z : (j == k ? a : I2) for j in 1:M]
        push!(out, reduce(kron, mats))
    end
    return out
end

"""
    test_functions_error(P1, w1, P2, w2; ks) -> Float64

Distance between two discrete measures `Σ w δ(ε - P[:,j])` probed by the
Fourier test functions `exp(i k·ε)` for `k` on a grid: zero iff they agree on
these probes (positions and weights).
"""
function test_functions_error(P1, w1, P2, w2; ks = (0.3, 1.7, -2.2, 5.0, 11.0))
    L = size(P1, 1)
    err = 0.0
    for k in Iterators.product(ntuple(_ -> ks, L)...)
        kv = collect(k)
        a = sum(w1 .* exp.(im .* vec(kv' * P1)); init = 0.0im)
        b = sum(w2 .* exp.(im .* vec(kv' * P2)); init = 0.0im)
        err = max(err, abs(a - b))
    end
    return err
end

"""    dict_maxdiff(a, b) — max |a[k] - b[k]| over the union of keys (sparse histograms)."""
function dict_maxdiff(a::AbstractDict, b::AbstractDict)
    m = 0.0
    for k in union(keys(a), keys(b))
        m = max(m, abs(get(a, k, 0.0) - get(b, k, 0.0)))
    end
    return m
end

"""    hist_dict(r) — a binned PSFResult as a Dict of nonzero bins."""
hist_dict(r) = Dict(k => v for (k, v) in MultipointNRG.nonzero_bins(r))

"""    decoupled_tail(chains, nc) — the chains with t_n = ε_n = 0 for n > nc."""
decoupled_tail(chs, nc) = [MultipointNRG.WilsonChain([n ≤ nc ? c.eps[n + 1] : 0.0 for n in 0:(length(c.eps) - 1)],
                                                     [n ≤ nc ? c.t[n + 1] : 0.0 for n in 0:(length(c.t) - 1)],
                                                     c.hyb, c.disc, c.nint, c.precision_check) for c in chs]

"""    model_ed(m, chains, N, T) — reference ED of the chain with sites -1..N."""
function model_ed(m, chs, N, T)
    H = hamiltonian_expr(m, chs, N)
    ms = chain_modes(n -> site_modes(m, n), N)
    nimp = length(impurity_modes(m))
    allowed = chain_allowed(m.impurity.basis.states, nimp, length(ms))
    return ed_reference(H, ms, charges(m, N); β = 1 / T, allowed = allowed)
end

# operators of the SIAM used throughout
const dup = Mode(-1, :up); const ddn = Mode(-1, :dn)
d_(σ) = cop(Mode(-1, σ)); dd_(σ) = cdag(Mode(-1, σ)); n_(σ) = num(Mode(-1, σ))
f0_(σ) = cop(Mode(0, σ))
