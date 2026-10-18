# Tridiagonalization of the star into the Wilson chain (task sheet Sec. 5.3;
# Bulla, Pruschke & Hewson 1997; Bulla, Costi & Pruschke RMP 80, 395, Sec. II B).
#
# Lee indexing (Sec. III A): the impurity is site -1, bath sites are
# 0, 1, …, N; site n has on-site energy ε_n and t_n couples site n to n - 1.
# Thus t_0 = sqrt(Σ_k |V_k|²) is the impurity coupling.

"""
    lanczos_chain(star, nsites; prec = 256) -> (ε, t)

Lanczos tridiagonalization of the star Hamiltonian `diag(E_k)` started from the
normalized coupling vector `V/|V|`, in `BigFloat` with `prec` bits and full
reorthogonalization (double precision loses orthogonality because
`t_n ~ Λ^{-n/2}`). Returns `ε[n+1] = ε_n` (n = 0..nsites-1) and
`t[n+1] = t_n` (n = 0..nsites-1), with `t_0 = |V|`.
"""
function lanczos_chain(star::StarBath, nsites::Integer; prec::Integer = 256)
    setprecision(BigFloat, prec) do
        E = BigFloat.(star.E); V = BigFloat.(star.V)
        K = length(E)
        nsites ≤ K || throw(ArgumentError("chain of $nsites sites needs at least $nsites star levels, have $K"))
        t0 = sqrt(sum(V .^ 2))
        Q = zeros(BigFloat, K, nsites)
        Q[:, 1] = V / t0
        ε = zeros(BigFloat, nsites); t = zeros(BigFloat, nsites)
        t[1] = t0
        for n in 1:nsites
            w = E .* Q[:, n]
            ε[n] = dot(Q[:, n], w)
            n == nsites && break
            w .-= ε[n] .* Q[:, n]
            n > 1 && (w .-= t[n] .* Q[:, n - 1])
            for _ in 1:2, m in 1:n                 # full reorthogonalization, twice
                w .-= dot(Q[:, m], w) .* Q[:, m]
            end
            t[n + 1] = sqrt(dot(w, w))
            t[n + 1] > 0 || throw(ArgumentError("Lanczos breakdown at site $n: star too small"))
            Q[:, n + 1] = w / t[n + 1]
        end
        return ε, t
    end
end

"""
    WilsonChain

The Wilson chain of one flavor: `eps[n+1] = ε_n`, `t[n+1] = t_n` for
`n = 0..N` (`t_0` is the impurity coupling), plus the discretization data and
the two-precision check record.
"""
struct WilsonChain
    eps::Vector{Float64}
    t::Vector{Float64}
    hyb::Hybridization
    disc::Discretization
    nint::Int
    precision_check::NamedTuple
end

nsites(c::WilsonChain) = length(c.eps)
"""    chainlength(c) — the largest site index `N` (sites `0..N`)."""
chainlength(c::WilsonChain) = length(c.eps) - 1

"""
    wilson_chain(h, N; Λ = 4, z = 1, scheme = :zitko, nint = N + 12,
                 prec = (256, 512), tol = 1e-12) -> WilsonChain

Discretize hybridization `h` and tridiagonalize the star into a chain with
sites `0..N`. The tridiagonalization runs twice, in `BigFloat` with `prec[1]`
and `prec[2]` bits (two-precision self-check): every returned site must agree
between the two runs to relative accuracy `tol` (relative to `t_n` for both
`ε_n` and `t_n`), otherwise an error names the first failing site.
"""
function wilson_chain(h::Hybridization, N::Integer; Λ::Real = 4.0, z::Real = 1.0, scheme::Symbol = :zitko,
                      nint::Integer = N + 12, prec = (256, 512), tol::Real = 1e-12)
    d = Discretization(Λ = Λ, z = z, scheme = scheme, nint = nint)
    star = setprecision(() -> discretize(h, d, nint), BigFloat, prec[2])
    ε1, t1 = lanczos_chain(star, N + 1; prec = prec[1])
    ε2, t2 = lanczos_chain(star, N + 1; prec = prec[2])
    agree = Int[]
    worst = 0.0
    for n in 1:(N + 1)
        scale = abs(t2[n])
        r = Float64(max(abs(ε1[n] - ε2[n]), abs(t1[n] - t2[n])) / scale)
        worst = max(worst, r)
        r ≤ tol && push!(agree, n - 1)
        r ≤ tol || throw(ErrorException("two-precision check failed at chain site $(n - 1): relative deviation $r > $tol"))
    end
    return WilsonChain(Float64.(ε2), Float64.(t2), h, d, nint,
                       (precisions = prec, max_relative_deviation = worst, sites_accepted = length(agree)))
end

"""
    chain_star(c::WilsonChain, N = chainlength(c)) -> (E, V)

The star parameters of the **finite** chain with sites `0..N`: the eigenvalues
`E_k` of the chain hopping matrix and `V_k = t_0 U_{0k}` (U its eigenvectors).
The impurity coupled to this star is unitarily equivalent to the impurity
coupled to the chain, so impurity PSFs are identical (used for V3 against
`HubbardAtom.AIMModel`).
"""
function chain_star(c::WilsonChain, N::Integer = chainlength(c))
    Hc = SymTridiagonal(c.eps[1:(N + 1)], c.t[2:(N + 1)])
    F = eigen(Hc)
    V = c.t[1] .* F.vectors[1, :]
    return F.values, V
end

"""
    star_moments(star, m) / chain_moments(ε, t, m)

Moments `μ_m = Σ_k |V_k|² E_k^m` of the star and `t_0² [(H_chain)^m]_{00}` of a
chain, in `BigFloat`. A chain with sites `0..N` from Lanczos reproduces the
star moments for `m ≤ 2N + 1` (Gauss quadrature property; V2).
"""
star_moments(star::StarBath, m::Integer) = sum(big.(star.V) .^ 2 .* big.(star.E) .^ m)
function chain_moments(ε::AbstractVector, t::AbstractVector, m::Integer)
    n = length(ε)
    H = zeros(BigFloat, n, n)
    for i in 1:n
        H[i, i] = ε[i]
        i < n && (H[i, i + 1] = H[i + 1, i] = t[i + 1])
    end
    v = zeros(BigFloat, n); v[1] = 1
    for _ in 1:m
        v = H * v
    end
    return big(t[1])^2 * v[1]
end
