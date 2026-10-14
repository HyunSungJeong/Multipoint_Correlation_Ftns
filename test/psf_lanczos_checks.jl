# The checks L1-L12 of the PSF Lanczos task, as functions that return the
# achieved errors. test/test_psf_lanczos.jl asserts them against the sheet's
# tolerances; notebooks/psf_lanczos.ipynb prints them. These call the public API
# of src/psf_lanczos.jl and the references of psf_lanczos_reference.jl.
#
# Errors are relative to the natural scale: energies to the largest |E_a - E_b|,
# weights to the largest weight, unless stated otherwise.

include(joinpath(@__DIR__, "psf_lanczos_reference.jl"))

"""Everything a fixture needs: input data, algorithm output, operator-space reference."""
function run_fixture(fx; ref::Bool = true)
    inp = psf_krylov_input(fx.sp, fx.O1s, fx.O2, fx.O3s; otol = fx.otol)
    K = psf_krylov(inp)
    return (fx = fx, inp = inp, K = K, ref = ref ? ref_krylov(fx) : nothing)
end

maxabs(x) = isempty(x) ? 0.0 : maximum(abs, x)
_relerr(a, b, s) = maxabs(a - b) / s
deflated(widths, M) = any(<(M), widths[1:(end - 1)])          # narrowing before the last block

"""normalised weights 𝐰_β = 𝐍⁻¹𝐚_β𝐍⁻¹ of a measure (Σ_β 𝐰_β = 𝟏)"""
function normalised_weights(weights)
    F = eigen(Hermitian(sum(weights)))
    Ninv = F.vectors * Diagonal(1 ./ sqrt.(F.values)) * F.vectors'
    return [Ninv * a * Ninv for a in weights]
end

# --------------------------------------------------------------------------
# L1 — input contract
# --------------------------------------------------------------------------

function check_L1(r)
    fx, K = r.fx, r.K
    sp = fx.sp
    raw = ref_raw_psf(fx)
    pos = maximum(maxabs(t.position - [sp.E[t.states[2]] - sp.E[t.states[1]], sp.E[t.states[3]] - sp.E[t.states[1]]])
                  for T in raw for t in T; init = 0.0) / energy_scale(sp)
    G3 = ref_gram(sp, fx.O3s, fx.O3s)
    G1 = ref_gram(sp, [Matrix(adjoint(O)) for O in fx.O1s], [Matrix(adjoint(O)) for O in fx.O1s])
    gram = max(_relerr(sum(r.inp.weights3), G3, maxabs(G3)), _relerr(sum(r.inp.weights1), G1, maxabs(G1)),
               _relerr(K.N3 * K.N3, G3, maxabs(G3)), _relerr(K.N1 * K.N1, G1, maxabs(G1)))
    tot = maximum(abs(psf_total_weight(raw[i, j]) - thermal_average(sp, fx.O1s[i] * fx.O2 * fx.O3s[j])) /
                  sum(t -> abs(t.weight), raw[i, j]; init = floatmin())
                  for i in axes(raw, 1), j in axes(raw, 2))
    return (positions = pos, gram = gram, total_weight = tot)
end

# --------------------------------------------------------------------------
# L2 — analytic anchor, Hubbard atom
# --------------------------------------------------------------------------

function check_L2(U::Real, β::Real)
    r = run_fixture(fixture(:F2; U = U, β = β); ref = false)
    K, inp = r.K, r.inp
    u = U / 2
    p = exp(β * u) / (1 + exp(β * u))
    w = real.(only.(normalised_weights(inp.weights3)))
    sc = abs(u)
    ip, im = argmin(abs.(inp.points3 .- u)), argmin(abs.(inp.points3 .+ u))
    peaks = length(inp.points3) == 2 ?
            max(abs(w[ip] - p), abs(w[im] - (1 - p)), abs(inp.points3[ip] - u) / sc, abs(inp.points3[im] + u) / sc) : Inf
    N3sq = abs(only(K.N3)^2 - 0.5)
    chain = length(K.eps3) == 2 ?
            max(abs(only(K.eps3[1]) - u * tanh(β * u / 2)), abs(only(K.t3[2]) - abs(u) / cosh(β * u / 2)),
                abs(only(K.eps3[2]) + u * tanh(β * u / 2))) / sc : Inf
    same = (length(K.eps1) == length(K.eps3)) ?
           max(maximum(abs(only(K.eps1[n]) - only(K.eps3[n])) for n in eachindex(K.eps1)),
               maximum(abs(only(K.t1[n]) - only(K.t3[n])) for n in 2:length(K.t1))) / sc : Inf
    return (peaks = peaks, N3sq = N3sq, chain = chain, bra_eq_ket = same, nlevels = length(K.eps3))
end

# --------------------------------------------------------------------------
# L3 — Jacobi blocks vs operator space
# --------------------------------------------------------------------------

"""
Krylov dimension of a chain counted as on the PSF side: eigenvalues of `𝕁` grouped
into support points (rounded keys of `postol`), each group counted with the
rank of its first-block weight `Σ x x†`.
"""
function effective_dimension(J::AbstractMatrix, M::Integer; postol = 1e-10, ranktol = 1e-12)
    F = eigen(Hermitian((J + J') / 2))
    digits = max(0, round(Int, -log10(postol)))
    groups = Dict{Float64,Vector{Int}}()
    for (i, λ) in enumerate(F.values)
        push!(get!(groups, round(λ; digits = digits) + 0.0, Int[]), i)
    end
    d = 0
    for g in values(groups)
        X = F.vectors[1:M, g]
        λs = eigvals(Hermitian(X * X'))
        d += count(>(ranktol * maximum(λs)), λs)
    end
    return d, length(groups)
end

function _chain_errors(eps, t, widths, epsr, tr, widthsr, M, levels)
    n = 0
    ee, et = 0.0, 0.0
    for k in 1:min(levels, length(eps), length(epsr))
        (widths[k] == M && widthsr[k] == M) || break     # gauge fixed only without deflation (P6)
        ee = max(ee, maxabs(eps[k] - epsr[k]))
        k > 1 && (et = max(et, maxabs(t[k] - tr[k])))
        n = k
    end
    return ee, et, n
end

function check_L3(r; levels = 20)
    K, ref = r.K, r.ref
    sc = energy_scale(r.fx.sp)
    e3, t3, n3 = _chain_errors(K.eps3, K.t3, K.widths3, ref.eps3, ref.t3, ref.widths3, K.M3, levels)
    e1, t1, n1 = _chain_errors(K.eps1, K.t1, K.widths1, ref.eps1, ref.t1, ref.widths1, K.M1, levels)
    d3, _ = effective_dimension(ref_jacobi(ref.eps3, ref.t3), K.M3)
    d1, _ = effective_dimension(ref_jacobi(ref.eps1, ref.t1), K.M1)
    return (eps = max(e3, e1) / sc, t = max(t3, t1) / sc, levels = min(n3, n1),
            R = (sum(K.widths1), sum(K.widths3)), R_ref = (sum(ref.widths1), sum(ref.widths3)),
            R_ref_eff = (d1, d3), deflation = (deflated(K.widths1, K.M1), deflated(K.widths3, K.M3)),
            deflation_ref = (deflated(ref.widths1, K.M1), deflated(ref.widths3, K.M3)))
end

# --------------------------------------------------------------------------
# L4 — Gauss quadrature (Golub-Welsch)
# --------------------------------------------------------------------------

"""
Eigenvalues of `𝕁` against the support points `b_β`, and the first-block parts
of its eigenvectors, summed over the eigenvalues at each point, against `𝐰_β`.
"""
function gauss_errors(J::AbstractMatrix, M::Integer, points, weights)
    F = eigen(Hermitian((J + J') / 2))
    w = normalised_weights(weights)
    acc = [zeros(ComplexF64, M, M) for _ in points]
    eig = 0.0
    for (k, λ) in enumerate(F.values)
        β = argmin(abs.(points .- λ))
        eig = max(eig, abs(points[β] - λ))
        x = F.vectors[1:M, k]
        acc[β] += x * x'
    end
    return eig, maximum(maxabs(acc[β] - w[β]) for β in eachindex(points))
end

function check_L4(r)
    K, inp, ref = r.K, r.inp, r.ref
    sc = energy_scale(r.fx.sp)
    out = Dict{Symbol,Float64}()
    for (name, J, M, pts, ws) in ((:ket, jacobi_matrix(K.eps3, K.t3), K.M3, inp.points3, inp.weights3),
                                  (:bra, jacobi_matrix(K.eps1, K.t1), K.M1, inp.points1, inp.weights1),
                                  (:ket_ref, ref_jacobi(ref.eps3, ref.t3), K.M3, inp.points3, inp.weights3),
                                  (:bra_ref, ref_jacobi(ref.eps1, ref.t1), K.M1, inp.points1, inp.weights1))
        e, w = gauss_errors(J, M, pts, ws)
        out[Symbol(name, :_eig)] = e / sc
        out[Symbol(name, :_w)] = w
    end
    return (eig = max(out[:ket_eig], out[:bra_eig]), w = max(out[:ket_w], out[:bra_w]),
            eig_ref = max(out[:ket_ref_eig], out[:bra_ref_eig]), w_ref = max(out[:ket_ref_w], out[:bra_ref_w]))
end

# --------------------------------------------------------------------------
# L5 — orthonormality and polynomial consistency
# --------------------------------------------------------------------------

"""
The matrix polynomials `𝐩_n(b)` of a chain by the recursion
`ω𝐩_n = 𝐩_n𝛜_n + 𝐩_{n+1}𝐭_{n+1} + 𝐩_{n-1}𝐭_n†`, `𝐩₁ = 𝟏`, at the point `b`.
"""
function chain_polynomials(eps, t, b::Real, nblocks::Integer)
    M = size(eps[1], 1)
    p = Matrix{ComplexF64}[Matrix{ComplexF64}(I, M, M)]
    for n in 1:(min(nblocks, length(eps)) - 1)
        X = b * p[n] - p[n] * eps[n]
        n > 1 && (X -= p[n - 1] * t[n]')
        T = t[n + 1]
        push!(p, size(T, 1) == size(T, 2) ? X / T : X * pinv(T))
    end
    return p
end

function _poly_error(eps, t, U, B, rows, points, widths, nblocks)
    off = cumsum([0; widths])
    err = 0.0
    for β in eachindex(points)
        isempty(rows[β]) && continue
        p = chain_polynomials(eps, t, points[β], nblocks)
        for n in eachindex(p)
            err = max(err, maxabs(B[β] * p[n] - U[rows[β], (off[n] + 1):off[n + 1]]))
        end
    end
    return err
end

function check_L5(r; nblocks = 10)
    K = r.K
    orth = max(maxabs(K.U3' * K.U3 - I), maxabs(K.U1' * K.U1 - I))
    poly = max(_poly_error(K.eps3, K.t3, K.U3, K.B3, K.rows3, K.points3, K.widths3, nblocks),
               _poly_error(K.eps1, K.t1, K.U1, K.B1, K.rows1, K.points1, K.widths1, nblocks))
    return (orth = orth, poly = poly)
end

# --------------------------------------------------------------------------
# L6 — coupling matrix
# --------------------------------------------------------------------------

function check_L6(r; levels = 20)
    K, ref, fx = r.K, r.ref, r.fx
    _, _, n3 = _chain_errors(K.eps3, K.t3, K.widths3, ref.eps3, ref.t3, ref.widths3, K.M3, levels)
    _, _, n1 = _chain_errors(K.eps1, K.t1, K.widths1, ref.eps1, ref.t1, ref.widths1, K.M1, levels)
    c1, c3 = sum(K.widths1[1:n1]), sum(K.widths3[1:n3])
    sc = maxabs(ref.V)
    V = _relerr(K.V[1:c1, 1:c3], ref.V[1:c1, 1:c3], sc)
    expect = [thermal_average(fx.sp, O1 * fx.O2 * O3) for O1 in fx.O1s, O3 in fx.O3s]
    sumrule = _relerr(K.N1 * K.V[1:K.M1, 1:K.M3] * K.N3, expect, max(maxabs(expect), maxabs(K.N1) * maxabs(K.N3) * sc))
    return (V = V, blocks = (n1, n3), sumrule = sumrule)
end

# --------------------------------------------------------------------------
# L7 — reconstruction, Eq. (12)
# --------------------------------------------------------------------------

"""
Compare reconstructed term lists with the aggregated input PSF `S`, peak by peak
(positions matched by rounded keys, nearest point within `1e-8` otherwise).
Returns (weight error on the input peaks, largest weight elsewhere, position
error), all relative.
"""
function compare_psf(rec::Matrix{Vector{PSFTerm}}, S::Matrix{Vector{PSFTerm}}, digits::Integer, escale::Real)
    smax = maximum(maxabs([t.weight for t in T]) for T in S)
    werr, extra, perr = 0.0, 0.0, 0.0
    for i in eachindex(S)
        P = [t.position for t in rec[i]]
        used = falses(length(P))
        index = Dict(round.(p; digits = digits) .+ 0.0 => k for (k, p) in enumerate(P))
        for t in S[i]
            k = get(index, round.(t.position; digits = digits) .+ 0.0, 0)
            if k == 0 && !isempty(P)
                d, k2 = findmin(p -> maximum(abs.(p - t.position)), P)
                k = d < 1e-8 ? k2 : 0
            end
            if k == 0
                werr = max(werr, abs(t.weight))
            else
                used[k] = true
                werr = max(werr, abs(rec[i][k].weight - t.weight))
                perr = max(perr, maxabs(rec[i][k].position - t.position))
            end
        end
        extra = max(extra, maxabs([rec[i][k].weight for k in findall(!, used)]))
    end
    return werr / smax, extra / smax, perr / escale
end

function check_L7(r)
    K, ref, inp = r.K, r.ref, r.inp
    sc = energy_scale(r.fx.sp)
    # (J_ref, J′_ref, V_ref): operator side only, any gauge
    rec_ref = reconstruct_psf(ref_jacobi(ref.eps1, ref.t1), ref_jacobi(ref.eps3, ref.t3), ref.V, ref.N1, ref.N3)
    w_ref, x_ref, p_ref = compare_psf(rec_ref, inp.S, inp.digits, sc)
    # (J, J′, V_ref): the algorithm's chains with the operator-space coupling,
    # only where the two share a gauge (same dimensions, no deflation)
    same = K.widths1 == ref.widths1 && K.widths3 == ref.widths3 &&
           !deflated(K.widths1, K.M1) && !deflated(K.widths3, K.M3)
    w_alg, x_alg = same ? compare_psf(reconstruct_psf(K; V = ref.V), inp.S, inp.digits, sc)[1:2] : (NaN, NaN)
    w_own, x_own = compare_psf(reconstruct_psf(K), inp.S, inp.digits, sc)[1:2]
    return (ref_chains = max(w_ref, x_ref), ref_positions = p_ref, alg_chains_Vref = max(w_alg, x_alg),
            own_V = max(w_own, x_own), gauge_shared = same)
end

# --------------------------------------------------------------------------
# L8 — resolvent
# --------------------------------------------------------------------------

"""The (z₁, z₂) grid of L8: four half-plane combinations, |Im z| ∈ {1, 0.1, 0.01}."""
function resolvent_grid(sc::Real)
    xs = [(-0.37, 0.52), (0.11, -0.23), (0.61, 0.08)] .|> x -> sc .* x
    return [(complex(x1, s1 * η), complex(x2, s2 * η)) for (x1, x2) in xs, s1 in (1, -1), s2 in (1, -1),
            η in (1.0, 0.1, 0.01)] |> vec
end

function check_L8(r)
    K, fx = r.K, r.fx
    raw = ref_raw_psf(fx)
    err_k, err_t = 0.0, 0.0
    for (z1, z2) in resolvent_grid(energy_scale(fx.sp))
        Fl = ref_resolvent_lehmann(fx, z1, z2; terms = raw)
        s = maxabs(Fl)
        err_k = max(err_k, _relerr(resolvent_krylov(K, z1, z2), Fl, s))
        err_t = max(err_t, _relerr(ref_resolvent_trace(fx, z1, z2), Fl, s))
    end
    cf = 0.0
    for (eps, t, M) in ((K.eps3, K.t3, K.M3), (K.eps1, K.t1, K.M1)), z in (0.3 + 0.01im, -0.7 - 0.1im, 1.1 + 1im)
        G = inv(z * I - jacobi_matrix(eps, t))
        col = reduce(vcat, resolvent_column(eps, t, z))
        cf = max(cf, _relerr(col, G[:, 1:M], maxabs(G[:, 1:M])))
    end
    return (krylov = err_k, trace = err_t, cf = cf)
end

# --------------------------------------------------------------------------
# L9 — truncation and moments
# --------------------------------------------------------------------------

"""Exact moments `𝐌_ab = Σ w ε₁^a ε₂^b` from the raw 3p terms, `M′×M` each."""
exact_moment(raw, a, b) = [sum(t -> t.weight * t.position[1]^a * t.position[2]^b, T; init = zero(ComplexF64)) for T in raw]

"""
`𝐍₁ Σ_mn [(𝕁′_{N′})^a]_{1m} 𝐕_mn [(𝕁_N)^b]_{n1} 𝐍₃` with chains truncated after
`Nbra`, `Nket` blocks.
"""
function krylov_moment(K, a, b, Nbra, Nket)
    J1, J3 = jacobi_matrix(K.eps1, K.t1; N = Nbra), jacobi_matrix(K.eps3, K.t3; N = Nket)
    x1 = Matrix{ComplexF64}(I, size(J1, 1), K.M1)
    x3 = Matrix{ComplexF64}(I, size(J3, 1), K.M3)
    for _ in 1:a; x1 = J1 * x1; end
    for _ in 1:b; x3 = J3 * x3; end
    return K.N1 * x1' * K.V[1:size(J1, 1), 1:size(J3, 1)] * x3 * K.N3
end

"""
Relative moment errors for `a ≤ amax`, `b ≤ bmax` with chains truncated at
`(Nbra, Nket)`; relative to `Σ|w| max|ε₁|^a max|ε₂|^b`.
"""
function moment_errors(r, Nbra, Nket, amax, bmax; raw = ref_raw_psf(r.fx))
    W = maximum(sum(t -> abs(t.weight), T; init = 0.0) for T in raw)
    e1 = maximum(abs(t.position[1]) for T in raw for t in T)
    e2 = maximum(abs(t.position[2]) for T in raw for t in T)
    E = zeros(amax + 1, bmax + 1)
    for a in 0:amax, b in 0:bmax
        E[a + 1, b + 1] = _relerr(krylov_moment(r.K, a, b, Nbra, Nket), exact_moment(raw, a, b), W * e1^a * e2^b)
    end
    return E
end

function check_L9(r; truncations = [(2, 3), (4, 2), (5, 5)])
    raw = ref_raw_psf(r.fx)
    inside, boundary = 0.0, Inf
    for (Nb, Nk) in truncations
        E = moment_errors(r, Nb, Nk, Nb, Nk; raw = raw)
        inside = max(inside, maximum(E[1:Nb, 1:Nk]))
        boundary = min(boundary, maximum(vcat(E[Nb + 1, 1:Nk], E[1:Nb, Nk + 1])))
    end
    return (inside = inside, boundary = boundary)
end

# --------------------------------------------------------------------------
# L10 — factorization lemma
# --------------------------------------------------------------------------

function check_L10(r)
    K, fx, inp = r.K, r.fx, r.inp
    digits = inp.digits
    key(x) = round(x; digits = digits) + 0.0
    k1 = Set(key(t.position[1]) for O in fx.O1s, P in fx.O1s for t in psf(fx.sp, O, adjoint(P); otol = fx.otol))
    k3 = Set(key(t.position[1]) for O in fx.O3s, P in fx.O3s for t in psf(fx.sp, adjoint(O), P; otol = fx.otol))
    outside = count(t -> !(key(t.position[1]) in k1 && key(t.position[2]) in k3), (t for T in inp.S for t in T))
    N1i, N3i = inv(K.N1), inv(K.N3)
    st = [N1i * w * N3i for (_, _, w) in K.s]
    err = maximum(maxabs(st[i] - K.B1[a]' * K.kernel[K.rows1[a], K.rows3[b]] * K.B3[b])
                  for (i, (a, b, _)) in enumerate(K.s))
    return (outside = outside, factorization = err / maximum(maxabs, st))
end

# --------------------------------------------------------------------------
# L11 — non-uniqueness (negative control)
# --------------------------------------------------------------------------

"""The input with the ket measure replaced by uniform weights on the same support."""
function wrong_measure(inp::PSFKrylovInput)
    P = length(inp.points3)
    A = sum(inp.weights3)
    return PSFKrylovInput(inp.points1, inp.weights1, inp.points3, [A / P for _ in 1:P], inp.s, inp.S, inp.digits)
end

function check_L11(r)
    Kw = psf_krylov(wrong_measure(r.inp))
    sc = energy_scale(r.fx.sp)
    rec = compare_psf(reconstruct_psf(Kw), r.inp.S, r.inp.digits, sc)
    ref = r.ref
    n = min(length(Kw.eps3), length(ref.eps3), 20)
    chain = max(maximum(maxabs(Kw.eps3[k] - ref.eps3[k]) for k in 1:n),
                n > 1 ? maximum(maxabs(Kw.t3[k] - ref.t3[k]) for k in 2:n) : 0.0) / sc
    return (reconstruction = max(rec[1], rec[2]), chain = chain, K = Kw)
end
