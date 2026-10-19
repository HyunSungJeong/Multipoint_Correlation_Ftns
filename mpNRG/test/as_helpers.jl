# V4: exactness with truncation (task sheet Sec. 6, V4). Tiny chains with N_keep
# well below the full dimension, so that KK, KD, DK and DD sectors and all
# refinement terms occur. The Anders-Schiller (AS) basis is built explicitly as
# dense vectors on the full chain (sites -1..N, mode order [N; …; -1]).

using LinearAlgebra

# dense AS data: per shell n, the dense shell eigenvectors U_n (on sites ≤ n)
# and the environment dimension d^{N-n}
function as_basis(run)
    N = run.params.N
    U = Dict{Int,Any}()
    for n in -1:N
        U[n] = dense_embedding(run, n)
    end
    return U
end

# dense projector P^n_X (X = :K, :D) on the full chain: 1_env ⊗ Σ_s |s⟩⟨s|
function as_projector(run, U, n, X)
    N = run.params.N
    sh = shell(run, n)
    denv = 4^(N - n)                     # SIAM: 4 states per Wilson site
    cols = reduce(hcat, [U[n].cols[q][:, xrange(sh, q, X)] for q in sectors(sh.space)])
    return kron(Matrix(1.0I, denv, denv), cols * cols')
end

# environment parity (-1)^{N_e} on the Fock states of sites N..n+1
env_parity(N, n) = Diagonal([isodd(count_ones(i)) ? -1.0 : 1.0 for i in 0:(4^(N - n) - 1)])

# the shell-n blocks O^n_{XX̄} of a shell operator embedded on the full chain. With the
# creators of later sites to the left (|se⟩ = c†(e)|s⟩), an operator of parity p
# on sites ≤ n acts on the environment as (-1)^{p N_e}: its AS matrix elements
# are δ_{eē} (-1)^{p N_e} [O^n]_{ss̄} (Lee Eq. (30) writes δ_{eē}; the sign
# cancels in every trace, since the products traced are even).
function as_embed(run, U, O, n, X, Xb)
    N = run.params.N
    sh = shell(run, n)
    Df = size(U[n].cols[first(sectors(sh.space))], 1)
    M = zeros(Df, Df)
    for q in sectors(sh.space)
        r = target(O, q)
        haskey(sh.space, r) || continue
        rr = xrange(sh, r, X); cc = xrange(sh, q, Xb)
        (isempty(rr) || isempty(cc)) && continue
        M .+= U[n].cols[r][:, rr] * getblock(O, q)[rr, cc] * U[n].cols[q][:, cc]'
    end
    Zenv = parity(O.shift) == 1 ? Matrix(env_parity(N, n)) : Matrix(1.0I, 4^(N - n), 4^(N - n))
    return kron(Zenv, M)
end

# dense FDM density matrix in the AS basis: Σ_n Σ_{s∈D} e^{-β(E^n_s - E_ref)}/Z |se⟩⟨se|
function as_rho(run, U, f)
    N = run.params.N
    ρ = zeros(4^(N + 2), 4^(N + 2))
    for n in -1:N
        sh = shell(run, n)
        for q in sectors(sh.space)
            dr = drange(sh, q)
            isempty(dr) && continue
            V = U[n].cols[q][:, dr]
            wts = exp.(-f.β .* (sh.E[q][dr] .+ sh.shift .- f.Eref) .- f.logZ)
            ρ .+= kron(Matrix(1.0I, 4^(N - n), 4^(N - n)), V * Diagonal(wts) * V')
        end
    end
    return ρ
end

