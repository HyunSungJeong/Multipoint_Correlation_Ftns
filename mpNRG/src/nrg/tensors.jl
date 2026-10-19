# The NRG tensor layer (refactor sheet TNOPS_REFACTOR.md, Sec. 4).
#
# The only tensor operations `nrg/`, `operators/`, `fdm/`, `psf/`, `hooks/`
# and `observables/` may call. Each is a named function built from TNOps
# primitives; no code outside TNOps and this file touches block dictionaries,
# except the declared dense kernels of `psf/engine.jl`, which read dense
# blocks through `block_view` only.
#
# Objects (legs of the isometry in this order):
#
#   * site leg    the local space of site n, untagged, `Out`;
#   * shell leg   all eigenstates of shell n: for every label q the sectors
#                 (q, K) and (q, D), tags `KTAG = 1` and `DTAG = 2`, dimensions
#                 `nkept(q)` and `ndisc(q)`; kept states first in every label;
#   * kept leg    the K sectors of a shell leg;
#   * A^n         the isometry of Lee Eq. (15) as a rank-3 tensor with legs
#                 (old = kept leg of shell n-1, `Out`; site, `Out`;
#                 new = shell leg of shell n, `In`), neutral flux. Its block
#                 ((q,K), (σ,0), (q′,X)) is the part of the pre-refactor matrix
#                 `Shell.A[q′]` with the rows of the pair (σ, q) of the fusion map
#                 and the X columns, reshaped to (dq, dσ, dX): the product index
#                 `s + (iσ-1) dq` makes `old` the fast index. At n = -1, A is rank
#                 2 (site = impurity states, new);
#   * operators   of shell n are rank-2 tensors on (shell leg `Out`, shell leg
#                 `In`), keyed by tagged sectors; kept-kept composites live on
#                 kept legs; sliced operators are `SlicedTensor`s.

const KTAG = 1      # kept states of a shell
const DTAG = 2      # discarded states of a shell

xtag(X::Symbol) = X === :K ? KTAG : (X === :D ? DTAG : throw(ArgumentError("sector string entries are :K or :D, got $X")))

"""    site_leg(loc) — the local space of a site as an untagged `Out` leg."""
site_leg(L::LocalSpace{S}) where {S} = Leg(Dict(q => length(v) for (q, v) in L.sectors); name = :site)

"""
    shell_leg(dims, nkeep) -> Leg

The tagged leg of all eigenstates of a shell: `(q, K)` with `nkeep[q]` and
`(q, D)` with `dims[q] - nkeep[q]` states.
"""
function shell_leg(dims::AbstractDict{S,<:Integer}, nk::AbstractDict{S,<:Integer}) where {S}
    secs = LegSector{S}[]; ds = Int[]
    for (q, d) in dims
        k = get(nk, q, 0)
        push!(secs, LegSector(q, KTAG)); push!(ds, k)
        push!(secs, LegSector(q, DTAG)); push!(ds, d - k)
    end
    return Leg(secs, ds; name = :shell)
end

"""    kept_leg(l) — the kept (K) sectors of a shell leg, as an `Out` leg named `:old`."""
kept_leg(l::Leg) = TNOps.withname(TNOps.witharrow(subleg(l, s -> s.tag == KTAG), Out), :old)

"""
    fuse_site(loc, prev) -> FusionMap

The product basis `{|σ_n⟩ ⊗ |s⟩^{n-1}_K}` of shell n (Lee Sec. III A): the site
leg (outer, slow index σ) fused with the kept leg of shell n-1 (inner, fast
index s), in the fermionic order of Lee p. 5 (`|σ_n⟩ ⊗ |s⟩ := c†(σ_n)|s⟩`).
"""
fuse_site(loc::Leg{S}, prev::Leg{S}) where {S} = fusionmap(loc, prev)

"""    diag_tensor(l, E, T) — the neutral diagonal operator with entries `E[q]` on the untagged or kept leg `l`."""
function diag_tensor(l::Leg{S}, E::AbstractDict{S,<:AbstractVector}, ::Type{T}) where {S,T}
    D = BlockTensor{T}(l, l, neutral(S))
    for (s, d) in zip(legsectors(l), l.dims)
        setblock!(D, (s, s), Matrix{T}(Diagonal(E[s.label][1:d])))
    end
    return D
end

"""
    product_hamiltonian(Fu, Ekept, hloc, hops) -> BlockTensor

`H^n` (n ≥ 1) on the product space of `Fu` (Lee Sec. III A):

    H^n = 1 ⊗ E^{n-1}_K + h_n ⊗ 1 + Σ_ν t_{nν} [(f†_{nν} Z_n) ⊗ f_{n-1,ν} + h.c.] ,

with the kept energies `Ekept` of shell n-1 (relative to its shift), the
on-site term `hloc` on the site leg and `hops = [(t, f†_{nν} Z_n, f_{n-1,ν}|_KK), …]`.
The hopping is the Kronecker product `(f† Z) ⊗ f` of rule (R1) (`kron_fused`);
no sign is entered by hand.
"""
function product_hamiltonian(Fu::FusionMap{S}, Ekept::AbstractDict{S,<:AbstractVector}, hloc::BlockTensor{T,2,S},
                             hops::AbstractVector, ::Type{T}) where {S,T}
    Il = identity_tensor(Fu.outer, T); Ik = identity_tensor(Fu.inner, T)
    Ek = diag_tensor(Fu.inner, Ekept, T)
    H = kron_fused(Il, Ek, Fu, Fu) + kron_fused(hloc, Ik, Fu, Fu)
    for (t, fdagZ, fprev) in hops
        hop = t * kron_fused(fdagZ, fprev, Fu, Fu)
        H = H + hop + hop'
    end
    return H
end

"""
    product_hamiltonian_dense(Fu, cols, Hd) -> BlockTensor

`H^0` on the product basis of `Fu` from a dense Hamiltonian `Hd` on a Fock
space: block `q′` is `cols[q′]' * Hd * cols[q′]`, `cols[q′]` the dense vectors of
the product states of sector `q′` (site 0 and the impurity, `_product_embedding`).
"""
function product_hamiltonian_dense(Fu::FusionMap{S}, cols::AbstractDict{S,<:AbstractMatrix}, Hd::AbstractMatrix{T}) where {S,T}
    Hp = BlockTensor{T}(Fu.out, Fu.out, neutral(S))
    for q in sectors(Fu.out)
        U = cols[q]
        setblock!(Hp, q, U' * Hd * U)
    end
    return Hp
end

"""
    diagonalize_shell(H, Fu; cut) -> (E, nkeep, A)

Diagonalize `H^n` per symmetry block (`TNOps.eigen`, threaded over blocks;
each block symmetrized as `(M+M')/2`), choose the kept states with
`cut(E) -> nkeep` (the global cut, `truncate_shell` → `TNOps.global_cut`), and
return the energies `E[q]` (ascending), the kept counts and the isometry
`A^n` with the K/D-tagged new leg (Lee Sec. III A, Eq. (15)). With `Fu =
nothing` (shell -1) `A` is rank 2 on the leg of `H`.
"""
function diagonalize_shell(H::BlockTensor{T,2,S}, Fu; cut) where {T,S}
    vals, U = eigen(H)
    E = Dict{S,Vector{Float64}}(s.label => v for (s, v) in vals)
    nk = cut(E)
    U2 = retag(U, 2, (s, d) -> [(KTAG, nk[s.label]), (DTAG, d - nk[s.label])])
    U2 = _named(U2, (Fu === nothing ? :site : :product, :new))
    A = Fu === nothing ? U2 : _named(split_legs(U2, 1, Fu), (:old, :site, :new))
    return E, nk, A
end

function _named(A::BlockTensor{T,N,S}, names::NTuple{N,Symbol}) where {T,N,S}
    ls = ntuple(i -> TNOps.withname(TNOps.leg(A, i), names[i]), Val(N))
    B = BlockTensor{T}(ls, TNOps.flux(A))
    for (k, M) in eachblock(A)
        setblock!(B, k, M)
    end
    return B
end

"""
    isometry_adjoint(A) — `A†` (legs: new `Out`, then old and site `In`), computed once per shell.
"""
isometry_adjoint(A::BlockTensor{T,3}) where {T} = adjoint(A, (1, 2), (3,))
isometry_adjoint(A::BlockTensor{T,2}) where {T} = A'

"""
    isometry_matrix(A, Fu) -> BlockTensor

`A^n` as a rank-2 tensor from the product leg of `Fu` to the shell leg (the
pre-refactor `Shell.A`): the old and site legs fused back.
"""
isometry_matrix(A::BlockTensor{T,3}, Fu::FusionMap) where {T} = fuse_legs(A, (1, 2))[1]
isometry_matrix(A::BlockTensor{T,2}, ::Nothing) where {T} = A

"""
    isometry_block(A, Fu, q) -> Matrix

The pre-refactor matrix `Shell.A[q]`: rows the product basis of sector `q`
(rows of the impurity states at n = -1), columns all eigenstates of `q`, kept
first.
"""
isometry_block(A::BlockTensor, Fu, q) = getblock(isometry_matrix(A, Fu), q)

"""
    rotate(A, P, Fu) -> BlockTensor

`A† P A` for an operator `P` on the product space of `Fu` (rotation into the
eigenbasis of shell n).
"""
function rotate(A::BlockTensor, P::BlockTensor{T,2}, Fu) where {T}
    U = isometry_matrix(A, Fu)
    return U' * P * U
end

"""
    lift_site(A, Adag, On) -> BlockTensor

`A† (O_n ⊗ 1) A` for an operator `O_n` on site n alone: rule (R2), `O_n` acts on
the site leg of `A` (no parity string).
"""
function lift_site(A::BlockTensor{TA,3}, Adag::BlockTensor{TA,3}, On::BlockTensor{TO,2}) where {TA,TO}
    T = contract(On, (2,), A, (2,))                 # (site′, old, new)
    T = permute(T, (2, 1, 3))                        # (old, site′, new)
    return contract(Adag, (2, 3), T, (1, 2))         # (new′, new)
end

"""
    refine_KK(A, Adag, O) -> BlockTensor

The refinement map of Lee Eqs. (31), (34): the KK block of an operator `O` of
shell n-1 mapped into all sectors of shell n,

    [O^n_{XX̄}]_{s s̄} = Σ [A^{σ†}_{XK}]_{s s′} [(Z_n)^p ⊗ O^{n-1}_{KK}]_{s′ s̄′} [A^{σ}_{KX̄}]_{s̄′ s̄} ,

with the parity string of rule (R1) (`p` the parity of the flux of `O`),
applied with `apply_parity` on the site leg. `O` may live on the full shell
leg of n-1 or on its kept leg. For a `SlicedTensor` the map acts slice by
slice.
"""
function refine_KK(A::BlockTensor{TA,3,S}, Adag::BlockTensor{TA,3,S}, O::BlockTensor{TO,2,S}) where {TA,TO,S}
    Okk = subtensor(O, (s -> s.tag == KTAG, s -> s.tag == KTAG))
    Okk = _onlegs(Okk, TNOps.leg(A, 1))
    T = contract(Okk, (2,), A, (1,))                # (old′, site, new)
    T = apply_parity(T, 2; power = parity(TNOps.flux(O)))
    return contract(Adag, (2, 3), T, (1, 2))         # (new′, new)
end
refine_KK(A::BlockTensor, Adag::BlockTensor, O::SlicedTensor) = mapslices(B -> refine_KK(A, Adag, B), O)

# the kept-kept operator re-labelled onto the kept leg of the isometry (same
# space; names and arrows aligned so that the contraction checks pass)
function _onlegs(O::BlockTensor{T,2,S}, kept::Leg{S}) where {T,S}
    TNOps.samespace(TNOps.leg(O, 2), kept) && TNOps.samespace(TNOps.leg(O, 1), kept) ||
        throw(DimensionMismatch("the KK operator does not live on the kept states of the isometry"))
    B = BlockTensor{T}(kept, kept, TNOps.flux(O))
    for (k, M) in eachblock(O)
        setblock!(B, k, M)
    end
    return B
end

"""
    rho_full(leg, ρK, ρD) -> BlockTensor

`ρ̄^n = blockdiag(ρ̄^n_{KK}, ρ̄^n_{DD})` on the shell leg (sector diagonal,
Lee Eq. (42)): the KK part a tensor on the kept leg, the DD part diagonal
(`ρD[q]` the Boltzmann weights of the discarded states, Lee Eq. (44)).
"""
function rho_full(l::Leg{S}, ρK::Union{Nothing,BlockTensor}, ρD::AbstractDict{S,<:AbstractVector}, ::Type{T}) where {S,T}
    R = BlockTensor{T}(l, l, neutral(S))
    if ρK !== nothing
        for (k, M) in eachblock(ρK)
            setblock!(R, k, M)
        end
    end
    for (q, d) in ρD
        isempty(d) && continue
        s = LegSector(q, DTAG)
        setblock!(R, (s, s), Matrix{T}(Diagonal(d)))
    end
    return R
end

"""
    backward_step(A, Adag, ρ) -> BlockTensor

One step of the FDM backward sweep (Lee Eqs. (40)-(45); Weichselbaum & von
Delft (2007), Eq. (8)): `ρ̄^{n-1}_{KK} = Σ_σ A^σ ρ̄^n (A^σ)†`, the partial trace
over the site leg of the shell-n reduced density matrix `ρ` (all sectors,
[`rho_full`](@ref)); the result lives on the kept leg of shell n-1.
"""
function backward_step(A::BlockTensor{TA,3}, Adag::BlockTensor{TA,3}, ρ::BlockTensor{TR,2}) where {TA,TR}
    T = contract(A, (3,), ρ, (1,))                  # (old, site, new′)
    return contract(T, (2, 3), Adag, (3, 1))        # (old, old′)
end

"""
    shell_trace(ρ, ops...) -> Number

`Σ_q degeneracy(q) tr[ρ̄^n_q O¹ ⋯]` for sector-projected shell-n operators
(Lee Eqs. (46)-(48)); exact zero if the product is not neutral.
"""
function shell_trace(ρ::BlockTensor, ops::BlockTensor...)
    P = ρ
    for O in ops
        P = P * O
    end
    return tr(P)
end

"""
    block_view(O, q, X, X̄) -> Matrix or nothing

The dense block of a shell operator with columns in the `X̄` states of label
`q` and rows in the `X` states of its target label (`X, X̄ ∈ {:K, :D}`);
`nothing` if not stored. Read only. The declared dense kernels of
`psf/engine.jl` see operators through this function only.
"""
function block_view(O::BlockTensor{T,2,S}, q::S, X::Symbol, Xb::Symbol) where {T,S}
    return TNOps.blockornothing(O, (LegSector(target(O, q), xtag(X)), LegSector(q, xtag(Xb))))
end

"""
    shell_tensor(l, Δ, blocks, T) -> BlockTensor

A shell operator on the tagged leg `l` (rows and columns) from full label
blocks: `blocks[q]` is the dense matrix from all states of label `q` to all
states of `q ⊕ Δ`, kept states first; it is split into its tagged blocks.
"""
function shell_tensor(l::Leg{S}, Δ::S, blocks::AbstractDict{S,<:AbstractMatrix}, ::Type{T}) where {S,T}
    A = BlockTensor{T}(l, l, Δ)
    for (q, M) in blocks
        setblock!(A, q, M)
    end
    return A
end

"""
    kk_tensor(kept, Δ, blocks, T) -> BlockTensor

A kept-kept operator (composite of the PSF engine) on the kept leg `kept`
from dense blocks keyed by column label (`nkept(q⊕Δ) × nkept(q)`); all-zero
blocks are not stored.
"""
function kk_tensor(kept::Leg{S}, Δ::S, blocks::AbstractDict{S,<:AbstractMatrix}, ::Type{T}) where {S,T}
    A = BlockTensor{T}(kept, kept, Δ)
    for (q, M) in blocks
        any(!iszero, M) && setblock!(A, (LegSector(target(A, q), KTAG), LegSector(q, KTAG)), M)
    end
    return A
end
