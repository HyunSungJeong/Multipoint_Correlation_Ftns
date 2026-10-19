# Shell-sector projections of local operators (task sheet Sec. 5.5; Lee
# Sec. IV A-B).

"""
    LocalOperator

A local operator for PSF and thermal-average calls: its expression, its
homogeneous components keyed by shift (Sec. 5.2.3), and the shell `n_op` at
which it is first represented exactly (the largest site it acts on; Lee
Sec. IV B treats operators acting nontrivially only on sites `-1..n₀` as
local).
"""
struct LocalOperator{S<:Sector}
    expr::OpExpr
    comps::Dict{S,OpExpr}
    nop::Int
    name::String
end

"""
    local_operator(run, expr; name = string(expr)) -> LocalOperator

Wrap an operator expression; checks that it acts only on sites `≤ n₀`, the
last shell without truncation, where its matrix elements are exact.
"""
function local_operator(run::NRGRun{S}, expr::OpExpr; name::AbstractString = string(expr)) where {S}
    ch = charges(run.model, max(maxsite(expr), 0))
    comps = Dict{S,OpExpr}(components(expr, ch))
    nop = max(maxsite(expr), -1)
    nop ≤ run.n0 || throw(ArgumentError(
        "operator acts on site $nop beyond n₀ = $(run.n0), the last shell without truncation; it is not local (Lee Sec. IV B)"))
    return LocalOperator{S}(expr, comps, nop, String(name))
end
local_operator(run::NRGRun, op::LocalOperator; kwargs...) = op

ishomogeneous(o::LocalOperator) = length(o.comps) ≤ 1
"""    opshift(o::LocalOperator) — the shift of a homogeneous operator."""
function opshift(o::LocalOperator{S}) where {S}
    isempty(o.comps) && return neutral(S)
    length(o.comps) == 1 || throw(ArgumentError("operator $(o.name) is inhomogeneous; use psf_multilinear"))
    return first(keys(o.comps))
end

"""
    shell_operator(run, expr, n = maxsite) -> BlockTensor

Matrix elements `[O^n]_{s s̄}` (all states, both sectors) of a homogeneous local
operator in the eigenbasis of shell `n ≤ n₀`, from the dense embedding of the
shell (exact: no truncation has happened yet); a tensor on the shell leg.
"""
function shell_operator(run::NRGRun{S,T}, expr::OpExpr, n::Integer = max(maxsite(expr), -1)) where {S,T}
    n ≤ run.n0 || throw(ArgumentError("shell $n is beyond n₀ = $(run.n0)"))
    emb = dense_embedding(run, n)
    ch = charges(run.model, max(n, 0))
    Δ = opshift(expr, ch)
    Tt = promote_type(T, eltype(expr) <: Real ? Float64 : ComplexF64)
    Od = densematrix(expr, emb.basis; T = Tt)
    sh = shell(run, n)
    blocks = Dict{S,Matrix{Tt}}()
    for q in sectors(sh.leg)
        r = fuse(q, Δ)
        haskey(sh.leg, r) || continue
        blocks[q] = emb.cols[r]' * (Od * emb.cols[q])
    end
    return shell_tensor(sh.leg, Δ, blocks, Tt)
end

"""
    refine(run, O, n) -> BlockTensor

The refinement map of a KK operator of shell `n-1` into all sectors of shell
`n` (Lee Eqs. (31), (34)):

    [O^n_{X X̄}]_{s s̄} = Σ [M^{σ_n †}_{XK}]_{s s′} [O^{n-1}_{KK}]_{s′ s̄′} [M^{σ_n}_{KX̄}]_{s̄′ s̄} ,

with the fermionic parity string of rule (R1): an operator of parity `p` on
earlier sites acts on the product basis as `Z_n^p ⊗ O`. Only the KK block of
`O` enters; `O` may live on the full leg of shell `n-1` or on its kept leg.
[`refine_KK`](@ref) with the isometry of shell `n`; a `SlicedTensor` is
refined slice by slice.
"""
function refine(run::NRGRun, O::Union{BlockTensor,SlicedTensor}, n::Integer)
    sh = shell(run, n)
    return refine_KK(sh.A, sh.Adag, O)
end

"""
    project(run, op::LocalOperator, n) -> BlockTensor

The homogeneous local operator in the basis of shell `n ≥ n_op`: built exactly
at `n_op` ([`shell_operator`](@ref)) and refined KK → next shell
([`refine`](@ref)) up to `n`. For `n ≤ n₀` this equals the exact matrix; for
later shells it is the shell-diagonal projection `O^n` of Lee Eq. (30), whose
non-KK sectors make up the exact expansion Eq. (33). Results are cached.
"""
function project(run::NRGRun, op::LocalOperator, n::Integer)
    expr = first(values(op.comps))
    length(op.comps) == 1 || throw(ArgumentError("project needs a homogeneous operator"))
    return project(run, expr, n)
end
function project(run::NRGRun, expr::OpExpr, n::Integer)
    # keyed by content (normal-ordered terms), never by object identity: object
    # ids are reused after garbage collection
    key = (:proj, normalorder(expr).terms, n)
    haskey(run.cache, key) && return run.cache[key]
    nop = max(maxsite(expr), -1)
    n ≥ nop || throw(ArgumentError("operator lives on sites up to $nop > $n"))
    O = n == nop ? shell_operator(run, expr, n) : refine(run, project(run, expr, n - 1), n)
    run.cache[key] = O
    return O
end

"""
    sector_view(run, O, n, q, X, X̄)

The block of `O^n` with columns in the `X̄` states of sector `q` and rows in
the `X` states of the target sector (`X, X̄ ∈ {:K, :D}`); `nothing` if absent.
[`block_view`](@ref).
"""
sector_view(run::NRGRun, O::BlockTensor, n::Integer, q, X::Symbol, Xb::Symbol) = block_view(O, q, X, Xb)
