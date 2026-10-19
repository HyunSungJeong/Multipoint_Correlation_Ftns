# Symbolic fermionic operator algebra on named modes (task sheet Sec. 5.1).
#
# Operators are polynomials in creation and annihilation operators, stored as
# a dictionary from words (products of ladder operators, left to right) to
# coefficients. They can be multiplied, added, conjugated, commuted and normal
# ordered symbolically, and evaluated into dense matrices on any Fock space
# that contains their modes (`fockspace.jl`). Composite operators such as
# q_σ = [d_σ, H_int] are therefore first-class.

"""
    Mode(site, name)

A fermionic mode: `site = -1` for the impurity, `site = n ≥ 0` for Wilson-chain
site `n` (Lee Sec. III A indexing), and a flavor `name` (e.g. `:up`).
"""
struct Mode
    site::Int
    name::Symbol
end

Base.isless(a::Mode, b::Mode) = isless((a.site, String(a.name)), (b.site, String(b.name)))
Base.show(io::IO, m::Mode) = m.site < 0 ? print(io, "d_", m.name) : print(io, "f", m.site, "_", m.name)

"""    Ladder(mode, dag) — `c†_mode` if `dag`, else `c_mode`."""
struct Ladder
    mode::Mode
    dag::Bool
end

Base.show(io::IO, l::Ladder) = (show(io, l.mode); l.dag && print(io, "†"))

# Content hashes. The default hash of a struct with a Symbol field depends on
# where the Symbol lives in memory, which changes with the code that is
# loaded; the iteration order of the term dictionaries, and with it the
# summation order of `densematrix` and of the operator algebra, would then
# change between builds (round-off differences of ~1e-16 that low-temperature
# FDM weights and 4p cancellations amplify to ~1e-12 of a PSF bin).
Base.hash(m::Mode, h::UInt) = hash(m.name, hash(m.site, hash(:Mode, h)))
Base.hash(l::Ladder, h::UInt) = hash(l.dag, hash(l.mode, hash(:Ladder, h)))

const Word = Vector{Ladder}

"""
    OpExpr{T}

A fermionic operator as a sum of words: `terms[w]` is the coefficient of the
product `w[1] w[2] ⋯ w[end]` (the empty word is the identity). Build with
[`cdag`](@ref), [`cop`](@ref), [`num`](@ref) and the arithmetic operators.
"""
struct OpExpr{T<:Number}
    terms::Dict{Word,T}
end

OpExpr{T}() where {T} = OpExpr{T}(Dict{Word,T}())
Base.eltype(::OpExpr{T}) where {T} = T

"""    cdag(m) — the creation operator of mode `m`."""
cdag(m::Mode) = OpExpr{Float64}(Dict([Ladder(m, true)] => 1.0))
"""    cop(m) — the annihilation operator of mode `m`."""
cop(m::Mode) = OpExpr{Float64}(Dict([Ladder(m, false)] => 1.0))
"""    num(m) — the occupation `c†_m c_m`."""
num(m::Mode) = cdag(m) * cop(m)
"""    identity_expr(T = Float64) — the identity operator."""
identity_expr(::Type{T} = Float64) where {T} = OpExpr{T}(Dict(Ladder[] => one(T)))

function _add!(d::Dict{Word,T}, w::Word, c) where {T}
    v = get(d, w, zero(T)) + c
    if iszero(v)
        delete!(d, w)
    else
        d[w] = v
    end
    return d
end

function Base.:+(a::OpExpr{TA}, b::OpExpr{TB}) where {TA,TB}
    T = promote_type(TA, TB)
    d = Dict{Word,T}(a.terms)
    for (w, c) in b.terms
        _add!(d, w, c)
    end
    return OpExpr{T}(d)
end
Base.:+(a::OpExpr, x::Number) = a + x * identity_expr()
Base.:+(x::Number, a::OpExpr) = a + x
Base.:-(a::OpExpr) = (-1) * a
Base.:-(a::OpExpr, b::OpExpr) = a + (-1) * b
Base.:-(a::OpExpr, x::Number) = a + (-x)
Base.:-(x::Number, a::OpExpr) = x + (-1) * a

function Base.:*(x::Number, a::OpExpr{TA}) where {TA}
    T = promote_type(typeof(x), TA)
    iszero(x) && return OpExpr{T}()
    return OpExpr{T}(Dict{Word,T}(w => x * c for (w, c) in a.terms))
end
Base.:*(a::OpExpr, x::Number) = x * a
Base.:/(a::OpExpr, x::Number) = inv(x) * a

function Base.:*(a::OpExpr{TA}, b::OpExpr{TB}) where {TA,TB}
    T = promote_type(TA, TB)
    d = Dict{Word,T}()
    for (wa, ca) in a.terms, (wb, cb) in b.terms
        _add!(d, vcat(wa, wb), ca * cb)
    end
    return normalorder(OpExpr{T}(d))
end

Base.:^(a::OpExpr, n::Integer) = n == 0 ? identity_expr(eltype(a)) : (n == 1 ? a : a * a^(n - 1))

"""
    adjoint(a)

Hermitian conjugate: words are reversed, daggers flipped and coefficients
conjugated.
"""
Base.adjoint(a::OpExpr{T}) where {T} =
    OpExpr{T}(Dict{Word,T}([Ladder(l.mode, !l.dag) for l in reverse(w)] => conj(c) for (w, c) in a.terms))

"""    comm(a, b) — the commutator `[a, b] = ab - ba`."""
comm(a::OpExpr, b::OpExpr) = a * b - b * a
"""    acomm(a, b) — the anticommutator `{a, b} = ab + ba`."""
acomm(a::OpExpr, b::OpExpr) = a * b + b * a

Base.iszero(a::OpExpr) = isempty(normalorder(a).terms)
Base.:(==)(a::OpExpr, b::OpExpr) = iszero(a - b)

"""    modes(a) — the sorted list of modes `a` acts on."""
modes(a::OpExpr) = sort!(unique!([l.mode for w in keys(a.terms) for l in w]))

"""
    maxsite(a) -> Int

The largest site index among the modes of `a` (`-1` for impurity-only
operators, `typemin(Int)` for multiples of the identity). An operator is local
in the sense of Lee Sec. IV B if `maxsite(a) ≤ n₀`.
"""
maxsite(a::OpExpr) = maximum((m.site for m in modes(a)); init = typemin(Int))

# Canonical order of ladder operators: creators left of annihilators,
# creators ascending in mode order, annihilators descending.
_lkey(l::Ladder) = l.dag ? (0, l.mode) : (1, l.mode)
function _inorder(a::Ladder, b::Ladder)
    a.dag != b.dag && return a.dag
    return a.dag ? isless(a.mode, b.mode) : isless(b.mode, a.mode)
end

"""
    normalorder(a) -> OpExpr

Normal order every word using `{c_i, c†_j} = δ_ij` and `{c_i, c_j} = 0`:
creators to the left (ascending mode order), annihilators to the right
(descending), with repeated equal ladders set to zero. Equal operators have
equal normal forms, which is how `==` and `iszero` are decided.
"""
function normalorder(a::OpExpr{T}) where {T}
    out = Dict{Word,T}()
    stack = Tuple{Word,T}[(w, c) for (w, c) in a.terms]
    while !isempty(stack)
        w, c = pop!(stack)
        iszero(c) && continue
        done = true
        for i in 1:(length(w) - 1)
            x, y = w[i], w[i + 1]
            if x.mode == y.mode && x.dag == y.dag
                done = true; c = zero(T)
                break
            end
            if !_inorder(x, y)
                # x y = -y x + {x, y}
                w2 = copy(w); w2[i] = y; w2[i + 1] = x
                push!(stack, (w2, -c))
                if x.mode == y.mode # {c, c†} = 1
                    push!(stack, (vcat(w[1:(i - 1)], w[(i + 2):end]), c))
                end
                done = false
                break
            end
        end
        done && !iszero(c) && _add!(out, w, c)
    end
    return OpExpr{T}(out)
end

function Base.show(io::IO, a::OpExpr)
    b = normalorder(a)
    if isempty(b.terms)
        print(io, "0")
        return
    end
    ws = sort!(collect(keys(b.terms)); by = w -> (length(w), map(_lkey, w)))
    first = true
    for w in ws
        c = b.terms[w]
        first || print(io, " + ")
        first = false
        print(io, "(", c, ")")
        if isempty(w)
            print(io, "·1")
        else
            for l in w
                print(io, " ", l)
            end
        end
    end
end

"""
    wordshift(w, charges) -> Sector

Symmetry shift of a word: `c†_m` contributes the charge `Q_m` of the mode, `c_m`
its dual (Sec. 5.2.3).
"""
function wordshift(w::Word, charges::AbstractDict{Mode,S}) where {S<:Sector}
    q = neutral(S)
    for l in w
        Q = get(charges, l.mode, nothing)
        Q === nothing && throw(ArgumentError("no charge assigned to mode $(l.mode)"))
        q = fuse(q, l.dag ? Q : dual(Q))
    end
    return q
end

"""
    components(a, charges) -> Dict{Sector, OpExpr}

Decompose `a` into homogeneous components, each with a definite shift, by
collecting the normal-ordered words by their shift. A homogeneous operator has
a single component (Sec. 5.2.3).
"""
function components(a::OpExpr{T}, charges::AbstractDict{Mode,S}) where {T,S<:Sector}
    out = Dict{S,OpExpr{T}}()
    for (w, c) in normalorder(a).terms
        q = wordshift(w, charges)
        e = get!(() -> OpExpr{T}(), out, q)
        _add!(e.terms, w, c)
    end
    return out
end

"""
    opshift(a, charges) -> Sector

The shift of a homogeneous operator; an `ArgumentError` names the components if
`a` is inhomogeneous.
"""
function opshift(a::OpExpr, charges::AbstractDict{Mode,S}) where {S<:Sector}
    cs = components(a, charges)
    isempty(cs) && return neutral(S)
    length(cs) == 1 || throw(ArgumentError(
        "operator is inhomogeneous: components with shifts $(collect(keys(cs))); use `components` or `psf_multilinear`"))
    return first(keys(cs))
end
