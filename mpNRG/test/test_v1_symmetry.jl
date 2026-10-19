# V1 (local parts). The generic symmetry-layer tests (V1a labels, V1b block
# algebra) moved to TNOps/test/test_labels_blockop.jl in the TNOps refactor.

using Test, MultipointNRG, LinearAlgebra, Random

@testset "V1c local operator algebra, shifts and parity of composites" begin
    G = SymmetryGroup(U1(:N), U1(:twoSz))                         # = spin_symmetry(:charge_spin)
    Q = σ -> sector(G; N = 1, twoSz = σ == :up ? 1 : -1)
    S = labeltype(G)
    ch = Dict{Mode,S}(Mode(-1, :up) => Q(:up), Mode(-1, :dn) => Q(:dn), Mode(0, :up) => Q(:up), Mode(0, :dn) => Q(:dn))
    du = cop(Mode(-1, :up)); dd = cop(Mode(-1, :dn))
    U = 0.7
    Hint = U * num(Mode(-1, :up)) * num(Mode(-1, :dn))
    q_up = comm(du, Hint); q_dn = comm(dd, Hint)
    @test q_up == U * du * num(Mode(-1, :dn))                     # [d↑, U n↑ n↓] = U d↑ n↓
    @test opshift(q_up, ch) == opshift(du, ch)                    # same shift as d↑ (H_int neutral)
    b_same = acomm(q_up, du')
    b_opp = acomm(q_up, dd')
    @test isneutral(opshift(b_same, ch))                          # {q↑, d†↑} is neutral
    @test opshift(b_opp, ch) == sector(G; N = 0, twoSz = -2)      # {q↑, d†↓} carries 2S_z = -2
    @test parity(opshift(b_opp, ch)) == 0                         # bosonic
    @test parity(opshift(q_up, ch)) == 1
    x = cdag(Mode(-1, :dn)) * cop(Mode(0, :dn))                   # d†↓ f_{0↓}
    @test isneutral(opshift(x, ch)) && maxsite(x) == 0
    # inhomogeneous operator: components with their shifts
    cs = components(du + dd, ch)
    @test length(cs) == 2
    @test_throws ArgumentError opshift(du + dd, ch)
    # canonical anticommutators of the local modes
    for a in (du, dd), b in (du, dd)
        @test acomm(a, b') == (a == b ? identity_expr() : OpExpr{Float64}())
        @test iszero(acomm(a, b))
    end
    # dense evaluation of the expression algebra agrees with the independent JW
    B = FockBasis([Mode(-1, :up), Mode(-1, :dn)])
    c = jw_annihilators(2)
    @test densematrix(du, B) == c[1] && densematrix(dd, B) == c[2]
    @test densematrix(q_up, B) ≈ U * c[1] * (c[2]' * c[2])
end
