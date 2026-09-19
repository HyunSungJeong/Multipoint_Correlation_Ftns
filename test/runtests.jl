# Step 1 of Block 4: verification of the Hubbard-atom spectrum.
#
# Reference: Kugler, Lee & von Delft, Phys. Rev. X 11, 041006 (2021).
# Run with:  julia --project test/runtests.jl

using Test
using LinearAlgebra

include(joinpath(@__DIR__, "..", "src", "HubbardAtom.jl"))
using .HubbardAtom

const PARAMS = [(1.0, 1.0), (2.0, 5.0), (4.0, 0.3), (0.0, 2.0), (8.0, 10.0),
                (-2.0, 1.5)]

@testset "Hubbard atom, Step 1: spectrum" begin

    @testset "constructor guards" begin
        @test_throws ArgumentError HubbardAtomModel(1.0, 0.0)
        @test_throws ArgumentError HubbardAtomModel(1.0, -1.0)
        @test_throws ArgumentError HubbardAtomModel(Inf, 1.0)
        m = HubbardAtomModel(3, 2)          # integers are promoted
        @test m.U == 3.0 && m.β == 2.0
    end

    @testset "fermionic algebra" begin
        ops = operators(HubbardAtomModel(1.7, 0.9))
        I4 = Matrix(1.0I, DIM, DIM)
        Z4 = zeros(DIM, DIM)

        annihilators = (ops.d_up, ops.d_dn)
        creators = (ops.dag_up, ops.dag_dn)

        # {dσ, d†σ'} = δσσ' and {dσ, dσ'} = 0
        for (a, da) in enumerate(annihilators), (b, db) in enumerate(annihilators)
            anti_mixed = da * creators[b] + creators[b] * da
            @test anti_mixed ≈ (a == b ? I4 : Z4)
            @test da * db + db * da ≈ Z4
            @test creators[a] * creators[b] + creators[b] * creators[a] ≈ Z4
        end

        # nilpotency
        @test ops.d_up * ops.d_up ≈ Z4
        @test ops.d_dn * ops.d_dn ≈ Z4

        # the sign the Jordan-Wigner string exists to produce:
        #   d↓|↑↓⟩ = d↓ d†↑ d†↓|0⟩ = -d†↑ d↓ d†↓|0⟩ = -|↑⟩
        up_dn = [0.0, 0.0, 0.0, 1.0]
        up = [0.0, 1.0, 0.0, 0.0]
        @test ops.d_dn * up_dn ≈ -up
        # while the ↑ mode, having no string, carries no sign
        @test ops.d_up * up_dn ≈ [0.0, 0.0, 1.0, 0.0]

        # creation from the vacuum reproduces the documented basis order
        vac = [1.0, 0.0, 0.0, 0.0]
        @test ops.dag_up * vac ≈ up
        @test ops.dag_dn * vac ≈ [0.0, 0.0, 1.0, 0.0]
        @test ops.dag_up * (ops.dag_dn * vac) ≈ up_dn

        # occupations are projectors, diagonal in the Fock basis
        @test ops.n_up ≈ Diagonal([0.0, 1.0, 0.0, 1.0])
        @test ops.n_dn ≈ Diagonal([0.0, 0.0, 1.0, 1.0])
        @test ops.n_up * ops.n_up ≈ ops.n_up
        @test ops.n_dn * ops.n_dn ≈ ops.n_dn
    end

    @testset "Hamiltonian structure: U=$U, β=$β" for (U, β) in PARAMS
        m = HubbardAtomModel(U, β)
        ops = operators(m)

        @test ops.H ≈ permutedims(ops.H)                    # hermitian
        @test ops.H * ops.N ≈ ops.N * ops.H                 # [H, N] = 0
        @test ops.H * ops.Sz ≈ ops.Sz * ops.H               # [H, Sz] = 0
        @test level_position(m) ≈ -U / 2
        @test half_interaction(m) ≈ U / 2

        # H is diagonal in the Fock basis with the closed-form entries
        u = U / 2
        @test ops.H ≈ Diagonal([0.0, -u, -u, 0.0])
    end

    @testset "diagonalisation: U=$U, β=$β" for (U, β) in PARAMS
        m = HubbardAtomModel(U, β)
        ops = operators(m)
        sp = spectrum(m)

        # eigenvalues against the closed form
        @test sp.E ≈ eigenenergies_analytic(m)
        @test issorted(sp.E)

        # a genuine eigendecomposition, in an orthonormal basis
        @test permutedims(sp.V) * sp.V ≈ Matrix(1.0I, DIM, DIM)
        @test ops.H * sp.V ≈ sp.V * Diagonal(sp.E)
        @test to_eigenbasis(sp, ops.H) ≈ Diagonal(sp.E)

        # Z checked by hand: Z = 2(1 + e^{βu})
        @test sp.Z ≈ partition_function_analytic(m) rtol = 1e-12
        @test sp.Z ≈ sum(exp.(-β .* sp.E)) rtol = 1e-12

        # Boltzmann weights
        @test sum(sp.ρ) ≈ 1.0
        @test all(>(0.0), sp.ρ)
        @test sp.ρ ≈ exp.(-β .* sp.E) ./ sp.Z
        # weights are monotone in energy: lower energy, larger weight
        @test issorted(sp.ρ, rev = true)

        # eigenvector signs are pinned: first significant component positive
        for j in 1:DIM
            col = sp.V[:, j]
            idx = findfirst(x -> abs(x) > 1e-12, col)
            @test idx !== nothing
            @test col[idx] > 0
        end

        # degenerate blocks carry sharp quantum numbers after canonicalisation
        Nvals, Szvals = quantum_numbers(sp, ops)
        @test all(n -> isapprox(n, round(n); atol = 1e-10), Nvals)
        @test all(s -> isapprox(2s, round(2s); atol = 1e-10), Szvals)
        @test sort(Nvals) ≈ [0.0, 1.0, 1.0, 2.0]
        @test sort(Szvals) ≈ [-0.5, 0.0, 0.0, 0.5]
    end

    @testset "thermodynamics: U=$U, β=$β" for (U, β) in PARAMS
        m = HubbardAtomModel(U, β)
        ops = operators(m)
        sp = spectrum(m)

        # particle-hole symmetry at εd = -U/2 pins the atom to half filling
        @test thermal_average(sp, ops.n_up) ≈ 0.5
        @test thermal_average(sp, ops.n_dn) ≈ 0.5
        @test thermal_average(sp, ops.N) ≈ 1.0
        # paramagnetic
        @test thermal_average(sp, ops.Sz) ≈ 0.0 atol = 1e-12
        # ⟨1⟩ = tr ρ = 1
        @test thermal_average(sp, Matrix(1.0I, DIM, DIM)) ≈ 1.0
        # ⟨H⟩ = Σ ρᵢ Eᵢ
        @test thermal_average(sp, ops.H) ≈ sum(sp.ρ .* sp.E)
    end

    @testset "closed forms and limits" begin
        # U = 0: four degenerate states, Z = 4, uniform weights
        m0 = HubbardAtomModel(0.0, 2.0)
        sp0 = spectrum(m0)
        @test sp0.E ≈ zeros(DIM)
        @test sp0.Z ≈ 4.0
        @test sp0.ρ ≈ fill(0.25, DIM)

        # double occupancy: ⟨n↑n↓⟩ = 1/Z, suppressed as βU grows
        for (U, β) in PARAMS
            m = HubbardAtomModel(U, β)
            ops = operators(m)
            sp = spectrum(m)
            @test thermal_average(sp, ops.n_up * ops.n_dn) ≈ 1 / sp.Z
        end

        # the tanh(βu/2) of Kugler Eq. (85) is this spectrum's polarisation
        for (U, β) in PARAMS
            m = HubbardAtomModel(U, β)
            u = half_interaction(m)
            singly = 2 * exp(β * u) / partition_function_analytic(m)
            @test 2 * singly - 1 ≈ tanh(β * u / 2)
        end

        # large βU: the singly occupied doublet exhausts the weight
        mbig = HubbardAtomModel(20.0, 20.0)
        spbig = spectrum(mbig)
        @test isfinite(spbig.Z)                      # shifted form does not overflow
        @test sum(spbig.ρ[1:2]) ≈ 1.0 atol = 1e-8
    end

    @testset "eigenbasis is usable by Step 2" begin
        # Eq. (28) needs (O)_{i,i+1} in the energy eigenbasis; check the
        # rotation is consistent and that dσ connects adjacent charge sectors.
        m = HubbardAtomModel(2.0, 3.0)
        ops = operators(m)
        sp = spectrum(m)

        d_rot = to_eigenbasis(sp, ops.d_up)
        dag_rot = to_eigenbasis(sp, ops.dag_up)
        @test dag_rot ≈ permutedims(d_rot)

        # rotation preserves the algebra
        I4 = Matrix(1.0I, DIM, DIM)
        @test d_rot * dag_rot + dag_rot * d_rot ≈ I4

        # dσ lowers N by one: nonzero elements only between N and N-1 sectors
        Nvals, _ = quantum_numbers(sp, ops)
        for i in 1:DIM, j in 1:DIM
            if abs(d_rot[i, j]) > 1e-10
                @test Nvals[j] - Nvals[i] ≈ 1.0
            end
        end
    end
end
