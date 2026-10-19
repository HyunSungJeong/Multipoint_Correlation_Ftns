using Test, MultipointNRG, LinearAlgebra, Random

const TESTDIR = @__DIR__
include(joinpath(TESTDIR, "helpers.jl"))

# One file per validation item of the task sheet, Sec. 6.
const FILES = [
    # P1
    "test_v1_symmetry.jl",
    # P2
    "test_v2_chain.jl",
    # P3
    "test_v3_spectra.jl",
    # P4
    "test_v1_chain.jl", "test_v4_operators.jl",
    # P5
    "test_v3_thermal.jl", "test_v4_fdm.jl", "test_v7_thermo.jl",
    # P6-P8
    "test_v3_psf.jl", "test_v4_psf.jl", "test_v4e_dense.jl", "test_symmetry.jl", "test_v5_rlm.jl",
    "test_v6_consistency.jl", "test_v7_spectral.jl", "test_hubbardatom.jl", "test_io.jl", "test_v8_external.jl",
    # P9
    "test_hooks_models.jl",
    # TNOps refactor: regression against the pre-tensor-refactor baseline
    "baseline/test_baseline.jl",
]

@testset "MultipointNRG" begin
    for f in FILES
        @testset "$f" begin
            include(joinpath(TESTDIR, f))
        end
    end
end
