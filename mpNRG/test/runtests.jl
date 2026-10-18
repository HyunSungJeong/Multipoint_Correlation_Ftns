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
]

@testset "MultipointNRG" begin
    for f in FILES
        @testset "$f" begin
            include(joinpath(TESTDIR, f))
        end
    end
end
