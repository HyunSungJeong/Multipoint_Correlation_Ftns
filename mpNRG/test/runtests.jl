using Test, MultipointNRG, LinearAlgebra, Random

const TESTDIR = @__DIR__
include(joinpath(TESTDIR, "helpers.jl"))

# One file per validation item of the task sheet, Sec. 6.
const FILES = [
    # P1
    "test_v1_symmetry.jl",
]

@testset "MultipointNRG" begin
    for f in FILES
        @testset "$f" begin
            include(joinpath(TESTDIR, f))
        end
    end
end
