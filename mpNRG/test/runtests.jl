using Test, MultipointNRG, LinearAlgebra, Random

const TESTDIR = @__DIR__

# One file per validation item of the task sheet, Sec. 6.
const FILES = [
]

@testset "MultipointNRG" begin
    for f in FILES
        @testset "$f" begin
            include(joinpath(TESTDIR, f))
        end
    end
end
