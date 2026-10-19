using Test, TNOps, LinearAlgebra, Random

const FILES = [
    "test_labels_blockop.jl",
    "test_tensor.jl",
    "test_fusion_factorize.jl",
]

@testset "TNOps" begin
    for f in FILES
        @testset "$f" begin
            include(joinpath(@__DIR__, f))
        end
    end
end
