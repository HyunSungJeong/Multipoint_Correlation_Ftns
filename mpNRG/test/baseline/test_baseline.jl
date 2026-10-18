# Regression test of the TNOps refactor (TNOPS_REFACTOR.md, Sec. 8): recompute
# every baseline quantity with the current code and compare with the frozen
# baseline written at the tag `pre-tensor-refactor` by make_baseline.jl.
# Run on its own or from runtests.jl:
#
#   julia -t 4 --project=mpNRG/test mpNRG/test/baseline/test_baseline.jl

using Test, JLD2
include(joinpath(@__DIR__, "cases.jl"))
include(joinpath(@__DIR__, "compare.jl"))

@testset "refactor regression against the pre-tensor-refactor baseline" begin
    old = load(joinpath(@__DIR__, "baseline.jld2"), "baseline")
    new = compute_baseline(; verbose = false)
    c = compare_baseline(new, old)
    print_comparison(c)
    @test c.kept_identical
    @test isempty(c.flips)
    for r in values(c.rows)
        @test r.pass
    end
end
