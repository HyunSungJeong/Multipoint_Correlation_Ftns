# First-call and warm timings for the heat-map production runs, in a fresh Julia
# process, so the first call really includes compilation (inside the notebook,
# the test cell has already compiled everything).
#
#   julia --project scripts/heatmap_timing.jl

using Printf
t_load = @elapsed begin
    include(joinpath(@__DIR__, "..", "src", "HubbardAtom.jl"))
    using .HubbardAtom
end
using BenchmarkTools

U = 1.0; T = U / 50
prm = HeatmapParams(U, 1 / T, T)
m = HubbardAtomModel(U, 1 / T)
t_tab = @elapsed tabs = Dict((s, f) => heatmap_tables(m, s, f) for s in (:updn, :upup), f in (:MF, :KF))

ms = [2n + 1 for n in -24:23]
νs = collect(range(-1, 1; length = 401))
t_mf1 = @elapsed vertex_grid(ms, ms, 0, prm, tabs[(:updn, :MF)])
t_mf2 = @belapsed vertex_grid($ms, $ms, 0, $prm, $(tabs[(:updn, :MF)]))
t_kf1 = @elapsed vertex_grid(νs, νs, 0.0, prm, tabs[(:updn, :KF)])
t_kf2 = @belapsed vertex_grid($νs, $νs, 0.0, $prm, $(tabs[(:updn, :KF)])) samples = 3 evals = 1

t_mk = @elapsed @eval using CairoMakie
F = randn(401, 401)
tmp = tempname() * ".png"
t_pl1 = @elapsed save(tmp, heatmap(νs, νs, F))
t_pl2 = @elapsed save(tmp, heatmap(νs, νs, F))

@printf("load src: %.2f s   build PSF tables (both spins, MF + KF): %.2f s   threads: %d\n", t_load, t_tab, Threads.nthreads())
@printf("P1  MF grid  48 × 48, one spin:              first call %8.3f s   warm %8.4f s\n", t_mf1, t_mf2)
@printf("P2  KF grid 401 × 401 × 16, one spin:        first call %8.3f s   warm %8.3f s\n", t_kf1, t_kf2)
@printf("    using CairoMakie: %.1f s   first heatmap + save: %.1f s   second: %.2f s\n", t_mk, t_pl1, t_pl2)
