# vertex_krylov V3c (task sheet Sec. 6, Step 2): truncated NRG on the physical chain.
#
#   julia -t 4 --project=mpNRG/test mpNRG/scripts/vertex_krylov_v3c.jl
#
# Builds the DiscreteData of every V3c case with the adapter (vk_v3c_data: siam(U = 0.5,
# εd = -0.1, Δ = 0.1), Λ = 4, z = 1, N = chain_length_for(T; Λ = 4), LogGrid(emin = 1e-9,
# emax = 10, ndec = 16)) and caches it in notebooks/data/vertex_krylov_v3c/ (one JLD2 file
# per case; existing files are reused, so the script resumes). Then runs Algorithm 2 and all
# checks (vk_case_v3c; report only) and prints a one-line summary per case. The notebook
# (Sec. 6) loads the same cached data and reruns the algorithm and the checks itself.

using MultipointNRG, LinearAlgebra, Printf, JLD2
isdefined(@__MODULE__, :ROOT) || (const ROOT = normpath(joinpath(@__DIR__, "..", "..")))
isdefined(@__MODULE__, :model_ed) || include(joinpath(ROOT, "mpNRG", "test", "helpers.jl"))
isdefined(@__MODULE__, :vk_v3c_data) || include(joinpath(ROOT, "mpNRG", "test", "vertex_krylov_adapter.jl"))

const V3C_DIR = joinpath(ROOT, "notebooks", "data", "vertex_krylov_v3c")
const V3C_TS = (1e-2, 1e-4)
const V3C_NKEEPS = (64, 128, 256, 512)
v3c_file(T, Nk, chan) = joinpath(V3C_DIR, @sprintf("T=%.0e_Nkeep=%d_%s.jld2", T, Nk, chan))

function v3c_build_all()
    mkpath(V3C_DIR)
    for T in V3C_TS, Nk in V3C_NKEEPS, chan in (:updn, :upup)
        file = v3c_file(T, Nk, chan)
        isfile(file) && continue
        t0 = time()
        d = vk_v3c_data(T, Nk, chan)
        jldsave(file; data = vk_strip(d))
        @printf("built T=%.0e Nkeep=%3d %-4s: N=%d, %.1f s (NRG %.1f s, PSFs %.1f s), clamped %d\n", T, Nk, chan,
                vk_v3c_length(T), time() - t0, d.meta[:time_nrg], d.meta[:time_data], d.meta[:nclamped])
        flush(stdout)
        GC.gc()
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    v3c_build_all()
    for T in V3C_TS, Nk in V3C_NKEEPS, chan in (:updn, :upup)
        d = load(v3c_file(T, Nk, chan), "data")
        t0 = time()
        o = vk_case_v3c(d)
        c = o[:checks]
        @printf("T=%.0e Nkeep=%3d %-4s: R=%4d L=%4d L_B=%4d C1 %.1e C2 %.1e C3 %.1e C4 %.1e C5 %.1e C6 %.1e C7 %.1e C8 %.1e | BigFloat L %d n_agree %d | %.1f s\n",
                T, Nk, chan, c[:R], c[:L], c[:LB], c[:C1], c[:C2], c[:C3], c[:C4], c[:C5], c[:C6], c[:C7], c[:C8],
                o[:big].L, o[:big].n_agree, time() - t0)
        flush(stdout)
    end
end
