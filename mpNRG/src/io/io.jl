# Output files and z-averaging (task sheet Sec. 5.8).

"""
    git_commit(dir = @__DIR__) -> String

The git commit of the source tree (or `"unknown"`).
"""
function git_commit(dir::AbstractString = @__DIR__)
    try
        return strip(read(Cmd(`git rev-parse HEAD`; dir = dir), String))
    catch
        return "unknown"
    end
end

package_version() = string(pkgversion(@__MODULE__))

"""
    run_metadata(run, f) -> Dict

All model and NRG parameters as plain data: model name and parameters,
symmetry group, local Hamiltonian (symbolic), chain coefficients,
discretization (Λ, z, scheme, intervals), N, Nkeep, Etrunc, gap tolerance,
temperature, n₀, git commit and package version.
"""
function run_metadata(run::NRGRun, f::Union{Nothing,FDM} = nothing)
    p = run.params
    d = Dict{String,Any}(
        "model" => run.model.name,
        "model_params" => Dict(string(k) => (v isa Number || v isa AbstractString ? v : string(v)) for (k, v) in run.model.params),
        "symmetry" => string(run.model.G),
        "Himp" => string(run.model.Himp),
        "Hsite0" => string(run.model.Hsite0),
        "channels" => [string(c.name) for c in run.model.channels],
        "chain_eps" => [c.eps for c in run.chains],
        "chain_t" => [c.t for c in run.chains],
        "Lambda" => p.Λ, "z" => p.z, "scheme" => string(p.scheme),
        "nint" => [c.nint for c in run.chains],
        "N" => p.N, "Nkeep" => p.Nkeep, "Etrunc" => p.Etrunc === nothing ? NaN : p.Etrunc,
        "gaptol" => p.gaptol, "n0" => run.n0,
        "kept_per_shell" => [totalkept(sh) for sh in run.shells],
        "git_commit" => git_commit(), "package_version" => package_version(),
    )
    f === nothing || (d["T"] = f.temp; d["fdm_checks"] = Dict(string(k) => v for (k, v) in pairs(f.info[:checks])))
    return d
end

"""
    psf_dict(r::PSFResult) -> Dict

A PSF as plain arrays (for other languages): operator names and shifts,
`ℓ`, grid parameters and bin centres, and either the dense histogram or the
nonzero bins (`bins` as an `nnz × (ℓ-1)` index matrix, `values`), the
`|weight|` histogram if present, the total weight, the FDM sum-rule reference
and the cost record; exact mode stores `positions`, `weights`, `labels`.
"""
function psf_dict(r::PSFResult)
    d = Dict{String,Any}("ops" => r.ops, "shifts" => [string(s) for s in r.shifts], "l" => r.ℓ,
                         "mode" => string(r.mode), "total" => r.total,
                         "sum_rule" => get(r.meta, :sum_rule, NaN), "pruned" => get(r.meta, :pruned, false))
    haskey(r.meta, :cost) && (d["cost"] = Dict(string(k) => v for (k, v) in r.meta[:cost]))
    if r.mode === :binned
        g = r.grid
        d["grid"] = Dict("emin" => g.emin, "emax" => g.emax, "ndec" => g.ndec, "ztol" => g.ztol, "centres" => centres(g))
        if r.data isa AbstractDict
            ks = collect(keys(r.data))
            d["bins"] = isempty(ks) ? zeros(Int, 0, r.ℓ - 1) : reduce(vcat, [collect(k)' for k in ks])
            d["values"] = [r.data[k] for k in ks]
            if r.absdata !== nothing
                d["abs_values"] = [get(r.absdata, k, 0.0) for k in ks]
            end
        else
            d["histogram"] = r.data
            r.absdata === nothing || (d["abs_histogram"] = r.absdata)
        end
    else
        d["positions"] = r.positions; d["weights"] = r.weights
        d["labels"] = [collect(l) for l in r.labels]
    end
    return d
end

"""
    save_results(path, run, f, psfs; thermal = Dict())

Write a JLD2 file with the run metadata ([`run_metadata`](@ref)), every PSF
([`psf_dict`](@ref), keyed by its permutation or name) and the thermal averages
used as sum-rule references.
"""
function save_results(path::AbstractString, run::NRGRun, f::FDM, psfs::AbstractDict; thermal::AbstractDict = Dict())
    jldopen(path, "w") do io
        io["metadata"] = run_metadata(run, f)
        io["psfs"] = Dict(string(k) => psf_dict(v) for (k, v) in psfs)
        io["thermal_averages"] = Dict(string(k) => v for (k, v) in thermal)
    end
    return path
end

"""    load_results(path) -> Dict — read a file written by `save_results`."""
load_results(path::AbstractString) = jldopen(io -> Dict(k => io[k] for k in keys(io)), path, "r")

"""
    zaverage(results) -> PSFResult

Average binned PSFs bin by bin over runs with different discretization
shifts `z` (Lee p. 23: `z = 1/n_z, …, 1`). Refuses results with different
grids, operators or `ℓ`.
"""
function zaverage(rs::AbstractVector{<:PSFResult})
    isempty(rs) && throw(ArgumentError("nothing to average"))
    a = rs[1]
    for r in rs
        r.mode === :binned || throw(ArgumentError("z-averaging needs binned PSFs"))
        r.grid == a.grid || throw(ArgumentError("cannot z-average PSFs on different grids"))
        (r.ops == a.ops && r.ℓ == a.ℓ) || throw(ArgumentError("cannot z-average PSFs of different operators"))
    end
    s = reduce(add_psf, rs)
    nz = length(rs)
    data = s.data isa AbstractDict ? Dict(k => v / nz for (k, v) in s.data) : s.data ./ nz
    meta = Dict{Symbol,Any}(:zaveraged => nz, :T => a.meta[:T],
                            :sum_rule => sum(r.meta[:sum_rule] for r in rs) / nz)
    return PSFResult{eltype(s.weights),a.ℓ - 1}(a.ops, a.shifts, a.ℓ, :binned, a.grid, data, nothing,
                                                 zeros(a.ℓ - 1, 0), eltype(s.weights)[], NTuple{3,Int}[], s.total / nz, meta)
end

"""
    zaveraged_psfs(model, params, ops; T, grid, nz, kwargs...) -> Dict{Vector{Int},PSFResult}

Run NRG and FDM for `z = 1/n_z, …, 1` and return the z-averaged PSFs of all
requested permutations (keyword arguments as for [`psfs`](@ref)).
"""
function zaveraged_psfs(model::ImpurityModel, p::NRGParams, ops::Tuple; T::Real, nz::Integer, kwargs...)
    per = Dict{Vector{Int},Vector{PSFResult}}()
    for k in 1:nz
        z = k / nz
        pk = NRGParams(N = p.N, Nkeep = p.Nkeep, Etrunc = p.Etrunc, gaptol = p.gaptol, Λ = p.Λ, z = z,
                       scheme = p.scheme, dense_max = p.dense_max)
        run = run_nrg(model, pk)
        f = fdm(run; T = T)
        for (perm, r) in psfs(run, f, ops; kwargs...)
            push!(get!(() -> PSFResult[], per, perm), r)
        end
    end
    return Dict(perm => zaverage(rs) for (perm, rs) in per)
end
