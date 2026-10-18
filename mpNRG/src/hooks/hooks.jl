# Per-shell callback interface (task sheet Sec. 5.10).
#
# A future in-sweep compression of the far-field KKD/KDK terms needs the
# per-shell internals of a PSF call; this file exposes them read-only. No
# compression is implemented.

"""
    CarriedComposite

One sliced composite operator being carried at shell `n`:

  - `born`: the shell where it was created (its sliced frequencies have the
    resolution `Λ^{-born/2}`); `depth`: nesting level (1 for the composites of
    3p KKD/KDK and 4p KKDD/KDKD/KDDK/KKKD/KKDK/KDKK terms, 2 for the composites
    of composites inside the 4p → 3p → 2p recursion);
  - `label`: the sector string that created it; `position`: its slot in the
    reduced PSF;
  - `tags`: for each original frequency εᵢ, whether it is sliced in this
    operator (`1` measured from state 1, `2` from the operator's left
    endpoint, `0` not sliced here);
  - `slices`: `Dict(slice key => BlockOp)`, the open bin legs. Each `BlockOp`
    lives on the full space of shell `n` (rows and columns per symmetry block,
    K states first, then D states), with the composite's symmetry shift; the
    slice key is a tuple over the original frequencies with bin indices (or
    exact energies in exact mode) in the sliced slots.
"""
struct CarriedComposite
    born::Int
    depth::Int
    label::String
    position::Int
    tags::Tuple
    slices::AbstractDict
end

"""
    ShellContext

Read-only view of the internals at shell `n` of a PSF forward sweep, passed to
every hook as `callback(run, n, ctx)`:

  - `isometries(ctx)`: `Dict(sector => A)` of shell `n` (Lee Eq. (15); rows in
    the product basis of the shell's fusion map, `fusionmap(ctx)`);
  - `rhoKK(ctx)`: `Dict(sector => ρ̄^n_{KK})` (Lee Eq. (45));
  - `call_operators(ctx)`: the sector-projected operators `O^n` of the call
    (top-level local operators, one `BlockOp` each);
  - `refinement_map(ctx)`: the function `O -> O^n` mapping a KK operator of
    shell `n-1` into all sectors of shell `n` (Lee Eq. (34));
  - `carried(ctx)`: the [`CarriedComposite`](@ref)s being carried at shell `n`.

Memory layout: every operator is a `BlockOp` keyed by column sector; within a
sector the states of the shell are ordered K first (`krange`) then D
(`drange`). Nothing may be mutated.
"""
struct ShellContext
    n::Int
    run::Any
    fdm::Any
    tasks::Vector{Any}
end

isometries(c::ShellContext) = shell(c.run, c.n).A
fusionmap(c::ShellContext) = shell(c.run, c.n).fusion
rhoKK(c::ShellContext) = rhoK(c.fdm, c.n)
refinement_map(c::ShellContext) = O -> refine(c.run, O, c.n)
function call_operators(c::ShellContext)
    out = BlockOp[]
    for t in c.tasks
        t.depth == 0 || continue
        for op in t.ops
            append!(out, values(op.slices))
        end
    end
    return out
end
function carried(c::ShellContext)
    out = CarriedComposite[]
    for t in c.tasks
        t.depth ≥ 1 || continue
        t.n == c.n || continue
        for (i, op) in enumerate(t.ops)
            push!(out, CarriedComposite(op.born, t.depth, t.label, i, op.tags, op.slices))
        end
    end
    return out
end

"""
    on_shell(callback, run, n, ctx)

Invoke a hook. The PSF engine calls this once per shell of the forward sweep
for every hook passed to [`psf`](@ref) (`hooks = [callback, …]`); with
`schedule = :breadth` the context lists every composite carried at shell `n`
(all creation shells at once), with `:depth` the composites of the task
currently being swept.
"""
on_shell(callback, run, n::Integer, ctx::ShellContext) = callback(run, n, ctx)

function _call_hooks(ctx, n, tasks)
    isempty(ctx.hooks) && return nothing
    sc = ShellContext(n, ctx.run, ctx.fdm, collect(Any, tasks))
    for h in ctx.hooks
        on_shell(h, ctx.run, n, sc)
    end
    return nothing
end

"""
    SingularValueRecorder()

Example hook: records, per shell, the singular values of every carried
composite slice (block by block). It demonstrates the interface only; no
compression is performed. `rec.data[n]` is a vector of
`(born, depth, label, position, slicekey, singular values)` tuples.
"""
struct SingularValueRecorder
    data::Dict{Int,Vector{Any}}
end
SingularValueRecorder() = SingularValueRecorder(Dict{Int,Vector{Any}}())

function (rec::SingularValueRecorder)(run, n, ctx::ShellContext)
    v = get!(() -> Any[], rec.data, n)
    for c in carried(ctx)
        for (k, B) in c.slices
            svals = reduce(vcat, [s.S for s in values(blocksvd(B))]; init = Float64[])
            push!(v, (c.born, c.depth, c.label, c.position, k, sort!(svals; rev = true)))
        end
    end
    return nothing
end
