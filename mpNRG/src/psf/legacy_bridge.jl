# Temporary (TNOps refactor R5c-R5e): the PSF engine still carries operators in
# the pre-refactor layout (BlockOp keyed by column label, kept states first).
# Projections and refinements are TNOps tensors from R5c on; this file converts
# at the two places where the engine obtains them (top-level projections and
# the refinement of carried operators). Deleted when psf/ is ported (R5e).

# the tensor as a label-keyed BlockOp on the full or the kept space of shell sh
function to_legacy(O::BlockTensor{T,2,S}, sh::Shell{S}) where {T,S}
    full = TNOps.samespace(TNOps.leg(O, 2), sh.leg)
    sp = full ? sh.space : keptspace(sh)
    B = BlockOp{T}(sp, sp, TNOps.flux(O))
    for q in unique(k[2].label for k in blockkeys(O))
        B.blocks[q] = getblock(O, q)
    end
    return B
end

# the BlockOp of shell sh (on its full or its kept space) as a tensor
function from_legacy(O::BlockOp{T,S}, sh::Shell{S}) where {T,S}
    O.col == sh.space && return shell_tensor(sh.leg, O.shift, O.blocks, T)
    return kk_tensor(kept_leg(sh.leg), O.shift, O.blocks, T)
end

refine(run::NRGRun, O::BlockOp, n::Integer) = to_legacy(refine(run, from_legacy(O, shell(run, n - 1)), n), shell(run, n))
