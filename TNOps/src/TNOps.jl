"""
    TNOps

Basic tensor-network operations on abelian-symmetric block-sparse tensors:
symmetry labels (`U1`, `Zn`, the fermion parity `Z2(:Pf)` always appended),
sector spaces, block-sparse operators and their fusion. No NRG, Wilson-chain,
PSF or model code.
"""
module TNOps

using LinearAlgebra
using Printf

include("symmetry.jl")
include("tensor.jl")
include("fusion.jl")

export SymFactor, U1, Zn, Z2, Sector, SymmetryGroup, sector, labeltype, factor_index, factornames,
       fuse, dual, neutral, isneutral, parity, degeneracy,
       SectorSpace, sectors, dim, offsets, BlockOp, getblock, setblock!, target, blockeigen, blocksvd,
       todense, identity_op, restrict, SlicedBlockOp, mapslices_op, nslices,
       FusionPair, FusionMap, fusionmap, kron_op

end # module
