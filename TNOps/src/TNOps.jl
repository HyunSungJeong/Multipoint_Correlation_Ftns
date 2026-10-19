"""
    TNOps

Basic tensor-network operations on abelian-symmetric block-sparse tensors:
symmetry labels (`U1`, `Zn`, the fermion parity `Z2(:Pf)` always appended as
the last factor), tagged legs with arrows, block tensors of any rank with a
flux, sliced tensors, block-wise linear algebra, fusion of legs,
factorizations and a global degeneracy-safe truncation. No NRG, Wilson-chain,
PSF or model code. Conventions: README.md.
"""
module TNOps

using LinearAlgebra
using Printf

include("symmetry.jl")
include("legs.jl")
include("tensor.jl")
include("sliced.jl")
include("linalg.jl")
include("dense.jl")
include("legacy.jl")
include("fusion.jl")

export SymFactor, U1, Zn, Z2, Sector, SymmetryGroup, sector, labeltype, factor_index, factornames,
       fuse, dual, neutral, isneutral, parity, degeneracy
# legs and block tensors
export Arrow, In, Out, arrow, LegSector, Leg, legsectors, sectors, dim, offsets, charge, samespace, isdual,
       BlockTensor, getblock, setblock!, hasblock, blockkeys, eachblock, nblocks, allowed_keys, target,
       randtensor, identity_tensor, subtensor, restrict, convert_eltype,
       SlicedTensor, setslice!, nslices,
       permute, contract, outer, todense, fromdense
# legacy rank-2 layer (until MultipointNRG is ported)
export SectorSpace, BlockOp, blockeigen, blocksvd, identity_op, SlicedBlockOp, mapslices_op,
       FusionPair, FusionMap, fusionmap, kron_op

end # module
