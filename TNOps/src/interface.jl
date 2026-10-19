# The symmetry interface.
#
# Code above TNOps (and every structural operation inside it) uses only the
# following functions for anything that depends on the symmetry; a future
# non-abelian layer must supply them (with multiplet labels, a fusion that
# returns several outcomes, and recoupling coefficients in contract, fuse_legs
# and the factorizations, which are not implemented here):
#
#   fuse(a, b)        label of a product state / shift of a product of operators
#   dual(a)           conjugate label (and dual(leg): flip the arrow)
#   neutral(S)        the trivial label; isneutral(a)
#   parity(a)         fermion parity component (0 or 1) of a label
#   degeneracy(a)     number of states represented by one block row of label a
#                     (1 for abelian labels); every state count goes through it
#   the block rule    keyflux(legs, key) == flux (charges of a block fuse to the
#                     flux), with allowed_keys(legs, flux) the enumeration of
#                     the blocks it permits
#
# Abelian labels make all of these elementwise and block sizes equal to
# multiplet counts; nothing else of the label structure may be used outside
# symmetry.jl.

"""
    SYMMETRY_INTERFACE

The functions code above `TNOps` may use for anything symmetry-dependent
(see `interface.jl`): `fuse`, `dual`, `neutral`, `isneutral`, `parity`,
`degeneracy`, and the block rule `keyflux` / `allowed_keys`.
"""
const SYMMETRY_INTERFACE = (:fuse, :dual, :neutral, :isneutral, :parity, :degeneracy, :keyflux, :allowed_keys)
