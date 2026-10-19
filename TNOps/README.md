# TNOps

Basic tensor-network operations on **abelian-symmetric block-sparse tensors**
in Julia: symmetry labels, tagged legs with arrows, block tensors of any rank
with a flux, sliced tensors, block-wise linear algebra and contractions, fusion
of legs, factorizations on a leg partition, a global degeneracy-safe
truncation, fermionic parity, and dense embeddings for testing.

`TNOps` contains no NRG, Wilson-chain, PSF or model code. It is the tensor
layer of [`MultipointNRG`](../mpNRG/README.md), which was refactored onto it
(task sheet [`mpNRG/docs/TNOPS_REFACTOR.md`](../mpNRG/docs/TNOPS_REFACTOR.md)).
Dependencies: `LinearAlgebra` and `Printf` (tests also use `Random`). Dense
kernels are BLAS/LAPACK on blocks; no external tensor library is used.

The verification record is [`STATUS.md`](STATUS.md); the executed checks
notebook is [`notebooks/tnops_checks.ipynb`](notebooks/tnops_checks.ipynb)
(PDF alongside). Tests: `julia --project=TNOps/test TNOps/test/runtests.jl`.

## Conventions

**Labels.** A symmetry is a product of `U1` and `Zn` factors; the fermion
parity `Z2(:Pf)` is always appended as the last factor. A label `Sector{M,K}`
is an `isbits` tuple in canonical form (Z_n components reduced mod n);
`fuse` adds componentwise, `dual` negates, `parity(a)` is the last component,
and `degeneracy(a) = 1` (abelian).

**Legs.** A `Leg` is a *physical* sector space — sectors `LegSector(label, tag)`
sorted by `(label, tag)`, each with a dimension — plus an arrow (`Out` or
`In`) and an optional name. The tag distinguishes blocks that share a label
(tag `0` = untagged); MultipointNRG uses tag 1 for the kept and tag 2 for the
discarded states of a shell. A sector carries the **charge** `label` on an
`Out` leg and `dual(label)` on an `In` leg. `dual(leg)` flips the arrow, which
dualizes the charges while the physical space stays the same. **A leg
contracts only with its dual** (same sectors and dimensions, opposite arrows);
mismatches raise an error naming the legs. (In IJulia, `In` and `Out` are the
notebook's cell history in `Main`; write `TNOps.In` and `TNOps.Out` there.)

**Block rule (the one symmetry statement).** A block of a tensor with flux Δ
may exist only if the charges of its sectors fuse to Δ, i.e.

    ⊕ labels of the Out legs  =  ⊕ labels of the In legs  ⊕  Δ .

Absent blocks are zero and are not stored; `setblock!` refuses forbidden
blocks and `contract` never computes one (`keyflux`, `allowed_keys`).

**Rank 2 = operators.** With legs (`Out` row, `In` column) a rank-2 tensor is
the block operator of the pre-refactor layer: the flux is the operator's shift,
`target(A, q) = q ⊕ Δ` is the row label reached from column label `q`, and
`getblock(A, q)` / `setblock!(A, q, M)` are keyed by the column label (with
all tags of a label stacked in ascending tag order). `BlockTensor{T}(row,
col, Δ)` builds such an operator from two legs. `A * B` is
`contract(A, (2,), B, (1,))`; `A'` swaps the legs (each dualized) and
conjugate-transposes the blocks.

**Leg order and index maps.** Blocks are dense `Array{T,N}` in the leg order
of the tensor. `todense` stacks the sectors of each leg in sorted `(label,
tag)` order: sector `s` of leg `l` occupies the dense indices
`offsets(l)[s] .+ (1:dim(l, s))`. `fromdense` is the inverse and errors on
weight outside the allowed blocks. `permute` is a pure leg permutation and
**bosonic**: it never inserts a fermionic sign.

**Fermionic signs** enter only through `parity_tensor(l)` and
`apply_parity(A, i; power)`, which multiplies every block by
`(-1)^(p · power)` with `p` the parity of the block's sector on leg `i`.

**Fusion.** `fusionmap(outer, inner)` fuses two legs with the same arrow. The
fused sector `q′ = σ ⊕ q` collects the pairs `(σ, q)` with outer sectors in
sorted order, then inner sectors in sorted order; pair `(σ, q)` occupies the
fused indices `offset .+ (1:dσ*dq)`, and the product state `(iσ, s)` has index
`offset + (iσ-1)*dq + s` (`kron(e_iσ, e_s)`; column-major over (inner,
outer)). `fuse_legs(A, (a, b))` fuses leg `a` (inner, fast) and leg `b`
(outer, slow) into one leg at position `min(a, b)`; `split_legs` is its
inverse. `kron_fused(X, Y, Frow, Fcol)` is `X ⊗ Y` on fused legs (`X` on the
outer, `Y` on the inner leg).

**Sliced tensors.** `SlicedTensor{Kt,T,N,S}` maps slice keys of any type
(tuples of bin indices, `Float64` energies, …) to block tensors with common
legs and flux; `mapslices(f, st)` applies a contraction to every slice. Slice
keys are not tensor legs.

**Factorizations** read a tensor as a map from its column legs to its row legs
(`eigen(A, rows, cols)`, `svd`, `qr`; rank 2 needs no partition). Multiple legs
on a side are fused and split back; the tensor must be in operator form (row
legs `Out`, column legs `In`). `eigen` needs a hermitian tensor with
neutral flux, symmetrizes each label block as `(M + M')/2` and runs the label
blocks in parallel; the results carry a new bond leg (`In`, untagged, one
sector per row label) and sector-tagged spectra.

**Global cut.** `global_cut(spectra; nkeep, threshold, gaptol)` sorts all
values of all sectors together (each counted `degeneracy` times), places the
cut at `nkeep` or at the threshold (whichever keeps fewer), and moves it up
into the nearest gap larger than `gaptol`, so that no degenerate multiplet is
split even across sectors. It returns the kept count per sector and the
record `(requested, actual, gap, cut_moved, total)`.

**Non-abelian interface.** `interface.jl` lists the only functions code above
`TNOps` may use for anything symmetry-dependent: `fuse`, `dual`, `neutral`,
`isneutral`, `parity`, `degeneracy` and the block rule. Every state count goes
through `degeneracy`. Non-abelian symmetries are not implemented.

## Example

```julia
using TNOps, LinearAlgebra, Random

G = SymmetryGroup(U1(:N), U1(:twoSz))             # factors (N, twoSz, Pf)
q(n, s) = sector(G; N = n, twoSz = s)             # Pf follows from N
site = Leg(Dict(q(0, 0) => 1, q(1, 1) => 1, q(1, -1) => 1, q(2, 0) => 1); name = :site)
bath = Leg(Dict(q(0, 0) => 2, q(1, 1) => 3, q(1, -1) => 3); name = :bath)

# fuse two legs and build a neutral hermitian operator on the product
F = fusionmap(site, bath)                          # outer = site, inner = bath
rng = MersenneTwister(1)
X = randtensor(rng, Float64, (F.out, dual(F.out)), neutral(labeltype(G)))
H = X + X'

# diagonalize per label, keep 6 states globally without splitting multiplets
vals, U = eigen(H)
kept, rec = global_cut(vals; nkeep = 6, gaptol = 1e-10)

# split the product leg of U back into (bath, site): a rank-3 isometry
A = split_legs(U, 1, F)                            # legs (bath, site, bond)
contract(A, (1, 2), adjoint(A, (1, 2), (3,)), (2, 3))   # = 1 on the bond
```

## Layout

```
src/TNOps.jl      module, exports
src/symmetry.jl   U1, Zn, Z2, Sector, SymmetryGroup, fuse, dual, neutral, parity, degeneracy
src/legs.jl       Leg, LegSector, arrows, dual(leg), charges, retag, subleg
src/tensor.jl     BlockTensor: storage, block rule, construction, block access, rank-2 layout
src/sliced.jl     SlicedTensor, mapslices
src/linalg.jl     arithmetic, adjoint, norm, dot, permute, contract, tr, outer
src/dense.jl      todense / fromdense (index map)
src/fusion.jl     FusionMap, fusionmap, kron_fused, fuse_legs, split_legs
src/factorize.jl  eigen, svd, qr on a leg partition; global_cut
src/parity.jl     parity_tensor, apply_parity
src/ncon.jl       ncon with a caller-given contraction order
src/interface.jl  the symmetry interface a non-abelian layer must supply
test/             tests against dense arrays and against frozen output of the old layer
notebooks/        tnops_checks.ipynb (executed) and its PDF
```
