# TNOps — status

Task sheet: [`../mpNRG/docs/TNOPS_REFACTOR.md`](../mpNRG/docs/TNOPS_REFACTOR.md).
`TNOps` is the abelian block-sparse tensor package that `MultipointNRG` is
refactored onto.

## Starting point and environment

- Refactor base: `bb8b1976c98027c7e229de493614add5444d6a05` (`bb8b197`) on
  `claude/practical-archimedes`, tagged locally as `pre-tensor-refactor`. The
  tag push was refused by the session's git proxy (it accepts branch pushes
  only), so the hash is recorded here and in `mpNRG/STATUS.md`.
- Julia 1.13.0 (conda-forge), packages from the General registry via git
  (`JULIA_PKG_SERVER=""`), 4 threads.

## Verification record

Tests: `julia -t 4 --project=TNOps/test TNOps/test/runtests.jl` — **2354
assertions, all passing** (R4: `retag` of a tensor leg added). Every operation is checked against the same
operation on `todense` arrays to 1e-13 relative, for random tensors of rank
1–5 with 2–5 sectors per leg (tagged and untagged), random fluxes, the
symmetries U1×U1×Z2, Z3 (×Pf) and parity only, real and complex elements;
the random seeds are fixed integers (an earlier version seeded with
`hash(name)` and its assertion count varied between runs). Achieved (notebook `notebooks/tnops_checks.ipynb`, largest deviation):

| Check | Achieved |
|:--|:--|
| `fromdense(todense(A)) == A`; stored forbidden blocks | 0; none (90 tensors) |
| `+`, `-`, scalar `*`, `conj`, `norm`, `dot` vs dense | 4.0e-16 |
| `permute` vs `permutedims` | 0 |
| `contract` (ranks 1–4, 0–2 pairs) vs explicit loops; forbidden output blocks | 3.1e-16; none (90 contractions) |
| `ncon` (three tensors, explicit order) vs explicit loops | 2.4e-16 |
| `tr` (rank 2, partial) vs dense | 1.0e-16 |
| `split_legs(fuse_legs(A)) - A` | 0 |
| `fusionmap` vs the frozen pre-refactor `fusionmap` (252 pairs, 6 shells) | identical (order and offsets) |
| `kron_fused` vs the frozen pre-refactor `kron_op` (54 blocks) | bit-identical |
| `eigen` (rank 2; rank-4 partition) reconstruction | 2.4e-15; 1.9e-15 |
| `svd`, `qr` (rank 2 with flux; rank-3 partition) reconstruction | ≤ 2.6e-15 |
| `global_cut` vs the frozen `truncate_shell` (20 spectra of the 4 baseline runs) | kept counts and records identical |
| `apply_parity`, `parity_tensor` vs the dense parity matrix | 0 |
| duality errors for mismatched legs | raised, naming the legs |

The frozen fixtures are `test/fixtures/old_layer_fixtures.jl`, written at R0 by
`mpNRG/test/baseline/make_baseline.jl` with the pre-refactor code as Julia
literals (TNOps has no JLD2).

## Design decisions and deviations

1. **Arrow convention.** A leg is a physical sector space plus an arrow; the
   charge of a sector is its label on an `Out` leg and the dual label on an
   `In` leg. `dual(leg)` flips the arrow (so the charges are dualized while
   the labels, tags and order stay). This makes "a leg contracts only with its
   dual" coincide with the matrix product of operators (`A.col` is `In`,
   `B.row` the same space `Out`) and keeps the sorted sector order of a
   `SectorSpace` for both legs of an operator.
2. **Fusion of two legs.** `FusionMap` fuses exactly two legs (outer, inner)
   with the pre-refactor layout; more legs are fused by repeated calls
   (`fuse_legs`, factorizations on a partition).
3. **Factorizations need operator form** (row legs `Out`, column legs `In`)
   and work per label, all tags of a label together: a neutral tensor couples
   sectors that share a label. (The first `eigen` worked per tagged sector and
   failed the reconstruction test for tagged legs with parity-only symmetry;
   fixed.) The bond legs are untagged.
4. **`In` and `Out` in IJulia** are the notebook's cell history in `Main` and
   shadow the exported arrows; notebooks write `TNOps.In` / `TNOps.Out`.
5. **Legacy layer.** Until MultipointNRG is ported (R5), the pre-refactor
   `SectorSpace`, `BlockOp`, `SlicedBlockOp`, `fusionmap(::SectorSpace, …)` and
   `kron_op` live unchanged in `src/legacy.jl`; the old `FusionMap` /
   `FusionPair` are renamed `OldFusionMap` / `OldFusionPair` there so that the
   generic fusion takes the names. `legacy.jl` is removed in R6.

## Phase log and pushes

Every phase is pushed to `claude/practical-archimedes` as a fast-forward and
checked with `git ls-remote origin refs/heads/claude/practical-archimedes`.

| Phase | Content | Pushed commit |
|:--|:--|:--|
| R0 | regression baseline and fixtures in `mpNRG/test/baseline/` | `debf22af2b97225347e05c0670c5fa7fd6453896` |
| R1 | `TNOps/` package: `git mv` of the symmetry layer, module wrapper, tests V1a/V1b | `9fb953612984b11a392d8d1d41ca3dcfd359d232` |
| R2 | legs, `BlockTensor`, `SlicedTensor`, arithmetic, `permute`, `contract`, `tr`, `outer`, `dense.jl` | `268d6cae63cb9749a13945816b499c2c65474966` |
| R3 | fusion, factorizations, `global_cut`, parity, `ncon`, interface, README, checks notebook | `1c38847989e9c787cd271af8c4d208b8f36ef528` |
