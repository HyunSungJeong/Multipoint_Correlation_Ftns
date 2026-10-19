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

Tests: `julia -t 4 --project=TNOps/test TNOps/test/runtests.jl` — **2352
assertions, all passing** (R6; 2354 at R4–R5, see decision 7 for why the
count of the random contraction tests changed). Every operation is checked against the same
operation on `todense` arrays to 1e-13 relative, for random tensors of rank
1–5 with 2–5 sectors per leg (tagged and untagged), random fluxes, the
symmetries U1×U1×Z2, Z3 (×Pf) and parity only, real and complex elements;
the random seeds are fixed integers (an earlier version seeded with
`hash(name)` and its assertion count varied between runs). Achieved (notebook `notebooks/tnops_checks.ipynb`, re-executed in R6 on the final code, largest deviation):

| Check | Achieved |
|:--|:--|
| `fromdense(todense(A)) == A`; stored forbidden blocks | 0; none (90 tensors) |
| `+`, `-`, scalar `*`, `conj`, `norm`, `dot` vs dense | 3.7e-16 |
| `permute` vs `permutedims` | 0 |
| `contract` (ranks 1–4, 0–2 pairs) vs explicit loops; forbidden output blocks | 3.1e-16; none (90 contractions) |
| `ncon` (three tensors, explicit order) vs explicit loops | 1.4e-16 |
| `tr` (rank 2, partial) vs dense | 1.0e-16 |
| `split_legs(fuse_legs(A)) - A` | 0 |
| `fusionmap` vs the frozen pre-refactor `fusionmap` (252 pairs, 6 shells) | identical (order and offsets) |
| `kron_fused` vs the frozen pre-refactor `kron_op` (54 blocks) | bit-identical |
| `eigen` (rank 2; rank-4 partition) reconstruction | 6.8e-16; 1.1e-15 |
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
5. **Legacy layer, then compatibility names.** Until MultipointNRG was ported
   (R1–R5), the pre-refactor `SectorSpace`, `BlockOp`, `SlicedBlockOp`,
   `fusionmap(::SectorSpace, …)` and `kron_op` lived unchanged in
   `src/legacy.jl` (the old fusion types renamed `OldFusionMap` /
   `OldFusionPair`). R6 removed it; `src/compat.jl` keeps the names as aliases
   and thin wrappers: `SectorSpace === Leg` (an untagged `Out` leg from a
   label ⇒ dimension dictionary), `BlockOp{T,S} = BlockTensor{T,2,S}` with the
   constructor `BlockOp{T}(row, col, Δ)` and the old 4-argument form,
   `SlicedBlockOp{T,S,Kt} = SlicedTensor{Kt,T,2,S}`, `mapslices_op = mapslices`,
   `blockeigen` / `blocksvd` (per label, returning the old dictionaries),
   `identity_op = identity_tensor`, `kron_op = kron_fused`. The V1a/V1b tests
   moved from mpNRG run unchanged on them.
6. **`eigen` uses the divide-and-conquer driver** (`LAPACK.syevd!`, through
   `eigen(H; alg = LinearAlgebra.DivideAndConquer())` where available), not
   Julia's default MRRR driver (`syevr`). Found in R5b: the eigenvectors MRRR
   returned for one 24×24 shell block of the V1d chain test were orthonormal
   only to ‖V'V − 1‖ = 1.6e-13, above the test's 1e-13 (the pre-refactor
   code, which happened to call the same driver on a differently assembled but
   equal matrix, gave 2.7e-14). QR (`syev`) and divide-and-conquer give
   ~5e-15 on the same matrix; with divide-and-conquer the largest deviation
   over the test is 1.1e-14. No tolerance was changed.
7. **Content hashes for `Sector`, `LegSector` and `Arrow`.** Julia's default
   hash of a struct value (and of an enum) mixes in the identity of its type,
   which differs between builds (the same source compiled with other code
   loaded): `hash(LegSector(q, 0))` was found to differ between the R5b and the
   R6 builds. Every block dictionary of a tensor is keyed by such sectors, so
   iteration orders — and with them the order in which contractions add
   contributions, hence the round-off — depended on unrelated code. The same
   mechanism in mpNRG's operator algebra caused the R5c finding
   (`mpNRG/STATUS.md`); it probably also explains the unidentified
   rounding-level offset of the first R1 comparison. With content hashes the
   order is the same in every build. A side effect: the random legs of the
   contraction tests were built by collecting a `Set` of sectors, so their
   composition, and the assertion count (2354 → 2352), depended on the build;
   the test now sorts the sectors first.
8. **`setslice!` compares legs by identity first** (then by value): slices
   produced by one contraction share their leg objects, and the value
   comparison of sector vectors showed up in the PSF engine's profile.

## Phase log and pushes

Every phase is pushed to `claude/practical-archimedes` as a fast-forward and
checked with `git ls-remote origin refs/heads/claude/practical-archimedes`.

| Phase | Content | Pushed commit |
|:--|:--|:--|
| R0 | regression baseline and fixtures in `mpNRG/test/baseline/` | `debf22af2b97225347e05c0670c5fa7fd6453896` |
| R1 | `TNOps/` package: `git mv` of the symmetry layer, module wrapper, tests V1a/V1b | `9fb953612984b11a392d8d1d41ca3dcfd359d232` |
| R2 | legs, `BlockTensor`, `SlicedTensor`, arithmetic, `permute`, `contract`, `tr`, `outer`, `dense.jl` | `268d6cae63cb9749a13945816b499c2c65474966` |
| R3 | fusion, factorizations, `global_cut`, parity, `ncon`, interface, README, checks notebook | `1c38847989e9c787cd271af8c4d208b8f36ef528` |
| R4 | mpNRG `nrg/tensors.jl` (named NRG contractions) and `test_nrg_tensors.jl`; TNOps `retag` | `6c119a66ca55633d370ab6f687a9827c56c78618` |
| R5a | mpNRG `fock/`, `reference/` build TNOps tensors | `02f2d104f0a76faf20bed48646c77c6182cc3561` |
| R5b | mpNRG forward sweep on TNOps tensors (rank-3 isometries, K/D tags); TNOps `eigen` on divide-and-conquer (decision 6) | `138a7cdc40c80fd99b8dd9ab529ab451b9b6d8d2` |
| R5c | mpNRG `operators/` on TNOps tensors; content hashes for `Mode`/`Ladder` (finding in `mpNRG/STATUS.md`) | `1c225feb59fc9a6017c8162903e72d1d10428aef` |
| R5d | mpNRG FDM backward sweep on TNOps tensors | `9aad3f41275184579066dbf3c862b998cd2fb7ad` |
| R5e | mpNRG PSF engine on `SlicedTensor`s and `block_view` | `e7f9f68f4f6bb5dd076fb5c171df0ac9b9101f66` |
| R5f | mpNRG hooks on tensor objects; grep test for block access | `ed7ba5438a3b3291e8199ddaf24eac76cbfd275c` |
| R6a | `legacy.jl` replaced by `compat.jl`; content hashes (decision 7); mpNRG README changes, `timing_compare.jl` | `0c10fe1962959f9ed19d3859cbad6282910c8e1f` |
| R6b | mpNRG `RefinePlan` (engine performance, `mpNRG/STATUS.md`); cheaper hashes; `setslice!` identity check (decision 8) | `4cf33a3f15beb35f283454c1b58e2493ab7ae917` |
| R6c | mpNRG `timing_compare.jl`: minimum over repetitions after a full GC | `839b151bd9a0d5bc2283424b47806aa71546eb39` |
| R6d | TNOps notebook re-executed, PDF; timing comparison data; README and STATUS records | `26e16909d9e57f9f5fa54017c1f00b2381de13f8` |
| R6e | V8 re-run; `mpnrg_validation.ipynb` re-executed with the regression section; PDF and figures; records | `45427709489be9e3bda7630fe5d6c0ad62710786` |
