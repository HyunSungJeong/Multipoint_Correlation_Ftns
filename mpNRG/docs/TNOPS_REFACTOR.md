# Task sheet: `TNOps`, a separate tensor-operations package, and the refactor of `mpNRG` onto it

Repository: `https://github.com/HyunSungJeong/Multipoint_Correlation_Ftns`
Branch: **`claude/practical-archimedes`** (existing), starting from commit **`bb8b197`** ("mpNRG: STATUS branch note, work lives on claude/practical-archimedes").

**All commits of this task go on the existing branch `claude/practical-archimedes`, and every push goes to that branch on `origin`.** The user explicitly permits pushing to it. Do not create, push to or leave work on any other branch. If the session was started on an automatically created branch (for example `claude/<random-name>`), switch first with `git fetch origin claude/practical-archimedes && git checkout claude/practical-archimedes`, and work there. Branch-push rules are in Section 1, rule 5.

Companion documents in the repository:
- `mpNRG/docs/TASK.md`, the original mpNRG task sheet;
- `mpNRG/STATUS.md`, the phase log, verification record, deviations, findings and open issues;
- `mpNRG/README.md`.

Read all three before starting. This sheet amends `mpNRG/docs/TASK.md` only where Section 11 says so.

---

## 0. Starting point and goal

The mpNRG task is complete at `bb8b197` (phases P0–P10, 2897 test assertions passing, executed validation notebook with its PDF). The package `MultipointNRG` in `mpNRG/` contains its own abelian symmetry layer:

- `mpNRG/src/symmetry/labels.jl` (181 lines): `SymFactor`, `U1`, `Zn`, `Z2`, `Sector{M,K}`, `SymmetryGroup`, `fuse`, `dual`, `neutral`, `isneutral`, `parity`, `degeneracy`.
- `mpNRG/src/symmetry/blockop.jl` (297 lines): `SectorSpace`, `BlockOp{T,S}` (rank-2 block-sparse operator with a shift), `getblock`, `setblock!`, `target`, `blockeigen`, `blocksvd`, `todense`, `identity_op`, `restrict`, `SlicedBlockOp` (a sparse `Dict` from slice key to `BlockOp`), `mapslices_op`.
- `mpNRG/src/symmetry/fusion.jl` (109 lines): `FusionPair`, `FusionMap`, `fusionmap`, `kron_op`.

The tensor contractions of the NRG are written by hand on raw blocks in the modules above that layer:

- `nrg/sweep.jl`: `Shell.A::Dict{S,Matrix}` (isometries per product sector, rows in the fusion-map product basis, columns = all eigenstates of shell $n$, K first then D); `_rotate`; the product Hamiltonians built with `kron_op`; `_diag_blocks`; `truncate_shell` (global, degeneracy-safe cut);
- `operators/projection.jl`: `refine` (Lee Eqs. (31), (34), with the parity string of rule (R1)); `sector_view`; `krange`/`drange`/`xrange` in `nrg/sweep.jl`;
- `fdm/fdm.jl`: the backward sweep for $\bar\rho^n_{KK}$ (loops over fusion pairs with `view(A, rows, :)`); `shell_trace`;
- `psf/engine.jl`: `_blk` and the row-wise kernels `_trace_step!`, `_comp_left_step!`, `_comp_right_step!`, `_addrow!`, `_addcol!`;
- `fock/fockspace.jl` (`localop`, `parity_op`), `reference/ed.jl` (`ed_operator`, `ed_psf`), `hooks/hooks.jl`, `psf/api.jl` (`_convert_op`).

**Goal.**

1. Create a **separate Julia package `TNOps`** in a new top-level folder **`TNOps/`** for basic tensor-network operations on abelian-symmetric block-sparse tensors. It must contain no NRG, Wilson-chain, PSF or model code.
2. Move the symmetry layer of `mpNRG/src/symmetry/` into it, and generalize it from rank-2 operators to tensors of any rank.
3. Refactor `MultipointNRG` so that **every tensor data structure and every structural tensor operation comes from `TNOps`**. NRG-specific contractions become a thin, named layer `mpNRG/src/nrg/tensors.jl` built from `TNOps` primitives.
4. Results must not change beyond floating-point round-off (Section 8), the public API stays compatible (Section 6), and the executed notebooks and their PDFs must document the refactored code (Section 9).

---

## 1. Ground rules

1. **Footprint.**
   - New code goes in `TNOps/`. Changes inside `mpNRG/` are refactoring only, plus the new regression material of Sections 8–9.
   - `HubbardAtom` (`src/`, `test/`, `notebooks/`) is not touched, apart from the root `README.md` status block and `.gitignore`.
2. **Separate package.**
   - `TNOps` has its own `Project.toml` (new UUID), `src/`, `test/`, `README.md`, `STATUS.md` and `notebooks/`.
   - It depends only on `LinearAlgebra`, plus `Printf`; `Random` is allowed in tests only.
   - It must **not** depend on `MultipointNRG` or `HubbardAtom`.
   - Add it to `mpNRG/Project.toml`, `mpNRG/test/Project.toml` and `mpNRG/notebooks/Project.toml` through path sources, in the style already used there: `[sources] TNOps = {path = "../TNOps"}` from `mpNRG/`, and `{path = "../../TNOps"}` from `mpNRG/test/` and `mpNRG/notebooks/`.
3. **No external tensor libraries** (TensorKit, ITensors, TensorOperations, QSpace, …). Dense kernels are `LinearAlgebra` (BLAS/LAPACK) on blocks.
4. **History.** Move `mpNRG/src/symmetry/*.jl` with `git mv` into `TNOps/src/`, commit the move unchanged, then edit in later commits, so that `git log --follow` works.
5. **Commits and pushes.**
   - One commit per refactor phase (Section 10), in the repository's style, e.g. `TNOps R2: rank-k block tensors, contract, permute`.
   - The full `mpNRG` test suite must pass at every commit from R4 on.
   - **Push to the existing branch `claude/practical-archimedes` after every phase:** `git push origin HEAD:claude/practical-archimedes`. Push as a fast-forward only; never force-push.
   - After each push, confirm with `git ls-remote origin refs/heads/claude/practical-archimedes` that the remote head equals the local `HEAD`, and record the pushed hash in `TNOps/STATUS.md`.
   - Also push the tag `pre-tensor-refactor` (`git push origin pre-tensor-refactor`). If tag pushes are refused, record the tagged hash in both `STATUS.md` files instead.
   - **Never push to another branch.** If pushing to `claude/practical-archimedes` is refused (permissions, or a remote change that makes a fast-forward impossible), stop and report to the user. Do not push elsewhere, merge or rebase on your own initiative.
6. **Stop rule.** Stop and find the cause before going on, never adjusting tolerances, signs or reference data to make the refactor pass, if any of these happens:
   - an `mpNRG` test that passed at `bb8b197` fails;
   - a regression comparison (Section 8) exceeds its tolerance;
   - a kept count changes.
7. **Environment**, from `mpNRG/STATUS.md`:
   - Julia 1.13.0 came from conda-forge, and packages from the General registry via git (`JULIA_PKG_SERVER=""`), because `julialang-s3.julialang.org` and `pkg.julialang.org` were blocked.
   - `mpNRG/refs/` is gitignored, so a fresh session must re-download the papers (arXiv) and the LMU Open Data 613 dataset (`data.ub.uni-muenchen.de`) for V8. These domains must be in the session's network allowlist.
   - If the dataset cannot be downloaded, follow Section 9.3.

---

## 2. Where each existing piece goes

| Now (at `bb8b197`) | After the refactor |
|:--|:--|
| `symmetry/labels.jl` (all) | `TNOps/src/symmetry.jl`, behaviour unchanged |
| `SectorSpace` | `TNOps.Leg` (Section 3.2). `SectorSpace` remains as the constructor of an untagged leg with the default arrow |
| `BlockOp{T,S}` | rank-2 `TNOps.BlockTensor`. `const BlockOp{T,S} = BlockTensor{T,2,S}` is kept as an alias |
| `getblock`, `setblock!`, `target`, `todense`, `identity_op`, `restrict` | `TNOps` functions for tensors of any rank, with rank-2 methods matching the current behaviour |
| `blockeigen`, `blocksvd`, `_diag_blocks` | `TNOps.eigen` / `TNOps.svd` on a leg partition (Section 3.5) |
| `truncate_shell` (sorting, global cut, gap search) | the selection part becomes `TNOps.global_cut` (Section 3.5). `truncate_shell` stays in `mpNRG` as a thin wrapper that adds the NRG scale `ω` and the `last`-shell rule |
| `SlicedBlockOp`, `mapslices_op`, `nslices` | `TNOps.SlicedTensor` (Section 3.4) |
| `FusionPair`, `FusionMap`, `fusionmap` | `TNOps` fusion of two legs (Section 3.5), with the same pair/offset layout |
| `kron_op` | `TNOps.kron_fused(X, Y, Frow, Fcol)`, generic |
| `Shell.A::Dict{S,Matrix}` | a rank-3 `BlockTensor` per shell (Section 4.2) |
| `_rotate`, `refine`, FDM backward loop, `shell_trace`, `sector_view`, `_blk` | named functions in `mpNRG/src/nrg/tensors.jl` (Section 4.3) |
| `localop`, `parity_op` (`fock/`) | build rank-2 `BlockTensor`s through `TNOps` constructors |
| `ed_operator`, `ed_psf` (`reference/`) | may use `TNOps` containers (see Section 7 on independence) |
| engine row-wise kernels | stay in `psf/engine.jl` as declared dense kernels (Section 4.4) |
| `test/test_symmetry.jl`, generic parts of `test/test_v1_symmetry.jl` | `TNOps/test/` |

---

## 3. Specification of `TNOps`

### 3.1 Layout

```
TNOps/
├── Project.toml
├── README.md          # conventions (arrows, flux, block rule, parity, leg order), examples
├── STATUS.md          # phase log, verification record
├── src/
│   ├── TNOps.jl       # module, exports
│   ├── symmetry.jl    # moved labels.jl (U1, Zn, Z2, Sector, SymmetryGroup, fuse, dual, …)
│   ├── legs.jl        # Leg, LegSector, arrows, dual(leg)
│   ├── tensor.jl      # BlockTensor{T,N,S}: storage, construction, block access, flux
│   ├── sliced.jl      # SlicedTensor: sparse map from slice key to BlockTensor
│   ├── linalg.jl      # arithmetic, permute, contract, tr, outer, kron_fused
│   ├── fusion.jl      # moved fusion.jl, generalized: fuse/split legs, FusionMap
│   ├── factorize.jl   # eigen, svd, qr on a leg partition; global_cut
│   ├── parity.jl      # parity_tensor, apply_parity
│   ├── ncon.jl        # multi-tensor contraction with a caller-given order
│   ├── dense.jl       # todense / fromdense with explicit index maps
│   └── interface.jl   # the functions a future non-abelian layer must supply
├── test/
└── notebooks/
    ├── tnops_checks.ipynb   # executed, committed with outputs
    └── tnops_checks.pdf
```

### 3.2 Symmetry and legs

- `symmetry.jl` is `labels.jl` moved unchanged: U(1) and Z_n factors, canonical `isbits` `Sector{M,K}` labels, the fermion parity `Z2(:Pf)` always appended as the last factor, and `degeneracy(::Sector) = 1`. Do not rename `Sector`; it is part of `MultipointNRG`'s public API.
- A `Leg` is an ordered list of `LegSector`s, each a pair (charge label `::Sector`, tag `::Int`) with a dimension, plus an arrow (`In`/`Out`) and an optional name.
  - The tag distinguishes blocks that share a charge label. Its main use is the K/D split of a shell (Section 4.2). Tag `0` means untagged.
  - Sector order is deterministic: sorted by (label, tag), as `SectorSpace` sorts labels today.
- `dual(leg)` flips the arrow and dualizes the labels. Contraction requires one leg to be the dual of the other, and mismatch errors name the legs.

### 3.3 Block tensors

- `BlockTensor{T,N,S}`: `N` legs, a flux $\Delta::S$, and a `Dict` from `NTuple{N,LegSector}` to dense `Array{T,N}`.
- Block rule: a block may exist only if the fused labels of the `Out` legs equal the fused labels of the `In` legs fused with $\Delta$. State this once in the README.
- Absent blocks are zero and not stored. `T` promotes only through explicit arithmetic, as in `BlockOp` today.
- For rank 2 with legs (`Out` row, `In` column), the flux is the current `shift`. All current `BlockOp` semantics must hold for the alias, including `target(A, q)`, the column-sector keyed `getblock`, and `todense` with sectors stacked in sorted order.

### 3.4 Sliced tensors

`SlicedTensor{Kt,T,N,S}` is a sparse map from slice key to `BlockTensor`, with common legs and flux. It generalizes `SlicedBlockOp`, and `mapslices` applies a contraction to every slice.

The key type is arbitrary: tuples of bin indices in binned mode, and **`Float64` energies in exact mode** (`ExactMode` in `psf/engine.jl`). For that reason slices are **not** turned into tensor legs. Exact-mode keys are not bin indices, and the sparse map is what the engine fills (mpNRG `STATUS.md`, deviation 2).

### 3.5 Operations

All operations are block-wise, with no dense intermediates outside `dense.jl`.

- **Arithmetic:** `+`, `-` (equal legs and flux), scalar `*`, `conj`, `adjoint` (rank 2, or with an explicit leg partition), `norm`, `dot`, `isapprox`.
- **`permute(A, perm)`:** a pure leg permutation that is **bosonic**, i.e. it never inserts a fermionic sign.
- **`contract(A, legsA, B, legsB)`:**
  - checks the duality of each leg pair;
  - the output legs are the uncontracted legs of `A`, then those of `B`, and the output flux is the fused flux;
  - per matching block pair it permutes, reshapes, calls `mul!` and accumulates;
  - it threads over output blocks;
  - it never computes a block the flux forbids.
- **`tr`** over a dual leg pair. A full trace of a tensor with non-neutral flux is an exact zero, computed without touching blocks.
- **`outer`**.
- **Fusion.** `fuse_legs(A, legs) -> (A′, FusionMap)` and `split_legs(A′, leg, fmap)`.
  - `FusionMap` keeps the current layout: per fused sector, the ordered pairs with offsets, product index `offset + (iσ-1)*dq + s`, and a `lookup`.
  - `fusionmap(Vloc, Vprev)` and `kron_fused(X, Y, Frow, Fcol)` reproduce `fusionmap` and `kron_op` exactly, including the order of pairs, so that product bases do not change.
- **Factorizations** on a leg partition: `eigen` (hermitian, neutral flux), `svd` and `qr`.
  - Each returns sector-tagged spectra and tensors with a new bond leg.
  - `eigen` keeps today's symmetrization `(M + M')/2` and threads over blocks like `_diag_blocks`.
- **`global_cut(spectra; nkeep, threshold, gaptol)`:** this is the selection logic of `truncate_shell`, moved, and it must give **identical kept counts**.
  - Sort all values across sectors, counting each `degeneracy(label)` times.
  - Place the cut at `nkeep` or at the threshold, whichever keeps fewer.
  - Move the cut up into the nearest gap larger than `gaptol`.
  - Return the kept count per sector and the truncation record.
- **Parity:** `parity_tensor(leg)` and `apply_parity(A, leg; power)`, which multiplies blocks by $(-1)^{p\cdot{\rm power}}$ according to the parity of their sector on that leg. These are the only way fermionic signs enter: `refine`'s `sg` factor and rule (R1) must be expressed with them.
- **`ncon`** with an explicit contraction order. No automatic order optimization.
- **`todense` / `fromdense`** with documented index maps. `fromdense` errors on weight outside the allowed blocks.

### 3.6 Non-abelian interface

`interface.jl` lists the only functions code above `TNOps` may use for anything symmetry-dependent: `fuse`, `dual`, `neutral`, `isneutral`, `parity`, `degeneracy`, and the block rule. Every state count goes through `degeneracy`, as it does today in `fdm/fdm.jl` and `truncate_shell`. Do not implement non-abelian symmetry.

### 3.7 Tests

Every operation is tested against dense arrays through `todense`/`fromdense`, to $10^{-13}$ relative:

- ranks 1–5, 2–5 sectors per leg, random fluxes;
- symmetries U1×U1×Z2, Z_3, and parity only;
- real and complex element types.

Specific checks:

- `contract` and `ncon` against explicit dense loops;
- `fuse_legs`∘`split_legs` is the identity;
- `fusionmap`/`kron_fused` against the **frozen output of the old `fusionmap`/`kron_op`**: save small fixtures at the `pre-tensor-refactor` tag and compare pair lists and blocks exactly;
- factorizations reconstruct the tensor;
- `global_cut` gives the same counts as the old `truncate_shell` on the spectra recorded in the baseline (Section 8);
- forbidden blocks are never allocated (check the keys);
- `apply_parity` against the dense parity matrix;
- duality errors are raised for mismatched legs.

Move the generic tests of `mpNRG/test/test_symmetry.jl` and `test_v1_symmetry.jl` here; V1a and V1b of the mpNRG verification record become `TNOps` tests. The mpNRG-specific parts stay in `mpNRG/test/`: shifts of composites, symmetry violation, selection rules, symmetry independence.

---

## 4. The NRG tensor layer inside `mpNRG`

### 4.1 Location and rule

`mpNRG/src/nrg/tensors.jl` defines the **only** tensor operations that `nrg/`, `operators/`, `fdm/`, `psf/`, `hooks/` and `observables/` may call. Each is a named function built from `TNOps` primitives, with a docstring citing the equation (Lee 2021 numbering, as in the current docstrings).

Outside `TNOps` and this file, no code accesses block dictionaries directly (no `.blocks`, no `getblock` on raw sectors). The only exception is the declared kernels of Section 4.4. Add a test that greps `mpNRG/src` for `.blocks` and fails on any occurrence outside `nrg/tensors.jl` and the declared kernels.

### 4.2 Objects

- **Shell isometry.** $A^n$ is a rank-3 `BlockTensor` with legs `site` (local space of site $n$), `old` (kept space of shell $n-1$) and `new` (all eigenstates of shell $n$). Its arrows are documented, and it is Lee Eq. (15).
  - At $n=-1$ it is rank 2 (impurity states → eigenstates).
  - The product-basis order must be the one of the current `FusionMap`, so that $A^n$ contains exactly the numbers of today's `Shell.A[q]`, reshaped.
- **K/D split by tags.** The `new` leg carries, for every label $q$, the sectors $(q,\mathrm K)$ and $(q,\mathrm D)$ with dimensions `nkept` and `ndisc`. Operators in the shell-$n$ basis then have blocks keyed by tagged sectors.
  - `sector_view(run, O, n, q, X, X̄)` becomes a block lookup.
  - `krange`, `drange` and `xrange` remain as exported wrappers for compatibility.
  - If tagging measurably slows the engine (Section 10, R6 timing), the alternative is to keep untagged sectors with K-first ordering and index ranges. Choose by measurement and record the choice in `STATUS.md`.
- **Operators** are rank-2 tensors (`BlockOp` alias); sliced operators are `SlicedTensor`s.

### 4.3 Named contractions (replacing hand-written code)

| Function | Replaces | Content | Source |
|:--|:--|:--|:--|
| `fuse_site(loc, prev)` | `fusionmap(Vloc, keptspace(prev))` in `run_nrg` | Product basis of shell $n$ and its `FusionMap` (fermionic order of Lee p. 5) | Lee Sec. III A |
| `product_hamiltonian(...)` | `_site0_product_hamiltonian`, `_chain_product_hamiltonian` | $H_n$ on the product space from the lifted old-shell energies, the on-site terms and the hopping $(f^\dagger_{n\nu}Z_n)\otimes f_{n-1,\nu}$ via `kron_fused` | Lee Sec. III A; rule (R1) |
| `diagonalize_shell(H; cut)` | `_diag_blocks`, `_finish_shell` | `TNOps.eigen` plus `global_cut` (through `truncate_shell`) | Lee Sec. III A |
| `rotate(A, P)` | `_rotate` | $A^\dagger P A$ for a product-space operator | |
| `lift_site(A, O_n)` | `kron_op(fl, Ik, …)` then `_rotate` in `run_nrg` | $A^\dagger(O_n\otimes\mathbb 1)A$ (rule (R2)) | |
| `refine_KK(A, O)` | `refine` | $A^\dagger[(Z_n)^{p}\otimes O_{KK}]A$ with `apply_parity` on the site leg; works slice by slice for `SlicedTensor` | Lee Eqs. (31), (34); rule (R1) |
| `shell_operator(run, expr, n)` | same | Exact matrix for $n\le n_0$ from `dense_embedding` (unchanged in content; output as `BlockTensor`) | |
| `backward_step(A, ρ̄_KK, ρ_D)` | the loop over `Fu.pairs` in `fdm` | $\bar\rho^{\,n-1}_{KK}=\sum_{\sigma}A^{\sigma}\,\rho^{\rm full}\,(A^{\sigma})^\dagger$, a partial trace over the `site` leg | Lee Eqs. (40)–(45); Weichselbaum & von Delft (2007), Eq. (8) |
| `shell_trace(ρ̄, ops...)` | `shell_trace` in `fdm/fdm.jl` | Shell trace of a product of sector-projected operators | Lee Eqs. (46)–(48) |
| `block_view(O, q, X, X̄)` | `_blk`, `sector_view` | Read-only dense block of tagged sectors, for the kernels of 4.4 | |

Test each function in `mpNRG/test/test_nrg_tensors.jl`:

- against the old implementation at the `pre-tensor-refactor` tag, through saved fixtures, to $10^{-13}$;
- against a dense Kronecker/Jordan–Wigner reference for chains of up to 4 sites, reusing the construction of `test/test_v1_chain.jl`.

### 4.4 Declared dense kernels (allowed exception)

The PSF engine evaluates shell traces and composite refinements **row by row**: it bins each row's eigenvalue differences and groups states by bin key (`_trace_step!`, `_comp_left_step!`, `_comp_right_step!`, `_addrow!`, `_addcol!`, and the row loops in `shell_trace!`, `_composite_left`, `_composite_right`). This is spectral bookkeeping on dense data, not a structural tensor operation. It stays in `psf/engine.jl` as dense code on matrices obtained **only** through `block_view`.

List these functions in a comment at the top of `psf/engine.jl` as "declared dense kernels", and exempt only them in the grep test of 4.1. Do not rewrite them as generic contractions, because that would destroy the binning-per-row structure that the performance work of P10 relies on.

---

## 5. What does not change

- **Algorithms and numerics.** Lee's formulas, the slicing conventions (`:lee`, `:left`), schedules (`:depth`, `:breadth`), `ztol`, `wtol`, grids, the PSF convention, and the order of product bases and of pairs in fusion maps.
- **The independent references.**
  - `test/test_v4e_dense.jl` (dense brute force without `BlockOp`) and the dense Kronecker Jordan–Wigner construction in `test/test_v1_chain.jl` must stay free of `TNOps`.
  - The `HubbardAtom` comparisons (V3b, V3c, export) are unchanged.
  - `reference/ed.jl` may use `TNOps` containers, but its numerics stay dense.
- **Output files.** The JLD2 output format of `io/io.jl` and `load_results` of files written at `bb8b197` must keep working. Add a test that loads a file written by the old code (a fixture made in R0).

---

## 6. Public API compatibility

- **Re-exports.** `MultipointNRG` keeps every name it exports at `bb8b197`. Names that now live in `TNOps` (`SymFactor`, `U1`, `Zn`, `Z2`, `Sector`, `SymmetryGroup`, `sector`, `labeltype`, `factor_index`, `factornames`, `fuse`, `dual`, `neutral`, `isneutral`, `parity`, `degeneracy`, `SectorSpace`, `sectors`, `dim`, `offsets`, `BlockOp`, `getblock`, `setblock!`, `target`, `blockeigen`, `blocksvd`, `todense`, `identity_op`, `restrict`, `SlicedBlockOp`, `mapslices_op`, `nslices`, `FusionPair`, `FusionMap`, `fusionmap`, `kron_op`) are re-exported. They become aliases or thin wrappers where the type changed.
- **Changed internals.**
  - `Shell` fields may change: `A` becomes a tensor, and the K/D information moves into tags.
  - The accessors `shell`, `keptspace`, `discspace`, `krange`, `drange`, `xrange`, `nkept`, `ndisc` and `sector_view` keep their signatures and meanings.
  - The hook accessors (`isometries`, `fusionmap`, `rhoKK`, `call_operators`, `refinement_map`, `carried`) keep their names. Their return types change from `Dict(sector => Matrix)`/`BlockOp` to `TNOps` objects.
  - Document every such change in a "Changes in the TNOps refactor" section of `mpNRG/README.md`, and update the docstrings of `hooks/hooks.jl` (the memory-layout paragraph).
- **Updating callers.** Update `examples/*.jl`, `scripts/*.jl` and `notebooks/build_notebook.py` wherever they use changed internals. For example, `build_notebook.py` constructs `BlockOp{Float64}(V, V, Δ)` in the V1 section.

---

## 7. Independence of the checks

`TNOps` is verified against dense arrays (Section 3.7); `mpNRG` is verified against ED and `HubbardAtom`. Keep at least these checks free of `TNOps` so that a `TNOps` bug cannot hide in both sides of a comparison:

- V1d (dense Kronecker Jordan–Wigner);
- V4e (dense brute force of Lee's recursion);
- the `HubbardAtom` comparisons.

---

## 8. Regression baseline

**R0, before changing any code.**

1. Check out `bb8b197`, run the full test suite, and record the result.
2. Tag the commit `pre-tensor-refactor`. Push the tag if allowed; otherwise record the hash.
3. Write `mpNRG/test/baseline/make_baseline.jl`. It runs a fixed set of small calculations with the pre-refactor code and stores the results in `mpNRG/test/baseline/baseline.jld2`. Commit both files.

**Contents of the baseline:**

- SIAM, $\Lambda=4$, short chains, with and without truncation, two symmetry choices (`Z2(:Pf)` only and `U1×U1×Z2`), one complex-valued model if one exists among the built-ins (otherwise skip);
- per shell: energies, kept counts per sector, truncation records, ground-state shifts;
- FDM: shell weights, $\log Z$, eigenvalues of every $\bar\rho^n_{KK}$;
- thermal averages, and thermodynamics (`fdm_thermodynamics`);
- 2p, 3p and 4p PSFs in binned mode (all permutations of one operator tuple per $\ell$, including a composite $q=[d,H_{\rm int}]$), in exact mode for one small case, and with `abs = true` for one case;
- the `:left` slicing form for one 3p case;
- the singular values recorded by `SingularValueRecorder` for one 3p call;
- frozen fixtures for Section 3.7 (`fusionmap`, `kron_op`, `truncate_shell`) and Section 4.3 (`refine`, `_rotate`, the FDM backward step), and one output file of `save_results` for Section 5.

**Compare only basis-independent quantities.** Do not compare isometries or $\bar\rho$ matrices elementwise. Within degenerate subspaces, `eigen` may return a different basis after any change in summation order, which leaves all physical results unchanged.

**Comparison.** `mpNRG/test/baseline/test_baseline.jl`, run after every phase from R4 on:

- energies, weights, $\log Z$, averages and $\bar\rho$ eigenvalues to $10^{-12}$ relative;
- PSF histograms bin by bin to $10^{-12}$ of the largest weight;
- exact-mode positions to $10^{-12}$;
- identical sets of nonzero bins.

**Kept counts must be identical.** A difference is a stop condition (Section 1, rule 6).

**Bin flips.** If a weight moves to an adjacent bin, it is acceptable only if the energy difference concerned lies within $10^{-12}$ of a bin edge (or of `ztol` for the zero bin). The test must demonstrate this case by case and report it. Anything else is a stop condition.

---

## 9. Notebooks and PDFs

### 9.1 `TNOps` notebook

Create `TNOps/notebooks/tnops_checks.ipynb`, executed with the Julia 1.13 kernel and committed with outputs, together with its PDF `TNOps/notebooks/tnops_checks.pdf`. It contains one section per group of Section 3.7, printing the achieved accuracies, plus a short usage example (build legs, contract, fuse, factorize with `global_cut`).

### 9.2 Re-execute the mpNRG validation notebook (required)

After the refactor, **re-execute `mpNRG/notebooks/mpnrg_validation.ipynb` on the refactored code and regenerate `mpNRG/notebooks/mpnrg_validation.pdf`.** Otherwise the committed outputs would still document the pre-refactor code.

- Change the notebook only through `mpNRG/notebooks/build_notebook.py`, then rebuild and execute it as documented at the top of that script (`jupyter nbconvert --to notebook --execute --inplace …`, 4 threads). Never hand-edit outputs.
- Render the PDF the same way as the existing one. The current PDF was produced by headless Chrome, which suggests `jupyter nbconvert --to webpdf`; check how it was made and use the same method.
- Verify that every code cell has outputs.
- Check that the printed accuracies agree with the verification record in `mpNRG/STATUS.md` (V1–V8 and export). Any change beyond round-off is a stop condition.
- Regenerate the figures in `notebooks/figures/mpnrg/`.

### 9.3 V8 results

The V8 section of the notebook only *reads* `notebooks/figures/mpnrg/v8_results.jld2`, which `scripts/v8_benchmark.jl` produces from the LMU 613 dataset. Re-execution alone therefore does **not** re-test V8.

- Download the dataset (Section 1, rule 7), regenerate `v8_results.jld2` with the refactored code, and run `test/test_v8_external.jl`. Confirm the V8 numbers of `STATUS.md`.
- If the dataset cannot be downloaded, leave `v8_results.jld2` unchanged, and state explicitly in the notebook (V8 section) and in `mpNRG/STATUS.md` that the V8 results predate the refactor and were not re-run.

### 9.4 Regression and timing section (required)

Add a section **"Refactor regression (TNOps)"** to `mpnrg_validation.ipynb`, through `build_notebook.py`. It contains:

1. **Baseline comparison.** It loads `test/baseline/baseline.jld2`, recomputes every quantity of Section 8 with the refactored code, and prints a table with columns: quantity, maximum deviation, tolerance, pass/fail. It also prints the kept-count identity check and any bin flips with their distance to the bin edge.
2. **Timing comparison.** The performance table of `STATUS.md` (P10: sweep, 2p, 3p, 4p versus $N_{\rm keep}=25,50,100$, SIAM $U=0.2$, $\Delta=0.04$, $T=10^{-4}$, $n_{\rm dec}=8$, 4 threads) is measured for **both** the `pre-tensor-refactor` commit and the refactored code, on the same machine and the same number of threads.
   - Use a `git worktree` of the tag for the old code, run by a script `mpNRG/scripts/timing_compare.jl` that writes `notebooks/figures/mpnrg/timing_compare.jld2`. The notebook reads that file and shows both columns and their ratio.
   - Report the machine and thread count. Do not reuse the old numbers in `STATUS.md`, which were measured on a different machine and session.

---

## 10. Phases

| Phase | Content | Acceptance |
|:--|:--|:--|
| R0 | Check out `claude/practical-archimedes` (Section 0 header); environment (Section 1, rule 7); full suite at `bb8b197`; tag `pre-tensor-refactor`; baseline and fixtures (Section 8); old-format output file (Section 5) | Baseline committed and pushed to `claude/practical-archimedes`; `STATUS.md` (both) record the tag and environment |
| R1 | `TNOps/` skeleton; `git mv` of `symmetry/*.jl` (unchanged commit), then the module wrapper; `mpNRG` uses `TNOps` for these names (re-exports) | `TNOps` label tests pass; full `mpNRG` suite and baseline pass |
| R2 | `Leg`/`LegSector`, `BlockTensor` (with `BlockOp` alias), `SlicedTensor`, arithmetic, `permute`, `contract`, `tr`, `outer`, `dense.jl` | Section 3.7 tests for these; `mpNRG` suite and baseline pass |
| R3 | Fusion (`fusionmap`, `kron_fused`, `fuse_legs`/`split_legs`), factorizations, `global_cut`, parity, `ncon`, `interface.jl`; `TNOps` README and notebook with PDF | All of Section 3.7, including the frozen-fixture comparisons; `tnops_checks` executed |
| R4 | `mpNRG/src/nrg/tensors.jl` (Section 4.3) with `test_nrg_tensors.jl` | Each named function matches its fixture and the dense reference |
| R5 | Port module by module, running the full suite and the baseline after each: (a) `fock/`, `reference/`; (b) `nrg/sweep.jl` (rank-3 $A^n$, K/D tags); (c) `operators/`; (d) `fdm/`; (e) `psf/` (`block_view`, `SlicedTensor`, declared kernels); (f) `hooks/`, `observables/`, `io/`, `psf/api.jl`; then the grep test of 4.1 | Suite and baseline pass after every sub-step (one commit each is fine) |
| R6 | Remove `mpNRG/src/symmetry/`; README "Changes" section; update examples, scripts and `build_notebook.py`; V8 re-run (9.3); regression and timing section (9.4); re-execute `mpnrg_validation.ipynb` and regenerate its PDF (9.2); amend `docs/TASK.md` (Section 11); root `README.md` rows | No old symmetry code; notebook and PDF regenerated with all outputs; timing ratio recorded; any slowdown above 1.2× in the engine (2p/3p/4p columns) explained in `mpNRG/STATUS.md`, and above 1.5× fixed before finishing |

---

## 11. Amendments to `mpNRG/docs/TASK.md`

Apply these in R6, and put a dated note at the top pointing to this sheet (copy it to `mpNRG/docs/TNOPS_REFACTOR.md`).

1. **Section 1, rule 1.** Code may also live in `TNOps/`.
2. **Section 1, rule 3.** The abelian block-sparse layer is the in-repository package `TNOps/`; no external tensor libraries.
3. **Section 3.** `symmetry/` leaves `mpNRG/src`; add `nrg/tensors.jl` and the sibling package `TNOps/`.
4. **Section 5.2.** The behaviour stays authoritative, and the locations change:
   - 5.2.1–5.2.2 and 5.2.9 → `TNOps/src/symmetry.jl` and `interface.jl`;
   - 5.2.4 → `TNOps` legs, `BlockTensor` and `SlicedTensor`;
   - 5.2.5 → `TNOps` fusion and parity, plus `nrg/tensors.jl` (`fuse_site`, `lift_site`, `refine_KK`);
   - 5.2.7 → `TNOps.global_cut`, called by `truncate_shell`;
   - 5.2.3 and 5.2.6 stay in `mpNRG`.
5. **Section 5.2.10.** The generic tests move to `TNOps/test/`; the physics-facing tests stay.
6. **Section 8.** Deliverables now include the regenerated notebook and PDF and the regression section of 9.4.

---

## 12. Deliverables

1. **`TNOps/`:** tests, `README.md`, `STATUS.md`, and `notebooks/tnops_checks.ipynb` (executed) with its PDF.
2. **`MultipointNRG` on `TNOps`:**
   - `nrg/tensors.jl`, and no `mpNRG/src/symmetry/`;
   - public API compatible (Section 6);
   - all tests passing (the V8 test with the dataset, or recorded as not re-run);
   - baseline comparison passing.
3. **`mpNRG/notebooks/mpnrg_validation.ipynb`**, re-executed on the refactored code with all outputs, with the new "Refactor regression (TNOps)" section (baseline table and timing comparison), and **`mpnrg_validation.pdf` regenerated**.
4. **`mpNRG/STATUS.md` updated:**
   - a refactor phase log (R0–R6 with commits);
   - the verification record re-confirmed;
   - the timing comparison;
   - every deviation, bin flip and finding.
5. `mpNRG/docs/TASK.md` amended (Section 11), with this sheet copied to `mpNRG/docs/TNOPS_REFACTOR.md`; the root `README.md` gains a `TNOps` block in the status section.
6. Everything committed **and pushed to the existing branch `claude/practical-archimedes`** (verified with `git ls-remote`, Section 1, rule 5), with no other branch created or pushed.
