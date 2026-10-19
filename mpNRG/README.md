# MultipointNRG

A general-purpose full-density-matrix numerical renormalization group (fdm-NRG)
in Julia for quantum impurity models with a finite local Fock space. It computes
thermodynamics, thermal expectation values, and **local 2p, 3p and 4p partial
spectral functions (PSFs)**, following

> S.-S. B. Lee, F. B. Kugler and J. von Delft, *Computing local multipoint
> correlators using the numerical renormalization group*,
> [Phys. Rev. X **11**, 041007 (2021)](https://doi.org/10.1103/PhysRevX.11.041007) — "Lee" below,

with the PSF convention of Kugler, Lee & von Delft,
[PRX **11**, 041006 (2021)](https://doi.org/10.1103/PhysRevX.11.041006), Eq. (28) — "Kugler" —
which is the convention of the repository's `HubbardAtom` package.

The package is independent of `HubbardAtom` (its `src/` uses only `LinearAlgebra`,
`Printf`, `JLD2` and the sibling package [`TNOps`](../TNOps), the abelian
block-sparse tensor layer); the tests use `HubbardAtom` as an independent
reference, and PSFs can be exported to its Matsubara and Keldysh kernels.

The verification record (every validation item, with achieved accuracies) is in
[`STATUS.md`](STATUS.md); the executed validation notebook is
[`notebooks/mpnrg_validation.ipynb`](notebooks/mpnrg_validation.ipynb) (PDF alongside).

## Quick start

```julia
using MultipointNRG

m = siam(U = 0.5, Δ = 0.04)                     # εd = -U/2, box Δ θ(D - |ω|), D = 1
T = 1e-6
p = NRGParams(N = chain_length_for(T; Λ = 4), Nkeep = 800, Λ = 4.0, z = 1.0)
run = run_nrg(m, p)                              # forward sweep (shells -1..N)
f = fdm(run; T = T)                              # full density matrix (backward sweep)

d(σ) = cop(Mode(-1, σ))                          # impurity annihilator d_σ
g = LogGrid(emin = 1e-9, emax = 10.0, ndec = 32) # one grid for all PSFs of a run

S2 = psf(run, f, (d(:up), d(:up)'); grid = g)                          # 2p, dense histogram
S3 = psf(run, f, (d(:up), d(:up)', num(Mode(-1, :dn))); grid = g)     # 3p, Dict of nonzero bins
S4 = psfs(run, f, (d(:up), d(:up)', d(:dn), d(:dn)'); grid = g)       # all 24 permutations (4p); costly:
                                                 # minutes per 4p PSF at N_keep = 100 (STATUS.md, timings)

S2.total ≈ S2.meta[:sum_rule]                   # Σ weight = Tr[ϱ O¹⋯O^ℓ] (exact)
A = broaden(S2, ω) .+ broaden(psf(run, f, (d(:up)', d(:up)); grid = g), -ω)   # spectral function
```

Composite operators are ordinary operator expressions:

```julia
Hint = 0.5 * num(Mode(-1, :up)) * num(Mode(-1, :dn))
q = comm(d(:up), Hint)                          # q_↑ = [d_↑, H_int]
b = acomm(q, d(:dn)')                           # bosonic {q_↑, d†_↓}
x = d(:dn)' * cop(Mode(0, :dn))                 # impurity–site-0 operator d†_↓ f_{0↓}
psf(run, f, (q, x, x', q'); grid = g)
```

Options of `psf`: `exact = true` (unbinned Lehmann terms with exact NRG energy
differences, with a hard size guard), `abs = true` (histogram of `|weight|`),
`sparse`, `slicing = :lee | :left` (alternative trace form, V6), `schedule = :depth | :breadth`,
`hooks` (per-shell callbacks, see `hooks/`). Inhomogeneous operators (e.g.
`d↑ + d↓` with conserved `S_z`) go through `psf_multilinear`.

Built-in models: `siam`, `resonant_level`, `kondo`, `two_orbital` (with an
example `kanamori` interaction), or a user-defined `ImpurityModel(G, impurity,
Himp, channels; Hsite0)` with any finite local space, any local Hamiltonian, any
diagonal hybridization (`BoxHybridization`, `FunctionHybridization`,
`TabulatedHybridization`), and any abelian symmetry group `G` built from `U1` and
`Zn` factors (the fermion parity `Z2(:Pf)` is always added). See `examples/`.

Tests: `julia -t 2 --project=mpNRG/test mpNRG/test/runtests.jl` (2897 assertions, about 16 min on 2 threads).

## Conventions (task sheet Sec. 4)

1. **Units.** Energies are in units of the half-bandwidth `D = 1` unless the
   hybridization says otherwise. Temperature is passed as `T` (`β = 1/T`).
   Shell `n` is rescaled by `ω_n = D Λ^{-(n-1)/2}` only for truncation and flow
   diagrams; all energies passed to PSFs and FDM weights are **absolute**. Every
   shell stores its accumulated ground-state shift (`shell(run, n).shift`).
2. **Wilson chain** (Lee Sec. III A): impurity = site `-1`, bath sites `0..N`,
   on-site energies `ε_n`, `t_n` couples site `n` to `n-1`, so `t_0` is the
   impurity coupling. `chain_length_for(T)` chooses `N` with `Λ^{-N/2} ≪ T` and a
   margin. At `n = N` all states are discarded.
3. **Fermionic order.** The product basis is built "as if filling the chain in
   the order `c†_N ⋯ c†_{-1}|0⟩`" (Lee p. 5): `|σ_n⟩ ⊗ |s⟩^{n-1} := c†(σ_n)|s⟩^{n-1}`,
   the creators of the new site to the left of all earlier ones; within a site
   the modes are in the order of the local space definition. Two rules follow
   (derivation in `STATUS.md`): (R1) an operator of parity `p` on earlier sites
   acts as `Z_n^p ⊗ O`, `Z_n = (-1)^{N_n}`; (R2) an operator on site `n` acts as
   `O_n ⊗ 1`. Signs are always obtained from the `Pf` labels.
4. **PSF convention** (Kugler Eq. (28), Lee Eq. (5)). Operators are passed **in
   permuted order** `(O¹, …, O^ℓ)`; the cycle of eigenstates `(1, …, ℓ)` has weight
   `ρ₁ ∏ᵢ (Oⁱ)_{i,i+1}` (cyclic) at `εᵢ = E_{i+1} - E₁`, `i = 1..ℓ-1`. Fermionic
   permutation signs `ζ_p` belong to the kernels and are **not** applied.
5. **Logarithmic grid** (Lee p. 13): `ε[±m] = ±10^{(|m|-1)/n_dec} ε_min`,
   `ε[0] = 0`, `|ε[m]| ≤ ε_max`; each grid point is the centre of its bin on the
   log axis; the zero bin collects `|ε| ≤ ztol`. One `LogGrid` object is shared by
   all permutations and all `ℓ` of a run; combining histograms on different
   grids is an error.
6. **Histograms.** 2p dense; 3p dense (optional sparse); 4p sparse by default
   (`Dict` of nonzero bin triples). Weights are `Float64` for real models and
   `ComplexF64` when the model or an operator is complex.

## Approximations and where they enter

| Approximation | Where | Function |
|:--|:--|:--|
| Logarithmic discretization of the bath (Λ, z; Žitko–Pruschke energies) | `bath/` | `discretize`, `wilson_chain` |
| Truncation to the kept sector (global cut moved into a gap) | `nrg/` | `truncate_shell` |
| NRG approximation `H\|se⟩ ≈ E^n_s \|se⟩` (Lee Eq. (19)) | `fdm/`, `psf/` | `fdm`, `psf` |
| Shell-diagonal expansion of nonlocal operators (Lee Eq. (35)) | `psf/` | the engine |
| Binning: weights moved to bin centres (Lee Sec. V B) | `psf/` | `LogGrid`, `BinnedMode` |
| Slicing: sliced frequencies keep the resolution of their shell (Lee Eq. (60)); for the 4p KKDK term ε₂ is assembled as `centre(ε₂-ε₁) + ε₁` (Lee Eq. (71b)) | `psf/` | `composites` |
| Broadening of 2p data (log-Gaussian + Fermi, Lee App. C) | `observables/` | `broaden` |
| z-averaging | `io/` | `zaverage`, `zaveraged_psfs` |

Total PSF weights are **not** approximated beyond the FDM itself: for every
`ℓ`, `Σ weight = Tr[ϱ O¹ ⋯ O^ℓ]` with the FDM `ϱ`, to rounding (V4a).

## Exporting PSFs to `HubbardAtom` kernels

`src/` returns plain arrays (`histogram(r)`, `nonzero_bins(r)`, `r.positions`,
`r.weights`, `psf_dict(r)`), so it does not depend on `HubbardAtom`. The
converter lives in `test/test_hubbardatom.jl` and the notebook:

```julia
import HubbardAtom as HA
to_psfterms(r) = r.mode === :exact ?
    [HA.PSFTerm(r.positions[:, j], ComplexF64(r.weights[j]), Int[]) for j in eachindex(r.weights)] :
    [HA.PSFTerm([centres(r.grid)[i] for i in k], ComplexF64(v), Int[]) for (k, v) in nonzero_bins(r)]
cache = [HA.PermutedPSF(p, HA.permutation_sign(p, isfermionic), to_psfterms(r)) for (p, r) in psfs(run, f, ops; grid = g)]
G = HA.correlator(model, ops_dense, matsubara_freqs; cache = cache)   # HubbardAtom's kernels
```

## Layout

```
../TNOps/       (sibling package) abelian labels, legs, BlockTensor, SlicedTensor, fusion, factorizations
src/nrg/tensors.jl  the named NRG contractions on TNOps tensors (the only tensor operations of mpNRG)
src/fock/       modes, symbolic operator algebra, Jordan-Wigner dense matrices, local spaces
src/bath/       hybridizations, logarithmic discretization, BigFloat Lanczos, star of a finite chain
src/model/      ImpurityModel, symmetry checks, built-in models
src/nrg/        forward sweep, global truncation, shell storage, dense embeddings
src/operators/  shell projections of local operators, the KK refinement map
src/fdm/        shell weights, reduced density matrices, thermal averages
src/psf/        grid, the generic ℓ = 2, 3, 4 engine (slicing, nested refinement), API
src/observables/ thermodynamics (FDM and shell-wise), 2p broadening
src/io/         JLD2 output with full metadata, z-averaging
src/hooks/      per-shell callback interface
src/reference/  dense ED of finite Wilson chains (independent of nrg/)
```

## Changes in the TNOps refactor

The abelian block-sparse layer moved out of `src/symmetry/` into the sibling
package `TNOps` (rank-k tensors with tagged, oriented legs; see
[`../TNOps/README.md`](../TNOps/README.md)), and every tensor operation of
`mpNRG` goes through the named functions of `src/nrg/tensors.jl` (refactor
sheet [`docs/TNOPS_REFACTOR.md`](docs/TNOPS_REFACTOR.md)). Results are
unchanged up to round-off (regression baseline `test/baseline/`, notebook
section "Refactor regression (TNOps)"). For code that uses the package:

- **Re-exported names keep working.** `SectorSpace(Dict(q => d))` builds an
  untagged `Leg` (`SectorSpace === Leg`); `BlockOp{T,S}` is an alias of
  `BlockTensor{T,2,S}` and `BlockOp{T}(row, col, Δ)` still constructs an
  empty operator; `A.shift`, `A.row`, `A.col`, `target(A, q)`,
  `getblock(A, q)` / `setblock!(A, q, M)` (keyed by column label), `todense`,
  `*`, `+`, `'`, `tr`, `norm`, `restrict` behave as before;
  `SlicedBlockOp{T,S,Kt}` is an alias of `SlicedTensor{Kt,T,2,S}` and
  `mapslices_op(f, A) = mapslices(f, A)`; `blockeigen`, `blocksvd`,
  `identity_op` and `kron_op` are thin wrappers (of `eigen`/`svd` per label,
  `identity_tensor`, `kron_fused`); `FusionMap`, `FusionPair`, `fusionmap`
  are the generic two-leg fusion of TNOps with the same layout (outer leg
  slow, inner fast).
- **Block dictionaries are keyed by tagged sectors.** `A.blocks` is now a
  `Dict` from tuples of `LegSector`s to arrays; code that indexed
  `A.blocks[q]` by label must use `getblock(A, q)` (all tags of the label,
  kept states first) or iterate `eachblock(A)`.
- **Shells.** `Shell.A` is the isometry `A^n` as a rank-3 `BlockTensor` with
  legs (old: the kept leg of shell `n-1`, site: the local leg, new: the shell
  leg), rank 2 at `n = -1`; `Shell.Adag` is its adjoint and `Shell.leg` the
  shell leg, on which every label `q` carries the tagged sectors `(q, K)`
  (tag 1, `nkept` states) and `(q, D)` (tag 2). `Shell.F` holds rank-2
  tensors on `Shell.leg`; `Shell.fusion` is a TNOps `FusionMap` (`.loc`/`.outer`
  the site leg, `.prev`/`.inner` the kept leg, `.out` the product leg;
  `pairs` keyed by `LegSector`, with `p.σ`, `p.q` now `LegSector`s). `shell`, `keptspace`, `discspace`, `krange`, `drange`, `xrange`,
  `nkept`, `ndisc` and `sector_view` keep their signatures and meanings;
  `Shell.space` is an untagged leg with the dimensions of the old
  `SectorSpace`.
- **Operators.** `shell_operator`, `refine` and `project` return rank-2
  tensors on the shell leg (`refine` also accepts a `SlicedTensor` and acts
  slice by slice); `sector_view(run, O, n, q, X, X̄)` is
  `block_view(O, q, X, X̄)`, the stored tagged block or `nothing`.
- **FDM.** `rhoK(f, n)` is a tensor on the kept leg of shell `n` (it was a
  `Dict` of matrices); `getblock(rhoK(f, n), q)` gives the old matrix.
  `rhoD`, `weight`, `rho_full(f, run, n, q)` are unchanged.
- **Hooks.** `isometries(ctx)` returns the rank-3 isometry, `rhoKK(ctx)` the
  kept-leg tensor, `call_operators(ctx)` rank-2 tensors, `carried(ctx)`
  composites whose `slices` are `SlicedTensor`s; `fusionmap(ctx)` and
  `refinement_map(ctx)` keep their roles (memory layout: the docstring of
  `ShellContext`).
- **Determinism.** `Mode`, `Ladder`, `Sector`, `LegSector` and `Arrow` have
  content hashes, so dictionary orders, and with them summation orders and
  round-off, no longer change between builds (findings R5c and R6 in
  `STATUS.md`). `TNOps.eigen` uses LAPACK's divide-and-conquer driver
  (finding R5b).
- **Output files** of `save_results` are unchanged; files written by the
  pre-refactor code load as before (`test/test_io.jl`).

