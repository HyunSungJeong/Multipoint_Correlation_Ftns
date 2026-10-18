# mpNRG — status

Task sheet: [`docs/TASK.md`](docs/TASK.md). Package: `MultipointNRG` (this folder).

## Environment and bookkeeping

- **Branch.** The task sheet names `claude/practical-archimedes`; this session was
  configured to develop on `claude/cool-sagan-txmva3` and must not push
  elsewhere without explicit permission. Both branches pointed at the same
  commit `541d40a` when the work started, so the history here applies to either.
  All commits are on `claude/cool-sagan-txmva3`.
- **Julia.** `julialang-s3.julialang.org` and `pkg.julialang.org` are blocked by
  the environment's network policy. Julia 1.13.0 was installed from conda-forge;
  packages come from the General registry via git (`JULIA_PKG_SERVER=""`).
  `julia = "1.11"` is the declared minimum; everything was run with 1.13.0.
- **Papers.** Downloaded from arXiv into `refs/` (gitignored): PDFs and the
  **LaTeX sources** of Lee 2021 (2101.00708), Kugler 2021, Weichselbaum & von Delft
  2007, Žitko & Pruschke 2009, Lee & Weichselbaum 2016, Weichselbaum 2012,
  Bulla, Costi & Pruschke 2008, Frankenbach et al. 2025, and others. Following the
  `academic-pdf-reading` skill ("source outranks rendering"), every equation,
  sign, index and overbar was read from the LaTeX source, never from a PDF text
  layer. (Equation numbers below are the printed ones, Lee 2021 printed page =
  PDF page.)
- **External data (V8).** LMU Open Data 613 (Frankenbach et al.), dataset
  `siam05_U0.05_T0.05_Delta0.0318`, downloaded into `refs/lmu613/` and extracted by
  `scripts/extract_lmu613.py` into a plain HDF5 file read with JLD2. Dataset:
  "mpNRG PSFs used in: Computing and compressing local vertex functions ... using
  quantics tensor cross interpolation", LMU Open Data,
  [doi:10.5282/ubm/data.613](https://doi.org/10.5282/ubm/data.613), license
  CC BY-SA 4.0. The committed `notebooks/figures/mpnrg/v8_results.jld2` contains
  arrays derived from it (its 2p PSFs and the marginals and totals of its 4p
  PSFs, next to ours) and is distributed under the same license.

## Phase log

| Phase | Content | Commit |
|:--|:--|:--|
| P0 | skeleton, Project.toml, test env (HubbardAtom path source), conventions | `mpNRG P0` |
| P1 | `symmetry/`, `fock/` | `mpNRG P1` |
| P2 | `bath/` | `mpNRG P2` |
| P3 | `model/`, `nrg/`, `reference/` | `mpNRG P3` |
| P4 | `operators/` | `mpNRG P4` |
| P5 | `fdm/`, thermodynamics | `mpNRG P5` |
| P6–P8 | `psf/` (one generic engine for ℓ = 2, 3, 4), `io/`, z-averaging, 2p broadening | `mpNRG P6–P8` |
| P9 | `hooks/`, examples, notebook, root README rows | `mpNRG P9` |
| P10 | timing table and performance notes | `mpNRG P10` |

The code was developed in one pass and the history was then split into these
commits; each commit contains the files of its phase and its tests pass on
that commit. P6, P7 and P8 are a single commit: the design requirement of Lee
p. 18 ("one 2p and one 3p routine ... irrespective of how their input operators
have been generated", refinements "through nested function calls") is met by a
single recursive routine for all ℓ, so there is no separate 2p, 3p and 4p code
to commit separately. The notebook was written and executed at P9 (it has one
section per validation item). The performance work (threaded depth-first
schedule, dense accumulation of histograms, grouping of composites by their
key) was done during development, before the split, so it is part of the P6–P8
code; the P10 commit holds the measured timing table and the notes on cost.
Test file moves forced by the split: the V1d chain test needs `project` and so
runs from P4 on; V1c builds its symmetry group directly instead of through the
P3 model layer. The P6–P8 commit carries a stub for the hook dispatcher (no-op
without hooks), replaced by `hooks/` at P9.

## Equations implemented

Lee = Lee, Kugler & von Delft, PRX 11, 041007 (2021). Read from the LaTeX source.

- **P2.** Žitko & Pruschke PRB 79, 085106 (2009), Eq. (22) (`E_1^z` for the box)
  and App. B, Eqs. (B1)–(B4) (ODE for an arbitrary Δ); Campo & Oliveira;
  Wilson's prescription (Bulla RMP 80, 395). Lanczos tridiagonalization of the
  star (Bulla, Pruschke & Hewson 1997). Lee Sec. III A indexing.
- **P3.** Lee Eq. (15) (shell eigenstates as MPS from the kept sector, the
  `A`/`M` isometries), Eq. (16) (MPS form), Sec. III A truncation, `n₀` (p. 5).
- **P4.** Lee Eq. (21) (AS overlaps), Eqs. (22)–(27) (shell projectors,
  refinement identity, projector products), Eqs. (28)–(31) (local operators,
  shell-diagonal projections), Eq. (31) recursion = the KK refinement map,
  Eqs. (32)–(34) (top-down expansion; exact for local operators, p. 9).
- **P5.** Lee Eqs. (40)–(45) (FDM in the AS basis with Boltzmann weights,
  sector projections, reduced density matrices, shell weights `w_n`, backward
  sweep), Eqs. (46)–(48) (thermal averages through shell traces), Eq. (50)
  (expansion of operator products) as an independent route; Weichselbaum & von
  Delft PRL 99, 076402 (2007), Eq. (8). Thermodynamics: Krishna-murthy, Wilkins
  & Wilson PRB 21, 1003 (1980).
- **P6–P8.** Lee Eq. (5) / Kugler Eq. (28) (PSF), Eqs. (11)–(12) (nested form),
  Eqs. (55)–(59) (2p), Eq. (60) and Sec. V B (binning, slicing, grid), Eqs.
  (61)–(66) (3p: seven sectors, KKD/KDK through 2p PSFs of composites, Eqs.
  (64b), (65b), (66a), (66b)), Eqs. (67)–(71) (4p: 15 sectors, KKDD/KDKD/KDDK
  through 2p PSFs, Eqs. (70a)–(70c), KKKD/KKDK/KDKK through 3p PSFs, Eqs.
  (71a)–(71c)), App. C, Eq. (C1) (log-Gaussian + Fermi broadening of 2p data).

## Derivation of the fermionic rules (Sec. 5.2.5)

Convention (Lee p. 5): `|σ_n⟩ ⊗ |s⟩^{n-1} := c†(σ_n) |s⟩^{n-1}`, where `c†(σ_n)` is
the ordered product of the creators of site `n` occupied in `σ_n`, standing to the
**left** of all creators of earlier sites. Let `N_n(σ_n)` be the number of
fermions in `σ_n`, `Z_n = (-1)^{N_n}`.

- **(R1)** Let `O` act on sites `< n` only, with definite parity `p_O` (it is a
  sum of words with `p_O` mod 2 ladder operators). Moving a single ladder
  operator `a` of an earlier site through `c†(σ_n)` costs one sign per creator,
  `a c†(σ_n) = (-1)^{N_n} c†(σ_n) a`; for a word of `p_O` such operators the
  sign is `(-1)^{p_O N_n}`. Hence
  `O (|σ_n⟩ ⊗ |s⟩) = (-1)^{p_O N_n} c†(σ_n) O|s⟩`, i.e. `O ↦ Z_n^{p_O} ⊗ O`.
- **(R2)** An operator `O_n` acting on site `n` only stands to the left of the
  earlier creators already: `O_n c†(σ_n)|s⟩ = (O_n c†(σ_n)|0⟩_n)`-part times
  `|s⟩`, i.e. `O_n ↦ O_n ⊗ 1`, with the matrix of `O_n` in the local basis.
- Consequently the hopping `f†_{nν} f_{n-1,ν}` is `(f†_{nν} ⊗ 1)(Z_n ⊗ f_{n-1,ν}) =
  (f†_{nν} Z_n) ⊗ f_{n-1,ν}`, assembled by `kron_op` from block operators.
- Within a site the creators are ordered as in the local-space definition; the
  dense Fock basis of sites `n, …, -1` (mode list `[site n; …; site -1]`) then
  factorises as `kron(site n, …, site -1)`, which is the "documented basis
  permutation" used in V1: the dense embedding of shell `n` is
  `U_n = (product embedding) · A_n`.
- **AS matrix elements.** For the full chain, `|se⟩ = c†(e)|s⟩` with the
  environment creators to the left, so by (R1) an operator of parity `p` on
  sites `≤ n` has `⟨se|O|s̄ē⟩ = δ_{eē} (-1)^{p N_e} [O^n]_{ss̄}`. Lee Eq. (30)
  writes `δ_{eē}`; the sign cancels in every trace (traced products are even),
  so all formulas are unaffected. Verified in V4b with the explicit AS basis.

Verified in V1d (all mode operators of a 3-site chain through the shell
construction vs an independent Kronecker-product Jordan-Wigner construction) and
in V4b.

## Verification record

Tests: `julia --project=mpNRG/test mpNRG/test/runtests.jl` (4 threads: `-t 4`).
"≤ x" means the largest deviation over all cases of the item.

Numbers from the executed notebook (`notebooks/mpnrg_validation.ipynb`, section
per item) and the `@info` lines of the tests. The tests assert the exactness
items (V1–V4) at 1e-12 to 1e-10 depending on the quantity (each tolerance is
stated at its assertion in `test/`), the physics items (V5–V8) at the
tolerances quoted in the findings below. Full suite: **2897 assertions, all passing** (Julia 1.13.0,
16 min on 2 threads; V8 is skipped without the dataset).

| Item | Check | Achieved |
|:--|:--|:--|
| V1a | labels: group axioms, canonical forms, duals | exact |
| V1b | block products / traces / eig / svd vs `todense` | 7.9e-18 rel. / 9.7e-16 |
| V1c | shifts and parity of composites `[d,H_int]`, `{q,d†}`, `d†f₀` | exact |
| V1d | 8 mode operators of a 3-site chain through the shells vs an independent Kronecker JW; `{c_i,c_j†} = δ_ij`; `H_N` from blocks | 1.4e-15; 6.7e-14; 1.9e-14 |
| V1e, V1f | a symmetry-breaking term is detected; truncation never splits a multiplet spread over blocks | yes; yes |
| V2a | `Σ\|V_k\|² = (1/π)∫Δ`, all z, three schemes | 0 (to rounding) |
| V2b | Žitko–Pruschke ODE (App. B, RK4, h = 1/512) vs closed form Eq. (22) | 1.3e-13 |
| V2c | chain moments vs star moments up to order 2N+1 (BigFloat) | 1.5e-154 |
| V2d | two-precision check, 256 vs 512 bits; `t_n Λ^{n/2}` → const | 7.1e-77 |
| V3a | shell spectra without truncation vs reference ED; `ln Z` | 5.3e-15; 1.2e-14 |
| V3b | thermal averages vs ED; vs `HubbardAtom.AIMModel` on the star (2p, 3p, 4p) | 4.3e-15; 4.7e-14 |
| V3c | PSFs without truncation vs ED, all permutations, ℓ = 2, 3, 4 (binned / exact positions) | 3.4e-15 / 1.1e-14, 2.3e-15 / 8.5e-15, 3.1e-15 / 1.4e-14 |
| V3c | atom limit `V → 0` vs `HubbardAtom` exact atom PSFs | 2.5e-14 |
| V4a | `Σ weight = Tr[ϱ O¹⋯Oˡ]` with truncation, ℓ = 2, 3, 4 (FDM and brute force in the AS basis) | 6.3e-15 |
| V4b | shell-diagonal expansion reassembles local operators (explicit AS basis) | 1.4e-15 |
| V4c | `Tr ϱ = 1`, `ρ̄ⁿ` = partial trace of `ϱ`, `ρ̄ ≥ 0` | 8.6e-16; min eigenvalue ≥ 0 (3e-41) |
| V4d | decoupled tail: truncated NRG = ED, positions and weights (refinements carry up to 64 % of `\|weight\|`) | 2.1e-14 |
| V4e | independent dense brute force of Lee's recursion with truncation (mutation check: a perturbed refinement fails) | ≤ 1e-13 |
| — | selection rules: pruned exactly where the shift arithmetic says; Z2 vs U1×Z2 vs U1×U1×Z2 give identical PSFs | exact; dropped `\|w\|` ≤ 6e-11 at `wtol = 1e-14` |
| V5 | RLM: per-decade weights vs the exact discrete model; `⟨n⟩` | 7.1e-7; 2.9e-8 |
| V5 | broadened `A(ω)` vs continuum (single z, b = 0.6 / `n_z = 4`, b = 0.3) | 0.34 / 0.097 (test, asserted ≤ 0.12) and 0.13 (notebook, \|ω\| ≤ 0.25) |
| V6a | `:lee` vs `:left` trace forms: totals; per decade; `:depth` vs `:breadth` | 1.6e-15; ≤ 1.7 % of the total (3p ≤ 1.4 %); 4.3e-18 |
| V6b | `⟨{d,d†}⟩ = 1` from PSF totals for z = 0.25…1 | 3.6e-15 |
| V6c | test-function distance to ED vs `N_keep` = 8, 16, 32, 64, 128, ≥ 256 (2p) | 0.29, 0.076, 0.040, 5.4e-3, 7.2e-4, 1.8e-15 |
| V7 | SIAM entropy ln 4 → ln 2 → 0 | 1.3814, 0.6831, 1.1e-5 |
| V7 | `T χ`: 1/8 → 1/4 → 0 | 0.1335, 0.2068, 3.9e-6 |
| V7 | `⟨n_d⟩ = 1` at particle-hole symmetry; Friedel `πΔA` plateau / smallest ω | 1.1e-15; 0.995 / 0.956 |
| V7 | `U = 0`: `S₄(ε₁,0,ε₃) = S₂(ε₁) S₂(ε₃)` (binned, truncated) | 4.1e-4 of max |
| V8 | LMU 613: 2p totals (8 PSFs incl. composites `q = [d,H_int]`) | ≤ 7e-6 |
| V8 | 2p bins, dataset's "bary" binning / Lee's centre binning (`N_keep` = 400, 800) | 7–15 % / 22–39 % of max bin |
| V8 | connected 4p totals, all 24 permutations × 2 spin flavors | 1.5e-4 abs., 0.4 % rel. |
| V8 | ε₁, ε₃ marginals of connected 4p PSFs, σσ′, 24 permutations (`N_keep` = 100, smoothed) | ≤ 10.8 %, = the 2p-level difference (9.8 %) |
| V8 | σσ: connected marginals vanish exactly (single mode) | ≤ 1.8e-3 of the σσ′ scale (dataset ≤ 1e-16) |
| V8 | our 4p marginal identity `Σ_{ε₂,ε₃} S[O¹..O⁴] = S[O¹, O²O³O⁴](ε₁)` (and ε₃) | 7.9e-4 (ε₁), 3e-14 (ε₃) |
| export | `G(iν)` through `HubbardAtom` kernels from exported PSFs vs `HubbardAtom` | 5.5e-14 |


## Performance (P10)

Notebook, section "Performance": SIAM `U = 0.2`, `Δ = 0.04`, `T = 1e-4`, `Λ = 4`,
`N = 18`, grid `n_dec = 8`, one z, 4 threads, abelian `U(1)×U(1)×Z₂`
(`N_keep` counts states). Operators `d↑, d†↑` (2p), `+ n↓` (3p),
`d↑, d†↑, d↓, d†↓` (4p, one permutation).

| `N_keep` | sweep [s] | 2p [s] | 3p [s] | 4p [s] | 4p cost: shell traces / composites / slice refinements |
|--:|--:|--:|--:|--:|:--|
| 25 | 0.18 | 0.01 | 0.28 | 29.8 | 249,503 / 1,917 / 373,258 |
| 50 | 0.14 | 0.01 | 0.67 | 70.6 | 320,227 / 1,917 / 453,614 |
| 100 | 0.19 | 0.02 | 1.51 | 248 | 313,028 / 1,725 / 631,569 |

Other measured costs: the V8 4p marginals (`N = 12`, `N_keep = 100`, dense
3D histograms on the dataset's grid) took 80–200 s per PSF on 2 threads; the
SIAM example (`T = 1e-3`, `N_keep = 100`, `n_dec = 16`) 22 min on 2 threads, of
which the 4p PSF is almost all; the full test suite 16 min on 2 threads.

Where the time goes: 2p and 3p are cheap (`~N` shell traces, `~N²`
refinements); the 4p KKKD/KKDK/KDKK strings need a 3p PSF of composites per
shell (`~N³`, Lee p. 18), and each slice refinement is a block product whose
cost grows with `N_keep`. Implemented: threads over the independent composite
PSFs (depth-first schedule), grouping of composites with equal keys, dense
accumulation when the histogram fits (`nb^ℓ ≤ 2²⁵` bins). Not implemented:
SU(2) (open issue 1), compression of carried slices (the hook interface
exposes their singular values, see the notebook).

## Deviations from Lee 2021 and why

1. **One generic engine.** Lee describes 2p, 3p and 4p separately and asks for
   one 2p and one 3p routine called in nested fashion. Here one recursive
   routine handles any number `m` of carried operators: at each shell it
   evaluates the sector strings with `X₁ = D` or a single K as shell traces and
   turns every other string into the PSF of the composites between consecutive K
   states, evaluated by the same routine over the later shells. For `ℓ ≤ 4` this
   reproduces Lee's case list exactly (3p: KKD, KDK; 4p: KKDD, KDKD, KDDK via 2p,
   KKKD, KKDK, KDKK via 3p) and Lee's choice of slicing references (composites
   starting at state 1 sliced from the left, those ending at state 1 from the
   right, the KKDK middle composite relative to its left endpoint, ε₂ − ε₁, Lee
   Eq. (71b)).
2. **Slicing storage.** `SlicedBlockOp` and the engine store slices as a sparse
   map from bin key to block operator (only bins with weight). The dense
   `rows × cols × bins` alternative of the task sheet was not implemented: with
   the per-row evaluation used here the sparse map is what is filled.
3. **`|weight|` histogram.** A weight is one `(shell, sector string, s₁, s_ℓ,
   slice)` contribution, the finest decomposition the sliced evaluation
   provides; internal states sharing all bins are summed before the modulus.
4. **Zero bin.** `ztol = min(1e-14, ε_min/10)` by default (above the rounding
   noise of differences of degenerate energies at the early shells).
5. **Alternative trace form (V6).** `slicing = :left` slices every composite
   relative to its left endpoint. With truncation it is a *different*
   approximation from Lee's form (ε_i is assembled from a shell-n offset and a
   later-shell energy instead of being sliced directly); totals agree exactly,
   positions to NRG accuracy, and both are exact when the NRG approximation is
   (V4d).
6. **Additional exactness tests** beyond the task sheet: V4d (decoupled tail:
   Lee's formulas are exact with truncation, positions included) and V4e (an
   independent dense brute-force implementation of the recursion in the
   explicit AS basis, with a mutation check).

## Findings

1. **Projection cache keyed by `objectid` (bug, fixed).** The first version
   cached shell projections of operator expressions by `objectid`. Julia reuses
   object ids of collected expressions, so a later, different operator could hit
   a stale entry (wrong PSFs and, in one case, a `BoundsError`). The key is now
   the normal-ordered term list plus the shell, `(:proj, normalorder(expr).terms, n)`.
2. **Environment parity in the explicit AS basis.** Lee Eq. (30) writes
   `⟨se|O|s̄ē⟩ = δ_{eē} [O^n]_{ss̄}`. In the fermionic order of Lee p. 5 an
   operator of parity `p` picks up `(-1)^{p N_e}` (derivation above). The first
   version of the V4b test omitted it for odd operators and failed; the
   production code never needs it because every trace is over an even product.
   The test now embeds `kron(Z_env^p, M)`.
3. **V3c needs the untruncated limit.** With truncation, NRG PSFs are not exact
   even for the atomic limit of the impurity, because which states are kept
   depends on the impurity; exactness tests therefore run without truncation
   (V3c) or on a decoupled tail where Lee's formulas are exact with truncation
   (V4d).
4. **Zero bin.** Differences of degenerate energies at the early shells carry
   rounding noise of order `1e-15`; the default `ztol = min(1e-14, ε_min/10)`
   puts them in the zero bin. Weights below `wtol` can be dropped (default 0);
   the dropped `|weight|` is reported in `meta` (V3: ≤ 6e-11 for `wtol = 1e-14`
   in the 3p tests).
5. **Friedel sum rule near ω = 0.** `πΔA(ω)` shows a dip at the lowest
   frequencies that is present at `U = 0` too, so it is a discretization and
   broadening artifact, not interaction physics; V7 tests the plateau
   log-averaged over `nz = 4` (0.995) and the value at the smallest frequency
   (0.956).
6. **V5 needs z-averaging.** The broadened single-`z` resonant-level spectrum
   (b = 0.6) deviates from the continuum by up to 34 %; with `nz = 4` and
   b = 0.3 the deviation is 9.7 %. The per-decade weights against the exact
   discrete model agree to 7e-7.
7. **Trace forms (V6).** `slicing = :left` is a different approximation with
   truncation; totals agree to 1e-13, per-decade weights to ≤ 1.7 % of the
   total (4p; ≤ 1.4 % for 3p), asserted ≤ 5 %.
8. **V8 binning.** The LMU 613 data are binned "bary": each weight is split
   between the two neighbouring grid points linearly in `ln|ε|`. With Lee's
   centre binning our 2p histograms differ from the dataset by 22–39 % of the
   largest bin; with the same "bary" rule by 7–15 %. Totals agree to 1e-5
   (2p) and 4e-3 relative (connected 4p). The dataset's `Rhoin_max` is `Δ/π`;
   its README gives `Δ = 0.0318`.

## Open issues

1. **Non-abelian symmetries.** Only abelian symmetries (U1, Z_n, parity) are
   implemented. SU(2) spin symmetry would reduce the cost per kept state by
   roughly the multiplet dimension; V8 compares abelian `N_keep` states with
   the dataset's 300 SU(2) multiplets (about 900 states).
2. **Dense slice storage.** The `rows × cols × bins` slice layout of the task
   sheet was not implemented (deviation 2).
3. **`|weight|` histogram.** Its value depends on the decomposition into
   weights (deviation 3); it is a diagnostic of cancellations, not a
   physical quantity.
4. **V8 4p comparison.** The 4p comparison uses the connected totals (all
   48 PSFs) and the `ε₁` and `ε₃` marginals of the connected PSFs (all 48,
   `N_keep = 100`, both z), not the full three-dimensional histograms: the
   dataset bins in three variables with its own rule. The dataset's marginals
   are exactly combinations of its own 2p PSFs (to 1e-14), e.g.
   `Σ_{ε₂,ε₃} S_con[d↑,d†↑,d↓,d†↓] = S[d↑, d†↑(1-n↓)] - S[d↑,d†↑]⟨d↓d†↓⟩`, and
   ours agree with the same 2p combination of our 2p PSFs to 8e-4, so the
   11 % marginal difference is the 2p-level difference of item 5, not a 4p
   issue. The bin-by-bin 3D comparison remains open.
5. **Residual V8 2p difference.** The 7–15 % bin-wise difference after
   identical binning does not decrease from `N_keep = 400` to 800 states, so
   it is not our truncation. Candidates, not resolved: details of the
   dataset's discretization and z-grid, its SU(2) truncation (300 multiplets),
   or its handling of the lowest shells. Totals, which do not depend on these,
   agree.
6. **Cost of 4p PSFs.** The 4p terms that need a 3p PSF at every shell scale as
   `N³` in the chain length (Lee p. 18) and dominate the run time; see the
   timing table.
