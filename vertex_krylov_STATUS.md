# vertex_krylov — status (Steps 1 and 2: phases V0–V4)

Discrete Krylov representation of the Matsubara regular part of the 3p vertex,
[VD] Algorithm 2. **All results live in the notebook**
[`notebooks/vertex_krylov.ipynb`](notebooks/vertex_krylov.ipynb) (executed, with
outputs, executed from a fresh kernel; [`notebooks/vertex_krylov.pdf`](notebooks/vertex_krylov.pdf)
alongside; rebuilt from `notebooks/build_vertex_krylov_notebook.py`). Its Section 0 holds
the full test matrix of Steps 1 and 2 (532 rows), Section 6 the V3c results, Section 7
the findings with computed evidence. This file holds only
the phase log, the deviations from the task sheet, the timings and the verdict.

## Phase log

| phase | content | commit |
|:--|:--|:--|
| V0+V1 | environment; `src/vertex_krylov.jl` (data layer, Stages 1–5, assembly, reference estimator, checks C1–C8); `test/test_vertex_krylov.jl` (414 assertions) | `8ee954c` |
| V2 (in progress) | `mpNRG/test/vertex_krylov_adapter.jl`, `mpNRG/test/test_vertex_krylov_nrg.jl`, notebook builder | `eb0d8bc` |
| V2, V3a, V3b | Wilson-chain ED (exact and binned), N = 2 block test with field, stabilization, Zacinskis check, untruncated NRG, decoupled-tail truncated NRG; performance and memory fixes | `da7a2c7` |
| V4 (Step 1) | notebook Sections 0–5 and 7 with outputs, PDF, figures | `48fea2b` |
| V4 (Step 1) | this file, README row block | `3201add` |
| amendment | task-sheet amendment A1–A3 after the Step-1 review: stabilized chain as the Stage-2 default for exact data, amended criteria in tests and notebook builder, Step-1 verdict regenerated | `5975e8f` |
| V3c | physical chain, truncated NRG: `mpNRG/scripts/vertex_krylov_v3c.jl` (data for 16 cases, cached), adapter (`vk_v3c_data`, `vk_case_v3c`, `vk_G0_chain`), Stage-1 option `gram = :general` and anti-Hermitian report, V3c code-path test, notebook Section 6 in the builder | `1c89b05` |
| V4 (Step 2) | notebook Sections 0–7 re-executed from a fresh kernel (amended criteria, V3c rows), PDF, this file finalized, README row block | see `git log` |

V0 and V1 were developed together in one file and committed together.

## Task-sheet amendment (after the Step-1 review)

The reviewer classified finding F1 as a mis-specification of the task sheet, not a
failure: for exact data with a bath, C4 is guaranteed only for the stabilized chain
([VD] Prop. 13). The task sheet is amended as follows; the code, the tests and the
notebook builder implement it.

- **A1 (C4).** C4 of the unstabilized particle chain (Π = 1 − vv†) is "report". For
  exact data the guaranteed C4 (tolerance 1e-9 in V2a, 1e-11 for the atom, V1) is
  that of the stabilized chain (Π = 1 − ΩΩ†).
- **A2 (Stage 2).** For exact data with a bath the stabilized particle chain is the
  default output of Stage 2 (`HA.vertex_krylov(d; stabilize = :auto)` stabilizes when
  `d.meta[:grid_kind] === :exact`; without a bath Ω = v and nothing changes). The
  unstabilized chain is kept as a diagnostic (`vk_run` → `out[:unstab]`: C4, C1,
  length, leakage profile, and the change of Σ^S and K̃). For binned and NRG data
  Prop. 13 does not apply and Π = 1 − vv†.
- **A3 (C2).** The invariants of the Zacinskis route (Σ^S, spectra and first-block
  weights of Ω′, S = N̄², start block) are guaranteed at 1e-10. The Jacobi-block
  agreement (`n_agree`, finding F2) is "report".

## Deviations from the task sheet

1. **Environment.** The setup script had installed Julia and Jupyter but no Julia
   registry, no packages and no IJulia, and the network policy blocks the Julia
   package servers (`pkg.julialang.org` redirects to `storage.julialang.net`: 403).
   Worked around without changing any project file: the General registry was cloned
   with git from GitHub, packages were installed with `JULIA_PKG_SERVER=""` (git
   clones and GitHub release artifacts), and IJulia was added to the default
   environment (kernel `julia-1.13`). The notebook's first cell activates
   `mpNRG/notebooks`.
2. **PDF rendering.** No LaTeX is installed and the MathJax CDN (`cdnjs.cloudflare.com`)
   is blocked. The PDF was printed from the nbconvert HTML export with the
   pre-installed Chromium, with MathJax 2.7.9 served locally from the npm registry,
   and a print stylesheet for the wide text tables.
3. **Atom level position.** `HubbardAtomModel` pins `εd = -U/2`; the task needs
   `εd = -0.3`. Added `atom_hamiltonian` / `atom_spectrum(m; εd, h)` (same gauge
   fixing as `spectrum`); `HubbardAtomModel` is unchanged. `discrete_data_atom`
   takes `εd`, `h` and checks that `sp` diagonalizes that Hamiltonian.
4. **C8 normalization when `K^direct ≡ 0`.** In the atom's (↑↑) channel
   `B = q_{↑↑} = U n_↓` commutes with `H`, `d_↑` and `q_↑`, so every correlator
   `G[B, x, y]` is independent of `τ₁` and its regular part vanishes at `ω12 ≠ 0`:
   `K^(12) ≡ 0` (max |K^direct| = 2e-16) while the six permutation terms are O(1)
   and cancel (also with operator-level `regular_sum`). The ratio
   `max|ΔK| / max|K^direct|` is then 0/0. `check_C8` uses
   `max|ΔK| / max_p max|K_p^direct|` when `max|K^direct| < 1e-10 · max_p|K_p|`
   (flag `degenerate`, marked `†` in the summary; only these two atom rows). It
   also reports the per-permutation error, which is asserted everywhere.
5. **Sparse 3p storage.** On an ExactGrid the Wilson-chain data have up to 12 163
   bins, so the six 3p families are stored as sparse `(α, β) => block`
   dictionaries (`s1[(α,β)][a,b]` instead of `s1[α,β][a,b]`, `N×2N` rows for
   families 3/5 and `2N×N` columns for 4/6 so that `N > 1` works), and the kernels as
   block-sparse matrices. Content as in Sec. 2.4.
6. **C2 split** (now amendment A3). `check_C2` returns the largest deviation of the
   invariants of the Zacinskis route (`S - N̄²`, start block, spectra and first-block
   weights of `Ω′`, `Σ^S`; ≤ 1.3e-12 everywhere); the Jacobi-block comparison is
   reported separately as a profile and as `n_agree` (finding F2).
7. **BigFloat.** The particle chain is run in BigFloat (precision 128) for the atom
   and the V2a chains with `Nch ≤ 1`. For `Nch = 2` on the ExactGrid
   (R = 2616–3840 coordinate rows, L ≈ R steps with full reorthogonalization,
   ~10¹¹ BigFloat operations) it was skipped. `DoubleFloats` /
   `GenericLinearAlgebra` are not in any environment, so `N > 1` in high precision
   is skipped as well.
8. **Extra cases.** Besides the matrix of Sec. 6: an `N = 2` atom in a rotated,
   complex orbital basis (`K` non-diagonal and asymmetric, so the `p₂` transposition
   (VD8) and the contractions of (VD5) are exercised non-trivially; with spin
   conservation `K` is diagonal and the transposition is a no-op); and the 16-state
   chain (`Nch = 0`) as the smallest reproducing case of F1. V3b, run as in mpNRG
   V4d, has no field variant.
9. **Pre-existing test failure.** `test/test_psf_lanczos.jl` (not touched by this
   task) fails one assertion on this Julia 1.13 build (`c.R_ref == (244, 244)`
   evaluates `(242, 242)`; it counts eigenvalue splittings at the 1e-15 level).
   It fails identically on a clean checkout of the branch head. Because
   `test/runtests.jl` stops at the first failing file, `test_vertex_krylov.jl` is
   included before it.
10. **NRG test not in the default suite.** `mpNRG/test/test_vertex_krylov_nrg.jl`
    takes 90 min and needs up to ~8 GB (a first run was OOM-killed while the
    notebook ran in parallel). It is not in `mpNRG/test/runtests.jl`;
    `VK_FAST=1` restricts it to `Nch = 1` (5 min).

11. **V3c grid.** `LogGrid(emin = 1e-9, emax = 10, ndec = 16)` instead of `emin = 1e-5`:
    the V3c chains (N = 15, 22) resolve energies far below 1e-5, and with `emin = 1e-5`
    one 3p PSF at `T = 1e-4` has 5.1e6 (`N_keep = 64`) to 2.7e8 (`N_keep = 512`)
    clamped contributions. With `emin = 1e-9` the engine still clamps positions between
    `ztol` = 1e-14 and the first bin (~9e-10); their counts are reported per case
    (`meta[:nclamped]`, notebook Sec. 6; the engine records counts, not weights).
12. **V3c chain.** Model, Λ = 4, z = 1 and length `chain_length_for(T; Λ = 4)` (margin 8)
    as in the mpNRG validation runs: N = 15 (`T = 1e-2`), 22 (`T = 1e-4`); no field.
13. **Truncated NRG data violate two preconditions of Stage 1** (exact and binned ED
    data satisfy them to rounding): the Pelz weights `W̃_β` are not Hermitian bin by
    bin (anti-Hermitian part 0.2–1.4 % of max‖W̃‖; the sum over bins is Hermitian to
    1e-17), and their Hermitian part is not positive semidefinite (negative mass
    2e-13 to 9e-4). Stage 1 factors the Hermitian part and drops its negative
    eigenvalues; both are now reported (`coords` diag `:antiherm_*`,
    `:rank_dropped_negative`). The factored measure then has `v†v ≠ 1`, and the
    specified assertion `‖v†v - 1‖ ≤ 1e-12` fails in 12 of the 16 V3c cases. For V3c
    only, Stage 1 runs with the opt-in `gram = :general` where the assertion would
    fail: `P = 1 - v(v†v)^{-1}v†`, `σ = (v†v)^{-1}v†w`, identical to the specification
    when `v†v = 1` (recorded per case in `diag[:gram_general]`). Every other phase
    keeps the hard assertion. All V3c checks compare with the raw data, so C1 and C8
    include the cost of what Stage 1 drops. Negative bosonic weights (`T = 1e-4`, down
    to −1.1e-5) are clamped to 0 and reported, as specified.
14. **V3c data cache not committed.** The V3c `DiscreteData` are built once by
    `mpNRG/scripts/vertex_krylov_v3c.jl` (2 h) and cached in
    `notebooks/data/vertex_krylov_v3c/` (1.9 GB, git-ignored). The notebook loads the
    cache, builds missing cases with the same function (`v3c_build_all`), and runs
    the algorithm and all checks itself.
15. **V3c in the test file.** V3c is report only, so `test_vertex_krylov_nrg.jl`
    exercises only the code path on the smallest V3c case (`T = 1e-2`,
    `N_keep = 64`, (↑↓)): it runs, the values are finite, and they are logged.

16. **Summary-table names after the amendment.** For exact data the rows are
    "C4 (stabilized, default)" (guaranteed), "C4 (unstabilized, diagnostic)" and
    "C4 (BigFloat, unstabilized)" (report), "C4 (BigFloat, default chain)"
    (guaranteed); "C2" holds the invariants (guaranteed, A3) and
    "C2 blocks n_agree/L" the Jacobi-block agreement (report).

## Timings

4 threads (`JULIA_NUM_THREADS=4`), Julia 1.13.3.

| item | time |
|:--|:--|
| `test/test_vertex_krylov.jl` (V0, V1) | 57–69 s |
| `mpNRG/test/test_vertex_krylov_nrg.jl`, `VK_FAST=1` (`Nch = 1` only) | 5.2 min |
| `mpNRG/test/test_vertex_krylov_nrg.jl`, full, before the amendment: 841 pass, 8 broken (F1) | 90 min |
| `mpNRG/test/test_vertex_krylov_nrg.jl`, full, amended criteria and V3c code path: 872 pass | 77 min |
| notebook, Step 1 only (first execution) | 97 min |
| notebook, final execution from a fresh kernel (all cases incl. V3c from the cache; computation cell) | 112 min (whole notebook 113 min) |
| atom cases (V1, 13 runs incl. BigFloat and stabilization) | 64 s |
| V2a `Nch = 1` per case (data, algorithm, checks, stabilization, BigFloat) | 12–27 s |
| V2a `Nch = 2` per case, no field (7 901 bins, R = 2 616, L = 2 615) | 12 min |
| V2a `Nch = 2` per case, field 0.05 (12 163 bins, R = 3 840, L = 3 839) | 32–35 min |
| V2b / V3a per case (LogGrid, 195 bins) | 1.5–8 s (+ NRG data: 3–15 s, first call 30–40 s incl. compilation) |
| V3b per case | 4–5 s |

| V3c data (NRG + 34 PSFs per case), `T = 1e-2`, `N_keep` = 64 / 128 / 256 / 512 | 18–49 s / 44 s / 3 min / 12 min |
| V3c data, `T = 1e-4`, `N_keep` = 64 / 128 / 256 / 512 | 35 s / 83 s / 6.5 min / 25 min |
| V3c algorithm and checks per case (incl. BigFloat chain), from the cache | 1.2–3.7 min |

In the `Nch = 2` ExactGrid cases the cost is the particle chain (full
reorthogonalization, ~2 min), its stabilized rerun, the couplings (GEMMs of size
`L_B × nbins × L`), and the Zacinskis check (dense eigenproblems of size R).

## Verdict

**Step 1 (V0, V1, V2, V3a, V3b): every guaranteed check passes under the amended
criteria A1–A3.** Final execution of the notebook from a fresh kernel: 356 Step-1
rows, **252 pass, 0 FAIL, 104 report**; largest stabilized C4 1.21e-14, largest C2
invariant deviation 1.2e-12. The test files confirm it with the same functions:
`test/test_vertex_krylov.jl` 414/414, `mpNRG/test/test_vertex_krylov_nrg.jl` 872/872.
Before the amendment the Step-1 verdict was "not every guaranteed check passed"
(C4 of the unstabilized chain, finding F1, smallest case: 16-state chain, C4 = 1.03 in
Float64 and BigFloat); the reviewer classified this as a mis-specification of the task
sheet (amendment above). The regeneration of the Step-1 verdict from the stored
outputs (`notebooks/reclassify_step1_summary.py`, 314 rows, 242 pass, 0 FAIL, 70
report) is superseded by the fresh execution, which also adds the BigFloat runs of
the default chain and the `n_agree` rows.

**Step 2 (V3c): reported, no pass/fail** (176 rows, all report; notebook Sec. 6 and
finding F4). In brief:

- Truncated NRG data violate two preconditions of Stage 1 that exact data satisfy:
  the Pelz weights are not Hermitian bin by bin, and their Hermitian part is not
  positive semidefinite. The algorithm as specified stops at its Stage-1 assertion
  in 12 of 16 cases; those ran with the reported variant `gram = :general`
  (deviation 13).
- The C2 invariants (≤ 6e-13) and C5 (0) hold as for any Hermitian data; C1
  (2e-6 to 1.4e-2) measures what Stage 1 drops.
- C7 and C8 both decrease with `N_keep`. At `T = 1e-2`, C7 falls to 2.4e-8–5.1e-8 for
  `N_keep ≥ 256` while C8 levels off at 8.9e-5–1.5e-4. At `T = 1e-4`, C7 is
  2.0e-4–2.5e-4 and C8 4.5e-4–7.3e-4 at `N_keep ≥ 256`. At `N_keep = 64`, C8 is
  0.08–0.58.
- `K̃` changes by 2e-3–7e-3 (relative) from `N_keep = 256` to 512, against
  6e-2–4e-1 from 64 to 512.
- C3, C4 (report) are of the size of the binned-ED values of V2b; C6 is at rounding
  level at `T = 1e-2` and 6e-7–1.4e-5 at `T = 1e-4`, where negative bosonic weights
  are clamped.

Open findings (no guaranteed check involved):

- **F1.** The unstabilized chain leaks into 𝕂₀ ⊖ d† (C4 report under A1), seeded by
  the ~1e-16 accuracy with which Float64 ED data satisfy the equation of motion;
  C1, C3, C8 are unaffected.
- **F2.** The Jacobi blocks of Stage 2 and of the Zacinskis route agree only on the
  leading blocks (`n_agree`), although the invariants agree; late Jacobi
  coefficients of measures whose weights span many decades are ill-conditioned
  (report under A3).
- **F3.** `K^(12) ≡ 0` in the atom's (↑↑) channel (deviation 4).
- **F4.** V3c: non-Hermitian, non-PSD truncated NRG weights (above). At large
  `N_keep` C8 is not controlled by C7; the attribution to the dropped
  anti-Hermitian part is consistent with the data but not isolated.
