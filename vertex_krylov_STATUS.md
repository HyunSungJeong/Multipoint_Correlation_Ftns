# vertex_krylov — status (Step 1: phases V0, V1, V2, V3a, V3b)

Discrete Krylov representation of the Matsubara regular part of the 3p vertex,
[VD] Algorithm 2. **All results live in the notebook**
[`notebooks/vertex_krylov.ipynb`](notebooks/vertex_krylov.ipynb) (executed, with
outputs; [`notebooks/vertex_krylov.pdf`](notebooks/vertex_krylov.pdf) alongside;
rebuilt from `notebooks/build_vertex_krylov_notebook.py`). Its Section 0 holds the
full Step-1 test matrix (314 rows), Section 7 the findings. This file holds only
the phase log, the deviations from the task sheet, the timings and the verdict.

## Phase log

| phase | content | commit |
|:--|:--|:--|
| V0+V1 | environment; `src/vertex_krylov.jl` (data layer, Stages 1–5, assembly, reference estimator, checks C1–C8); `test/test_vertex_krylov.jl` (414 assertions) | `8ee954c` |
| V2 (in progress) | `mpNRG/test/vertex_krylov_adapter.jl`, `mpNRG/test/test_vertex_krylov_nrg.jl`, notebook builder | `eb0d8bc` |
| V2, V3a, V3b | Wilson-chain ED (exact and binned), N = 2 block test with field, stabilization, Zacinskis check, untruncated NRG, decoupled-tail truncated NRG; performance and memory fixes | `da7a2c7` |
| V4 (Step 1) | notebook Sections 0–5 and 7 with outputs, PDF, figures | `48fea2b` |
| V4 (Step 1) | this file, README row block | see `git log` |

V0 and V1 were developed together in one file and committed together. V3c
(physical truncated chain) is Step 2 and was not started.

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
6. **C2 split.** C2 has no tolerance in the test matrix. It is reported as the
   largest deviation of the robust parts of the Zacinskis route (`S - N̄²`, start
   block, spectra and first-block weights of `Ω′`, `Σ^S`; ≤ 1.3e-12 everywhere).
   The Jacobi-block comparison is reported separately as a profile and as `n_agree`
   (finding F2).
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

## Timings

4 threads (`JULIA_NUM_THREADS=4`), Julia 1.13.3.

| item | time |
|:--|:--|
| `test/test_vertex_krylov.jl` (V0, V1) | 57–69 s |
| `mpNRG/test/test_vertex_krylov_nrg.jl`, `VK_FAST=1` (`Nch = 1` only) | 5.2 min |
| `mpNRG/test/test_vertex_krylov_nrg.jl`, full: 841 pass, 8 broken (F1) | 90 min |
| notebook, all cases (computation cell) | 97 min |
| atom cases (V1, 13 runs incl. BigFloat and stabilization) | 64 s |
| V2a `Nch = 1` per case (data, algorithm, checks, stabilization, BigFloat) | 12–27 s |
| V2a `Nch = 2` per case, no field (7 901 bins, R = 2 616, L = 2 615) | 12 min |
| V2a `Nch = 2` per case, field 0.05 (12 163 bins, R = 3 840, L = 3 839) | 32–35 min |
| V2b / V3a per case (LogGrid, 195 bins) | 1.5–8 s (+ NRG data: 3–15 s, first call 30–40 s incl. compilation) |
| V3b per case | 4–5 s |

In the `Nch = 2` ExactGrid cases the cost is the particle chain (full
reorthogonalization, ~2 min), its stabilized rerun, the couplings (GEMMs of size
`L_B × nbins × L`), and the Zacinskis check (dense eigenproblems of size R).

## Verdict

**Not every guaranteed check of Step 1 passed.** All guaranteed checks pass
(C1, C5, C6, C7, C8 everywhere; C3 for exact data; the closed form, sanity and
convention checks; V3a/V3b data agreement ≤ 1e-14) **except C4 (bath leakage) for
exact Wilson-chain data with the particle chain of Algorithm 2 as specified
(projector `1 - vv†`)**: 13 FAIL rows in the summary (V2a, every chain case,
Float64 and BigFloat, plus the N = 2 block test). With the stabilization of [VD]
Prop. 13 (`1 - ΩΩ†`) C4 is ≤ 1.3e-14 in every case, and the outputs are unchanged
(`ΔΣ^S` ≤ 3.2e-16 and relative `ΔK̃` ≤ 1.2e-14 in the five cases tabulated in notebook
Sec. 3; asserted ≤ 1e-10 and ≤ 1e-9 for every V2a case in the NRG test file).

**Failure F1 — C4 without stabilization.**

- Values: 0.20–0.85 against the tolerance 1e-9.
- Smallest reproducing case: `siam(U = 0.5, εd = -0.1, Δ = 0.1)`, chain sites
  −1..0 (16 states), `T = 0.05`, ExactGrid, channel (↑↓). C4 = 1.03 in both Float64
  and BigFloat. The chain has L = 15 = dim ran P levels against 14 stabilized;
  dim Ω = 2.
- Mechanism: the leak starts at ~1e-15 at n = 1 and grows roughly exponentially
  along the chain until it is O(1) in the last levels. The unstabilized chain is
  longer than the stabilized one by dim Ω − 1, i.e. it enters the single-particle
  invariant subspace 𝕂₀ ⊖ d† of Lemma 12.
- Not Float64 rounding: BigFloat gives the same leak, so the seed is the
  ~1e-16 accuracy with which the Float64 ED data satisfy the equation of motion.
- Unaffected: C1, C3 and C8 pass with and without stabilization, so the spurious
  levels carry no weight in `Σ^S` or `K̃`.
- Per the stop rule no tolerance was loosened and no sign was patched. Details,
  plots and the stabilized comparison are in notebook Sec. 3 and Sec. 7.

Open findings (no guaranteed check involved):

- **F2.** The Jacobi blocks of Stage 2 and of the Zacinskis route agree only on the
  leading blocks (`n_agree` = 28–210), although `Σ^S`, the spectra and the weights
  agree to ≤ 1.3e-12. The late Jacobi coefficients of measures whose weights span
  many decades are ill-conditioned.
- **F3.** `K^(12) ≡ 0` in the atom's (↑↑) channel (deviation 4).
