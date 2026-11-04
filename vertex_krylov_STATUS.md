# vertex_krylov — status (Step 1: phases V0, V1, V2, V3a, V3b)

Discrete Krylov representation of the Matsubara regular part of the 3p vertex,
[VD] Algorithm 2. **All results live in the notebook**
[`notebooks/vertex_krylov.ipynb`](notebooks/vertex_krylov.ipynb) (with outputs;
PDF alongside). This file holds only the phase log, the deviations from the
task sheet, the timings, and the final verdict.

## Phase log

| phase | content | commit |
|:--|:--|:--|
| V0+V1 | environment; `src/vertex_krylov.jl` (data layer, Stages 1-5, assembly, reference estimator, checks C1-C8); `test/test_vertex_krylov.jl` | (pending) |

## Deviations from the task sheet

1. **Environment.** The setup script had installed Julia and Jupyter but no Julia
   registry, packages or IJulia, and the environment's network policy blocks the
   Julia package servers (`pkg.julialang.org` redirects to `storage.julialang.net`,
   403). Worked around without changing any project file: the General registry was
   cloned with git from GitHub, packages were installed with `JULIA_PKG_SERVER=""`
   (git clones and GitHub release artifacts), and IJulia was added to the default
   environment.
2. **Atom level position.** `HubbardAtomModel` pins `εd = -U/2`; the task needs
   `εd = -0.3`. Added `atom_hamiltonian` / `atom_spectrum(m; εd, h)` in
   `src/vertex_krylov.jl` (same gauge fixing as `spectrum`); `HubbardAtomModel`
   is unchanged. `discrete_data_atom` takes `εd`, `h` and checks that `sp`
   diagonalizes that Hamiltonian.
3. **Sparse 3p storage.** On an ExactGrid the Wilson-chain data have hundreds to
   thousands of bins, so the six 3p families are stored as sparse
   `(α, β) => block` dictionaries (`s1[(α,β)][a,b]` instead of `s1[α,β][a,b]`), and
   the kernels as block-sparse matrices. Same content as Sec. 2.4.
4. **C8 normalization when `K^direct ≡ 0`.** In the atom's (↑↑) channel
   `B = q_{↑↑} = U n_↓` commutes with `H`, `d_↑` and `q_↑`, so every correlator
   `G[B, x, y]` is independent of `τ₁` and its regular part vanishes at `ω12 ≠ 0`:
   `K^(12) ≡ 0` while the six permutation terms are O(1) and cancel (verified with
   operator-level `regular_sum`). The task-sheet ratio `max|ΔK|/max|K^direct|` is
   then 0/0. `check_C8` reports it, and when `max|K^direct| < 1e-10 · max_p|K_p|`
   uses `max|ΔK| / max_p max|K_p^direct|` instead (flag `degenerate`). It also
   reports the per-permutation error (Prop. 10 holds term by term), which is
   tested everywhere.
5. **Extra N = 2 test.** Besides the spin-resolved block test, the atom is run in a
   rotated, complex orbital basis `d'_i = Σ_σ rot[i,σ] d_σ`, where `K` is
   non-diagonal and asymmetric, so the `p₂` transposition (VD8) and the index
   contractions (VD5) are exercised non-trivially (with spin conservation `K` is
   diagonal and the transposition is a no-op).
6. **Pre-existing test failure.** `test/test_psf_lanczos.jl` (not touched by this
   task) fails one assertion on this Julia 1.13 build (`c.R_ref == (244, 244)`
   evaluates `(242, 242)`; it counts eigenvalue splittings at the 1e-15 level).
   Verified identical on a clean checkout of the branch head. Because
   `test/runtests.jl` stops at the first failing file, `test_vertex_krylov.jl` is
   included before it.

## Timings

(see below)

## Verdict

(pending)
