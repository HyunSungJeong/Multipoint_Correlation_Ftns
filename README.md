# Multipoint correlation functions from exact diagonalisation

**Hubbard atom · Hubbard dimer · Anderson impurity model with a discrete bath**

Numerical companion to

> F. B. Kugler, S.-S. B. Lee and J. von Delft,
> *Multipoint Correlation Functions: Spectral Representation and Numerical Evaluation*,
> [Phys. Rev. X **11**, 041006 (2021)](https://doi.org/10.1103/PhysRevX.11.041006)

**Design principle.** Every correlator here is assembled from its spectral
representation. The partial spectral functions (PSFs) of Eq. (28) are computed
once, from an exact diagonalisation, as exact δ peaks — no broadening, no
binning, no Matsubara summation. They are then shared by every formalism: the
Matsubara kernel of Eq. (46) and the Keldysh kernels of Eqs. (52), (67b) act on
the same PSFs, and the disconnected part is removed at the PSF level, Eq. (31).
Everything downstream of `spectrum(m)` is model independent. A new model only
supplies its Hamiltonian and operators; the atom, the dimer and the Anderson
impurity model (AIM) all run through the same code.

The Julia module is still called `HubbardAtom`, for compatibility; it now
covers all three models.

| Model | States | Notebook (committed with outputs, PDF alongside) |
|:--|:--|:--|
| Hubbard atom, MF and KF, heat maps of Figs. 10, 11 | 4 | `notebooks/hubbard_atom_spectral_representation.ipynb` |
| Hubbard dimer | 16 | `notebooks/hubbard_dimer.ipynb` |
| AIM, `N_b = 1, 2, 3` bath sites | 16, 64, 256 | `notebooks/aim.ipynb` |

## Status

Each row names the task sheet it answers, the check, and the accuracy achieved.
The long per-step narratives are in the notebooks.

### Hubbard atom

`H = εd (n↑ + n↓) + U n↑ n↓`, `εd = -U/2`: the `U/Δ → ∞` limit of Eq. (79).

| Sheet | Item | Check | Accuracy |
|:--|:--|:--|:--|
| `today.pdf` | Step 1 | spectrum, `Z = 2(1 + e^(βu))` | ≤ 2e-16 |
| `today.pdf` | Step 2 | PSFs, Eq. (28); sum rule `Σ w = ⟨O₁⋯O_ℓ⟩` | exact |
| `today.pdf` | Step 3 | kernel, Eq. (46), two-branch form | no Inf/NaN by construction |
| `today.pdf` | Step 4 | `G(iν) = ½ Σ± (iν ± U/2)⁻¹`, App. E | ≤ 4e-16 |
| `today.pdf` | Step 5 | 4p vertex vs Eqs. (85a), (85b) | ≤ 8e-13 |
| `today.pdf` | Step 6 | `U = 0` null test | ≤ 2e-15 |
| `keldysh_task.pdf` | K1–K4 | `ω^[η]`, `μⱼ = p⁻¹(ηⱼ)`, kernels (52), (67b), assembly (67a) | exact; Eq. (67b) = Eq. (63) over 440 `(ℓ,k,p)` |
| `keldysh_task.pdf` | V1, V2 | `G^{1⋯1} = 0`; `G^R`, `G^A`, `G^K` vs App. C; FDT (C4) | 0.0; ≤ 9e-16; `O(γ₀²)` |
| `keldysh_task.pdf` | V3 | Eq. (69), ℓ = 2, 3, 4 | ≤ 2e-15 |
| `keldysh_task.pdf` | V4–V6 | finite / `→0` / δ-like classification; `U = 0`; `U`-scaling vs Eq. (85) | ≤ 7e-12; exponents 3.008, 1.953, 2.061, 2.010, 2.010 |
| `heatmap_task.pdf` | H1 | allocation-free `vertex_point!`, `vertex_grid` | 0 bytes; = reference to 1e-11 on values ≤ 5e3 |
| `heatmap_task.pdf` | H2 | `S^con = S - S^dis` at the PSF level | MF = Eq. (73); KF `Σ ζ K^[η] * S^dis` ≤ 4e-16 relative |
| `heatmap_task.pdf` | H3 | Keldysh amputation, Eq. (76), legs from Eq. (D2) | `F^2222 ≡ 0`, `F^[η]_↑↑ ≡ 0` |
| `heatmap_task.pdf` | H4 / P3 | `F^[η]_↑↓` vs the continuation of Eq. (85a), 401² grid | ≤ 3e-15 relative |
| `heatmap_task.pdf` | P1, P2 | Figs. 10, 11 reproduced | P1 targets 0.9510, 8.76e-5, −0.3439 met |
| `heatmap_task.pdf` | P4, P5 | leg conditioning; `γ₀` scaling | `|G^R(0)| = 4γ₀/U²`; `G^con` poles ∝ `γ₀^-3.00` |

### Hubbard dimer

Two sites, 16 states, momentum `k ∈ {0, π}`; closed forms from the task sheet,
checked against exact diagonalisation.

| Sheet | Item | Check | Accuracy |
|:--|:--|:--|:--|
| `dimer_task.pdf` | D1 | spectrum, degeneracies, `Z` | ≤ 5e-15 |
| `dimer_task.pdf` | D2, D3 | `G_k` vs closed form; `G₁₁`, `G₁₂` from site operators | ≤ 4e-15; ≤ 1e-15 |
| `dimer_task.pdf` | D4 | Keldysh `G^R`, `G^A`; FDT at all 8 poles | ≤ 1e-13 relative; `O(γ₀²)` |
| `dimer_task.pdf` | D5, D6 | momentum selection rule; `U = 0` | exactly 0; ≤ 3e-14 |
| `dimer_task.pdf` | D7 | `F↑↓/U → ½ δ_{k₁-k₂+k₃-k₄}` | ≤ 6e-6 |
| `dimer_task.pdf` | D8 | `t → 0`: `F → ½ F^atom δ` | exact at `t = 0`; ∝ `t²` |
| `dimer_task.pdf` | D9, D10 | crossing; Keldysh selection rule, `U = 0`, Eq. (69) | ≤ 5e-14; exact / ≤ 5e-14 / exact |
| `dimer_task.pdf` | D11 | δ(ω₁₂) term vs `β(c-U/2)` | observation, see Findings |

### Anderson impurity model

`H = -U/2 Σ_σ n_dσ + U n_d↑ n_d↓ + Σ_{nσ} [ε_n b†_nσ b_nσ + V_n (d†_σ b_nσ + h.c.)]`,
Eq. (79) with `N_b` bath sites from moment matching of the box hybridisation
(`D = 1`, presets `{-D, D}`, `{-D, 0, D}`, `{-D, -D/3, D/3, D}`).

| Sheet | Item | Check | Accuracy |
|:--|:--|:--|:--|
| `aim_task.pdf` | Sec. 1 | sum rule, `ε_n → -ε_{N_b+1-n}`, anticommutators of all modes, `⟨n_dσ⟩ = ½` | exact / ≤ 1e-15 / exact / ≤ 6e-14 |
| `aim_task.pdf` | I1 | chain-walk PSFs = dense PSFs (atom, dimer, `N_b = 1, 2`; 24 permutations; full and connected) | identical peaks, weights to 1e-16 |
| `aim_task.pdf` | I2 | Dict peak merging = pairwise merging; no straddled cells | identical, same order |
| `aim_task.pdf` | I3 | `ImpurityModel` interface, `otol` on every PSF path | existing 58,708 assertions unchanged |
| `aim_task.pdf` | A1 | `N_b = 1` levels, `Z`, 8 poles, `T → 0` residues | ≤ 7e-15; 0.172914 / 0.327086 |
| `aim_task.pdf` | A2 | `G_d(iπT)`, 12 reference values, `N_b = 1, 2, 3` | ≤ 3.4e-14 relative, `|Re G_d|` ≤ 1.4e-13 |
| `aim_task.pdf` | A3 | `U = 0`: `[z - Δ(z)]⁻¹`, MF and KF | MF ≤ 4e-14; KF ≤ 4e-11 on values up to `1/γ₀` |
| `aim_task.pdf` | B1 | `U = 0`: `S^con`, `F` (MF and 16 KF components) | `S^con` ≤ 2e-15, `F ≡ 0` |
| `aim_task.pdf` | B2 | `V → 0`: `F → F^atom` | ≤ 1.2e-13 at `V = 0`; ∝ `V²` |
| `aim_task.pdf` | B3 | Eq. (84) with the discrete-bath `χ₀` | ≤ 1.3e-5 on brackets up to 1.9 |
| `aim_task.pdf` | B4 | crossing, both forms | ≤ 4e-13 |
| `aim_task.pdf` | B5 | `G^1111 = 0`, `F^2222 ≡ 0`, `F^[η]` vs regular Matsubara sums | exact / exact / ≤ 7e-13 relative |
| `aim_task.pdf` | B6 | cost of tables and grids | see Performance |
| `aim_task.pdf` | O1 | local `χ_loc`, sheet table | all printed digits |
| `aim_task.pdf` | O2, O3 | δ(ω₁₂) term vs `βΔ_ST`; heat maps next to the atom's | observations, see Findings |

## Findings

**Eq. (85) is right; the PDF text layer is not.** The text layer puts both
products of the anomalous term on one line and loses the fraction bar; read
literally it is energy⁻⁷. The printed factor is `∏ᵢ(iωᵢ+u)/∏ᵢ(iωᵢ)`, confirmed
from the page image, by dimensions, by Appendix E's second-order vertex and by
a brute-force τ integration. Every equation used anywhere in the repository was
read from the rendered page, not the text layer.

**`p⁻¹`, not `p`** (p. 13). The worked example `k = [24]`, `p = (4123)` cannot
tell the two apart (both give `k_p = 2121`); only `[μ₁μ₂] = [31]` does. Against a
convention-free digit permutation, the `p` rule is wrong in 12/48 (ℓ = 3) and
168/384 (ℓ = 4) cases, never for ℓ = 2.

**A trap that passes `α ≤ 1`.** Pairing the alternating signs of Eq. (67b) with
the external order instead of the permuted one agrees exactly for every
`α ≤ 1` component, so Eq. (69) cannot see it; it changes 44 of 64 entries of the
↑↑ V4 table. Kept in the tests as a negative control.

**Eq. (59b) is not implemented**: in its last line both imaginary-part signs are
flipped relative to the γ convention (`+2πiδ` where `-2πiδ` follows).

**The FDT (C4) holds only as `γ₀ → 0`**, with an `O(γ₀²)` error (a factor 100
per decade), atom and dimer alike. **V5 has no teeth for K2**: at `U = 0`,
`S^con ≡ 0`, so any kernel gives `G^con = 0`.

**Fig. 11 caption classes.** The unplotted components are exact symmetry images
of the plotted ones (to rounding) except 1212 and 2121, which form their own
class, about 3× larger than 2112 and similar to it only visually.

**`γ₀` scaling.** The `γ₀⁻³` of p. 19 is the correlator's pole scale — confirmed,
`max|G^con,k| ∝ γ₀^-3.00`. The vertex maxima scale faster: `F^[η]` as `γ₀⁻⁴`,
exactly as Eq. (3) implies at the origin, 2112 and 1122 as about `γ₀⁻⁵`.

**D8: the two branches of Eq. (46).** For the dimer tuple 0101 all 384 δ-carrying
terms take the anomalous branch at `t = 0` and none at `t ≠ 0`; the vertex is
continuous, deviating as `t²`. Below `t ≈ 3×10⁻⁴` roundoff ∝ `1/t²` takes over.

**D11: the dimer's δ(ω₁₂) term** is the free-moment value for `β(c-U/2) ≪ 1` and
`∝ β e^{-β(c-U/2)}` beyond the crossover — the triplet's Boltzmann weight.

**I1: chain-walk cost.** The chain walk reproduces the dense PSFs exactly and
turns `N_b = 3` from impossible (4.3×10⁹ tuples per permutation) into tens of
seconds: 0.6–0.8×10⁶ surviving cycles and about 3×10⁵ merged peaks per
permutation (table under Performance).

**B2: the AIM approaches the atom as `V²`**, exactly at `V = 0`. For `N_b = 2` at
`V = 10⁻⁴` the law breaks: levels split by ~`V²` ≈ 3×10⁻⁸ let LAPACK mix
eigenvectors at ε/gap ~ 10⁻⁸, above the `1e-12` filter; `otol = 1e-8` restores
the `V²` law. The same mechanism as D8.

**O1: screening depends on the parity of `N_b`.** Half filling holds `N_b + 1`
electrons. For `N_b = 1, 3` the ground state is a singlet and `χ_loc` saturates
(2.02467 at `N_b = 1`, `U = 2`, `Δ = 0.1`); for `N_b = 2` it is a Kramers doublet
and `χ_loc ∝ β` at any coupling.

**O2: the AIM's δ(ω₁₂) term is cut off at the doublet gap, not `Δ_ST`.** For
`βΔ_ST ≪ 1` it is 0.75 of the atom's free-moment value. Beyond the crossover the
local decay rate is 0.74–0.78 `Δ_ST` for all three couplings, i.e. the gap `a - b`
to the four-fold odd-charge level, which lies below the triplet for `N_b = 1`
(`(a-b)/Δ_ST = 0.715–0.745`). The term changes sign at `βΔ_ST ≈ 24.5–26.5`; a two-point
rate across that node mimics the dimer's `Δ_ST` law. So the dimer's
`β e^{-βΔ_ST}` does not carry over.

**O3: heat maps.** With the moment screened (`N_b = 1`, `βΔ_ST = 15.7`) the MF
vertex drops to 2×10⁻⁵ of the atom's and the KF maps lose the atom's cross; with
the moment free, or for the unscreened `N_b = 2` doublet, the atom's structure
survives. `F^[η]_↑↑ ≡ 0` is atom-specific: nonzero for every AIM case.

## Performance

**Atom heat maps** (`scripts/heatmap_timing.jl`, fresh process, one thread, one
spin): MF 48² first call 0.25 s, warm 8 ms; KF 401² × 16 components first call
2.5 s, **warm 2.2 s** — the number to compare with the heat-map sheet's one-hour
figure. 84 merged connected peaks in total (↑↓, 24 permutations, `U = 1`, `T = U/50`).

**PSF construction (I1)**, all 24 connected tables, one thread, `notebooks/aim.ipynb`
(atom `U = 2`, `β = 5`; dimer `t = 0.7`, `U = 2.3`, `β = 3`; AIM `U = 2`, `β = 10`):

| system | states | dense, 24 tables | chain, 24 tables | cycles / permutation | merged peaks / permutation (Σ) |
|---|---:|---:|---:|---:|---:|
| atom | 4 | 0.001 s | 0.016 s | 1–1 | 4–5 (104) |
| dimer (`k = 0000`) | 16 | 0.018 s | 0.000 s | 8–13 | 20–21 (488) |
| AIM `N_b = 1` | 16 | 0.025 s | 0.002 s | 64–100 | 110–126 (2896) |
| AIM `N_b = 2` | 64 | 2.830 s | 0.147 s | 4046–5537 | 4918–5249 (123068) |
| AIM `N_b = 3` | 256 | skipped | 25.021 s | 646304–814512 | 301835–346159 (7915652) |

**AIM grids (B6)**, `U = 1`, `T = U/50`, `γ₀ = T`, `Δ = 0.1`, 4 threads:

| model | 4 tables | merged peaks / permutation (Σ) | MF 48², warm | KF 401² × 16, first | KF 401² × 16, warm |
|---|---:|---:|---:|---:|---:|
| AIM `N_b = 1` | 0.14 s | 53–73 (1556) | 0.032 s | 9.0 s | 9.1 s |
| AIM `N_b = 2` | 0.58 s | 630–744 (16364) | 0.315 s | 94.1 s | 101.2 s |

The cost per frequency point is proportional to the number of merged peaks,
which is why the AIM slices cost 4× (`N_b = 1`) and 45× (`N_b = 2`) the atom's
wall time despite four threads.

## Conventions and design decisions

- **Jordan–Wigner operators.** Every model builds its mode operators with
  global strings (`jordan_wigner`), and the Hamiltonian as products of those
  matrices; no matrix element is entered by hand, so string signs (e.g. in
  `d†_σ b_nσ`) come out automatically. Anticommutators are asserted first.
- **Degenerate eigenspaces** are gauge-fixed by diagonalising a symmetry
  operator (`N + √2 Sz`, plus the site exchange for the dimer), and eigenvector
  signs are pinned: immaterial for Eq. (28), which closes a cycle, but
  reproducible.
- **Boltzmann weights** are shifted by the ground-state energy, so large `β`
  never overflows.
- **Matsubara frequencies carry an integer** (`ω = mπ/β`), so the vanishing of a
  composite in Eq. (46) is an exact test, and `TOL_DEG = 1e-10` decides spectral
  degeneracies explicitly.
- **`ζp` counts fermionic transpositions only.**
- **Matrix-element filter `otol`.** When `H` is not diagonal in the Fock basis,
  symmetry-forbidden matrix elements come out of the rotation as roundoff; they
  are zeroed below `otol` (`1e-12` for the dimer and the AIM, `0` for the atom).
- **PSF methods.** The dense `psf` is the reference; the chain walk (`psf_chain`,
  `psf4_tables_chain`) is used automatically above 16 states and gives the same
  peaks.
- **Appendix D.** Its imaginary shifts do not enter the Matsubara vertex, whose
  legs sit on discrete frequencies. They do enter the Keldysh amputation: the
  legs of Eq. (76) use the Eq. (D2) shifts `3γ₀`/`γ₀` (H3, `leg`).

## Layout

```
src/HubbardAtom.jl      module; AbstractModel, ImpurityModel; the atom's operators and spectrum
src/psf.jl              PSFs, Eq. (28), dense reference; Eq. (31) split
src/kernel.jl           Matsubara kernel, Eq. (46)
src/correlator.jl       Eq. (39); propagator, 4p correlator, vertex for any ImpurityModel;
                        regular sum (Eq. 68b); anomalous part of Eq. (45)
src/keldysh.jl          Keldysh: Eqs. (49), (52), (63), (67a,b)
src/dimer.jl            two-site model: operators, closed forms, k-basis vertex
src/heatmap.jl          grid evaluation, PSF-level subtraction, KF amputation (D2 legs)
src/psf_chain.jl        chain-walk PSFs (I1), Dict peak merging (I2)
src/aim.jl              AIM: discretisation, operators, closed forms, χ₀, χ_loc
scripts/heatmap_timing.jl  first-call / warm timings in a fresh process
test/runtests.jl        atom spectrum; includes all other test files
test/test_psf.jl, test_kernel.jl, test_correlator.jl   atom, Matsubara
test/test_keldysh.jl    K1-K3, V1-V6
test/test_dimer.jl      D1-D11
test/test_heatmap.jl    H1-H4, P1 targets
test/test_aim.jl        AIM task: Sec. 1, I1-I2, A1-A3, B1-B5, O1
notebooks/hubbard_atom_spectral_representation.ipynb, .pdf
notebooks/hubbard_dimer.ipynb, .pdf
notebooks/aim.ipynb, .pdf
notebooks/figures/heatmaps/   atom P1-P5 (PNG, PDF, .jld2)
notebooks/figures/aim/        AIM O2, O3 (PNG, PDF, .jld2)
```

## Reproducibility

Every number quoted in this README is produced by a test (`test/`) or a cell of
a committed notebook (outputs included). `julia --project test/runtests.jl`
runs **59,353 assertions**, all passing, on Julia 1.13.0. The one
file not committed is the 82 MB array of all 16 atom KF components
(`p2_kf_vertex_full.jld2`), which the atom notebook regenerates in seconds.

## Running

```
julia --project -e 'using Pkg; Pkg.instantiate()'
julia --project test/runtests.jl
```

Notebooks need [IJulia](https://github.com/JuliaLang/IJulia.jl); the kernelspec
is `julia-1.13`. `notebooks/aim.ipynb` takes about 15 minutes with
`JULIA_NUM_THREADS=4` (mostly the `N_b = 2` Keldysh slices); the others take
under a minute to a few minutes. Dependencies: `LinearAlgebra`, `Printf`, `Test`
(stdlib), `StaticArrays`; the heat-map and AIM notebooks also use `CairoMakie`,
`JLD2`, `BenchmarkTools`.
