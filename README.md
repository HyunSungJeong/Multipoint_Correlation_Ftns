# Multipoint correlation functions of the Hubbard atom

Numerical companion to

> F. B. Kugler, S.-S. B. Lee and J. von Delft,
> *Multipoint Correlation Functions: Spectral Representation and Numerical Evaluation*,
> [Phys. Rev. X **11**, 041006 (2021)](https://doi.org/10.1103/PhysRevX.11.041006)

Matsubara and Keldysh correlators of the half-filled Hubbard atom and the
two-site Hubbard model (dimer), assembled directly from their spectral
representation and checked against every closed form available. The partial
spectral functions are computed once and shared by both formalisms; only the
kernel changes. The same code serves both models: everything downstream of
`spectrum(m)` accepts any `AbstractModel`.

Model — the `U/Δ → ∞` limit of the symmetric AIM, Eq. (79):

```
H = εd (n↑ + n↓) + U n↑ n↓ ,   εd = -U/2
```

on the four states `|0⟩, |↑⟩, |↓⟩, |↑↓⟩`.

## Status — Block 4 complete

| Step | Content | Benchmark | Result |
|:--|:--|:--|:--|
| **1** | spectrum: operators, diagonalisation, `ρᵢ = e^(-βEᵢ)/Z` | `Z = 2(1 + e^(βu))` by hand | ≤ 2e-16 |
| **2** | partial spectral functions, Eq. (28), general in ℓ | sum rule `Σ w = ⟨O₁⋯O_ℓ⟩` | exact |
| **3** | the kernel, Eq. (46), two-branch form | finite by construction | no Inf/NaN |
| **4** | ℓ = 2: assemble Eq. (39) | `G(iν) = ½ Σ± (iν ± U/2)⁻¹`, App. E | ≤ 4e-16 |
| **5** | ℓ = 4: 24 permutations, Eqs. (73), (74) | Eqs. (85a), (85b) | ≤ 8e-13 |
| **6** | null test at `U = 0` | every connected object vanishes | ≤ 2e-15 |

## Status — Keldysh formalism complete

Same PSFs `S`, new kernel. Real frequencies `ω` with `Σωᵢ = 0`; `γ₀` is an
explicit argument everywhere, and nothing is broadened or put on a grid.

| Task | Content | Check | Result |
|:--|:--|:--|:--|
| **K1** | `ω^[η]`, Eq. (49) | slot `η` gets `+i(ℓ-1)γ₀`, others `-iγ₀`; `Σω^[η] = Σω` | exact |
| **K2** | permuted Keldysh index, `μⱼ = p⁻¹(ηⱼ)` | p. 13 example `k = [24]`, `p = (4123)` → `[μ₁μ₂] = [31]`, `k_p = 2121`; convention-free digit permutation | 8/8, 48/48, 384/384 |
| **K3** | retarded kernel Eq. (52), general kernel Eq. (67b) | `α = 0` → 0; `α = 1` vs Eq. (52); Eq. (67b) vs Eq. (63) over 440 `(ℓ,k,p)` | 0.0 |
| **K4** | assembly Eq. (67a), prefactor `2^(1-ℓ/2)` | — | — |
| **V1** | `G^{1⋯1} = 0` | ℓ = 2, 3, 4 | 0.0 |
| **V2** | ℓ = 2, App. C: `G^[1] = G^R`, `G^[2] = G^A`, `G^[12] = G^K` | Lehmann forms; atom closed forms; FDT (C4) recovered as `O(γ₀²)` | ≤ 9e-16; 4.9e-5 → 4.9e-9 |
| **V3** | Eq. (69): `2^(ℓ/2-1) G^[η](ω) = G̃(iω → ω^[η])` | ℓ = 2, 3, 4, via a regular-kernel flag on the Matsubara routine | ≤ 2e-15 |
| **V4** | which components are finite / `→0` / `δ`-like as `γ₀ → 0` | table at generic and `ωᵢⱼ = 0` points, ↑↓ and ↑↑ | see notebook |
| **V5** | null test at `U = 0` | `S^con ≡ 0`; `G^con,k` for all 16 `k` | ≤ 7e-12 |
| **V6** | `δ`-like components against Eq. (85) | `U`-scaling exponents 3, 2, 2 (↑↓) and 2, 2 (↑↑); App. E `(n,n)` | 3.008, 1.953, 2.061, 2.010, 2.010 |

## Status — Hubbard dimer complete

`notebooks/hubbard_dimer.ipynb` (PDF alongside). Two sites, 16 states, operators
from global Jordan–Wigner strings in mode order `1↑, 1↓, 2↑, 2↓`, momentum
`k ∈ {0, π}` carried as an integer. The closed forms (spectrum, `Z`, `G_k`) are
the task sheet's own; they are checked here against exact diagonalisation.

| Check | Content | Result |
|:--|:--|:--|
| **D1** | spectrum, degeneracies, `Z`; shifted weights at `β = 1000` | ≤ 5e-15 |
| **D2** | `G_k(iν)` from ℓ = 2 PSFs vs closed form, 3 parameter sets; U = 0, t = 0, T → 0 limits | ≤ 4e-15 |
| **D3** | `G₁₁ = ½(G₀+G_π)`, `G₁₂ = ½(G₀-G_π)` from site operators; `G₁₁` purely imaginary | ≤ 1e-15 |
| **D4** | Keldysh ℓ = 2: `G^R`, `G^A` vs continued closed form; FDT (C4) at all 8 poles | ≤ 1e-13 rel.; `O(γ₀²)` |
| **D5** | momentum selection rule, 8 forbidden tuples | exactly 0 (≤ 8e-16 unfiltered) |
| **D6** | U = 0: `G^con`, `F` for 16 tuples × 2 spins | ≤ 3e-14 |
| **D7** | `F↑↓/U → ½ δ_{k₁-k₂+k₃-k₄}`, `F↑↑/U → 0` (Richardson from U = 1e-2, 5e-3) | ≤ 6e-6 |
| **D8** | t → 0: `F → ½ F^atom δ`; anomalous vs regular branch of Eq. (46) | exact at t = 0; ∝ t² |
| **D9** | crossing, both `ω₁↔ω₃` and `ω₂↔ω₄` forms | ≤ 5e-14 |
| **D10** | Keldysh: selection rule, U = 0, Eq. (69) for all fully retarded components | exact / ≤ 5e-14 / exact |
| **D11** | size of the `δ_{ω₁₂}` term vs `β(c-U/2)` | free-moment value for x ≪ 1; `∝ β e^{-β(c-U/2)}` for x ≫ 1 |

## Status — vertex heat maps (Figs. 10, 11) complete

The last section of `notebooks/hubbard_atom_spectral_representation.ipynb` evaluates the
atom's 4p vertex on 2D frequency grids at `U = 1`, `T = U/50`, `γ₀ = T`, `ω = 0`, and
reproduces Fig. 10 (MF) and Fig. 11 (KF) with error maps. Figures (PNG + PDF) and
arrays (`.jld2`) are written to `notebooks/figures/heatmaps/`.

| Item | Content | Result |
|:--|:--|:--|
| **H1** | allocation-free `vertex_point!` + grid driver `vertex_grid` | 0 bytes, type-stable; = reference code to 1e-11 on values up to 5e3 |
| **H2** | `S^con = S - S^dis` at the PSF level, coincident peaks merged | MF = Eq. (73) line 1; KF `Σ ζ K^[η] * S^dis = 0` to 4e-16 relative |
| **H3** | Keldysh amputation, Eq. (76), legs from Eq. (D2) (shifts 3γ₀/γ₀) | `leg(ν,1,1)` = ℓ = 2 routine; `F^2222 ≡ 0`, `F^[η]_↑↑ ≡ 0` |
| **H4 / P3** | `F^[η]_↑↓` against the continuation of Eq. (85a), 401 × 401 | max relative error 3e-15 |
| **P1** | Fig. 10, 48 × 48 Matsubara grid | targets 0.9510, 8.76e-5, −0.3439 met; ≤ 6e-11 vs Eq. (85) |
| **P2** | Fig. 11, 2 × 7 panels, all 16 components saved | 1122, 1111 purely imaginary to 7e-16 |
| **P4, P5** | leg conditioning; `γ₀` scaling | `|G^R(0)| = 4γ₀/U²`; `G^con` poles ∝ `γ₀^-3.00` |

Timings, fresh process, one spin, one thread (`scripts/heatmap_timing.jl`): MF 48²
first call 0.24 s, warm 9 ms; KF 401² × 16 components first call 2.6 s, warm 2.2 s.

Findings: the `γ₀⁻³` of p. 19 describes the correlator poles, which the data confirm;
the vertex maxima scale as `γ₀⁻³` to `γ₀⁻⁵` depending on the component (`F^[η]` as
`γ₀⁻⁴`, as Eq. (3) implies). In the Fig. 11 caption, "other components with two 1's
and two 2's are similar to 2112" is a symmetry statement for 1221, but 1212 and 2121
form their own class (about 3× larger), similar only visually. The task sheet's
`ref_plots.jl` was not supplied; `symlog`, the Eq. (85) map and `Fret_ud` are
implemented in `src/heatmap.jl` / the notebook. The 82 MB file with all 16 KF
components (`p2_kf_vertex_full.jld2`) is regenerated by the notebook and git-ignored.

`julia --project test/runtests.jl` runs **58,708 assertions**, all passing, on
Julia 1.13.0. `notebooks/hubbard_atom_spectral_representation.ipynb` executes end to end through the
`julia-1.13` IJulia kernel — 30 code cells, no errors, about 25 s — and is committed with
its outputs, so the numbers are readable on GitHub without running anything.
Strip them with `jupyter nbconvert --clear-output --inplace notebooks/hubbard_atom_spectral_representation.ipynb`
if you prefer a clean file.

`notebooks/hubbard_atom_spectral_representation.pdf` is the executed notebook rendered to PDF (34 pages), for
reading or printing without a Julia kernel. `notebooks/hubbard_dimer.ipynb`
(about a minute, 18 code cells) and `notebooks/hubbard_dimer.pdf` (20 pages) do
the same for the dimer.

## Layout

```
src/HubbardAtom.jl      module; AbstractModel; Step 1: operators, spectrum, Boltzmann weights
src/psf.jl              Step 2: partial spectral functions, Eq. (28)
src/kernel.jl           Step 3: the Matsubara kernel, Eq. (46)
src/correlator.jl       Steps 4-6: Eq. (39), the propagator, the 4p vertex;
                        regular-kernel flag and G̃ of Eq. (68b) for V3
src/keldysh.jl          Keldysh: Eqs. (49), (52), (63), (67a,b)
src/dimer.jl            the two-site model: operators, closed forms, k-basis vertex
src/heatmap.jl          grid evaluation, PSF-level subtraction, KF amputation (D2 legs)
scripts/heatmap_timing.jl  first-call / warm timings in a fresh process
test/runtests.jl        Step 1 checks; includes the other test files
test/test_psf.jl        Step 2 checks
test/test_kernel.jl     Step 3 checks
test/test_correlator.jl Steps 4-6 checks
test/test_keldysh.jl    K1-K3 and V1-V6
test/test_dimer.jl      D1-D11, traps 1-3
test/test_heatmap.jl    H1-H4, P1 targets, symlog
notebooks/hubbard_atom_spectral_representation.ipynb
                        narrated walkthrough: six Matsubara steps, then Keldysh
notebooks/hubbard_atom_spectral_representation.pdf
                        the executed notebook, rendered to PDF
notebooks/hubbard_dimer.ipynb, notebooks/hubbard_dimer.pdf
                        the dimer task, D1-D11
notebooks/figures/heatmaps/
                        P1-P5 figures (PNG, PDF) and arrays (.jld2)
```

## Conventions and design decisions

**Fock basis** indexed `i = 1 + n↑ + 2 n↓`, i.e. `|0⟩, |↑⟩, |↓⟩, |↑↓⟩`, with
`|↑↓⟩ = d†↑ d†↓ |0⟩`. Operators come from a Jordan–Wigner transform in
fermionic mode order `(↑, ↓)`,

```
d↑ = 1 ⊗ f ,   d↓ = f ⊗ z ,   f = [0 1; 0 0] ,   z = [1 0; 0 -1]
```

so the parity string over the `↑` mode supplies the sign in `d↓|↑↓⟩ = -|↑⟩`
rather than it being patched in by hand. That sign propagates into every
permutation sign `ζp` of Eq. (39).

**Degenerate eigenspaces** are gauge-fixed by rotating to diagonalise
`N + √2 Sz`, distinct on every `(N, Sz)` sector, and each eigenvector's first
significant component is pinned positive. Physically immaterial — Eq. (28)
closes a cycle over the eigenbasis, which `test_psf.jl` verifies explicitly by
flipping eigenvector signs — but it makes results reproducible across LAPACK
versions and gives every eigenstate sharp charge and spin.

**Boltzmann weights** use the shifted form `e^(-β(Eᵢ-E₀))/Σⱼ e^(-β(Eⱼ-E₀))`, so
large `βU` does not overflow.

**Matsubara frequencies carry an integer.** `MatsubaraFreq` stores `m` in
`ω = mπ/β` (odd = fermionic, even = bosonic). The composite
`Ω_{1̄⋯ī} = iω_{1̄⋯ī} - ω'_{1̄⋯ī}` of Eq. (46) vanishes only when *both* parts
vanish, and the Matsubara part vanishes exactly or not at all — so that test is
an integer comparison. Float-comparing `Ω` to zero would confuse a genuinely
small frequency (`π/β` is tiny at large `β`) with a composite that is
identically zero and needs the anomalous branch. `test_kernel.jl` exercises this
at `β = 1e12`, where a nonzero bosonic composite has `|Ω| = 6e-12`, well under
any sane tolerance.

**`ζp` counts fermionic transpositions only**, not the parity of `p`: a bosonic
operator such as `n_σ` commutes through and must not contribute a sign.

**Appendix D does not enter.** Its imaginary frequency shifts are needed only
for the Keldysh amputation of Eq. (76), where the legs sit at real frequencies.
The Matsubara part has its legs exactly on discrete frequencies, with nothing to
shift; the Keldysh part stops at the full and connected correlators and does
not amputate.

## Keldysh: notes and pitfalls

All Keldysh equations were read from page images, not the text layer.

**`p⁻¹`, not `p`.** The permuted Keldysh index uses `μⱼ = p⁻¹(ηⱼ)` (p. 13).
The worked example on p. 13 cannot tell the two apart by itself: `p` and `p⁻¹`
both map `{2,4}` onto `{1,3}`, so the wrong rule also gives `k_p = 2121`. Only
the intermediate `[μ₁μ₂] = [31]` discriminates. The tests therefore compare
against a convention-free reference, permuting the digits of `k` directly. The
`p` rule would be wrong in 0/8 (ℓ = 2, every permutation an involution),
12/48 and 168/384 cases.

**External vs. permuted labels.** In Eq. (67b) the `ω^[η]` are evaluated at the
external labels `η̄̂ⱼ = p(k_p,ⱼ)`, and the alternating sign `(-1)^(j-1)` runs over
the sorted permuted slots. The tests check that `η̄̂` is a permutation of `k`
for every `(k, p)`.

**A trap that passes `α ≤ 1`.** Pairing the alternating signs with the external
ordering instead of the permuted one agrees exactly with every `α ≤ 1`
component, so V3 cannot see it. It changes 44 of 64 entries of the ↑↑ V4 table.
The test suite keeps it as a negative control.

**Eq. (59b) is not implemented.** In its last line both imaginary-part signs
are flipped relative to the `γ` convention; as printed it gives `+2πiδ` where
`-2πiδ` follows from the definition. Nothing here depends on it.

**The FDT (C4) holds only as `γ₀ → 0`.** At finite `γ₀` the ratio
`G^K/(G^R - G^A)` deviates from `tanh(βω/2)` by `O(γ₀²)`: a factor 100 per
decade, as checked in V2.

**V4.** For ↑↓, `α = 1` and `α = 3` components stay finite everywhere; `α = 2`
components and `2222` go to zero at generic points; at `ωₐᵦ = 0` exactly the
`α = 2` components with a single `2` in `{a, b}` diverge like `1/γ₀`. For ↑↑,
there is no `δ` at `ω₁₃`, matching Eq. (85b), and `F↑↑` is purely anomalous, so
its retarded components vanish identically. Some ↑↑ `α = 1` entries that looked
divergent were roundoff (`|G|/Σ|terms| ≈ 1e-16`); the classification is now
conditioning-aware.

**V5 has no teeth for K2.** At `U = 0`, `S^con ≡ 0`, so every kernel, correct or
not, gives `G^con = 0`. Even the full `G` of the trap kernel vanishes at a
generic point. V5 is a check on the PSF split of Eq. (31), not on the kernel.

## Dimer: notes and findings

**Momentum-basis amputation.** In the site basis the legs are 2×2 matrices, so
the scalar division of Eq. (74) is wrong for `t ≠ 0` (5–14% off at
`t = 0.7`); it is done leg by leg in the `k` basis, where `G` is diagonal, and
site components follow by transforming the four legs of `F`. Eq. (73) gets a
momentum Kronecker delta per Wick pairing and is cross-checked against the
kernel-free PSF split of Eq. (31).

**Matrix-element filter (`otol`).** `H` is not diagonal in the Fock basis, so
symmetry-forbidden matrix elements come out of the eigenbasis rotation as
roundoff (`~1e-17`), not zeros. Unfiltered, 99.9% of the `1.6×10⁶` cycles of a
4p PSF survive with weights `~1e-33` (397,408 terms instead of 272).
`psf(...; otol)` zeroes rotated elements below `otol` first; `DimerContext`
uses `1e-12`, and `otol = 0` reproduces the unfiltered sum (identical results
for allowed tuples; forbidden ones at 1e-16 instead of exactly 0). The atom's
default is `0`, so its results are unchanged.

**D8, both branches of Eq. (46).** Site exchange `P` is conserved and the
bonding/antibonding levels have opposite `P`, so which terms switch branch at
`t = 0` depends on the tuple: for `0101` all 384 δ-carrying terms are anomalous
at `t = 0` and none are at `t ≠ 0`, the contribution moving to the regular
branch through `±2t` denominators. The deviation from `½F^atom` is a clean `t²`
law for `10⁻³ ≤ t ≤ 10⁻¹`. Below `t ≈ 3×10⁻⁴` roundoff growing as `1/t²` takes
over in `0101` (1e-7 at `t = 1e-4`, 1e-5 at `t = 1e-5`); at `t = 1e-5` the
singlet–triplet gap `4t²/U ≈ 2e-10` also lets LAPACK mix those eigenvectors.

**D11, the δ(ω₁₂) term.** Defined without fitting as the anomalous term of
Eq. (45) (`anomalous_part`), restricted to one frequency channel, on the
connected PSF, amputated. On the atom it reproduces the three `βu²δ_ω` terms
of Eq. (85a) channel by channel. For the dimer, relative to the free-moment
value it is `≈ 1` for `β(c-U/2) ≪ 1`, crosses over at `β(c-U/2) ≈ 1`, and
beyond that `|D₁₂| ≈ C β e^{-β(c-U/2)}`: the local decay rate is 0.95 of the
singlet–triplet gap at `x ≈ 20`, with the shortfall × β ≈ 1 for all three `t`.
The task sheet's expectation holds; the rate is the triplet's Boltzmann weight.

**Trap 3.** At `t = U/2` the triplet meets `-U/2 - t` and the anomalous branch
fires unexpectedly; with the explicit tolerance the vertex is continuous there
(`|F(1) - F(1±10⁻⁶)| ≤ 6e-13`).

## Closed forms used as checks

With `u = U/2`, the spectrum is `E(|0⟩) = 0`, `E(|↑⟩) = E(|↓⟩) = -u`,
`E(|↑↓⟩) = 2εd + U = 0`, hence

```
Z = 1 + e^(βu) + e^(βu) + 1 = 2(1 + e^(βu))
```

the hand check for Step 1. The `th = tanh(βu/2)` of Eq. (85) is the
polarisation `(e^(βu) - 1)/(e^(βu) + 1)` of this same spectrum. At `εd = -U/2`
the atom sits at half filling, `⟨n↑⟩ = ⟨n↓⟩ = 1/2`, with double occupancy
`⟨n↑n↓⟩ = 1/Z`.

For Step 2, summing Eq. (28) over every eigenstate cycle collapses the cyclic
product of matrix elements to a trace,

```
Σ weights = tr(ρ O_1̄ O_2̄ ⋯ O_ℓ̄) = ⟨O_1̄ O_2̄ ⋯ O_ℓ̄⟩
```

so the total weight of a PSF is the equal-time expectation value of the
operator product in the same order — for every `ℓ`. Sharper at `ℓ = 2`: the
local spectral function `A(ω) = S[d,d†](ω) + S[d†,d](-ω)` comes out as
`½δ(ω-u) + ½δ(ω+u)`, exactly the two poles of Appendix E's
`G(iν) = ½ Σ± (iν ± U/2)⁻¹`, with the strong temperature dependence of the
individual PSFs cancelling.

Note that PSFs can vanish identically rather than merely cancel: since `n↑` is a
projector and `d↑d†↑ = 1 - n↑`, every cyclic arrangement of `(n↑, d↑, d†↑)` has
`n↑(1 - n↑) = 0`, so `psf` returns no terms at all. Interleaving differently,
`d↑ n↑ d†↑ = (1 - n↑)² = 1 - n↑`, does not vanish.

### Eq. (85) and the PDF text layer

**The published Eq. (85) is correct.** What is broken is the PDF *text layer*:
it renders both products of the anomalous term on one line and loses the
fraction bar. Extracted literally it reads

```
β u² [δ_{ω12} th + δ_{ω13}(th-1) + δ_{ω14}(th+1)] / [∏ᵢ(iωᵢ+u) ∏ᵢ(iωᵢ)]
```

which cannot be right: `βu²` has dimension of energy, so the remaining factor
must be dimensionless, while that denominator carries energy⁸ — the expression
would be energy⁻⁷. Building against that form made Step 5 fail *only* in the
`βδ_ω` terms, while matching to 1e-16 at every generic frequency point.

The printed factor is `∏ᵢ(iωᵢ+u) / ∏ᵢ(iωᵢ)`:

```
F↑↓ = 2u + u³ Σᵢ(iωᵢ)²/∏ᵢ(iωᵢ) - 6u⁵/∏ᵢ(iωᵢ)
      + β u² [δ_{ω12} th + δ_{ω13}(th-1) + δ_{ω14}(th+1)] ∏ᵢ(iωᵢ+u)/∏ᵢ(iωᵢ)

F↑↑ = β u² (δ_{ω14} - δ_{ω12}) ∏ᵢ(iωᵢ+u)/∏ᵢ(iωᵢ)
```

This was first reconstructed from three independent arguments and then
**confirmed by rendering p. 16 of the PDF to an image and reading the printed
equation** (`pdftoppm -f 16 -r 400`). The three arguments, none of them
circular, were:

1. **Dimensions.** Both terms come out as energy, as a vertex must.
2. **Appendix E.** It separately states that expanding the vertex to second
   order gives `F↑↓ = U + ¼βU²(δ_{ω14} - δ_{ω13})` and
   `F↑↑ = ¼βU²(δ_{ω14} - δ_{ω12})`. The corrected form reproduces both, with
   the remainder shrinking as `O(U³)` — checked in `test_correlator.jl`.
3. **Brute force.** A direct numerical `τ`-integration of Eq. (71), independent
   of the whole Eq. (39)/Eq. (46) machinery, reproduces the computed `G⁽⁴⁾` at
   the very frequency points where the text-layer form disagreed.

Whenever any `δ` fires the frequencies are `±`-paired, `(ν,-ν,ν',-ν')`, so
`∏ᵢ(iωᵢ+u) = ∏ᵢ(iωᵢ-u)` and the sign of `u` in that product is not observable.

Eqs. (28), (39), (45), (46), (73), (74) and the Appendix E results were all
likewise checked against the page images; only Eq. (85) was affected.

`F↑↑` is also checked against crossing symmetry,
`F↑↑(iω) = F↑↓(iω) - F↑↓(iω′)` with `ω₁ ↔ ω₃` exchanged, independently of the
numerics.

## Running

```
julia --project test/runtests.jl
```

Notebook — needs [IJulia](https://github.com/JuliaLang/IJulia.jl):

```
julia --project -e 'using Pkg; Pkg.add("IJulia")'
jupyter lab notebooks/hubbard_atom_spectral_representation.ipynb
```

The kernelspec is `julia-1.13`; adjust it if your installed kernel is named
differently. Dependencies: `LinearAlgebra`, `Printf`, `Test` (stdlib) and
`StaticArrays`; the heat-map section also needs `CairoMakie`, `JLD2`,
`BenchmarkTools`. All are in `Project.toml`: `julia --project -e 'using Pkg; Pkg.instantiate()'`.
