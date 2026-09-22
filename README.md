# Multipoint correlation functions of the Hubbard atom

Numerical companion to

> F. B. Kugler, S.-S. B. Lee and J. von Delft,
> *Multipoint Correlation Functions: Spectral Representation and Numerical Evaluation*,
> [Phys. Rev. X **11**, 041006 (2021)](https://doi.org/10.1103/PhysRevX.11.041006)

Matsubara and Keldysh correlators of the half-filled Hubbard atom, assembled
directly from their spectral representation and checked against every closed
form the paper quotes. The partial spectral functions are computed once and
shared by both formalisms; only the kernel changes.

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

`julia --project test/runtests.jl` runs **52,067 assertions**, all passing, on
Julia 1.13.0. `notebooks/hubbard_atom_spectral_representation.ipynb` executes end to end through the
`julia-1.13` IJulia kernel — 30 code cells, no errors, about 25 s — and is committed with
its outputs, so the numbers are readable on GitHub without running anything.
Strip them with `jupyter nbconvert --clear-output --inplace notebooks/hubbard_atom_spectral_representation.ipynb`
if you prefer a clean file.

`notebooks/hubbard_atom_spectral_representation.pdf` is the executed notebook rendered to PDF (20 pages), for
reading or printing without a Julia kernel.

## Layout

```
src/HubbardAtom.jl      module; Step 1: operators, spectrum, Boltzmann weights
src/psf.jl              Step 2: partial spectral functions, Eq. (28)
src/kernel.jl           Step 3: the Matsubara kernel, Eq. (46)
src/correlator.jl       Steps 4-6: Eq. (39), the propagator, the 4p vertex;
                        regular-kernel flag and G̃ of Eq. (68b) for V3
src/keldysh.jl          Keldysh: Eqs. (49), (52), (63), (67a,b)
test/runtests.jl        Step 1 checks; includes the other test files
test/test_psf.jl        Step 2 checks
test/test_kernel.jl     Step 3 checks
test/test_correlator.jl Steps 4-6 checks
test/test_keldysh.jl    K1-K3 and V1-V6
notebooks/hubbard_atom_spectral_representation.ipynb
                        narrated walkthrough: six Matsubara steps, then Keldysh
notebooks/hubbard_atom_spectral_representation.pdf
                        the executed notebook, rendered to PDF
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
differently. Dependencies are stdlib only (`LinearAlgebra`, `Printf`, `Test`).
