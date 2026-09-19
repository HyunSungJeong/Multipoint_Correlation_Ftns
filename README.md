# Multipoint correlation functions of the Hubbard atom

Numerical companion to

> F. B. Kugler, S.-S. B. Lee and J. von Delft,
> *Multipoint Correlation Functions: Spectral Representation and Numerical Evaluation*,
> [Phys. Rev. X **11**, 041006 (2021)](https://doi.org/10.1103/PhysRevX.11.041006)

The goal is to assemble Matsubara correlators of the half-filled Hubbard atom
directly from their spectral representation, Eq. (39), and check them against the
closed forms the paper quotes.

Model — the `U/Δ → ∞` limit of the symmetric AIM, Eq. (79):

```
H = εd (n↑ + n↓) + U n↑ n↓ ,   εd = -U/2
```

on the four states `|0⟩, |↑⟩, |↓⟩, |↑↓⟩`.

## Status

| Step | Content | Benchmark | State |
|:--|:--|:--|:--|
| **1** | spectrum: operators, diagonalisation, `ρᵢ = e^(-βEᵢ)/Z` | `Z = 2(1 + e^(βu))` by hand | **done** |
| 2 | partial spectral functions, Eq. (28), general in ℓ | — | not started |
| 3 | the kernel, Eq. (46), two-branch form | finite by construction | not started |
| 4 | ℓ = 2: assemble Eq. (39) | `G(iν) = ½ Σ± (iν ± U/2)⁻¹`, App. E | not started |
| 5 | ℓ = 4: 24 permutations, Eq. (73), Eq. (76), App. D | Eqs. (85a), (85b) | not started |
| 6 | null test at `U = 0` | every connected object vanishes | not started |

## Layout

```
src/HubbardAtom.jl    Step 1: operators, spectrum, Boltzmann weights
test/runtests.jl      verification of Step 1
notebooks/block4.ipynb  narrated walkthrough; Step 1 filled, Steps 2-6 stubbed
```

## Running

Tests:

```
julia --project test/runtests.jl
```

Notebook — needs [IJulia](https://github.com/JuliaLang/IJulia.jl):

```
julia --project -e 'using Pkg; Pkg.add("IJulia")'
jupyter lab notebooks/block4.ipynb
```

The notebook's kernelspec is `julia-1.13`; adjust it to your installed kernel
name if Jupyter reports a missing kernel.

Step 1 was developed and checked against Julia 1.13.0: `test/runtests.jl`
passes 258 assertions, and every code cell of `notebooks/block4.ipynb`
executes cleanly.

## Conventions

The Fock basis is indexed `i = 1 + n↑ + 2 n↓`:

| index | state | `(N, Sz)` |
|:--|:--|:--|
| 1 | `\|0⟩` | `(0, 0)` |
| 2 | `\|↑⟩` | `(1, +1/2)` |
| 3 | `\|↓⟩` | `(1, -1/2)` |
| 4 | `\|↑↓⟩` | `(2, 0)` |

with `|↑↓⟩ = d†↑ d†↓ |0⟩`. Operators come from a Jordan–Wigner transform with
fermionic mode order `(↑, ↓)`:

```
d↑ = 1 ⊗ f ,   d↓ = f ⊗ z ,   f = [0 1; 0 0] ,   z = [1 0; 0 -1]
```

so the parity string `z` over the `↑` mode supplies the sign in
`d↓|↑↓⟩ = -|↑⟩` rather than it being patched in by hand. That sign propagates
into every permutation sign `ζp` of Eq. (39), so `test/runtests.jl` checks it
explicitly.

`spectrum` diagonalises `H` with `eigen` and then fixes the gauge inside each
degenerate eigenspace by rotating to diagonalise `N + √2 Sz`, which takes a
distinct value on every `(N, Sz)` sector. The choice is physically immaterial —
Eq. (39) sums over a complete eigenbasis — but it makes the eigenvectors
reproducible and gives every eigenstate sharp charge and spin, which matters for
debugging Steps 2-6. Boltzmann weights are evaluated in the shifted form
`e^(-β(Eᵢ-E₀))/Σⱼ e^(-β(Eⱼ-E₀))` so that large `βU` does not overflow.

## Closed forms used as checks

With `u = U/2`, the spectrum is `E(|0⟩) = 0`, `E(|↑⟩) = E(|↓⟩) = -u`,
`E(|↑↓⟩) = 2εd + U = 0`, hence

```
Z = 1 + e^(βu) + e^(βu) + 1 = 2(1 + e^(βu))
```

which is the hand check Step 1 is required to reproduce. The
`th = tanh(βu/2)` appearing in Eq. (85) is the polarisation
`(e^(βu) - 1)/(e^(βu) + 1)` of this same spectrum, so getting `Z` right here is a
prerequisite for the Step 5 benchmark. At `εd = -U/2` the atom sits at half
filling, `⟨n↑⟩ = ⟨n↓⟩ = 1/2`, and the double occupancy is `⟨n↑n↓⟩ = 1/Z`.
