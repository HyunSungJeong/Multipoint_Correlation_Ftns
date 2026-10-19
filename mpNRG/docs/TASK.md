# Task sheet: `mpNRG`, a general-purpose multipoint NRG code base in Julia

> **Note (2026-10-19).** This sheet was amended by the TNOps refactor, whose own sheet is
> [`TNOPS_REFACTOR.md`](TNOPS_REFACTOR.md) (Section 11 there lists the amendments). The abelian block-sparse
> layer now lives in the sibling package `TNOps/` at the repository root, and `mpNRG/src/nrg/tensors.jl` holds the
> only tensor operations of `mpNRG`. Amended passages are marked **[Amended 2026-10-19, TNOps refactor]**.

Repository: `https://github.com/HyunSungJeong/Multipoint_Correlation_Ftns`
Branch: **`claude/practical-archimedes`** (existing; HEAD `541d40a` at the time of writing). Do not create a new branch.
Target folder: **`mpNRG/`** at the repository root. All NRG code lives there.
Language: Julia only.

---

## 0. What this is and what it is not

The repository today contains the Julia package `HubbardAtom` (`src/`, `test/`, `notebooks/`). It computes partial spectral functions (PSFs) exactly from exact diagonalization (ED), and then evaluates Matsubara and Keldysh kernels, vertices and the PSF Lanczos construction on them.

This task adds a **separate, general-purpose numerical renormalization group (NRG) package** in `mpNRG/`. It is a full-density-matrix (FDM) NRG for quantum impurity models that computes thermodynamics, thermal expectation values and **local 2p, 3p and 4p PSFs**. It follows the multipoint NRG of Lee, Kugler & von Delft, PRX 11, 041007 (2021) [R17].

It is a code base for computing physics, not a test harness:

- **Models.** Any impurity model with a finite local Fock space must be expressible, not just the single-impurity Anderson model (SIAM).
- **Operators.** Any local operator must be usable, including composite operators such as $q=[d,H_{\rm int}]$.
- **Hybridization.** Any hybridization function must be accepted, so the package can serve as a DMFT impurity solver later.

Tests and validation are one layer on top of that.

Downstream uses that must be *possible* with the public API. Do not special-case them in the core.

- Discrete, unbroadened 2p/3p/4p PSF histograms on one common logarithmic grid, for every operator permutation. These are needed for the project's Krylov/Liouvillian algorithms, which need unbroadened data.
- PSFs of composite local operators: $q_\sigma=[d_\sigma,H_{\rm int}]$, bosonic bilinears such as $\{q_1,d_2^\dagger\}$, and operators on the impurity plus Wilson site 0, such as $d^\dagger_\downarrow f_{0\downarrow}$.
- Optional extra outputs per PSF: the histogram of absolute values (the sum of $|{\rm weight}|$ per bin), and an unbinned Lehmann-term mode for small systems.
- Access to per-shell internals (isometries, reduced density matrices, sector-projected operators, sliced composite operators) through a documented hook interface. A future in-sweep compression scheme will need them. Do **not** implement that scheme now.
- Export of PSFs to the existing `HubbardAtom` kernels (Matsubara, Keldysh), so correlators and vertices can be built with code that already exists and is tested.

Out of scope for this task (Section 9 lists future extensions the design must not preclude):

- non-abelian symmetries;
- matrix-valued (off-diagonal) hybridization;
- bosonic baths;
- time-dependent NRG;
- broadening of 3p/4p PSFs;
- Matsubara or Keldysh kernels (they exist in `HubbardAtom`);
- symmetric improved estimators (downstream).

---

## 1. Ground rules

1. **Branch and footprint.**
   - Work and commit on `claude/practical-archimedes`.
   - Do not modify `src/`, `test/` or `notebooks/` of `HubbardAtom`.
   - Allowed outside `mpNRG/`: one new row block in the root `README.md` status section, and root `.gitignore` entries.
   - **[Amended 2026-10-19, TNOps refactor]** Code may also live in `TNOps/` (the abelian block-sparse tensor package, a sibling of `mpNRG/`).
   - Push after each phase if the environment permits; otherwise leave the commits local and say so in `STATUS.md`.
2. **Separate package.**
   - `mpNRG/` is its own Julia package, `MultipointNRG`, with its own `Project.toml` and UUID.
   - Its `src/` must **not** depend on `HubbardAtom`.
   - Tests and notebooks may depend on `HubbardAtom` as an independent reference implementation. Use the test environment with a path source (`[sources] HubbardAtom = {path = ".."}`, which needs Julia ≥ 1.11). If the installed Julia is older, use `Pkg.develop(path="..")` in the test setup.
   - The repository has been run with Julia 1.13 (commit `41d5f09`). Set `julia = "1.11"` as the minimum unless that is impossible.
3. **Dependencies.** Keep them minimal:
   - `LinearAlgebra`, `JLD2` (already used by the repository) and `Printf`;
   - optionally `DoubleFloats` for extended-precision tridiagonalization; `BigFloat` from Base is also acceptable;
   - optionally `ThreadsX` or plain `Threads` for parallel loops.

   Do not add QSpace, ITensors, TensorKit or similar. The abelian block-sparse layer is written in this package so that it stays transparent.
   **[Amended 2026-10-19, TNOps refactor]** The abelian block-sparse layer is the in-repository package `TNOps/` (dependencies `LinearAlgebra`, `Printf`); still no external tensor libraries.
4. **Papers.**
   - Paper PDFs are not in the repository. Ask the user to place them in `mpNRG/refs/`, which must be in `.gitignore` (never commit PDFs). If network access allows, download arXiv versions instead; the arXiv IDs are in Section 2.
   - **Read every equation, sign, index and overbar from the page images, never from the PDF text layer.** The text layer of Lee 2021 is known to corrupt brackets, primes, overbars and underlines (`]`, `⋯` and the underlines of sliced frequencies are affected).
   - If the user's `academic-pdf-reading` skill is installed (GitHub `HyunSungJeong/Academic_Claude_Skills`), follow it.
   - Lee 2021 printed page = PDF page (offset 0).
5. **Provenance in code.** Every function that implements a formula carries a docstring citing it in the style already used in the repository, e.g. "Lee Eq. (59)", "Weichselbaum & von Delft (2007) Eq. (8)". Every NRG *approximation* (truncation, the NRG approximation Lee Eq. (19), binning, slicing) is stated in the docstring of the function that makes it.
6. **Stop rule.** If an exactness test (Section 6, V1–V4) fails, or an implementation disagrees with the paper, stop and report the discrepancy in `STATUS.md`. Never loosen a tolerance or change a sign convention to make a test pass without writing down why. A sign that "works" but is not derived is a bug.
7. **Bookkeeping.**
   - Keep `mpNRG/STATUS.md` up to date: phase, what is done, what was verified and to what accuracy, open issues.
   - Commit once per phase, with messages in the repository's style (e.g. `mpNRG P6: 2p PSFs, Lee Eq. (59)`).
   - Verification code also runs in a committed Jupyter notebook (Julia kernel, executed with outputs, rendered to PDF alongside), as for the other models in the repository.

---

## 2. References

All entries were confirmed by web search on 2026-10-18 (title, authors, journal, volume, page; arXiv ID where available). "Use" says what each one is needed for.

| Key | Reference | Use |
|---|---|---|
| R1 | K. G. Wilson, *The renormalization group: Critical phenomena and the Kondo problem*, Rev. Mod. Phys. **47**, 773 (1975). DOI 10.1103/RevModPhys.47.773 | Logarithmic discretization, Wilson chain, iterative diagonalization, rescaling |
| R2 | H. R. Krishna-murthy, J. W. Wilkins, K. G. Wilson, *Renormalization-group approach to the Anderson model of dilute magnetic alloys. I. Static properties for the symmetric case*, Phys. Rev. B **21**, 1003 (1980); *II. Static properties for the asymmetric case*, Phys. Rev. B **21**, 1044 (1980) | NRG for the SIAM; thermodynamics (entropy, susceptibility); fixed points; benchmarks |
| R3 | M. Yoshida, M. A. Whitaker, L. N. Oliveira, *Renormalization-group calculation of excitation properties for impurity models*, Phys. Rev. B **41**, 9403 (1990) | Origin of the discretization shift $z$ (interleaving) and dynamical NRG |
| R4 | W. C. Oliveira, L. N. Oliveira, *Generalized numerical renormalization-group method to calculate the thermodynamical properties of impurities in metals*, Phys. Rev. B **49**, 11986 (1994). DOI 10.1103/PhysRevB.49.11986 | $z$-averaging |
| R5 | R. Bulla, Th. Pruschke, A. C. Hewson, *Anderson impurity in pseudo-gap Fermi systems*, J. Phys.: Condens. Matter **9**, 10463 (1997); arXiv:cond-mat/9707010 | Wilson chain for an arbitrary (energy-dependent) hybridization |
| R6 | R. Bulla, A. C. Hewson, Th. Pruschke, *Numerical renormalization group calculations for the self-energy of the impurity Anderson model*, J. Phys.: Condens. Matter **10**, 8365 (1998); arXiv:cond-mat/9804224 | Composite-operator correlators for the self-energy (asymmetric estimator); motivates the composite-operator API |
| R7 | R. Žitko, Th. Pruschke, *Energy resolution and discretization artefacts in the numerical renormalization group*, Phys. Rev. B **79**, 085106 (2009); arXiv:0810.4765 | Discretization prescription used by Lee 2021 (their Ref. [23]); $z$-averaging; band-edge artefacts |
| R8 | V. L. Campo Jr., L. N. Oliveira, *Alternative discretization in the numerical renormalization-group method*, Phys. Rev. B **72**, 104432 (2005) | Alternative discretization scheme (optional) |
| R9 | R. Žitko, *Adaptive logarithmic discretization for numerical renormalization group methods*, Comput. Phys. Commun. **180**, 1271 (2009); arXiv:0902.3124 | Discretization of arbitrary (e.g. DMFT) hybridizations; reference for the differential-equation scheme |
| R10 | F. B. Anders, A. Schiller, *Real-time dynamics in quantum-impurity systems: A time-dependent numerical renormalization-group approach*, Phys. Rev. Lett. **95**, 196801 (2005); arXiv:cond-mat/0505553 | Complete basis of discarded states (Anders–Schiller basis) |
| R11 | F. B. Anders, A. Schiller, *Spin precession and real-time dynamics in the Kondo model: Time-dependent numerical renormalization-group study*, Phys. Rev. B **74**, 245113 (2006); arXiv:cond-mat/0604517 | Detailed derivation of the complete basis |
| R12 | R. Peters, Th. Pruschke, F. B. Anders, *Numerical renormalization group approach to Green's functions for quantum impurity models*, Phys. Rev. B **74**, 245114 (2006); arXiv:cond-mat/0607494 | Complete-basis Green's functions (Lee's Ref. [21]) |
| R13 | A. Weichselbaum, J. von Delft, *Sum-rule conserving spectral functions from the numerical renormalization group*, Phys. Rev. Lett. **99**, 076402 (2007); arXiv:cond-mat/0607497 | **FDM-NRG**: full density matrix, shell weights, 2p spectral functions, sum rules |
| R14 | R. Bulla, T. A. Costi, Th. Pruschke, *Numerical renormalization group method for quantum impurity systems*, Rev. Mod. Phys. **80**, 395 (2008); arXiv:cond-mat/0701105 | General NRG reference: discretization, chain mapping, truncation, thermodynamics, spectral functions, broadening |
| R15 | A. Weichselbaum, *Tensor networks and the numerical renormalization group*, Phys. Rev. B **86**, 245124 (2012); arXiv:1209.2062 | MPS formulation of NRG, FDM in tensor-network language, fermionic signs, implementation details |
| R16 | F. B. Kugler, S.-S. B. Lee, J. von Delft, *Multipoint correlation functions: Spectral representation and numerical evaluation*, Phys. Rev. X **11**, 041006 (2021); arXiv:2101.00707 | Definition and conventions of PSFs (Eq. (28)); kernels (already implemented in `HubbardAtom`) |
| R17 | S.-S. B. Lee, F. B. Kugler, J. von Delft, *Computing local multipoint correlators using the numerical renormalization group*, Phys. Rev. X **11**, 041007 (2021); arXiv:2101.00708 | **Primary reference** for everything multipoint: shell projectors, operator expansions, density matrix, thermal averages, 2p/3p/4p PSFs, binning and slicing |
| R18 | S.-S. B. Lee, A. Weichselbaum, *Adaptive broadening to improve spectral resolution in the numerical renormalization group*, Phys. Rev. B **94**, 235127 (2016); arXiv:1609.01248 | Broadening of 2p data (log-Gaussian to linear crossover at $T$; adaptive scheme optional) |
| R19 | F. B. Kugler, *Improved estimator for numerical renormalization group calculations of the self-energy*, Phys. Rev. B **105**, 245132 (2022); arXiv:2202.04063 | Symmetric self-energy estimator; needs 2p PSFs of $d$ and $q$ (downstream use of the API) |
| R20 | J.-M. Lihm, J. Halbinger, J. Shim, J. von Delft, F. B. Kugler, S.-S. B. Lee, *Symmetric improved estimators for multipoint vertex functions*, Phys. Rev. B **109**, 125138 (2024); arXiv:2310.12098 | Auxiliary 3p/4p correlators of composite operators (downstream use of the API) |
| R21 | A. Ge, J. Halbinger, S.-S. B. Lee, J. von Delft, F. B. Kugler, *Analytic continuation of multipoint correlation functions*, Ann. Phys. (Berlin) **536**, 2300504 (2024); arXiv:2311.11389 | PSF conventions across formalisms (context only) |
| R22 | A. Weichselbaum, *QSpace – An open-source tensor library for Abelian and non-Abelian symmetries*, SciPost Phys. Codebases **40** (2024); arXiv:2405.06632 | The tensor library behind the reference mpNRG code (MuNRG); design reference for a future non-abelian layer |
| R23 | M. Frankenbach, M. Ritter, M. Pelz, N. Ritz, J. von Delft, A. Ge, *Compressing local vertex functions from the multipoint numerical renormalization group using quantics tensor cross interpolation*, arXiv:2506.13359, published in Phys. Rev. Research (DOI 10.1103/jx7h-lsqk). Their mpNRG PSFs are published at LMU Open Data, https://data.ub.uni-muenchen.de/613/ | **External benchmark data** (published mpNRG PSFs) |
| R24 | A. Calvo-Fernández, M. Blanco-Rey, A. Eiguren, *The PointGroupNRG code for numerical renormalization group calculations with discrete point-group symmetries*, Comput. Phys. Commun. **296**, 109032 (2024); arXiv:2307.03658 | An existing open-source **Julia** NRG code; design comparison only, do not copy code |
| R25 | R. Žitko, *NRG Ljubljana* (software), https://github.com/rokzitko/nrgljubljana, snapshot DOI 10.5281/zenodo.4841076 | Optional independent 2p cross-check |
| R26 | A. Zacinskis, F. T. Ebel, M. Pelz, F. B. Kugler, K. Held, J. von Delft, M. W. Haverkort, A. Gleis, *Symmetric estimator for discrete self-energy of discrete many-body systems*, arXiv:2605.04974 (2026) | Downstream consumer of discrete, unbroadened data (context only) |

### Map of Lee 2021 [R17]

Tag pages were located on the text layer; read the equations themselves from the page images.

| Section (printed page) | Content | Equations |
|---|---|---|
| II D (p. 4) | Nested structure of PSFs, the $\varepsilon_i$ variables | (11)–(14) |
| III A (p. 5) | Wilson shells, kept/discarded sectors, MPS form | (15)–(17) |
| III B–C (p. 6) | Anders–Schiller basis, NRG approximation, preview of 2p PSFs | (18)–(20) |
| IV A (p. 7) | Shell projectors | (21)–(27) |
| IV B (p. 8) | Expanding local operators (the shell-diagonal expansion is **exact** for local operators) | (28)–(37) |
| IV C (p. 9) | Expanding nonlocal (frequency-dependent) operators (approximate) | (35), (38)–(39) |
| IV D (p. 10) | Density matrix, reduced density matrices, backward sweep | (40)–(45) |
| IV E–F (p. 11) | Thermal averages; expanding operator products | (46)–(54) |
| V A (pp. 12–13) | 2p PSFs | (55)–(59) |
| V B (pp. 13–14) | Binning and slicing (logarithmic grid, sliced tensors) | (60) |
| V C (pp. 14–16) | 3p PSFs: the seven sectors, KKD/KDK refinement through 2p PSFs of composite operators | (61)–(66) |
| V D (pp. 16–18) | 4p PSFs: 15 sectors, refinements through 2p and 3p PSFs of sliced composites, nested recursion, cost estimates | (67)–(71) |
| VI (pp. 18–22) | Histogram representation, kernels, disconnected parts, broadening, equation of motion | (72)–(87) |
| VII A (p. 23) | AIM benchmark model and NRG parameters | (88) |
| App. A (p. 33) | Bottom-up vs top-down; converting shell-off-diagonal products | (A1)–(A2) |
| App. C (p. 35) | Log-Gaussian broadening kernel | (C1)–(C2) |

---

## 3. Architecture

```
mpNRG/
├── Project.toml              # package MultipointNRG
├── README.md                 # user-facing overview, quick start, conventions
├── STATUS.md                 # phase log, verification record, open issues
├── docs/TASK.md              # a copy of this task sheet
├── refs/                     # paper PDFs, gitignored
├── src/
│   ├── MultipointNRG.jl      # module, exports
│   ├── fock/                 # local Fock spaces, fermionic operators, operator algebra
│   ├── (symmetry/)           # [amended] moved to the sibling package TNOps/ (see below)
│   ├── bath/                 # hybridization functions, discretization, tridiagonalization
│   ├── model/                # ImpurityModel, built-in models
│   ├── nrg/                  # forward sweep, truncation, shell storage (Anders–Schiller basis);
│   │                         #   [amended] nrg/tensors.jl: the named NRG contractions on TNOps tensors
│   ├── operators/            # shell-sector projections, local/nonlocal expansions, products
│   ├── fdm/                  # backward sweep, reduced density matrices, thermal averages
│   ├── psf/                  # grids, histograms, binning, slicing, 2p/3p/4p engines
│   ├── observables/          # thermodynamics, flow diagrams, 2p broadening
│   ├── io/                   # JLD2 format with full metadata, z-averaging
│   ├── hooks/                # per-shell callback interface
│   └── reference/            # dense brute-force ED on finite Wilson chains (testing aid, but public)
├── test/
│   ├── runtests.jl
│   ├── Project.toml          # test env: Test, HubbardAtom (path source)
│   └── ...                   # one file per validation item of Section 6
├── examples/                 # SIAM, Kondo, resonant level, two-orbital, arbitrary Δ(ω)
└── notebooks/
    ├── mpnrg_validation.ipynb   # executed, committed with outputs
    └── mpnrg_validation.pdf
```

**[Amended 2026-10-19, TNOps refactor]** The abelian layer is the sibling package `TNOps/` (`TNOps/src`: symmetry, legs, `BlockTensor`,
`SlicedTensor`, fusion, factorizations, `global_cut`, parity, `ncon`, interface; `TNOps/test`, `TNOps/notebooks`).
`mpNRG/src/symmetry/` no longer exists; `mpNRG/src/nrg/tensors.jl` is the only place where `mpNRG` operates on
tensors (refactor sheet Sec. 4).

Layering rule: `psf/` and `fdm/` depend on `nrg/` and `operators/`, never the reverse. `reference/` depends only on `fock/` and `bath/`, so it is an independent check of `nrg/`.

---

## 4. Conventions (fix once, document in `mpNRG/README.md`)

1. **Units.**
   - Energies are in units of the half-bandwidth $D=1$ unless the hybridization says otherwise.
   - Temperature is given as $T$, with $\beta=1/T$ derived from it.
   - Internally, shell $n$ uses the usual rescaled energies, but **all energies passed to PSFs and FDM weights are absolute**.
   - Track the accumulated ground-state shift of every shell exactly. The FDM shell weights need energies of different shells on one common scale [R13, R17 Sec. IV D].
2. **Wilson chain indexing** follows Lee [R17, Sec. III A]: the impurity is site $-1$, bath sites are $0,1,\dots,N$, on-site energies are $\epsilon_n$, and $t_n$ couples site $n$ to site $n-1$.
   - Choose $N$ so that $\Lambda^{-N/2}\ll T$ (Lee, p. 5). Make the margin a parameter.
   - At $n=N$ all states are discarded (Lee, p. 5).
3. **Fermionic order and signs.**
   - Lee builds the product basis "from right to left, as if filling the chain in the order $c^\dagger_N\cdots c^\dagger_{-1}|0\rangle$" (p. 5, following Weichselbaum & von Delft [R13]). Use this convention, or another one that is written down.
   - Implement signs through an explicit parity operator per site, not by hand-patched signs.
   - Unit-test the canonical anticommutators across shells (Section 6, V1).
4. **PSF convention.** This is identical to Kugler Eq. (28) [R16] and to the existing `HubbardAtom.psf` (`src/psf.jl`, docstring of `PSFTerm`).
   - For operators given **already in permuted order** $(\mathcal O^1,\dots,\mathcal O^\ell)$, the weight of the cycle $(1,\dots,\ell)$ of eigenstates is $\rho_1\prod_i(\mathcal O^i)_{i,i+1}$, cyclically.
   - Its positions are $\varepsilon_i=E_{i+1}-E_1$, $i=1,\dots,\ell-1$.
   - Lee writes the same object in nested form (Eqs. (11), (12), (55), (61), (67)) and uses the same $\varepsilon_i$.
   - Fermionic permutation signs $\zeta_p$ belong to the kernels, **not** to the PSFs. Store raw PSFs per permutation.
5. **Logarithmic grid** (Lee p. 13, read from the page image):
   - $\underline\varepsilon[\pm m]=\pm10^{(|m|-1)/n_{\rm dec}}\varepsilon_{\min}$ for $m\neq0$, and $\underline\varepsilon[0]=0$;
   - $|\underline\varepsilon[m]|\le\varepsilon_{\max}$;
   - the zero bin collects weight at zero frequency up to a numerical tolerance (a parameter);
   - each grid point is the centre of its bin (Lee p. 13, Fig. 1(a)).

   Lee's defaults are $\Lambda=4$ and $n_{\rm dec}=32$ (p. 13). **One grid object is shared by all permutations and all $\ell$ in a run** (Lee p. 18); the API must make it hard to mix grids.
6. **Storage of histograms.**
   - 2p: dense.
   - 3p: dense, with optional sparse storage.
   - 4p: sparse by default (sorted COO of nonzero bins, or a dictionary keyed by bin triples). A dense 4p histogram on a fine grid does not fit in memory.
   - Weights are complex in general; use a type parameter so real-symmetric runs stay real.

---

## 5. Component specifications

### 5.1 Local Fock spaces and operators (`fock/`)

- A local space is a set of named fermionic modes (for example `(:d,:up)`, `(:d,:dn)`, or several orbitals). Bosonic local degrees of freedom need not be supported now, but must not be ruled out by the types.
- Provide a small operator algebra on the local space: creation and annihilation operators, sums, products, scalar multiples, adjoint, commutator and anticommutator, and normal-ordered display. Each operator carries its quantum-number shift and its fermionic parity.
- Composite operators are first-class:
  - $q_\sigma=[d_\sigma,H_{\rm int}]$ is constructed from the user's local Hamiltonian;
  - bosonic composites like $\{q_1,d_2^\dagger\}$ are products and anticommutators of local operators.
- Operators acting on the impurity **and** Wilson site 0 must also be expressible, e.g. $d^\dagger_\downarrow f_{0\downarrow}$. Lee treats operators acting nontrivially only on sites $-1$ to $n_0$ as local (Sec. IV B, p. 8).

### 5.2 Abelian symmetries (`symmetry/`)

**[Amended 2026-10-19, TNOps refactor]** The behaviour specified in 5.2 stays authoritative; the locations change: 5.2.1–5.2.2 and 5.2.9 →
`TNOps/src/symmetry.jl` and `interface.jl`; 5.2.4 → the `TNOps` legs, `BlockTensor` and `SlicedTensor`; 5.2.5 → the
`TNOps` fusion and parity plus `nrg/tensors.jl` (`fuse_site`, `lift_site`, `refine_KK`); 5.2.7 → `TNOps.global_cut`,
called by `truncate_shell`; 5.2.3 and 5.2.6 stay in `mpNRG`.

This layer is written inside the package (ground rule 3). Everything above it (`nrg/`, `operators/`, `fdm/`, `psf/`) talks to it only through the interface of Section 5.2.9, so that a non-abelian layer can replace it later.

#### 5.2.1 Symmetry factors and sector labels

- A symmetry is a product $G=G_1\times\cdots\times G_K$ of factors of two kinds:
  - `U1(name)`, with integer labels, fused by addition;
  - `Zn(name, n)`, with labels in $\{0,\dots,n-1\}$, fused by addition mod $n$. `Z2` is the case $n=2$.
- Spin is handled as a `U1(:twoSz)` factor with integer label $2S_z$, so half-integers never appear.
- Charge can be declared as total charge `U1(:N)` or per flavor (for example `U1(:Nup)`, `U1(:Ndn)`, or one charge per orbital).
- A sector label is an `isbits` `NTuple{K,Int}` in canonical form (Z_n components reduced mod $n$). It must be hashable and cheap to compare.
- Provide:
  - `fuse(a, b)`: componentwise sum, mod $n$ for Z_n factors;
  - `dual(a)`: componentwise negation, mod $n$ for Z_n factors;
  - `neutral(G)`;
  - `isneutral(a)`.

  Unit-test associativity, commutativity, identity and inverse on random labels.

#### 5.2.2 Fermion parity is always present

- The layer **always** carries a fermion-parity factor `Z2(:Pf)`, even if the user declares no symmetry at all, or declares charges that already determine parity. The redundancy costs nothing.
- Parity is needed for two reasons:
  - fermionic signs (5.2.5) are computed from it;
  - it gives an exact selection rule: a PSF vanishes unless the number of odd operators is even.
- At construction, check that every local basis state has a parity label consistent with $(-1)^{N}$. If total charge is declared, check that `Pf` equals $N$ mod 2 on every sector.

#### 5.2.3 Charges of local modes and operator shifts

- The user assigns to each local fermionic mode $c_m$ its label $Q_m\in G$, i.e. the quantum numbers of $c^\dagger_m|0\rangle$. For the SIAM with $G={\rm U1}(N)\times{\rm U1}(2S_z)\times{\rm Z2}(P_f)$: $Q_{d\uparrow}=(1,+1,1)$ and $Q_{d\downarrow}=(1,-1,1)$. The bath-channel modes $f_{n\nu}$ get the label of the flavor they couple to.
- Every operator with a definite symmetry behaviour carries a **shift** $\Delta\in G$: it maps sector $q$ to sector $q\oplus\Delta$. Its fermionic parity is the `Pf` component of $\Delta$.
- Shifts compose in the operator algebra of Section 5.1:
  - $c^\dagger_m$ has shift $Q_m$, and $c_m$ has ${\rm dual}(Q_m)$;
  - a product has the fused shift of its factors;
  - an adjoint has the dual shift;
  - a commutator or anticommutator of homogeneous operators has the fused shift.

  For example, $q_\sigma=[d_\sigma,H_{\rm int}]$ has the shift of $d_\sigma$ whenever $H_{\rm int}$ is neutral. The composite $\{q_1,d^\dagger_2\}$ has ${\rm fuse}(\Delta_{q_1},\Delta_{d^\dagger_2})$, which is neutral for equal spins and carries $2S_z=\mp2$ for opposite spins.
- **Inhomogeneous operators.** A sum of terms with different shifts (for example $d_\uparrow+d_\downarrow$ under conserved $S_z$) has no single shift. `components(op)` decomposes any local operator into homogeneous components, each with its own shift.
  - The core PSF routines accept homogeneous operators only.
  - A convenience wrapper `psf_multilinear` expands an inhomogeneous call into the sum over component tuples.
  - Component tuples whose total shift is not neutral are skipped exactly. The trace forces $\Delta_1\oplus\cdots\oplus\Delta_\ell=0$, so they cannot contribute.
- The same selection rule prunes the permutation loop of `psfs` before any NRG work is done. Record the pruned tuples in the output metadata so that a zero is visibly a symmetry zero.

#### 5.2.4 Spaces and block-sparse operators

- A `SectorSpace` is an ordered list of (label, dimension) pairs, with a deterministic order (sorted labels) so that runs are reproducible.
- A `BlockOp{T}` holds:
  - a row space and a column space;
  - a shift $\Delta$;
  - a dictionary from column label $q$ to a dense `Matrix{T}` of size ${\rm dim}(q\oplus\Delta)\times{\rm dim}(q)$.

  Absent blocks are zero and are not stored. `T` is `Float64` unless the model or an operator is genuinely complex, and is then `ComplexF64`. Do not promote silently.
- Required operations:
  - `*`, with shifts fused; the inner spaces must match;
  - `+`, requiring equal shifts;
  - scalar multiplication;
  - `adjoint`, which dualizes the shift and conjugate-transposes the blocks;
  - `tr`, nonzero only for a neutral operator;
  - block-wise `eigen` for hermitian neutral operators, returning eigenvalues tagged with their sector;
  - block-wise `svd`, needed by the hook example of Section 5.10;
  - `todense(op, basis)`, which embeds into an explicit full matrix with a documented sector-to-index map, for tests only.
- **Operators with frequency legs.** A sliced operator carries one or two extra dense legs (bin indices) next to its two symmetry legs (Lee Eq. (60) and Sec. V D). Provide `SlicedBlockOp`, which stores per symmetry block either a `rows × cols × bins` array or a sparse map from bin to matrix; choose by measured fill.
  - Contractions over the symmetry legs act block-wise.
  - Bin legs pass through untouched, so that all bins are processed in parallel, as Lee does (p. 14).
  - Operators with two open bin legs (4p composites sliced in two frequencies) use the same structure with a bin pair.

#### 5.2.5 Adding a Wilson site: fusion and fermionic signs

- Adding site $n$ fuses the kept space of shell $n-1$ (sectors $q$) with the local space of site $n$ (sectors $\sigma$). The new sector is $q'=\sigma\oplus q$.
- For every $q'$, store the ordered list of contributing pairs $(\sigma,q)$ with their row offsets: the fusion map. It defines the product basis of every block of $H_n$ and is saved with the shell's isometries, because the operator expansions of Section 5.5 need it.
- Within a pair, order the product basis states exactly as in the fermionic convention of Section 4.3. With Lee's convention (basis built "as if filling the chain in the order $c^\dagger_N\cdots c^\dagger_{-1}|0\rangle$", p. 5), the new site's creators stand to the **left** of those of all earlier sites. Two consequences follow:
  - an operator $O$ acting on earlier sites, with parity $p_O\in\{0,1\}$, is represented in the new product basis as $(Z_n)^{p_O}\otimes O$, where $Z_n=(-1)^{N_n}$ is the parity of the new site;
  - an operator acting on the new site alone is represented as $O_n\otimes\mathbb 1$.

  Both factors are read off the `Pf` labels of the sectors involved, so **no sign is ever entered by hand**. Derive these two rules in `STATUS.md` from the convention before coding them, and verify them in V1. If the session adopts a different ordering, it must rederive and document the rules.
- Within a single site with several modes (for example $\uparrow,\downarrow$), the local Jordan–Wigner order is part of the local-space definition of Section 5.1 and is used consistently for the impurity and for every bath site.
- The hopping term $t_n\sum_\nu(f^\dagger_{n\nu}f_{n-1,\nu}+{\rm h.c.})$ is assembled from these representations as a product of block operators, never as a hand-signed matrix.

#### 5.2.6 Checking that the requested symmetry holds

- **Hamiltonian.** At model construction, embed the local Hamiltonian, every impurity–bath coupling term and every on-site bath term into the dense local Fock space. Check $\|[H,Q_k]\|\le$ tol for every U1 generator $Q_k$, and invariance under every Z_n factor.
  - On failure, the error names the offending term and the broken factor, and suggests dropping that factor. Examples: a transverse field breaks `U1(:twoSz)`; pairing terms break `U1(:N)`; spin–orbit coupling breaks `U1(:twoSz)`.
  - Never fall back to a smaller symmetry silently.
- **Operators.** Every operator passed to a PSF call must decompose into homogeneous components (5.2.3); the check is exact, by sector bookkeeping on the dense local matrix. Report the shift of each component in the call's metadata.
- **Hybridization.** Per-flavor charges are legitimate only if the hybridization is diagonal in flavor, which is the scope of this task (Section 5.3). Check that each bath channel is coupled only to impurity modes with the same label.

#### 5.2.7 Truncation across blocks

- The NRG cut is **global**. Collect the eigenvalues of all blocks of $H_n$, sort them together, take the $N_{\rm keep}$-th level (or the energy cutoff), and move the cut up into the nearest gap larger than a relative tolerance, so that no degenerate multiplet is split.
- Multiplets are often spread over several blocks: a spin doublet sits in $2S_z=\pm1$, and a triplet in $2S_z=-2,0,2$. A per-block cut would split them and break the symmetry the abelian labels cannot see. This is the reason the cut must be global.
- Record per shell:
  - the number kept per block;
  - the requested and actual $N_{\rm keep}$;
  - the gap at the cut.

  Expose them in the flow-diagram data.

#### 5.2.8 Performance

- Block dictionaries keyed by `isbits` labels; contiguous dense blocks; no per-element access in inner loops.
- Fusion maps are precomputed once per shell.
- Threads over blocks in eigendecomposition and in contractions (Section 5.4). Report the size of the largest block and the distribution of block sizes per shell, since the largest block sets the cost.

#### 5.2.9 Interface for a future non-abelian layer

`nrg/`, `operators/`, `fdm/` and `psf/` may use only:

- `fuse`, `dual`, `neutral`, `isneutral`, `parity`;
- the `SectorSpace` and `BlockOp`/`SlicedBlockOp` operations of 5.2.4;
- the fusion map of 5.2.5;
- `degeneracy(label)`, which returns 1 for every abelian label.

A non-abelian implementation would supply multiplet dimensions greater than 1, fusion outer multiplicities and Clebsch–Gordan data behind the same functions (see [R22] for what such a layer involves). Do not implement it now. Do not let any module above `symmetry/` assume that a block row is a single state rather than a multiplet: route every state count through `degeneracy`, including partition functions, traces and FDM shell weights.

#### 5.2.10 Tests (part of V1, phase P1)

**[Amended 2026-10-19, TNOps refactor]** The generic tests (labels, block algebra, fusion, factorizations) move to `TNOps/test/`; the
physics-facing tests (fermionic signs, symmetry checks, the shell construction) stay in `mpNRG/test/`.

- **Labels:** group axioms for `fuse` and `dual`; canonical forms of Z_n labels.
- **Block algebra:** for random `BlockOp`s with random shifts, compare against `todense`:
  - products, sums, adjoints and traces to $10^{-14}$;
  - block-wise `eigen` against dense `eigen` of the embedded matrix;
  - that a product of non-neutral total shift has zero trace exactly.
- **Fusion and signs:** build the first three Wilson sites both through the fusion map and through a dense Jordan–Wigner construction (independent code in `test/`). Check:
  - that the two agree after the documented basis permutation;
  - $\{c_i,c^\dagger_j\}=\delta_{ij}$ and $\{c_i,c_j\}=0$ for all impurity and chain modes;
  - that $H_n$ from blocks equals the dense $H_n$.
- **Symmetry violation:** a transverse field with `U1(:twoSz)` declared, and a pairing term with `U1(:N)` declared, must both raise the documented error.
- **Selection rules:** a 3p call with total shift $\neq0$ is pruned and returns an exact zero, recorded in the metadata. A 4p call such as $(d_\uparrow,d^\dagger_\uparrow,d_\downarrow,d^\dagger_\downarrow)$ in all 24 permutations is pruned exactly where the shift arithmetic says it should be.
- **Truncation:** a toy shell with an exact doublet split across $2S_z=\pm1$ and the cut falling inside it must keep or drop both members, and must record the moved cut.
- **Symmetry independence:** the same SIAM run with $G={\rm Z2}(P_f)$ only, with ${\rm U1}(N)\times{\rm Z2}(P_f)$, and with ${\rm U1}(N)\times{\rm U1}(2S_z)\times{\rm Z2}(P_f)$ must give identical energies, thermal averages and PSFs to $10^{-10}$. Use no truncation, or an $N_{\rm keep}$ cut that falls in the same gap in all three runs. This shows the symmetry layer changes efficiency, not results.

### 5.3 Bath, discretization and Wilson chain (`bath/`)

- Hybridization input:
  - a box $\Delta(\omega)=\Delta\,\theta(D-|\omega|)$, the model of Kugler Eq. (79) and Lee Sec. VII A;
  - an analytic function $\Delta(\omega)$;
  - tabulated data on a real-frequency grid (DMFT use);
  - one hybridization per flavor, diagonal in flavor.
- Discretization: logarithmic intervals bounded by $\pm D\Lambda^{-k-z}$ with shift $z\in(0,1]$ (Lee p. 23). The representative energies follow Žitko & Pruschke [R7], the prescription Lee uses (his Ref. [23]). Wilson's original prescription [R1, R14] is a selectable option; Campo–Oliveira [R8] is optional.
- Tridiagonalization of the star into the chain [R5, R14]:
  - run Lanczos or Householder in **extended precision** (`BigFloat` or `Double64`), because $t_n\sim\Lambda^{-n/2}$ and double precision loses orthogonality;
  - provide a two-precision self-check and accept only sites on which they agree.
- Outputs:
  - the chain $\{\epsilon_n,t_n\}$ and the impurity coupling;
  - the star parameters $\{E_k,V_k\}$ of the **finite** chain, i.e. the eigen-decomposition of the chain hopping matrix. These are used for ED cross-checks (Section 6, V3).

### 5.4 Iterative diagonalization (`nrg/`)

- Forward sweep: add one site, build $H_n$ per symmetry block, and diagonalize with dense `eigen` per block.
- Truncation by $N_{\rm keep}$ or by an energy cutoff in rescaled units, or both. **Never split a degenerate multiplet**: move the cut into the nearest gap larger than a relative tolerance, and record where it was moved.
- Store for every shell, per block:
  - the isometries ($A$ tensors, Lee Eq. (15));
  - energies with sector labels (K or D);
  - the shell's ground-state shift.

  $n_0$ is the last shell diagonalized without truncation (Lee p. 5).
- Expose flow-diagram data (lowest rescaled levels per symmetry sector versus $n$) and memory/time statistics per shell.
- Thread over blocks. Do not thread over shells, because the sweep is sequential.

### 5.5 Operator expansions (`operators/`)

- Sector projections $\mathcal O^n_{X\bar X}$, with $X,\bar X\in\{K,D\}$, of a local operator in the shell-$n$ basis (Lee Sec. IV A–B, Eqs. (21)–(37)). These are built in the forward sweep from the local operator, with fermionic parity strings handled by the convention of Section 4.3.
- Lee states that the shell-diagonal expansion is **exact for local operators** (p. 9). Make this a test (V4b).
- Products of operators expanded shell-diagonally (Lee Sec. IV F, Eqs. (49)–(54)), and frequency-dependent (nonlocal) operators $\mathcal O_\varepsilon$ (Lee Sec. IV C).
- Composite KK operators carried from shell $n$ to later shells, i.e. the refinement map of a KK block of shell $n-1$ into all sectors of shell $n$ (Lee Eq. (34)). This map is the workhorse of the 3p and 4p refinements, and it is also what the hook interface must expose.

### 5.6 Full density matrix (`fdm/`)

- Backward sweep from $N$ to $n_0$:
  - the density matrix in the Anders–Schiller basis, with the shell weights of Weichselbaum & von Delft [R13];
  - the reduced density matrices $\bar\rho^n_{XX}$, in particular $\bar\rho^n_{KK}$ (Lee Sec. IV D, Eqs. (40)–(45)).
- Thermal averages $\langle AB\cdots\rangle$ of products of local operators through shell traces (Lee Sec. IV E, Eqs. (46)–(48)).
- Checks: ${\rm Tr}\,\varrho=1$, $\bar\rho\ge0$ (smallest eigenvalue reported), and shell weights summing to one.

### 5.7 PSF engine, $\ell=2,3,4$ (`psf/`)

- **2p** (Lee Sec. V A): the shell expansion of Eq. (56), with traces Eq. (57)/(58) and the final formula Eq. (59), summed over all shells $\ge n_0$ and sectors $X\bar X\neq KK$.
- **Binning and slicing** (Lee Sec. V B):
  - binning moves each weight to its bin centre;
  - slicing stores a frequency-resolved matrix $([\mathcal O^n_{X\bar X}]_{s\bar s})_{\underline\varepsilon}$ with an **open frequency leg**, Eq. (60), so that all bins are processed in parallel.
- **3p** (Lee Sec. V C):
  - five of the seven sectors ($D\bar X\hat X$ and KDD) are evaluated as shell traces, binning both frequencies;
  - the KKD and KDK terms are refined as 2p PSFs of a single and a composite operator (Eqs. (64b), (65b)), expanded over later shells $\bar n>n$ (Eqs. (66a), (66b));
  - only the KD and DK sectors are sliced, never DD (p. 15).
- **4p** (Lee Sec. V D):
  - nine of the 15 sectors are evaluated as shell traces;
  - three (KKDD, KDKD, KDDK) go through 2p PSFs of sliced composite doubles or triples (Eqs. (70a)–(70c));
  - three (KKKD, KKDK, KDKK) go through 3p PSFs of sliced composites (Eqs. (71a)–(71c)), which recurse once more into 2p refinements.
- **Design requirement**, taken from Lee p. 18: write *one* 2p routine and *one* 3p routine "which work irrespective of how their input operators have been generated and whether they are sliced or not". The refinements then arise from **nested calls**, not from separate hand-written routines per diagram.
- Also follow Lee's advice (p. 15) to associate the original frequencies $\varepsilon_i$ with operators rather than $-\varepsilon_i$ wherever possible. This keeps the recursion organized.
- **Equivalent trace forms.** Lee gives two forms for 2p (Eqs. (55a)/(55b)), two for 3p (Eqs. (61a)/(61b)) and five for 4p (Eq. (67)). Implement the forms Lee uses for each sector, and expose an option to evaluate a PSF through an alternative form for diagnostics (V6).
- **Interface:**
  - `psf(run, ops::Tuple; grid, kwargs...)` takes operators **in permuted order** and returns one histogram object.
  - `psfs(run, ops::Tuple; grid, perms = :all)` loops over the requested permutations.
  - Options: `abs = true` also returns the histogram of $|{\rm weight}|$ per bin. `exact = true` returns unbinned Lehmann terms (positions with exact NRG energy differences, weights, and sector/shell labels) for small runs, with a hard memory guard.
- Report the cost per $\ell$ and compare it with Lee's estimates ($\sim N$ for the adequately refined terms, $\sim N^2$ for 2p refinements, $\sim N^3$ for the 4p terms that need a 3p PSF for every $n$; p. 18).

### 5.8 Outputs and I/O (`io/`)

- Each run file (JLD2) stores:
  - all model parameters and the full NRG parameters ($\Lambda$, $z$, discretization scheme, $N$, $N_{\rm keep}$, truncation energy, $T$, tolerances);
  - the grid;
  - the git commit and package version;
  - the operators (as symbolic expressions);
  - per-permutation histograms;
  - thermal averages used as sum-rule references.
- $z$-averaging: average histograms bin by bin over runs with $z=1/n_z,\dots,1$ (Lee p. 23). Refuse to average runs whose grids or operators differ.
- Export:
  - histograms to `Vector{HubbardAtom.PSFTerm}`-compatible data (bin centre → position, bin weight → weight). Write the converter in `test/` and the notebook so that `src/` stays independent of `HubbardAtom`; `src/` returns plain arrays;
  - plain `Array`/`Dict` dumps for other languages.

### 5.9 Observables (`observables/`)

- Thermodynamics:
  - impurity entropy and susceptibility from FDM thermal averages, and from the conventional shell-wise formulas [R2, R14]; both, cross-checked;
  - occupation and double occupancy.
- 2p spectral functions with broadening. Implement log-Gaussian broadening with the linear crossover at $T$ (Lee App. C, Eqs. (C1)–(C2); [R13, R18]). The adaptive scheme [R18] is optional.
- Do **not** broaden 3p/4p PSFs (Lee Sec. VI B: broadening must not spoil cancellations in the permutation sum; out of scope).

### 5.10 Hooks (`hooks/`)

A per-shell callback interface, invoked during the PSF forward sweep: `on_shell(callback, run, shell_index, ctx)`. `ctx` exposes read-only views of:

- the shell isometries;
- $\bar\rho^n_{KK}$;
- the sector-projected operators of the current PSF call;
- the refinement map of KK operators into shell $n$;
- the sliced composite operators that are being carried (with their open bin index and creation shell).

Document the memory layout. This is the interface a future in-sweep compression of the far-field KKD/KDK terms will use. Provide one example callback that only records the singular values of the carried composites per shell, to demonstrate the interface. Do not implement any compression.

### 5.11 Built-in models (`model/`)

- SIAM with arbitrary $\epsilon_d$, $U$, magnetic field and per-spin hybridization.
- Resonant level model ($U=0$).
- Kondo (s-d) model.
- Two-orbital Anderson model with a generic local interaction given as an operator expression (no built-in Kanamori required, but provide an example).
- A user-defined model from (local space, local Hamiltonian, flavor → channel coupling, hybridizations, symmetries).

### 5.12 Reference ED (`reference/`)

- Dense full-Fock-space ED of a **finite** Wilson chain (or its star form), with exact Lehmann PSFs for $\ell=2,3,4$ in the convention of Section 4.4.
- It must reuse only `fock/` and `bath/`, not `nrg/`, so that it is an independent check.
- It is small and slow by design. Guard against sizes beyond a few thousand states.

---

## 6. Validation ladder

Each item is a test file and a notebook section. Tolerances are relative to the largest weight unless stated. **V1–V4 are exactness tests: a failure stops the work (rule 6).**

**V1. Algebra.** Includes the symmetry-layer tests of Section 5.2.10.
- Canonical anticommutators $\{c_i,c_j^\dagger\}=\delta_{ij}$ for impurity and chain modes, built through the shell construction, on chains up to $N=4$.
- Hermiticity of $H_n$; block structure consistent with the declared symmetries; parity of composite operators.

**V2. Chain.**
- Sum rule: $\sum_k|V_k|^2$ equals $\frac1\pi\int d\omega\,\Delta(\omega)$ for every $z$. For the box this is $2D\Delta/\pi$, as in the repository's `discretize_box`.
- The spectral measure of the chain at site 0 reproduces the discretized star exactly: moments to $10^{-12}$ in extended precision.
- $t_n\Lambda^{n/2}$ approaches a constant.
- The two-precision self-check passes.

**V3. No-truncation exactness against ED.** Short chain; $N_{\rm keep}$ at least the full dimension at every shell, so that every state is discarded only at $n=N$ and the Anders–Schiller basis is the exact eigenbasis. Compare:
- energies;
- partition function;
- thermal averages;
- 2p, 3p and 4p PSFs for all permutations, in both `exact = true` mode and binned mode (against identically binned ED data).

References:
- (a) `reference/` ED, for any model and any $\epsilon_d$;
- (b) the repository's `HubbardAtom.AIMModel(U, β, ε, V)`, which is fixed at $\epsilon_d=-U/2$, with $(\varepsilon,V)$ = the star parameters of the finite chain (Section 5.3). Unitary equivalence of chain and star makes the impurity PSFs identical;
- (c) the Hubbard-atom limit ($V\to0$), against `HubbardAtom`'s exact atom PSFs.

Operators:
- $d_\sigma$, $d^\dagger_\sigma$, $n_{d\sigma}$;
- the composite $q_\sigma=[d_\sigma,H_{\rm int}]$;
- the bosonic composite $\{q_1,d^\dagger_2\}$;
- the impurity–site-0 operator $d^\dagger_\downarrow f_{0\downarrow}$.

Target $\le10^{-10}$. For 4p use $N_b\le3$ star sites, or the repository's sparse `psf_chain`/`psf4_tables_chain` routines, to keep ED affordable.

V3 exercises no kept/discarded bookkeeping (everything sits in the last shell), hence V4.

**V4. Exactness with truncation.** Tiny chains (full Hilbert space $\lesssim 4^6$) with $N_{\rm keep}$ well below the full dimension, so that KK, KD, DK and DD sectors and all refinement terms occur.
- (a) **Weight sum rules.** The total weight of every PSF equals the FDM thermal average ${\rm Tr}[\varrho\,\mathcal O^1\cdots\mathcal O^\ell]$, computed both by `fdm/` and by brute force in the explicitly constructed Anders–Schiller basis with $\varrho$ assembled from the stored shell data. Binning and slicing move positions only, so this is expected to hold to $\sim10^{-12}$. If it does not, find out why before going on, and record the finding.
- (b) The shell-diagonal expansion of each local operator reassembles the exact full-chain matrix (Lee p. 9: exact for local operators).
- (c) ${\rm Tr}\,\varrho=1$; $\bar\rho^n\ge0$.

**V5. Resonant level model ($U=0$).**
- The single-particle problem of the finite chain is solvable exactly. Compare NRG 2p PSFs (with truncation) bin by bin with the exact discrete spectral function; total weights must be exact (V4a).
- After broadening, compare with the continuum $A(\omega)$ of the box model.
- Report deviations; there is no hard threshold beyond V4a.

**V6. Internal consistency.**
- The equivalent trace forms (Section 5.7) agree.
- Total weights are independent of $z$.
- Increasing $N_{\rm keep}$ converges, with the convergence reported.

**V7. Physics benchmarks (SIAM, box hybridization).**
- Entropy flow $\ln4\to\ln2\to0$.
- $T\chi$ crossover; $\langle n_d\rangle=1$ at particle–hole symmetry.
- Friedel sum rule $\pi\Delta A(0)\approx1$ for $T\ll T_K$ at particle–hole symmetry.
- Optionally, a 2p cross-check with NRG Ljubljana [R25].

**V8. External multipoint benchmark.**
- Download the published mpNRG PSFs of Frankenbach et al. [R23]. First inspect and document their format, parameters and conventions.
- Run the same model and parameters, and compare 4p (and any 2p/3p) PSFs at the level the data allow: total weights, marginals, and histograms after identical binning.
- Agreement is not a hard gate if their NRG used non-abelian symmetries or settings this code cannot reproduce. Report what differs and why.

---

## 7. Phases

Each phase ends with:

- tests passing (`julia --project=mpNRG/test mpNRG/test/runtests.jl`, or the equivalent);
- an updated `STATUS.md`;
- a notebook section;
- one commit.

Before coding a phase, read the cited pages of the papers from the page images and list in `STATUS.md` the equations implemented.

| Phase | Content | Acceptance |
|---|---|---|
| P0 | Package skeleton in `mpNRG/`, `Project.toml`, test environment with `HubbardAtom` path source, `.gitignore` for `refs/`, `README.md` with conventions (Section 4), `STATUS.md`, `docs/TASK.md` | `using MultipointNRG` works; empty test suite runs |
| P1 | `fock/`, `symmetry/` (Section 5.2) | V1 (local parts), all of Section 5.2.10 |
| P2 | `bath/`: hybridizations, discretizations, extended-precision tridiagonalization, star of the finite chain | V2 |
| P3 | `nrg/`: forward sweep, truncation, shell storage, flow diagrams; `reference/` ED | V1 (chain parts); spectra in V3 |
| P4 | `operators/`: sector projections, local and site-0 operators, products, KK refinement map | V4b |
| P5 | `fdm/`: backward sweep, reduced density matrices, thermal averages; thermodynamics | V3 (thermal averages), V4c, V7 thermodynamics |
| P6 | `psf/`: grid, histograms, binning, 2p PSFs; `io/`, $z$-averaging; 2p broadening | V3 (2p), V4a (2p), V5, V7 |
| P7 | 3p PSFs with slicing, refinement of KKD and KDK | V3 (3p), V4a (3p), V6 |
| P8 | 4p PSFs via nested 2p/3p calls | V3 (4p), V4a (4p), V6, V8 |
| P9 | `hooks/`, built-in models, examples, export to `HubbardAtom`, root `README.md` status rows, final notebook rendered to PDF | All tests green; notebook executed and committed |
| P10 (optional) | Performance: profiling, threading, memory of sliced tensors, timing table for SIAM at $\Lambda=4$ versus $N_{\rm keep}$ and $\ell$ | Table in `STATUS.md` |

Practical guidance:

- **Kept states.** Lee keeps 300 *multiplets* with U(1) charge × SU(2) spin symmetry (p. 23). With abelian symmetries only, more *states* are needed for comparable accuracy. Determine this by convergence (V6) rather than assuming a number.
- **Chain length.** Choose the length from $T$ (Section 4.2).
- **Validation in the notebook** uses $\Lambda=4$, $n_{\rm dec}=32$ and $n_z=2$ or $4$, as in Lee.

---

## 8. Deliverables

1. `mpNRG/` package as in Section 3, all tests passing, commits on `claude/practical-archimedes`.
2. `mpNRG/README.md` covering:
   - quick start (SIAM 2p, 3p, 4p);
   - the conventions of Section 4;
   - the list of approximations and where they enter;
   - how to export PSFs to `HubbardAtom` kernels.
3. `mpNRG/STATUS.md` with:
   - the verification record, item by item from Section 6, with achieved accuracies;
   - every deviation from Lee 2021 and why;
   - open issues.
4. `mpNRG/notebooks/mpnrg_validation.ipynb` (executed, with outputs) and its PDF.
5. Root `README.md`: a short `mpNRG` block in the status section, in the existing table style (sheet / check / accuracy).
6. **[Amended 2026-10-19, TNOps refactor]** The validation notebook re-executed on the refactored code, its PDF regenerated, and its section
   "Refactor regression (TNOps)" (baseline comparison and timing, refactor sheet Sec. 9.4).

---

## 9. Future extensions the design must not block

These are **not** part of this task.

- **Non-abelian symmetries** (SU(2) spin, SU(N) orbital) through a replaceable symmetry layer [R22].
- **Matrix-valued hybridization** (multi-orbital DMFT, clusters), which needs block tridiagonalization.
- **In-sweep compression** of the KKD/KDK far field through a truncated running basis of carried composites (uses `hooks/`).
- **Broadening of multipoint PSFs**, connected parts and symmetric improved estimators [R19, R20], built downstream on exported histograms.
- **Bosonic baths** and other local degrees of freedom.
