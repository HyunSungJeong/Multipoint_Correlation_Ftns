"""
    MultipointNRG

A general-purpose full-density-matrix numerical renormalization group (fdm-NRG)
for quantum impurity models with a finite local Fock space. It computes
thermodynamics, thermal expectation values and local 2p, 3p and 4p partial
spectral functions (PSFs) following

    S.-S. B. Lee, F. B. Kugler and J. von Delft, *Computing local multipoint
    correlators using the numerical renormalization group*,
    Phys. Rev. X **11**, 041007 (2021)   ["Lee" in the docstrings]

with the PSF convention of Kugler, Lee & von Delft, PRX **11**, 041006 (2021),
Eq. (28) ["Kugler"], identical to `HubbardAtom.psf`.

Layers (task sheet Sec. 3): `symmetry/` (abelian labels, block-sparse
operators), `fock/` (modes, operator algebra, Jordan-Wigner), `bath/`
(hybridizations, discretization, Wilson chain), `model/`, `nrg/` (forward
sweep), `operators/` (shell projections), `fdm/` (density matrix), `psf/`
(grids, histograms, the 2p/3p/4p engine), `observables/`, `io/`, `hooks/`, and
`reference/` (dense ED of finite Wilson chains, an independent check).
"""
module MultipointNRG

using LinearAlgebra
using Printf
using JLD2


end # module
