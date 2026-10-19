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

Layers (task sheet Sec. 3): the sibling package `TNOps` (abelian labels,
block-sparse tensors, fusion; re-exported), `fock/` (modes, operator algebra, Jordan-Wigner), `bath/`
(hybridizations, discretization, Wilson chain), `model/`, `nrg/` (forward
sweep), `operators/` (shell projections), `fdm/` (density matrix), `psf/`
(grids, histograms, the 2p/3p/4p engine), `observables/`, `io/`, `hooks/`, and
`reference/` (dense ED of finite Wilson chains, an independent check).
"""
module MultipointNRG

using LinearAlgebra
using Printf
using JLD2
# the abelian block-sparse layer (symmetry labels, sector spaces, block
# operators, fusion) is the sibling package TNOps; its names are re-exported
using TNOps
import TNOps: fusionmap
# local Fock spaces and operator algebra
include("fock/operators.jl")
include("fock/fockspace.jl")
# bath, discretization, Wilson chain
include("bath/hybridization.jl")
include("bath/discretization.jl")
include("bath/chain.jl")
# models
include("model/model.jl")
include("model/builtin.jl")
# the NRG tensor layer (named contractions on TNOps tensors), forward sweep
include("nrg/tensors.jl")
include("nrg/sweep.jl")
# operator projections, density matrix
include("operators/projection.jl")
include("fdm/fdm.jl")
# partial spectral functions
include("psf/grid.jl")
include("psf/engine.jl")
include("psf/api.jl")
include("hooks/hooks.jl")
# observables and I/O
include("observables/thermo.jl")
include("observables/broadening.jl")
include("io/io.jl")
# reference ED (independent of nrg/)
include("reference/ed.jl")

export SymFactor, U1, Zn, Z2, Sector, SymmetryGroup, sector, labeltype, factor_index, factornames,
       fuse, dual, neutral, isneutral, parity, degeneracy,
       SectorSpace, sectors, dim, offsets, BlockOp, getblock, setblock!, target, blockeigen, blocksvd,
       todense, identity_op, restrict, SlicedBlockOp, mapslices_op, nslices,
       FusionPair, FusionMap, fusionmap, kron_op,
       LegSector, Leg, BlockTensor, SlicedTensor, identity_tensor, eachblock, blockkeys, block_view,
       Mode, Ladder, OpExpr, cdag, cop, num, identity_expr, comm, acomm, normalorder, modes, maxsite,
       wordshift, components, opshift,
       FockBasis, densematrix, statecharge, occupation, LocalSpace, localop, parity_op, localspace_sectors,
       Hybridization, BoxHybridization, FunctionHybridization, TabulatedHybridization, integrate, total_coupling,
       bandwidth, Discretization, interval_edges, StarBath, discretize, zitko_energies,
       lanczos_chain, WilsonChain, wilson_chain, chain_star, star_moments, chain_moments, chainlength, nsites,
       BathChannel, ImpurityModel, impurity_modes, site_modes, site_space, charges, hamiltonian_expr,
       site0_hamiltonian, check_symmetry, check_model_symmetry, wilson_chains,
       spin_symmetry, siam, resonant_level, kondo, two_orbital, kanamori, spin_ops,
       NRGParams, chain_length_for, Shell, NRGRun, run_nrg, shell, keptspace, discspace, krange, drange,
       nkept, ndisc, totalkept, totaldisc, energy_scale, truncate_shell, dense_embedding, ProductEmbedding,
       LocalOperator, local_operator, shell_operator, refine, project, sector_view, xrange,
       FDM, fdm, weight, rhoD, rhoK, rho_full, fdm_checks, thermal_average, shell_trace, thermal_average_product,
       LogGrid, nbins, zerobin, centres, binindex, PSFResult, psf, psfs, psf_multilinear, add_psf, all_perms,
       histogram, nonzero_bins, bin_exact, bin_terms, total_weight, npoint,
       CarriedComposite, ShellContext, isometries, fusionmap, rhoKK, refinement_map, call_operators, carried,
       on_shell, SingularValueRecorder,
       bath_model, fdm_thermodynamics, impurity_thermodynamics_fdm, shell_thermodynamics,
       impurity_thermodynamics_shell, evenodd_average, impurity_thermodynamics_zavg, delta_SL, delta_F, broaden,
       git_commit, run_metadata, psf_dict, save_results, load_results, zaverage, zaveraged_psfs,
       EDResult, ed_reference, ed_operator, ed_thermal_average, ed_psf, chain_modes, chain_allowed

end # module
