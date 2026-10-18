"""Extract the mpNRG PSFs of Frankenbach et al. (LMU Open Data 613,
https://data.ub.uni-muenchen.de/613/) into a plain HDF5 file readable by JLD2.

Usage: python3 extract_lmu613.py <dataset dir> <output.h5>

The dataset directory is one unzipped set, e.g. siam05_U0.05_T0.05_Delta0.0318.
Its .mat files are MATLAB 7.3 (HDF5). We copy:
  - parameters of mpNRG.mat (Lambda, N, Nkeep, nz, T, U, epsd, Delta field,
    max of Rhoin = true Delta / pi, emin, emax, estep),
  - the frequency grid `odisc`,
  - the 2p PSFs (F1,F1dag), (F1dag,F1), (Q1,Q1dag), (Q1dag,Q1), (F1,Q1dag), (Q1dag,F1),
    (Q1,F1dag), (F1dag,Q1) (z-averaged; directory PSF_nz=2_conn_zavg),
  - every 4p PSF of the 24 permutations of (F1,F1dag,F3,F3dag) in 4pt/ (connected
    parts, two flavor components each: [0] = sigma sigma, [1] = sigma sigma').
Arrays are stored such that Julia (JLD2) reads them with MATLAB's index order.
"""
import sys, os, itertools
import h5py, numpy as np

src, out = sys.argv[1], sys.argv[2]
psfdir = os.path.join(src, "PSF_nz=2_conn_zavg")

def cell_arrays(f, key):
    return [f[r][()] for r in f[key][()].ravel()]

with h5py.File(out, "w") as o:
    m = h5py.File(os.path.join(src, "mpNRG.mat"), "r")
    for k in ["Lambda", "N", "Nkeep", "nz", "T", "U", "epsd", "Delta", "emin", "emax", "estep"]:
        o["params/" + k] = float(m[k][()].ravel()[0])
    o["params/Rhoin_max"] = float(m["Rhoin"][()].max())
    names2 = ["F1,F1dag", "F1dag,F1", "Q1,Q1dag", "Q1dag,Q1", "F1,Q1dag", "Q1dag,F1", "Q1,F1dag", "F1dag,Q1"]
    for nm in names2:
        f = h5py.File(os.path.join(psfdir, "PSF_((%s)).mat" % nm), "r")
        if "odisc" not in o:
            o["odisc"] = f["odisc"][()].ravel()
        a = cell_arrays(f, "Adisc")
        o["psf2/" + nm.replace(",", "_")] = np.asarray(a[0]).ravel()
    ops = ["F1", "F1dag", "F3", "F3dag"]
    for p in itertools.permutations(range(4)):
        nm = ",".join(ops[i] for i in p)
        fn = os.path.join(psfdir, "4pt", "PSF_((%s)).mat" % nm)
        f = h5py.File(fn, "r")
        a = cell_arrays(f, "Adisc")
        for j, arr in enumerate(a):
            # h5py shows MATLAB's column-major array with reversed axes; Julia
            # (column-major) reverses them again on reading, so storing the array
            # as read gives Julia the original MATLAB index order (eps1, eps2, eps3)
            o["psf4/%s/flavor%d" % ("".join(str(i + 1) for i in p), j + 1)] = np.ascontiguousarray(arr)
print("written", out)
