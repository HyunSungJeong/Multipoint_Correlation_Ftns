# TNOps — status

Task sheet: [`../mpNRG/docs/TNOPS_REFACTOR.md`](../mpNRG/docs/TNOPS_REFACTOR.md).
`TNOps` is the abelian block-sparse tensor package that `MultipointNRG` is
refactored onto.

## Starting point and environment

- Refactor base: `bb8b1976c98027c7e229de493614add5444d6a05` (`bb8b197`) on
  `claude/practical-archimedes`, tagged locally as `pre-tensor-refactor`. The
  tag push was refused by the session's git proxy (it accepts branch pushes
  only), so the hash is recorded here and in `mpNRG/STATUS.md`.
- Julia 1.13.0 (conda-forge), packages from the General registry via git
  (`JULIA_PKG_SERVER=""`), 4 threads.

## Phase log and pushes

Every phase is pushed to `claude/practical-archimedes` as a fast-forward and
checked with `git ls-remote origin refs/heads/claude/practical-archimedes`.

| Phase | Content | Pushed commit |
|:--|:--|:--|
| R0 | regression baseline and fixtures in `mpNRG/test/baseline/` | `debf22af2b97225347e05c0670c5fa7fd6453896` |
| R1 | `TNOps/` package: `git mv` of the symmetry layer, module wrapper, tests V1a/V1b | `9fb953612984b11a392d8d1d41ca3dcfd359d232` |
