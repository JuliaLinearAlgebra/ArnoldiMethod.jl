# Implicitly Restarted Arnoldi Method (IRAM) in Julia

## Goal
Remove the ARPACK dependency of Julia and make `eigs` a native Julia function.

## Status
Work in progress!

- [x] Minimal working example of implicit restart in complex arithmetic
- [ ] Efficient QR iterations in implicit restart: Given's rotations or maybe LAPACK methods.
- [ ] Real arithmetic with real matrices; handle conjugate eigenpairs in the Schur decomp
- [ ] Targeting of eigenvalues
- [ ] Locking of converged Ritz vectors
- [ ] Generalized eigenvalue problems
