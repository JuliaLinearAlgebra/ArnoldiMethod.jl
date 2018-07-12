# Implicitly Restarted Arnoldi Method (IRAM) in Julia

[![Build Status](https://travis-ci.org/haampie/IRAM.jl.svg?branch=master)](https://travis-ci.org/haampie/IRAM.jl) [![codecov](https://codecov.io/gh/haampie/IRAM.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/haampie/IRAM.jl)

## Docs
[![](https://img.shields.io/badge/docs-latest-blue.svg)](https://haampie.github.io/IRAM.jl/latest)

## Goal
Remove the ARPACK dependency of Julia and make `eigs` a native Julia function.

## Status
Work in progress!

- [x] Minimal working example of implicit restart in complex arithmetic
- [x] Efficient QR iterations in implicit restart: Given's rotations or maybe LAPACK methods.
- [x] Real arithmetic with real matrices; handle conjugate eigenpairs in the Schur decomp
- [ ] Targeting of eigenvalues
- [x] Locking of converged Ritz vectors
- [ ] Generalized eigenvalue problems
