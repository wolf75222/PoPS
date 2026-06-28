# Validated boundaries

This page lists boundaries that are part of the current contract. It is not a
place to document missing plumbing for public APIs. If a public feature lacks
codegen or runtime support, fix the implementation or remove the public surface.

## Dimension

The core is two-dimensional today. This is a structural invariant of the C++
storage, index space, finite-volume kernels, Poisson operators, and AMR
hierarchy.

## GPU validation

GPU validation is performed on GH200 outside the standard CI matrix. The public
docs should cite the backend coverage matrix for exact evidence rather than
making broad claims.

## Python compile toolchain

The symbolic and time-program routes compile generated C++ against the installed
headers and the loaded `_pops` ABI. A working C++ toolchain is required for those
routes.

## FFT field solves

FFT-style field solvers are mathematical descriptors with stricter requirements
than geometric multigrid. They may require a uniform periodic mesh and constant
coefficients. Their descriptors must reject incompatible layouts before runtime.

## AMR

AMR is a public layout. Missing codegen, missing bindings, or missing install
paths are not documented limitations. Public AMR incompatibilities must come
from descriptors that declare concrete mathematical or backend requirements.

## Experimental helpers

`pops.experimental` is not production API. It may contain host debug helpers or
prototype utilities. Tutorials should not depend on it.
