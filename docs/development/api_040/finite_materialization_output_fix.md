# Finite materialization: full Program output authority and scope

The first installed reception on MAIN 70c4724 failed compiling the complete
M09 Program. The earlier host tests compiled the genuine emitted residual and
reconstruction bodies but did not cover the surrounding Program allocation and
status-consumption scopes. They remain useful, but were insufficient integration
coverage. The native failed.cpp artifacts in
`outputs/installed-finite-amr-native-serial` are preserved as the original failure.

Two production defects were fixed, without changing the equations, residual
threshold, native numerical provider or headers:

1. The finite co-location vote enclosed the entire kernel body in a local block.
   `expression_active_ID` was declared inside, then used by status reduction
   outside. Close only the vote's local block before declaring the mask and
   executing the kernel. The layout vote still precedes every `fab(li)` access.
2. Top-level `pointwise_expression` still allocated both scratches from its first
   input. The original potential residual has an 8-component first input but an
   explicit 4-component potential output. A common `pointwise_output_template`
   helper authenticates the output metadata and now drives the output scratch,
   scalar status scratch and kernel source/template. Existing ordinary pointwise
   expressions retain their previous first-input allocation.

`test_finite_m09_full_program.py` first failed 4/4: two source assertions exposed
8→4 prototype mismatch and two complete C++ translation units exposed out-of-scope
masks. After the fix all four pass. It compiles the entire generated monolithic
and condensed Program with real repository/Kokkos headers, shared-exception ABI
contract, Dim 2 and OpenMP flags when required by installed Kokkos. MPI headers,
when available, additionally enable the actual MPI branches. This is syntax-only,
not DSO installation or a native execution claim.

The coherent finite source/host suite passed 16/16 before the additional MPI syntax
configuration; the four checks were then rerun with MPI branches enabled and
passed 4/4 (9.36s).
Installed serial/MPI reception remains owned by the primary worker. No package
installation or shared build was performed here.
