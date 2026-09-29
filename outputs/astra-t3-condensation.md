# T3: singular public condensation

The public API accepts a model-owned `local_linear_operator`, registered through
`model.operator`, and `Program.condensed_coeffs` on its exact state. For
`J=diag(0,1,1)`, subset `(1,2)`, `th_dt=1`, the eliminated matrix is zero.
This is a valid authoring request whose numerical elimination must fail. The
complete coupled system may still be invertible; the condensation has a
strictly smaller domain, as stated by C17 and the handoff's condensation contract.

The generated coefficient, flux and reconstruction kernels ignored the bool
returned by `block_inverse`/`block_apply_inverse`. A singular inverse leaves
the output array unwritten, contrary to the old emitter comment promising
nonfinite output. Compiling the actual emitted coefficient cell with the actual
native header and clang's `-ftrivial-auto-var-init=zero` produced a finite identity
tensor from this singular block. Initial receipt: **1 failed in 2.59 s**,
`outputs/t3-condensation-red.xml`.

The bounded emitter correction consumes the bool before any inverse read. Each
kernel uses the existing native `for_each_cell_reduce_max`, including the AMR
grown boxes it actually evaluates, then calls
`consume_pointwise_evaluation_status` before halos or subsequent uses. Nonfinite
inverse and final kernel outputs also fail. Successful inverse arithmetic is
unchanged. Reconstruction remains inside the existing accepted-step transaction;
this patch does not introduce a separate publication protocol.

Validation: **29 passed in 14.03 s** across the new public/source/native-fragment
tests and the existing mapped-condensation suite. Six tests compile and execute
small actual inverse/apply fragments; six inspect the three production routes
through Uniform and AMR emitters. No complete native mesh, MPI, or GPU execution
was performed. Receipt: `outputs/t3-condensation-green.xml`.

A separate adversarial intrinsic finding was retained: a large well-conditioned
diagonal matrix can overflow its determinant and return a finite zero inverse.
The bool-consumption patch alone does not solve that issue. An independent
intrinsic scaling correction and numerical-policy decision follow separately;
the overflow case is not presented as qualified by this emitter commit.

The general mixed-product global residual API, original full-system residual
after condensation, and arbitrary cross-state condensations remain outside this
bounded correction. No dense global mesh matrix or model-specific opcode was
introduced.
