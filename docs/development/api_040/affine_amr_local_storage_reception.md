# Affine moments on AMR local storage

The public affine moment body can be evaluated at a Program stage and inside a
`LocalResidual` without a physical face flux. Such a block resolves to the exact
`StateStorage` authority of its Case block. Previously AMR resolution rejected it
before hierarchy construction with `AMR resolution requires resolved numerics`.

AMR resolution now carries the resolved block/state pair when that block has
`StateStorage` and no numerical flux plan. The hierarchy derives its stencil
buffer from that storage's ghost depth and records zero reflux need. The same
authority sets the coarse/fine fill accuracy requirement; an unowned transfer
subject still fails. Direct typed value tagging works; a discrete gradient tag
continues to require a resolved spatial discretization. No `ResolvedNumerics`,
flux, or reflux operation is synthesized.

`test_affine_push_forward_amr_runtime.py` is the installed reception. Four
weighted 2V particles define six moments of total order at most two. The
public body applies a fixed affine velocity map; a distinct 1.1-times seed
solves its original residual against a captured mapped stage. The state has
exact linear-in-x cell averages. At one and two AMR levels, the test checks
initial and accepted moments against independently moved particles on every
valid cell; the two-level case requires partial fine coverage. The acceptance
limit is absolute `3e-12`, relative zero, chosen before native execution. A
separate impossible first residual tests collective refusal, unchanged levels,
patches, time and bitwise states on two attempts.

Source reception: six tests in `test_local_state_storage_amr.py` pass
validate/resolve/Program emission, exact storage and nesting checks. The three
installed variants are collected but have **not** been executed by this
worktree; native compilation, MPI and rollback qualification belong to the
central installed build. This is a local Program/AMR transfer witness, not a
claim about transport fluxes or reflux.
