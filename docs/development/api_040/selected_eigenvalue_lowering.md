# Selected-State eigenvalue lowering

The multi-StateSpace physics facade stores authored `waves` in the shared
Module's `_eigenvalues`, but `_module_to_model` builds a private native emitter
for one exact `StateSpace`. Before this change, lowering the A view rebound B's
wave expression even when only B owned a grid flux. With a B-dependent wave,
the A→B→A sequence failed on A with `single-state lowering cannot read a
foreign qualified quantity`. Four test permutations reproduced that failure;
the same four without waves already passed.

The selected view already computes `applicable_grid_names` from operator
signatures and the exact selected State. It now installs the Module wave law
only when that view has an applicable grid flux. B still receives its wave law,
while storage-only A has neither a flux nor eigenvalues. This does not change
the shared authoring Module, its hash, or the single-StateSpace validation of
an orphan wave declaration. A selected grid flux with a genuinely foreign
wave expression still fails closed during symbol rebinding.

The focused source test covers both State declaration orders, A→B→A and B→A→A
lowering orders, with and without waves, plus the single-state orphan refusal
and a selected A flux that must reject B's foreign wave law. It passes 10/10 after the correction; the initial four wave-bearing cases were
red. Native package rebuild, bind, MPI, and a full multi-state transport run
remain outside this source receipt. The authoring contract still has a single
shared Module wave law; independent per-StateSpace wave declarations would
require a separate contract change.
