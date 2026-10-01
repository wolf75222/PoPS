# Integrated IR16/IR17 source reception — 2026-10-01

The explicit global-field history storage port and the direct spatial interaction
port are integrated in the real PoPS tree. Their extensions are conditional:
`pops.program.global-field-history-storage@1` selects IR16 and
`pops.spatial-interaction@1` selects IR17. Programs that use neither extension
retain their previous IR/C++ identities. The model equations, field spaces,
physical kernel, cell-volume measure, midpoint quadrature, realization workspace
and publication remain distinct declarations.

ROOT replayed the combined source contracts on native-reception tree
`a1d4a27d353a7d5edbe150a747ea4e25b45a7709`, whose tracked tree is identical to
integration tree `b2c365cde11720a619585e64e492b7084f61488a`. All 91 tests passed,
with zero failures, errors or skips in 112.993 seconds. The interpreter came
from the preserved `pops-api040` environment, but the test bootstrap explicitly
imported the source package in that checkout. This is source and actual-header
host reception; it is not an installed native DSO or scientific reception.
The exact log and JUnit digests are recorded in the companion JSON.

The nonlocal independent review receives 24 additional source/host tests against
author gel `13a3cabe5c1d6830e3d6a9fe3357c945d243c0eb`. Four fresh legacy profiles
retain all eight IR/C++ hashes. The global history independent review receives
44 source/host tests and exact old-profile parity at a controlled shared callsite.
These counts overlap the combined ROOT selection and must not be added as
independent scientific witnesses.

Actual C++ header checks exposed a Float32 compilation regression in the first
nonlocal implementation. The repaired transport uses native uint32/uint64 IEEE
words with 16-bit integer limbs, including signed zero, without floating SUM
normalization. Actual header/AMR-engine host checks run 61 assertions for each
Real type; independent extracted consumer checks run 27 for each. They do not
establish a Float32 native DSO, GPU or distributed execution.

The generic direct realization uses linear snapshot storage and quadratic work,
includes physical self-pairs, supports component subsets/permutations and signed
nonsymmetric kernels, and uses active finest-cell volume times actual EB volume
fraction. Its declared workspace is not an RSS/allocator-overhead bound. No
model-specific kernel or degree-of-freedom cap is introduced. Multi-level
candidate snapshots still require an explicit simultaneous source barrier.

The first AMR Stage archive capture labelled raw history slots backwards. The
fixture repair receives actual POPSHID1 sample bytes and explicitly labels the
published depth-two ring as latest slot 1 and previous slot 0. The equations and
seven original solver controls remain unchanged. Actual AMR Stage execution is
pending the next SDK build. A separate actual-header probe also demonstrates
that the legacy all-ring history matcher refuses a valid retained slot while a
new front-slot store is pending. The nonlocal consumer needs a dedicated
selected-sample closure; the legacy guard must remain intact. The next SDK
freeze waits for that closure and its independent review.

ROOT preserved SDK2e4 and its previously sealed scientific archives. The cloned
`pops-api040-ir17` environment is reserved for real repository-script compilation;
copied native binaries are not new qualification. The first source replay in the
integration checkout was terminated during a filesystem/import stall before any
test result. The equal-tree replay in the fast native checkout is the closed
91-test reception. No GitHub CI or complete M26 PDE qualification is claimed.
