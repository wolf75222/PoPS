# Independent Source review: AMR object-byte gather v2

Reviewed ROOT d1e52b4e8530eaa45b35d17999531960b55c13e0 and corrective
023fc0c3cbffe4db4baf715a7b20772d56086f8f in the separate ROOT worktree.
This report preserves the preceding eight-step reader commit 0ad0dab.
No Native execution, rebuild, scientific approval, or MPI qualification is claimed.

## Decision and authority

Principles 1.1/1.3/1.5: the change is a common storage transport operation,
independent of physical equations, names, component order, or method.
Principle 1.6: the motivating SDK18 MPI experiment 732028 remains FAILED;
source review cannot promote the replacement binary. Principle 1.7: the same
payload cells and components are transported as bytes; no measured performance
claim is made here.

LevelLayout::validate_ requires exact coarse tiling and disjoint fine patches
inside the domain (level_layout.hpp 132–155). Distribution validates its owner
list against the layout/rank space, and MultiFab constructs only those local
owner patches. Thus partitioned valid cells have one contributor; replicated
storage contributes only rank zero. Fine holes and empty ranks contribute
explicit zero object bytes. Ghost cells are deliberately outside this getter.

The first patch had a material pre-existing allocation between the preparation
failure vote and exact agreement: the descriptor vector. Corrective 023fc0c
uses the existing initializer-list/span API. The consensus implementation votes
both its length-vector and payload-vector allocation failures before proceeding.
Byte count multiplication is checked inside voted preparation; MPI byte OR is
chunked below INT_MAX by the existing ExecutionLane helper.

Corrective 023fc0c also explicitly zeroes the OR identity and includes the native
object bytes of double 1.0 in the exact contract before payload transfer. This
rejects heterogeneous representation/endianness for the supported IEEE-double
scope. The operation preserves the prepared double object bytes, including
signed zero. Real-to-double conversion is still the existing API operation;
arbitrary native Real NaN payload preservation is not promised. Serial remains
an identity transport. No remaining material Source blocker was found.

## Required actual oracle after rebuild

Re-run the authentic failing M16 MPI2 experiment with its retained signed-zero
carrier versus public getter comparison. In addition, an infrastructure run
should exercise distributed multi-box coarse storage, canonical replicated
coarse storage, a partially covered fine level, and an empty owner rank. Compare
component-major public getter bytes with independently decoded native valid
carrier bytes; explicitly expect +0 in uncovered fine cells. Keep finite ±0,
positive/negative exactly representable values and component permutation. Do not
use a mocked reduction or claim full ghost transport from this valid-cell getter.

Existing test_mpi_system_io_gather.cpp exercises actual MPI marshaling in
Dimensions 1/2/3, including signed zero and partitioned/replicated storage, but
calls the Uniform helper; it does not independently receive this AMR change.
No new uncompiled setup test is submitted as evidence. ROOT owns the rebuilt
AMR public-API test and Native reception.

## Commands and status

Bounded git diff d1e52b4..023fc0c and targeted reads of amr_system.cpp,
level_layout.hpp, distribution.hpp, multifab.hpp, comm.hpp and
exact_field_marshaling.hpp: SOURCE REVIEW COMPLETE. ROOT worktree clean at
inspection. No SDK/ENV/Main/raw capture was modified. Full TU/MPI/runtime tests:
NOT RUN by this reviewer.
