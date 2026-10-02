# Independent Halo2 clock relay review

Reviewed source: fb530d5 +863fa6b6 +72384a6d6b24acf1399e1e2993462348f9646ec1.
Private review branch only; no production edits, Native runtime, ENV writes or JIT.
Conclusion: no additional blocker in periodic State accepted-halo topology relay.
This is a Source/host review, not an executed topology/rollback qualification.

Primary authority: actual public Program checkpoint C++ emitter exports the exact
qualified temporal primary clock. Loader module_metadata.hpp authenticates a
nonempty exported primary against the complete logical-clock registry, rejects
duplicates there, and does not infer primary from sorted position. Historical
artifacts without the optional symbol remain loadable; only the opted stronger
accepted-halo request requires an installed primary authority. Legacy execution
identity and synthetic topology clock path remain available when not opted.

accepted_halo_topology_points authenticates decoded accepted epoch, generation,
actual hierarchy level-clock count, primary logical tick, every accepted level's
macro tick, physical time and zero rational phase before constructing points.
Noninitial zero duration and nonfinite/negative duration are refused. Each
capacity transition uses its real declared ParentChildClockRelation and the LAST
partition interval, not a guessed spatial ratio or dt/level. It partitions a
normalized [0, macro_dt] window and scales exact rational phase spans, avoiding
catastrophic cancellation at large accepted physical times. Published execution
points retain the actual accepted time, primary namespace and macro tick.
Five-level independent header probe covers ratios2,5/2,3,7/3 with final spans
1/2,1/10,1/30,1/210; contiguous coverage and explicit remainder markers checked.
Adversaries: wrong parent/child level, ratio<1, missing remainder permission,
wrong parent level, zero/reversed phase range, empty physical interval, alpha
foreign macro tick/level/outside phase. The probe compiles and executes the REAL
public amr_clock.hpp with /usr/bin/clang++ C++20, no extracted implementation,
Kokkos, DSO, fake Native module or heavy build. Temporary output is isolated.
The primitive itself does not authenticate arbitrary differing end macro ticks;
the reviewed relay creates both stamps from the same already-authenticated tick.

Ordinary Refine/Rebalance publication: outer accepted transaction snapshot is
prepared collectively. Clock/admission failures vote before topology publication;
prepare_topology_field_order runs before mutation. Actual new hierarchy is then
prepared, candidates cover all blocks and levels, and full-field copies vote
before returning. Regrid-parent uses the same admission/preflight ordering,
refreshes Program hierarchy/history before candidate halo preparation and only
then publishes tagging checkpoint. Zero-duration initialization runs boundary
preflight before topology mutation, so effects requiring positive dt reject.

Rollback: execute_transaction catches failures and restores the accepted snapshot,
whose scope includes engine/carriers, provider storage/registries, Program context,
accepted bytes/revision/clocks/durations, histories, field potential/boundary
points and tagging. Restore uses the surviving carrier lane rather than stale
post-regrid graph, votes preparation and execution, publishes prepared restore,
recreates hierarchy and refreshes Program resources. This source ordering is
reviewed; failure injection and byte equality on a real backend remain ROOT's
Native obligation. No claim that Source probes execute these private C++ methods.

Boundary scope: the field setter point alias is not a fresh FieldBC producer.
This review does NOT qualify nonperiodic FieldBC local-dt evaluation. Hooke's
separate stronger producer/effect work and its independent review remain required.

Reproduce Source suite with env -u PYTHONPATH and private WT/python inserted:
`tests/review/test_sol61_halo_clock_relay_independent.py`
`tests/python/unit/amr/test_accepted_halo_preparation.py`
`tests/python/unit/codegen/test_program_graph_lowering.py`
`tests/python/unit/runtime/test_amr_checkpoint_contract.py`
The isolated independent pair passed5.03s; combined suite: 93 PASS14.09s, zero skips. Actual Source import
path asserted and `_pops` absent. Historical production files unchanged.
