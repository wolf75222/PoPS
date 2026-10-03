# ExternalFieldSolver@4: installed dispatch fault witnesses

Two parameterized Native nodes use the real public validate/resolve/compile/bind/run pipeline and FieldSolver@2/FieldTopology@2 component ABI tables, through the current native-memory adapter contract @4. The test DSO exposes typed arm/counter functions. After bind its actual installed path is obtained from public inspect(), checked byte-for-byte against the artifact-owned binary, and opened at that same path. No production fault hook is added.

`callback_throw` makes the selected rank write actual candidate Field values and then throw a C++ runtime_error. Every rank must fail the solve; the Native adapter joins its callback completion before its vote. Counters must show one dispatch on each rank and writes where patches exist. `table_preparation` truncates only the selected rank's component table header after bind, so the actual ComponentLibrary::table preparation lookup must fail before any callback executes anywhere (counter zero). Disarming restores only the test table authority; it never restores numerical storage.

Both tests retain ALL model and Program actual C25 source/binary/manifest evidence before bind, the test component binary/metadata, local and complete public POPSCAR1 storage, immediate State and Field NPYs, clocks and consumer cursors. The before/after full-grown storage, valid State bits, Field bits, clock and cursor values must be exactly equal. Retry must accept one step, add exactly one callback per rank, retain the passive State and publish the actual constant Field value 7. This is a transaction/dispatch engineering witness; it does not qualify a Poisson residual or a general physical model.

This preparation has not executed Native. Source/Host checks compile the actual test DSO, verify typed invalid arming and header truncation/restoration, retain identity bytes losslessly, validate/resolve the real composition, and check adapter ordering. A Host control test is not Native MPI qualification. CPU wrapper/allocation failure remains missing: the provider cannot inject DefaultExecutionSpace construction. The table fault covers only actual pre-callback table preparation, and is not relabeled as allocation/wrapping.

Installed execution (ROOT-owned SDK29):

```
env -u PYTHONPATH PYTHONDONTWRITEBYTECODE=1 "$SDK29/bin/python" -m pytest tests/python/integration/native_loader/test_external_field_dispatch_faults_runtime.py -q --junitxml=dispatch-world1.xml
mpiexec -n 2 env -u PYTHONPATH PYTHONDONTWRITEBYTECODE=1 "$SDK29/bin/python" -m pytest tests/python/integration/native_loader/test_external_field_dispatch_faults_runtime.py -q --junitxml=dispatch-rank.xml
```

Use the repository installed runner/outer identity guard to authenticate the exact SDK, Source, dimension, Native DSO, MPI world and per-rank XML; give each MPI rank a separate XML through that runner. The bare commands specify the nodes, not a substitute identity/launch authority. No result is claimed before those runs.

The brief AMR bind(resources=...) review of 72eb41ae confirms the public pops.bind signature accepts resources, and the execution context is the existing artifact-derived typed object under that resource key. The test-only change removes an unsupported top-level execution_context keyword; it does not modify the AMR/Core contract or qualify its future run.

The Source snapshot prerequisite is isolated in `e6a3a946d727671e55debec51cba975be201793c` plus `a548763973524949a5ba38cc7215cdf4eea19355`; its earlier registry-cycle and callable-first-projection failures remain preserved. The coherent affected Source/Host cohort passes 52 tests, and exactly the two installed nodes collect successfully. Independent snapshot review remains a prerequisite for integration. These checks do not receive either runtime fault.
