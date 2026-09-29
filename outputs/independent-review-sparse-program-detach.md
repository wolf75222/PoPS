# Independent review of c38623b: sparse Program detachment

Reviewed MAIN `c38623b` source-only. No native build, JIT or installed acceptance was run. The installed SDK 027 product/H05 failures remain the pre-rebuild red evidence.

The no-drop/no-nonidentity-alias branch now copies each original `ProgramValue.id` and initializes the output `_next_id` from the source. `clone` applies that choice to nested values as well as flat values. A genuine dropped node or nonidentity alias retains the earlier compact-renumbering branch. The detached Program still has to pass the existing IR identity guard; neither serialization nor that guard was weakened. The second change skips **optional** User face inspection when `node_model is None`, while a present model still traverses `_model_impl` and authenticated User face handling.

I independently reran the new sparse-detach, existing IR-optimizer and compiled-detach suites: **28/28 passed in 3.69 s**. The separate authored User face and qualified routing suites passed **14/14 in 5.44 s**. Both invocations used `env -u PYTHONPATH`, `pops-api040/bin/python` and `-o pythonpath=python` from MAIN, totaling the same 42 affected tests as the root receipt.

An additional independent test, `test_sparse_nested_detach_independent.py`, reserves a nonexecuting SSA-id gap, then authors nested public `Program.branch` regions and an actual commit. Its no-op rebuild and `detach_compiled_program` retain serialized IR, hash, flat IDs, `_next_id` and reowned Program values. This is a synthetic gap combined with real nested regions; the root's public LocalResidual-product test supplies the separate real producer of a gap. The new test passed **1/1** when `pops` was explicitly preimported from MAIN `python/pops/__init__.py` before pytest. An earlier unqualified invocation imported the older isolated checkout and failed the old detach guard; it is not evidence against `c38623b`.

No source-level blocker was found in the four-file diff. The source checks do not establish native System/AMR publication, rollback, MPI or the rebuilt installed SDK 027 product/H05 replay.
