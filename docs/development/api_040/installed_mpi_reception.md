# Installed MPI reception receipts, version 2

`run_installed_mpi_checks.py --dimension 1|2|3` selects an actual installed
native dimension before authentication and on every MPI worker. The default
remains dimension 2 for existing commands. An absent native variant is an
authentication failure; the runner never substitutes a higher-dimensional
extrusion. Dimensions, ranks, and OpenMP threads are separate settings.

Receipt schema 2 adds `dimension` to `result.json` and each rank identity and
passes the selected dimension explicitly in the recorded MPI command. The
runner still checks the installed package path, native hash, shipped source
manifest, per-rank test parity, and unchanged test sources. Passing requires
at least one test and zero failures, errors, or skips on every rank. Counts
are reported per rank and are not added together as independent experiments.

This extends the evidence format, not the PoPS semantic IR or native ABI.
Historical schema-1 receipts retain their original bytes and describe Dim2.
Selecting dimension 3 is a supported reception option, not evidence that a
Dim3 artifact has been built or executed on this machine.
