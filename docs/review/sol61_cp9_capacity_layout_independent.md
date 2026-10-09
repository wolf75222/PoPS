# Independent CP9 capacity Layout review

Author 492743f9aceffc340339054f3209cea532f8c6b7; exact baseline 99c805690cbca380abe5d08b157aafaa1959cb3e. Verdict Source/host: accepted. The production delta only replaces unsupported range iteration by the actual BoxArray size/index API. No public Layout/header change or numerical contract/version change. Native build732222 FAILED remains historical; this review is not a successful replacement build.

Independent test executes the entire actual System checkpoint_state_carriers_capacity method against the real mesh::BoxArray and codec headers, with explicit host Field/System/lane stand-ins. It reproduces baseline compilation refusal in Dim1/2/3 (3 RED,1.98s), then unchanged tests pass after the author patch. Coherent author+independent cohort gives 6 PASS5.31s. No Kokkos, device, real MPI or installed Native backend was executed.

Distinct State blocks carry different component counts, global box cardinalities and anisotropic ghosts. Independent capacity arithmetic matches the full method; reversing block names preserves the bound. The actual encoder's maximal one-cell fragmentation reaches that bound exactly. A rank with zero local materialization still has the global layout metadata; a three-rank archive with no rank2 rows fits the same capacity. Missing all State blocks refuses. Global max(valid-cell count) across block layouts remains a conservative per-block bound, not a local-rank count. Existing checked arithmetic and voted error handling are unchanged.

The original restore host adapter returned std::vector from layout(), explaining why that separate probe did not catch this interface mismatch; it did not qualify the full System translation unit. New independent probes use the actual noniterable BoxArray. ROOT must rebuild the genuine GCC15.3/Kokkos/MPICH translation unit and SDK before any Native reception.

Command: env -u PYTHONPATH PYTHONDONTWRITEBYTECODE=1 /Users/romaindespoulain/miniforge3/envs/pops-api040-ir17/bin/python -m pytest --noconftest -p no:cacheprovider tests/review/test_sol61_cp9_capacity_independent.py tests/review/test_sol61_uniform_capacity_actual_layout.py -q --tb=short --junitxml=/tmp/sol61-cp9-layout-independent-green.xml

XML RED /tmp/sol61-cp9-layout-independent-red.xml SHA292070f68d767f3ac032335a25fbdf0dc9d95eea2eea4635b8a14ee18025f088; GREEN /tmp/sol61-cp9-layout-independent-green.xml SHAad9a7c7d5ee5df58301a18cfb57c6a5ff99cecff81f47f678231ef47d149b921. Compiler selected /usr/bin/clang++, C++20, Wall/Wextra/Werror. This is host evidence, not the failed job's GCC toolchain.
