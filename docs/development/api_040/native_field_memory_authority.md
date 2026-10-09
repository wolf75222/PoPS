# Native Field memory authority, revision 1

`runtime-field-memory-authority@1` defines the existing runtime report
`field_memory_space` from `Kokkos::DefaultExecutionSpace::memory_space`, the
default allocation space of the real `Fab<Dim>` and `MultiFab<Dim>` storage.
It does not derive Field residence from the separately reported SharedSpace
arena. The report vocabulary and C ABI structures are unchanged; the changed
public header must be rebuilt and authenticated by its SDK/header signature.

The classification is `host` for HostSpace, `managed` for another memory space
accessible to DefaultHostExecutionSpace, and `device` otherwise. Here
`managed` expresses host accessibility; it does not certify migration or a
particular unified-memory allocator. `allocator_mode` and
`native_shared_space_identity()` still describe the distinct process arena.

The integration test compares the report to the actual Fab alias, allocates a
two-component Fab, writes its storage with the default execution space, fences,
and checks the values through Fab's explicit host mirror. Run that same test in
each configured backend. A CPU/OpenMP pass does not qualify CUDA, HIP, SYCL or
OpenMPTarget; a device build must authenticate its real execution/memory spaces
and compiler flags as well as execute the allocation/kernel/copy sequence.

Earlier receipts that report SharedSpace as Field storage remain historical
evidence for their exact source/native artifacts. They do not establish the
memory authority of a rebuilt artifact with this correction.
