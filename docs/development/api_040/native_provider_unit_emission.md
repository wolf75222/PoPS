# Native provider unit emission

SDK27R2 job732785 genuinely failed C++ compilation before bind: provider contracts emitted a structured PhysicalDimension as an unquoted C++ dictionary. This is an emitter defect, not a numerical or native runtime result.

The shared optional-unit emitter serializes PhysicalDimension to its canonical compact JSON string and then quotes that string as a C++ literal. None remains std::nullopt; historical named unit strings retain their existing literal bytes. Canonical structured metadata is validated; no str(dict) conversion or unit removal occurs. Model provider routes and Program consumer plans use the same helper. No contract or ABI version changes: the intended optional native string representation is restored.

Source validation compiles the complete emitted loaders for all three genuine resolved thermal Models with current repository headers and read-only SDK14 Kokkos/MPI dependencies. This is host syntax qualification only, without linking/loading a PoPS native package or running a simulation. ROOT owns SDK28 rebuild and actual reception; the SDK27R2 failure is preserved.

The corrective UTF-8 literal encoder preserves non-BMP and control bytes with fixed three-digit octal escapes. Embedded NUL uses the length-bearing std::string constructor, so its suffix remains present. Ordinary ASCII named-unit literals and canonical structured-unit JSON bytes remain unchanged. Tiny host executables compare the actual optional-string bytes, including Unicode and NUL; they do not execute a PoPS runtime. The previous non-BMP compiler refusal is retained as review evidence.
