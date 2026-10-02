# M19 provenance correction independent review — SOURCE_ONLY

Exact reviewed gel: 6b8f158dc92eb402551e1c8927ff4bfdc3317bf1, parent a491ad65, author worktree PoPS-sol61-m19-provenance-capture. No Source blocker found.

Principles 1.1/1.4/1.6 → retain actual compiler authority and expose missing evidence → fixture `_capture_components`: absent CompiledModel `_generated_cpp` becomes explicit null; ModuleManifest uses its real `to_dict`. Each enumerated model/program requires a pinned binary and sidecar; model manifest is mandatory. Program CPP must be exact str before `dump_cpp`, preventing its regeneration fallback; IR and nonempty hash remain mandatory. FRAME.to_dict, quadrature.to_data and PlatformManifest.to_data match current interfaces. No physical equation, threshold, axis ordering or restart comparison changed.

Four files independently compared byte-for-byte against installed ENV9: codegen/_loader_model.py, codegen/loader.py, codegen/_loader_dump.py, model/_module_manifest.py; all identical. These checks read files and never load/execute a DSO.

Independent command in author worktree: `env -u PYTHONPATH PYTHONDONTWRITEBYTECODE=1 /Users/romaindespoulain/miniforge3/envs/pops-api040-ir17/bin/python -m pytest --noconftest -p no:cacheprovider -o pythonpath=python tests/review/test_sol61_m19_provenance_capture.py -q` → 9 PASS, 1.31s, zero skips. Genuine Python classes use explicitly synthetic metadata files; they provide no Native authenticity or scientific receipt.

Limitations: model CPP absence remains an explicit provenance gap; file pins do not prove a compiler-to-DSO graph. Fresh ROOT Native reception remains required. Historical failures a649/a491 remain unchanged. No BGK or Vlasov-Poisson qualification, Native/JIT/build or SDK mutation.
