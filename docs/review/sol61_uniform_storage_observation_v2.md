# Uniform observation profile @2

New standalone Native node `tests/python/integration/runtime/test_uniform_accepted_storage_observation_v2.py::test_installed_uniform_accepted_storage_observation_v2`; @1 remains unchanged.

Before bind, public compile policy requires retained actual source for every model. Exact verified Uniform layout Program C++/IR and all model C++/DSO provenance are persisted without regeneration. Every rank captures and immediately writes both successive rank-local and complete POPSCAR1 archives, valid NPY, owner clock and SHA-pinned phase receipts before requesting the next observation. Even empty ranks retain their actual shard. Every Native and I/O phase is collectively entered.

This tests readonly initial observation equality and actual valid/full-grown correspondence. It performs no step, restart or rollback and issues no SCI approval. Source persistence adversaries use explicit synthetic byte strings; they do not qualify Native or decode POPSCAR1. The production C++ codec remains the full validator. Native Serial/MPI execution belongs to ROOT.
