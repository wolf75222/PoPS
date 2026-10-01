# Full AMR state carrier persistence — independent Source review

Review worktree based on MAIN 3b2b0509. Root production candidates copied into this private worktree as uncommitted review inputs; this commit owns only the independent tests and this note. M19 freeze de48143 is preserved in its separate worktree.

The principles 1.1–1.8 were already read from MAIN. This repair extends generic accepted-state persistence of actual AMR storage, without assigning physical roles to names or changing the physical equations. Python transports an opaque native image and never reconstructs ghosts. No performance equivalence, native codec validity or restored AMR scientific acceptance is established here.

Reviewed guards: exact bytes plus POPSCAR1 envelope, uint8 one-dimensional archive, absent key, absent native capture/validate/restore methods. The method/archive preflight votes before native capture or validation. A fallible native validation propagates to the outer restart preparation consensus before begin_restart_transaction; subsequent exchange/diagnostic preparation performs local validation, not additional collectives. The native validator must remain nonmutating and receive source geometry rather than require current target topology. Its C++ implementation is outside this Python receipt.

Lifecycle ordering is source-verified: history replay, accepted semantic/exchange restoration, cadence and final set_clock, materialized field warmstarts, full state carrier restoration, exact restored-contract validation, then optional regrid. Carrier publication remains within the enclosing rollback transaction. AMR payload 11 is refused by the exact release-version gate before prepare_v3; current schema/generated contracts specify 12.

Independent test result: **16 PASS, 0.24s**. Tests import actual source pops from this worktree/python with an explicit pops.__file__ assertion and assert absence of pops._pops and dimension-native modules. Method recorders check transport/frontier behavior only; their opaque image is deliberately synthetic and does not claim a native codec or MPI execution. Covered attacks include nonbytes envelope, obsolete magic, header-only image, archive dtype/rank, missing image/method and source-validator failure. Lifecycle and outer consensus ordering are source AST/text checks.

No Python production defect found in this bounded review. Actual Native dimension-2 save/restart carrier parity remains pending ROOT execution. No setup, build, JIT, environment mutation or native execution occurred.

Reviewed input SHA256 pins:

- `python/pops/runtime/_checkpoint_state_carriers.py`: `8cf9e516d413312cfe4cc235b91aeaf11e43d5cc308b1ffc46e0f8fcc81d8f71`
- `python/pops/runtime/_amr_checkpoint_v3.py`: `cfa80e271a6564feba14bac6f76fc6ee76f348e6497be549456d3b3fdfe24db9`
- `schemas/release_contract.v2.json`: `406e080808a1214bcddde3440c42fb54232e649a00a03a4cf836a59f0b1cda9c`
- `python/pops/_generated_release_contract.py`: `f258d08970c237396919261fb9c805004a7fffd1b2aa098d02e88738416bd87f`
- `include/pops/runtime/config/generated_release_contract.hpp`: `1448aa6d0ca1d84b01b84b425ac9d1cc34d823c9312c7f42690429a67f73287d`
