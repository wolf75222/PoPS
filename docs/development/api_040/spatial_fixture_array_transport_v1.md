# Spatial fixture array transport @1 — source/host receipt

Base fixture: `e7da8f75a6f142ef627b1e4dccd2ca073e27796b`. This correction changes only test capture transport. Program equations, Native collective producers/encoder, InitialConditionPlan admission, nonlocal/history contracts, archive keys, checkpoint anchors and receipt @1 remain unchanged.

## Authentic failures retained

The first e7 campaign failed all six cases at bind because of double initialization authority. e7da corrected that fixture error. The next authentic campaign, `installed-sdk9f57-spatial-interaction-serial-after-initial-authority-dim2`, successfully bound but still closed six FAIL / zero error / zero skip after 214 s. Native SDK9f57, Python378, Native source b1675566. Its capture attempted to send a NumPy array through `allgather_value`: `native allgather encoding failed: rank0 TypeError native collective value at $.pieces.active[0].values cannot encode ndarray`. It is preserved as RED, without scientific qualification.

The existing structured control-value codec also deliberately refuses `bytes`, so the auxiliary tuple's transport needed the same test-only conversion. Neither core guard is changed.

## Closed transport and convergence

`pops.spatial-interaction-fixture-array-wire@1` is an envelope containing explicitly tagged arrays, opaque bytes, exact Python scalars, tuples, lists and string-key dictionaries. Arrays carry precisely `{dtype, shape, Cbytes}`; Cbytes is lowercase hexadecimal for the array's C-order raw bytes. Dictionary entries retain insertion order explicitly, so the post-decode piece manifest and NPZ arrays are byte exact to the pre-transport capture. No object, pickle, structured/string dtype or arbitrary protocol conversion is admitted.

The decoder checks the closed envelope/tag/key sets, canonical NumPy dtype spelling and explicit endian, exact integer nonnegative shape, at most 32 tensor axes, `sys.maxsize` byte/stride arithmetic (including empty axes), and exact payload length/hex alphabet before `frombuffer` or array copying. Supported plain bool/integer words are 1/2/4/8 bytes; floating words are 4/8 bytes. It copies reconstructed arrays into independent C-contiguous buffers without numerical conversion. Nonfinite raw values, NaN payloads and signed zeros remain evidence; the wire does not approve them as admissible physics. Host metadata limits here are fixture decoder bounds, not a new production DOF cap.

Local encoding runs inside a separate `collective_call` vote **before** payload allgather. Only after successful all-rank encoding does the unchanged `allgather_value` transmit the closed wire. Decoding has its own subsequent convergence boundary. A simulated peer encoder failure proves no payload entry follows a failed vote. This cannot prove absence of a hang inside a real MPI collective; Root receives the fresh six-case Serial/MPI campaign.

This wire creates Python raw bytes, hexadecimal strings, the existing control-plane encoded payload, gather buffers and decoded arrays. It makes no RSS/workspace claim and is not charged to the scientific DirectSpatialInteraction workspace. No C++/SDK API is changed.

## Verification

```sh
rtk proxy env -u PYTHONPATH -u POPS_NATIVE_DIM PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=python /Users/romaindespoulain/miniforge3/envs/pops-api040/bin/python -B -m pytest -q tests/review/test_sol61_spatial_fixture_array_wire.py tests/review/test_sol61_spatial_interaction_native_fixture.py tests/python/unit/runtime/test_collective_fixture_checks.py tests/python/unit/runtime/test_native_collectives.py
```

The final coherent run received **89 SOURCE/HOST PASS in 9.08 s**, including two explicit Float32/Float64 signaling/quiet NaN payload-bit probes. Ruff and `git diff --check` pass. The tests execute the real Python `pops._native_collectives.encode_value/decode_value` on four piece schemas and two explicitly synthetic rank payloads, including histories, masks, owners, empty pieces and auxiliary bytes. They discriminate the old ndarray/bytes refusals, malformed lengths/shape/endian/types before allocation, duplicate keys and unknown contracts/tags. They verify reconstructed archive NPZ arrays and JSON manifests byte exactly, plus actual public Case/validate/resolve/emit admission from e7da. The host collective transport seam is named explicitly and is not an MPI communicator or a native positive.

No PoPS Native execution, SDK build, JIT, MPI run, donor mutation or ROOT owner approval was performed here. All positive transport fixtures are synthetic source/host tests. Existing Native RED archives remain immutable; fresh Native reception remains pending Root. Frontier's independent scientific reader must continue requiring actual new archive files and external owner seals.
