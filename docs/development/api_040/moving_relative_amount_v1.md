# Canonical relative face amount and moving checkpoint revision 4

The genuine installed SDK20d reception at source `470dfb08` passed six ALE
tests and refused two vector-with-source cases. The diagnostic-only rebuild
at `5f2076e3ff9ee8c1558022a3b0295bb39737d6f9`, SDK
`3f1286c0514edba336044d55c5e00b7965917455740adc31d0e21c621e30757d`,
reproduced both refusals. In the task workspace,
`outputs/installed-public-ale-exact-receipt-diagnostic-dim1-sdk3f12-20260930`
authenticates the true installed Python/native package and 1111 production
files. Its identity SHA is
`8a6ae0c675b7f8cfb6e1a46c8d2d83f81fb07d48e3a6106dd6255f979729795e`;
log SHA is `664db77d1564d387b718dac928946294f9d350f95796930bc93476db7fcd88e5`.
Native Dim1 SHA is `1cb02ada9cc8f0be2ed9dbc511becf77af37a243224f3d10cb88f0ae41995733`.

The two actual discrepancies match independent rational arithmetic witnesses:
`abc` component 1, cell 9/right, stores `-0x1.7ee75a8e3ebaap-11` while
validation reconstructs `-0x1.7ee75a8e3ebabp-11`; `cab` component 2,
cell 1/right, stores `0x1.fa10cfbb6fc96p-10` while validation reconstructs
`0x1.fa10cfbb6fc95p-10`. Separate multiply/subtract rounds twice; compiler
contraction rounds once. The independent review uses exact `Fraction` inputs,
not PoPS's arithmetic or a native numerical result invented by a source test.

The new realization `pops.moving.relative-face-amount@1` computes
`fma(-density, swept_volume, integrated_physical_flux)` on the host explicitly.
The native producer and receipt verifier use this same named realization.
It changes no physical flux, source, Reynolds/GCL equation, temporal method,
Kokkos storage, topology, acceptance owner, or residual tolerance. Ledger
quantities still require exact equality; exterior/support/point guards remain.
The independent offline checker derives the same correctly rounded quantity
from exact rational representations of the saved binary64 inputs.

`POPSEX04` adds one unsigned 64-bit relative-amount realization tag immediately
after each present receipt's geometry tolerance: `1` is the explicit fused
realization; `0` names the retained historical compiler expression. Unknown
tags refuse. The new producer always issues tag 1. Per-receipt tags allow a
composite retained image to contain an older receipt without reinterpreting its
quantity. Cross-rank checkpoint metadata includes both wire and realization
versions, before restore publication.

`POPSEX03` remains readable using its original exact expression; no alternate
rounding or ULP tolerance is accepted for it. Decoded purely historical
geometry re-exports revision 3, including an initial image with no receipt.
A subsequent newly accepted interval issues revision 4. Older artifact/SDK
compatibility checks still apply: availability of a low-level historical codec
does not authorize restarting an incompatible compiled artifact. The inner
`POPSEX02` ledger, native Module ABI5 and SystemPackage ABI7 remain unchanged.
The shipped header signature changes and native binaries must be rebuilt.

The repair's serial/MPI runtime qualification and revision-4 codec injections
are pending until a coherent rebuilt package is received. The red receipts,
historical archives and their source identities are preserved separately.
