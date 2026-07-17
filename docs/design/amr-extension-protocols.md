# AMR extension protocols

## Native materialization action

An AMR transfer or materialization package is extended by composition, not by inheriting a PoPS
implementation class.  Its resolved action implements exactly this structural method:

```python
def native_amr_materialization(*, key):
    return NativeAMRMaterializationDescriptor(...)
```

The supported provider SDK is exported from `pops.amr`:

- `NativeAMRMaterializationDescriptor` is the versioned, immutable native IR;
- `NativeAMRActionKind` and `NativeAMRMaterializationKind` select closed native effects;
- `NativeAMRMaterializationCapabilities` carries explicit capability identities;
- `TransferCapabilities` carries the typed order, halo, dimension, conservation, temporal and
  refinement-ratio envelope for a physical transfer;
- `CanonicalOptions` freezes provider identity and route options into deterministic data.

Validation and runtime preparation inspect only the returned descriptor.  They never dispatch on
the action or provider class and do not require a PoPS base class.  Built-in actions implement the
same method as external packages.

The descriptor authenticates all of the following:

- `schema_version == 1`;
- the action and materialization kinds;
- the exact transfer-key identity and operation;
- an owner/package-qualified provider identity;
- one non-empty `native_route`, repeated in immutable route options;
- sorted, unique capability IDs for both the materialization family and operation;
- physical transfer capabilities when, and only when, the materialization is physical.

PoPS invokes the extension protocol twice while preparing the closed resolved IR.  Different
canonical results are rejected as non-deterministic.  Returning a mapping, a descriptor for another
key, an unsupported schema version, non-canonical data, missing capabilities, or contradictory
materialization/action kinds fails before artifact creation or native mutation.

Exact type checks remain valid only after this boundary: `ResolvedTransfer`,
`NativeAMRMaterializationDescriptor` and its nested capability values are closed internal IR
containers.  Open extension objects themselves must never appear in a central `type(...)` or
`isinstance(...)` dispatch.

## Minimal external physical action

```python
from dataclasses import dataclass

from pops.amr import (
    CanonicalOptions,
    NativeAMRActionKind,
    NativeAMRMaterializationCapabilities,
    NativeAMRMaterializationDescriptor,
    NativeAMRMaterializationKind,
    TransferCapabilities,
)


@dataclass(frozen=True, slots=True)
class ExternalConservativeTransfer:
    provider_id: str

    def native_amr_materialization(self, *, key):
        kind = NativeAMRMaterializationKind.PHYSICAL
        transfer = TransferCapabilities(
            order=2,
            ghost_depth=(1,),
            dimensions=(2,),
            conservative=True,
            refinement_ratios=(2,),
        )
        options = CanonicalOptions({"native_route": "conservative_linear"})
        return NativeAMRMaterializationDescriptor(
            schema_version=1,
            action=NativeAMRActionKind.APPLY_TRANSFER_PROVIDER,
            materialization=kind,
            operation=key.operation,
            transfer_key_identity=key.identity,
            provider_qualified_id=self.provider_id,
            provider_identity=CanonicalOptions({
                "qualified_id": self.provider_id,
                "authority": "external.package",
                "options": {},
            }),
            options=options,
            native_route="conservative_linear",
            capabilities=NativeAMRMaterializationCapabilities.for_materialization(
                kind,
                key.operation,
                transfer=transfer,
            ),
        )
```

This class has no PoPS superclass.  Acceptance is determined entirely by the immutable descriptor
and the currently declared native capability envelope.
