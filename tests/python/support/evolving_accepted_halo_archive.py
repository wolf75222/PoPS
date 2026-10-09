"""Negative checkpoint copy: retain current ZIP format and authenticated original lifecycle."""
import numpy as np
from pops.runtime._checkpoint_manifest import (MANIFEST_KEY,IDENTITY_KEY,
    inspect_checkpoint_payload_integrity,_identity_from_json,_seal_checkpoint_payload_with_identities)

def truncated_carrier_archive(source,destination):
    with np.load(source,allow_pickle=False) as image:
        manifest,_=inspect_checkpoint_payload_integrity(image,runtime_kind="amr")
        payload={k:image[k].copy() for k in image.files if k not in (MANIFEST_KEY,IDENTITY_KEY)}
    carrier=payload["state_carriers_checkpoint"]
    assert carrier.dtype==np.uint8 and carrier.ndim==1 and carrier.size>8
    payload["state_carriers_checkpoint"]=carrier[:-1].copy()
    # Preserve the accepted checkpoint's authentic run, not the later receiver's run.
    _seal_checkpoint_payload_with_identities(payload,runtime_kind="amr",
        semantic=_identity_from_json(manifest["semantic_identity"]),
        artifact=_identity_from_json(manifest["artifact_identity"]),
        bind=_identity_from_json(manifest["bind_identity"]),
        run=_identity_from_json(manifest["run_identity"]),origin=manifest.get("origin"))
    # Same DEFLATED member contract as the genuine repository checkpoint writers.
    with open(destination,"xb") as stream:np.savez_compressed(stream,**payload)
    with np.load(destination,allow_pickle=False) as image:
        revised,_=inspect_checkpoint_payload_integrity(image,runtime_kind="amr")
    for key in ("semantic_identity","artifact_identity","bind_identity","run_identity","clock"):
        assert revised[key]==manifest[key]
    return payload
