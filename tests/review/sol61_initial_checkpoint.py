# ruff: noqa: E402 -- select and authenticate the source package before importing it
"""Independent lifecycle/envelope protocol probes; no native capture or PDE executes."""

import argparse
from copy import deepcopy
from dataclasses import replace
import hashlib
from io import BytesIO
import json
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import patch

parser = argparse.ArgumentParser()
parser.add_argument("--source", type=Path, required=True)
parser.add_argument("--receipt", type=Path, required=True)
parser.add_argument("--legacy-only", action="store_true")
args = parser.parse_args()
source = args.source.resolve()
manifest = json.loads((source / "source_manifest.json").read_text())
for name, digest in manifest["files"].items():
    assert hashlib.sha256((source / name).read_bytes()).hexdigest() == digest, name
sys.path.insert(0, str(source / "python"))
import numpy as np
import pops

sys.modules["pops.runtime._engine_descriptors"] = SimpleNamespace(
    abi_key=lambda: "independent-protocol-abi"
)  # ABI equality seam only, no native module.
from pops.identity import make_identity
from pops.output._checkpoint_contract import MANIFEST_KEY, IDENTITY_KEY
from pops.runtime import _checkpoint_manifest as protocol
from pops.runtime._lifecycle import _LifecycleMixin
from pops.runtime._run_manifest import begin_run
from pops.runtime._temporal_restart import TemporalRestartState
from pops.time import FixedDt

assert Path(pops.__file__).resolve() == source / "python/pops/__init__.py"
identities = tuple(
    make_identity(domain, {"independent": "initial-checkpoint"})
    for domain in ("semantic", "artifact", "bind")
)
legacy_run = make_identity("run", {"independent": "legacy-envelope-fixture"})


class Owner(_LifecycleMixin):
    def __init__(self):
        self._bound_snapshot = SimpleNamespace(
            semantic_identity=identities[0],
            artifact_identity=identities[1],
            bind_identity=identities[2],
        )
        self._temporal_restart_state = TemporalRestartState()
        program = pops.Program("independent_initial_checkpoint")
        strategy = FixedDt(0.125)
        program.step_strategy(strategy)
        self._temporal_restart_state.configure_program(
            program.temporal_manifest(), time=0.0, macro_step=0, strategy=strategy
        )
        self.t, self.step = 0.0, 0

    def time(self):
        return self.t

    def macro_step(self):
        return self.step


def payload(owner):
    return {
        "t": np.asarray(owner.time()),
        "macro_step": np.asarray(owner.macro_step(), dtype=np.int64),
        "temporal_restart_state": owner._temporal_restart_state.checkpoint_json(
            time=owner.time(), macro_step=owner.macro_step()
        ),
        "state": np.asarray([0.7, -0.2], dtype=np.float64),
        "abi_key": "independent-protocol-abi",
    }


legacy = {
    "t": np.asarray(0.25),
    "macro_step": np.asarray(2, dtype=np.int64),
    "state": np.asarray([0.7, -0.2], dtype=np.float64),
}
legacy_identity = protocol._seal_checkpoint_payload_with_identities(
    legacy,
    runtime_kind="uniform",
    semantic=identities[0],
    artifact=identities[1],
    bind=identities[2],
    run=legacy_run,
)
legacy_manifest, _ = protocol.inspect_checkpoint_payload_integrity(legacy, runtime_kind="uniform")
assert legacy_manifest["schema_version"] == 1 and "origin" not in legacy_manifest
assert protocol.checkpoint_run_identity(legacy) == legacy_run
observations = {
    "legacy_manifest_sha256": hashlib.sha256(legacy[MANIFEST_KEY].encode()).hexdigest(),
    "legacy_restart_token": legacy_identity.token,
}


def must_refuse(action, label):
    try:
        action()
    except (ValueError, TypeError, RuntimeError, KeyError):
        return
    raise AssertionError("inadmissible initial envelope accepted: " + label)


if not args.legacy_only:
    from pops.output._checkpoint_collective import _require_manifest_restart_identity

    owner = Owner()
    before = deepcopy(owner._temporal_restart_state.__dict__)
    evidence = protocol.checkpoint_lifecycle_evidence(owner)
    assert evidence == {
        "run_identity": None,
        "origin": {"schema_version": 1, "kind": "bound_initial"},
    }
    for kind in ("uniform", "amr"):
        image = payload(owner)
        restart = protocol.seal_checkpoint_payload(owner, image, runtime_kind=kind)
        data, authenticated = protocol.inspect_checkpoint_payload_integrity(
            image, runtime_kind=kind
        )
        assert authenticated == restart and data["schema_version"] == 2
        assert data["run_identity"] is None and data["origin"] == evidence["origin"]
        assert protocol.checkpoint_run_identity(image) is None
        assert protocol.authenticate_checkpoint_payload(owner, image, runtime_kind=kind) == restart
        _require_manifest_restart_identity(data, restart.token)
        assert owner.last_run_identity is None and owner.last_run_manifest is None
        assert owner._temporal_restart_state.__dict__ == before
        for clock_key, clock_value in (
            ("time", (-0.0).hex()),
            ("time", "0x0p+0"),
            ("macro_step", "0"),
            ("macro_step", 0.0),
            ("macro_step", False),
        ):
            forged = deepcopy(image)
            row = json.loads(forged[MANIFEST_KEY])
            row["clock"][clock_key] = clock_value
            base = {key: value for key, value in row.items() if key != "restart_identity"}
            try:
                resigned = make_identity("restart", base)
            except TypeError:
                assert clock_key == "macro_step" and type(clock_value) is float
                continue  # Binary floats are inadmissible in canonical identity metadata.
            row["restart_identity"] = protocol._identity_json(resigned)
            forged[IDENTITY_KEY] = resigned.token
            forged[MANIFEST_KEY] = json.dumps(row, sort_keys=True, separators=(",", ":"))
            must_refuse(
                lambda forged=forged, kind=kind: protocol.inspect_checkpoint_payload_integrity(
                    forged, runtime_kind=kind
                ),
                kind + ":noncanonical-clock:" + repr(clock_value),
            )
        for mutation in ("origin", "run", "schema", "time", "controller", "stats", "array"):
            forged = deepcopy(image)
            row = json.loads(forged[MANIFEST_KEY])
            if mutation == "origin":
                row["origin"]["kind"] = "run"
            elif mutation == "run":
                row["run_identity"] = protocol._identity_json(legacy_run)
            elif mutation == "schema":
                row["schema_version"] = 1
            elif mutation == "time":
                forged["t"] = np.asarray(0.25)
            elif mutation == "array":
                forged["state"][0] += 1
            else:
                temporal = json.loads(forged["temporal_restart_state"])
                if mutation == "controller":
                    temporal["controller_state"]["last_accepted_dt"] = (0.125).hex()
                else:
                    temporal["transaction_stats"]["rejected"] = 1
                forged["temporal_restart_state"] = json.dumps(temporal)
            forged[MANIFEST_KEY] = json.dumps(row, sort_keys=True, separators=(",", ":"))
            must_refuse(
                lambda forged=forged, kind=kind: protocol.inspect_checkpoint_payload_integrity(
                    forged, runtime_kind=kind
                ),
                kind + ":" + mutation,
            )
            if mutation != "array":
                # Recompute the content identity: test semantic refusal beyond checksum mismatch.
                if mutation == "time":
                    row["clock"]["time"] = (0.25).hex()
                row["arrays"] = {
                    name: protocol._array_evidence(forged[name]) for name in row["arrays"]
                }
                base = {key: value for key, value in row.items() if key != "restart_identity"}
                resigned = make_identity("restart", base)
                row["restart_identity"] = protocol._identity_json(resigned)
                forged[IDENTITY_KEY] = resigned.token
                forged[MANIFEST_KEY] = json.dumps(row, sort_keys=True, separators=(",", ":"))
                must_refuse(
                    lambda forged=forged, kind=kind: protocol.inspect_checkpoint_payload_integrity(
                        forged, runtime_kind=kind
                    ),
                    kind + ":resealed:" + mutation,
                )
    for mutation in (
        "noninitial_native_clock",
        "run_manifest_without_identity",
        "unbound",
        "attempted",
    ):
        invalid = Owner()
        if mutation == "noninitial_native_clock":
            invalid.t = 0.25
        elif mutation == "run_manifest_without_identity":
            invalid._last_run_manifest = object()
        elif mutation == "unbound":
            invalid._bound_snapshot = None
        else:
            invalid._temporal_restart_state.transaction_stats["failed"] = 1
        must_refuse(
            lambda invalid=invalid: protocol.checkpoint_lifecycle_evidence(invalid), mutation
        )
    owner._last_run_identity = legacy_run
    owner._restart_lineage_identity = legacy_run
    owner._last_run_manifest = object()
    owner._restore_checkpoint_run_identity(None)
    assert owner.last_run_identity is None and owner.last_run_manifest is None
    assert owner._restart_lineage_identity is None
    run = begin_run(owner, t_end=1.0, step_transaction={}, max_steps=16, output_dir=None)
    assert run.continuation_identity is None and owner.last_run_identity == run.run_identity
    assert owner.last_run_manifest is run
    image = payload(owner)
    protocol.seal_checkpoint_payload(owner, image, runtime_kind="uniform")
    assert json.loads(image[MANIFEST_KEY])["schema_version"] == 1
    observations["initial"] = (
        "Uniform/AMR envelope v2, no synthetic run, immutable temporal authority"
    )
    observations["negative"] = (
        "14 tamper + 12 content-resealed semantic tamper + 10 noncanonical clock + 4 invalid owners refused"
    )
    observations["run_request"] = (
        "initial restore clears prior lineage; begin_run creates real request identity; next envelope v1"
    )
    from pops.output import _checkpoint_collective as collective
    from pops.output._checkpoint_contract import CheckpointResourceBudget
    from pops.runtime._multi_layout_executor import _CompositeTemporalRestartState

    def archive(image):
        stream = BytesIO()
        np.savez_compressed(stream, **image)
        return stream.getvalue()

    for child_kind in ("uniform", "amr"):
        composite_owner = Owner()
        leaves = {name: Owner()._temporal_restart_state for name in ("a", "b")}
        composite_owner._temporal_restart_state = _CompositeTemporalRestartState(leaves)
        children = []
        for _ in leaves:
            child_owner = Owner()
            child = payload(child_owner)
            protocol.seal_checkpoint_payload(child_owner, child, runtime_kind=child_kind)
            children.append(np.frombuffer(archive(child), dtype=np.uint8))
        composite = {
            "t": np.asarray(0.0),
            "macro_step": np.asarray(0, dtype=np.int64),
            "layout_ids": np.asarray(tuple(leaves)),
            "abi_key": "independent-protocol-abi",
        }
        composite.update(
            {"layout_checkpoint_%d" % index: child for index, child in enumerate(children)}
        )
        kind = "multi_layout_" + child_kind
        protocol.seal_checkpoint_payload(composite_owner, composite, runtime_kind=kind)
        data, restart = protocol.inspect_checkpoint_payload_integrity(composite, runtime_kind=kind)
        assert data["schema_version"] == 2 and data["run_identity"] is None
        _require_manifest_restart_identity(data, restart.token)
        blob = archive(composite)
        budget = CheckpointResourceBudget(
            kind,
            max_members=32,
            max_manifest_characters=65536,
            max_array_bytes=1 << 20,
            max_uncompressed_bytes=2 << 20,
            max_archive_bytes=2 << 20,
            authority="independent fixture budget; no native authority claimed",
        )
        decoded = collective.decode_checkpoint_bytes(blob, budget)
        protocol.inspect_checkpoint_payload_integrity(decoded, runtime_kind=kind)
        admitted_children = tuple(
            SimpleNamespace(
                payload=collective.decode_checkpoint_bytes(
                    child.tobytes(), replace(budget, runtime_kind=child_kind)
                )
            )
            for child in children
        )
        protocol.require_bound_initial_children(
            admitted_children, composite_owner._temporal_restart_state
        )
        assert len(composite[MANIFEST_KEY]) <= collective._manifest_character_budget(
            tuple(name for name in composite if name not in {MANIFEST_KEY, IDENTITY_KEY})
        )
        with patch.object(
            collective,
            "_read_npy_header",
            side_effect=AssertionError("header decoded before admission"),
        ):
            must_refuse(
                lambda blob=blob, budget=budget: collective.decode_checkpoint_bytes(
                    blob, replace(budget, max_archive_bytes=len(blob) - 1)
                ),
                "archive bound before NPY",
            )
        with patch.object(
            collective,
            "_read_npy_array",
            side_effect=AssertionError("array decoded before manifest bound"),
        ):
            must_refuse(
                lambda blob=blob, budget=budget, composite=composite: (
                    collective.decode_checkpoint_bytes(
                        blob,
                        replace(budget, max_manifest_characters=len(composite[MANIFEST_KEY]) - 1),
                    )
                ),
                "manifest bound before array allocation",
            )
        noninitial = {
            name: deepcopy(value)
            for name, value in legacy.items()
            if name not in {MANIFEST_KEY, IDENTITY_KEY}
        }
        protocol._seal_checkpoint_payload_with_identities(
            noninitial,
            runtime_kind=child_kind,
            semantic=identities[0],
            artifact=identities[1],
            bind=identities[2],
            run=legacy_run,
        )
        admitted_noninitial = SimpleNamespace(
            payload=collective.decode_checkpoint_bytes(
                archive(noninitial), replace(budget, runtime_kind=child_kind)
            )
        )
        must_refuse(
            lambda admitted_noninitial=admitted_noninitial, composite_owner=composite_owner: (
                protocol.require_bound_initial_children(
                    (admitted_noninitial,), composite_owner._temporal_restart_state
                )
            ),
            "run-origin child inside initial composite",
        )
    observations["composite"] = (
        "Uniform/AMR child envelopes decoded with exact v2 origin; noninitial child refused"
    )
    observations["budget"] = (
        "actual NPZ decoder rejects archive before header decode and manifest before array decode"
    )

receipt = {
    "source_object": manifest["base_commit"],
    "source_tree_sha256": manifest["source_tree_sha256"],
    "probe_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    "scope": "source lifecycle/envelope protocol only; fake bound identity provider and ABI equality seam; no native capture/MPI/PDE",
    "observations": observations,
}
args.receipt.parent.mkdir(parents=True, exist_ok=True)
args.receipt.write_text(json.dumps(receipt, indent=2) + "\n")
print(json.dumps(receipt, indent=2))
