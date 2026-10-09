"""Produce actual native outputs in an isolated process, before running the oracle."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time
import tempfile

import numpy as np

from . import compiler_relay
from .provenance import (
    core_identity,
    digest,
    exclusive_directory,
    loaded_paths,
    tree_manifest,
    write_json,
)
from .reference import VARIANTS, budget, require


def event(path, data):
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(data, sort_keys=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def capture(variant, output, source, wheel):
    import pops
    from .author import DT, author_case, literal_inputs

    p = VARIANTS[variant]
    with exclusive_directory(output) as directory:
        before = core_identity(source, wheel)
        write_json(directory / "pre-native-budget.json", budget(p))
        # Fresh per-process caches force real compilation for every named witness.
        os.environ.update(
            {
                "POPS_CXX": str(Path(compiler_relay.__file__).resolve()),
                "POPS_WITNESS_CXX_REAL": before["compiler"],
                "POPS_WITNESS_COMPILER_EVIDENCE": str(directory / "compiler"),
                "POPS_CACHE_DIR": str(directory / "cache"),
                "POPS_NATIVE_CACHE_DIR": str(directory / "native-cache"),
                "POPS_CODEGEN_DIR": str(directory / "generated"),
                "POPS_KEEP_GENERATED": "1",
            }
        )
        labels = {}
        if variant.endswith("-renamed"):
            labels = {
                role: "witness_renamed_" + role
                for role in (
                    "domain",
                    "donor_model",
                    "donor_state",
                    "receiver_model",
                    "receiver_state",
                    "phi_input",
                    "psi_input",
                    "case",
                    "donor_block",
                    "receiver_block",
                    "field_owner",
                    "phi",
                    "psi",
                    "field_problem",
                    "method",
                )
            }
        # Values cast here are the declared rational witness parameters, not fitted coefficients.
        from fractions import Fraction

        authored = author_case(
            epsilon=Fraction(str(p.epsilon)), beta=Fraction(str(p.beta)), labels=labels
        )
        inputs = literal_inputs()
        np.savez(directory / "actual-bind-inputs.npz", **inputs)
        producer = {
            "pid": os.getpid(),
            "job_id": os.environ.get("SLURM_JOB_ID"),
            "hostname": os.uname().nodename,
            "python": sys.executable,
            "variant": variant,
        }
        event(
            directory / "execution.jsonl",
            {
                **producer,
                "event": "start",
                "time_ns": time.time_ns(),
                "input_sha256": digest(directory / "actual-bind-inputs.npz"),
            },
        )
        resolved = pops.resolve(pops.validate(authored.case), layout=authored.layout)
        require(resolved.resolved_dimension == 2, "unexpected dimension")
        artifact = pops.compile(resolved)
        directory.joinpath("dumps").mkdir()
        for i, block in enumerate(artifact.blocks):
            block.model.dump_cpp(directory / "dumps" / ("model%d.cpp" % i))
        for i, row in enumerate(artifact.layout_programs):
            row.program.dump_cpp(directory / "dumps" / ("program%d.cpp" % i))
        receiver, donor = (
            labels.get("receiver_block", "receiver"),
            labels.get("donor_block", "reservoir"),
        )
        # Serial launch accepts empty resources through the public bind API.
        runtime = pops.bind(
            artifact, initial_state={receiver: inputs["receiver"], donor: inputs["donor"]}
        )
        report = pops.run(runtime, t_end=float(DT), max_steps=1, console=False)
        shape = inputs["receiver"].shape
        arrays = {
            "qfinal": np.asarray(runtime.state_global(receiver)).reshape(shape).copy(),
            "d_final": np.asarray(runtime.state_global(donor)).reshape(shape).copy(),
            "d_initial": inputs["donor"].copy(),
        }
        histories = {}
        for stage in range(3):
            for key, name in [("q" + str(stage), "receiver-stage-%d" % stage)] + [
                (role + str(stage), "observed-stage-%d-%s" % (stage, role))
                for role in ("phi", "psi")
            ]:
                arrays[key] = np.asarray(runtime.history_global(name, 1)).reshape(shape).copy()
                histories[name] = {
                    "depth": runtime.history_depth(name),
                    "dt": runtime.history_slot_dt(name, 1),
                }
        require(set(runtime.history_names()) == set(histories), "history contract changed")
        require(
            report.accepted_steps == 1
            and runtime.macro_step() == 1
            and runtime.time() == float(DT),
            "not one accepted native step",
        )
        require(
            all(a.dtype == np.float64 and np.isfinite(a).all() for a in arrays.values()),
            "invalid native arrays",
        )
        np.savez(directory / "actual-arrays.npz", **arrays)
        # The core's rename-no-replace contract is unsupported by ROMEO GPFS.
        # Execute it on the compute node's local filesystem and retain every original byte.
        checkpoint_dir = Path(tempfile.mkdtemp(prefix="pops-witness-checkpoint-", dir="/tmp"))
        checkpoint = Path(runtime.checkpoint(checkpoint_dir / "accepted-checkpoint"))
        require(checkpoint.resolve().is_relative_to(checkpoint_dir.resolve()), "checkpoint path")
        shutil.copytree(checkpoint_dir, directory / "checkpoint")
        checkpoint_receipt = {
            "original_path": str(checkpoint),
            "original_directory": str(checkpoint_dir),
            "retained": str(Path("checkpoint") / checkpoint.relative_to(checkpoint_dir)),
            "files": tree_manifest(checkpoint_dir),
        }
        maps = loaded_paths()
        (directory / "loaded-maps.txt").write_text(
            Path("/proc/self/maps").read_text(), encoding="utf-8"
        )
        paths = {before["native_path"]: "core"}
        for row in artifact.blocks:
            paths[str(Path(row.model.so_path).resolve())] = "model"
        for path in artifact.layout_program_paths.values():
            paths[str(Path(path).resolve())] = "program"
        binaries = []
        (directory / "binaries").mkdir()
        for path, role in paths.items():
            value = digest(path)
            # Model loading seals a private image in /tmp/pops-native-*.
            # Identify that actually mapped image by its bytes, never by catalogue name alone.
            matches = [
                mapped
                for mapped in maps
                if ".so" in mapped and Path(mapped).is_file() and digest(mapped) == value
            ]
            require(matches, "no loaded image matches artifact bytes: " + path)
            mapped_path = sorted(matches)[0]
            dest = "binaries/" + value + ".so"
            shutil.copyfile(mapped_path, directory / dest)
            binaries.append(
                {
                    "original_path": mapped_path,
                    "artifact_path": path,
                    "sealed_copy": mapped_path != path,
                    "retained": dest,
                    "sha256": value,
                    "role": role,
                }
            )
        # Authenticate the loaded support-library closure by original path and current bytes.
        closure = {
            path: digest(path) for path in sorted(maps) if ".so" in path and Path(path).is_file()
        }
        after = core_identity(source, wheel)
        require(after == before, "core changed during witness")
        cpp_inputs = []
        for invocation in sorted((directory / "compiler").glob("*/invocation.json")):
            record = json.loads(invocation.read_text())
            for item in record["inputs"]:
                cpp_inputs.append(item["sha256"])
        receipt = {
            **producer,
            "physics": {"epsilon": p.epsilon, "beta": p.beta},
            "core_before": before,
            "core_after": after,
            "loaded_binaries": binaries,
            "loaded_paths": sorted(maps),
            "loader_closure": closure,
            "compiler_inputs_digest": hashlib.sha256(
                "".join(sorted(cpp_inputs)).encode()
            ).hexdigest(),
            "library_files": tree_manifest(Path(__file__).parent),
            "histories": histories,
            "checkpoint": checkpoint_receipt,
            "installed_program_hash": runtime.installed_program_hash(),
            "run_report": report.to_data(),
            "program_report": runtime.program_report().to_dict(),
            "scope": "Observed process + retained raw files; not a cryptographic proof of computational origin.",
            "missing_contracts": [
                "Persistent public successful-CG iteration/residual readout.",
                "Persistent public publication-stage timestamp readout from an accepted checkpoint.",
            ],
        }
        write_json(directory / "capture.json", receipt)
        originals = tree_manifest(directory)
        originals.pop("execution.jsonl", None)
        event(
            directory / "execution.jsonl",
            {
                **producer,
                "event": "finish",
                "time_ns": time.time_ns(),
                "accepted_steps": report.accepted_steps,
                "output_sha256": digest(directory / "actual-arrays.npz"),
                "receipt_sha256": digest(directory / "capture.json"),
                "files": originals,
            },
        )
        # Seal the original files before an independent process reads them.
        for path in directory.rglob("*"):
            if path.is_file():
                path.chmod(0o555 if os.access(path, os.X_OK) else 0o444)
        print(json.dumps({"variant": variant, "directory": str(directory), "producer": producer}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", required=True, choices=tuple(VARIANTS))
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--wheel", required=True, type=Path)
    args = parser.parse_args()
    capture(args.variant, args.out.resolve(), args.source.resolve(), args.wheel.resolve())


if __name__ == "__main__":
    main()
