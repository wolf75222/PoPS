#!/usr/bin/env python3
"""Exact MPI execution resources are exercised only under a real MPI launcher."""
from __future__ import annotations

from dataclasses import replace
import os
import sys


_REQUIRE_MPI = os.environ.get("POPS_REQUIRE_MPI_TESTS") == "1"

try:
    from mpi4py import MPI

    from pops._platform_contracts import (
        CapabilityProof,
        ExecutionContext,
        ExecutionResource,
        proven_serial_manifest,
    )
    from pops.codegen._compiled_artifact import CompiledSimulationArtifact
    from pops.runtime._component_execution_context import component_execution_data
except Exception as exc:  # noqa: BLE001 -- optional outside the required MPI lane
    if _REQUIRE_MPI:
        raise RuntimeError("required MPI execution-context contract could not import") from exc
    print("skip test_execution_context_mpi_world (MPI runtime unavailable: %s)" % exc)
    sys.exit(0)


if MPI.COMM_WORLD.Get_size() < 2:
    if _REQUIRE_MPI:
        raise RuntimeError("MPI execution-context contract requires at least two ranks")
    print("skip test_execution_context_mpi_world (needs mpiexec -n 2)")
    sys.exit(0)


platform = replace(
    proven_serial_manifest(
        backend="production", target="system", abi="test|c++|c++23"
    ),
    communicator=CapabilityProof.proven(
        "MPI_COMM_WORLD", "test.real-mpi-world"
    ),
)
artifact = object.__new__(CompiledSimulationArtifact)
object.__setattr__(artifact, "platform_manifest", platform)
context = ExecutionContext.mpi_world(artifact, MPI.COMM_WORLD)
projected = component_execution_data(context)

assert context.communicator.handle is MPI.COMM_WORLD
assert context.datatype.handle is MPI.DOUBLE
assert projected["communicator_f_handle"] == int(MPI.COMM_WORLD.py2f())
assert projected["communicator_datatype_f_handle"] == int(MPI.DOUBLE.py2f())
assert projected["communicator_identity"] == "MPI_COMM_WORLD"
assert projected["communicator_datatype_identity"] == "MPI_DOUBLE"

duplicate = MPI.COMM_WORLD.Dup()
try:
    try:
        ExecutionContext.mpi_world(artifact, duplicate)
    except ValueError as exc:
        assert "only mpi4py.MPI.COMM_WORLD" in str(exc)
    else:
        raise AssertionError("a duplicated communicator was accepted")
finally:
    duplicate.Free()

datatype = MPI.DOUBLE.Create_contiguous(1)
datatype.Commit()
try:
    changed = replace(
        context,
        datatype=ExecutionResource("datatype", "float64", handle=datatype),
    )
    try:
        component_execution_data(changed)
    except ValueError as exc:
        assert "exact mpi4py.MPI.DOUBLE" in str(exc)
    else:
        raise AssertionError("a noncanonical MPI datatype was accepted")
finally:
    datatype.Free()

if MPI.COMM_WORLD.Get_rank() == 0:
    print("PASS test_execution_context_mpi_world")
