"""Small source-only fixtures for the exact ADC-660 phase records."""
from __future__ import annotations

from pops.codegen._plans import ResolvedBlock, ResolvedSimulationPlan
from pops.codegen.compiled_artifact import CompiledBlockArtifact, CompiledSimulationArtifact
from pops.identity import make_identity
from pops.model.bind_schema import BindSchema
from pops.problem._snapshot import AuthoringSnapshot


class CanonicalValue:
    def __init__(self, name):
        self.name = name

    def to_data(self):
        return {"name": self.name}


class CompiledComponent:
    def __init__(self, name, *, target):
        self.name = name
        self.program = None
        self.program_name = name
        self.program_hash = None
        self.cache_key = None
        self.so_path = "/tmp/%s.so" % name
        self.target = target
        self.backend = "production"
        self.caps = {"cpu": True, "amr": target == "amr_system", "mpi": False, "gpu": False}
        self.abi_key = "test-headers|clang++|c++23"
        self.cxx = "clang++"
        self.std = "c++23"
        self.artifact_identity = make_identity("artifact", {"component": name})

    def inspect(self):
        return {"component": self.name, "status": "compiled"}

    def requirements(self):
        return {"component": self.name, "backend": self.backend}

    def manifest(self):
        return {"component": self.name, "target": self.target}

    def arguments(self):
        return {"component": self.name}

    def capability_matrix(self):
        return {"cpu": True, "amr": self.target == "amr_system"}


def artifact_fixture(*, target="system", block_names=("fluid",), bind_schema=None):
    source_models = tuple(CanonicalValue("source-" + name) for name in block_names)
    spatial = tuple({"mesh": {"block": name}} for name in block_names)
    schema = BindSchema() if bind_schema is None else bind_schema
    plan = ResolvedSimulationPlan(
        snapshot=AuthoringSnapshot({"case": "typed-artifact", "target": target}),
        target=target,
        backend="production",
        layout={"kind": "amr" if target == "amr_system" else "uniform"},
        time=None if target == "amr_system" else CanonicalValue("rk2"),
        blocks=tuple(
            ResolvedBlock(name, model, block_spatial, "production")
            for name, model, block_spatial in zip(
                block_names, source_models, spatial, strict=True)
        ),
        bind_schema=schema,
        compile_values=schema.resolve_compile(),
        field_solvers={},
        outputs=(),
        diagnostics=(),
        libraries=(),
        requirements={"amr": target == "amr_system"},
        capabilities={"cpu": True, "amr": target == "amr_system"},
    )
    components = tuple(CompiledComponent(name, target=target) for name in block_names)
    blocks = tuple(
        CompiledBlockArtifact(name, component, resolved.spatial)
        for name, component, resolved in zip(block_names, components, plan.blocks, strict=True)
    )
    program = None if target == "amr_system" else CompiledComponent("program", target=target)
    return CompiledSimulationArtifact(plan=plan, program=program, blocks=blocks)
