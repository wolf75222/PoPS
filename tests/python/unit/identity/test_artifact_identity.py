from __future__ import annotations

from pops.identity import (
    artifact_identity,
    artifact_spec_identity,
    binary_identity,
    semantic_identity,
)


def _spec(semantic, *, target="system", routes=None):
    return artifact_spec_identity(
        semantic,
        target=target,
        backend="production",
        precision="double",
        abi="abi-v1",
        toolchain="clang++|c++20",
        routes=routes or {"registry": "v1", "feature": "serial"},
        components={"generated_source": b"source-digest"},
        flags=("-O3", "-DNDEBUG"),
        libraries=("library-content-digest",),
    )


def test_target_and_routes_change_artifact_spec_not_semantics():
    semantic = semantic_identity({"equation": "transport", "order": 2})

    system = _spec(semantic)
    amr = _spec(semantic, target="amr_system")
    routes = _spec(semantic, routes={"feature": "mpi", "registry": "v1"})

    assert system != amr
    assert system != routes
    assert semantic == semantic_identity({"order": 2, "equation": "transport"})


def test_final_artifact_authenticates_exact_binary_bytes(tmp_path):
    semantic = semantic_identity({"equation": "transport"})
    spec = _spec(semantic)
    path = tmp_path / "problem.so"
    path.write_bytes(b"first binary")
    first_binary = binary_identity(path)
    first = artifact_identity(spec, first_binary)

    path.write_bytes(b"second binary")
    second_binary = binary_identity(path)
    second = artifact_identity(spec, second_binary)

    assert first_binary != second_binary
    assert first != second
