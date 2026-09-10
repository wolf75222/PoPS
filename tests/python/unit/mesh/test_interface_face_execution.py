"""Exact shared-interface face composition retains every unowned physical closure."""
from copy import deepcopy
from dataclasses import dataclass

import pytest

from pops.mesh.boundaries.interface_authoring import _InterfaceFaceExecutionAuthority
from pops.model import Handle, OwnerPath


@dataclass(frozen=True)
class _PhysicalAuthority:
    data: dict

    def canonical_identity(self):
        return {"physical": self.data}

    def compile_boundary_data(self):
        return self.data

    def runtime_boundary_data(self, params):
        assert params == {"bound": 2.0}
        return self.data


def _authority():
    owner = OwnerPath.case("interface_execution")
    provider = Handle("inlet", kind="boundary_provider", owner=owner)
    interface = Handle("join", kind="multiblock_interface", owner=owner)
    base = _PhysicalAuthority({
        "identity": "original-authority",
        "required_depth": 3,
        "state": "exact-left-state",
        "faces": [{
            "ordinal": ordinal,
            "producer": provider.qualified_id if ordinal == 1 else "other-%d" % ordinal,
            "geometry": {"frame": "left", "ordinal": ordinal},
            "type": "dirichlet",
            "representation": "primitive",
            "converter": "exact-converter",
            "values": [float(ordinal + 1)],
            "analytic_programs": [{"exact": ordinal}],
            "analytic_clock": "main",
        } for ordinal in range(4)],
    })
    return _InterfaceFaceExecutionAuthority(base, provider, interface, 1)


@pytest.mark.parametrize("runtime", [False, True])
def test_only_the_exact_owned_face_is_reserved_without_mutating_physical_authority(runtime):
    authority = _authority()
    before = deepcopy(authority.base.data)
    identity = deepcopy(authority.canonical_identity())
    assert identity != authority.base.canonical_identity()
    assert identity["physical_provider"] == authority.physical_provider.canonical_identity()
    assert identity["interface"] == authority.interface.canonical_identity()
    assert identity["face_ordinal"] == 1
    result = (authority.runtime_boundary_data({"bound": 2.0}) if runtime
              else authority.compile_boundary_data())
    assert authority.base.data == before
    assert authority.canonical_identity() == identity
    assert {key: value for key, value in result.items() if key != "faces"} == {
        key: value for key, value in before.items() if key != "faces"}
    for ordinal in (0, 2, 3):
        assert result["faces"][ordinal] == before["faces"][ordinal]
    selected = result["faces"][1]
    assert selected == {**before["faces"][1], "type": "external", "values": [],
                        "representation": "conservative", "converter": None,
                        "analytic_programs": [], "analytic_clock": None}
    selected["geometry"]["frame"] = "changed-result"
    assert authority.base.data == before


@pytest.mark.parametrize("fault", ["foreign-provider", "wrong-face", "duplicate", "periodic",
                                  "already-external"])
def test_interface_face_requires_unique_exact_physical_ownership(fault):
    authority = _authority()
    face = authority.base.data["faces"][1]
    if fault == "foreign-provider":
        face["producer"] = "another-owner::inlet"
    elif fault == "wrong-face":
        face["ordinal"] = 2
    elif fault == "duplicate":
        authority.base.data["faces"].append(deepcopy(face))
    else:
        face["type"] = "periodic" if fault == "periodic" else "external"
    before = deepcopy(authority.base.data)
    with pytest.raises(ValueError, match="owned physical face|already consumed"):
        authority.compile_boundary_data()
    assert authority.base.data == before
