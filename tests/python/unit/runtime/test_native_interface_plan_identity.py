"""Native boundary descriptors preserve full authority across JSON detachment."""
from copy import deepcopy
import json

import pytest

from pops import interfaces
from pops.runtime._runtime_authorities import _native_interface_identity_matches


@pytest.mark.parametrize("interface", [interfaces.GhostBoundary, interfaces.NumericalFlux])
def test_exact_native_interface_survives_ordered_tuple_list_round_trip(interface):
    binding = {"native_interface": interface.to_data(), "interface_version": interface.version}
    detached = json.loads(json.dumps(binding))
    assert binding["native_interface"]["operations"] == tuple(
        detached["native_interface"]["operations"])
    assert _native_interface_identity_matches(binding, interface)
    assert _native_interface_identity_matches(detached, interface)


@pytest.mark.parametrize("field", ["uri", "version", "catalog_sha256", "protocol_abi",
                                  "cpp_table", "id", "name", "hot_path", "operations"])
def test_changed_native_interface_descriptor_is_refused(field):
    interface = interfaces.GhostBoundary
    binding = json.loads(json.dumps({"native_interface": interface.to_data(),
                                    "interface_version": interface.version}))
    value = binding["native_interface"][field]
    replacement = (["another_operation"] if field == "operations" else
                   not value if type(value) is bool else
                   value + 1 if type(value) is int else value + "/changed")
    binding["native_interface"][field] = replacement
    assert not _native_interface_identity_matches(binding, interface)


@pytest.mark.parametrize("version", [True, "1", 2])
def test_outer_interface_version_remains_exact(version):
    interface = interfaces.GhostBoundary
    binding = {"native_interface": deepcopy(interface.to_data()), "interface_version": version}
    assert not _native_interface_identity_matches(binding, interface)
