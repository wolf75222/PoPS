"""Native admission for the versioned consumed-Field mapping contract."""
from collections.abc import Mapping

from pops._generated_release_contract import NATIVE_ABI_VERSION

# The versioned endpoint capabilities remain published by the ABI11 release.
# Never infer compatibility with an ABI newer than this Python release.
_MAPPED_OUTPUT_FIRST_ABI = 9
_MAPPED_AMR_FIRST_ABI = 10


def requires_mapped_consumed_field_output(value):
    """Inspect the exact serialized Program, without inferring from its maximum IR version."""
    if isinstance(value, Mapping):
        if value.get("contract") == "mapped-consumed-output@1" or value.get("op") == "field_map_pack":
            return True
        return any(requires_mapped_consumed_field_output(item) for item in value.values())
    if isinstance(value, (tuple, list)):
        return any(requires_mapped_consumed_field_output(item) for item in value)
    return False


def require_mapped_consumed_field_output(artifact, *, capability_reader=None):
    """Refuse incompatible native facts before provider installation or System mutation."""
    required = False
    adaptive = False
    for row in artifact.layout_programs:
        program = row.program.program
        if program is None:
            raise ValueError("compiled layout Program has no retained Program authority")
        row_required = requires_mapped_consumed_field_output(program._serialize())
        required |= row_required
        if row_required and row.target not in ("system", "amr_system"):
            raise ValueError("mapped Field layout Program has no exact native target")
        adaptive |= row_required and row.target == "amr_system"
    if not required:
        return
    require_mapped_field_native_facts(capability_reader=capability_reader, adaptive=adaptive)


def require_mapped_field_native_facts(*, capability_reader=None, adaptive=False):
    if capability_reader is None:
        from pops._capabilities_report import _module_capabilities
        capability_reader = _module_capabilities
    facts = capability_reader("production")
    if not isinstance(facts, Mapping) or type(facts.get("abi_version")) is not int \
            or not (_MAPPED_OUTPUT_FIRST_ABI <= facts["abi_version"] <= NATIVE_ABI_VERSION) \
            or facts.get("mapped_consumed_field_output") is not True:
        raise RuntimeError(
            f"mapped-consumed-output@1 requires Native ABI9..{NATIVE_ABI_VERSION} and "
            "mapped_consumed_field_output; rebuild/install this exact source before bind"
        )

    if adaptive and (facts["abi_version"] < _MAPPED_AMR_FIRST_ABI or facts.get("mapped_consumed_field_output_amr") is not True):
        raise RuntimeError(
            f"AMR scalar endpoint@1 requires Native ABI10..{NATIVE_ABI_VERSION} and "
            "mapped_consumed_field_output_amr"
        )
