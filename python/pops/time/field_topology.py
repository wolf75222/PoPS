"""Pure schema validation for ephemeral native topology capabilities."""
from collections.abc import Mapping


def validate_topology_signature(signature):
    if (not isinstance(signature, Mapping) or set(signature) != {
            "vtype", "field_signature", "side", "dimension"}
            or signature["vtype"] != "field_topology"
            or signature["side"] not in ("source", "target")
            or type(signature["dimension"]) is not int
            or not 1 <= signature["dimension"] <= 3):
        raise TypeError("FieldTopology requires an exact native dimension and source/target authority")
    field = signature["field_signature"]
    if (not isinstance(field, Mapping) or field.get("vtype") not in ("state", "scalar_field")
            or field.get("block") is None or field.get("space") is None
            or not getattr(field["space"], "components", ())):
        raise TypeError("FieldTopology requires an authenticated complete Field signature")
    return signature
