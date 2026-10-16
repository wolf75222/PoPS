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
    field_components(field)
    return signature


def field_components(field):
    """Storage width follows the exact scientific producer, never its State neighbour."""
    if not isinstance(field, Mapping) or field.get("vtype") not in ("state", "scalar_field"):
        raise TypeError("FieldTopology requires an authenticated complete Field signature")
    authority = field.get("field_authority")
    if authority is not None:
        if (field.get("vtype") != "scalar_field" or field.get("block") is not None
                or not isinstance(authority, Mapping) or authority.get("kind") != "field_problem"
                or authority.get("version") != 1 or type(authority.get("components")) is not int
                or authority["components"] < 1):
            raise TypeError("FieldTopology has an invalid independent Field provider")
        return authority["components"]
    space = field.get("space")
    if field.get("block") is None or space is None or not getattr(space, "components", ()):
        raise TypeError("FieldTopology requires an authenticated complete Field signature")
    return len(space.components)
