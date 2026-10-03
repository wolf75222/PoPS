"""Canonical physical units at the native optional-string contract boundary."""
import json
from collections.abc import Mapping
from pops.model import PhysicalDimension

def _utf8_string_cpp(text):
    """A C++ string expression preserving every declared UTF-8 byte."""
    payload = text.encode("utf-8")
    escapes = {8: r"\b", 9: r"\t", 10: r"\n", 12: r"\f", 13: r"\r",
               34: r'\"', 92: r"\\"}
    literal = '"' + "".join(escapes.get(byte, chr(byte) if 32 <= byte < 127
                                        else "\\%03o" % byte)
                             for byte in payload) + '"'
    # The const-char* constructor would silently truncate an embedded NUL.
    return "std::string{%s, %d}" % (literal, len(payload)) if 0 in payload else literal


def optional_unit_cpp(unit):
    if unit is None:
        return "std::nullopt"
    if isinstance(unit, Mapping):
        raw = dict(unit)
        unit = PhysicalDimension.from_data(raw)
        if unit.to_data() != raw:
            raise ValueError("native physical dimension must be canonical")
    if isinstance(unit, PhysicalDimension):
        text = json.dumps(unit.to_data(), sort_keys=True, separators=(",", ":"))
    elif type(unit) is str and unit:
        text = unit
    else:
        raise TypeError("native unit requires a physical dimension, named unit, or None")
    return "std::optional<std::string>{%s}" % _utf8_string_cpp(text)
