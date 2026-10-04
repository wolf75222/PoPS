"""Exact public UTF-8 text at C++ string boundaries (cpp-public-text@2)."""
import json

CONTRACT = "cpp-public-text@2"


def cpp_string_literal(text, *, ensure_ascii=True):
    """C++ literal, retaining legacy JSON spellings where they are valid C++.

    JSON surrogate pairs and low universal-character escapes are not C++ escapes.
    Encode those characters without changing their declared UTF-8 bytes. Escaping
    per character also keeps an authored backslash-u sequence literal.
    """
    if not isinstance(text, str):
        raise TypeError("C++ text requires a string")
    parts = []
    for char in text:
        code = ord(char)
        if 0xD800 <= code <= 0xDFFF:
            # This is not a UTF-8 public identity (encode fails at the authority too).
            char.encode("utf8")
        if (code < 32 and code not in (8, 9, 10, 12, 13)) or 127 <= code < 160:
            parts.append("".join("\\%03o" % byte for byte in char.encode("utf8")))
        elif code > 0xFFFF and ensure_ascii:
            parts.append("\\U%08x" % code)
        else:
            parts.append(json.dumps(char, ensure_ascii=ensure_ascii)[1:-1])
    return '"' + "".join(parts) + '"'


def cpp_string_expression(text, *, ensure_ascii=True):
    literal = cpp_string_literal(text, ensure_ascii=ensure_ascii)
    return "std::string{%s, %d}" % (literal, len(text.encode("utf8"))) if "\0" in text else literal


def cpp_string_view_expression(text):
    """A borrowed view of a static literal, never of a temporary owned string."""
    literal = cpp_string_literal(text)
    return "std::string_view{%s, %d}" % (literal, len(text.encode("utf8"))) if "\0" in text else literal
