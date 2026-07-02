"""ADC-586: source-only parity gate between the Python and C++ builtin brick catalogs, plus the
acceptance gate that no NEW hand-coded brick-token list may live in an error message string.

The builtin native brick catalog lives in two mirrored places: python/pops/runtime/brick_catalog.py
(the import-free Python inspection surface) and include/pops/runtime/builders/factory/brick_catalog.hpp
(the light C++ table the codegen / dispatch reference). They must carry the SAME categories, the SAME
ordered ids, the SAME native entries / params / n_vars / polar_ok; otherwise a native brick means one
thing in Python and another in C++.

brick_catalog.hpp already locks ITSELF against the two single sources (model_registry.hpp and
route_ids.hpp) with compile-time static_asserts, and the two single sources are already locked to each
other by test_route_registry_parity.py. This test closes the remaining loop -- it locks
brick_catalog.py against brick_catalog.hpp -- WITHOUT a build, by loading brick_catalog.py standalone
(deliberately import-free) and parsing the C++ table with the same tolerant comment/brace-aware scan
test_route_registry_parity.py uses.

It ALSO carries the ADC-586 acceptance gate: adding a central brick must not force a new hand-coded
public brick-token list in an ERROR message. The gate scans include/ and python/bindings/ for a
brick-token enumeration ("exb|compressible", "charge|background", ...) INSIDE a quoted string, after
stripping comments, and asserts every such hit lives in one of the three single sources
(model_registry.hpp / route_ids.hpp / brick_catalog.hpp). A legitimate new hit is reported, never
allowlisted.
"""
import importlib.util
import pathlib
import re

import pytest


REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
CATALOG_PY = REPO_ROOT / "python" / "pops" / "runtime" / "brick_catalog.py"
CATALOG_HPP = REPO_ROOT / "include" / "pops" / "runtime" / "builders" / "factory" / "brick_catalog.hpp"

# The three single sources of the builtin brick token lists: a brick-token enumeration inside a
# quoted string is allowed ONLY in these files (they ARE the single sources the messages derive from).
SINGLE_SOURCE_FILES = {
    REPO_ROOT / "include" / "pops" / "runtime" / "dynamic" / "model_registry.hpp",
    REPO_ROOT / "include" / "pops" / "runtime" / "config" / "route_ids.hpp",
    CATALOG_HPP,
}

_TABLE_RE = re.compile(r"kBrickCatalog\s*\[\s*\]\s*=\s*\{")
_STRING_RE = re.compile(r'"((?:[^"\\]|\\.)*)"')
_INT_RE = re.compile(r"-?\d+")


# --- brick_catalog.py (Python side): load standalone, no pops import --------------------------
def _load_catalog_module():
    """Load brick_catalog.py by path, without importing the pops package or the compiled _pops."""
    spec = importlib.util.spec_from_file_location("_pops_brick_catalog_parity", CATALOG_PY)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _parse_python_catalog():
    module = _load_catalog_module()
    rows = []
    for entry in module.brick_catalog():
        rows.append({
            "category": entry["category"],
            "id": entry["id"],
            "route_index": entry["route_index"],
            "native_entry": entry["native_entry"],
            "params": tuple(entry["params"]),
            "n_vars": entry["n_vars"],
            "polar_ok": bool(entry["polar_ok"]),
        })
    return rows, module


# --- brick_catalog.hpp (C++ side): tolerant, build-free table parsing -------------------------
def _strip_comments(text):
    """Drop // and /* */ comments while preserving string literals verbatim (as route parity)."""
    out = []
    i = 0
    n = len(text)
    in_str = False
    while i < n:
        c = text[i]
        if in_str:
            out.append(c)
            if c == "\\" and i + 1 < n:
                out.append(text[i + 1])
                i += 2
                continue
            if c == '"':
                in_str = False
            i += 1
            continue
        if c == '"':
            in_str = True
            out.append(c)
            i += 1
            continue
        if c == "/" and i + 1 < n and text[i + 1] == "/":
            while i < n and text[i] != "\n":
                i += 1
            continue
        if c == "/" and i + 1 < n and text[i + 1] == "*":
            i += 2
            while i + 1 < n and not (text[i] == "*" and text[i + 1] == "/"):
                i += 1
            i += 2
            continue
        out.append(c)
        i += 1
    return "".join(out)


def _scan_braced(text, open_index):
    """Return the text inside the brace group whose opening brace is at @p open_index."""
    depth = 0
    i = open_index
    n = len(text)
    in_str = False
    while i < n:
        c = text[i]
        if in_str:
            if c == "\\":
                i += 2
                continue
            if c == '"':
                in_str = False
            i += 1
            continue
        if c == '"':
            in_str = True
        elif c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return text[open_index + 1:i]
        i += 1
    raise ValueError("unbalanced braces starting at offset %d" % open_index)


def _split_rows(block):
    """Split a table body into its top-level `{...}` row groups (string/brace-aware)."""
    rows = []
    i = 0
    n = len(block)
    while i < n:
        if block[i] == "{":
            inner = _scan_braced(block, i)
            rows.append(inner)
            i += len(inner) + 2
        else:
            i += 1
    return rows


def _tokenize_row(inner):
    """Tokenize a row body into integers, string literals, identifiers (true/false) and commas."""
    tokens = []
    i = 0
    n = len(inner)
    while i < n:
        c = inner[i]
        if c.isspace():
            i += 1
            continue
        if c == ",":
            tokens.append(("comma", None))
            i += 1
            continue
        if c == '"':
            m = _STRING_RE.match(inner, i)
            if m is None:
                raise ValueError("brick_catalog.hpp row: unterminated string in %r" % inner)
            tokens.append(("str", m.group(1)))
            i = m.end()
            continue
        m = _INT_RE.match(inner, i)
        if m is not None:
            tokens.append(("int", int(m.group(0))))
            i = m.end()
            continue
        m = re.match(r"[A-Za-z_]\w*", inner[i:])
        if m is not None:
            tokens.append(("ident", m.group(0)))
            i += m.end()
            continue
        i += 1  # stray punctuation (none expected after comment stripping)
    return tokens


def _interpret_cpp_row(row_text):
    """Turn one BrickCatalogEntry row into a comparable dict (10 top-level fields).

    Row shape: {"id", "category", route_index, "native_entry", "params", n_vars, polar_ok,
    "requirements", "capabilities", "summary"}. Adjacent string literals inside one field are
    concatenated (C++ literal concatenation); commas inside a string literal are NOT separators.
    """
    fields = [[]]
    for kind, value in _tokenize_row(row_text):
        if kind == "comma":
            fields.append([])
        else:
            fields[-1].append((kind, value))
    assert len(fields) == 10, (
        "brick_catalog.hpp row does not have 10 fields "
        "(id, category, route_index, native_entry, params, n_vars, polar_ok, requirements, "
        "capabilities, summary): %r" % row_text.strip())

    def field_str(field):
        return "".join(v for kind, v in field if kind == "str")

    def field_int(field):
        ints = [v for kind, v in field if kind == "int"]
        return ints[0] if ints else None

    def field_bool(field):
        idents = [v for kind, v in field if kind == "ident"]
        return idents[0] == "true"

    return {
        "id": field_str(fields[0]),
        "category": field_str(fields[1]),
        "route_index": field_int(fields[2]),
        "native_entry": field_str(fields[3]),
        "params": tuple(p for p in field_str(fields[4]).split(",") if p),
        "n_vars": field_int(fields[5]),
        "polar_ok": field_bool(fields[6]),
    }


def _parse_cpp_catalog():
    raw = CATALOG_HPP.read_text(encoding="utf-8")
    text = _strip_comments(raw)
    m = _TABLE_RE.search(text)
    assert m is not None, "brick_catalog.hpp is missing the `kBrickCatalog[] = {` table"
    inner = _scan_braced(text, m.end() - 1)
    return [_interpret_cpp_row(row) for row in _split_rows(inner)], raw


# --- The parity gates -------------------------------------------------------------------------
def test_catalog_module_loads_standalone_and_stays_import_free():
    """brick_catalog.py must load by path, without importing pops, so this gate needs no build."""
    source = CATALOG_PY.read_text(encoding="utf-8")
    offender = re.search(r"(?m)^\s*(?:import\s+pops|from\s+pops)\b", source)
    assert offender is None, (
        "python/pops/runtime/brick_catalog.py must stay import-free of the pops package: the "
        "source-only architecture gate loads it standalone via importlib, BEFORE the compiled "
        "_pops extension exists; found %r" % (offender.group(0) if offender else None))

    module = _load_catalog_module()
    catalog = module.brick_catalog()
    assert catalog, "brick_catalog.py exposed an empty catalog"
    assert all({"id", "category", "native_entry", "params", "n_vars"} <= set(row)
               for row in catalog)


def test_python_and_cpp_expose_the_same_categories_and_order():
    py_rows, _ = _parse_python_catalog()
    cpp_rows, _ = _parse_cpp_catalog()
    assert len(py_rows) == len(cpp_rows) == 11, (
        "brick catalog row-count drift: brick_catalog.py has %d rows, brick_catalog.hpp has %d "
        "(expected 11: 3 transports + 5 canonical sources + 3 elliptics)"
        % (len(py_rows), len(cpp_rows)))
    py_key = [(r["category"], r["id"]) for r in py_rows]
    cpp_key = [(r["category"], r["id"]) for r in cpp_rows]
    assert py_key == cpp_key, (
        "brick catalog (category, id) order drift between the two mirrors:\n"
        "  brick_catalog.py  = %s\n  brick_catalog.hpp = %s" % (py_key, cpp_key))


def test_python_and_cpp_columns_match_per_row():
    py_rows, _ = _parse_python_catalog()
    cpp_rows, _ = _parse_cpp_catalog()
    for i, (py_row, cpp_row) in enumerate(zip(py_rows, cpp_rows, strict=True)):
        for col in ("route_index", "native_entry", "params", "n_vars", "polar_ok"):
            assert py_row[col] == cpp_row[col], (
                "brick catalog %r column %r mismatch at row %d (id %r):\n"
                "  brick_catalog.py  = %r\n  brick_catalog.hpp = %r"
                % (py_row["category"], col, i, py_row["id"], py_row[col], cpp_row[col]))


def test_no_new_hand_coded_brick_token_list_in_error_strings():
    """ADC-586 acceptance gate: no NEW hand-coded public brick-token list in an error message.

    Scan include/ and python/bindings/ for a brick-token enumeration inside a quoted string, after
    stripping comments; every such hit must live in one of the three single sources. A brick-token
    enumeration is two adjacent canonical brick ids joined by ``|`` (the message form) -- the exact
    shape a hand-written rejection list would take. A legitimate new hit (a genuine new single
    source) is REPORTED here, never silently allowlisted.
    """
    module = _load_catalog_module()
    ids_by_cat = {}
    for row in module.brick_catalog():
        ids_by_cat.setdefault(row["category"], []).append(row["id"])
    # Pipe-joined pairs of adjacent canonical ids: the token-list shape a message would hand-code.
    pair_res = []
    for cat_ids in ids_by_cat.values():
        for a, b in zip(cat_ids, cat_ids[1:], strict=False):
            pair_res.append(re.compile(re.escape(a) + r"\s*\|\s*" + re.escape(b)))

    scan_roots = [REPO_ROOT / "include", REPO_ROOT / "python" / "bindings"]
    offenders = []
    for root in scan_roots:
        for path in sorted(root.rglob("*")):
            if not path.is_file() or path.suffix not in (".hpp", ".h", ".cpp", ".cc"):
                continue
            if path.resolve() in SINGLE_SOURCE_FILES:
                continue
            text = _strip_comments(path.read_text(encoding="utf-8", errors="replace"))
            for literal in _STRING_RE.findall(text):
                for pair_re in pair_res:
                    if pair_re.search(literal):
                        offenders.append((str(path.relative_to(REPO_ROOT)), literal))
                        break

    assert not offenders, (
        "ADC-586: a hand-coded brick-token list appears in an error-message string OUTSIDE the "
        "single sources (model_registry.hpp / route_ids.hpp / brick_catalog.hpp). A central brick "
        "must not force a new public token list; derive the message from the catalog / registry "
        "csv helpers instead. If this is a genuine new single source, report it (do NOT allowlist):"
        "\n" + "\n".join("  %s: %r" % (p, lit) for p, lit in offenders))


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
