"""ADC-584: source-only parity gate between the Python and C++ route registries.

The typed native route IDs live in two mirrored registries: python/pops/runtime/routes.py
(the Python currency of the lowering layer) and include/pops/runtime/config/route_ids.hpp
(the C++ typed identities that cross the runtime ABI). They must carry the SAME families, the
SAME ordered tokens, the SAME native entry points and the SAME requirement/limitation contracts;
otherwise a behavior route means one thing in Python and another in C++.

route_ids.hpp already locks ITSELF against the historical single-source tag tables
(kLimiters / kRiemanns / kTransports / kSources / kElliptics) with compile-time static_asserts.
This test closes the remaining half of the loop -- it locks routes.py against route_ids.hpp --
WITHOUT a build.  It loads routes.py standalone (the module is deliberately import-free, stdlib
only) and parses the C++ table text with a tolerant regex, so it runs in the source-only
architecture gate before the compiled _pops extension exists.

The C++ rows can wrap across lines and split a field into adjacent string literals
("abc" "def"); the parser strips comments (respecting string literals) and concatenates adjacent
literals so the comparison stays faithful to what the compiler sees, not to the line layout.
"""
import importlib.util
import pathlib
import re

import pytest


REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
ROUTES_PY = REPO_ROOT / "python" / "pops" / "runtime" / "routes.py"
ROUTE_IDS_HPP = REPO_ROOT / "include" / "pops" / "runtime" / "config" / "route_ids.hpp"

# C++ `k<Name>Routes[]` table -> Python family key (routes.py `_TABLES` key).
CPP_TABLE_TO_FAMILY = {
    "Riemann": "riemann",
    "Limiter": "limiter",
    "Recon": "recon",
    "Time": "time",
    "Splitting": "splitting",
    "FieldSolver": "field_solver",
    "PoissonBc": "poisson_bc",
    "Layout": "layout",
    "Transport": "transport",
    "Source": "source",
    "Elliptic": "elliptic",
    "SourceStage": "source_stage",
    "PoissonRhs": "poisson_rhs",
    "Wall": "wall",
}

# The historical alias spellings, mirror of routes.py `_ALIASES` and the parse_*_route branches
# in route_ids.hpp.  The two alias sets must evolve together (see test below).
EXPECTED_ALIASES = {
    "source": {"lorentz": "magnetic", "potential_lorentz": "potential_magnetic"},
    "time": {"ssprk2": "explicit"},
}

_TABLE_RE = re.compile(r"k(\w+)Routes\s*\[\s*\]\s*=\s*\{")
_STRING_RE = re.compile(r'"((?:[^"\\]|\\.)*)"')
_INT_RE = re.compile(r"-?\d+")


# --- routes.py (Python side): load standalone, no pops import --------------------------------
def _load_routes_module():
    """Load routes.py by path, without importing the pops package or the compiled _pops."""
    spec = importlib.util.spec_from_file_location("_pops_routes_registry_parity", ROUTES_PY)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _parse_python_registry():
    module = _load_routes_module()
    families = {}
    for row in module.route_manifest():
        families.setdefault(row["family"], []).append({
            "token": row["token"],
            "native_entry": row["native_entry"],
            "requirements": tuple(row["requirements"]),
            "limitations": tuple(row["limitations"]),
        })
    return families, module


# --- route_ids.hpp (C++ side): tolerant, build-free table parsing ----------------------------
def _strip_comments(text):
    """Drop // and /* */ comments while preserving string literals verbatim.

    A commented-out row (e.g. `// {0, "x", ...}`) must not be parsed as a live route, and a `//`
    or `/*` that happens to sit inside a string literal must survive untouched.
    """
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
    """Return the text inside the brace group whose opening brace is at @p open_index.

    Braces inside string literals are ignored so a native entry / limitation containing a brace
    cannot unbalance the scan.
    """
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


def _tokenize_row(inner, family):
    """Tokenize a row body into integers, string literals and commas."""
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
                raise ValueError("route_ids.hpp %s row: unterminated string in %r"
                                 % (family, inner))
            tokens.append(("str", m.group(1)))
            i = m.end()
            continue
        m = _INT_RE.match(inner, i)
        if m is not None:
            tokens.append(("int", int(m.group(0))))
            i = m.end()
            continue
        i += 1  # stray punctuation (none expected after comment stripping)
    return tokens


def _split_csv(value):
    """Mirror routes.py `_split`: split on ',' and drop empty fields."""
    return tuple(part for part in value.split(",") if part)


def _interpret_cpp_row(row_text, family):
    """Turn one `{index, "token", "entry", "req", "lim"}` row into a comparable dict.

    Fields are comma-separated at the top level; adjacent string literals inside one field are
    concatenated (C++ literal concatenation), commas inside a string literal are NOT separators.
    """
    fields = [[]]
    for kind, value in _tokenize_row(row_text, family):
        if kind == "comma":
            fields.append([])
        else:
            fields[-1].append((kind, value))
    if len(fields) != 5:
        raise AssertionError(
            "route_ids.hpp %s row does not have 5 fields "
            "(index, token, native_entry, requirements, limitations): %r"
            % (family, row_text.strip()))

    def field_str(field):
        return "".join(v for kind, v in field if kind == "str")

    index_ints = [v for kind, v in fields[0] if kind == "int"]
    return {
        "index": index_ints[0] if index_ints else -1,
        "token": field_str(fields[1]),
        "native_entry": field_str(fields[2]),
        "requirements": _split_csv(field_str(fields[3])),
        "limitations": _split_csv(field_str(fields[4])),
    }


def _parse_cpp_registry():
    raw = ROUTE_IDS_HPP.read_text(encoding="utf-8")
    text = _strip_comments(raw)
    families = {}
    for m in _TABLE_RE.finditer(text):
        name = m.group(1)
        family = CPP_TABLE_TO_FAMILY.get(name)
        assert family is not None, (
            "route_ids.hpp defines k%sRoutes[] with no Python family mapping; the two registries "
            "must map 1:1, so add it to CPP_TABLE_TO_FAMILY and routes.py together" % name)
        inner = _scan_braced(text, m.end() - 1)
        families[family] = [_interpret_cpp_row(row, family) for row in _split_rows(inner)]
    return families, raw


# --- The parity gates -------------------------------------------------------------------------
def test_routes_module_loads_standalone_and_stays_import_free():
    """routes.py must load by path, without importing pops, so this gate needs no build."""
    source = ROUTES_PY.read_text(encoding="utf-8")
    offender = re.search(r"(?m)^\s*(?:import\s+pops|from\s+pops)\b", source)
    assert offender is None, (
        "python/pops/runtime/routes.py must stay import-free of the pops package: the source-only "
        "architecture gate loads it standalone via importlib.spec_from_file_location, BEFORE the "
        "compiled _pops extension exists, so any `import pops` / `from pops` would break the "
        "load; found %r" % (offender.group(0) if offender else None))

    module = _load_routes_module()
    manifest = module.route_manifest()
    assert manifest, "routes.py exposed an empty route manifest"
    assert all({"family", "token", "native_entry"} <= set(row) for row in manifest)
    assert isinstance(module._ALIASES, dict)


def test_python_and_cpp_expose_the_same_route_families():
    py_families, _ = _parse_python_registry()
    cpp_families, _ = _parse_cpp_registry()
    py_set, cpp_set = set(py_families), set(cpp_families)
    assert py_set == cpp_set, (
        "route family set drift between the two mirrored registries:\n"
        "  only in routes.py:      %s\n"
        "  only in route_ids.hpp:  %s"
        % (sorted(py_set - cpp_set), sorted(cpp_set - py_set)))


def test_ordered_tokens_and_native_entries_match_per_family():
    py_families, _ = _parse_python_registry()
    cpp_families, _ = _parse_cpp_registry()
    for family in sorted(set(py_families) & set(cpp_families)):
        py_rows = py_families[family]
        cpp_rows = cpp_families[family]
        assert len(py_rows) == len(cpp_rows), (
            "family %r row-count drift: routes.py has %d rows, route_ids.hpp has %d"
            % (family, len(py_rows), len(cpp_rows)))
        for i, (py_row, cpp_row) in enumerate(zip(py_rows, cpp_rows, strict=True)):
            assert cpp_row["index"] == i, (
                "route_ids.hpp family %r row %d carries index %d (expected %d); the C++ index "
                "column must mirror the enumerator position"
                % (family, i, cpp_row["index"], i))
            assert py_row["token"] == cpp_row["token"], (
                "family %r token mismatch at row %d: routes.py=%r route_ids.hpp=%r"
                % (family, i, py_row["token"], cpp_row["token"]))
            assert py_row["native_entry"] == cpp_row["native_entry"], (
                "family %r native_entry mismatch at row %d (token %r): "
                "routes.py=%r route_ids.hpp=%r"
                % (family, i, py_row["token"], py_row["native_entry"], cpp_row["native_entry"]))


def test_requirements_and_limitations_parity_per_family():
    py_families, _ = _parse_python_registry()
    cpp_families, _ = _parse_cpp_registry()
    for family in sorted(set(py_families) & set(cpp_families)):
        for i, (py_row, cpp_row) in enumerate(
                zip(py_families[family], cpp_families[family], strict=True)):
            assert py_row["requirements"] == cpp_row["requirements"], (
                "family %r requirements mismatch at row %d (token %r):\n"
                "  routes.py=%r\n  route_ids.hpp=%r"
                % (family, i, py_row["token"], py_row["requirements"], cpp_row["requirements"]))
            assert py_row["limitations"] == cpp_row["limitations"], (
                "family %r limitations mismatch at row %d (token %r):\n"
                "  routes.py=%r\n  route_ids.hpp=%r"
                % (family, i, py_row["token"], py_row["limitations"], cpp_row["limitations"]))


def test_alias_maps_evolve_together():
    module = _load_routes_module()
    aliases = {family: dict(mapping) for family, mapping in module._ALIASES.items()}
    assert aliases == EXPECTED_ALIASES, (
        "routes.py._ALIASES drifted from the historical alias spellings; the Python alias map and "
        "the C++ parse_*_route alias branches are one set and must evolve together:\n"
        "  routes.py._ALIASES = %r\n  expected           = %r" % (aliases, EXPECTED_ALIASES))

    raw = ROUTE_IDS_HPP.read_text(encoding="utf-8")
    for needle in ('token == "lorentz"', 'token == "ssprk2"'):
        assert needle in raw, (
            "route_ids.hpp is missing the alias branch `%s`; the C++ parse_*_route alias "
            "resolution and the Python _ALIASES map are one set and must evolve together" % needle)


def test_embedded_route_manifest_signature_and_version_parity():
    """ADC-599: the EMBEDDED route manifest (signature + registry version) is one across the mirror.

    ``route_registry_signature()`` ("family:count,..." in registry order) is baked verbatim into
    every generated artifact (pops_compiled_route_manifest / pops_program_route_manifest) and
    compared at load time by pops::verify_route_manifest.  The Python producer
    (routes.py::route_registry_signature) and the C++ consumer (route_ids.hpp::route_registry_signature)
    must emit the SAME string, and ROUTE_REGISTRY_VERSION must equal kRouteRegistryVersion; otherwise
    a freshly built .so would be refused (or wrongly accepted) against its own headers.  We recompute
    the signature from the already-parsed tables on BOTH sides (registry order = table order) and
    cross-check the Python value against routes.py's own function."""
    py_families, module = _parse_python_registry()
    cpp_families, cpp_raw = _parse_cpp_registry()

    def _signature(families):
        return ",".join("%s:%d" % (family, len(rows)) for family, rows in families.items())

    py_sig = _signature(py_families)
    cpp_sig = _signature(cpp_families)
    # The Python-side function itself must agree with the recomputed table signature (guards a
    # divergence between _TABLES iteration order and route_registry_signature()).
    assert module.route_registry_signature() == py_sig, (
        "routes.py::route_registry_signature() %r disagrees with its own _TABLES row counts %r"
        % (module.route_registry_signature(), py_sig))
    assert py_sig == cpp_sig, (
        "embedded route manifest signature drift between the two mirrored registries:\n"
        "  routes.py::route_registry_signature() = %r\n"
        "  route_ids.hpp::route_registry_signature() = %r\n"
        "a generated artifact would be refused against its own pops headers" % (py_sig, cpp_sig))

    version_match = re.search(r"kRouteRegistryVersion\s*=\s*(-?\d+)", cpp_raw)
    assert version_match is not None, (
        "route_ids.hpp is missing `kRouteRegistryVersion = <n>`; it mirrors "
        "routes.py::ROUTE_REGISTRY_VERSION and both must be present")
    assert module.ROUTE_REGISTRY_VERSION == int(version_match.group(1)), (
        "route registry version drift: routes.py ROUTE_REGISTRY_VERSION=%d vs "
        "route_ids.hpp kRouteRegistryVersion=%s; bump BOTH on an incompatible registry change"
        % (module.ROUTE_REGISTRY_VERSION, version_match.group(1)))


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
