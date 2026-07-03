"""pops.runtime._bind_validation -- the pure bind-time refusal core (ADC-537).

``pops.bind`` refuses a bad install BEFORE the native artifact is loaded: an initial state of the
wrong shape / dtype / component count / ghost depth, a runtime param outside its typed domain, an
aux field a lowered operator requires but the state omits, and an ABI / Kokkos / MPI / layout
manifest mismatch. Every refusal is a HARD error with precise context; there is NO Python-runtime
fallback when the native load fails (that decision lives in :mod:`pops.codegen.orchestration`).

This module is the PURE core of those gates: each function takes plain metadata (manifest /
arguments / layout / declared params / supplied state) and returns one actionable refusal line per
violation (empty list = ok). No ``_pops`` and no numpy at module scope (arrays are duck-typed via
``.shape`` / ``.dtype``), so the whole refusal surface is host-testable with plain dicts;
:func:`aggregate_bind_refusals` folds the per-gate lines into one error.

Per the phase-6 cross-stream contract (decisions 4-5): per-block ghost depth and the ABI / Kokkos /
MPI feature tokens come from the compiled MANIFEST; a fresh artifact always carries them, and a
manifest lacking a field it must carry is refused as ABI-incomplete (fail loud, never skipped).

The ABI comparison is LIKE-WITH-LIKE: the artifact key (``<headers-sha>|<cxx>|<std>``) and the
runtime env key (``compiler=..;std=..;headers=..;kokkos=..;stdlib=..``) are parsed into components
and only the comparable ones are compared -- the headers signature (the real header-ABI anchor) and
the normalized C++ standard (``c++20`` == ``202002L``). Incomparable tokens (compiler path vs
version) are never compared; a token spelled ``unknown`` is an honest-unknown, skipped like ``None``.
"""
import re

# The runtime env-format abi key: 'compiler=..;std=202002L;headers=<sha>;kokkos=..;stdlib=..'.
_ENV_HEADERS_RE = re.compile(r"(?:^|;)\s*headers=([^;]+)")
_ENV_STD_RE = re.compile(r"(?:^|;)\s*std=(\d{6})L?(?:;|$)")
# Normalized C++ standard: a 'c++NN' / 'gnu++NN' flag token -> the 6-digit __cplusplus year value.
_CXX_FLAG_RE = re.compile(r"^(?:c|gnu)\+\+(\d{2})$")
_STD_YEARS = {"11": "201103", "14": "201402", "17": "201703", "20": "202002", "23": "202302"}
# String tokens that mean honest-unknown on either side (skipped, exactly like None).
_UNKNOWN_TOKENS = ("unknown", "")


def _normalize_std(token):
    """Normalize a C++ standard token to its 6-digit year form (``c++20``/``202002L`` -> ``202002``).

    Accepts the flag spelling (``c++20`` / ``gnu++23``) and the ``__cplusplus`` spelling
    (``202002L`` / ``202002``). An unparseable token returns ``None`` -- honest-unknown, skipped by
    the comparison (never a refusal on a token the parser does not understand)."""
    if token is None:
        return None
    text = str(token).strip().lower()
    year = re.match(r"^(\d{6})l?$", text)
    if year:
        return year.group(1)
    flag = _CXX_FLAG_RE.match(text)
    if flag:
        return _STD_YEARS.get(flag.group(1))
    return None


def _abi_components(abi_key):
    """Parse an abi key in EITHER representation into ``(headers_signature, normalized_std)``.

    Artifact form (:class:`~pops.codegen.loader.CompiledProblem`): ``<headers-sha>|<cxx>|<std>``
    (pipe-delimited; the first field is the headers signature, the third the ``c++NN`` standard).
    Runtime form (``pops.abi_key()``): the env string
    ``compiler=..;std=202002L;headers=<sha>;kokkos=..;stdlib=..``. An OPAQUE token (neither form)
    anchors the comparison on the whole string. A component a side cannot supply is ``None``
    (honest-unknown, skipped). The compiler token is deliberately NOT extracted: a compiler PATH
    (``/usr/bin/c++``) and a compiler VERSION (``13.3.0``) are not comparable, and the headers
    signature already covers the header ABI."""
    text = str(abi_key)
    env_headers = _ENV_HEADERS_RE.search(text)
    if env_headers:
        env_std = _ENV_STD_RE.search(text)
        return env_headers.group(1).strip(), (env_std.group(1) if env_std else None)
    if "|" in text:
        parts = [p.strip() for p in text.split("|")]
        headers = parts[0] or None
        std = _normalize_std(parts[2]) if len(parts) > 2 else None
        return headers, std
    return (text.strip() or None), None


def _is_unknown_token(value):
    """True when a string token is the honest-unknown spelling (``unknown`` / empty)."""
    return str(value).strip().lower() in _UNKNOWN_TOKENS


def _shape_of(array):
    """The ``.shape`` tuple of @p array, or ``None`` when it exposes none (a bare list / scalar)."""
    shape = getattr(array, "shape", None)
    if shape is None:
        return None
    return tuple(int(s) for s in shape)


def _dtype_name(array):
    """The dtype name of @p array (``arr.dtype.name`` / ``str(arr.dtype)``), or ``None``."""
    dtype = getattr(array, "dtype", None)
    if dtype is None:
        return None
    return getattr(dtype, "name", None) or str(dtype)


def _precision_dtype_names(precision):
    """The accepted dtype name(s) for a manifest ``precision`` token (``double`` -> float64)."""
    table = {"double": ("float64",), "single": ("float32",), "float": ("float32",)}
    return table.get(str(precision or "double").lower(), ("float64",))


def validate_initial_state(manifest, arguments, layout, initial_state):
    """Refuse an initial state that does not match the artifact + mesh (ADC-537 gate d / G4).

    For each supplied block array, check -- against the MANIFEST (the ABI truth) and the mesh
    LAYOUT -- the mesh shape (n x n cells, optionally with a ghost ring), the dtype (the artifact's
    declared real precision), the component count (the model's conservative variable count) and the
    ghost depth. A supplied block name the artifact does not declare is also refused. Returns one
    actionable line per mismatch (empty list = ok).

    Sourcing:
      - the declared blocks + component count come from @p arguments (``instances``);
      - the mesh extent comes from @p layout (``Uniform.mesh`` / ``AMR.base`` -> a 2D n x n grid);
      - the ghost depth + real precision come from @p manifest (``ghost_depth`` / ``precision``);
        a manifest that carries no ``ghost_depth`` is refused as ABI-incomplete (never guessed).
    """
    lines = []
    if not initial_state:
        return lines
    instances = dict(getattr(arguments, "instances", {}) or {})
    declared = set(instances)
    for name in sorted(set(initial_state) - declared):
        lines.append("initial state for unknown block %r; the artifact declares block(s) %s"
                     % (name, sorted(declared) or "(none)"))
    mesh = _layout_mesh(layout)
    ghost = getattr(manifest, "ghost_depth", None)
    if ghost is None and initial_state:
        lines.append("the compiled manifest carries no ghost_depth; it is ABI-incomplete and cannot "
                     "be bound safely (rebuild the artifact so its manifest records the halo depth)")
    accepted_dtypes = _precision_dtype_names(getattr(manifest, "precision", None))
    for name in sorted(set(initial_state) & declared):
        array = initial_state[name]
        spec = instances[name]
        _check_one_initial_state(lines, name, array, spec, mesh, ghost, accepted_dtypes)
    return lines


def _check_one_initial_state(lines, name, array, spec, mesh, ghost, accepted_dtypes):
    """Append the shape / dtype / component refusals for ONE block's supplied @p array."""
    components = int(spec.get("components", 0) or 0)
    shape = _shape_of(array)
    if shape is None:
        lines.append("initial state for block %r is not an array (no .shape); pass a numpy array of "
                     "shape (%s, n, n)" % (name, components or "n_components"))
        return
    if mesh is not None and ghost is not None:
        expected = _expected_shapes(components, mesh, ghost)
        if shape not in expected:
            lines.append("initial state for block %r has shape %s; expected one of %s (n=%d cells "
                         "per axis, %d component(s), ghost depth %d)"
                         % (name, shape, sorted(expected), mesh, components, ghost))
    dtype = _dtype_name(array)
    if dtype is not None and dtype not in accepted_dtypes:
        lines.append("initial state for block %r has dtype %r; the artifact's declared precision "
                     "expects %s" % (name, dtype, " or ".join(accepted_dtypes)))


def _expected_shapes(components, mesh, ghost):
    """The set of accepted (..., n, n) shapes for @p components on an @p mesh (valid or +ghost ring)."""
    n = int(mesh)
    valid = (n, n)
    haloed = (n + 2 * int(ghost), n + 2 * int(ghost))
    shapes = set()
    for grid in (valid, haloed):
        if components and components > 1:
            shapes.add((components,) + grid)
        else:
            shapes.add((1,) + grid)
            shapes.add(grid)  # a single-component block may be a bare (n, n) array
    return shapes


def _layout_mesh(layout):
    """The 2D cell count ``n`` (an n x n grid) of @p layout, or ``None`` when it carries no mesh."""
    if layout is None:
        return None
    mesh = getattr(layout, "mesh", None) or getattr(layout, "base", None)
    n = getattr(mesh, "n", None)
    if n is None:
        return None
    if isinstance(n, (tuple, list)):
        return int(n[0]) if n else None
    return int(n)


def validate_runtime_param_domains(declared_params, params):
    """Refuse a supplied runtime param outside its declared typed domain (ADC-537 gate c / G3).

    @p declared_params maps a param name to its typed declaration (a
    :class:`pops.params.runtime.RuntimeParam` exposing ``domain`` / ``check_bind``). For each
    SUPPLIED value in @p params whose name is declared with a domain, run the domain check; a
    violation becomes a hard line naming the param, the expected domain, the received value and the
    phase (``bind``) -- the 4-part ADC-541 message. A supplied name that is declared but NOT a
    runtime param (a const) is refused too (a const is frozen at compile, not settable at bind). A
    supplied name declared by nothing is left to the artifact's own unknown-param refusal (this gate
    does not duplicate it). Returns one line per violation (empty list = ok).
    """
    lines = []
    for name in sorted(params or {}):
        decl = (declared_params or {}).get(name)
        if decl is None:
            continue  # unknown-name refusal belongs to the artifact's own param check
        check_bind = getattr(decl, "check_bind", None)
        if not callable(check_bind):
            continue  # a non-runtime declaration (const) carries no bind-time domain check
        try:
            check_bind(params[name])
        except ValueError as exc:
            lines.append(str(exc))
    return lines


def validate_bind_manifest(manifest, runtime_facts):
    """Refuse an ABI / Kokkos / MPI / layout manifest mismatch at the bind front door (gate b / G2).

    Compares the compiled MANIFEST against the loaded runtime facts (@p runtime_facts, a plain dict
    of the feature tokens the _pops build reports: ``abi_key`` / ``supports_mpi`` / ``supports_gpu``
    / ``precision`` / ``communicator``). Policy (phase-6 contract decision 5): a token the manifest
    MUST carry but does not is refused as ABI-unverifiable (fail loud, never skipped); a token BOTH
    sides report is compared and a definite mismatch is a hard line naming both sides. An
    honest-unknown token on the RUNTIME side (the facts dict does not report it) is skipped -- the
    runtime cannot adjudicate what it does not know -- which is NOT a fallback. Returns one line per
    mismatch (empty list = ok).
    """
    lines = []
    facts = runtime_facts or {}
    # ABI key: the manifest MUST carry one; a fresh artifact always does. An absent key is unverifiable.
    manifest_abi = getattr(manifest, "abi_key", None)
    if not manifest_abi:
        lines.append("the compiled manifest carries no abi_key; the artifact is ABI-unverifiable and "
                     "cannot be bound (rebuild it so its manifest stamps the ABI key)")
    else:
        _compare_abi(lines, manifest_abi, facts.get("abi_key"))
    _compare_feature(lines, "supports_mpi", getattr(manifest, "supports_mpi", None),
                     facts.get("supports_mpi"), "MPI")
    _compare_feature(lines, "supports_gpu", getattr(manifest, "supports_gpu", None),
                     facts.get("supports_gpu"), "GPU / Kokkos device")
    _compare_str(lines, "precision", getattr(manifest, "precision", None),
                 facts.get("precision"))
    _compare_communicator(lines, getattr(manifest, "communicator", None),
                          facts.get("communicator"))
    return lines


def _compare_abi(lines, manifest_abi, runtime_abi):
    """Compare the two abi keys COMPONENT-WISE (like-with-like), never as raw strings.

    The artifact key (``<headers-sha>|<cxx>|<std>``) and the runtime key (the env string with
    ``headers=`` / ``std=`` tokens) are DIFFERENT representations of the same identity: both are
    parsed (:func:`_abi_components`) and only the comparable components are adjudicated -- the
    headers signature (the real header-ABI anchor) and the normalized C++ standard. A component a
    side cannot supply is honest-unknown and skipped; the compiler path-vs-version token is never
    compared (incomparable; the headers signature covers the header ABI)."""
    if not runtime_abi:
        return  # the runtime cannot state its ABI: not adjudicable, the manifest is still stamped
    m_headers, m_std = _abi_components(manifest_abi)
    r_headers, r_std = _abi_components(runtime_abi)
    if m_headers and r_headers and m_headers != r_headers:
        lines.append("ABI mismatch: the artifact was built against headers signature %r but the "
                     "loaded pops runtime reports %r (artifact abi_key %r vs runtime %r); rebuild "
                     "the artifact against this runtime"
                     % (m_headers, r_headers, manifest_abi, runtime_abi))
        return  # the headers anchor already adjudicated; do not stack a redundant std line
    if m_std and r_std and m_std != r_std:
        lines.append("C++ standard mismatch: the artifact was built for std %s but the loaded pops "
                     "runtime reports %s (artifact abi_key %r vs runtime %r); rebuild the artifact "
                     "against this runtime" % (m_std, r_std, manifest_abi, runtime_abi))


def _compare_communicator(lines, manifest_value, runtime_value):
    """Directional communicator check: refuse only what the artifact NEEDS and the runtime LACKS.

    ``unknown`` (either side) is an honest-unknown token, skipped exactly like ``None`` -- an
    artifact that does not state its communicator is never refused on it. An artifact declaring
    ``serial`` binds under ANY runtime (a serial artifact needs no communicator the runtime could
    lack -- the more-capable-runtime direction is fine). Only an artifact that DECLARES a specific
    parallel communicator the runtime reports it cannot provide (a known, different token) is
    refused."""
    if manifest_value is None or runtime_value is None:
        return
    if _is_unknown_token(manifest_value) or _is_unknown_token(runtime_value):
        return  # honest-unknown on either side: not adjudicable, not a fallback
    needed = str(manifest_value).strip().lower()
    provided = str(runtime_value).strip().lower()
    if needed == "serial":
        return  # a serial artifact binds anywhere; a more-capable runtime is never a mismatch
    if needed != provided:
        lines.append("communicator mismatch: the artifact requires communicator=%r but the loaded "
                     "pops runtime provides %r; bind under a matching runtime or rebuild the "
                     "artifact" % (manifest_value, runtime_value))


def _compare_feature(lines, field, manifest_value, runtime_value, human):
    """Refuse a boolean feature the artifact REQUIRES but the runtime LACKS (directional).

    A refusal is only ``manifest=True`` and ``runtime=False``: the artifact needs a feature the
    loaded runtime does not provide. The reverse (a more-capable runtime than the CPU-only artifact
    uses) is NOT a mismatch -- a CPU artifact binds fine on a Kokkos/MPI-capable runtime. An
    honest-unknown (``None``) on either side is not adjudicable and skipped."""
    if manifest_value is None or runtime_value is None:
        return  # honest-unknown on either side: not adjudicable, not a fallback
    if bool(manifest_value) and not bool(runtime_value):
        lines.append("%s support mismatch: the artifact requires %s=True but the loaded pops runtime "
                     "reports %s=False; bind under a matching runtime or rebuild the artifact"
                     % (human, field, field))


def _compare_str(lines, field, manifest_value, runtime_value):
    """Refuse a definite string-token mismatch (both sides KNOWN and different).

    ``None`` and the ``unknown`` spelling are honest-unknown on either side and skipped (not
    adjudicable, not a fallback); only two known, different tokens refuse."""
    if manifest_value is None or runtime_value is None:
        return
    if _is_unknown_token(manifest_value) or _is_unknown_token(runtime_value):
        return  # honest-unknown token: skipped exactly like None
    if str(manifest_value) != str(runtime_value):
        lines.append("%s mismatch: the artifact declares %s=%r but the loaded pops runtime reports "
                     "%r; bind under a matching runtime or rebuild the artifact"
                     % (field, field, manifest_value, runtime_value))


def operator_required_aux(manifest):
    """The aux fields a lowered OPERATOR requires, unioned from the module manifest (gate a / G1).

    An aux a lowered operator reads must be supplied even when the model's ``aux_extra_names`` omits
    it. The manifest's ``aux_required`` already unions the model-declared aux; a richer manifest that
    records per-operator aux requirements (``operators`` -> ``aux``) widens the set. Returns a sorted
    list of aux names the artifact requires at bind (a superset feeding the missing-argument check).
    """
    required = set(getattr(manifest, "aux_required", None) or [])
    operators = getattr(manifest, "operators", None) or []
    for op in operators:
        if isinstance(op, dict):
            required.update(op.get("aux", []) or [])
        else:
            required.update(getattr(op, "aux", None) or [])
    return sorted(required)


def validate_operator_aux(manifest, aux, provided_named_aux=()):
    """Refuse an aux a lowered operator requires but the bind omits (ADC-537 gate a / G1).

    Unions the operator-required aux (:func:`operator_required_aux`) and refuses each name neither
    supplied via ``pops.bind(aux=...)`` nor already declared on the sim. Returns one line per missing
    required aux (empty list = ok)."""
    lines = []
    supplied = set(aux or {}) | set(provided_named_aux or ())
    for name in operator_required_aux(manifest):
        if name not in supplied:
            lines.append("aux field %r is required by a lowered operator but was not supplied; pass "
                         "pops.bind(aux={%r: <array>})" % (name, name))
    return lines


def loaded_runtime_facts():
    """The feature tokens the LOADED pops runtime reports, for the manifest bind gate (G2).

    Reads the live runtime's ABI key (``pops.abi_key()``) and its environment report
    (``runtime_environment_report()``: ``precision`` / ``communicator`` / ``mpi_compiled`` /
    ``has_kokkos`` / ``kokkos_backend``) into the plain dict :func:`validate_bind_manifest` compares
    against the manifest. A token the runtime does not know (the conservative static fallback reports
    ``mpi_compiled``/``has_kokkos`` as ``None``) stays ``None`` so the gate skips it -- the runtime
    cannot adjudicate what it does not know. Imports lazily so this module's scope stays _pops-free.
    """
    facts = {}
    try:
        from pops._bootstrap import abi_key
        facts["abi_key"] = abi_key()
    except Exception:  # noqa: BLE001 -- an unreadable abi key leaves the manifest gate to catch it
        facts["abi_key"] = None
    try:
        from pops.runtime_environment import runtime_environment_report
        env = runtime_environment_report()
    except Exception:  # noqa: BLE001 -- no env report: every runtime token is honest-unknown
        env = {}
    facts["precision"] = env.get("precision")
    facts["communicator"] = env.get("communicator")
    facts["supports_mpi"] = env.get("mpi_compiled")
    facts["supports_gpu"] = env.get("has_kokkos")
    return facts


def aggregate_bind_refusals(groups):
    """Fold the per-gate refusal lines into ONE ``ValueError`` message, or ``None`` when all pass.

    @p groups is an iterable of ``(gate_label, [lines])``; a non-empty aggregate returns the message
    string (the caller raises), listing each gate's lines under its label. Empty -> ``None``."""
    flat = []
    for label, lines in groups:
        for line in lines or []:
            flat.append("[%s] %s" % (label, line))
    if not flat:
        return None
    return ("pops.bind: %d refusal(s) before the native artifact is loaded:\n  "
            % len(flat)) + "\n  ".join(flat)


def collect_missing_arguments(args, provided_blocks, provided_params, provided_aux,
                              provided_solvers):
    """Pure core of the early bind-input check (Spec 5 sec.10); no engine call -> host-testable.

    Compare an :class:`pops.codegen.inspect_compiled.Arguments` against what an install supplies and
    return one actionable line per MISSING required argument (empty list when everything required is
    met). Shared by ``System._install_compiled`` and ``AmrSystem._install_compiled`` so both enforce
    the SAME contract.

    Only entries whose ``required`` flag is true are enforced: an input the artifact marks optional
    (a const param, an unrequired solver -- the default Poisson field has a working default and is
    NOT flagged required by ``arguments()``) is never demanded, so a previously valid install passes
    through unchanged. ``provided_*`` are the supplied sets (block names, param names, aux names,
    solver fields); a block already added on the sim counts as provided. Each line names EXACTLY what
    is missing and the matching ``pops.bind`` keyword to supply it."""
    missing = []
    for name, spec in sorted(getattr(args, "instances", {}).items()):
        if spec.get("required") and name not in provided_blocks:
            missing.append("instance %r (a state block the program advances); supply its initial "
                           "state via pops.bind(state={%r: <array>})" % (name, name))
    for name, spec in sorted(getattr(args, "params", {}).items()):
        if spec.get("required") and name not in provided_params:
            missing.append("runtime param %r; pass pops.bind(params={%r: <value>})" % (name, name))
    for name, spec in sorted(getattr(args, "aux", {}).items()):
        if spec.get("required") and name not in provided_aux:
            missing.append("aux field %r; pass pops.bind(aux={%r: <array>})" % (name, name))
    for name, spec in sorted(getattr(args, "solvers", {}).items()):
        if spec.get("required") and name not in provided_solvers:
            missing.append("solver for field %r; pass pops.bind(solvers={%r: <Solver>})"
                           % (name, name))
    return missing


def validate_install_arguments(sim, compiled, instances, params, aux, solvers):
    """Early bind-input validation (Spec 5 sec.10) for a COMPILED install on @p sim (System OR
    AmrSystem): reject -- BEFORE any native mutation -- an install missing a REQUIRED argument the
    artifact declares, with one clear actionable error aggregating every missing input.

    Reads ``compiled.arguments()`` (the inert metadata the .so DECLARES) and confirms every argument
    marked ``required`` is supplied by this install call (@p instances / params / aux / solvers) OR
    already wired on the sim (an added block, a declared named aux). A NATIVE install
    (``compiled is None``) carries no declared arguments and is skipped; a handle whose
    ``arguments()`` is unavailable or unreadable is skipped too (conservative -- a missing check
    never breaks a working install)."""
    if compiled is None or not hasattr(compiled, "arguments"):
        return
    try:
        args = compiled.arguments()
    except Exception:  # noqa: BLE001 -- introspection must never break a valid install
        return
    provided_blocks = set(instances)
    try:
        provided_blocks |= set(sim.block_names())
    except Exception:  # noqa: BLE001 -- block_names is a convenience; absence is not a failure
        pass
    # Named aux already declared on the sim (B_z has no queryable trace, so it must come via aux=).
    provided_named_aux = set()
    for table in getattr(sim, "_aux_field_index", {}).values():
        provided_named_aux |= set(table)
    missing = collect_missing_arguments(
        args, provided_blocks, set(params), set(aux) | provided_named_aux, set(solvers))
    if missing:
        raise ValueError("pops.bind: the compiled artifact is missing required argument(s):\n  "
                         + "\n  ".join(missing))


def run_bind_gates(compiled, problem, layout, initial, params, aux):
    """Run the four ADC-537 bind refusal gates, raising ONE aggregated ``ValueError`` on any violation.

    The ``pops.bind`` front-door check (called from :mod:`pops.codegen.orchestration` BEFORE the
    native artifact is loaded). Reads the compiled artifact's MANIFEST
    (:meth:`~pops.codegen.loader.CompiledProblem.manifest` -- ghost depth / precision / abi_key /
    supports_* per the phase-6 contract) and its ``arguments()`` (the declared blocks / components),
    the live runtime facts (:func:`loaded_runtime_facts`) and the Problem's typed param declarations
    (``problem._param_declarations``), then folds the four gates. Every refusal is a HARD error with
    precise context; there is NO fallback. A degraded handle that exposes no manifest / arguments is
    left to the adapter's own native install check (this gate never fabricates one)."""
    manifest = compiled.manifest() if hasattr(compiled, "manifest") else None
    arguments = compiled.arguments() if hasattr(compiled, "arguments") else None
    if manifest is None or arguments is None:
        return  # degraded handle: the native install raises its own clear error
    declared_params = getattr(problem, "_param_declarations", None) or {}
    groups = [
        ("aux-required-by-operator", validate_operator_aux(manifest, aux)),
        ("manifest-abi", validate_bind_manifest(manifest, loaded_runtime_facts())),
        ("runtime-param-domain", validate_runtime_param_domains(declared_params, params)),
        ("initial-state", validate_initial_state(manifest, arguments, layout, initial)),
    ]
    message = aggregate_bind_refusals(groups)
    if message is not None:
        raise ValueError(message)


__all__ = ["validate_initial_state", "validate_runtime_param_domains", "validate_bind_manifest",
           "validate_operator_aux", "operator_required_aux", "loaded_runtime_facts",
           "aggregate_bind_refusals", "run_bind_gates",
           "collect_missing_arguments", "validate_install_arguments"]
