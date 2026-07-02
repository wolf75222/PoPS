"""Typed FieldContext for a Program field solve (ADC-588).

Today ``P.solve_fields(...)`` "returns a FieldContext" only in prose: the IR node is a plain
``Value`` and the downstream RHS reads the shared aux by convention. This module makes the
FieldContext a real, inert descriptor attached to that value so a stage's field solve is
identifiable and a cross-stage / cross-block read fails loud instead of silently reading a stale
solve. It mirrors the C++ ``pops::FieldContext`` (include/pops/runtime/context/field_context.hpp):
a validity token, not a container -- it holds no field data and changes no numerics.
"""

# The default single field problem's name (the shared Poisson coupling). A named elliptic field
# uses its own name; ``None`` here means "the default phi solve". Kept as the reserved sentinel the
# operator-first lowering already uses (program_core._lower_call: fields_from_state -> default).
DEFAULT_FIELD_PROBLEM = "phi"


class FieldContext:
    """Provenance + validity token for one field solve.

    Attributes:
        field_problem: the field problem's name (``"phi"`` for the default shared Poisson, or a
            named elliptic field). Never ``None`` -- the default resolves to
            :data:`DEFAULT_FIELD_PROBLEM` so a report always names a problem.
        block: the owning block name (the block whose state was solved from).
        stage_source: a stable identifier of the stage state this solve consumed (the input
            State value's id). Two solves of the same problem/block from DIFFERENT stage states
            are distinct contexts; reading one where the other is expected is a stage mismatch.
        outputs: the ordered output handle names this solve produces (``("phi", "grad_x",
            "grad_y")`` for the default), for reports / structured-output lookup.
    """

    __slots__ = ("field_problem", "block", "stage_source", "outputs")

    def __init__(self, field_problem, block, stage_source, outputs=()):
        self.field_problem = field_problem or DEFAULT_FIELD_PROBLEM
        self.block = block
        self.stage_source = stage_source
        self.outputs = tuple(outputs)

    def matches(self, field_problem, block, stage_source):
        """True when this context was produced by exactly the requested triple.

        A ``None`` ``field_problem`` matches any problem (the default single-field case), mirroring
        the negative-``req_field`` rule of the C++ ``FieldContext::matches``.
        """
        return ((field_problem is None or self.field_problem == field_problem)
                and self.block == block and self.stage_source == stage_source)

    def require_read(self, field_problem, block, stage_source):
        """Assert a downstream read targets THIS solve, else raise a structured error naming the
        field problem, the block and the stage that mismatched (the ADC-588 incompatible-context
        contract). Returns ``self`` so it composes in an expression.
        """
        if not self.matches(field_problem, block, stage_source):
            raise ValueError(
                "incompatible field context: output of field problem %r solved for block %r "
                "(stage source %r) cannot be read as problem %r / block %r / stage source %r"
                % (self.field_problem, self.block, self.stage_source,
                   field_problem, block, stage_source))
        return self

    def output(self, handle):
        """Resolve an output handle, raising a structured error listing the known outputs when the
        handle is unknown (never a silent miss). The default problem exposes ``phi`` /
        ``grad_x`` / ``grad_y``; a named field exposes the handles its problem declared.
        """
        if handle in self.outputs:
            return handle
        raise KeyError(
            "unknown field output %r of problem %r; known outputs: %s"
            % (handle, self.field_problem, list(self.outputs)))

    def __repr__(self):
        return ("FieldContext(field_problem=%r, block=%r, stage_source=%r, outputs=%r)"
                % (self.field_problem, self.block, self.stage_source, self.outputs))
