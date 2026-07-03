"""pops.problem.handles -- stable authoring handles for a Problem's parts (ADC-526).

When a user declares a block or a field on a :class:`pops.problem.Problem`, the setter returns a
STABLE handle: an inert reference (name + kind + stable id) the user can hold to inspect that part
or wire a later reference to it, without reaching into the Problem's internal registries. The id is
stable for the life of the Problem so a handle keeps pointing at the same declared part even as more
are added (mirrors the :mod:`pops.physics.board_handles` pattern for the model board).

A handle owns NO runtime data and computes nothing: it is a typed name. Two handles compare equal
when they name the same part of the same Problem (id + kind), so a handle can key a lookup.
"""


class ProblemHandle:
    """Base of the stable problem-part handles: a name, a kind and a stable id.

    ``kind`` is the family the handle points into (``block`` / ``state`` / ``field`` / ``operator``);
    ``name`` is the declared identifier; ``handle_id`` is a stable ``kind:name`` slug that does not
    change as the Problem grows. Inert -- it carries a weak reference to the owning Problem for
    :meth:`inspect` only, never the runtime.
    """

    kind = "handle"

    def __init__(self, name, *, owner=None):
        self._name = str(name)
        self._owner = owner

    @property
    def name(self):
        return self._name

    @property
    def handle_id(self):
        return "%s:%s" % (self.kind, self._name)

    def inspect(self):
        """A plain ``{kind, name, id}`` view of the handle (no build, no compile)."""
        return {"kind": self.kind, "name": self._name, "id": self.handle_id}

    def __eq__(self, other):
        return isinstance(other, ProblemHandle) and other.kind == self.kind \
            and other._name == self._name

    def __hash__(self):
        return hash((self.kind, self._name))

    def __repr__(self):
        return "%s(%r)" % (type(self).__name__, self._name)


class BlockHandle(ProblemHandle):
    """A stable reference to a declared physics block (``problem.add_block(...)`` returns one)."""

    kind = "block"

    def state(self, name):
        """A :class:`StateHandle` for a named state component of this block (inert reference)."""
        return StateHandle("%s.%s" % (self._name, name), owner=self._owner)


class StateHandle(ProblemHandle):
    """A stable reference to a block's state component (``block.state('ne')``)."""

    kind = "state"


class FieldHandle(ProblemHandle):
    """A stable reference to a declared elliptic field problem (``problem.field(...)`` returns one)."""

    kind = "field"


class OperatorHandle(ProblemHandle):
    """A stable reference to a declared coupling / local operator on the Problem (inert)."""

    kind = "operator"


__all__ = ["ProblemHandle", "BlockHandle", "StateHandle", "FieldHandle", "OperatorHandle"]
