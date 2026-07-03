"""pops.lib.presets -- ready-to-run compose-and-go bundles (ADC-524, criterion 7).

A PRESET pairs a provided model (:mod:`pops.lib.models`) with a provided time scheme
(:mod:`pops.lib.time`) so a user drops both into a :class:`pops.Case` in one step instead of
re-deriving the pairing every time. This is the ONE namespace for such bundles: ``pops.lib`` keeps
only things that are ready to use (``lib.time`` schemes, ``lib.models`` models, ``lib.presets``
bundles); the generic building blocks live in the top-level central packages (``pops.numerics`` /
``pops.solvers`` / ...).

A preset composes DESCRIPTORS ONLY -- a provided model factory and a time-scheme macro. It never
touches the runtime, the codegen driver, or a mesh layout: WHERE the bundle runs (``Uniform`` /
``AMR``) stays the user's choice on the Case, and the ``.so`` compile stays ``pops.compile``'s job.
That keeps ``pops.lib`` a leaf of the import graph (``lib`` may import ir / model / time / physics /
moments, never codegen or runtime).

Usage::

    import pops
    from pops.lib.presets import vlasov_poisson_magnetic_euler
    from pops.mesh.layouts import Uniform
    from pops.mesh.cartesian import CartesianMesh

    preset = vlasov_poisson_magnetic_euler()
    case = (pops.Case(layout=Uniform(CartesianMesh(n=96)), name="plasma")
            .block("f", physics=preset.model())
            .time(preset.time_scheme("f")))
    compiled = pops.compile(case)
"""


class Preset:
    """A ready-to-run pairing of a provided model factory and a provided time-scheme macro.

    ``model()`` builds the provided physics model (a ``pops.physics`` / ``pops.lib.models``
    composition ready for a Case block); ``time_scheme(block)`` builds the matching
    ``pops.time.Program`` for that block name. Both are DESCRIPTOR-level authoring objects the user
    hands to a :class:`pops.Case`; the preset carries no mesh, no runtime, and no compiled ``.so``.

    Args:
        name: The preset's identifier (shown in ``repr`` and ``inspect()``).
        model_factory: A zero-argument callable returning the provided model.
        time_factory: A callable ``block -> pops.time.Program`` building the time scheme for a block.
        description: A one-line human summary of what the bundle solves.
    """

    category = "preset"

    def __init__(self, name, model_factory, time_factory, description=""):
        self._name = name
        self._model_factory = model_factory
        self._time_factory = time_factory
        self._description = description

    @property
    def name(self):
        return self._name

    def model(self):
        """Build the provided model for a Case block (``.block(name, physics=preset.model())``)."""
        return self._model_factory()

    def time_scheme(self, block):
        """Build the matching ``pops.time.Program`` for @p block (``.time(preset.time_scheme(name))``)."""
        return self._time_factory(block)

    def inspect(self):
        """An inert ``{name, category, description}`` view of the bundle (no build, no compile)."""
        return {"name": self._name, "category": self.category, "description": self._description}

    def __repr__(self):
        return "Preset(%r)" % self._name


def vlasov_poisson_magnetic_euler(*, order=4):
    """The HyQMOM15 Vlasov-Poisson-magnetic model paired with a forward-Euler step.

    Composes :meth:`pops.lib.models.moments.HyQMOM15.vlasov_poisson_magnetic` (transport flux +
    Poisson coupling + Vlasov electric source + magnetic Lorentz source) with the forward-Euler macro
    (``pops.lib.time.forward_euler``), which builds a fresh, inspectable ``pops.time.Program`` from the
    block name alone. The user picks the layout (Uniform / AMR) on the Case; nothing here is mesh- or
    runtime-bound.
    """
    from pops.lib.models import HyQMOM15
    from pops.lib.time import forward_euler

    return Preset(
        "vlasov_poisson_magnetic_euler",
        model_factory=lambda: HyQMOM15.vlasov_poisson_magnetic(order=order),
        time_factory=lambda block: forward_euler(block),
        description="HyQMOM15 Vlasov-Poisson-magnetic with a forward-Euler time step.")


__all__ = ["Preset", "vlasov_poisson_magnetic_euler"]
