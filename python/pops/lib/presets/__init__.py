"""pops.lib.presets -- ready-to-run compose-and-go bundles (ADC-524, criterion 7).

A PRESET pairs a provided model (:mod:`pops.lib.models`) with a provided time scheme
(:mod:`pops.lib.time`) so a user drops both into a :class:`pops.Problem` in one step instead of
re-deriving the pairing every time. This is the ONE namespace for such bundles: ``pops.lib`` keeps
only things that are ready to use (``lib.time`` schemes, ``lib.models`` models, ``lib.presets``
bundles); the generic building blocks live in the top-level central packages (``pops.numerics`` /
``pops.solvers`` / ...).

A preset composes DESCRIPTORS ONLY -- a provided model factory and a time-scheme macro. It never
touches the runtime, the codegen driver, or a mesh layout: WHERE the bundle runs (``Uniform`` /
``AMR``) stays the user's choice at ``pops.compile(problem, layout=...)``, and the ``.so`` compile
stays ``pops.compile``'s job. That keeps ``pops.lib`` a leaf of the import graph (``lib`` may import
ir / model / time / physics / moments, never codegen or runtime).

Usage::

    import pops
    from pops.lib.presets import vlasov_poisson_magnetic_euler
    from pops.mesh.layouts import Uniform
    from pops.mesh.cartesian import CartesianMesh

    preset = vlasov_poisson_magnetic_euler()
    model = preset.model()
    problem = pops.Problem(name="plasma")
    block = problem.add_block("f", model)
    state = next(
        handle for handle in model.module.declaration_index().records()
        if handle.kind == "state"
    )
    problem.time(preset.time_scheme(block, state))
    compiled = pops.compile(problem, layout=Uniform(CartesianMesh(n=96)))
"""
from __future__ import annotations

from typing import Any


class Preset:
    """A ready-to-run pairing of a provided model factory and a provided time-scheme macro.

    ``model()`` builds the provided physics model (a ``pops.physics`` / ``pops.lib.models``
    composition ready for a Problem block); ``time_scheme(block, state)`` builds the matching
    ``pops.time.Program`` from the authoritative ``BlockHandle`` and model state ``Handle``. Both
    are DESCRIPTOR-level authoring objects the user hands to a :class:`pops.Problem`; the preset
    carries no mesh, no runtime, and no compiled ``.so``.

    Args:
        name: The preset's identifier (shown in ``repr`` and ``inspect()``).
        model_factory: A zero-argument callable returning the provided model.
        time_factory: A callable ``(block, state) -> pops.time.Program`` building the time scheme
            from typed semantic references.
        description: A one-line human summary of what the bundle solves.
    """

    category = "preset"

    def __init__(self, name: Any, model_factory: Any, time_factory: Any,
                 description: Any = "") -> None:
        self._name = name
        self._model_factory = model_factory
        self._time_factory = time_factory
        self._description = description

    @property
    def name(self) -> Any:
        return self._name

    def model(self) -> Any:
        """Build the provided model for a Case block (``.block(name, physics=preset.model())``)."""
        return self._model_factory()

    def time_scheme(self, block: Any, state: Any) -> Any:
        """Build the matching Program from a BlockHandle and model state Handle.

        Free block/state names are intentionally not accepted: the delegated ``pops.lib.time``
        builder authenticates both references against the Case registry before producing IR.
        """
        return self._time_factory(block, state)

    def inspect(self) -> dict:
        """An inert ``{name, category, description}`` view of the bundle (no build, no compile)."""
        return {"name": self._name, "category": self.category, "description": self._description}

    def __repr__(self) -> str:
        return "Preset(%r)" % self._name


def vlasov_poisson_magnetic_euler(*, order: Any = 4) -> Any:
    """The HyQMOM15 Vlasov-Poisson-magnetic model paired with a forward-Euler step.

    Composes :meth:`pops.lib.models.moments.HyQMOM15.vlasov_poisson_magnetic` (transport flux +
    Poisson coupling + Vlasov electric source + magnetic Lorentz source) with the forward-Euler macro
    (``pops.lib.time.forward_euler``), which builds a fresh, inspectable ``pops.time.Program`` from the
    typed block and state references. The user picks the layout (Uniform / AMR) on the Case; nothing
    here is mesh- or runtime-bound.
    """
    from pops.lib.models import HyQMOM15
    from pops.lib.time import forward_euler

    return Preset(
        "vlasov_poisson_magnetic_euler",
        model_factory=lambda: HyQMOM15.vlasov_poisson_magnetic(order=order),
        time_factory=lambda block, state: forward_euler(block, state),
        description="HyQMOM15 Vlasov-Poisson-magnetic with a forward-Euler time step.")


__all__ = ["Preset", "vlasov_poisson_magnetic_euler"]
