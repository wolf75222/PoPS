"""pops.codegen.compiled_artifact -- the public inspectable handle protocol (ADC-523).

``pops.compile(...)`` returns a compiled handle. Its concrete runtime-coupled class
(``pops.codegen.loader.CompiledProblem``) is an INTERNAL detail: users never import or
construct it. This module names the STRUCTURAL surface that handle promises -- an on-disk
``.so`` path plus the inert ``inspect()`` / ``requirements()`` reports -- as a
:class:`typing.Protocol`, re-exported at ``pops.CompiledArtifact`` for type annotations.

Annotating with ``CompiledArtifact`` (not ``CompiledProblem``) keeps the front door narrow:
callers depend on the inspectable contract, not on the concrete loader class that also carries
the bind / install machinery.
"""
from typing import Protocol, runtime_checkable


@runtime_checkable
class CompiledArtifact(Protocol):
    """Structural type of the handle returned by :func:`pops.compile`.

    A handle is a ``CompiledArtifact`` when it carries an on-disk ``.so`` path and can produce
    the inert compile-time reports. The concrete class (``pops.codegen.CompiledProblem``) satisfies
    it; this Protocol is what public code should annotate against, so the runtime-coupled loader
    class stays off the public surface.
    """

    @property
    def so_path(self) -> str:
        """Path to the compiled ``.so`` artifact on disk."""
        ...

    def inspect(self):
        """A printable :class:`pops.codegen.inspect_report.CompiledReport` of this artifact."""
        ...

    def requirements(self):
        """The compile-time :class:`pops.codegen.inspect_report.RequirementsReport`."""
        ...

    def manifest(self):
        """The rich :class:`pops.external.CompiledArtifactManifest` self-description (ADC-536).

        The ABI identity, blocks / variables / roles, required aux, const / runtime params, per-block
        ghost depth, field outputs and ``supports_*`` capability flags -- what a runtime needs to bind
        the ``.so`` safely. Metadata-only (no bind / dlopen)."""
        ...

    def arguments(self):
        """The runtime inputs the artifact expects at bind (ADC-536).

        An :class:`pops.codegen.inspect_compiled.Arguments` listing the instances / params / aux /
        solvers / outputs and runtime layout to SUPPLY at :func:`pops.bind`. Metadata-only."""
        ...

    def capability_matrix(self):
        """The ADC-549 native route matrix for this artifact (ADC-536).

        The route support columns (feature / layout / backend / platform / MPI / GPU / status /
        limitation) built from the carried manifest. Metadata-only (no dlopen / bind)."""
        ...


__all__ = ["CompiledArtifact"]
