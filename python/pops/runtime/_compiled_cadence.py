"""Private compiled-Program cadence record.

This is a runtime install detail, not part of the public ``pops.time`` language.
Ready-made time schemes belong in ``pops.lib.time``; a user-authored
``pops.time.Program`` lowers to C++ and the runtime may carry this small record
while binding the compiled shared object.
"""


class CompiledProgramCadence:
    """Record the macro-step cadence applied around an installed compiled Program."""

    def __init__(self, substeps=1, stride=1, cfl="default"):
        if not isinstance(substeps, int) or substeps < 1:
            raise ValueError(
                "CompiledProgramCadence: substeps must be a positive int "
                "(got %r)" % (substeps,))
        if not isinstance(stride, int) or stride < 1:
            raise ValueError(
                "CompiledProgramCadence: stride must be a positive int "
                "(got %r)" % (stride,))
        if cfl != "default" and cfl != "program" and not isinstance(cfl, (int, float)):
            raise ValueError(
                "CompiledProgramCadence: cfl must be 'default', 'program', "
                "or a numeric value; got %r" % (cfl,))
        self.substeps = substeps
        self.stride = stride
        self.cfl = cfl
        self.kind = "compiled"

    def __repr__(self):
        return "CompiledProgramCadence(substeps=%d, stride=%d, cfl=%r)" % (
            self.substeps, self.stride, self.cfl)
