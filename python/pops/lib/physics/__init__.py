"""pops.lib.physics -- ready-made physics-library authoring helpers (ADC-637).

These helpers author a REUSABLE piece of physics on a model (a coupled-source linearization, a
coupling) so a scheme macro can lower it generically. The first entry is the electrostatic-Lorentz
linearization the condensed-implicit solve eliminates -- the DSL spelling of the retiring hand-written
Schur brick's rotation operator ``B``.

Public API
----------
    author_electrostatic_lorentz  -- author J = [[0, B_z], [-B_z, 0]] on the momentum subset of a model.
    LORENTZ_J_NAME                 -- the canonical operator name the condensed_schur macro references.
"""

from .electrostatic_lorentz import LORENTZ_J_NAME, author_electrostatic_lorentz

__all__ = ["author_electrostatic_lorentz", "LORENTZ_J_NAME"]
