"""Private Program authoring and immutable-IR implementation.

The public temporal façade is :mod:`pops.time`; internal consumers import the
specific leaf module they need so this package initializer never creates a
second Program authority or an import cycle.
"""

__all__: tuple[str, ...] = ()
