"""pops.output -- output/checkpoint surface (the general policy API is removed).

The general ``OutputPolicy`` / ``CheckpointPolicy`` (with ``HDF5`` / ``Plotfile`` formats and
``AllLevels`` / ``CoarseOnly`` / ``SelectedLevels`` level selection) was a DECORATIVE surface:
it stored a policy but had zero codegen and zero C++ runtime wiring (a multi-week build), so
``Case.output(...)`` only ever rejected at validate. Per the no-decorative-API rule it has been
REMOVED (ADC-509 tracks the general output/checkpoint runtime) rather than left as an inert reject.

The WIRED, narrower AMR-output surface lives in :mod:`pops.mesh.amr` (Spec 5 sec.8.11):
``pops.mesh.amr.AMROutput`` / ``pops.mesh.amr.CheckpointPolicy`` (and the AMR-local ``AllLevels`` /
``CoarseOnly`` / ``SelectedLevels``). Use those; this package exports no general policy.
"""

__all__ = []
