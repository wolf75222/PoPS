"""pops.output.formats -- typed output-format descriptors (Spec 5 sec.5.14).

A format is a typed object (``HDF5()``), not a string ``format="hdf5"``. Inert; the runtime
writes the actual files. AMReX plotfile output is not exported until there is a real native
writer; exposing ``Plotfile()`` without one would violate the Spec-5 no-decorative-API rule.
"""
from pops.descriptors import Descriptor


class NPZ(Descriptor):
    """Compressed NumPy archive output, dependency-free default."""

    category = "output_format"
    native_token = "npz"


class VTK(Descriptor):
    """VTK ImageData output for visualization tools."""

    category = "output_format"
    native_token = "vtk"


class HDF5(Descriptor):
    """HDF5 output. ``parallel=True`` requests the parallel-HDF5 path (build-dependent)."""

    category = "output_format"
    native_token = "hdf5"

    def __init__(self, parallel=False):
        self.parallel = bool(parallel)

    def options(self):
        return {"parallel": self.parallel}

    def requirements(self):
        return {"parallel_io": True} if self.parallel else {}


__all__ = ["NPZ", "VTK", "HDF5"]
