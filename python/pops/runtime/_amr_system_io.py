"""AMR visualization and the single strict content-addressed checkpoint route."""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pops.runtime._amr_system_contract import _AmrSystem
else:
    _AmrSystem = object


class _AmrSystemIO(_AmrSystem):
    """Output / checkpoint / restart methods of AmrSystem."""

    def set_history_persistence(self, mapping: Any) -> Any:
        self._history_persistence = dict(mapping or {})
        return self

    def last_restart_report(self) -> Any:
        return getattr(self, "_last_restart_report", None)

    def write(self, path: Any, format: str = "npz", step: Any = None) -> Any:
        """Write coarse visualization fields; this output is not a restart artifact."""
        import os
        import numpy as np

        n = self._s.nx()
        suffix = ("_%06d" % int(step)) if step is not None else ""
        names = list(self._s.block_names()) or [""]
        if format == "npz":
            out = {
                "t": self._s.time(), "n": n,
                "patch_rectangles": np.array(self.patch_rectangles(), dtype=np.float64)
                if self.patch_rectangles() else np.zeros((0, 4)),
            }
            for block in names:
                key = block or "block"
                out["density_" + key] = np.asarray(
                    self.density(block) if block else self.density(), dtype=np.float64)
            out["phi"] = np.asarray(self.potential(), dtype=np.float64)
            target = path + suffix + ".npz"
            tmp = target + ".tmp"
            with open(tmp, "wb") as handle:
                np.savez_compressed(handle, **out)
            os.replace(tmp, target)
            return target
        if format == "vtk":
            target = path + suffix + ".vti"
            arrays, labels = [], []
            for block in names:
                key = block or "block"
                arrays.append(np.asarray(
                    self.density(block) if block else self.density(),
                    dtype=np.float64).reshape(n, n))
                labels.append("%s_density" % key)
            arrays.append(np.asarray(self.potential(), dtype=np.float64).reshape(n, n))
            labels.append("phi")
            lines = [
                '<?xml version="1.0"?>',
                '<VTKFile type="ImageData" version="0.1" byte_order="LittleEndian">',
                '  <ImageData WholeExtent="0 %d 0 %d 0 0" Origin="0 0 0" '
                'Spacing="%.17g %.17g 1">' % (n, n, self._L / n, self._L / n),
                '    <Piece Extent="0 %d 0 %d 0 0">' % (n, n), '      <CellData>',
            ]
            for name, array in zip(labels, arrays, strict=True):
                lines.append('        <DataArray type="Float64" Name="%s" format="ascii">' % name)
                lines.append("          " + " ".join("%.17g" % value for value in array.ravel()))
                lines.append("        </DataArray>")
            lines += ["      </CellData>", "    </Piece>", "  </ImageData>", "</VTKFile>", ""]
            tmp = target + ".tmp"
            with open(tmp, "w", encoding="utf-8") as handle:
                handle.write("\n".join(lines))
            os.replace(tmp, target)
            return target
        raise ValueError("AmrSystem.write: format must be 'npz' or 'vtk'")

    def checkpoint(self, path: Any) -> Any:
        """Write the only supported AMR checkpoint schema, for frozen or active regridding."""
        from pops.runtime._amr_checkpoint_v3 import write_v3

        return write_v3(
            self, self._s, path, self._L, self._regrid_every,
            getattr(self, "_history_persistence", None) or {})

    def restart(self, path: Any) -> Any:
        """Authenticate and restore the current AMR checkpoint schema; no historical fallback."""
        import numpy as np

        target = path if path.endswith(".npz") else path + ".npz"
        data = np.load(target, allow_pickle=False)
        from pops.runtime._checkpoint_manifest import authenticate_checkpoint_payload
        self._last_restart_identity = authenticate_checkpoint_payload(
            self, data, runtime_kind="amr")
        version = int(data["pops_amr_checkpoint_version"])
        if version != 3:
            raise ValueError(
                "restart: AMR checkpoint version %r unsupported; expected exactly 3" % version)
        from pops.runtime._amr_checkpoint_v3 import restart_v3

        self._last_restart_report = restart_v3(self._s, data, self._L)


__all__ = ["_AmrSystemIO"]
