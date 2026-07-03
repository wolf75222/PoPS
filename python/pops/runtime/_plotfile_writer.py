"""AMReX-style plotfile writer (addendum C.1, ADC-542).

Writes an AMReX plotfile directory: a ``Header`` (time, level count, domains, refinement ratios) plus
per-level ``Level_k/`` directories carrying the cell data and the box index files. Pure host
formatting on rank 0 -- no native symbol is fabricated (output is a runtime concern, not a kernel
concern). On a Uniform System the plotfile is the single-level (level 0) plotfile; on an AMR hierarchy
it carries the selected levels. The former Plotfile refusals are DELETED: both layouts write.

This writer uses the block density fields + the fine-patch footprints the engine already exposes; it
is a faithful, reader-parseable plotfile Header + per-level data, not a re-derivation of the state.
"""


def write_plotfile(sim, prefix, step=None, levels=None):
    """Write an AMReX plotfile under @p prefix (a directory). Returns the directory path.

    @p levels is the resolved level index list (from the AMR output driver); ``None`` -> all levels a
    Uniform / AMR ``sim`` reports (``n_levels`` or 1). @p step numbers the plotfile directory.
    """
    import os
    import numpy as np

    n_levels = int(sim.n_levels()) if hasattr(sim, "n_levels") else 1
    if levels is None:
        levels = list(range(n_levels))
    n = int(sim.nx())
    L = float(getattr(sim, "_L", None) if getattr(sim, "_L", None) is not None else n)
    t = float(sim.time())
    suffix = ("_%06d" % int(step)) if step is not None else ""
    root = prefix + suffix
    os.makedirs(root, exist_ok=True)

    # Block names (parity with the AMR writer): each block's coarse density is one plotfile variable.
    names = list(sim.block_names()) if hasattr(sim, "block_names") else []
    if not names:
        names = ["block"]

    # AMReX plotfile Header (v1 layout): version, n_vars, var names, dim, time, finest level,
    # domain lo/hi, ref ratios, per-level geometry, then the box lists.
    header = ["HyperCLaw-V1.1", str(len(names))]
    header += ["%s_density" % (b if b else "block") for b in names]
    header.append("2")               # spatial dimension
    header.append("%.17g" % t)       # time
    header.append(str(len(levels) - 1))  # finest level index
    header.append("0 0")             # domain lo (physical)
    header.append("%.17g %.17g" % (L, L))  # domain hi (physical)
    header.append(" ".join("2" for _ in levels[1:]))  # ref ratios (2 per finer level)
    for k in levels:
        cells = n << k
        header.append("((0,0) (%d,%d) (0,0))" % (cells - 1, cells - 1))
    header.append(" ".join("0" for _ in levels))  # step numbers per level (uniform macro clock)
    for k in levels:
        dx = (L / n) / float(1 << k)
        header.append("%.17g %.17g" % (dx, dx))

    with open(os.path.join(root, "Header"), "w") as f:
        f.write("\n".join(header) + "\n")

    # Per-level data directories. Level 0 carries the coarse block densities + phi; finer levels carry
    # their patch footprints (the box index files), the multi-resolution cell data staged as npz next
    # to the level Header so a reader picks up the refined region.
    for k in levels:
        ldir = os.path.join(root, "Level_%d" % k)
        os.makedirs(ldir, exist_ok=True)
        cells = n << k
        boxes = [(ilo, jlo, ihi, jhi)
                 for (lvl, ilo, jlo, ihi, jhi) in (sim.patch_boxes() if hasattr(sim, "patch_boxes")
                                                   else [])
                 if lvl == k]
        if k == 0:
            boxes = [(0, 0, cells - 1, cells - 1)]
        lhdr = ["%d" % len(boxes), "%.17g" % t]
        for (ilo, jlo, ihi, jhi) in boxes:
            lhdr.append("((%d,%d) (%d,%d) (0,0))" % (ilo, jlo, ihi, jhi))
        with open(os.path.join(ldir, "Level_Header"), "w") as f:
            f.write("\n".join(lhdr) + "\n")
        # Per-level cell payload. An AMR sim carries per-level accessors (the same ones the v3
        # checkpoint gathers): EVERY selected level emits its full per-block state + shared phi. A
        # Uniform sim (no per-level accessors) writes its single-level fields at level 0.
        out = {"t": t, "n": cells}
        if hasattr(sim, "level_potential"):
            multi = sim.n_blocks() != 1
            for b in names:
                st = (sim.block_level_state(b, k) if multi else sim.level_state(k))
                out["state_%s" % (b if b else "block")] = np.asarray(st, dtype=np.float64)
            out["phi"] = np.asarray(sim.level_potential(k), dtype=np.float64)
        elif k == 0:
            for b in names:
                key = b if b else "block"
                out["density_" + key] = np.asarray(sim.density(b) if b else sim.density(),
                                                    dtype=np.float64)
            if hasattr(sim, "potential"):
                out["phi"] = np.asarray(sim.potential(), dtype=np.float64)
        with open(os.path.join(ldir, "Cell.npz"), "wb") as f:
            np.savez_compressed(f, **out)
    return root


__all__ = ["write_plotfile"]
