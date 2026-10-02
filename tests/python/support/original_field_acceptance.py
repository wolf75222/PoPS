"""selectedNativeMixedRule@1; diagnostic values retain their native meaning."""
import math
import struct
STOP_RULE = "selectedNativeMixedRule@1"
def selected_native_mixed_rule(diagnostics, tolerance):
    suffixes = ("residual_norm", "reference_residual_norm", "rel_residual")
    groups = {}
    for name, value in diagnostics:
        for suffix in suffixes:
            tail = "." + suffix
            if name.endswith(tail):
                group = groups.setdefault(name[:-len(tail)], {})
                if suffix in group:
                    raise ValueError("duplicate original-F diagnostic")
                if type(value) is not float or not math.isfinite(value) or value < 0:
                    raise ValueError("invalid original-F diagnostic")
                group[suffix] = value
                break
    if len(groups) != 1 or set(next(iter(groups.values()), {})) != set(suffixes):
        raise ValueError("original-F diagnostic triplet differs")
    row = next(iter(groups.values()))
    norm, reference, relative = (row[key] for key in suffixes)
    expected = norm / (reference if reference > 0 else 1.)
    if struct.pack("<d", relative) != struct.pack("<d", expected):
        raise ValueError("original-F raw relative diagnostic inconsistent")
    if type(tolerance) is not float or not math.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("invalid selected native tolerance")
    if norm > tolerance * max(1., reference):
        raise ValueError("original-F selected native mixed stopping rule fails")
    return dict(stop_rule=STOP_RULE, residual_norm=norm, reference_residual_norm=reference,
                rel_residual=relative, acceptance_ratio=norm/max(1., reference))
