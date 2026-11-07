"""Independent periodic tag/cell geometry oracle; no PoPS or derived config."""
import math
import numpy as np


def physical_tags(shape):
    nx, ny = shape
    # Exact Cartesian conservative cell average of the declared cosine.
    marker = np.sinc(1/nx)*np.cos(2*math.pi*(np.arange(nx)+.5)/nx)
    return np.broadcast_to(marker[None, :] > .7, (ny, nx)).copy()


def periodic_dilation(tags, radius):
    result = np.zeros_like(tags)
    for dy in range(-radius, radius+1):
        for dx in range(-radius, radius+1):
            result |= np.roll(np.roll(tags, dy, axis=0), dx, axis=1)
    return result


def fine_coverage(boxes, shape):
    nx, ny = shape
    mask = np.zeros((2*ny, 2*nx), dtype=bool)
    for level, lower, upper in boxes:
        if level != 1:
            continue
        xlo, ylo = lower; xhi, yhi = upper
        if not (0 <= xlo <= xhi < 2*nx and 0 <= ylo <= yhi < 2*ny):
            raise ValueError("native fine patch outside declared periodic cell geometry")
        region = mask[ylo:yhi+1, xlo:xhi+1]
        if region.any():
            raise ValueError("duplicate native fine patch ownership")
        region[:] = True
    return mask


def receive(boxes, shape, tag_buffer):
    expected = periodic_dilation(physical_tags(shape), tag_buffer)
    fine = fine_coverage(boxes, shape)
    target = np.repeat(np.repeat(expected, 2, axis=0), 2, axis=1)
    if not np.array_equal(fine, target):
        raise ValueError("native fine coverage differs from authored periodic tag selection")
    # This witness is rectangular per periodic strip, aligned at ratio2, minbox1,
    # efficiency1. Exact coverage is not asserted for arbitrary sparse clusters.
    if not expected.any() or expected.all():
        raise ValueError("witness must retain both refined and coarse active cells")
    return ~expected, fine
