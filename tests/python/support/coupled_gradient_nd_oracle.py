"""Independent Cartesian component-gradient oracle; NumPy only.

Axes follow the explicit cell tuple, not a native flattening convention. D/R
act on components; directional anisotropy here comes from the mesh spacings.
"""
import numpy as np


def fourier_cell_means(cells, lengths, modes, amplitude):
    cells, lengths, modes = tuple(cells), tuple(lengths), tuple(modes)
    phase = np.zeros(cells)
    average_factor = 1.
    for axis, (count, length, mode) in enumerate(zip(cells, lengths, modes, strict=True)):
        if count <= 0 or length <= 0:
            raise ValueError("positive cells and lengths required")
        shape = [1] * len(cells)
        shape[axis] = count
        phase += 2*np.pi*mode*(np.arange(count).reshape(shape)+.5)/count
        average_factor *= np.sinc(mode/count)
    return (np.asarray(amplitude).reshape((-1,)+(1,)*len(cells))
            * np.exp(1j*phase) * average_factor).real


def laplacian_symbol(cells, lengths, modes):
    return -sum(4*np.sin(np.pi*m/n)**2/(length/n)**2
                for n, length, m in zip(cells, lengths, modes, strict=True))


def rate(values, matrix, spacings):
    values = np.asarray(values)
    laplacian = np.zeros_like(values)
    for axis, h in enumerate(spacings, start=1):
        laplacian += (np.roll(values, 1, axis)-2*values+np.roll(values, -1, axis))/h**2
    return np.einsum("ij,j...->i...", matrix, laplacian)


def energy_rate(values, dissipative, spacings):
    volume = float(np.prod(spacings))
    result = 0.
    for axis, h in enumerate(spacings, start=1):
        jump = np.roll(values, -1, axis)-values
        result -= volume/h**2*np.sum(jump*np.einsum("ij,j...->i...", dissipative, jump))
    return result


def face_records(values, matrix, spacings, temporal_weight):
    """Each cell's two outward-oriented records per axis, as a mathematical oracle."""
    volume = float(np.prod(spacings))
    for cell in np.ndindex(values.shape[1:]):
        for axis, h in enumerate(spacings):
            for side in (0, 1):
                left, right = list(cell), list(cell)
                if side == 0:
                    left[axis] = (left[axis]-1) % values.shape[axis+1]
                else:
                    right[axis] = (right[axis]+1) % values.shape[axis+1]
                flux = matrix @ (values[(slice(None), *right)]-values[(slice(None), *left)]) / h
                for component, value in enumerate(flux):
                    yield {"cell": cell, "axis": axis, "side": side, "component": component,
                           "orientation": 2*side-1, "face_measure": volume/h,
                           "numerical_flux": float(value), "temporal_weight": temporal_weight}


def ledger_increment(records, shape, spacings):
    increment = np.zeros(shape)
    volume = float(np.prod(spacings))
    for row in records:
        increment[(row["component"], *row["cell"])] += (
            row["orientation"]*row["face_measure"]*row["temporal_weight"]
            * row["numerical_flux"] / volume)
    return increment
