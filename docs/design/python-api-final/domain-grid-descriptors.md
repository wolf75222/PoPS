# Domain, frame, and Cartesian-grid descriptors

This layer is pure Python. It defines scientific geometry before layout resolution and never loads
`pops._pops`.

```python
from pops.domain import Rectangle, RectangleBoundaryNames
from pops.frames import Cartesian2D
from pops.mesh import CartesianGrid, PeriodicAxes

domain = Rectangle(
    "unit_square",
    lower=(0.0, 0.0),
    upper=(1.0, 1.0),
    boundaries=RectangleBoundaryNames(
        x_min="inlet",
        x_max="outlet",
        y_min="bottom",
        y_max="top",
    ),
).tag("fluid")

frame = domain.frame(Cartesian2D())
x, y = frame.axes
inlet = frame.boundaries.x_min
outlet = frame.boundaries.x_max
grid = CartesianGrid(frame=frame, cells=(128, 128))
periodic_grid = CartesianGrid(
    frame=frame,
    cells=(128, 128),
    periodic=PeriodicAxes(frame.axes),
)
```

`CartesianAxis`, `DomainBoundary`, `Rectangle`, `RectangleFrame`, and `CartesianGrid` are immutable,
hashable values. Axis and boundary selection is typed: use `frame.coordinates.x`,
`frame.boundaries.x_min`, or `frame.boundaries.pair(x)`. There is deliberately no
`boundaries["x_min"]` or `pair("x")` runtime selector.

`Rectangle.frame(...)` is a binding operation, not mutation. It returns a `RectangleFrame` that
contains the rectangle and its coordinate system. This makes the two required grid inputs complete:
`CartesianGrid(frame=..., cells=...)` derives all of the following without duplicated user choices:

- axis order from the typed frame (`x`, then `y`);
- physical extent from the rectangle;
- the periodic/physical axis partition from its four stable boundaries and optional typed
  `PeriodicAxes` value;
- cell widths from extent and cell counts.

Every descriptor exposes an exact `to_dict()` / `from_dict()` round trip and a domain-separated
`canonical_id`. Decoders reject missing, extra, stale, reordered, or inconsistent derived fields.
Boundary geometry identities intentionally remain stable when semantic domain tags are added; Case
qualification happens later during resolution. Periodicity is never a backend-shaped boolean:
`PeriodicAxes` accepts only canonical typed axes, and the grid identity records both the periodic
axes and the derived physical complement. A backend unable to preserve that partition must reject
a one-axis topology rather than widening it silently.

`CartesianGrid` is the only public Cartesian-grid descriptor. APIs that consume a grid do not
accept an integer, a shape tuple, or a square-mesh compatibility object: domain, frame, extent,
cells, and topology remain explicit. The real annular backend remains available separately through
the advanced `pops.mesh.PolarMesh` descriptor; it is not a root-level shortcut and does not replace
the framed Cartesian path. Adaptive authoring is imported from `pops.amr`; the internal
The implementation package is private at `pops.mesh._amr`; `pops.mesh.amr` is not importable.

This foundation does not claim a native lowering route. A platform-specific grid component must
consume and authenticate this canonical descriptor during `resolve`/`compile`; authoring it never
runs a numerical kernel or opens a native library.
