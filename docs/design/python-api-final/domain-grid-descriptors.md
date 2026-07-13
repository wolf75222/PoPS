# Domain, frame, and Cartesian-grid descriptors

This layer is pure Python. It defines scientific geometry before layout resolution and never loads
`pops._pops`.

```python
from pops.domain import Rectangle, RectangleBoundaryNames
from pops.frames import Cartesian2D
from pops.mesh.grid import CartesianGrid

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
- the bounded topology from its four stable boundaries;
- cell widths from extent and cell counts.

Every descriptor exposes an exact `to_dict()` / `from_dict()` round trip and a domain-separated
`canonical_id`. Decoders reject missing, extra, stale, reordered, or inconsistent derived fields.
Boundary geometry identities intentionally remain stable when semantic domain tags are added; Case
qualification and periodic/physical classification happen later during resolution.

This foundation does not claim a native lowering route. A platform-specific grid component must
consume and authenticate this canonical descriptor during `resolve`/`compile`; authoring it never
runs a numerical kernel or opens a native library.
