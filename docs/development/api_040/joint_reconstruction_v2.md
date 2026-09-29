# Joint source reconstruction contract v2

Status: implemented source contract; native package reception is separate.
The scalar `pops.generated.reconstruction.stencil/v1` contract is preserved.

## Public construction and authority

```python
recipe = reconstruction.User(
    lambda sample: (
        sample(0)[0] + alpha * (sample(1, V)[2] - sample(-1, V)[0]),
        sample(0)[1] + alpha * (sample(1, V)[1] - sample(-1)[0]),
    ),
    state=U, sampling=(V,), formal_order=1,
)
```

`U` and every sampling entry are exact StateHandles, with no duplicate entry.
`sample(offset)` returns the fixed-width `VectorExpr` of U;
`sample(offset, V)` returns V in V's declared component order. The return value
is a tuple/VectorExpr of exactly `len(U)` scalar common-IR expressions. Python
runs once at construction, never inside a cell/face loop. Omitting `state`
retains the original scalar-per-component API and its v1 identity.

The sample input packing is `(U, *sampling)`, each in declaration order. It is
not the final storage packing. After Case resolution each sampled handle is
matched exactly to the principal group's resolved states, and an integer mapping
translates its components to the native group's packing. A foreign Model or an
ambiguous unqualified instance is not selected by name. Source capabilities are
retained and authenticated before their normal Case qualification; the frozen
resolved numerical descriptor includes the canonical handles. The source-body
digest excludes opaque authoring IDs. It includes ordered local state names,
component counts, expressions, active sample reads, order, and runtime captures.

Each finite-volume row owns its output and its RuntimeParams context. The body
can read every declared row input, including other states; scalar and vector rows
may coexist. The group reconstructs a complete state and evaluates the existing
shared conservative face flux. Runtime captures are inferred from all output
roots, authenticated against their model and, when explicitly qualified, their
exact block instance. Equal display names do not select a parameter carrier.

## Native evaluation

`pops.generated.reconstruction.joint-stencil/v2` emits a policy with
`stencil_face_state(sample) -> std::array<Real,n_components>` and fixed positive
`formal_order`, `n_ghost`, `stencil_min_offset`, `stencil_max_offset`. The existing
`StencilReconstruction` capability accepts exactly one scalar or joint protocol.
An ambiguous policy implementing both is rejected by `ReconstructionPolicy`.

Native joint samples use `(offset, component)`. The existing compile-time axis
and orientation displace each source-cell index exactly once. A principal wrapper
calls each vector row once, sets that row's parameter carrier, maps its inputs and
assembles its outputs. A scalar row retains its componentwise invocation. Arrays
and mappings have compile-time sizes; there is no per-cell heap allocation.

The halo is inferred from reads that remain in the expression, including both
possible branches: `max(1, 1-min_offset, max_offset+1)`. The complete group uses
the maximum row halo and the union envelope. Signed int32 offset/depth/span bounds
express the native representation; there is no 16-cell or model-specific limit.

Primitive variables use the existing common physical conversion. A selected
offset is converted once and cached across all components and rows. Inactive
`where` branches do not convert their samples. A failed active conversion cannot
be masked by a later branch or minimum: its native status remains failed. The
joint output is converted back through the same physical inverse. Common CSE
guards every evaluated intermediate/output; active nonfinite arithmetic produces
nonfinite traces, consumed by the existing native face status and collective
rejection path. No new publication authority or local success bypass is added.

## Guarantees and limits

This opens vector and cross-state mathematics; it supplies no automatic vector
TVD, positivity, realizability, consistency, formal-order proof, eigenbasis, or
stability theorem. `formal_order` remains an author assertion. Existing composed
stability guards and numerical flux authority remain in force. The body can
implement characteristic algebra with common scalar/vector operations, but the
compiler does not choose that algebra by model name.

Native calls without a fallible status contract remain rejected. Samples are
axis-oriented integer stencil offsets, not arbitrary multidimensional offsets.
The first profile returns one vector row per authored body; the complete principal
state is assembled from these rows with distinct capture contexts. Scalar Python
captures become literals; typed RuntimeParam captures remain bind/rebind reads.
This decision does not claim native MPI, GPU or full C11/C12 qualification before
reception of the exact rebuilt headers and package.
