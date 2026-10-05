# AMR scalar mapped consumed output

The conditional IR24 graph now has an AMR realization, authenticated as scalar-field-endpoint@1. ABI10 adds the scalar geometry capacity entry and mapped_consumed_field_output_amr capability. ABI9 remains sufficient for the existing Uniform realization.

Python admission bounds these existing endpoint contracts by the generated
`NATIVE_ABI_VERSION`: Uniform accepts ABI9 through the published release and AMR
accepts ABI10 through that release. ABI11 includes accepted-state DTO and physical
evaluation-point layout changes and publishes Uniform payload9 and AMR payload12,
retaining `mapped-consumed-output@1` and `scalar-field-endpoint@1`; their explicit
Native capabilities remain required. This endpoint admission does not permit
mixing binary artifacts across ABI/header identities: the compiler and loader's
existing exact ABI key and shipped-header checks still apply.
An unknown newer ABI, a noninteger ABI, or a missing/false/nonboolean capability
is refused. Compile checks the AMR capability for `target="amr_system"` before
provider source emission; bind checks the exact serialized layout Programs
before installation. This compatibility admission qualifies neither a Native
artifact nor numerical execution.

Private scalar endpoints have width one independently of State widths. Geometry and active hierarchy coverage determine capacity before allocation. Actual Program source/destination fields must match layout, distribution, rank, hierarchy and attempt generation; accepted State buffers cannot serve as destination candidates. Scalar candidates are checked finite collectively before publication. Existing composite publication, hierarchy epoch validation and transaction compensation remain in force.

The public Source witness uses two different two-level hierarchies and equation-owned Field ports. Resolve, slicing and actual AMR emission pass three checks; no native build or runtime has qualified this realization. Existing HOST and common-time execution restrictions remain explicit. This does not qualify GPU, multirate or scientific M19 coupling.
