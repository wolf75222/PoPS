# AMR scalar mapped consumed output

The conditional IR24 graph now has an AMR realization, authenticated as scalar-field-endpoint@1. ABI10 adds the scalar geometry capacity entry and mapped_consumed_field_output_amr capability. ABI9 remains sufficient for the existing Uniform realization.

Private scalar endpoints have width one independently of State widths. Geometry and active hierarchy coverage determine capacity before allocation. Actual Program source/destination fields must match layout, distribution, rank, hierarchy and attempt generation; accepted State buffers cannot serve as destination candidates. Scalar candidates are checked finite collectively before publication. Existing composite publication, hierarchy epoch validation and transaction compensation remain in force.

The public Source witness uses two different two-level hierarchies and equation-owned Field ports. Resolve, slicing and actual AMR emission pass three checks; no native build or runtime has qualified this realization. Existing HOST and common-time execution restrictions remain explicit. This does not qualify GPU, multirate or scientific M19 coupling.
