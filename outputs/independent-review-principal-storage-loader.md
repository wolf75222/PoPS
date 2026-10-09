# Independent review: principal storage loader, 3d06cab

Scope: `python/pops/codegen/_compile_emit.py` on MAIN commit `3d06cab`;
read-only review and source-only checks. No native compilation by this reviewer.

The change gates `emit_user_reconstruction_policy` and
`emit_user_face_policy` only when the lowered model implementation has
`_program_only_storage_axes`. This attribute is set on the private carrier by
principal lowering. Its native package therefore supplies state storage and
halos without constructing a standalone FV policy against an incomplete row
physics model. `_emit_bricks`, model hash, ABI metadata, provider routes, and
the Program's principal helper are unchanged.

Independent source probe: build the actual `principal_amr_case((2,3),
levels=1,reverse=True)`, validate/resolve, lower both blocks, emit both
`amr_system` loaders and the Program. Both row models have the storage flag;
neither loader defines `UserReconstructionPolicy` or `UserFacePolicy`, while
each retains the compiled route manifest. The Program body is 47,386
characters and contains four `ReconstructionRow0` references, two
`NumericalRow0` references, and five `model.parameter_sets` references.
`PrincipalFiniteVolumeGroup._payload` still hashes the complete method
descriptors. `principal_lowering._authenticate_numerical_captures` still
checks owner/block identity. The helper emits one policy per row and binds
that row's parameter set; it evaluates complete physical flux and shared
stability through the packed group model.

Autonomous User routes: `test_user_numerical_capture_routing.py` plus
`test_principal_group_codegen.py` passed 10/10 in 40.65 s. The former
asserts both source identities and both User policy arguments remain in
`system` and `amr_system` loaders when no storage-only marker is present.
Root's larger source suite reported 29/29. This review found no source
regression or omitted group capture. It does not establish native acceptance;
the seven red cases from the preceding installed run must be retested against
the rebuilt 3d06cab artifact.
