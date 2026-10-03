# Explicit Module physical frame

`Module(name, frame=cartesian_frame)` declares the physical chart used by all of its StateSpaces. Their default frame is this declared authority; a foreign explicit frame is refused. No State or layout is selected to infer it. FieldSpace and physical support declarations remain independent.

ModuleManifest schema 12 authenticates `physical_frame` and every StateSpace frame. Modules without the option retain schema 10 (or existing global-quantity schema 11) and their previous serialization. This is Python declaration/codegen authority, without a new native ABI entry. It is separable from the future AMR scalar endpoint/ABI10.

The thermal helper now declares the existing chart on each of its three Modules; equations, units, axes, quadratures and numerical guards are unchanged. Source checks do not qualify any backend.
