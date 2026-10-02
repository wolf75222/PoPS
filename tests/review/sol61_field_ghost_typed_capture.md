# Typed provenance capture correction@4 and saved reader@3

Actual SDK14 Serial raw profile@3 Native fixture passed but exported inferred_boundary_expression as a Python mappingproxy repr string: shallow dict(signature) plus json.dumps(default=str) lost typed JSON. Root28 external pins remain unchanged. No reader may receive/requalify that raw by eval, AST reconstruction, repr decoding or replacing provenance. Experimental uncommitted parser was removed before this freeze. Explicit typed-object refusal replaces accidental TypeError for new reader; old readers are retained.

The public ComponentManifest API is to_data(), not ModuleManifest.to_dict(). Genuine Source resolve test executes this exact serializer, verifies nested inferred signature remains dict/list JSON, and demonstrates the old shallow/default=str route loses it. Fixture capture now uses to_data()['signature'] and dumps without default=str, allow_nan=False. Qualification profile advances public@3→public@4; new saved reader@3 accepts typed signature objects only and refuses duplicate JSON keys/nonfinite/malformed values. Source guards/equations/candidate point/consumed xmin/OriginalF1e-10 are unchanged.

A second actual-format mismatch was isolated: public block_level_state_global exports flat component-major float64 lengths128/512, while fields and masks are8x8/16x16. New profile normalization requires precisely these flat shapes and reshapes to2xnynx before exact bit comparison with carrier valid cells. It does not infer dimensions or accept other array layouts. Field cache remains bits-only across restart, candidate residuals are initial/accepted only.

No raw files altered, no Native/JIT/build/ENV/Main mutation, no scientific reception. External refusal report sol61-sdk14-initial-field-ghost-lossy-capture-refusal.json SHA9ad7046538da3e310951d7bee7ff31adee429c60a54e9fe0220910bab4be59f7 rehashes28 originals, root_scientific_approval=false, no PoPS import. Full reader refuses old contract; explicit decoder additionally refuses its actual repr. New authentic typed capture is ROOT-owned and pending. Signature semantic identity is pinned externally with genuine component authority; this correction does not reconstruct manifest/DSO graphs from metadata labels.

Coherent Source/offline validation:39 passed6.56s zeroSkip; actual public ComponentManifest.to_data positive, lossy output/type/schema/tuple/duplicate/nonfinite/malformed negatives, exact flat layout counterexamples and existing math/point/strip regressions.


## Writer-axis contract correction

Actual typed@4 receive exposed that writer boxes are half-open NumPy(y,x), while POPSCAR1 is native closed(x,y). Existing output_geometry_binding.hpp explicitly uses native_axis=Dim-1-array_axis and hi+1, preserving BoxArray order. The reader now uses exactly [ylo,xlo,yhi+1,xhi+1], without sorting or guessing a transpose from data. Rectangular Source geometry and coordinate-dependent component-major bit probes distinguish this from square/uniform cases. New reader@4 preserves reader@3 bytes and corrects its implementation of the existing writer-axis contract, not physical acceptance or capture format. Old lossy capture remains refused; typed28 originals remain unmodified. Actual typed initial/accepted candidate F and exact restart checks pass offline after this correction, no PoPS import/Native rerun. ROOT approval remains false pending independent review/receipt.

Exact final reader@4 Source/offline cohort:41 passed7.35s zeroSkip. Reader@3 historical source bytes preserved.
