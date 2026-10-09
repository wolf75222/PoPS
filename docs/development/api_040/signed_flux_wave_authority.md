# Signed flux wave authority @2

`FluxWaveLaw(output_state, values, signed_bounds=None)` remains byte-compatible in its canonical @1 projection when `signed_bounds` is absent. A supplied axis mapping of exact `(lower, upper)` symbolic pairs selects `pops.flux-wave-law@2`. Its axes must equal the authored spectrum and the exact flux/output State; component arity still belongs to the spectrum, not to the two bounding expressions. The pair is a physical declaration bounding the spectrum, not a claim inferred from a model name or a pressure variable. General symbolic inequalities cannot be certified by this constructor.

For a diagonal two-component transport one may declare the spectrum `(a,b)` and pairs `(Minimum(a,b), Maximum(a,b))` with the existing pointwise symbolic operations. Insertion order of Cartesian axis keys is canonicalized. The left/right pair order remains mathematical and is never reordered by the compiler. Every bound expression contributes to declaration-reference authentication and the conditional manifest codec. A selected native flux view requires one unambiguous pair authority across all its contributing fluxes; mixed absent/present or different laws fail closed. Independently selected States remain independent. A global spectrum is not borrowed as a signed pair from a different State.

The lowering passes these pairs to the existing native-emitter `Model.wave_speeds` API. Compiled metadata therefore derives `explicit_pair` from the actual emitted source, and the ordinary HLL capability guard remains unchanged. No native ABI or Riemann method changes. Existing spectra continue to drive the same maximum-wave-speed calculation. The new signed pairs undergo normal expression lowering/CSE; no Jacobian, eigenvalue approximation, zero clipping or numerical tolerance is substituted. Model codegen/source identity changes and requires a new official installed build; old model DSOs cannot qualify this code.

The real VP job733028 compiled all components and then failed at the bind HLL guard, before any Native step: the active transport had eigenvalues but no emitted signed pair. Its spectator and reduced-density models were already authenticated StateStorage and were not the rejected flux. VP now explicitly declares the exact scalar pairs `(charge*electric, charge*electric)` and `(velocity, velocity)` from the same expressions as its original flux and spectrum. HLL, SSPRK2, force, meshes, quadrature, oracle and `2e-11` threshold remain unchanged. This fixes missing operator-first library/lowering expressivity; it does not weaken the runtime guard.

Source/Host tests include a distinct nonkinetic two-component transport, reversed axis insertion, negative/positive/zero characteristic speeds, legacy refusal, codec/roundtrip negatives, all three actual VP resolved models and the genuine emitted C++ translation unit syntax against read-only public headers. A real CompiledModel object is used only as inert Source metadata for the actual wave-capability guard; no Native binary or engine is simulated or qualified. Native execution remains ROOT's next rebuilt-SDK reception.

The initial detached @2 decoder checked only DAG envelopes and roots; the independent
946 counterexample replaced a signed-bound node with an unknown opcode and was accepted.
The follow-up uses the common structural Expr DAG validator, without reconstructing
or evaluating expressions. It checks every descriptor, its arity, backward child IDs,
exact integer indices, canonical scalar payloads and complete root reachability. Unknown
structural extension descriptors require a corresponding common schema before detached
admission. This validation does not certify symbolic bound inequalities or foreign
owner authority by itself; existing declaration resolution remains responsible for owners.
Legacy @1 wire output and its admission behavior are preserved. No Native bypass was
observed in the detached-decoder counterexample.
