# Exact AMR checkpoint image

The current AMR checkpoint is one sealed accepted-state image. It is not a collection of optional
``level_aux`` arrays. The outer checkpoint manifest authenticates every array and lifecycle identity
before restart preflight; the AMR codec then validates the complete logical image before opening the
native restart transaction.

The image preserves:

- the selected native dimension, exact-rank coarse shape, bounds, periodicity, configured depth,
  active depth, and one refinement-ratio vector per configured transition;
- level-qualified patch boxes and owner maps for every recorded rank;
- every block's component count and complete level state;
- one native ``POPSAUX1`` auxiliary image per active level, including the sealed registry contract,
  accepted generation, storage groups, owner-qualified ``ComponentKey`` values, group addresses,
  and accepted provider evaluation points;
- primary and logical clocks, rational temporal relations, synchronization evidence, conservative
  ledgers, transfer provenance, history descriptors and persisted slots;
- every rank's exact accepted Program image, including its temporal partition and persistent
  accepted state.

Scalar refinement ratios and inferred two-dimensional shapes are rejected. Auxiliary values are not
flattened: a component remains addressed by its storage-group identity and component offset, with
its semantic key and provider owner alongside it.

## Restore boundary

Python validates the sealed container, spatial specialization, hierarchy, every dense payload,
history policy, temporal contract, Program image presence, and the presence of one opaque native
auxiliary image per active level. Python does not decode ``POPSAUX1`` or define a second
``ComponentKey`` authority. ``AmrSystem::restore_auxiliary_checkpoint_accepted_state`` owns that
format and preflights the complete level set before it mutates any registry. The existing native
restart transaction remains the atomic publication boundary and rolls back topology, state,
auxiliary registries, histories, clocks, and Program state together on failure.

If Python ever needs inspectable auxiliary metadata, the required seam is a native versioned
``to_data`` binding; reconstructing native binary framing in Python is not allowed. The final
consolidation into one native ``restore_checkpoint_image(request)`` call likewise requires a new
binding on ``AmrSystem``. Until that binding exists, the prepared request is applied through the
current transaction-scoped setters; no intermediate state is externally visible.
