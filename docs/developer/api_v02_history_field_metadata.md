# Retained Field descriptions in History

`Program(history_field_metadata_budget=(retained_bytes, workspace_bytes))` sets the storage for immutable scientific producer descriptions and representation obligations carried by History. An authored pair of positive integer capacities is part of the Program identity and takes precedence. Omitting it is the normal case: the plan installs the default capacities (8 MiB retained, 16 MiB workspace, `DEFAULT_HISTORY_FIELD_METADATA_BUDGET`) for every Program with at least one History ring, so no case writes an infrastructure budget to run (spec 0.3, B3). A Program without History rings installs none. The prelude, the AMR checkpoint exports and the exchange checkpoint bound read the same resolved pair. A description that exceeds either capacity still fails before publication.

The retained capacity counts deduplicated support/obligation objects, their string payloads, ring descriptors and slot references. The workspace capacity bounds requested temporary table storage and the simultaneously live wire images used by this codec, including the borrowed input wire on decode. These are logical capacities of this service, not a claim about process RSS or the numerical MultiFab allocation. Existing numerical/history and checkpoint budgets remain separate. No flux budget supplies metadata storage.

A slot stores its `ContinuationFieldSupport` and `FieldRepresentationObligation` with its actual numerical value. Rotation, accepted snapshots and rollback move all three together. A lookup of an exact native History slot takes precedence over an address registry, including when its metadata is empty. Restart and MPI repartition keep the producer description unchanged and separately authenticate the new live allocation. Metadata bytes alone never authorize a different field.

For a consumed independent field, declare its allocation owner explicitly:

```python
p.store_history("potential", observation, depth=2, owner_block=block)
previous = p.history("potential", lag=1)
```

The read keeps the original independent provider and FieldSpace, with no State owner. Its private authored storage declaration, maximum lag and frozen DSO tuple are verified. `owner_block` allocates storage; it does not turn the observation into that block's State or grant access to a foreign level. The same read can feed ordinary field algebra and reductions.

Exact temporal endpoint copies preserve their metadata. Nonidentity interpolation or spatial remapping of coordinates with an outstanding representation obligation is rejected before numerical transformation: interpolating raw coordinates need not commute with their authored nonlinear physical map. The complete numerical route is to discharge that map on the source, remap physical quantities using the authored Python equations, and reconstruct target coordinates. Clean Field supports receive a reconstruction origin engaging the source descriptions, exact transfer coefficients, target sample and target geometry. No current-point label replaces an old producer point.

Uniform checkpoint envelope `POPSEX06` retains the previous exchange image, exact samples and one canonical shared metadata table. AMR `POPSHFM1` wraps the existing FLX image verbatim, the metadata table and exact samples. A Program without History rings has no metadata budget, and its wire images remain unchanged. Restore validates tuple, sample identity, canonical aliases, program identity and budgets before publication; an enabled metadata contract cannot silently restore a legacy image missing its metadata.

## Qualification status

The new source tests exercise canonical budgets, detached/frozen declarations, independent reads and negative authority mutations. Native C++ cases cover metadata codec/alias rejection, joint rotation/rollback, repartition preparation and interpolation refusal. Public receivers cover the independent screened PDE with old-sample consumption, real AMR reconstruction, checkpoint/restart and parent rejection. At the introduction of cohort 017 these receivers are authored and pending execution against its installed native SDK; syntax/source checks alone are not native or scientific qualification. Execution receipts and their exact source identities belong in the reception matrix after the run.
