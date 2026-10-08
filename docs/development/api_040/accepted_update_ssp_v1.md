# Exact accepted-update convex proof

`accepted-update-ssp@1` proves a temporal coefficient-one convex decomposition
from the actual typed affine expression and its contributing rate/state inputs.
It does not recognize a method, model, operator name, fingerprint, or SSA spelling.
It neither replaces an expression nor changes its emitted floating-point arithmetic.

For each independently owned accepted State, the proof reconstructs exact rational
stage rows `A`, accepted weights `b`, and evaluation coordinates `c`. Every stage
must preserve its own initial State, refer only to earlier contributing rates,
and have its actual coordinate equal the row sum. State weights are constant and
rate weights carry exactly one power of dt. The accepted window spans one step
and the rate weights sum to one. Boolean coefficient metadata is not real scalar
arithmetic. Other State commits may evolve under independent per-owner proofs or
retain their exact initial algebraic image.

Let `K` append `b` to the strictly lower-triangular explicit stage matrix. At
coefficient one, the algorithm computes the exact inverse of `I+K` and checks
nonnegativity of `v=(I+K)^-1 1` and `alpha=I-(I+K)^-1`. These identities give
`q = v U0 + alpha (q + dt F(q))`: each row is a convex combination of the initial
State and forward-Euler images. This is a constructive coefficient-one certificate,
not an optimal SSP coefficient, global method order, or constitutive-law theorem.

The certificate is conditional on the actual forward-Euler spatial premise.
Existing constitutive, transport, positivity/stability, conservation and source
guards remain unchanged. Different authored rate expressions are allowed; their
mathematical definitions and each actual evaluation remain in the generated code.
Accepted diffusive exchange weights are still derived from the authored expression.

An independent Field solve, diagnostic or history operation remains executed.
Its presence alone does not make the accepted affine expression an unknown method.
Its typed effects must not mutate live accepted-State inputs. Field feedback and
extra evaluation reads are not silently removed: absent a proved frozen read
closure, they leave the forward-Euler acceptance premise unproved. The operation
itself still has its ordinary lowering; the acceptance diagnostic identifies the
missing obligation, without selecting a recipe from an identity.

Field publication distinguishes numerical output observations from consumer-State
binding/shape authority. The exact constitutive read union must be supplied by
authenticated publication components before shape-only links can be excluded from
that mathematical dependency trace. Other prerequisite reads are not ignored.
Matrix-free Field reads retain their actual input/output binders and free captures.
Issued object identity, integer SSA identifiers, role-qualified cache keys and cycle
checks prevent detached references from supplying a frozen-input certificate.
Owned primitive recipes are expanded for read analysis; a State-only derived
expression is not treated as a Field read merely because it is a primitive.

Global `UnknownOrder` remains descriptive for a mixed executed Program. This local
certificate does not change that result. Failure to prove an acceptance obligation
is reported separately from the translated mathematical operation.

A typed terminal acceptance guard keeps its actual Bool condition and its
existing FailRun or RejectAttempt action. On the successful path its result is
an alias of the exact same type, owner, clock, point, space and storage/context
value. It is not an accepted-State write. The proof and exchange trace may follow
that primary value only after authenticating the terminal action and all issued
input references; the generated condition still executes before commit. Arbitrary
actions, non-Bool conditions, detached inputs and projection mutations cannot
supply this contract. A failed condition withdraws the attempt through the
existing runtime failure path.

The same successful-value alias rule applies when classifying an unchanged
accepted driver and when tracing a Field input. A terminal predicate remains
executed control; it is not an additional mathematical coefficient input on the
successful path. All issued condition references and their effects are still
checked before this classification. A changed donor value or mutating condition
cannot obtain the unchanged-driver premise through that rule.

Source tests cover both expanded and retained-stage public two-stage expressions,
exact exchange weights, legacy frozen-driver Euler, independent owners, different
rate expressions, renamed models/operators/stages, reordered independent observations
and their reference IDs, rational coefficient/stage perturbations, Field feedback,
detached/cyclic references and binders, Boolean metadata, and live input mutation.
The actual public one-level AMR example resolves and emits both equivalent two-stage
spellings with `b=(1/2,1/2)`. These are Source proofs; installed Native execution,
saved-state numerical reception, MPI/GPU coverage and scientific qualification
remain distinct work.


The production compiler proves constitutive read closure using the exact owner-qualified
`ProgramModelGraph` source Module already supplied to Program emission. It authenticates
the block's model owner and physical State target, then resolves the actual operator bodies
and primitive recipes from that Module. Canonical compiled Program detachment intentionally
removes live authoring registries; their absence neither implies empty reads nor selects a
representative model. A missing or foreign source authority leaves the acceptance obligation
unproved with an explicit diagnostic. The low-level authoring-only proof route may use its
still-issued registry; installed artifact publication always carries the model graph authority.
This boundary changes proof inputs only: the mathematical expressions, owner/stage contracts,
coefficient arithmetic and emitted numerical operations remain unchanged.

Consumed publication uses a shared compiler resolver for its destination space.
It authenticates the exact destination block, requires that block's model owner
to equal the declaration owner, authenticates the declaration in that source
Module, and retrieves its declared FieldSpace. Proof and emission use the same
resolver through the existing frontend validator's `target_space` callback.
Missing or foreign graph/module/declaration authorities remain explicit failures;
compiled detachment does not require reconstructing an authoring Case registry.
The stored global-field history witness uses its canonical compiled Handle.
This fixes an authoring-identity suffix in storage metadata and changes no
mathematical expression or floating-point operation.

These are corrections to the existing version-1 authority and alias semantics.
The operation meanings, wire contracts and requested coefficient-one guarantee
are unchanged. Each integrated correction receives a new core fingerprint and
fresh native qualification; earlier finite receipts do not qualify the new core.
