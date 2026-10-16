# Inventory provenance and current dependencies

`Increment.apply` and `Increment.pack` construct ordinary finite PoPS graphs.
The compiler automatically checks their relationship at `Candidate`/`Trial`
freeze, again after graph reloading, and before lowering the provenance alias.
The native operation is the same numerical binding; no physical formula is
implemented a second time in C++.

```python
from pops.contributions import Contribution, Delivery, Increment, FiniteInterval, LinearMap
from pops.graph_functions import graph
from pops.time.variant_types import Product, type_of

q = before["q"]                 # exact projection of continuation_read(base)
window = FiniteInterval(before["time"], before["time"] + h, "physical")

@graph(inputs=Product(value=type_of(q)), output=type_of(q))
def identity(p, a):
    return a["value"]

B = LinearMap.differential(identity, at=p.product(value=q), wrt="value")
J = Contribution("exchange", integrated_value, window)
ledger = Increment((J,), (Delivery(J, q, B),))
q1 = ledger.apply(p, q, name="updated_inventory")
packet = ledger.pack(p)
# Candidate(... continuation=p.product(q=q1, ...), increment=packet, ...)
```

Replacing `J` constructs a new ledger. A candidate retaining the old application
with the new packet is rejected without requiring a voluntary
`require_application` call. A complete old branch remains valid: replacement is
functional, not mutation of every old value. Old stages can still appear in
predictors, observations and histories. An application hidden only in those
places cannot certify the retained recipient path.

References retain their exact Program, recipient, path, correlated branch and
native loop invocation. Two instances of the same graph or two equal-shaped
recipients are not interchangeable. Products can transport or reconstruct an
entire packet; changing its numerical coordinates, dropping a contribution or
renaming coordinates does not silently erase its declared provenance.

The proof establishes provenance of the actual graph, not symbolic equivalence
or physical conservation for an arbitrary equation. The declared linear action
is the actual graph differential. Reassociation or another algorithm must
construct its own valid graph and proof. Plain Products supplied as increments
remain `uncertified_user_product`; their field names confer no ledger authority.
Consumed accepted child increments can be composed into a new plain Product,
which likewise makes no automatic claim to be a newly verified ledger.

## Nonlinear accumulation

When the stored variable is q but the inventory is Q(q), define Q once as a
closed `GraphFunction`. `checked_inventory_decode` reuses that same definition
for the old inventory and the proposed reconstructed value. It checks the
original residual, requested accuracy, inventory domain, selected branch and
budget. The decoder supplies a candidate value, never a trusted residual.

```python
from pops.inventory_decode import checked_inventory_decode
from pops.ledger_provenance import InventoryBinding

checked = p.require_success(checked_inventory_decode(
    p, recipient=Q0, application=Q1, base_value=before["q"],
    encode=Q, decode=reconstruct, inventory_domain=inventory_domain,
    admissible=admissible, selection=selection,
    before_data=old_data, after_data=new_data, budget=budget,
    atol=absolute_tolerance, rtol=relative_tolerance))
residual = Contribution("conversion_residual", checked["defect"], window, kind="defect")
final_ledger = Increment((*ledger.contributions, residual),
    (*ledger.deliveries, Delivery(residual, Q0, identity_action)))
packet = final_ledger.pack(p, bindings=(
    InventoryBinding(Q0, ("q",), conversion=checked["value"]),))
# Retain checked["value"] at q. Do not apply residual again.
```

The defect is exactly `Q(q_checked) - Q1`. It is declared once with identity
delivery to the same recipient, even when numerically zero. With positive
tolerance the defect stays in the ledger; it is not silently discarded or
applied twice. Each recipient can have its own reconstruction and defect.
Examples with `Q(q)=q+q*q` are in `examples/api_v02/inventory_publication_v15.py`.

## Declared current derived values

A retained z=f(q) can be declared explicitly. This requirement belongs to the
Continuation schema and cannot be omitted by a later candidate.

```python
from pops.current_dependencies import CurrentDependency

current_z = CurrentDependency(f, target=("z",), inputs={"value": ("q",)})
schema = ContinuationSchema(initial, time=("time",), step=("step",),
                            current_dependencies=(current_z,))
z1 = current_z.evaluate(p, current=p.product(q=q1))
# Retain q1 and z1 together. Retaining f(q_old) as current z is refused.
```

The compiler does not infer this scientific relation from a name or a role.
Undeclared z is user data and proves no current dependency. Explicit old-stage
observations remain legal. The schema digest includes these declarations and
scientific types; retaining an old identity after stripping declarations is
rejected. Editing all equations and re-authoring a new schema remains an
explicit model change, not a forged certificate of the old model.

## Scope

This contract applies to the finite Scalar/Product/Array inventory route with
explicit `FiniteInterval` endpoints. Existing Field/Interval adapters retain
their separate authority rules and are not silently promoted to this contract.
Backend capacities and the existing bounded control-flow provenance analysis
still apply and fail closed if the association cannot be established. No native
execution or scientific model acceptance follows merely from source validation.
JVP/VJP preserve the exact checked primal binding; tangents carry the numerical
derivative and do not inherit the primal ledger certificate.
