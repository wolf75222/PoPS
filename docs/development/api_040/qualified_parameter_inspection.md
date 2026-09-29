# Qualified parameter inspection correction

The installed `4bb0639` reception compiled both seven-unknown product variants,
then failed before bind because inspection merged three block-local `gain`
declarations and rejected their different defaults. The owners were distinct
and the compiled BindSchema already carried their qualified identities.

Public artifact argument and memory reports now use that immutable BindSchema
directly. Each qualified slot retains its own default, type, domain, kind and
handle. Model-local metadata is not a second parameter authority. Low-level
component inspection without a BindSchema retains the legacy conflict check.
No parameter value, equation or native parameter route is changed.

This repairs the existing qualified-parameter contract. The argument report's
qualified keys and manifest format are unchanged; no semantic IR, package ABI
or checkpoint version changes. Tests exercise different homonymous defaults,
both block orders, argument and manifest construction, aggregate memory
metadata, and continued rejection of an ambiguous low-level declaration merge.
The native product/rebind fixture remains the end-to-end acceptance test.
