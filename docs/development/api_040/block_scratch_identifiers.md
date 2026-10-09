# Internal Block scratch identifiers

`block-scratch-identifiers@1` qualifies C++ internal scratch tokens independently of
exact public Case Block labels. The old explicit and implicit coupled lowerings
embedded labels verbatim: accepted `z-first` emitted `cr2_z-first`, invalid C++.
Sanitization alone would alias accepted `a-b` and `a_b`.

The shared table uses the genuine C++ printer sanitizer on the complete prefixed
token, deterministically gives valid spellings priority, and escapes colliding
UTF-8 labels. Every scratch and its appended FieldView `A` reserves a token family;
actual typed scalar locals are reserved too. Authored labels resembling escapes
remain distinct. Safe noncolliding legacy tokens remain byte-identical.

Both explicit rates and existing implicit products use this table. Bundle order,
scratch IDs/subslots, equations, provider ownership, registry labels and accepted
publication/rollback behavior are unchanged. Source/Host tests compile complete
public Programs with labels, collisions, reversed order, Unicode and handle-family
collisions. An original heterogeneous implicit residual is unchanged.

These are Source/Host checks. A fresh actual build/C25 is required for Native;
no Native, MPI, GPU or physical qualification follows from compilation.
