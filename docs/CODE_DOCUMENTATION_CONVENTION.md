# Code documentation convention

Goal: document public interfaces, invariants and layer boundaries without
paraphrasing the code. Good documentation explains when to use a brick, what assumptions
it makes, and what contracts it exposes to the rest of the library.

This convention is aligned with existing standards, adapted to `adc_cpp`:

- [Doxygen, "Documenting the code"](https://www.doxygen.nl/manual/docblocks.html): `///` blocks,
  brief/details separation, `@file`, `@brief`, `@param`, `@return`.
- [Google C++ Style Guide, Comments](https://google.github.io/styleguide/cppguide.html#Comments):
  comments on non-trivial classes, declaration comments for usage, implementation
  comments for the "how/why", synchronization invariants.
- [C++ Core Guidelines, Interfaces](https://isocpp.github.io/CppCoreGuidelines/CppCoreGuidelines#S-interfaces):
  an interface is a contract; preconditions, postconditions and template parameters must
  be explicit.
- [PEP 257](https://peps.python.org/pep-0257/): Python docstrings as the first statement of the module,
  the class or the public function, with a short summary then details.

## 1. File header

Each `.hpp` / `.cpp` must start, after `#pragma once` or the required includes, with a
`@file` block of this kind:

```cpp
/// @file
/// @brief Short role of the file.
///
/// Couche : `include/adc/<dossier>`.
/// Role : ...
/// Contrat : ...
///
/// Invariants :
/// - invariant important ;
/// - contrainte importante ;
/// - relation avec les autres couches.
```

Doxygen requires a `@file` to correctly document the free functions, typedefs, enums and macros
of a header. So every header with public symbols must have one. Use `///` blocks to
stay compatible with clang-format and the style already present in the repo.

Rule: the header explains the **architectural meaning**. The history of a PR stays out of the header,
except when it documents a durable technical safeguard.

## 2. Class / struct header

Each public class or non-trivial struct must have an interface comment. The format is not
a mandatory form; keep only the lines useful to the type at hand:

```cpp
/// Role court de la classe.
///
/// Usage : ...
/// Contrat : ...
/// Invariants : ...
/// Preconditions : ...
/// Postconditions : ...
/// Contraintes : ...
class Foo { ... };
```

Adapted Google C++ rule: any non-obvious class/struct must let the reader know
when to use it and what assumptions it makes. For classes touched by MPI, Kokkos or AMR,
explicitly document the synchronization, ownership and conservation assumptions.

Adapted C++ Core Guidelines rule: if the interface is template, the parameters must be
documented by a concept (`PhysicalModel`, `EllipticSolver`, `CoupledSystemLike`, etc.) or by a
sentence saying what the type must provide. Prefer a `concept` over a comment if possible.

For a small trivial policy (`NoSlope`, empty tag, small functor), one sentence is enough. Do not
over-document obvious types.

## 3. Function comments

Comment a function when:

- it crosses a layer (`System` -> elliptic, AMR -> reflux, Python loader -> C++);
- it has a conservation invariant;
- it has an MPI/GPU constraint;
- it deliberately refuses a case;
- it has subtle temporal semantics (`substeps`, `stride`, IMEX).

In a header, the declaration comment describes **usage and contract**:

```cpp
/// Installe un bloc evolue sur la hierarchie AMR commune.
/// @param name nom unique du bloc.
/// @param substeps nombre de sous-pas ; doit etre >= 1.
/// @throws std::runtime_error si le nom existe deja.
```

In a `.cpp` or in the body of a function, the definition comment describes **the non-trivial
implementation**: order of operations, numerical choice, MPI guard, reason for a trade-off. Do not
repeat word for word the declaration comment.

Use `@param`, `@return`, `@throws` when the function is part of the public API or of an important
seam (`System`, `AmrSystem`, DSL loader, solvers, AMR). For obvious internal functions,
one sentence is enough.

Avoid:

```cpp
// Increment i
++i;
```

Prefer:

```cpp
// Tous les rangs doivent appeler ce solve collectif ; les rangs sans fab local font no-op
// sur les boucles locales mais participent aux reductions.
```

## 4. Comment levels

| Level | What to comment | Example |
|---|---|---|
| Folder/file | Architectural role, boundaries | `runtime/System` orchestrates, does not contain the physics formulas. |
| Class | Usage, contract, invariants, constraints | `AmrSystem` orchestrates a common AMR hierarchy. |
| Public method | User/API contract, `@param`, `@return`, `@throws` if useful | `add_block` accepts several time policies. |
| Complex block | Why the order of operations matters | Poisson then aux then RHS. |
| Line | Rare, only bug/trick | `local_size()==0` MPI guard. |

Density principle: comment interfaces and layer boundaries more heavily than the
small internal loops. Scientific code needs readable contracts, not noise.

## 5. Formulas and numerics

When the code implements a formula, write the formula once near the kernel or the builder:

```cpp
/// Assemble L(phi) = -div(A grad phi) + kappa phi.
/// A est centre cellule ; les flux de face utilisent la moyenne definie plus bas.
```

If the formula comes from a paper, cite the local document (`docs/SCHUR_CONDENSATION_DESIGN.md`) rather
than putting a long bibliography in the header.

## 6. MPI / GPU

Any function called under MPI/Kokkos must make the following two points explicit if relevant:

- collective or local;
- behavior of ranks without local data.
- order of fences if the function goes from device kernels to host read;
- reason for a named functor if the code is instantiated from another translation unit.

Example:

```cpp
/// Collectif : tous les rangs appellent la fonction. Les boucles sur `local_size()` sont no-op
/// sur les rangs sans box locale.
```

For Kokkos/CUDA, note the places where named functors are required:

```cpp
/// Foncteur nomme : evite les lambdas device cross-TU qui cassent nvcc.
```

If a function is collective, say so in its public comment. If it is local, also say so
when the caller might believe it synchronizes. The most dangerous MPI bugs come from this
ambiguity.

## 7. Multi-block AMR

Documentation and design rule:

```text
AMR multi-blocs conservatif = hierarchie commune, cellules co-localisees, regrid par union des tags.
```

Never document as a near target:

```text
une espece absente localement d'un patch raffine
```

except with a complete conservative projection plan. Otherwise the coupled sources, the Poisson RHS and
the reflux are not conservative.

## 8. Python

Python files must have a module docstring:

```python
"""Role public du module.

Ce module expose ...
Chemins supportes : ...
Contraintes : ...
"""
```

PEP 257 rules retained:

- the docstring is the first statement of the module, the class or the public function;
- first line short, then a blank line, then details;
- public functions document arguments, return, side effects, exceptions and call restrictions
  if it is not obvious;
- do not copy the signature into the docstring.

Python API classes must say whether they are:

- production path;
- CPU prototype;
- compatibility/legacy;
- simple configuration object.

For `adc` in particular, each API class must say whether it keeps the GPU/MPI path or whether it
falls back through a host/prototype path. This is a user contract, not a detail.

## 9. Application process

Do not make a giant patch that adds comments everywhere without reading the files. Apply in
batches:

1. one coherent folder;
2. file header + main classes;
3. no behavior change;
4. `git diff` check;
5. tests only if the patch touches executable code.

Recommended order:

1. `include/adc/amr`;
2. `include/adc/mesh`;
3. `include/adc/core`;
4. `include/adc/numerics/time`;
5. `include/adc/numerics/elliptic`;
6. `include/adc/runtime`;
7. `python/system.cpp` by extraction/refactor, not only comments.
