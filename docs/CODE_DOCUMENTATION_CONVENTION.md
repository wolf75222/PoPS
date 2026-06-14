# Convention de documentation du code

Objectif : documenter les interfaces publiques, les invariants et les frontieres de couches sans
paraphraser le code. Une bonne documentation explique quand utiliser une brique, quelles hypotheses
elle prend, et quels contrats elle expose aux autres parties de la bibliotheque.

Cette convention est alignee sur des standards existants, adaptes a `adc_cpp` :

- [Doxygen, "Documenting the code"](https://www.doxygen.nl/manual/docblocks.html) : blocs `///`,
  separation brief/details, `@file`, `@brief`, `@param`, `@return`.
- [Google C++ Style Guide, Comments](https://google.github.io/styleguide/cppguide.html#Comments) :
  commentaires de classes non triviales, commentaires de declarations pour l'usage, commentaires
  d'implementation pour le "comment/pourquoi", invariants de synchronisation.
- [C++ Core Guidelines, Interfaces](https://isocpp.github.io/CppCoreGuidelines/CppCoreGuidelines#S-interfaces) :
  une interface est un contrat ; les preconditions, postconditions et parametres de templates doivent
  etre explicites.
- [PEP 257](https://peps.python.org/pep-0257/) : docstrings Python en premiere instruction du module,
  de la classe ou de la fonction publique, avec resume court puis details.

## 1. Entete de fichier

Chaque `.hpp` / `.cpp` doit commencer, apres `#pragma once` ou les includes obligatoires, par un bloc
`@file` de ce type :

```cpp
/// @file
/// @brief Role court du fichier.
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

Doxygen exige un `@file` pour documenter correctement les fonctions libres, typedefs, enums et macros
d'un header. Donc tout header avec symboles publics doit en avoir un. Utiliser les blocs `///` pour
rester compatibles avec clang-format et le style deja present dans le repo.

Regle : l'entete explique le **sens architectural**. L'historique d'une PR reste hors du header,
sauf quand il documente un garde-fou technique durable.

## 2. Entete de classe / struct

Chaque classe publique ou struct non triviale doit avoir un commentaire d'interface. Le format n'est
pas une fiche obligatoire ; garder seulement les lignes utiles au type concerne :

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

Regle Google C++ adaptee : toute classe/struct non evidente doit permettre au lecteur de savoir
quand l'utiliser et quelles hypotheses elle fait. Pour les classes touchees par MPI, Kokkos ou AMR,
documenter explicitement les hypotheses de synchronisation, de possession et de conservation.

Regle C++ Core Guidelines adaptee : si l'interface est template, les parametres doivent etre
documentes par un concept (`PhysicalModel`, `EllipticSolver`, `CoupledSystemLike`, etc.) ou par une
phrase qui dit ce que le type doit fournir. Preferer un `concept` a un commentaire si c'est possible.

Pour une petite policy triviale (`NoSlope`, tag vide, petit foncteur), une phrase suffit. Ne pas
sur-documenter les types evidents.

## 3. Commentaires de fonction

Commenter une fonction quand :

- elle traverse une couche (`System` -> elliptique, AMR -> reflux, Python loader -> C++) ;
- elle a un invariant de conservation ;
- elle a une contrainte MPI/GPU ;
- elle refuse volontairement un cas ;
- elle a une semantique temporelle subtile (`substeps`, `stride`, IMEX).

Dans un header, le commentaire de declaration decrit **l'usage et le contrat** :

```cpp
/// Installe un bloc evolue sur la hierarchie AMR commune.
/// @param name nom unique du bloc.
/// @param substeps nombre de sous-pas ; doit etre >= 1.
/// @throws std::runtime_error si le nom existe deja.
```

Dans un `.cpp` ou dans le corps d'une fonction, le commentaire de definition decrit **l'implementation
non triviale** : ordre des operations, choix numerique, garde MPI, raison d'un compromis. Ne pas
repeter mot pour mot le commentaire de declaration.

Utiliser `@param`, `@return`, `@throws` quand la fonction fait partie de l'API publique ou d'un seam
important (`System`, `AmrSystem`, DSL loader, solveurs, AMR). Pour les fonctions internes evidentes,
une phrase suffit.

Eviter :

```cpp
// Increment i
++i;
```

Preferer :

```cpp
// Tous les rangs doivent appeler ce solve collectif ; les rangs sans fab local font no-op
// sur les boucles locales mais participent aux reductions.
```

## 4. Niveaux de commentaire

| Niveau | A commenter | Exemple |
|---|---|---|
| Dossier/fichier | Role architectural, limites | `runtime/System` orchestre, ne contient pas les formules physiques. |
| Classe | Usage, contrat, invariants, contraintes | `AmrSystem` orchestre une hierarchie AMR commune. |
| Methode publique | Contrat utilisateur/API, `@param`, `@return`, `@throws` si utile | `add_block` accepte plusieurs time policies. |
| Bloc complexe | Pourquoi l'ordre des operations compte | Poisson puis aux puis RHS. |
| Ligne | Rare, seulement bug/astuce | Garde `local_size()==0` MPI. |

Principe de densite : commenter plus fortement les interfaces et les frontieres de couches que les
petites boucles internes. Le code scientifique a besoin de contrats lisibles, pas de bruit.

## 5. Formules et numerique

Quand le code implemente une formule, ecrire la formule une fois pres du kernel ou du builder :

```cpp
/// Assemble L(phi) = -div(A grad phi) + kappa phi.
/// A est centre cellule ; les flux de face utilisent la moyenne definie plus bas.
```

Si la formule vient d'un papier, citer le document local (`docs/SCHUR_CONDENSATION_DESIGN.md`) plutot
que mettre une longue bibliographie dans le header.

## 6. MPI / GPU

Toute fonction appelee sous MPI/Kokkos doit expliciter les deux points suivants si pertinents :

- collective ou locale ;
- comportement des rangs sans donnees locales.
- ordre des fences si la fonction passe de kernels device a lecture hote ;
- raison d'un foncteur nomme si le code est instancie depuis une autre unite de traduction.

Exemple :

```cpp
/// Collectif : tous les rangs appellent la fonction. Les boucles sur `local_size()` sont no-op
/// sur les rangs sans box locale.
```

Pour Kokkos/CUDA, noter les endroits ou les foncteurs nommes sont obligatoires :

```cpp
/// Foncteur nomme : evite les lambdas device cross-TU qui cassent nvcc.
```

Si une fonction est collective, le dire dans son commentaire public. Si elle est locale, le dire aussi
quand l'appelant pourrait croire qu'elle synchronise. Les bugs MPI les plus dangereux viennent de cette
ambiguite.

## 7. AMR multi-blocs

Regle de documentation et de design :

```text
AMR multi-blocs conservatif = hierarchie commune, cellules co-localisees, regrid par union des tags.
```

Ne jamais documenter comme cible proche :

```text
une espece absente localement d'un patch raffine
```

sauf avec un plan de projection conservative complet. Sinon les sources couplees, le RHS Poisson et
le reflux ne sont pas conservatifs.

## 8. Python

Les fichiers Python doivent avoir une docstring de module :

```python
"""Role public du module.

Ce module expose ...
Chemins supportes : ...
Contraintes : ...
"""
```

Regles PEP 257 retenues :

- la docstring est la premiere instruction du module, de la classe ou de la fonction publique ;
- premiere ligne courte, puis ligne vide, puis details ;
- les fonctions publiques documentent arguments, retour, effets de bord, exceptions et restrictions
  d'appel si ce n'est pas evident ;
- ne pas recopier la signature dans la docstring.

Les classes d'API Python doivent dire si elles sont :

- chemin production ;
- prototype CPU ;
- compatibilite/legacy ;
- simple objet de configuration.

Pour `adc` en particulier, chaque classe d'API doit dire si elle garde le chemin GPU/MPI ou si elle
repasse par un chemin hote/prototype. C'est un contrat utilisateur, pas un detail.

## 9. Processus d'application

Ne pas faire un patch geant qui ajoute des commentaires partout sans lire les fichiers. Appliquer par
lots :

1. un dossier coherent ;
2. entete fichier + classes principales ;
3. aucun changement de comportement ;
4. verification `git diff` ;
5. tests seulement si le patch touche du code executable.

Ordre conseille :

1. `include/adc/amr`;
2. `include/adc/mesh`;
3. `include/adc/core`;
4. `include/adc/numerics/time`;
5. `include/adc/numerics/elliptic`;
6. `include/adc/runtime`;
7. `python/system.cpp` par extraction/refactor, pas seulement commentaires.
