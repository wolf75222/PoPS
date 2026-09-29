# C11 — coordonnées primitives conjointes

Base isolée : f81be74, branche codex/api040-principal-group. Aucun build/install natif exécuté par cet agent ; MAIN préservé.

## Obligation et réalisation

C11 (`PoPS_Codex_handoff_0.4.0/reference/PoPS_API_v0.4.0/document/02_contracts.tex`, ligne 38) impose que plusieurs quantités puissent participer au même flux principal sans perdre les dérivées croisées lorsque le stockage est séparé. La reconstruction et la borne restent celles du système complet. La présente tranche lève le refus artificiel des variables primitives pour ce groupe.

`Model.primitive_state(*coordinates, states=(...), conservative=(...))` déclare un ordre physique de coordonnées, leur récupération et leur inverse explicite. `Model.recovery_admissibility(states=(...), **predicates)` complète le domaine. Ces objets sont immuables, indépendants du layout, authentifiés par états/variables/paramètres exacts et inclus dans le hash du Module. Une capture homonyme étrangère est refusée. Les composants mixtes et la permutation de l'ordre physique ne changent pas le pack natif par lignes.

La réalisation réutilise l'IR commun, son CSE contrôlé et les conversions natives avec statut. Chaque ligne utilise son propre tableau RuntimeParam. Les contraintes et tous les intermédiaires actifs sont contrôlés avant un statut Success. Les branches `where` suivent l'évaluation paresseuse existante. Aucune inverse implicite, callback par cellule, duplication de reconstruction ou plafond de cardinalité.

Une conversion au sample d'une ligne lit l'état conjoint complet : le rayon maximal des reconstructions primitives est donc propagé à toutes les lignes, au stockage généré et au contrat de nidification AMR. Un témoin User à offsets 0/2 vérifie cette propagation, puis prépare une réception native contre oracle.

## Contrat sérialisé

ModuleManifest passe de schema 9 à 10, afin d'authentifier `expressions.primitive_coordinates`. Les images schema 9 sont refusées explicitement ; ce changement ne modifie pas l'ABI native du fournisseur. Le nouveau mode de `PreparedPrincipalFlux` fait partie des headers recompilés par le build central.

## Preuves locales

- Rouge initial : `primitive_state(..., states=...)` levait TypeError avant la tranche.
- 44 tests source : coordonnées conjointes, ownership, permutation, groupes 1+1/2+3/1+4, capture numérique, manifest et groupes principaux ; dont 3 contre-tests indépendants Sol (3+4 composantes, permutation, hash inverse/domaine, schema 9 refusé/schema 10 rond-trip).
- 7 tests existants StateStorage réussis.
- Script canonique `test_module_manifest.py` via runpy sur checkout : réussi.
- C++ complet généré System et AmrSystem avec paramètres et domaine : syntax-only réussi avec les flags MPI/Kokkos réels du build parent.
- C++ complet User avec rayons distincts : syntax-only dans `../../outputs/principal-primitive-user-syntax.log`.
- 7 tests natifs collectés, non exécutés : 5 combinaisons de groupement/permutation et rebind du même artefact, 1 domaine invalide localisé avec rollback de toutes les lignes/temps, 1 reconstruction User avec halo joint asymétrique. Oracle NumPy FV indépendant sur les deux axes ; moyennes de cellules exactes initiales.
- Contre-revue indépendante : `outputs/sol6-primitive-independent-review.md`.

## Portée à ne pas dépasser

`conservative=` est une assertion explicite de l'auteur : aucune preuve algébrique générale de inverse(forward(U))=U n'est revendiquée. Avec des paramètres différents par ligne, les formules doivent employer le contexte propre à la composante de sortie ; les tests exercent une famille non linéaire où cette assertion est exacte et les valeurs de paramètres changent réellement la reconstruction.

La réception native préparée est Uniform Dim2. L'émission AMR syntaxiquement valide n'est pas une preuve d'exécution AMR, MPI ou de conservation composite. Le groupe garde les obligations AMR précédentes (bords périodiques et fenêtres synchrones) ; les bords généraux restent à réaliser. PathConservative n'est pas modifié par cette tranche.
