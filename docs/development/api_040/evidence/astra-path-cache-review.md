# Contre-revue Astra : chemin symbolique et cache natif

## SymbolicPath

Fichiers lus : `numerics/symbolic_path.py`, `numerics/nonconservative.py`,
`codegen/module_emit_symbolic_path.py`, `codegen/nonconservative_lowering.py`,
`codegen/module_emit_path.py`, et `substitute_quantities` dans `_ir/application.py`.
Pas de modification des fichiers core root.

Trois findings transmis à root :

1. **P1, état uniforme.** L'ancien bend vectoriel imposait seulement les
   extrémités, pas Psi(s;U,U)=U. Exemple bend=s, B=1, midpoint : état gauche=droite=0
   donnait une intégrale .25. Root a remplacé bend par matrice K et correction
   s(1-s)K(R-L), assurant chemin constant sur diagonale pour corps fini.
2. **P1, matrice constante modifiée par quadrature.** Même après cette correction,
   K(s)=s et midpoint produisaient 1.25 B*jump. Trois tests indépendants ont échoué
   réellement avec B=((2,-1),(0,0)), puis sont passés après correction root :
   partie B(L)*jump intégrée exactement, quadrature sur (B(Psi)-B(L))*dPsi.
   Les échecs étaient 2.5 au lieu de 2, 6.5 au lieu de 2, et erreur relative
   persistante sur saut 1e-6. Ce n'était pas une erreur d'arrondi.
3. **P1, paramètre runtime dans le chemin seulement.** Déclarer un paramètre
   `path_speed_only` et le lire uniquement dans la borne passe validate/resolve,
   puis échoue à l'émission native : RuntimeParamRef index not assigned. Reproduit
   par test indépendant ; communiqué à root pour raccord de collecte/schema.

Le fichier `tests/python/unit/numerics/test_symbolic_path_consistency.py` porte
8 cas : trois états uniformes, trois sauts pour B constant, substitution par
identité conservant le DAG, paramètre runtime exclusivement dans la borne.
Dernier résultat observé avant correction du troisième finding : 7 pass / 1 fail.

La fermeture des quantités étrangères est refusée ; les Vars libres homonymes
ne peuvent se substituer aux arguments. La loi retenue est réauthentifiée dans
le lowerer commun. Les bornes restent une obligation mathématique auteur portant
DF+B sur toute la courbe ; aucune preuve spectrale automatique n'est revendiquée.
Les gardes d'intermédiaires CSE et les contrôles finaux de direction/intégrale/
borne empêchent la conversion silencieuse d'un NaN en résultat fini.

## PreparedResourceCache

Diff et header complet inspectés, ainsi que les nouveaux tests. Aucun finding
bloquant sur la correction : ancien holder gardé vivant, allocation candidate
avant constructeur et gate collectif, construction puis gate collectif, enfin
publication shared_ptr sans allocation ni move de Resource. Les tests couvrent
la ressource non déplaçable, l'identité d'adresse du buffer et son contenu après
échec sur un seul rang, puis retry ; construction initiale refusée aussi.

Les succès serial/MPI2 appartiennent aux reçus d'Astra_PROTOCOLS, et n'ont pas été
rejoués par cette contre-revue. La frontière reste correcte : un constructeur
contenant lui-même des collectifs doit coordonner ses allocations/échecs internes.
Un wrapper externe ne peut réparer un rang déjà bloqué dans un collectif interne.
