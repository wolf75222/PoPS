# Tranche sémantique native T1a/T3a

Implémentée dans le vrai `python/pops`, sans runtime scientifique alternatif.
L'algebra scalaire Expr existante porte les feuilles de composants des valeurs
SSA Program. Les corps sur `q[i]` sont réutilisables sur déclarations physiques,
valeurs courantes, stages et inconnues locales. Produits/divisions composantiels
non affines, puissances et opérations scalaires déjà reconnues sont matérialisés
par `Program.value`. Les coefficients de méthode rejoignent cette même algèbre.

Les lectures authentifient le Program vivant, les références qualifiées, le
bloc, le Space, la région et l'horloge. Une définition de stage refusée restaure
l'auteur atomiquement. La capture conserve les occurrences du corps ; elle ne
normalise pas les occurrences de termes de taux par ensemble.

Le graphe sérialisé est une table topologique avec références entières, sans
expansion d'un DAG partagé en arbre. Le lowering rétablit le DAG Expr puis passe
par la CSE commune `materialize_all=True`. Chaque feuille et intermédiaire est
contrôlé fini. Les noyaux C++ écrivent un statut par cellule, le réduisent sur la
voie d'exécution native et refusent le pas avant publication. Le masque actif
préserve les cellules inactives. Les résidus Newton invalides signalent NaN au
provider natif qui possède la convergence, les garde-fous et les diagnostics.

`LocalResidual(..., captures={...})` sépare le seed Newton des données gelées
de l'équation. La signature est `residual(P, unknown, **captures)` ; la signature
historique reste disponible lorsque captures n'est pas fourni. Les placeholders
conservent point, state_ref et contexte de champ. Le lowering `source/apply` dans
le résidu lit son argument réel (inconnue, seed ou capture), au lieu de relire
systématiquement l'itéré.

## Vérification avant installation commune

85 tests passent avec l'interpréteur `pops-api040`, `env -u PYTHONPATH`, et import
explicite du checkout. Ce sont des preuves d'auteur/IR/codegen, pas des résultats
de runtime installé. Les fichiers sont `test_program_expressions.py`,
`test_program_expressions_adversarial.py`, `test_program_value_data_model.py`,
`test_temporal_handle_data_model.py`, `test_general_solve_request.py` et
`test_clock_propagation.py`.

La contre-revue Sol6 a reproduit l'absence de promotion de `dt * q[i]` ; corrigé
et couvert dans les deux ordres. Root et Sol6 ont signalé `minimum(0/0, q)` qui
masquait NaN ; contrôle de tous les intermédiaires ajouté. La réutilisation d'un
sous-graphe 32 fois conserve 34 nœuds et 32 temporaires CSE.

Le fichier d'intégration `test_program_expression_runtime.py` contient cinq cas
à exécuter après le rebuild/install commun root : H(2)=6 ; H(T)-H(old)=1 avec
old=0 et seeds 0.25/2, donnant la même racine (sqrt(5)-1)/2 ; rejet avec état
inchangé pour division invalide et overflow masqués par minimum. La tentative
avant installation commune s'est arrêtée au manifest absent du checkout avant
exécution ; aucun succès natif de ces cinq cas n'est revendiqué ici.

## Frontière

Cette tranche n'achève pas la 0.4.0 globale. Les matériaux, physiques générales,
globaux, `where` scientifique commun, agrégats structurés et corpus scientifique
ne sont pas qualifiés par ces tests. Aucun résultat prototype n'est comptabilisé.
L'environnement/build natif, l'intégration et la qualification finale restent
sous responsabilité root. Le rapport de contre-revue native Sol6 est séparé dans
`astra-cross-review-native.md`.
