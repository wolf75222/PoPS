# M17 : représentant Native futur de la composition publique

Contrat de test `pops.fan-li15-public-composition-native-fixture@1`. Ce lot est uniquement **Source** ; aucune compilation JIT ni exécution Native n'a été effectuée par l'auteur. ROOT doit exécuter le futur test dans son SDK installé authentifié.

## Équations et méthode

Le support test expose le modèle Fan–Li15 original : deux vitesses, base totale degré4, quinze moments, dix équations conservatives et cinq terminales régularisées. Il appelle `fan_li15_expressions` et `fan_li15_path`, qui construit le public `NormalizedPolynomialPath`. Flux, matrices B et covecteurs sont visibles ensemble dans le support ; ni le nom du modèle ni celui des composantes ne sélectionnent un émetteur central.

Le domaine périodique est `(0,1)^2`, grille16×16. Les données sont les moyennes exactes de la mixture originale de deux gaussiennes, fonction de x et extrudée en y. Le programme est l'original SSPRK2 explicitement composé, avec gardes rho/SPD avant les deux évaluations et à l'endpoint, dt=`1e-4`. Le nouveau représentant fait **deux** étapes, tandis que le corpus historique Gauss4 conserve ses huit étapes : cette tranche courte ne reçoit pas le corpus historique entier.

Les coefficients initiaux ne s'annulent pas : h3max=`0.017545356253353284`, h4max=`0.005277108301697017` ; produit régularisé maximal aux faces=`0.00756390007852085`. La référence après une étape B actif diffère de l'équation distincte B=0 de `1.1669082960352739e-5`. B=0 est ici une référence séparée de discrimination ; aucun run Native B=0 n'est reçu.

## Oracle et captures

L'oracle physique primaire indépendant existant calcule flux et B(U)dU, sans utiliser le DAG ou le noyau émis. La référence FV est explicitement Rusanov premier ordre sur le segment brut, majorant complet des moments seconds, puis le même SSPRK2. Quadratures24/48 indépendantes sont comparées. Les critères originaux demeurent : erreur état `3e-8`, inventaire des dix composantes conservatives `3e-13`, invariance y `1e-12`, initial `1e-13`, temps `1e-14`, activation B>`1e-8`. Aucun seuil n'est ajusté à une marge observée.

Avant bind sont persistés les déclarations, le vrai C++/IR du programme de layout sélectionné et le manifeste compilé. La sélection exige un véritable `CompiledSimulationArtifact.verify()`, exactement un layout gas, sa ligne vérifiée et l'identité du programme wrapper/ligne. L'initialisation est un `InitialCondition(...BindArray(),ConservativeCellAverage())` public, avec son unique sujet qualifié authentifié. Chaque phase initial/accepted1/accepted2 est enregistrée avant l'étape suivante : tous les quinze arrays16×16, horloge physique et macrostep. Une erreur bind/run est enregistrée collectivement avant toute autre opération Native et l'exception locale d'origine est relancée.

Le reçu contient les chemins/SHA du paquet Native réellement chargé et du programme réellement compilé, les clés ABI/problème/cache, le layout et les hashes des fichiers. Il garde `root_scientific_approval=false` et `full_M17_qualification=false`. ROOT fournit indépendamment l'inventaire SDK, l'identité avant/après et les sceaux.

Ces captures sont les valeurs intégrales du calcul **Uniform**, pas un checkpoint AMR/ghost entier. Ce lot ne ferme pas CP12/restart, rollback, Riemann/chocs, changement d'ordre du modèle Fan–Li, convergence spatiale/temporelle, GPU ou performance applicative. Les deux nœuds canonical/reverse exercent le rangement ; une comparaison croisée des archives Native reste à recevoir par un non-auteur.

## Défaut de raccord réellement trouvé

La première construction Source originale normalized-analytic refusait son flux : déclaration `physical.x/y` contre contrat exact `physical.directional_flux(g)`, soit des DAG différents pour la même expression mathématique. Le gel distinct `5c74303` corrige **uniquement le script example** : flux et chemin de l'option normalisée partagent explicitement les expressions directionnelles/covecteurs. La voie Gauss4 conserve physical.x/y et la quadrature originale. Ni bibliothèque centrale ni authentification ne sont relâchées.

RED : cohorte test2P/1F (`ValueError: path realization does not match its exact retained physical expressions`) avant resolve. GREEN : construction publique des deux options passe, puis support typed complet validate→resolve→lower→C++ réel : **3 PASS en155.63 s**, XML `/tmp/sol61-m17-future-fixture-source.xml`. Deux nœuds Native sont seulement collectés (0.61 s), sans exécution.

Commandes :

```sh
rtk proxy env -u PYTHONPATH PYTHONDONTWRITEBYTECODE=1 /Users/romaindespoulain/miniforge3/envs/pops/bin/python -m pytest --noconftest -p no:cacheprovider -o pythonpath=python tests/review/test_m17_future_fixture_source.py -q --tb=short --junitxml=/tmp/sol61-m17-future-fixture-source.xml
# ROOT seulement, dans son vrai SDK installé et PYTHONPATH absent :
env -u PYTHONPATH python -m pytest tests/python/integration/runtime/test_fan_li15_public_composition_runtime.py -q --tb=short
```

Le corpus original authentifié a été relu dans `original-handoff-readable-copy-20261002/context/CORPUS_ORIGINAL.md`135–170 : chemin explicite, régularisation différentielle réelle, rangement/ordre sans émetteur dédié. La réception scientifique exige les nouveaux résultats authentiques sauvegardés ; elle n'est pas déduite de cette validation Source.

## Delta JSON strict

Le véritable `CompiledLayoutProgram.to_data()` contient un digest bytes32. Le témoin ne l'encode pas avec `default=str` : il persiste explicitement layout_id, target, block_names, identity.token et artifact.artifact_identity.token, après vérification de la ligne. Un test sur les vraies classes wrapper, composants explicitement Source metadata-only, reproduit le refus de json.dumps(row.to_data()) puis vérifie le roundtrip du nouveau contrat. Aucun SDK/DSO n'est chargé par ce test.
