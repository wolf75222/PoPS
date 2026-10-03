# SDK26 M18/W09 et CP9 dimensionnel — réception bornée, 2026-10-03

ROOT a reçu les scopes ci-dessous après rejeux purs et contre-revues indépendantes. Cette actualisation ne clôt ni M_N général, ni les 94 obligations de la mission. Les anciens textes, reçus et échecs restent historiques. Le [nouvel index](sdk26_m18_w09_dimensional_reception_20261003.index.json) donne les chemins originaux, sept SHA256 ROOT complets et copies byte-identiques ; aucun reçu n'est régénéré. Cette documentation n'effectue aucune nouvelle exécution.

## Domaines

| Domaine | Source / Native propre | Jobs et comptes observés | Réception exacte |
|---|---|---|---|
| Build SDK26 Dim2 | Source `51b0eec9dcfd660098cc5531eed6c400185c8dad`, DSO `6a30996f85c8aef24ef9233fdcbd0850c7b20ed44a70b85511dd20ffc00030ed` | 732416 ; 24 objets CXX | Build/install/doctor seulement ; ROOT `a95c9ee9…0d8c1b` |
| M18/W09 déclaré | Même SDK26 Dim2, CPU et MPI2 sur un même nœud | 732422/732423 ; 6 profils logiques, 9 nodes JUnit de rang | Trois cibles .49/.5/.9, math positive et refus/restauration ; ROOT `7eee84fa…795033b` |
| Ancien M18 cinq nœuds / trois moments | Même SDK26 Dim2 ; quadrature historique distincte | 732440/732442 ; 2 profils logiques, 3 nodes de rang ; 20 cellules intérieures par profil | Résidu original, populations/entropie, CP9/restart/replay et refus ; ROOT `ac39ea99…e4eec60` |
| CP9 Uniform Dim1/Dim3 | Source `717936347672aeb4bc9146e2b3065bfd42cf1ef5` ; DSO Dim1 `2d3fa405796dfed0fdf706a579281b15692be317996ae166e63c89a15be23e89`, Dim3 `06fbfad4560ea22172f4ec7c561fc45c752d0cc6c9b8238e1e4c1f553634a03e` | Builds732365/732370 ; runs732408–732411 ; 4 CP9 profils + 2 baselines CPU, 8 nodes / 6 XML | Deux States de largeur2/3, stockage/horloges CP9/replay/refus engineering ; ROOT `bbf0b191…fa2b0fd5`, builds `3dffc209…14dc3` |
| Catalogue CI Dim1/2/3 | Douze chemins intégrés et contre-revus Source, indépendants des paquets exécutés ci-dessus | Reçus Source `3899e73f…ce2bd` et intégration `7943b077…34663` | Routing/build/download et admission artefact CI vérifiés Source ; aucun nouveau Native/CI exécuté |

SDK26 distingue 1 193 chemins de production, 1 149 shipped, 1 158 installed ; ces domaines ne sont pas interchangeables. Les builds Dim1/3 ont 1 192 chemins de production, 1 158 installed après build. Header commun `8ee6d09b428cfe172d08008253c2ea490eb733f9b1458f6f2ed8da1584effbf5` ; contrats ABI natif8 / UniformCP9 / AMR12 distincts. Une signature identique n'autorise pas l'héritage d'une qualification d'un DSO à l'autre. Main documentaire de référence `d2129eb897b548b1a31974b45c7f2a86733f70bf`, Native intégré `4778814b58a578f10357951b3b995f190af15fed` : ces intégrations ultérieures ne sont pas de nouvelles exécutions.

## M18-W09

La bibliothèque et le script linéaire déclarent explicitement les nœuds `(-.5,0,.5)`, poids positifs `(1/3,1/3,1/3)`, base `(1,v)` et covecteurs. Original W09 impose l'intervalle et la cible `(1,.9)`, pas une table unique de poids. Le séparateur `(.5,-1)` est nonnégatif à tous les nœuds et donne la marge -.4 : refus certifié de cette cible sans réparation. À `(1,.5)`, une mesure conique concentrée à l'extrémité est faisable, mais aucune population exponentielle strictement positive à multiplicateurs finis ne la réalise. Cette dernière conclusion est mathématique ; le runtime refuse conservativement par `finite_dual_not_certified`, sans conclure « frontière exacte » d'un zéro flottant.

`Certificate@2` normalise par puissance de deux exactement, vérifie chaque coefficient et chaque nœud ; perte de direction non représentable refusée. Les enveloppes d'arrondi restent appariées cellule par cellule. Les indicateurs de finitude précèdent toute réduction ; un NaN/Inf local ne peut être caché. Les deux RED Source indépendants (échelle sous-normale et min d'enveloppe mal apparié), leurs correctifs et traces demeurent conservés. Les opérations natives communes `P.value/P.min/P.guard` précèdent `LocalResidual` ; aucune recette physique ou méthode choisie par le cœur n'est ajoutée.

Le cas `(1,.49)` conserve la cible originale, reconstruit des populations finies strictement positives et les deux moments : résidu maximal `4.085620730620576e-14`, écart au dual analytique `1.539213201340317e-12`. L'original-residual guard reste `2e-11` avec rtol0 ; la perturbation de moment nul a une entropie supérieure. Les cas .5/.9 gardent leurs exceptions contrôlées sur tous les rangs, stockage accepted valid+grown, horloges et `consumer_cursors` exactement restaurés. Les CP9 sauvegardés sont joints aux bits State ; ce fixture ne réalise pas restart/replay.

Fixture@2 prépare Target/quadrature/certificats avant bind ; C25 ALL Model/Program, sources réelles et binaires compagnons sont retenus avant bind. Raw local/complete précède chaque getter valide et sa persistance immédiate. Le hash f91 exécuté avant/après est `d8533e19434279b287cadf119e6b4a068d4f5eee90b3886b927e0ddc26e95a12`. Les lecteurs stricts @3 ajoutent les jointures réelles d'archives/CP9/origines et demeurent distincts des préparations @1/@2. `consumer_cursors` n'est pas une HistoryVariable ; aucune histoire n'est inventée.

## Non-régression historique

Le profil cinq nœuds sur[-1,1], trois moments et 20 cibles modérées n'est pas remplacé par W09. Residual maximal `4.543698750580916e-12`, écart dual `1.9385557395446007e-11`, entropie gap minimal `0.0002657273170826313` : critères inchangés `2e-11` pour le résidu, `1e-8` pour le dual, gap strict `>1e-7`. Accepted/restored et continuous/replayed ont leurs bits full-grown/valid CP9 et horloges exacts. La cible historique impossible `(1,0,1.1)` dans une cellule est refusée deux fois avec restauration ; le safe rebind est explicitement un autre bind.

Les jobs431/436 sont conservés : échec du garde d'alias d'origine avant bind, aucune science reçue depuis ces échecs. Les jobs440/442 corrigent seulement le chemin canonique du wrapper, pas le fixture ou les tolérances. Les lecteurs/rapports RED précédents restent immuables. Ce profil ne reçoit pas la policy C25 require ALL Model/Program, même si des actual Model CPP existent dans le cache ; il ne reçoit pas non plus diagnostics/cache, formule Ghost ou nouvelle physique W09.

## Dimension et catalogue

Les quatre CP9 Dim1/3 réels reçoivent 13 captures par profil, shards/complete et projection valide, trois attaques, restauration CP9/replay et deux States indépendantes2/3 dans un témoin FE dyadique constant. Les deux CPU ajoutent la baseline deux horloges. L'autorité du fichier CPU est la jointure pré-soumission/post-quatre-jobs SHA `16c11e77a4dc0cb54c48485d73539c3c3b763960e1a67314267268a8ed659965` ; il n'existe pas d'export test-source CPU mesuré dans chaque job. Cette limite est explicite, contrairement au before/after f91 des nouveaux W09.

Les orchestration R1 jobs382–385 échoués et le catalogue478 RED ne sont pas requalifiés. Le catalogue reçu Source remplace la paramétrisation multi-DSO par trois fichiers et un support commun ; Dim1/3 explicites, Dim2 défaut. Le build/prewarm et download Dim3 sont réels dans le plan CI, mais aucune exécution CI n'est reçue. L'artefact CI hors préfixe doit joindre origine exacte, couverture/hash de toutes sources Python, manifeste/DSO, signature native des headers ; la voie installée historique reste distincte. Le nouveau hash de fixture ne reçoit pas automatiquement les anciennes preuves Native16c11.

## Reproduction autorisée

Depuis un worktree isolé correspondant au Source exact : `scripts/setup_env.sh --cpu --dim 2` une fois, ENV dédié ; puis `scripts/build_python.sh --dim 2 --mpi --wheel-dir "$WHEELS"` incrémental. Les builds reçus proviennent des clones offline/copy des paquets figés, doctor et inventaires avant/après ; SDK25 original et SDK3/Home sont préservés.

Sur le véritable paquet activé, sans Source PYTHONPATH :

```bash
env -u PYTHONPATH POPS_NATIVE_DIM=2 POPS_REQUIRE_NATIVE_TESTS=1 python docs/development/api_040/run_installed_checks.py --output "$RUN" --test tests/python/integration/runtime/test_m18_w09_declared_domain_runtime.py::test_declared_quadrature_near_boundary_and_w09
env -u PYTHONPATH POPS_NATIVE_DIM=2 POPS_REQUIRE_NATIVE_TESTS=1 python docs/development/api_040/run_installed_mpi_checks.py --output "$RUN_MPI" --ranks 2 --dimension 2 --threads 1 --timeout 6000 --test tests/python/integration/runtime/test_m18_w09_declared_domain_runtime.py::test_declared_quadrature_near_boundary_and_w09
```

Pour la non-régression, remplacer seulement le node par `tests/python/integration/runtime/test_m18_discrete_entropy_runtime.py::test_twenty_interior_targets_and_outside_cone_refusal`. Les wrappers réellement utilisés sont copiés byte-identiques dans le paquet documentaire ; les submitters déploient seulement les scripts autorisés et vérifient le fixture déjà figé. Les nouveaux nodes catalogue `test_uniform_cp9_dim${D}_runtime.py::test_installed_dimensional_cp9_two_states` doivent être exécutés séparément avec la véritable dimension D ; aucune exécution de ces nouveaux nodes n'est revendiquée ici.

Les scopes reçus sont CPU x86, Kokkos/MPICH, MPI2 sur un même nœud et physiques/témoins nommés. Ni AMR, GPU, inter-nœuds, performance, CI, arbitrairement proche de frontière, quadrature universelle, transport/continuum M_N, ni réception complète94 n'en découle. Les auteurs et contre-revieweurs de cette tranche travaillent en GPT‑6.1 Sol selon l'orchestration ROOT ; aucune qualification technique n'est déduite du modèle d'agent.
