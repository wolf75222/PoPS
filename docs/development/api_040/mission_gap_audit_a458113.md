# Audit indépendant des écarts de mission à a458113

Référence examinée : `a458113e` (2026-09-29), et non une qualification du HEAD
mobile. MAIN était déjà `986fb25` lors de la lecture ; les registres ont été lus
avec `git show a458113:…`. Les fichiers de production cités ci-dessous étaient
inchangés entre ces deux révisions. Aucune campagne, compilation ou modification
du cœur n'a été effectuée pour cet audit.

Autorités : handoff `PROMPT_CODEX.md`, sections 5–10 ; contrats C01–C40 dans
`reference/PoPS_API_v0.4.0/document/{02_contracts,03_protocols}.tex` ; registre
intégré [contracts.csv](contracts.csv), [corpus.json](corpus.json) et
[CHECKPOINT.md](CHECKPOINT.md). Les reçus historiques restent attachés à leur
source, SDK, dimension et backend propres. Une intégration source n'actualise pas
automatiquement leur qualification.

## Ce qui empêche une clôture

| Tranche | Mécanismes établis dans le code / preuves bornées | Obligations encore ouvertes |
|---|---|---|
| T1, C05–09 | Expressions communes, captures qualifiées, branches paresseuses et gardes flottantes ; tests source/host et certains consommateurs natifs. L'exponentielle publique est intégrée. | Couverture de tous les consommateurs/opérations et profils flottants ; représentation non linéaire des moyennes, unités/supports hétérogènes. Une fonction de moyenne n'est pas une moyenne de fonction. |
| T2, C10–14 | User reconstruction joint, flux utilisateur, groupe principal et Primitive ; routes AMR choisies reçues. SymbolicPath reçoit M17 actif. Diffusion diagonale **et TensorDiffusion SPD** existent. | Nonconservatif d'ordre supérieur avec contribution intérieure ; caractéristiques sur le symbole complet/oblique ; tenseurs/joints hors contrats existants et traces compatibles. Un callback User ne reçoit aucune garantie universelle d'ordre, positivité ou réalisabilité. |
| T3, C15–19 | LocalNewton contrôle le résidu original ; produits locaux de tailles quelconques sélectionnées, P.source/P.apply et captures gelées ; petits solves robustes. M11 et M18 sont composés publiquement. | Réception native récente produits/H05/M11/M18 ; résidus globaux mixtes, contraintes/domaines/noyaux et condensation publique générale. Les providers dépendant du candidat et plusieurs snapshots d'un même provider restent explicitement refusés. |
| T4, C29–37 | Outcomes consommables, quiescence, leases/completion/drain natifs et rejets collectifs ; cas C34 réels de publication NPZ ; correction MPI M15 intégrée. | Pas de preuve générale de conflits read/write de racines PDE concurrentes, ni de composition de toutes les annulations/complétions tardives ; GPU non exécuté. Le nouveau refus MPI M15 attend le paquet reconstruit. |
| T5, C20–24/C38–40 | Histoires, continuations et transferts ; C38 AND9 trois niveaux avec rollback/retry dans une réception C++/MPI2 choisie ; C22 ExternalTimeGrid exact. | Frontières calculées/multi-horloges et échanges d'intervalle généraux ; regrid après échange provisoire/remplacement et propriétés numériques du transfert. C22 ne couvre pas un endpoint de relaxation RK calculé. Une copie conservatrice ne prouve ni admissibilité ni commutation différentielle. |
| T6, C25–28 | Corpus partiel avec archives d'états et oracles ; CPU OpenMP et MPICH réellement utilisés ; protocole performance historique comparable. | Corpus complet et méthodes prescrites encore incomplets ; pas de GPU, OpenMPI, CI distante ou HPC nouvellement qualifiés ; pas de campagne PDE 3D générale. Les compteurs absents restent indisponibles. |

Le checkpoint rapporte, sur le SDK `02723ae9` antérieur aux derniers correctifs,
92 CTest passés et trois gardes ignorées, avec quatre agrégats MPI2 passés. Cela
ne qualifie pas `a458113`. La réception produits après JIT a encore dix échecs
sur vingt-et-un : huit halos manquants et deux collisions de paramètres, corrigés
ensuite en source. Le hang MPI M15 a aussi un correctif source, pas encore un
nouveau reçu au point d'audit. Les résultats source/host ne sont pas des pas
MultiFab exécutés.

## Dette précise de registre

Le registre n'est ni vide ni globalement vert, mais certaines lignes retardent
sur les ajouts du checkpoint. Son `current_head=5d11268` ne peut pas servir de
révision d'exécution universelle.

- W03 conserve les sept échecs de compilation AMR historiques sans relier les
  sept routes User/principal réussies ensuite, déjà mentionnées par C11/C23.
- C32 présente le rerun M08 corrigé comme ouvert ; C20/C24 et le checkpoint
  portent le reçu source/stage `cfef849`. Le cache général reste ouvert.
- C39 présente AND9 comme en attente ; C38/C40 décrivent la réception choisie
  passée. Cela ne ferme pas les propriétés numériques de tous les transferts.
- M11/W10 et M18/W09 restent `not_executed_for_this_mapping` avec « candidate
  symbols only », alors que leurs bibliothèques, tests et contre-revues sont
  intégrés. Il faut ajouter leur preuve source/host sans inventer un résultat natif.
- M10 est reçu sur un et deux rangs selon le checkpoint, mais sa ligne garde
  MPI parmi les gaps sans distinguer le témoin d'un pas de la campagne complète.
- Certaines lignes W01/W02 disent encore « original witness remains open »
  immédiatement après une qualification historique bornée : préciser la variante
  exacte manquante au lieu d'effacer le reçu ou de promouvoir l'ensemble.

Actions de traçabilité : relier chaque correction à son ancien rouge et à son
nouveau reçu, séparer `implemented/source/host/native/backend` de `passed/failed/
not_executed`, et garder une colonne d'obligation résiduelle. Ne pas régénérer
aveuglément les tableaux avec le bootstrap historique `build_registry.py`.

## Corpus : préserver les différences mathématiques

- M01–M03 ont des balayages historiques complets dans leurs variantes Dim2
  extrudées ; ils ne valent pas toute dimension/AMR ni le HEAD courant.
- M04 ForwardEuler échoue toujours au critère d'ordre `.60635 < .7` ; le succès
  SSPRK2 est une autre méthode, pas la réparation de ce reçu.
- M06 est homogène en enthalpie, pas un front de Stefan ; M13 est une chaîne
  réactionnelle homogène, pas un streamer/photoionisation non locale.
- M07 reçoit le lac entièrement mouillé à l'équilibre en vraie Dim1, serial/MPI2 ;
  pas wet/dry ou écoulement perturbé. M08 reçoit deux pas/stages choisis. M10
  reçoit un pas Poisson–SG, pas la campagne de convergence complète.
- M09/W06 : l'oracle fini 12×12 n'est pas un solve PoPS ; le tutoriel disque FV
  n'est pas le même opérateur d'incidence aléatoire 8×4.
- M11/W10 : friction rang deux et système augmenté 4×4 composés ; la garde
  **L j−f originale** empêche le multiplicateur d'accepter une charge projetée.
  Les masses molaires/fermetures du mélange Poisson–Maxwell–Stefan restent à fixer.
- M15 : cinq moments B.1 axiaux, pas la hiérarchie multiordre ; MPI rouge conservé.
  M16 : le contre-exemple oblique établit une obstruction MATH de la variante
  étudiée, pas l'impossibilité de tout transport de moments. M17 est un transport
  Fan–Li15 réellement non conservatif, mais Dim2 extrudé, pas AMR ou ordre supérieur.
- M18 possède désormais les trois multiplicateurs seuls avec une cible gelée
  read-only (`6b45bda0`), suite au correctif d'inventaire bind `629a2126`.
  Son exponentielle/fermeture discrète reste à recevoir nativement.
- M19 : les réductions et relèvements **existent** (`PhysicalSupportMap`, tests
  Uniform/AMR/interstage). Le registre ne contient pas pour autant de campagne
  complète Vlasov/BGK ; il faut recevoir la chaîne prescrite puis le transport.
- M20/M21/M24/M25 ont des données physiques constitutives encore non fermées ;
  elles restent des obligations explicites, sans remplacement par une ODE choisie.
  M22/H05 est un échange linéaire fermé de deux réservoirs, pas une PDE M1/Marshak.
- M05, M14/W11, M23, M26, M27, M28 contiennent des sous-cas fermés utilisables
  immédiatement pour développer des mécanismes communs ; ils ne doivent pas être
  supprimés sous une étiquette de périmètre. W12 reste notamment non reçu.

## Trois extensions communes prioritaires

### 1. Produits globaux d'inconnues et élimination/reconstruction composables

Impact : T3/C15–19, W06, M09 puis M27, et couplages champ–matière. Commencer par
le **vrai** témoin fini M09 avec son application G:4→8 et D=−Gᵀ, déclaré comme
petit opérateur à supports/dimensions explicites. L'extension ne doit pas
transformer G en gradient de maillage ou dérouler un domaine HPC en code.

Réutiliser le provider global/matrix-free et les problèmes de champ existants ;
donner au résidu fermé ses inconnues, captures, applications et relations de bord
exactes. Une condensation devient une composition de bibliothèque sous
obligation d'inversibilité vérifiée ; reconstruire puis contrôler **toutes** les
équations originales. Conserver un solve monolithique lorsque le pivot choisi
est singulier mais le système complet inversible.

Première réception : M09 mono/condensé <1e-11, deux seeds/captures distincts,
permutation, pivot singulier, noyau compatible/incompatible, erreur rang-local
sans publication. Puis sous-cas M27 mixte périodique `(c,mu)` au lieu de cacher
un opérateur d'ordre quatre. Seams : `time/_program/local_product.py`,
`codegen/program_emit_local_product.py:product_operator_environments`,
`fields/residual.py`, `codegen/program_emit_solve.py` et l'ancien
`program_emit_condensed.py`. Les refus de provider au candidat et snapshots
multiples sont de vraies obligations IMPL, pas une preuve que le problème est local.

### 2. Opérateurs différentiels couplés avec parties dissipative et dispersive distinctes

Impact : T2/C14, M05, M23, puis composantes de M12/M20 quand leur physique sera
fermée. Ne pas réimplémenter la diffusion diagonale ou TensorDiffusion déjà
présentes. Étendre les constructions gradient/divergence et flux joints avec un
contrat d'appariement, de mesure et de traces explicite pour les composantes.

Le premier discriminant fermé est M23 : la rotation de Fourier transverse
prescrite, k=2, etaH=.3, temps=.2, norme et phase séparées, limite etaH=0.
L'antisymétrie concerne ici le couplage des composantes ; un tenseur spatial
antisymétrique scalaire constant n'est pas un substitut. Il faut préserver son
symbole dispersif, sans le forcer dans un validateur SPD ni ajouter une dissipation
pour passer le test. En miroir, le sous-cas M05 de cisaillement diffusif doit
vérifier l'identité de dissipation et les flux de bord déclarés.

Seams : `numerics/diffusion.py:TensorDiffusion` exige actuellement A et dW/dU SPD
et des frontières périodiques ; `codegen/program_emit_diffusion.py` refuse
l'explicite AMR sans borne composite prouvée. Étendre la réalisation et ses
preuves, pas supprimer ces gardes. Contrôles : symbole/phase indépendant,
permutation, rotation, bilan partagé, trace/conormal et refus collectif ; preuve
AMR d'adjoint/énergie distincte du seul reflux conservatif.

### 3. Frontières atteintes et échanges d'intervalle au travers des transactions

Impact : T4/T5, C20–24/C29/C33–40, M14/W11, W12 puis M28. Le premier incrément
fermé est le témoin circuit normalisé : C=1, charge initiale .7, enfant produisant
un courant puis parent rejeté. Le courant intégré, la charge, les historiques,
le ledger et le front temporel doivent tous rester inchangés ; seconde
consommation interdite. Aucun nouveau modèle de gaine n'est nécessaire.

Ajouter le témoin W12 dépendant de la **durée effective** : après rejet/retry avec
une autre durée, aucune contribution mise en cache sur la seule date ne doit
survivre. Ensuite composition avec sous-cyclage, remplacement d'échange, regrid
préparé puis rejeté et reprise. Exiger une règle publique pour tout endpoint
calculé/relaxé ; l'ExternalTimeGrid exact actuel ne fournit pas cette règle.

Réutiliser `program_cadence_continuation.inc`, les continuations de hiérarchie,
les leases et les journaux réels. `program_emit_hierarchy_regions.py` refuse
encore le Path planifié/niché et la coexistence de certains solves spatiaux ; ces
restrictions restent IMPL. Ne pas prendre les transactions de sortie NPZ comme
preuve de conflits read/write de racines PDE. Tester abandon d'enfant, rejet
parent, lecture périmée, MPI rang seul et destruction avec travail physiquement
inachevé. La preuve GPU devra attendre un événement réel sur matériel disponible.

## Ordre de réception et verdict

Avant d'élargir le cœur : terminer la réception actuellement lancée des halos,
paramètres qualifiés, captures read-only, M15 MPI, M11 et M18, puis mettre le
registre à jour avec ces reçus exacts. Les trois extensions ci-dessus peuvent
ensuite partir de cette base reçue, chacune avec rouge, mécanisme commun,
contre-revue et campagne native limitée avant élargissement.

Le benchmark 3d06cab→cf6dace (ratio médian .978179) reste utile pour ce calcul
historique uniquement. `kernels=36` mesure des batches Program ; scratch, halo,
MPI et lancements GPU absents ne valent pas zéro. Aucune validation performance
globale, GPU, OpenMPI, résilience à la perte d'un processus ou durabilité après
coupure n'est acquise. La mission entière demeure ouverte pour des raisons
identifiables d'implémentation, de réception et, pour certains cas, de définition
scientifique manquante.
