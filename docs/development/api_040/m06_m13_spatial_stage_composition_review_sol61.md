# M06/M13 : composition d'un stage avec le résidu spatial original

Réception indépendante **source seulement**, sur production
`bfae73f34174079bc6f2d8f4d50aa11d08eca5b5`. Checkout exclusif :
`/Users/romaindespoulain/dev/tmp/pops-sol61-m06-spatial-composition-review`.
Aucun build, JIT, appel natif, installation, modification de MAIN ou de
l'environnement. Deux fichiers tests/docs ; aucune correction de production.

## Contrat original

Le handoff intact `PoPS_Codex_handoff_0.4.0/context/CORPUS_ORIGINAL.md`,
lignes 186 et 200, demande respectivement accumulation énergétique puis front
de fusion (M06), et chimie, transport, Poisson et contribution radiative non
locale distincts (M13). Les sous-cas homogènes existants ne ferment aucun de ces
fronts spatiaux.

Les Notes originales `math/PoPS_Abstraction_Notes/notes_on_the_abstraction_of_pops.tex`,
lignes 402–446, donnent

\[
U=\mathcal Q(q),\quad \partial_tH(T)=\nabla\!\cdot(k(T)\nabla T)+s,
\qquad H(T^+)-H(T^n)=\tau P.
\]

Pour `H(T)=T+T²`, `Tn=0`, `tau*P=1`, le vrai bilan donne
`T+=(sqrt(5)-1)/2` et une énergie de 1 ; la substitution par
`H'(T+)*(T+-Tn)=tau*P` donne `T+=1/2`, énergie `3/4`.
La relation de phase reste nécessaire quand T n'est pas une coordonnée complète.
Ces équations ne justifient pas une inversion imposée de H.

Les Notes 558–569 conservent séparément la relation globale
`A(Phi;U,eta,Y,d,t)=ell(U,eta,Y,d,t)`, ses conditions et sa normalisation.
Les signatures originales `spec/public_reference.pyi:104,158–168` exposent
`Model.field_problem(...)`, `Program.dt`, inconnues, captures `bind(...)` et
solve des relations. C15/C16 de `document/02_contracts.tex:64–70` distinguent
inconnues, données gelées et expressions qui restent dans le résidu.
Le corpus M13 ne fournit pas une fermeture quantitative de photoionisation :
aucun Helmholtz, coefficient chimique ou budget photonique n'est inventé ici.

## Verdict précis sur les routes actuelles

Six contrôles publics source passent :

* `ImplicitStage` possède déjà `Q(q)-U_n-tau*R(Q(q))`. Deux cas réellement
  validés/résolus/émis conservent `tau=Program.dt` et `tau=Program.dt/2`, avec
  polynôme exact `{1: 1}` / `{1: 1/2}`. Ils évaluent la conduction sur Q(q),
  consomment le résultat, conservent T en histoire et committent **Q(T+)**.
  La primitive inverse utilisée dans ce contrôle est un choix de ce modèle
  lisse, pas une exigence générale ou une solution pour le plateau latent.
* Le `FieldProblem` original
  `Reaction(T,1+T)-DivCoeffGrad(T,.01)==H_old`
  forme exactement `T+T²-.01*div(grad T)-H_old`. Le capture H_old est le vrai
  état au point n, séparé du seed. Case/validate/resolve/émission passent et
  une température consommée est observable en histoire. Le contrôle conserve
  volontairement l'ancien H : il **n'atteste pas** un nouveau pas énergétique.
* La projection `cell_mean_state` existante est reçue positivement sur le
  problème linéaire public M27, avec owner/StateSpace/endpoint exacts.
* Remplacer `.01` par le vrai `Program.dt` dans le coefficient du problème
  original est refusé : `FieldProblemError`,
  `diffusion requires exact State captures; unknown-dependent D has no full spatial nonlinear JVP realization`.
  Ici la donnée refusée est le coefficient temporel SSA, **pas** k(T).
  Le diagnostic actuel ne distingue pas ces deux extensions.
* Le résultat original non linéaire est une observation `scalar_field` sans
  owner/StateSpace de l'état évolué. `cell_mean_state` refuse la méthode non
  linéaire, même avec support explicite, `cell_average` et endpoint corrects :
  `projection requires the native cell-centred finite-volume stencil`.
  Aucun node ni commit n'est créé après ce refus. Déqualifier les gardes ou
  recopier cette observation comme énergie serait incorrect.

Il manque donc la **liaison d'un stage temporel avec le problème spatial
original et sa projection d'accumulation**, pour cette route. Cela ne signifie
pas que M06 lisse est impossible : la route ImplicitStage précédente existe.
`D(candidate)` est une extension distincte confiée à l'auteur Euler ; aucun
correctif ni nouvelle réception de cette extension n'est compris ici.

## Une seule correction de production proposée

Ajouter un adaptateur versionné, par exemple `EvolvedOriginalFieldStage@1`, qui
lie le tuple d'inconnues du problème original à : l'état conservé n gelé,
la relation Q déclarée, les captures exactes, le polynôme de durée du Program
et l'endpoint conservé. Il doit préparer le résidu **original**
`Q(q)-U_n-tau*R(q,Phi;d)` et conserver les contraintes spatiales de Phi dans
le même problème, puis autoriser une projection explicite `Q(q+)` vers l'état
après consumption. L'adaptateur ne doit pas ajouter une température évoluée
fictive, un carrier State de dt, un dt RuntimeParam égal par convention, une
division par un dt littéral, ni remplacer le résidu par une chaîne différentielle.

Fichiers concernés, à examiner dans une future implémentation cohérente :

* Nouveau `python/pops/time/evolved_field_stage.py`, export
  `python/pops/time/__init__.py`, dispatch
  `python/pops/time/_program/solve_request.py` : binding explicite du stage,
  des coordonnées et des captures ; ancien ImplicitStage inchangé.
* `python/pops/fields/_program_nonlinear_problem.py` et `_program_expression.py` :
  contrat versionné de durée SSA exacte, distinction des captures State gelées
  et du coefficient temporel ; identité scientifique indépendante du seed.
* `python/pops/fields/_program_problem.py` et `_observation_contract.py` :
  projection versionnée de l'accumulation et de son support vers l'endpoint.
  Ne pas élargir implicitement l'identité linéaire de moyenne de cellule à
  `mean(H(T))=H(mean(T))`.
* `python/pops/codegen/program_emit_nonlinear_field.py` et
  `program_emit_amr_original_field.py` : lier la durée effective du trial/stage,
  conserver le full-residual JVP, le recheck original et le staging commun de
  Q(q+), champs, histoires et échanges. Réutiliser les workspaces/providers
  existants ; tout nouveau seam natif doit rester versionné et explicitement reçu.

Coût de ce binding : une transformation Q par cellule pour le résidu/endpoint,
des métadonnées O(nombre de captures/composantes), et le staging de la sortie
physique ; les solves globaux et échanges de halos restent ceux du problème
déclaré. Aucune prétention de coût fixe pour une fermeture radiative ou un
préconditionneur non linéaire encore non spécifiés. Les votes de finitude,
contrôle et publication doivent rester convergés, y compris empty ranks ;
aucune collective dans le callback Kokkos local. Conserver byte-identiques
les anciennes routes sans ce binding. Rollback inclut état, champs, histoire,
ledger, cache et durée ; aucun transfert n'est publié par une simple terminaison
du solve.

M13 pourrait utiliser cette composition pour un choix **déclaré** de couplage
implicite avec des champs auxiliaires. Les solves de Poisson/nonlocal au stage
connu et leurs publications existent déjà ; ni leur présence ni ce futur
adaptateur ne qualifient le streamer, la radiativité, ses frontières ou son
bilan photonique. Le choix des équations et de l'intégration reste séparé.

## Identité et commande

Git blobs exacts reçus sur bfae73f3 :

| Fichier | Git blob |
|---|---|
| `python/pops/time/implicit_stage.py` | `e6cce4cd62374856eb9e1654ae0ca25727db1e20` |
| `python/pops/time/_program/spatial_solve.py` | `3a64faa9597be96e186e5d110c70177be06a541f` |
| `python/pops/fields/_program_problem.py` | `a007a948f595a8111ffbf51cb8f8f04fc34fece4` |
| `python/pops/fields/_program_nonlinear_problem.py` | `0050ae51999c6ef1e53559bc90bb51620e8aa049` |

```sh
cd /Users/romaindespoulain/dev/tmp/pops-sol61-m06-spatial-composition-review
rtk proxy env -u PYTHONPATH -u POPS_NATIVE_DIM PYTHONPATH=python \
  PYTHONDONTWRITEBYTECODE=1 \
  /Users/romaindespoulain/miniforge3/envs/pops-api040/bin/python -B -m pytest -q \
  tests/review/test_sol61_m06_spatial_composition.py --tb=short
```

Résultat final : **6 PASS, 5.39 s** ; Ruff et `git diff --cached --check` passent.
Le premier import depuis le checkout Documents a été arrêté par SIGINT après
environ trois minutes dans `importlib.get_data` avant tout cas ; il n'est pas
compté comme résultat numérique. Le contre-probe source hors Documents bfae
a fini en moins d'une seconde, sans module `pops._bootstrap`/`pops._pops`
chargé. La suite ci-dessus ne prétend pas qualifier une bibliothèque native,
un pas exécuté, MPI, AMR, Stefan, ni M13 complet. Aucun owner pin d'un vrai
run n'est créé ou remplacé.
