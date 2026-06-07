# Manifeste des assets (adc_cpp)

Ce document recense les images suivies par git sous `docs/` du dépôt `adc_cpp`,
leur surface de référence réelle, leur producteur connu et une décision de
gestion. Il existe parce que la quasi-totalité de ces assets ont été produits
hors de leur chemin committé et **ne portent aucune provenance enregistrée**
(SHA `adc_cpp`, backend, résolution, commande de génération). Le seul jeu
d'assets *avec* provenance traçable est celui du tutoriel canonique
`docs/sphinx/tutorials/_assets/`, documenté en section finale.

## Périmètre

Le glob `docs/*.png` + `docs/*.gif` (racine `docs/`, hors `docs/_build/`,
hors `docs/sphinx/tutorials/_assets/`) compte **33 images** : **20 PNG + 13
GIF**. La colonne « Référencé par » provient d'un `grep` des fichiers `.md`
(en excluant `docs/_build/`). `docs/DOC_REFONTE_AUDIT.md` est le document
d'audit qui *catalogue* tous ces fichiers ; il n'est pas une surface de doc
vivante et n'est donc pas compté comme référence d'affichage ci-dessous.

## État de la provenance

**Aucune** des 33 images de `docs/` ne porte de provenance enregistrée. Pour
chacune, on ignore : le SHA `adc_cpp` au moment de la génération, le backend
(prototype / aot / production), la résolution de la grille, le nombre de pas,
et la commande exacte qui l'a produite. Les figures `tut_*` et la galerie
`fig_*`/`anim_*` ont été committées comme artefacts sortis d'un pipeline
local, pas reconstruites à leur chemin de dépôt.

La surface de doc *vivante* ne référence pour affichage que **deux** de ces
33 fichiers :

- `anim_romeo_diocotron_amr3.gif` — embed HTML du hero `README.md:12` ;
- `fig_openmp_scaling.png` — embed markdown `docs/PERFORMANCE.md:99`.

Tout le reste est soit archive-only (`docs/archive/*.md`), soit orphelin
(ne reste référencé que dans le document d'audit, ou plus du tout).

## Légende des décisions

- **keep** : asset d'une surface vivante ; à conserver. Provenance à
  enregistrer (faute de quoi il reste non reproductible).
- **regenerate-with-provenance** : à reconstruire via un script versionné
  émettant un `provenance.json`, si l'asset doit revenir dans la doc.
- **move-to-archive** : asset uniquement utile aux pages d'archive ; à
  conserver avec l'archive (idéalement sous `docs/archive/assets/`).
- **delete-orphan** : plus aucune référence vivante ; candidat à suppression.

## GIF (13)

| Fichier | Référencé par (hors `_build`, hors audit) | Producteur | Décision |
|---|---|---|---|
| `anim_romeo_diocotron_amr3.gif` | `README.md` (hero, l.12) | inconnu — run ROMEO/GH200 supposé, non documenté | **keep** — seul GIF de la surface vivante ; provenance ROMEO à enregistrer |
| `anim_magnetic_diocotron.gif` | `docs/archive/ROADMAP.md` | inconnu | **move-to-archive** |
| `anim_diocotron.gif` | aucune (audit seul) | inconnu — ex-galerie Sphinx morte | **delete-orphan** ou regenerate-with-provenance si réutilisé |
| `anim_diocotron_column.gif` | aucune (audit seul) | inconnu — ex-galerie morte | **delete-orphan** |
| `anim_diocotron_amr3.gif` | aucune (audit seul) | inconnu — ex-galerie morte | **delete-orphan** |
| `anim_diocotron_multipatch.gif` | aucune (audit seul) | inconnu — ex-galerie morte | **delete-orphan** |
| `anim_diocotron_amr.gif` | aucune | inconnu | **delete-orphan** |
| `anim_diocotron_mpi.gif` | aucune | inconnu | **delete-orphan** |
| `anim_python_amr.gif` | aucune | inconnu | **delete-orphan** |
| `tut_diocotron_py.gif` | aucune | inconnu — ex-tutoriels Sphinx (supprimés, commit 194c63f) | **delete-orphan** ou regenerate-with-provenance |
| `tut_diocotron_ring.gif` | aucune | inconnu — ex-tutoriels | **delete-orphan** ou regenerate-with-provenance |
| `tut_ep_collapse.gif` | aucune | inconnu — ex-tutoriels | **delete-orphan** ou regenerate-with-provenance |
| `tut_tfap_field.gif` | aucune | inconnu — ex-tutoriels | **delete-orphan** ou regenerate-with-provenance |

## PNG (20)

| Fichier | Référencé par (hors `_build`, hors audit) | Producteur | Décision |
|---|---|---|---|
| `fig_openmp_scaling.png` | `docs/PERFORMANCE.md` (l.99) | `scripts/plot_bench_scaling.py` (cité dans PERFORMANCE.md:92) | **keep** — seule PNG d'une surface vivante ; suit la décision PERFORMANCE.md, provenance à enregistrer |
| `fig_diocotron_amr_vs_uniforme.png` | `docs/archive/ROADMAP.md` | inconnu | **move-to-archive** |
| `fig_diocotron_conv_modes.png` | `docs/archive/DIOCOTRON_GROWTH_RATE.md` | inconnu | **move-to-archive** |
| `fig_diocotron_highorder.png` | `docs/archive/DIOCOTRON_GROWTH_RATE.md` | inconnu | **move-to-archive** |
| `fig_diocotron_invariants.png` | `docs/archive/DIOCOTRON_GROWTH_RATE.md` | inconnu | **move-to-archive** |
| `fig_diocotron_ml_convergence.png` | `docs/archive/ROADMAP.md` | inconnu | **move-to-archive** |
| `fig_diocotron_reproduction.png` | `docs/archive/ROADMAP.md` | inconnu | **move-to-archive** |
| `romeo_amr_efficiency.png` | `docs/archive/ROMEO.md` | inconnu — run ROMEO supposé | **move-to-archive** |
| `romeo_growth_mode4.png` | `docs/archive/ROMEO.md` | inconnu — run ROMEO supposé | **move-to-archive** |
| `romeo_highorder_convergence.png` | `docs/archive/ROMEO.md` | inconnu — run ROMEO supposé | **move-to-archive** |
| `fig_diocotron_growth.png` | aucune (audit seul) | inconnu — ex-galerie morte | **delete-orphan** |
| `fig_diocotron_modes.png` | aucune (audit seul) | inconnu — ex-galerie morte | **delete-orphan** |
| `fig_diocotron_column_growth.png` | aucune | inconnu | **delete-orphan** |
| `fig_diocotron_theory.png` | aucune | inconnu | **delete-orphan** |
| `tut_diocotron_growth.png` | aucune | inconnu — ex-tutoriels (commit 194c63f) | **delete-orphan** ou regenerate-with-provenance |
| `tut_diocotron_sequence.png` | aucune | inconnu — ex-tutoriels | **delete-orphan** ou regenerate-with-provenance |
| `tut_euler_poisson.png` | aucune | inconnu — ex-tutoriels | **delete-orphan** ou regenerate-with-provenance |
| `tut_plasma.png` | aucune | inconnu — ex-tutoriels | **delete-orphan** ou regenerate-with-provenance |
| `tut_poisson_backends.png` | aucune | inconnu — ex-tutoriels | **delete-orphan** ou regenerate-with-provenance |
| `tut_tfap_ap.png` | aucune | inconnu — ex-tutoriels | **delete-orphan** ou regenerate-with-provenance |

## Synthèse

- **2 keep** : `anim_romeo_diocotron_amr3.gif` (README), `fig_openmp_scaling.png`
  (PERFORMANCE.md). Surface vivante ; provenance à enregistrer.
- **10 move-to-archive** : figures `fig_diocotron_*` et `romeo_*` plus
  `anim_magnetic_diocotron.gif`, référencées uniquement par `docs/archive/*.md`.
- **21 orphelins** : les 10 fichiers `tut_*` (ex-pool des tutoriels Sphinx
  partis vers `adc_cases`, suppression `tutorials/` commit 194c63f) plus les
  ex-images de la galerie morte et autres `anim_*`/`fig_*` sans référence. Pour
  chacun : **delete-orphan**, ou **regenerate-with-provenance** si l'asset doit
  revenir dans la nouvelle galerie/tutoriel.

Les `tut_*` n'ont **aucune** provenance et ne sont plus référencés nulle part :
ils sont déjà entièrement orphelins indépendamment de toute refonte.

## Assets du tutoriel canonique (avec provenance)

Contrairement à ce qui précède, le tutoriel A→Z vit sous
`docs/sphinx/tutorials/` et **embarque sa provenance**. Le script
`docs/sphinx/tutorials/diocotron_tutorial.py` régénère ses 4 images et écrit
`docs/sphinx/tutorials/_assets/provenance.json` à chaque exécution.

Provenance commune (extraite de `provenance.json`) :

- script : `docs/sphinx/tutorials/diocotron_tutorial.py`
- commande : `python diocotron_tutorial.py --n 96 --steps 60`
- SHA `adc_cpp` : `e58b513d2245c9258a8720b91830b9ee95cafde9`
- backend de compilation : `aot`
- backend d'exécution : `serial` (défaut ; cf. getting_started pour Kokkos/MPI)
- résolution : `96x96`, `steps=60`, `cfl=0.4`, Python `3.12.2`
- métriques de contrôle : `growth_factor=1.5212313128`,
  `mass_drift=1.81e-16`, `amr_uniform_max_delta=0.0717869334`

| Fichier | Dimensions | Provenance |
|---|---|---|
| `docs/sphinx/tutorials/_assets/diocotron_growth.png` | 1104x432 | `provenance.json` (clé `assets`) |
| `docs/sphinx/tutorials/_assets/diocotron_cover.png` | 456x432 | `provenance.json` (clé `assets`) |
| `docs/sphinx/tutorials/_assets/diocotron.gif` | 380x360 | `provenance.json` (clé `assets`) |
| `docs/sphinx/tutorials/_assets/diocotron_uniform_vs_amr.png` | 912x432 | `provenance.json` (clé `assets`) |

Le dossier contient aussi les `.so` compilés associés au run
(`diocotron_aot.so`, `diocotron_production.so`), artefacts du même pipeline.

Ce jeu est le modèle à suivre pour toute régénération des assets ci-dessus
marqués **regenerate-with-provenance** : un script versionné, une commande
reproductible, et un `provenance.json` committé à côté des images.
