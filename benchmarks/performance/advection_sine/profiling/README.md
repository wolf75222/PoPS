# Profilage macOS — protocole complet, hors scaling

Cette acquisition répond à une question différente du scaling : quelles piles
CPU dominent le cycle de vie Python public sur macOS ? Elle n’écrit jamais dans
`results/` et ses temps ne peuvent pas être ajoutés aux summaries de scaling.

Le seul point autorisé est lu depuis `../campaigns/strong_openmp.json` :
`t8`, 3D `128³`, blocs `32`, CFL `0.40`, `32` pas, un warmup, cinq répétitions,
huit threads et un rang. Toute dérive du JSON versionné est refusée.

## Acquisition

Le runner n’accepte aucune commande de cas fournie par l’appelant : il accepte
seulement `--python`, `--campaign` et `--output`, puis construit lui-même la
liste exacte d’arguments de `advection_sine.py` pour le point canonique. Chaque
feuille reçoit son propre `rank-output/`, donc les dix processus ne peuvent pas
réutiliser le même JSON de rang.

Avant toute cible, le runner crée une archive et un manifeste content-addressed
du tree Git-visible, extrait cette archive dans la nouvelle sortie et vérifie
l’extraction. La cible s’exécute depuis cet export, pas depuis le worktree qui
pourrait changer. READY lie le `tree_sha256`/manifeste ainsi que les fichiers du
cas, support, helpers et protocole. Il joint également le reçu de la variante
native réellement chargée : SHA de l’extension, SHA de `variants.json`,
dimension, MPI, Kokkos, version et ABI. Le collecteur exige ces faits identiques
sur les dix feuilles, ainsi qu'OpenMP/Kokkos à huit threads, MPI inactif et un
seul rang pour le point `t8`.

La variante native est reconstruite par le runner depuis cet export, dans
`<output>/native-build`, avant toute cible. Il faut uniquement fournir
`POPS_MACOS_PROFILE_KOKKOS_ROOT`; `POPS_MACOS_PROFILE_CMAKE`,
`POPS_MACOS_PROFILE_NINJA` et `POPS_MACOS_PROFILE_CXX` peuvent préciser les
outils absolus. La configuration est Release, Dim3, Kokkos/OpenMP, sans MPI ni
HDF5 et avec le Python `--python` exact. Le runner fixe
`POPS_INCLUDE=source-tree/include`, `POPS_CACHE_DIR` et `XDG_CACHE_HOME` sous
la nouvelle sortie, puis écrit `build.receipt.json`. `CMAKE_HOME_DIRECTORY`
doit désigner ce `source-tree` fraîchement exporté : une variante native
extérieure ne peut donc pas être présentée comme construite depuis la source
profilée.
La compilation utilise quatre jobs au plus par défaut; régler
`POPS_MACOS_PROFILE_BUILD_JOBS` entre 1 et 32 ne change que les ressources de
build, jamais le workload profilé.

Le processus de cas public doit, après compilation, bind et warmup, appeler
`ready_go.ready_after_bind_warmup`, attendre `ready_go.await_go`, exécuter une
seule répétition complète, puis appeler `completed_public_lifecycle`. Le nonce
READY/GO est unique; les fichiers préexistants sont refusés.

`run_macos_profile.sh` démarre cinq nouveaux processus par outil :

- `/usr/bin/sample` : cinq feuilles `sample/rep01` à `rep05` ;
- `/usr/bin/xctrace record --template 'Time Profiler' --attach PID` : cinq
  bundles `xctrace/rep01/time-profiler.trace` et leurs TOC exportés.

Le READY contient obligatoirement la campagne/hash, le SHA et l’état dirty de
la source, l’identité sémantique de l’artifact, le rapport runtime, l’hôte et
la liste/hash de commande canonique. Le script vérifie ces éléments, le reçu
`command.json` et le résultat `rank-00000.json` de chaque feuille, puis produit
des reçus sans écrasement.

Pour `sample`, la durée plafond est longue (600 s par défaut), avec `-mayDie`
et `-fullPaths`. Le reçu d’acquisition prouve que le processus public complet
s’est terminé et a été attendu par le lanceur *pendant* cet intervalle, et non après une
capture d’une seconde. Avant GO, le lanceur arrête la cible, attend l’état
arrêté, puis exige l’en-tête écrit par `sample`; seulement ensuite il publie GO
et relance la cible. `kill -0` seul n’est donc pas accepté comme preuve
d’attachement.
Pour Time Profiler, `notifyutil` attend le signal réel
`--notify-tracing-started` de `xctrace`; seulement alors le script publie GO.
`--quiet`, `--no-prompt` et `--time-limit` rendent l’acquisition non
interactive et bornée. L’apparition d’un dossier `.trace` n’est jamais une
preuve d’attachement.

L'attente READY vaut 1200 s par défaut (configurable avec
`POPS_MACOS_PROFILE_READY_TIMEOUT_SECONDS`, minimum 300 s) pour laisser le
temps au vrai cas 128³ de compiler/binder/chauffer. Elle ne réduit jamais le
workload. Un trap réveille puis termine une cible arrêtée ou en attente de GO
si l'acquisition échoue/interrompt le script.

Le script attend le READY post-warmup, vérifie le PID, attache l’outil, crée GO
avec le nonce identique, puis exige les reçus de processus et d’acquisition. Il
ne possède pas de solution numérique alternative.

```bash
bash benchmarks/performance/advection_sine/profiling/run_macos_profile.sh \
  --python /chemin/python-authentifie \
  --output /chemin/nouveau/profile-YYYYMMDD
```

## Collecte et figures

`collect_profiles.py` exige les dix feuilles, cinq reçus complets par outil,
des nonces uniques, une provenance identique, des rapports `sample` non vides,
et les TOC Time Profiler dont la table est strictement `time-profile`. Chaque
bundle `.trace` est inventorié et hashé récursivement. Le parser `sample` ne
compte que les feuilles repliées de sa section `Call graph`, jamais les compteurs
inclusifs des ancêtres.

`plot_profiles.py` crée PNG/SVG du top-15, de la composition par image, et un
icicle des piles de feuilles `sample`. Cet icicle dérive seulement des
indentations effectivement présentes dans `Call graph` et de leurs poids
exclusifs : il ne reconstitue pas une hiérarchie à partir de compteurs isolés.
Aucun fichier image n’est livré avant une collecte réelle. Après collecte,
`COMPLETE.json` scelle cinq `sample` + cinq `xctrace`, source, build, extension,
reçus, traces et `summary.json`. Les figures sont ensuite produites dans un
staging puis publiées atomiquement avec manifeste et SHA-256 dans le répertoire
frère `<profil>.figures`, jamais sous la racine scellée `<profil>`. Le runner
revérifie `COMPLETE.json` après cette publication; le traceur refuse un
`summary.json` non couvert par ce seal.
