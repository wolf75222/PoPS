# Contre-réception AMR ghosts — 30 septembre 2026

Candidate exacte : `def8c752fde989734cfc4aace76a447ce3d0d63d`.
Checkout exclusif `PoPS-sol61-dot-all-ghost-review`, branche
`codex/api040-sol61-dot-all-ghost-review`. Aucune mutation de production,
de MAIN ou de l'environnement partagé.

## 46 propriétés déjà reçues, rejouées

Le runner réutilise le scaffold indépendant figé dans
`98e6246c3daab41df8a846168ffa9f3a7da4dd32`, extrait du git object exact.
Il sélectionne le nouveau SHA, un nouveau répertoire de sortie, remplace
l'ancienne assertion d'acceptation des ghosts par un refus et ajoute un
contre-cas replica. Les 46 autres assertions restent inchangées et passent :
couverture AMR 26→13, EB/coverage, composants supplémentaires, identités
Scratch/History/Direct, foreign owner, layouts/largeurs/rank-space,
NaN actif/couvert, produit/sommes locaux/globaux non finis, replicas et rangs
locaux vides. Les vrais noyau/helper/providers/visiteurs et templates natifs
d'identité sont exécutés avec de vraies FieldView/Box/Index.

Les six méthodes legacy Uniform/AMR et le visiteur legacy restent
byte-identiques à 21a56b9. Les hashes du noyau, du helper, du nouveau visiteur
finest, des validators et du provider Uniform sont également identiques au
receipt a77. Seul le provider AMR extrait change :
`4ad5a7017633fb517ed1eeff010bcf57acf3e34f15be08f87728ac5d6e8d7285`.

## Contrat ghosts reçu séparément

Le champ Direct droit possède le même layout, les mêmes composants,
distribution et rank-space que le champ fine gauche, mais ghosts=99 contre 0.
Il refusait auparavant seulement dans Uniform ; il refuse désormais aussi
dans AMR. Les deux nouvelles assertions exigent exactement
`std::invalid_argument("AMR Program dot_all ghost count differs")` et
la séquence `V` sans `S` : convergence de l'erreur locale avant somme.
La seconde injection concerne lane.rank()==1 d'une distribution répliquée,
donc un processus qui ne contribue pas à la contraction.

Le diff de def8c752 place les deux lignes de guard uniquement dans le callback
du nouveau `dot_all`, dans le try déjà convergé ; aucun helper ni visiteur
legacy n'est modifié. La fixture C++ auteur injecte la RHS de ghosts différents
uniquement sur rang zéro et exige le refus collectif. Elle est inspectée ici,
mais son exécution MPI appartient à l'intégrateur.

Résultat : **48 assertions host PASS = 46 propriétés antérieures + 2 refus
ghosts exacts**. ASan/UBSan reçoit le même programme et les mêmes 48 assertions.
Ruff reçoit le runner.

## Captures publiques dt_bound, lot distinct

Les deux contre-cas d'émission publique restent figés dans
`a7ecd3ca47b636aa058514a382786b9d75c58e93` : ancien `u.n` principal capturé
et bloc readonly query-only absent du map des updates. Ce lot ghosts ne reçoit
pas leur réparation. Leur contre-réception attend le prochain gel Python/codegen
et sera accompagnée d'une fixture native publique compile/bind/stabilité/run.

## Reproduction et portée

```sh
rtk proxy /Users/romaindespoulain/miniforge3/envs/pops-api040/bin/python tests/review/sol61_dot_all_ghost_review.py
rtk proxy /usr/bin/clang++ -std=c++20 -O0 -g -fsanitize=address,undefined -Iinclude -I/Users/romaindespoulain/miniforge3/envs/pops-api040/include outputs/sol61-dot-all-ghost-independent/probe.cpp -o outputs/sol61-dot-all-ghost-independent/probe-sanitized
rtk proxy outputs/sol61-dot-all-ghost-independent/probe-sanitized
```

Archive candidate `include` + `python` :
`c42ca112de46f099d89a1063249552d2f79d8a7e2a28f19a4df7ba809d9e234a`.
TU host : `76c8a880474a7e6ed02cdd7331666b250db0f44da32099c5f5d3feef5f26280b`.
Receipt régénérable sous `outputs/sol61-dot-all-ghost-independent/receipt.json`.

Le stockage MultiFab, les facades de hiérarchie, les lanes, collectives et
l'ordonnancement du noyau sont des substituts host explicites. Le refus est
reçu dans les branches extraites et l'ordre des appels simulés ; ce résultat
ne qualifie ni progrès MPI réel, ni Kokkos/GPU natif, ni PDE, ni checkpoint.
