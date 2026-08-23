# Campagnes scientifiques de vérification

Cette arborescence, exclusivement sous `benchmarks/verification/` (et non sous un
`verification/` à la racine), contient des campagnes reproductibles, distinctes du harness C++ de
performance situé directement sous `benchmarks/`. Elles ne publient aucun seuil de performance.
Chaque campagne écrit des données structurées, puis un script de post-traitement indépendant les
relit pour produire les figures.

Le premier cas est [`advection/sine_wave`](advection/sine_wave/README.md). Les étapes futures
proches seront des répertoires de physique semblables (`euler/`, `poisson/`, `amr/`) lorsqu’un cas
scientifique et ses données de référence seront prêts ; aucun répertoire vide n’est créé ici.
