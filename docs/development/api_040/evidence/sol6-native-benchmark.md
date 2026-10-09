# Comparaison microbenchmark des reconstructions natives

2026-09-29, checkout `work/PoPS`, baseline `3a93ba7f` via `git show HEAD:include/pops/numerics/fv/reconstruction.hpp`; actuel = diff de la branche partagée avant commit. Code benchmark et CSV dans `outputs/reconstruction_bench.cpp`, `reconstruction-bench-baseline.csv`, `reconstruction-bench-current-optimized.csv`. Le TU C++ est compilé deux fois avec le même clang++ macOS, `-O3 -std=c++20 -Iinclude`; aucune option MPI/Kokkos dans ce probe CPU isolé. Seed `0x6040`, 4096 échantillons × 512 passages par répétition, un échauffement puis cinq mesures. Les appels de pente voient des données aléatoires bornées; WENO lisse voit `1+0.1 sin` et WENO extrême une pente d'amplitude voisine de `max/16`.

Critère fixé avant mesure: signaler une médiane du cas ordinaire plus lente de >20 % avec des intervalles interquartiles séparés. Ce seuil est un déclencheur d'analyse, pas un budget de réception établi par le dépôt. La validation numérique prime; aucune tolérance de test n'a été adaptée aux temps.

| Formule ordinaire | Baseline médiane (ns/appel) | Actuel médiane (ns/appel) | Ratio indicatif |
| --- | ---: | ---: | ---: |
| Minmod | 1.082 | 2.251 | 2.08× |
| Van Leer | 1.359 | 2.454 | 1.81× |
| MC | 1.306 | 2.221 | 1.70× |
| Superbee | 1.311 | 2.338 | 1.78× |
| WENO5-Z lisse | 3.249 | 10.141 | 3.12× |

Il y a des valeurs extrêmes de temps dans la baseline (MC 8.647 ns, Superbee 5.275 ns) dues au bruit du système ou à la contention; les médianes ne sont donc pas des garanties. Le WENO lisse est plus stable: baseline 3.160–3.501 ns, actuel 10.141–11.131 ns dans les deux petits relevés successifs. Le chemin actuel ajoute validations et sélection de repli avant l'ancien calcul. Une variante évaluant les magnitudes à chaque appel a été mesurée et écartée, car WENO lisse passait à ~12.526 ns médian (`reconstruction-bench-current-magnitudes.csv`). Le code conservé correspond au relevé `current-optimized.csv`.

Le binaire benchmark Mach-O fait 34432 octets avec le header baseline, 34496 avec le header actuel; la section `__TEXT` de `size` est arrondie à 16 KiB dans les deux cas et n'explique pas le coût des instructions. Le test WENO extrême n'est pas comparable en temps sémantique : les appels baseline produisent NaN, les appels actuels produisent des faces individuellement finies; l'accumulateur du benchmark déborde vers Inf après des millions d'additions. Les probes directs `weno_red_probe.cpp` vérifient la valeur finie de la reconstruction extrême indépendamment de cet accumulateur.

Ce résultat mesure des fonctions CPU isolées, pas un pas PDE, un noyau Kokkos, MPI ou GPU. Le surcoût mesuré est matériel au niveau formule et justifie un benchmark solver avant un verdict de performance. Il ne réfute pas les corrections mathématiques : `denorm_min`, `max`, le bord WENO float32 et `eps<0` étaient réellement défectueux en baseline.
