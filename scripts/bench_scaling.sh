#!/usr/bin/env bash
# Balayage de scaling OpenMP du banc bench_amr (deux-fluides AP mono-grille).
#
# Lance bench_amr (mode "tf", sans I/O) pour quelques tailles de grille et quelques
# nombres de threads OpenMP, et ecrit sur stdout un CSV au format attendu par
# scripts/plot_bench_scaling.py :
#
#     n,threads,mcells_per_s
#
# C'est purement INFORMATIF (mesure de debit), jamais une garde bloquante : aucune
# assertion de temps ici. Les gardes DURES de non-regression vivent en CTest
# (bench_amr_smoke : conservation de masse ; arena_kokkos_no_malloc : 0 allocation).
#
# Usage :
#   bash scripts/bench_scaling.sh > /tmp/scaling.csv
#   python scripts/plot_bench_scaling.py /tmp/scaling.csv docs/fig_openmp_scaling.png
#
# Binaire : 1er argument, sinon $BENCH_AMR, sinon build-omp/bin/bench_amr (relatif au
# depot). Listes ajustables : $SIZES (tailles n), $THREADS (threads), $NSTEPS (pas).

set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo="$(cd "${here}/.." && pwd)"

bench="${1:-${BENCH_AMR:-${repo}/build-omp/bin/bench_amr}}"
if [[ ! -x "${bench}" ]]; then
  echo "bench_amr introuvable ou non executable : ${bench}" >&2
  echo "compile d'abord (ex. cmake --build build-omp --target bench_amr) ou passe le chemin en argument." >&2
  exit 1
fi

# Defauts modestes (rapides sur un portable) ; surchargeables par l'environnement.
sizes="${SIZES:-128 256 512}"
threads="${THREADS:-1 2 4 8}"
nsteps="${NSTEPS:-50}"

echo "n,threads,mcells_per_s"
for n in ${sizes}; do
  for t in ${threads}; do
    # mode "tf" : deux-fluides AP mono-grille seul, pas d'AMR, pas d'I/O.
    line="$(OMP_NUM_THREADS="${t}" "${bench}" "${n}" "${nsteps}" tf \
            | grep 'two-fluid AP (MG)' || true)"
    # Extrait le debit : champ juste avant "M mailles-MAJ/s".
    mcells="$(printf '%s\n' "${line}" \
              | sed -n 's/.*| \([0-9.]*\) M mailles-MAJ\/s.*/\1/p')"
    if [[ -n "${mcells}" ]]; then
      echo "${n},${t},${mcells}"
    else
      echo "ligne MG non analysable pour n=${n} threads=${t}" >&2
    fi
  done
done
