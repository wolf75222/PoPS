"""Spectre du taux de croissance diocotron : eigensolver radial vs theorie.

Trace gamma_l(l) calcule par l'eigensolver radial (examples/diocotron_theory,
fichier CSV) et superpose les valeurs de reference de Hoffart-Maier-Shadid-Tomas
(arXiv:2510.11808, fig 5.4) pour la geometrie (a,b,R)=(6,8,16).

Usage :
  ./build/bin/diocotron_theory 6 8 16 /tmp/theory.csv
  python scripts/plot_diocotron_theory.py /tmp/theory.csv docs/fig_diocotron_theory.png
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# valeurs theoriques publiees (Davidson-Felice / Hoffart et al. fig 5.4)
PAPER = {3: 0.772, 4: 0.911, 5: 0.683}


def main() -> int:
    if len(sys.argv) < 3:
        print(__doc__, file=sys.stderr)
        return 1
    csv = Path(sys.argv[1])
    out = Path(sys.argv[2])
    out.parent.mkdir(parents=True, exist_ok=True)

    data = np.loadtxt(csv, delimiter=",", skiprows=2)
    ell, gamma = data[:, 0], data[:, 1]
    keep = gamma > 1e-6  # modes instables (l >= 6 stables ici)

    fig, ax = plt.subplots(figsize=(7, 4.6), constrained_layout=True)
    fig.patch.set_alpha(0)
    ax.set_facecolor("none")
    ax.plot(ell[keep], gamma[keep], "o-", color="C0", lw=2, ms=7,
            label="eigensolver radial (nous)")
    ax.plot(list(PAPER.keys()), list(PAPER.values()), "kx", ms=11, mew=2.5,
            label="Hoffart et al. 2025 (fig 5.4)")
    ax.set_xlabel(r"mode azimutal $\ell$")
    ax.set_ylabel(r"$\gamma_\ell / \omega_D$")
    ax.set_title("Taux de croissance diocotron : theorie lineaire (a=6, b=8, R=16)")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.savefig(out, dpi=130, transparent=True)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
