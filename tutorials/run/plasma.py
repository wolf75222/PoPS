#!/usr/bin/env python3
"""Tutoriel exécutable : Euler pour les plasmas (couplage électrostatique répulsif).

Le modèle Euler-Poisson sert la gravité ET le plasma : seul le signe de la source
elliptique change (interaction = Gravity ou Plasma). On le montre sur la dispersion
d'une perturbation acoustique au repos delta_rho = eps rho0 cos(kx).

  panneau A (champ modéré, four_pi_G = 20) : les deux régimes OSCILLENT, mais à des
  fréquences différentes. omega^2 = c_s^2 k^2 -+ omega_p^2 : signe - en gravité (Jeans
  amollit le son), signe + en plasma (Bohm-Gross le durcit). On mesure omega par le
  premier passage à zéro du mode et on compare à la théorie.

  panneau B (champ fort, four_pi_G = 120, omega_p^2 > c_s^2 k^2) : la gravité devient
  INSTABLE (effondrement de Jeans, le mode croît en exp(gamma t)), tandis que le plasma
  reste borné. Un plasma mono-espèce est inconditionnellement stable, jamais d'instabilité.

Usage :
  PYTHONPATH=build-py/python python3 tutorials/run/plasma.py
Sortie : docs/tut_plasma.png
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import adc

ROOT = Path(__file__).resolve().parents[2]

GAMMA, RHO0, P0, L, N = 5.0 / 3.0, 1.0, 1.0, 1.0, 64
CS2 = GAMMA * P0 / RHO0
K = 2.0 * np.pi / L


def mode_series(kind, four_pi_G, eps, nsteps, dt):
    """Amplitude du mode cos(kx) de (rho - rho0) au cours du temps."""
    cfg = adc.EulerPoissonConfig()
    cfg.n, cfg.L, cfg.gamma = N, L, GAMMA
    cfg.four_pi_G, cfg.rho0, cfg.p0 = four_pi_G, RHO0, P0
    cfg.interaction, cfg.eps, cfg.use_fft = kind, eps, False
    s = adc.EulerPoissonSolver(cfg)
    x = (np.arange(N) + 0.5) / N * L
    cosk = np.cos(K * x)[None, :]
    t, m = [], []
    for _ in range(nsteps):
        s.step(dt)
        d = np.asarray(s.density())  # (ny, nx), perturbation en x
        t.append(s.time())
        m.append(2.0 * np.mean((d - RHO0) * cosk))
    return np.array(t), np.array(m)


def first_zero(t, m):
    """Premier passage + -> - du mode, interpolé : omega = pi / (2 t_zero)."""
    for i in range(1, len(m)):
        if m[i - 1] > 0.0 >= m[i]:
            return t[i - 1] + (t[i] - t[i - 1]) * m[i - 1] / (m[i - 1] - m[i])
    return None


def omega_theory(four_pi_G, sign):
    """sign = +1 plasma (Bohm-Gross, +omega_p^2), -1 gravité (Jeans, -omega_p^2)."""
    return np.sqrt(CS2 * K * K + sign * four_pi_G * RHO0)


def main() -> int:
    dt = 0.35 * (L / N) / (np.sqrt(CS2) + 0.1)

    # --- A : champ modéré, les deux oscillent (Jeans stable vs Bohm-Gross) ---
    nA = int(1.0 / dt)
    tg, mg = mode_series(adc.InteractionKind.Gravity, 20.0, 1e-3, nA, dt)
    tp, mp = mode_series(adc.InteractionKind.Plasma, 20.0, 1e-3, nA, dt)
    wg_th, wp_th = omega_theory(20.0, -1), omega_theory(20.0, +1)
    wg_me = np.pi / (2 * first_zero(tg, mg))
    wp_me = np.pi / (2 * first_zero(tp, mp))
    eg = abs(wg_me - wg_th) / wg_th
    ep = abs(wp_me - wp_th) / wp_th
    print(f"A gravité (Jeans)  : omega_th={wg_th:.3f} mesuré={wg_me:.3f} ({100*eg:.1f}%)")
    print(f"A plasma (Bohm-Gross): omega_th={wp_th:.3f} mesuré={wp_me:.3f} ({100*ep:.1f}%)")

    # --- B : champ fort, gravité instable (Jeans) vs plasma borné ---
    nB = int(0.6 / dt)
    tgi, mgi = mode_series(adc.InteractionKind.Gravity, 120.0, 1e-4, nB, dt)
    tpi, mpi = mode_series(adc.InteractionKind.Plasma, 120.0, 1e-4, nB, dt)
    gamma = np.sqrt(120.0 * RHO0 - CS2 * K * K)  # taux de croissance de Jeans
    growth = abs(mgi[-1]) / abs(mgi[0])
    bound = np.max(np.abs(mpi)) / abs(mpi[0])
    print(f"B gravité : |mode| x{growth:.1f} (instable, gamma_th={gamma:.2f}) ; "
          f"plasma borné à x{bound:.2f}")

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.2, 3.7), constrained_layout=True)
    fig.patch.set_alpha(0)
    a1.plot(tg, mg / 1e-3, "C0-", lw=1.6, label=f"gravité  $\\omega$={wg_me:.2f}")
    a1.plot(tp, mp / 1e-3, "C3-", lw=1.6, label=f"plasma  $\\omega$={wp_me:.2f}")
    a1.axhline(0, color="k", lw=0.6, alpha=0.4)
    a1.set(xlabel="temps", ylabel=r"mode $/\,\varepsilon$",
           title="A. champ modéré : Jeans vs Bohm-Gross")
    a1.legend(frameon=False, fontsize=9)
    a1.grid(True, alpha=0.25)

    a2.semilogy(tgi, np.abs(mgi), "C0-", lw=1.6, label="gravité (instable)")
    a2.semilogy(tgi, np.abs(mgi[0]) * np.cosh(gamma * tgi), "k--", lw=1.0,
                label=r"$\cosh(\gamma t)$ théorie")
    a2.semilogy(tpi, np.clip(np.abs(mpi), 1e-12, None), "C3-", lw=1.6,
                label="plasma (borné)")
    a2.set(xlabel="temps", ylabel=r"$|$mode$|$",
           title="B. champ fort : effondrement vs stabilité")
    a2.legend(frameon=False, fontsize=9)
    a2.grid(True, which="both", alpha=0.25)

    out = ROOT / "docs" / "tut_plasma.png"
    fig.savefig(out, dpi=130)
    print(f"écrit {out}")

    assert eg < 0.1, "la fréquence de Jeans (gravité) doit suivre la théorie"
    assert ep < 0.1, "la fréquence de Bohm-Gross (plasma) doit suivre la théorie"
    assert wp_me > wg_me, "le plasma durcit le son : omega_plasma > omega_gravité"
    assert growth > 5.0, "la gravité doit être instable (Jeans) à champ fort"
    assert bound < 3.0, "le plasma doit rester borné (toujours stable)"
    return 0


if __name__ == "__main__":
    sys.exit(main())
