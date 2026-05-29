"""Teste le module Python `adc` (binding de la facade libadc).

Verifie qu'on pilote les solveurs concrets depuis Python sans rien savoir des
templates C++ : construction par config, pas de temps, invariants physiques
(masse conservee, quantite de mouvement nulle pour la gravite), et champ rendu
en tableau numpy de la bonne forme. PYTHONPATH pointe sur le dossier du .so.
"""
import sys
import numpy as np
import adc

fails = 0


def chk(cond, what):
    global fails
    if not cond:
        print("FAIL", what)
        fails += 1


# --- DiocotronSolver ---
cfg = adc.DiocotronConfig()
cfg.n = 64
ds = adc.DiocotronSolver(cfg)
m0 = ds.mass()
for _ in range(5):
    ds.step(0.01)
rho = ds.density()
print(f"DiocotronSolver : shape={rho.shape} masse {m0:.6e} -> {ds.mass():.6e}")
chk(isinstance(rho, np.ndarray) and rho.shape == (64, 64), "diocotron_density_numpy")
chk(abs(ds.mass() - m0) < 1e-9, "diocotron_masse_conservee")

# --- EulerPoissonSolver, backend FFT ---
ec = adc.EulerPoissonConfig()
ec.n = 64
ec.use_fft = True
es = adc.EulerPoissonSolver(ec)
em0 = es.mass()
for _ in range(5):
    es.step(0.004)
print(f"EulerPoissonSolver(FFT) : masse={es.mass():.6e} "
      f"p=({es.total_momentum(0):.2e}, {es.total_momentum(1):.2e})")
chk(abs(es.mass() - em0) < 1e-9, "ep_masse_conservee")
chk(abs(es.total_momentum(0)) < 1e-9, "ep_qte_mouvement_nulle")
chk(es.density().shape == (64, 64), "ep_density_numpy")

if fails == 0:
    print("OK test_bindings")
sys.exit(0 if fails == 0 else 1)
