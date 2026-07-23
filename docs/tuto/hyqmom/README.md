# HyQMOM a 15 moments

Ce dossier reprend les cas principaux de `RieMOM2D_Electrostatic_periodic` avec l'API publique
PoPS. Chaque fichier est separe pour rester lisible : pas de selecteur de cas, pas de factory
commune cachee. Les scripts declarent le domaine, le modele, les numeriques, le programme de temps
et les donnees initiales dans l'ordre ou ils sont utilises.

| Fichier | Cas |
|---|---|
| `01_openmp_diocotron_hll.py` | diocotron periodique avec Poisson FFT et source cyclotron |
| `02_openmp_constant_hll.py` | etat Maxwellien constant et fermeture HyQMOM ecrite en Python |
| `03_openmp_fluid_wave_hll.py` | onde fluide lineaire avec flux HLL |
| `04_openmp_electrostatic_wave_hll.py` | onde electrostatique avec Poisson FFT |
| `05_openmp_magnetic_wave_hll.py` | onde magnetique avec Poisson FFT et source cyclotron |
| `06_openmp_shock_tube_hll.py` | tube a choc 2D avec relaxation15 |
| `07_openmp_crossing_jets_hll.py` | jets croises 2D avec relaxation15 |

Le cas diocotron conserve le systeme physique, les parametres et les choix numeriques effectivement
utilises par `main.m` :

| Element | Valeur |
|---|---|
| Domaine | $[-0.5,0.5]^2$, periodique |
| Maillage | $128 \times 128$ |
| Moments | 15 moments cartesiens, jusqu'a l'ordre quatre |
| Fermeture | HyQMOM d'ordre quatre, six moments fermes d'ordre cinq |
| Flux numerique | HLL, spectre du Jacobien complet |
| Reconstruction | constante par cellule |
| Temps | Euler explicite, CFL $0.5$, $t_{max}=1$ |
| Champ | Poisson FFT spectral, potentiel de moyenne nulle |
| Frequences | $\omega_p=20$, $\omega_c=-20$ |

Les scripts n'utilisent pas HLLC.

## Systeme de moments

Pour $p+q\leq 4$, les moments bruts sont

```math
M_{pq}(x,y,t)=\int_{\mathbb{R}^2} f(x,y,v_x,v_y,t)v_x^p v_y^q\,dv_xdv_y.
```

Ils verifient

```math
\partial_t M_{pq}+\partial_x M_{p+1,q}+\partial_y M_{p,q+1}
=pE_xM_{p-1,q}+qE_yM_{p,q-1}
+\omega_c\left(pM_{p-1,q+1}-qM_{p+1,q-1}\right).
```

Les quinze composantes sont rangees exactement comme dans le code MATLAB :

```text
M00 M10 M20 M30 M40 M01 M11 M21 M31 M02 M12 M22 M03 M13 M04
```

Les flux contiennent six moments d'ordre cinq. `HyQMOM15Closure()` utilise les relations
polynomiales du PDF et de `closureS5.m`, par exemple

```math
S_{50}=\frac{S_{30}}{2}\left(5S_{40}-3S_{30}^2-1\right),
```

```math
S_{32}=\frac{1}{2}\left(2S_{40}-3S_{30}^2\right)S_{12}
+\frac{1}{2}\left(3S_{22}-1\right)S_{30}.
```

Cette fermeture est appliquee aux expressions symboliques lors de la construction du modele. Les
flux obtenus sont ensuite compiles ; aucune fermeture Python n'est appelee dans les cellules.

### Fermeture utilisateur complete

Le script [`02_openmp_constant_hll.py`](02_openmp_constant_hll.py) n'appelle
pas `HyQMOM15Closure()`. Il ecrit explicitement les six relations du PDF dans
une fermeture utilisateur :

```python
@closure(4)
def user_hyqmom15_closure(S):
    s30 = S["S30"]
    s40 = S["S40"]
    # ... S41, S32, S23 et S14 sont egalement definis dans le script.
    return {
        "S50": 0.5 * s30 * (5.0 * s40 - 3.0 * s30 * s30 - 1.0),
        "S41": ...,
        "S32": ...,
        "S23": ...,
        "S14": ...,
        "S05": ...,
    }
```

`CartesianVelocityMoments(4, closure=user_hyqmom15_closure)` construit ensuite
en Python les quinze inconnues, les transformations des moments bruts vers les
moments centres puis standardises, les flux fermes et leurs Jacobiennes. La
fonction utilisateur est evaluee une seule fois sur les expressions symboliques
pendant la construction. Son arithmetique est incorporee dans l'AST compile :
il n'existe aucun callback Python dans les cellules ou les pas de temps.

L'exemple conserve `HyQMOM15Closure()` dans les autres cas afin de montrer la
version concise de la meme physique. Les deux chemins utilisent le meme
protocole generique de fermeture locale.

## Suivre les simulations

Les cinq premiers cas possedent un `ConsoleMonitor` configurable :

```python
ConsoleMonitor(
    schedule=every(MONITOR_EVERY, clock=program.clock),
    diagnostics=(
        StepChangeNorm(L2(), block=plasma),
        Integral(role=Density(), block=plasma),
    ),
    template=(
        "step={step} t={time:.4e} dt={dt:.3e} "
        "dU_L2={plasma.step_change_l2:.3e} "
        "mass={plasma.integral:.6e}"
    ),
    enabled=ENABLE_MONITOR,
)
```

Le residu `dU_L2` porte sur les quinze moments. `mass` selectionne uniquement
la composante ayant le role `Density`, donc $M_{00}$. La cadence par defaut est
de 100 pas acceptes, car le diagnostic global d'un etat a quinze composantes
est plus couteux que celui de l'advection scalaire. Les reductions restent
natives et OpenMP ; seul l'affichage des scalaires a lieu en Python.
`ENABLE_MONITOR = False` retire completement le monitor avant compilation.

## Initialisations

Les scripts `constant`, `shock_tube` et `crossing_jets` construisent directement des moments reels
Maxwelliens. Les scripts `fluid_wave`, `electrostatic_wave` et `magnetic_wave` suivent les fichiers
MATLAB : un vecteur propre du systeme linearise sert a fabriquer la perturbation initiale.

Pour `magnetic_wave`, le vecteur propre est complexe. PoPS n'evolue pas un etat complexe : les
moments physiques restent reels. Le script prend donc la partie reelle du mode harmonique
complexe. Le complexe est seulement un outil d'initialisation lineaire.

## Diocotron

La densite initiale est un anneau perturbe :

```math
\rho(r,\theta)=
\begin{cases}
0.9+0.1\sin(4\theta), & 0.35\leq r\leq 0.40,\\
10^{-4}, & \text{ailleurs}.
\end{cases}
```

Le domaine reste entier. L'anneau n'est ni un mur ni une frontiere embarquee.

Le potentiel initial satisfait

```math
\Delta\phi=\omega_p^2\left(\rho-\langle\rho\rangle\right),
\qquad \langle\phi\rangle=0,
```

puis la vitesse moyenne initiale est la derive utilisee par le MATLAB :

```math
u_x=-\frac{\partial_y\phi}{\omega_c},
\qquad
u_y=\frac{\partial_x\phi}{\omega_c}.
```

Les quinze moments sont ceux d'une gaussienne de covariance identite, translatee par cette
vitesse et multipliee par $\rho$.

## Ce qui est natif

PoPS execute le flux HLL, l'evaluation du Jacobien $15\times15$, les sources electrique et
cyclotron, la FFT periodique, le programme Euler, la borne de pas et les mises a jour avec le
backend C++ natif. Kokkos execute les kernels de cellules.

Dans le tube a choc et les jets croises, `HyQMOM15Relaxation` declare la transformation symbolique.
Le programme l'applique une fois apres Euler dans un kernel C++/Kokkos, avant le commit de l'etat.

La neutralisation du second membre reste explicite pendant toute l'evolution. Le modele soustrait
le `RuntimeParam("neutralizing_density")`, lie a la moyenne discrete de la densite initiale. Le flux
conservatif doit maintenir cette compatibilite. `ConstantNullspace()` ne modifie jamais le second
membre : si l'accumulation d'arrondi lui fait perdre sa moyenne nulle, le calcul echoue au lieu de
projeter silencieusement. `MeanValueGauge(0.0)` fixe ensuite la moyenne du potentiel.

## Differences et limites par rapport a l'archive

Le tutoriel ne reproduit pas deux erreurs du MATLAB :

- `main.m` appelle un diagnostic L2 qui ne connait pas le cas diocotron ;
- `meshgrid` transpose l'angle dans l'initialisation et inverse ici le signe de la perturbation.

PoPS conserve l'orientation physique $(x,y)$. Le MATLAB exploite une factorisation particuliere du
Jacobien pour calculer les bornes HLL ; PoPS calcule le spectre du Jacobien complet. Le backend dense
refuse maintenant les vitesses HLL issues d'un spectre non reel ou non converge : le run echoue au
lieu d'utiliser une borne qui masquerait une perte d'hyperbolicite.

## Execution

Depuis la racine du depot :

```bash
bash scripts/setup_env.sh
bash scripts/build_python.sh
conda activate pops
python docs/tuto/hyqmom/01_openmp_diocotron_hll.py
python docs/tuto/hyqmom/02_openmp_constant_hll.py
python docs/tuto/hyqmom/03_openmp_fluid_wave_hll.py
python docs/tuto/hyqmom/04_openmp_electrostatic_wave_hll.py
python docs/tuto/hyqmom/05_openmp_magnetic_wave_hll.py
python docs/tuto/hyqmom/06_openmp_shock_tube_hll.py
python docs/tuto/hyqmom/07_openmp_crossing_jets_hll.py
```

Les cas `diocotron`, `electrostatic_wave`, `magnetic_wave`, `shock_tube` et `crossing_jets`
reprennent des tailles proches du MATLAB et peuvent etre nettement plus longs que les petits
tutoriels d'advection. Les constantes sont rassemblees au debut de chaque fichier si l'on souhaite
d'abord faire un essai plus court.

Le script ecrit `results/01_openmp_diocotron_hll.npz`. Le potentiel sauvegarde est celui du dernier
debut de pas, puisque le programme resout le champ sur $U^n$ avant de committer $U^{n+1}$, comme
`main.m`. Pour produire les cartes de densite, de ce potentiel et de vitesse moyenne :

```bash
python docs/tuto/hyqmom/plot_openmp_results.py
```
