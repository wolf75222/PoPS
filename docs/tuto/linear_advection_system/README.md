# Systeme d'advection lineaire

Ce tutoriel resout un systeme conservatif a deux composantes :

```math
\frac{\partial Q}{\partial t}
+\frac{\partial (A_x Q)}{\partial x}
+\frac{\partial (A_y Q)}{\partial y}=0.
```

Les matrices $A_x$ et $A_y$ sont pleines et ne commutent pas. Il n'existe donc pas une seule base
qui transforme le probleme 2D en deux advections scalaires independantes. Chaque direction possede
ses propres vitesses caracteristiques :

```math
\lambda(A_x)=(-0.5,1.0), \qquad \lambda(A_y)=(-0.2,0.4).
```

## Flux upwind caracteristique

[`01_openmp_characteristic_upwind.py`](01_openmp_characteristic_upwind.py) declare le flux physique
$F_x=A_xQ$ et $F_y=A_yQ$. `model.roe_from_jacobian()` derive ensuite le Jacobien de chaque flux. Le
runtime calcule sa decomposition caracteristique

```math
A=R\Lambda R^{-1}, \qquad |A|=R|\Lambda|R^{-1}.
```

Le flux de Roe natif applique alors la dissipation $|A|$ dans cette base :

```math
\widehat F(Q_L,Q_R)
=\frac{1}{2}\left[A(Q_L+Q_R)-|A|(Q_R-Q_L)\right]
=A^+Q_L+A^-Q_R.
```

Les modes de vitesse positive prennent l'etat gauche. Ceux de vitesse negative prennent l'etat
droit. C'est le flux upwind du systeme, et non un flux scalaire applique separement a chaque
composante.

```python
model.roe_from_jacobian()

finite_volume = FiniteVolume(
    flux=physical_flux,
    variables=variables.Conservative(Q),
    reconstruction=reconstruction.FirstOrder(),
    riemann=riemann.Roe(),
)
```

Le domaine est periodique. Le fichier verifie que les integrales des deux composantes restent
constantes et que le couplage des matrices cree une composante $q_2$ a partir d'un etat initial ou
seule $q_1$ est non nulle.

```bash
python docs/tuto/linear_advection_system/01_openmp_characteristic_upwind.py
```
