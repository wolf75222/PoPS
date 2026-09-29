# M08: periodic guiding-center variant and stage-field evidence

The [M08 corpus entry](corpus.json) specifies a periodic `[0,2π]²` square, zero-mean charge, a manufactured vortex, `32²` and `64²` grids, SSPRK2, two states at the same physical time, a field residual at most `1e-10`, and a compatible velocity-divergence check. It marks the historical reference as `not_demonstrated`: the 0.4.0 reference has a Poisson solve and gradient, but no complete two-dimensional guiding-center assembly. The reference's `document/01_review.tex` F08 also requires cache keys to distinguish different stages at one time. This example is a new, explicitly named variant; it does not claim a replay of a missing reference result.

The law is `-Δφ=q`, `u=(∂yφ,-∂xφ)`, `∂tq+∇·(qu)=0`, `∂tc+∇·(cu)=0`, with periodic boundaries and zero-mean gauge. Charge and tracer are different Model and Case blocks, each with a scalar state and its own authored flux. One Poisson solve per physical SSPRK2 stage consumes the charge State and publishes the resulting gradient to both blocks. The two RHS calls at each stage carry the same field provenance; stage 1 solves again from the predicted charge. This is spatial transport, not a prescribed or frozen velocity. The initial cell averages are those of `q=0.1 cos(x)cos(y)` and `c=1+0.2 cos(x)`. The continuum initial charge is stationary along its own streamlines, while the tracer moves. First-order Rusanov transport makes the discrete charge change slightly, which makes stage-1 field freshness observable.

The corpus does not fix amplitude, time step, end time, or a tracer accuracy tolerance. This variant declares amplitudes `0.1` and `0.2`, `dt=0.02`, two steps, and `t_end=0.04` in source before any native run. It uses a separate NumPy FFT inverse of the periodic five-point `-Δh`, centered-difference velocity/divergence, and separate first-order Rusanov + SSPRK2 update as its numerical oracle. The native C++ field solver and its cache do not participate in this oracle. The source also computes a `1.1q` probe at a distinct stage with **the same time as stage 0**. That probe affects no physical update: a stale reuse of the stage-0 potential yields a Poisson residual of order `1e-2` and violates the predeclared `2e-9` linear-response threshold. The source test checks three distinct equation identities for stage 0, the same-time probe, and stage 1.

All bounds appear in `api040_m08_guiding_center.py::CRITERIA` before compilation: field residual `1e-10`; independent FFT potential `1e-9`; observed-gradient agreement `2e-10`; compatible centered divergence `1e-10`; probe linear response `2e-9` with minimum `1e-3` potential separation; physical stage-field separation `1e-8`; full separate FV/SSPRK2 state agreement `3e-7`; separate charge and tracer mean defects `2e-10`; initial binding and final time `1e-12`. No field value is clipped. The receipt is based on reopened native state/history NPZ files and includes package/native identity, ABI, execution context, solver diagnostics, and archive hash. A source-only validate/resolve/emit result is not a native qualification.

The source-only run first exposed a genuine publication defect: `_program_publication._states` traversed through a stage State into an older Poisson solve and misreported sequential states as conflicting current inputs. The bounded fix stops the current solve's provenance walk at State and nested solve boundaries while retaining exact current load/coefficient inputs. Dedicated tests keep both negative boundaries: conflicting current states, and explicit use of the older field for a new stage. Installed Dim=2 execution and MPI receipt are pending the root's rebuilt native artifact.


### Discrimination du second champ de SSPRK2

La réception fixe l'erreur FV maximale à **3e-11**, avant toute exécution native.
Le seuil précédent 3e-7 acceptait la trajectoire volontairement incorrecte qui réutilise
le champ initial dans le second RHS : les écarts finaux aux résolutions 32 et 64 sont
respectivement 4.44054e-9 et 2.21704e-9. Le nouvel écart minimal vaut donc plus de
73 fois la tolérance. Les deux oracles sont calculés avant compilation ; leur séparation
doit dépasser 21 fois la tolérance. Les états sauvegardés doivent ensuite être à moins
de 3e-11 de l'oracle frais et à au moins 20 fois cette tolérance de l'oracle périmé.
L'inégalité triangulaire relie explicitement ces deux critères. Les références périmées,
les erreurs et la marge sont conservées dans le NPZ et le reçu.

Le seuil 3e-11 est une obligation de précision supplémentaire, et **ne découle pas**
du seul seuil de résidu maximal 1e-10. La CG demande rel_tol=1e-12 et abs_tol=1e-13 ;
le mode périodique non nul le plus lent de -Laplacien a une valeur propre proche de 1,
les amplitudes sont 0.1/0.2 et le temps intégré 0.04. Cela motive une réception nettement
plus stricte que 3e-7, avec une marge pratique par rapport aux tolérances de solveur.
La réception native doit démontrer cette précision ; un échec reste un échec, sans
ajustement automatique du seuil ou des paramètres physiques.
