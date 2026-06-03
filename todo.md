# TODO — adc_cpp

> Liste de travail vivante. Synthese de (1) l'objectif initial du chantier (canal `aux` extensible
> + parite AMR + cablage runtime / Python / DSL), (2) ce que `docs/ROADMAP.md` marque "en file",
> (3) ce que les agents ont explicitement note comme "reste a faire".
> Convention : `[x]` fait et sur `master`, `[~]` partiel, `[ ]` a faire.

## 1. Chantier "Aux extensible" (champs auxiliaires au-dela de phi / grad)

Objectif : un modele declare/lit des champs aux SUPPLEMENTAIRES (B_z magnetique, T_e electronique)
sans casser l'existant, en retro-compat bit-exacte (`n_aux` defaut = 3 -> strictement identique).

- [x] **Inc. 1 — Lecture** : `adc::Aux` + `B_z` (comp 3), `kAuxBaseComps=3`, `aux_comps<Model>()`,
      `load_aux<NComp>`. Les foncteurs nommes lisent `load_aux<aux_comps<Model>()>`. (#24)
- [x] **Inc. 2 — Peuplement Coupler** mono-bloc : `fill_bz`, aux alloue a `aux_comps<Model>()`. (#24)
- [x] **Inc. 3 — Peuplement SystemAssembler** multi-blocs (aux = max sur les blocs). (#25)
- [x] **Inc. 4 — `CompositeModel::n_aux`** = max des briques ; `aux_comps` deplace dans
      `physical_model.hpp` (header contrat). (#26)
- [x] **Inc. 5 — runtime `System`** : `ensure_aux_width` + `set_magnetic_field` (binding Python
      calque sur `set_epsilon_field`). Chemin natif `add_compiled_model` complet. (#29)
- [x] **Inc. 6 — DSL** : emet `n_aux` quand une formule lit `aux('B_z')` (`AUX_CANONICAL`). (#30)
- [x] **Inc. 7 — chemin JIT** `add_dynamic_block` : `IModel::n_aux()` virtuel + marshaling
      `aux_ncomp_` -> B_z transporte, Python end-to-end. (#32)
- [ ] **Inc. 8 — chemin AOT compile** `add_compiled_block` : l'ABI `compiled_block_abi.hpp`
      (`adc_compiled_*`) marshale encore 3 composantes en dur -> B_z absent par ce chemin.
      Symetrique de l'inc. 7 cote ABI `extern "C"`.
- [ ] **T_e — 2e champ extra DERIVE** : T = p/rho calculee par le `System` depuis un bloc fluide
      designe a chaque solve (comp 4, populate cote System, pas user-fourni comme B_z). Valide la
      generalisation a 2 champs aux.
- [ ] **AMR / implicite** : `advance_amr` et le stepper implicite gardent `load_aux` defaut (3) ;
      etendre a la largeur du modele pour un B_z sur AMR (bit-identique aujourd'hui).

## 2. Chantier "EPM elliptique generique" (operateur elliptique composable)

- [x] Permittivite variable `eps(x)` : `GeometricMG::set_epsilon` + `System::set_epsilon_field`
      (binding Python) sur `master`.
- [x] `EllipticProblem` / `FieldPostProcess` nommes (coeff, CL, nullspace, convention `E = -grad phi`).
- [ ] Autres operateurs / coefficients composables (au-dela d'eps(x)).
- [ ] Recabler les sites en forme `/(2*dx)` vers la forme multiplicative `*cx` (`amr_coupler`,
      `amr_coupler_mp`, `spectral_coupler`) — differe au dernier bit, donc hors perimetre tant
      qu'on veut le bit-identique.

## 3. Durcissement de l'architecture (`docs/ROADMAP.md` "en file")

- [~] **Moteur AMR unifie** : `advance_amr(LevelHierarchy&)`, `FluxRegister`, `CoverageMask` promus
      en vrais types. RESTE : promouvoir `PatchRange`, le routage bordant de `CoarseFineInterface`,
      `SubcyclingSchedule`, `RegridPolicy` (encore inlines dans `subcycle_level_mp`) et y replier la
      famille `amr_step_*` (qui encode le cas dans le nom).
- [~] **API memoire explicite** : `for_each_cell_reduce_{sum,max}`, `sum`/`norm_inf` faits.
      RESTE : `sync_host` / `sync_device` explicites.
- [~] **Familles de ghosts** : `fill_physical_bc` / `fill_boundary` / `mf_fill_fine_ghosts` separes.
      RESTE : remonter le coarse-fine en helper nomme de premier niveau.
- [~] **VariableRole** : couplages inter-especes par role faits (#18). RESTE : usage plein dans le
      runtime / le DSL (noms de variables generes par role).
- [x] AMR multi-patch distribue MPI (2 et N niveaux), `CouplingPolicy` mince, suite de validation
      numerique coeur, decoupage elliptique (operateur / solveur / probleme).

## 4. GPU (GH200) — integration

- [x] Composants valides SEPAREMENT et bit-identiques au CPU sur GH200 : System mono-grille, ops de
      champ AMR, halos MPI multi-GPU, backend AOT d'un modele DSL, `load_aux<4>` (B_z device).
- [ ] **Validation INTEGREE** `AmrSystem` + MPI + GPU en un seul run ; perf full-device a travailler.
- [ ] `add_compiled_model` (chemin AOT externe) n'a PAS d'API dans `AmrSystem` (System seul,
      mono-box mono-rang) — l'etendre si besoin AMR.

## 5. Physique magnetisee

- [x] Push de Boris E+B combine (`tfap_boris`, cyclotron exact, derive ExB sans croissance seculaire).
- [ ] Reformulation AP tensorielle sous champ fort.

## 6. Reproduction Hoffart (arXiv:2510.11808) — APPLICATIF, cote `adc_cases`

- [~] **M1** : taux de croissance numerique vs analytique (diocotron). Pipeline valide ; `gamma_norm`
      croit vers 0.911 mais limite par la diffusion numerique du bord d'anneau (-> motive l'AMR).
- [x] **M2 / M2b** : AMR sur le bord d'anneau (triple le taux a base egale) + Poisson multi-niveau.
- [ ] Montee en resolution / convergence vers le taux analytique ; integration SAMRAI ulterieure.
