# Static FAC reference and qualified seeds

Private follow-up to 50655394; Source/offline only, no new Native data.

The static composite route in `composite_fac_poisson.hpp` calls `composite_forcing_norm_()` before measuring the initial candidate residual (lines 417–441). That helper zeroes prepared operator scratch, fills the exact transfer ghosts, evaluates the residual and refluxes it (1972–1987). It never zeroes or measures the warm candidate as the reference. Thus R(0) means the zero iterate, not the start-of-solve residual. The dynamic Newton/boundary-kernel branch is separate and this conclusion does not qualify it.

For the unchanged screened equation, homogeneous Neumann conditions and constant qualified m, zero-iterate Original F is −8m on every active cell, including coarse/fine interfaces: zero interpolation and face flux remain zero. Initial m=2 gives reference 16; the one-step endpoint m=2(1+1/64) gives 16.25. ForwardEuler stage t_n uses the former. Warm seeds 0, 2, the exact current m, or m+1e−12 change the initial residual, never this reference. The independent stencil test covers both states and all four seeds.

Consequently the existing relative cutoff lies between 1.2307692307692308e−11 and 1.25e−11; it cannot collapse to relative×0.25 or relative×a tiny warm residual. No absolute-floor correction is required for this static board. Keep authenticated `fac.abs_tol=0`, the existing relative coefficient and Original F guard 1e−10. Adding the proposed absolute budget would be bounded here, but changes the authored policy without addressing an actual warm-start collapse. Native solver execution remains ROOT-owned and pending; Source tests do not establish attained Native accuracy.
