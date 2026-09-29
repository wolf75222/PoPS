# M04 temporal variants and preserved failure

The native execution with `diag(0.01, 0)` and Forward Euler now reaches `t=0.1`
on all three required grids. The directional face-frequency correction removes
the previous invalid double restriction. The physical equation, exact initial
cell averages, N=32/64/128, step `0.9/(1/h+0.02/h²)` and error thresholds stay
unchanged. Nevertheless, its first observed order is 0.60635, below the original
0.7 criterion. `m04-xonly-220b-openmp1` remains a failed scientific reception.

This failure belongs to the selected time method at these finite resolutions.
Reopening and hashing the actual native states, an independent Fourier product
for upwind/central Forward Euler reproduces them to 4.45e-16. Temporal
antidiffusion partly cancels the spatial upwind diffusion, by different amounts
on the three grids. Its analytical finite-grid orders are 0.60635 and 0.70177.
This observation does not waive the acceptance criterion or establish an
asymptotic convergence campaign.

The additional `m04-ssprk2` case changes only the explicitly authored temporal
method to SSPRK2. This is a distinct `(P,D,M)` experiment, and is the method used
by the supplied 0.4.0 reference `examples/cases.py::advection_diffusion` (whose
other constants differ). Before native execution its independent discrete
Fourier oracle predicts errors 0.00732003/0.00371589/0.00187236 and orders
0.97814/0.98885, under the same unchanged acceptance thresholds. Those predictions
are not native results. W02 remains a Forward Euler negative impulse witness.

Scientific receipt schema 3 records `spatial_method` and `temporal_method`
separately and adds the error against an independent Fourier solution of the
exact selected discrete method, including the final shortened step. The latter
is checked at 3e-12, separately from the continuous PDE error and conservation.
The original `m04` and `m04-isotropic` commands explicitly retain Forward Euler.
