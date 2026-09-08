import pytest
from pops.numerics.diffusion import explicit_diffusion_dt_bound


def test_combined_bound_refuses_individually_stable_courant_and_diffusion_numbers():
    # dt=h=1, c=.6, r=.3: each independent bound passes, their sum is 1.2.
    transport_only=explicit_diffusion_dt_bound(spacings=(1,),diffusivities=(0,),speeds=(.6,))
    diffusion_only=explicit_diffusion_dt_bound(spacings=(1,),diffusivities=(.3,))
    combined=explicit_diffusion_dt_bound(spacings=(1,),diffusivities=(.3,),speeds=(.6,))
    assert min(transport_only,diffusion_only)>1
    assert combined==pytest.approx(1/1.2)
    assert explicit_diffusion_dt_bound(spacings=(1,),diffusivities=(.1,),speeds=(.6,))>1


@pytest.mark.parametrize("ratio",[2,4])
def test_amr_uses_local_h_and_actual_substep_ratio(ratio):
    coarse=explicit_diffusion_dt_bound(spacings=(1.,),diffusivities=(.25,))
    fine=explicit_diffusion_dt_bound(spacings=(1/ratio,),diffusivities=(.25,),time_ratio=ratio)
    assert fine==coarse/ratio
    assert explicit_diffusion_dt_bound(spacings=(1/ratio,),diffusivities=(.25,),time_ratio=ratio*ratio)==coarse
