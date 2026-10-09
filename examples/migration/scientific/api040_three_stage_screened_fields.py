"""Two stage-dependent screened fields, one frozen donor, public SSPRK3 composition.

Periodic [0,2]x[0,3], cell averages on 16x12 cells:
  d_t=0; (-Delta+3)phi=q+d; (-Delta+5)psi=q-3d/4;
  q_t=.15Delta q-.8q+.2phi+.4psi.

Run as a script or module. This example reports an actual execution, not independent
scientific qualification. Tests use a separately authored scalar/FFT oracle.
"""
from pathlib import Path
import argparse
import json
import numpy as np
import pops

if __package__:
    from .api040_stage_fields_library import author_case, literal_inputs, DT
    from .runtime import _execution_resources
else:
    from api040_stage_fields_library import author_case, literal_inputs, DT
    from runtime import _execution_resources


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',required=True,type=Path)
    args=parser.parse_args()
    assert not args.out.exists()
    args.out.mkdir(mode=0o700,parents=True)

    # Physics: the reusable scientific library contains the visible equations above,
    # the two positive screenings and the declared, unchanged CG solver controls.
    authored=author_case()
    initial=literal_inputs()

    # Spatial realization: periodic two-dimensional finite volumes, no manual kernel.
    validated=pops.validate(authored.case)
    resolved=pops.resolve(validated,layout=authored.layout)

    # Time: q1=q0+hF0; q2=3q0/4+(q1+hF1)/4;
    # q_next=q0/3+2(q2+hF2)/3. F0,F1,F2 each solve and publish both Fields.
    # The same public Program is consumed by compile; no second technical graph.
    artifact=pops.compile(resolved)
    simulation=pops.bind(artifact,
        initial_state={'receiver':initial['receiver'],'reservoir':initial['donor']},
        resources=_execution_resources(artifact))
    report=pops.run(simulation,t_end=float(DT),max_steps=1,console=False)

    # Save real states; field clocks/solver reports are not inferred from these arrays.
    np.savez(args.out/'actual-states.npz',
        receiver=np.asarray(simulation.state_global('receiver')),
        donor=np.asarray(simulation.state_global('reservoir')))
    simulation.checkpoint(args.out/'accepted-checkpoint')
    print(json.dumps({'accepted_steps':report.accepted_steps,'time':simulation.time(),
        'artifact':artifact.artifact_identity.token,
        'scope':'One actual public step; independent scientific reception is separate.'}))
