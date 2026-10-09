"""Real resolved fault plan; recording consumer tests Source ordering, never Native."""
from pathlib import Path
import sys
import pytest
import pops
from pops.identity import make_identity
from pops.runtime._amr_bootstrap_execution import BootstrapReceipt,execute_bootstrap
from tests.python.support.initial_field_ghost_native_case import build
from tests.python.support.initial_ghost_failure_component import load_component,InitialFailureBoundary

@pytest.mark.parametrize("recompute_level",[0,1])
def test_genuine_field_action_failure_aborts_before_finalizer(tmp_path,recompute_level):
    root=Path(__file__).resolve().parents[2]
    assert Path(pops.__file__).resolve().is_relative_to(root/'python')
    assert 'pops._pops' not in sys.modules
    component=load_component(tmp_path/'component')
    case,layout=build(boundary_composer=lambda base:InitialFailureBoundary(base,component))
    resolved=pops.resolve(pops.validate(case),layout=layout,components=(component,));resolved.verify()
    plan=resolved.bootstrap_plan
    assert [(a.operation,a.level) for a in plan.actions if a.operation=='recompute']==[('recompute',0),('recompute',1)]
    events=[]
    class RecordingConsumer:
        bootstrap_consumer_identity=make_identity('source-recording-bootstrap-consumer',{'schema_version':1})
        def consume_bootstrap_action(self,action):
            events.append((action.operation,action.level))
            if action.operation=='recompute' and action.level==recompute_level:
                raise RuntimeError('independent source field-action refusal')
            return BootstrapReceipt(action.identity,self.bootstrap_consumer_identity,{})
        def finalize_bootstrap(self):events.append(('finalize',None))
        def abort_bootstrap(self):events.append(('abort',None))
    with pytest.raises(RuntimeError,match='source field-action refusal'):
        execute_bootstrap(plan,RecordingConsumer())
    assert events[-2:]==[('recompute',recompute_level),('abort',None)]
    assert ('finalize',None) not in events
    assert 'pops._pops' not in sys.modules


def test_diagnostic_accepts_genuine_collective_attempt_shape(tmp_path):
    from tests.python.support.collective_checks import collective_attempt
    from tests.python.support.initial_ghost_failure_receipt import save_early_bind_receipt
    def refused():raise RuntimeError('independent genuine helper refusal')
    _,failures=collective_attempt(None,refused)
    assert type(failures) is tuple
    path=save_early_bind_receipt(tmp_path,rank=0,world_size=1,failures=failures,
        owner_count=0,image_count=0,target_count=0,
        native={'path':'SourceOnly','sha256':'SourceOnly'},
        artifact_identity='SourceOnly',component_manifest={'scope':'SourceOnly'})
    import json
    assert json.loads(path.read_text())['failures']==[['RuntimeError','independent genuine helper refusal',True]]
