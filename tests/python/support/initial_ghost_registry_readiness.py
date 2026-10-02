"""Strict test-only readonly readiness disposition; never substitutes registry contents."""
EXPECTED_REFUSAL='AMR rank-local carrier manifest requires all fields to be materialized exactly'

def classify_registry_disposition(slots, flags, manifest, failures, payload, phase, world_size):
    if type(world_size) is not int or world_size<1:raise ValueError('invalid rank authority')
    if phase not in ('bootstrap-baseline','after-bootstrap-abort','parent-before','parent-rejected'):
        raise ValueError('unknown registry observation phase')
    if type(slots) is not list or not slots or any(type(s) is not str or not s for s in slots) or len(set(slots))!=len(slots):
        raise ValueError('invalid qualified Field slot authority')
    if type(flags) is not list or len(flags)!=len(slots) or any(type(f) is not bool for f in flags):
        raise ValueError('Field materialization flags must be exact bool')
    if type(manifest) is not list or len(manifest)!=len(slots):raise ValueError('Field manifest inventory differs')
    byslot={}
    for row in manifest:
        if type(row) is not list or len(row)<9 or row[0]!='pops.amr.field-provider-checkpoint-manifest@1' or row[1] in byslot or row[8] not in ('materialized','unmaterialized'):
            raise ValueError('Field manifest readiness authority differs')
        byslot[row[1]]=row[8]=='materialized'
    if set(byslot)!=set(slots) or [byslot[s] for s in slots]!=flags:raise ValueError('Field readiness authorities contradict')
    expected_ready=phase in ('parent-before','parent-rejected')
    if any(f!=expected_ready for f in flags):raise ValueError('unexpected Field materialization for observation phase')
    if type(failures) not in (tuple,list) or len(failures)!=world_size:raise ValueError('missing collective registry outcome')
    authority={'slots':slots,'materialized':flags,'field_manifest':manifest}
    if expected_ready:
        if any(f is not None for f in failures):raise ValueError('ready registry read failed')
        if type(payload) is not list or not payload or any(type(r) is not list or any(type(v) is not str for v in r) for r in payload):raise ValueError('available registry has invalid typed rows')
        return {'schema':'sol61.initial-ghost-registry-disposition@1','disposition':'available','readiness':authority,'rows':payload}
    exact=('RuntimeError',EXPECTED_REFUSAL,True)
    if payload is not None or any(type(f) not in (tuple,list) or len(f)!=3 or type(f[0]) is not str or type(f[1]) is not str or type(f[2]) is not bool or tuple(f)!=exact for f in failures):
        raise ValueError('unmaterialized registry did not return its exact normative refusal')
    return {'schema':'sol61.initial-ghost-registry-disposition@1','disposition':'readiness-refused','readiness':authority,'refusal':{'exception':'RuntimeError','message':EXPECTED_REFUSAL,'rank_count':len(failures)},'scope':'registry contents unavailable; no payload received'}

def capture_registry_disposition(world, native_owner, phase):
    from tests.python.support.collective_checks import collective_call,collective_attempt
    from pops._native_collectives import allgather_value
    slots=collective_call(world,native_owner.field_provider_slots)
    size=world.size if world is not None else 1
    gathered=allgather_value(world,slots)
    def require_slots():
        if type(gathered) not in (list,tuple) or len(gathered)!=size:raise ValueError('Field slot consensus missing rank')
        if any(type(row) is not list or row!=slots for row in gathered):raise ValueError('Field slot authority differs between ranks')
    collective_call(world,require_slots)
    flags=[collective_call(world,lambda s=s:native_owner.field_provider_materialized(s)) for s in slots]
    manifest=collective_call(world,native_owner.field_provider_checkpoint_manifest)
    payload,failures=collective_attempt(world,native_owner.checkpoint_rank_local_carrier_manifest)
    return collective_call(world,lambda:classify_registry_disposition(slots,flags,manifest,failures,payload,phase,size))
