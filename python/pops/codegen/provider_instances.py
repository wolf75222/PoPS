"""Declared components and their separately authenticated runtime instance keys.

ProviderPack remains the Module-owned declaration authority. Only the native storage
address is projected; slots, typed dependencies and physical producer evidence remain
unchanged. The opt-in contract enters the resolved operation identity and cache.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace

CONTRACT = 'consumed-provider-instance@1'
EVIDENCE = 'native_provider_instance'


def validate_instance_contract(contract, module=None, *, owner_qid=None):
    if not isinstance(contract, Mapping) or set(contract) != {'contract', 'definition_owner_qid', 'instance_owner_qid'} \
            or contract['contract'] != CONTRACT:
        raise ValueError('invalid consumed provider instance contract')
    definition = contract['definition_owner_qid']
    instance = contract['instance_owner_qid']
    from ._plans import authenticate_block_instance_owner
    from pops.model.ownership import OwnerPath
    _, _, model_owner = authenticate_block_instance_owner(instance, allow_unscoped=False)
    if type(definition) is not str or not definition or definition == instance:
        raise ValueError('provider instance lost its independent declaration owner')
    if str(OwnerPath.from_data(model_owner)) != definition:
        raise ValueError('provider instance does not instantiate its declaration owner')
    if module is not None and definition != str(module.owner_path.canonical()):
        raise ValueError('provider instance changed its Module declaration owner')
    if owner_qid is not None and instance != owner_qid:
        raise ValueError('provider instance changed its Case block owner')
    return contract


def require_instance_contract(plan, module=None, *, owner_qid=None):
    contract = None if plan is None else plan.provider_evidence.get(EVIDENCE)
    if contract is None:
        return None
    validate_instance_contract(contract, module, owner_qid=owner_qid)
    if any(op.guarantees.get('block_instance') != contract['instance_owner_qid'] for op in plan.operations):
        raise ValueError('provider instance differs from its resolved operation owner')
    return contract


def runtime_key(key, contract):
    """Project only local declarations; foreign Case-owned providers stay exact."""
    if contract is None or key.owner_qid != contract['definition_owner_qid']:
        return key
    return replace(key, owner_qid=contract['instance_owner_qid'])


def emitter_contract(model, *, owner_qid=None):
    plan = getattr(model, '_resolved_operations', None)
    if plan is not None:
        return require_instance_contract(plan, owner_qid=owner_qid)
    contract = getattr(model, '_native_provider_instance', None)
    if contract is None:
        return None
    from .component_provider_packs import require_emitter_provider_carrier
    require_emitter_provider_carrier(model, where='native provider instance projection')
    return validate_instance_contract(contract, owner_qid=owner_qid)


def qualify_publication_instances(blocks):
    """Select the complete duplicate-owner group, including unpublished siblings."""
    from ._compiler_lowering import require_compiler_lowering
    modules = {block.name: require_compiler_lowering(block.model).source_module for block in blocks}
    owners = {block.name: str(modules[block.name].owner_path.canonical()) for block in blocks}
    published = {owners[b.name] for b in blocks
                 if b.resolved_operations.provider_evidence.get('program_field_publications')}
    duplicated = {owner for owner in published if sum(o == owner for o in owners.values()) > 1}
    result = []
    for block in blocks:
        if owners[block.name] not in duplicated:
            result.append(block)
            continue
        plan = block.resolved_operations
        contract = {'contract': CONTRACT, 'definition_owner_qid': owners[block.name],
                    'instance_owner_qid': block.instance_owner_qid}
        plan = replace(plan, provider_evidence={**plan.provider_evidence, EVIDENCE: contract})
        require_instance_contract(plan, modules[block.name], owner_qid=block.instance_owner_qid)
        result.append(replace(block, resolved_operations=plan))
    return tuple(result)


def graph_instance_contracts(graph):
    from .program_models import ProgramModelGraph
    if type(graph) is not ProgramModelGraph:
        return ()
    return tuple(contract for model in graph._models_by_block.values()
                 if (contract := emitter_contract(model)) is not None)
