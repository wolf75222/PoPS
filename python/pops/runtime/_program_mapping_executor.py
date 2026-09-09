"""Drive native Program continuations and prepared maps in one outer transaction."""
from __future__ import annotations


def execute_program_maps(owner, dt, generation, attempt, receipts, captured):
    from pops.runtime._native_step_target import native_program_region_target
    routes = owner._transfer_routes
    stage = {route.program_invocation: route for route in routes if route.program_invocation}
    accepted = [route for route in routes if not route.program_invocation
                and route.transfer.synchronization_uri == "pops://synchronization/before-step@1"]
    after = [route for route in routes if not route.program_invocation and route not in accepted]
    # Accepted input snapshots precede all region execution and target writes.
    for route in accepted:
        route.session.capture(generation, attempt)
        captured.append(route)

    def apply(route, *, already_captured=False):
        if not already_captured:
            route.session.capture(generation, attempt)
            if all(previous is not route for previous in captured):
                captured.append(route)
        receipt = route.session.apply(generation, attempt)
        owner._authenticate_mapping_receipt(route, receipt, generation=generation, attempt=attempt)
        receipts.append(receipt)

    for route in accepted:
        apply(route, already_captured=True)
    waiting = {}  # layout -> None runnable, qualified port identity suspended, or False complete
    for layout in owner._engines:
        waiting[layout] = None
    pending_after = list(after)
    while any(value is not False for value in waiting.values()) or pending_after:
        progress = False
        for route in tuple(pending_after):
            if waiting[route.transfer.source_layout_id] is False:
                apply(route)
                pending_after.remove(route)
                progress = True
        blocked = {route.transfer.target_layout_id for route in pending_after}
        for layout in sorted(waiting):
            if waiting[layout] is not None or layout in blocked:
                continue
            port = native_program_region_target(owner._engines[layout])._advance_program_region(dt)
            if port and (port not in stage or layout not in (
                    stage[port].transfer.source_layout_id, stage[port].transfer.target_layout_id)):
                raise RuntimeError("native Program reached an unbound map port")
            waiting[layout] = port or False
            progress = True
        for identity, route in sorted(stage.items()):
            source, target = route.transfer.source_layout_id, route.transfer.target_layout_id
            if waiting[source] == identity and waiting[target] == identity:
                apply(route)
                waiting[source] = waiting[target] = None
                progress = True
        if not progress:
            raise RuntimeError("native Program map regions cannot satisfy their pending dependencies")
