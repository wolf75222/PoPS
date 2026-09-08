"""Resolve declared coupled output roles to admitted owner-qualified input blocks."""
from __future__ import annotations


def coupled_output_inputs(bundle, values):
    from pops.time.references import block_name
    result = {}
    for output, rate in bundle.items():
        matches = [value for value in values if value.space == rate.base_space]
        if len(matches) != 1:
            # The historical RateBundle key is an explicit output role. When several instances
            # share one StateSpace, resolve that role among exact admitted BlockHandles, never
            # choose an arbitrary first instance or confuse type equality with owner identity.
            matches = [value for value in matches if block_name(value.block) == output]
        if len(matches) != 1:
            raise ValueError("coupled output %r requires one exact owner-qualified input binding" % output)
        result[output] = matches[0]
    if len({value.block for value in result.values()}) != len(result):
        raise ValueError("coupled output roles must not alias the same input block")
    return result
