"""Authenticate two-stage diffusion receipts against their retained Program."""


def ssprk2_stage_indices(program, contexts):
    rates = tuple(value for value in program._values if value.op == "diffusive_rhs")
    assert len(rates) == 2 and len(contexts) == 2
    # The initial evaluation keeps the accepted TimePoint; the second reads the
    # authored forward-Euler predictor. Neither relies on ledger record order.
    assert rates[0].inputs[0].op == "state"
    assert rates[1].inputs[0].op == "linear_combine"
    assert all(rate.point == rate.inputs[0].point for rate in rates)
    tags = tuple("stage:" + str(rate.point) + "/evaluation:" + str(rate.id)
                 for rate in rates)
    expected = tuple("/" + str(len(tag.encode("utf-8"))) + ":" + tag for tag in tags)
    result = {}
    for context in contexts:
        matched = tuple(index for index, tag in enumerate(expected)
                        if context.endswith(tag))
        assert len(matched) == 1, (context, expected)
        result[context] = matched[0]
    assert set(result.values()) == {0, 1}
    return result
