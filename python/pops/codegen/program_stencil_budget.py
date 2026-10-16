"""Resource ceilings for retained Uniform face expressions, including closed regions."""


def uniform_stencil_expression_budget(program):
    from pops.codegen.program_emit_amr import _flux_expression_budgets
    from pops.time._program.resource_budgets import validate_flux_expression_abi
    budgets = _flux_expression_budgets(program)
    validate_flux_expression_abi(budgets)
    bases, terms = sum(row[0] for row in budgets), max((row[1] for row in budgets), default=0)
    if program.flux_expression_budget is None:
        evaluations = 0
        retained = sum(depth+1 for depth in program._histories.values())
        pending = [(tuple(program._values), 1)]
        while pending:
            values, multiplicity = pending.pop()
            for value in values:
                if value.op == "stencil_flux_divergence":
                    evaluations += multiplicity
                if value.op == "continuation_open":
                    retained += len(value.attrs["schema"].get("retained_fields", ()))
                repetitions = 1
                if value.op in ("structured_fold", "structured_iterate"):
                    bound = value.inputs[-1]
                    if bound.op == "int64_constant":
                        repetitions = max(0, bound.attrs["value"])
                    # A runtime-valued bound retains the default finite ceiling. It is
                    # never promoted to compile-time allocation authority; authors may
                    # supply flux_expression_budget for a larger dynamic retained set.
                elif value.op in ("range", "subcycle"):
                    repetitions = value.attrs["count"]
                for key in ("cond_block", "body_block", "true_block", "false_block",
                            "apply_block", "residual_block"):
                    block = value.attrs.get(key)
                    if block:
                        pending.append((block, multiplicity*repetitions))
        bases = max(bases, evaluations*(1+retained), 1)
        terms = max(terms, 1)
    validate_flux_expression_abi(((bases, terms),))
    return bases, terms
