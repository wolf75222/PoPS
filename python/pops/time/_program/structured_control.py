"""Typed products and bounded native regions, elaborated into the existing Program IR.

Products are structural values, not physics-specific containers.  An iteration result is opaque:
only ``match`` or ``require_success`` can expose its successful state.  Python callbacks below run
once while building the graph; generated C++ and PoPS field storage execute all iterations.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, cast
from collections.abc import Callable

if TYPE_CHECKING:
    from pops.time._program.contract import _ProgramBase
else:
    _ProgramBase = object

from pops.time._authoring import atomic_authoring
from pops.time.values import ProgramValue
from pops.time.value_signatures import value_signature, same_signature

if TYPE_CHECKING:
    from pops.time.value_signatures import ValueSignature
from pops.time._program.value_validation import require_owned, require_region, validate_input_regions


def product_signature(fields) -> ValueSignature:
    return cast("ValueSignature", {"vtype": "product", "members": tuple(fields)})


def scalar_signature(kind) -> ValueSignature:
    return cast("ValueSignature", dict(vtype=kind, block=None, space=None, state_ref=None, field_context=None))


def validate_boolean_operation(inputs, attrs, signature):
    operation = attrs.get("operation")
    if (operation not in ("and", "or", "not") or signature["vtype"] != "bool"
            or len(inputs) != (1 if operation == "not" else 2)
            or any(value.vtype != "bool" for value in inputs)):
        raise TypeError("boolean algebra requires exact Bool operands and arity")


def new_typed(program, signature, op, inputs, attrs, *, point=None, name=None) -> ProgramValue:
    attrs = dict(attrs)
    if signature.get("field_authority") is not None:
        attrs["field_authority"] = signature["field_authority"]
    if signature["vtype"] in ("product", "variant", "index_set", "index", "array", "continuation_base", "continuation_candidate", "continuation_assessment", "continuation_accepted", "field_topology"):
        attrs["signature"] = signature
        if point is None:
            point = signature.get("point")
    elif signature["vtype"] == "result":
        attrs["success_signature"] = signature["success"]
        if point is None:
            point = signature.get("point")
    elif point is None:
        point = signature.get("point")
    return program._new(
        signature["vtype"], op, inputs, attrs, name, signature.get("block"),
        space=signature.get("space"), field_context=signature.get("field_context"),
        state_ref=signature.get("state_ref"), point=point, inherit_state_ref=False)


def iteration_signature(result) -> ValueSignature:
    return product_signature((
        ("state", result.attrs["state_signature"]),
        ("iterations", {**scalar_signature("int64"), "point": result.point}),
    ))


def success_signature(result) -> ValueSignature:
    if result.vtype == "variant":
        members = dict(value_signature(result)["members"])
        if set(members) == {"ok", "err"}:
            return members["ok"]
        raise TypeError("require_success requires Variant(ok=T, err=E)")
    if result.vtype == "iteration_result":
        return iteration_signature(result)
    if result.vtype == "result":
        return result.attrs["success_signature"]
    raise TypeError("expected a typed Result or IterationResult")


def failure_signature(result) -> ValueSignature:
    if result.vtype == "variant":
        success_signature(result)
        return dict(value_signature(result)["members"])["err"]
    code = ("code", {**scalar_signature("int64"), "point": result.point})
    origin = ("origin", {**scalar_signature("int64"), "point": result.point})
    return product_signature((code, origin, ("iterations", {**scalar_signature("int64"), "point": result.point}))
                             if result.vtype == "iteration_result" else (code, origin))


def _error_members(signature):
    return (dict(signature["members"]) if signature["vtype"] == "variant"
            else {"user": signature})


def _typed_capture_errors(block, output):
    """Infer the closed typed error union at this capture, stopping at child captures."""
    from pops.time.variant_types import RuntimeFailure, instantiate
    members = {"runtime": instantiate(RuntimeFailure, output.point).signature}
    found = []
    def visit(nodes):
        for node in nodes:
            if node.op == "variant_require_success":
                found.append(failure_signature(node.inputs[0]))
            if node.op != "capture_result":
                for key in ("body_block", "cond_block", "true_block", "false_block"):
                    visit(node.attrs.get(key, ()))
    visit(block)
    is_result = output.vtype == "variant" and set(dict(value_signature(output)["members"])) == {"ok", "err"}
    if is_result:
        found.append(failure_signature(output))
    if not found:
        return None, False
    for signature in found:
        for tag, payload in _error_members(signature).items():
            if tag in members and not same_signature(members[tag], payload):
                raise TypeError("capture error tag %r has incompatible payload types; name errors explicitly" % tag)
            members[tag] = payload
    return {"vtype": "variant", "members": tuple(sorted(members.items()))}, is_result


def _record_region(program, callback: Callable[..., ProgramValue], initial, *, with_index=False):
    block = []
    region = program._region_for_block(block)
    for parent in program._recording:
        program._allow_region_capture(program._region_for_block(parent), region)
    if initial.region:
        program._allow_region_capture(initial.region, region)
    program._recording.append(block)
    try:
        argument = new_typed(program, value_signature(initial), "region_argument", (initial,),
                             {}, point=initial.point)
        if with_index:
            index = new_typed(program, scalar_signature("int64"), "region_index", (), {},
                              point=initial.point)
            result = callback(program, argument, index)
        else:
            index = None
            result = callback(program, argument)
        require_region(program, result, region, "structured region result", allow=(argument,))
    finally:
        program._recording.pop()
    return block, region, argument, index, result


class _ProgramStructuredControl(_ProgramBase):
    @atomic_authoring
    def variant(self, variant_type, tag, payload):
        """Inject a typed native payload into a closed, case-authored tagged sum."""
        from pops.time.variant_types import instantiate_variant_payload
        from pops.time.points import TimePoint
        require_owned(self, payload, "variant payload")
        # The envelope is structural; infer unbound scalar parameters from leaves,
        # never from a nested sum envelope or by changing an exact authority.
        point = TimePoint(self.clock)
        signature = instantiate_variant_payload(
            variant_type, tag, value_signature(payload), point).signature
        if signature["vtype"] != "variant":
            raise TypeError("variant requires a Variant type")
        members = dict(signature["members"])
        if not isinstance(tag, str) or tag not in members:
            raise ValueError("unknown Variant tag %r" % tag)
        require_owned(self, payload, "variant payload")
        if not same_signature(members[tag], value_signature(payload)):
            raise TypeError("variant payload must have the tag's exact structural type and authority")
        return new_typed(self, signature, "variant_construct", (payload,), {"tag": tag},
                         point=point)

    def _match_variant(self, value, arms):
        members = dict(value_signature(value)["members"])
        if set(arms) != set(members):
            raise TypeError("match requires exactly every Variant tag; missing=%r extra=%r"
                            % (sorted(set(members)-set(arms)), sorted(set(arms)-set(members))))
        if any(not callable(arm) for arm in arms.values()):
            raise TypeError("match arms must be callable")
        tags = tuple(members)
        expected = []
        expected_points = []
        def dispatch(index):
            tag = tags[index]
            condition = new_typed(self, scalar_signature("bool"), "variant_is", (value,), {"tag": tag})
            def selected(program):
                payload = new_typed(program, members[tag], "variant_payload", (value,), {"tag": tag})
                result = arms[tag](program, payload)
                require_owned(program, result, "match %s result" % tag)
                signature = value_signature(result)
                if expected and not same_signature(expected[0], signature):
                    raise TypeError("match arms must return the same structural type and authorities")
                if not expected:
                    expected.append(signature)
                    expected_points.append(result.point)
                return result
            def otherwise(program):
                if index+1 < len(tags):
                    return dispatch(index+1)
                return new_typed(program, expected[0], "variant_unreachable", (value,), {}, point=expected_points[0])
            return self.branch(condition, selected, otherwise)
        return dispatch(0)

    @atomic_authoring
    def product(self, **fields):
        """Construct an immutable named product; fields may themselves be products."""
        for name, value in fields.items():
            if not isinstance(name, str) or not name:
                raise TypeError("product field names must be non-empty strings")
            require_owned(self, value, "product field %s" % name)
            if value.vtype in ("result", "iteration_result"):
                raise TypeError("product fields require handled Result values")
        signature = product_signature(tuple((name, value_signature(value))
                                            for name, value in fields.items()))
        return new_typed(self, signature, "product", tuple(fields.values()),
                         {"fields": tuple(fields)})

    @atomic_authoring
    def _product_get(self, value, key):
        require_owned(self, value, "product projection", vtype="product")
        if not isinstance(key, str):
            raise TypeError("product projection requires a field name")
        fields = dict(value_signature(value)["members"])
        if key not in fields:
            raise KeyError(key)
        return new_typed(self, fields[key], "product_get", (value,), {"field": key})

    @atomic_authoring
    def int64(self, value):
        """An exact Int64 constant or checked native conversion from a Scalar.

        A Scalar must be finite, integral and in [-2**63,2**63); no truncation or wrapping occurs.
        """
        if isinstance(value, ProgramValue):
            require_owned(self, value, "int64")
            if value.vtype == "int64":
                return value
            if value.vtype != "scalar":
                raise TypeError("int64 requires a Python int or native Scalar")
            return new_typed(self, scalar_signature("int64"), "int64_cast", (value,), {})
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError("int64 requires a Python int or native Scalar")
        if not -(1 << 63) <= value < (1 << 63):
            raise ValueError("int64 constant is out of range")
        return new_typed(self, scalar_signature("int64"), "int64_constant", (), {"value": value})

    @atomic_authoring
    def boolean(self, value):
        if type(value) is not bool:
            raise TypeError("boolean requires an exact Python bool")
        return new_typed(self, scalar_signature("bool"), "bool_constant", (), {"value": value})

    @atomic_authoring
    def _boolean_op(self, values, operation):
        """Pure algebra over already-evaluated Bool values, with no implicit truth conversion."""
        for value in values:
            if type(value) is bool:
                continue
            require_owned(self, value, "boolean algebra")
            if value.vtype != "bool":
                raise TypeError("boolean algebra requires Bool operands")
        inputs = tuple(self.boolean(value) if type(value) is bool else value for value in values)
        signature, attrs = scalar_signature("bool"), {"operation": operation}
        validate_boolean_operation(inputs, attrs, signature)
        return new_typed(self, signature, "bool_op", inputs, attrs)

    @atomic_authoring
    def _int64_binop(self, a, b, operation):
        a, b = self.int64(a), self.int64(b)
        if operation not in ("add", "sub", "mul", "floordiv", "mod"):
            raise TypeError("unsupported Int64 operation %r" % operation)
        return new_typed(self, scalar_signature("int64"), "int64_op", (a, b),
                         {"operation": operation})

    @atomic_authoring
    def _int64_compare(self, a, b, comparison):
        a, b = self.int64(a), self.int64(b)
        if comparison not in ("<", "<=", ">", ">=", "==", "!="):
            raise TypeError("unsupported Int64 comparison")
        return new_typed(self, scalar_signature("bool"), "int64_compare", (a, b),
                         {"comparison": comparison})

    @atomic_authoring
    def fold(self, initial, count, body_fn, *, max_steps=None):
        """Fold a bounded native range: ``body_fn(P, carried_value, int64_index)``.

        A dynamic count requires an explicit finite ``max_steps`` capacity.  All leaves are updated
        simultaneously, so swaps and cross-coupled fields do not depend on field order.
        """
        require_owned(self, initial, "fold initial")
        signature = value_signature(initial)
        if initial.vtype == "result":
            raise TypeError("fold initial requires a handled Result value")
        validate_input_regions(self, (initial,), self._current_region(), "fold")
        if not callable(body_fn):
            raise TypeError("fold body must be callable")
        if max_steps is None:
            if type(count) is not int:
                raise ValueError("fold with a native count requires max_steps")
            max_steps = count
        count_value, budget = self.int64(count), self.int64(max_steps)
        if type(count) is int and count < 0:
            raise ValueError("fold count must be non-negative")
        if type(max_steps) is int and max_steps < 0:
            raise ValueError("fold max_steps must be non-negative")
        block, region, argument, index, result = _record_region(
            self, body_fn, initial, with_index=True)
        if not same_signature(signature, value_signature(result)):
            raise TypeError("fold body must preserve every carried field's type and authority")
        return new_typed(self, signature, "structured_fold", (initial, count_value, budget),
                         {"body_block": block, "body_region": region, "body": result,
                          "body_argument": argument, "body_index": index}, point=initial.point)

    @atomic_authoring
    def iterate(self, initial, step: Callable[..., ProgramValue], stop: Callable[..., ProgramValue], *, max_steps):
        """Bounded native iteration returning an opaque success/failure sum.

        ``stop(P, x)`` is evaluated before the first step and after each successful step.  A zero
        budget may succeed only if the initial stop holds.  Success contains ``state`` and
        ``iterations``; failure contains ``code`` and ``iterations``.  Codes: 1 budget exhausted,
        2 invalid budget, 3 rejected native evaluation.  Access is through match/require_success.
        """
        require_owned(self, initial, "iterate initial")
        signature = value_signature(initial)
        if initial.vtype == "result":
            raise TypeError("iterate initial requires a handled Result value")
        validate_input_regions(self, (initial,), self._current_region(), "iterate")
        if not callable(step) or not callable(stop):
            raise TypeError("iterate step and stop must be callable")
        budget = self.int64(max_steps)
        cond_block, cond_region, cond_argument, _, condition = _record_region(self, stop, initial)
        if condition.vtype != "bool":
            raise TypeError("iterate stop must return a native Bool")
        body, body_region, body_argument, _, next_value = _record_region(self, step, initial)
        typed_result = next_value.vtype == "variant" and set(dict(value_signature(next_value)["members"])) == {"ok", "err"}
        next_signature = (success_signature(next_value) if next_value.vtype == "result" or typed_result
                          else value_signature(next_value))
        if not same_signature(signature, next_signature):
            raise TypeError("iterate step must preserve every carried field's type and authority")
        attrs = {"state_signature": signature, "cond_block": cond_block, "cond_region": cond_region,
             "cond": condition, "cond_argument": cond_argument,
             "body_block": body, "body_region": body_region, "body": next_value,
             "body_argument": body_argument, "body_is_result": next_value.vtype == "result",
             "body_is_variant_result": typed_result}
        errors, _ = _typed_capture_errors(cond_block + body, next_value)
        if errors is not None:
            success = product_signature((("state", signature),
                                         ("iterations", {**scalar_signature("int64"), "point": initial.point})))
            attrs["error_signature"] = errors
            return new_typed(self, {"vtype": "variant", "members": (("err", errors), ("ok", success))},
                             "structured_iterate", (initial, budget), attrs, point=initial.point)
        return self._new("iteration_result", "structured_iterate", (initial, budget), attrs,
            None, None, point=initial.point,
            inherit_state_ref=False)

    @atomic_authoring
    def success(self, value):
        """Construct Result[T, code:Int64]. Its successful value is only exposed by match/guard."""
        require_owned(self, value, "success")
        if value.vtype in ("result", "iteration_result"):
            raise TypeError("success requires a handled Result value; nested Results are not implicit")
        return self._new("result", "result_ok", (value,),
                         {"success_signature": value_signature(value)}, None, None,
                         point=value.point, inherit_state_ref=False)

    @atomic_authoring
    def failure(self, success_template, code):
        """Construct an opaque failure of the template's type; a nonzero Int64 is required.

        The error payload is ``(code: Int64, origin: Int64)``: origin 1 means user failure, origin 2
        means a checked runtime failure. User codes never collide semantically with runtime codes.
        The template supplies the success
        type and storage geometry, never a substitute successful value. No Python object payload
        or callback is installed in native code.
        """
        require_owned(self, success_template, "failure success template")
        if success_template.vtype in ("result", "iteration_result"):
            raise TypeError("failure template requires a handled Result value")
        if type(code) is int and code == 0:
            raise ValueError("failure code must be nonzero")
        error = self.int64(code)
        return self._new("result", "result_error", (success_template, error),
                         {"success_signature": value_signature(success_template)}, None, None,
                         point=success_template.point, inherit_state_ref=False)

    @atomic_authoring
    def capture(self, body: Callable[..., ProgramValue], *, _continuation_scope=False):
        """Capture recoverable failures in one closed native region, without nested Result sugar."""
        if not callable(body):
            raise TypeError("capture body must be callable accepting Program")
        block = []
        region = self._region_for_block(block)
        for parent in self._recording:
            self._allow_region_capture(self._region_for_block(parent), region)
        self._recording.append(block)
        try:
            output = body(self)
            require_owned(self, output, "capture result")
            validate_input_regions(self, (output,), region, "capture result")
        finally:
            self._recording.pop()
        signature = (success_signature(output) if output.vtype == "result"
                     else value_signature(output))
        errors, is_typed_result = _typed_capture_errors(block, output)
        if errors is not None:
            if is_typed_result:
                signature = success_signature(output)
            result_signature = {"vtype": "variant", "members": (("err", errors), ("ok", signature))}
            return new_typed(self, result_signature, "capture_result", (),
                             {"continuation_scope": bool(_continuation_scope), "success_signature": signature, "error_signature": errors,
                              "body_block": block, "body_region": region, "body": output,
                              "body_is_variant_result": is_typed_result,
                              "body_is_result": output.vtype == "result"}, point=output.point)
        return self._new("result", "capture_result", (),
                         {"continuation_scope": bool(_continuation_scope), "success_signature": signature, "body_block": block,
                          "body_region": region, "body": output,
                          "body_is_result": output.vtype == "result"}, None, None,
                         point=output.point, inherit_state_ref=False)

    @atomic_authoring
    def require_success(self, result):
        """Expose (state, iterations) only after a native success check; otherwise reject attempt."""
        require_owned(self, result, "require_success")
        signature = success_signature(result)
        if result.vtype == "variant":
            return new_typed(self, signature, "variant_require_success", (result,), {})
        return new_typed(self, signature, "iteration_success" if result.vtype == "iteration_result"
                         else "result_success", (result,), {})

    @atomic_authoring
    def is_ok(self, result):
        require_owned(self, result, "is_ok")
        success_signature(result)
        if result.vtype == "variant":
            return new_typed(self, scalar_signature("bool"), "variant_is", (result,), {"tag": "ok"})
        return new_typed(self, scalar_signature("bool"), "iteration_is_ok" if result.vtype == "iteration_result"
                         else "result_is_ok", (result,), {})

    @atomic_authoring
    def match(self, result, **arms):
        """Lazily handle both variants with callbacks ``arm(P, payload)``.

        Only the ok region receives the iteration state.  Both arms must return the same structural
        type; callbacks cannot retain a payload for use outside its authorized region.
        """
        require_owned(self, result, "match")
        validate_input_regions(self, (result,), self._current_region(), "match")
        if result.vtype == "variant":
            return self._match_variant(result, arms)
        if set(arms) != {"ok", "err"}:
            raise TypeError("Result match requires exactly ok and err arms")
        ok, err = arms["ok"], arms["err"]
        success_type = success_signature(result)
        validate_input_regions(self, (result,), self._current_region(), "match")
        if not callable(ok) or not callable(err):
            raise TypeError("match arms must be callable")
        condition = self.is_ok(result)
        failure = failure_signature(result)
        records = []
        for variant, callback, signature in (("ok", ok, success_type),
                                              ("err", err, failure)):
            block = []
            region = self._region_for_block(block)
            for parent in self._recording:
                self._allow_region_capture(self._region_for_block(parent), region)
            if result.region:
                self._allow_region_capture(result.region, region)
            self._recording.append(block)
            try:
                payload = new_typed(self, signature, "iteration_payload" if result.vtype == "iteration_result"
                                    else "result_payload", (result,),
                                    {"variant": variant}, point=result.point)
                output = callback(self, payload)
                require_owned(self, output, "match %s result" % variant)
                validate_input_regions(self, (output,), region, "match %s result" % variant)
            finally:
                self._recording.pop()
            records.append((block, region, output))
        (left, left_region, a), (right, right_region, b) = records
        if not same_signature(value_signature(a), value_signature(b)):
            raise TypeError("match arms must return the same structural type and authorities")
        return new_typed(self, value_signature(a), "result_match", (result, condition),
                         {"true_block": left, "true_region": left_region, "true_result": a,
                          "false_block": right, "false_region": right_region, "false_result": b},
                         point=a.point)


__all__ = ["_ProgramStructuredControl", "value_signature", "same_signature"]
