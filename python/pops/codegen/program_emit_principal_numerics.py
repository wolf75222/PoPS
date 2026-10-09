"""Compose existing authored numerical policies without merging their parameter contexts."""
from types import SimpleNamespace


def _carrier(entry, **policies):
    from pops._ir.values import set_runtime_param_indices
    return SimpleNamespace(**policies, assign_runtime_indices=lambda:
                           set_runtime_param_indices(dict(entry["parameter_indices"])))


def emit_principal_reconstruction(entry):
    from pops.identity.scalar import scalar_cpp
    group, name = entry["group"], entry["cpp_name"]
    method = group.methods[0]
    if method.reconstruction.scheme != "source_stencil":
        policy = method.reconstruction.native_id
        if not policy:
            raise ValueError("principal reconstruction lacks an authenticated native implementation")
        epsilon = method.reconstruction.options.get("epsilon")
        return "", policy + "{" + ("" if epsilon is None else scalar_cpp(epsilon)) + "}"
    from .user_reconstruction_lowering import emit_user_reconstruction_policy
    wrapper = name + "Reconstruction"
    definitions, policies = [], []
    for row, selected in enumerate(group.methods):
        policy = wrapper + "Row%d" % row
        definitions.append(emit_user_reconstruction_policy(
            _carrier(entry, _user_reconstruction=selected.reconstruction)).replace(
                "UserReconstructionPolicy", policy))
        policies.append("pops_generated::" + policy)
    descriptors = tuple(selected.reconstruction for selected in group.methods)
    if any(item.capabilities.get("vector_row") for item in descriptors):
        return _emit_joint_reconstruction(entry, wrapper, definitions, policies, descriptors)
    lines = ["struct %s {" % wrapper,
        "  static constexpr int formal_order=%d;" % min(item.options["formal_order"] for item in descriptors),
        "  static constexpr int n_ghost=%d;" % max(item.options["ghost_depth"] for item in descriptors),
        "  static constexpr int stencil_min_offset=%d;" % min(item.options["stencil_min_offset"] for item in descriptors),
        "  static constexpr int stencil_max_offset=%d;" % max(item.options["stencil_max_offset"] for item in descriptors),
        "  std::array<pops::RuntimeParams,%d> parameter_sets{};" % len(group.states),
        "  template<class Sample> POPS_HD pops::Real stencil_face_value(const Sample& sample) const {"]
    end = 0
    for row, (policy, count, descriptor) in enumerate(zip(policies, group.component_counts, descriptors, strict=True)):
        end += count
        lines += ["    if (sample.component >= 0 && sample.component < %d) {" % end,
                  "      %s policy{};" % policy]
        if descriptor.options.get("runtime_captures"):
            lines.append("      policy.params = parameter_sets[%d];" % row)
        lines += ["      return policy.stencil_face_value(sample);", "    }"]
    lines += ["    return std::numeric_limits<pops::Real>::quiet_NaN();", "  }", "};"]
    return "".join(definitions) + "\n".join(lines) + "\n", wrapper + "{model.parameter_sets}"


def emit_principal_numerical_flux(entry):
    group, name = entry["group"], entry["cpp_name"]
    method = group.methods[0]
    if method.riemann.scheme != "source_face":
        numerical = method.riemann.native_id
        if numerical != "pops::RusanovFlux":
            raise NotImplementedError("principal numerical flux requires a realized common bound")
        return "", numerical + "{}"
    from .user_riemann_lowering import emit_user_face_policy
    wrapper = name + "Numerical"
    definitions, policies = [], []
    for row, selected in enumerate(group.methods):
        policy = wrapper + "Row%d" % row
        definitions.append(emit_user_face_policy(
            _carrier(entry, _user_face=selected.riemann)).replace("UserFacePolicy", policy))
        policies.append("pops_generated::" + policy)
    lines = ["struct %s {" % wrapper,
        "  std::array<pops::RuntimeParams,%d> parameter_sets{};" % len(group.states),
        "  template<int Count> struct RowPhysical {",
        "    static constexpr int n_vars=Count; using State=pops::StateVec<Count>;",
        "    struct ProviderPack {};",
        "    struct Trace { State state{}; pops::FluxDensity<State> density{}; pops::StabilityBound bound{}; };",
        "    POPS_HD pops::FluxDensity<State> evaluate(const Trace& trace,const pops::FaceContext&) const { return trace.density; }",
        "    POPS_HD pops::StabilityBound stability(const Trace& trace,const pops::FaceContext&) const { return trace.bound; }",
        "  };",
        "  template<pops::OrdinaryPhysicalFlux Physical>",
        "  POPS_HD pops::FluxEvaluation<typename Physical::State> operator()(const Physical& physical,",
        "      const typename Physical::Trace& left,const typename Physical::Trace& right,const pops::FaceContext& face) const {",
        "    using State=typename Physical::State;",
        "    if(face.orientation==pops::FaceOrientation::kNegative)",
        "      return pops::detail::canonical_evaluation(*this,physical,left,right,face);",
        "    const auto left_density=physical.evaluate(left,face), right_density=physical.evaluate(right,face);",
        "    auto failure=pops::FluxEvaluation<State>::failed(0);",
        "    if(pops::detail::physical_pair_failed(left_density,right_density,failure)) return failure;",
        "    const auto left_bound=physical.stability(left,face), right_bound=physical.stability(right,face);",
        "    State output{}; pops::StabilityBound bound{};"]
    offset = 0
    for row, (policy, count, selected) in enumerate(zip(policies, group.component_counts, group.methods, strict=True)):
        lines += ["    {", "      RowPhysical<%d> row_physical{};" % count,
            "      typename RowPhysical<%d>::Trace row_left{}, row_right{};" % count,
            "      row_left.bound=left_bound; row_right.bound=right_bound;",
            "      for(int i=0;i<%d;++i) {" % count,
            "        row_left.state[i]=left.state[%d+i]; row_right.state[i]=right.state[%d+i];" % (offset,offset),
            "        row_left.density.value[i]=left_density.value[%d+i]; row_right.density.value[i]=right_density.value[%d+i];" % (offset,offset),
            "      }", "      %s policy{};" % policy]
        if selected.riemann.options.get("runtime_captures"):
            lines.append("      policy.params = parameter_sets[%d];" % row)
        lines += ["      const auto row_evaluation=policy(row_physical,row_left,row_right,face);",
            "      if(!row_evaluation.succeeded()) {",
            "        failure.status=row_evaluation.status; failure.reason_code=row_evaluation.reason_code; return failure;",
            "      }",
            "      pops::StabilityBound combined{};",
            "      if(!pops::detail::max_normal_stability_bound(bound,row_evaluation.stability,combined))",
            "        return pops::FluxEvaluation<State>::reject(pops::RiemannFailureCause::kUserInvalidStability);",
            "      bound=combined; const auto density=row_evaluation.checked_density().value;",
            "      for(int i=0;i<%d;++i) output[%d+i]=density[i];" % (count,offset), "    }"]
        offset += count
    lines += ["    return pops::FluxEvaluation<State>::ok(output,bound);", "  }", "};"]
    return "".join(definitions) + "\n".join(lines) + "\n", wrapper + "{model.parameter_sets}"


def _emit_joint_reconstruction(entry, wrapper, definitions, policies, descriptors):
    group = entry["group"]
    starts, cursor = {}, 0
    for state, count in zip(group.states, group.component_counts, strict=True):
        starts[state] = cursor
        cursor += count
    lines = [
        "struct %s {" % wrapper,
        "  static constexpr int n_components=%d;" % cursor,
        "  static constexpr int formal_order=%d;"
        % min(d.options["formal_order"] for d in descriptors),
        "  static constexpr int n_ghost=%d;" % max(d.options["ghost_depth"] for d in descriptors),
        "  static constexpr int stencil_min_offset=%d;"
        % min(d.options["stencil_min_offset"] for d in descriptors),
        "  static constexpr int stencil_max_offset=%d;"
        % max(d.options["stencil_max_offset"] for d in descriptors),
        "  std::array<pops::RuntimeParams,%d> parameter_sets{};" % len(group.states),
        "  template<class Sample,int Count> struct MappedSample {",
        "    const Sample& source; std::array<int,Count> components; int component=0;",
        "    POPS_HD pops::Real operator()(int offset,int index) const { return source(offset,components[index]); }",
        "    POPS_HD pops::Real operator()(int offset) const { return (*this)(offset,component); }",
        "  };",
        "  template<class Sample> POPS_HD std::array<pops::Real,n_components> stencil_face_state(const Sample& sample) const {",
        "    std::array<pops::Real,n_components> output{};",
    ]
    for row, (state, count, policy, descriptor) in enumerate(
        zip(group.states, group.component_counts, policies, descriptors, strict=True)
    ):
        joint = descriptor.capabilities.get("vector_row", False)
        sources = (
            (descriptor.options["state"], *descriptor.options["sampling"]) if joint else (state,)
        )
        mapping = [
            starts[source] + component
            for source in sources
            for component in range(group.component_counts[group.states.index(source)])
        ]
        lines += ["    {", "      %s policy{};" % policy]
        if descriptor.options["runtime_captures"]:
            lines.append("      policy.params=parameter_sets[%d];" % row)
        lines.append(
            "      MappedSample<Sample,%d> mapped{sample,{{%s}},0};"
            % (len(mapping), ",".join(map(str, mapping)))
        )
        if joint:
            lines += [
                "      const auto trace=policy.stencil_face_state(mapped);",
                "      for(int i=0;i<%d;++i) output[%d+i]=trace[i];" % (count, starts[state]),
            ]
        else:
            lines += [
                "      for(int i=0;i<%d;++i) {" % count,
                "        mapped.component=i; output[%d+i]=policy.stencil_face_value(mapped);"
                % starts[state],
                "      }",
            ]
        lines.append("    }")
    lines += ["    return output;", "  }", "};"]
    return "".join(definitions) + "\n".join(lines) + "\n", wrapper + "{model.parameter_sets}"
