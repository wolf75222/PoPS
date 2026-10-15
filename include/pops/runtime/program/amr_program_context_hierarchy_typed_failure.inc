// Immutable values of a captured typed failure. The scheduler owns control;
// the Python-authored Result owns its mathematical payload.
public:
struct HierarchyTypedFailure {
  ContinuationType type;
  ContinuationValueRef<Dim> value;
  std::shared_ptr<const ContinuationClosure<Dim>> closure;
  finite_value_lease producer_lease;
  PreparedResourceAttempt producer_attempt;
  std::string schema, dispatch_identity;
  std::int64_t operation = -1;
  int producer_level = -1;
  std::uint64_t epoch = 0, generation = 0;
};
template <class Prepare>
[[noreturn]] void raise_hierarchy_typed_failure(std::int64_t operation, Prepare&& prepare,
                                                std::uint64_t byte_capacity) const {
  auto prepared = continuation_local_phase(
      [&] {
        require_hierarchy_failure_handlers_(operation);
        if (!mapping_continuation_->executing || active_level_ < 0)
          throw std::logic_error("typed hierarchy error has no executing producer");
        auto [type, value, schema] = prepare();
        auto result = std::make_shared<HierarchyTypedFailure>();
        result->type = std::move(type);
        result->value = std::move(value);
        result->schema = std::move(schema);
        result->operation = operation;
        result->producer_level = active_level_;
        result->producer_attempt = resource_attempt();
        result->epoch = runtime_->topology_epoch();
        result->generation = runtime_->materialization_generation();
        // Opaque transport keeps each Field's own producer support. The capsule
        // producer need not be its numerical reader or its original author.
        continuation_detail::walk_value<Dim>(
            result->value, byte_capacity,
            [&](const auto& leaf) {
              if (leaf.kind() == ContinuationKind::Field) {
                const auto authority = continuation_field_storage_authority_(leaf.field());
                const auto& support = leaf.field_support();
                if (!authority || !support || !authority->support ||
                    *support != *authority->support ||
                    authority->runtime_owner !=
                        (leaf.field_provider().empty() ? sys_block(leaf.field_owner()) : -1) ||
                    support->field_provider != leaf.field_provider() ||
                    authority->level != support->producer_level ||
                    authority->epoch != runtime_->topology_epoch() ||
                    authority->generation != runtime_->materialization_generation())
                  throw std::invalid_argument(
                      "typed hierarchy Field has no exact retained producer support");
              }
              return true;
            },
            [](const auto&) {});
        ExactContractBuilder contract;
        contract.text("pops.hierarchy.typed-capture@1")
            .bytes(mapping_continuation_->failures.front().back().identity)
            .scalar(mapping_continuation_->internal_sequence)
            .scalar(active_level_)
            .scalar(result->producer_attempt.ordinal())
            .text(result->schema);
        result->dispatch_identity = std::move(contract).release();
        return result;
      },
      "typed hierarchy error preparation");
  prepared->producer_lease = current_finite_value_lease();
  auto& live = finite_values_();
  auto registries = continuation_local_phase(
      [&] {
        auto result = live.protected_distributed;
        if (live.distributed &&
            std::find(result.begin(), result.end(), live.distributed) == result.end())
          result.push_back(live.distributed);
        return result;
      },
      "typed hierarchy error dependency authority");
  prepared->closure = continuation_collective_closure<Dim>(prepared->value, live.distributed,
                                                           prepared_execution_lane(), byte_capacity,
                                                           live.finite.realm(), registries);
  auto identity = continuation_local_phase(
      [&] {
        ContinuationCodec<Dim> codec(
            prepared->type, prepared->schema, byte_capacity,
            [&](const ContinuationType& type,
                const ContinuationFieldSupport::Ref& support) -> const field_type& {
              if (!support)
                throw std::invalid_argument("typed hierarchy Field omitted its producer support");
              return continuation_field_prototype_(type, prepared->producer_level, support);
            },
            &prepared_execution_lane(), live.distributed.get(), prepared->closure.get());
        return codec.collective_identity(prepared->value, 0);
      },
      "typed hierarchy error value validation");
  const std::array<std::pair<std::string_view, std::string_view>, 2> contract{
      {{"dispatch", prepared->dispatch_identity}, {"value", identity}}};
  if (!all_ranks_agree_exact_ordered_byte_pairs(contract, prepared_execution_lane()))
    throw std::logic_error("typed hierarchy errors disagree across the execution lane");
  auto rejection = continuation_local_phase(
      [&] {
        return std::make_exception_ptr(
            StepAttemptRejected(SolveStatus::kInvalidEvaluation, StepAttemptDisposition::kReject, 0,
                                "structured_control", "captured typed hierarchy error",
                                "pops.hierarchy.typed-capture@1", prepared->dispatch_identity));
      },
      "typed hierarchy error dispatch preparation");
  mapping_continuation_->typed_failure = std::move(prepared);
  std::rethrow_exception(rejection);
}

hierarchy_typed_failure hierarchy_failure_value(const StepAttemptRejected& failure,
                                                std::int64_t operation, std::string_view schema,
                                                std::uint64_t byte_capacity) const {
  auto payload = continuation_local_phase(
      [&] {
        if (!mapping_continuation_ || !mapping_continuation_->typed_failure)
          throw std::logic_error("typed hierarchy error capsule is absent");
        auto result = mapping_continuation_->typed_failure;
        if (failure.typed_failure_schema() != "pops.hierarchy.typed-capture@1" ||
            result->dispatch_identity != failure.typed_failure_payload() ||
            result->operation != operation || result->schema != schema ||
            !result->producer_attempt.visible() || result->epoch != runtime_->topology_epoch() ||
            result->generation != runtime_->materialization_generation())
          throw std::logic_error("typed hierarchy error lost its exact producer authority");
        return result;
      },
      "typed hierarchy error capsule authority");
  continuation_local_phase(
      [&] {
        auto& live = finite_values_();
        for (const auto& registry : payload->closure->registries) {
          live.transport.bind_distributed_registry(
              *registry, [registry](distributed::Identity version, distributed::DomainRef domain,
                                    const ContinuationType& element) {
                registry->authenticate(domain);
                auto array = std::static_pointer_cast<const ContinuationDistributedArray<Dim>>(
                    registry->resolve_value(version, distributed::array_type_tag<Dim>()));
                if (!array || array->domain != domain || array->element != element ||
                    array->version.issuer != version.issuer ||
                    array->version.ordinal != version.ordinal)
                  throw std::invalid_argument("typed failure Array has no exact retained version");
                return array;
              });
          if (std::find(live.protected_distributed.begin(), live.protected_distributed.end(),
                        registry) == live.protected_distributed.end())
            live.protected_distributed.push_back(registry);
        }
        live.transport.capture(payload->value, byte_capacity);
        for (const auto& array : payload->closure->arrays)
          live.transport.capture(ContinuationValue<Dim>::distributed_array(array), byte_capacity);
      },
      "typed hierarchy error dependency retention");
  return payload;
}
