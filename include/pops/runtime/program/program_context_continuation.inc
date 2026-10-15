// Included in ProgramContext<Dim>. All accepted data lives in RuntimeState::continuation_;
// the context owns only non-restorable live capabilities and burned instance ordinals.
public:
using continuation_authority_type = ContinuationAuthority<Dim>;
using continuation_ref = ContinuationValueRef<Dim>;

ContinuationTransportSession<Dim>& continuation_transport_session() const {
  return algebra_lifetime_->transport;
}

std::uint64_t continuation_byte_capacity() const {
  return runtime_state().continuation_.configuration().byte_capacity;
}

template <class Function>
auto continuation_local_phase(Function&& function, const char* label) const {
  using Result = std::invoke_result_t<Function>;
  std::exception_ptr error;
  if constexpr (std::is_void_v<Result>) {
    try {
      function();
    } catch (...) {
      error = std::current_exception();
    }
    try {
      device_fence();
    } catch (...) {
      error = std::current_exception();
    }
    continuation_rethrow_(error, label);
  } else {
    std::optional<Result> result;
    try {
      result.emplace(function());
    } catch (...) {
      error = std::current_exception();
    }
    try {
      device_fence();
    } catch (...) {
      error = std::current_exception();
    }
    continuation_rethrow_(error, label);
    return std::move(*result);
  }
}

void continuation_control_consensus() const {
  const auto state = algebra_lifetime_->continuation.logical_state();
  for (auto word : state)
    continuation_word_agrees_(word, "continuation dynamic control differs across ranks");
}

std::int64_t next_finite_domain_version() const {
  // Burned counters may differ after a failed local allocation. Synchronize upward before
  // issuance; never rewind a surviving rank or convert Int64 to floating point.
  const auto local = static_cast<std::uint64_t>(continuation_local_phase(
      [&] { return algebra_lifetime_->finite.next_version(); }, "finite-domain ordinal issuance"));
  const auto& lane = prepared_execution_lane();
  constexpr std::uint64_t mask = (UINT64_C(1) << 21) - 1;
  const auto high = all_reduce_max(static_cast<long>(local >> 42), lane);
  const auto middle = all_reduce_max(
      static_cast<long>((local >> 42) == std::uint64_t(high) ? (local >> 21) & mask : 0), lane);
  const auto low =
      all_reduce_max(static_cast<long>((local >> 42) == std::uint64_t(high) &&
                                               ((local >> 21) & mask) == std::uint64_t(middle)
                                           ? local & mask
                                           : 0),
                     lane);
  const auto shared = static_cast<std::int64_t>((std::uint64_t(high) << 42) |
                                                (std::uint64_t(middle) << 21) | std::uint64_t(low));
  algebra_lifetime_->finite.import_high_water(shared);
  return shared;
}

FiniteDomainRef continuation_new_domain(std::int64_t count, std::int64_t capacity,
                                        std::int64_t family) const {
  const auto& store = runtime_state().continuation_;
  algebra_lifetime_->finite.import_high_water(store.domain_high_water());
  // Agree the requested replicated domain before allocation. Signed words retain all bits.
  continuation_word_agrees_(std::bit_cast<std::uint64_t>(count),
                            "finite-domain size differs across ranks");
  continuation_word_agrees_(std::bit_cast<std::uint64_t>(capacity),
                            "finite-domain capacity differs across ranks");
  continuation_word_agrees_(std::bit_cast<std::uint64_t>(family),
                            "finite-domain family differs across ranks");
  (void)next_finite_domain_version();
  auto domain = continuation_local_phase(
      [&] {
        return algebra_lifetime_->finite.issue(count, capacity, family, store.domain_realm());
      },
      "finite-domain allocation");
  continuation_word_agrees_(static_cast<std::uint64_t>(domain->version),
                            "finite-domain ordinal differs across ranks");
  continuation_synchronize_realm_(domain->realm());
  continuation_local_phase(
      [&] { algebra_lifetime_->transport.capture(ContinuationValue<Dim>::domain(domain)); },
      "finite-domain transport registration");
  return domain;
}

// Only a rank-local algebra region may call this factory. Its issuer is
// deliberately separate from the replicated collective domain allocator.
// Allocation failure is handled by the enclosing local region before its
// next communication phase; there is no hidden rendezvous in this method.
FiniteDomainRef continuation_new_local_domain(std::int64_t count, std::int64_t capacity,
                                              std::int64_t family) const {
  auto domain = algebra_lifetime_->local_finite.issue(count, capacity, family);
  algebra_lifetime_->transport.capture(ContinuationValue<Dim>::domain(domain));
  return domain;
}

void continuation_configure(ContinuationConfiguration configuration) const {
  continuation_local_phase(
      [&] {
        configuration.validate();
        if (configuration.fields.size() != static_cast<std::size_t>(n_blocks()))
          throw std::invalid_argument(
              "Continuation must bind every native state block exactly once");
        std::set<int> owners;
        for (const auto& binding : configuration.fields) {
          if (!owners.insert(sys_block(binding.program_block)).second ||
              binding.clock_identity !=
                  runtime_state()
                      .checkpoint_metadata_.uniform_auxiliary_clocks.primary_clock_identity)
            throw std::invalid_argument("Continuation binding has foreign block/clock authority");
          const auto& type = configuration.at(binding.path);
          if (type.field_components != state(binding.program_block).ncomp())
            throw std::invalid_argument(
                "Continuation binding changes the carrier component schema");
        }
        continuation_require_profile_();
        runtime_state().continuation_.configure(std::move(configuration));
      },
      "continuation schema installation");
}

bool continuation_initialized() const {
  const bool initialized = runtime_state().continuation_.initialized();
  continuation_word_agrees_(initialized ? 1 : 0,
                            "continuation initialization differs across ranks");
  return initialized;
}

void continuation_initialize(continuation_ref initial) const {
  initial =
      continuation_local_phase([&] { return continuation_accepted_origins_(std::move(initial)); },
                               "continuation initial field origins");
  auto closure = continuation_collective_closure<Dim>(
      initial, algebra_lifetime_->distributed.get(), prepared_execution_lane(),
      runtime_state().continuation_.configuration().byte_capacity,
      runtime_state().continuation_.domain_realm() ? runtime_state().continuation_.domain_realm()
                                                   : algebra_lifetime_->finite.realm());
  auto prepared = continuation_local_phase(
      [&] {
        auto& store = runtime_state().continuation_;
        continuation_require_profile_();
        auto out = store.prepare(std::move(initial), continuation_prototypes_(), true,
                                 algebra_lifetime_->distributed, std::move(closure));
        continuation_require_frontier_(out.value, system_->time(), macro_step());
        continuation_require_projection_(out.value);
        return out;
      },
      "continuation initial image");
  continuation_image_consensus_(prepared);
  runtime_state().continuation_.publish(std::move(prepared));
}

typename continuation_authority_type::Base continuation_read() const {
  const auto realm = runtime_state().continuation_.domain_realm();
  continuation_word_agrees_(realm ? 1 : 0,
                            "continuation finite realm presence differs across ranks");
  // A checkpoint reconstructs private capabilities independently on each rank.
  // Establish a common transport origin before any restored alias can escape.
  if (realm)
    continuation_synchronize_realm_(realm);
  auto base = continuation_local_phase(
      [&] {
        auto& store = runtime_state().continuation_;
        continuation_require_profile_();
        store.require_frontier(system_->time(), macro_step());
        continuation_require_projection_(store.read());
        algebra_lifetime_->finite.import_high_water(store.domain_high_water());
        for (const auto& restored_realm : store.domain_realms())
          algebra_lifetime_->finite.adopt(restored_realm);
        if (auto registry = algebra_lifetime_->distributed) {
          algebra_lifetime_->transport.bind_distributed_registry(
              *registry, [registry](distributed::Identity version, distributed::DomainRef domain,
                                    const ContinuationType& element) {
                registry->authenticate(domain);
                auto array = std::static_pointer_cast<const ContinuationDistributedArray<Dim>>(
                    registry->resolve_value(version, distributed::array_type_tag<Dim>()));
                if (!array || array->domain != domain || array->element != element ||
                    array->version.issuer != version.issuer ||
                    array->version.ordinal != version.ordinal)
                  throw std::invalid_argument(
                      "restored distributed Array differs from its exact vault version");
                return array;
              });
        }
        algebra_lifetime_->transport.capture(store.read());
        if (store.closure())
          for (const auto& array : store.closure()->arrays)
            algebra_lifetime_->transport.capture(ContinuationValue<Dim>::distributed_array(array));
        return algebra_lifetime_->continuation.read(store);
      },
      "continuation accepted read");
  continuation_control_consensus();
  return base;
}

continuation_authority_type& continuation_authority() const {
  return algebra_lifetime_->continuation;
}

void continuation_publish(const typename continuation_authority_type::Accepted& accepted) const {
  continuation_control_consensus();
  const auto selected = continuation_local_phase(
      [&] { return algebra_lifetime_->continuation.publication_state(accepted); },
      "continuation selected candidate authority");
  for (auto word : selected)
    continuation_word_agrees_(word, "different dynamic candidates selected for root publication");
  std::vector<std::pair<field_type*, const field_type*>> projections;
  auto value = continuation_local_phase(
      [&] {
        auto& runtime = runtime_state();
        continuation_require_profile_();
        if (!runtime.cadence_continuation_ || !runtime.cadence_continuation_->executing ||
            runtime.stride_ != 1 || runtime.substeps_ != 1)
          throw std::logic_error(
              "Continuation publication requires one active complete root frontier");
        const auto& frame = *runtime.cadence_continuation_;
        auto value = algebra_lifetime_->continuation.require_root_publication(
            accepted, runtime.continuation_);
        if (frame.accepted_macro_step == std::numeric_limits<int>::max())
          throw std::overflow_error("Continuation macro-step cursor exhausted");
        continuation_require_frontier_(value, frame.partition.end,
                                       static_cast<std::int64_t>(frame.accepted_macro_step) + 1);
        return value;
      },
      "continuation root candidate frontier");
  // Acceptance turns only the declared State projections into new origins. Auxiliary Field
  // aliases retain their numerical expression; no field data or authored decision is changed.
  value = continuation_local_phase([&] { return continuation_accepted_origins_(std::move(value)); },
                                   "continuation accepted field origins");
  auto closure = continuation_collective_closure<Dim>(
      value, algebra_lifetime_->distributed.get(), prepared_execution_lane(),
      runtime_state().continuation_.configuration().byte_capacity,
      runtime_state().continuation_.domain_realm() ? runtime_state().continuation_.domain_realm()
                                                   : algebra_lifetime_->finite.realm());
  auto prepared = continuation_local_phase(
      [&] {
        auto& runtime = runtime_state();
        auto out =
            runtime.continuation_.prepare(std::move(value), continuation_prototypes_(), false,
                                          algebra_lifetime_->distributed, std::move(closure));
        projections.reserve(runtime.continuation_.configuration().fields.size());
        for (const auto& binding : runtime.continuation_.configuration().fields)
          projections.emplace_back(&state(binding.program_block),
                                   &continuation_at<Dim>(out.value, binding.path)->field());
        return out;
      },
      "continuation root publication preparation");
  // commit_many stages all field copies and converges every fallible operation first. The
  // following image swap cannot fail; the enclosing StepTransaction still owns rollback.
  continuation_image_consensus_(prepared);
  commit_many(std::span<const std::pair<field_type*, const field_type*>>(projections));
  runtime_state().continuation_.publish(std::move(prepared));
  algebra_lifetime_->continuation.revoke();
}

private:
continuation_ref continuation_accepted_origins_(continuation_ref value) const {
  for (const auto& binding : runtime_state().continuation_.configuration().fields) {
    auto leaf = value;
    std::vector<continuation_ref> parents;
    parents.reserve(binding.path.size());
    for (auto index : binding.path) {
      if (leaf->kind() != ContinuationKind::Product || index >= leaf->members().size())
        throw std::invalid_argument("accepted field projection changes its installed path");
      parents.push_back(leaf);
      leaf = leaf->members()[index];
    }
    if (!leaf->field_exchange_expression())
      continue;
    auto replacement = ContinuationValue<Dim>::with_field_exchange_expression(leaf, {});
    for (std::size_t depth = binding.path.size(); depth-- > 0;) {
      auto children = parents[depth]->members();
      children[binding.path[depth]] = std::move(replacement);
      replacement = ContinuationValue<Dim>::product(std::move(children));
    }
    value = std::move(replacement);
  }
  std::vector<FieldExchangeExpressions::Ref> roots;
  std::set<const ContinuationValue<Dim>*> seen;
  continuation_detail::walk_value<Dim>(
      value, continuation_byte_capacity(),
      [&](const auto& node) {
        if (!seen.insert(&node).second)
          return false;
        if (node.kind() == ContinuationKind::Field)
          roots.push_back(node.field_exchange_expression());
        return true;
      },
      [](const auto&) {});
  if (field_exchanges_->limits().max_bases)
    (void)FieldExchangeExpressions::encode_roots(roots, field_exchanges_->limits());
  return value;
}

void continuation_synchronize_realm_(const FiniteDomainRealmRef& realm) const {
  const auto local = continuation_local_phase(
      [&] {
        if (!realm || !realm->visible())
          throw std::logic_error("finite realm is unavailable");
        return realm->unexported_identity();
      },
      "finite-domain transport origin");
  const auto& lane = prepared_execution_lane();
  std::array<std::uint64_t, 2> common{};
  // Portable exact words: each reduced summand is at most 2^21-1 and only
  // rank zero contributes. No floating-point conversion or world fallback.
  constexpr std::uint64_t mask = (UINT64_C(1) << 21) - 1;
  for (std::size_t n = 0; n < common.size(); ++n)
    for (unsigned shift : {0U, 21U, 42U, 63U})
      common[n] |=
          std::uint64_t(all_reduce_sum(
              static_cast<long>(lane.rank() == 0 ? ((local[n] >> shift) & mask) : 0), lane))
          << shift;
  continuation_local_phase([&] { algebra_lifetime_->finite.synchronize_identity(realm, common); },
                           "finite-domain common transport origin");
}

void continuation_image_consensus_(
    const typename ContinuationStore<Dim>::Prepared& prepared) const {
  auto identity = continuation_local_phase(
      [&] {
        return runtime_state()
            .continuation_.codec(continuation_prototypes_())
            .collective_identity(prepared.value, prepared.publication);
      },
      "continuation replicated image identity");
  if (!all_ranks_agree_exact_ordered_byte_pairs({{"continuation", identity}},
                                                prepared_execution_lane()))
    throw std::runtime_error("continuation replicated values or ownership differ across ranks");
}
void continuation_word_agrees_(std::uint64_t value, const char* label) const {
  const auto& lane = prepared_execution_lane();
  for (unsigned shift : {0u, 21u, 42u, 63u}) {
    const auto word = static_cast<long>((value >> shift) & ((UINT64_C(1) << 21) - 1));
    if (all_reduce_min(word, lane) != all_reduce_max(word, lane))
      throw std::runtime_error(label);
  }
}
void continuation_rethrow_(const std::exception_ptr& error, const char* label) const {
  const auto& lane = prepared_execution_lane();
  if (all_reduce_max(error ? 1L : 0L, lane) == 0)
    return;
  if (lane.size() == 1 && error)
    std::rethrow_exception(error);
  throw std::runtime_error(std::string(label) + " failed collectively");
}
typename ContinuationCodec<Dim>::Prototypes continuation_prototypes_() const {
  return [this](int owner) -> const field_type& { return state(owner); };
}
void continuation_require_profile_() const {
  const auto& runtime = runtime_state();
  if (!runtime.hist_.histories.empty() || runtime.cache_.size() != 0 ||
      !runtime.moving_interval_geometry_.empty() ||
      !system_->configured_field_provider_slots().empty() ||
      !runtime.accepted_exchanges_.same_integral_declarations(AcceptedExchangeLedger{}))
    throw std::logic_error(
        "Continuation profile does not permit retained legacy history/cache/field/integral/moving "
        "storage");
}
void continuation_require_frontier_(const continuation_ref& value, double time,
                                    std::int64_t step) const {
  const auto& configuration = runtime_state().continuation_.configuration();
  if (std::bit_cast<std::uint64_t>(continuation_at<Dim>(value, configuration.time_path)->real()) !=
          std::bit_cast<std::uint64_t>(time) ||
      continuation_at<Dim>(value, configuration.step_path)->integer() != step)
    throw std::logic_error("Continuation image does not reach its authenticated native frontier");
}
void continuation_require_projection_(const continuation_ref& value) const {
  for (const auto& binding : runtime_state().continuation_.configuration().fields) {
    const auto leaf = continuation_at<Dim>(value, binding.path);
    if (leaf->field_exchange_expression())
      throw std::invalid_argument(
          "accepted State projection cannot replay its prior delivered face expression");
    require_continuation_projection<Dim>(leaf->field(), state(binding.program_block));
  }
}

public:
