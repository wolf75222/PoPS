// Included in AmrProgramContext<Dim>. The caller holds one scope per invocation.
public:
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

class FiniteValueScope;
class FiniteValueLease {
  friend class AmrProgramContext;
  friend class FiniteValueScope;
  const AmrProgramContext* owner_;
  std::shared_ptr<AmrFiniteValueLifetime<Dim>> lifetime_;
  PreparedResourceAttempt attempt_;
  std::uint64_t epoch_, generation_;
  int level_, substep_;
  std::int64_t tick_;
  std::string clock_;
  double start_, duration_;
  ::pops::amr::Rational begin_, end_;
  std::shared_ptr<FiniteValueLease> anchor_;
  std::uint64_t invocation_ = 0;
  bool parent_phase_ = false;
  bool owns_ = false;
  explicit FiniteValueLease(const AmrProgramContext& owner,
                            std::shared_ptr<AmrFiniteValueLifetime<Dim>> lifetime)
      : owner_(&owner),
        lifetime_(std::move(lifetime)),
        attempt_(owner.resource_attempt()),
        epoch_(owner.runtime_->topology_epoch()),
        generation_(owner.runtime_->materialization_generation()),
        level_(owner.active_level_),
        substep_(owner.logical_substep_),
        tick_(static_cast<std::int64_t>(owner.facade_->macro_step())),
        clock_(owner.primary_clock_),
        start_(owner.current_interval_start_time_),
        duration_(owner.current_dt_),
        begin_(owner.current_interval_begin_phase_),
        end_(owner.current_interval_end_phase_),
        parent_phase_(owner.continuation_parent_phase()) {}
  void require(const AmrProgramContext& owner, bool exact_resume_window = true) const {
    if ((!owns_ && !anchor_) || owner_ != &owner ||
        parent_phase_ != owner.continuation_parent_phase() || !lifetime_->active ||
        invocation_ != lifetime_->invocation || !attempt_.visible() ||
        !attempt_.same_attempt(owner.resource_attempt()) ||
        epoch_ != owner.runtime_->topology_epoch() ||
        generation_ != owner.runtime_->materialization_generation() ||
        level_ != owner.active_level_ ||
        (exact_resume_window &&
         (substep_ != owner.logical_substep_ ||
          tick_ != static_cast<std::int64_t>(owner.facade_->macro_step()) ||
          clock_ != owner.primary_clock_ ||
          std::bit_cast<std::uint64_t>(start_) !=
              std::bit_cast<std::uint64_t>(owner.current_interval_start_time_) ||
          std::bit_cast<std::uint64_t>(duration_) !=
              std::bit_cast<std::uint64_t>(owner.current_dt_) ||
          begin_ != owner.current_interval_begin_phase_ ||
          end_ != owner.current_interval_end_phase_)))
      throw std::logic_error("AMR finite-value lease lost its invocation authority");
  }

 public:
  ~FiniteValueLease() noexcept {
    if (owns_ && invocation_ == lifetime_->invocation)
      lifetime_->finish();
  }
};
using finite_value_lease = std::shared_ptr<FiniteValueLease>;
class FiniteValueScope {
  friend class AmrProgramContext;

 public:
  FiniteValueScope(const FiniteValueScope&) = delete;
  FiniteValueScope& operator=(const FiniteValueScope&) = delete;
  ~FiniteValueScope() noexcept { owner_->active_finite_invocation_ = std::move(prior_); }
  finite_value_lease lease() const { return owner_->current_finite_value_lease(); }

 private:
  FiniteValueScope(const AmrProgramContext& owner, finite_value_lease lease)
      : owner_(&owner), lease_(std::move(lease)), prior_(owner.active_finite_invocation_) {
    owner.continuation_local_phase([&] { lease_->require(owner); },
                                   "AMR finite-value invocation activation");
    owner.active_finite_invocation_ = lease_;
  }
  const AmrProgramContext* owner_;
  finite_value_lease lease_, prior_;
};
FiniteValueScope finite_value_scope() const {
  auto lease = continuation_local_phase(
      [&] {
        auto& slot = finite_value_lifetimes_[continuation_parent_phase() ? -1 : active_level_];
        if (!slot)
          slot = std::make_shared<AmrFiniteValueLifetime<Dim>>();
        return finite_value_lease(new FiniteValueLease(*this, slot));
      },
      "AMR finite-value lifetime allocation");
  continuation_local_phase(
      [&] {
        lease->lifetime_->begin();
        lease->invocation_ = lease->lifetime_->invocation;
        lease->owns_ = true;
      },
      "AMR finite-value invocation entry");
  return FiniteValueScope(*this, std::move(lease));
}
finite_value_lease current_finite_value_lease() const {
  return continuation_local_phase(
      [&] {
        (void)finite_values_();
        auto bookmark =
            finite_value_lease(new FiniteValueLease(*this, active_finite_invocation_->lifetime_));
        bookmark->anchor_ = active_finite_invocation_;
        bookmark->invocation_ = active_finite_invocation_->invocation_;
        return bookmark;
      },
      "AMR finite-value suspension bookmark");
}
FiniteValueScope resume_finite_value_scope(finite_value_lease lease) const {
  continuation_local_phase(
      [&] {
        if (!lease)
          throw std::invalid_argument("AMR finite-value resume requires a lease");
      },
      "AMR finite-value resume");
  return FiniteValueScope(*this, std::move(lease));
}
void revoke_finite_values_on_rollback() const noexcept {
  for (const auto& [level, lifetime] : finite_value_lifetimes_) {
    (void)level;
    lifetime->finish();
  }
  active_finite_invocation_.reset();
}
ContinuationTransportSession<Dim>& continuation_transport_session() const {
  return finite_values_().transport;
}
[[nodiscard]] distributed::Registry& distributed_registry() const {
  auto& live = finite_values_();
  const auto& lane = prepared_execution_lane();
  const long present = live.distributed ? 1L : 0L;
  if (all_reduce_min(present, lane) != all_reduce_max(present, lane))
    throw std::logic_error("AMR distributed issuer lifecycle differs across ranks");
  if (!live.distributed)
    live.distributed = std::make_shared<distributed::Registry>(lane);
  return *live.distributed;
}
[[nodiscard]] distributed::Registry* distributed_registry_if_present() const noexcept {
  return active_finite_invocation_ ? active_finite_invocation_->lifetime_->distributed.get()
                                   : nullptr;
}
std::int64_t next_finite_domain_version() const {
  // Burned counters may differ after a failed local allocation. Synchronize upward before
  // issuance; never rewind a surviving rank or convert Int64 to floating point.
  const auto local = static_cast<std::uint64_t>(continuation_local_phase(
      [&] { return finite_values_().finite.next_version(); }, "finite-domain ordinal issuance"));
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
  finite_values_().finite.import_high_water(shared);
  return shared;
}

FiniteDomainRef continuation_new_domain(std::int64_t count, std::int64_t capacity,
                                        std::int64_t family) const {
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
        return finite_values_().finite.issue(count, capacity, family,
                                             finite_values_().finite.realm());
      },
      "finite-domain allocation");
  continuation_word_agrees_(static_cast<std::uint64_t>(domain->version),
                            "finite-domain ordinal differs across ranks");
  continuation_synchronize_realm_(domain->realm());
  continuation_local_phase(
      [&] { finite_values_().transport.capture(ContinuationValue<Dim>::domain(domain)); },
      "finite-domain transport registration");
  return domain;
}

// Only a rank-local algebra region may call this factory. Its issuer is
// deliberately separate from the replicated collective domain allocator.
// Allocation failure is handled by the enclosing local region before its
// next communication phase; there is no hidden rendezvous in this method.
FiniteDomainRef continuation_new_local_domain(std::int64_t count, std::int64_t capacity,
                                              std::int64_t family) const {
  auto domain = finite_values_().local_finite.issue(count, capacity, family);
  finite_values_().transport.capture(ContinuationValue<Dim>::domain(domain));
  return domain;
}

private:
AmrFiniteValueLifetime<Dim>& finite_values_() const {
  if (!active_finite_invocation_)
    throw std::logic_error("AMR finite value requires its active invocation issuer");
  active_finite_invocation_->require(*this, false);
  return *active_finite_invocation_->lifetime_;
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
  continuation_local_phase([&] { finite_values_().finite.synchronize_identity(realm, common); },
                           "finite-domain common transport origin");
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

mutable std::map<int, std::shared_ptr<AmrFiniteValueLifetime<Dim>>> finite_value_lifetimes_;
mutable finite_value_lease active_finite_invocation_;
