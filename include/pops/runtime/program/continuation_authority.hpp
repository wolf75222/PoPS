#pragma once

#include <pops/runtime/program/continuation_image.hpp>
#include <pops/runtime/program/continuation_closure.hpp>
#include <pops/runtime/program/prepared_resource_lifetime.hpp>

#include <array>
#include <cmath>
#include <exception>
#include <set>
#include <unordered_set>

namespace pops::runtime::program {

using ContinuationPath = std::vector<std::size_t>;

template <int Dim>
ContinuationValueRef<Dim> continuation_at(ContinuationValueRef<Dim> root,
                                          const ContinuationPath& path) {
  for (auto item : path) {
    if (!root || root->kind() != ContinuationKind::Product || item >= root->members().size())
      throw std::invalid_argument("continuation path is not an installed Product projection");
    root = root->members()[item];
  }
  if (!root)
    throw std::invalid_argument("continuation path projects an absent value");
  return root;
}

// A carrier is a materialized projection: its valid cells must agree bit for bit with
// the authoritative image. Halo cells are derived storage and may be filled between reads.
template <int Dim>
void require_continuation_projection(const MultiFab<Dim>& image, const MultiFab<Dim>& carrier) {
  if (image.layout() != carrier.layout() || image.distribution() != carrier.distribution() ||
      image.local_rank() != carrier.local_rank() || image.ncomp() != carrier.ncomp() ||
      image.ghosts() != carrier.ghosts())
    throw std::logic_error("continuation carrier layout/ownership differs from image");
  using Bits = std::conditional_t<sizeof(Real) == 8, std::uint64_t, std::uint32_t>;
  for (std::size_t p = 0; p < image.local_size(); ++p) {
    const auto& left = image.fab(p);
    const auto& right = carrier.fab(p);
    auto a = left.create_host_mirror();
    auto b = right.create_host_mirror();
    left.copy_to_host(a);
    right.copy_to_host(b);
    const auto view = left.view();
    const auto extent = left.box().extent();
    for (std::int64_t cell = 0; cell < left.box().numPts(); ++cell) {
      auto remainder = cell;
      std::size_t offset = 0;
      for (int axis = 0; axis < Dim; ++axis) {
        const auto coordinate = left.box().lo[axis] + remainder % extent[axis];
        remainder /= extent[axis];
        offset += static_cast<std::size_t>(coordinate - view.origin[axis]) * view.strides[axis];
      }
      for (int component = 0; component < image.ncomp(); ++component) {
        const auto at = offset + component * view.component_stride;
        if (std::bit_cast<Bits>(a(at)) != std::bit_cast<Bits>(b(at)))
          throw std::logic_error(
              "continuation carrier was modified outside its authoritative image");
      }
    }
  }
}

struct ContinuationFieldBinding {
  ContinuationPath path;
  int program_block = -1;
  std::string clock_identity;
  // Exact frontend witnesses for the two legal views, never a cast dropping a point.
  std::string read_view;
  std::string write_view;
  bool operator==(const ContinuationFieldBinding&) const = default;
};

struct ContinuationConfiguration {
  ContinuationType type;
  std::string identity;
  std::uint64_t byte_capacity = 0;
  ContinuationPath time_path;
  ContinuationPath step_path;
  std::vector<ContinuationFieldBinding> fields;
  std::uint64_t distributed_storage_bytes = 0;
  std::uint64_t distributed_storage_versions = 0;
  bool operator==(const ContinuationConfiguration&) const = default;

  const ContinuationType& at(const ContinuationPath& path) const {
    const auto* out = &type;
    for (auto item : path) {
      if (out->kind != ContinuationKind::Product || item >= out->children.size())
        throw std::invalid_argument("continuation binding is not a static Product path");
      out = &out->children[item];
    }
    return *out;
  }
  void validate() const {
    type.validate(byte_capacity);
    if (bool(distributed_storage_bytes) != bool(distributed_storage_versions))
      throw std::invalid_argument("Continuation has an incomplete distributed storage policy");
    if (identity.empty() || byte_capacity < 32 || type.kind != ContinuationKind::Product ||
        at(time_path).kind != ContinuationKind::Float64 ||
        at(step_path).kind != ContinuationKind::Int64)
      throw std::invalid_argument(
          "continuation configuration requires Product schema and exact time/step paths");
    std::set<int> blocks;
    std::set<ContinuationPath> paths;
    for (const auto& binding : fields) {
      const auto& leaf = at(binding.path);
      if (binding.program_block < 0 || binding.clock_identity.empty() ||
          binding.read_view.empty() || binding.write_view.empty() ||
          leaf.kind != ContinuationKind::Field || leaf.field_owner != binding.program_block ||
          !blocks.insert(binding.program_block).second || !paths.insert(binding.path).second)
        throw std::invalid_argument(
            "continuation field binding is incomplete, duplicated or changes owner");
    }
  }
};

/// One schema-derived persistent image in ProgramRuntimeState. Copies retain immutable bytes/
/// buffers; no live scope, attempt, generated closure or mutable borrowed scratch is copied.
template <int Dim>
class ContinuationStore {
 public:
  using Ref = ContinuationValueRef<Dim>;
  using Prototypes = typename ContinuationCodec<Dim>::Prototypes;
  struct Prepared {
    std::shared_ptr<const ContinuationConfiguration> configuration;
    Ref value;
    std::uint64_t publication = 0;
    FiniteDomainRealmRef realm;
    std::int64_t domain_high_water = 0;
    std::vector<FiniteDomainRealmRef> realms;
    std::shared_ptr<distributed::Registry> distributed_registry;
    std::shared_ptr<const ContinuationClosure<Dim>> closure;
  };
  bool configured() const noexcept { return bool(configuration_); }
  bool initialized() const noexcept { return bool(value_); }
  const ContinuationConfiguration& configuration() const {
    if (!configuration_)
      throw std::logic_error("continuation has no installed schema");
    return *configuration_;
  }
  const Ref& read() const {
    if (!value_)
      throw std::logic_error("continuation has not been initialized");
    return value_;
  }
  std::uint64_t publication() const noexcept { return publication_; }
  const FiniteDomainRealmRef& domain_realm() const noexcept { return realm_; }
  const std::vector<FiniteDomainRealmRef>& domain_realms() const noexcept { return realms_; }
  const std::shared_ptr<distributed::Registry>& distributed_registry() const noexcept {
    return distributed_registry_;
  }
  const std::shared_ptr<const ContinuationClosure<Dim>>& closure() const noexcept {
    return closure_;
  }
  std::int64_t domain_high_water() const noexcept { return domain_high_water_; }
  void configure(ContinuationConfiguration configuration) {
    configuration.validate();
    if (configuration_ && *configuration_ != configuration)
      throw std::logic_error("continuation schema cannot change after artifact installation");
    if (!configuration_)
      configuration_ = std::make_shared<const ContinuationConfiguration>(std::move(configuration));
  }
  ContinuationCodec<Dim> codec(
      Prototypes prototypes, const ExecutionLane* lane = nullptr,
      const distributed::Registry* registry = nullptr,
      const ContinuationClosure<Dim>* closure = nullptr,

      typename ContinuationCodec<Dim>::DecodeContext* decode_context = nullptr,
      typename ContinuationCodec<Dim>::ArchiveFieldResolver archive_fields = {}) const {
    const auto& c = configuration();
    return ContinuationCodec<Dim>(c.type, c.identity, c.byte_capacity, std::move(prototypes), lane,
                                  registry ? registry : distributed_registry_.get(),

                                  closure ? closure : closure_.get(), decode_context,
                                  std::move(archive_fields));
  }
  Prepared prepare(Ref next, Prototypes prototypes, bool initialize = false,
                   std::shared_ptr<distributed::Registry> registry = {},
                   std::shared_ptr<const ContinuationClosure<Dim>> closure = {}) const {
    if (initialize != !initialized())
      throw std::logic_error(
          "continuation initialize/publish disagrees with accepted storage state");
    if (!initialize && publication_ == std::numeric_limits<std::uint64_t>::max())
      throw std::overflow_error("continuation publication epoch exhausted");
    const auto publication = initialize ? 0 : publication_ + 1;
    // Encoding validates recursive types, aliases, capacity and local field ownership before
    // publication. These bytes are deliberately not a second accepted authority.
    if (!registry)
      registry = distributed_registry_;
    if (registry && (registry->storage_bytes() != configuration().distributed_storage_bytes ||
                     registry->storage_versions() != configuration().distributed_storage_versions))
      throw std::invalid_argument(
          "Continuation distributed vault differs from installed storage policy");
    if (closure)
      for (const auto& authority : closure->registries)
        if (!authority || authority->storage_bytes() != configuration().distributed_storage_bytes ||
            authority->storage_versions() != configuration().distributed_storage_versions)
          throw std::invalid_argument(
              "Continuation distributed issuer differs from installed storage policy");
    (void)codec(std::move(prototypes), nullptr, registry.get(), closure.get())
        .encode(next, publication);
    FiniteDomainRealmRef realm = realm_;
    std::vector<FiniteDomainRealmRef> realms;
    if (closure && closure->replicated_realm) {
      if (realm && realm != closure->replicated_realm)
        throw std::invalid_argument("candidate changes its accepted finite allocator authority");
      realm = closure->replicated_realm;
      realms.push_back(realm);
    }
    std::int64_t high_water = domain_high_water_;
    auto inspect = [&](const Ref& root) {
      continuation_detail::walk_value<Dim>(
          root, configuration().byte_capacity,
          [&](const ContinuationValue<Dim>& node) {
            const auto* value = &node;
            if (const auto& domain = value->domain_ref()) {
              if (domain->realm()->replicated()) {
                if (!realm)
                  realm = domain->realm();
                if (realm == domain->realm())
                  high_water = std::max(high_water, domain->version);
              }
              if (std::find(realms.begin(), realms.end(), domain->realm()) == realms.end())
                realms.push_back(domain->realm());
            }
            return true;
          },
          [](const ContinuationValue<Dim>&) {});
    };
    inspect(next);
    if (closure)
      for (const auto& array : closure->arrays)
        inspect(ContinuationValue<Dim>::distributed_array(array));
    return {configuration_, std::move(next),   publication,         std::move(realm),
            high_water,     std::move(realms), std::move(registry), std::move(closure)};
  }
  Prepared prepare_restart(
      std::span<const std::uint8_t> bytes, Prototypes prototypes,
      const ExecutionLane* lane = nullptr,

      typename ContinuationCodec<Dim>::DecodeContext* decode_context = nullptr,
      typename ContinuationCodec<Dim>::ArchiveFieldResolver archive_fields = {}) const {
    auto decoded = codec(std::move(prototypes), lane, nullptr, nullptr, decode_context,
                         std::move(archive_fields))
                       .decode(bytes);

    if (decoded.distributed_registry && (decoded.distributed_registry->storage_bytes() !=
                                             configuration().distributed_storage_bytes ||
                                         decoded.distributed_registry->storage_versions() !=
                                             configuration().distributed_storage_versions))
      throw std::invalid_argument(
          "checkpoint distributed storage policy differs from installed code");
    if (decoded.closure)
      for (const auto& authority : decoded.closure->registries)
        if (!authority || authority->storage_bytes() != configuration().distributed_storage_bytes ||
            authority->storage_versions() != configuration().distributed_storage_versions)
          throw std::invalid_argument(
              "checkpoint distributed issuer policy differs from installed code");
    return {configuration_,
            std::move(decoded.root),
            decoded.publication,
            std::move(decoded.realm),
            decoded.max_domain_version,
            std::move(decoded.realms),
            std::move(decoded.distributed_registry),
            std::move(decoded.closure)};
  }
  std::vector<std::uint8_t> checkpoint(Prototypes prototypes) const {
    return codec(std::move(prototypes)).encode(read(), publication_);
  }
  void publish(Prepared&& prepared) noexcept {
    if (!prepared.configuration || prepared.configuration != configuration_ || !prepared.value)
      std::terminate();
    value_.swap(prepared.value);
    publication_ = prepared.publication;
    realm_.swap(prepared.realm);
    domain_high_water_ = prepared.domain_high_water;
    realms_.swap(prepared.realms);
    distributed_registry_.swap(prepared.distributed_registry);
    closure_.swap(prepared.closure);
  }
  void swap(ContinuationStore& other) noexcept {
    configuration_.swap(other.configuration_);
    value_.swap(other.value_);
    std::swap(publication_, other.publication_);
    realm_.swap(other.realm_);
    std::swap(domain_high_water_, other.domain_high_water_);
    realms_.swap(other.realms_);
    distributed_registry_.swap(other.distributed_registry_);
    closure_.swap(other.closure_);
  }
  void require_frontier(double time, std::int64_t step) const {
    const auto& c = configuration();
    const auto actual_time = continuation_at<Dim>(read(), c.time_path)->real();
    const auto actual_step = continuation_at<Dim>(read(), c.step_path)->integer();
    if (std::bit_cast<std::uint64_t>(time) != std::bit_cast<std::uint64_t>(actual_time) ||
        step != actual_step)
      throw std::logic_error("continuation image and accepted runtime frontier differ");
  }

 private:
  std::shared_ptr<const ContinuationConfiguration> configuration_;
  Ref value_;
  std::uint64_t publication_ = 0;
  FiniteDomainRealmRef realm_;
  std::int64_t domain_high_water_ = 0;
  std::vector<FiniteDomainRealmRef> realms_;
  std::shared_ptr<distributed::Registry> distributed_registry_;
  std::shared_ptr<const ContinuationClosure<Dim>> closure_;
};

/// Live ownership authority, intentionally outside accepted snapshots. No numeric version or
/// static SSA identifier supplied by a case can construct these capabilities.
template <int Dim>
class ContinuationAuthority {
 public:
  class Base;
  class Candidate;
  class Assessment;
  class Accepted;

 private:
  using Ref = ContinuationValueRef<Dim>;
  struct ScopeState {
    std::shared_ptr<ScopeState> parent;
    std::uint64_t ordinal = 0;
    bool revoked = false;
    std::set<std::uint64_t> consumed_increments;
    bool visible() const noexcept { return !revoked && (!parent || parent->visible()); }
  };
  struct State;
  struct CandidateData;
  struct AssessmentData;
  struct ChildDependency {
    std::shared_ptr<const AssessmentData> assessment;
    std::shared_ptr<ScopeState> accepted_scope;
  };
  struct BaseData {
    std::weak_ptr<State> issuer;
    std::shared_ptr<ScopeState> scope;
    PreparedResourceAttempt attempt;
    Ref value;
    Ref root_value;
    std::uint64_t publication = 0;
    std::shared_ptr<const CandidateData> source_candidate;
  };
  struct CandidateData {
    std::shared_ptr<const BaseData> base;
    std::shared_ptr<ScopeState> scope;
    std::shared_ptr<ScopeState> owner_scope;
    Ref root;
    Ref proposal;
    std::uint64_t ordinal = 0;
    bool superseded = false;
    std::vector<ChildDependency> children;
  };
  struct AssessmentData {
    std::shared_ptr<CandidateData> candidate;
    Ref checks;
    std::string graph_identity;
    bool accepted = false;
  };
  struct State {
    PreparedResourceAttempt attempt;
    std::shared_ptr<ScopeState> root_scope;
    std::shared_ptr<ScopeState> current_scope;
    std::shared_ptr<const BaseData> base;
    std::uint64_t next_ordinal = 0;
    std::int64_t next_domain_version = 0;
  };

 public:
  class Base {
   public:
    Base() = default;
    const Ref& value() const {
      require_();
      return data_->value;
    }
    const Ref& root() const { return value(); }

   private:
    friend class ContinuationAuthority;
    friend class Candidate;
    friend class Accepted;
    std::shared_ptr<const BaseData> data_;
    explicit Base(std::shared_ptr<const BaseData> data) : data_(std::move(data)) {}
    void require_() const {
      if (!data_)
        throw std::logic_error("empty continuation base capability");
      const auto issuer = data_->issuer.lock();
      if (!issuer || !data_->attempt.visible() || !data_->attempt.same_attempt(issuer->attempt) ||
          !data_->scope || !data_->scope->visible())
        throw std::logic_error("continuation base capability has expired");
      if (data_->source_candidate) {
        if (data_->source_candidate->superseded || !data_->source_candidate->scope->visible())
          throw std::logic_error("provisional base lost its exact accepted predecessor");
        Base(data_->source_candidate->base).require_();
      }
    }
  };
  class Candidate {
   public:
    Candidate() = default;
    const Ref& value() const {
      require_();
      return data_->root;
    }
    const Ref& root() const { return value(); }
    const Ref& proposal() const {
      require_();
      return data_->proposal;
    }
    Base base() const {
      require_();
      return Base(data_->base);
    }

   private:
    friend class ContinuationAuthority;
    friend class Accepted;
    friend class Assessment;
    std::shared_ptr<CandidateData> data_;
    explicit Candidate(std::shared_ptr<CandidateData> data) : data_(std::move(data)) {}
    void require_() const {
      std::vector<std::shared_ptr<const CandidateData>> pending{data_};
      std::unordered_set<const CandidateData*> visited;
      while (!pending.empty()) {
        auto candidate = std::move(pending.back());
        pending.pop_back();
        if (!candidate || candidate->superseded || !candidate->scope ||
            !candidate->scope->visible())
          throw std::logic_error("candidate is absent, superseded or revoked");
        if (!visited.insert(candidate.get()).second)
          continue;
        Base(candidate->base).require_();
        for (const auto& child : candidate->children) {
          if (!child.assessment || !child.assessment->accepted || !child.accepted_scope ||
              !child.accepted_scope->visible())
            throw std::logic_error("candidate lost an exact accepted child dependency");
          pending.push_back(child.assessment->candidate);
        }
      }
    }
  };
  class Assessment {
   public:
    Assessment() = default;
    bool accepted() const {
      if (!data_)
        throw std::logic_error("empty assessment capability");
      Candidate(data_->candidate).require_();
      return data_->accepted;
    }
    const Ref& checks() const {
      (void)accepted();
      return data_->checks;
    }

   private:
    friend class ContinuationAuthority;
    friend class Accepted;
    std::shared_ptr<const AssessmentData> data_;
    explicit Assessment(std::shared_ptr<const AssessmentData> data) : data_(std::move(data)) {}
  };
  class Accepted {
   public:
    Accepted() = default;
    const Ref& value() const {
      require_();
      return candidate_.data_->root;
    }
    const Candidate& candidate() const {
      require_();
      return candidate_;
    }

   private:
    friend class ContinuationAuthority;
    Candidate candidate_;
    Assessment assessment_;
    std::shared_ptr<ScopeState> scope_;
    Accepted(Candidate candidate, Assessment assessment, std::shared_ptr<ScopeState> scope)
        : candidate_(std::move(candidate)),
          assessment_(std::move(assessment)),
          scope_(std::move(scope)) {}
    void require_() const {
      candidate_.require_();
      if (!scope_ || !scope_->visible() || !assessment_.data_ || !assessment_.data_->accepted ||
          assessment_.data_->candidate != candidate_.data_)
        throw std::logic_error("accepted provisional lacks assessment on its exact candidate");
    }
  };
  class Scope {
   public:
    Scope() = default;
    Scope(const Scope&) = delete;
    Scope& operator=(const Scope&) = delete;
    Scope(Scope&& other) noexcept
        : issuer_(std::move(other.issuer_)), scope_(std::exchange(other.scope_, {})) {}
    Scope& operator=(Scope&& other) noexcept {
      if (this != &other) {
        revoke_();
        issuer_ = std::move(other.issuer_);
        scope_ = std::exchange(other.scope_, {});
      }
      return *this;
    }
    ~Scope() noexcept { revoke_(); }

   private:
    friend class ContinuationAuthority;
    std::weak_ptr<State> issuer_;
    std::shared_ptr<ScopeState> scope_;
    Scope(std::weak_ptr<State> issuer, std::shared_ptr<ScopeState> scope)
        : issuer_(std::move(issuer)), scope_(std::move(scope)) {}
    void revoke_() noexcept {
      if (!scope_)
        return;
      scope_->revoked = true;
      if (auto issuer = issuer_.lock(); issuer && issuer->current_scope == scope_)
        issuer->current_scope = scope_->parent;
      scope_.reset();
    }
  };

  ContinuationAuthority() : state_(std::make_shared<State>()) {}
  ContinuationAuthority(const ContinuationAuthority&) = delete;
  ContinuationAuthority& operator=(const ContinuationAuthority&) = delete;
  std::array<std::uint64_t, 3> logical_state() const noexcept {
    return {state_->next_ordinal, state_->root_scope ? state_->root_scope->ordinal : 0,
            state_->current_scope ? state_->current_scope->ordinal : 0};
  }
  void revoke() noexcept {
    if (state_->root_scope)
      state_->root_scope->revoked = true;
    state_->base.reset();
    state_->current_scope.reset();
    state_->root_scope.reset();
  }
  void begin(PreparedResourceAttempt attempt) {
    revoke();
    if (!attempt.visible())
      throw std::logic_error("continuation requires an active native attempt");
    const auto ordinal = fresh_();
    auto root = std::make_shared<ScopeState>();
    root->ordinal = ordinal;
    state_->attempt = std::move(attempt);
    state_->root_scope = root;
    state_->current_scope = std::move(root);
  }
  std::int64_t next_domain_version() {
    if (state_->next_domain_version == std::numeric_limits<std::int64_t>::max())
      throw std::overflow_error("native finite-domain version exhausted");
    return ++state_->next_domain_version;
  }
  void import_domain_high_water(std::int64_t accepted) {
    if (accepted < 0)
      throw std::invalid_argument("invalid restored finite-domain version");
    state_->next_domain_version = std::max(state_->next_domain_version, accepted);
  }
  Base read(const ContinuationStore<Dim>& store) {
    require_scope_();
    if (!state_->base) {
      auto base = std::make_shared<BaseData>();
      base->issuer = state_;
      base->scope = state_->root_scope;
      base->attempt = state_->attempt;
      base->value = store.read();
      base->root_value = store.read();
      base->publication = store.publication();
      state_->base = std::move(base);
    }
    if (state_->base->root_value != store.read() ||
        state_->base->publication != store.publication())
      throw std::logic_error("accepted continuation changed during an active pure attempt");
    return Base(state_->base);
  }
  Scope begin_scope() {
    require_scope_();
    const auto ordinal = fresh_();
    auto child = std::make_shared<ScopeState>();
    child->parent = state_->current_scope;
    child->ordinal = ordinal;
    state_->current_scope = child;
    return Scope(state_, std::move(child));
  }
  Candidate freeze(const Base& base, Ref proposal, Ref candidate_root,
                   std::span<const Accepted> children = {}) {
    require_base_(base);
    require_scope_();
    if (!proposal || !candidate_root || candidate_root->kind() != ContinuationKind::Product ||
        candidate_root->members().size() != 4)
      throw std::invalid_argument(
          "Candidate requires complete (continuation,increment,observations,checks) Product");
    const auto ordinal = fresh_();
    auto data = std::make_shared<CandidateData>();
    data->base = base.data_;
    data->scope = state_->current_scope;
    data->owner_scope = state_->current_scope;
    data->root = std::move(candidate_root);
    data->proposal = std::move(proposal);
    data->ordinal = ordinal;
    std::unordered_set<const CandidateData*> unique_children;
    data->children.reserve(children.size());
    for (const auto& child : children) {
      child.require_();
      if (!unique_children.insert(child.candidate_.data_.get()).second)
        throw std::invalid_argument("candidate lists the same accepted child twice");
      data->children.push_back({child.assessment_.data_, child.scope_});
    }
    return Candidate(std::move(data));
  }
  Assessment attest(const Candidate& candidate, Ref checks, bool accepted,
                    std::string graph_identity) {
    require_candidate_(candidate);
    require_scope_();
    if (!checks || graph_identity.empty())
      throw std::invalid_argument("Assessment requires executed checks and graph identity");
    if (candidate.data_->owner_scope != state_->current_scope)
      throw std::logic_error("Assessment must execute in its candidate scope");
    return Assessment(std::make_shared<const AssessmentData>(
        AssessmentData{candidate.data_, std::move(checks), std::move(graph_identity), accepted}));
  }
  Accepted accept(const Candidate& candidate, const Assessment& assessment) {
    require_candidate_(candidate);
    require_scope_();
    if (candidate.data_->owner_scope != state_->current_scope)
      throw std::logic_error("accept requires assessment in its current candidate scope");
    Accepted out(candidate, assessment, state_->current_scope);
    out.require_();
    return out;
  }
  void supersede(const Candidate& candidate) {
    require_candidate_(candidate);
    require_scope_();
    if (candidate.data_->owner_scope != state_->current_scope)
      throw std::logic_error("child cannot supersede a borrowed parent candidate");
    candidate.data_->superseded = true;
  }
  Accepted promote(Scope&& child, const Accepted& accepted) {
    require_scope_();
    accepted.require_();
    if (child.issuer_.lock() != state_ || !child.scope_ || child.scope_ != state_->current_scope ||
        accepted.scope_ != child.scope_ || !child.scope_->parent)
      throw std::logic_error("child promotion requires its exact current accepted scope");
    const auto parent = child.scope_->parent;
    // The immutable candidate/assessment identity does not change. The child remains linked
    // to its parent so a later parent rejection still revokes every descendant.
    auto out = Accepted(accepted.candidate_, accepted.assessment_, parent);
    accepted.candidate_.data_->owner_scope = parent;
    state_->current_scope = parent;
    child.scope_.reset();
    return out;
  }
  Base provisional_base(const Accepted& accepted) {
    accepted.require_();
    require_candidate_(accepted.candidate_);
    require_scope_();
    if (accepted.scope_ != state_->current_scope)
      throw std::logic_error("provisional base is outside the promoted child scope");
    auto data = std::make_shared<BaseData>(*accepted.candidate_.data_->base);
    data->scope = state_->current_scope;
    data->value = accepted.value()->members()[0];
    data->source_candidate = accepted.candidate_.data_;
    return Base(std::move(data));
  }
  Ref consume_increment(const Accepted& child) {
    child.require_();
    require_candidate_(child.candidate_);
    require_scope_();
    if (child.scope_ != state_->current_scope ||
        !state_->current_scope->consumed_increments.insert(child.candidate_.data_->ordinal).second)
      throw std::logic_error("child increment is foreign or already consumed in this parent scope");
    return child.value()->members()[1];
  }
  std::array<std::uint64_t, 5> publication_state(const Accepted& accepted) const {
    accepted.require_();
    require_candidate_(accepted.candidate_);
    require_scope_();
    const auto& candidate = *accepted.candidate_.data_;
    return {candidate.ordinal, candidate.base->publication, candidate.scope->ordinal,
            candidate.owner_scope->ordinal,
            candidate.base->source_candidate ? candidate.base->source_candidate->ordinal : 0};
  }
  void require_exact_children(const Accepted& accepted, std::span<const Accepted> expected) const {
    accepted.require_();
    const auto& children = accepted.candidate_.data_->children;
    if (children.size() != expected.size())
      throw std::logic_error("parent acceptance does not bind every exact child");
    for (std::size_t i = 0; i < expected.size(); ++i) {
      expected[i].require_();
      if (children[i].assessment != expected[i].assessment_.data_ ||
          children[i].accepted_scope != expected[i].scope_)
        throw std::logic_error("parent acceptance substitutes an accepted child dependency");
    }
  }

  Ref require_root_publication(const Accepted& accepted,
                               const ContinuationStore<Dim>& store) const {
    accepted.require_();
    require_candidate_(accepted.candidate_);
    require_scope_();
    if (state_->current_scope != state_->root_scope || accepted.scope_ != state_->root_scope ||
        accepted.candidate_.data_->base->root_value != store.read() ||
        accepted.candidate_.data_->base->publication != store.publication())
      throw std::logic_error(
          "root publication has stale base, foreign scope or changed generation");
    return accepted.value()->members()[0];
  }

 private:
  std::shared_ptr<State> state_;
  std::uint64_t fresh_() {
    if (state_->next_ordinal == std::numeric_limits<std::uint64_t>::max())
      throw std::overflow_error("continuation dynamic capability ordinal exhausted");
    return ++state_->next_ordinal;
  }
  void require_scope_() const {
    if (!state_->attempt.visible() || !state_->current_scope || !state_->current_scope->visible())
      throw std::logic_error("continuation operation has no active native scope");
  }
  void require_base_(const Base& base) const {
    base.require_();
    if (base.data_->issuer.lock() != state_)
      throw std::logic_error("continuation base belongs to another native context");
  }
  void require_candidate_(const Candidate& candidate) const {
    candidate.require_();
    require_base_(Base(candidate.data_->base));
  }
};

}  // namespace pops::runtime::program
