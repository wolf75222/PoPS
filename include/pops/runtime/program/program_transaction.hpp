#pragma once

/// @file
/// @brief Allocation-free accepted-state transaction kernel for compiled Programs.
///
/// This header contains the small host-side authority shared by the Uniform and AMR runtime
/// drivers.  It deliberately knows nothing about MPI, Kokkos, fields, or a numerical solver.  A
/// driver registers typed state participants at bind time, reserves their restore images and
/// effect slots, then drives one transaction through the following one-way protocol:
///
///   snapshot -> candidate -> solve/guard/effect-prepare -> hidden-publish ->
///   compensable-effects -> atomic-seal -> irreversible-finalize/fail-stop.
///
/// A transaction never consumes a SolveOutcome.  The integrating runtime owns that boundary and
/// must consume its SolveOutcome before calling `publish()`.  This keeps the solver's one-shot
/// publication contract (and its MPI consensus) authoritative instead of introducing a second
/// outcome type here.
///
/// The hot path has no std::function, associative container, string, or capacity-changing vector
/// operation.  Registration and bind are cold operations and may allocate.  After bind, participant
/// restore images and the prepared-effect vector are fixed; the no-throw registration probe and an
/// effect beyond the frozen budget are refused without vector growth or allocation.

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <concepts>
#include <exception>
#include <limits>
#include <mutex>
#include <shared_mutex>
#include <stdexcept>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>

namespace pops::runtime::program {

inline constexpr std::uint32_t kInvalidProgramTransactionIndex =
    std::numeric_limits<std::uint32_t>::max();

/// Exact lifecycle state.
enum class ProgramTransactionPhase : std::uint8_t {
  kUnbound = 0,
  kSnapshot = 1,
  kCandidate = 2,
  kSolveGuardEffectPrepare = 3,
  kHiddenPublish = 4,
  kCompensableEffects = 5,
  kAtomicSeal = 6,
  kIrreversibleFinalize = 7,
  kAccepted = 8,
  kRolledBack = 9,
  kFailStop = 10,
};

[[nodiscard]] inline constexpr const char* program_transaction_phase_name(
    ProgramTransactionPhase phase) noexcept {
  switch (phase) {
    case ProgramTransactionPhase::kUnbound:
      return "unbound";
    case ProgramTransactionPhase::kSnapshot:
      return "snapshot";
    case ProgramTransactionPhase::kCandidate:
      return "candidate";
    case ProgramTransactionPhase::kSolveGuardEffectPrepare:
      return "solve_guard_effect_prepare";
    case ProgramTransactionPhase::kHiddenPublish:
      return "hidden_publish";
    case ProgramTransactionPhase::kCompensableEffects:
      return "compensable_effects";
    case ProgramTransactionPhase::kAtomicSeal:
      return "atomic_seal";
    case ProgramTransactionPhase::kIrreversibleFinalize:
      return "irreversible_finalize";
    case ProgramTransactionPhase::kAccepted:
      return "accepted";
    case ProgramTransactionPhase::kRolledBack:
      return "rolled_back";
    case ProgramTransactionPhase::kFailStop:
      return "fail_stop";
  }
  return "invalid";
}

/// Cold/hot failure classification.  The ordinal and reason code are fixed-width so a failure can
/// be carried across an MPI adapter without allocating a diagnostic string.
enum class ProgramTransactionFailure : std::uint8_t {
  kNone = 0,
  kNotBound = 1,
  kAlreadyActive = 2,
  kRegistration = 3,
  kBudget = 4,
  kSnapshot = 5,
  kCandidate = 6,
  kSolve = 7,
  kGuard = 8,
  kEffectPrepare = 9,
  kHiddenPublish = 10,
  // Covers a failed compensable-effect publication or its collective agreement.
  kCompensation = 11,
  kAtomicSeal = 12,
  kFinalize = 13,
  kFailStop = 14,
};

/// Fixed-size failure witness.  `phase` identifies where the fault occurred; `failure` identifies
/// the contract that rejected the operation; `ordinal` is a participant/effect index or the
/// invalid-index sentinel; `reason_code` is supplied by the callback/consensus adapter.
struct ProgramTransactionFault final {
  ProgramTransactionPhase phase = ProgramTransactionPhase::kUnbound;
  ProgramTransactionFailure failure = ProgramTransactionFailure::kNone;
  std::uint32_t ordinal = kInvalidProgramTransactionIndex;
  std::uint32_t reason_code = 0;

  [[nodiscard]] constexpr bool failed() const noexcept {
    return failure != ProgramTransactionFailure::kNone;
  }
};

/// Monotone accepted-state generation.  It is intentionally not an integer alias: a read lease or
/// publication receipt must carry the authenticated generation, not an arbitrary counter.
struct AcceptedGeneration final {
  std::uint64_t value = 0;

  [[nodiscard]] constexpr bool valid() const noexcept { return value != 0; }
  [[nodiscard]] constexpr explicit operator std::uint64_t() const noexcept { return value; }
  friend constexpr bool operator==(AcceptedGeneration, AcceptedGeneration) = default;
  friend constexpr bool operator!=(AcceptedGeneration, AcceptedGeneration) = default;
  friend constexpr bool operator<(AcceptedGeneration left, AcceptedGeneration right) noexcept {
    return left.value < right.value;
  }
};

/// Per-participant storage promised before bind.  `restore_bytes` is the exact size passed to the
/// snapshot/restore callbacks.  Candidate bytes are an accounting budget for a runtime-owned
/// candidate carrier; this kernel does not allocate or interpret that carrier.
struct ProgramParticipantBudget final {
  std::size_t restore_bytes = 0;
  std::size_t candidate_bytes = 0;
};

/// Frozen participant metadata exposed for bind-time audit and checkpoint provenance.
struct ProgramParticipantInfo final {
  std::uint32_t order = kInvalidProgramTransactionIndex;
  ProgramParticipantBudget budget{};
};

/// Immutable bind status for one compensable-effect slot. A slot is declared during cold setup
/// and becomes frozen exactly once at `bind()`. Per-step submission/prepared/published bits are
/// candidate state and must never redefine this bind authority.
enum class ProgramEffectSlotStatus : std::uint8_t {
  kUnregistered = 0,
  kDeclared = 1,
  kFrozen = 2,
};

/// Frozen compensable-effect identity exposed for bind-time audit and collective provenance.
struct ProgramEffectInfo final {
  std::uint64_t identity = 0;
  std::uint32_t order = kInvalidProgramTransactionIndex;
  ProgramEffectSlotStatus status = ProgramEffectSlotStatus::kUnregistered;
};

/// Aggregate budget frozen by `bind()`.  A zero maximum means "the exact count/total declared at
/// bind", not unlimited growth.  This makes an accidental late registration impossible in either
/// interpretation of a zero-valued generated manifest.
struct ProgramTransactionBudget final {
  std::size_t max_participants = 0;
  std::size_t max_restore_bytes = 0;
  std::size_t max_candidate_bytes = 0;
  std::size_t max_effects = 0;
};

/// MPI-neutral collective agreement callback.  The callback receives the phase and a local status
/// code and returns true only when the communicator has agreed on that status.  A null callback is
/// the serial identity.  No communicator, MPI header, or reduction implementation is retained.
struct ProgramTransactionConsensus final {
  using Function = bool (*)(void* context, std::uint32_t phase, std::uint32_t status) noexcept;

  Function function = nullptr;
  void* context = nullptr;

  [[nodiscard]] bool agree(ProgramTransactionPhase phase, std::uint32_t status) const noexcept {
    return function == nullptr || function(context, static_cast<std::uint32_t>(phase), status);
  }
};

/// Erased participant lifecycle.  The registry's typed handle (see `ParticipantHandle<T>`) keeps
/// the authoring side type-safe; these five pointers are the only calls retained in the hot path.
/// `publish` returns false before mutating the accepted image.  Rollback is allowed to restore a
/// participant from the pre-step image and must never throw.
struct ProgramParticipantOps final {
  using Snapshot = bool (*)(void* object, void* image, std::size_t image_bytes) noexcept;
  using Restore = void (*)(void* object, const void* image, std::size_t image_bytes) noexcept;
  using Publish = bool (*)(void* object) noexcept;
  using Rollback = void (*)(void* object, const void* image, std::size_t image_bytes) noexcept;
  /// Optional detached candidate carrier.  When supplied, `ProvisionalView<T>` points at this
  /// carrier, keeping the accepted object readable until hidden publication.  A null pointer is
  /// permitted for runtimes whose participant object already owns its candidate storage.
  using Candidate = void* (*)(void* object) noexcept;

  Snapshot snapshot = nullptr;
  Restore restore = nullptr;
  Publish publish = nullptr;
  Rollback rollback = nullptr;
  Candidate candidate = nullptr;
};

/// A typed, non-owning participant token.  It is cheap to copy and cannot be forged for another
/// T without the registry rejecting its type token.  The pointed-to object and callbacks must outlive
/// the registry.
template <class T>
struct ParticipantHandle final {
  std::uint32_t index = kInvalidProgramTransactionIndex;
  const void* type_token = nullptr;

  [[nodiscard]] constexpr bool valid() const noexcept {
    return index != kInvalidProgramTransactionIndex && type_token != nullptr;
  }
  [[nodiscard]] constexpr explicit operator bool() const noexcept { return valid(); }
};

/// Bind-time identity for one compensable-effect slot.  Effect order and identity are frozen with
/// the participant registry; a step may only populate these slots in their declared order.
struct EffectHandle final {
  std::uint32_t index = kInvalidProgramTransactionIndex;
  std::uint64_t identity = 0;

  [[nodiscard]] constexpr bool valid() const noexcept {
    return index != kInvalidProgramTransactionIndex && identity != 0;
  }
  [[nodiscard]] constexpr explicit operator bool() const noexcept { return valid(); }
};

class ProgramTransactionRegistry;
class ProgramTransaction;
class ProvisionalReadLease;

namespace detail {

template <class T>
[[nodiscard]] inline const void* participant_type_token() noexcept {
  static const unsigned char token = 0;
  return &token;
}

template <class T>
concept MethodParticipant =
    requires(T& object, void* image, const void* const_image, std::size_t bytes) {
      { object.snapshot(image, bytes) } noexcept -> std::same_as<bool>;
      { object.restore(const_image, bytes) } noexcept;
      { object.publish() } noexcept -> std::same_as<bool>;
    };

template <class T>
concept MethodParticipantCandidate = requires(T& object) {
  { object.candidate() } noexcept -> std::same_as<T*>;
};

template <class T>
  requires MethodParticipant<T>
[[nodiscard]] inline ProgramParticipantOps method_participant_ops() noexcept {
  ProgramParticipantOps ops{
      [](void* object, void* image, std::size_t bytes) noexcept -> bool {
        return static_cast<T*>(object)->snapshot(image, bytes);
      },
      [](void* object, const void* image, std::size_t bytes) noexcept {
        static_cast<T*>(object)->restore(image, bytes);
      },
      [](void* object) noexcept -> bool { return static_cast<T*>(object)->publish(); },
      [](void* object, const void* image, std::size_t bytes) noexcept {
        static_cast<T*>(object)->restore(image, bytes);
      },
  };
  if constexpr (MethodParticipantCandidate<T>)
    ops.candidate = [](void* object) noexcept -> void* {
      return static_cast<void*>(static_cast<T*>(object)->candidate());
    };
  return ops;
}

}  // namespace detail

/// A view into one registered candidate.  It carries the transaction's base generation so a
/// runtime can authenticate that a field/clock read belongs to the same accepted boundary.  The
/// view is deliberately non-owning and move-only; it must not escape the transaction callback.
template <class T>
class ProvisionalView final {
 public:
  ProvisionalView() noexcept = default;
  ProvisionalView(const ProvisionalView&) = delete;
  ProvisionalView& operator=(const ProvisionalView&) = delete;
  ProvisionalView(ProvisionalView&& other) noexcept
      : object_(std::exchange(other.object_, nullptr)),
        generation_(other.generation_),
        participant_(std::exchange(other.participant_, kInvalidProgramTransactionIndex)) {}
  ProvisionalView& operator=(ProvisionalView&& other) noexcept {
    if (this != &other) {
      object_ = std::exchange(other.object_, nullptr);
      generation_ = other.generation_;
      participant_ = std::exchange(other.participant_, kInvalidProgramTransactionIndex);
    }
    return *this;
  }

  [[nodiscard]] bool valid() const noexcept { return object_ != nullptr; }
  [[nodiscard]] explicit operator bool() const noexcept { return valid(); }
  [[nodiscard]] T* get() const noexcept { return object_; }
  [[nodiscard]] T& operator*() const noexcept { return *object_; }
  [[nodiscard]] T* operator->() const noexcept { return object_; }
  [[nodiscard]] AcceptedGeneration base_generation() const noexcept { return generation_; }
  [[nodiscard]] std::uint32_t participant_index() const noexcept { return participant_; }

 private:
  friend class ProgramTransaction;

  ProvisionalView(T* object, AcceptedGeneration generation, std::uint32_t participant) noexcept
      : object_(object), generation_(generation), participant_(participant) {}

  T* object_ = nullptr;
  AcceptedGeneration generation_{};
  std::uint32_t participant_ = kInvalidProgramTransactionIndex;
};

/// A read-side critical section.  Readers that acquired this lease before hidden publication keep
/// seeing the accepted object and generation.  Once hidden publication takes the writer lock, new
/// leases block until seal or rollback.  No heap ownership is involved.
class AcceptedReadLease final {
 public:
  AcceptedReadLease() noexcept = default;
  AcceptedReadLease(const AcceptedReadLease&) = delete;
  AcceptedReadLease& operator=(const AcceptedReadLease&) = delete;
  AcceptedReadLease(AcceptedReadLease&& other) noexcept { move_from_(other); }
  AcceptedReadLease& operator=(AcceptedReadLease&& other) noexcept {
    if (this != &other) {
      release_();
      move_from_(other);
    }
    return *this;
  }
  ~AcceptedReadLease() noexcept { release_(); }

  [[nodiscard]] bool valid() const noexcept;
  [[nodiscard]] explicit operator bool() const noexcept { return valid(); }
  [[nodiscard]] AcceptedGeneration generation() const noexcept { return generation_; }

  template <class T>
  [[nodiscard]] const T* read(const ParticipantHandle<T>& handle) const noexcept;

  template <class T>
  [[nodiscard]] const T* view(const ParticipantHandle<T>& handle) const noexcept {
    return read(handle);
  }

 private:
  friend class ProgramTransactionRegistry;

  AcceptedReadLease(const void* registry, std::shared_lock<std::shared_mutex>&& lock,
                    AcceptedGeneration generation) noexcept;
  AcceptedReadLease(const void* registry, AcceptedGeneration generation,
                    const ProvisionalReadLease* provisional_scope) noexcept
      : registry_(registry), generation_(generation), provisional_scope_(provisional_scope) {}

  AcceptedReadLease(const void* registry, AcceptedGeneration generation,
                    AcceptedReadLease* owning_root) noexcept;
  static AcceptedReadLease* find_(const void* registry) noexcept;
  static bool contains_(const AcceptedReadLease* lease) noexcept;
  [[nodiscard]] bool authenticated_root_(const void* registry) const noexcept;
  void link_root_() noexcept;
  void unlink_root_() noexcept;
  void release_() noexcept;
  void move_from_(AcceptedReadLease& other) noexcept;

  const void* registry_ = nullptr;
  std::shared_lock<std::shared_mutex> lock_{};
  AcceptedGeneration generation_{};
  const ProvisionalReadLease* provisional_scope_ = nullptr;
  // Accepted roots are the only nodes linked into this TLS stack. Borrowed leases retain the
  // authenticated root and increment its fixed in-object count; they never acquire a mutex.
  AcceptedReadLease* root_ = nullptr;
  AcceptedReadLease* next_ = nullptr;
  std::thread::id owner_thread_{};
  std::size_t borrowed_count_ = 0;
  bool linked_ = false;
  bool borrowed_ = false;
  inline static thread_local AcceptedReadLease* scope_head_ = nullptr;
};

/// Explicit writer-thread access to the resident candidate during candidate preparation.  The lease
/// owns no lock and no heap storage: it links one fixed node into a thread-local scope chain.  A
/// public accepted reader bypasses the shared mutex only while this exact scope is linked and the
/// registry still reports one of the two authenticated candidate-preparation phases.
class ProvisionalReadLease final {
 public:
  ProvisionalReadLease() noexcept = default;
  ProvisionalReadLease(const ProvisionalReadLease&) = delete;
  ProvisionalReadLease& operator=(const ProvisionalReadLease&) = delete;
  ProvisionalReadLease(ProvisionalReadLease&& other) noexcept;
  ProvisionalReadLease& operator=(ProvisionalReadLease&& other) noexcept;
  ~ProvisionalReadLease() noexcept;

  /// The lease is valid only on its creating thread, while its registry remains in candidate or
  /// solve/guard/effect-preparation phase and the lease is still linked.  It becomes invalid at
  /// hidden publication, seal, or rollback, so a stale context cannot authorize a post-publication
  /// read.
  [[nodiscard]] bool valid() const noexcept;
  [[nodiscard]] explicit operator bool() const noexcept { return valid(); }

  /// Release this scope on its creating thread.  Releasing from another thread or releasing an
  /// already moved/released lease is a deterministic logic error; the noexcept destructor remains
  /// strict and terminates on a cross-thread escape instead of silently leaving a TLS node linked.
  void release();

 private:
  friend class ProgramTransactionRegistry;
  friend class AcceptedReadLease;

  explicit ProvisionalReadLease(const ProgramTransactionRegistry* registry) noexcept;
  static ProvisionalReadLease* find_(const ProgramTransactionRegistry* registry) noexcept;
  static bool contains_(const ProvisionalReadLease* lease) noexcept;
  void unlink_() noexcept;
  void move_link_(ProvisionalReadLease& other) noexcept;

  inline static thread_local ProvisionalReadLease* scope_head_ = nullptr;
  const ProgramTransactionRegistry* registry_ = nullptr;
  std::thread::id owner_thread_{};
  ProvisionalReadLease* next_ = nullptr;
  bool linked_ = false;
};

/// A writer-side visibility lease for cold savepoint capture and external rollback.  It shares the
/// registry mutex with the step transaction, so a public reader can never observe the aggregate
/// swaps performed by an external restore.  The lease owns no transaction phase and is therefore
/// not a second execution engine.
class AcceptedWriteLease final {
 public:
  AcceptedWriteLease() noexcept = default;
  AcceptedWriteLease(const AcceptedWriteLease&) = delete;
  AcceptedWriteLease& operator=(const AcceptedWriteLease&) = delete;
  AcceptedWriteLease(AcceptedWriteLease&& other) noexcept;
  AcceptedWriteLease& operator=(AcceptedWriteLease&& other) noexcept;
  ~AcceptedWriteLease() noexcept;

  [[nodiscard]] bool valid() const noexcept { return lock_.owns_lock(); }
  [[nodiscard]] explicit operator bool() const noexcept { return valid(); }

 private:
  friend class ProgramTransactionRegistry;
  AcceptedWriteLease(const void* registry, std::unique_lock<std::shared_mutex>&& lock) noexcept
      : registry_(registry), lock_(std::move(lock)) {}
  void release_() noexcept;

  const void* registry_ = nullptr;
  std::unique_lock<std::shared_mutex> lock_{};
};

/// A prepared output/effect that can be published, compensated in reverse order, and finalized
/// exactly once.  The context belongs to the integrating runtime; this value owns no heap storage.
class PreparedCompensableEffect final {
 public:
  using Prepare = bool (*)(void* context) noexcept;
  using Publish = bool (*)(void* context) noexcept;
  using Compensate = void (*)(void* context) noexcept;
  using Discard = void (*)(void* context) noexcept;
  using Finalize = bool (*)(void* context) noexcept;

  PreparedCompensableEffect() noexcept = default;
  PreparedCompensableEffect(void* context, Publish publish, Compensate compensate,
                            Finalize finalize = nullptr, Discard discard = nullptr,
                            Prepare prepare = nullptr, std::uint64_t identity = 0) noexcept
      : context_(context),
        prepare_(prepare),
        publish_(publish),
        compensate_(compensate),
        discard_(discard),
        finalize_(finalize),
        identity_(identity) {}

  PreparedCompensableEffect(const PreparedCompensableEffect&) = delete;
  PreparedCompensableEffect& operator=(const PreparedCompensableEffect&) = delete;
  PreparedCompensableEffect(PreparedCompensableEffect&& other) noexcept
      : context_(std::exchange(other.context_, nullptr)),
        prepare_(std::exchange(other.prepare_, nullptr)),
        publish_(std::exchange(other.publish_, nullptr)),
        compensate_(std::exchange(other.compensate_, nullptr)),
        discard_(std::exchange(other.discard_, nullptr)),
        finalize_(std::exchange(other.finalize_, nullptr)),
        identity_(std::exchange(other.identity_, 0)) {}
  PreparedCompensableEffect& operator=(PreparedCompensableEffect&& other) noexcept {
    if (this != &other) {
      context_ = std::exchange(other.context_, nullptr);
      prepare_ = std::exchange(other.prepare_, nullptr);
      publish_ = std::exchange(other.publish_, nullptr);
      compensate_ = std::exchange(other.compensate_, nullptr);
      discard_ = std::exchange(other.discard_, nullptr);
      finalize_ = std::exchange(other.finalize_, nullptr);
      identity_ = std::exchange(other.identity_, 0);
    }
    return *this;
  }
  ~PreparedCompensableEffect() = default;

  [[nodiscard]] bool valid() const noexcept {
    return publish_ != nullptr && compensate_ != nullptr;
  }
  [[nodiscard]] std::uint64_t identity() const noexcept { return identity_; }

 private:
  friend class ProgramTransactionRegistry;
  friend class ProgramTransaction;

  void* context_ = nullptr;
  Prepare prepare_ = nullptr;
  Publish publish_ = nullptr;
  Compensate compensate_ = nullptr;
  Discard discard_ = nullptr;
  Finalize finalize_ = nullptr;
  std::uint64_t identity_ = 0;
};

/// Receipt for one effect.  The kernel writes the state bits exactly once; callers may persist the
/// receipt after seal as proof that the effect belongs to the accepted generation.
struct EffectReceipt final {
  std::uint64_t identity = 0;
  AcceptedGeneration generation{};
  std::uint32_t ordinal = kInvalidProgramTransactionIndex;
  bool published = false;
  bool compensated = false;
  bool finalized = false;
  bool finalize_attempted = false;

  [[nodiscard]] bool valid() const noexcept {
    return published && generation.valid() && ordinal != kInvalidProgramTransactionIndex;
  }
};

/// Summary receipt returned by `publish()`.  `operator bool` makes the no-throw API convenient in a
/// runtime while retaining the authenticated generation and counts for diagnostics.
struct ProgramPublishReceipt final {
  AcceptedGeneration generation{};
  std::uint32_t participant_count = 0;
  std::uint32_t effect_count = 0;
  ProgramTransactionPhase phase = ProgramTransactionPhase::kRolledBack;
  bool published = false;

  [[nodiscard]] explicit operator bool() const noexcept { return published; }
  [[nodiscard]] bool valid() const noexcept { return published; }
};

/// Summary receipt returned by `seal()`.  Sealing publishes the candidate generation and releases
/// the writer lock; it never finalizes an effect.
struct ProgramSealReceipt final {
  AcceptedGeneration generation{};
  std::uint32_t effect_count = 0;
  ProgramTransactionPhase phase = ProgramTransactionPhase::kRolledBack;
  bool sealed = false;

  [[nodiscard]] explicit operator bool() const noexcept { return sealed; }
  [[nodiscard]] bool valid() const noexcept { return sealed && generation.valid(); }
};

/// Summary receipt returned by `finalize()`.  A failed finalizer sets `fail_stop` while preserving
/// the accepted generation; it deliberately does not reopen or roll back scientific state.
struct ProgramFinalizeReceipt final {
  AcceptedGeneration generation{};
  std::uint32_t finalized_effects = 0;
  std::uint32_t failed_effects = 0;
  ProgramTransactionPhase phase = ProgramTransactionPhase::kFailStop;
  bool finalized = false;
  bool fail_stop = false;

  [[nodiscard]] explicit operator bool() const noexcept { return finalized && !fail_stop; }
  [[nodiscard]] bool accepted() const noexcept { return generation.valid(); }
};

/// One move-only transaction lease.  All mutating methods after begin are noexcept; misuse or a
/// callback failure is represented by `false` plus `fault()`, so the hot driver can report a
/// collective phase without constructing an exception.  The cold `begin()` boundary may throw if
/// a snapshot cannot be captured or if the registry is not bound.
class ProgramTransaction final {
 public:
  ProgramTransaction(const ProgramTransaction&) = delete;
  ProgramTransaction& operator=(const ProgramTransaction&) = delete;
  ProgramTransaction(ProgramTransaction&& other) noexcept;
  ProgramTransaction& operator=(ProgramTransaction&&) = delete;
  ~ProgramTransaction() noexcept;

  [[nodiscard]] ProgramTransactionPhase phase() const noexcept { return phase_; }
  [[nodiscard]] AcceptedGeneration base_generation() const noexcept { return base_generation_; }
  [[nodiscard]] AcceptedGeneration generation() const noexcept { return sealed_generation_; }
  [[nodiscard]] bool active() const noexcept { return active_; }
  [[nodiscard]] bool sealed() const noexcept {
    return phase_ == ProgramTransactionPhase::kAtomicSeal ||
           phase_ == ProgramTransactionPhase::kIrreversibleFinalize ||
           phase_ == ProgramTransactionPhase::kAccepted ||
           phase_ == ProgramTransactionPhase::kFailStop;
  }
  [[nodiscard]] const ProgramTransactionFault& fault() const noexcept { return fault_; }

  /// Transition snapshot -> candidate. A registry configured with
  /// ``set_candidate_visibility_lock(true)`` acquires its writer lease here, before the first
  /// candidate mutation, so accepted readers cannot observe a live candidate.
  bool begin_candidate() noexcept;
  bool candidate() noexcept { return begin_candidate(); }

  /// Transition candidate -> solve/guard/effect preparation.  The integrating runtime must consume
  /// its `pops::SolveOutcome` before entering `publish()`; this class intentionally has no solver API.
  bool begin_solve_guard_effect_prepare() noexcept;
  bool begin_prepare() noexcept { return begin_solve_guard_effect_prepare(); }
  bool prepare_phase() noexcept { return begin_solve_guard_effect_prepare(); }

  template <class T>
  [[nodiscard]] ProvisionalView<T> provisional(const ParticipantHandle<T>& handle) noexcept;

  template <class T>
  [[nodiscard]] ProvisionalView<T> provisional_view(const ParticipantHandle<T>& handle) noexcept {
    return provisional(handle);
  }

  /// Populate the next bind-frozen effect slot.  The optional preparation callback is deferred to
  /// `publish()`, where every rank validates every slot with the same fixed collective sequence.
  /// Structural mismatches are recorded in the slot and fail collectively before hidden publish.
  bool prepare_effect(EffectHandle handle, PreparedCompensableEffect effect) noexcept;
  bool add_effect(EffectHandle handle, PreparedCompensableEffect effect) noexcept {
    return prepare_effect(handle, std::move(effect));
  }

  /// Publish participant candidates while readers still see the accepted generation, then publish
  /// effects.  On any failure all already-published effects are compensated reverse-order and every
  /// participant is restored before returning false.  The writer lock remains held only after full
  /// success, blocking new readers until `seal()` or `rollback()`.
  [[nodiscard]] ProgramPublishReceipt publish() noexcept;
  [[nodiscard]] ProgramPublishReceipt hidden_publish() noexcept { return publish(); }

  /// Atomically seal the accepted generation.  A successful seal is irreversible: rollback becomes
  /// a no-op and scientific state remains accepted even when a later finalizer reports failure.
  [[nodiscard]] ProgramSealReceipt seal() noexcept;
  [[nodiscard]] ProgramSealReceipt atomic_seal() noexcept { return seal(); }

  /// Finalize each published effect at most once.  A failed finalizer records fail-stop and never
  /// invokes participant rollback or effect compensation.
  [[nodiscard]] ProgramFinalizeReceipt finalize() noexcept;
  [[nodiscard]] ProgramFinalizeReceipt irreversible_finalize() noexcept { return finalize(); }

  /// Restore the base image and compensate/discard prepared effects.  This operation is idempotent
  /// and intentionally does nothing after atomic seal.
  void rollback() noexcept;
  void reject() noexcept { rollback(); }

  [[nodiscard]] std::uint32_t effect_count() const noexcept;
  [[nodiscard]] const EffectReceipt* effect_receipt(std::uint32_t ordinal) const noexcept;

 private:
  friend class ProgramTransactionRegistry;

  explicit ProgramTransaction(ProgramTransactionRegistry& registry,
                              AcceptedGeneration generation) noexcept;
  void snapshot_();
  void fail_(ProgramTransactionPhase phase, ProgramTransactionFailure failure,
             std::uint32_t ordinal, std::uint32_t reason_code = 0) noexcept;
  [[nodiscard]] bool consensus_(ProgramTransactionPhase phase, std::uint32_t status) const noexcept;
  [[nodiscard]] bool prepare_effects_collectively_() noexcept;
  void restore_participants_() noexcept;
  void discard_prepared_() noexcept;
  void compensate_published_() noexcept;
  void close_rollback_() noexcept;
  void unlock_publication_() noexcept;
  [[nodiscard]] ProgramPublishReceipt failed_publish_() const noexcept;

  ProgramTransactionRegistry* registry_ = nullptr;
  AcceptedGeneration base_generation_{};
  AcceptedGeneration sealed_generation_{};
  ProgramTransactionPhase phase_ = ProgramTransactionPhase::kSnapshot;
  ProgramTransactionFault fault_{};
  std::uint32_t published_participants_ = 0;
  std::uint32_t next_effect_submission_ = 0;
  std::uint32_t effect_submission_protocol_status_ = 0;
  bool active_ = false;
  bool publication_complete_ = false;
  bool publication_lock_borrowed_ = false;
  std::unique_lock<std::shared_mutex> publication_lock_{};
};

/// Registry of state participants and preallocated transaction storage.  It is intentionally
/// non-copyable/non-movable because leases refer to its visibility mutex and participant addresses.
class ProgramTransactionRegistry final {
 public:
  explicit ProgramTransactionRegistry(ProgramTransactionBudget budget = {},
                                      ProgramTransactionConsensus consensus = {}) noexcept
      : budget_(budget), consensus_callback_(consensus) {}
  ProgramTransactionRegistry(const ProgramTransactionRegistry&) = delete;
  ProgramTransactionRegistry& operator=(const ProgramTransactionRegistry&) = delete;
  ProgramTransactionRegistry(ProgramTransactionRegistry&&) = delete;
  ProgramTransactionRegistry& operator=(ProgramTransactionRegistry&&) = delete;
  ~ProgramTransactionRegistry() = default;

  /// Register one typed object before bind.  The erased callbacks receive the exact object pointer
  /// associated with the typed handle.  This overload does not retain a callback object or allocate
  /// during a step.
  template <class T>
  [[nodiscard]] ParticipantHandle<T> register_participant(T& object, ProgramParticipantOps ops,
                                                          ProgramParticipantBudget budget = {}) {
    return register_participant_impl_<T>(static_cast<void*>(&object),
                                         detail::participant_type_token<T>(), ops, budget);
  }

  /// No-throw registration probe for generated bind paths.  It is useful when a runtime wants to
  /// turn a late-registration attempt into a fixed status code without constructing an exception;
  /// in particular, the bound fast path returns before touching any vector or allocator.  The
  /// throwing `register_participant` overload remains the diagnostic API for cold setup code.
  template <class T>
  [[nodiscard]] ParticipantHandle<T> try_register_participant(
      T& object, ProgramParticipantOps ops, ProgramParticipantBudget budget = {}) noexcept {
    if (bound_ || ops.snapshot == nullptr || ops.restore == nullptr || ops.publish == nullptr)
      return {};
    try {
      return register_participant_impl_<T>(static_cast<void*>(&object),
                                           detail::participant_type_token<T>(), ops, budget);
    } catch (...) {
      return {};
    }
  }

  template <class T>
    requires detail::MethodParticipant<T>
  [[nodiscard]] ParticipantHandle<T> try_register_participant(
      T& object, ProgramParticipantBudget budget = {}) noexcept {
    if (budget.restore_bytes == 0 && budget.candidate_bytes == 0)
      budget = {sizeof(T), sizeof(T)};
    return try_register_participant(object, detail::method_participant_ops<T>(), budget);
  }

  template <class T>
    requires detail::MethodParticipant<T>
  [[nodiscard]] ParticipantHandle<T> try_register_participant(
      T& object, std::size_t restore_bytes, std::size_t candidate_bytes = 0) noexcept {
    return try_register_participant(object, detail::method_participant_ops<T>(),
                                    ProgramParticipantBudget{restore_bytes, candidate_bytes});
  }

  /// Convenience registration for a state exposing noexcept `snapshot`, `restore`, and `publish`
  /// methods.  A candidate-aware state may implement those methods by swapping its own detached
  /// candidate carrier at `publish()`.
  template <class T>
    requires detail::MethodParticipant<T>
  [[nodiscard]] ParticipantHandle<T> register_participant(T& object,
                                                          ProgramParticipantBudget budget = {}) {
    if (budget.restore_bytes == 0 && budget.candidate_bytes == 0)
      budget = {sizeof(T), sizeof(T)};
    return register_participant(object, detail::method_participant_ops<T>(), budget);
  }

  template <class T>
    requires detail::MethodParticipant<T>
  [[nodiscard]] ParticipantHandle<T> register_participant(T& object, std::size_t restore_bytes,
                                                          std::size_t candidate_bytes = 0) {
    return register_participant(object, detail::method_participant_ops<T>(),
                                ProgramParticipantBudget{restore_bytes, candidate_bytes});
  }

  /// Declare one compensable-effect slot before bind.  Identities are non-zero, unique and ordered;
  /// after bind a transaction can only populate these exact slots and cannot grow the registry.
  [[nodiscard]] EffectHandle register_effect(std::uint64_t identity);
  [[nodiscard]] EffectHandle try_register_effect(std::uint64_t identity) noexcept;

  /// Reserve the exact participant image/effect capacity and freeze registration order and
  /// budgets.  No operation after this method can grow a vector or allocate a restore image.
  void bind();

  [[nodiscard]] bool bound() const noexcept { return bound_; }
  [[nodiscard]] bool fail_stop() const noexcept { return fail_stop_; }
  [[nodiscard]] std::size_t participant_count() const noexcept { return participants_.size(); }
  [[nodiscard]] ProgramParticipantInfo participant_info(std::uint32_t index) const noexcept {
    if (index >= participants_.size())
      return {};
    const ParticipantEntry& participant = participants_[index];
    return {participant.order, participant.budget};
  }
  [[nodiscard]] std::size_t effect_capacity() const noexcept { return effects_.size(); }
  [[nodiscard]] ProgramEffectInfo effect_info(std::uint32_t index) const noexcept {
    if (index >= effects_.size())
      return {};
    const auto& effect = effects_[index];
    return {effect.identity, effect.order, effect.bind_status};
  }
  [[nodiscard]] std::size_t restore_bytes() const noexcept { return restore_bytes_; }
  [[nodiscard]] std::size_t candidate_bytes() const noexcept { return candidate_bytes_; }
  [[nodiscard]] ProgramTransactionBudget budget() const noexcept { return budget_; }
  [[nodiscard]] AcceptedGeneration accepted_generation() const noexcept {
    return AcceptedGeneration{accepted_generation_.load(std::memory_order_acquire)};
  }
  [[nodiscard]] ProgramTransactionPhase phase() const noexcept {
    return fail_stop_ ? ProgramTransactionPhase::kFailStop
                      : (bound_ ? (active_ ? active_phase_ : ProgramTransactionPhase::kAccepted)
                                : ProgramTransactionPhase::kUnbound);
  }
  [[nodiscard]] const ProgramTransactionFault& last_fault() const noexcept { return last_fault_; }

  /// Acquire a read lease for the last sealed generation.  A live-carrier registry may hold its
  /// visibility writer for the complete Candidate interval; in that mode a foreign reader blocks
  /// until seal/rollback and a same-thread reader requires an explicit ProvisionalReadLease.
  [[nodiscard]] AcceptedReadLease acquire_read() const;

  /// Acquire an explicit writer-thread scope for reading the resident candidate.  This scope is
  /// accepted only while this registry owns its visibility writer for an active Candidate or
  /// SolveGuardEffectPrepare transaction.  It is deliberately separate from `acquire_read()` so a
  /// public reader cannot accidentally bypass the shared mutex through same-thread recursion.
  [[nodiscard]] ProvisionalReadLease acquire_provisional_read() const;

  /// Acquire the same visibility writer used by a step transaction.  This is for cold external
  /// savepoint capture and rollback only; it does not alter the registry phase or generation.
  [[nodiscard]] AcceptedWriteLease acquire_write() const;

  /// Update the MPI-neutral callback only before bind.  The callback is copied, never owned.
  void set_consensus(ProgramTransactionConsensus consensus);
  [[nodiscard]] bool try_set_consensus(ProgramTransactionConsensus consensus) noexcept {
    if (bound_)
      return false;
    consensus_callback_ = consensus;
    return true;
  }
  /// Freeze the visibility policy before bind. Live-carrier runtimes hold the writer lease for the
  /// complete candidate/rollback interval; detached participants may retain the publish-time lock.
  void set_candidate_visibility_lock(bool enabled);
  [[nodiscard]] ProgramTransactionConsensus consensus() const noexcept {
    return consensus_callback_;
  }

  [[nodiscard]] ProgramTransaction begin();

 private:
  friend class ProgramTransaction;
  friend class AcceptedReadLease;
  friend class AcceptedWriteLease;
  friend class ProvisionalReadLease;

  struct ParticipantEntry final {
    void* object = nullptr;
    const void* type_token = nullptr;
    ProgramParticipantOps ops{};
    ProgramParticipantBudget budget{};
    std::vector<std::byte> restore_image{};
    std::uint32_t order = kInvalidProgramTransactionIndex;
  };

  struct EffectEntry final {
    std::uint64_t identity = 0;
    std::uint32_t order = kInvalidProgramTransactionIndex;
    ProgramEffectSlotStatus bind_status = ProgramEffectSlotStatus::kUnregistered;
    PreparedCompensableEffect effect{};
    EffectReceipt receipt{};
    std::uint32_t submission_status = 0;
    bool submitted = false;
    bool prepared = false;
    bool discarded = false;
  };

  template <class T>
  [[nodiscard]] ParticipantHandle<T> register_participant_impl_(void* object,
                                                                const void* type_token,
                                                                ProgramParticipantOps ops,
                                                                ProgramParticipantBudget budget) {
    if (bound_)
      throw std::logic_error(
          "ProgramTransactionRegistry refuses participant registration after bind");
    if (object == nullptr || type_token == nullptr || ops.snapshot == nullptr ||
        ops.restore == nullptr || ops.publish == nullptr)
      throw std::invalid_argument(
          "ProgramTransactionRegistry requires complete participant callbacks");
    if (participants_.size() == std::numeric_limits<std::uint32_t>::max())
      throw std::length_error("ProgramTransactionRegistry participant index overflow");
    for (const ParticipantEntry& existing : participants_)
      if (existing.object == object)
        throw std::invalid_argument(
            "ProgramTransactionRegistry refuses duplicate participant object");
    ParticipantEntry entry;
    entry.object = object;
    entry.type_token = type_token;
    entry.ops = ops;
    entry.budget = budget;
    entry.order = static_cast<std::uint32_t>(participants_.size());
    participants_.push_back(std::move(entry));
    return ParticipantHandle<T>{static_cast<std::uint32_t>(participants_.size() - 1), type_token};
  }

  template <class T>
  [[nodiscard]] const T* read_(const ParticipantHandle<T>& handle) const noexcept {
    if (!handle.valid() || handle.index >= participants_.size())
      return nullptr;
    const ParticipantEntry& entry = participants_[handle.index];
    if (entry.type_token != handle.type_token)
      return nullptr;
    return static_cast<const T*>(entry.object);
  }

  template <class T>
  [[nodiscard]] T* provisional_(const ParticipantHandle<T>& handle) noexcept {
    if (!handle.valid() || handle.index >= participants_.size())
      return nullptr;
    ParticipantEntry& entry = participants_[handle.index];
    if (entry.type_token != handle.type_token)
      return nullptr;
    return static_cast<T*>(entry.ops.candidate == nullptr ? entry.object
                                                          : entry.ops.candidate(entry.object));
  }

  template <class T>
  [[nodiscard]] const T* provisional_read_(const ParticipantHandle<T>& handle,
                                           const ProvisionalReadLease* scope) const noexcept {
    if (!provisional_scope_allowed_(scope) || !handle.valid() ||
        handle.index >= participants_.size())
      return nullptr;
    const ParticipantEntry& entry = participants_[handle.index];
    if (entry.type_token != handle.type_token)
      return nullptr;
    return static_cast<const T*>(
        entry.ops.candidate == nullptr ? entry.object : entry.ops.candidate(entry.object));
  }

  [[nodiscard]] bool provisional_scope_allowed_(const ProvisionalReadLease* scope) const noexcept;

  struct WriterThreadMarker final {
    const ProgramTransactionRegistry* registry = nullptr;
    WriterThreadMarker* next = nullptr;
    bool linked = false;
  };

  [[nodiscard]] static bool writer_thread_owns_(
      const ProgramTransactionRegistry* registry) noexcept {
    for (WriterThreadMarker* marker = writer_thread_head_; marker != nullptr;
         marker = marker->next) {
      if (marker->registry == registry)
        return true;
    }
    return false;
  }

  void link_writer_thread_() const noexcept;
  void unlink_writer_thread_() const noexcept;

  [[nodiscard]] bool consensus_(ProgramTransactionPhase phase,
                                std::uint32_t status) const noexcept {
    return consensus_callback_.agree(phase, status);
  }

  void clear_active_() noexcept {
    active_ = false;
    active_phase_ = ProgramTransactionPhase::kAccepted;
  }

  ProgramTransactionBudget budget_{};
  ProgramTransactionConsensus consensus_callback_{};
  std::vector<ParticipantEntry> participants_{};
  std::vector<EffectEntry> effects_{};
  mutable std::shared_mutex visibility_mutex_{};
  std::atomic<std::uint64_t> accepted_generation_{0};
  std::size_t restore_bytes_ = 0;
  std::size_t candidate_bytes_ = 0;
  bool bound_ = false;
  bool candidate_visibility_lock_ = false;
  bool active_ = false;
  bool fail_stop_ = false;
  ProgramTransactionPhase active_phase_ = ProgramTransactionPhase::kUnbound;
  ProgramTransactionFault last_fault_{};
  // One fixed marker per registry is linked into this thread's stack while its visibility writer
  // is held. A linked stack, rather than one TLS pointer, supports nested multi-layout registries.
  mutable WriterThreadMarker writer_thread_marker_{};
  inline static thread_local WriterThreadMarker* writer_thread_head_ = nullptr;
};

inline AcceptedReadLease::AcceptedReadLease(const void* registry,
                                            std::shared_lock<std::shared_mutex>&& lock,
                                            AcceptedGeneration generation) noexcept
    : registry_(registry),
      lock_(std::move(lock)),
      generation_(generation),
      root_(this),
      owner_thread_(std::this_thread::get_id()),
      linked_(true) {
  if (registry_ == nullptr || !lock_.owns_lock())
    std::terminate();
  link_root_();
}

inline AcceptedReadLease::AcceptedReadLease(const void* registry, AcceptedGeneration generation,
                                            AcceptedReadLease* owning_root) noexcept
    : registry_(registry),
      generation_(generation),
      root_(owning_root),
      owner_thread_(std::this_thread::get_id()),
      borrowed_(true) {
  if (owning_root == nullptr || !owning_root->authenticated_root_(registry_) ||
      owning_root->borrowed_count_ == std::numeric_limits<std::size_t>::max())
    std::terminate();
  ++owning_root->borrowed_count_;
}

inline AcceptedReadLease* AcceptedReadLease::find_(const void* registry) noexcept {
  for (AcceptedReadLease* lease = scope_head_; lease != nullptr; lease = lease->next_) {
    if (lease->linked_ && !lease->borrowed_ && lease->root_ == lease &&
        lease->registry_ == registry)
      return lease;
  }
  return nullptr;
}

inline bool AcceptedReadLease::contains_(const AcceptedReadLease* lease) noexcept {
  for (const AcceptedReadLease* current = scope_head_; current != nullptr;
       current = current->next_) {
    if (current == lease)
      return current->linked_ && !current->borrowed_ && current->root_ == current;
  }
  return false;
}

inline bool AcceptedReadLease::authenticated_root_(const void* registry) const noexcept {
  return registry_ != nullptr && registry_ == registry && root_ == this && !borrowed_ && linked_ &&
         owner_thread_ == std::this_thread::get_id() && lock_.owns_lock() && contains_(this);
}

inline void AcceptedReadLease::link_root_() noexcept {
  if (registry_ == nullptr || root_ != this || borrowed_ || !linked_ ||
      owner_thread_ != std::this_thread::get_id() || !lock_.owns_lock() ||
      find_(registry_) != nullptr)
    std::terminate();
  next_ = scope_head_;
  scope_head_ = this;
}

inline void AcceptedReadLease::unlink_root_() noexcept {
  if (!linked_)
    return;
  if (!authenticated_root_(registry_) || borrowed_count_ != 0)
    std::terminate();
  if (scope_head_ == this) {
    scope_head_ = next_;
  } else {
    AcceptedReadLease* previous = scope_head_;
    while (previous != nullptr && previous->next_ != this)
      previous = previous->next_;
    if (previous == nullptr)
      std::terminate();
    previous->next_ = next_;
  }
  linked_ = false;
  root_ = nullptr;
  next_ = nullptr;
}

inline void AcceptedReadLease::release_() noexcept {
  if (borrowed_) {
    if (root_ == nullptr || owner_thread_ != std::this_thread::get_id() ||
        !root_->authenticated_root_(registry_) || root_->borrowed_count_ == 0)
      std::terminate();
    --root_->borrowed_count_;
    registry_ = nullptr;
    root_ = nullptr;
    owner_thread_ = {};
    borrowed_ = false;
    generation_ = {};
    return;
  }

  if (provisional_scope_ != nullptr) {
    if (lock_.owns_lock() || root_ != nullptr || linked_ || borrowed_)
      std::terminate();
    registry_ = nullptr;
    provisional_scope_ = nullptr;
    generation_ = {};
    return;
  }

  if (root_ != nullptr) {
    if (root_ != this || !linked_ || !authenticated_root_(registry_) || borrowed_count_ != 0)
      std::terminate();
    unlink_root_();
    if (!lock_.owns_lock())
      std::terminate();
    lock_.unlock();
  } else if (registry_ != nullptr || linked_ || lock_.owns_lock()) {
    std::terminate();
  }

  registry_ = nullptr;
  root_ = nullptr;
  next_ = nullptr;
  owner_thread_ = {};
  linked_ = false;
  borrowed_ = false;
  borrowed_count_ = 0;
  generation_ = {};
}

inline void AcceptedReadLease::move_from_(AcceptedReadLease& other) noexcept {
  if (this == &other)
    return;

  if (other.provisional_scope_ != nullptr) {
    if (other.lock_.owns_lock() || other.root_ != nullptr || other.linked_ || other.borrowed_)
      std::terminate();
    registry_ = std::exchange(other.registry_, nullptr);
    generation_ = other.generation_;
    provisional_scope_ = std::exchange(other.provisional_scope_, nullptr);
    other.generation_ = {};
    return;
  }

  if (other.borrowed_) {
    if (other.owner_thread_ != std::this_thread::get_id() || other.root_ == nullptr ||
        !other.root_->authenticated_root_(other.registry_))
      std::terminate();
    registry_ = std::exchange(other.registry_, nullptr);
    generation_ = other.generation_;
    root_ = other.root_;
    owner_thread_ = other.owner_thread_;
    borrowed_ = true;
    other.root_ = nullptr;
    other.owner_thread_ = {};
    other.borrowed_ = false;
    other.generation_ = {};
    return;
  }

  if (other.linked_) {
    if (other.root_ != &other || other.owner_thread_ != std::this_thread::get_id() ||
        other.borrowed_count_ != 0 || !other.authenticated_root_(other.registry_))
      std::terminate();

    registry_ = other.registry_;
    generation_ = other.generation_;
    root_ = this;
    next_ = other.next_;
    owner_thread_ = other.owner_thread_;
    linked_ = true;
    lock_ = std::move(other.lock_);

    if (scope_head_ == &other) {
      scope_head_ = this;
    } else {
      AcceptedReadLease* previous = scope_head_;
      while (previous != nullptr && previous->next_ != &other)
        previous = previous->next_;
      if (previous == nullptr)
        std::terminate();
      previous->next_ = this;
    }
    other.registry_ = nullptr;
    other.root_ = nullptr;
    other.next_ = nullptr;
    other.owner_thread_ = {};
    other.linked_ = false;
    other.generation_ = {};
    return;
  }

  if (other.registry_ != nullptr || other.root_ != nullptr || other.lock_.owns_lock() ||
      other.borrowed_count_ != 0)
    std::terminate();
}

inline ProvisionalReadLease::ProvisionalReadLease(
    const ProgramTransactionRegistry* registry) noexcept
    : registry_(registry),
      owner_thread_(std::this_thread::get_id()),
      next_(scope_head_),
      linked_(registry != nullptr) {
  if (linked_)
    scope_head_ = this;
  else
    owner_thread_ = {};
}

inline ProvisionalReadLease* ProvisionalReadLease::find_(
    const ProgramTransactionRegistry* registry) noexcept {
  for (ProvisionalReadLease* scope = scope_head_; scope != nullptr; scope = scope->next_) {
    if (scope->linked_ && scope->registry_ == registry)
      return scope;
  }
  return nullptr;
}

inline bool ProvisionalReadLease::contains_(const ProvisionalReadLease* lease) noexcept {
  for (const ProvisionalReadLease* scope = scope_head_; scope != nullptr; scope = scope->next_) {
    if (scope == lease)
      return scope->linked_;
  }
  return false;
}

inline void ProvisionalReadLease::unlink_() noexcept {
  if (!linked_)
    return;
  if (scope_head_ == this) {
    scope_head_ = next_;
  } else {
    ProvisionalReadLease* previous = scope_head_;
    while (previous != nullptr && previous->next_ != this)
      previous = previous->next_;
    if (previous == nullptr)
      std::terminate();
    previous->next_ = next_;
  }
  registry_ = nullptr;
  owner_thread_ = {};
  next_ = nullptr;
  linked_ = false;
}

inline void ProvisionalReadLease::move_link_(ProvisionalReadLease& other) noexcept {
  if (!other.linked_)
    return;
  if (other.owner_thread_ != std::this_thread::get_id())
    std::terminate();

  registry_ = other.registry_;
  owner_thread_ = other.owner_thread_;
  next_ = other.next_;
  linked_ = true;
  if (scope_head_ == &other) {
    scope_head_ = this;
  } else {
    ProvisionalReadLease* previous = scope_head_;
    while (previous != nullptr && previous->next_ != &other)
      previous = previous->next_;
    if (previous == nullptr)
      std::terminate();
    previous->next_ = this;
  }
  other.registry_ = nullptr;
  other.owner_thread_ = {};
  other.next_ = nullptr;
  other.linked_ = false;
}

inline ProvisionalReadLease::ProvisionalReadLease(ProvisionalReadLease&& other) noexcept {
  move_link_(other);
}

inline ProvisionalReadLease& ProvisionalReadLease::operator=(
    ProvisionalReadLease&& other) noexcept {
  if (this == &other)
    return *this;
  if (linked_) {
    if (owner_thread_ != std::this_thread::get_id())
      std::terminate();
    unlink_();
  }
  registry_ = nullptr;
  owner_thread_ = {};
  next_ = nullptr;
  linked_ = false;
  move_link_(other);
  return *this;
}

inline ProvisionalReadLease::~ProvisionalReadLease() noexcept {
  if (!linked_)
    return;
  if (owner_thread_ != std::this_thread::get_id())
    std::terminate();
  unlink_();
}

inline bool ProvisionalReadLease::valid() const noexcept {
  if (!linked_ || registry_ == nullptr || owner_thread_ != std::this_thread::get_id() ||
      !contains_(this))
    return false;
  return registry_->provisional_scope_allowed_(this);
}

inline void ProvisionalReadLease::release() {
  if (!linked_)
    throw std::logic_error("Program provisional read scope is already released");
  if (owner_thread_ != std::this_thread::get_id())
    throw std::logic_error("Program provisional read scope must be released on its owner thread");
  unlink_();
}

inline bool ProgramTransactionRegistry::provisional_scope_allowed_(
    const ProvisionalReadLease* scope) const noexcept {
  if (scope == nullptr || !ProvisionalReadLease::contains_(scope))
    return false;
  if (!scope->linked_ || scope->registry_ != this ||
      scope->owner_thread_ != std::this_thread::get_id())
    return false;
  if (!bound_ || fail_stop_ || !active_ ||
      last_fault_.failure != ProgramTransactionFailure::kNone || !writer_thread_owns_(this))
    return false;
  return active_phase_ == ProgramTransactionPhase::kCandidate ||
         active_phase_ == ProgramTransactionPhase::kSolveGuardEffectPrepare;
}

inline void ProgramTransactionRegistry::link_writer_thread_() const noexcept {
  WriterThreadMarker& marker = writer_thread_marker_;
  if (marker.linked || writer_thread_owns_(this))
    std::terminate();
  marker.registry = this;
  marker.next = writer_thread_head_;
  marker.linked = true;
  writer_thread_head_ = &marker;
}

inline void ProgramTransactionRegistry::unlink_writer_thread_() const noexcept {
  WriterThreadMarker& marker = writer_thread_marker_;
  if (!marker.linked)
    return;
  if (marker.registry != this)
    std::terminate();
  if (writer_thread_head_ == &marker) {
    writer_thread_head_ = marker.next;
  } else {
    WriterThreadMarker* previous = writer_thread_head_;
    while (previous != nullptr && previous->next != &marker)
      previous = previous->next;
    if (previous == nullptr)
      std::terminate();
    previous->next = marker.next;
  }
  marker.registry = nullptr;
  marker.next = nullptr;
  marker.linked = false;
}

inline bool AcceptedReadLease::valid() const noexcept {
  if (provisional_scope_ != nullptr) {
    if (!ProvisionalReadLease::contains_(provisional_scope_))
      return false;
    return registry_ != nullptr &&
           static_cast<const ProgramTransactionRegistry*>(registry_)->provisional_scope_allowed_(
               provisional_scope_);
  }
  if (borrowed_)
    return root_ != nullptr && owner_thread_ == std::this_thread::get_id() &&
           root_->authenticated_root_(registry_) && root_->borrowed_count_ != 0;
  return authenticated_root_(registry_);
}

inline AcceptedWriteLease::AcceptedWriteLease(AcceptedWriteLease&& other) noexcept
    : registry_(std::exchange(other.registry_, nullptr)), lock_(std::move(other.lock_)) {}

inline AcceptedWriteLease& AcceptedWriteLease::operator=(AcceptedWriteLease&& other) noexcept {
  if (this == &other)
    return *this;
  release_();
  registry_ = std::exchange(other.registry_, nullptr);
  lock_ = std::move(other.lock_);
  return *this;
}

inline AcceptedWriteLease::~AcceptedWriteLease() noexcept {
  release_();
}

inline void AcceptedWriteLease::release_() noexcept {
  if (lock_.owns_lock())
    lock_.unlock();
  if (registry_ != nullptr)
    static_cast<const ProgramTransactionRegistry*>(registry_)->unlink_writer_thread_();
  registry_ = nullptr;
}

inline ProgramTransaction::ProgramTransaction(ProgramTransactionRegistry& registry,
                                              AcceptedGeneration generation) noexcept
    : registry_(&registry),
      base_generation_(generation),
      phase_(ProgramTransactionPhase::kSnapshot),
      active_(true) {}

inline ProgramTransaction::ProgramTransaction(ProgramTransaction&& other) noexcept
    : registry_(std::exchange(other.registry_, nullptr)),
      base_generation_(other.base_generation_),
      sealed_generation_(other.sealed_generation_),
      phase_(other.phase_),
      fault_(other.fault_),
      published_participants_(other.published_participants_),
      next_effect_submission_(other.next_effect_submission_),
      effect_submission_protocol_status_(other.effect_submission_protocol_status_),
      active_(std::exchange(other.active_, false)),
      publication_complete_(other.publication_complete_),
      publication_lock_borrowed_(other.publication_lock_borrowed_),
      publication_lock_(std::move(other.publication_lock_)) {
  if (registry_ != nullptr && active_)
    registry_->active_phase_ = phase_;
}

inline ProgramTransaction::~ProgramTransaction() noexcept {
  if (!active_)
    return;
  if (phase_ == ProgramTransactionPhase::kAtomicSeal ||
      phase_ == ProgramTransactionPhase::kIrreversibleFinalize ||
      phase_ == ProgramTransactionPhase::kFailStop) {
    (void)finalize();
  } else {
    rollback();
  }
}

inline void ProgramTransaction::snapshot_() {
  if (registry_ == nullptr)
    throw std::logic_error("ProgramTransaction has no registry");
  const bool writer_reentrant = ProgramTransactionRegistry::writer_thread_owns_(registry_);
  std::shared_lock<std::shared_mutex> read_lock;
  if (!writer_reentrant)
    read_lock = std::shared_lock<std::shared_mutex>(registry_->visibility_mutex_);
  bool local_success = true;
  std::uint32_t local_failure = kInvalidProgramTransactionIndex;
  for (std::size_t index = 0; index < registry_->participants_.size(); ++index) {
    auto& participant = registry_->participants_[index];
    const bool captured = participant.ops.snapshot(
        participant.object, participant.restore_image.data(), participant.restore_image.size());
    if (!captured && local_success) {
      local_success = false;
      local_failure = static_cast<std::uint32_t>(index);
    }
  }
  // Snapshot is a collective phase too.  Every rank reaches this agreement before any local
  // exception escapes, so a failed capture cannot strand peers inside candidate execution.
  const bool collectively_success =
      consensus_(ProgramTransactionPhase::kSnapshot, local_success ? 0u : 1u);
  if (!local_success || !collectively_success) {
    fail_(ProgramTransactionPhase::kSnapshot, ProgramTransactionFailure::kSnapshot,
          local_success ? kInvalidProgramTransactionIndex : local_failure,
          collectively_success ? 0u : 1u);
    if (read_lock.owns_lock())
      read_lock.unlock();
    // No candidate has been entered and the callback contract says snapshot is read-only with
    // respect to the participant.  Do not restore partially written image bytes: the next begin
    // recaptures the complete image before it can enter candidate.
    phase_ = ProgramTransactionPhase::kRolledBack;
    active_ = false;
    if (registry_ != nullptr)
      registry_->clear_active_();
    throw std::runtime_error("ProgramTransaction participant snapshot failed collectively");
  }
}

inline ProgramTransaction ProgramTransactionRegistry::begin() {
  if (!bound_)
    throw std::logic_error("ProgramTransactionRegistry::begin requires bind");
  if (fail_stop_)
    throw std::logic_error("ProgramTransactionRegistry is fail-stop");
  if (AcceptedReadLease* root = AcceptedReadLease::find_(this)) {
    if (!root->authenticated_root_(this))
      std::terminate();
    throw std::logic_error("Program transaction cannot begin while an accepted read lease is held");
  }
  if (active_)
    throw std::logic_error("ProgramTransactionRegistry already has an active transaction");
  active_ = true;
  active_phase_ = ProgramTransactionPhase::kSnapshot;
  last_fault_ = {};
  for (EffectEntry& entry : effects_) {
    entry.effect = {};
    entry.receipt = {};
    entry.receipt.identity = entry.identity;
    entry.receipt.ordinal = entry.order;
    entry.submission_status = 0;
    entry.submitted = false;
    entry.prepared = false;
    entry.discarded = false;
  }
  ProgramTransaction transaction(*this, accepted_generation());
  try {
    transaction.snapshot_();
  } catch (...) {
    active_ = false;
    active_phase_ = ProgramTransactionPhase::kRolledBack;
    throw;
  }
  return transaction;
}

inline bool ProgramTransaction::consensus_(ProgramTransactionPhase phase,
                                           std::uint32_t status) const noexcept {
  return registry_ == nullptr || registry_->consensus_(phase, status);
}

inline void ProgramTransaction::fail_(ProgramTransactionPhase phase,
                                      ProgramTransactionFailure failure, std::uint32_t ordinal,
                                      std::uint32_t reason_code) noexcept {
  fault_ = {phase, failure, ordinal, reason_code};
  if (registry_ != nullptr)
    registry_->last_fault_ = fault_;
}

inline bool ProgramTransaction::begin_candidate() noexcept {
  if (!active_ || phase_ != ProgramTransactionPhase::kSnapshot)
    return false;
  phase_ = ProgramTransactionPhase::kCandidate;
  registry_->active_phase_ = phase_;
  std::uint32_t lock_status = 0;
  bool acquired_writer = false;
  if (registry_->candidate_visibility_lock_) {
    try {
      if (!ProgramTransactionRegistry::writer_thread_owns_(registry_)) {
        publication_lock_ =
            std::unique_lock<std::shared_mutex>(registry_->visibility_mutex_, std::try_to_lock);
        acquired_writer = publication_lock_.owns_lock();
        if (!acquired_writer)
          lock_status = 1;
      } else {
        // An external savepoint already owns this mutex. Hidden publication borrows that writer;
        // attempting another unique_lock here would self-deadlock.
        publication_lock_borrowed_ = true;
      }
    } catch (...) {
      lock_status = 2;
    }
  }
  const bool lock_consensus = consensus_(ProgramTransactionPhase::kCandidate, lock_status);
  if (lock_status != 0 || !lock_consensus) {
    if (publication_lock_.owns_lock())
      publication_lock_.unlock();
    publication_lock_borrowed_ = false;
    fail_(ProgramTransactionPhase::kCandidate, ProgramTransactionFailure::kCandidate,
          kInvalidProgramTransactionIndex, lock_status == 0 ? 3U : lock_status);
    return false;
  }
  if (acquired_writer) {
    // This marker is only a writer-recursion detector.  It never grants a read lease: a public
    // reader accidentally called from the writer callback is refused deterministically below.
    registry_->link_writer_thread_();
  }
  return true;
}

inline bool ProgramTransaction::begin_solve_guard_effect_prepare() noexcept {
  if (!active_ || phase_ != ProgramTransactionPhase::kCandidate)
    return false;
  phase_ = ProgramTransactionPhase::kSolveGuardEffectPrepare;
  registry_->active_phase_ = phase_;
  if (!consensus_(ProgramTransactionPhase::kSolveGuardEffectPrepare, 0)) {
    fail_(ProgramTransactionPhase::kSolveGuardEffectPrepare, ProgramTransactionFailure::kSolve,
          kInvalidProgramTransactionIndex, 1);
    return false;
  }
  return true;
}

template <class T>
inline ProvisionalView<T> ProgramTransaction::provisional(
    const ParticipantHandle<T>& handle) noexcept {
  if (!active_ || (phase_ != ProgramTransactionPhase::kCandidate &&
                   phase_ != ProgramTransactionPhase::kSolveGuardEffectPrepare))
    return {};
  T* object = registry_ == nullptr ? nullptr : registry_->provisional_(handle);
  return ProvisionalView<T>(object, base_generation_,
                            object == nullptr ? kInvalidProgramTransactionIndex : handle.index);
}

inline bool ProgramTransaction::prepare_effect(EffectHandle handle,
                                               PreparedCompensableEffect effect) noexcept {
  if (!active_ || phase_ != ProgramTransactionPhase::kSolveGuardEffectPrepare) {
    if (active_ && effect_submission_protocol_status_ == 0)
      effect_submission_protocol_status_ = 1;
    if (effect.discard_ != nullptr)
      effect.discard_(effect.context_);
    return false;
  }
  if (next_effect_submission_ >= registry_->effects_.size()) {
    if (effect_submission_protocol_status_ == 0)
      effect_submission_protocol_status_ = 2;
    if (effect.discard_ != nullptr)
      effect.discard_(effect.context_);
    return false;
  }
  auto& entry = registry_->effects_[next_effect_submission_];
  if (entry.bind_status != ProgramEffectSlotStatus::kFrozen) {
    if (effect_submission_protocol_status_ == 0)
      effect_submission_protocol_status_ = 3;
    if (effect.discard_ != nullptr)
      effect.discard_(effect.context_);
    return false;
  }
  entry.effect = std::move(effect);
  entry.submitted = true;
  entry.submission_status = 0;
  if (!handle.valid() || handle.index != entry.order || handle.identity != entry.identity)
    entry.submission_status = 2;
  else if (!entry.effect.valid())
    entry.submission_status = 3;
  else if (entry.effect.identity() != entry.identity)
    entry.submission_status = 4;
  ++next_effect_submission_;
  return true;
}

inline bool ProgramTransaction::prepare_effects_collectively_() noexcept {
  if (registry_ == nullptr)
    return false;
  const std::size_t effect_count = registry_->effects_.size();
  if (effect_count > std::numeric_limits<std::uint32_t>::max())
    std::terminate();

  // Agree on the frozen slot count first.  A count mismatch returns after one common collective,
  // rather than letting the shorter rank fall out of a per-slot reduction loop.
  const bool count_agrees = consensus_(ProgramTransactionPhase::kSolveGuardEffectPrepare,
                                       static_cast<std::uint32_t>(effect_count));
  const bool protocol_agrees = consensus_(ProgramTransactionPhase::kSolveGuardEffectPrepare,
                                          effect_submission_protocol_status_);
  if (!count_agrees) {
    fail_(ProgramTransactionPhase::kSolveGuardEffectPrepare,
          ProgramTransactionFailure::kEffectPrepare, kInvalidProgramTransactionIndex, 5);
    return false;
  }
  const bool protocol_ok = protocol_agrees && effect_submission_protocol_status_ == 0;

  bool collectively_prepared = protocol_ok;
  std::uint32_t first_failure = kInvalidProgramTransactionIndex;
  std::uint32_t first_reason = !protocol_agrees ? 8U : effect_submission_protocol_status_;
  for (std::uint32_t index = 0; index < effect_count; ++index) {
    auto& entry = registry_->effects_[index];
    std::uint32_t submission_status = entry.submission_status;
    if (!entry.submitted)
      submission_status = 1;

    // Authenticate every immutable slot property before invoking its preparation callback. A
    // preparation may itself enter a provider-owned collective, so letting one rank call it while
    // another rank has already rejected its bind status, handle, or identity would deadlock the
    // lane. Never short-circuit these reductions: every rank executes the same six consensus
    // calls for every frozen slot.
    const bool bind_status_agrees = consensus_(ProgramTransactionPhase::kSolveGuardEffectPrepare,
                                               static_cast<std::uint32_t>(entry.bind_status));
    const bool ordinal_agrees =
        consensus_(ProgramTransactionPhase::kSolveGuardEffectPrepare, entry.order);
    const bool identity_low_agrees =
        consensus_(ProgramTransactionPhase::kSolveGuardEffectPrepare,
                   static_cast<std::uint32_t>(entry.identity & 0xffffffffULL));
    const bool identity_high_agrees =
        consensus_(ProgramTransactionPhase::kSolveGuardEffectPrepare,
                   static_cast<std::uint32_t>((entry.identity >> 32U) & 0xffffffffULL));
    const bool submission_status_agrees =
        consensus_(ProgramTransactionPhase::kSolveGuardEffectPrepare, submission_status);
    const bool bind_status_ok = entry.bind_status == ProgramEffectSlotStatus::kFrozen;
    const bool submission_contract_agrees = protocol_ok && bind_status_agrees && bind_status_ok &&
                                            ordinal_agrees && identity_low_agrees &&
                                            identity_high_agrees && submission_status_agrees;

    std::uint32_t preparation_status = 0;
    if (!submission_contract_agrees)
      preparation_status = 7;
    else if (submission_status != 0)
      preparation_status = submission_status;
    else if (entry.effect.prepare_ != nullptr && !entry.effect.prepare_(entry.effect.context_))
      preparation_status = 6;
    const bool preparation_status_agrees =
        consensus_(ProgramTransactionPhase::kSolveGuardEffectPrepare, preparation_status);
    const bool slot_ok = submission_contract_agrees && submission_status == 0 &&
                         preparation_status == 0 && preparation_status_agrees;
    entry.prepared = slot_ok;
    if (!slot_ok && collectively_prepared) {
      collectively_prepared = false;
      first_failure = index;
      first_reason =
          !submission_contract_agrees || !preparation_status_agrees ? 7 : preparation_status;
    }
  }
  if (!collectively_prepared) {
    const ProgramTransactionFailure failure =
        protocol_agrees && effect_submission_protocol_status_ == 2
            ? ProgramTransactionFailure::kBudget
            : ProgramTransactionFailure::kEffectPrepare;
    fail_(ProgramTransactionPhase::kSolveGuardEffectPrepare, failure, first_failure, first_reason);
  }
  return collectively_prepared;
}

inline ProgramPublishReceipt ProgramTransaction::failed_publish_() const noexcept {
  return {base_generation_, 0,
          static_cast<std::uint32_t>(registry_ == nullptr ? 0 : registry_->effects_.size()),
          ProgramTransactionPhase::kRolledBack, false};
}

inline void ProgramTransaction::unlock_publication_() noexcept {
  if (publication_lock_.owns_lock()) {
    publication_lock_.unlock();
    if (registry_ != nullptr)
      registry_->unlink_writer_thread_();
  }
}

inline void ProgramTransaction::compensate_published_() noexcept {
  if (registry_ == nullptr)
    return;
  for (std::size_t index = registry_->effects_.size(); index-- > 0;) {
    auto& entry = registry_->effects_[index];
    if (!entry.receipt.published || entry.receipt.compensated)
      continue;
    entry.effect.compensate_(entry.effect.context_);
    entry.receipt.compensated = true;
  }
}

inline void ProgramTransaction::discard_prepared_() noexcept {
  if (registry_ == nullptr)
    return;
  for (std::size_t index = registry_->effects_.size(); index-- > 0;) {
    auto& entry = registry_->effects_[index];
    if (entry.receipt.published || entry.discarded)
      continue;
    if (entry.effect.discard_ != nullptr)
      entry.effect.discard_(entry.effect.context_);
    entry.discarded = true;
  }
}

inline void ProgramTransaction::restore_participants_() noexcept {
  if (registry_ == nullptr)
    return;
  for (std::size_t index = registry_->participants_.size(); index-- > 0;) {
    auto& participant = registry_->participants_[index];
    if (participant.ops.rollback != nullptr)
      participant.ops.rollback(participant.object, participant.restore_image.data(),
                               participant.restore_image.size());
    else
      participant.ops.restore(participant.object, participant.restore_image.data(),
                              participant.restore_image.size());
  }
}

inline void ProgramTransaction::close_rollback_() noexcept {
  if (!active_)
    return;
  compensate_published_();
  discard_prepared_();
  restore_participants_();
  unlock_publication_();
  phase_ = ProgramTransactionPhase::kRolledBack;
  publication_complete_ = false;
  if (registry_ != nullptr)
    registry_->clear_active_();
  active_ = false;
}

inline ProgramPublishReceipt ProgramTransaction::publish() noexcept {
  if (!active_ || phase_ != ProgramTransactionPhase::kSolveGuardEffectPrepare)
    return failed_publish_();
  if (!prepare_effects_collectively_()) {
    close_rollback_();
    return failed_publish_();
  }
  phase_ = ProgramTransactionPhase::kHiddenPublish;
  registry_->active_phase_ = phase_;
  std::uint32_t lock_status = 0;
  bool acquired_writer = false;
  if (!publication_lock_.owns_lock() && !publication_lock_borrowed_) {
    try {
      publication_lock_ =
          std::unique_lock<std::shared_mutex>(registry_->visibility_mutex_, std::try_to_lock);
      acquired_writer = publication_lock_.owns_lock();
      if (!acquired_writer)
        lock_status = 1;
    } catch (...) {
      lock_status = 2;
    }
  }
  const bool lock_consensus = consensus_(ProgramTransactionPhase::kHiddenPublish, lock_status);
  if (lock_status != 0 || !lock_consensus) {
    if (publication_lock_.owns_lock())
      publication_lock_.unlock();
    fail_(ProgramTransactionPhase::kHiddenPublish, ProgramTransactionFailure::kHiddenPublish,
          kInvalidProgramTransactionIndex, lock_status == 0 ? 3U : lock_status);
    close_rollback_();
    return failed_publish_();
  }
  if (acquired_writer)
    registry_->link_writer_thread_();
  for (std::size_t index = 0; index < registry_->participants_.size(); ++index) {
    auto& participant = registry_->participants_[index];
    const bool participant_published = participant.ops.publish(participant.object);
    const std::uint32_t participant_status =
        participant_published ? 0U : static_cast<std::uint32_t>(index) + 1U;
    // Authenticate every frozen participant immediately after its local swap.  Waiting until the
    // end of the loop lets a successful rank enter the next participant (which may itself own a
    // collective) while a failed rank has already stopped publishing.  The per-slot reduction
    // gives every rank the same exit point and exact failure ordinal.
    const bool participant_consensus =
        consensus_(ProgramTransactionPhase::kHiddenPublish, participant_status);
    if (!participant_published || !participant_consensus) {
      fail_(ProgramTransactionPhase::kHiddenPublish, ProgramTransactionFailure::kHiddenPublish,
            static_cast<std::uint32_t>(index), participant_consensus ? 0U : 2U);
      close_rollback_();
      return failed_publish_();
    }
    ++published_participants_;
  }
  phase_ = ProgramTransactionPhase::kCompensableEffects;
  registry_->active_phase_ = phase_;
  for (std::size_t index = 0; index < registry_->effects_.size(); ++index) {
    auto& entry = registry_->effects_[index];
    // A compensable publisher is allowed to discover failure after it has changed external
    // state.  Once invoked, it therefore belongs to the rollback set irrespective of its return
    // value.  Publishers must provide an idempotent noexcept compensation for every attempted
    // publication; effects not invoked because an earlier one failed remain prepared-only and are
    // discarded below.
    entry.receipt.generation = base_generation_;
    entry.receipt.published = true;
    const bool effect_published = entry.effect.publish_(entry.effect.context_);
    const std::uint32_t effect_status =
        effect_published ? 0U : static_cast<std::uint32_t>(index) + 1U;
    // Effects can wrap provider-owned collectives.  Consensus per frozen ordinal prevents a rank
    // that succeeded locally from entering the next effect after another rank has failed.
    const bool effect_consensus =
        consensus_(ProgramTransactionPhase::kCompensableEffects, effect_status);
    if (!effect_published || !effect_consensus) {
      fail_(ProgramTransactionPhase::kCompensableEffects, ProgramTransactionFailure::kCompensation,
            static_cast<std::uint32_t>(index), effect_consensus ? 0U : 1U);
      close_rollback_();
      return failed_publish_();
    }
  }
  publication_complete_ = true;
  return {base_generation_, static_cast<std::uint32_t>(registry_->participants_.size()),
          static_cast<std::uint32_t>(registry_->effects_.size()), phase_, true};
}

inline ProgramSealReceipt ProgramTransaction::seal() noexcept {
  if (!active_ || phase_ != ProgramTransactionPhase::kCompensableEffects)
    return {sealed_generation_,
            static_cast<std::uint32_t>(registry_ == nullptr ? 0 : registry_->effects_.size()),
            phase_, false};
  phase_ = ProgramTransactionPhase::kAtomicSeal;
  registry_->active_phase_ = phase_;
  if (!consensus_(ProgramTransactionPhase::kAtomicSeal, 0)) {
    fail_(ProgramTransactionPhase::kAtomicSeal, ProgramTransactionFailure::kAtomicSeal,
          kInvalidProgramTransactionIndex, 1);
    close_rollback_();
    return {base_generation_, static_cast<std::uint32_t>(registry_->effects_.size()), phase_,
            false};
  }
  const std::uint64_t old_generation =
      registry_->accepted_generation_.load(std::memory_order_relaxed);
  if (old_generation == std::numeric_limits<std::uint64_t>::max()) {
    fail_(ProgramTransactionPhase::kAtomicSeal, ProgramTransactionFailure::kAtomicSeal,
          kInvalidProgramTransactionIndex, 2);
    close_rollback_();
    return {base_generation_, static_cast<std::uint32_t>(registry_->effects_.size()), phase_,
            false};
  }
  sealed_generation_ = AcceptedGeneration{old_generation + 1};
  for (auto& entry : registry_->effects_)
    if (entry.receipt.published)
      entry.receipt.generation = sealed_generation_;
  registry_->accepted_generation_.store(sealed_generation_.value, std::memory_order_release);
  unlock_publication_();
  return {sealed_generation_, static_cast<std::uint32_t>(registry_->effects_.size()), phase_, true};
}

inline ProgramFinalizeReceipt ProgramTransaction::finalize() noexcept {
  if (!active_)
    return {sealed_generation_,
            0,
            0,
            phase_,
            phase_ == ProgramTransactionPhase::kAccepted,
            phase_ == ProgramTransactionPhase::kFailStop};
  if (phase_ != ProgramTransactionPhase::kAtomicSeal &&
      phase_ != ProgramTransactionPhase::kIrreversibleFinalize &&
      phase_ != ProgramTransactionPhase::kAccepted && phase_ != ProgramTransactionPhase::kFailStop)
    return {sealed_generation_, 0, 0, phase_, false, false};
  if (phase_ == ProgramTransactionPhase::kAccepted || phase_ == ProgramTransactionPhase::kFailStop)
    return {sealed_generation_,
            0,
            0,
            phase_,
            phase_ == ProgramTransactionPhase::kAccepted,
            phase_ == ProgramTransactionPhase::kFailStop};
  phase_ = ProgramTransactionPhase::kIrreversibleFinalize;
  registry_->active_phase_ = phase_;
  std::uint32_t finalized = 0;
  std::uint32_t failed = 0;
  for (std::size_t index = 0; index < registry_->effects_.size(); ++index) {
    auto& entry = registry_->effects_[index];
    if (!entry.receipt.published || entry.receipt.finalize_attempted)
      continue;
    entry.receipt.finalize_attempted = true;
    const bool ok =
        entry.effect.finalize_ == nullptr || entry.effect.finalize_(entry.effect.context_);
    const std::uint32_t finalizer_status = ok ? 0U : static_cast<std::uint32_t>(index) + 1U;
    const bool finalizer_consensus =
        consensus_(ProgramTransactionPhase::kIrreversibleFinalize, finalizer_status);
    if (ok && finalizer_consensus) {
      entry.receipt.finalized = true;
      ++finalized;
    } else {
      ++failed;
    }
  }
  // Preserve one phase witness even for an empty effect table and authenticate the exact failure
  // count after every fixed ordinal has run.  Finalizers are irreversible, so disagreement is
  // fail-stop and never triggers scientific rollback.
  if (!consensus_(ProgramTransactionPhase::kIrreversibleFinalize, failed))
    ++failed;
  if (failed != 0) {
    phase_ = ProgramTransactionPhase::kFailStop;
    registry_->fail_stop_ = true;
    fail_(ProgramTransactionPhase::kIrreversibleFinalize, ProgramTransactionFailure::kFinalize,
          kInvalidProgramTransactionIndex, failed);
  } else {
    phase_ = ProgramTransactionPhase::kAccepted;
  }
  registry_->clear_active_();
  active_ = false;
  return {sealed_generation_, finalized, failed, phase_, failed == 0, failed != 0};
}

inline void ProgramTransaction::rollback() noexcept {
  if (!active_ || sealed())
    return;
  close_rollback_();
}

inline std::uint32_t ProgramTransaction::effect_count() const noexcept {
  return registry_ == nullptr ? 0 : static_cast<std::uint32_t>(registry_->effects_.size());
}

inline const EffectReceipt* ProgramTransaction::effect_receipt(
    std::uint32_t ordinal) const noexcept {
  if (registry_ == nullptr || ordinal >= registry_->effects_.size())
    return nullptr;
  return &registry_->effects_[ordinal].receipt;
}

inline EffectHandle ProgramTransactionRegistry::register_effect(std::uint64_t identity) {
  if (bound_)
    throw std::logic_error("ProgramTransactionRegistry refuses effect registration after bind");
  if (identity == 0)
    throw std::invalid_argument("ProgramTransactionRegistry effect identity must be non-zero");
  if (effects_.size() == std::numeric_limits<std::uint32_t>::max())
    throw std::length_error("ProgramTransactionRegistry effect index overflow");
  for (const EffectEntry& existing : effects_)
    if (existing.identity == identity)
      throw std::invalid_argument("ProgramTransactionRegistry refuses duplicate effect identity");
  if (budget_.max_effects != 0 && effects_.size() >= budget_.max_effects)
    throw std::length_error("ProgramTransactionRegistry effect budget exceeded");
  EffectEntry entry;
  entry.identity = identity;
  entry.order = static_cast<std::uint32_t>(effects_.size());
  entry.bind_status = ProgramEffectSlotStatus::kDeclared;
  entry.receipt.identity = identity;
  entry.receipt.ordinal = entry.order;
  effects_.push_back(std::move(entry));
  return {static_cast<std::uint32_t>(effects_.size() - 1), identity};
}

inline EffectHandle ProgramTransactionRegistry::try_register_effect(
    std::uint64_t identity) noexcept {
  try {
    return register_effect(identity);
  } catch (...) {
    return {};
  }
}

inline void ProgramTransactionRegistry::bind() {
  if (bound_)
    throw std::logic_error("ProgramTransactionRegistry::bind called twice");
  if (budget_.max_participants != 0 && participants_.size() > budget_.max_participants)
    throw std::length_error("ProgramTransactionRegistry participant budget exceeded");
  if (participants_.size() > std::numeric_limits<std::uint32_t>::max())
    throw std::length_error("ProgramTransactionRegistry participant count is not representable");
  if (budget_.max_effects > std::numeric_limits<std::uint32_t>::max())
    throw std::length_error("ProgramTransactionRegistry effect count is not representable");
  if (budget_.max_effects != 0 && effects_.size() > budget_.max_effects)
    throw std::length_error("ProgramTransactionRegistry effect budget exceeded");
  std::size_t restore_total = 0;
  std::size_t candidate_total = 0;
  for (auto& participant : participants_) {
    if (participant.budget.restore_bytes >
            std::numeric_limits<std::size_t>::max() - restore_total ||
        participant.budget.candidate_bytes >
            std::numeric_limits<std::size_t>::max() - candidate_total)
      throw std::length_error("ProgramTransactionRegistry participant budget overflow");
    restore_total += participant.budget.restore_bytes;
    candidate_total += participant.budget.candidate_bytes;
  }
  if (budget_.max_restore_bytes != 0 && restore_total > budget_.max_restore_bytes)
    throw std::length_error("ProgramTransactionRegistry restore budget exceeded");
  if (budget_.max_candidate_bytes != 0 && candidate_total > budget_.max_candidate_bytes)
    throw std::length_error("ProgramTransactionRegistry candidate budget exceeded");
  participants_.reserve(participants_.size());
  effects_.reserve(effects_.size());
  try {
    for (auto& participant : participants_)
      participant.restore_image.resize(participant.budget.restore_bytes);
  } catch (...) {
    for (auto& participant : participants_)
      participant.restore_image.clear();
    throw;
  }
  restore_bytes_ = restore_total;
  candidate_bytes_ = candidate_total;
  for (std::size_t index = 0; index < effects_.size(); ++index) {
    EffectEntry& effect = effects_[index];
    if (effect.identity == 0 || effect.order != index ||
        effect.bind_status != ProgramEffectSlotStatus::kDeclared)
      throw std::logic_error("ProgramTransactionRegistry effect slots are not bind-canonical");
    effect.bind_status = ProgramEffectSlotStatus::kFrozen;
  }
  bound_ = true;
  active_phase_ = ProgramTransactionPhase::kAccepted;
}

inline AcceptedReadLease ProgramTransactionRegistry::acquire_read() const {
  if (ProvisionalReadLease* scope = ProvisionalReadLease::find_(this)) {
    if (!provisional_scope_allowed_(scope))
      throw std::logic_error(
          "Program accepted reader provisional scope is no longer authenticated");
    return AcceptedReadLease(this, accepted_generation(), scope);
  }
  if (writer_thread_owns_(this))
    throw std::logic_error(
        "Program accepted reader requires an explicit provisional read scope during a candidate");
  if (AcceptedReadLease* root = AcceptedReadLease::find_(this)) {
    if (!root->authenticated_root_(this))
      std::terminate();
    return AcceptedReadLease(this, root->generation_, root);
  }
  std::shared_lock<std::shared_mutex> lock(visibility_mutex_);
  return AcceptedReadLease(this, std::move(lock), accepted_generation());
}

inline ProvisionalReadLease ProgramTransactionRegistry::acquire_provisional_read() const {
  if (!bound_ || fail_stop_ || !active_ ||
      last_fault_.failure != ProgramTransactionFailure::kNone || !writer_thread_owns_(this) ||
      (active_phase_ != ProgramTransactionPhase::kCandidate &&
       active_phase_ != ProgramTransactionPhase::kSolveGuardEffectPrepare))
    throw std::logic_error(
        "Program provisional read scope requires an active writer-owned candidate transaction");
  return ProvisionalReadLease(this);
}

inline AcceptedWriteLease ProgramTransactionRegistry::acquire_write() const {
  if (AcceptedReadLease* root = AcceptedReadLease::find_(this)) {
    if (!root->authenticated_root_(this))
      std::terminate();
    throw std::logic_error(
        "Program transaction writer cannot be acquired while an accepted read lease is held");
  }
  if (writer_thread_owns_(this))
    throw std::logic_error(
        "Program transaction writer cannot be acquired recursively by an external savepoint");
  std::unique_lock<std::shared_mutex> lock(visibility_mutex_);
  link_writer_thread_();
  return AcceptedWriteLease(this, std::move(lock));
}

inline void ProgramTransactionRegistry::set_consensus(ProgramTransactionConsensus consensus) {
  if (bound_)
    throw std::logic_error("ProgramTransactionRegistry refuses a consensus change after bind");
  consensus_callback_ = consensus;
}

inline void ProgramTransactionRegistry::set_candidate_visibility_lock(bool enabled) {
  if (bound_)
    throw std::logic_error(
        "ProgramTransactionRegistry refuses a visibility policy change after bind");
  candidate_visibility_lock_ = enabled;
}

template <class T>
inline const T* AcceptedReadLease::read(const ParticipantHandle<T>& handle) const noexcept {
  if (!valid())
    return nullptr;
  const auto* registry = static_cast<const ProgramTransactionRegistry*>(registry_);
  return provisional_scope_ == nullptr ? registry->read_(handle)
                                       : registry->provisional_read_(handle, provisional_scope_);
}

}  // namespace pops::runtime::program
