#pragma once

#include <functional>
#include <memory>
#include <stdexcept>
#include <thread>
#include <utility>

namespace pops {
template <int Dim>
class SystemBlockStore;

/// One exact group core invocation, issued only by the owning System block store.
/// The provider cannot substitute a block, field, point, lane, or transport session.
template <int Dim>
class SystemInterfaceCoreSession final {
 public:
  using pointer = std::shared_ptr<SystemInterfaceCoreSession>;
  SystemInterfaceCoreSession(const SystemInterfaceCoreSession&) = delete;
  SystemInterfaceCoreSession& operator=(const SystemInterfaceCoreSession&) = delete;

  void evaluate() {
    if (std::this_thread::get_id() != owner_thread_)
      throw std::logic_error("shared-interface core session belongs to another execution thread");
    if (!active_ || *active_slot_ != this)
      throw std::logic_error("shared-interface core session is stale or belongs to another group");
    if (used_)
      throw std::logic_error("shared-interface core session was already consumed");
    used_ = true;
    core_();
    completed_ = true;
  }

 private:
  friend class SystemBlockStore<Dim>;
  explicit SystemInterfaceCoreSession(std::function<void()> core)
      : core_(std::move(core)),
        owner_thread_(std::this_thread::get_id()),
        active_slot_(&current_) {}

  template <class Preflight, class Core, class Provider, class Admission>
  static void dispatch(Preflight&& preflight, Core&& core, Provider&& provider,
                       Admission&& admission) {
    pointer session;
    auto prepare = [&] {
      preflight();
      session = pointer(new SystemInterfaceCoreSession(std::forward<Core>(core)));
    };
    // The action is a stack-owned POD thunk: even std::function conversion and token/control-
    // block allocation occur inside the owning lane's collective admission.
    admission([](void* payload) { (*static_cast<decltype(prepare)*>(payload))(); }, &prepare);
    auto* previous = current_;
    current_ = session.get();
    session->active_ = true;
    const auto close = [&] {
      session->active_ = false;
      session->core_ = {};
      current_ = previous;
    };
    try {
      provider(session);
      if (!session->completed_)
        throw std::logic_error("shared-interface provider did not complete its exact core session");
    } catch (...) {
      close();
      throw;
    }
    close();
  }

  std::function<void()> core_;
  const std::thread::id owner_thread_;
  SystemInterfaceCoreSession** const active_slot_;
  bool active_ = false;
  bool completed_ = false;
  bool used_ = false;
  inline static thread_local SystemInterfaceCoreSession* current_ = nullptr;
};
}  // namespace pops
