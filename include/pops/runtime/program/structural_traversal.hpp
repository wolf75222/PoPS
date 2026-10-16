#pragma once

#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory_resource>
#include <stdexcept>
#include <unordered_set>
#include <vector>

namespace pops::runtime::program::continuation_detail {

// A traversal consumes finite work and bounds its live frame buffer using the
// owning codec/registry's declared resource capacity. Neither is a depth limit.
class StructuralBudget {
 public:
  explicit StructuralBudget(std::uint64_t capacity) : remaining_(capacity), capacity_(capacity) {}

  void consume(std::uint64_t units = 1) {
    if (units > remaining_)
      throw std::length_error("structural traversal exceeds its work capacity");
    remaining_ -= units;
  }

  void frames(std::size_t count, std::size_t width) const {
    if (width != 0 && count > capacity_ / width)
      throw std::length_error("structural traversal exceeds its frame capacity");
  }

 private:
  std::uint64_t remaining_;
  std::uint64_t capacity_;
};

// Account for the bytes requested by a temporary container (both nodes and
// buckets), before each upstream allocation. Allocator implementation overhead
// is outside this requested-byte budget, as for the other native containers.
class StructuralWorkspaceResource final : public std::pmr::memory_resource {
 public:
  explicit StructuralWorkspaceResource(std::uint64_t capacity, std::pmr::memory_resource* upstream =
                                                                   std::pmr::get_default_resource())
      : capacity_(capacity), upstream_(upstream) {
    if (!upstream_)
      throw std::invalid_argument("structural workspace needs an allocator");
  }
  std::uint64_t used() const noexcept { return used_; }

 private:
  void* do_allocate(std::size_t bytes, std::size_t alignment) override {
    if (bytes > capacity_ - used_)
      throw std::length_error("structural workspace exceeds its byte capacity");
    void* storage = upstream_->allocate(bytes, alignment);
    used_ += bytes;
    return storage;
  }
  void do_deallocate(void* storage, std::size_t bytes, std::size_t alignment) override {
    upstream_->deallocate(storage, bytes, alignment);
    used_ -= bytes;
  }
  bool do_is_equal(const std::pmr::memory_resource& other) const noexcept override {
    return this == &other;
  }
  std::uint64_t capacity_;
  std::uint64_t used_ = 0;
  std::pmr::memory_resource* upstream_;
};

// Borrow every node. In particular no shared_ptr copy may perturb a GC census.
// enter=false prunes an already authenticated alias; active-path cycles always
// fail. Children and exit retain the recursive traversal's exact ordinal order.
template <class Node, class Count, class Child, class Enter, class Exit>
void walk_structure(const Node* root, Count count, Child child, Enter enter, Exit exit,
                    StructuralBudget& budget) {
  struct Frame {
    const Node* node;
    std::size_t next;
  };
  std::vector<Frame> stack;
  std::unordered_set<const Node*> active;
  auto push = [&](const Node* node) {
    budget.consume();
    if (!node)
      throw std::invalid_argument("structural traversal contains a missing node");
    if (active.contains(node))
      throw std::invalid_argument("structural value graph is cyclic");
    if (!enter(*node)) {
      exit(*node);
      return;
    }
    // A leaf cannot lead back to an active ancestor. Preserve both callbacks
    // and the actual child/lifetime check without allocating a DFS frame or
    // an active-set node for every Index in a large Requests Array.
    if (count(*node) == 0) {
      exit(*node);
      return;
    }
    budget.frames(stack.size() + 1, sizeof(Frame));
    stack.push_back({node, 0});
    active.insert(node);
  };
  push(root);
  while (!stack.empty()) {
    auto& frame = stack.back();
    if (frame.next < count(*frame.node)) {
      const Node* next = child(*frame.node, frame.next++);
      push(next);
    } else {
      const Node* node = frame.node;
      exit(*node);
      active.erase(node);
      stack.pop_back();
    }
  }
}

}  // namespace pops::runtime::program::continuation_detail
