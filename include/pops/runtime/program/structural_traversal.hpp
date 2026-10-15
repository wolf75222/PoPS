#pragma once

#include <cstddef>
#include <cstdint>
#include <limits>
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
