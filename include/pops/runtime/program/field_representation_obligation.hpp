#pragma once

#include <memory>
#include <stdexcept>
#include <string>

namespace pops::runtime::program {

// Description retained by an immutable value; not permission to transform or read it.
// The contract includes the exact authored transformation and its closed bindings.
// The origin names the producing evaluation, independently of a live storage pointer.
struct FieldRepresentationObligation {
  using Ref = std::shared_ptr<const FieldRepresentationObligation>;
  const std::string contract;
  const std::string origin;

  void validate() const {
    if (contract.empty() || origin.empty())
      throw std::invalid_argument("Field representation obligation is incomplete");
  }
  friend bool operator==(const FieldRepresentationObligation&,
                         const FieldRepresentationObligation&) = default;
};

}  // namespace pops::runtime::program
